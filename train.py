"""
train.py — Stage-1 training loop for CERTIFY-BTC (Phase 3).

Stage 1 (from config.STAGE1): freeze EfficientNetB4, train only CBAM + fusion + head with
Adam (lr 1e-3), Focal Loss (glioma weight 2.0), and a cosine LR schedule.

Built for a 6GB GPU:
  - Automatic Mixed Precision (fp16) halves the memory of activations.
  - Gradient accumulation makes ACCUM_STEPS small batches act like one big batch, so the
    effective batch size is BATCH_SIZE * ACCUM_STEPS without the memory cost.
  - A checkpoint is saved EVERY epoch to checkpoints/ (never lose a run again).

Run it:  python train.py
In local mode this does a fast 2-epoch smoke run on ~200 images.
"""

import os
import sys
import time
import random

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from modules.datasets import build_dataloaders
from modules.model import CertifyBTC, count_params
from modules.losses import FocalLoss


def set_seed(seed):
    """Pin every RNG so a run is reproducible — important for a paper."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(state, path):
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, device=None):
    """Load a checkpoint safely across machines. map_location moves GPU-saved tensors onto
    THIS device (CPU or a different GPU) so it never crashes on load. weights_only=False
    because our checkpoint also stores optimizer state + ints (it's our own trusted file)."""
    device = device or config.DEVICE
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def _autocast(device):
    """Context manager for mixed precision — enabled only on CUDA in local/cloud config."""
    use_amp = config.USE_AMP and device == "cuda"
    return torch.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=use_amp)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Return (avg loss, overall accuracy, per-class accuracy list) on a loader."""
    model.eval()
    total_loss, n, correct = 0.0, 0, 0
    per_correct = [0] * config.NUM_CLASSES
    per_total = [0] * config.NUM_CLASSES
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with _autocast(device):
            logits = model(x)
            loss = loss_fn(logits, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        for c in range(config.NUM_CLASSES):
            mask = y == c
            per_total[c] += int(mask.sum())
            per_correct[c] += int((pred[mask] == c).sum())
    acc = correct / max(n, 1)
    per_acc = [per_correct[c] / per_total[c] if per_total[c] else float("nan")
               for c in range(config.NUM_CLASSES)]
    return total_loss / max(n, 1), acc, per_acc


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, accum_steps):
    """One pass over the training data with AMP + gradient accumulation."""
    model.train()
    running, n = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    n_batches = len(loader)
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        with _autocast(device):
            logits = model(x)
            # Divide by accum_steps so the summed gradients equal one big-batch gradient.
            loss = loss_fn(logits, y) / accum_steps
        scaler.scale(loss).backward()

        # Step once every accum_steps batches (and flush any remainder at the end).
        if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running += loss.item() * accum_steps * x.size(0)  # undo the division for reporting
        n += x.size(0)
    return running / max(n, 1)


def main():
    set_seed(config.SEED)
    device = config.DEVICE
    print(f"Device: {device} | MACHINE={config.MACHINE} | AMP={config.USE_AMP and device=='cuda'}")
    print(f"Effective batch = {config.BATCH_SIZE} x {config.ACCUM_STEPS} accum "
          f"= {config.BATCH_SIZE * config.ACCUM_STEPS}\n")

    train_loader, val_loader, _test_loader = build_dataloaders()

    model = CertifyBTC().to(device)
    model.freeze_backbone(True)  # Stage 1: only CBAM + fusion + head train
    total, trainable = count_params(model)
    print(f"\nParams: {total:,} total | {trainable:,} trainable (backbone frozen)\n")

    alpha = torch.tensor(config.CLASS_WEIGHTS_LIST, device=device)
    loss_fn = FocalLoss(alpha=alpha, gamma=config.FOCAL_GAMMA)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=config.STAGE1["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config.STAGE1_EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=config.USE_AMP and device == "cuda")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_val = -1.0

    for epoch in range(1, config.STAGE1_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler,
                                     device, config.ACCUM_STEPS)
        val_loss, val_acc, per_acc = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        pca = "  ".join(f"{config.CLASS_NAMES[c][:5]}={per_acc[c]:.2f}"
                        for c in range(config.NUM_CLASSES))
        print(f"epoch {epoch}/{config.STAGE1_EPOCHS} ({time.time()-t0:.0f}s) | "
              f"train {train_loss:.4f} | val {val_loss:.4f} acc {val_acc:.3f} | {pca}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_acc": val_acc,
            "machine": config.MACHINE,
        }
        save_checkpoint(ckpt, os.path.join(config.CHECKPOINT_DIR, f"stage1_epoch{epoch}.pth"))
        if val_acc >= best_val:
            best_val = val_acc
            save_checkpoint(ckpt, os.path.join(config.CHECKPOINT_DIR, "stage1_best.pth"))
            print(f"   -> new best (val acc {val_acc:.3f}) saved to stage1_best.pth")

    print(f"\nStage 1 complete. Best val acc: {best_val:.3f}")
    print(f"Checkpoints in: {config.CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
