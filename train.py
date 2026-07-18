"""
train.py — two-stage training for CERTIFY-BTC (Phases 3 & 5).

  Stage 1 (python train.py            or  --stage 1):
      freeze EfficientNetB4, train CBAM + fusion + head. Adam 1e-3, Focal Loss (glioma 2.0),
      cosine LR.
  Stage 2 (python train.py --stage 2):
      resume Stage 1, unfreeze the last 2 blocks (lr 1e-4), and add domain-adversarial training
      (Gradient Reversal + domain head, alpha ramps 0 -> 1) for multi-site robustness.

Built for a 6GB GPU: Automatic Mixed Precision (fp16) + gradient accumulation
(effective batch = BATCH_SIZE * ACCUM_STEPS). A checkpoint is saved EVERY epoch.
"""

import os
import sys
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from modules.datasets import build_dataloaders
from modules.model import CertifyBTC, count_params
from modules.losses import FocalLoss
from modules.domain_adversarial import DomainClassifier, grl_alpha


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
    """Mixed-precision context — enabled only on CUDA per config."""
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


# ---------------------------------------------------------------------------
# STAGE 1
# ---------------------------------------------------------------------------
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
            loss = loss_fn(logits, y) / accum_steps  # scale so summed grads == big-batch grad
        scaler.scale(loss).backward()
        if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        running += loss.item() * accum_steps * x.size(0)
        n += x.size(0)
    return running / max(n, 1)


def train_stage1():
    set_seed(config.SEED)
    device = config.DEVICE
    print(f"[Stage 1] Device: {device} | MACHINE={config.MACHINE} | "
          f"AMP={config.USE_AMP and device=='cuda'}")
    print(f"Effective batch = {config.BATCH_SIZE} x {config.ACCUM_STEPS} "
          f"= {config.BATCH_SIZE * config.ACCUM_STEPS}\n")

    train_loader, val_loader, _ = build_dataloaders()

    model = CertifyBTC().to(device)
    model.freeze_backbone(True)
    total, trainable = count_params(model)
    print(f"\nParams: {total:,} total | {trainable:,} trainable (backbone frozen)\n")

    alpha = torch.tensor(config.CLASS_WEIGHTS_LIST, device=device)
    loss_fn = FocalLoss(alpha=alpha, gamma=config.FOCAL_GAMMA)

    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=config.STAGE1["lr"])
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
        _save_epoch(model, optimizer, epoch, val_acc, "stage1", best_val)
        best_val = max(best_val, val_acc)
    print(f"\n[Stage 1] complete. Best val acc: {best_val:.3f}")


# ---------------------------------------------------------------------------
# STAGE 2 (domain-adversarial)
# ---------------------------------------------------------------------------
def train_stage2(num_domains=2):
    set_seed(config.SEED)
    device = config.DEVICE
    print(f"[Stage 2] Device: {device} | domain-adversarial (GRL) | "
          f"unfreeze last {config.STAGE2['unfreeze_last_blocks']} blocks\n")

    train_loader, val_loader, _ = build_dataloaders()

    model = CertifyBTC().to(device)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "stage1_best.pth")
    if os.path.exists(ckpt_path):
        load_checkpoint(ckpt_path, model, device=device)
        print(f"Resumed from {ckpt_path}")
    else:
        print("No Stage-1 checkpoint found — training Stage 2 from ImageNet init.")
    model.unfreeze_last_blocks(config.STAGE2["unfreeze_last_blocks"])

    # NOTE: with one dataset (local) there is only one real domain, so we SIMULATE `num_domains`
    # domains to exercise the machinery. On the cloud, domain labels come from the datasets.
    domain_clf = DomainClassifier(in_dim=512, num_domains=num_domains).to(device)
    print("(!) LOCAL mechanism check: domains are SIMULATED — real domains need >=2 datasets.")

    class_loss_fn = FocalLoss(alpha=torch.tensor(config.CLASS_WEIGHTS_LIST, device=device),
                              gamma=config.FOCAL_GAMMA)
    domain_loss_fn = nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad] + list(domain_clf.parameters())
    optimizer = Adam(params, lr=config.STAGE2["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config.STAGE2_EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=config.USE_AMP and device == "cuda")

    total, trainable = count_params(model)
    print(f"Params: {total:,} total | {trainable:,} trainable (last blocks unfrozen)\n")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    total_steps = config.STAGE2_EPOCHS * len(train_loader)
    best_val = -1.0

    for epoch in range(1, config.STAGE2_EPOCHS + 1):
        model.train(); domain_clf.train()
        c_run, d_run, n, d_correct = 0.0, 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        nb = len(train_loader)
        alpha = 0.0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            # Simulated domain labels for the local mechanism check.
            d = torch.randint(0, num_domains, (x.size(0),), device=device)
            # Ramp alpha 0 -> 1 across the whole run (per-step so it moves even in a 1-epoch run).
            alpha = grl_alpha((epoch - 1) * nb + i, total_steps)

            with _autocast(device):
                logits, fused, _ = model(x, return_maps=True)
                class_loss = class_loss_fn(logits, y)
                domain_logits = domain_clf(fused, alpha=alpha)
                domain_loss = domain_loss_fn(domain_logits, d)
                loss = (class_loss + domain_loss) / config.ACCUM_STEPS
            scaler.scale(loss).backward()
            if (i + 1) % config.ACCUM_STEPS == 0 or (i + 1) == nb:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            c_run += class_loss.item() * x.size(0)
            d_run += domain_loss.item() * x.size(0)
            d_correct += int((domain_logits.argmax(1) == d).sum())
            n += x.size(0)

        scheduler.step()
        val_loss, val_acc, per_acc = evaluate(model, val_loader, class_loss_fn, device)
        pca = "  ".join(f"{config.CLASS_NAMES[c][:5]}={per_acc[c]:.2f}"
                        for c in range(config.NUM_CLASSES))
        print(f"epoch {epoch}/{config.STAGE2_EPOCHS} | alpha {alpha:.2f} | "
              f"class {c_run/n:.4f}  domain {d_run/n:.4f} (dom_acc {d_correct/n:.2f}) | "
              f"val acc {val_acc:.3f} | {pca}")
        _save_epoch(model, optimizer, epoch, val_acc, "stage2", best_val,
                    extra={"domain_clf": domain_clf.state_dict()})
        best_val = max(best_val, val_acc)

    print(f"\n[Stage 2] complete. Best val acc: {best_val:.3f}")
    print("(dom_acc near 1/num_domains means domains are indistinguishable — the GOAL. Here it's "
          "simulated, so treat this as a plumbing check only.)")


# ---------------------------------------------------------------------------
def _save_epoch(model, optimizer, epoch, val_acc, tag, best_val, extra=None):
    ckpt = {"epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "val_acc": val_acc, "machine": config.MACHINE}
    if extra:
        ckpt.update(extra)
    save_checkpoint(ckpt, os.path.join(config.CHECKPOINT_DIR, f"{tag}_epoch{epoch}.pth"))
    if val_acc >= best_val:
        save_checkpoint(ckpt, os.path.join(config.CHECKPOINT_DIR, f"{tag}_best.pth"))
        print(f"   -> new best (val acc {val_acc:.3f}) saved to {tag}_best.pth")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CERTIFY-BTC training")
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2],
                    help="1 = frozen backbone (Focal); 2 = unfreeze + domain-adversarial")
    args = ap.parse_args()
    (train_stage1 if args.stage == 1 else train_stage2)()
