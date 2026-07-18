"""
train.py — two-stage training for CERTIFY-BTC (Phases 3 & 5).

  Stage 1 (python train.py            or  --stage 1):
      freeze EfficientNetB4, train CBAM + fusion + head. Adam 1e-3, Focal Loss (glioma 2.0),
      cosine LR.
  Stage 2 (python train.py --stage 2):
      resume Stage 1, unfreeze the last 2 blocks (lr 1e-4). With >=2 datasets it adds
      domain-adversarial training (GRL + domain head); with ONE dataset it's plain fine-tuning.

Built for a 6GB GPU: AMP (fp16) + gradient accumulation (effective batch = BATCH_SIZE *
ACCUM_STEPS). A checkpoint is saved EVERY epoch, and each stage reports TEST accuracy at the end.

The run size is controlled by config.MACHINE (set via the CERTIFY_MACHINE env var):
  local (debug, 200 imgs)  |  local_full (all Nickparvar, full epochs)  |  cloud (everything).
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
    """Load a checkpoint safely across machines. map_location moves GPU-saved tensors onto THIS
    device so it never crashes on load. weights_only=False because our checkpoint also stores
    optimizer state + ints (it's our own trusted file)."""
    device = device or config.DEVICE
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def _autocast(device):
    use_amp = config.USE_AMP and device == "cuda"
    return torch.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=use_amp)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Return (avg loss, overall accuracy, per-class accuracy=recall list) on a loader."""
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


def _report_test(model, test_loader, loss_fn, device, tag):
    """Load the best checkpoint of this stage and report its TEST accuracy + per-class recall.
    Per-class accuracy IS recall here — watch glioma (HXAI-BTC's weak class at 0.82)."""
    best = os.path.join(config.CHECKPOINT_DIR, f"{tag}_best.pth")
    if os.path.exists(best):
        load_checkpoint(best, model, device=device)
    _, acc, per = evaluate(model, test_loader, loss_fn, device)
    pca = "  ".join(f"{config.CLASS_NAMES[c]}={per[c]:.3f}" for c in range(config.NUM_CLASSES))
    print("=" * 60)
    print(f"[{tag}] TEST accuracy      : {acc:.4f}")
    print(f"[{tag}] TEST recall/class  : {pca}")
    print("=" * 60)
    return acc, per


def _save_epoch(model, optimizer, epoch, val_acc, tag, best_val, extra=None):
    ckpt = {"epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "val_acc": val_acc, "machine": config.MACHINE}
    if extra:
        ckpt.update(extra)
    save_checkpoint(ckpt, os.path.join(config.CHECKPOINT_DIR, f"{tag}_epoch{epoch}.pth"))
    if val_acc >= best_val:
        save_checkpoint(ckpt, os.path.join(config.CHECKPOINT_DIR, f"{tag}_best.pth"))
        print(f"   -> new best (val acc {val_acc:.3f}) saved to {tag}_best.pth")


# ---------------------------------------------------------------------------
# STAGE 1
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, accum_steps):
    model.train()
    running, n = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    n_batches = len(loader)
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        with _autocast(device):
            logits = model(x)
            loss = loss_fn(logits, y) / accum_steps
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
          f"= {config.BATCH_SIZE * config.ACCUM_STEPS} | epochs={config.STAGE1_EPOCHS}\n")

    train_loader, val_loader, test_loader = build_dataloaders()

    model = CertifyBTC().to(device)
    model.freeze_backbone(True)
    total, trainable = count_params(model)
    print(f"\nParams: {total:,} total | {trainable:,} trainable (backbone frozen)\n")

    loss_fn = FocalLoss(alpha=torch.tensor(config.CLASS_WEIGHTS_LIST, device=device),
                        gamma=config.FOCAL_GAMMA)
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
    _report_test(model, test_loader, loss_fn, device, "stage1")


# ---------------------------------------------------------------------------
# STAGE 2 (fine-tune; domain-adversarial only when >= 2 datasets)
# ---------------------------------------------------------------------------
def train_stage2(num_domains=2):
    set_seed(config.SEED)
    device = config.DEVICE
    use_domain = len(config.ACTIVE_DATASETS) > 1     # GRL only helps with multiple datasets
    print(f"[Stage 2] Device: {device} | unfreeze last {config.STAGE2['unfreeze_last_blocks']} "
          f"blocks | domain-adversarial: {'ON' if use_domain else 'OFF (single dataset)'} | "
          f"epochs={config.STAGE2_EPOCHS}\n")

    train_loader, val_loader, test_loader = build_dataloaders()

    model = CertifyBTC().to(device)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "stage1_best.pth")
    if os.path.exists(ckpt_path):
        load_checkpoint(ckpt_path, model, device=device)
        print(f"Resumed from {ckpt_path}")
    else:
        print("No Stage-1 checkpoint found — training Stage 2 from ImageNet init.")
    model.unfreeze_last_blocks(config.STAGE2["unfreeze_last_blocks"])

    class_loss_fn = FocalLoss(alpha=torch.tensor(config.CLASS_WEIGHTS_LIST, device=device),
                              gamma=config.FOCAL_GAMMA)
    params = [p for p in model.parameters() if p.requires_grad]
    domain_clf, domain_loss_fn = None, None
    if use_domain:
        domain_clf = DomainClassifier(in_dim=512, num_domains=num_domains).to(device)
        domain_loss_fn = nn.CrossEntropyLoss()
        params = params + list(domain_clf.parameters())
    else:
        print("(single dataset -> Stage 2 is plain fine-tuning: unfreeze + Focal, no GRL)")

    optimizer = Adam(params, lr=config.STAGE2["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config.STAGE2_EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=config.USE_AMP and device == "cuda")

    total, trainable = count_params(model)
    print(f"Params: {total:,} total | {trainable:,} trainable\n")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    total_steps = config.STAGE2_EPOCHS * len(train_loader)
    best_val = -1.0

    for epoch in range(1, config.STAGE2_EPOCHS + 1):
        model.train()
        if domain_clf is not None:
            domain_clf.train()
        c_run, d_run, n, d_correct = 0.0, 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        nb = len(train_loader)
        alpha, t0 = 0.0, time.time()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            with _autocast(device):
                if use_domain:
                    logits, fused, _ = model(x, return_maps=True)
                else:
                    logits = model(x)
                class_loss = class_loss_fn(logits, y)
                loss = class_loss
                if use_domain:
                    d = torch.randint(0, num_domains, (x.size(0),), device=device)
                    alpha = grl_alpha((epoch - 1) * nb + i, total_steps)
                    domain_logits = domain_clf(fused, alpha=alpha)
                    domain_loss = domain_loss_fn(domain_logits, d)
                    loss = class_loss + domain_loss
                loss = loss / config.ACCUM_STEPS
            scaler.scale(loss).backward()
            if (i + 1) % config.ACCUM_STEPS == 0 or (i + 1) == nb:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            c_run += class_loss.item() * x.size(0)
            n += x.size(0)
            if use_domain:
                d_run += domain_loss.item() * x.size(0)
                d_correct += int((domain_logits.argmax(1) == d).sum())

        scheduler.step()
        val_loss, val_acc, per_acc = evaluate(model, val_loader, class_loss_fn, device)
        pca = "  ".join(f"{config.CLASS_NAMES[c][:5]}={per_acc[c]:.2f}"
                        for c in range(config.NUM_CLASSES))
        dom_txt = f"domain {d_run/n:.3f} (dom_acc {d_correct/n:.2f}) | " if use_domain else ""
        print(f"epoch {epoch}/{config.STAGE2_EPOCHS} ({time.time()-t0:.0f}s) | "
              f"class {c_run/n:.4f} | {dom_txt}val acc {val_acc:.3f} | {pca}")
        extra = {"domain_clf": domain_clf.state_dict()} if domain_clf is not None else None
        _save_epoch(model, optimizer, epoch, val_acc, "stage2", best_val, extra=extra)
        best_val = max(best_val, val_acc)

    print(f"\n[Stage 2] complete. Best val acc: {best_val:.3f}")
    _report_test(model, test_loader, class_loss_fn, device, "stage2")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CERTIFY-BTC training")
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2],
                    help="1 = frozen backbone (Focal); 2 = unfreeze (+ domain-adversarial if >=2 datasets)")
    args = ap.parse_args()
    (train_stage1 if args.stage == 1 else train_stage2)()
