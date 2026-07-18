"""
counterfactual.py — gradient-based counterfactuals (Phase 7).

Question a clinician actually asks: "what would have to change for this to look healthy?"
We answer it by gently nudging the input image — via gradient descent ON THE PIXELS — toward
the 'notumor' class, while an L2 penalty keeps the change minimal. The DIFFERENCE map
(|counterfactual - original|) then highlights the regions the model relies on to call it a
tumor: erase/soften those and the prediction flips to healthy.

This is a form of explanation and a sanity check: the difference should concentrate ON the
tumor, not scattered noise. (Like the other viz, it only looks clean once the model is trained.)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

import config


def counterfactual(model, x, target_class, steps=60, lr=0.01, l2=0.05, eps=1.5):
    """Nudge x toward target_class. Returns (counterfactual x, difference map (H,W), history).

    steps/lr : how far and how fast we move the pixels.
    l2       : how strongly we insist the change stay small (bigger = subtler edit).
    eps      : L-inf bound on the per-pixel change. Projecting the perturbation into this ball
               each step keeps the edit small AND stops a brittle model from being driven into
               an adversarial blow-up regime.
    """
    device = x.device
    x0 = x.clone().detach()
    x_cf = x.clone().detach().requires_grad_(True)
    target = torch.full((x.size(0),), int(target_class), dtype=torch.long, device=device)
    opt = torch.optim.Adam([x_cf], lr=lr)

    model.eval()
    history = []
    for _ in range(steps):
        opt.zero_grad()
        logits = model(x_cf)
        # push toward target class, but penalize drifting away from the original image.
        loss = F.cross_entropy(logits, target) + l2 * ((x_cf - x0) ** 2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            # Project the change into an L-inf ball around x0, then clamp to a valid image range.
            x_cf.data = x0 + torch.clamp(x_cf.data - x0, -eps, eps)
            x_cf.data = torch.clamp(x_cf.data, -3.0, 3.0)
            p_target = F.softmax(model(x_cf), dim=1)[0, int(target_class)].item()
        history.append(p_target)

    diff = (x_cf - x0).detach().abs().sum(dim=1)[0].cpu().numpy()  # (H,W)
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    return x_cf.detach(), diff, history


if __name__ == "__main__":
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from modules.model import CertifyBTC
    from modules.datasets import build_dataloaders
    from modules.preprocessing import denormalize
    from train import load_checkpoint

    device = config.DEVICE
    model = CertifyBTC().to(device)
    ck = os.path.join(config.CHECKPOINT_DIR, "stage1_best.pth")
    if os.path.exists(ck):
        load_checkpoint(ck, model, device=device)
    model.eval()   # BatchNorm needs eval mode for single-image (batch=1) inference

    notumor_idx = config.CLASS_NAMES.index("notumor")

    # Pick a test image the model currently thinks is a tumor (not 'notumor').
    xb, yb = next(iter(build_dataloaders()[2]))
    x = None
    for i in range(xb.shape[0]):
        xi = xb[i:i + 1].to(device)
        if int(model(xi).argmax(1)) != notumor_idx:
            x, true_lbl = xi, config.CLASS_NAMES[yb[i]]
            break
    if x is None:
        x, true_lbl = xb[:1].to(device), config.CLASS_NAMES[yb[0]]

    with torch.no_grad():
        p_before = F.softmax(model(x), 1)[0, notumor_idx].item()
    x_cf, diff, history = counterfactual(model, x, notumor_idx)
    p_after = history[-1]

    print(f"Counterfactual toward 'notumor'  (true label: {true_lbl})")
    print(f"  P(notumor) before : {p_before:.3f}")
    print(f"  P(notumor) after  : {p_after:.3f}   (should increase)")
    print(f"  trajectory        : {[round(h,3) for h in history[::12]]}")

    img_before = denormalize(x[0])
    img_after = denormalize(x_cf[0])
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img_before); ax[0].set_title(f"original ({true_lbl})"); ax[0].axis("off")
    ax[1].imshow(img_after); ax[1].set_title(f"counterfactual\nP(notumor) {p_after:.2f}"); ax[1].axis("off")
    ax[2].imshow(img_before); ax[2].imshow(diff, cmap="jet", alpha=0.5)
    ax[2].set_title("difference (what changed)"); ax[2].axis("off")
    plt.tight_layout()
    out = os.path.join(config.LOG_DIR, "phase7_counterfactual_demo.png")
    plt.savefig(out, dpi=90)
    print(f"Saved -> {out}")
    print("counterfactual OK.")
