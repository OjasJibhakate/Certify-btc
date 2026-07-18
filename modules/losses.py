"""
losses.py — training losses for CERTIFY-BTC (Phase 3).

Focal Loss (Lin et al., 2017) is cross-entropy with two upgrades that directly target our
glioma weakness:
  - a per-class weight (alpha): we set glioma=2.0 so mistakes on glioma cost more, pushing
    the model to stop under-detecting it (its recall was only 0.82 in HXAI-BTC).
  - a focusing term (1 - p_t)^gamma: examples the model already gets right are down-weighted,
    so its effort concentrates on the HARD cases (the borderline glioma/meningioma scans).

The evidential loss (single-pass uncertainty) arrives in Phase 6; only Focal is used now.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        """
        alpha : optional 1-D tensor of per-class weights (length = num_classes), placed on the
                model's device. None -> every class weighted equally.
        gamma : focusing strength. gamma=0 gives plain weighted cross-entropy; higher values
                focus harder on misclassified examples. We use config.FOCAL_GAMMA (2.0).
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        # log-probability of every class, then pull out the TRUE class's log-prob.
        logp = F.log_softmax(logits, dim=1)                    # (B, C)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)  # (B,)  = log p_t
        pt = logpt.exp()                                        # (B,)  = p_t in [0, 1]

        focal = (1.0 - pt) ** self.gamma   # ~0 for easy (p_t≈1), ~1 for hard (p_t≈0)
        loss = -focal * logpt              # base focal loss, before class weighting

        if self.alpha is not None:
            at = self.alpha.gather(0, target)  # (B,) each sample's class weight
            loss = at * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # 'none' -> per-sample losses


if __name__ == "__main__":
    import config
    torch.manual_seed(0)
    C = config.NUM_CLASSES
    alpha = torch.tensor(config.CLASS_WEIGHTS_LIST)
    loss_fn = FocalLoss(alpha=alpha, gamma=config.FOCAL_GAMMA)

    print("Focal loss smoke test")
    print(f"  class weights (alpha): {config.CLASS_WEIGHTS_LIST}")
    print(f"  gamma                : {config.FOCAL_GAMMA}")

    logits = torch.randn(8, C)
    target = torch.randint(0, C, (8,))
    print(f"  loss (random logits) : {loss_fn(logits, target).item():.4f}")

    # Sanity: a confident-CORRECT prediction -> near-zero loss; confident-WRONG -> large.
    good = torch.zeros(1, C); good[0, 1] = 10.0   # very sure it's class 1
    bad  = torch.zeros(1, C); bad[0, 0] = 10.0    # very sure it's class 0
    tgt  = torch.tensor([1])                       # ...truth is class 1
    print(f"  confident-correct    : {loss_fn(good, tgt).item():.5f}  (should be ~0)")
    print(f"  confident-wrong      : {loss_fn(bad, tgt).item():.3f}   (should be large)")
