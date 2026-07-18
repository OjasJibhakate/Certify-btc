"""
domain_adversarial.py — Gradient Reversal Layer + domain classifier (Phase 5).

Goal: make the 512-d fused feature DATASET-INVARIANT, so the tumor classifier behaves the
same on a scan from any hospital/scanner (multi-site robustness).

How (Ganin & Lempitsky, 2015 — "DANN"): attach a second head that tries to predict WHICH
dataset a sample came from (its 'domain'). Between the features and this domain head sits a
Gradient Reversal Layer (GRL): in the forward pass it does nothing, but in the backward pass
it MULTIPLIES the gradient by -alpha. Effect: the domain head learns to tell datasets apart,
while the backbone is pushed the OPPOSITE way — to ERASE dataset-identifying information.
The tug-of-war leaves only tumor-relevant, site-invariant features.

alpha ramps 0 -> 1 over training so the adversarial pressure turns on gradually (full pressure
from step 1 destabilizes training).

NOTE: this needs >= 2 datasets to do anything real. With Nickparvar alone it's a mechanism
check; the actual domain labels come from combining Figshare/BraTS/TCGA on the cloud.
"""

import math

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradientReversal(Function):
    """Identity forward; negated-and-scaled gradient backward."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)               # identity

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient. The second None is because `alpha` gets no gradient.
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    """Functional wrapper for the Gradient Reversal Layer."""
    return _GradientReversal.apply(x, alpha)


class DomainClassifier(nn.Module):
    """Small MLP that predicts the domain (dataset) id from the fused feature, THROUGH a GRL.
    Because of the GRL, training it also teaches the backbone to hide domain cues."""

    def __init__(self, in_dim=512, num_domains=2, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_domains),
        )

    def forward(self, feat, alpha=1.0):
        return self.net(grad_reverse(feat, alpha))


def grl_alpha(step, total_steps):
    """DANN schedule: smoothly ramp alpha 0 -> 1 as training progresses."""
    p = step / max(total_steps, 1)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


if __name__ == "__main__":
    # 1. GRL mechanism: forward is identity, backward negates & scales the gradient.
    x = torch.randn(3, 4, requires_grad=True)
    out = grad_reverse(x, alpha=2.0)
    print("GRL forward is identity :", torch.allclose(out, x))
    out.sum().backward()
    # d(sum)/dx would normally be +1 everywhere; GRL with alpha=2 makes it -2.
    print("GRL backward grad       :", x.grad[0].tolist(), " (expect all -2.0)")

    # 2. Domain classifier output shape.
    feat = torch.randn(8, 512)
    dc = DomainClassifier(in_dim=512, num_domains=2)
    dlog = dc(feat, alpha=1.0)
    print("domain logits shape     :", tuple(dlog.shape), " (expect (8, 2))")

    # 3. Alpha ramp across training.
    print("alpha ramp 0->1         :", [round(grl_alpha(s, 10), 2) for s in range(11)])
    print("domain_adversarial OK.")
