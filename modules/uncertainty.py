"""
uncertainty.py — how sure is the model? (Phase 6)

Two complementary ways to measure uncertainty:

1) Evidential Deep Learning (EDL, Sensoy et al. 2018) — SINGLE-PASS uncertainty.
   Instead of softmax, read the network output as 'evidence' e_k >= 0 for each class. That
   defines a Dirichlet distribution with parameters alpha_k = e_k + 1. From it:
       total strength S = sum_k alpha_k
       probability    p_k = alpha_k / S
       uncertainty    u = K / S        (K = #classes)
   Low evidence -> S ≈ K -> u ≈ 1 (the model says "I don't know"); high evidence -> u → 0.
   To make the model produce meaningful evidence you TRAIN with edl_mse_loss (a drop-in swap
   for Focal in the training loop). The functions here compute evidence/uncertainty + the loss.

2) MC-Dropout — keep dropout ON at test time, run N forward passes, and measure how much the
   predictions disagree. Cheap, model-agnostic; used as the comparison baseline in the paper.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# --- Evidential (Dirichlet) --------------------------------------------------

def logits_to_evidence(logits):
    """Map raw logits to non-negative evidence. softplus is smooth and never dies to 0."""
    return F.softplus(logits)


def dirichlet_from_logits(logits):
    """alpha = evidence + 1 (the '+1' is the uniform Dirichlet prior)."""
    return logits_to_evidence(logits) + 1.0


def dirichlet_uncertainty(alpha):
    """Return (probabilities, uncertainty u in [0,1]). u = K / S."""
    S = alpha.sum(dim=1, keepdim=True)
    K = alpha.shape[1]
    probs = alpha / S
    u = (K / S).squeeze(1)
    return probs, u


def kl_dirichlet_to_uniform(alpha):
    """KL( Dir(alpha) || Dir(1) ) per sample — penalizes evidence away from 'uniform'."""
    ones = torch.ones_like(alpha)
    S_alpha = alpha.sum(dim=1, keepdim=True)
    term1 = (torch.lgamma(S_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
             + torch.lgamma(ones).sum(dim=1, keepdim=True)
             - torch.lgamma(ones.sum(dim=1, keepdim=True)))
    term2 = ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(S_alpha))).sum(dim=1, keepdim=True)
    return (term1 + term2).squeeze(1)


def edl_mse_loss(logits, targets, epoch=1, total_epochs=10):
    """Evidential loss (Sensoy MSE/Bayes-risk form) + annealed KL on misleading evidence."""
    K = logits.shape[1]
    alpha = dirichlet_from_logits(logits)
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    y = F.one_hot(targets, K).float()

    err = ((y - p) ** 2).sum(dim=1, keepdim=True)          # fit term
    var = (p * (1 - p) / (S + 1)).sum(dim=1, keepdim=True)  # variance term
    # Remove the true class's evidence before penalizing, so only WRONG evidence is punished.
    alpha_tilde = y + (1 - y) * alpha
    annealing = min(1.0, epoch / max(total_epochs, 1))      # ramp KL weight 0 -> 1
    kl = kl_dirichlet_to_uniform(alpha_tilde).unsqueeze(1)
    return (err + var + annealing * kl).mean()


# --- MC-Dropout --------------------------------------------------------------

def enable_dropout(model):
    """Put ONLY the dropout layers back into train mode (rest stays eval)."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_predict(model, x, n_samples=30):
    """Run N stochastic forward passes. Returns (mean prob, predictive entropy, mutual info).
    Predictive entropy = total uncertainty; mutual info = epistemic (model) uncertainty."""
    model.eval()
    enable_dropout(model)
    probs = torch.stack([F.softmax(model(x), dim=1) for _ in range(n_samples)])  # (N,B,K)
    mean_p = probs.mean(0)                                        # (B,K)
    entropy = -(mean_p * (mean_p + 1e-12).log()).sum(1)          # total uncertainty
    exp_entropy = -(probs * (probs + 1e-12).log()).sum(2).mean(0)
    mutual_info = entropy - exp_entropy                          # epistemic uncertainty
    return mean_p, entropy, mutual_info


if __name__ == "__main__":
    torch.manual_seed(0)

    # 1) Dirichlet uncertainty: confident logits -> low u; near-zero logits -> high u.
    for name, lg in [("confident", torch.tensor([[12.0, 0.0, 0.0, 0.0]])),
                     ("unsure",    torch.tensor([[0.3, 0.2, 0.25, 0.1]]))]:
        alpha = dirichlet_from_logits(lg)
        p, u = dirichlet_uncertainty(alpha)
        print(f"  {name:9s} -> p_max {p.max().item():.3f}  uncertainty {u.item():.3f}")

    # 2) EDL loss computes on a batch.
    logits = torch.randn(8, config.NUM_CLASSES)
    tgt = torch.randint(0, config.NUM_CLASSES, (8,))
    print(f"  edl_mse_loss: {edl_mse_loss(logits, tgt, epoch=5, total_epochs=10).item():.4f}")

    # 3) MC-Dropout on the real trained model.
    from modules.model import CertifyBTC
    from modules.datasets import build_dataloaders
    from train import load_checkpoint
    device = config.DEVICE
    model = CertifyBTC().to(device)
    ck = os.path.join(config.CHECKPOINT_DIR, "stage1_best.pth")
    if os.path.exists(ck):
        load_checkpoint(ck, model, device=device)
    xb, _ = next(iter(build_dataloaders()[2]))       # test loader
    mean_p, ent, mi = mc_dropout_predict(model, xb[:4].to(device), n_samples=30)
    print("  MC-dropout entropy (4)     :", [round(e, 3) for e in ent.tolist()])
    print("  MC-dropout mutual-info (4) :", [round(m, 3) for m in mi.tolist()])
    print("uncertainty OK.")
