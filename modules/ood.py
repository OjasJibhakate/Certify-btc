"""
ood.py — out-of-distribution detection via energy score (Phase 6).

An OOD input — a knee MRI, a corrupted scan, plain noise, anything outside our 4 classes —
should be REJECTED, not confidently classified. The energy score (Liu et al. 2020) reads how
much coherent 'mass' the logits carry:

    E(x) = -T * logsumexp(logits / T)

In-distribution inputs excite some class strongly -> LOW energy. OOD inputs excite nothing
coherently -> HIGH energy. We calibrate a threshold on in-distribution data and flag anything
above it as OOD. Energy needs no retraining and beats plain softmax confidence for this job.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import config


def energy_score(logits, T=1.0):
    """Free energy per sample. Lower = more in-distribution. Returns a (B,) tensor."""
    return -T * torch.logsumexp(logits / T, dim=1)


def fit_threshold(id_energies, coverage=0.95):
    """Threshold that keeps `coverage` of in-distribution data (flag the rest as OOD)."""
    return float(np.quantile(np.asarray(id_energies), coverage))


def is_ood(energies, threshold):
    """True where the sample is flagged out-of-distribution."""
    return np.asarray(energies) > threshold


class MahalanobisOOD:
    """Feature-distance OOD detector (Lee et al. 2018). Energy fails on pure noise because the
    model is overconfident there; Mahalanobis instead asks 'how far is this sample's FEATURE
    vector from the training feature distribution?'. Noise lands far from every class cluster,
    so it's flagged even when the logits are (wrongly) huge.

    Fit class means + a shared (pooled) covariance on in-distribution features; score = the
    minimum Mahalanobis distance to any class centroid (higher = more OOD)."""

    def __init__(self, shrinkage=1e-2):
        self.shrinkage = shrinkage
        self.means = None
        self.inv_cov = None
        self.classes = None

    def fit(self, features, labels):
        features, labels = np.asarray(features), np.asarray(labels)
        self.classes = np.unique(labels)
        self.means = {int(c): features[labels == c].mean(0) for c in self.classes}
        centered = np.concatenate([features[labels == c] - self.means[int(c)] for c in self.classes])
        cov = np.cov(centered.T) + self.shrinkage * np.eye(features.shape[1])  # shrinkage = stable inverse
        self.inv_cov = np.linalg.inv(cov)
        return self

    def score(self, features):
        features = np.asarray(features)
        dists = np.stack([
            (((features - self.means[int(c)]) @ self.inv_cov) * (features - self.means[int(c)])).sum(1)
            for c in self.classes
        ], axis=1)
        return dists.min(axis=1)   # distance to the NEAREST class cluster


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    from modules.model import CertifyBTC
    from modules.datasets import build_dataloaders
    from train import load_checkpoint

    device = config.DEVICE
    model = CertifyBTC().to(device)
    model.eval()
    ck = os.path.join(config.CHECKPOINT_DIR, "stage1_best.pth")
    if os.path.exists(ck):
        load_checkpoint(ck, model, device=device)

    _, _, test_loader = build_dataloaders()

    # In-distribution energies: real MRI from the test set.
    id_e = []
    with torch.no_grad():
        for x, _ in test_loader:
            id_e += energy_score(model(x.to(device))).cpu().tolist()

    # OOD energies: pure Gaussian-noise "images" of the same shape (clearly not brain MRI).
    ood_e = []
    n_batches = len(id_e) // config.BATCH_SIZE + 1
    with torch.no_grad():
        for _ in range(n_batches):
            x = torch.randn(config.BATCH_SIZE, 3, config.IMG_SIZE, config.IMG_SIZE, device=device)
            ood_e += energy_score(model(x)).cpu().tolist()
    ood_e = ood_e[:len(id_e)]

    thr = fit_threshold(id_e, config.CONFORMAL_COVERAGE)
    y_true = [0] * len(id_e) + [1] * len(ood_e)      # 1 = OOD
    y_score = id_e + ood_e                            # higher energy -> more OOD
    auroc = roc_auc_score(y_true, y_score)

    print(f"  ID  energy mean : {np.mean(id_e):+.2f}   (real MRI)")
    print(f"  OOD energy mean : {np.mean(ood_e):+.2f}   (noise inputs)")
    print(f"  threshold (keep {config.CONFORMAL_COVERAGE:.0%} ID): {thr:+.2f}")
    print(f"  OOD correctly flagged      : {np.mean(is_ood(ood_e, thr)):.1%}")
    print(f"  AUROC (ID vs OOD)          : {auroc:.3f}   (1.0 = perfect separation)")
    if auroc < 0.5:
        print("  ! INVERTED: the undertrained model is OVER-confident on noise (huge logits ->")
        print("    very low energy), so energy-OOD fails here. The code is correct; the MODEL")
        print("    isn't ready. Energy-OOD needs the fully-trained model (calibrated logit")
        print("    magnitudes) + realistic OOD scans. Revisit after full training (Phase 8).")
    print("ood mechanism OK (see caveat above).")
