"""
analyze.py — full evaluation of a trained CERTIFY-BTC model on the test set.

Prints, for checkpoints/stage2_best.pth:
  - test accuracy + per-class recall, confusion matrix
  - certification: uncertainty separation on missed gliomas, conformal glioma coverage,
    Mahalanobis OOD (AUROC vs noise)
  - the glioma operating-point sweep (trade precision for sensitivity)

Run after training:   CERTIFY_MACHINE=local_full python analyze.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CERTIFY_MACHINE", "local_full")   # evaluate on the full test set

import numpy as np
import torch
import torch.nn.functional as F

import config
from modules.model import CertifyBTC
from modules.datasets import build_dataloaders
from train import load_checkpoint
from modules.conformal import ConformalRAPS, coverage_and_size
from modules.ood import MahalanobisOOD
from sklearn.metrics import confusion_matrix, roc_auc_score

device = config.DEVICE
model = CertifyBTC().to(device)
model.eval()
load_checkpoint(os.path.join(config.CHECKPOINT_DIR, "stage2_best.pth"), model, device=device)
_, val_loader, test_loader = build_dataloaders()
CN = config.CLASS_NAMES
G = CN.index("glioma")


@torch.no_grad()
def collect(loader):
    L, P, Fe, Y = [], [], [], []
    for x, y in loader:
        lg, fu, _ = model(x.to(device), return_maps=True)
        L.append(lg.float().cpu().numpy()); P.append(F.softmax(lg, 1).float().cpu().numpy())
        Fe.append(fu.float().cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(L), np.concatenate(P), np.concatenate(Fe), np.concatenate(Y)


val_logits, val_p, val_f, val_y = collect(val_loader)
test_logits, test_p, test_f, test_y = collect(test_loader)
test_pred = test_p.argmax(1)
gl = test_y == G
missed = gl & (test_pred != G)
correct = gl & (test_pred == G)

print("=" * 60)
print(f"TEST accuracy: {(test_pred==test_y).mean():.4f}   (n={len(test_y)})")
from sklearn.metrics import precision_recall_fscore_support
prec, rec, _, _ = precision_recall_fscore_support(test_y, test_pred, labels=range(config.NUM_CLASSES))
for i, c in enumerate(CN):
    print(f"  {c:<11} recall {rec[i]:.3f}  precision {prec[i]:.3f}")
print("confusion (rows=true, cols=pred):")
print(confusion_matrix(test_y, test_pred))

print("\n--- Certification ---")
ent = -(test_p * np.log(test_p + 1e-12)).sum(1)
print(f"A1 uncertainty: correct-glioma {ent[correct].mean():.3f} vs "
      f"MISSED-glioma {ent[missed].mean():.3f} ({ent[missed].mean()/max(ent[correct].mean(),1e-6):.1f}x)")

cp = ConformalRAPS(coverage=0.95); cp.calibrate(val_p, val_y)
sets = cp.predict(test_p)
cov, size = coverage_and_size(sets, test_y)
gl_cov = np.mean([G in sets[i] for i in np.where(gl)[0]])
miss_cov = np.mean([G in sets[i] for i in np.where(missed)[0]]) if missed.sum() else float("nan")
print(f"A2 conformal: overall cov {cov:.3f} | glioma set-cov {gl_cov:.3f} | "
      f"missed-glioma recovered {miss_cov:.1%}")

maha = MahalanobisOOD().fit(val_f, val_y)
noise_f = []
with torch.no_grad():
    for _ in range(60):
        _, fu, _ = model(torch.randn(16, 3, config.IMG_SIZE, config.IMG_SIZE, device=device), return_maps=True)
        noise_f.append(fu.float().cpu().numpy())
noise_f = np.concatenate(noise_f)[:len(test_f)]
id_s, ood_s = maha.score(test_f), maha.score(noise_f)
auroc = roc_auc_score([0]*len(id_s) + [1]*len(ood_s), np.concatenate([id_s, ood_s]))
print(f"A3 OOD (Mahalanobis vs noise): AUROC {auroc:.3f}")

print("\n--- Glioma operating-point sweep (bias added to glioma logit) ---")
print(f"  {'bias':>5} {'gl_recall':>10} {'gl_prec':>8} {'overall_acc':>12}")
for b in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    adj = test_logits.copy(); adj[:, G] += b
    pr = adj.argmax(1)
    r = (pr[gl] == G).mean()
    p = (test_y[pr == G] == G).mean() if (pr == G).any() else 0.0
    a = (pr == test_y).mean()
    print(f"  {b:>5.1f} {r:>10.3f} {p:>8.3f} {a:>12.3f}")
print("=" * 60)
