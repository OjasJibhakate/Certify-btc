"""
xai.py — explainability: Grad-CAM + LIME + SHAP consensus, and Dice validation (Phase 7).

Different explainers see different things, so any single one can mislead. We combine three into
a CONSENSUS heatmap (weights from the plan: Grad-CAM 0.45, LIME 0.30, SHAP 0.25):
  - Grad-CAM : gradient of the class score w.r.t. a deep feature map -> where the CNN 'looked'.
  - LIME     : perturb superpixels, see which ones swing the prediction (model-agnostic).
  - SHAP     : Shapley-value pixel attributions (game-theoretic, gradient-based here).

We also provide dice_score(): in Phase 8, on the cloud, we score the CBAM pseudo-mask against
BraTS ground-truth tumor masks with it (2*overlap / total area).

LIME and SHAP are slow and version-sensitive, so each is wrapped in try/except: if one fails,
the consensus is formed from whatever succeeded, and we report which ran.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import torch
import torch.nn.functional as F

import config
from modules.preprocessing import IMAGENET_MEAN, IMAGENET_STD, denormalize


# --- helpers -----------------------------------------------------------------

def _norm01(a):
    a = np.asarray(a, dtype=np.float32)
    mn, mx = float(a.min()), float(a.max())
    return (a - mn) / (mx - mn) if mx > mn else np.zeros_like(a)


def image01_to_tensor(img01, device):
    """HxWx3 float in [0,1] -> normalized (1,3,H,W) tensor the model expects."""
    x = (img01.astype(np.float32) - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
    return x.to(device)


# --- Grad-CAM ----------------------------------------------------------------

class GradCAM:
    """Grad-CAM on a chosen module's output feature map.

    The forward hook is registered ONLY for the duration of a call and removed afterwards, so
    it can't interfere with later no_grad forward passes (e.g. LIME/SHAP)."""

    def __init__(self, model, target_module):
        self.model = model
        self.target_module = target_module
        self.activation = None

    def _hook(self, module, inp, out):
        # our CBAM returns (refined_features, spatial_map); take the features.
        act = out[0] if isinstance(out, tuple) else out
        if act.requires_grad:              # only when we're actually computing gradients
            act.retain_grad()
        self.activation = act

    def __call__(self, x, class_idx=None):
        handle = self.target_module.register_forward_hook(self._hook)
        try:
            self.model.eval()
            self.model.zero_grad()
            logits = self.model(x)                       # fires the hook
            if class_idx is None:
                class_idx = logits.argmax(1)
            score = logits.gather(1, class_idx.view(-1, 1)).sum()
            score.backward()
            grads = self.activation.grad                 # (B,C,h,w)
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * self.activation).sum(1))   # (B,h,w)
            cam = cam.detach().cpu().numpy()[0]
        finally:
            handle.remove()                              # never leave the hook attached
        return _norm01(cam)


# --- LIME --------------------------------------------------------------------

def lime_heatmap(model, img01, device, num_samples=300):
    from lime import lime_image

    def predict_fn(images):
        # LIME hands us many perturbed HxWx3 [0,1] images; run them in small chunks so a big
        # batch can't blow the 6GB budget.
        out = []
        for i in range(0, len(images), 8):
            chunk = images[i:i + 8]
            xs = torch.cat([image01_to_tensor(im, device) for im in chunk], dim=0)
            with torch.no_grad():
                out.append(F.softmax(model(xs), dim=1).cpu().numpy())
        return np.concatenate(out, axis=0)

    explainer = lime_image.LimeImageExplainer()
    expl = explainer.explain_instance(img01.astype(np.float64), predict_fn,
                                      top_labels=1, hide_color=0, num_samples=num_samples)
    label = expl.top_labels[0]
    # dict_heatmap: superpixel -> weight; paint it back into an image-sized map.
    seg = expl.segments
    weights = dict(expl.local_exp[label])
    heat = np.vectorize(lambda s: weights.get(s, 0.0))(seg)
    return _norm01(np.abs(heat))


# --- SHAP --------------------------------------------------------------------

def shap_heatmap(model, x_tensor, background, class_idx, nsamples=20):
    import shap
    explainer = shap.GradientExplainer(model, background)
    sv = explainer.shap_values(x_tensor, nsamples=nsamples)
    arr = np.asarray(sv)
    # Normalize to (num_classes, 3, H, W) for our single input, robust to SHAP versions.
    if arr.ndim == 5 and arr.shape[-1] == config.NUM_CLASSES:   # (n, 3, H, W, C)
        chosen = arr[0, :, :, :, int(class_idx)]                 # (3,H,W)
    elif arr.ndim == 5:                                          # (C, n, 3, H, W)
        chosen = arr[int(class_idx), 0]                          # (3,H,W)
    else:
        chosen = arr[0] if arr.ndim == 4 else arr
    heat = np.abs(chosen).sum(axis=0)                            # sum over channels -> (H,W)
    return _norm01(heat)


# --- consensus + dice --------------------------------------------------------

def consensus_heatmap(maps, weights, size=None):
    """Weighted blend of available heatmaps (keys map to weights). Missing methods are dropped
    and the remaining weights renormalized."""
    if size is None:
        size = config.IMG_SIZE
    avail = {k: v for k, v in maps.items() if v is not None}
    if not avail:
        return None
    wsum = sum(weights[k] for k in avail)
    out = np.zeros((size, size), dtype=np.float32)
    for k, m in avail.items():
        m = cv2.resize(_norm01(m), (size, size))
        out += (weights[k] / wsum) * m
    return _norm01(out)


def dice_score(pred_mask, gt_mask):
    """Dice = 2|A∩B| / (|A|+|B|). 1.0 = perfect overlap. Both are binary masks."""
    a = np.asarray(pred_mask).astype(bool)
    b = np.asarray(gt_mask).astype(bool)
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else float(2.0 * np.logical_and(a, b).sum() / denom)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from modules.model import CertifyBTC
    from modules.datasets import build_dataloaders
    from train import load_checkpoint

    device = config.DEVICE
    model = CertifyBTC().to(device)
    ck = os.path.join(config.CHECKPOINT_DIR, "stage1_best.pth")
    if os.path.exists(ck):
        load_checkpoint(ck, model, device=device)
    model.eval()   # BatchNorm needs eval mode for single-image (batch=1) inference

    xb, yb = next(iter(build_dataloaders()[2]))     # test loader
    x = xb[:1].to(device)
    img01 = denormalize(xb[0])                        # HxWx3 [0,1]
    pred = int(model(x).argmax(1).item())
    print(f"Explaining one image (true={config.CLASS_NAMES[yb[0]]}, pred={config.CLASS_NAMES[pred]})")

    maps = {"gradcam": None, "lime": None, "shap": None}
    weights = {"gradcam": 0.45, "lime": 0.30, "shap": 0.25}

    # Grad-CAM (always).
    try:
        t = time.time()
        maps["gradcam"] = GradCAM(model, model.cbam["b7"])(x)
        print(f"  Grad-CAM ok ({time.time()-t:.1f}s)")
    except Exception as e:
        print(f"  Grad-CAM FAILED: {e}")

    if device == "cuda":
        torch.cuda.empty_cache()

    # LIME (reduced samples for speed, chunked forwards for memory).
    try:
        t = time.time()
        maps["lime"] = lime_heatmap(model, img01, device, num_samples=150)
        print(f"  LIME ok ({time.time()-t:.1f}s)")
    except Exception as e:
        print(f"  LIME FAILED: {e}")

    if device == "cuda":
        torch.cuda.empty_cache()

    # SHAP (single background image + few samples). Heavy for B4@380 on 6GB — if it OOMs we
    # skip it (cloud-only) and the consensus uses the methods that ran.
    try:
        t = time.time()
        bg = xb[1:2].to(device)
        maps["shap"] = shap_heatmap(model, x, bg, pred, nsamples=20)
        print(f"  SHAP ok ({time.time()-t:.1f}s)")
    except Exception as e:
        print(f"  SHAP FAILED (likely 6GB OOM — runs on cloud): {str(e)[:80]}")

    consensus = consensus_heatmap(maps, weights)

    # dice_score self-check on two synthetic overlapping disks.
    A = np.zeros((100, 100), np.uint8); cv2.circle(A, (50, 50), 30, 1, -1)
    B = np.zeros((100, 100), np.uint8); cv2.circle(B, (60, 50), 30, 1, -1)
    print(f"  dice_score self-check (overlapping disks): {dice_score(A, B):.3f}")

    panels = [("input", img01)] + [(k, maps[k]) for k in ("gradcam", "lime", "shap")] + \
             [("consensus", consensus)]
    fig, ax = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3))
    for a, (name, m) in zip(ax, panels):
        if m is None:
            a.text(0.5, 0.5, f"{name}\n(failed)", ha="center", va="center"); a.axis("off"); continue
        if name == "input":
            a.imshow(m)
        else:
            a.imshow(img01); a.imshow(cv2.resize(m, img01.shape[:2]), cmap="jet", alpha=0.5)
        a.set_title(name); a.axis("off")
    plt.tight_layout()
    out = os.path.join(config.LOG_DIR, "phase7_xai_demo.png")
    plt.savefig(out, dpi=90)
    print(f"Saved -> {out}")
    print("xai OK.")
