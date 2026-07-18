"""
cbam_mask.py — turn CBAM spatial attention into a pseudo-segmentation mask (Phase 4).

Nickparvar has no ground-truth tumor outlines. But the CBAM spatial attention map already
highlights WHERE the model looks. This module converts that soft heat map into a clean binary
tumor mask with classic image processing — no extra labels, no training:

    attention -> upsample -> Gaussian blur -> Otsu threshold -> CLOSE then OPEN -> largest blob

Why each step:
- upsample : the deep attention map is tiny (e.g. 12x12); blow it up to image size.
- blur     : smooth the blocky upsampling so the threshold gives a clean boundary.
- Otsu     : auto-pick the foreground/background cutoff — no magic threshold to tune.
- close->open (morphology): CLOSE fills small holes inside the blob; OPEN removes stray
             specks. Order matters: fill first, then denoise.
- largest connected component: a tumor is one region, so keep only the biggest blob.

In Phase 7 we score these masks against BraTS ground truth using the Dice metric.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes

import config


def to_numpy(attn):
    """Accept a torch tensor or ndarray of shape (H,W)/(1,H,W)/(B,1,H,W)[single] -> (H,W)."""
    if hasattr(attn, "detach"):
        attn = attn.detach().cpu().float().numpy()
    attn = np.asarray(attn, dtype=np.float32)
    attn = np.squeeze(attn)
    if attn.ndim != 2:
        raise ValueError(f"Expected a 2-D attention map after squeeze, got shape {attn.shape}")
    return attn


def upsample_map(attn, size):
    """Resize a (small) attention map to size x size (bilinear) and rescale to [0,1]."""
    a = cv2.resize(to_numpy(attn), (size, size), interpolation=cv2.INTER_LINEAR)
    mn, mx = float(a.min()), float(a.max())
    return (a - mn) / (mx - mn) if mx > mn else np.zeros_like(a)


def combine_maps(maps, size):
    """Average several attention maps (from different scales) after upsampling to a common
    size. Fusing scales gives a more robust localizer than any single map."""
    ups = [upsample_map(m, size) for m in maps]
    a = np.mean(ups, axis=0)
    mn, mx = float(a.min()), float(a.max())
    return (a - mn) / (mx - mn) if mx > mn else a


def keep_largest_cc(binary):
    """Keep only the largest connected white blob (a tumor is a single region)."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if num <= 1:                       # only background
        return binary.astype(np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]   # skip label 0 (background)
    largest = 1 + int(np.argmax(areas))
    return (labels == largest).astype(np.uint8)


def brain_region(gray):
    """Rough brain/foreground mask = the non-black part of the scan.

    A tumor lives inside the brain, so restricting the attention mask to here removes the
    border/background artifacts that weak attention loves to fire on. Otsu separates the dark
    background from the bright brain; we then close, fill holes, and keep the largest blob."""
    g = np.asarray(gray, dtype=np.uint8)
    try:
        t = threshold_otsu(g)
    except ValueError:
        t = 20
    fg = (g > max(int(t), 10)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    fg = keep_largest_cc(fg)
    fg = binary_fill_holes(fg).astype(np.uint8)   # a brain is solid, no holes
    return fg


def make_pseudo_mask(attn, out_size=None, blur_ksize=15, close_ks=7, open_ks=5, brain_mask=None):
    """Full attention -> binary mask pipeline. Returns a uint8 array with values {0,1}.

    If brain_mask is given, the Otsu cutoff is computed from IN-BRAIN attention only and the
    result is intersected with the brain — so the mask can only land on brain tissue."""
    if out_size is None:
        out_size = config.IMG_SIZE

    heat = upsample_map(attn, out_size)                 # [0,1]
    heat_u8 = (heat * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(heat_u8, (blur_ksize, blur_ksize), 0)

    # Choose the threshold. With a brain mask, use only in-brain pixels so the background
    # (all zeros) can't skew Otsu into selecting the whole brain.
    try:
        if brain_mask is not None and brain_mask.any():
            t = threshold_otsu(blur[brain_mask > 0])
        else:
            t = threshold_otsu(blur)
    except ValueError:
        t = 127
    binary = (blur > t).astype(np.uint8)
    if brain_mask is not None:
        binary = (binary & (brain_mask > 0)).astype(np.uint8)

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ks, open_ks))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k)

    return keep_largest_cc(binary)


def masks_from_model_maps(maps_dict, out_size=None, grays=None):
    """Given the model's returned attention dict {name: (B,1,H,W)}, produce per-sample
    (combined_heatmap, binary_mask) for the whole batch.

    grays : optional list of per-sample grayscale uint8 images (size out_size). If given, a
            brain mask is computed per image and the tumor mask is confined to the brain.
    Returns two lists of length B."""
    if out_size is None:
        out_size = config.IMG_SIZE
    names = list(maps_dict.keys())
    batch = maps_dict[names[0]].shape[0]
    heats, masks = [], []
    for i in range(batch):
        per_scale = [maps_dict[k][i] for k in names]     # list of (1,H,W)
        heat = combine_maps(per_scale, out_size)
        bmask = brain_region(grays[i]) if grays is not None else None
        heats.append(heat)
        masks.append(make_pseudo_mask(heat, out_size, brain_mask=bmask))
    return heats, masks


if __name__ == "__main__":
    # Demo: run one batch through the (Phase-3 trained) model, build pseudo-masks, and save a
    # picture so you can SEE the mask land on the tumor.
    import torch
    import matplotlib
    matplotlib.use("Agg")               # headless: save to file, no popup window
    import matplotlib.pyplot as plt

    from modules.model import CertifyBTC
    from modules.datasets import build_dataloaders
    from modules.preprocessing import denormalize
    from train import load_checkpoint

    device = config.DEVICE
    model = CertifyBTC().to(device)

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "stage1_best.pth")
    if os.path.exists(ckpt_path):
        load_checkpoint(ckpt_path, model, device=device)
        print(f"Loaded trained weights: {ckpt_path}")
    else:
        print("No checkpoint found — using untrained attention (masks will look rough).")

    train_loader, _, _ = build_dataloaders()
    xb, yb = next(iter(train_loader))
    xb = xb.to(device)

    model.eval()
    with torch.no_grad():
        _, _, maps = model(xb, return_maps=True)

    # Per-image grayscale (0..255) so the mask can be confined to the brain.
    grays = [(denormalize(xb[i])[:, :, 0] * 255).astype("uint8") for i in range(xb.shape[0])]
    heats, masks = masks_from_model_maps(maps, out_size=config.IMG_SIZE, grays=grays)

    n = min(4, xb.shape[0])
    fig, ax = plt.subplots(3, n, figsize=(3 * n, 9))
    for i in range(n):
        img = denormalize(xb[i])                    # HxWx3 in [0,1]
        ax[0, i].imshow(img); ax[0, i].set_title(config.CLASS_NAMES[yb[i].item()]); ax[0, i].axis("off")
        ax[1, i].imshow(heats[i], cmap="jet"); ax[1, i].set_title("attention"); ax[1, i].axis("off")
        ax[2, i].imshow(img); ax[2, i].imshow(masks[i], cmap="Reds", alpha=0.4)
        cov = 100.0 * masks[i].mean()
        ax[2, i].set_title(f"mask ({cov:.1f}%)"); ax[2, i].axis("off")
    ax[0, 0].set_ylabel("input");  ax[1, 0].set_ylabel("attention"); ax[2, 0].set_ylabel("overlay")
    plt.tight_layout()

    out = os.path.join(config.LOG_DIR, "phase4_mask_demo.png")
    plt.savefig(out, dpi=90)
    print(f"Saved demo figure -> {out}")
    print("Mask coverage per image (% of frame):",
          [round(100.0 * float(m.mean()), 1) for m in masks])
    print("cbam_mask OK.")
