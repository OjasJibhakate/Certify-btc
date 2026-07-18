"""
radiomics_fusion.py — radiomic feature extraction + fusion (Phase 4).

Deep features tell you 'what the tumor looks like'; radiomic features quantify the tumor
region as NUMBERS — its size, shape, intensity statistics, and texture. Fusing both gives the
classifier hand-crafted measurements it can't easily read off raw pixels.

PyRadiomics note: the standard 107-feature IBSI library does NOT install on this Windows +
NumPy-2.x setup (its 3.1.0 release is broken and it predates NumPy 2). So this module uses a
dependency-light extractor on scikit-image/scipy that runs everywhere and produces the SAME
categories of features (first-order intensity + 2-D shape + GLCM texture, ~40 features). On a
Linux cloud box with numpy<2 you can later swap in real pyradiomics behind extract_features().

Pipeline: (image, pseudo-mask) -> ~40 features -> standardize -> PCA(32) -> concat with the
512-d deep feature -> 544-d fused vector for the final classifier.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import skew, kurtosis
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config

# ----------------------------------------------------------------------------
# Feature groups. Each returns (names, values) so the vector is self-describing.
# ----------------------------------------------------------------------------

_FIRST_ORDER_NAMES = [
    "fo_mean", "fo_std", "fo_min", "fo_max", "fo_median", "fo_range",
    "fo_p10", "fo_p25", "fo_p75", "fo_p90", "fo_mad", "fo_rms",
    "fo_skew", "fo_kurtosis", "fo_energy", "fo_entropy", "fo_uniformity",
]

def first_order_features(pixels):
    """Intensity statistics over the pixels INSIDE the mask."""
    if pixels.size == 0:
        return _FIRST_ORDER_NAMES, [0.0] * len(_FIRST_ORDER_NAMES)
    p = pixels.astype(np.float64)
    hist, _ = np.histogram(p, bins=32, range=(0, 255), density=True)
    hist = hist + 1e-12                       # avoid log(0)
    prob = hist / hist.sum()
    vals = [
        p.mean(), p.std(), p.min(), p.max(), np.median(p), p.max() - p.min(),
        np.percentile(p, 10), np.percentile(p, 25), np.percentile(p, 75), np.percentile(p, 90),
        np.mean(np.abs(p - p.mean())),                 # mean absolute deviation
        np.sqrt(np.mean(p ** 2)),                       # root mean square
        float(skew(p)) if p.std() > 0 else 0.0,
        float(kurtosis(p)) if p.std() > 0 else 0.0,
        np.sum((p / 255.0) ** 2),                       # energy
        float(-np.sum(prob * np.log2(prob))),           # entropy
        float(np.sum(prob ** 2)),                       # uniformity
    ]
    return _FIRST_ORDER_NAMES, [float(v) for v in vals]


_SHAPE_NAMES = [
    "sh_area", "sh_perimeter", "sh_eccentricity", "sh_solidity", "sh_extent",
    "sh_equiv_diam", "sh_major_axis", "sh_minor_axis", "sh_orientation", "sh_compactness",
]

def shape_features(mask):
    """2-D shape descriptors of the mask's largest region (via skimage regionprops)."""
    lbl = label(mask.astype(np.uint8))
    props = regionprops(lbl)
    if not props:
        return _SHAPE_NAMES, [0.0] * len(_SHAPE_NAMES)
    r = max(props, key=lambda x: x.area)               # largest region
    perim = r.perimeter if r.perimeter > 0 else 1.0
    compactness = (perim ** 2) / (4.0 * np.pi * r.area) if r.area > 0 else 0.0  # 1.0 = circle
    vals = [
        r.area, r.perimeter, r.eccentricity, r.solidity, r.extent,
        r.equivalent_diameter_area, r.axis_major_length, r.axis_minor_length,
        r.orientation, compactness,
    ]
    return _SHAPE_NAMES, [float(v) for v in vals]


_GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]
_GLCM_DISTANCES = [1, 3]
_GLCM_NAMES = [f"tx_{p}_d{d}" for d in _GLCM_DISTANCES for p in _GLCM_PROPS]

def glcm_features(gray, mask, levels=32):
    """Gray-Level Co-occurrence Matrix texture on the tumor's bounding box.

    GLCM counts how often intensity pairs occur next to each other -> texture. We average
    over 4 directions (0/45/90/135 deg) at 2 distances, then read 6 standard properties.
    (Approximate: computed on the bbox crop; a small amount of background is included.)"""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return _GLCM_NAMES, [0.0] * len(_GLCM_NAMES)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = gray[y0:y1, x0:x1]
    q = (crop.astype(np.float32) / 256.0 * levels).astype(np.uint8)   # quantize to `levels`
    q = np.clip(q, 0, levels - 1)
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(q, distances=_GLCM_DISTANCES, angles=angles,
                        levels=levels, symmetric=True, normed=True)
    vals = []
    for di in range(len(_GLCM_DISTANCES)):
        for prop in _GLCM_PROPS:
            # graycoprops returns (n_distances, n_angles); average over the 4 angles.
            vals.append(float(graycoprops(glcm, prop)[di].mean()))
    return _GLCM_NAMES, vals


def extract_features(gray, mask):
    """Full radiomic vector for one (grayscale image, binary mask). Returns (names, values)."""
    pixels = gray[mask > 0]
    names, values = [], []
    for fn_names, fn_vals in (first_order_features(pixels),
                              shape_features(mask),
                              glcm_features(gray, mask)):
        names += fn_names
        values += fn_vals
    return names, np.asarray(values, dtype=np.float32)


# ----------------------------------------------------------------------------
# Dimensionality reduction + fusion
# ----------------------------------------------------------------------------

class RadiomicReducer:
    """Standardize the ~40 features, then PCA down to n_components (32 in the plan).

    We standardize first because the features live on wildly different scales (area in the
    thousands, eccentricity in [0,1]); without it, PCA would be dominated by the big-number
    features. Fit on the TRAIN set only, then transform train/val/test."""

    def __init__(self, n_components=32):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = None

    def fit(self, X):
        Xs = self.scaler.fit_transform(X)
        k = min(self.n_components, Xs.shape[0], Xs.shape[1])
        if k < self.n_components:
            print(f"[RadiomicReducer] only {k} components possible "
                  f"(need >= {self.n_components} samples & features); using {k}.")
        self.pca = PCA(n_components=k)
        self.pca.fit(Xs)
        return self

    def transform(self, X):
        return self.pca.transform(self.scaler.transform(X)).astype(np.float32)

    @property
    def explained(self):
        return float(self.pca.explained_variance_ratio_.sum()) if self.pca else 0.0


def fuse(deep_features, radiomic_features):
    """Concatenate deep (B, 512) and radiomic (B, k) into (B, 512+k). Accepts numpy arrays."""
    deep = np.asarray(deep_features, dtype=np.float32)
    rad = np.asarray(radiomic_features, dtype=np.float32)
    return np.concatenate([deep, rad], axis=1)


if __name__ == "__main__":
    # End-to-end demo: pull N images, build pseudo-masks, extract features, PCA(32), fuse
    # with the model's 512-d deep feature -> 544-d vector.
    import torch
    from modules.model import CertifyBTC
    from modules.datasets import build_dataloaders
    from modules.preprocessing import denormalize
    from modules.cbam_mask import masks_from_model_maps
    from train import load_checkpoint

    device = config.DEVICE
    model = CertifyBTC().to(device)
    ckpt = os.path.join(config.CHECKPOINT_DIR, "stage1_best.pth")
    if os.path.exists(ckpt):
        load_checkpoint(ckpt, model, device=device)
    model.eval()

    train_loader, _, _ = build_dataloaders()

    feat_matrix, deep_matrix, feat_names = [], [], None
    collected = 0
    target = 48                                     # enough samples for a 32-dim PCA
    with torch.no_grad():
        for xb, _yb in train_loader:
            xb = xb.to(device)
            logits, fused, maps = model(xb, return_maps=True)
            grays = [(denormalize(xb[i])[:, :, 0] * 255).astype(np.uint8) for i in range(xb.shape[0])]
            _heats, masks = masks_from_model_maps(maps, out_size=config.IMG_SIZE, grays=grays)
            for i in range(xb.shape[0]):
                gray = grays[i]
                names, values = extract_features(gray, masks[i])
                feat_names = names
                feat_matrix.append(values)
                deep_matrix.append(fused[i].cpu().numpy())
                collected += 1
            if collected >= target:
                break

    X = np.stack(feat_matrix)          # (N, ~40)
    D = np.stack(deep_matrix)          # (N, 512)
    print("=" * 56)
    print("  radiomics_fusion demo")
    print("=" * 56)
    print(f"  samples             : {X.shape[0]}")
    print(f"  radiomic features   : {X.shape[1]}  ({len(feat_names)} named)")
    print(f"  example names       : {feat_names[:4]} ... {feat_names[-3:]}")

    reducer = RadiomicReducer(n_components=32).fit(X)
    R = reducer.transform(X)           # (N, 32)
    print(f"  after PCA           : {R.shape}  | variance kept = {reducer.explained:.2%}")

    F = fuse(D, R)                     # (N, 544)
    print(f"  deep features       : {D.shape}")
    print(f"  fused (deep+radiom) : {F.shape}  (expect (*, 544))")
    print("radiomics_fusion OK.")
