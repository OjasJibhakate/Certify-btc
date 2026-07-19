"""
preprocessing.py — image preprocessing for CERTIFY-BTC (Phase 1).

For the 2D Nickparvar JPGs the pipeline is:
    grayscale read -> CLAHE (optional) -> 3-channel -> resize -> ImageNet-normalize -> tensor

WHY each step:
- grayscale: an MRI is intrinsically single-channel intensity. Reading as grayscale
  drops any accidental color and gives one clean channel to enhance.
- CLAHE: scanners differ in brightness/contrast. CLAHE equalizes LOCAL contrast so
  tumor boundaries stand out, without blowing out the whole image the way global
  histogram equalization would.
- 3-channel: EfficientNetB4 was pretrained on RGB ImageNet, so its first conv layer
  expects 3 input channels. We replicate the single MRI channel three times.
- ImageNet-normalize: shift/scale to the mean/std the backbone was trained on, so the
  pretrained weights see inputs in the distribution they expect.

NOTE on skull-strip / N4 (config.PREPROCESS): those operate on 3D volumes (BraTS/TCGA)
and need HD-BET / SimpleITK. The Nickparvar files are already 2D slices, so we skip
them here. They get wired in later, cloud-only, for the volumetric datasets.
"""

import os
import sys

# Make the project root importable so `import config` works whether this file is run
# as `python -m modules.preprocessing`, `python modules/preprocessing.py`, or imported
# from a notebook. We add the parent of this file's folder (= project root) to sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch

import config

# ImageNet statistics EfficientNet was pretrained with (RGB order). Because our three
# channels are identical (grayscale replicated), this just rescales intensities.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def apply_clahe(gray, clip_limit=2.0, tile_grid=8):
    """Contrast-Limited Adaptive Histogram Equalization on a single-channel image.

    clip_limit caps how much any local histogram bin can be amplified (prevents noise
    blow-up); tile_grid splits the image into that many regions for the 'adaptive'
    part. 8x8 tiles with clip 2.0 is a gentle, widely-used default.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(gray)


def _augment_geometry(gray, rng):
    """Geometry augmentation (applied BEFORE CLAHE) — flip, rotate, scale, occasional blur.

    Wider than the Phase-1 version: it exposes the model to more tumor poses and scales, which
    directly targets the train->test generalization gap (glioma especially). Still label-
    preserving — axial brain MRI is roughly L-R symmetric and a small tilt/zoom is realistic.
    """
    h, w = gray.shape
    if rng.random() < 0.5:
        gray = cv2.flip(gray, 1)                        # horizontal flip
    angle = rng.uniform(-15, 15)                        # head tilt (was ±10)
    scale = rng.uniform(0.9, 1.1)                       # zoom in/out a little
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    gray = cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.3:                              # mimic acquisition softness
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def _augment_intensity(gray, rng):
    """Intensity augmentation (applied AFTER CLAHE, so CLAHE can't equalize it away) —
    brightness / contrast / gamma jitter to mimic scanner-to-scanner intensity differences,
    a likely driver of the glioma train->test shift."""
    g = gray.astype(np.float32)
    g = g * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)     # contrast * x + brightness
    gamma = rng.uniform(0.8, 1.25)                            # non-linear tone shift
    g = 255.0 * np.clip(g / 255.0, 0.0, 1.0) ** gamma
    return np.clip(g, 0, 255).astype(np.uint8)


def preprocess_image(path, img_size=None, use_clahe=None, augment=False, rng=None):
    """Load one image and return a normalized CHW float tensor of shape (3, S, S)."""
    if img_size is None:
        img_size = config.IMG_SIZE
    if use_clahe is None:
        use_clahe = config.PREPROCESS["clahe"]

    # 1. Read as single-channel grayscale.
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    # 2. Optional GEOMETRY augmentation (before CLAHE).
    if augment:
        rng = rng if rng is not None else np.random.default_rng()
        gray = _augment_geometry(gray, rng)

    # 3. Local contrast enhancement.
    if use_clahe:
        gray = apply_clahe(gray)

    # 3b. Optional INTENSITY augmentation (after CLAHE, so it isn't equalized away).
    if augment:
        gray = _augment_intensity(gray, rng)

    # 4. Resize. INTER_AREA is the best resampler when SHRINKING an image.
    gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # 5. Grayscale -> 3 identical channels (what the pretrained backbone expects).
    rgb = np.stack([gray, gray, gray], axis=-1)  # H x W x 3

    # 6. Scale to [0,1], then apply ImageNet normalization.
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD

    # 7. HWC -> CHW and to tensor (PyTorch wants channels first).
    x = torch.from_numpy(x).permute(2, 0, 1).contiguous().float()
    return x


def denormalize(x):
    """Undo ImageNet normalization for DISPLAY only. Returns an HxWx3 float array in
    [0,1] — handy in the notebook to show what the model actually 'sees'."""
    x = x.detach().cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    x = x * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(x, 0.0, 1.0)


if __name__ == "__main__":
    # Standalone smoke test: preprocess the first glioma image we can find.
    import glob
    pattern = os.path.join(config.DATASET_PATHS["nickparvar"], "Training", "glioma", "*.jpg")
    files = glob.glob(pattern)
    if not files:
        print(f"No images found at {pattern} — is the dataset in place?")
    else:
        t = preprocess_image(files[0])
        print(f"OK  preprocessed {os.path.basename(files[0])}")
        print(f"    tensor shape : {tuple(t.shape)}  (expect (3, {config.IMG_SIZE}, {config.IMG_SIZE}))")
        print(f"    dtype        : {t.dtype}")
        print(f"    min / max    : {t.min():.3f} / {t.max():.3f}  (normalized -> roughly -2..+2)")
