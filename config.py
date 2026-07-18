"""
config.py — CERTIFY-BTC single source of truth.

Everything that changes between your RTX 4050 (debugging) and a cloud GPU
(full runs) is controlled by ONE switch: the MACHINE variable below.
No other file should hard-code a batch size, epoch count, or path.

Why a single config file?
- Reproducibility: a reviewer can read one file and know exactly how a run
  was configured. IEEE papers live or die on reproducibility.
- Safety: you can't accidentally launch a 30-epoch cloud run on your laptop,
  or a tiny debug run on a rented GPU, because the knobs live in one place.
"""

import os
import torch

# ---------------------------------------------------------------------------
# 1. THE MASTER SWITCH
# ---------------------------------------------------------------------------
# "local"      -> RTX 4050 6GB. Tiny batches, 2 epochs, one dataset, heavy modules OFF.
#                 Purpose: DEBUG that the code runs end-to-end without crashing.
# "local_full" -> RTX 4050 but the REAL run: ALL Nickparvar data, full epochs, heavy modules
#                 still OFF (no HD-BET/pyradiomics). Use for the headline-accuracy run.
# "cloud"      -> rented GPU. Full batches, full epochs, all datasets, everything ON.
#
# Set via the CERTIFY_MACHINE env var (default "local") so you never edit this file to launch a
# big run. In PowerShell:  $env:CERTIFY_MACHINE="local_full"; python train.py --stage 1
MACHINE = os.environ.get("CERTIFY_MACHINE", "local")

assert MACHINE in ("local", "local_full", "cloud"), \
    f"MACHINE must be 'local', 'local_full', or 'cloud', got {MACHINE!r}"

# ---------------------------------------------------------------------------
# 2. DEVICE
# ---------------------------------------------------------------------------
# Detect CUDA once, here, and import DEVICE everywhere else. When you load a
# checkpoint you MUST pass map_location=DEVICE so a GPU-saved file can be read
# back on CPU (or a different GPU) without crashing.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# 3. PATHS
# ---------------------------------------------------------------------------
# All paths are derived from this file's location, so the project works no
# matter where you clone it (laptop vs cloud) without editing absolute paths.
ROOT_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(ROOT_DIR, "data")
CHECKPOINT_DIR  = os.path.join(ROOT_DIR, "checkpoints")
LOG_DIR         = os.path.join(ROOT_DIR, "logs")

# Per-dataset roots. These folders won't exist yet — we create the loaders in
# Phase 1. Kept here so there is ONE place to point at your data.
DATASET_PATHS = {
    "nickparvar": os.path.join(DATA_DIR, "nickparvar"),  # primary 4-class (Kaggle)
    "figshare":   os.path.join(DATA_DIR, "figshare"),    # 3-class, cross-dataset val
    "brats":      os.path.join(DATA_DIR, "brats"),        # has real segmentation masks
    "tcga":       os.path.join(DATA_DIR, "tcga"),         # multi-site robustness
}

# Make sure the output folders exist. We never auto-create data folders —
# you place data there yourself so a typo can't silently create an empty set.
for _p in (CHECKPOINT_DIR, LOG_DIR):
    os.makedirs(_p, exist_ok=True)

# ---------------------------------------------------------------------------
# 4. CLASSES
# ---------------------------------------------------------------------------
# Canonical class order for the WHOLE project. Index = integer label the model
# outputs. Every dataset loader maps ITS labels onto this order (Phase 1), so
# "glioma" is always class 0 no matter which dataset a sample came from.
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

# Class weights for Focal Loss. Glioma was the weak class in HXAI-BTC (~82%),
# so it gets extra weight to push the model to care about it more.
CLASS_WEIGHTS = {
    "glioma":     2.0,
    "meningioma": 1.0,
    "notumor":    1.0,
    "pituitary":  1.0,
}
# As a plain list aligned to CLASS_NAMES order, ready to hand to a loss fn.
CLASS_WEIGHTS_LIST = [CLASS_WEIGHTS[c] for c in CLASS_NAMES]

# ---------------------------------------------------------------------------
# 5. IMAGE / PREPROCESSING
# ---------------------------------------------------------------------------
IMG_SIZE = 380   # EfficientNetB4's native resolution

# Feature flags for the expensive preprocessing steps. In local/debug mode we
# turn the slow ones off so you can iterate fast; cloud turns them on.
PREPROCESS = {
    "skull_strip": MACHINE == "cloud",   # HD-BET is slow + heavy; cloud only
    "n4_bias":     MACHINE == "cloud",   # N4 bias-field correction (SimpleITK)
    "clahe":       True,                  # cheap contrast boost; fine on laptop
}

# ---------------------------------------------------------------------------
# 6. MODE-DEPENDENT HYPERPARAMETERS
# ---------------------------------------------------------------------------
# This dict is the heart of the local/cloud split. We pick ONE block based on
# MACHINE and expose its values as module-level names below.
_MODE = {
    "local": {
        "BATCH_SIZE":        4,      # 6GB VRAM: keep it tiny
        "ACCUM_STEPS":       4,      # grad accumulation -> effective batch 16
        "NUM_WORKERS":       0,      # Windows + small runs: 0 avoids worker bugs
        "USE_AMP":           True,   # mixed precision saves VRAM even in debug
        "DATASETS":          ["nickparvar"],   # one dataset while debugging
        "HEAVY_MODULES":     False,  # diffusion / full radiomics OFF
        "STAGE1_EPOCHS":     2,      # just enough to prove the loop works
        "STAGE2_EPOCHS":     1,
        "LIMIT_SAMPLES":     200,    # cap dataset size for a fast smoke test
    },
    "local_full": {                  # the REAL run on the 4050: full Nickparvar, full epochs
        "BATCH_SIZE":        16,     # VRAM-probed safe (Stage 2 peaks < 1GB at this size)
        "ACCUM_STEPS":       2,      # effective batch 32
        "NUM_WORKERS":       0,      # 0 = rock-solid on Windows for an unattended run
        "USE_AMP":           True,
        "DATASETS":          ["nickparvar"],   # only dataset we have locally
        "HEAVY_MODULES":     False,  # no pyradiomics/SHAP/HD-BET on 6GB
        "STAGE1_EPOCHS":     15,
        "STAGE2_EPOCHS":     12,
        "LIMIT_SAMPLES":     None,   # ALL the data
    },
    "cloud": {
        "BATCH_SIZE":        32,
        "ACCUM_STEPS":       1,      # big enough batch, no accumulation needed
        "NUM_WORKERS":       8,
        "USE_AMP":           True,
        "DATASETS":          ["nickparvar", "figshare", "brats", "tcga"],
        "HEAVY_MODULES":     True,
        "STAGE1_EPOCHS":     15,
        "STAGE2_EPOCHS":     12,
        "LIMIT_SAMPLES":     None,   # use the whole dataset
    },
}[MACHINE]

# Expose the selected block as top-level names so other files just do
# `from config import BATCH_SIZE` etc.
BATCH_SIZE      = _MODE["BATCH_SIZE"]
ACCUM_STEPS     = _MODE["ACCUM_STEPS"]
NUM_WORKERS     = _MODE["NUM_WORKERS"]
USE_AMP         = _MODE["USE_AMP"]
ACTIVE_DATASETS = _MODE["DATASETS"]
HEAVY_MODULES   = _MODE["HEAVY_MODULES"]
STAGE1_EPOCHS   = _MODE["STAGE1_EPOCHS"]
STAGE2_EPOCHS   = _MODE["STAGE2_EPOCHS"]
LIMIT_SAMPLES   = _MODE["LIMIT_SAMPLES"]

# ---------------------------------------------------------------------------
# 7. TRAINING (mode-independent)
# ---------------------------------------------------------------------------
# Two-stage schedule from the plan. Stage 1 trains the new parts with the
# backbone frozen; Stage 2 unfreezes the last blocks and adds the domain-
# adversarial loss. These values are the same on laptop and cloud — only the
# EPOCH COUNTS above differ.
STAGE1 = {
    "lr":              1e-3,
    "freeze_backbone": True,
    "optimizer":       "adam",
    "scheduler":       "cosine",   # CosineAnnealingLR
    "loss":            "focal",
}
STAGE2 = {
    "lr":                    1e-4,
    "unfreeze_last_blocks":  2,
    "optimizer":             "adam",
    "scheduler":             "cosine",
    "loss":                  "focal",
    "domain_adversarial":    True,  # GRL alpha ramps 0 -> 1 across the stage
}

FOCAL_GAMMA = 2.0   # focusing parameter; higher = more focus on hard examples
SEED        = 42    # set everywhere for reproducibility

# ---------------------------------------------------------------------------
# 8. CERTIFICATION LAYER TARGETS (used from Phase 6 on)
# ---------------------------------------------------------------------------
CONFORMAL_COVERAGE = 0.95   # RAPS target coverage
OOD_METHOD         = "energy"

# ---------------------------------------------------------------------------
# 9. SANITY PRINT
# ---------------------------------------------------------------------------
def summary():
    """Print the active configuration. Run `python config.py` to see it."""
    print("=" * 60)
    print(f"  CERTIFY-BTC config   |   MACHINE = {MACHINE}")
    print("=" * 60)
    print(f"  DEVICE           : {DEVICE}")
    print(f"  Active datasets  : {ACTIVE_DATASETS}")
    print(f"  Batch size       : {BATCH_SIZE}  (accum x{ACCUM_STEPS} "
          f"-> effective {BATCH_SIZE * ACCUM_STEPS})")
    print(f"  AMP (fp16)       : {USE_AMP}")
    print(f"  Heavy modules    : {HEAVY_MODULES}")
    print(f"  Stage-1 epochs   : {STAGE1_EPOCHS}")
    print(f"  Stage-2 epochs   : {STAGE2_EPOCHS}")
    print(f"  Sample cap       : {LIMIT_SAMPLES}")
    print(f"  Image size       : {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Classes          : {CLASS_NAMES}")
    print("=" * 60)


if __name__ == "__main__":
    summary()
