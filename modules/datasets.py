"""
datasets.py — flexible multi-dataset loader for CERTIFY-BTC (Phase 1).

Today it loads Nickparvar. It is written so adding Figshare/BraTS/TCGA later is just a
new entry in DATASET_CLASS_MAP: every dataset's native folder names get mapped onto the
canonical config.CLASS_NAMES order, so "glioma" is always class 0 regardless of source.

Nickparvar layout on disk:
    data/nickparvar/Training/<class>/*.jpg
    data/nickparvar/Testing/<class>/*.jpg
"""

import os
import sys
import glob

# Project root on path (see the same note in preprocessing.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import config
from modules.preprocessing import preprocess_image

# --- Per-dataset label mapping -> canonical CLASS_NAMES ----------------------
# Key = dataset name; value = {folder_name_in_that_dataset: canonical_class_name}.
# Nickparvar already uses our names. Figshare (3-class, no 'notumor') will add its
# own entry later; a class a dataset doesn't have simply isn't produced by it.
DATASET_CLASS_MAP = {
    "nickparvar": {
        "glioma":     "glioma",
        "meningioma": "meningioma",
        "notumor":    "notumor",
        "pituitary":  "pituitary",
    },
}

# canonical class name -> integer label (the number the model outputs)
CLASS_TO_IDX = {name: i for i, name in enumerate(config.CLASS_NAMES)}


def scan_split(dataset_name, split):
    """Return a list of (image_path, canonical_label_idx) for one split folder.

    `split` is the on-disk sub-folder, e.g. 'Training' or 'Testing'.
    """
    root = config.DATASET_PATHS[dataset_name]
    mapping = DATASET_CLASS_MAP[dataset_name]
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"Missing split folder: {split_dir}\n"
            f"Expected the dataset under {root} with '{split}/<class>/*.jpg'."
        )

    samples = []
    for folder_name, canonical in mapping.items():
        label_idx = CLASS_TO_IDX[canonical]
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for path in glob.glob(os.path.join(split_dir, folder_name, ext)):
                samples.append((path, label_idx))
    return samples


def _stratified_subset(samples, n, seed):
    """Take a ~n-sample, class-balanced subset (for fast local smoke tests).

    We keep every class represented — critical because glioma is our target class and
    must never vanish from a tiny debug subset.
    """
    if n is None or n >= len(samples):
        return samples
    labels = [lbl for _, lbl in samples]
    subset, _ = train_test_split(samples, train_size=n, stratify=labels, random_state=seed)
    return subset


class BrainMRIDataset(Dataset):
    """Wraps a list of (path, label) and preprocesses on-the-fly in __getitem__."""

    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment
        # One RNG per dataset instance for reproducible augmentation. NOTE: with
        # num_workers>0 (cloud) each worker copies this RNG, so we re-seed it per
        # worker in Phase 3's training loop. Local mode uses 0 workers, so it's fine.
        self.rng = np.random.default_rng(config.SEED)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        x = preprocess_image(path, augment=self.augment, rng=self.rng)
        return x, label


def build_dataloaders(val_fraction=0.15):
    """Build train / val / test DataLoaders for the active datasets.

    - Nickparvar ships as Training/ + Testing/. We hold out `val_fraction` of Training
      (stratified) as validation, and keep Testing untouched as the test set.
    - In local mode config.LIMIT_SAMPLES shrinks every split so a smoke test finishes
      in seconds.
    Returns (train_loader, val_loader, test_loader) and prints a count summary.
    """
    train_pool, test_pool = [], []
    for name in config.ACTIVE_DATASETS:
        if name not in DATASET_CLASS_MAP:
            print(f"[skip] '{name}' not wired up yet (comes online in a later phase).")
            continue
        train_pool += scan_split(name, "Training")
        test_pool  += scan_split(name, "Testing")

    if not train_pool:
        raise RuntimeError("No training samples found. Is the dataset in data/ ?")

    # Stratified train/val split of the Training pool.
    train_labels = [lbl for _, lbl in train_pool]
    train_s, val_s = train_test_split(
        train_pool, test_size=val_fraction, stratify=train_labels,
        random_state=config.SEED,
    )
    test_s = test_pool

    # Local debug: shrink each split. Val/test get a small floor so every class survives.
    if config.LIMIT_SAMPLES is not None:
        floor = max(config.LIMIT_SAMPLES // 4, config.NUM_CLASSES * 4)
        train_s = _stratified_subset(train_s, config.LIMIT_SAMPLES, config.SEED)
        val_s   = _stratified_subset(val_s, floor, config.SEED)
        test_s  = _stratified_subset(test_s, floor, config.SEED)

    # Only the training set is augmented.
    train_ds = BrainMRIDataset(train_s, augment=True)
    val_ds   = BrainMRIDataset(val_s, augment=False)
    test_ds  = BrainMRIDataset(test_s, augment=False)

    common = dict(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
                  pin_memory=(config.DEVICE == "cuda"))
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader   = DataLoader(val_ds, shuffle=False, **common)
    test_loader  = DataLoader(test_ds, shuffle=False, **common)

    _print_summary(train_s, val_s, test_s)
    return train_loader, val_loader, test_loader


def _print_summary(train_s, val_s, test_s):
    def counts(samples):
        c = [0] * config.NUM_CLASSES
        for _, lbl in samples:
            c[lbl] += 1
        return c
    print("=" * 60)
    print(f"  Datasets: {config.ACTIVE_DATASETS}  |  MACHINE={config.MACHINE}")
    print("=" * 60)
    print("  split      total  " + "  ".join(f"{n[:5]:>5}" for n in config.CLASS_NAMES))
    for name, s in (("train", train_s), ("val", val_s), ("test", test_s)):
        c = counts(s)
        print(f"  {name:<9} {len(s):>5}  " + "  ".join(f"{v:>5}" for v in c))
    print("=" * 60)


if __name__ == "__main__":
    # Standalone smoke test: build the loaders and pull ONE batch.
    tr, va, te = build_dataloaders()
    xb, yb = next(iter(tr))
    print(f"\nOne train batch: images {tuple(xb.shape)}, labels {tuple(yb.shape)}")
    print(f"  image dtype {xb.dtype}, label dtype {yb.dtype}")
    print(f"  labels in this batch: {yb.tolist()}")
    print("Phase 1 dataloader OK.")
