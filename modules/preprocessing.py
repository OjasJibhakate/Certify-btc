"""
preprocessing.py  —  Phase 1 (not implemented yet).

Planned: skull-strip (HD-BET, cloud only), N4 bias correction (SimpleITK),
CLAHE contrast, resize to config.IMG_SIZE. Each step gated by config.PREPROCESS
so the slow ones stay off on the RTX 4050.
"""
