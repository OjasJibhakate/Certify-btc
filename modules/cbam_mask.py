"""
cbam_mask.py  —  Phase 4 (not implemented yet).

Planned: turn the CBAM spatial attention map into a pseudo-segmentation mask:
bilinear upsample -> Gaussian blur -> Otsu threshold -> morphological CLOSE then
OPEN -> keep the largest connected component.
"""
