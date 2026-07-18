"""
model.py  —  Phase 2 (not implemented yet).

Planned: multi-scale EfficientNetB4. Extract Block 3 / 5 / 7 feature maps, apply
a CBAM to each, pool + concatenate -> FC(512), then two heads (evidential tumor
classifier + domain classifier with gradient reversal).
"""
