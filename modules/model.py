"""
model.py — CertifyBTC multi-scale backbone (Phase 2).

Idea: a single-scale classifier sees the tumor at one resolution. Gliomas are diffuse and
infiltrative; meningiomas are compact and well-defined. Reading features at THREE depths of
EfficientNetB4 — Block 3 (fine texture), Block 5 (mid structure), Block 7 (coarse semantics)
— each cleaned by its own CBAM, gives the classifier complementary views. This is the main
lever we pull to fix the glioma/meningioma confusion that limited HXAI-BTC.

Pipeline:
    image -> EfficientNetB4 -> {Block3, Block5, Block7 feature maps}
          -> CBAM per block -> global-avg-pool each -> concat -> FC(512) -> classifier

Later phases attach to the 512-d fused feature: the domain-adversarial head (Phase 5) and
the evidential head (Phase 6). For now the head is a plain linear classifier, trained with
Focal Loss in Phase 3.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models.feature_extraction import create_feature_extractor

import config
from modules.cbam import CBAM

# Which EfficientNet stages we tap. In torchvision's efficientnet, `features[i]` is stage i;
# stages 3/5/7 are the "Block 3/5/7" from the plan.
RETURN_NODES = {"features.3": "b3", "features.5": "b5", "features.7": "b7"}


class CertifyBTC(nn.Module):
    def __init__(self, num_classes=None, pretrained=True, fusion_dim=512, dropout=0.3):
        super().__init__()
        if num_classes is None:
            num_classes = config.NUM_CLASSES

        # 1. Pretrained EfficientNetB4, wrapped so it returns our 3 intermediate maps.
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b4(weights=weights)
        self.features = create_feature_extractor(backbone, return_nodes=RETURN_NODES)

        # 2. Discover each tapped stage's channel count with a dry run (robust to B4 details).
        ch = self._infer_channels()
        self._channels = ch

        # 3. One CBAM per scale (keys must be valid module names -> no dots).
        self.cbam = nn.ModuleDict({k: CBAM(ch[k]) for k in ch})

        # 4. Fuse the three pooled vectors into a 512-d shared feature.
        self.pool = nn.AdaptiveAvgPool2d(1)
        concat_dim = sum(ch.values())
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 5. Classifier head (plain for now; swapped for evidential in Phase 6).
        self.classifier = nn.Linear(fusion_dim, num_classes)

    @torch.no_grad()
    def _infer_channels(self):
        """Push a zero tensor through and read the channel dim of each returned map."""
        self.features.eval()
        dummy = torch.zeros(1, 3, config.IMG_SIZE, config.IMG_SIZE)
        feats = self.features(dummy)
        return {k: v.shape[1] for k, v in feats.items()}

    def forward(self, x, return_maps=False):
        feats = self.features(x)                     # dict: {b3, b5, b7} feature maps
        pooled, maps = [], {}
        for k, fmap in feats.items():
            refined, sp_map = self.cbam[k](fmap, return_map=True)
            pooled.append(self.pool(refined).flatten(1))  # (B, C_k)
            maps[k] = sp_map
        fused = self.fusion(torch.cat(pooled, dim=1))     # (B, 512)
        logits = self.classifier(fused)                    # (B, num_classes)
        if return_maps:
            return logits, fused, maps
        return logits

    def freeze_backbone(self, freeze=True):
        """Stage 1 freezes EfficientNet; CBAM + fusion + head stay trainable.
        Stage 2 unfreezes the last blocks via unfreeze_last_blocks()."""
        for p in self.features.parameters():
            p.requires_grad = not freeze

    def unfreeze_last_blocks(self, n=2):
        """Stage 2: unfreeze only the last n EfficientNet stages (e.g. features.6 & features.7),
        keeping earlier stages frozen. CBAM/fusion/head are already trainable. Fine-tuning just
        the deep stages adapts high-level features to MRI without disturbing the low-level
        (edge/texture) filters that transfer well from ImageNet."""
        for p in self.features.parameters():
            p.requires_grad = False
        keep = {f"features.{7 - i}" for i in range(n)}   # n=2 -> {'features.7', 'features.6'}
        for name, p in self.features.named_parameters():
            stage = ".".join(name.split(".")[:2])         # e.g. 'features.6'
            if stage in keep:
                p.requires_grad = True
        return keep


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    device = config.DEVICE
    model = CertifyBTC().to(device)
    model.freeze_backbone(True)   # Stage-1 configuration

    # Dummy batch on the real device; run under AMP exactly like training will.
    x = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE, device=device)
    use_amp = config.USE_AMP and device == "cuda"
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=use_amp):
        logits, fused, maps = model(x, return_maps=True)

    total, trainable = count_params(model)
    print("=" * 58)
    print("  CertifyBTC forward-pass smoke test")
    print("=" * 58)
    print(f"  device            : {device}")
    print(f"  tapped channels   : {model._channels}")
    print(f"  input             : {tuple(x.shape)}")
    for k, m in maps.items():
        print(f"  {k} spatial map    : {tuple(m.shape)}")
    print(f"  fused feature     : {tuple(fused.shape)}  (expect (2, 512))")
    print(f"  logits            : {tuple(logits.shape)}  (expect (2, {config.NUM_CLASSES}))")
    print(f"  params total      : {total:,}")
    print(f"  params trainable  : {trainable:,}  (backbone frozen -> Stage 1)")
    print("Phase 2 model OK.")
