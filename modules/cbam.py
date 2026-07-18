"""
cbam.py — Convolutional Block Attention Module (Phase 2).

CBAM makes the network 'pay attention' in two complementary ways:
  1. Channel attention — WHICH feature channels matter (e.g. edge-like vs blob-like).
  2. Spatial attention  — WHERE in the image to look (which pixels).
We apply channel first, then spatial (the order from Woo et al., 2018).

Why it matters for us: the SPATIAL attention map doubles as a rough tumor localizer.
In Phase 4 we turn that map into a pseudo-segmentation mask — so CBAM isn't only an
accuracy booster, it's also how the model 'points at' the tumor for explainability.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Learns a weight in [0,1] for each channel.

    It squeezes every channel to a single number two ways — global average-pool and
    global max-pool over height/width — runs both through the SAME small MLP, adds them,
    and sigmoids. Using both pools captures 'typical' and 'peak' activation of a channel.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)  # bottleneck; floor so tiny stages don't vanish
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = x.mean(dim=(2, 3))               # (B, C) global average pool
        mx  = x.amax(dim=(2, 3))               # (B, C) global max pool
        attn = self.mlp(avg) + self.mlp(mx)    # shared MLP on both, summed
        attn = torch.sigmoid(attn).view(b, c, 1, 1)
        return x * attn                        # broadcast-scale each channel


class SpatialAttention(nn.Module):
    """Learns a weight in [0,1] for each pixel.

    Collapses the channel dimension to 2 maps (average and max over channels), convolves
    them into a single map, and sigmoids -> a heat map over the image. This map is what
    we later reuse as a pseudo-mask.
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2  # 'same' padding so the map keeps the input H,W
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)         # (B,1,H,W)
        mx  = x.amax(dim=1, keepdim=True)         # (B,1,H,W)
        attn = torch.cat([avg, mx], dim=1)        # (B,2,H,W)
        attn = torch.sigmoid(self.conv(attn))     # (B,1,H,W) heat map in [0,1]
        return x * attn, attn                     # scaled features + the map itself


class CBAM(nn.Module):
    """Channel attention -> Spatial attention. Returns refined features, and optionally
    the spatial heat map (which we reuse as a pseudo-mask in Phase 4)."""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x, return_map=False):
        x = self.channel(x)
        x, sp_map = self.spatial(x)
        if return_map:
            return x, sp_map
        return x


if __name__ == "__main__":
    # Standalone smoke test on a dummy feature map (like an EfficientNet block would emit).
    x = torch.randn(2, 160, 12, 12)   # (batch, channels, H, W)
    cbam = CBAM(160)
    y, m = cbam(x, return_map=True)
    print("CBAM smoke test")
    print(f"  in          : {tuple(x.shape)}")
    print(f"  out         : {tuple(y.shape)}  (same shape — attention refines, not resizes)")
    print(f"  spatial map : {tuple(m.shape)}  (1-channel heat map)")
    print(f"  map min/max : {m.min():.3f} / {m.max():.3f}  (should be within 0..1)")
