"""
U-Net architecture for soil mapping (V3-F5).

A standard encoder–decoder U-Net with configurable input channels, depth, and
width, exposing two optional output heads on a shared decoder:

* a **classification** head → per-pixel class logits (``n_classes`` channels),
* a **regression** head → a single continuous channel (e.g. clay %).

The model is device-agnostic (CPU / CUDA / Apple MPS via :func:`resolve_device`)
so the same code trains locally on CPU/MPS and on a Colab/Kaggle GPU (V3-T6).
Inputs of arbitrary spatial size are handled by padding to a multiple of
``2**depth`` and cropping the output back, so 256×256 tiles and edge-clamped
tiles both pass through unchanged in shape.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a device string. ``"auto"`` prefers CUDA, then MPS, then CPU."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


class _DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    Encoder–decoder U-Net with optional classification + regression heads.

    Args:
        in_channels:    number of input feature channels
        base_channels:  channel width of the first encoder block (doubles each level)
        depth:          number of down/up-sampling levels
        n_classes:      classification head output channels; 0 disables the head
        regression:     if True, add a 1-channel regression head
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        depth: int = 4,
        n_classes: int = 6,
        regression: bool = False,
    ):
        super().__init__()
        if n_classes == 0 and not regression:
            raise ValueError("at least one head required (n_classes>0 or regression=True)")
        self.depth = depth
        self.n_classes = n_classes
        self.regression = regression

        # Encoder
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        ch = in_channels
        widths = [base_channels * (2 ** i) for i in range(depth)]
        for w in widths:
            self.downs.append(_DoubleConv(ch, w))
            ch = w

        # Bottleneck
        self.bottleneck = _DoubleConv(ch, ch * 2)
        ch = ch * 2

        # Decoder
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for w in reversed(widths):
            self.ups.append(nn.ConvTranspose2d(ch, w, 2, stride=2))
            self.up_convs.append(_DoubleConv(w * 2, w))
            ch = w

        if n_classes > 0:
            self.class_head = nn.Conv2d(ch, n_classes, 1)
        if regression:
            self.reg_head = nn.Conv2d(ch, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Pad to a multiple of 2**depth so pooling/upsampling round-trips.
        _, _, h, w = x.shape
        mult = 2 ** self.depth
        pad_h = (mult - h % mult) % mult
        pad_w = (mult - w % mult) % mult
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, up_conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = up_conv(x)

        # Crop back to the original input size.
        x = x[:, :, :h, :w]

        out: dict[str, torch.Tensor] = {}
        if self.n_classes > 0:
            out["class_logits"] = self.class_head(x)
        if self.regression:
            out["regression"] = self.reg_head(x)
        return out
