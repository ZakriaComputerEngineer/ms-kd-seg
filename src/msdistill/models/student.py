"""Lightweight convolutional student and the projection heads used for distillation.

The student is a U-Net with the channel width scaled down from a base of 32 to a
base of 8, giving 487,316 trainable parameters. Its topology is deliberately
unchanged from the widely used baseline so that the ablation isolates the
training objective rather than confounding it with an architecture search.

Projection heads
----------------
Feature distillation needs the student and teacher representations to live in a
common space. Only the *student* side is projected. This is not a stylistic
choice: if the teacher side is also given a learnable projection, minimizing
`MSE(W_s f_s, W_t f_t)` has the trivial optimum `W_s = W_t = 0`, so the loss can
be driven to zero while transferring nothing -- and the gradient that reaches the
student in the meantime collapses its features. We observed exactly that failure
(a 0.11 Dice drop and two diverged folds) before fixing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class StudentOutput:
    logits: torch.Tensor                   # (B, C, H, W)
    feat: Optional[torch.Tensor] = None    # (B, 4*base, H/4, W/4) decoder feature
    pooled: Optional[torch.Tensor] = None  # (B, 16*base) pooled bottleneck


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyUNetStudent(nn.Module):
    """U-Net topology at base width 8. 487K parameters at `base_ch=8`."""

    def __init__(self, in_channels: int, num_classes: int, base_ch: int = 8):
        super().__init__()
        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]

        self.enc1 = ConvBlock(in_channels, chs[0])
        self.enc2 = ConvBlock(chs[0], chs[1])
        self.enc3 = ConvBlock(chs[1], chs[2])
        self.enc4 = ConvBlock(chs[2], chs[3])
        self.bottleneck = ConvBlock(chs[3], chs[4])
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(chs[4], chs[3], 2, stride=2)
        self.dec4 = ConvBlock(chs[4], chs[3])
        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.dec3 = ConvBlock(chs[3], chs[2])
        self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec2 = ConvBlock(chs[2], chs[1])
        self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.dec1 = ConvBlock(chs[1], chs[0])

        self.out_conv = nn.Conv2d(chs[0], num_classes, 1)

        # Distillation tap dimensions.
        self.feature_dim = chs[2]   # stride-4 decoder feature
        self.pooled_dim = chs[4]    # bottleneck width

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))   # stride 4
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.out_conv(d1)

        if not return_features:
            return logits
        return StudentOutput(logits=logits, feat=d3, pooled=b.mean(dim=(2, 3)))


class SpatialFeatureProjector(nn.Module):
    """Maps the student's stride-4 decoder feature into the teacher's channel
    space with a 1x1 convolution, preserving spatial layout.

    Discarded after training: it exists only to make the two feature maps
    comparable, and contributes nothing to inference cost.
    """

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(student_dim, teacher_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(teacher_dim),
        )

    def forward(self, student_feat: torch.Tensor) -> torch.Tensor:
        return self.proj(student_feat)


class PooledFeatureProjector(nn.Module):
    """Student-side regressor for the pooled-feature (FitNets-style) baseline."""

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(student_dim, teacher_dim),
            nn.BatchNorm1d(teacher_dim),
        )

    def forward(self, student_pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(student_pooled)


def build_student(cfg, base_ch: Optional[int] = None) -> TinyUNetStudent:
    """`base_ch` overrides the configured width, used by the capacity baseline."""
    return TinyUNetStudent(
        in_channels=len(cfg.modalities),
        num_classes=cfg.num_classes,
        base_ch=base_ch if base_ch is not None else cfg.student_base_channels,
    )


def student_builder(base_ch: Optional[int] = None):
    """A one-argument builder, so ensembles of differently sized students can be
    loaded through the same `load_ensemble(builder, ...)` interface."""
    def build(cfg) -> TinyUNetStudent:
        return build_student(cfg, base_ch)
    return build


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
