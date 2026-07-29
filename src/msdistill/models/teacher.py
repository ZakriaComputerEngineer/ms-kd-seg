"""Transformer teacher: a Mix Transformer encoder with a full-resolution decoder.

Why not stock SegFormer
-----------------------
SegFormer's All-MLP decoder emits logits at stride 4 and relies on a single
bilinear upsample to reach input resolution. That is fine for ADE20K, where
objects span hundreds of pixels, and fatal here: at 256x256 an MS lesion is
frequently 2-6 pixels across, so a stride-4 prediction cannot represent it at
all. Measured on this cohort, a stock SegFormer-B0 teacher reached 0.36 Dice on
normal WMH while a 487K-parameter U-Net student reached 0.57 -- the "teacher" was
the weaker model, which makes distillation meaningless.

The fix is a two-step refinement path that carries the decoder feature back to
stride 1, fusing it with shallow full-resolution features computed directly from
the input. The transformer encoder still does the semantic work; the refinement
head only restores spatial precision. It adds 198K parameters at the reported
`hr_channels=64`, roughly 0.7% of the 27.5M teacher.

Distillation taps
-----------------
`forward(..., return_features=True)` returns a `TeacherOutput` exposing:
  * `logits`      -- stride-1 class logits, the target for logit distillation
  * `feat`        -- the stride-4 decoder feature, the target for spatially
                     resolved feature distillation
  * `pooled`      -- globally averaged last-stage encoder feature, used only to
                     reproduce the pooled-feature (FitNets-style) baseline
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Encoder presets, matching the official Mix Transformer configurations.
#
# `depths` must be listed explicitly. `SegformerConfig` defaults to (2,2,2,2),
# which is correct only for B0 and B1; B2 is (3,4,6,3). Relying on the default
# builds a 16.3M-parameter model where B2 is 27.4M, and -- worse -- a checkpoint
# saved from the downloaded B2 then fails to load into the offline-constructed
# model, which is how this surfaced.
MIT_PRESETS = {
    "b0": {"hf_encoder": "nvidia/mit-b0",
           "hf_segmentation": "nvidia/segformer-b0-finetuned-ade-512-512",
           "hidden_sizes": (32, 64, 160, 256), "depths": (2, 2, 2, 2), "decoder_dim": 256},
    "b1": {"hf_encoder": "nvidia/mit-b1",
           "hf_segmentation": "nvidia/segformer-b1-finetuned-ade-512-512",
           "hidden_sizes": (64, 128, 320, 512), "depths": (2, 2, 2, 2), "decoder_dim": 256},
    "b2": {"hf_encoder": "nvidia/mit-b2",
           "hf_segmentation": "nvidia/segformer-b2-finetuned-ade-512-512",
           "hidden_sizes": (64, 128, 320, 512), "depths": (3, 4, 6, 3), "decoder_dim": 768},
}


@dataclass
class TeacherOutput:
    logits: torch.Tensor                  # (B, C, H, W)
    feat: Optional[torch.Tensor] = None   # (B, D, H/4, W/4)
    pooled: Optional[torch.Tensor] = None  # (B, hidden_sizes[-1])


def _first_conv(module: nn.Module) -> Optional[nn.Conv2d]:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            return m
    return None


def _adapt_input_channels(encoder: nn.Module, in_channels: int) -> None:
    """Replace the patch-embedding stem so it accepts `in_channels` modalities.

    Pretrained RGB filters are reused for the first three channels; any extra
    channels are initialized from the mean of the RGB filters, which starts them
    at a sensible grayscale-like response rather than at noise.
    """
    old = _first_conv(encoder)
    if old is None or old.in_channels == in_channels:
        return
    new = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=old.bias is not None)
    with torch.no_grad():
        n_copy = min(in_channels, old.in_channels)
        new.weight[:, :n_copy] = old.weight[:, :n_copy]
        if in_channels > old.in_channels:
            new.weight[:, old.in_channels:] = old.weight.mean(dim=1, keepdim=True)
        if old.bias is not None:
            new.bias.copy_(old.bias)

    # Locate and replace the module in its parent.
    for parent in encoder.modules():
        for name, child in parent.named_children():
            if child is old:
                setattr(parent, name, new)
                return


class MLPProject(nn.Module):
    """SegFormer's per-pixel linear projection of one encoder stage."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)     # (B, HW, C)
        x = self.proj(x)
        return x.transpose(1, 2).reshape(b, -1, h, w)


class AllMLPDecoder(nn.Module):
    """SegFormer's All-MLP head: project every stage to a common width, upsample
    to stride 4, concatenate, and fuse with a 1x1 convolution."""

    def __init__(self, hidden_sizes: Tuple[int, ...], decoder_dim: int, dropout: float = 0.1):
        super().__init__()
        self.projections = nn.ModuleList([MLPProject(c, decoder_dim) for c in hidden_sizes])
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * len(hidden_sizes), decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.out_dim = decoder_dim

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        target_hw = features[0].shape[-2:]
        projected = []
        for proj, feat in zip(self.projections, features):
            p = proj(feat)
            if p.shape[-2:] != target_hw:
                p = F.interpolate(p, size=target_hw, mode="bilinear", align_corners=False)
            projected.append(p)
        # Deepest-first concatenation matches the reference implementation.
        fused = self.fuse(torch.cat(projected[::-1], dim=1))
        return self.dropout(fused)


class HRRefinement(nn.Module):
    """Lifts the stride-4 decoder feature back to stride 1.

    Two shallow convolutional streams computed straight from the input supply the
    high-frequency detail that the transformer encoder discarded when it
    downsampled; the decoder feature supplies the semantics.
    """

    def __init__(self, in_dim: int, in_channels: int, hr_channels: int = 64):
        super().__init__()
        half = max(hr_channels // 2, 8)

        # Stride-1 and stride-2 detail streams over the raw input.
        self.stem_full = nn.Sequential(
            nn.Conv2d(in_channels, half, 3, padding=1, bias=False),
            nn.BatchNorm2d(half), nn.ReLU(inplace=True),
        )
        self.stem_half = nn.Sequential(
            nn.Conv2d(half, hr_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hr_channels), nn.ReLU(inplace=True),
        )

        self.reduce = nn.Sequential(
            nn.Conv2d(in_dim, hr_channels, 1, bias=False),
            nn.BatchNorm2d(hr_channels), nn.ReLU(inplace=True),
        )
        self.fuse_half = nn.Sequential(
            nn.Conv2d(hr_channels * 2, hr_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hr_channels), nn.ReLU(inplace=True),
        )
        self.fuse_full = nn.Sequential(
            nn.Conv2d(hr_channels + half, hr_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hr_channels), nn.ReLU(inplace=True),
        )
        self.out_dim = hr_channels

    def forward(self, decoder_feat: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        s1 = self.stem_full(image)          # stride 1
        s2 = self.stem_half(s1)             # stride 2

        x = self.reduce(decoder_feat)                                        # stride 4
        x = F.interpolate(x, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse_half(torch.cat([x, s2], dim=1))                        # stride 2
        x = F.interpolate(x, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse_full(torch.cat([x, s1], dim=1))                        # stride 1
        return x


class SegFormerHR(nn.Module):
    """Mix Transformer encoder + All-MLP decoder + optional HR refinement."""

    def __init__(self, num_classes: int, in_channels: int = 3, variant: str = "b2",
                 hr_decoder: bool = True, hr_channels: int = 64, pretrained: bool = True):
        super().__init__()
        if variant not in MIT_PRESETS:
            raise ValueError(f"variant must be one of {sorted(MIT_PRESETS)}, got {variant!r}")
        preset = MIT_PRESETS[variant]
        self.variant = variant
        self.num_classes = num_classes
        self.hidden_sizes = tuple(preset["hidden_sizes"])

        self.encoder = self._build_encoder(preset, pretrained)
        _adapt_input_channels(self.encoder, in_channels)

        self.decoder = AllMLPDecoder(self.hidden_sizes, preset["decoder_dim"])
        self.feature_dim = self.decoder.out_dim
        self.pooled_dim = self.hidden_sizes[-1]

        self.hr = HRRefinement(self.decoder.out_dim, in_channels, hr_channels) if hr_decoder else None
        head_dim = self.hr.out_dim if self.hr is not None else self.decoder.out_dim
        self.classifier = nn.Conv2d(head_dim, num_classes, kernel_size=1)

    @staticmethod
    def _encoder_config(preset: dict):
        from transformers import SegformerConfig
        return SegformerConfig(
            hidden_sizes=list(preset["hidden_sizes"]),
            depths=list(preset["depths"]),
        )

    @classmethod
    def _build_encoder(cls, preset: dict, pretrained: bool) -> nn.Module:
        try:
            from transformers import SegformerModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The teacher requires `transformers`. Install it with "
                "`pip install transformers`."
            ) from exc

        if not pretrained:
            return SegformerModel(cls._encoder_config(preset))

        # Prefer the ADE20K-finetuned checkpoint, whose features are already
        # adapted to dense prediction; fall back to the ImageNet-only encoder,
        # then to random initialization so an offline environment still produces
        # a runnable -- if much weaker -- model.
        last_error: Optional[Exception] = None
        for name in (preset["hf_segmentation"], preset["hf_encoder"]):
            try:
                if "finetuned" in name:
                    from transformers import SegformerForSemanticSegmentation
                    return SegformerForSemanticSegmentation.from_pretrained(name).segformer
                return SegformerModel.from_pretrained(name)
            except Exception as exc:  # network, auth, or hub-layout failure
                last_error = exc
                continue

        print(f"  [teacher] WARNING: could not download pretrained weights "
              f"({type(last_error).__name__}: {last_error}). Falling back to random "
              f"initialization -- the teacher will be far weaker than reported, and "
              f"the distillation results will not be meaningful.")
        return SegformerModel(cls._encoder_config(preset))

    def encoder_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = self.encoder(pixel_values=x, output_hidden_states=True, return_dict=True)
        states = list(out.hidden_states)
        # Some transformers versions return an extra final-norm state; keep the
        # four pyramid levels whose channel widths match the preset.
        if len(states) > len(self.hidden_sizes):
            states = [s for s in states if s.shape[1] in self.hidden_sizes][-len(self.hidden_sizes):]
        return states

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.encoder_features(x)
        decoded = self.decoder(features)

        if self.hr is not None:
            head_in = self.hr(decoded, x)
            logits = self.classifier(head_in)
        else:
            logits = self.classifier(decoded)

        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if not return_features:
            return logits
        return TeacherOutput(logits=logits, feat=decoded, pooled=features[-1].mean(dim=(2, 3)))


class FrozenTeacher(nn.Module):
    """Inference-only wrapper around a trained `SegFormerHR`.

    Freezing is enforced structurally rather than by convention: parameters have
    `requires_grad=False`, `train()` is a no-op that keeps the module in eval
    mode, and `forward` runs under `torch.no_grad()`. A training loop cannot
    accidentally update the teacher or leak BatchNorm statistics into it.
    """

    def __init__(self, model: SegFormerHR):
        super().__init__()
        self.model = model
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self.feature_dim = model.feature_dim
        self.pooled_dim = model.pooled_dim

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, num_classes: int, in_channels: int,
                        variant: str, hr_decoder: bool, hr_channels: int,
                        map_location="cpu") -> "FrozenTeacher":
        import os
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Teacher checkpoint not found: {checkpoint_path}. Refusing to "
                "construct a randomly initialized teacher -- that silently produces "
                "meaningless distillation targets."
            )
        model = SegFormerHR(num_classes=num_classes, in_channels=in_channels, variant=variant,
                            hr_decoder=hr_decoder, hr_channels=hr_channels, pretrained=False)
        state = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(
                f"Teacher checkpoint {checkpoint_path} is missing {len(missing)} tensors "
                f"(first few: {missing[:5]}). The checkpoint does not match "
                f"variant={variant!r}, hr_decoder={hr_decoder}."
            )
        if unexpected:
            print(f"  [teacher] ignoring {len(unexpected)} unexpected tensors in checkpoint")
        return cls(model)

    def train(self, mode: bool = True) -> "FrozenTeacher":
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, return_features: bool = False):
        return self.model(x, return_features=return_features)


def build_teacher(cfg, pretrained: bool = True) -> SegFormerHR:
    return SegFormerHR(
        num_classes=cfg.num_classes,
        in_channels=len(cfg.modalities),
        variant=cfg.teacher_variant,
        hr_decoder=cfg.teacher_hr_decoder,
        hr_channels=cfg.teacher_hr_channels,
        pretrained=pretrained,
    )


def load_frozen_teacher(cfg, checkpoint_path: str, device) -> FrozenTeacher:
    teacher = FrozenTeacher.from_checkpoint(
        checkpoint_path,
        num_classes=cfg.num_classes,
        in_channels=len(cfg.modalities),
        variant=cfg.teacher_variant,
        hr_decoder=cfg.teacher_hr_decoder,
        hr_channels=cfg.teacher_hr_channels,
        map_location="cpu",
    )
    return teacher.to(device).eval()
