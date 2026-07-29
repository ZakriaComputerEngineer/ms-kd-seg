"""Segmentation and distillation objectives.

Everything here is computed in float32 even when the surrounding forward pass runs
under autocast. Temperature-softened KL divergences on half-precision logits
underflow badly: with T=3 and four classes the softened probabilities routinely
land near 1e-4, where fp16 has roughly two significant digits left.

Distillation terms
------------------
`logit`   Per-pixel temperature-softened KL, weighted uniformly. This is the
          classical Hinton objective applied densely, and is the baseline the
          paper argues is inadequate for this task.

`region`  The same KL, but with the pixel weights decoupled into a lesion region
          and a background region. On this cohort background occupies about 96%
          of voxels, so a uniform average is dominated by voxels where teacher
          and student already agree and where agreement carries no clinical
          information. Upweighting the (dilated) annotated foreground restores
          gradient mass to the voxels the task is actually about.

`cwd`     Channel-wise distillation: each class channel is turned into a spatial
          probability distribution and matched to the teacher's. Because the
          normalization is per channel rather than per pixel, a rare class
          contributes as much signal as a common one -- which is precisely the
          property a per-pixel KL lacks under severe class imbalance.

`feat`    Spatially resolved feature alignment on the stride-4 decoder feature.
          Features are L2-normalized per pixel before comparison, so the loss
          constrains the direction of each feature vector rather than its
          magnitude and cannot be minimized by shrinking activations.

`fitnets` Pooled-feature regression, retained as a comparison point representing
          the common practice of matching globally averaged features.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Supervised terms
# --------------------------------------------------------------------------- #

class SoftDiceLoss(nn.Module):
    """Multi-class soft Dice, averaged over classes present in the batch.

    Classes absent from both prediction and reference are excluded from the mean.
    Including them contributes a term that is identically ~1.0 regardless of model
    quality, which dilutes the loss by a factor that varies with batch
    composition and makes the reported value incomparable across batches.
    """

    def __init__(self, num_classes: int, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits.float(), dim=1)
        onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * onehot).sum(dims)
        cardinality = probs.sum(dims) + onehot.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        present = (onehot.sum(dims) > 0)
        if present.any():
            return 1.0 - dice[present].mean()
        return 1.0 - dice.mean()


class HardLabelLoss(nn.Module):
    """Class-weighted cross entropy combined with soft Dice."""

    def __init__(self, cfg):
        super().__init__()
        self.register_buffer("class_weights",
                             torch.tensor(cfg.class_ce_weights, dtype=torch.float32))
        self.dice = SoftDiceLoss(cfg.num_classes)
        self.ce_weight = cfg.ce_weight
        self.dice_weight = cfg.dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        ce = F.cross_entropy(logits, target, weight=self.class_weights.to(logits.dtype))
        return self.ce_weight * ce + self.dice_weight * self.dice(logits, target)


# --------------------------------------------------------------------------- #
# Region weighting
# --------------------------------------------------------------------------- #

def region_weight_map(target: torch.Tensor, foreground_classes: Sequence[int],
                      fg_weight: float, bg_weight: float,
                      dilation: int = 2) -> torch.Tensor:
    """Per-pixel weights that upweight the annotated foreground and its margin.

    The dilation matters: lesion boundaries are exactly where teacher and student
    disagree most and where the teacher's soft distribution carries the most
    information, so the weighted region is grown a few voxels beyond the
    annotation rather than stopping at it.

    Weights are renormalized to mean 1.0 so that changing `fg_weight` changes the
    *distribution* of gradient over space without also rescaling the loss, which
    keeps the term's magnitude comparable across ablation rows.
    """
    fg = torch.zeros_like(target, dtype=torch.float32)
    for c in foreground_classes:
        fg = torch.maximum(fg, (target == c).float())

    if dilation > 0:
        k = 2 * dilation + 1
        fg = F.max_pool2d(fg.unsqueeze(1), kernel_size=k, stride=1, padding=dilation).squeeze(1)

    weights = bg_weight + (fg_weight - bg_weight) * fg
    return weights / weights.mean().clamp_min(1e-8)


# --------------------------------------------------------------------------- #
# Distillation terms
# --------------------------------------------------------------------------- #

def logit_distillation(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                       temperature: float,
                       weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Temperature-softened KL between dense per-pixel class distributions.

    The divergence is averaged over every spatial position as well as the batch.
    Folding H*W into the reduction is not optional: with 256x256 inputs, reducing
    only over the batch inflates the term by 65,536x relative to the supervised
    loss and the optimizer simply ignores the labels.
    """
    s = student_logits.float()
    t = teacher_logits.float().detach()
    T = temperature

    log_p_s = F.log_softmax(s / T, dim=1)
    p_t = F.softmax(t / T, dim=1)
    # (B, H, W) pointwise KL, before spatial reduction.
    kl = (p_t * (torch.log(p_t.clamp_min(1e-8)) - log_p_s)).sum(dim=1)

    if weight_map is not None:
        kl = kl * weight_map
    return kl.mean() * (T ** 2)


def channel_wise_distillation(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                              temperature: float) -> torch.Tensor:
    """Match each class channel's spatial activation distribution (CWD).

    Normalizing across space *within* a channel makes the term scale-free in the
    class frequency: the abnormal-WMH channel, which covers well under one
    percent of voxels, produces a spatial distribution just as peaked and just as
    informative as the background channel's.
    """
    s = student_logits.float()
    t = teacher_logits.float().detach()
    B, C, H, W = s.shape
    T = temperature

    s_flat = (s / T).reshape(B * C, H * W)
    t_flat = (t / T).reshape(B * C, H * W)

    log_p_s = F.log_softmax(s_flat, dim=1)
    p_t = F.softmax(t_flat, dim=1)
    kl = (p_t * (torch.log(p_t.clamp_min(1e-8)) - log_p_s)).sum(dim=1)
    return kl.mean() * (T ** 2)


def spatial_feature_distillation(student_feat_projected: torch.Tensor,
                                 teacher_feat: torch.Tensor,
                                 weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Direction-only alignment of spatially resolved features.

    Both maps are L2-normalized along the channel axis at every spatial location,
    so the objective asks the student to reproduce *what* the teacher encodes at
    each position, not how strongly. Without the normalization the loss admits a
    shrink-to-zero solution and destabilizes the student's decoder.
    """
    s = student_feat_projected.float()
    t = teacher_feat.float().detach()

    if s.shape[-2:] != t.shape[-2:]:
        s = F.interpolate(s, size=t.shape[-2:], mode="bilinear", align_corners=False)

    s = F.normalize(s, p=2, dim=1)
    t = F.normalize(t, p=2, dim=1)

    per_pixel = (s - t).pow(2).sum(dim=1)   # (B, h, w)
    if weight_map is not None:
        if weight_map.shape[-2:] != per_pixel.shape[-2:]:
            weight_map = F.interpolate(weight_map.unsqueeze(1), size=per_pixel.shape[-2:],
                                       mode="bilinear", align_corners=False).squeeze(1)
        per_pixel = per_pixel * weight_map
    return per_pixel.mean()


def pooled_feature_distillation(student_pooled_projected: torch.Tensor,
                                teacher_pooled: torch.Tensor) -> torch.Tensor:
    """FitNets-style regression onto globally averaged teacher features."""
    return F.mse_loss(student_pooled_projected.float(), teacher_pooled.float().detach())


# --------------------------------------------------------------------------- #
# Assembled criterion
# --------------------------------------------------------------------------- #

class DistillationCriterion(nn.Module):
    """Builds the total objective for one ablation variant.

    `terms` selects which distillation components are active; the supervised term
    is always present. Every distillation term is ramped linearly from zero over
    `cfg.kd_warmup_epochs`, because a randomly initialized student pushed hard
    toward teacher statistics before it can segment anything at all converges to
    a visibly worse optimum.
    """

    def __init__(self, cfg, terms: Sequence[str]):
        super().__init__()
        unknown = set(terms) - {"logit", "region", "cwd", "feat", "fitnets"}
        if unknown:
            raise ValueError(f"unknown distillation terms: {sorted(unknown)}")

        self.cfg = cfg
        self.terms = tuple(terms)
        self.hard_loss = HardLabelLoss(cfg)
        self._ramp = 1.0

    # -- warmup ------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        warmup = max(int(self.cfg.kd_warmup_epochs), 0)
        self._ramp = 1.0 if warmup == 0 else min(1.0, (epoch + 1) / float(warmup))

    @property
    def uses_teacher(self) -> bool:
        return len(self.terms) > 0

    @property
    def needs_spatial_feature(self) -> bool:
        return "feat" in self.terms

    @property
    def needs_pooled_feature(self) -> bool:
        return "fitnets" in self.terms

    # -- forward -----------------------------------------------------------
    def forward(self, student_logits: torch.Tensor, target: torch.Tensor,
                teacher_logits: Optional[torch.Tensor] = None,
                student_feat_projected: Optional[torch.Tensor] = None,
                teacher_feat: Optional[torch.Tensor] = None,
                student_pooled_projected: Optional[torch.Tensor] = None,
                teacher_pooled: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        hard = self.hard_loss(student_logits, target)
        total = cfg.kd_alpha_hard * hard
        parts: Dict[str, float] = {"hard": float(hard.detach())}

        if not self.terms:
            return total, parts

        if teacher_logits is None:
            raise ValueError(f"variant with terms {self.terms} requires teacher logits")

        ramp = self._ramp
        weights = None
        if "region" in self.terms or "feat" in self.terms:
            weights = region_weight_map(
                target, cfg.foreground_classes, cfg.kd_fg_weight, cfg.kd_bg_weight,
                cfg.kd_region_dilation,
            )

        if "logit" in self.terms:
            term = logit_distillation(student_logits, teacher_logits, cfg.kd_temperature)
            total = total + ramp * cfg.kd_logit_weight * term
            parts["logit"] = float(term.detach())

        if "region" in self.terms:
            term = logit_distillation(student_logits, teacher_logits, cfg.kd_temperature,
                                      weight_map=weights)
            total = total + ramp * cfg.kd_logit_weight * term
            parts["region"] = float(term.detach())

        if "cwd" in self.terms:
            term = channel_wise_distillation(student_logits, teacher_logits,
                                             cfg.kd_cwd_temperature)
            total = total + ramp * cfg.kd_cwd_weight * term
            parts["cwd"] = float(term.detach())

        if "feat" in self.terms:
            if student_feat_projected is None or teacher_feat is None:
                raise ValueError("the 'feat' term requires projected student and teacher features")
            term = spatial_feature_distillation(student_feat_projected, teacher_feat,
                                                weight_map=weights)
            total = total + ramp * cfg.kd_feature_weight * term
            parts["feat"] = float(term.detach())

        if "fitnets" in self.terms:
            if student_pooled_projected is None or teacher_pooled is None:
                raise ValueError("the 'fitnets' term requires pooled student and teacher features")
            term = pooled_feature_distillation(student_pooled_projected, teacher_pooled)
            total = total + ramp * cfg.kd_feature_weight * term
            parts["fitnets"] = float(term.detach())

        parts["ramp"] = ramp
        return total, parts
