"""Experiment configuration for the MS lesion segmentation distillation study.

Every hyperparameter the paper reports lives here so that a single object fully
determines a run. `Config.fingerprint()` hashes the fields that affect training,
and the training loop refuses to reuse cached fold results whose fingerprint does
not match -- this is what prevents the stale-resume contamination that silently
mixed results across experiment variants in the previous pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


# Fields deliberately excluded from the fingerprint: they change where output goes
# or how fast it runs, but not what is computed.
_NON_SEMANTIC_FIELDS = frozenset({
    "data_root", "output_dir", "num_workers", "use_amp", "max_hours_per_fold",
    "progress", "benchmark_trials", "benchmark_warmup", "pin_memory",
    "resume_every_epochs", "parallel_folds",
})


@dataclass
class Config:
    # ------------------------------------------------------------------ paths
    data_root: str = "/kaggle/input/ms3seg-mri-tri-mask-lesion-segmentation"
    output_dir: str = "/kaggle/working/msdistill_out"

    # ----------------------------------------------------------- label scheme
    class_names: Tuple[str, ...] = ("background", "ventricles", "normal_wmh", "abnormal_wmh")
    num_classes: int = 4
    # Classes that carry clinical meaning; background is excluded from summary metrics.
    foreground_classes: Tuple[int, ...] = (1, 2, 3)
    # The class whose segmentation the paper treats as the primary endpoint.
    primary_class: int = 3  # abnormal (pathological) WMH
    # Classes for which lesion-wise detection metrics are computed.
    lesion_classes: Tuple[int, ...] = (2, 3)

    modalities: Tuple[str, ...] = ("T1", "T2", "FLAIR")

    # --------------------------------------------------------- preprocessing
    target_size: Tuple[int, int] = (256, 256)
    clip_percentiles: Tuple[float, float] = (0.5, 99.5)
    # Slices whose foreground (non-air) fraction falls below this are dropped from
    # training only. Evaluation always uses every slice of every test volume.
    min_brain_fraction: float = 0.01
    # Training slices containing any lesion voxel are sampled this many times more
    # often than lesion-free slices. 1.0 disables oversampling.
    lesion_oversample_factor: float = 3.0

    # ------------------------------------------------------------- partitions
    test_fraction: float = 0.20
    n_folds: int = 3
    seed: int = 42
    # Extra seeds for the seed-variance study. The student is cheap enough that
    # repeating it is affordable; the teacher is trained once per fold only.
    student_seeds: Tuple[int, ...] = (42, 1337, 2024)

    # --------------------------------------------------------------- training
    batch_size: int = 8
    num_epochs: int = 60
    early_stop_patience: int = 12
    num_workers: int = 2
    pin_memory: bool = True
    use_amp: bool = True
    grad_clip_norm: float = 5.0
    # Wall-clock guard per fold, independent of the epoch budget. On a Kaggle T4
    # a teacher fold runs roughly 1.5-2.5 h and a co-trained student fold roughly
    # 2-3 h, so 3.0 lets a healthy fold finish while still capping a stuck one.
    # Hitting the guard saves a snapshot and marks the fold incomplete, so
    # rerunning resumes rather than silently reporting a truncated run.
    max_hours_per_fold: float = 3.0
    progress: bool = True
    # Snapshot the whole co-trained student stage every N epochs so a killed
    # session loses at most N epochs rather than a whole fold. The snapshot holds
    # every variant's weights, optimizer moments and best-so-far state, so it is
    # large; writing it every epoch costs more in I/O than it saves.
    # 0 disables mid-fold snapshots.
    resume_every_epochs: int = 5

    # Run this many cross-validation folds concurrently, one per GPU. Folds are
    # completely independent -- no parameter or gradient crosses them -- so this
    # is near-linear scaling with zero inter-GPU communication.
    #
    # This is deliberately NOT nn.DataParallel, which re-broadcasts the entire
    # model to every device on every step. For a 27.5M teacher on PCIe-connected
    # T4s with no NVLink that is ~110 MB of traffic per step against ~120 ms of
    # compute, and it measured *slower* than a single GPU in an earlier version
    # of this pipeline.
    #
    # 1 disables it. "auto" is not offered on purpose: silently changing how a
    # long run executes based on what hardware happened to be allocated makes
    # results hard to reproduce. Excluded from the fingerprint -- it changes
    # scheduling, not arithmetic, so cached folds stay valid either way.
    parallel_folds: int = 1

    # Student optimizer (trained from random initialization).
    student_lr: float = 3e-3
    student_weight_decay: float = 1e-4
    student_warmup_epochs: int = 3

    # Teacher optimizer (fine-tuning pretrained weights).
    teacher_lr: float = 6e-5
    teacher_head_lr: float = 6e-4
    teacher_weight_decay: float = 1e-2
    teacher_warmup_epochs: int = 2
    teacher_epochs: int = 45
    teacher_early_stop_patience: int = 12

    # ------------------------------------------------------------ hard labels
    ce_weight: float = 0.5
    dice_weight: float = 0.5
    # Inverse-square-root-frequency weights, normalized so ventricles sit at 1.0.
    # The cohort's voxel shares are background 99.55%, ventricles 0.30%,
    # abnormal WMH 0.10%, normal WMH 0.05%, giving 1/sqrt(f) ratios of
    # 0.06 : 1.00 : 2.44 : 1.73 in (background, ventricles, normal, abnormal)
    # order. Plain inverse frequency would put a weight of ~2000 on normal WMH
    # and make training diverge; the square root is the usual compromise.
    class_ce_weights: Tuple[float, ...] = (0.05, 1.0, 2.4, 1.7)

    # ---------------------------------------------------------- architectures
    # Teacher encoder: "b0" | "b1" | "b2". b2 is the reported configuration;
    # b0/b1 exist as documented fallbacks for tighter GPU budgets.
    teacher_variant: str = "b2"
    # Adds the full-resolution refinement decoder that lifts SegFormer's native
    # stride-4 output back to stride 1. Without it the teacher cannot resolve
    # thin periventricular lesions and is *worse* than the tiny student.
    teacher_hr_decoder: bool = True
    teacher_hr_channels: int = 64

    student_base_channels: int = 8

    # ----------------------------------------------------- distillation terms
    kd_temperature: float = 3.0
    # Weight on the hard-label term; distillation terms are added on top with
    # their own weights rather than sharing (1 - alpha), so each ablation row
    # changes exactly one thing.
    kd_alpha_hard: float = 1.0
    kd_logit_weight: float = 3.0
    kd_cwd_weight: float = 3.0
    kd_cwd_temperature: float = 4.0
    kd_feature_weight: float = 1.0
    # Region-decoupled logit distillation: how much the foreground region (any
    # non-background voxel in the ground truth, dilated by `kd_region_dilation`)
    # is upweighted relative to background in the per-pixel KL.
    kd_fg_weight: float = 5.0
    kd_bg_weight: float = 1.0
    kd_region_dilation: int = 2
    # Linear ramp of every distillation term from 0 to full weight over this many
    # epochs, so an under-trained student is not dragged around by teacher noise
    # before it has learned anything.
    kd_warmup_epochs: int = 5

    # ------------------------------------------------------------- evaluation
    # Which held-out-test aggregation the tables report as headline numbers.
    # "volume" = Dice recomputed per patient over the whole 3-D volume, then
    # averaged across patients (the protocol Metrics Reloaded recommends).
    report_aggregation: str = "volume"
    # Overlap needed for a predicted connected component to count as detecting a
    # reference lesion, for lesion-wise F1.
    lesion_match_iou: float = 0.10
    # Connected components smaller than this many voxels are ignored in the
    # lesion-wise analysis (annotation noise floor).
    min_lesion_voxels: int = 3
    bootstrap_samples: int = 10000
    alpha_level: float = 0.05

    # ---------------------------------------------------------- benchmarking
    benchmark_warmup: int = 10
    benchmark_trials: int = 50
    benchmark_batch_sizes: Tuple[int, ...] = (1, 8)
    benchmark_cpu: bool = True

    # ------------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.teacher_variant not in {"b0", "b1", "b2"}:
            raise ValueError(f"teacher_variant must be b0/b1/b2, got {self.teacher_variant!r}")
        if len(self.class_ce_weights) != self.num_classes:
            raise ValueError("class_ce_weights must have one entry per class")
        if self.report_aggregation not in {"volume", "global"}:
            raise ValueError("report_aggregation must be 'volume' or 'global'")

    # ------------------------------------------------------------- directories
    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.output_dir, "checkpoints")

    @property
    def results_dir(self) -> str:
        return os.path.join(self.output_dir, "results")

    @property
    def figures_dir(self) -> str:
        return os.path.join(self.output_dir, "figures")

    @property
    def tables_dir(self) -> str:
        return os.path.join(self.output_dir, "tables")

    def make_dirs(self) -> "Config":
        for d in (self.checkpoint_dir, self.results_dir, self.figures_dir, self.tables_dir):
            os.makedirs(d, exist_ok=True)
        return self

    # ------------------------------------------------------------ fingerprint
    def semantic_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if k not in _NON_SEMANTIC_FIELDS}

    def fingerprint(self) -> str:
        blob = json.dumps(self.semantic_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"config": asdict(self), "fingerprint": self.fingerprint()},
                      f, indent=2, default=str)

    def describe(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


# ---------------------------------------------------------------------------
# The ablation grid.
#
# Every row trains the *same* student architecture from the *same* random
# initialization on the *same* batches; only `terms` differs. That is what makes
# the ladder an ablation rather than a collection of unrelated runs.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    # Subset of {"logit", "region", "cwd", "feat", "fitnets"}. "logit" is the
    # plain Hinton per-pixel KL; "region" replaces it with the region-decoupled
    # version; they are mutually exclusive.
    terms: Tuple[str, ...]
    needs_teacher: bool
    # Short description used in the generated LaTeX table and the README.
    note: str = ""
    # Compact objective description printed in the results table.
    objective: str = ""
    # Citation key for rows that reproduce a published distillation objective.
    # Printing these in the table is what distinguishes "we compared against
    # three published methods" from "we ablated our own loss".
    citation: str = ""
    # Overrides Config.student_base_channels. Used only by the capacity baseline.
    base_channels: Optional[int] = None

    def __post_init__(self) -> None:
        if "logit" in self.terms and "region" in self.terms:
            raise ValueError(f"{self.key}: 'logit' and 'region' are mutually exclusive")


ABLATION_VARIANTS: Tuple[Variant, ...] = (
    Variant("unet32", "U-Net (base 32)", (), False,
            "Conventionally sized U-Net trained under the identical protocol. "
            "Supplies a same-protocol convolutional reference point, which a "
            "published number obtained under a different protocol cannot.",
            objective="hard labels", base_channels=32),
    Variant("scratch", "Student-Scratch", (), False,
            "Hard labels only. Lower bound for the compact architecture.",
            objective="hard labels"),
    Variant("kd_vanilla", "Student + Hinton KD", ("logit",), True,
            "Per-pixel temperature-softened KL applied uniformly over all voxels.",
            objective="uniform per-pixel KL", citation="hinton2015"),
    Variant("kd_fitnets", "Student + FitNets", ("fitnets",), True,
            "Globally pooled bottleneck feature matched by MSE -- the most widely "
            "reproduced feature-distillation recipe in medical segmentation.",
            objective="pooled feature MSE", citation="romero2015fitnets"),
    Variant("kd_cwd", "Student + CWD", ("cwd",), True,
            "Channel-wise distribution matching over the logit maps.",
            objective="channel-wise KL", citation="shu2021cwd"),
    Variant("kd_region", "Student + region-decoupled KL", ("region",), True,
            "Per-pixel KL with the divergence decoupled by anatomical region.",
            objective="region-weighted KL"),
    Variant("kd_region_cwd", "Student + region KL + CWD", ("region", "cwd"), True,
            "Adds channel-wise distribution matching on top of the region term.",
            objective="region KL + channel-wise"),
    Variant("kd_full", "Student-Full (proposed)", ("region", "cwd", "feat"), True,
            "Complete objective, including spatially resolved feature alignment.",
            objective="region KL + CWD + spatial feature"),
)

VARIANTS_BY_KEY: Dict[str, Variant] = {v.key: v for v in ABLATION_VARIANTS}

# A reduced grid for tight compute budgets. Keeps the capacity baseline, the
# lower bound, the textbook method and the full objective. The capacity baseline
# stays in even under pressure: a same-protocol conventional U-Net is worth more
# to a reviewer than two extra ablation rungs.
QUICK_VARIANT_KEYS: Tuple[str, ...] = ("unet32", "scratch", "kd_vanilla", "kd_cwd", "kd_full")

# Variants used for the seed-variance study, which only needs the rows the
# headline claim rests on.
SEED_STUDY_KEYS: Tuple[str, ...] = ("scratch", "kd_vanilla", "kd_full")

# The order the ablation ladder is reported in, and the order consecutive
# significance tests walk. Each step adds exactly one component.
LADDER_KEYS: Tuple[str, ...] = ("scratch", "kd_vanilla", "kd_region", "kd_region_cwd", "kd_full")


def select_variants(keys: Optional[List[str]] = None) -> List[Variant]:
    if keys is None:
        return list(ABLATION_VARIANTS)
    missing = [k for k in keys if k not in VARIANTS_BY_KEY]
    if missing:
        raise KeyError(f"unknown variant keys: {missing}")
    return [VARIANTS_BY_KEY[k] for k in keys]
