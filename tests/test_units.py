"""Unit tests for the pieces where a silent error would corrupt reported numbers.

`scripts/smoke_test.py` covers the pipeline end to end. These target the specific
behaviours that previously produced wrong results without failing: the empty-class
metric convention, the degenerate feature-loss optimum, loss reductions, the
teacher freeze contract, and the cache-fingerprint guard.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from msdistill.config import ABLATION_VARIANTS, Config, Variant, select_variants
from msdistill.data import LabelRemapper, augment_slice, clip_and_normalize
from msdistill.losses import (DistillationCriterion, SoftDiceLoss, channel_wise_distillation,
                              logit_distillation, region_weight_map,
                              spatial_feature_distillation)
from msdistill.metrics import (dice_binary, evaluate_volume, hausdorff95, iou_binary,
                               lesion_wise_counts, normalized_surface_dice, summarize)
from msdistill.models.student import (SpatialFeatureProjector, build_student, count_parameters)
from msdistill.models.teacher import FrozenTeacher, build_teacher
from msdistill.stats import holm_bonferroni, paired_valid, wilcoxon_compare


@pytest.fixture
def cfg():
    return Config(target_size=(32, 32), teacher_variant="b0", teacher_hr_channels=8)


# --------------------------------------------------------------------------- #
# Metrics: the defect that invalidated the previous run
# --------------------------------------------------------------------------- #

class TestEmptyClassConvention:
    def test_empty_reference_and_prediction_is_undefined(self):
        empty = np.zeros((8, 8), dtype=bool)
        assert np.isnan(dice_binary(empty, empty)), (
            "an absent class must be undefined, not a free 1.0 -- this exact "
            "convention error inflated every rare-class score in the earlier pipeline"
        )

    def test_false_positive_only_scores_zero(self):
        ref = np.zeros((8, 8), dtype=bool)
        pred = np.zeros((8, 8), dtype=bool)
        pred[2:4, 2:4] = True
        assert dice_binary(pred, ref) == 0.0

    def test_perfect_overlap(self):
        m = np.zeros((8, 8), dtype=bool)
        m[1:5, 1:5] = True
        assert dice_binary(m, m) == pytest.approx(1.0)

    def test_half_overlap(self):
        ref = np.zeros((10, 10), dtype=bool)
        pred = np.zeros((10, 10), dtype=bool)
        ref[:, :4] = True     # 40 voxels
        pred[:, 2:6] = True   # 40 voxels, 20 shared
        assert dice_binary(pred, ref) == pytest.approx(2 * 20 / 80)
        assert iou_binary(pred, ref) == pytest.approx(20 / 60)

    def test_undefined_excluded_from_summary(self, cfg):
        """A patient missing a class must not drag that class's mean toward 1.0."""
        ref = np.zeros((8, 8, 3), dtype=np.int64)
        ref[0:2, 0:2, 0] = 3
        pred = ref.copy()
        case_full = evaluate_volume(pred, ref, cfg, "p1", compute_boundary=False)

        empty = np.zeros_like(ref)
        case_empty = evaluate_volume(empty, empty, cfg, "p2", compute_boundary=False)

        assert np.isnan(case_empty.dice["abnormal_wmh"])
        stats = summarize([case_full, case_empty], cfg)
        assert stats["dice"]["abnormal_wmh"]["n"] == 1
        assert stats["dice"]["abnormal_wmh"]["n_undefined"] == 1


class TestBoundaryMetrics:
    """Regressions for three defects an adversarial review found in this module.
    Each produced plausible-looking numbers rather than an error."""

    def test_surface_distance_uses_the_other_surface_not_the_other_object(self):
        """A hollow reference must not score a perfect boundary match.

        Building the distance transform from the complement of the *filled* mask
        records any surface voxel inside the other object at 0 mm, so a
        prediction nowhere near the reference boundary scored NSD 1.0.
        """
        ref = np.zeros((21, 21, 21), dtype=bool)
        ref[5:16, 5:16, 5:16] = True
        ref[7:14, 7:14, 7:14] = False          # hollow shell

        pred = np.zeros_like(ref)
        pred[8:13, 8:13, 8:13] = True          # sits inside the cavity

        nsd = normalized_surface_dice(pred, ref, (1.0, 1.0, 1.0), tolerance_mm=1.0)
        hd = hausdorff95(pred, ref, (1.0, 1.0, 1.0))
        assert nsd < 0.99, f"hollow reference scored NSD {nsd:.3f}; surfaces are not adjacent"
        assert hd > 0.5, f"hollow reference scored HD95 {hd:.3f} mm"

    def test_identical_masks_have_zero_boundary_error(self):
        m = np.zeros((16, 16, 8), dtype=bool)
        m[4:12, 4:12, 2:6] = True
        assert hausdorff95(m, m, (1.0, 1.0, 1.0)) == pytest.approx(0.0)
        assert normalized_surface_dice(m, m, (1.0, 1.0, 1.0)) == pytest.approx(1.0)

    def test_total_miss_is_scored_not_dropped(self):
        """Predicting nothing is the worst outcome, not an undefined one.

        Returning nan here removed the total misses from the mean and from every
        paired test, which flatters whichever variant segments least -- the exact
        axis the ablation varies.
        """
        ref = np.zeros((16, 16, 4), dtype=bool)
        ref[6:10, 6:10, 1:3] = True
        empty = np.zeros_like(ref)
        spacing = (0.9, 0.9, 5.73)

        assert normalized_surface_dice(empty, ref, spacing) == 0.0
        hd = hausdorff95(empty, ref, spacing)
        assert not np.isnan(hd) and hd > 0
        # Genuinely undefined only when there was nothing to find.
        assert np.isnan(hausdorff95(empty, empty, spacing))
        assert np.isnan(normalized_surface_dice(empty, empty, spacing))

    def test_boundary_metrics_scale_with_spacing(self):
        ref = np.zeros((16, 16, 16), dtype=bool)
        ref[4:12, 4:12, 4:12] = True
        pred = np.zeros_like(ref)
        pred[6:14, 4:12, 4:12] = True          # shifted 2 voxels along axis 0
        isotropic = hausdorff95(pred, ref, (1.0, 1.0, 1.0))
        stretched = hausdorff95(pred, ref, (2.0, 1.0, 1.0))
        assert stretched > isotropic


class TestLesionWise:
    def test_merged_blob_cannot_claim_every_lesion(self):
        """One-to-one matching. Scoring each reference component against the
        union of all predictions let a single whole-slice blob claim unlimited
        true positives at a cost of one false positive, out-scoring an accurate
        prediction."""
        ref = np.zeros((40, 40, 1), dtype=bool)
        for i in range(5):
            ref[5:9, 4 + 7 * i:8 + 7 * i, 0] = True

        everything = np.ones_like(ref)
        accurate = np.zeros_like(ref)
        for i in range(4):
            accurate[5:9, 4 + 7 * i:8 + 7 * i, 0] = True

        blob = lesion_wise_counts(everything, ref, 0.10, 3)
        good = lesion_wise_counts(accurate, ref, 0.10, 3)
        assert good.f1 > blob.f1, (
            f"whole-slice prediction scored F1 {blob.f1:.3f} against an accurate "
            f"prediction's {good.f1:.3f}")
        assert blob.tp <= 1, f"one merged component claimed {blob.tp} detections"

    def test_prediction_on_a_subthreshold_lesion_is_ignored_not_penalised(self):
        ref = np.zeros((40, 40, 1), dtype=bool)
        ref[10:12, 10:11, 0] = True            # 2 voxels, below min_voxels=3

        on_target = np.zeros_like(ref)
        on_target[9:12, 9:12, 0] = True        # lands on the tiny real lesion
        elsewhere = np.zeros_like(ref)
        elsewhere[30:33, 30:33, 0] = True      # empty background

        assert lesion_wise_counts(on_target, ref, 0.10, 3).fp == 0
        assert lesion_wise_counts(elsewhere, ref, 0.10, 3).fp == 1

    def test_counts_distinct_components(self):
        ref = np.zeros((20, 20, 3), dtype=bool)
        ref[2:6, 2:6, 1] = True      # lesion A
        ref[12:16, 12:16, 1] = True  # lesion B
        pred = np.zeros_like(ref)
        pred[2:6, 2:6, 1] = True     # finds A only
        pred[0:4, 15:19, 2] = True   # spurious component

        counts = lesion_wise_counts(pred, ref, overlap_threshold=0.10, min_voxels=3)
        assert counts.n_ref == 2 and counts.tp == 1 and counts.fn == 1 and counts.fp == 1
        assert counts.f1 == pytest.approx(2 * 1 / (2 * 1 + 1 + 1))
        assert counts.tpr == pytest.approx(0.5)

    def test_voxel_dice_can_disagree_with_detection(self):
        """The reason both are reported: outlining well and finding everything
        are different achievements."""
        ref = np.zeros((30, 30, 1), dtype=bool)
        ref[2:8, 2:8, 0] = True       # large lesion, 36 voxels
        for i in range(5):            # five small lesions, 9 voxels each
            ref[20:23, 2 + 5 * i:5 + 5 * i, 0] = True
        pred = np.zeros_like(ref)
        pred[2:8, 2:8, 0] = True      # finds only the large one

        assert dice_binary(pred, ref) > 0.5           # respectable overlap
        assert lesion_wise_counts(pred, ref).tpr == pytest.approx(1 / 6)  # poor detection


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #

class TestLosses:
    def test_logit_kl_is_zero_for_identical_logits(self):
        z = torch.randn(2, 4, 16, 16)
        assert float(logit_distillation(z, z, temperature=3.0)) == pytest.approx(0.0, abs=1e-5)

    def test_logit_kl_reduces_over_space_not_only_batch(self):
        """Reducing only over the batch inflates the term by H*W and the
        optimizer stops attending to the labels."""
        torch.manual_seed(0)
        small = logit_distillation(torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8), 3.0)
        torch.manual_seed(0)
        large_s = torch.randn(2, 4, 8, 8).repeat(1, 1, 4, 4)
        large_t = torch.randn(2, 4, 8, 8).repeat(1, 1, 4, 4)
        large = logit_distillation(large_s, large_t, 3.0)
        # 16x the spatial positions, same distribution -> same magnitude.
        assert float(large) == pytest.approx(float(small), rel=0.5)

    def test_cwd_is_zero_for_identical_logits(self):
        z = torch.randn(2, 4, 16, 16)
        assert float(channel_wise_distillation(z, z, 4.0)) == pytest.approx(0.0, abs=1e-5)

    def test_region_weights_have_unit_mean(self):
        """Renormalization is what keeps the term's magnitude comparable across
        ablation rows when kd_fg_weight changes."""
        target = torch.zeros(2, 32, 32, dtype=torch.long)
        target[:, 10:14, 10:14] = 3
        for fg in (2.0, 5.0, 20.0):
            w = region_weight_map(target, (1, 2, 3), fg_weight=fg, bg_weight=1.0, dilation=2)
            assert float(w.mean()) == pytest.approx(1.0, abs=1e-4)

    def test_region_weights_upweight_foreground(self):
        target = torch.zeros(1, 32, 32, dtype=torch.long)
        target[:, 10:14, 10:14] = 3
        w = region_weight_map(target, (1, 2, 3), fg_weight=5.0, bg_weight=1.0, dilation=0)
        assert float(w[0, 11, 11]) > float(w[0, 0, 0])

    def test_region_dilation_grows_the_weighted_area(self):
        target = torch.zeros(1, 32, 32, dtype=torch.long)
        target[:, 15:17, 15:17] = 3
        tight = region_weight_map(target, (1, 2, 3), 5.0, 1.0, dilation=0)
        grown = region_weight_map(target, (1, 2, 3), 5.0, 1.0, dilation=3)
        assert (grown > grown.min() + 1e-6).sum() > (tight > tight.min() + 1e-6).sum()

    def test_spatial_feature_loss_is_scale_invariant(self):
        """L2 normalization is what removes the shrink-to-zero direction."""
        s = torch.randn(2, 8, 16, 16)
        t = torch.randn(2, 8, 16, 16)
        base = float(spatial_feature_distillation(s, t))
        scaled = float(spatial_feature_distillation(s * 100.0, t))
        assert scaled == pytest.approx(base, rel=1e-4)

    def test_spatial_feature_loss_zero_for_matching_direction(self):
        t = torch.randn(2, 8, 16, 16)
        assert float(spatial_feature_distillation(t * 3.0, t)) == pytest.approx(0.0, abs=1e-5)

    def test_soft_dice_ignores_absent_classes(self):
        loss = SoftDiceLoss(num_classes=4)
        logits = torch.zeros(1, 4, 8, 8)
        logits[:, 0] = 10.0
        target = torch.zeros(1, 8, 8, dtype=torch.long)
        assert float(loss(logits, target)) < 0.05

    def test_criterion_warmup_ramps_from_zero(self, cfg):
        cfg.kd_warmup_epochs = 4
        criterion = DistillationCriterion(cfg, ("region", "cwd"))
        criterion.set_epoch(0)
        assert criterion._ramp == pytest.approx(0.25)
        criterion.set_epoch(3)
        assert criterion._ramp == pytest.approx(1.0)
        criterion.set_epoch(99)
        assert criterion._ramp == pytest.approx(1.0)

    def test_criterion_rejects_unknown_term(self, cfg):
        with pytest.raises(ValueError):
            DistillationCriterion(cfg, ("nonexistent",))

    def test_criterion_requires_teacher_when_terms_present(self, cfg):
        criterion = DistillationCriterion(cfg, ("region",))
        with pytest.raises(ValueError):
            criterion(torch.randn(1, 4, 8, 8), torch.zeros(1, 8, 8, dtype=torch.long))

    def test_scratch_variant_needs_no_teacher(self, cfg):
        criterion = DistillationCriterion(cfg, ())
        total, parts = criterion(torch.randn(1, 4, 8, 8), torch.zeros(1, 8, 8, dtype=torch.long))
        assert torch.isfinite(total) and set(parts) == {"hard"}


class TestProjectionIsOneSided:
    def test_only_the_student_side_is_learnable(self):
        """Both sides learnable admits W_s = W_t = 0, which drives the loss to
        zero while transferring nothing and collapsing the student's features."""
        projector = SpatialFeatureProjector(student_dim=32, teacher_dim=64)
        teacher_feat = torch.randn(2, 64, 8, 8)
        student_feat = torch.randn(2, 32, 8, 8, requires_grad=True)

        loss = spatial_feature_distillation(projector(student_feat), teacher_feat)
        loss.backward()
        assert teacher_feat.grad is None, "no gradient may reach the teacher feature"
        assert student_feat.grad is not None and torch.isfinite(student_feat.grad).all()

    def test_zeroing_the_projection_does_not_zero_the_loss(self):
        """The degenerate optimum must not exist: with the teacher target fixed,
        a zero projection is maximally wrong, not free."""
        projector = SpatialFeatureProjector(student_dim=32, teacher_dim=64)
        with torch.no_grad():
            for p in projector.parameters():
                p.zero_()
        loss = spatial_feature_distillation(projector(torch.randn(2, 32, 8, 8)),
                                            torch.randn(2, 64, 8, 8))
        assert float(loss.detach()) > 0.5


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

class TestModels:
    def test_student_parameter_count_is_the_reported_figure(self):
        student = build_student(Config())
        assert count_parameters(student) == 487_316

    def test_base32_student_matches_the_conventional_unet(self):
        assert count_parameters(build_student(Config(), 32)) == 7_766_084

    def test_student_output_shapes(self):
        cfg = Config(target_size=(64, 64))
        out = build_student(cfg)(torch.randn(2, 3, 64, 64), return_features=True)
        assert out.logits.shape == (2, 4, 64, 64)
        assert out.feat.shape[-2:] == (16, 16)      # stride 4
        assert out.pooled.shape == (2, 128)

    def test_teacher_emits_full_resolution_logits(self):
        """The correction: without the refinement decoder the teacher predicts at
        stride 4 and cannot represent a lesion a few pixels across."""
        cfg = Config(teacher_variant="b0", teacher_hr_channels=8)
        teacher = build_teacher(cfg, pretrained=False).eval()
        with torch.no_grad():
            out = teacher(torch.randn(1, 3, 64, 64), return_features=True)
        assert out.logits.shape == (1, 4, 64, 64)
        assert out.feat.shape[-2:] == (16, 16)

    def test_frozen_teacher_cannot_be_unfrozen(self, tmp_path):
        cfg = Config(teacher_variant="b0", teacher_hr_channels=8)
        model = build_teacher(cfg, pretrained=False)
        path = tmp_path / "t.pt"
        torch.save(model.state_dict(), path)

        teacher = FrozenTeacher.from_checkpoint(str(path), 4, 3, "b0", True, 8)
        teacher.train(True)
        assert not teacher.training
        assert all(not p.requires_grad for p in teacher.parameters())

    def test_missing_checkpoint_raises_rather_than_using_random_weights(self):
        with pytest.raises(FileNotFoundError):
            FrozenTeacher.from_checkpoint("/nonexistent/teacher.pt", 4, 3, "b0", True, 8)

    def test_b2_preset_uses_the_correct_depths(self):
        from msdistill.models.teacher import MIT_PRESETS
        assert MIT_PRESETS["b2"]["depths"] == (3, 4, 6, 3), (
            "SegformerConfig defaults to (2,2,2,2); using it for B2 builds a 16.3M "
            "model instead of 27.5M and breaks checkpoint loading"
        )


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

class TestData:
    def test_label_remapping_snaps_to_nearest(self):
        remapper = LabelRemapper(np.array([0.0, 16448.0, 49087.0, 65535.0]))
        raw = np.array([0.0, 16448.2, 49086.7, 65535.0, 100.0])
        assert remapper(raw).tolist() == [0, 1, 2, 3, 0]

    def test_normalization_uses_brain_voxels(self):
        volume = np.zeros((32, 32, 4), dtype=np.float32)
        volume[8:24, 8:24, :] = np.random.default_rng(0).normal(500, 50, (16, 16, 4))
        out = clip_and_normalize(volume, 0.5, 99.5)
        brain = out[8:24, 8:24, :]
        assert abs(float(brain.mean())) < 0.5      # brain roughly centred
        assert np.isfinite(out).all()

    def test_augmentation_keeps_labels_discrete_and_aligned(self):
        import random
        image = np.random.default_rng(0).normal(0, 1, (3, 32, 32)).astype(np.float32)
        mask = np.zeros((32, 32), dtype=np.int64)
        mask[10:20, 10:20] = 3
        for seed in range(8):
            aug_img, aug_mask = augment_slice(image, mask, random.Random(seed))
            assert aug_img.shape == image.shape and aug_mask.shape == mask.shape
            assert set(np.unique(aug_mask).tolist()) <= {0, 3}, "interpolation invented labels"
            assert np.isfinite(aug_img).all()

    def test_augmentation_is_reproducible_for_a_given_seed(self):
        """Variants are compared on identical batches; this is what guarantees it."""
        import random
        image = np.random.default_rng(1).normal(0, 1, (3, 32, 32)).astype(np.float32)
        mask = np.zeros((32, 32), dtype=np.int64)
        a = augment_slice(image, mask, random.Random(7))
        b = augment_slice(image, mask, random.Random(7))
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


# --------------------------------------------------------------------------- #
# Config and statistics
# --------------------------------------------------------------------------- #

class TestParallelFolds:
    """Running folds concurrently must change scheduling and nothing else."""

    def test_parallel_folds_does_not_change_the_fingerprint(self):
        """If it did, switching to two GPUs would discard every cached fold."""
        assert Config().fingerprint() == Config(parallel_folds=2).fingerprint()

    def test_device_plan_is_round_robin_and_clamped(self):
        from msdistill.train import plan_fold_devices
        # No CUDA in CI, so every fold lands on cpu; the point is that the plan
        # has one entry per fold and never indexes past the visible devices.
        plan = plan_fold_devices(3, Config(parallel_folds=2))
        assert len(plan) == 3
        assert all(d.type in ("cpu", "cuda") for d in plan)

    def test_concurrent_result_writes_do_not_lose_folds(self, tmp_path):
        """`save_results` is called from several folds; the lock plus atomic
        rename must leave every fold present and the file parseable."""
        import json
        import threading

        from msdistill.train import load_cached_results, save_results

        cfg = Config(output_dir=str(tmp_path)).make_dirs()
        accumulated: List[dict] = []
        lock = threading.Lock()

        def worker(fold: int) -> None:
            for _ in range(20):
                with lock:
                    accumulated[:] = [r for r in accumulated if r["fold"] != fold]
                    accumulated.append({"fold": fold, "best_val_dice": 0.5,
                                        "checkpoint": "", "complete": True})
                    snapshot = list(accumulated)
                save_results(cfg, "concurrent", snapshot)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        restored = load_cached_results(cfg, "concurrent")
        assert {r["fold"] for r in restored} == {0, 1, 2, 3}
        # The file must be valid JSON, not a half-written truncation.
        with open(os.path.join(cfg.results_dir, "concurrent_cv.json"), encoding="utf-8") as f:
            json.load(f)

    def test_seeded_construction_is_serialised(self):
        """Two threads building students concurrently must still get identical
        initial weights for the same seed -- otherwise the ablation's 'same
        initialization' guarantee silently breaks under parallel_folds > 1."""
        import threading

        from msdistill.train import _build_variant_state

        cfg = Config(target_size=(32, 32))
        variant = select_variants(["scratch"])[0]
        device = torch.device("cpu")
        collected: Dict[int, torch.Tensor] = {}
        barrier = threading.Barrier(2)

        def build(idx: int) -> None:
            barrier.wait()                       # maximise the chance of interleaving
            state = _build_variant_state(cfg, variant, None, device, seed=1234)
            collected[idx] = state.model.enc1.block[0].weight.detach().clone()

        threads = [threading.Thread(target=build, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert torch.allclose(collected[0], collected[1]), (
            "concurrent construction produced different initial weights; the RNG "
            "lock is not holding"
        )


class TestConfig:
    def test_fingerprint_changes_with_semantic_fields(self):
        assert Config().fingerprint() != Config(kd_fg_weight=9.0).fingerprint()
        assert Config().fingerprint() != Config(teacher_variant="b1").fingerprint()

    def test_fingerprint_ignores_non_semantic_fields(self):
        """Changing an output path or worker count must not force a retrain."""
        assert Config().fingerprint() == Config(output_dir="/elsewhere",
                                                num_workers=8).fingerprint()

    def test_variant_rejects_mutually_exclusive_terms(self):
        with pytest.raises(ValueError):
            Variant("bad", "Bad", ("logit", "region"), True)

    def test_ablation_grid_contains_the_required_rows(self):
        keys = {v.key for v in ABLATION_VARIANTS}
        assert {"unet32", "scratch", "kd_vanilla", "kd_fitnets", "kd_cwd", "kd_full"} <= keys

    def test_prior_art_rows_carry_citations(self):
        for key in ("kd_vanilla", "kd_fitnets", "kd_cwd"):
            variant = select_variants([key])[0]
            assert variant.citation, f"{key} reproduces published work and must cite it"

    def test_invalid_teacher_variant_rejected(self):
        with pytest.raises(ValueError):
            Config(teacher_variant="b7")


class TestStats:
    def test_pairing_drops_only_undefined_pairs(self):
        a = [0.5, float("nan"), 0.7, 0.9]
        b = [0.4, 0.6, float("nan"), 0.8]
        va, vb = paired_valid(a, b)
        assert va.tolist() == [0.5, 0.9] and vb.tolist() == [0.4, 0.8]

    def test_identical_vectors_are_not_significant(self):
        values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.55]
        result = wilcoxon_compare(values, values, "a", "b", "dice", "abnormal_wmh",
                                  n_resamples=200)
        assert result.p_value == pytest.approx(1.0)
        assert result.mean_difference == pytest.approx(0.0)
        assert result.n_effective == 0

    def test_minimum_sample_guard_counts_non_tied_pairs(self):
        """`zero_method="wilcox"` deletes ties and tests at the reduced size, so
        gating on the raw pair count let 18 ties plus 2 differing pairs emit a
        normal-approximation p-value whose exact minimum is 0.5."""
        baseline = [0.5] * 20
        method = list(baseline)
        method[0] += 0.2
        method[1] += 0.2                      # only 2 non-tied pairs out of 20

        r = wilcoxon_compare(method, baseline, "m", "b", "dice", "abnormal_wmh",
                             n_resamples=200)
        assert r.n_pairs == 20 and r.n_effective == 2
        assert np.isnan(r.p_value), "an effective n of 2 must not yield a p-value"

    def test_enough_non_tied_pairs_does_produce_a_p_value(self):
        baseline = [0.5] * 20
        method = [v + (0.1 if i < 6 else 0.0) for i, v in enumerate(baseline)]
        r = wilcoxon_compare(method, baseline, "m", "b", "dice", "abnormal_wmh",
                             n_resamples=200)
        assert r.n_effective == 6 and not np.isnan(r.p_value)

    def test_consistent_improvement_is_detected(self):
        baseline = [0.40, 0.45, 0.50, 0.55, 0.60, 0.42, 0.48, 0.52]
        better = [v + 0.10 for v in baseline]
        result = wilcoxon_compare(better, baseline, "ours", "base", "dice", "abnormal_wmh",
                                  n_resamples=500)
        assert result.mean_difference == pytest.approx(0.10, abs=1e-6)
        assert result.p_value < 0.05
        assert result.effect_size == pytest.approx(1.0)
        assert result.ci_low > 0.0

    def test_holm_adjustment_is_monotone_and_conservative(self):
        results = [
            wilcoxon_compare([0.5] * 8, [0.5] * 8, f"m{i}", "b", "dice", "c", n_resamples=100)
            for i in range(3)
        ]
        for r, p in zip(results, (0.01, 0.02, 0.04)):
            r.p_value = p
        adjusted = [r.p_adjusted for r in holm_bonferroni(results, alpha=0.05)]
        assert all(a >= b for a, b in zip(adjusted, [0.01, 0.02, 0.04]))
        assert adjusted == sorted(adjusted)


class TestGapRecovery:
    def test_undefined_when_the_teacher_does_not_lead(self):
        """Reporting 'recovered X% of the gap' with a negative gap is a standard
        way to make a failed distillation look successful."""
        from msdistill.evaluate import ModelEvaluation, teacher_student_gap_recovery

        def make(name, dice):
            ev = ModelEvaluation(name=name, label=name, n_params=1)
            ev.summary = {"dice": {"abnormal_wmh": {"mean": dice, "std": 0.0, "n": 5}}}
            return ev

        teacher_weak = make("teacher", 0.50)
        scratch = make("scratch", 0.60)
        variant = make("kd", 0.65)
        assert np.isnan(teacher_student_gap_recovery(teacher_weak, scratch, variant,
                                                     "abnormal_wmh"))

        teacher_strong = make("teacher", 0.80)
        assert teacher_student_gap_recovery(teacher_strong, scratch, variant,
                                            "abnormal_wmh") == pytest.approx(0.25)
