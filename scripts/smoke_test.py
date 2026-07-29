#!/usr/bin/env python
"""End-to-end smoke test on synthetic data.

Builds a miniature dataset with the same directory layout, NIfTI structure and
16-bit label encoding as MS3SEG, then runs every stage of the real pipeline at
toy scale: discovery, label inference, caching, patient-level splitting, teacher
fine-tuning, co-trained student ablation, held-out evaluation, significance
testing, efficiency profiling, LaTeX generation and figure rendering.

It runs on CPU in a couple of minutes and needs no network access, so it can gate
every change. If this passes, the notebook will not fail on a syntax error, a
shape mismatch or a missing key after an hour of GPU training.

    python scripts/smoke_test.py [--keep]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
import traceback
from typing import Dict, List

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import nibabel as nib
import torch

from msdistill.config import (LADDER_KEYS, Config, select_variants)
from msdistill.data import (LabelRemapper, PatientVolumeCache, build_patient_index,
                            infer_label_values, make_fold_splits, split_test_and_dev,
                            summarize_class_balance)
from msdistill.efficiency import environment_summary, profile_model
from msdistill.evaluate import (evaluate_with_single_model_stats,
                                load_ensemble, save_evaluations, teacher_student_gap_recovery)
from msdistill.models.student import build_student, count_parameters, student_builder
from msdistill.models.teacher import FrozenTeacher, build_teacher
from msdistill.report import (MS3SEG_PUBLISHED, ablation_table, build_macros, dataset_table,
                              detection_table, efficiency_table, seed_variance_table,
                              significance_table, standard_extra_macros, write_macros,
                              write_markdown_summary, write_table)
from msdistill.stats import compare_all, ladder_comparisons
from msdistill.train import (checkpoint_path, get_device, run_student_stage, set_seed)
from msdistill.viz import (plot_ablation_bars, plot_accuracy_vs_cost, plot_class_balance,
                           plot_paired_differences, plot_qualitative, plot_training_curves)

# The raw 16-bit values the distributed masks use for the four labels.
RAW_LABEL_VALUES = (0.0, 16448.0, 49087.0, 65535.0)


def synthesize_dataset(root: str, n_patients: int = 12, n_slices: int = 6,
                       size: int = 64, seed: int = 0) -> None:
    """Write a miniature cohort in the MS3SEG directory layout."""
    rng = np.random.default_rng(seed)
    images_dir = os.path.join(root, "MS_100_patient_registered")
    masks_dir = os.path.join(root, "MS_100_model_input", "man_4L_masks_new")
    os.makedirs(masks_dir, exist_ok=True)

    yy, xx = np.mgrid[0:size, 0:size]
    affine = np.diag([1.0, 1.0, 3.0, 1.0])

    for p in range(n_patients):
        pid = f"{p + 1:03d}"
        pdir = os.path.join(images_dir, pid)
        os.makedirs(pdir, exist_ok=True)

        labels = np.zeros((size, size, n_slices), dtype=np.float32)
        centre = size / 2.0
        brain = ((xx - centre) ** 2 + (yy - centre) ** 2) < (0.42 * size) ** 2

        for s in range(n_slices):
            sl = np.zeros((size, size), dtype=np.int64)
            # Ventricles: a central blob, present on most slices.
            vent = ((xx - centre) ** 2 / 1.8 + (yy - centre) ** 2) < (0.10 * size) ** 2
            sl[vent & brain] = 1
            # A handful of small hyperintensities of each pathological kind.
            for class_idx, count in ((2, rng.integers(1, 4)), (3, rng.integers(0, 3))):
                for _ in range(int(count)):
                    cy = rng.integers(int(0.2 * size), int(0.8 * size))
                    cx = rng.integers(int(0.2 * size), int(0.8 * size))
                    r = rng.integers(2, 4)
                    blob = ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2
                    sl[blob & brain] = class_idx
            labels[:, :, s] = sl

        raw = np.zeros_like(labels)
        for idx, value in enumerate(RAW_LABEL_VALUES):
            raw[labels == idx] = value
        nib.save(nib.Nifti1Image(raw, affine), os.path.join(masks_dir, f"{pid}.nii"))

        # Modality intensities correlate with the label map so the task is learnable.
        for fname, gain in ((f"{pid}_T1WI_reg.nii", -0.6),
                            (f"{pid}_T2WI_reg.nii", 0.8),
                            (f"{pid}_FLAIR.nii", 1.4)):
            vol = np.zeros((size, size, n_slices), dtype=np.float32)
            for s in range(n_slices):
                base = 300.0 * brain + rng.normal(0, 12, (size, size))
                sl = labels[:, :, s]
                base = base + gain * 220.0 * (sl == 2) + gain * 300.0 * (sl == 3)
                base = base - 0.4 * gain * 180.0 * (sl == 1)
                vol[:, :, s] = np.clip(base, 0, None)
            nib.save(nib.Nifti1Image(vol, affine), os.path.join(pdir, fname))


def build_config(root: str, out: str) -> Config:
    cfg = Config(
        data_root=root,
        output_dir=out,
        target_size=(64, 64),
        n_folds=2,
        test_fraction=0.25,
        batch_size=4,
        num_epochs=2,
        early_stop_patience=5,
        teacher_epochs=2,
        teacher_early_stop_patience=5,
        teacher_variant="b0",
        teacher_hr_channels=16,
        num_workers=0,
        use_amp=False,
        progress=False,
        kd_warmup_epochs=1,
        student_warmup_epochs=0,
        teacher_warmup_epochs=0,
        bootstrap_samples=200,
        benchmark_warmup=1,
        benchmark_trials=3,
        benchmark_batch_sizes=(1, 4),
        benchmark_cpu=True,
        student_seeds=(42, 1337),
        lesion_oversample_factor=2.0,
    )
    return cfg.make_dirs()


def train_toy_teachers(cfg: Config, fold_splits, cache, device) -> Dict[int, FrozenTeacher]:
    """Fine-tune one tiny, randomly initialized teacher per fold.

    `pretrained=False` keeps the smoke test offline. The real pipeline always
    downloads pretrained weights; this only exercises the code path.
    """
    from msdistill.losses import HardLabelLoss
    from msdistill.data import make_dataloaders

    teachers: Dict[int, FrozenTeacher] = {}
    for fold, split in enumerate(fold_splits):
        set_seed(cfg.seed + fold)
        model = build_teacher(cfg, pretrained=False).to(device)
        loader, _ = make_dataloaders(split["train"], split["val"], cache, cfg, seed=cfg.seed)
        criterion = HardLabelLoss(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        for epoch in range(cfg.teacher_epochs):
            loader.dataset.set_epoch(epoch)
            for images, masks in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images.to(device)), masks.to(device))
                loss.backward()
                optimizer.step()
        path = checkpoint_path(cfg, "teacher", fold)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
        teachers[fold] = FrozenTeacher.from_checkpoint(
            path, cfg.num_classes, len(cfg.modalities), cfg.teacher_variant,
            cfg.teacher_hr_decoder, cfg.teacher_hr_channels).to(device).eval()
    return teachers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="keep the temporary workspace")
    args = parser.parse_args()

    workspace = tempfile.mkdtemp(prefix="msdistill_smoke_")
    data_root = os.path.join(workspace, "data")
    out_dir = os.path.join(workspace, "out")
    checks: List[str] = []

    try:
        print(f"workspace: {workspace}")
        device = get_device()
        print(f"device: {device}")

        # -- 1. synthetic cohort ------------------------------------------
        synthesize_dataset(data_root)
        cfg = build_config(data_root, out_dir)
        checks.append("config + fingerprint " + cfg.fingerprint())

        # -- 2. discovery and label inference ------------------------------
        index = build_patient_index(cfg.data_root, cfg.modalities)
        assert len(index) == 12, f"expected 12 patients, found {len(index)}"
        raw_values = infer_label_values([e["mask"] for e in index.values()], cfg.num_classes)
        assert np.allclose(sorted(raw_values), sorted(RAW_LABEL_VALUES)), \
            f"label inference returned {raw_values}"
        remapper = LabelRemapper(raw_values)
        checks.append(f"discovered {len(index)} patients; inferred labels {raw_values.tolist()}")

        # -- 3. splits and cache -------------------------------------------
        dev_ids, test_ids = split_test_and_dev(sorted(index), cfg.test_fraction, cfg.seed)
        fold_splits = make_fold_splits(dev_ids, cfg.n_folds, cfg.seed)
        assert not (set(dev_ids) & set(test_ids)), "test patients leaked into development"
        for f, split in enumerate(fold_splits):
            assert not (set(split["train"]) & set(split["val"])), f"fold {f} train/val overlap"
        cache = PatientVolumeCache(sorted(index), index, remapper, cfg, verbose=False)
        balance = summarize_class_balance(cache, cfg)
        assert abs(sum(balance.values()) - 1.0) < 1e-6
        checks.append(f"{len(dev_ids)} dev / {len(test_ids)} test patients, "
                      f"class balance {[f'{v:.3f}' for v in balance.values()]}")

        # -- 4. teachers ----------------------------------------------------
        teachers = train_toy_teachers(cfg, fold_splits, cache, device)
        checks.append(f"trained {len(teachers)} fold-matched teachers")

        # verify the teacher really is frozen
        teacher = teachers[0]
        teacher.train(True)
        assert not teacher.training, "FrozenTeacher.train(True) must not enable training mode"
        assert all(not p.requires_grad for p in teacher.parameters()), "teacher params not frozen"
        checks.append("teacher freeze contract holds")

        # -- 5. co-trained student ablation ---------------------------------
        variants = select_variants()
        results = run_student_stage(cfg, variants, fold_splits, cache, teachers, device)
        for v in variants:
            assert len(results[v.key]) == cfg.n_folds, f"{v.key}: missing folds"
        checks.append(f"co-trained {len(variants)} variants x {cfg.n_folds} folds")

        # -- 6. held-out evaluation ------------------------------------------
        evaluations = {}
        predictions: Dict[str, np.ndarray] = {}
        store_for = test_ids[:2]

        teacher_models = [
            FrozenTeacher.from_checkpoint(
                checkpoint_path(cfg, "teacher", f), cfg.num_classes, len(cfg.modalities),
                cfg.teacher_variant, cfg.teacher_hr_decoder, cfg.teacher_hr_channels
            ).to(device).eval() for f in range(cfg.n_folds)
        ]
        evaluations["teacher"] = evaluate_with_single_model_stats(
            teacher_models, cache, test_ids, cfg, device, "teacher",
            f"Teacher (MiT-{cfg.teacher_variant.upper()}+HR)",
            count_parameters(teacher_models[0].model, trainable_only=False),
            store_predictions_for=store_for, prediction_store=predictions)
        evaluations["teacher"].objective_note = "hard labels"

        for v in variants:
            models = load_ensemble(student_builder(v.base_channels),
                                   [r["checkpoint"] for r in results[v.key]], device, cfg)
            ev = evaluate_with_single_model_stats(
                models, cache, test_ids, cfg, device, v.key, v.label,
                count_parameters(models[0]),
                store_predictions_for=store_for, prediction_store=predictions)
            ev.objective_note = v.objective
            ev.citation = v.citation
            evaluations[v.key] = ev

        # The single-model statistics must be populated for every ensembled model;
        # without them the accuracy table and the efficiency table describe
        # different systems.
        for key, ev in evaluations.items():
            assert ev.single_model_dice, f"{key}: no single-model statistics recorded"

        for key, ev in evaluations.items():
            assert len(ev.cases) == len(test_ids), f"{key}: wrong number of evaluated cases"
            assert [c.patient_id for c in ev.cases] == list(test_ids), \
                f"{key}: case order does not match the test list -- pairing would be invalid"
            assert "dice" in ev.summary, f"{key}: no dice summary"
        checks.append(f"evaluated {len(evaluations)} models on {len(test_ids)} test patients")

        # the empty-class convention must not award a free 1.0
        from msdistill.metrics import dice_binary
        empty = np.zeros((4, 4), dtype=bool)
        assert np.isnan(dice_binary(empty, empty)), "empty/empty Dice must be undefined"
        assert dice_binary(~empty, empty) == 0.0, "false-positive-only Dice must be 0"
        checks.append("empty-class Dice convention verified")

        # -- 7. statistics ----------------------------------------------------
        primary = cfg.class_names[cfg.primary_class]
        comparisons = {}
        for class_idx in cfg.foreground_classes:
            name = cfg.class_names[class_idx]
            comparisons[name] = compare_all(evaluations, "scratch", "dice", name, cfg,
                                            exclude=["teacher"])
        ladder = ladder_comparisons(evaluations, list(LADDER_KEYS), "dice", primary, cfg)
        assert all(0.0 <= c.p_value <= 1.0 or np.isnan(c.p_value)
                   for c in comparisons[primary]), "invalid p-values"
        checks.append(f"{sum(len(v) for v in comparisons.values())} paired comparisons + "
                      f"{len(ladder)} ladder steps")

        gap = teacher_student_gap_recovery(evaluations["teacher"], evaluations["scratch"],
                                           evaluations["kd_full"], primary)
        checks.append(f"gap recovery on {primary}: {gap:.3f}" if not np.isnan(gap)
                      else "gap recovery undefined (teacher did not beat scratch) -- handled")

        # -- 8. efficiency -----------------------------------------------------
        eff_teacher = profile_model(build_teacher(cfg, pretrained=False).to(device), "Teacher",
                                    cfg, device, checkpoint=checkpoint_path(cfg, "teacher", 0))
        eff_unet32 = profile_model(build_student(cfg, 32).to(device), "U-Net (base 32)", cfg,
                                   device, checkpoint=results["unet32"][0]["checkpoint"])
        eff_student = profile_model(build_student(cfg).to(device), "Student", cfg, device,
                                    checkpoint=results["scratch"][0]["checkpoint"])
        # The smoke test uses the b0 teacher to stay offline, and b0 (~3.7M) is
        # *smaller* than the base-32 U-Net (~7.8M) -- which is precisely why b0 is
        # not a usable teacher for a compression paper and why the reported
        # configuration is b2 (~27M). Only the student ordering is asserted here.
        assert eff_student.params_total < eff_unet32.params_total, \
            "the compact student must be smaller than the base-32 U-Net"
        assert eff_student.params_total < eff_teacher.params_total, \
            "the student must be smaller than the teacher"
        assert eff_student.gpu_latency_ms, "no latency samples recorded"
        assert eff_student.cpu_latency_ms, "no CPU latency recorded -- it is the headline metric"
        checks.append(f"efficiency: teacher {eff_teacher.params_total:,}p "
                      f"{eff_teacher.gmacs:.3f}GMACs via {eff_teacher.flops_method}; "
                      f"unet32 {eff_unet32.params_total:,}p; "
                      f"student {eff_student.params_total:,}p {eff_student.gmacs:.3f}GMACs")

        # -- 9. LaTeX ----------------------------------------------------------
        ordered = ["teacher"] + [v.key for v in variants]
        eff_reports = [eff_teacher, eff_unet32, eff_student]
        tables = {
            "table_ablation.tex": ablation_table(evaluations, ordered, cfg, comparisons,
                                                 published=MS3SEG_PUBLISHED),
            "table_detection.tex": detection_table(evaluations, ordered, cfg),
            "table_efficiency.tex": efficiency_table(eff_reports, "Teacher", cfg,
                                                     ensemble_size=cfg.n_folds,
                                                     environment=environment_summary(device),
                                                     slices_per_volume=cache.n_slices(test_ids[0])),
            "table_significance.tex": significance_table(comparisons[primary] + ladder, cfg),
            "table_dataset.tex": dataset_table(balance, len(index), len(test_ids), cfg.n_folds,
                                               cache.n_slices(test_ids[0]), cfg),
            "table_seeds.tex": seed_variance_table(
                {v.key: {s: evaluations[v.key].dice(primary) for s in cfg.student_seeds}
                 for v in variants}, cfg),
        }
        for fname, content in tables.items():
            path = write_table(content, os.path.join(cfg.tables_dir, fname))
            assert os.path.getsize(path) > 100, f"{fname} looks empty"
            assert content.count(r"\begin{tabular}") == content.count(r"\end{tabular}"), \
                f"{fname}: unbalanced tabular environment"

        macros = build_macros(
            evaluations, eff_reports, comparisons, cfg,
            extra=standard_extra_macros(cfg, len(index), len(dev_ids), len(test_ids),
                                        cache.n_slices(test_ids[0]), balance, eff_reports),
            ladder=ladder,
        )
        write_macros(macros, os.path.join(cfg.tables_dir, "results_macros.tex"))
        assert len(macros) > 30, f"only {len(macros)} macros generated"
        for name in macros:
            assert name.isalpha(), f"macro {name!r} contains characters LaTeX will reject"

        # The manuscript cites measured values through these macros. If the
        # pipeline does not emit one, the compiled PDF shows a red [PENDING]
        # marker where a number should be -- catch that here, not at submission.
        paper_tex = os.path.join(os.path.dirname(REPO_ROOT), "main.tex")
        if os.path.exists(paper_tex):
            cited = set(re.findall(r"\\(Res[A-Za-z]+)",
                                   open(paper_tex, encoding="utf-8").read()))
            uncovered = sorted(cited - set(macros) - {"ResPending"})
            assert not uncovered, (
                f"main.tex cites {len(uncovered)} macro(s) the pipeline never "
                f"produces: {uncovered}")
            checks.append(f"all {len(cited)} macros cited by main.tex are produced")
        else:
            checks.append("main.tex not found beside the repo; macro coverage not checked")
        write_markdown_summary(evaluations, ordered, cfg,
                               os.path.join(cfg.results_dir, "RESULTS.md"))
        checks.append(f"{len(tables)} LaTeX tables + {len(macros)} prose macros")

        # -- 10. figures --------------------------------------------------------
        figs = [
            plot_training_curves({v.key: results[v.key] for v in variants[:4]},
                                 evaluations["teacher"].mean_foreground_dice(),
                                 os.path.join(cfg.figures_dir, "training_curves.png")),
            plot_ablation_bars(evaluations, ordered, cfg,
                               os.path.join(cfg.figures_dir, "ablation.png")),
            plot_paired_differences(evaluations, "kd_full", "scratch", primary,
                                    os.path.join(cfg.figures_dir, "paired.png")),
            plot_accuracy_vs_cost(
                {k: evaluations[k] for k in ("teacher", "unet32", "scratch", "kd_full")},
                {"teacher": eff_teacher, "unet32": eff_unet32,
                 "scratch": eff_student, "kd_full": eff_student}, cfg,
                os.path.join(cfg.figures_dir, "accuracy_vs_cost.png")),
            plot_class_balance(balance, os.path.join(cfg.figures_dir, "class_balance.png")),
            plot_qualitative(cache, predictions, store_for, [2, 2],
                             [("teacher", "Teacher"), ("scratch", "Scratch"),
                              ("kd_vanilla", "Hinton KD"), ("kd_full", "Proposed")],
                             cfg, os.path.join(cfg.figures_dir, "qualitative.png")),
        ]
        for f in figs:
            assert os.path.exists(f) and os.path.getsize(f) > 2000, f"figure {f} is empty"
        checks.append(f"{len(figs)} figures rendered")

        save_evaluations(cfg, evaluations)
        cfg.to_json(os.path.join(cfg.results_dir, "config.json"))
        checks.append(f"environment: {environment_summary(device)}")

        print("\n" + "=" * 72)
        for i, c in enumerate(checks, 1):
            print(f"  {i:>2}. {c}")
        print("=" * 72)
        print("SMOKE TEST PASSED")
        return 0

    except Exception:
        print("\n" + "=" * 72)
        for i, c in enumerate(checks, 1):
            print(f"  {i:>2}. {c}")
        print("=" * 72)
        print("SMOKE TEST FAILED\n")
        traceback.print_exc()
        return 1
    finally:
        if args.keep:
            print(f"\nworkspace kept at {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
