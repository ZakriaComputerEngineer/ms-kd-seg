#!/usr/bin/env python
"""Run the whole study headlessly.

The notebook is the primary interface because the experiments were run on Kaggle,
but a notebook is the wrong tool for a machine you can SSH into: it cannot be
resumed from a shell, its output is not greppable, and `nohup` does not apply.
This runs the identical pipeline as a script.

    python scripts/run_all.py --data-root /data/ms3seg --output-dir runs/b2

    # tighter budget
    python scripts/run_all.py --data-root /data/ms3seg --quick --teacher b1

    # stages can be run separately; each resumes from its own checkpoints
    python scripts/run_all.py --data-root /data/ms3seg --stages teacher
    python scripts/run_all.py --data-root /data/ms3seg --stages students,evaluate,report

Every stage is resumable and guarded by the same configuration fingerprint as the
notebook, so interrupting and rerunning continues rather than restarting.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from msdistill.config import (LADDER_KEYS, QUICK_VARIANT_KEYS, SEED_STUDY_KEYS, Config,
                              select_variants)
from msdistill.data import (LabelRemapper, PatientVolumeCache, build_patient_index,
                            infer_label_values, locate_dataset, make_fold_splits,
                            split_test_and_dev, summarize_class_balance)
from msdistill.efficiency import environment_summary, profile_model
from msdistill.evaluate import (evaluate_ensemble, evaluate_with_single_model_stats,
                                load_ensemble, save_evaluations, teacher_student_gap_recovery)
from msdistill.models.student import build_student, count_parameters, student_builder
from msdistill.models.teacher import build_teacher, load_frozen_teacher
from msdistill.report import (MS3SEG_PUBLISHED, ablation_table, build_macros, dataset_table,
                              detection_table, efficiency_table, seed_variance_table,
                              significance_table, standard_extra_macros, write_macros,
                              write_markdown_summary, write_table)
from msdistill.stats import compare_all, format_p, ladder_comparisons, significance_marker
from msdistill.train import (get_device, run_student_stage, run_teacher_stage, set_seed)
from msdistill.viz import (plot_ablation_bars, plot_accuracy_vs_cost, plot_class_balance,
                           plot_paired_differences, plot_qualitative, plot_training_curves)

ALL_STAGES = ("teacher", "students", "evaluate", "report", "seeds")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True,
                   help="directory containing MS_100_patient_registered/")
    p.add_argument("--output-dir", default="runs/default")
    p.add_argument("--teacher", choices=["b0", "b1", "b2"], default="b2")
    p.add_argument("--no-hr", action="store_true",
                   help="disable the full-resolution decoder. Do not use: a stride-4 "
                        "teacher is weaker than the student and the study becomes meaningless.")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=None, help="student epoch budget")
    p.add_argument("--teacher-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-hours-per-fold", type=float, default=3.0)
    p.add_argument("--parallel-folds", type=int, default=1,
                   help="run this many folds concurrently, one per GPU (2 for Kaggle T4 x2)")
    p.add_argument("--quick", action="store_true",
                   help=f"reduced variant grid: {', '.join(QUICK_VARIANT_KEYS)}")
    p.add_argument("--stages", default=",".join(ALL_STAGES),
                   help=f"comma-separated subset of {ALL_STAGES}")
    p.add_argument("--no-progress", action="store_true", help="suppress progress bars")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    kwargs = dict(
        data_root=args.data_root,
        output_dir=args.output_dir,
        teacher_variant=args.teacher,
        teacher_hr_decoder=not args.no_hr,
        n_folds=args.folds,
        batch_size=args.batch_size,
        num_workers=args.workers,
        max_hours_per_fold=args.max_hours_per_fold,
        parallel_folds=args.parallel_folds,
        progress=not args.no_progress,
    )
    if args.epochs is not None:
        kwargs["num_epochs"] = args.epochs
    if args.teacher_epochs is not None:
        kwargs["teacher_epochs"] = args.teacher_epochs
    return Config(**kwargs).make_dirs()


def banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}", flush=True)


def main() -> int:
    args = parse_args()
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    unknown = stages - set(ALL_STAGES)
    if unknown:
        print(f"unknown stage(s): {sorted(unknown)}; valid: {ALL_STAGES}")
        return 2

    if args.no_hr:
        print("WARNING: --no-hr disables the full-resolution decoder. The teacher will "
              "predict at stride 4 and is expected to score BELOW the student on small "
              "lesions. This exists to reproduce the failure, not to run the study.\n")

    cfg = build_config(args)
    device = get_device()
    set_seed(cfg.seed)

    banner(f"MS lesion distillation | teacher MiT-{cfg.teacher_variant.upper()}"
           f"{'+HR' if cfg.teacher_hr_decoder else ''} | device {device}")
    print(f"fingerprint {cfg.fingerprint()}   output {cfg.output_dir}")
    print(f"environment {environment_summary(device)}")
    cfg.to_json(os.path.join(cfg.results_dir, "config.json"))
    started = time.time()

    # ---------------------------------------------------------------- data
    banner("Data")
    cfg.data_root = locate_dataset(cfg.data_root)
    patient_index = build_patient_index(cfg.data_root, cfg.modalities)
    remapper = LabelRemapper(infer_label_values(
        [e["mask"] for e in patient_index.values()], cfg.num_classes))
    dev_patients, test_patients = split_test_and_dev(sorted(patient_index),
                                                     cfg.test_fraction, cfg.seed)
    fold_splits = make_fold_splits(dev_patients, cfg.n_folds, cfg.seed)
    assert not (set(dev_patients) & set(test_patients))

    cache = PatientVolumeCache(sorted(patient_index), patient_index, remapper, cfg,
                               verbose=cfg.progress)
    class_balance = summarize_class_balance(cache, cfg)
    slices_per_volume = cache.n_slices(test_patients[0])
    print(f"{len(patient_index)} patients | {len(dev_patients)} dev / {len(test_patients)} test "
          f"| cache {cache.memory_bytes() / 1e9:.2f} GB")
    print("class balance: " + ", ".join(f"{k}={100 * v:.2f}%" for k, v in class_balance.items()))
    plot_class_balance(class_balance, os.path.join(cfg.figures_dir, "fig_class_balance.png"))

    variants = select_variants(list(QUICK_VARIANT_KEYS) if args.quick else None)
    primary_class = cfg.class_names[cfg.primary_class]

    # ------------------------------------------------------------- teacher
    teacher_results: List[Dict] = []
    if "teacher" in stages:
        banner("Stage 0 - teacher (one per fold)")
        teacher_results = run_teacher_stage(cfg, fold_splits, cache, device)
    else:
        from msdistill.train import load_cached_results
        teacher_results = load_cached_results(cfg, "teacher")
        if not teacher_results:
            print("no cached teachers; run --stages teacher first")
            return 1

    teachers_by_fold = {r["fold"]: load_frozen_teacher(cfg, r["checkpoint"], device)
                        for r in teacher_results}

    # ------------------------------------------------------------- students
    student_results: Dict[str, List[Dict]] = {}
    if "students" in stages:
        banner(f"Stage 1 - {len(variants)} co-trained student variants")
        student_results = run_student_stage(cfg, variants, fold_splits, cache,
                                            teachers_by_fold, device)
        curve_keys = [k for k in ("scratch", "kd_vanilla", "kd_cwd", "kd_region", "kd_full")
                      if k in student_results]
        plot_training_curves(
            {k: student_results[k] for k in curve_keys},
            float(np.mean([r["best_val_dice"] for r in teacher_results])),
            os.path.join(cfg.figures_dir, "fig_training_curves.png"))
    else:
        from msdistill.train import load_cached_results
        student_results = {v.key: load_cached_results(cfg, v.key) for v in variants}
        if any(not r for r in student_results.values()):
            print("missing cached student folds; run --stages students first")
            return 1

    if "evaluate" not in stages:
        print(f"\ndone in {(time.time() - started) / 3600:.2f} h")
        return 0

    # ----------------------------------------------------------- evaluation
    banner("Held-out evaluation")
    evaluations, predictions = {}, {}
    qualitative = test_patients[:3]

    teacher_models = [load_frozen_teacher(cfg, r["checkpoint"], device) for r in teacher_results]
    evaluations["teacher"] = evaluate_with_single_model_stats(
        teacher_models, cache, test_patients, cfg, device, "teacher",
        f"Teacher (MiT-{cfg.teacher_variant.upper()}" + ("+HR)" if cfg.teacher_hr_decoder else ")"),
        count_parameters(teacher_models[0].model, trainable_only=False),
        store_predictions_for=qualitative, prediction_store=predictions)
    evaluations["teacher"].objective_note = "hard labels"

    for v in variants:
        models = load_ensemble(student_builder(v.base_channels),
                               [r["checkpoint"] for r in student_results[v.key]], device, cfg)
        ev = evaluate_with_single_model_stats(
            models, cache, test_patients, cfg, device, v.key, v.label,
            count_parameters(models[0]),
            store_predictions_for=qualitative, prediction_store=predictions)
        ev.objective_note, ev.citation = v.objective, v.citation
        evaluations[v.key] = ev

    ordered_keys = ["teacher"] + [v.key for v in variants]
    print(f"\n{'model':<34}{'params':>10}{'normal WMH':>14}{'abnormal WMH':>14}{'mean FG':>9}")
    for key in ordered_keys:
        ev = evaluations[key]
        print(f"{ev.label[:33]:<34}{ev.n_params / 1e6:>9.2f}M"
              f"{ev.dice('normal_wmh'):>14.3f}{ev.dice('abnormal_wmh'):>14.3f}"
              f"{ev.mean_foreground_dice():>9.3f}")

    # If the teacher does not lead, the compression framing does not hold and the
    # paper needs the cross-architecture-transfer reframing instead. Better to see
    # it here than in review.
    t = evaluations["teacher"].dice(primary_class)
    s = evaluations["scratch"].dice(primary_class)
    verdict = ("teacher leads; compression framing holds" if t > s else
               "TEACHER DOES NOT LEAD -- see docs/DESIGN.md section 4 for the reframing")
    print(f"\npremise check on {primary_class}: teacher {t:.3f} vs scratch {s:.3f} -> {verdict}")
    save_evaluations(cfg, evaluations)

    # ---------------------------------------------------------- statistics
    banner("Significance")
    comparisons = {cfg.class_names[c]: compare_all(evaluations, "scratch", "dice",
                                                   cfg.class_names[c], cfg, exclude=["teacher"])
                   for c in cfg.foreground_classes}
    ladder = ladder_comparisons(evaluations, [k for k in LADDER_KEYS if k in evaluations],
                                "dice", primary_class, cfg)
    for c in comparisons[primary_class]:
        print(f"  {c.name_a:<16} vs scratch  delta={c.mean_difference:+.3f} "
              f"CI[{c.ci_low:+.3f},{c.ci_high:+.3f}] p={format_p(c.p_adjusted)}"
              f"{significance_marker(c)}")
    print("  -- ladder --")
    for c in ladder:
        print(f"  {c.name_b:<16} -> {c.name_a:<16} delta={c.mean_difference:+.3f} "
              f"p={format_p(c.p_adjusted)}{significance_marker(c)}")

    gap = teacher_student_gap_recovery(evaluations["teacher"], evaluations["scratch"],
                                       evaluations["kd_full"], primary_class)
    print("\ngap recovered by kd_full: "
          + (f"{100 * gap:.1f}%" if not np.isnan(gap) else "undefined (teacher does not lead)"))

    # ---------------------------------------------------------- efficiency
    banner("Efficiency")
    efficiency_reports = [
        profile_model(build_teacher(cfg, pretrained=False).to(device), "Teacher", cfg, device,
                      checkpoint=teacher_results[0]["checkpoint"],
                      slices_per_volume=slices_per_volume),
        profile_model(build_student(cfg, 32).to(device), "U-Net (base 32)", cfg, device,
                      slices_per_volume=slices_per_volume),
        profile_model(build_student(cfg).to(device), "Student", cfg, device,
                      checkpoint=student_results["scratch"][0]["checkpoint"],
                      slices_per_volume=slices_per_volume),
    ]
    for r in efficiency_reports:
        print(f"  {r.name:<18}{r.params_total:>12,}p {r.gmacs:>7.2f} GMACs  "
              f"GPU@b1 {r.gpu_latency_ms.get(1, float('nan')):>7.2f} ms  "
              f"CPU@b1 {r.cpu_latency_ms.get(1, float('nan')):>8.1f} ms")

    if "report" not in stages:
        print(f"\ndone in {(time.time() - started) / 3600:.2f} h")
        return 0

    # -------------------------------------------------------------- report
    banner("Paper artefacts")
    tables = {
        "table_ablation.tex": ablation_table(evaluations, ordered_keys, cfg, comparisons,
                                             published=MS3SEG_PUBLISHED),
        "table_detection.tex": detection_table(evaluations, ordered_keys, cfg),
        "table_efficiency.tex": efficiency_table(efficiency_reports, "Teacher", cfg,
                                                 ensemble_size=cfg.n_folds,
                                                 environment=environment_summary(device),
                                                 slices_per_volume=slices_per_volume),
        "table_significance.tex": significance_table(comparisons[primary_class] + ladder, cfg),
        "table_dataset.tex": dataset_table(class_balance, len(patient_index),
                                           len(test_patients), cfg.n_folds,
                                           slices_per_volume, cfg),
    }
    for name, content in tables.items():
        write_table(content, os.path.join(cfg.tables_dir, name))

    macros = build_macros(
        evaluations, efficiency_reports, comparisons, cfg,
        extra=standard_extra_macros(cfg, len(patient_index), len(dev_patients),
                                    len(test_patients), slices_per_volume,
                                    class_balance, efficiency_reports),
        ladder=ladder)
    write_macros(macros, os.path.join(cfg.tables_dir, "results_macros.tex"))
    write_markdown_summary(evaluations, ordered_keys, cfg,
                           os.path.join(cfg.results_dir, "RESULTS.md"))

    cost_by_key = {"teacher": efficiency_reports[0], "unet32": efficiency_reports[1]}
    for v in variants:
        if v.key != "unet32":
            cost_by_key[v.key] = efficiency_reports[2]
    plot_ablation_bars(evaluations, ordered_keys, cfg,
                       os.path.join(cfg.figures_dir, "fig_ablation.png"))
    plot_paired_differences(evaluations, "kd_full", "scratch", primary_class,
                            os.path.join(cfg.figures_dir, "fig_paired_delta.png"))
    plot_accuracy_vs_cost({k: evaluations[k] for k in ordered_keys if k in cost_by_key},
                          cost_by_key, cfg,
                          os.path.join(cfg.figures_dir, "fig_accuracy_vs_cost.png"))
    plot_qualitative(cache, predictions, qualitative,
                     [cache.n_slices(p) // 2 for p in qualitative],
                     [("teacher", "Teacher"), ("scratch", "Scratch"),
                      ("kd_vanilla", "Hinton KD"), ("kd_full", "Proposed")],
                     cfg, os.path.join(cfg.figures_dir, "fig_qualitative.png"))
    print(f"  {len(tables)} tables, {len(macros)} macros, 5 figures -> {cfg.output_dir}")

    # --------------------------------------------------------------- seeds
    if "seeds" in stages and len(cfg.student_seeds) > 1:
        banner("Seed variance")
        seed_variants = select_variants(list(SEED_STUDY_KEYS))
        per_seed: Dict[str, Dict[int, float]] = {}
        for seed in cfg.student_seeds:
            results_for_seed = (
                {v.key: student_results[v.key] for v in seed_variants} if seed == cfg.seed
                else run_student_stage(cfg, seed_variants, fold_splits, cache,
                                       teachers_by_fold, device, seed=seed))
            for v in seed_variants:
                models = load_ensemble(student_builder(v.base_channels),
                                       [r["checkpoint"] for r in results_for_seed[v.key]],
                                       device, cfg)
                ev = evaluate_ensemble(models, cache, test_patients, cfg, device,
                                       f"{v.key}_s{seed}", v.label, count_parameters(models[0]),
                                       compute_boundary=False)
                per_seed.setdefault(v.key, {})[seed] = ev.dice(primary_class)
        write_table(seed_variance_table(per_seed, cfg),
                    os.path.join(cfg.tables_dir, "table_seeds.tex"))
        for key, values in per_seed.items():
            vals = [v for v in values.values() if not np.isnan(v)]
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            print(f"  {key:<14}" + "  ".join(f"s{s}={v:.3f}" for s, v in values.items())
                  + f"   sd={sd:.4f}")

    print(f"\ndone in {(time.time() - started) / 3600:.2f} h")
    print(f"copy {cfg.tables_dir} and {cfg.figures_dir} next to main.tex, then compile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
