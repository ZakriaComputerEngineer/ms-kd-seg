#!/usr/bin/env python
"""Assemble the Kaggle notebook from the package sources.

The notebook must be self-contained -- a Kaggle session cannot import a local
package that has not been uploaded -- but keeping a second copy of the code
inside a `.ipynb` guarantees the two drift apart. So the notebook is *generated*:
this script inlines each module in dependency order, strips the intra-package
imports that inlining makes redundant, and wraps the result in the narrative and
driver cells.

Regenerate after any change to `src/msdistill`:

    python scripts/build_notebook.py
    python scripts/validate_notebook.py
"""

from __future__ import annotations

import ast
import json
import os
from typing import Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src", "msdistill")
OUT = os.path.join(REPO_ROOT, "notebooks", "ms_kd_segmentation.ipynb")

# Dependency order. Every name a module references must be defined by an earlier
# entry, because the notebook has no import machinery to resolve cycles.
MODULE_ORDER = [
    ("config.py", "Configuration and the ablation grid"),
    ("data.py", "Dataset discovery, preprocessing and loaders"),
    ("models/teacher.py", "Teacher: Mix Transformer encoder with a full-resolution decoder"),
    ("models/student.py", "Student and the distillation projection heads"),
    ("losses.py", "Supervised and distillation objectives"),
    ("metrics.py", "Volume-level segmentation metrics"),
    ("train.py", "Training engine"),
    ("evaluate.py", "Held-out evaluation"),
    ("stats.py", "Significance testing"),
    ("efficiency.py", "Efficiency benchmarking"),
    ("report.py", "LaTeX table and macro generation"),
    ("viz.py", "Publication figures"),
]

def strip_relative_imports(source: str) -> str:
    """Replace intra-package imports with comments, preserving indentation.

    Uses the parse tree rather than a line regex, because several of these
    imports wrap across lines and a regex that misses a continuation leaves a
    dangling parenthesized name list behind.

    Comments rather than deletions: some of these are the first statement inside
    a function, and deleting the line outright would risk emptying the block.
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    replace: Dict[int, str] = {}   # 0-based line -> replacement
    drop: set = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.level:
            continue
        start = node.lineno - 1
        end = (node.end_lineno or node.lineno) - 1
        indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
        replace[start] = f"{indent}# (inlined above)"
        drop.update(range(start + 1, end + 1))

    out = [replace.get(i, line) for i, line in enumerate(lines) if i not in drop]
    return "\n".join(out).rstrip() + "\n"


def md(source: str) -> Dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str, role: str = "driver") -> Dict:
    """`role` is "library" (pure definitions, safe to execute anywhere) or
    "driver" (touches the dataset). `scripts/validate_notebook.py` reads it to
    decide which cells it can execute and which it can only analyse statically;
    inferring the boundary from the source text was unreliable."""
    return {"cell_type": "code", "execution_count": None,
            "metadata": {"msdistill_role": role},
            "outputs": [], "source": source.rstrip().splitlines(keepends=True)}


# --------------------------------------------------------------------------- #
# Narrative
# --------------------------------------------------------------------------- #

HEADER = """\
# Region-Decoupled Distillation of a Transformer Teacher for CPU-Deployable Multiple Sclerosis Lesion Segmentation

Reference implementation. This notebook is generated from `src/msdistill` by
`scripts/build_notebook.py` — edit the package, not the notebook.

## What this runs

A four-class segmentation problem on MS3SEG (background, ventricles, incidental
"normal" white-matter hyperintensity, pathological "abnormal" WMH), and a
controlled study of what a small convolutional network can inherit from a
transformer teacher.

| Stage | What happens |
|---|---|
| 0 | Fine-tune one teacher **per cross-validation fold** on that fold's training patients |
| 1 | Co-train every student ablation variant against its fold-matched frozen teacher |
| 2 | Evaluate all models on the held-out patients under one identical ensemble protocol |
| 3 | Paired significance tests, efficiency profiling, LaTeX tables and figures |

## Three corrections that changed the conclusions

Earlier runs of this study produced results that contradicted its own premise.
The causes were mechanical, and each is fixed here.

**The teacher was the weaker model.** Stock SegFormer emits logits at stride 4 and
upsamples ×4 to reach input resolution. At 256×256 an MS lesion is often 2–6
pixels across, so that prediction cannot represent one. Measured: SegFormer-B0
reached 0.36 Dice on normal WMH where the 487K-parameter student reached 0.57.
Distilling from it was distilling downward. The teacher here uses a **MiT-B2
encoder with a full-resolution refinement decoder**, which restores stride-1
output and makes the compression ratio (≈56× parameters) meaningful.

**The reported metric awarded free points.** Dice was accumulated per mini-batch
and averaged. With `smooth=1e-5`, a batch containing none of a class scores
`(0+ε)/(0+ε) = 1.0` — a perfect score for predicting nothing. Since abnormal WMH
is absent from most slices, the rare-class numbers were inflated and the ranking
of methods was wrong. Metrics here are computed **once per patient over the whole
volume**, with absent classes recorded as undefined rather than perfect.

**The feature loss had a trivial optimum.** `MSE(W_s·f_s, W_t·f_t)` with *both*
projections learnable is minimized by `W_s = W_t = 0`. The loss went to zero, the
student's bottleneck collapsed, and two folds diverged. Only the student side is
projected here, features keep their spatial layout, and they are L2-normalized so
the objective constrains direction rather than magnitude.

A fourth issue was procedural: cached fold results from a previous configuration
were silently reused, which is why two different objectives once reported
byte-identical validation scores. Every results file now carries a configuration
fingerprint and a mismatch triggers retraining.

## The ablation ladder

| Variant | Objective | Question it answers |
|---|---|---|
| `scratch` | hard labels only | What can this architecture do alone? |
| `kd_vanilla` | + uniform per-pixel KL | Does textbook distillation help here? |
| `kd_fitnets` | + pooled-feature MSE | Does matching globally averaged features help? |
| `kd_cwd` | + channel-wise distillation | Does prior dense-prediction KD help? |
| `kd_region` | + lesion-region-weighted KL | Does decoupling foreground from background help? |
| `kd_region_cwd` | + channel-wise | Is the channel-wise term additive? |
| `kd_full` | + spatial feature alignment | Full proposed objective |

Every row trains the **same architecture from the same initialization on the same
batches in the same order**; only the objective differs. Differences between rows
therefore cannot be attributed to data ordering or initialization, and the
per-patient scores are properly paired for significance testing.
"""

SETUP_MD = """\
## 1. Environment

`transformers` supplies the pretrained Mix Transformer encoder; `nibabel` reads
NIfTI volumes; `scipy` provides the connected-component and distance-transform
routines behind the lesion-wise and boundary metrics.
"""

SETUP_CODE = '''\
import importlib, subprocess, sys

def ensure(package: str, import_name: str = None) -> None:
    """Install only what is missing, so a warm environment starts instantly."""
    try:
        importlib.import_module(import_name or package.split("==")[0].replace("-", "_"))
    except ImportError:
        print(f"installing {package} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)

for pkg, mod in [("nibabel", "nibabel"), ("transformers", "transformers"),
                 ("scipy", "scipy"), ("tqdm", "tqdm"), ("scikit-learn", "sklearn")]:
    ensure(pkg, mod)

import os, json, math, time, copy, random, hashlib, warnings, re, platform, statistics
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
'''

LIBRARY_MD = """\
---

# Part I — Library

The cells below are the contents of `src/msdistill`, inlined in dependency order.
They define everything; nothing is executed against data until Part II.
"""

RUN_MD = """\
---

# Part II — Experiment

## 2. Configuration

`data_root` is the only value that normally needs changing. Set
`teacher_variant` to `"b1"` or `"b0"` if the GPU budget is tight — `b2` is the
configuration the paper reports.
"""

RUN_CONFIG = '''\
cfg = Config(
    # Point this at the directory containing MS_100_patient_registered/ and
    # MS_100_model_input/. On Kaggle that is somewhere under /kaggle/input/; the
    # exact depth varies with how the dataset was attached -- one mirror sits at
    # /kaggle/input/datasets/nguynvntnpht/ms3seg-mri-tri-mask-lesion-segmentation.
    # If this path is wrong the next cell searches for it, so a mismatch is not fatal.
    data_root="/kaggle/input/ms3seg-mri-tri-mask-lesion-segmentation",
    output_dir="/kaggle/working/msdistill_out",

    teacher_variant="b2",       # "b0" / "b1" are documented lower-cost fallbacks
    teacher_hr_decoder=True,    # the correction that makes the teacher competitive
    n_folds=3,
    num_epochs=60,
    teacher_epochs=45,
    batch_size=8,
    num_workers=2,
    max_hours_per_fold=3.0,     # wall-clock guard; a stopped fold resumes, it is not lost

    # Run this many folds at once, one per GPU. Set to 2 on Kaggle's "GPU T4 x2"
    # accelerator; leave at 1 for a single GPU. Folds share nothing but the
    # read-only volume cache, so this scales nearly linearly -- unlike
    # nn.DataParallel, which re-broadcasts the whole model every step and
    # measured *slower* than one GPU on these PCIe-connected T4s.
    # With 3 folds on 2 GPUs the wall time is that of 2 folds, so about 1.5x.
    parallel_folds=1,
).make_dirs()

if torch.cuda.device_count() > 1 and cfg.parallel_folds == 1:
    print(f"NOTE: {torch.cuda.device_count()} GPUs visible but parallel_folds=1. "
          f"Set parallel_folds={torch.cuda.device_count()} above to use them.")

# Locate the dataset wherever it actually landed. Mirrors nest the same folders
# under different slugs, and Kaggle exposes attached datasets as symlinks -- which
# a plain os.walk will not descend into. If this fails it prints a listing of what
# it did find, which distinguishes "not attached" from "attached somewhere else".
cfg.data_root = locate_dataset(cfg.data_root)

DEVICE = get_device()
set_seed(cfg.seed)
require_scipy()   # without it every p-value silently becomes a dash in the tables
cfg.to_json(os.path.join(cfg.results_dir, "config.json"))
print(f"device={DEVICE}  fingerprint={cfg.fingerprint()}")
print(f"outputs -> {cfg.output_dir}")
'''

RUN_DATA_MD = """\
## 3. Data

Label values are **inferred** rather than hard-coded. The distributed masks store
the four classes as widely separated 16-bit values, and the constants differ
between mirrors of the dataset; scanning a few volumes and taking the sorted
unique values is robust to that.
"""

RUN_DATA = '''\
patient_index = build_patient_index(cfg.data_root, cfg.modalities)
print(f"patients with all modalities + mask: {len(patient_index)}")

raw_values = infer_label_values([e["mask"] for e in patient_index.values()], cfg.num_classes)
remapper = LabelRemapper(raw_values)
print(f"inferred raw label values: {raw_values.tolist()}")

dev_patients, test_patients = split_test_and_dev(sorted(patient_index), cfg.test_fraction, cfg.seed)
fold_splits = make_fold_splits(dev_patients, cfg.n_folds, cfg.seed)

# Patient-level partitioning is what prevents slices of one subject appearing in
# both training and evaluation. Assert it rather than trusting it.
assert not (set(dev_patients) & set(test_patients))
for i, split in enumerate(fold_splits):
    assert not (set(split["train"]) & set(split["val"])), f"fold {i} overlaps"
    print(f"fold {i}: train={len(split['train'])} val={len(split['val'])}")
print(f"\\ndevelopment: {len(dev_patients)}  held-out test: {len(test_patients)}")
'''

RUN_CACHE = '''\
cache = PatientVolumeCache(sorted(patient_index), patient_index, remapper, cfg)
class_balance = summarize_class_balance(cache, cfg)

print("voxel share by class:")
for name, fraction in class_balance.items():
    print(f"  {name:<14} {100 * fraction:6.3f}%")

example = sorted(patient_index)[0]
print(f"\\n{example}: volume {cache.images[example].shape}, "
      f"spacing {tuple(round(s, 2) for s in cache.spacing[example])} mm")
plot_class_balance(class_balance, os.path.join(cfg.figures_dir, "fig_class_balance.png"))
'''

RUN_TEACHER_MD = """\
## 4. Stage 0 — Teacher

One teacher per fold, trained on exactly that fold's training patients. Reusing a
single teacher across folds would mean distilling into fold *k* from a model that
had already trained on fold *k*'s validation patients, inflating the measured
benefit on precisely the cases used to measure it.

This is the expensive stage. Progress is checkpointed per fold and guarded by
`max_hours_per_fold`; rerunning the cell resumes rather than restarting.
"""

RUN_TEACHER = '''\
teacher_results = run_teacher_stage(cfg, fold_splits, cache, DEVICE)

teachers_by_fold = {
    r["fold"]: load_frozen_teacher(cfg, r["checkpoint"], DEVICE)
    for r in teacher_results
}
print(f"\\n{len(teachers_by_fold)} frozen fold-matched teachers ready")

# The freeze is a contract the training loop depends on -- verify it holds.
probe = teachers_by_fold[0]
probe.train(True)
assert not probe.training and all(not p.requires_grad for p in probe.parameters())
print("teacher freeze verified: eval mode locked, zero trainable parameters")
'''

RUN_STUDENT_MD = """\
## 5. Stage 1 — Student ablation

All variants for a fold train in one pass. The teacher forward runs **once per
batch** and its outputs are shared, which turns a cost of
`n_variants × (teacher + student)` into `teacher + n_variants × student`.

The scientific reason matters more than the speed: every variant sees an
identical batch sequence with identical augmentation, so the rows differ only in
their objective.
"""

RUN_STUDENT = '''\
variants = select_variants()          # select_variants(list(QUICK_VARIANT_KEYS)) for a shorter run
student_results = run_student_stage(cfg, variants, fold_splits, cache, teachers_by_fold, DEVICE)

curve_keys = [k for k in ("scratch", "kd_vanilla", "kd_cwd", "kd_region", "kd_full")
              if k in student_results]
plot_training_curves(
    {k: student_results[k] for k in curve_keys},
    teacher_dice=float(np.mean([r["best_val_dice"] for r in teacher_results])),
    path=os.path.join(cfg.figures_dir, "fig_training_curves.png"),
)
'''

RUN_EVAL_MD = """\
## 6. Held-out evaluation

Every model — teacher included — is evaluated identically: the K fold
checkpoints are ensembled by averaging softmax probabilities, and metrics are
computed per patient over the complete volume. Comparing a single-fold teacher
against fold-ensembled students, as an earlier version did, handicaps the teacher
for reasons unrelated to distillation.
"""

RUN_EVAL = '''\
evaluations, predictions = {}, {}
qualitative_patients = test_patients[:3]
primary_class = cfg.class_names[cfg.primary_class]

teacher_models = [load_frozen_teacher(cfg, r["checkpoint"], DEVICE) for r in teacher_results]
evaluations["teacher"] = evaluate_with_single_model_stats(
    teacher_models, cache, test_patients, cfg, DEVICE,
    name="teacher",
    label=f"Teacher (MiT-{cfg.teacher_variant.upper()}" + ("+HR)" if cfg.teacher_hr_decoder else ")"),
    n_params=count_parameters(teacher_models[0].model, trainable_only=False),
    store_predictions_for=qualitative_patients, prediction_store=predictions,
)
evaluations["teacher"].objective_note = "hard labels"

for v in variants:
    builder = student_builder(v.base_channels)
    models = load_ensemble(builder, [r["checkpoint"] for r in student_results[v.key]],
                           DEVICE, cfg)
    ev = evaluate_with_single_model_stats(
        models, cache, test_patients, cfg, DEVICE, v.key, v.label,
        count_parameters(models[0]),
        store_predictions_for=qualitative_patients, prediction_store=predictions)
    ev.objective_note = v.objective
    ev.citation = v.citation
    evaluations[v.key] = ev

ordered_keys = ["teacher"] + [v.key for v in variants]

header = f"{'model':<34}{'params':>9}{'normal WMH':>15}{'abnormal WMH':>15}{'mean FG':>9}"
print(header + "\\n" + "-" * len(header))
for key in ordered_keys:
    ev = evaluations[key]
    print(f"{ev.label[:33]:<34}{ev.n_params/1e6:>8.2f}M"
          f"{ev.dice('normal_wmh'):>9.3f} ±{ev.std('dice','normal_wmh'):.3f}"
          f"{ev.dice('abnormal_wmh'):>9.3f} ±{ev.std('dice','abnormal_wmh'):.3f}"
          f"{ev.mean_foreground_dice():>9.3f}")

# The premise check. If the teacher does not exceed the compact baseline the
# compression framing does not hold, and the paper must be reframed around
# cross-architecture transfer instead -- better to discover this here than in
# review.
t = evaluations["teacher"].dice(primary_class)
s = evaluations["scratch"].dice(primary_class)
print(f"\\nteacher {t:.3f} vs scratch {s:.3f} on {primary_class}: "
      + ("teacher leads, compression framing holds"
         if t > s else "TEACHER DOES NOT LEAD -- see the reframing note at the end"))

save_evaluations(cfg, evaluations)
'''

RUN_STATS_MD = """\
## 7. Significance

With 20 test patients, "0.62 versus 0.59" is not a result on its own. Each
comparison is paired across patients (Wilcoxon signed-rank), reported with a
percentile bootstrap confidence interval on the mean difference, and corrected
for multiplicity within each class using Holm–Bonferroni.

The ladder comparisons are separate on purpose: they test whether *each added
component* contributes, which a comparison against the baseline cannot show.
"""

RUN_STATS = '''\
comparisons = {
    cfg.class_names[c]: compare_all(evaluations, "scratch", "dice", cfg.class_names[c], cfg,
                                    exclude=["teacher"])
    for c in cfg.foreground_classes
}

ladder = ladder_comparisons(evaluations, [k for k in LADDER_KEYS if k in evaluations],
                            "dice", primary_class, cfg)

print(f"--- vs Student-Scratch on {primary_class} ---")
for c in comparisons[primary_class]:
    print(f"  {c.name_a:<16} delta={c.mean_difference:+.3f}  "
          f"CI[{c.ci_low:+.3f},{c.ci_high:+.3f}]  p_holm={format_p(c.p_adjusted)}"
          f"{significance_marker(c)}")

print(f"\\n--- consecutive ablation steps on {primary_class} ---")
for c in ladder:
    print(f"  {c.name_b:<16} -> {c.name_a:<16} delta={c.mean_difference:+.3f}  "
          f"p_holm={format_p(c.p_adjusted)}{significance_marker(c)}")

gap = teacher_student_gap_recovery(evaluations["teacher"], evaluations["scratch"],
                                   evaluations["kd_full"], primary_class)
print(f"\\nteacher-scratch gap recovered by kd_full: "
      + (f"{100 * gap:.1f}%" if not np.isnan(gap)
         else "undefined (teacher did not exceed the scratch student on this class)"))
'''

RUN_EFF_MD = """\
## 8. Efficiency

Parameter count is a poor proxy for deployment cost: the earlier 7.6× parameter
reduction bought only 1.6× in latency, because the small U-Net does all its work
at full resolution while the transformer downsamples immediately. Multiply-
accumulate cost, GPU and CPU wall-clock at batch size 1, and peak memory are all
measured.
"""

RUN_EFF = '''\
slices_per_volume = cache.n_slices(test_patients[0])
environment = environment_summary(DEVICE)

efficiency_reports = [
    profile_model(build_teacher(cfg, pretrained=False).to(DEVICE), "Teacher", cfg, DEVICE,
                  checkpoint=teacher_results[0]["checkpoint"],
                  slices_per_volume=slices_per_volume),
    profile_model(build_student(cfg, 32).to(DEVICE), "U-Net (base 32)", cfg, DEVICE,
                  checkpoint=student_results.get("unet32", [{}])[0].get("checkpoint"),
                  slices_per_volume=slices_per_volume),
    profile_model(build_student(cfg).to(DEVICE), "Student", cfg, DEVICE,
                  checkpoint=student_results["scratch"][0]["checkpoint"],
                  slices_per_volume=slices_per_volume),
]

for r in efficiency_reports:
    print(f"{r.name:<18} {r.params_total:>11,}p  {r.gmacs:>7.2f} GMACs  "
          f"GPU@b1 {r.gpu_latency_ms.get(1, float('nan')):>7.2f} ms  "
          f"CPU@b1 {r.cpu_latency_ms.get(1, float('nan')):>8.1f} ms  "
          f"peak {r.peak_gpu_memory_mb:>6.0f} MB")

teacher_eff, student_eff = efficiency_reports[0], efficiency_reports[-1]
print(f"\\nMAC accounting: {teacher_eff.flops_method}")
print(f"environment: {environment}")
print(f"\\ncompression teacher -> student: "
      f"{teacher_eff.params_total / student_eff.params_total:.0f}x parameters, "
      f"{teacher_eff.gmacs / student_eff.gmacs:.1f}x MACs, "
      f"{teacher_eff.cpu_latency_ms.get(1, float('nan')) / student_eff.cpu_latency_ms.get(1, float('nan')):.1f}x CPU latency")
print("The three ratios differ because the student computes at full input "
      "resolution while the transformer downsamples at the stem. Report all three.")
print(f"\\none patient study ({slices_per_volume} slices) on 4 CPU threads: "
      f"{student_eff.cpu_latency_ms.get(1, float('nan')) * slices_per_volume / 1000:.2f} s "
      f"single model, "
      f"{student_eff.cpu_latency_ms.get(1, float('nan')) * slices_per_volume * cfg.n_folds / 1000:.2f} s "
      f"as a {cfg.n_folds}-fold ensemble")
'''

RUN_OUTPUT_MD = """\
## 9. Paper artefacts

Every table and every number quoted in the manuscript's prose is written here.
`results_macros.tex` defines a LaTeX command per quantity, so the paper contains
no hand-typed figures and cannot go stale relative to the experiments.
"""

RUN_OUTPUT = '''\
tables = {
    "table_ablation.tex":     ablation_table(evaluations, ordered_keys, cfg, comparisons,
                                             published=MS3SEG_PUBLISHED),
    "table_detection.tex":    detection_table(evaluations, ordered_keys, cfg),
    "table_efficiency.tex":   efficiency_table(efficiency_reports, "Teacher", cfg,
                                               ensemble_size=cfg.n_folds,
                                               environment=environment,
                                               slices_per_volume=slices_per_volume),
    "table_significance.tex": significance_table(comparisons[primary_class] + ladder, cfg),
    "table_dataset.tex":      dataset_table(class_balance, len(patient_index),
                                            len(test_patients), cfg.n_folds,
                                            slices_per_volume, cfg),
}
for filename, content in tables.items():
    write_table(content, os.path.join(cfg.tables_dir, filename))

macros = build_macros(
    evaluations, efficiency_reports, comparisons, cfg,
    extra=standard_extra_macros(cfg, len(patient_index), len(dev_patients),
                                len(test_patients), slices_per_volume,
                                class_balance, efficiency_reports),
    ladder=ladder,
)
write_macros(macros, os.path.join(cfg.tables_dir, "results_macros.tex"))
write_markdown_summary(evaluations, ordered_keys, cfg,
                       os.path.join(cfg.results_dir, "RESULTS.md"))

print(f"{len(tables)} tables and {len(macros)} macros written to {cfg.tables_dir}")
print("copy tables/ into paper/ and \\\\input them; no number is typed by hand")
'''

RUN_FIGS = '''\
cost_by_key = {"teacher": efficiency_reports[0], "unet32": efficiency_reports[1]}
for v in variants:
    if v.key != "unet32":
        cost_by_key[v.key] = efficiency_reports[-1]   # every student shares one architecture

figures = [
    plot_ablation_bars(evaluations, ordered_keys, cfg,
                       os.path.join(cfg.figures_dir, "fig_ablation.png")),
    plot_paired_differences(evaluations, "kd_full", "scratch", primary_class,
                            os.path.join(cfg.figures_dir, "fig_paired_delta.png")),
    plot_accuracy_vs_cost(
        {k: evaluations[k] for k in ("teacher", "unet32", "scratch", "kd_full")
         if k in evaluations},
        cost_by_key, cfg,
        os.path.join(cfg.figures_dir, "fig_accuracy_vs_cost.png")),
    plot_qualitative(
        cache, predictions, qualitative_patients,
        [cache.n_slices(p) // 2 for p in qualitative_patients],
        [("teacher", "Teacher"), ("scratch", "Scratch"),
         ("kd_vanilla", "Hinton KD"), ("kd_full", "Proposed")],
        cfg, os.path.join(cfg.figures_dir, "fig_qualitative.png")),
]

for path in figures:
    print("wrote", path)
    display(__import__("IPython").display.Image(filename=path))
'''

RUN_SEEDS_MD = """\
## 10. Seed variance (optional)

A difference smaller than the seed-to-seed spread is not a finding. The student
is cheap enough to repeat, so the headline comparison is checked against
independent initializations. Skip this cell if the GPU budget is exhausted — the
main results do not depend on it, but the paper is stronger with it.
"""

RUN_SEEDS = '''\
RUN_SEED_STUDY = True   # set False to skip

per_seed = {}
if RUN_SEED_STUDY and len(cfg.student_seeds) > 1:
    seed_variants = select_variants(list(SEED_STUDY_KEYS))
    for seed in cfg.student_seeds:
        if seed == cfg.seed:
            results_for_seed = {v.key: student_results[v.key] for v in seed_variants}
        else:
            results_for_seed = run_student_stage(cfg, seed_variants, fold_splits, cache,
                                                 teachers_by_fold, DEVICE, seed=seed)
        for v in seed_variants:
            models = load_ensemble(student_builder(v.base_channels),
                                   [r["checkpoint"] for r in results_for_seed[v.key]],
                                   DEVICE, cfg)
            ev = evaluate_ensemble(models, cache, test_patients, cfg, DEVICE,
                                   f"{v.key}_s{seed}", v.label, count_parameters(models[0]),
                                   compute_boundary=False)
            per_seed.setdefault(v.key, {})[seed] = ev.dice(primary_class)

    write_table(seed_variance_table(per_seed, cfg),
                os.path.join(cfg.tables_dir, "table_seeds.tex"))

    spreads = []
    for key, values in per_seed.items():
        vals = [v for v in values.values() if not np.isnan(v)]
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        spreads.append(sd)
        print(f"{key:<12} " + "  ".join(f"seed{s}={v:.3f}" for s, v in values.items())
              + f"   sd={sd:.4f}")

    # This number is the interpretability floor for the whole results section:
    # a difference between objectives smaller than the spread across seeds of a
    # single objective is not evidence of anything.
    floor = max(spreads) if spreads else 0.0
    gain = (evaluations["kd_full"].dice(primary_class)
            - evaluations["scratch"].dice(primary_class))
    print(f"\\nseed-to-seed spread (worst variant): {floor:.4f} Dice")
    print(f"proposed-minus-baseline gain on {primary_class}: {gain:+.4f} Dice")
    print("-> " + ("gain exceeds the seed floor; interpret it"
                   if abs(gain) > floor else
                   "gain is WITHIN the seed floor; report it as not interpretable"))
'''

RUN_EXPORT_MD = """\
## 11. Download the results

A Kaggle session is deleted when it ends. Everything the paper needs is small --
tables, macros, figures and the per-patient JSON total a couple of megabytes,
and `scripts/regenerate_tables.py` can rebuild every table from the JSON alone
without a GPU. The checkpoints are the bulk and are only needed to re-run
inference.
"""

RUN_EXPORT = '''\
import shutil

# The small artefacts: everything needed to rebuild the paper.
shutil.make_archive("/kaggle/working/paper_artifacts", "zip", cfg.output_dir,
                    base_dir=None, root_dir=cfg.output_dir)
for sub in ("results", "tables", "figures"):
    src = os.path.join(cfg.output_dir, sub)
    if os.path.isdir(src):
        shutil.copytree(src, f"/kaggle/working/export/{sub}", dirs_exist_ok=True)
shutil.make_archive("/kaggle/working/paper_artifacts", "zip", "/kaggle/working/export")
size = os.path.getsize("/kaggle/working/paper_artifacts.zip") / 1e6
print(f"paper_artifacts.zip  {size:.1f} MB  <- download this one")

# The checkpoints, separately, because they are large and rarely needed again.
ckpt = os.path.join(cfg.output_dir, "checkpoints")
if os.path.isdir(ckpt):
    total = sum(os.path.getsize(os.path.join(ckpt, f)) for f in os.listdir(ckpt)) / 1e6
    print(f"checkpoints/         {total:.0f} MB  (zip separately only if you need them)")
    print("  !zip -r /kaggle/working/checkpoints.zip " + ckpt)
'''

CLOSING_MD = """\
---

## 12. What to copy into the paper

| Artefact | Destination |
|---|---|
| `tables/results_macros.tex` | `\\input` at the top of `main.tex`; every number in the prose is a macro from here |
| `tables/table_*.tex` | `\\input` at each table position |
| `figures/fig_*.png` | `paper/figures/` |
| `results/RESULTS.md` | the repository README |
| `results/test_evaluations.json` | archived per-patient scores, for reproducibility |

### Reading the outcome honestly

Three results are possible and the paper has a defensible framing for each.

**The distilled student beats the scratch student significantly.** The headline
claim holds; report the ladder to show which component earned it.

**Only some components help.** Report that. An ablation whose rows are all
positive is less credible than one that identifies a component contributing
nothing — and the negative row is itself a finding about dense-prediction
distillation under extreme class imbalance.

**The student matches or beats the teacher.** This is common with small cohorts
and a high-capacity teacher, and it does not invalidate the work: the efficiency
result stands on its own, and the finding becomes "at this data scale, teacher
capacity is not the binding constraint" — which is worth reporting, provided the
paper says so plainly rather than burying it.
"""


def build() -> Dict:
    cells: List[Dict] = [md(HEADER), md(SETUP_MD), code(SETUP_CODE, "library"), md(LIBRARY_MD)]

    for relative, title in MODULE_ORDER:
        path = os.path.join(SRC, relative)
        with open(path, encoding="utf-8") as f:
            source = f.read()
        cells.append(md(f"### `{relative}` — {title}"))
        cells.append(code(strip_relative_imports(source), "library"))

    cells += [
        md(RUN_MD), code(RUN_CONFIG),
        md(RUN_DATA_MD), code(RUN_DATA), code(RUN_CACHE),
        md(RUN_TEACHER_MD), code(RUN_TEACHER),
        md(RUN_STUDENT_MD), code(RUN_STUDENT),
        md(RUN_EVAL_MD), code(RUN_EVAL),
        md(RUN_STATS_MD), code(RUN_STATS),
        md(RUN_EFF_MD), code(RUN_EFF),
        md(RUN_OUTPUT_MD), code(RUN_OUTPUT), code(RUN_FIGS),
        md(RUN_SEEDS_MD), code(RUN_SEEDS),
        md(RUN_EXPORT_MD), code(RUN_EXPORT),
        md(CLOSING_MD),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    notebook = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # newline="\n" so the file is byte-identical on every platform. Without it
    # Windows writes CRLF and CI's "notebook is in sync with src/" check fails on
    # line endings alone, which is a maddening thing to debug.
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
        f.write("\n")
    n_code = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")
    n_md = sum(1 for c in notebook["cells"] if c["cell_type"] == "markdown")
    print(f"wrote {OUT}\n  {len(notebook['cells'])} cells ({n_code} code, {n_md} markdown)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
