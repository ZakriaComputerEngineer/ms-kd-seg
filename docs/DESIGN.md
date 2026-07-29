# Design record

Why the pipeline is shaped the way it is. Written for whoever picks this up next — including future
us — so that decisions that look arbitrary are not silently undone.

---

## 1. Diagnosis of the previous pipeline

The prior notebook ran to completion and produced numbers that contradicted its own thesis: the
teacher scored below the student on the clinically important class, and adding the feature term made
things worse. Those were not bad luck. Fourteen defects were found; the six that changed conclusions
are below.

### D1 — Batch-averaged Dice awards a perfect score for predicting nothing (fatal)

`dice_sum += per_class_dice(logits, masks); ... / n_batches` with `smooth=1e-5`. A batch containing
no voxels of class *c* in either prediction or reference computes `(2·0 + 1e-5)/(0 + 1e-5) = 1.0`.

Pathological WMH appears in a minority of slices, so a large fraction of batches contributed a free
1.0 to that class. The reported rare-class scores were inflated by an amount that varied with how
often each model predicted an empty mask — which is model-dependent, so the *ranking* was affected,
not just the level. **Every number the previous run reported for the rare classes was invalid**, and
checkpoint selection was driven by the same broken quantity.

*Fix:* `metrics.py` computes each metric once per patient over the whole volume; absent classes are
`nan` (excluded) rather than 1.0. `train.validation_dice` uses the same definition, so selection and
reporting agree.

### D2 — Stride-4 teacher cannot represent the target (fatal)

SegFormer's decode head outputs at H/4 and bilinearly upsamples ×4. At 256×256 an MS lesion is often
2–6 px across. Measured: teacher 0.362 Dice on incidental WMH, student 0.571.

Encoder capacity was never the issue. B0→B2 alone would not have fixed it.

*Fix:* `HRRefinement` in `models/teacher.py` returns the decoder feature to stride 1 via two shallow
streams computed directly from the input (~0.2M params). `teacher_hr_decoder=True` is not optional.

### D3 — Two-sided learnable projection has a zero optimum (fatal)

`MSE(W_s·GAP(f_s), W_t·GAP(f_t))` with both projections trainable is minimized at `W_s = W_t = 0`.
The loss reaches zero transferring nothing, while the gradient reaching the student collapses its
bottleneck. Observed: KD-Full 0.476 vs KD-Soft 0.586; folds at 0.10 and 0.21.

Independently, `GAP` over a 128×16×16 bottleneck destroys all spatial information — for a dense
prediction task.

*Fix:* student-side projection only; spatial layout preserved; per-pixel L2 normalization removes
the shrink direction. `losses.spatial_feature_distillation`.

### D4 — Stale cached results silently merged across variants (fatal to trust)

`student_kd_soft` and `student_kd_full` reported **byte-identical** validation Dice on folds 0 and 1
(0.6287, 0.1045). Two different objectives cannot do that. `run_cross_validation` loaded a results
JSON from a previous session and skipped those folds as "already logged".

*Fix:* `Config.fingerprint()` hashes every semantic field; `load_cached_results` discards a cache
whose fingerprint differs and retrains. Non-semantic fields (paths, worker count, AMP) are excluded
so that changing them does not force a rerun.

### D5 — Asymmetric evaluation protocol

The teacher was evaluated as a single best-fold model; students as 3-fold ensembles. That
handicapped the teacher for reasons unrelated to distillation.

*Fix:* one protocol for everything. `evaluate_with_single_model_stats` additionally reports
single-fold accuracy, so the accuracy table and the efficiency table describe the same artefact.

### D6 — Parameter count used as an efficiency proxy

7.6× fewer parameters bought 1.6× latency, because the student computes at full resolution while the
transformer downsamples at the stem. Reporting only the parameter ratio overstated the practical
gain by roughly an order of magnitude.

*Fix:* `efficiency.py` reports parameters, MACs, GPU and CPU latency, peak memory, throughput and
per-study wall time. The paper explains the discrepancy rather than hiding it.

### Also fixed, lower impact

- Arbitrary 90° rotations in augmentation — anatomically impossible for axial brain slices.
- Percentile normalization computed over `volume > 0`, on volumes containing negatives, then
  zeroing background *after* standardization, manufacturing an intensity edge at the skull.
- KL computed in fp16 under autocast; softened probabilities near 1e-4 have ~2 significant digits
  there. All losses now compute in fp32.
- Hard-coded raw mask values `[0, 16448, 49087, 65535]`; these differ between dataset mirrors.
  Now inferred by scanning.
- `SegFormerTeacher` printed a warning and continued with **random weights** when a checkpoint was
  missing. Now raises.
- `SegformerConfig` built with default `depths=(2,2,2,2)`, which is wrong for B2 `(3,4,6,3)`. This
  produced a 16.3M model where B2 is 27.5M and would have made checkpoint loading fail at run time.

---

## 1b. Defects found in the *rewrite*, by adversarial review

The rewrite was then reviewed by independent agents told to break it. They found three more bugs of
exactly the same character as D1 — plausible numbers, no error. All three are evaluation-only, so
none invalidated any training. Each now has a regression test.

### R1 — Surface distances measured to the wrong thing

`_surface_distances` built the distance transform from the complement of the *filled* mask instead
of the other *surface*. Any surface voxel lying inside the other object was recorded at 0 mm.

> A hollow reference scored `hd95=0.000, nsd=1.000` against a truth of `5.000 / 0.919`. A routine
> shifted-sphere overlap scored `nsd=0.796` against a truth of `0.692`.

NSD was inflated on essentially every case and HD95 biased low. Two-line fix.

### R2 — Lesion matching had no one-to-one assignment

True positives were scored per reference component against the *union* of all predictions, so one
merged blob could claim unlimited detections at a cost of one false positive.

> A whole-slice prediction scored lesion-F1 **0.909** against five reference lesions — beating an
> accurate four-of-five prediction at 0.800 — while its voxel Dice was 0.072.

Now greedy one-to-one by IoU, which also makes `lesion_match_iou` finally mean what it is named. A
prediction landing on a reference component excluded for being under `min_voxels` is ignored rather
than penalised.

### R3 — A total miss was recorded as "undefined"

`hausdorff95` and `normalized_surface_dice` returned `nan` when the prediction was empty even with a
non-empty reference, and both `summarize` and `paired_valid` drop `nan`. The worst possible
outcome was therefore deleted from the mean and from every paired test — and the missingness is a
function of the outcome being measured, so it systematically flattered whichever variant segmented
least. That is the exact axis this ablation varies. Empty predictions now score the volume diagonal
(HD95) and 0.0 (NSD), following the BraTS/nnU-Net convention.

### R4 — The Wilcoxon guard used the wrong sample size

`zero_method="wilcox"` deletes tied pairs and tests at the reduced size, but the `n >= 5` guard
checked the pre-tie count. Eighteen ties plus two differing pairs still emitted `p=0.18`, a
normal-approximation value for an effective *n* of 2 whose exact two-sided minimum is 0.5. Ties are
the expected case on the rare classes, where both models often predict the same empty mask. The
guard now checks `n_effective`, which is recorded on every result and printed in the table.

### R5 — Latency protocol did not match the caption

The CPU path ran 3 warmups and 10 trials while the auto-generated caption claimed 50/10. Since CPU
latency is the headline deployment number, the paper would have stated a protocol it did not use.

Also fixed: the manuscript would not have compiled (unescaped underscores in the placeholder
branches), the abstract paired ensemble accuracy with single-model latency, and Contribution 2
claimed a causal finding — "output stride, *not encoder capacity*" — that no ablation in the grid
isolates. That claim is now stated as a design decision, and the pilot numbers behind it are
labelled as a pilot.

---

## 2. Design decisions

### Teacher: MiT-B2 + HR refinement (27.5M, 19.6 GMACs)

| Option | Verdict |
|---|---|
| Stock SegFormer-B0 (3.7M) | Rejected — D2, and smaller than the 7.8M U-Net baseline, so "compression" would be incoherent |
| B0 + HR (3.9M) | Fallback only. Fixes D2 but gives an 8× ratio |
| **B2 + HR (27.5M)** | **Chosen.** 56× parameters, 25× MACs |
| Ensemble / nnU-Net teacher | Rejected — breaks fold-matched training, days of compute |

The HR decoder is the load-bearing change. Encoder size is the tunable one.

### Student: unchanged TinyUNet (487K)

Deliberately identical to the previous baseline so the study isolates the *objective*. Changing
architecture and objective together would make the ablation uninterpretable.

### Loss composition

Three terms, each answering a distinct objection:

- **Region-decoupled KL** — the direct answer to 99.55% background. Weights renormalized to unit
  mean so `kd_fg_weight` redistributes gradient without rescaling the term, keeping magnitudes
  comparable across ablation rows.
- **Channel-wise (CWD)** — scale-free in class frequency; also the citable answer to "why not plain
  per-pixel KL", which reviewers will ask.
- **Spatial feature alignment** — the corrected form of D3.

Warmup ramp on all three: a randomly initialized student driven toward teacher statistics before it
can segment anything converges worse.

### Co-training all variants in one pass

Primary motivation is scientific, not speed: identical batch sequence and augmentation across rows,
which makes per-patient scores properly paired. Speed is the bonus —
`teacher + n·student` instead of `n·(teacher + student)`, roughly 4× for eight rows.

*Constraint this imposes:* all variants must share one data loader, so per-variant augmentation
policies are impossible. Acceptable; nothing needs them.

### Mid-fold snapshots for the co-trained stage

Co-training the grid has one downside: the variants also *fail* together. An interruption partway
through a 60-epoch fold would discard all eight at once, where the previous per-variant pipeline
would have lost only the variant in flight.

`_save_snapshot` therefore persists every variant's weights, optimizer moments, scaler state and
best-so-far checkpoint every `resume_every_epochs` epochs (default 5), written to a temporary file
and moved into place atomically so a killed write cannot corrupt it. The snapshot is roughly 180 MB,
which is why it is not written every epoch.

Two details that are easy to get wrong and are covered here:

- A fold cut short by the wall-clock guard is marked `complete: False`. It still writes its
  best-so-far checkpoints, but `run_student_stage` will not skip it on the next run. Checking only
  for checkpoint existence would silently report a truncated run as finished.
- The learning-rate schedule is not serialised. `WarmupCosine.step(epoch)` computes the factor from
  the absolute epoch number and multiplies the base rates captured at construction, so resuming at
  epoch 20 restores the correct rate without any saved scheduler state — and the optimizer's own
  `load_state_dict`, which overwrites `param_groups["lr"]`, is corrected on the next step.

### Notebook generated from the package

Kaggle cannot import an un-uploaded local package, so the notebook must be self-contained. Keeping a
second copy of the code inside a `.ipynb` guarantees drift. `build_notebook.py` inlines the modules
in dependency order and strips intra-package imports via the parse tree (a line regex missed
multi-line imports and left dangling parenthesized name lists).

`validate_notebook.py` executes every library cell in a fresh module and statically checks that
driver cells reference only names the library provides — the failure this prevents is a NameError in
cell 40 after an hour of GPU training.

---

## 3. Deliberate non-goals

| Not done | Why |
|---|---|
| Multi-scale decoder distillation | Marginal expected gain; an extra ablation row there is no page budget for |
| Logit standardization (CVPR 2024) | Cited, not implemented — opens a hyperparameter that cannot be tuned before the deadline |
| CIRKD / MGD / SKD as implemented comparators | Three implemented comparators is already above the median for a 6-page paper |
| Boundary-aware distillation | Attractive given NSD/HD95 reporting; no time to tune |
| 3-D or 2.5-D modelling | 5.73 mm slices; through-plane context is thin and the annotation is per-slice |
| Test-time augmentation | Would widen the accuracy/efficiency gap the paper is trying to report honestly |

---

## 4. Interpreting the outcome

Three outcomes are possible. Each has a defensible framing, written **before** seeing results so
that the narrative is not chosen to fit them.

**Teacher leads, distillation helps, ladder climbs.** The intended result. Report the ladder to show
which component earned it.

**The ladder does not climb monotonically.** The most likely outcome. Reframe from "our combined
objective wins" to "which distillation signal survives extreme foreground sparsity". A component
analysis with a clean negative result is stronger than a monotone ladder — provided it is framed as
a finding. **Do not drop the losing row from the table**; that is the specific behaviour reviewers
are trained to detect.

**The student matches or beats the teacher.** Common with small cohorts, and consistent with MS3SEG's
own transformer baselines losing to U-Net. Reframe from *compression* to *cross-architecture
representation transfer*: the teacher is not a stronger predictor but a differently-inductive one,
and if a student distilled from a teacher it outperforms still beats its hard-label twin, what
transfers is the teacher's uncertainty structure rather than its accuracy. The efficiency result
stands regardless.

`teacher_student_gap_recovery` returns `nan` rather than a percentage when the teacher does not lead,
because "recovered X% of the gap" is meaningless with a negative gap — and is a standard way to make
a failed distillation look successful.

---

## 5. Verification

`scripts/smoke_test.py` gates every change. It asserts, among other things:

- inferred label values match the known encoding
- no patient appears in two partitions
- the teacher freeze contract holds (`train(True)` is a no-op, zero trainable params)
- empty/empty Dice is `nan` and false-positive-only Dice is `0.0` — D1 cannot regress
- single-model statistics exist for every ensembled model — D5 cannot regress
- every LaTeX table has balanced environments and every macro name is LaTeX-legal
- **every `\Res…` macro cited by `main.tex` is actually produced** — so the compiled PDF cannot
  contain a `[PENDING]` marker where a number belongs
