# Runbook

Operational guide for producing the submission. Read `docs/DESIGN.md` for *why*; this is *how*.

---

## Submission target

**FIT 2026** — 23rd International Conference on Frontiers of Information Technology, COMSATS
University Islamabad.

| Item | Value |
|---|---|
| Paper deadline | **31 July 2026** |
| Notification | 15 October 2026 |
| Camera-ready | 30 October 2026 (final paper 8 November) |
| Conference | 14–15 December 2026, COMSTECH Secretariat, Islamabad |
| Page limit | **6 pages**, two-column IEEE conference format |
| Submission | PDF only, EasyChair: `https://easychair.org/conferences/?conf=fit26` |
| Track | **Computer Vision and Image Processing** (lists Medical Imaging) |
| Theme | AI for Societal Transformation in the Global South |
| Publication | Technically co-sponsored by IEEE Islamabad Section; submitted for IEEE Xplore |

Assumed, not stated by the conference: references count inside the 6 pages (plan for all-inclusive —
planning for "6 + refs" and being wrong is unrecoverable); single-blind, so include author names;
no purchasable extra pages.

---

## Order of operations

Teacher training is the blocking item; everything else can proceed in parallel with it.

```
1. Teachers        3 folds × MiT-B2+HR                        GPU, ~4–7 h
2. Student grid    8 variants × 3 folds, single shared pass   GPU, ~6–10 h
3. Seed study      3 variants × 3 folds × 2 extra seeds       GPU, ~4 h   (optional)
4. Evaluation      ensemble + per-fold, all metrics           ~20 min
5. Artefacts       tables, macros, figures                    ~2 min
6. Paper           copy tables/ and figures/, compile         —
```

### Using two GPUs

On Kaggle's **GPU T4 × 2** accelerator, set `parallel_folds=2` in the configuration cell. With 3
folds on 2 GPUs the wall time becomes that of 2 folds — about **1.5×** on both stages.

Folds are completely independent: no parameter, gradient or batch crosses them, so this needs no
inter-GPU communication at all. The driver runs each fold in its own thread on its own device.

**It is deliberately not `nn.DataParallel`.** That re-broadcasts the entire model to every device on
every step; for the 27.5M teacher on PCIe-connected T4s with no NVLink that is ~110 MB of traffic
per step against ~120 ms of compute, and it measured *slower* than a single GPU in an earlier
version of this pipeline. `parallel_folds` is excluded from the configuration fingerprint, so
switching between one and two GPUs never invalidates cached folds — you can start on one and finish
on two.

Two consequences to expect: per-epoch log lines from different folds interleave, and peak host RAM
is unchanged (the volume cache is shared read-only) while each GPU holds its own teacher and
students.

**Budget.** Roughly **10–17 GPU-hours** for steps 1–2 on a single T4, or **7–11** with
`parallel_folds=2`, against a Kaggle free-tier allowance of about 30 GPU-hours per week. Note that
the allowance is charged per GPU-hour, so two GPUs for 7 hours costs about the same quota as one for
14 — this buys wall-clock time, not quota. It fits, with the seed study, but not twice — so avoid configuration changes
that invalidate the fingerprint after the teachers are trained.

The cost split is worth understanding before cutting anything. The teacher is ~19.6 GMACs per slice
against the student's 0.77, so a teacher forward dominates every KD batch. Co-training the whole
grid in one pass is what keeps step 2 affordable: eight variants cost
`teacher + 8 × student` per batch rather than `8 × (teacher + student)`, about 4× less. Adding a
ninth variant therefore costs very little; adding a second teacher configuration costs a great deal.

Memory: the cached cohort is ~1.7 GB of RAM (100 patients × 3 modalities × 256×256×20 float32, masks
as uint8), and MiT-B2 at batch 8 with mixed precision peaks near 5 GB of GPU memory. Both fit a
standard Kaggle session comfortably.

While (1) runs, draft the prose. Every number is a macro, so the text can be finished before the
numbers exist.

### Budget fallback ladder

Apply in this order if compute runs short:

1. Skip the seed study (step 3).
2. `select_variants(list(QUICK_VARIANT_KEYS))` — 5 rows instead of 8.
3. `teacher_variant="b1"` (13.8M, still 28×).
4. `teacher_variant="b0"` **with `teacher_hr_decoder=True`** (3.9M, 8×).
5. `n_folds=2`.

**Never** drop `teacher_hr_decoder`, and never drop the `unet32` row. The HR decoder is the fix; a
stock stride-4 teacher loses to the student and is unusable at any encoder size. The `unet32` row is
the only like-for-like comparison in the paper.

**Never** disable the fingerprint check to reuse cached folds across a config change. That is
exactly how the previous run produced byte-identical scores for two different objectives.

---

## Running it

### On Kaggle

1. New notebook, GPU accelerator, internet **on** (the teacher downloads pretrained weights).
2. Add the MS3SEG dataset.
3. Upload `notebooks/ms_kd_segmentation.ipynb`.
4. Set `cfg.data_root` in the configuration cell — or leave it; the cell auto-detects any directory
   under `/kaggle/input` containing `MS_100_patient_registered/`.
5. Run all.

Sessions time out. `max_hours_per_fold` stops a fold cleanly and every stage resumes from its
checkpoints, so rerunning the notebook continues rather than restarting. Download
`/kaggle/working/msdistill_out/` between sessions.

### Locally

```bash
pip install -r requirements.txt
python scripts/smoke_test.py          # ~3 min, CPU, no dataset needed
```

---

## Producing the paper

```bash
# after the notebook has run
cp -r msdistill_out/tables  ../tables
cp -r msdistill_out/figures ../figures

python scripts/make_fallbacks.py           # regenerate placeholders for any new prose macros
python scripts/make_fallbacks.py --check   # MUST exit 0 before submitting

cd .. && pdflatex main && pdflatex main
```

`--check` exits non-zero while any cited macro lacks a measured value. A PDF containing a red
`[PENDING]` marker must never be uploaded.

---

## Pre-submission checklist

- [ ] `python scripts/make_fallbacks.py --check` exits 0
- [ ] No `[PENDING]` anywhere in the compiled PDF
- [ ] Exactly 6 pages; no overfull `\hbox` warnings

  `python scripts/check_latex.py` estimates length statically and currently reports ~6.2 pages —
  within the heuristic's error, so the real answer comes from compiling. If the compiled PDF runs
  over, cut in this order:

  1. the IoU columns from Table I
  2. rows from Table II, down to Teacher / Scratch / Hinton KD / Proposed
  3. Fig. 2 (per-patient deltas), reporting "improved in *k* of 20 patients" in prose instead
  4. the Related Work paragraph on cross-image / masked-generative / intra-class distillation

  Do **not** cut the ablation table, the efficiency table, or any losing ablation row.
- [ ] Compiles clean from a fresh directory
- [ ] Published-baseline rows sit behind their own rule with the protocol caveat in the caption
- [ ] MS3SEG Table 7 (four-class) and Table 9 (binary) numbers are **not** merged into a range
- [ ] No parameter count attributed to the MS3SEG baselines — that paper reports none
- [ ] Accuracy table and efficiency table describe the same system (ensemble rows present)
- [ ] HD95 / NSD in millimetres using real header spacing, not unit voxels
- [ ] Any ablation row that lost is still in the table
- [ ] **Every directional claim re-checked against the numbers.** The values in the paper are
      macros and update themselves; the *directions* are prose and do not. The list of claims to
      verify is a comment block at the top of the Results section in `main.tex`. A sentence the
      table contradicts is the single most damaging thing a reviewer can find.
- [ ] Author names included (single-blind assumption)
- [ ] Repository URL in the paper resolves
- [ ] Submitted with a day of slack, not on the deadline

---

## If the numbers disappoint

`docs/DESIGN.md` §4 has the pre-written framing for each outcome, including the two that are more
likely than the intended one. The short version:

- **Ladder does not climb monotonically** — reframe as *which signal survives sparsity*; keep the
  losing row in the table.
- **Teacher does not beat the student** — reframe from compression to cross-architecture transfer;
  the efficiency result is unaffected.
- **Nothing reaches significance** — lead with effect sizes, confidence intervals and seed spread;
  state the power limitation in Limitations before a reviewer states it for you.

Decide none of this after seeing the numbers. That is what the pre-written framings are for.
