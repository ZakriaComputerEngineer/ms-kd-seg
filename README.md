# Region-Decoupled Distillation for CPU-Deployable MS Lesion Segmentation

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-MS3SEG%20CC--BY--4.0-green.svg)](https://doi.org/10.6084/m9.figshare.30393475)

Reference implementation for a study of **knowledge distillation from a transformer teacher into a
0.49M-parameter U-Net** for four-class multiple sclerosis lesion segmentation on
[MS3SEG](https://doi.org/10.1038/s41597-026-07184-5). Accompanies a paper submitted to IEEE FIT 2026.

The task separates *incidental* age-related white matter hyperintensity (WMH) from *pathological*
demyelinating lesion — a distinction that matters clinically and is usually collapsed. The
deployment target is a clinical workstation CPU, not a GPU.

> **New to the topic?** [`docs/PAPER_EXPLAINED.md`](docs/PAPER_EXPLAINED.md) explains the whole
> study in plain language, defines every acronym used here and in the paper, and answers the
> questions a reviewer is most likely to ask.

```
Teacher   MiT-B2 + full-resolution refinement decoder   27.5M params   19.6 GMACs
Student   U-Net, base width 8                            0.49M params    0.77 GMACs
                                                         56× params     25× MACs
```

---

## What is actually being tested

Under a background prior above 99%, the components of a standard segmentation distillation
objective do not contribute equally. The ablation is built to say **which one earns the gain**:

| Variant | Objective | Question |
|---|---|---|
| `unet32` | hard labels, base width 32 | Same-protocol conventional U-Net (7.8M) |
| `scratch` | hard labels | What can the compact net do alone? |
| `kd_vanilla` | uniform per-pixel KL | Does textbook distillation help? |
| `kd_fitnets` | pooled feature MSE | Does the standard feature recipe help? |
| `kd_cwd` | channel-wise KL | Does prior dense-prediction KD help? |
| `kd_region` | region-decoupled KL | Does decoupling foreground from background help? |
| `kd_region_cwd` | region KL + channel-wise | Is the channel-wise term additive? |
| `kd_full` | + spatial feature alignment | Full proposed objective |

Three rows (`kd_vanilla`, `kd_fitnets`, `kd_cwd`) reproduce published objectives — Hinton KD,
FitNets and CWD — so the comparison is against prior art, not only against our own ablation.

Every student row trains **the same architecture from the same initialization on the same batches
in the same order**. All variants for a fold train in a single pass sharing one data loader and one
teacher forward per batch. Differences between rows cannot be attributed to data ordering, and the
per-patient scores are properly paired for significance testing.

---

## Three corrections that changed the conclusions

An earlier version of this pipeline produced results that contradicted its own premise. The causes
were mechanical. They are documented here because each is easy to reproduce by accident.

### 1. The teacher was the weaker model

Stock SegFormer emits logits at **stride 4** and reaches input resolution by one bilinear upsample.
At 256×256 an MS lesion is often 2–6 pixels across, so a stride-4 prediction cannot represent one.

> Measured: SegFormer-B0 reached **0.36** Dice on incidental WMH where the 487K student reached
> **0.57**. Distillation was transferring downward.

Consistent with the MS3SEG authors' own baselines, where UNETR and Swin UNETR trail a plain U-Net by
0.05–0.14 Dice on every task. The fix is a refinement path that returns the decoder feature to
stride 1 (`models/teacher.py: HRRefinement`) — about 0.2M parameters. **Output stride, not encoder
capacity, was the binding constraint.**

### 2. The reported metric awarded free points

Dice was accumulated per mini-batch and averaged. With a smoothing constant, a batch containing
none of a class scores `(0+ε)/(0+ε) = 1.0` — a perfect score for predicting nothing. Pathological
WMH is absent from most individual slices, so this inflated the rare-class numbers and **reordered
the ranking of methods**.

Metrics are now computed once per patient over the complete volume, with a class absent from a
patient's reference recorded as undefined rather than as a free 1.0 (`metrics.py: dice_binary`).

### 3. The feature loss had a trivial optimum

`MSE(W_s·f_s, W_t·f_t)` with **both** projections learnable is minimized by `W_s = W_t = 0`. The
loss reaches zero having transferred nothing, and the gradient reaching the student meanwhile
collapses its bottleneck.

> Measured: a 0.11 Dice drop and two diverged folds.

Only the student side is projected now, features keep their spatial layout, and they are
L2-normalized per pixel so the objective constrains direction rather than magnitude
(`losses.py: spatial_feature_distillation`).

**A fourth issue was procedural.** Cached fold results from a previous configuration were silently
reused, which is why two different objectives once reported byte-identical validation scores. Every
results file now carries a configuration fingerprint and a mismatch triggers retraining
(`train.py: load_cached_results`).

---

## Quick start

### Kaggle / Colab

Open `notebooks/ms_kd_segmentation.ipynb`, attach the MS3SEG dataset, set `cfg.data_root`, run all.
The notebook is self-contained — it installs its own dependencies and needs nothing from this repo.

### Local

```bash
pip install -r requirements.txt

# End-to-end verification on synthetic data (~3 min, CPU, no network, no dataset).
python scripts/smoke_test.py

# Regenerate the notebook from src/ and verify it will run.
python scripts/build_notebook.py
python scripts/validate_notebook.py
```

`smoke_test.py` builds a miniature cohort with the same directory layout, NIfTI structure and 16-bit
label encoding as MS3SEG, then exercises every stage: discovery, label inference, caching,
patient-level splitting, teacher fine-tuning, co-trained ablation, held-out evaluation, significance
testing, efficiency profiling, LaTeX generation and figure rendering. **If it passes, the notebook
will not die on a shape mismatch after an hour of GPU training.**

---

## Repository layout

```
src/msdistill/
  config.py            Config dataclass, ablation grid, fingerprinting
  data.py              Discovery, label inference, preprocessing, augmentation, loaders
  models/teacher.py    MiT encoder + All-MLP decoder + HR refinement; frozen wrapper
  models/student.py    TinyUNet + distillation projection heads
  losses.py            Supervised + region / channel-wise / spatial-feature distillation
  metrics.py           Volume-level Dice, IoU, HD95, NSD, lesion-wise detection
  train.py             Teacher stage, co-trained student stage, resumable
  evaluate.py          Held-out evaluation, ensemble + single-model statistics
  stats.py             Wilcoxon, Holm–Bonferroni, bootstrap CI, rank-biserial
  efficiency.py        Params, MACs, GPU/CPU latency, memory, throughput
  report.py            LaTeX tables and prose macros
  viz.py               Publication figures

scripts/
  smoke_test.py        End-to-end synthetic-data verification
  build_notebook.py    Assemble the notebook from src/ (never edit the .ipynb)
  validate_notebook.py Compile + execute-check every cell
  make_fallbacks.py    Placeholder macros so the paper compiles before results exist

notebooks/ms_kd_segmentation.ipynb    Generated. Do not edit by hand.
```

**The notebook is generated.** Editing `src/` and rerunning `build_notebook.py` is the only
supported workflow; a hand-edited `.ipynb` will be overwritten and the two copies would otherwise
drift apart.

---

## Evaluation protocol

Designed against the objections a reviewer raises about small-cohort segmentation studies.

**Aggregation.** Per patient, over the complete 3-D volume. Pooled dataset-level Dice is reported as
a secondary number, because it weights patients by lesion load whereas the per-patient mean weights
subjects equally — papers reporting only one can look substantially better or worse than they are.

**Beyond overlap.** Voxel Dice answers *how much lesion tissue was outlined*. Radiologists track
*how many distinct lesions* appear over time. Lesion-wise F1 / TPR / FDR from connected-component
matching are reported alongside HD95 and normalized surface Dice at 2 mm, computed with each
volume's true header spacing.

**Significance.** Paired Wilcoxon signed-rank across test patients — not a *t*-test, since
per-patient Dice on rare classes is bounded and spikes at zero — with a percentile bootstrap 95% CI
on the mean paired difference and a matched-pairs rank-biserial effect size. Holm–Bonferroni within
each class. Two families reported separately: *versus baseline* (does distillation help at all) and
*consecutive ladder rungs* (does each component help).

**Seed spread as a floor.** Differences smaller than the spread across independent seeds are
reported but not interpreted.

**Efficiency.** Median of 50 timed passes after 10 warmup iterations, with device synchronization
inside the timed region; CPU timing at 4 threads. Parameters, MACs, GPU and CPU latency, peak
memory and per-study wall time — because the three ratios differ by more than an order of magnitude
and a parameter-count-only claim is not reliable. The K-fold ensemble's cost is reported too, so the
accuracy and efficiency tables describe the same system.

---

## Configuration

Everything lives in `Config` (`src/msdistill/config.py`). `Config.fingerprint()` hashes the fields
that affect training; cached fold results whose fingerprint differs are discarded rather than merged.

| Field | Default | Note |
|---|---|---|
| `teacher_variant` | `"b2"` | `"b1"` (13.8M) / `"b0"` (3.9M) are lower-cost fallbacks |
| `teacher_hr_decoder` | `True` | **Do not disable.** This is the correction, not the encoder size |
| `n_folds` | `3` | Patient-level |
| `kd_fg_weight` | `5.0` | Foreground upweighting in the region-decoupled divergence |
| `kd_region_dilation` | `2` | Weighted region grown past the annotation, where teacher and student disagree most |
| `kd_warmup_epochs` | `5` | Linear ramp on every distillation term |
| `max_hours_per_fold` | `3.0` | Wall-clock guard; a stopped fold resumes, it is not lost |
| `parallel_folds` | `1` | Set to `2` on Kaggle's *GPU T4 × 2*. Folds are independent, so this needs no inter-GPU communication — unlike `nn.DataParallel`, which was measurably **slower** here |

Reduced grids: `QUICK_VARIANT_KEYS` (5 rows) for tight budgets. Keep `unet32` even under pressure —
a same-protocol conventional U-Net is worth more to a reviewer than two extra ablation rungs.

---

## Reproducing the paper's numbers

No number in the manuscript is typed by hand.

1. Run the notebook.
2. Copy `results/tables/` into the paper directory.
3. `python scripts/make_fallbacks.py --check` — exits non-zero if any cited macro lacks a measured
   value, so a PDF containing `[PENDING]` can never be submitted.
4. Compile.

`results_macros.tex` defines a LaTeX command per reported quantity and the manuscript cites those
commands, so rerunning the experiments updates the paper.

---

## Dataset

MS3SEG — 100 MS patients, single 1.5 T Toshiba/Canon Vantage, Tabriz, Iran; co-registered T1, T2 and
T2-FLAIR at 0.90 × 0.90 × 5.73 mm; four-class expert annotation.

- Paper: Bashiri Bawil et al., *Scientific Data* **13**, 867 (2026).
  [doi:10.1038/s41597-026-07184-5](https://doi.org/10.1038/s41597-026-07184-5)
- Data: [doi:10.6084/m9.figshare.30393475](https://doi.org/10.6084/m9.figshare.30393475) (CC BY 4.0)
- Upstream code: [github.com/Mahdi-Bashiri/MS3SEG](https://github.com/Mahdi-Bashiri/MS3SEG) (MIT)

**No patient data is redistributed here.** Ethics approval (IR.TBZMED.REC.1402.902, Tabriz
University of Medical Sciences) and consent are recorded in the data descriptor.

### Getting it

**On Kaggle** — add the mirrored dataset as an input (search `ms3seg`); the notebook locates it
automatically, following symlinks, and prints a directory listing if it cannot.

**Elsewhere** — download the Figshare record and extract `MS_100_patient_registered.rar` and
`MS_100_model_input.rar`. Point `Config.data_root` at the directory containing both. The layout the
code expects:

```
<data_root>/
  MS_100_patient_registered/<id>/<id>_{T1WI_reg,T2WI_reg,FLAIR}.nii
  MS_100_model_input/man_4L_masks_new/<id>.nii
```

You do not need the other archives (`_full`, `_nifti`, `_gifs`) — about 5 GB of the 6.7 GB total.

Class balance is extreme, which is what every component of the objective is designed around. The
data descriptor reports background 99.55%, ventricles 0.30%, pathological WMH 0.10%, incidental WMH
0.05%; measured on the 256×256 resampled volumes this pipeline trains on, it is 99.33 / 0.47 / 0.15
/ 0.05. The paper quotes the measured figures, since those are what the loss actually sees.

> **Note on published baselines.** Numbers from the MS3SEG paper's Table 7 (four-class:
> U-Net 0.8897 / 0.6452 / 0.6686) and Table 9 (binary lesion-only: U-Net 0.7469) are **different
> experiments** and must not be merged into a range. That paper reports no parameter counts. Its
> GitHub README results table diverges from the published article on incidental WMH for every
> architecture — cite the article. Its protocol also differs from ours (single-channel FLAIR, focal
> loss, 30 epochs, no augmentation, 5-fold, slice-level aggregation), which is why the
> `unet32` row exists.

---

## Citation

```bibtex
@inproceedings{msdistill2026,
  title     = {Region-Decoupled Distillation of a Transformer Teacher for
               Multiple Sclerosis Lesion Segmentation},
  author    = {Mehmood, M. Zakria},
  booktitle = {Proc. 23rd Int. Conf. Frontiers of Information Technology (FIT)},
  year      = {2026},
  url       = {https://github.com/ZakriaComputerEngineer/ms-kd-seg},
}
```

Please cite the dataset as well — this work reports on it but did not create it:

```bibtex
@article{ms3seg2026,
  title   = {A multiple sclerosis {MRI} dataset with tri-mask annotations for
             lesion segmentation},
  author  = {Bashiri Bawil, Mahdi and Shamsi, Mousa and Ghalehasadi, Aydin and
             Fahmi Jafargholkhanloo, Ali and Shakeri Bavil, Abolhassan},
  journal = {Scientific Data},
  volume  = {13},
  pages   = {867},
  year    = {2026},
  doi     = {10.1038/s41597-026-07184-5},
}
```

## License

MIT for this code — see [LICENSE](LICENSE). MS3SEG is CC BY 4.0 and is **not** redistributed here;
obtain it from Figshare. Pretrained Mix Transformer weights are downloaded at run time from the
Hugging Face Hub and remain under NVIDIA's terms for those checkpoints.
