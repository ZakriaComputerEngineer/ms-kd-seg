# The paper in plain language

What this work is, why it matters, every acronym defined, and answers to the questions a reviewer
or examiner is most likely to ask. Written to be read start to finish by someone who has not
followed the distillation literature.

---

## The one-paragraph version

MS lesion segmentation models are built and benchmarked on high-end scanners and need a GPU to run,
which excludes the clinics that most need them. We take a 27.5-million-parameter transformer and
distil it into a 487-thousand-parameter network that processes a full patient scan in half a second
on a CPU, matching a conventional model sixteen times its size. Along the way we show two things:
what disqualifies a transformer as a teacher for tiny lesions is its *output resolution*, not its
size — and that when 99% of your pixels are background, distillation components do not combine the
way the literature assumes. One technique that is actively harmful on its own turns out to help in
combination, which means the usual practice of validating components separately gives the wrong
answer.

---

## 1. The medical problem

Multiple sclerosis damages the myelin sheath insulating nerve fibres in the brain. On an MRI that
damage appears as bright spots in the white matter, called **hyperintensities** — literally
"brighter than they should be."

The complication: **not every bright spot is MS.** As people age they accumulate harmless bright
spots from ordinary small-vessel changes. A radiologist tracking a patient's MS needs to count only
the *disease* lesions, because that count drives treatment decisions. Confusing the two either
invents disease progression that is not there, or hides progression that is.

The task therefore has four labels per pixel:

| Label | What it is | Share of voxels |
|---|---|---|
| Background | Everything else | 99.33% |
| Ventricles | Fluid-filled cavities — bright, and easily confused with lesions | 0.47% |
| **Incidental WMH** | Harmless age-related bright spots | 0.05% |
| **Pathological WMH** | Actual MS lesions | 0.15% |

Separating the last two is the hard part. They look similar, and both are tiny — frequently 2–6
pixels across at the working resolution. The extreme class imbalance in that final column is what
every design decision in the paper is reacting to.

---

## 2. The computing problem

The strongest models for this are large transformers needing a serious GPU. The dataset used here,
**MS3SEG**, was collected in Tabriz, Iran, on a 1.5-tesla Toshiba scanner in routine clinical
practice — deliberately not flagship equipment. Clinics producing data like this generally do not
have a GPU in radiology.

That is the gap the paper addresses: **accuracy is reported on hardware the target sites do not
own.**

---

## 3. What we built

A **teacher–student** setup, the standard approach for model compression. Train a large accurate
model (the teacher), then train a small model (the student) to imitate it.

The key idea is that the student learns more than the labels. It learns the teacher's *confidence* —
that a given pixel was 70/30 lesion-versus-background rather than a flat 100/0. That shading is
information the raw labels do not contain, and it is what "knowledge" means in knowledge
distillation.

| | Parameters | Compute (GMACs) | CPU time per scan |
|---|---|---|---|
| Teacher (MiT-B2+HR) | 27.5M | 19.79 | 8.14 s |
| Conventional U-Net | 7.8M | 12.08 | 2.89 s |
| **Our student** | **487K** | **0.77** | **0.51 s** |

56× fewer parameters than the teacher, 26× less compute, 16× faster on CPU — and it scores as well
as the U-Net sixteen times its size.

---

## 4. Two findings that were not obvious

### The off-the-shelf transformer was a bad teacher, and size was not the reason

SegFormer predicts at "stride 4": it makes decisions on a coarse grid four pixels wide, then
stretches the result back up. An MS lesion is 2–6 pixels. It physically cannot represent one.

Measured in a pilot run: the teacher scored **0.36** Dice on incidental WMH where the tiny student
scored **0.57**. The teacher was worse than the model it was supposed to be teaching, so
distillation was transferring downward.

Adding a small decoder (0.2M parameters, under 1% of the model) that restores full resolution fixed
it. **The binding constraint was output resolution, not capacity** — which matters because the
instinctive fix is to reach for a bigger encoder, and that would not have worked.

### Distillation components do not add up

Three published techniques and three combinations were tested under one protocol. The result:

- Each technique alone recovers roughly half the available gain.
- The ladder is **not monotonic** — region decoupling alone is no better than plain per-pixel
  distillation.
- **Channel-wise distillation makes the student significantly *worse* on its own** (−0.018 Dice,
  p = 0.009 on incidental WMH) **yet improves the result when combined with another term.**

Since standard practice is to validate each technique in isolation, that practice would have
discarded a component that works. This is the paper's most transferable claim.

---

## 5. Results

Held-out test set, 20 patients, Dice per patient over the complete volume:

| Model | Params | Ventricles | Incidental WMH | Pathological WMH | Mean |
|---|---|---|---|---|---|
| Teacher | 27.5M | 0.877 | 0.683 | 0.749 | 0.770 |
| U-Net (base 32) | 7.8M | 0.870 | 0.661 | 0.724 | 0.751 |
| Student, no distillation | 487K | 0.856 | 0.649 | 0.704 | 0.736 |
| Student + Hinton KD | 487K | 0.861 | 0.668 | 0.725 | 0.751 |
| Student + FitNets | 487K | 0.856 | 0.662 | 0.726 | 0.748 |
| Student + CWD | 487K | 0.859 | **0.631** | 0.713 | 0.734 |
| Student + region KL | 487K | 0.862 | 0.666 | 0.719 | 0.749 |
| Student + region + CWD | 487K | 0.864 | 0.669 | 0.714 | 0.749 |
| **Student, full objective** | **487K** | **0.866** | **0.675** | **0.728** | **0.756** |

Two things to read off it: the full objective is the best student on every class, and it beats the
16× larger U-Net on mean score. CWD alone is the only row below the no-distillation baseline.

---

## 6. The honest weak spot

**Pathological WMH** was declared the primary outcome in advance. On that class the improvement
(+0.024 Dice) is real in direction but **not statistically significant** — with 20 test patients,
variation between patients exceeds the difference between methods. On the other two classes the
result *is* significant (p < 0.001 on incidental WMH).

The paper states this plainly rather than switching to the class where the result is strongest.
Post-hoc endpoint switching is exactly what reviewers are trained to catch, and the honest version
costs less than being caught.

A related disclosure: for three single-component variants, the positive *average* is produced by a
few patients with large gains while the majority regress slightly. This shows up as a negative
rank-biserial effect size beside a positive mean difference, and it is why those rows are reported
but not interpreted.

---

## 7. Glossary

### Medical and imaging

| Term | Meaning |
|---|---|
| **MS** | Multiple sclerosis |
| **MRI** | Magnetic resonance imaging |
| **WMH** | White matter hyperintensity — a bright spot in brain white matter |
| **Incidental WMH** | Harmless, age-related. The dataset calls this class "normal" |
| **Pathological WMH** | An actual MS lesion. The dataset calls this class "abnormal" |
| **T1 / T2** | Two MRI sequences; different tissues appear bright in each |
| **FLAIR** | FLuid-Attenuated Inversion Recovery — an MRI sequence that suppresses the bright cerebrospinal-fluid signal so lesions stand out. The most informative of the three inputs |
| **MS3SEG** | The dataset. 100 patients, single 1.5 T Toshiba scanner, published April 2026 |
| **Voxel** | A 3-D pixel — one sample in a volume |

### Architectures

| Term | Meaning |
|---|---|
| **CNN** | Convolutional neural network |
| **U-Net** | The standard medical segmentation architecture, named for its U shape: downsample, then upsample with shortcut connections |
| **Transformer** | An architecture built on attention, where every location can consult every other |
| **SegFormer** | A transformer designed for segmentation |
| **MiT** | Mix Transformer — SegFormer's encoder. **B0/B1/B2** are size variants; B2 is ours |
| **HR** | High-Resolution — *our* addition. "MiT-B2+HR" means their encoder plus our refinement decoder |
| **MLP** | Multi-layer perceptron. SegFormer's decoder is "All-MLP" |
| **Stride** | How coarse a model's output grid is. Stride 4 means one prediction per 4×4 block — the flaw we fixed |
| **UNETR / Swin UNETR** | Transformer segmentation models; both lose to a plain U-Net on this dataset |
| **ADE20K / ImageNet** | Large natural-photo datasets used to pre-train the teacher before fine-tuning on brain scans |

### Distillation

| Term | Meaning |
|---|---|
| **KD** | Knowledge Distillation — the teacher→student idea as a whole |
| **KL** | Kullback–Leibler divergence — measures how different two probability distributions are. The loss for "student should predict what the teacher predicts" |
| **CWD** | Channel-wise Distillation — matches each class's spatial pattern separately. The component that hurts alone but helps in combination |
| **FitNets** | A 2015 method that matches internal features rather than outputs |
| **Region-decoupled** | Our term: weight the loss more heavily near lesions, so the 99% background does not drown the signal |
| **Temperature** | A softening control. Higher temperature makes the teacher's confidence less extreme, exposing more of its uncertainty |
| **Logits** | A model's raw output scores, before they are turned into probabilities |

### Metrics

| Term | Meaning |
|---|---|
| **Dice** | Overlap between prediction and truth. 0 = none, 1 = perfect. The main metric |
| **IoU** | Intersection over Union — same idea, harsher scale |
| **HD95** | 95th-percentile Hausdorff Distance — how far the boundary is off, in mm. **Lower is better** |
| **NSD** | Normalized Surface Dice — fraction of boundary within 2 mm of correct. Kinder than HD95 on tiny structures |
| **Lesion F1 / TPR / FDR** | Counting whole *lesions* rather than pixels. TPR = fraction found, FDR = fraction of findings that were false. Radiologists count lesions, so this matters clinically |
| **Sensitivity / Precision** | Fraction of true lesion tissue found / fraction of flagged tissue that was real |
| **FG** | Foreground — everything that is not background |
| **SD / CI** | Standard deviation / confidence interval |

### Statistics

| Term | Meaning |
|---|---|
| **Wilcoxon signed-rank** | A test comparing two methods on the same patients without assuming a bell-curve distribution |
| **Holm–Bonferroni** | A correction for running many tests at once. Test seven things and one looks good by luck; this compensates |
| **p-value** | Probability of seeing this result if there were no real effect. Below 0.05 is the convention |
| **Rank-biserial (r)** | Effect size. It can be **negative while the mean difference is positive** — meaning a few patients improved a lot while most got slightly worse. That happened here, and is why three variants are not interpreted |
| **Bootstrap** | Resampling patients repeatedly to estimate uncertainty |
| **Paired** | Comparing methods on the *same* patients, which cancels out between-patient variation |

### Compute

| Term | Meaning |
|---|---|
| **GMACs** | Billions of multiply-accumulate operations — the honest measure of compute cost |
| **FLOPs** | Floating-point operations; roughly 2 × MACs |
| **AMP** | Automatic Mixed Precision — using 16-bit numbers where safe, for speed |
| **AdamW** | The optimiser (Adam with decoupled weight decay) |
| **CV** | Cross-validation — rotating which patients are held out, so results do not hinge on one lucky split |
| **Fold** | One rotation of cross-validation |
| **Ensemble** | Averaging several models' predictions. More accurate, but costs one forward pass per member |
| **T4 / CUDA** | The NVIDIA GPU used for training / NVIDIA's GPU programming layer |

---

## 8. Questions you should expect, and the answers

**"Isn't distilling SegFormer into a U-Net obvious? Where is the novelty?"**
We do not claim a novel loss, and the paper says so. The contribution is the controlled study: three
published objectives and three combinations under one protocol, on a cohort where foreground is
under half a percent. The finding — that a component harmful in isolation helps in combination — is
not obtainable from the literature, because the literature validates these components separately.
Plus the first independent benchmark on this dataset and a deployable artefact.

**"Your primary endpoint isn't significant. Why should I believe the method works?"**
On two of three classes it is significant, including p < 0.001 on incidental WMH, where the
improvement exceeds that of a U-Net sixteen times larger. On the primary class we report a point
estimate with a confidence interval and explicitly decline to claim significance. Twenty test
patients is not enough to resolve a 0.024 Dice difference against this much between-patient
variance, and we say so rather than working around it.

**"Why is the teacher only 27.5M? That's not large."**
Large enough for a 56× compression ratio, and the paper's argument is that resolution rather than
capacity is the binding constraint — demonstrated by a 3.7M stock model scoring *below* the 487K
student. A larger teacher would raise the ceiling and is listed as future work.

**"Why not just use nnU-Net?"**
Different question. nnU-Net optimises accuracy given compute; we ask what accuracy survives when
compute is fixed at CPU-only. It would be a reasonable stronger teacher, which we say in Limitations.

**"Your HD95 values are enormous."**
5.73 mm slice thickness. One missed lesion in an adjacent slice contributes roughly 6 mm, and HD95
on scattered small structures is dominated by the single worst case. This affects every model
including the teacher, which is why normalised surface Dice is reported alongside it.

**"How do I know the numbers in the paper match the code?"**
No number in the paper is typed by hand. The evaluation emits a LaTeX macro per quantity and the
manuscript cites those macros; a pre-submission check fails if any cited macro lacks a measured
value. Every table is generated. See the repository README.
