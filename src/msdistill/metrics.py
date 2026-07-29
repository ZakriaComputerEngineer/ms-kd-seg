"""Segmentation metrics computed at the level of a whole patient volume.

The single most consequential correction relative to our earlier pipeline is
where the averaging happens. Accumulating Dice per mini-batch and averaging the
result is wrong for scattered, rare structures: a batch containing no voxels of a
class yields `(0 + eps) / (0 + eps) = 1.0`, a perfect score for having predicted
nothing. Since abnormal WMH is absent from most individual slices, that single
convention inflated the rare-class scores and reordered the ranking of methods.

Here every metric is computed once per patient over the complete 3-D volume, and
a class that is absent from a patient's reference is reported as undefined
(`nan`) for that patient rather than as a free 1.0. Patient-level values are then
averaged, which is also what makes paired significance testing across patients
possible.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import numpy as np

try:
    from scipy import ndimage as ndi
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    ndi = None
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------- #
# Overlap
# --------------------------------------------------------------------------- #

def dice_binary(pred: np.ndarray, ref: np.ndarray) -> float:
    """Dice for one binary volume pair.

    Returns `nan` when the reference is empty and the prediction is also empty --
    the metric is genuinely undefined there. Returns 0.0 when the reference is
    empty but the prediction is not, because that is a pure false positive and
    scoring it as undefined would hide it.
    """
    p, r = pred.sum(), ref.sum()
    if r == 0:
        return float("nan") if p == 0 else 0.0
    inter = np.logical_and(pred, ref).sum()
    return float(2.0 * inter / (p + r))


def iou_binary(pred: np.ndarray, ref: np.ndarray) -> float:
    p, r = pred.sum(), ref.sum()
    if r == 0:
        return float("nan") if p == 0 else 0.0
    inter = np.logical_and(pred, ref).sum()
    union = np.logical_or(pred, ref).sum()
    return float(inter / union) if union > 0 else float("nan")


def sensitivity_precision(pred: np.ndarray, ref: np.ndarray) -> Tuple[float, float]:
    tp = float(np.logical_and(pred, ref).sum())
    fn = float(np.logical_and(~pred, ref).sum())
    fp = float(np.logical_and(pred, ~ref).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return sens, prec


# --------------------------------------------------------------------------- #
# Boundary
# --------------------------------------------------------------------------- #

def _surface_distances(pred: np.ndarray, ref: np.ndarray,
                       spacing: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Distances from each surface to the *other surface*, in millimetres.

    The distance transform must be built from the complement of the other
    *surface*, not of the other filled object. Using the filled mask records any
    surface voxel lying inside the other object at 0 mm, which is not a surface
    distance at all: a hollow reference then scores a perfect NSD of 1.0 and an
    HD95 of 0 against a prediction that is nowhere near its boundary. That error
    inflates NSD on essentially every case and biases HD95 low.
    """
    structure = ndi.generate_binary_structure(pred.ndim, 1)
    pred_surface = np.logical_xor(pred, ndi.binary_erosion(pred, structure, border_value=0))
    ref_surface = np.logical_xor(ref, ndi.binary_erosion(ref, structure, border_value=0))
    if not pred_surface.any() or not ref_surface.any():
        return np.array([]), np.array([])

    dt_to_ref = ndi.distance_transform_edt(~ref_surface, sampling=spacing)
    dt_to_pred = ndi.distance_transform_edt(~pred_surface, sampling=spacing)
    return dt_to_ref[pred_surface], dt_to_pred[ref_surface]


def _max_distance(shape: Sequence[int], spacing: Sequence[float]) -> float:
    """Worst attainable surface distance in a volume: its diagonal, in mm."""
    return float(np.linalg.norm(np.asarray(shape, dtype=float)
                                * np.asarray(spacing, dtype=float)))


def hausdorff95(pred: np.ndarray, ref: np.ndarray, spacing: Sequence[float]) -> float:
    """95th-percentile symmetric Hausdorff distance, in millimetres.

    An empty prediction against a non-empty reference is scored at the volume
    diagonal, following the BraTS and nnU-Net convention, rather than as
    undefined. Returning `nan` there would delete the worst possible outcome --
    a total miss -- from the mean and from every paired test, which
    systematically flatters whichever variant segments least. That is precisely
    the axis this ablation varies.
    """
    if not _HAVE_SCIPY:
        return float("nan")
    p, r = pred.sum(), ref.sum()
    if p == 0 and r == 0:
        return float("nan")                        # genuinely undefined
    if p == 0 or r == 0:
        return _max_distance(pred.shape, spacing)  # total miss / pure false positive
    d_pr, d_rp = _surface_distances(pred, ref, spacing)
    if d_pr.size == 0 or d_rp.size == 0:
        return float("nan")
    return float(max(np.percentile(d_pr, 95), np.percentile(d_rp, 95)))


def normalized_surface_dice(pred: np.ndarray, ref: np.ndarray, spacing: Sequence[float],
                            tolerance_mm: float = 2.0) -> float:
    """Fraction of the two surfaces lying within `tolerance_mm` of each other.

    Preferred over Hausdorff for small structures because it degrades gracefully
    rather than being dominated by a single worst-case voxel.
    """
    if not _HAVE_SCIPY:
        return float("nan")
    p, r = pred.sum(), ref.sum()
    if p == 0 and r == 0:
        return float("nan")
    if p == 0 or r == 0:
        return 0.0      # a total miss agrees with nothing; see hausdorff95
    d_pr, d_rp = _surface_distances(pred, ref, spacing)
    if d_pr.size == 0 or d_rp.size == 0:
        return float("nan")
    hits = (d_pr <= tolerance_mm).sum() + (d_rp <= tolerance_mm).sum()
    return float(hits / (d_pr.size + d_rp.size))


# --------------------------------------------------------------------------- #
# Lesion-wise detection
# --------------------------------------------------------------------------- #

@dataclass
class LesionCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    n_ref: int = 0
    n_pred: int = 0

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return float(2 * self.tp / denom) if denom > 0 else float("nan")

    @property
    def tpr(self) -> float:
        return float(self.tp / self.n_ref) if self.n_ref > 0 else float("nan")

    @property
    def fdr(self) -> float:
        if self.n_pred > 0:
            return float(self.fp / self.n_pred)
        # Predicting nothing when there was something to find is a false-discovery
        # rate of 0, not an undefined value to be dropped from the average.
        return float("nan") if self.n_ref == 0 else 0.0


def lesion_wise_counts(pred: np.ndarray, ref: np.ndarray, overlap_threshold: float = 0.10,
                       min_voxels: int = 3) -> LesionCounts:
    """Connected-component detection statistics with one-to-one matching.

    Voxel Dice answers "how much of the lesion tissue did you outline"; this
    answers "how many distinct lesions did you find", which is the quantity MS
    radiologists actually track over time. A model can score well on one and
    poorly on the other, so the paper reports both.

    Matching is **one-to-one and greedy by IoU**. Scoring each reference
    component against the union of all predictions instead lets a single merged
    blob claim an unlimited number of true positives at a cost of one false
    positive: a prediction covering the entire slice scored lesion-F1 0.909
    against five reference lesions, beating an accurate four-of-five prediction
    at 0.800, while its voxel Dice was 0.072. Reference components below
    `min_voxels` are excluded from scoring, and a prediction that lands on one of
    them is ignored rather than penalised as a spurious finding.
    """
    counts = LesionCounts()
    if not _HAVE_SCIPY:
        return counts

    structure = ndi.generate_binary_structure(ref.ndim, 2)
    ref_lab, n_ref = ndi.label(ref, structure=structure)
    pred_lab, n_pred = ndi.label(pred, structure=structure)

    ref_sizes = np.bincount(ref_lab.ravel(), minlength=n_ref + 1)
    pred_sizes = np.bincount(pred_lab.ravel(), minlength=n_pred + 1)
    ref_ids = [i for i in range(1, n_ref + 1) if ref_sizes[i] >= min_voxels]
    pred_ids = [j for j in range(1, n_pred + 1) if pred_sizes[j] >= min_voxels]

    counts.n_ref = len(ref_ids)
    used_ref: set = set()
    used_pred: set = set()

    if ref_ids and pred_ids:
        # Every pairwise intersection in one pass, keyed by (ref_label, pred_label).
        both = (ref_lab > 0) & (pred_lab > 0)
        inter = np.bincount(
            ref_lab[both].astype(np.int64) * (n_pred + 1) + pred_lab[both].astype(np.int64),
            minlength=(n_ref + 1) * (n_pred + 1))

        kept_ref, kept_pred = set(ref_ids), set(pred_ids)
        candidates = []
        for code in np.nonzero(inter)[0]:
            i, j = divmod(int(code), n_pred + 1)
            if i in kept_ref and j in kept_pred:
                n_int = int(inter[code])
                iou = n_int / float(ref_sizes[i] + pred_sizes[j] - n_int)
                if iou >= overlap_threshold:
                    candidates.append((iou, i, j))

        candidates.sort(reverse=True)
        for _, i, j in candidates:
            if i in used_ref or j in used_pred:
                continue
            used_ref.add(i)
            used_pred.add(j)
            counts.tp += 1

    # A predicted component sitting on reference tissue that was excluded for
    # being under `min_voxels` earns no credit, but must not be counted against
    # the model either -- it found something real that we chose not to score.
    # This applies even when *every* reference component was subthreshold, which
    # is why it sits outside the matching branch above.
    if pred_ids:
        excluded = (ref_lab > 0) & ~np.isin(ref_lab, ref_ids)
        ignored = sum(1 for j in pred_ids
                      if j not in used_pred and np.logical_and(pred_lab == j, excluded).any())
    else:
        ignored = 0

    counts.fn = len(ref_ids) - len(used_ref)
    counts.fp = len(pred_ids) - len(used_pred) - ignored
    counts.n_pred = len(pred_ids) - ignored
    return counts


# --------------------------------------------------------------------------- #
# Per-case container
# --------------------------------------------------------------------------- #

@dataclass
class CaseMetrics:
    patient_id: str
    dice: Dict[str, float] = field(default_factory=dict)
    iou: Dict[str, float] = field(default_factory=dict)
    sensitivity: Dict[str, float] = field(default_factory=dict)
    precision: Dict[str, float] = field(default_factory=dict)
    hd95: Dict[str, float] = field(default_factory=dict)
    nsd: Dict[str, float] = field(default_factory=dict)
    lesion_f1: Dict[str, float] = field(default_factory=dict)
    lesion_tpr: Dict[str, float] = field(default_factory=dict)
    lesion_fdr: Dict[str, float] = field(default_factory=dict)
    reference_voxels: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "patient_id": self.patient_id,
            "dice": self.dice, "iou": self.iou,
            "sensitivity": self.sensitivity, "precision": self.precision,
            "hd95": self.hd95, "nsd": self.nsd,
            "lesion_f1": self.lesion_f1, "lesion_tpr": self.lesion_tpr,
            "lesion_fdr": self.lesion_fdr,
            "reference_voxels": self.reference_voxels,
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "CaseMetrics":
        """Rebuild from `test_evaluations.json`.

        Lets tables and figures be regenerated from a completed run without
        repeating inference -- which matters when the only GPU available is a
        session-limited one and a caption needs changing.
        """
        return cls(
            patient_id=payload["patient_id"],
            dice=payload.get("dice", {}), iou=payload.get("iou", {}),
            sensitivity=payload.get("sensitivity", {}), precision=payload.get("precision", {}),
            hd95=payload.get("hd95", {}), nsd=payload.get("nsd", {}),
            lesion_f1=payload.get("lesion_f1", {}), lesion_tpr=payload.get("lesion_tpr", {}),
            lesion_fdr=payload.get("lesion_fdr", {}),
            reference_voxels=payload.get("reference_voxels", {}),
        )


def evaluate_volume(prediction: np.ndarray, reference: np.ndarray, cfg,
                    patient_id: str, spacing: Sequence[float] = (1.0, 1.0, 1.0),
                    compute_boundary: bool = True) -> CaseMetrics:
    """All metrics for one patient's full 3-D label volume.

    `prediction` and `reference` are integer label volumes of identical shape.
    """
    if prediction.shape != reference.shape:
        raise ValueError(f"{patient_id}: prediction {prediction.shape} != reference {reference.shape}")

    case = CaseMetrics(patient_id=patient_id)
    for class_idx, name in enumerate(cfg.class_names):
        pred_c = prediction == class_idx
        ref_c = reference == class_idx

        case.reference_voxels[name] = int(ref_c.sum())
        case.dice[name] = dice_binary(pred_c, ref_c)
        case.iou[name] = iou_binary(pred_c, ref_c)
        sens, prec = sensitivity_precision(pred_c, ref_c)
        case.sensitivity[name] = sens
        case.precision[name] = prec

        # Boundary metrics on background are meaningless and expensive.
        if compute_boundary and class_idx in cfg.foreground_classes:
            case.hd95[name] = hausdorff95(pred_c, ref_c, spacing)
            case.nsd[name] = normalized_surface_dice(pred_c, ref_c, spacing, tolerance_mm=2.0)

        if class_idx in cfg.lesion_classes:
            counts = lesion_wise_counts(pred_c, ref_c, cfg.lesion_match_iou, cfg.min_lesion_voxels)
            case.lesion_f1[name] = counts.f1
            case.lesion_tpr[name] = counts.tpr
            case.lesion_fdr[name] = counts.fdr

    return case


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def nanmean(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values], dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(arr))


def collect_metric(cases: Sequence[CaseMetrics], metric: str, class_name: str) -> np.ndarray:
    """Per-patient vector for one (metric, class) pair, preserving patient order.

    Significance testing depends on this ordering: the Wilcoxon test in
    `stats.py` pairs entry i of two models' vectors, which is only valid if both
    were produced from the same patient list in the same order.
    """
    return np.array([getattr(c, metric).get(class_name, float("nan")) for c in cases],
                    dtype=np.float64)


def summarize(cases: Sequence[CaseMetrics], cfg) -> Dict[str, Dict[str, Dict[str, float]]]:
    """{metric: {class: {"mean":…, "std":…, "n":…}}} over the patient cohort."""
    metrics = ["dice", "iou", "sensitivity", "precision", "hd95", "nsd",
               "lesion_f1", "lesion_tpr", "lesion_fdr"]
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for metric in metrics:
        per_class: Dict[str, Dict[str, float]] = {}
        for name in cfg.class_names:
            values = collect_metric(cases, metric, name)
            defined = values[~np.isnan(values)]
            if defined.size == 0:
                continue
            per_class[name] = {
                "mean": float(defined.mean()),
                "std": float(defined.std(ddof=1)) if defined.size > 1 else 0.0,
                "median": float(np.median(defined)),
                "n": int(defined.size),
                "n_undefined": int(values.size - defined.size),
            }
        if per_class:
            out[metric] = per_class

    # Mean over the clinically meaningful classes, the headline scalar.
    fg_names = [cfg.class_names[i] for i in cfg.foreground_classes]
    if "dice" in out:
        present = [out["dice"][n]["mean"] for n in fg_names if n in out["dice"]]
        if present:
            out.setdefault("summary", {})["mean_foreground_dice"] = {
                "mean": float(np.mean(present)), "std": 0.0, "n": len(present),
            }
    return out


def global_dice(predictions: Sequence[np.ndarray], references: Sequence[np.ndarray],
                cfg) -> Dict[str, float]:
    """Dice computed once over the pooled cohort rather than per patient.

    Reported as a secondary number. It weights patients by lesion load, so it is
    dominated by the few subjects with the highest burden, whereas the per-patient
    mean weights every subject equally. Papers that report only one of the two can
    look substantially better or worse than they are; we give both.
    """
    inter = np.zeros(cfg.num_classes, dtype=np.float64)
    p_sum = np.zeros(cfg.num_classes, dtype=np.float64)
    r_sum = np.zeros(cfg.num_classes, dtype=np.float64)
    for pred, ref in zip(predictions, references):
        for c in range(cfg.num_classes):
            pc, rc = pred == c, ref == c
            inter[c] += np.logical_and(pc, rc).sum()
            p_sum[c] += pc.sum()
            r_sum[c] += rc.sum()
    out: Dict[str, float] = {}
    for c, name in enumerate(cfg.class_names):
        denom = p_sum[c] + r_sum[c]
        out[name] = float(2.0 * inter[c] / denom) if denom > 0 else float("nan")
    return out
