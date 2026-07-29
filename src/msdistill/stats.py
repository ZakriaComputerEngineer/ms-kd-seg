"""Significance testing and uncertainty estimates for paired model comparisons.

Reviewers of segmentation papers routinely reject "0.62 vs 0.59, therefore our
method is better" when n = 20 patients. Everything reported in the paper's
comparison table is therefore accompanied by a paired test and a confidence
interval.

The tests are *paired* because every model is evaluated on the same patients in
the same order, and the models are additionally trained on identical batch
sequences (see `train.py`). Pairing removes between-patient variance, which on
this cohort is far larger than the between-method differences we are trying to
resolve.

Wilcoxon signed-rank is used rather than a paired t-test: per-patient Dice is
bounded, skewed, and on the rare classes has a spike at zero, so normality does
not hold. Multiplicity is controlled with Holm-Bonferroni, which is uniformly
more powerful than plain Bonferroni and makes no independence assumption.

Ties are handled with `zero_method="wilcox"`: pairs where the two models produce
identical values are deleted and the test runs at the reduced sample size. That
convention matters here because on the rare classes both models frequently
predict the same empty mask, so `n_effective` -- the count of non-tied pairs --
can be far below the patient count. It is recorded on every result and is what
the minimum-sample guard checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import stats as sps
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    sps = None
    _HAVE_SCIPY = False


def require_scipy() -> None:
    """Fail loudly rather than emitting a complete-looking table of dashes.

    Without scipy every p-value is nan, `holm_bonferroni` sees an empty family,
    and the generated table renders with "--" in every significance column --
    which looks like a result rather than a missing dependency.
    """
    if not _HAVE_SCIPY:
        raise ImportError(
            "scipy is required for significance testing and for the lesion-wise and "
            "boundary metrics. Install it with `pip install scipy`; without it the "
            "generated tables would silently omit every p-value."
        )


@dataclass
class ComparisonResult:
    name_a: str
    name_b: str
    metric: str
    class_name: str
    n_pairs: int
    # Pairs with a non-zero difference. `zero_method="wilcox"` discards ties, so
    # this -- not `n_pairs` -- is the sample size the test actually runs at.
    n_effective: int
    mean_a: float
    mean_b: float
    mean_difference: float
    ci_low: float
    ci_high: float
    statistic: float
    p_value: float
    p_adjusted: float = float("nan")
    effect_size: float = float("nan")   # matched-pairs rank-biserial correlation
    significant: bool = False

    def as_row(self) -> Dict:
        return {
            "comparison": f"{self.name_a} vs {self.name_b}",
            "metric": self.metric, "class": self.class_name,
            "n": self.n_pairs, "n_effective": self.n_effective,
            "mean_a": self.mean_a, "mean_b": self.mean_b,
            "delta": self.mean_difference,
            "ci95": (self.ci_low, self.ci_high),
            "p": self.p_value, "p_holm": self.p_adjusted,
            "effect_size": self.effect_size, "significant": self.significant,
        }


def paired_valid(a: Sequence[float], b: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only patients where both models produced a defined value."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"paired vectors must align: {arr_a.shape} vs {arr_b.shape}")
    keep = ~(np.isnan(arr_a) | np.isnan(arr_b))
    return arr_a[keep], arr_b[keep]


def bootstrap_ci_of_difference(a: Sequence[float], b: Sequence[float], n_resamples: int = 10000,
                               alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean paired difference (a - b).

    Patients are resampled as units, so the interval reflects uncertainty about
    which patients happened to land in the test split -- the dominant source of
    uncertainty at n = 20.
    """
    arr_a, arr_b = paired_valid(a, b)
    if arr_a.size < 2:
        return float("nan"), float("nan")
    diff = arr_a - arr_b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_resamples, diff.size))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def rank_biserial(a: Sequence[float], b: Sequence[float]) -> float:
    """Matched-pairs rank-biserial correlation: the standardized effect size that
    accompanies a Wilcoxon signed-rank test. Ranges from -1 to +1."""
    arr_a, arr_b = paired_valid(a, b)
    diff = arr_a - arr_b
    diff = diff[diff != 0]
    if diff.size == 0:
        return 0.0
    ranks = sps.rankdata(np.abs(diff)) if _HAVE_SCIPY else np.argsort(np.argsort(np.abs(diff))) + 1.0
    total = ranks.sum()
    if total == 0:
        return 0.0
    return float((ranks[diff > 0].sum() - ranks[diff < 0].sum()) / total)


def wilcoxon_compare(a: Sequence[float], b: Sequence[float], name_a: str, name_b: str,
                     metric: str, class_name: str, n_resamples: int = 10000,
                     alpha: float = 0.05, seed: int = 42) -> ComparisonResult:
    arr_a, arr_b = paired_valid(a, b)
    n = int(arr_a.size)
    # `zero_method="wilcox"` deletes tied pairs and tests at the reduced sample
    # size, so the minimum-n guard has to look at the non-tied count. Gating on
    # the raw pair count lets 18 ties plus 2 differing pairs emit a
    # normal-approximation p-value whose exact two-sided minimum is 0.5. Ties are
    # the expected case on the rare classes, where both models often predict the
    # same empty mask.
    n_effective = int(np.count_nonzero(arr_a - arr_b)) if n else 0
    mean_a = float(arr_a.mean()) if n else float("nan")
    mean_b = float(arr_b.mean()) if n else float("nan")
    delta = mean_a - mean_b if n else float("nan")

    statistic, p_value = float("nan"), float("nan")
    if n_effective >= 5 and _HAVE_SCIPY:
        try:
            res = sps.wilcoxon(arr_a, arr_b, alternative="two-sided",
                               zero_method="wilcox", correction=False)
            statistic, p_value = float(res.statistic), float(res.pvalue)
        except ValueError:
            statistic, p_value = 0.0, 1.0
    elif n_effective == 0 and n > 0:
        statistic, p_value = 0.0, 1.0   # identical on every patient

    low, high = bootstrap_ci_of_difference(arr_a, arr_b, n_resamples, alpha, seed)
    return ComparisonResult(
        name_a=name_a, name_b=name_b, metric=metric, class_name=class_name,
        n_pairs=n, n_effective=n_effective,
        mean_a=mean_a, mean_b=mean_b, mean_difference=delta, ci_low=low, ci_high=high,
        statistic=statistic, p_value=p_value,
        effect_size=rank_biserial(arr_a, arr_b) if n else float("nan"),
    )


def holm_bonferroni(results: List[ComparisonResult], alpha: float = 0.05) -> List[ComparisonResult]:
    """Step-down multiplicity correction over a family of comparisons.

    The family is defined per (metric, class): correcting across classes as well
    would be needlessly conservative, since each class answers a separate
    clinical question.
    """
    testable = [r for r in results if not np.isnan(r.p_value)]
    order = sorted(range(len(testable)), key=lambda i: testable[i].p_value)
    m = len(testable)
    running_max = 0.0
    for rank, i in enumerate(order):
        adjusted = min(1.0, (m - rank) * testable[i].p_value)
        running_max = max(running_max, adjusted)   # enforce monotonicity
        testable[i].p_adjusted = running_max
        testable[i].significant = running_max <= alpha   # Holm rejects at <= alpha
    return results


def compare_all(evaluations: Dict[str, "object"], reference_key: str, metric: str,
                class_name: str, cfg, exclude: Optional[Sequence[str]] = None
                ) -> List[ComparisonResult]:
    """Compare every model against `reference_key` on one (metric, class) pair."""
    from .metrics import collect_metric

    exclude = set(exclude or ())
    reference = evaluations[reference_key]
    ref_ids = [c.patient_id for c in reference.cases]
    ref_values = collect_metric(reference.cases, metric, class_name)

    results: List[ComparisonResult] = []
    for key, ev in evaluations.items():
        if key == reference_key or key in exclude:
            continue
        ids = [c.patient_id for c in ev.cases]
        if ids != ref_ids:
            raise ValueError(
                f"cannot pair {key} against {reference_key}: evaluated on different "
                f"patients or in a different order"
            )
        results.append(wilcoxon_compare(
            collect_metric(ev.cases, metric, class_name), ref_values,
            name_a=key, name_b=reference_key, metric=metric, class_name=class_name,
            n_resamples=cfg.bootstrap_samples, alpha=cfg.alpha_level, seed=cfg.seed,
        ))
    return holm_bonferroni(results, cfg.alpha_level)


def ladder_comparisons(evaluations: Dict[str, "object"], ordered_keys: Sequence[str],
                       metric: str, class_name: str, cfg) -> List[ComparisonResult]:
    """Consecutive comparisons along the ablation ladder.

    This is the evidence that each *added component* helps, as distinct from the
    evidence that the full method beats the baseline. A method can beat its
    baseline overall while one of its components contributes nothing, and the
    ablation should say so.
    """
    from .metrics import collect_metric

    results: List[ComparisonResult] = []
    for previous, current in zip(ordered_keys[:-1], ordered_keys[1:]):
        if previous not in evaluations or current not in evaluations:
            continue
        results.append(wilcoxon_compare(
            collect_metric(evaluations[current].cases, metric, class_name),
            collect_metric(evaluations[previous].cases, metric, class_name),
            name_a=current, name_b=previous, metric=metric, class_name=class_name,
            n_resamples=cfg.bootstrap_samples, alpha=cfg.alpha_level, seed=cfg.seed,
        ))
    return holm_bonferroni(results, cfg.alpha_level)


def format_p(p: float) -> str:
    if np.isnan(p):
        return "--"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def significance_marker(result: ComparisonResult) -> str:
    if np.isnan(result.p_adjusted):
        return ""
    if result.p_adjusted < 0.001:
        return "***"
    if result.p_adjusted < 0.01:
        return "**"
    if result.p_adjusted < 0.05:
        return "*"
    return ""
