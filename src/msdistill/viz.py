"""Publication figures.

Colour choices are not stylistic. The categorical palette is the Okabe-Ito set
restricted to five hues whose pairwise separation stays above the deuteranopia
and tritanopia thresholds (worst adjacent pair dE 11.0 under deuteranopia), so the
figures survive colour-vision deficiency and greyscale printing. Identity is never
carried by colour alone: every series is also directly labelled or named on an
axis.

Two rules that shape the layouts:

* No dual y-axes anywhere. Accuracy and compute are plotted against each other as
  a scatter rather than overlaid on a shared x with two scales.
* Where eight models would need eight hues, the chart is redrawn as a magnitude
  comparison -- one neutral hue for reference methods, one accent for the
  proposed one -- because eight categorical hues cannot be told apart reliably.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# --- validated palette ------------------------------------------------------
SERIES = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9"]
ACCENT = "#0072B2"      # the proposed method
REFERENCE = "#9AA3AE"   # baselines and prior work
TEACHER = "#3F4650"     # the upper reference line
INK = "#1F2328"
MUTED = "#6B7280"
GRID = "#E3E6EA"

# Overlay colours for the three annotated foreground classes.
CLASS_COLORS = {
    0: (0.0, 0.0, 0.0, 0.0),      # background: fully transparent
    1: (0.337, 0.706, 0.914, 1),  # ventricles      #56B4E9
    2: (0.000, 0.620, 0.451, 1),  # normal WMH      #009E73
    3: (0.835, 0.369, 0.000, 1),  # abnormal WMH    #D55E00
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 320,
        "savefig.bbox": "tight",
        "font.size": 8.5,
        "font.family": "sans-serif",
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "axes.edgecolor": "#C7CCD3",
        "axes.linewidth": 0.7,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "lines.linewidth": 1.6,
    })


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def label_to_short(label: str) -> str:
    return (label.replace("Student-", "").replace("Ours-Full: ", "Ours-Full\n")
                 .replace("Ours: ", "Ours\n").replace("KD-", "KD-"))


# --------------------------------------------------------------------------- #
# 1. Training dynamics
# --------------------------------------------------------------------------- #

def plot_training_curves(histories: Dict[str, List[Dict]], teacher_dice: Optional[float],
                         path: str, max_series: int = 5,
                         title: str = "Validation Dice during student training") -> str:
    """Fold-averaged validation Dice per epoch.

    Restricted to `max_series` variants because beyond five lines the reader
    cannot match line to legend; the full grid lives in the tables.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.5))

    keys = list(histories.keys())[:max_series]
    for i, key in enumerate(keys):
        folds = histories[key]
        curves = [f["history"]["val_dice"] for f in folds if f.get("history", {}).get("val_dice")]
        if not curves:
            continue
        n = min(len(c) for c in curves)
        mean = np.mean([c[:n] for c in curves], axis=0)
        colour = SERIES[i % len(SERIES)]
        ax.plot(range(n), mean, color=colour, label=key, zorder=3)
        # Direct end label so identity does not depend on the legend alone.
        ax.annotate(key, xy=(n - 1, mean[-1]), xytext=(3, 0), textcoords="offset points",
                    color=colour, fontsize=6.5, va="center", zorder=4)

    if teacher_dice is not None and not np.isnan(teacher_dice):
        ax.axhline(teacher_dice, color=TEACHER, linestyle=(0, (4, 3)), linewidth=1.1, zorder=2)
        ax.annotate("teacher", xy=(0.01, teacher_dice), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", color=TEACHER, fontsize=6.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean foreground Dice")
    ax.set_title(title, color=INK, loc="left")
    ax.margins(x=0.18)
    ax.legend(loc="lower right", ncol=1)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 2. Ablation magnitudes
# --------------------------------------------------------------------------- #

def plot_ablation_bars(evaluations: Dict[str, "object"], ordered_keys: Sequence[str], cfg,
                       path: str, accent_keys: Sequence[str] = ("kd_full",),
                       teacher_key: str = "teacher") -> str:
    """One panel per clinically relevant class; bars are models.

    This is a magnitude comparison, not eight identities, so it uses a single
    neutral hue with the proposed method accented rather than eight categorical
    colours. Error bars are the standard deviation across test patients.
    """
    apply_style()
    class_indices = list(cfg.foreground_classes)
    fig, axes = plt.subplots(1, len(class_indices), figsize=(2.45 * len(class_indices), 2.9),
                             sharey=True)
    if len(class_indices) == 1:
        axes = [axes]

    plot_keys = [k for k in ordered_keys if k in evaluations and k != teacher_key]
    labels = [label_to_short(evaluations[k].label) for k in plot_keys]

    for ax, class_idx in zip(axes, class_indices):
        class_name = cfg.class_names[class_idx]
        means = np.array([evaluations[k].dice(class_name) for k in plot_keys])
        stds = np.array([evaluations[k].std("dice", class_name) for k in plot_keys])
        colours = [ACCENT if k in accent_keys else REFERENCE for k in plot_keys]

        y = np.arange(len(plot_keys))
        ax.barh(y, means, height=0.62, color=colours,
                xerr=stds, error_kw={"ecolor": "#8A9099", "elinewidth": 0.8, "capsize": 2},
                zorder=3)
        for yi, value in zip(y, means):
            if not np.isnan(value):
                ax.annotate(f"{value:.3f}", xy=(value, yi), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=6.5, color=INK)

        if teacher_key in evaluations:
            t = evaluations[teacher_key].dice(class_name)
            if not np.isnan(t):
                ax.axvline(t, color=TEACHER, linestyle=(0, (4, 3)), linewidth=1.1, zorder=4)

        ax.set_yticks(y)
        ax.set_yticklabels(labels if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_xlim(0, max(1.0, float(np.nanmax(means)) * 1.28))
        ax.set_xlabel("Dice")
        ax.set_title(class_name.replace("_", " ").title(), color=INK, loc="left")
        ax.grid(axis="y", visible=False)

    handles = [Patch(facecolor=ACCENT, label="Proposed"),
               Patch(facecolor=REFERENCE, label="Baseline / prior KD"),
               plt.Line2D([0], [0], color=TEACHER, linestyle=(0, (4, 3)), label="Teacher")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 3. Per-patient paired differences
# --------------------------------------------------------------------------- #

def plot_paired_differences(evaluations: Dict[str, "object"], method_key: str,
                            baseline_key: str, class_name: str, path: str,
                            title: Optional[str] = None) -> str:
    """Sorted per-patient difference between two models on one class.

    A mean improvement of a few Dice points is unconvincing on its own: it can be
    produced by one outlier patient. Showing every patient's signed difference
    makes consistency visible, which is the property that matters clinically.
    """
    from .metrics import collect_metric

    apply_style()
    method = collect_metric(evaluations[method_key].cases, "dice", class_name)
    baseline = collect_metric(evaluations[baseline_key].cases, "dice", class_name)
    ids = [c.patient_id for c in evaluations[method_key].cases]

    valid = ~(np.isnan(method) | np.isnan(baseline))
    diff = (method - baseline)[valid]
    ids = [p for p, keep in zip(ids, valid) if keep]
    order = np.argsort(diff)
    diff, ids = diff[order], [ids[i] for i in order]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    colours = [ACCENT if d > 0 else "#B4441F" for d in diff]
    y = np.arange(len(diff))
    ax.hlines(y, 0, diff, color=colours, linewidth=1.6, zorder=3)
    ax.scatter(diff, y, s=18, color=colours, zorder=4, edgecolor="white", linewidth=0.6)
    ax.axvline(0, color="#8A9099", linewidth=0.9, zorder=2)

    improved = int((diff > 0).sum())
    ax.set_yticks(y)
    ax.set_yticklabels(ids, fontsize=5.8)
    ax.set_xlabel(f"$\\Delta$ Dice  ({method_key} $-$ {baseline_key})")
    ax.set_ylabel("Test patient")
    ax.set_title(title or f"{class_name.replace('_', ' ').title()}: "
                          f"improved in {improved}/{len(diff)} patients",
                 color=INK, loc="left")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 4. Accuracy versus compute
# --------------------------------------------------------------------------- #

def plot_accuracy_vs_cost(evaluations: Dict[str, "object"],
                          efficiency: Dict[str, "object"], cfg, path: str,
                          class_name: Optional[str] = None) -> str:
    """Segmentation quality against measured compute.

    Deliberately a scatter rather than two lines on a shared x with separate y
    scales: the trade-off is a two-dimensional fact and a dual-axis chart would
    let the visual slope be set by arbitrary axis limits.
    """
    apply_style()
    class_name = class_name or cfg.class_names[cfg.primary_class]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    for key, ev in evaluations.items():
        eff = efficiency.get(key)
        if eff is None or np.isnan(eff.gmacs):
            continue
        dice = ev.dice(class_name)
        if np.isnan(dice):
            continue
        is_teacher = key == "teacher"
        colour = TEACHER if is_teacher else (ACCENT if key == "kd_full" else REFERENCE)
        ax.scatter(eff.gmacs, dice, s=54 if is_teacher else 44, color=colour,
                   edgecolor="white", linewidth=0.8, zorder=4)
        ax.annotate(label_to_short(ev.label).replace("\n", " "),
                    xy=(eff.gmacs, dice), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=6.2, color=INK, zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Compute per slice (GMACs, log scale)")
    ax.set_ylabel(f"Dice, {class_name.replace('_', ' ')}")
    ax.set_title("Segmentation quality against inference cost", color=INK, loc="left")
    ax.margins(x=0.28, y=0.22)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 5. Qualitative comparison
# --------------------------------------------------------------------------- #

def overlay_mask(gray: np.ndarray, labels: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend a label map over a normalized grayscale slice."""
    lo, hi = np.percentile(gray, [1, 99])
    base = np.clip((gray - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.repeat(base[:, :, None], 3, axis=2)
    for class_idx, colour in CLASS_COLORS.items():
        if colour[3] == 0:
            continue
        m = labels == class_idx
        if not m.any():
            continue
        rgb[m] = (1 - alpha) * rgb[m] + alpha * np.array(colour[:3])
    return np.clip(rgb, 0, 1)


def plot_qualitative(cache, predictions: Dict[str, np.ndarray], patient_ids: Sequence[str],
                     slice_indices: Sequence[int], model_order: Sequence[Tuple[str, str]],
                     cfg, path: str, zoom_box: Optional[Tuple[int, int, int, int]] = None) -> str:
    """Grid of FLAIR input, reference annotation and each model's prediction.

    `predictions` is keyed "<model_name>::<patient_id>" and holds (H, W, S)
    label volumes, matching what `evaluate_ensemble` stores.
    """
    apply_style()
    flair_idx = cfg.modalities.index("FLAIR") if "FLAIR" in cfg.modalities else 0
    n_rows = len(patient_ids)
    n_cols = 2 + len(model_order)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.42 * n_cols, 1.5 * n_rows))
    axes = np.atleast_2d(axes)

    for r, (pid, s) in enumerate(zip(patient_ids, slice_indices)):
        gray = cache.images[pid][flair_idx, :, :, s]
        ref = cache.masks[pid][:, :, s]

        panels = [("FLAIR", None), ("Reference", ref)]
        for key, label in model_order:
            volume = predictions.get(f"{key}::{pid}")
            panels.append((label, volume[:, :, s] if volume is not None else None))

        for c, (title, labels) in enumerate(panels):
            ax = axes[r, c]
            if labels is None and title == "FLAIR":
                lo, hi = np.percentile(gray, [1, 99])
                ax.imshow(np.clip((gray - lo) / max(hi - lo, 1e-6), 0, 1), cmap="gray")
            elif labels is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes,
                        color=MUTED, fontsize=7)
            else:
                ax.imshow(overlay_mask(gray, labels))
            if zoom_box:
                x0, y0, x1, y1 = zoom_box
                ax.set_xlim(x0, x1)
                ax.set_ylim(y1, y0)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=7.5, color=INK)
            if c == 0:
                ax.set_ylabel(pid, fontsize=7, color=MUTED, rotation=0,
                              ha="right", va="center", labelpad=10)

    handles = [Patch(facecolor=CLASS_COLORS[i][:3], label=cfg.class_names[i].replace("_", " "))
               for i in cfg.foreground_classes]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# 6. Class balance
# --------------------------------------------------------------------------- #

def plot_class_balance(class_balance: Dict[str, float], path: str) -> str:
    """Log-scaled voxel frequency, motivating every imbalance-aware design choice."""
    apply_style()
    names = list(class_balance.keys())
    values = np.array([100 * class_balance[n] for n in names])

    fig, ax = plt.subplots(figsize=(3.2, 1.9))
    colours = [REFERENCE if n == "background" else ACCENT for n in names]
    y = np.arange(len(names))
    ax.barh(y, values, height=0.6, color=colours, zorder=3)
    for yi, v in zip(y, values):
        ax.annotate(f"{v:.2f}%", xy=(v, yi), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=6.8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("_", " ") for n in names])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Share of annotated voxels (%, log scale)")
    ax.set_title("Class imbalance in the cohort", color=INK, loc="left")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.35)
    return _save(fig, path)
