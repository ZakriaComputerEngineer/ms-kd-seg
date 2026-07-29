"""Turns evaluation objects into the exact LaTeX the paper compiles.

The paper never contains a hand-typed number. Every quantity in the prose is a
macro defined in `results_macros.tex`, which this module regenerates from the
evaluation results, and every table is a `.tex` file the paper `\\input`s. Rerun
the notebook and the manuscript updates itself; there is no step at which a stale
figure can survive a change to the experiments.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import numpy as np

from .stats import ComparisonResult, format_p, significance_marker


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #

def fmt(value: float, decimals: int = 3, dash: str = "--") -> str:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return dash
    return f"{value:.{decimals}f}"


def fmt_pm(mean: float, std: float, decimals: int = 3) -> str:
    if mean is None or np.isnan(mean):
        return "--"
    if std is None or np.isnan(std):
        return fmt(mean, decimals)
    return f"{mean:.{decimals}f}\\,$\\pm$\\,{std:.{decimals}f}"


def fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def latex_escape(text: str) -> str:
    return (text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
                .replace("#", r"\#"))


# The dataset's raw class keys title-case into "Normal Wmh", which is wrong in
# two ways: WMH is an initialism, and "normal/abnormal" is the dataset's wording
# for what the clinical literature calls incidental and pathological.
CLASS_DISPLAY = {
    "background": "Background",
    "ventricles": "Ventricles",
    "normal_wmh": "Incidental WMH",
    "abnormal_wmh": "Pathological WMH",
}


def class_label(name: str) -> str:
    return CLASS_DISPLAY.get(name, name.replace("_", " ").title())


def _macro_name(*parts: str) -> str:
    """LaTeX command names may only contain letters, so digits are spelled out."""
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
              "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    raw = "".join(p.replace("_", " ").title().replace(" ", "") for p in parts)
    raw = "".join(digits.get(ch, ch) for ch in raw)
    return "Res" + re.sub(r"[^A-Za-z]", "", raw)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

def ablation_table(evaluations: Dict[str, "object"], ordered_keys: Sequence[str], cfg,
                   comparisons: Optional[Dict[str, List[ComparisonResult]]] = None,
                   published: Optional[Dict[str, Dict]] = None,
                   caption: str = "", label: str = "tab:ablation",
                   baseline_key: str = "scratch") -> str:
    """Main results table.

    Two conventions matter here. Published numbers obtained under a different
    protocol go in their own block behind a rule, with the protocol difference
    stated in the caption -- a reader who compares across that rule has been
    warned. And the deployable single model's accuracy is reported next to the
    fold ensemble's, so the accuracy and efficiency tables describe the same
    system.
    """
    report_classes = [cfg.class_names[i] for i in cfg.foreground_classes]
    primary = cfg.class_names[cfg.primary_class]
    # Direction matters. `significance_marker` reports only that a difference is
    # unlikely under the null; marking a significantly WORSE result with the same
    # symbol the caption defines as "better" would invert the reading. Channel-wise
    # distillation is significantly worse than the baseline on one class, so this
    # is not hypothetical.
    marker_by_key: Dict[str, Dict[str, str]] = {}
    if comparisons:
        for class_name, results in comparisons.items():
            for r in results:
                if not significance_marker(r):
                    continue
                mark = r"$^{\dagger}$" if r.mean_difference > 0 else r"$^{\ddagger}$"
                marker_by_key.setdefault(r.name_a, {})[class_name] = mark

    # Six columns, not eight. The "Objective" column was the widest thing in the
    # table and duplicated information already in the model name; the
    # single-model column moved to a footnote row. An eight-column table* at
    # 10pt overruns \textwidth, and an overfull float is what makes LaTeX scatter
    # tables to the end of the document.
    default_caption = (
        "Held-out test accuracy, Dice per patient over the complete volume, averaged across "
        "patients ($\\pm$ SD). Rows above the rule come from prior work under a different "
        "protocol and are not directly comparable; all rows below share one protocol. "
        "$\\dagger$ / $\\ddagger$: significantly better / worse than the compact baseline "
        "(paired Wilcoxon, Holm-corrected within class, $\\alpha=0.05$); no comparison on "
        "pathological WMH reaches significance. Single deployable fold models score "
        f"{fmt(evaluations['kd_full'].single_model_dice.get(primary, float('nan')))} "
        "(proposed) against "
        f"{fmt(evaluations[baseline_key].single_model_dice.get(primary, float('nan')))} "
        "(baseline) on " + class_label(primary) + "; "
        "Table~\\ref{tab:efficiency} times that configuration."
        if "kd_full" in evaluations and baseline_key in evaluations else
        "Held-out test accuracy, Dice per patient over the complete volume ($\\pm$ SD).")

    lines = [
        r"\begin{table*}[!t]", r"\centering",
        r"\caption{" + (caption or default_caption) + "}",
        r"\label{" + label + "}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lr" + "c" * len(report_classes) + "c}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Params} & "
        + " & ".join(r"\textbf{" + latex_escape(class_label(c)) + "}" for c in report_classes)
        + r" & \textbf{Mean FG} \\",
        r"\midrule",
    ]

    if published:
        for name, values in published.items():
            cells = [fmt(values.get(c, float("nan")), 3) for c in report_classes]
            lines.append(f"{latex_escape(name)} & {values.get('params', 'n/r')} & "
                         + " & ".join(cells)
                         + f" & {fmt(values.get('mean_fg', float('nan')))} \\\\")
        lines.append(r"\midrule")

    for key in ordered_keys:
        ev = evaluations.get(key)
        if ev is None:
            continue
        cells = []
        for class_name in report_classes:
            cell = fmt_pm(ev.dice(class_name), ev.std("dice", class_name))
            cell += marker_by_key.get(key, {}).get(class_name, "")
            cells.append(cell)

        # Prior-art rows carry their citation on the model name rather than in a
        # separate column.
        name = latex_escape(ev.label)
        if getattr(ev, "citation", ""):
            name += r"~\cite{" + ev.citation + "}"

        lines.append(f"{name} & {fmt_params(ev.n_params)} & " + " & ".join(cells)
                     + f" & {fmt(ev.mean_foreground_dice())} \\\\")
        if key in ("teacher", "unet32", baseline_key):
            lines.append(r"\addlinespace")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


# Verified against the MS3SEG data descriptor (Sci. Data 13:867, 2026), Table 7
# (four-class task) and Table 9 (binary lesion-only task). The two must not be
# merged into a single range: they are different experiments. Parameter counts
# are not reported in that paper, so the cell is "n/r" rather than a guess.
MS3SEG_PUBLISHED = {
    "U-Net~\\cite{ms3seg} (4-class)": {
        "params": "n/r", "objective": "hard labels",
        "ventricles": 0.890, "normal_wmh": 0.6452, "abnormal_wmh": 0.6686, "mean_fg": 0.735,
    },
}


def detection_table(evaluations: Dict[str, "object"], ordered_keys: Sequence[str], cfg,
                    caption: str = "", label: str = "tab:detection") -> str:
    """Boundary and lesion-wise detection metrics for the pathological class."""
    primary = cfg.class_names[cfg.primary_class]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{" + (caption or
            f"Boundary and lesion-wise detection quality on the {class_label(primary)} "
            "class. Voxel overlap alone can hide a model that outlines the lesions it finds "
            "well while missing many of them; lesion-wise F1 counts distinct lesions "
            "detected. HD95 and NSD are reported in millimetres using each volume's header "
            "spacing.") + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{HD95 (mm)} $\downarrow$ & \textbf{NSD@2mm} $\uparrow$ & "
        r"\textbf{Lesion F1} $\uparrow$ & \textbf{Lesion TPR} $\uparrow$ \\",
        r"\midrule",
    ]
    for key in ordered_keys:
        ev = evaluations.get(key)
        if ev is None:
            continue
        lines.append(
            f"{latex_escape(ev.label)} & "
            f"{fmt(ev.value('hd95', primary), 2)} & {fmt(ev.value('nsd', primary))} & "
            f"{fmt(ev.value('lesion_f1', primary))} & {fmt(ev.value('lesion_tpr', primary))} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def efficiency_table(reports: Sequence["object"], reference_name: str, cfg,
                     caption: str = "", label: str = "tab:efficiency",
                     ensemble_size: int = 1, environment: Optional[Dict[str, str]] = None,
                     slices_per_volume: int = 20) -> str:
    """Deployment cost.

    Three deliberate choices. CPU latency at batch size 1 is included because
    that, not GPU throughput, is the setting the paper's motivation describes.
    Multiply-accumulate cost is reported alongside parameters because the two
    ratios differ substantially -- a full-resolution U-Net does all its work at
    input scale while a hierarchical transformer downsamples at the stem -- and
    quoting only the flattering one would be misleading. And because the accuracy
    tables use a $K$-fold ensemble, the ensemble's cost is given its own rows
    rather than left for the reader to infer.
    """
    reference = next((r for r in reports if r.name == reference_name), None)

    default_caption = (
        f"Measured inference cost at $256\\times256$. Latency is the median of "
        f"{cfg.benchmark_trials} timed passes after {cfg.benchmark_warmup} warmup iterations, "
        f"with device synchronization inside the timed region; CPU timing uses 4 threads. "
        f"\\emph{{Ensemble}} rows give the cost of the {ensemble_size}-member fold ensemble the "
        f"accuracy tables report, so both tables describe the same system. Parameter count alone "
        f"overstates the advantage; the MAC and wall-clock columns are the honest ones.")
    if environment:
        default_caption += (" Hardware: "
                            + latex_escape(environment.get("gpu_name", "CPU only"))
                            + ", PyTorch " + latex_escape(environment.get("torch", "")) + ".")
    if reports and getattr(reports[0], "flops_method", None):
        default_caption += (" MACs via "
                            + latex_escape(str(reports[0].flops_method)) + ".")

    # Seven columns rather than nine: batch-8 GPU latency and peak memory were the
    # least load-bearing and pushed the float past \textwidth.
    lines = [
        r"\begin{table}[!t]", r"\centering",
        r"\caption{" + (caption or default_caption) + "}",
        r"\label{" + label + "}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Params} & \textbf{GMACs} & \textbf{GPU} & "
        r"\textbf{CPU} & \textbf{Study} \\",
        r" & & & (ms) & (ms) & (s) \\",
        r"\midrule",
    ]

    def row(r, multiplier: int, suffix: str = "") -> str:
        cpu = r.cpu_latency_ms.get(1, float("nan")) * multiplier
        study = cpu * slices_per_volume / 1000.0
        return (f"{latex_escape(r.name)}{suffix} & "
                f"{fmt_params(r.params_total * multiplier)} & "
                f"{fmt(r.gmacs * multiplier, 2)} & "
                f"{fmt(r.gpu_latency_ms.get(1, float('nan')) * multiplier, 2)} & "
                f"{fmt(cpu, 1)} & {fmt(study, 2)} \\\\")

    for r in reports:
        lines.append(row(r, 1))
    if ensemble_size > 1:
        lines.append(r"\addlinespace")
        for r in reports:
            lines.append(row(r, ensemble_size, f" ($\\times${ensemble_size})"))

    if reference is not None and reference.cpu_latency_ms.get(1):
        student = reports[-1]
        if student.cpu_latency_ms.get(1):
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{6}{l}{\emph{Student vs.\ teacher: }"
                         f"{reference.params_total / student.params_total:.0f}$\\times$ params, "
                         f"{reference.gmacs / student.gmacs:.0f}$\\times$ MACs, "
                         f"{reference.cpu_latency_ms[1] / student.cpu_latency_ms[1]:.1f}"
                         r"$\times$ CPU latency} \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def significance_table(comparisons: Sequence[ComparisonResult], cfg,
                       caption: str = "", label: str = "tab:significance") -> str:
    """Paired comparisons with their effective sample sizes.

    `n` is the number of patients where both models produced a defined value;
    `n'` is how many of those differed, which is the size the signed-rank test
    actually runs at once ties are deleted. On the rare classes the two can
    diverge sharply, and a reader cannot judge a $p$-value without the second.
    """
    n_patients = max((c.n_pairs for c in comparisons), default=0)
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{" + (caption or
            f"Paired comparisons across the held-out test cohort ($n \\leq {n_patients}$ "
            "patients per class, depending on where the class is present in the reference). "
            "$\\Delta$ is the mean paired difference in Dice with a percentile bootstrap 95\\% "
            "confidence interval; $n'$ is the number of non-tied pairs the Wilcoxon signed-rank "
            "test runs at; $p$ is Holm-corrected within class; $r$ is the matched-pairs "
            "rank-biserial effect size.") + "}",
        r"\label{" + label + "}",
        r"\footnotesize",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Comparison} & \textbf{Class} & $\Delta$\textbf{Dice} & \textbf{95\% CI} & "
        r"$n'$ & $p_{\text{Holm}}$ & $r$ \\",
        r"\midrule",
    ]
    for c in comparisons:
        lines.append(
            f"{latex_escape(c.name_a)} vs.\\ {latex_escape(c.name_b)} & "
            f"{latex_escape(class_label(c.class_name))} & "
            f"{fmt(c.mean_difference)} & "
            f"[{fmt(c.ci_low)}, {fmt(c.ci_high)}] & {c.n_effective} & "
            f"{format_p(c.p_adjusted)}{significance_marker(c)} & {fmt(c.effect_size, 2)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def dataset_table(class_balance: Dict[str, float], n_patients: int, n_test: int,
                  n_folds: int, slices_per_patient: int, cfg,
                  label: str = "tab:dataset") -> str:
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Cohort composition and class balance. The two clinically relevant "
        r"classes together occupy well under two percent of voxels, which is the "
        r"imbalance every component of the objective is designed around.}",
        r"\label{" + label + "}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"\textbf{Property} & \textbf{Value} \\",
        r"\midrule",
        f"Patients & {n_patients} \\\\",
        f"Held-out test patients & {n_test} \\\\",
        f"Development folds & {n_folds} \\\\",
        f"Axial slices per patient & {slices_per_patient} \\\\",
        f"Input modalities & {', '.join(cfg.modalities)} \\\\",
        r"\midrule",
    ]
    for name, fraction in class_balance.items():
        lines.append(f"Voxel fraction, {latex_escape(name.replace('_', ' '))} & "
                     f"{100 * fraction:.2f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def seed_variance_table(per_seed: Dict[str, Dict[int, float]], cfg,
                        label: str = "tab:seeds") -> str:
    """Spread of the headline metric across independent seeds.

    Included because a difference smaller than the seed-to-seed spread is not a
    finding, and a reader cannot tell which is which without this table.
    """
    seeds = sorted({s for values in per_seed.values() for s in values})
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Mean foreground Dice across independent random seeds. Differences "
        r"between objectives are reported only where they exceed this spread.}",
        r"\label{" + label + "}",
        r"\begin{tabular}{l" + "c" * len(seeds) + "cc}",
        r"\toprule",
        r"\textbf{Variant} & " + " & ".join(f"seed {s}" for s in seeds)
        + r" & \textbf{Mean} & \textbf{SD} \\",
        r"\midrule",
    ]
    for key, values in per_seed.items():
        row = [fmt(values.get(s, float("nan"))) for s in seeds]
        present = [v for v in values.values() if not np.isnan(v)]
        mean = float(np.mean(present)) if present else float("nan")
        std = float(np.std(present, ddof=1)) if len(present) > 1 else float("nan")
        lines.append(f"{latex_escape(key)} & " + " & ".join(row)
                     + f" & {fmt(mean)} & {fmt(std)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Prose macros
# --------------------------------------------------------------------------- #

def standard_extra_macros(cfg, n_patients: int, n_dev: int, n_test: int,
                          slices_per_volume: int, class_balance: Dict[str, float],
                          efficiency: Sequence["object"]) -> Dict[str, str]:
    """Cohort and protocol macros the manuscript's prose cites.

    Defined here rather than inline in the notebook so that the smoke test can
    assert the paper's macro set is fully covered without duplicating the list.
    """
    by_name = {r.name: r for r in efficiency}
    teacher = by_name.get("Teacher")
    student = by_name.get("Student")

    macros = {
        "ResNumPatients": str(n_patients),
        "ResNumDevPatients": str(n_dev),
        "ResNumTestPatients": str(n_test),
        "ResNumFolds": str(cfg.n_folds),
        "ResSlicesPerPatient": str(slices_per_volume),
        "ResTeacherVariant": f"MiT-{cfg.teacher_variant.upper()}",
        "ResBackgroundShare": f"{100 * class_balance.get('background', float('nan')):.2f}",
        "ResNormalShare": f"{100 * class_balance.get('normal_wmh', float('nan')):.2f}",
        "ResAbnormalShare": f"{100 * class_balance.get('abnormal_wmh', float('nan')):.2f}",
        "ResVentricleShare": f"{100 * class_balance.get('ventricles', float('nan')):.2f}",
    }

    if student is not None and student.cpu_latency_ms.get(1):
        per_study = student.cpu_latency_ms[1] * slices_per_volume / 1000.0
        # One deployable model. Pair this with the single-model Dice macros.
        macros["ResStudySeconds"] = f"{per_study:.1f}"
        # The K-fold ensemble the accuracy tables report. Pair this with the
        # ensemble Dice macros; mixing the two overstates the system by K.
        macros["ResStudySecondsEnsemble"] = f"{per_study * cfg.n_folds:.1f}"
    if teacher is not None and student is not None:
        if student.gmacs and not np.isnan(student.gmacs) and not np.isnan(teacher.gmacs):
            macros["ResMacRatioExact"] = f"{teacher.gmacs / student.gmacs:.1f}"
    return macros


def build_macros(evaluations: Dict[str, "object"], efficiency: Sequence["object"],
                 comparisons: Dict[str, List[ComparisonResult]], cfg,
                 extra: Optional[Dict[str, str]] = None,
                 ladder: Optional[Sequence[ComparisonResult]] = None) -> Dict[str, str]:
    """Every number the manuscript's prose cites, as a LaTeX macro.

    `ladder` results get a distinct `Ladder` prefix. They compare consecutive
    rungs rather than everything against the baseline, and are Holm-corrected
    within their own family, so a comparison appearing in both would otherwise
    collide on one macro name with two different adjusted p-values.
    """
    macros: Dict[str, str] = {}

    for key, ev in evaluations.items():
        for class_name in cfg.class_names:
            value = ev.dice(class_name)
            if not np.isnan(value):
                macros[_macro_name(key, class_name, "dice")] = fmt(value)
                macros[_macro_name(key, class_name, "dicesd")] = fmt(ev.std("dice", class_name))
            # The deployable single model's accuracy, so the prose can pair it
            # with the single model's latency. Quoting K-fold ensemble Dice
            # beside one model's inference time overstates the system by K.
            single = getattr(ev, "single_model_dice", {}).get(class_name, float("nan"))
            if not np.isnan(single):
                macros[_macro_name(key, class_name, "singledice")] = fmt(single)
                macros[_macro_name(key, class_name, "singledicesd")] = fmt(
                    getattr(ev, "single_model_dice_std", {}).get(class_name, float("nan")))
        macros[_macro_name(key, "meanfg")] = fmt(ev.mean_foreground_dice())
        macros[_macro_name(key, "params")] = fmt_params(ev.n_params)
        primary = cfg.class_names[cfg.primary_class]
        for metric in ("lesion_f1", "hd95", "nsd"):
            value = ev.value(metric, primary)
            if not np.isnan(value):
                macros[_macro_name(key, metric)] = fmt(value, 2 if metric == "hd95" else 3)

    for report in efficiency:
        macros[_macro_name(report.name, "gmacs")] = fmt(report.gmacs, 2)
        macros[_macro_name(report.name, "paramcount")] = fmt_params(report.params_total)
        if report.gpu_latency_ms.get(1):
            macros[_macro_name(report.name, "gpums")] = fmt(report.gpu_latency_ms[1], 2)
        if report.cpu_latency_ms.get(1):
            macros[_macro_name(report.name, "cpums")] = fmt(report.cpu_latency_ms[1], 1)

    # Ratios the abstract quotes directly.
    by_name = {r.name: r for r in efficiency}
    if "Teacher" in by_name and "Student" in by_name:
        t, s = by_name["Teacher"], by_name["Student"]
        if s.params_total:
            macros["ResParamRatio"] = f"{t.params_total / s.params_total:.0f}"
        if s.gmacs and not np.isnan(s.gmacs) and not np.isnan(t.gmacs):
            macros["ResMacRatio"] = f"{t.gmacs / s.gmacs:.0f}"
        if s.gpu_latency_ms.get(1) and t.gpu_latency_ms.get(1):
            macros["ResGpuSpeedup"] = f"{t.gpu_latency_ms[1] / s.gpu_latency_ms[1]:.1f}"
        if s.cpu_latency_ms.get(1) and t.cpu_latency_ms.get(1):
            macros["ResCpuSpeedup"] = f"{t.cpu_latency_ms[1] / s.cpu_latency_ms[1]:.1f}"

    for class_name, results in comparisons.items():
        for r in results:
            base = _macro_name(r.name_a, "vs", r.name_b, class_name)
            macros[base + "Delta"] = fmt(r.mean_difference)
            macros[base + "P"] = format_p(r.p_adjusted)
            macros[base + "CI"] = f"[{fmt(r.ci_low)}, {fmt(r.ci_high)}]"

    for r in (ladder or ()):
        base = "ResLadder" + _macro_name(r.name_a, "vs", r.name_b)[3:]
        macros[base + "Delta"] = fmt(r.mean_difference)
        macros[base + "P"] = format_p(r.p_adjusted)
        macros[base + "CI"] = f"[{fmt(r.ci_low)}, {fmt(r.ci_high)}]"

    if extra:
        macros.update({k: str(v) for k, v in extra.items()})
    return macros


def write_macros(macros: Dict[str, str], path: str) -> str:
    lines = [
        "% Auto-generated by msdistill.report -- do not edit by hand.",
        "% Regenerate by rerunning the evaluation notebook.",
        "",
    ]
    for name in sorted(macros):
        lines.append(f"\\newcommand{{\\{name}}}{{{macros[name]}}}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def write_table(content: str, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def write_markdown_summary(evaluations: Dict[str, "object"], ordered_keys: Sequence[str],
                           cfg, path: str) -> str:
    """Plain-text results digest for the repository README and for eyeballing."""
    report_classes = [cfg.class_names[i] for i in cfg.foreground_classes]
    lines = ["# Held-out test results", "",
             "Per-patient Dice over complete volumes, mean Â± SD across test patients.", "",
             "| Model | Params | " + " | ".join(c.replace("_", " ") for c in report_classes)
             + " | Mean FG |",
             "|---|---|" + "---|" * (len(report_classes) + 1)]
    for key in ordered_keys:
        ev = evaluations.get(key)
        if ev is None:
            continue
        cells = [f"{ev.dice(c):.3f} Â± {ev.std('dice', c):.3f}" if not np.isnan(ev.dice(c)) else "--"
                 for c in report_classes]
        lines.append(f"| {ev.label} | {fmt_params(ev.n_params)} | " + " | ".join(cells)
                     + f" | {ev.mean_foreground_dice():.3f} |")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
