#!/usr/bin/env python
"""Rebuild tables, macros and significance tests from a completed run.

Reads `results/test_evaluations.json` and re-emits everything downstream of it
without repeating inference. Changing a caption, dropping a column or fixing a
table that overruns \\textwidth should not cost a GPU session, and on a
session-limited accelerator it otherwise would.

Efficiency figures are re-measured on whatever machine this runs on unless
`--efficiency-from` points at a previous run's tables, in which case the
measured numbers are carried over verbatim -- latency measured on a laptop must
never silently replace latency measured on the reported hardware.

    python scripts/regenerate_tables.py --run msdistill_out
    python scripts/regenerate_tables.py --run msdistill_out --out ../tables
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from msdistill.config import LADDER_KEYS, Config  # noqa: E402
from msdistill.evaluate import load_evaluations  # noqa: E402
from msdistill.report import (MS3SEG_PUBLISHED, ablation_table, build_macros,  # noqa: E402
                              dataset_table, detection_table, efficiency_table,
                              significance_table, standard_extra_macros, write_macros,
                              write_markdown_summary, write_table)
from msdistill.stats import compare_all, ladder_comparisons  # noqa: E402


class CarriedEfficiency:
    """Efficiency numbers parsed out of a previously generated table.

    Re-measuring on a different machine would silently replace the hardware the
    paper claims, so the previous run's measurements are reused verbatim.
    """

    def __init__(self, name, params_total, gmacs, gpu_ms, cpu_ms):
        self.name = name
        self.params_total = params_total
        self.params_trainable = params_total
        self.gmacs = gmacs
        self.gflops = gmacs * 2
        self.flops_method = "torch.utils.flop_counter"
        self.gpu_latency_ms = {1: gpu_ms}
        self.gpu_latency_iqr_ms = {}
        self.gpu_throughput_ips = {}
        self.cpu_latency_ms = {1: cpu_ms}
        self.peak_gpu_memory_mb = float("nan")
        self.checkpoint_mb = float("nan")
        self.volume_latency_ms = float("nan")


def _parse_params(text: str) -> int:
    text = text.strip()
    if text.endswith("M"):
        return int(float(text[:-1]) * 1e6)
    if text.endswith("K"):
        return int(float(text[:-1]) * 1e3)
    return int(float(re.sub(r"[^0-9.]", "", text) or 0))


def parse_efficiency(macros_path: str) -> List[CarriedEfficiency]:
    """Recover measured efficiency from a previous run's `results_macros.tex`.

    Read from the macro definitions rather than by parsing the rendered table.
    A formatted table is a presentation artefact -- its column order changes when
    the layout changes, and a parser that silently grabs the wrong column puts a
    wrong number in the paper with no error. The macros are structured data with
    stable names.
    """
    if not os.path.exists(macros_path):
        return []
    text = open(macros_path, encoding="utf-8").read()

    def macro(name: str):
        m = re.search(r"\\newcommand\{\\" + name + r"\}\{([^}]*)\}", text)
        return m.group(1) if m else None

    rows = []
    for display, key in (("Teacher", "Teacher"),
                         ("U-Net (base 32)", "UNetBaseThreeTwo"),
                         ("Student", "Student")):
        params = macro(f"Res{key}Paramcount")
        gmacs = macro(f"Res{key}Gmacs")
        gpu = macro(f"Res{key}Gpums")
        cpu = macro(f"Res{key}Cpums")
        if None in (params, gmacs, gpu, cpu):
            continue
        rows.append(CarriedEfficiency(display, _parse_params(params),
                                      float(gmacs), float(gpu), float(cpu)))
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="msdistill_out", help="directory holding results/")
    p.add_argument("--out", default=None, help="where to write tables (default: <run>/tables)")
    p.add_argument("--efficiency-from", required=True,
                   help="a results_macros.tex from a run measured on the REPORTED hardware; "
                        "its efficiency numbers are carried over verbatim rather than "
                        "re-measured on this machine")
    args = p.parse_args()

    run = os.path.abspath(args.run)
    results_dir = os.path.join(run, "results")
    out_dir = os.path.abspath(args.out) if args.out else os.path.join(run, "tables")
    os.makedirs(out_dir, exist_ok=True)

    payload = json.load(open(os.path.join(results_dir, "config.json"), encoding="utf-8"))
    saved = payload["config"]
    cfg_fields = {k: v for k, v in saved.items() if k in Config.__dataclass_fields__}
    for k, v in list(cfg_fields.items()):
        ann = Config.__dataclass_fields__[k].type
        if isinstance(v, list) and "Tuple" in str(ann):
            cfg_fields[k] = tuple(v)
    cfg = Config(**cfg_fields)
    cfg.output_dir = run

    evaluations = load_evaluations(os.path.join(results_dir, "test_evaluations.json"))
    order = ["teacher", "unet32", "scratch", "kd_vanilla", "kd_fitnets", "kd_cwd",
             "kd_region", "kd_region_cwd", "kd_full"]
    ordered = [k for k in order if k in evaluations]
    primary = cfg.class_names[cfg.primary_class]
    n_test = len(evaluations[ordered[0]].cases)
    print(f"loaded {len(evaluations)} models, {n_test} test patients, fingerprint "
          f"{payload['fingerprint']}")

    comparisons = {cfg.class_names[c]: compare_all(evaluations, "scratch", "dice",
                                                   cfg.class_names[c], cfg, exclude=["teacher"])
                   for c in cfg.foreground_classes}
    ladder = ladder_comparisons(evaluations, [k for k in LADDER_KEYS if k in evaluations],
                                "dice", primary, cfg)

    # De-duplicate: `kd_vanilla vs scratch` is in both families, Holm-corrected
    # differently in each. Printing both in one table invites the reader to
    # compare two adjusted p-values for the same hypothesis.
    seen = {(c.name_a, c.name_b) for c in comparisons[primary]}
    ladder_unique = [c for c in ladder if (c.name_a, c.name_b) not in seen]

    eff = parse_efficiency(args.efficiency_from)
    if eff:
        print(f"carried {len(eff)} efficiency rows from {os.path.basename(args.efficiency_from)} "
              f"(measured on the reported hardware, not re-measured here):")
        for r in eff:
            print(f"    {r.name:<18} {r.params_total:>10,}p  {r.gmacs:>6.2f} GMACs  "
                  f"GPU {r.gpu_latency_ms[1]:>6.2f} ms  CPU {r.cpu_latency_ms[1]:>7.1f} ms")
    else:
        raise SystemExit(f"no efficiency macros found in {args.efficiency_from}; refusing to "
                         f"emit an efficiency table rather than fabricate one")

    balance = {c: payload.get("class_balance", {}).get(c, float("nan")) for c in cfg.class_names}
    if all(v != v for v in balance.values()):          # not stored; recover from macros
        macro_path = os.path.join(run, "tables", "results_macros.tex")
        if os.path.exists(macro_path):
            text = open(macro_path, encoding="utf-8").read()
            for cls, key in (("background", "ResBackgroundShare"),
                             ("ventricles", "ResVentricleShare"),
                             ("normal_wmh", "ResNormalShare"),
                             ("abnormal_wmh", "ResAbnormalShare")):
                m = re.search(r"\\newcommand\{\\" + key + r"\}\{([0-9.]+)\}", text)
                if m:
                    balance[cls] = float(m.group(1)) / 100.0

    slices = int(round(float(re.search(
        r"\\newcommand\{\\ResSlicesPerPatient\}\{(\d+)\}",
        open(os.path.join(run, "tables", "results_macros.tex"), encoding="utf-8").read()
    ).group(1))))

    tables = {
        "table_ablation.tex": ablation_table(evaluations, ordered, cfg, comparisons,
                                             published=MS3SEG_PUBLISHED),
        "table_detection.tex": detection_table(evaluations, ordered, cfg),
        "table_significance.tex": significance_table(comparisons[primary] + ladder_unique, cfg),
        "table_dataset.tex": dataset_table(balance, cfg.n_folds * 0 + 100, n_test,
                                           cfg.n_folds, slices, cfg),
    }
    if eff:
        tables["table_efficiency.tex"] = efficiency_table(
            eff, eff[0].name, cfg, ensemble_size=cfg.n_folds, slices_per_volume=slices)

    for name, content in tables.items():
        write_table(content, os.path.join(out_dir, name))
        print(f"  wrote {name}")

    macros = build_macros(
        evaluations, eff, comparisons, cfg,
        extra=standard_extra_macros(cfg, 100, 100 - n_test, n_test, slices, balance, eff),
        ladder=ladder)
    write_macros(macros, os.path.join(out_dir, "results_macros.tex"))
    write_markdown_summary(evaluations, ordered, cfg,
                           os.path.join(results_dir, "RESULTS.md"))
    print(f"  wrote results_macros.tex ({len(macros)} macros)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
