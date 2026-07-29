#!/usr/bin/env python
"""Static and semantic validation of the generated notebook.

Catches the failure mode that costs the most: a notebook that runs for forty
minutes of GPU training and then dies on a NameError in a later cell. Every
library cell is compiled and executed in a fresh namespace, and every driver cell
is compiled and checked for names the namespace does not provide.

    python scripts/validate_notebook.py
"""

from __future__ import annotations

import ast
import builtins
import json
import os
import sys
import traceback
import types
from typing import Dict, List, Set

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(REPO_ROOT, "notebooks", "ms_kd_segmentation.ipynb")

# Names supplied by the runtime rather than by a cell.
RUNTIME_PROVIDED = {"display", "get_ipython", "In", "Out", "__file__"}

REQUIRED_SYMBOLS = [
    "Config", "select_variants", "LADDER_KEYS", "SEED_STUDY_KEYS", "QUICK_VARIANT_KEYS",
    "student_builder", "evaluate_with_single_model_stats", "MS3SEG_PUBLISHED",
    "build_patient_index", "infer_label_values",
    "LabelRemapper", "locate_dataset", "describe_tree", "find_dataset_roots",
    "split_test_and_dev", "make_fold_splits", "PatientVolumeCache",
    "summarize_class_balance", "get_device", "set_seed", "run_teacher_stage",
    "run_student_stage", "load_frozen_teacher", "build_teacher", "build_student",
    "count_parameters", "load_ensemble", "evaluate_ensemble", "save_evaluations",
    "teacher_student_gap_recovery", "compare_all", "ladder_comparisons", "format_p",
    "significance_marker", "require_scipy", "profile_model", "environment_summary",
    "ablation_table",
    "detection_table", "efficiency_table", "significance_table", "dataset_table",
    "seed_variance_table", "build_macros", "write_macros", "write_table",
    "write_markdown_summary", "plot_class_balance", "plot_training_curves",
    "plot_ablation_bars", "plot_paired_differences", "plot_accuracy_vs_cost",
    "plot_qualitative",
]


def cell_source(cell: Dict) -> str:
    """Cell text with IPython magics neutralised.

    `!zip ...` and `%matplotlib inline` are valid in a notebook and a syntax
    error to `ast.parse`. They are replaced with a comment rather than dropped so
    line numbers in any reported error still line up with the cell.
    """
    lines = []
    for line in "".join(cell["source"]).splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("!", "%%", "%")) and not stripped.startswith("%%html"):
            lines.append(" " * (len(line) - len(stripped)) + "pass  # magic: " + stripped[:60])
        else:
            lines.append(line)
    return "\n".join(lines)


def collect_bindings(tree: ast.AST) -> Set[str]:
    """Every module-level name a cell defines."""
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.For, ast.While, ast.If, ast.With, ast.Try)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                    names.add(sub.id)
                elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(sub.name)
                elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for alias in sub.names:
                        names.add(alias.asname or alias.name.split(".")[0])
    return names


def names_bound_within(node: ast.AST) -> Set[str]:
    """Names a statement binds anywhere inside itself.

    Includes comprehension targets and lambda parameters, which live in their own
    scope: `[f(x) for x in xs]` reads `x` but does not require `x` to pre-exist,
    and counting it as a free name produces spurious failures.
    """
    bound: Set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.comprehension):
            bound.update(n.id for n in ast.walk(sub.target) if isinstance(n, ast.Name))
        elif isinstance(sub, ast.Lambda):
            a = sub.args
            args = list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)
            if a.vararg:
                args.append(a.vararg)
            if a.kwarg:
                args.append(a.kwarg)
            bound.update(arg.arg for arg in args)
        elif isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
            bound.add(sub.id)
        elif isinstance(sub, ast.ExceptHandler) and sub.name:
            bound.add(sub.name)
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(sub.name)
        elif isinstance(sub, (ast.Import, ast.ImportFrom)):
            bound.update(alias.asname or alias.name.split(".")[0] for alias in sub.names)
    return bound


def collect_free_names(tree: ast.AST) -> Set[str]:
    """Module-level names a cell reads before binding them.

    Deliberately shallow: a name used only inside a function body does not have
    to exist when the cell is executed, only when the function is called.
    """
    used: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        statement_used = {sub.id for sub in ast.walk(node)
                          if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)}
        used |= statement_used - names_bound_within(node)
    return used


def main() -> int:
    if not os.path.exists(NOTEBOOK):
        print(f"notebook not found: {NOTEBOOK}\nrun scripts/build_notebook.py first")
        return 1

    with open(NOTEBOOK, encoding="utf-8") as f:
        notebook = json.load(f)

    code_cells = [(i, cell_source(c)) for i, c in enumerate(notebook["cells"])
                  if c["cell_type"] == "code"]
    roles = {i: c.get("metadata", {}).get("msdistill_role", "driver")
             for i, c in enumerate(notebook["cells"]) if c["cell_type"] == "code"}
    n_library = sum(1 for r in roles.values() if r == "library")
    print(f"{len(notebook['cells'])} cells, {len(code_cells)} code "
          f"({n_library} library, {len(code_cells) - n_library} driver)")

    failures: List[str] = []

    # -- 1. every cell must parse ------------------------------------------
    trees: Dict[int, ast.AST] = {}
    for index, source in code_cells:
        try:
            trees[index] = ast.parse(source, filename=f"cell[{index}]")
        except SyntaxError as exc:
            failures.append(f"cell {index}: syntax error at line {exc.lineno}: {exc.msg}")
    if failures:
        for f in failures:
            print("  FAIL", f)
        return 1
    print("  [ok] all cells parse")

    # -- 2. no leftover intra-package imports -------------------------------
    import_failures = []
    for index, source in code_cells:
        for lineno, line in enumerate(source.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("from .", "from msdistill", "import msdistill")):
                import_failures.append(f"cell {index} line {lineno}: inlining missed "
                                       f"{stripped!r}")
    failures += import_failures
    print("  [ok] no unresolved package imports" if not import_failures
          else f"  FAIL {len(import_failures)} unresolved package imports")

    # -- 3. execute the library cells ----------------------------------------
    # Execute into a real module object registered in sys.modules, not a bare
    # dict. `dataclasses` resolves a class's annotations via
    # `sys.modules[cls.__module__].__dict__`, so a namespace with no backing
    # module raises AttributeError on the first @dataclass -- a failure of the
    # harness, not of the notebook, which runs inside a genuine `__main__`.
    module = types.ModuleType("__notebook__")
    module.__builtins__ = builtins
    sys.modules["__notebook__"] = module
    namespace: Dict[str, object] = module.__dict__
    executed = 0
    exec_failed = False
    for index, source in code_cells:
        if roles[index] != "library":
            continue
        # The environment cell is executed unmodified. `ensure()` is a no-op when
        # a package is already importable, so a warm environment installs nothing;
        # a cold one legitimately installs what the notebook needs.
        try:
            exec(compile(source, f"cell[{index}]", "exec"), namespace)
            executed += 1
        except Exception as exc:
            failures.append(f"cell {index}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            exec_failed = True
            break
    print(f"  [{'FAIL' if exec_failed else 'ok'}] executed {executed} library cells "
          f"({len(namespace)} names bound)")

    # -- 4. driver cells resolve against the library --------------------------
    available = set(namespace) | set(dir(builtins)) | RUNTIME_PROVIDED
    name_failures = []
    for index, _ in code_cells:
        if roles[index] == "library":
            continue
        # Names the cell binds itself (loop variables, locals, comprehension
        # targets) are not required to pre-exist.
        bindings = collect_bindings(trees[index])
        missing = sorted(collect_free_names(trees[index]) - available - bindings)
        if missing:
            name_failures.append(f"cell {index}: undefined at run time: {missing}")
        available |= bindings
    failures += name_failures
    print("  [ok] driver cells resolve against the library" if not name_failures
          else f"  FAIL {len(name_failures)} driver cells reference undefined names")

    # -- 5. required public surface -------------------------------------------
    absent = [name for name in REQUIRED_SYMBOLS if name not in namespace]
    if absent:
        failures.append(f"library does not define: {absent}")
    print(f"  [ok] all {len(REQUIRED_SYMBOLS)} required symbols defined" if not absent
          else f"  FAIL missing symbols: {absent}")

    # -- 6. structural sanity ---------------------------------------------------
    joined = "\n".join(s for _, s in code_cells)
    for needle, description in (("Config(", "constructs a Config"),
                                ("run_teacher_stage", "runs the teacher stage"),
                                ("run_student_stage", "runs the student stage"),
                                ("write_macros", "writes the prose macros")):
        if needle not in joined:
            failures.append(f"no cell {description}")

    print()
    if failures:
        print(f"VALIDATION FAILED ({len(failures)} problem(s))")
        for f in failures:
            print("  -", f)
        return 1
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
