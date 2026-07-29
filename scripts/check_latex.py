#!/usr/bin/env python
"""Static checks on the manuscript, for environments with no LaTeX toolchain.

Not a substitute for compiling. It catches the errors that are cheap to detect
and expensive to discover late: unbalanced environments, citations with no
bibliography entry, bibliography entries nothing cites, undefined custom macros,
and unresolved result placeholders.

    python scripts/check_latex.py [--paper ../main.tex]
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from typing import List, Set, Tuple

# `verbatim`-like environments would need different handling; none are used.
ENV_OPEN = re.compile(r"\\begin\{([A-Za-z*]+)\}")
ENV_CLOSE = re.compile(r"\\end\{([A-Za-z*]+)\}")
CITE = re.compile(r"\\cite\{([^}]*)\}")
BIBITEM = re.compile(r"\\bibitem\{([^}]*)\}")
LABEL = re.compile(r"\\label\{([^}]*)\}")
REF = re.compile(r"\\(?:ref|eqref|autoref)\{([^}]*)\}")
NEWCOMMAND = re.compile(r"\\(?:new|renew|provide)command\*?\{?\\([A-Za-z]+)\}?")
USERMACRO = re.compile(r"\\([A-Z][A-Za-z]*)")


def strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        idx, escaped = None, False
        for i, ch in enumerate(line):
            if ch == "\\":
                escaped = not escaped
            elif ch == "%" and not escaped:
                idx = i
                break
            else:
                escaped = False
        out.append(line if idx is None else line[:idx])
    return "\n".join(out)


# Commands provided by IEEEtran and the loaded packages. Anything capitalised and
# not in here, not defined in the document, and not a result macro is suspicious.
KNOWN_COMMANDS = {
    "IEEEauthorblockN", "IEEEauthorblockA", "IEEEauthorrefmark", "IEEEoverridecommandlockouts",
    "IEEEkeywords", "IEEEtran", "InputIfFileExists", "IfFileExists",
    # amsmath sizing and delimiters
    "Big", "Bigg", "Biggl", "Biggr", "Bigl", "Bigr", "Vert", "Delta", "Sigma", "Omega",
    "Leftarrow", "Rightarrow", "Leftrightarrow", "Longrightarrow",
    # words that happen to be capitalised inside bibliography entries
    "Proc", "Trans", "Sci", "Data", "Systems", "Intelligence", "Conf", "Knowledge",
    "Based", "Syst", "Artificial", "Neural", "Information", "Processing", "IEEE", "CVF",
    "CVPR", "ICCV", "ECCV", "ICLR", "MICCAI", "AAAI", "NeurIPS",
}


def check(path: str) -> Tuple[List[str], List[str]]:
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    text = strip_comments(raw)
    errors: List[str] = []
    warnings: List[str] = []

    # -- environments -------------------------------------------------------
    stack: List[Tuple[str, int]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for name in ENV_OPEN.findall(line):
            stack.append((name, lineno))
        for name in ENV_CLOSE.findall(line):
            if not stack:
                errors.append(f"line {lineno}: \\end{{{name}}} with nothing open")
            elif stack[-1][0] != name:
                errors.append(f"line {lineno}: \\end{{{name}}} closes "
                              f"\\begin{{{stack[-1][0]}}} from line {stack[-1][1]}")
                stack.pop()
            else:
                stack.pop()
    for name, lineno in stack:
        errors.append(f"line {lineno}: \\begin{{{name}}} never closed")

    # -- braces --------------------------------------------------------------
    depth, escaped = 0, False
    for i, ch in enumerate(text):
        if ch == "\\":
            escaped = not escaped
            continue
        if not escaped:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    errors.append(f"unbalanced closing brace near offset {i}")
                    depth = 0
        escaped = False
    if depth != 0:
        errors.append(f"{depth} unclosed brace(s) in the document")

    # -- math delimiters ------------------------------------------------------
    inline = len(re.findall(r"(?<!\\)\$", text))
    if inline % 2:
        errors.append(f"odd number of unescaped $ ({inline}) -- unbalanced inline math")

    # -- citations vs bibliography ---------------------------------------------
    cited: Set[str] = set()
    for group in CITE.findall(text):
        cited.update(k.strip() for k in group.split(",") if k.strip())
    defined = set(BIBITEM.findall(text))

    for key in sorted(cited - defined):
        errors.append(f"\\cite{{{key}}} has no \\bibitem")
    for key in sorted(defined - cited):
        warnings.append(f"\\bibitem{{{key}}} is never cited (wastes page budget)")

    # -- labels and references ---------------------------------------------------
    # Table labels live in the generated tables/*.tex files, which are \input at
    # compile time, so they must be scanned too or every table reference looks
    # dangling.
    labels = set(LABEL.findall(text))
    tables_dir_early = os.path.join(os.path.dirname(os.path.abspath(path)), "tables")
    generated = [n for n in os.listdir(tables_dir_early)
                 if n.startswith("table_") and n.endswith(".tex")] \
        if os.path.isdir(tables_dir_early) else []
    if generated:
        for name in generated:
            with open(os.path.join(tables_dir_early, name), encoding="utf-8") as f:
                labels.update(LABEL.findall(f.read()))
    else:
        warnings.append("no generated tables/table_*.tex yet; table labels are "
                        "unverifiable until the notebook has run")
        labels.update(REF.findall(text))   # do not flag every table as dangling

    for key in sorted(set(REF.findall(text)) - labels):
        errors.append(f"\\ref{{{key}}} has no \\label")
    duplicates = [k for k, n in Counter(LABEL.findall(text)).items() if n > 1]
    for key in duplicates:
        errors.append(f"\\label{{{key}}} defined more than once")

    # -- custom macros -------------------------------------------------------------
    defined_macros = set(NEWCOMMAND.findall(text))
    result_macros = {m for m in USERMACRO.findall(text) if m.startswith("Res")}
    used = set(USERMACRO.findall(text))
    unknown = sorted(used - defined_macros - result_macros - KNOWN_COMMANDS)
    for name in unknown:
        warnings.append(f"\\{name} is capitalised but neither defined here nor a Res* macro "
                        f"-- check it is provided by a loaded package")

    # -- result placeholders ---------------------------------------------------------
    tables_dir = os.path.join(os.path.dirname(os.path.abspath(path)), "tables")
    macros_file = os.path.join(tables_dir, "results_macros.tex")
    if os.path.exists(macros_file):
        with open(macros_file, encoding="utf-8") as f:
            measured = set(re.findall(r"\\newcommand\{\\(Res[A-Za-z]+)\}", f.read()))
        pending = sorted(result_macros - measured - {"ResPending"})
        for name in pending:
            errors.append(f"\\{name} has no measured value -- would print [PENDING]")
    else:
        warnings.append("tables/results_macros.tex not present: every result macro will "
                        "render as [PENDING]. Run the notebook before submitting.")

    # -- length estimate ------------------------------------------------------------
    body = text.split(r"\begin{document}", 1)[-1].split(r"\begin{thebibliography}", 1)[0]
    words = len(re.findall(r"[A-Za-z][A-Za-z'-]+", body))
    # Count floats in the body only: the preamble's fallback definitions contain
    # \begin{table} of their own and would otherwise be counted as content.
    # Starred floats span both columns and cost roughly twice the page area of an
    # unstarred one. Counting them alike over-estimated the layout by close to a
    # page once the paper had several single-column floats.
    #
    # \GeneratedTable emits its own table* environment, so it is counted from the
    # generated file rather than from main.tex; \GeneratedFigure always sits
    # inside a figure environment already counted here.
    n_wide_fig = len(re.findall(r"\\begin\{figure\*\}", body))
    n_narrow_fig = len(re.findall(r"\\begin\{figure\}", body))
    n_wide_tab = len(re.findall(r"\\begin\{table\*\}", body))
    n_narrow_tab = len(re.findall(r"\\begin\{table\}", body))

    tables_dir_f = os.path.join(os.path.dirname(os.path.abspath(path)), "tables")
    for name in re.findall(r"\\GeneratedTable\{([^}]*)\}", body):
        gen = os.path.join(tables_dir_f, name)
        if os.path.exists(gen):
            head = open(gen, encoding="utf-8").read(200)
            if r"\begin{table*}" in head:
                n_wide_tab += 1
            else:
                n_narrow_tab += 1
        else:
            n_wide_tab += 1          # assume the expensive case when unknown

    n_tables = n_wide_tab + n_narrow_tab
    n_figures = n_wide_fig + n_narrow_fig
    n_refs = len(defined)
    # IEEEtran two-column: roughly 950 words per page of pure prose; a
    # single-column float costs ~0.30 page and a double-column one ~0.55.
    prose_pages = words / 950.0
    float_pages = 0.30 * (n_wide_tab + n_wide_fig) + 0.15 * (n_narrow_tab + n_narrow_fig)
    ref_pages = n_refs / 34.0
    estimate = prose_pages + float_pages + ref_pages

    # The float term assumes every table is a full-width `table*` set at 10pt.
    # Single-column tables and \footnotesize ones are substantially cheaper, so
    # treat the figure as an upper bound with roughly +/-0.4 pages of slack.
    optimistic = words / 1080.0 + 0.85 * float_pages + ref_pages

    print(f"  words (body): {words}   tables: {n_tables}   figures: {n_figures}   "
          f"refs: {n_refs}")
    print(f"  rough length: {optimistic:.1f}-{estimate:.1f} pages "
          f"(prose {prose_pages:.1f} + floats {float_pages:.1f} + refs {ref_pages:.1f})")
    print("  this is a heuristic with roughly +/-0.4 pages of error; it assumes every "
          "float is\n  a full-width 10pt table. Compiling is the only authoritative answer.")
    if optimistic > 6.0:
        warnings.append(f"even the optimistic estimate is {optimistic:.1f} pages against a "
                        f"6-page limit -- cut roughly {int((optimistic - 6.0) * 1080)} words "
                        f"or a float")
    elif estimate > 6.0:
        warnings.append(f"length is borderline ({optimistic:.1f}-{estimate:.1f} pages against a "
                        f"6-page limit). Compile and count before cutting; if over, follow the "
                        f"cut order in docs/RUNBOOK.md rather than trimming at random")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", default=None)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.abspath(args.paper or os.path.join(os.path.dirname(repo_root), "main.tex"))
    if not os.path.exists(path):
        print(f"not found: {path}")
        return 1

    print(f"checking {path}")
    errors, warnings = check(path)

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print("  ~", w)
    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print("  x", e)
        print("\nLATEX CHECK FAILED")
        return 1

    print("\nLATEX CHECK PASSED (static only -- still compile before submitting)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
