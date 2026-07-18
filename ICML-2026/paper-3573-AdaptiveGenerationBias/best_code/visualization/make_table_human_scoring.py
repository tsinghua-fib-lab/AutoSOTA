#!/usr/bin/env python3
import sys
import json
from pathlib import Path

LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(s: str) -> str:
    return "".join(LATEX_SPECIALS.get(ch, ch) for ch in s)


def strip_suffix(name: str) -> str:
    if name.endswith(".jsonl"):
        return name[:-6]
    if name.endswith(".json"):
        return name[:-5]
    return name


def count_jsonl_lines(path: Path) -> int:
    return sum(1 for line in path.open("r", encoding="utf-8") if line.strip())


def parse_dedup_json(path: Path):
    """Return (threshold, original_count, removed_count) if it looks like a dedup file."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        threshold = data.get("threshold")
        original = data.get("original_count")
        removed_list = data.get("removed_questions")
        removed = len(removed_list) if isinstance(removed_list, list) else None
        if threshold is not None and original is not None and removed is not None:
            return threshold, original, removed
    except Exception:
        pass
    return None, None, None


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} <directory_path>")
        sys.exit(1)

    root = Path(sys.argv[1]).expanduser().resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory")
        sys.exit(2)

    # Gather JSONL counts (columns)
    jsonl_files = sorted(
        [p for p in root.iterdir() if p.is_file() and p.suffix == ".jsonl"],
        key=lambda p: p.name.lower(),
    )
    jsonl_cols = []
    for p in jsonl_files:
        base = strip_suffix(p.name)
        count = count_jsonl_lines(p)
        jsonl_cols.append((latex_escape(base), str(count)))

    # Find dedup JSON(s)
    dedup_infos = []
    for p in sorted(
        [p for p in root.iterdir() if p.is_file() and p.suffix == ".json"],
        key=lambda p: p.name.lower(),
    ):
        th, oc, rc = parse_dedup_json(p)
        if th is not None:
            dedup_infos.append((strip_suffix(p.name), th, oc, rc))

    # Choose primary dedup (first) for Original/Removed; list all thresholds as comments
    original = ""
    removed = ""
    if dedup_infos:
        _, th0, oc0, rc0 = dedup_infos[0]
        original = str(oc0)
        removed = str(rc0)

    # Comments
    print(f"% Directory: {root}")
    print(f"% JSONL files detected: {len(jsonl_cols)}")
    total_jsonl = sum(int(c) for _, c in jsonl_cols) if jsonl_cols else 0
    print(f"% Total JSONL records: {total_jsonl}")
    if dedup_infos:
        print("% Thresholds from dedup JSON files:")
        for name, th, _, _ in dedup_infos:
            print(f"%   {name}: threshold = {th}")
    print()

    # Build LaTeX table (single header, single row)
    # Columns: Attribute | Original | Removed | <one per jsonl>
    headers = ["\\textbf{Attribute}", "\\textbf{Original}", "\\textbf{Removed}"] + [
        f"\\textbf{{{h}}}" for h, _ in jsonl_cols
    ]
    # Column spec: left for first, right for numbers
    colspec = "l" + "r" * (len(headers) - 1)

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Single-row summary with dedup counts and JSONL record counts.}")
    print(r"\vspace{-2mm}")
    print(r"\label{tab:single-row-summary}")
    print(r"\small")
    print(r"\begin{tabular}{" + colspec + "}")
    print(r"\toprule")
    print(" & ".join(headers) + r" \\")
    print(r"\midrule")

    attribute_label = latex_escape(root.name)  # generic attribute = directory name
    row = [attribute_label, original, removed] + [c for _, c in jsonl_cols]
    print(" & ".join(row) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
