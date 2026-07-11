#!/usr/bin/env python
"""Aggregate per-dataset benchmark log files into per-algorithm mean tables.

Reads log files from ``logs/benchmark/<predictor>/<dataset>.log``,
parses both the standard results table (Hit / Miss / Total / Hit Rate /
Cost Ratio) and the RPB instrumentation table (L0-miss / Non-L0-miss /
Gate-Pass / Gate-Fail / Pass Rate / Pred-Used / OM-Used), then computes
per-algorithm mean (and std) across datasets.

Output:
  * one standard-metrics table per predictor
  * one RPB-instrumentation table per predictor (for algorithms that
    expose those counters)

Sibling of ``scripts/aggregate_results.py`` (which reads flat CSVs from
``stat/``). This script targets the richer per-run logs that include the
RPB instrumentation introduced after the CSV format was frozen.
"""
import argparse
import math
import os
import re
import sys
from collections import defaultdict


SPEC2006_DATASETS = [
    "astar", "bwaves", "bzip", "cactusadm", "gems",
    "lbm", "leslie3d", "libq", "mcf", "milc",
    "omnetpp", "sphinx3", "xalanc",
]

# Standard results table columns (output of `verbose=True` in benchmark/__main__.py)
STD_HEADER = ["Name", "Hit", "Miss", "Total", "Hit Rate", "Cost Ratio"]
STD_NUMERIC = ["Hit Rate", "Cost Ratio"]
STD_DISPLAY = ["Hit Rate", "Cost Ratio", "LRU-norm Cost Ratio"]

# RPB instrumentation table columns
RPB_HEADER = [
    "Name", "L0-miss", "Non-L0-miss", "Gate-Pass", "Gate-Fail",
    "Pass Rate", "Pred-Used (non-L0)", "OM-Used (non-L0)",
]
RPB_NUMERIC = [
    "L0-miss", "Non-L0-miss", "Gate-Pass", "Gate-Fail",
    "Pass Rate", "Pred-Used (non-L0)", "OM-Used (non-L0)",
]
RPB_DISPLAY = RPB_NUMERIC  # all of these are shown


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

TABLE_ROW_RE = re.compile(r"^\|(.*)\|\s*$")


def _parse_cell(value):
    """Convert a stripped table cell to float when possible, else return str."""
    try:
        return float(value)
    except ValueError:
        return value


def parse_log_tables(log_path):
    """Parse a benchmark log file. Returns a list of tables.

    Each table is a dict ``{algo_name: {col: value}}`` where numeric
    cells are floats and the algorithm name (first column "Name") is
    used as the key. Multiple tables are returned in document order.

    Robust to tqdm progress lines (they don't start with "|") and to
    arbitrary text interleaved with the tables.
    """
    tables = []
    current_header = None
    current_data = {}

    def flush():
        nonlocal current_header, current_data
        if current_header is not None and current_data:
            tables.append((current_header, current_data))
        current_header = None
        current_data = {}

    with open(log_path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = TABLE_ROW_RE.match(line)
            if not m:
                # Non-table line (including "+----+" separators inside a
                # table) is ignored. A table is only closed by a new
                # header row or end-of-file.
                continue
            parts = [c.strip() for c in m.group(1).split("|")]
            if not parts:
                continue
            if parts[0] == "Name":
                flush()
                current_header = parts
                current_data = {}
                continue
            if current_header is None:
                # Stray "|...|" line outside a table; ignore.
                continue
            if len(parts) != len(current_header):
                # Malformed row; skip but keep the table open.
                continue
            algo = parts[0]
            row = {}
            for col, raw in zip(current_header, parts):
                if col == "Name":
                    row[col] = raw
                else:
                    row[col] = _parse_cell(raw)
            current_data[algo] = row

    flush()
    return tables


def classify_table(header):
    """Return 'std', 'rpb', or 'other' based on header columns."""
    cols = set(header)
    if {"Hit Rate", "Cost Ratio"}.issubset(cols) and "Hit" in cols:
        return "std"
    if "Gate-Pass" in cols and "Gate-Fail" in cols:
        return "rpb"
    return "other"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def discover_predictors(logs_dir):
    return sorted(
        d for d in os.listdir(logs_dir)
        if os.path.isdir(os.path.join(logs_dir, d))
    )


def discover_datasets_for_predictor(logs_dir, predictor):
    """Return list of dataset names (without extension) inferred from
    ``<predictor>/*.log``. For oracle-style logs the filename is
    ``<dataset>_<noise>.log``; we keep the full stem so the caller can
    still aggregate distinct noise types.
    """
    pred_dir = os.path.join(logs_dir, predictor)
    if not os.path.isdir(pred_dir):
        return []
    out = []
    for f in os.listdir(pred_dir):
        if f.endswith(".log"):
            out.append(f.removesuffix(".log"))
    return sorted(out)


def aggregate_predictor(logs_dir, predictor, datasets):
    """Aggregate one predictor's logs across a dataset list.

    Returns ``(std_order, std_values, rpb_order, rpb_values, n_found, missing)``.
    """
    std_values = defaultdict(lambda: defaultdict(list))
    std_order = []
    rpb_values = defaultdict(lambda: defaultdict(list))
    rpb_order = []
    n_found = 0
    missing = []

    for ds in datasets:
        path = os.path.join(logs_dir, predictor, f"{ds}.log")
        if not os.path.isfile(path):
            missing.append(ds)
            continue
        n_found += 1
        tables = parse_log_tables(path)
        # Per-dataset LRU cost for normalization
        std_table = None
        for header, data in tables:
            if classify_table(header) == "std":
                std_table = data
                break
        lru_cost = None
        if std_table:
            lru_row = std_table.get("LRU")
            if lru_row and isinstance(lru_row.get("Cost Ratio"), float):
                lru_cost = lru_row["Cost Ratio"]

        for header, data in tables:
            kind = classify_table(header)
            if kind == "std":
                for algo, row in data.items():
                    hr = row.get("Hit Rate")
                    cr = row.get("Cost Ratio")
                    if not (isinstance(hr, float) and isinstance(cr, float)):
                        continue
                    if math.isnan(hr) or math.isnan(cr):
                        continue
                    if algo not in std_values:
                        std_order.append(algo)
                    std_values[algo]["Hit Rate"].append(hr)
                    std_values[algo]["Cost Ratio"].append(cr)
                    if lru_cost is not None and lru_cost != 1.0:
                        std_values[algo]["LRU-norm Cost Ratio"].append(
                            (cr - 1.0) / (lru_cost - 1.0)
                        )
                    else:
                        std_values[algo]["LRU-norm Cost Ratio"].append(0.0)
            elif kind == "rpb":
                for algo, row in data.items():
                    if algo not in rpb_values:
                        rpb_order.append(algo)
                    for col in RPB_NUMERIC:
                        v = row.get(col)
                        if isinstance(v, float) and not math.isnan(v):
                            rpb_values[algo][col].append(v)

    # deduplicate (defensive)
    std_order = list(dict.fromkeys(std_order))
    rpb_order = list(dict.fromkeys(rpb_order))
    return std_order, std_values, rpb_order, rpb_values, n_found, missing


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def std(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def _format_number(v):
    """Friendly number formatting for both small floats and integers."""
    if v >= 1000 or (v == int(v) and v >= 1):
        return f"{v:.1f}"
    if abs(v) >= 1.0:
        return f"{v:.4f}"
    return f"{v:.4f}"


def print_std_table(predictor, order, values, n_found):
    print(f"\n{'=' * 80}")
    print(f"  {predictor} — standard metrics  ({n_found} datasets)")
    print(f"{'=' * 80}")

    header = ["Name", "N"]
    for col in STD_DISPLAY:
        header.append(col)
        header.append("std")
    col_widths = [max(len(h), 40) if h == "Name" else (8 if h == "std" else max(len(h), 10))
                  for h in header]

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"

    def fmt(vals):
        return "| " + " | ".join(
            v.ljust(w) if isinstance(v, str) else v.rjust(w)
            for v, w in zip(vals, col_widths)
        ) + " |"

    print(sep)
    print(fmt(header))
    print(sep)
    for algo in order:
        per_col = values[algo]
        n = len(per_col.get("Hit Rate", []))
        row = [algo, str(n)]
        for col in STD_DISPLAY:
            vs = per_col.get(col, [])
            row.append(f"{mean(vs):.4f}")
            row.append(f"{std(vs):.4f}")
        print(fmt(row))
    print(sep)


def print_rpb_table(predictor, order, values, n_found):
    if not order:
        return
    print(f"\n{'-' * 80}")
    print(f"  {predictor} — RPB instrumentation  ({n_found} datasets, "
          f"{len(order)} variants with counters)")
    print(f"{'-' * 80}")

    header = ["Name", "N"]
    for col in RPB_DISPLAY:
        header.append(col)
    col_widths = [
        max(len(h), 40) if h == "Name"
        else max(len(h), 12)
        for h in header
    ]

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"

    def fmt(vals):
        return "| " + " | ".join(
            v.ljust(w) if isinstance(v, str) else v.rjust(w)
            for v, w in zip(vals, col_widths)
        ) + " |"

    print(sep)
    print(fmt(header))
    print(sep)
    for algo in order:
        per_col = values[algo]
        n = max(len(vs) for vs in per_col.values()) if per_col else 0
        row = [algo, str(n)]
        for col in RPB_DISPLAY:
            vs = per_col.get(col, [])
            if not vs:
                row.append("-")
            else:
                row.append(_format_number(mean(vs)))
        print(fmt(row))
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark logs from logs/benchmark/<predictor>/<dataset>.log "
                    "into per-algorithm mean tables (standard metrics + RPB instrumentation).")
    parser.add_argument("--logs_dir", default="logs/benchmark",
                        help="root logs directory (default: logs/benchmark)")
    parser.add_argument("--predictor", default=None,
                        help="comma-separated predictor names to aggregate "
                             "(default: auto-discover all subdirs of logs_dir)")
    parser.add_argument("--expected", default="spec2006",
                        choices=["spec2006", "auto"],
                        help="expected dataset set; 'auto' uses every *.log in the predictor dir "
                             "(default: spec2006)")
    parser.add_argument("--exclude", default=None,
                        help="regex; algorithms matching are dropped from the output tables")
    parser.add_argument("--include", default=None,
                        help="regex; only algorithms matching are kept in the output tables")
    args = parser.parse_args()

    if not os.path.isdir(args.logs_dir):
        print(f"ERROR: logs directory not found: {args.logs_dir}", file=sys.stderr)
        sys.exit(1)

    if args.predictor:
        predictors = [p.strip() for p in args.predictor.split(",") if p.strip()]
    else:
        predictors = discover_predictors(args.logs_dir)
        if not predictors:
            print(f"ERROR: no predictor subdirs in {args.logs_dir}/", file=sys.stderr)
            sys.exit(1)
        print(f"Auto-discovered predictors: {', '.join(predictors)}")

    include_re = re.compile(args.include) if args.include else None
    exclude_re = re.compile(args.exclude) if args.exclude else None

    def _filter(order):
        return [a for a in order
                if (include_re is None or include_re.search(a))
                and (exclude_re is None or not exclude_re.search(a))]

    for pred in predictors:
        if args.expected == "spec2006":
            datasets = SPEC2006_DATASETS
        else:
            datasets = discover_datasets_for_predictor(args.logs_dir, pred)

        std_order, std_values, rpb_order, rpb_values, n_found, missing = aggregate_predictor(
            args.logs_dir, pred, datasets)

        if n_found == 0:
            print(f"\nWARNING: no logs found for predictor '{pred}' under {args.logs_dir}/{pred}/")
            continue

        if missing:
            print(f"\nWARNING: missing logs for '{pred}': {', '.join(missing)}")

        std_order = _filter(std_order)
        rpb_order = _filter(rpb_order)

        print_std_table(pred, std_order, std_values, n_found)
        print_rpb_table(pred, rpb_order, rpb_values, n_found)


if __name__ == "__main__":
    main()
