#!/usr/bin/env python3
"""
Summarize cross-day, cross-model k-fold aggregated results into a LaTeX table.

Expected layout:
<root>/
    day_*/<model_name>/metrics/kfold_results.yaml

Outputs:
- LaTeX table saved to <root>/<table_file> (default: summary_table.tex)
- Pickled list of records [{day, model, mean_test_accuracy}] to <root>/<records_file>

Example call:
python3 scripts/python/summarize_day_model_results.py \
    --root /path/to/experiments \
    --uncertainty mean_std \
    --decimals 2 \
    --bold_best \
    --our_model_name "VDW-GNN"

"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
import sys
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    # Allow running the script from outside the repo root.
    sys.path.append(str(REPO_ROOT))

import results_utils as ru
from results_utils import (
    _round_value,
    collect_results,
    consistent_parameter_counts,
    day_sort_key,
    day_to_int,
    format_day_label,
    format_metric,
    model_sort_key,
    parameter_std_zero_everywhere,
    prettify_model_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize day/model k-fold results into LaTeX.")
    parser.add_argument(
        "--root", 
        type=Path, 
        default=Path("."), 
        help="Root directory containing day_* folders."
    )
    parser.add_argument(
        "--uncertainty",
        type=str,
        default="mean_std",
        choices=("mean_std", "median_range"),
        help="How to format uncertainty: mean ± std, or median[min,max].",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Number of decimals for non-integer metrics (except parameters/best_epoch).",
    )
    parser.add_argument(
        "--bold_best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bold the best test_accuracy within each day (default: True).",
    )
    parser.add_argument(
        "--table_file",
        type=str,
        default="summary_table.tex",
        help="Filename for the LaTeX table output (saved under --root).",
    )
    parser.add_argument(
        "--records_file",
        type=str,
        default="summary_records.pkl",
        help="Filename for pickled records (saved under --root).",
    )
    parser.add_argument(
        "--our_model_name",
        type=str,
        default="VDW-GNN",
        help="Our model name to use in the table.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model directory names to include (e.g., 'vdw_1,cebra,marble,lfads_1'). "
        "If omitted, all discovered models are included.",
    )
    parser.add_argument(
        "--single",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, emit a single summary table aggregated across all days.",
    )
    return parser.parse_args()



def build_table(
    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    mode: str,
    decimals: int,
    bold_best: bool,
    our_model_name: str,
    *,
    drop_params_column: bool = False,
) -> str:
    lines: List[str] = []
    if drop_params_column:
        header = [
            r"\begin{tabular}{llccc}",
            r"\toprule",
            r"Day & Model & Test Acc. & Mean Train Time & Best Epoch \\",
            r"\midrule",
        ]
    else:
        header = [
            r"\begin{tabular}{llcccc}",
            r"\toprule",
            r"Day & Model & Test Acc. & Mean Train Time & Best Epoch & Params \\",
            r"\midrule",
        ]
    lines.extend(header)

    suppress_param_uncertainty = (mode == "mean_std") and parameter_std_zero_everywhere(results)

    for day in sorted(results.keys(), key=day_sort_key):
        models = results[day]
        sorted_models = sorted(models.keys(), key=model_sort_key)
        # Identify best test accuracy (mean or median depending on mode)
        best_value = None
        if bold_best:
            for m in sorted_models:
                test_stats = models[m].get("test_accuracy", {})
                if mode == "mean_std":
                    candidate = test_stats.get("mean")
                else:
                    candidate = test_stats.get("median")
                if candidate is None:
                    continue
                if best_value is None or (candidate > best_value):
                    best_value = candidate

        first_row = True
        for m in sorted_models:
            stats = models[m]
            test_fmt = format_metric(stats.get("test_accuracy", {}), mode, decimals, force_int=False)
            if bold_best:
                center_val = stats.get("test_accuracy", {}).get("mean" if mode == "mean_std" else "median")
                if center_val is not None and best_value is not None and math.isfinite(center_val):
                    if math.isclose(center_val, best_value) or center_val == best_value:
                        test_fmt = r"\textbf{" + test_fmt + "}"

            train_time_fmt = format_metric(stats.get("mean_train_time", {}), mode, decimals, force_int=False)
            best_epoch_fmt = format_metric(stats.get("best_epoch", {}), mode, decimals, force_int=True)
            row_day = format_day_label(day) if first_row else ""
            first_row = False
            if drop_params_column:
                lines.append(
                    f"{row_day} & {prettify_model_name(m, our_model_name)} & {test_fmt} & {train_time_fmt} & {best_epoch_fmt} \\\\"
                )
            else:
                param_stats = stats.get("parameter_count", {})
                if suppress_param_uncertainty:
                    mean_param = param_stats.get("mean")
                    if mean_param is None or not math.isfinite(mean_param):
                        param_fmt = "n/a"
                    else:
                        param_fmt = _round_value(mean_param, decimals, force_int=True)
                else:
                    param_fmt = format_metric(param_stats, mode, decimals, force_int=True)

                lines.append(
                    f"{row_day} & {prettify_model_name(m, our_model_name)} & {test_fmt} & {train_time_fmt} & {best_epoch_fmt} & {param_fmt} \\\\"
                )
        lines.append(r"\midrule")

    # Replace trailing midrule with bottomrule
    if lines and lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def aggregate_grand_results(
    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    metrics: List[str],
    *,
    use_mean_std_for: set[str] | None = None,
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, bool]]:
    aggregated: Dict[str, Dict[str, Dict[str, float]]] = {}
    param_constant_map: Dict[str, bool] = {}
    model_names = sorted(
        {model for day_data in results.values() for model in day_data.keys()},
        key=model_sort_key,
    )
    use_mean_std_for = use_mean_std_for or set()
    for model in model_names:
        for metric in metrics:
            means: List[float] = []
            stds: List[float] = []
            for day_data in results.values():
                stats = day_data.get(model, {}).get(metric, {})
                mean_val = stats.get("mean")
                if mean_val is None or not math.isfinite(float(mean_val)):
                    continue
                means.append(float(mean_val))
                std_val = stats.get("std")
                if std_val is not None and math.isfinite(float(std_val)):
                    stds.append(float(std_val))
            if not means:
                continue
            metric_stats = ru.aggregate_values(means)
            if metric not in use_mean_std_for and stds:
                metric_stats["std"] = float(math.sqrt(sum(v ** 2 for v in stds)))
            aggregated.setdefault(model, {})[metric] = metric_stats

        param_means: List[float] = []
        param_stds: List[float] = []
        for day_data in results.values():
            stats = day_data.get(model, {}).get("parameter_count", {})
            mean_val = stats.get("mean")
            if mean_val is None or not math.isfinite(float(mean_val)):
                continue
            param_means.append(float(mean_val))
            std_val = stats.get("std")
            if std_val is not None and math.isfinite(float(std_val)):
                param_stds.append(float(std_val))
        if not param_means:
            param_constant_map[model] = False
        else:
            all_means_equal = all(
                math.isclose(mean_val, param_means[0]) for mean_val in param_means[1:]
            )
            all_std_zero = (
                bool(param_stds) and all(math.isclose(std_val, 0.0) for std_val in param_stds)
            )
            param_constant_map[model] = all_means_equal and all_std_zero

    return aggregated, param_constant_map


def build_single_table(
    aggregated: Dict[str, Dict[str, Dict[str, float]]],
    param_constant_map: Dict[str, bool],
    mode: str,
    decimals: int,
    bold_best: bool,
    our_model_name: str,
) -> str:
    columns = [
        ("test_accuracy", "Accuracy"),
        ("mean_train_time", "Sec/epoch"),
        ("best_epoch", "Best Epoch"),
        ("parameter_count", "Params"),
    ]
    lines: List[str] = []
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(
        " & ".join([r"\textbf{Model}"] + [f"\\textbf{{{label}}}" for _, label in columns])
        + r" \\"
    )
    lines.append(r"\midrule")

    models = sorted(aggregated.keys(), key=model_sort_key)
    best_vals: Dict[str, float] = {}
    if bold_best:
        for metric, _ in columns[:2]:
            best_val = None
            for model in models:
                stats = aggregated.get(model, {}).get(metric, {})
                center = stats.get("mean")
                if center is None or not math.isfinite(float(center)):
                    continue
                if metric == "test_accuracy":
                    if best_val is None or float(center) > best_val:
                        best_val = float(center)
                else:
                    if best_val is None or float(center) < best_val:
                        best_val = float(center)
            if best_val is not None:
                best_vals[metric] = best_val

    for model in models:
        row_cells = [prettify_model_name(model, our_model_name)]
        for metric, _ in columns:
            stats = aggregated.get(model, {}).get(metric, {})
            if metric == "parameter_count" and param_constant_map.get(model, False):
                mean_val = stats.get("mean")
                if mean_val is None or not math.isfinite(float(mean_val)):
                    cell = "n/a"
                else:
                    cell = _round_value(float(mean_val), decimals, force_int=True)
            else:
                force_int = metric in {"best_epoch", "parameter_count"}
                cell = format_metric(stats, mode, decimals, force_int=force_int)

            if bold_best and metric in {"test_accuracy", "mean_train_time"}:
                center_val = stats.get("mean")
                best_val = best_vals.get(metric)
                if (
                    center_val is not None
                    and best_val is not None
                    and math.isfinite(float(center_val))
                    and math.isclose(float(center_val), best_val)
                ):
                    cell = r"\textbf{" + cell + "}"
            row_cells.append(cell)
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    include_models = None
    if args.models:
        include_models = {m.strip() for m in args.models.split(",") if m.strip()}
    results = collect_results(root, include_models=include_models)
    if not results:
        print(f"No results found under {root}")
        return

    if args.single:
        metrics = ["test_accuracy", "mean_train_time", "best_epoch", "parameter_count"]
        aggregated, param_constant_map = aggregate_grand_results(
            results,
            metrics=metrics,
            use_mean_std_for={"mean_train_time", "test_accuracy"},
        )
        table = build_single_table(
            aggregated=aggregated,
            param_constant_map=param_constant_map,
            mode=args.uncertainty,
            decimals=args.decimals,
            bold_best=args.bold_best,
            our_model_name=args.our_model_name,
        )
        output_name = args.table_file
        if output_name == "summary_table.tex":
            output_name = "grand_means.txt"
        elif output_name.endswith(".tex"):
            output_name = output_name[:-4] + ".txt"
        table_path = root / output_name
        table_path.write_text(table, encoding="utf-8")
        print(f"Wrote single-table summary to {table_path}")
        return

    suffix = f"_{args.uncertainty}"
    table_filename = args.table_file
    records_filename = args.records_file
    if table_filename.endswith(".tex"):
        table_filename = table_filename[:-4] + suffix + ".tex"
    else:
        table_filename = table_filename + suffix
    if records_filename.endswith(".pkl"):
        records_filename = records_filename[:-4] + suffix + ".pkl"
    else:
        records_filename = records_filename + suffix

    params_constant, model_param_counts = consistent_parameter_counts(results)

    table = build_table(
        results=results,
        mode=args.uncertainty,
        decimals=args.decimals,
        bold_best=args.bold_best,
        our_model_name=args.our_model_name,
        drop_params_column=params_constant,
    )
    table_path = root / table_filename
    if params_constant:
        preamble_lines = ["Model parameter counts (constant across days):"]
        for model in sorted(model_param_counts.keys(), key=model_sort_key):
            count_val = model_param_counts[model]
            preamble_lines.append(f"{prettify_model_name(model, args.our_model_name)}: {_round_value(count_val, args.decimals, force_int=True)}")
        # Blank line between preamble and table
        output_text = "\n".join(preamble_lines + ["", table])
    else:
        output_text = table

    table_path.write_text(output_text, encoding="utf-8")
    print(f"Wrote LaTeX table to {table_path}")

    # Records for downstream use
    records: List[Dict[str, object]] = []
    center_key = "mean" if args.uncertainty == "mean_std" else "median"
    for day, models in results.items():
        for model, metrics in models.items():
            test_stats = metrics.get("test_accuracy", {})
            center_val = test_stats.get(center_key)
            records.append(
                {
                    "day": day_to_int(day),
                    "model": model,
                    "test_accuracy": center_val,
                    "uncertainty": args.uncertainty,
                }
            )
    records_path = root / records_filename
    with open(records_path, "wb") as f:
        pickle.dump(records, f)
    print(f"Wrote records to {records_path}")


if __name__ == "__main__":
    main()

