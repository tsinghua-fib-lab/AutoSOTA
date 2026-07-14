#!/usr/bin/env python3
"""
Summarize wind experiment results into LaTeX tables.

Given a root directory that contains per-model subdirectories, each with
combination subdirectories (e.g., vdw_modular/n100p10) holding
results_*.yaml files, this script aggregates metrics across replications
and emits two LaTeX tables: one for valid_mse and one for rotation_mse.

Tables are grouped by sample_n (rows: models; columns: mask_prop). The
default aggregation is mean ± std; you can choose median [min, max].

Output: root_dir/latex_tables.txt containing the two tables.

Example call:
python3 scripts/python/summarize_wind_tables.py \
    --root_dir=/path/to/experiments/wind

Single-combo table:
python3 scripts/python/summarize_wind_tables.py \
    --root_dir=/path/to/experiments/wind \
    --single=n2000p30
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Ensure project code directory is on the path so results_utils can be imported
CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

import results_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize wind experiment results into LaTeX tables."
    )
    parser.add_argument(
        "--root_dir",
        type=Path,
        required=True,
        help="Root directory containing per-model subdirectories with nXXXpYY combos.",
    )
    parser.add_argument(
        "--agg_mode",
        type=str,
        default="mean_std",
        choices=["mean_std", "median_range"],
        help="Aggregation display mode: mean_std (mean ± std) or median_range (median[min,max]).",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Number of decimals for metric formatting.",
    )
    parser.add_argument(
        "--bold_best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bold the best model (lowest metric) per sample_n & mask_prop.",
    )
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help=(
            "If provided (e.g., n2000p30), generate a single table for that combo "
            "with Model, Validation MSE, Test MSE, Rotated Test MSE, and Parameter Count."
        ),
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="latex_tables.txt",
        help="Output filename written inside root_dir.",
    )
    return parser.parse_args()


def parse_combo_dir_name(
    name: str,
) -> Tuple[str, str] | None:
    if not name.startswith("n") or "p" not in name:
        return None
    try:
        n_part, p_part = name[1:].split("p", 1)
        sample_n = str(int(n_part))
        mask_prop = str(int(p_part)) if p_part.isdigit() else p_part
        mask_prop_val = float(mask_prop) / 100 if mask_prop.isdigit() else float(mask_prop)
        mask_prop = f"{mask_prop_val:.2f}"
    except Exception:
        return None
    return sample_n, mask_prop


def parse_single_combo(
    combo: str,
) -> Tuple[str, str]:
    name = combo.strip()
    if not name.startswith("n"):
        name = f"n{name}"
    parsed = parse_combo_dir_name(name)
    if parsed is None:
        raise ValueError(f"Invalid combo name: {combo}")
    return parsed


def discover_results(
    root: Path,
    single_combo: str | None = None,
) -> Dict[str, Dict[str, Dict[str, Path]]]:
    """
    Traverse root/model/combination, collecting result files.

    Returns: {sample_n: {mask_prop: {model: [Path,...]}}}
    """
    layout: Dict[str, Dict[str, Dict[str, List[Path]]]] = {}
    desired_combo = parse_single_combo(single_combo) if single_combo else None
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        model = model_dir.name
        for combo_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            # Expect combo_dir name like n100p10
            parsed = parse_combo_dir_name(combo_dir.name)
            if parsed is None:
                continue
            if desired_combo is not None and parsed != desired_combo:
                continue
            sample_n, mask_prop = parsed
            yaml_paths = sorted(combo_dir.glob("results_*.yaml"))
            if not yaml_paths:
                continue
            layout.setdefault(sample_n, {}).setdefault(mask_prop, {}).setdefault(model, []).extend(
                yaml_paths
            )
    return layout


def metric_exists_in_layout(
    layout: Dict[str, Dict[str, Dict[str, List[Path]]]],
    metric: str,
) -> bool:
    """
    Returns True if any result file in the layout contains the requested metric.
    """
    for mp_dict in layout.values():
        for model_dict in mp_dict.values():
            for paths in model_dict.values():
                for path in paths:
                    try:
                        data = results_utils.read_yaml(path)
                    except Exception as exc:
                        print(f"[WARN] Failed to read {path}: {exc}")
                        continue
                    if metric in data:
                        return True
    return False


def aggregate(
    layout: Dict[str, Dict[str, Dict[str, List[Path]]]],
    metrics: List[str],
) -> Dict:
    aggregated = {}
    for sample_n, mp_dict in layout.items():
        for mask_prop, model_dict in mp_dict.items():
            for model, paths in model_dict.items():
                stats = results_utils.aggregate_replication_results(
                    result_paths=paths,
                    metrics=metrics,
                )
                aggregated.setdefault(sample_n, {}).setdefault(mask_prop, {})[model] = stats
    return aggregated


def extract_best_flags(
    aggregated: Dict,
    metric: str,
) -> Dict[Tuple[str, str], str]:
    """
    Identify best (lowest) mean per (sample_n, mask_prop).
    Returns map of (sample_n, mask_prop) -> model name that wins.
    """
    best: Dict[Tuple[str, str], str] = {}
    for sample_n, mp_dict in aggregated.items():
        for mask_prop, model_dict in mp_dict.items():
            best_model = None
            best_val = math.inf
            for model, stats in model_dict.items():
                mean_val = stats.get(metric, {}).get("mean")
                if mean_val is None or not math.isfinite(float(mean_val)):
                    continue
                if float(mean_val) < best_val:
                    best_val = float(mean_val)
                    best_model = model
            if best_model is not None:
                best[(sample_n, mask_prop)] = best_model
    return best


def format_cell(
    stats: Dict[str, float],
    mode: str,
    decimals: int,
) -> str:
    return results_utils.format_metric(stats, mode=mode, decimals=decimals, force_int=False)


def format_cell_for_metric(
    stats: Dict[str, float],
    mode: str,
    decimals: int,
    metric: str,
) -> str:
    force_int = metric in {"parameter_count", "best_epoch"}
    return results_utils.format_metric(stats, mode=mode, decimals=decimals, force_int=force_int)


def build_table(
    aggregated: Dict,
    metric: str,
    mode: str,
    decimals: int,
    bold_best: bool,
) -> str:
    sample_ns = sorted(aggregated.keys(), key=lambda x: int(x))
    # Collect all mask props to order columns numerically
    all_mask_props = sorted(
        {mp for mp_dict in aggregated.values() for mp in mp_dict.keys()},
        key=lambda x: float(x),
    )
    best_flags = extract_best_flags(aggregated, metric) if bold_best else {}

    lines: List[str] = []
    header_cols = (
        " & ".join(
            [r"\textbf{Sample size}", r"\textbf{Model}"] + [f"$p={mp}$" for mp in all_mask_props]
        )
        + r" \\"
    )
    lines.append(r"\begin{tabular}{ll" + "c" * len(all_mask_props) + "}")
    lines.append(r"\toprule")
    lines.append(header_cols)
    lines.append(r"\midrule")

    for sample_n in sample_ns:
        model_set = set()
        for mp_dict in aggregated[sample_n].values():
            model_set.update(mp_dict.keys())
        sorted_models = sorted(model_set)
        for idx, model in enumerate(sorted_models):
            sample_cell = rf"\multirow{{{len(sorted_models)}}}{{*}}{{n={sample_n}}}" if idx == 0 else ""
            row_cells = [sample_cell, model]
            for mp in all_mask_props:
                stats = aggregated[sample_n].get(mp, {}).get(model, {}).get(metric, {})
                cell = format_cell(stats, mode=mode, decimals=decimals) if stats else "n/a"
                if bold_best and best_flags.get((sample_n, mp)) == model and "n/a" not in cell:
                    cell = r"\textbf{" + cell + "}"
                row_cells.append(cell)
            lines.append(" & ".join(row_cells) + r" \\")
        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def build_wind_rot_table(
    aggregated: Dict,
    metrics: List[str],
    mode: str,
    decimals: int,
    bold_best: bool,
) -> str:
    """
    Build a single-combo table (models as rows, metrics as columns) for wind_rot.
    """
    sample_keys = sorted(aggregated.keys(), key=lambda x: int(x))
    if not sample_keys:
        raise RuntimeError("No aggregated entries available for wind_rot.")
    sample_key = sample_keys[0]
    mask_keys = sorted(aggregated[sample_key].keys(), key=lambda x: float(x))
    if not mask_keys:
        raise RuntimeError("No mask_prop entries available for wind_rot.")
    mask_key = mask_keys[0]
    model_dict = aggregated[sample_key].get(mask_key, {})
    models = sorted(model_dict.keys())

    def _escape_latex(
        text: str,
    ) -> str:
        return text.replace("_", "\\_")

    def _best_models() -> Dict[str, str]:
        best_map: Dict[str, str] = {}
        for metric in metrics:
            best_model = None
            best_val = math.inf
            for model in models:
                stats = model_dict.get(model, {}).get(metric, {})
                mean_val = stats.get("mean")
                if mean_val is None or not math.isfinite(float(mean_val)):
                    continue
                if float(mean_val) < best_val:
                    best_val = float(mean_val)
                    best_model = model
            if best_model is not None:
                best_map[metric] = best_model
        return best_map

    best_flags = _best_models() if bold_best else {}
    header_cells = ["\\textbf{Model}"] + [f"\\textbf{{{_escape_latex(m)}}}" for m in metrics]
    lines: List[str] = []
    lines.append(r"\begin{tabular}{" + "l" + "c" * len(metrics) + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    for model in models:
        row_cells = [model]
        for metric in metrics:
            stats = model_dict.get(model, {}).get(metric, {})
            cell = format_cell(stats, mode=mode, decimals=decimals)
            if bold_best and best_flags.get(metric) == model and "n/a" not in cell:
                cell = r"\textbf{" + cell + "}"
            row_cells.append(cell)
        lines.append(" & ".join(row_cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def build_single_combo_table(
    aggregated: Dict,
    mode: str,
    decimals: int,
    bold_best: bool,
) -> str:
    """
    Build a single-combo table (models as rows, metrics as columns).
    """
    sample_keys = sorted(aggregated.keys(), key=lambda x: int(x))
    if not sample_keys:
        raise RuntimeError("No aggregated entries available for single-combo table.")
    sample_key = sample_keys[0]
    mask_keys = sorted(aggregated[sample_key].keys(), key=lambda x: float(x))
    if not mask_keys:
        raise RuntimeError("No mask_prop entries available for single-combo table.")
    mask_key = mask_keys[0]
    model_dict = aggregated[sample_key].get(mask_key, {})
    models = sorted(model_dict.keys())

    columns = [
        ("test_mse", "Test MSE"),
        ("rotation_mse", "Rotated Test MSE"),
        ("mean_train_time", "Sec/epoch"),
        ("best_epoch", "Best Epoch"),
        ("parameter_count", "Parameter Count"),
    ]

    def _escape_latex(
        text: str,
    ) -> str:
        return text.replace("_", "\\_")

    def _best_models() -> Dict[str, str]:
        best_map: Dict[str, str] = {}
        for metric, _ in columns:
            best_model = None
            best_val = math.inf
            for model in models:
                stats = model_dict.get(model, {}).get(metric, {})
                mean_val = stats.get("mean")
                if mean_val is None or not math.isfinite(float(mean_val)):
                    continue
                if float(mean_val) < best_val:
                    best_val = float(mean_val)
                    best_model = model
            if best_model is not None:
                best_map[metric] = best_model
        return best_map

    def _param_std_zero_everywhere() -> bool:
        if not models:
            return False
        for model in models:
            param_stats = model_dict.get(model, {}).get("parameter_count", {})
            std_val = param_stats.get("std")
            if std_val is None:
                return False
            try:
                if not math.isclose(float(std_val), 0.0):
                    return False
            except (TypeError, ValueError):
                return False
        return True

    best_flags = _best_models() if bold_best else {}
    suppress_param_uncertainty = (mode == "mean_std") and _param_std_zero_everywhere()
    header_cells = ["\\textbf{Model}"] + [
        f"\\textbf{{{_escape_latex(label)}}}" for _, label in columns
    ]
    lines: List[str] = []
    lines.append(r"\begin{tabular}{" + "l" + "c" * len(columns) + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    for model in models:
        row_cells = [model]
        for metric, _ in columns:
            stats = model_dict.get(model, {}).get(metric, {})
            if metric == "parameter_count" and suppress_param_uncertainty:
                mean_param = stats.get("mean")
                if mean_param is None or not math.isfinite(float(mean_param)):
                    cell = "n/a"
                else:
                    cell = results_utils._round_value(mean_param, decimals, force_int=True)
            else:
                cell = format_cell_for_metric(stats, mode=mode, decimals=decimals, metric=metric)
            if bold_best and best_flags.get(metric) == model and "n/a" not in cell:
                cell = r"\textbf{" + cell + "}"
            row_cells.append(cell)
        lines.append(" & ".join(row_cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.root_dir.resolve()
    layout = discover_results(root, single_combo=args.single)
    if not layout:
        if args.single:
            raise RuntimeError(f"No result files found under {root} for {args.single}")
        raise RuntimeError(f"No result files found under {root}")

    mode = "mean_std" \
        if args.agg_mode == "mean_std" else "median_range"
    is_wind_rot = any(part == "wind_rot" for part in root.parts)

    if args.single:
        metrics = ["test_mse", "rotation_mse", "mean_train_time", "best_epoch", "parameter_count"]
        aggregated = aggregate(layout=layout, metrics=metrics)
        single_table = build_single_combo_table(
            aggregated=aggregated,
            mode=mode,
            decimals=args.decimals,
            bold_best=args.bold_best,
        )
        output_name = args.output_name
        if output_name == "latex_tables.txt":
            output_name = "single_latex.txt"
        out_path = root / output_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write("% Single-combo metrics table\n")
            f.write(single_table)
        print(f"[DONE] Wrote single-combo table to {out_path}")
    elif is_wind_rot:
        rotation_present = metric_exists_in_layout(layout, metric="rotation_mse")
        metrics = ["valid_mse", "test_mse"]
        if rotation_present:
            metrics.append("rotation_mse")
        metrics.append("offline_comp_time")

        aggregated = aggregate(layout=layout, metrics=metrics)
        wind_rot_table = build_wind_rot_table(
            aggregated=aggregated,
            metrics=metrics,
            mode=mode,
            decimals=args.decimals,
            bold_best=args.bold_best,
        )
        out_path = root / args.output_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write("% Wind_rot metrics table\n")
            f.write(wind_rot_table)
        print(f"[DONE] Wrote wind_rot table to {out_path}")
    else:
        rotation_present = metric_exists_in_layout(layout, metric="rotation_mse")
        metrics = ["valid_mse"] + (["rotation_mse"] if rotation_present else [])

        aggregated = aggregate(layout=layout, metrics=metrics)

        valid_table = build_table(
            aggregated=aggregated,
            metric="valid_mse",
            mode=mode,
            decimals=args.decimals,
            bold_best=args.bold_best,
        )
        rotation_table = build_table(
            aggregated=aggregated,
            metric="rotation_mse",
            mode=mode,
            decimals=args.decimals,
            bold_best=args.bold_best,
        ) if rotation_present else None

        out_path = root / args.output_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write("% Valid MSE table\n")
            f.write(valid_table)
            if rotation_table:
                f.write("\n\n% Rotation MSE table\n")
                f.write(rotation_table)
            else:
                f.write("\n\n% Rotation MSE table unavailable (no rotation_mse in results)\n")

        print(f"[DONE] Wrote LaTeX tables to {out_path}")


if __name__ == "__main__":
    main()

