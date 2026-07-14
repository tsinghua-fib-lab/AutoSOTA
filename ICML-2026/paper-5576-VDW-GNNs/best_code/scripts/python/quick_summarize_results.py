#!/usr/bin/env python3
"""
Quickly summarize k-fold cross-validation metrics for multiple models.

Given a root directory, an experiment subdirectory, and one or more metric names,
this script looks for per-fold results in:
    <root_dir>/<model>/<exp_subdir>/fold_*/metrics/results.yaml

It aggregates each requested metric across folds as mean ± standard deviation
and prints a simple table (one row per model).
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Aggregate k-fold CV metrics across models.',
    )
    parser.add_argument(
        '--root_dir',
        type=str,
        required=True,
        help='Root directory containing model subdirectories.',
    )
    parser.add_argument(
        '--exp_subdirs',
        '--exp_subdir',
        dest='exp_subdirs',
        nargs='+',
        required=True,
        help='One or more experiment subdirectories (comma or space separated).',
    )
    parser.add_argument(
        '--metrics',
        nargs='+',
        required=True,
        help='Metric keys to read from fold_*/metrics/results.yaml files.',
    )
    parser.add_argument(
        '--decimals',
        type=int,
        default=4,
        help='Number of decimal places for mean and standard deviation.',
    )
    return parser.parse_args()


def load_yaml(path: str) -> Dict:
    """
    Load a YAML file and return a dictionary; returns an empty dict on failure.
    """
    try:
        with open(path, 'r') as handle:
            return yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Failed to read yaml at {path}: {exc}", file=sys.stderr)
        return {}


def collect_fold_metrics(
    run_dir: str,
    metrics: List[str],
) -> Dict[str, List[float]]:
    """
    Gather metric values from all folds within a single model run directory.
    """
    metric_values: Dict[str, List[float]] = {m: [] for m in metrics}
    if not os.path.isdir(run_dir):
        return metric_values

    try:
        entries = os.listdir(run_dir)
    except Exception as exc:
        print(f"Could not list directory {run_dir}: {exc}", file=sys.stderr)
        return metric_values

    fold_dirs = [
        os.path.join(run_dir, name)
        for name in entries
        if name.startswith('fold_') and os.path.isdir(os.path.join(run_dir, name))
    ]

    for fold_dir in sorted(fold_dirs):
        results_path = os.path.join(fold_dir, 'metrics', 'results.yaml')
        results = load_yaml(results_path)
        if not results:
            continue
        for metric in metrics:
            if metric in results:
                try:
                    metric_values[metric].append(float(results[metric]))
                except (TypeError, ValueError) as exc:
                    print(
                        f"Non-numeric value for {metric} in {results_path}: {exc}",
                        file=sys.stderr,
                    )
    return metric_values


def format_mean_std(
    values: List[float],
    decimals: int,
) -> str:
    """
    Format a list of numeric values as mean ± standard deviation.
    """
    if len(values) == 0:
        return 'NA'
    if len(values) == 1:
        mean_val = float(values[0])
        return f"{mean_val:.{decimals}f} ± {0.0:.{decimals}f}"
    arr = np.array(values, dtype=float)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1))
    return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"


def build_table(
    rows: List[Dict[str, str]],
    headers: List[str],
) -> str:
    """
    Build a simple text table from rows and headers.
    """
    if len(rows) == 0:
        return 'No results found.'

    widths = {h: len(h) for h in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row.get(header, ''))))

    header_line = ' | '.join(header.ljust(widths[header]) for header in headers)
    separator = '-+-'.join('-' * widths[header] for header in headers)

    body_lines = []
    for row in rows:
        body = ' | '.join(str(row.get(header, '')).ljust(widths[header]) for header in headers)
        body_lines.append(body)

    return '\n'.join([header_line, separator] + body_lines)


def find_model_run_dirs(
    root_dir: str,
    exp_subdirs: List[str],
) -> Dict[str, List[str]]:
    """
    Identify model directories and their corresponding experiment run paths.
    """
    model_runs: Dict[str, List[str]] = {}
    if not os.path.isdir(root_dir):
        print(f"Root directory does not exist: {root_dir}", file=sys.stderr)
        return model_runs

    try:
        candidates = os.listdir(root_dir)
    except Exception as exc:
        print(f"Could not list root directory {root_dir}: {exc}", file=sys.stderr)
        return model_runs

    for name in sorted(candidates):
        model_root = os.path.join(root_dir, name)
        if not os.path.isdir(model_root):
            continue
        run_dirs_set = set()
        for exp_subdir in exp_subdirs:
            # Case 1: explicit subdir inside model_root
            if exp_subdir:
                run_dir = os.path.join(model_root, exp_subdir)
                if os.path.isdir(run_dir):
                    run_dirs_set.add(run_dir)
            # Case 2: allow root to already be the desired exp_subdir
            if os.path.basename(model_root) == exp_subdir and os.path.isdir(model_root):
                run_dirs_set.add(model_root)
            # Case 3: empty exp_subdir means use model_root directly
            if exp_subdir == '' and os.path.isdir(model_root):
                run_dirs_set.add(model_root)
        if len(run_dirs_set) > 0:
            model_runs[name] = sorted(run_dirs_set)
    return model_runs


def main() -> None:
    args = parse_args()
    metrics = [m.strip() for m in args.metrics if m.strip()]
    if len(metrics) == 0:
        print('No metrics provided.', file=sys.stderr)
        sys.exit(1)

    exp_subdirs: List[str] = []
    for raw in args.exp_subdirs:
        for part in raw.split(','):
            token = part.strip()
            if token == '':
                exp_subdirs.append('')
            else:
                exp_subdirs.append(token.lstrip(os.sep))
    if len(exp_subdirs) == 0:
        print('No experiment subdirectories provided.', file=sys.stderr)
        sys.exit(1)

    model_run_dirs = find_model_run_dirs(args.root_dir, exp_subdirs)
    sort_metric = None
    for metric_name in metrics:
        if 'acc' in metric_name.lower():
            sort_metric = metric_name
            break

    rows_with_keys: List[Tuple[float, Dict[str, str]]] = []

    for model_name, run_dirs in model_run_dirs.items():
        per_metric_lists: Dict[str, List[float]] = {m: [] for m in metrics}
        for run_dir in run_dirs:
            per_run = collect_fold_metrics(run_dir, metrics)
            for metric in metrics:
                per_metric_lists[metric].extend(per_run.get(metric, []))

        accuracy_sort_key = float('-inf')
        if sort_metric is not None:
            accuracy_values = per_metric_lists.get(sort_metric, [])
            if len(accuracy_values) > 0:
                accuracy_sort_key = float(np.mean(np.array(accuracy_values, dtype=float)))

        formatted = {
            metric: format_mean_std(values, args.decimals)
            for metric, values in per_metric_lists.items()
        }
        rows_with_keys.append((accuracy_sort_key, {'Model': model_name, **formatted}))

    headers = ['Model'] + metrics
    if sort_metric is not None:
        rows = [
            row for _, row in sorted(rows_with_keys, key=lambda item: item[0], reverse=True)
        ]
    else:
        rows = [row for _, row in rows_with_keys]

    table_str = build_table(rows, headers)
    print(table_str)


if __name__ == '__main__':
    main()

