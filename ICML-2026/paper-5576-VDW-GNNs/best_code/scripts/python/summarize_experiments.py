#!/usr/bin/env python3

"""
Summarize experiment results (CV or single runs) into a LaTeX table.

Inputs: one or more experiment directories. Each experiment dir is expected to
have subdirectories per model and, for CV, subdirectories per fold as 'fold_i'.

For ellipsoid experiments, we extract from logs/train_logs.pkl the best
validation MSE and its epoch, and compute mean train and inference (validation)
times from epoch 1 through best epoch. From metrics/results.pkl we extract
parameter counts and test metrics. For CV, aggregate metrics across folds and
report mean $\\pm$ std (default) or median [min, max] if --agg=median is used.

Example call:
python3 code/scripts/python/summarize_experiments.py \
  --exp_dirs experiments/ellipsoids_cv/ellipsoids/normals \
  --metrics mse \
  --agg median \
  --decimals 4 \
  --out_tex experiments/ellipsoids_cv/ellipsoids/ellipsoids_cv_test_median_metrics.tex \
  --out_csv experiments/ellipsoids_cv/ellipsoids/ellipsoids_cv_test_median_metrics.csv
"""

import os
import sys
import argparse
import pickle
from typing import List, Dict, Any, Tuple, Literal
import numpy as np
import pandas as pd

# Likely want to exclude epoch 0 by default to avoid influence of initialization
EXCLUDE_EPOCH_0 = True

# Metric comparison categories (for bolding best test metrics)
LOWER_IS_BETTER = {'mse', 'mae', 'rmse', 'loss'}
HIGHER_IS_BETTER = {
    'accuracy', 'acc', 'f1', 'precision', 'recall', 'sensitivity',
    'specificity', 'auroc', 'roc_auc', 'auc', 'r2'
}


def latex_process_underscores(
    text: str,
    mode: Literal['escape', 'replace'] = 'escape',
) -> str:
    """
    Process underscores in a string for LaTeX:
      - mode == 'escape': replace '_' with '\_'
      - mode == 'replace': replace '_' with space
    """
    if mode == 'escape':
        return str(text).replace('_', r"\_")
    elif mode == 'replace':
        return str(text).replace('_', ' ')
    else:
        raise ValueError(f"Invalid mode: {mode}")


def primary_number_for_comparison(
    value: Any,
) -> float:
    """
    Extract the primary numeric value from a string-formatted metric for comparison.
    Handles formats like:
      - "12.34" (single number)
      - "12.34 $\\pm$ 0.56" or "12.34 ± 0.56" (use the first number)
      - "12.34 [10.00, 15.00]" (use the first number)
    Returns NaN if parsing fails.
    """
    try:
        if isinstance(value, (int, float, np.floating)):
            return float(value)
        s = str(value).strip()
        if s == "":
            return float('nan')
        # Normalize pm symbol variants
        s_norm = s.replace('$\\pm$', '±')
        if '±' in s_norm:
            s_norm = s_norm.split('±', 1)[0].strip()
        # Median format: "median [min, max]"
        if ' [' in s_norm:
            s_norm = s_norm.split(' [', 1)[0].strip()
        return float(s_norm)
    except Exception:
        return float('nan')


def normalize_math_text(text: Any) -> str:
    """
    Prepare a cell's text for math mode by normalizing plus/minus and removing
    nested math markers. Replaces "$\\pm$" or the unicode '±' with "\\pm".
    """
    s = str(text)
    s = s.replace('$\\pm$', '\\pm')
    s = s.replace('±', '\\pm')
    return s


def make_math_cell(
    text: Any,
    bold: bool = False,
) -> str:
    """
    Wrap a cell value in LaTeX math mode. If bold is True, wrap content in \\mathbf{...}.
    """
    s = normalize_math_text(text)
    if bold:
        return f"$\\mathbf{{{s}}}$"
    return f"${s}$"


def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)


def is_cv_run(model_dir: str) -> bool:
    try:
        entries = os.listdir(model_dir)
    except FileNotFoundError:
        return False
    return any(
        name.startswith('fold_') \
        and os.path.isdir(os.path.join(model_dir, name)) \
            for name in entries
    )


def _to_float_safe(x: Any) -> float:
    try:
        arr = np.array(x)
        if arr.size == 1:
            return float(arr.squeeze())
        # fallback: try float directly
        return float(x)
    except Exception:
        return x  # return as-is


def collect_single_run_metrics(
    run_dir: str, 
    metrics: List[str]
) -> Dict[str, Any]:
    metrics_dir = os.path.join(run_dir, 'metrics')
    logs_dir = os.path.join(run_dir, 'logs')
    results_pkl = os.path.join(metrics_dir, 'results.pkl')
    train_logs_pkl = os.path.join(logs_dir, 'train_logs.pkl')

    out: Dict[str, Any] = {}

    # Results: includes param counts and requested test metrics only
    if os.path.exists(results_pkl):
        res = load_pickle(results_pkl)
        # Only keep trainable parameter count for the table
        out['num_params_trainable'] = res.get('num_params_trainable')
        # Normalize requested metrics for test (strip any _valid suffix)
        req_test_metrics = [(m[:-6] if m.endswith('_valid') else m) for m in metrics]
        for m in req_test_metrics:
            if m in res:
                out[f"test_{m}"] = _to_float_safe(res[m])

    # Training logs: best requested validation metrics, best epoch, mean times until best
    if os.path.exists(train_logs_pkl):
        logs = load_pickle(train_logs_pkl)
        # Expect list of dicts per epoch
        df = pd.DataFrame(logs)
        # Requested metric columns (append _valid suffix if needed)
        req_cols = [(m if m.endswith('_valid') else f"{m}_valid") for m in metrics]
        # Pick first available requested metric for best-epoch selection (fallback to mse_valid/loss_valid)
        metric_col = None
        for col in req_cols:
            if col in df.columns and not df[col].isna().all():
                metric_col = col
                break
        if metric_col is None:
            if 'mse_valid' in df.columns and not df['mse_valid'].isna().all():
                metric_col = 'mse_valid'
            elif 'loss_valid' in df.columns and not df['loss_valid'].isna().all():
                metric_col = 'loss_valid'

        if metric_col is not None:
            best_idx = df[metric_col].idxmin()
            best_row = df.loc[best_idx]
            # Record best value for each requested metric that exists
            for col in req_cols:
                if col in df.columns and not df[col].isna().all():
                    base = col[:-6] if col.endswith('_valid') else col
                    out[f'best_val_{base}'] = _to_float_safe(best_row[col])
            out['best_epoch'] = int(best_row.get('epoch', best_idx + 1))

            # Compute means from epoch 1 through best epoch (exclude epoch 0 by default)
            start_epoch = 1 if EXCLUDE_EPOCH_0 else 0
            mask = (df['epoch'] >= start_epoch) & (df['epoch'] <= out['best_epoch']) if 'epoch' in df.columns else df.index <= best_idx
            # Training time per epoch (sec)
            if 'train_time_sec' in df.columns:
                out['mean_train_time_sec'] = _to_float_safe(df.loc[mask, 'train_time_sec'].mean())
            # Validation (inference) time per validation epoch
            if 'valid_inference_time_sec' in df.columns:
                # Use nonzero entries to avoid counting non-validation epochs as zero
                valid_times = df.loc[mask, 'valid_inference_time_sec']
                out['mean_valid_inference_time_sec'] = _to_float_safe(valid_times[valid_times > 0].mean()) if (valid_times > 0).any() else 0.0

    return out


def collect_model_metrics(
    model_dir: str,
    agg: str = 'mean', 
    metrics: List[str] = None, 
    decimals: int = 4
) -> Dict[str, Any]:
    model_name = os.path.basename(model_dir)
    if metrics is None:
        metrics = ['mse']
    if is_cv_run(model_dir):
        fold_dirs = sorted([os.path.join(model_dir, d) for d in os.listdir(model_dir) if d.startswith('fold_')])
        per_fold = [collect_single_run_metrics(fd, metrics) for fd in fold_dirs]
        # Aggregate numeric keys
        keys = sorted({k for d in per_fold for k in d.keys()})
        agg_out: Dict[str, Any] = {'model': model_name}
        # Special-case parameter count: if identical across folds, show single number; else aggregate as below
        if 'num_params_trainable' in keys:
            pvals = [d.get('num_params_trainable') for d in per_fold if d.get('num_params_trainable') is not None]
            if len(pvals) > 0:
                if len(set(pvals)) == 1:
                    agg_out['num_params_trainable'] = int(pvals[0])
                else:
                    p_arr = np.array(pvals, dtype=float)
                    if agg == 'median':
                        agg_out['num_params_trainable'] = f"{np.median(p_arr):.{decimals}f} [{np.min(p_arr):.{decimals}f}, {np.max(p_arr):.{decimals}f}]"
                    else:
                        std_val = np.std(p_arr, ddof=1) if len(p_arr) > 1 else 0.0
                        agg_out['num_params_trainable'] = f"{np.mean(p_arr):.{decimals}f} $\\pm$ {std_val:.{decimals}f}"
        # Aggregate remaining numeric keys (skip param-total and skip repeated param-trainable if set already)
        for k in keys:
            if k in ('num_params_total', 'num_params_trainable'):
                continue
            vals = [d[k] for d in per_fold if k in d and d[k] is not None]
            if len(vals) == 0:
                continue
            vals_arr = np.array(vals, dtype=float)
            # Special-case best_epoch formatting
            if k == 'best_epoch':
                if agg == 'median':
                    agg_out[k] = f"{int(np.median(vals_arr))} [{int(np.min(vals_arr))}, {int(np.max(vals_arr))}]"
                else:
                    mean_epoch_int = int(np.rint(np.mean(vals_arr)))
                    std_val = np.std(vals_arr, ddof=1) if len(vals_arr) > 1 else 0.0
                    std_epoch_int = int(np.rint(std_val))
                    agg_out[k] = f"{mean_epoch_int} $\\pm$ {std_epoch_int}"
                continue

            # Weighted aggregation for time columns across folds (weight by number of epochs included)
            if k in ('mean_train_time_sec', 'mean_valid_inference_time_sec') and agg != 'median':
                weighted_values = []
                weights = []
                for d in per_fold:
                    if k in d and d[k] is not None \
                    and ('best_epoch' in d and d['best_epoch'] is not None):
                        v = float(d[k])
                        best_ep = int(d['best_epoch'])
                        start_ep = 1 if EXCLUDE_EPOCH_0 else 0
                        n_epochs = max(1, best_ep - start_ep)
                        weighted_values.append(v)
                        weights.append(n_epochs)
                if len(weights) > 0 and sum(weights) > 0:
                    w = np.array(weights, dtype=float)
                    x = np.array(weighted_values, dtype=float)
                    w_mean = np.average(x, weights=w)
                    # population-style weighted variance
                    w_var = np.average((x - w_mean) ** 2, weights=w)
                    w_std = float(np.sqrt(w_var))
                    agg_out[k] = f"{w_mean:.{decimals}f} $\\pm$ {w_std:.{decimals}f}"
                    continue

            if agg == 'median':
                agg_out[k] = f"{np.median(vals_arr):.{decimals}f} [{np.min(vals_arr):.{decimals}f}, {np.max(vals_arr):.{decimals}f}]"
            else:
                std_val = np.std(vals_arr, ddof=1) if len(vals_arr) > 1 else 0.0
                agg_out[k] = f"{np.mean(vals_arr):.{decimals}f} $\\pm$ {std_val:.{decimals}f}"
        return agg_out
    else:
        out = collect_single_run_metrics(model_dir, metrics)
        out['model'] = model_name
        # Format numeric values to requested decimals
        for k, v in list(out.items()):
            if k in ('model', 'num_params_trainable', 'best_epoch'):
                continue
            if isinstance(v, (int, float, np.floating)):
                out[k] = f"{float(v):.{decimals}f}"
        return out


def gather_experiments(
    exp_dirs: List[str], 
    agg: str = 'mean', 
    metrics: List[str] = None, 
    decimals: int = 4
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for exp_dir in exp_dirs:
        if not os.path.isdir(exp_dir):
            continue
        entries = [os.path.join(exp_dir, d) for d in os.listdir(exp_dir)]
        # If this directory itself looks like a model dir (contains folds or logs/metrics), treat it as a single model
        if is_cv_run(exp_dir) or os.path.isdir(os.path.join(exp_dir, 'logs')) or os.path.isdir(os.path.join(exp_dir, 'metrics')):
            rows.append(
                collect_model_metrics(
                    exp_dir,
                    agg=agg,
                    metrics=metrics,
                    decimals=decimals,
                )
            )
            continue
        # Otherwise, iterate immediate subdirectories as candidate model dirs
        for entry in entries:
            if not os.path.isdir(entry):
                continue
            rows.append(
                collect_model_metrics(
                    entry,
                    agg=agg,
                    metrics=metrics,
                    decimals=decimals,
                )
            )
    if len(rows) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Order columns with key metrics first if present
    preferred = ['model', 'best_epoch', 'mean_train_time_sec', 'mean_valid_inference_time_sec', 'num_params_trainable']
    if metrics:
        val_cols = []
        test_cols = []
        for m in metrics:
            base = m if not m.endswith('_valid') else m[:-6]
            val_cols.append(f'best_val_{base}')
            test_cols.append(f'test_{base}')
        preferred = ['model'] + val_cols + test_cols + ['best_epoch', 'mean_train_time_sec', 'mean_valid_inference_time_sec', 'num_params_trainable']
    others = [c for c in df.columns if c not in preferred]
    df = df[preferred + others]
    return df


def main():
    parser = argparse.ArgumentParser(description='Summarize experiment results to LaTeX table')
    parser.add_argument('--exp_dirs', nargs='+', required=True, help='One or more experiment directories')
    parser.add_argument('--agg', type=str, choices=['mean', 'median'], default='mean', help='Aggregation for CV runs')
    parser.add_argument('--metrics', nargs='+', default=['mse'], help='Metrics to extract: will read validation from train_logs and test from results (e.g., mse mae). Use names without _valid suffix')
    parser.add_argument('--decimals', type=int, default=4, help='Number of decimal places to round numeric outputs')
    parser.add_argument('--out_csv', type=str, default=None, help='Optional CSV output path')
    parser.add_argument('--out_tex', type=str, default=None, help='Optional LaTeX table output path')
    parser.add_argument('--bold_best', action='store_true', help='Bold best test metric scores across models')
    args = parser.parse_args()

    df = gather_experiments(
        args.exp_dirs,
        agg=args.agg,
        metrics=args.metrics,
        decimals=args.decimals,
    )
    if df.empty:
        print('No results found.')
        return

    print(df)
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        df.to_csv(args.out_csv, index=False)
    if args.out_tex:
        # Sort rows by model name
        df_sorted = df.sort_values(by='model') \
            if 'model' in df.columns else df.copy()

        # Split columns into two tables
        primary_cols_core = [c for c in ['best_val_mse', 'test_mse', 'num_params_trainable'] if c in df_sorted.columns]
        first_cols = ['model'] + primary_cols_core if 'model' in df_sorted.columns else primary_cols_core
        remaining_cols = [c for c in df_sorted.columns if c not in primary_cols_core and c != 'model']
        second_cols = (['model'] + remaining_cols) if 'model' in df_sorted.columns else remaining_cols

        def df_to_latex_table(df_in: pd.DataFrame) -> str:
            latex_df = df_in.copy()
            # Determine best rows per test column for optional bolding
            test_cols = [c for c in latex_df.columns if c.startswith('test_')]
            best_masks: Dict[str, np.ndarray] = {}
            for col in test_cols:
                metric_name = col[5:]
                numeric = np.array(
                    [primary_number_for_comparison(v) for v in latex_df[col].tolist()],
                    dtype=float,
                )
                if np.all(np.isnan(numeric)):
                    continue
                if (metric_name in LOWER_IS_BETTER) or any(k in metric_name for k in ['mse', 'mae', 'rmse', 'error', 'loss']):
                    best_val = np.nanmin(numeric)
                else:
                    best_val = np.nanmax(numeric)
                best_masks[col] = np.isclose(numeric, best_val, equal_nan=False)

            # Ensure non-model columns can accept string (object) assignments for math formatting
            for col in latex_df.columns:
                if col != 'model':
                    latex_df[col] = latex_df[col].astype('object')

            # Convert all non-model columns to math mode; apply bold where requested
            for row_idx in range(len(latex_df)):
                for col in latex_df.columns:
                    if col == 'model':
                        continue
                    bold = args.bold_best and (col in best_masks) and bool(best_masks[col][row_idx])
                    latex_df.iat[row_idx, latex_df.columns.get_loc(col)] = make_math_cell(latex_df.iat[row_idx, latex_df.columns.get_loc(col)], bold=bold)
            # Escape underscores in headers and model names for LaTeX rendering
            if 'model' in latex_df.columns:
                latex_df['model'] = latex_df['model'].apply(latex_process_underscores)
            latex_df.columns = [
                latex_process_underscores(str(col)) \
                    for col in latex_df.columns
                ]
            # Set LaTeX column alignment: first column left-aligned, rest right-aligned
            column_format = 'l' + 'r' * (len(latex_df.columns) - 1)
            return latex_df.to_latex(index=False, escape=False, column_format=column_format)

        # Build first table (validation MSE, test MSE, params)
        df_first = df_sorted[first_cols] if len(first_cols) > 0 else pd.DataFrame()
        latex_str_1 = df_to_latex_table(df_first)
        if args.agg == 'median':
            latex_str_1 += "\\caption{Results shown are median [min, max] metric scores across cross-validation folds.}"
        else:
            latex_str_1 += "\\caption{Results shown are mean $\\pm$ standard deviation of scores across cross-validation folds.}"
        latex_str_1 = "\\begin{table}[h!]\n" + latex_str_1 + "\n\\end{table}"

        # Build second table (remaining columns)
        df_second = df_sorted[second_cols] if len(second_cols) > 0 else pd.DataFrame()
        latex_str_2 = df_to_latex_table(df_second)
        if args.agg == 'median':
            latex_str_2 += "\\caption{Results shown are median [min, max] of scores across cross-validation folds.}"
        else:
            latex_str_2 += "\\caption{Results shown are mean $\\pm$ standard deviation metric scores across cross-validation folds.}"
        latex_str_2 = "\\begin{table}[h!]\n" + latex_str_2 + "\n\\end{table}"

        latex_str = latex_str_1 + "\n\n" + latex_str_2

        os.makedirs(os.path.dirname(args.out_tex), exist_ok=True)
        with open(args.out_tex, 'w') as f:
            f.write(latex_str)


if __name__ == '__main__':
    main()

