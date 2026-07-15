#!/usr/bin/env python3
"""
Run BERW-Hetero on a small bbob-noisy fixed-budget slice and save *internal residual-pool diagnostics*.

This complements the Hansen fixed-budget "money plot" evidence by making the theory's assumptions
measurable on real runs:
  - pool size
  - z clipping saturation
  - within-generation shape-shift proxy (bucketed KS on standardized residuals)
  - drift proxy (KS vs previous-generation residuals)
  - scale model fit R^2
  - split-median centering stability (scale-normalized)

Outputs:
  - state_index.csv + per-run state traces
  - diagnostics_summary.csv
  - diagnostics_summary.png
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import cocoex
import matplotlib.pyplot as plt
import numpy as np

from _project import BASE_DIR, repo_relpath
from algorithms.berw_es import my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero as berw_hetero


class ProblemWrapper:
    """
    Wrap a COCO problem so we can attach Python-side trace attributes.

    `cocoex.Problem` objects are implemented with restricted attributes; they do not allow
    setting arbitrary fields. This wrapper delegates everything via `__getattr__` while
    keeping a normal Python object surface for instrumentation.
    """

    def __init__(self, problem):
        self._p = problem

    def __call__(self, x):
        return float(self._p(x))

    def __getattr__(self, name):
        return getattr(self._p, name)


def parse_int_list(spec: str) -> list[int]:
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a_i = int(a.strip())
            b_i = int(b.strip())
            if b_i < a_i:
                a_i, b_i = b_i, a_i
            out.extend(range(a_i, b_i + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def format_filter(dims: list[int], funcs: list[int], instances: list[int]) -> str:
    dims_s = ",".join(str(d) for d in dims)
    funcs_s = ",".join(str(f) for f in funcs)
    inst_s = ",".join(str(i) for i in instances)
    return f"dimensions:{dims_s} function_indices:{funcs_s} instance_indices:{inst_s}"


STATE_HEADER = [
    "evals",
    "generation",
    "noise_level",
    "noise_ema",
    "temp_scale",
    "n_sched",
    "n_eff",
    "reeval_count",
    "gate_closed",
    "mueff",
    "mueff_target",
    "noise_s0",
    "noise_s1",
    "noise_z_pool_size",
    "noise_z_abs_median",
    "noise_z_clip_frac",
    "noise_shape_ks",
    "noise_shape_w1",
    "noise_drift_ks",
    "noise_drift_w1",
    "noise_scale_fit_r2",
    "noise_scale_pred_cv",
    "noise_center_split_rel",
    "noise_center_split_cv",
]


def write_csv(path: str, header: list[str], rows: list[tuple]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def summarize_state_trace(state_rows: list[tuple]) -> dict[str, float]:
    if not state_rows:
        return {}

    idx = {name: i for i, name in enumerate(STATE_HEADER)}

    def col(name: str) -> np.ndarray:
        j = idx[name]
        arr = np.array([float(r[j]) for r in state_rows], dtype=float)
        return arr

    out: dict[str, float] = {}
    for key in [
        "noise_z_pool_size",
        "noise_z_clip_frac",
        "noise_shape_ks",
        "noise_shape_w1",
        "noise_drift_ks",
        "noise_drift_w1",
        "noise_scale_fit_r2",
        "noise_scale_pred_cv",
        "noise_center_split_rel",
        "noise_center_split_cv",
    ]:
        series = col(key)
        finite = series[np.isfinite(series)]
        out[f"final_{key}"] = float(finite[-1]) if finite.size > 0 else float("nan")
        out[f"mean_{key}"] = float(np.mean(finite)) if finite.size > 0 else float("nan")
        out[f"max_{key}"] = float(np.max(finite)) if finite.size > 0 else float("nan")
    return out


def plot_summary(summary_rows: list[dict[str, str]], out_path: str) -> None:
    if not summary_rows:
        return

    def as_float_list(key: str) -> list[float]:
        out = []
        for r in summary_rows:
            try:
                out.append(float(r.get(key, "nan")))
            except Exception:
                out.append(float("nan"))
        return out

    funcs = [int(float(r["function_index"])) for r in summary_rows]
    uniq_funcs = sorted(set(funcs))

    metrics = [
        ("final_noise_z_pool_size", "final pool size"),
        ("mean_noise_z_clip_frac", "mean clip frac"),
        ("mean_noise_shape_w1", "mean shape W1"),
        ("mean_noise_drift_w1", "mean drift W1"),
        ("mean_noise_scale_fit_r2", "mean scale R2"),
        ("mean_noise_scale_pred_cv", "mean scale CV"),
        ("mean_noise_center_split_rel", "mean center split rel"),
        ("mean_noise_center_split_cv", "mean center split CV"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.6), dpi=180)
    axes = axes.reshape(-1)

    for ax, (key, title) in zip(axes, metrics):
        data_by_func = []
        labels = []
        for fidx in uniq_funcs:
            vals = []
            for r in summary_rows:
                if int(float(r["function_index"])) != int(fidx):
                    continue
                try:
                    v = float(r.get(key, "nan"))
                except Exception:
                    v = float("nan")
                if np.isfinite(v):
                    vals.append(v)
            if vals:
                data_by_func.append(vals)
                labels.append(str(fidx))
        if not data_by_func:
            ax.axis("off")
            continue
        try:
            ax.boxplot(data_by_func, tick_labels=labels, showfliers=False)
        except TypeError:
            ax.boxplot(data_by_func, labels=labels, showfliers=False)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=0, labelsize=8)

    fig.suptitle("Residual-pool diagnostics (BERW-Hetero, bbob-noisy fixed budget)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=os.path.join(BASE_DIR, "evidence", "hansen_test_fixed_budget", "diagnostics"))
    parser.add_argument("--dims", default="40")
    parser.add_argument(
        "--functions",
        default="8,10,11,13,14,16,17,19,20,22,23,25,26,28,29",
        help="bbob-noisy function indices (1-30). Default matches the Hansen fixed-budget high-misranking slice.",
    )
    parser.add_argument("--instances", default="1-5")
    parser.add_argument("--budget-mult", type=int, default=100, help="Budget multiplier (xD).")
    args = parser.parse_args()

    dims = parse_int_list(args.dims)
    funcs = parse_int_list(args.functions)
    inst = parse_int_list(args.instances)
    suite_filter = format_filter(dims, funcs, inst)

    out_dir = os.path.abspath(args.out_dir)
    traces_dir = os.path.join(out_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)

    state_index_path = os.path.join(out_dir, "state_index.csv")
    summary_path = os.path.join(out_dir, "diagnostics_summary.csv")

    state_index_rows = []
    summary_rows: list[dict[str, str]] = []

    suite = cocoex.Suite("bbob-noisy", "", suite_filter)
    total = len(suite)
    count = 0

    for problem in suite:
        count += 1
        budget = int(args.budget_mult) * int(problem.dimension)

        state_trace: list[tuple] = []
        wrapped = ProblemWrapper(problem)
        setattr(wrapped, "_berw_state_trace", state_trace)

        berw_hetero(wrapped, budget)

        algo = "BERW-Hetero"
        state_id = f"berw_hetero_B{args.budget_mult}_f{int(problem.id_function)}_d{int(problem.dimension)}_i{int(problem.id_instance)}"
        state_file = os.path.join(traces_dir, f"{state_id}.csv")
        write_csv(state_file, STATE_HEADER, state_trace)
        state_index_rows.append(
            {
                "state_id": state_id,
                "algorithm": algo,
                "budget_multiplier": str(int(args.budget_mult)),
                "function": str(int(problem.id_function)),
                "function_index": str(int(problem.id_function) - 100),
                "dimension": str(int(problem.dimension)),
                "instance": str(int(problem.id_instance)),
                "state_file": state_file,
            }
        )

        stats = summarize_state_trace(state_trace)
        row = {
            "algorithm": algo,
            "budget_multiplier": str(int(args.budget_mult)),
            "function": str(int(problem.id_function)),
            "function_index": str(int(problem.id_function) - 100),
            "dimension": str(int(problem.dimension)),
            "instance": str(int(problem.id_instance)),
            **{k: f"{v:.12g}" for k, v in stats.items()},
        }
        summary_rows.append(row)

        if count % 5 == 0 or count == total:
            print(f"[{count:3d}/{total}] f{int(problem.id_function):03d} d{int(problem.dimension):02d} i{int(problem.id_instance):02d}")

    # Write state index
    with open(state_index_path, "w", newline="") as f:
        fieldnames = list(state_index_rows[0].keys()) if state_index_rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(state_index_rows)

    with open(summary_path, "w", newline="") as f:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    plot_path = os.path.join(out_dir, "diagnostics_summary.png")
    plot_summary(summary_rows, plot_path)

    print("Wrote:", repo_relpath(state_index_path))
    print("Wrote:", repo_relpath(summary_path))
    print("Wrote:", repo_relpath(plot_path))


if __name__ == "__main__":
    main()
