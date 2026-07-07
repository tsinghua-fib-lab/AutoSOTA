"""
BMI (Benchmark for Mutual Information) evaluation.

Generates data on-the-fly via the `bmi` package (no pre-saved files needed).
All methods share the same generated samples for fair comparison.

Usage:
    # InfoAtlas only
    python -m evaluations.evaluate_bmi \
        --ckpt_path /path/to/last.ckpt

    # With baselines
    python -m evaluations.evaluate_bmi \
        --ckpt_path /path/to/last.ckpt \
        --methods InfoAtlas KSG MINE
"""
import os
import sys
import time
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bmi
import jax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from infer import load_ckpt, estimate_mi_batch
from evaluations._baseline_helper import AVAILABLE_BASELINES, estimate_mi_baseline, init_infonet_v1

CHOSEN_TASK_LIST = [
    "1v1-normal-0.75",
    "normal_cdf-1v1-normal-0.75",
    "1v1-additive-0.1",
    "1v1-additive-0.75",
    "wiggly-1v1-normal-0.75",
    "student-identity-1-1-1",
    "asinh-student-identity-1-1-1",
    "multinormal-dense-2-2-0.5",
    "multinormal-dense-3-3-0.5",
    "multinormal-dense-5-5-0.5",
    "multinormal-sparse-2-2-2-2.0",
    "multinormal-sparse-3-3-2-2.0",
    "multinormal-sparse-5-5-2-2.0",
    "student-identity-2-2-1",
    "student-identity-2-2-2",
    "student-identity-3-3-2",
    "student-identity-3-3-3",
    "student-identity-5-5-2",
    "student-identity-5-5-3",
    "normal_cdf-multinormal-sparse-3-3-2-2.0",
    "normal_cdf-multinormal-sparse-5-5-2-2.0",
    "half_cube-multinormal-sparse-3-3-2-2.0",
    "half_cube-multinormal-sparse-5-5-2-2.0",
    "spiral-multinormal-sparse-3-3-2-2.0",
    "spiral-multinormal-sparse-5-5-2-2.0",
    "spiral-normal_cdf-multinormal-sparse-3-3-2-2.0",
    "spiral-normal_cdf-multinormal-sparse-5-5-2-2.0",
    "asinh-student-identity-2-2-1",
    "asinh-student-identity-3-3-2",
    "asinh-student-identity-5-5-2",
]


# ============================================================
# Data generation helpers
# ============================================================

def _to_numpy(a) -> np.ndarray:
    """JAX DeviceArray -> numpy.ndarray"""
    return np.asarray(jax.device_get(a))


def _ensure_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a


def generate_bmi_data(
    task_list: List[str],
    sample_size: int,
    seeds: List[int],
) -> Dict[str, dict]:
    """
    Generate BMI task data on-the-fly.

    Returns:
        data_cache: {task_name: {"gt_mi": float, "samples": [(X, Y), ...]}}
            where each X, Y is np.ndarray of shape [sample_size, d].
    """
    data_cache = {}
    for task_name in task_list:
        if task_name not in bmi.benchmark.BENCHMARK_TASKS:
            print(f"[WARN] Task not found in bmi: {task_name}, skipping.")
            continue

        task = bmi.benchmark.BENCHMARK_TASKS[task_name]
        gt_mi = float(task.mutual_information)
        samples = []
        for seed in seeds:
            X, Y = task.sample(sample_size, seed=seed)
            X = _ensure_2d(_to_numpy(X)).astype(np.float32)
            Y = _ensure_2d(_to_numpy(Y)).astype(np.float32)
            samples.append((X, Y))

        data_cache[task_name] = {"gt_mi": gt_mi, "samples": samples}

    return data_cache


# ============================================================
# InfoAtlas evaluation
# ============================================================

def evaluate_bmi_infoatlas(
    model, max_dim, softrank_reg, data_cache,
    parallel_bs=10, gauss_copula=True,
):
    model.eval()
    results_dict = {}

    for task_name, task_data in data_cache.items():
        gt_mi = task_data["gt_mi"]
        samples = task_data["samples"]

        # Skip tasks whose native dimension exceeds this checkpoint's max_dim
        # (e.g. dim-5 tasks on a max_dim=3 model), which InfoAtlas cannot ingest.
        task_dim = samples[0][0].shape[1] if samples else 0
        if task_dim > max_dim:
            results_dict[task_name] = {"gt_mi": gt_mi, "mean_est": float("nan")}
            print(f"  [InfoAtlas] {task_name:<55s} skipped (dim {task_dim} > max_dim {max_dim})")
            continue

        task_estimates = []

        for start_idx in range(0, len(samples), parallel_bs):
            batch = samples[start_idx:start_idx + parallel_bs]
            x_list = [torch.from_numpy(X).float().unsqueeze(0) for X, Y in batch]
            y_list = [torch.from_numpy(Y).float().unsqueeze(0) for X, Y in batch]

            mi_batch = estimate_mi_batch(
                torch.cat(x_list, 0), torch.cat(y_list, 0),
                model=model, max_dim=max_dim, softrank_reg=softrank_reg,
                gauss_copula=gauss_copula,
            )
            task_estimates.extend(mi_batch.numpy().tolist())

        mean_est = float(np.mean(task_estimates))
        results_dict[task_name] = {"gt_mi": gt_mi, "mean_est": mean_est}
        print(f"  [InfoAtlas] {task_name:<55s} GT={gt_mi:>10.6f} | Est={mean_est:>10.6f}")

    return results_dict


# ============================================================
# Baseline evaluation
# ============================================================

def evaluate_bmi_baseline(method_name, data_cache, device):
    results_dict = {}

    for task_name, task_data in data_cache.items():
        gt_mi = task_data["gt_mi"]
        samples = task_data["samples"]
        task_estimates = []

        for X, Y in samples:
            mi = estimate_mi_baseline(method_name, X, Y, device=device)
            task_estimates.append(float(mi))

        mean_est = float(np.mean(task_estimates))
        results_dict[task_name] = {"gt_mi": gt_mi, "mean_est": mean_est}
        print(f"  [{method_name}] {task_name:<55s} GT={gt_mi:>10.6f} | Est={mean_est:>10.6f}")

    return results_dict


# ============================================================
# Main
# ============================================================

def run_bmi_evaluation(args):
    output_dir = os.path.join(args.output_dir, "bmi")
    os.makedirs(output_dir, exist_ok=True)

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    seeds = list(range(42, 42 + args.repeats))

    # ---- Generate data once for all methods ----
    print("=" * 100)
    print(f"[BMI EVAL] Generating data: sample_size={args.sample_size}, repeats={args.repeats}, seeds={seeds}")
    print("=" * 100)
    t0 = time.perf_counter()
    data_cache = generate_bmi_data(CHOSEN_TASK_LIST, args.sample_size, seeds)
    print(f"[BMI EVAL] Data generation finished in {time.perf_counter() - t0:.2f}s")

    # Load InfoNet V1 model if needed
    if "InfoNet" in args.methods:
        init_infonet_v1(args.infonet_config_path, args.infonet_ckpt_path, device=dev)

    # Load InfoAtlas model if needed
    model, max_dim, softrank_reg, gauss_copula = None, None, None, True
    if "InfoAtlas" in args.methods:
        module_wrapper, model, cfg, device = load_ckpt(
            ckpt_path=args.ckpt_path, cfg_path=args.cfg_path, device=dev, verbose=True,
        )
        max_dim = int(cfg.input_dim_x)
        softrank_reg = float(cfg.softrank_reg)
        gauss_copula = bool(cfg.gauss_copula) if hasattr(cfg, "gauss_copula") else True

    # Run evaluation for each method
    all_method_results: Dict[str, Dict] = {}

    for method in args.methods:
        print("=" * 100)
        print(f"[BMI EVAL] Running method: {method}")
        print("=" * 100)
        t0 = time.perf_counter()

        if method == "InfoAtlas":
            results = evaluate_bmi_infoatlas(
                model=model, max_dim=max_dim, softrank_reg=softrank_reg,
                data_cache=data_cache, parallel_bs=args.parallel_bs,
                gauss_copula=gauss_copula,
            )
        else:
            results = evaluate_bmi_baseline(
                method_name=method, data_cache=data_cache, device=dev,
            )

        elapsed = time.perf_counter() - t0
        all_method_results[method] = results
        print(f"[BMI EVAL] {method} finished in {elapsed:.2f}s")

    # ---- Build result table ----
    rows = []
    for task_name in CHOSEN_TASK_LIST:
        if task_name not in data_cache:
            continue
        gt_mi = None
        row = {"task_name": task_name}
        for method in args.methods:
            r = all_method_results[method][task_name]
            if gt_mi is None:
                gt_mi = r["gt_mi"]
            row["gt_mi"] = gt_mi
            row[f"{method}_est"] = r["mean_est"]
            row[f"{method}_abs_bias"] = abs(r["mean_est"] - gt_mi)
            if abs(gt_mi) > 1e-12:
                row[f"{method}_bias_rate"] = abs(r["mean_est"] - gt_mi) / abs(gt_mi)
            else:
                row[f"{method}_bias_rate"] = float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)

    # ---- Print summary ----
    print("\n" + "=" * 100)
    for method in args.methods:
        gt = df["gt_mi"].values
        est = df[f"{method}_est"].values
        mean_abs_bias = df[f"{method}_abs_bias"].mean()
        valid = np.abs(gt) > 1e-12
        mean_bias_rate = np.mean(np.abs(est[valid] - gt[valid]) / np.abs(gt[valid]))
        print(f"[BMI EVAL] {method:<15s} mean_abs_bias = {mean_abs_bias:.8f}, mean_bias_rate = {mean_bias_rate:.6f}")
    print("=" * 100)

    # ---- Save CSV ----
    csv_path = os.path.join(output_dir, "bmi_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[BMI EVAL] CSV saved to {csv_path}")

    # ---- Save plot ----
    if len(args.methods) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Abs Bias per task
        x_pos = np.arange(len(df))
        width = 0.8 / max(len(args.methods), 1)
        for i, method in enumerate(args.methods):
            biases = df[f"{method}_abs_bias"].values
            axes[0].bar(x_pos + i * width, biases, width, label=method, alpha=0.8)
        axes[0].set_xticks(x_pos + width * (len(args.methods) - 1) / 2)
        axes[0].set_xticklabels([t[:20] + "..." if len(t) > 20 else t for t in df["task_name"]],
                                rotation=90, fontsize=6)
        axes[0].set_ylabel("Absolute Bias")
        axes[0].set_title("BMI: Absolute Bias per Task")
        axes[0].legend(fontsize=8)

        # Plot 2: Summary bar chart (mean abs bias & mean bias rate)
        methods = args.methods
        mean_abs_biases = [df[f"{m}_abs_bias"].mean() for m in methods]
        mean_bias_rates = [df[f"{m}_bias_rate"].mean() for m in methods]

        x = np.arange(len(methods))
        w = 0.35
        axes[1].bar(x - w / 2, mean_abs_biases, w, label="Mean Abs Bias", alpha=0.8)
        axes[1].bar(x + w / 2, mean_bias_rates, w, label="Mean Bias Rate", alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(methods, rotation=45, ha="right")
        axes[1].set_title("BMI: Summary Metrics")
        axes[1].legend()

        plt.tight_layout()
        png_path = os.path.join(output_dir, "bmi_results.png")
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[BMI EVAL] Plot saved to {png_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMI Evaluation")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint .ckpt file (required for InfoAtlas)")
    parser.add_argument("--cfg_path", type=str, default=None, help="Path to config .yaml file (optional)")
    parser.add_argument("--methods", type=str, nargs="+", default=["InfoAtlas"],
                        help=f"Methods to evaluate. Available: InfoAtlas, {', '.join(AVAILABLE_BASELINES)}")
    parser.add_argument("--output_dir", type=str, default="results", help="Root output directory")
    parser.add_argument("--parallel_bs", type=int, default=200,
                        help="Batch size for InfoAtlas parallel inference (number of XY pairs per batch)")
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--infonet_config_path", type=str, default=None,
                        help="Path to InfoNet V1 config yaml (required if InfoNet in --methods)")
    parser.add_argument("--infonet_ckpt_path", type=str, default=None,
                        help="Path to InfoNet V1 checkpoint (required if InfoNet in --methods)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if "InfoAtlas" in args.methods and args.ckpt_path is None:
        parser.error("--ckpt_path is required when InfoAtlas is in --methods")
    if "InfoNet" in args.methods and (args.infonet_config_path is None or args.infonet_ckpt_path is None):
        parser.error("--infonet_config_path and --infonet_ckpt_path are required when InfoNet is in --methods")

    run_bmi_evaluation(args)
