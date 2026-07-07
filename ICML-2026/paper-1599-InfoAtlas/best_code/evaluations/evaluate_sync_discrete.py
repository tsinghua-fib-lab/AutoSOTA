"""
Discrete sync evaluation.

Usage:
    python -m evaluations.evaluate_sync_discrete \
        --ckpt_path /path/to/last.ckpt \
        --methods InfoAtlas KSG MINE
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from infer import load_ckpt, estimate_mi_batch
from evaluations._baseline_helper import AVAILABLE_BASELINES, estimate_mi_baseline, init_infonet_v1

EPS = 1e-12


# ============================================================
# Ground-truth MI for discrete distributions
# ============================================================

def gt_mi_from_joint_pmf(pxy):
    pxy = np.asarray(pxy, dtype=np.float64)
    pxy = pxy / pxy.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denom = np.clip(px @ py, EPS, None)
    ratio = pxy / denom
    mask = pxy > 0
    return float(np.sum(pxy[mask] * np.log(np.clip(ratio[mask], EPS, None))))

def build_kary_symmetric_joint_pmf(k, q):
    pxy = np.full((k, k), (1.0 - q) / (k * (k - 1)), dtype=np.float64)
    np.fill_diagonal(pxy, q / k)
    return pxy

def gt_mi_kary_symmetric(k, q):
    term0 = np.log(k)
    term1 = 0.0 if q == 0 else q * np.log(q)
    term2 = 0.0 if q == 1 else (1.0 - q) * np.log((1.0 - q) / (k - 1))
    return float(term0 + term1 + term2)


# ============================================================
# Sampling
# ============================================================

def sample_from_joint_pmf_1d(n, pxy, rng):
    pxy = np.asarray(pxy, dtype=np.float64)
    pxy = pxy / pxy.sum()
    nx, ny = pxy.shape
    flat = pxy.reshape(-1)
    idx = rng.choice(flat.size, size=n, replace=True, p=flat)
    return (idx // ny).astype(np.float32), (idx % ny).astype(np.float32)

def sample_categorical_independent_dims(n, dim, pxy_1d, rng):
    xs, ys = [], []
    for _ in range(dim):
        x1, y1 = sample_from_joint_pmf_1d(n=n, pxy=pxy_1d, rng=rng)
        xs.append(x1[:, None])
        ys.append(y1[:, None])
    return np.concatenate(xs, axis=1).astype(np.float32), np.concatenate(ys, axis=1).astype(np.float32)

# ============================================================
# Task builder
# ============================================================

def build_sync_tasks():
    tasks = []
    categorical_specs = [
        {"task_name": "categorical-k3-q0.45-1d", "k": 3, "q": 0.45, "dim": 1},
        {"task_name": "categorical-k5-q0.42-1d", "k": 5, "q": 0.42, "dim": 1},
        {"task_name": "categorical-k8-q0.35-1d", "k": 8, "q": 0.35, "dim": 1},
        {"task_name": "categorical-k5-q0.44-5d", "k": 5, "q": 0.44, "dim": 5},
    ]
    for spec in categorical_specs:
        pxy_1d = build_kary_symmetric_joint_pmf(k=spec["k"], q=spec["q"])
        gt_1d = gt_mi_kary_symmetric(k=spec["k"], q=spec["q"])
        tasks.append({
            "task_name": spec["task_name"], "family": "categorical_symmetric",
            "dim": spec["dim"], "k": spec["k"], "q": spec["q"],
            "pxy_1d": pxy_1d, "gt_mi": float(spec["dim"] * gt_1d),
        })
    custom_tables = [
        ("custom-table-3x3-a-1d", 1, np.array([
            [0.18, 0.08, 0.07], [0.08, 0.17, 0.08], [0.07, 0.08, 0.19]], dtype=np.float64)),
        ("custom-table-4x4-b-1d", 1, np.array([
            [0.12, 0.05, 0.04, 0.04], [0.05, 0.11, 0.05, 0.04],
            [0.04, 0.05, 0.11, 0.05], [0.04, 0.04, 0.05, 0.12]], dtype=np.float64)),
        ("custom-table-5x5-c-1d", 1, np.array([
            [0.09, 0.04, 0.03, 0.02, 0.02], [0.04, 0.08, 0.04, 0.03, 0.02],
            [0.03, 0.04, 0.08, 0.04, 0.03], [0.02, 0.03, 0.04, 0.08, 0.04],
            [0.02, 0.02, 0.03, 0.04, 0.09]], dtype=np.float64)),
        ("custom-table-3x3-d-5d", 5, np.array([
            [0.20, 0.07, 0.06], [0.07, 0.18, 0.07], [0.06, 0.07, 0.22]], dtype=np.float64)),
        ("custom-table-4x4-e-5d", 5, np.array([
            [0.13, 0.05, 0.03, 0.02], [0.05, 0.12, 0.05, 0.03],
            [0.03, 0.05, 0.12, 0.05], [0.02, 0.03, 0.05, 0.14]], dtype=np.float64)),
    ]
    for task_name, dim, pxy_1d in custom_tables:
        pxy_1d = pxy_1d / pxy_1d.sum()
        tasks.append({
            "task_name": task_name, "family": "categorical_custom",
            "dim": dim, "pxy_1d": pxy_1d, "gt_mi": float(dim * gt_mi_from_joint_pmf(pxy_1d)),
        })
    return tasks

def sample_from_task(task, sample_size, seed):
    rng = np.random.default_rng(seed)
    return sample_categorical_independent_dims(n=sample_size, dim=task["dim"], pxy_1d=task["pxy_1d"], rng=rng)


# ============================================================
# Per-method evaluation
# ============================================================

def eval_discrete_infoatlas(model, max_dim, softrank_reg, tasks, sample_size, repeats,
                            parallel_bs, seeds, gauss_copula):
    model.eval()
    results = {}
    for task in tasks:
        tn = task["task_name"]
        # Skip tasks whose native dimension exceeds this checkpoint's max_dim.
        if task["dim"] > max_dim:
            results[tn] = {"gt_mi": task["gt_mi"], "est": float("nan")}
            print(f"  [InfoAtlas] {tn:<36s} skipped (dim {task['dim']} > max_dim {max_dim})")
            continue
        ests = []
        for start_idx in range(0, repeats, parallel_bs):
            batch_seeds = seeds[start_idx:start_idx + parallel_bs]
            x_l, y_l = [], []
            for seed in batch_seeds:
                X, Y = sample_from_task(task, sample_size, seed)
                x_l.append(torch.from_numpy(X).float().unsqueeze(0))
                y_l.append(torch.from_numpy(Y).float().unsqueeze(0))
            mi = estimate_mi_batch(torch.cat(x_l, 0), torch.cat(y_l, 0),
                                   model=model, max_dim=max_dim, softrank_reg=softrank_reg, gauss_copula=gauss_copula)
            ests.extend(mi.numpy().tolist())
        results[tn] = {"gt_mi": task["gt_mi"], "est": float(np.mean(ests))}
        print(f"  [InfoAtlas] {tn:<36s} GT={task['gt_mi']:>9.6f} | Est={results[tn]['est']:>9.6f}")
    return results

def eval_discrete_baseline(method_name, tasks, sample_size, repeats, seeds, device):
    results = {}
    for task in tasks:
        tn = task["task_name"]
        ests = []
        for seed in seeds:
            X, Y = sample_from_task(task, sample_size, seed)
            mi = estimate_mi_baseline(method_name, X, Y, device=device)
            ests.append(float(mi))
        results[tn] = {"gt_mi": task["gt_mi"], "est": float(np.mean(ests))}
        print(f"  [{method_name}] {tn:<36s} GT={task['gt_mi']:>9.6f} | Est={results[tn]['est']:>9.6f}")
    return results


# ============================================================
# Main
# ============================================================

def run_sync_discrete_evaluation(args):
    output_dir = os.path.join(args.output_dir, "sync_discrete")
    os.makedirs(output_dir, exist_ok=True)

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    seeds = list(range(42, 42 + args.repeats))
    tasks = build_sync_tasks()

    if "InfoNet" in args.methods:
        init_infonet_v1(args.infonet_config_path, args.infonet_ckpt_path, device=dev)

    model, max_dim, softrank_reg, gauss_copula = None, None, None, True
    if "InfoAtlas" in args.methods:
        module_wrapper, model, cfg, device = load_ckpt(
            ckpt_path=args.ckpt_path, cfg_path=args.cfg_path, device=dev, verbose=True,
        )
        max_dim = int(cfg.input_dim_x)
        softrank_reg = float(cfg.softrank_reg)
        gauss_copula = bool(cfg.gauss_copula) if hasattr(cfg, "gauss_copula") else True

    all_method_results = {}
    for method in args.methods:
        print("=" * 100)
        print(f"[SYNC DISCRETE] Running method: {method}")
        print("=" * 100)
        t0 = time.perf_counter()
        if method == "InfoAtlas":
            results = eval_discrete_infoatlas(model, max_dim, softrank_reg, tasks, args.sample_size,
                                              args.repeats, args.parallel_bs, seeds, gauss_copula)
        else:
            results = eval_discrete_baseline(method, tasks, args.sample_size, args.repeats, seeds, dev)
        all_method_results[method] = results
        print(f"[SYNC DISCRETE] {method} finished in {time.perf_counter() - t0:.2f}s")

    # Build result table
    rows = []
    for task in tasks:
        tn = task["task_name"]
        row = {"task_name": tn, "gt_mi": task["gt_mi"]}
        for method in args.methods:
            r = all_method_results[method][tn]
            est = r["est"]
            row[f"{method}_est"] = est
            row[f"{method}_abs_bias"] = abs(est - task["gt_mi"])
        rows.append(row)
    df = pd.DataFrame(rows)

    # Print summary
    print("\n" + "=" * 100)
    for method in args.methods:
        print(f"[SYNC DISCRETE] {method:<15s} mean_abs_bias = {df[f'{method}_abs_bias'].mean():.8f}")
    print("=" * 100)

    # Save CSV
    csv_path = os.path.join(output_dir, "sync_discrete_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[SYNC DISCRETE] CSV saved to {csv_path}")

    # Save plot
    fig, ax = plt.subplots(figsize=(7, 5))
    mean_biases = [df[f"{m}_abs_bias"].mean() for m in args.methods]
    x = np.arange(len(args.methods))
    ax.bar(x, mean_biases, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(args.methods, rotation=45, ha="right")
    ax.set_ylabel("Mean Absolute Bias")
    ax.set_title("Sync Discrete")
    plt.tight_layout()
    png_path = os.path.join(output_dir, "sync_discrete_results.png")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SYNC DISCRETE] Plot saved to {png_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discrete Sync Evaluation")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--cfg_path", type=str, default=None)
    parser.add_argument("--methods", type=str, nargs="+", default=["InfoAtlas"],
                        help=f"Methods to evaluate. Available: InfoAtlas, {', '.join(AVAILABLE_BASELINES)}")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--parallel_bs", type=int, default=10)
    parser.add_argument("--sample_size", type=int, default=1000)
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

    run_sync_discrete_evaluation(args)
