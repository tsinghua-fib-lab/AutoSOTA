"""
Sync evaluation on continuous synthetic tasks (Beta/Gamma copula, Bernoulli, Poisson).

Usage:
    python -m evaluations.evaluate_sync \
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
from scipy.stats import norm, beta as beta_dist, gamma as gamma_dist, poisson
from scipy.special import logsumexp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from infer import load_ckpt, estimate_mi_batch
from evaluations._baseline_helper import AVAILABLE_BASELINES, estimate_mi_baseline, init_infonet_v1

EPS = 1e-12


# ============================================================
# Ground-truth MI
# ============================================================

def gt_mi_gaussian_copula_1d(rho):
    return float(-0.5 * np.log(1.0 - rho ** 2))

def gt_mi_gaussian_copula_diag(dim, rho):
    return float(dim * gt_mi_gaussian_copula_1d(rho))

def gt_mi_bernoulli_1d(q):
    term1 = 0.0 if q == 0 else q * np.log(2.0 * q)
    term2 = 0.0 if q == 1 else (1.0 - q) * np.log(2.0 * (1.0 - q))
    return float(term1 + term2)

def gt_mi_bernoulli_independent_dims(dim, q):
    return float(dim * gt_mi_bernoulli_1d(q))

def joint_pmf_poisson_shared_component(max_x, max_y, lam0, lam1, lam2):
    log_const = -(lam0 + lam1 + lam2)
    pxy = np.zeros((max_x + 1, max_y + 1), dtype=np.float64)
    max_n = max(max_x, max_y)
    log_fact = np.zeros(max_n + 1, dtype=np.float64)
    for n in range(1, max_n + 1):
        log_fact[n] = log_fact[n - 1] + np.log(n)
    def log_p_term(u, x, y):
        val = log_const
        if lam0 == 0:
            if u != 0: return -np.inf
        else:
            val += u * np.log(lam0)
        val -= log_fact[u]
        xv = x - u
        if lam1 == 0:
            if xv != 0: return -np.inf
        else:
            val += xv * np.log(lam1)
        val -= log_fact[xv]
        yw = y - u
        if lam2 == 0:
            if yw != 0: return -np.inf
        else:
            val += yw * np.log(lam2)
        val -= log_fact[yw]
        return val
    for x in range(max_x + 1):
        for y in range(max_y + 1):
            m = min(x, y)
            logs = [log_p_term(u, x, y) for u in range(m + 1)]
            pxy[x, y] = np.exp(logsumexp(logs))
    return pxy

def gt_mi_poisson_shared_component_1d(lam0, lam1, lam2, tail_mass_tol=1e-12, extra_buffer=20):
    lam_x, lam_y = lam0 + lam1, lam0 + lam2
    q = 1.0 - tail_mass_tol
    max_x = int(poisson.ppf(q, lam_x))
    max_y = int(poisson.ppf(q, lam_y))
    if not np.isfinite(max_x):
        max_x = int(lam_x + 10.0 * np.sqrt(max(lam_x, 1.0)) + extra_buffer)
    if not np.isfinite(max_y):
        max_y = int(lam_y + 10.0 * np.sqrt(max(lam_y, 1.0)) + extra_buffer)
    max_x += extra_buffer
    max_y += extra_buffer
    pxy = joint_pmf_poisson_shared_component(max_x, max_y, lam0, lam1, lam2)
    pxy /= pxy.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    ratio = pxy / np.clip(px @ py, EPS, None)
    return float(np.sum(pxy * np.log(np.clip(ratio, EPS, None))))

def gt_mi_poisson_shared_component_independent_dims(dim, lam0, lam1, lam2, **kw):
    return float(dim * gt_mi_poisson_shared_component_1d(lam0, lam1, lam2, **kw))


# ============================================================
# Sampling
# ============================================================

def sample_gaussian_copula_beta(n, dim, rho, alpha_x, beta_x, alpha_y, beta_y, rng):
    cov = np.block([[np.eye(dim), rho * np.eye(dim)], [rho * np.eye(dim), np.eye(dim)]])
    z = rng.multivariate_normal(np.zeros(2 * dim), cov, size=n)
    ux = np.clip(norm.cdf(z[:, :dim]), EPS, 1.0 - EPS)
    uy = np.clip(norm.cdf(z[:, dim:]), EPS, 1.0 - EPS)
    return beta_dist.ppf(ux, a=alpha_x, b=beta_x).astype(np.float32), \
           beta_dist.ppf(uy, a=alpha_y, b=beta_y).astype(np.float32)

def sample_gaussian_copula_gamma(n, dim, rho, shape_x, scale_x, shape_y, scale_y, rng):
    cov = np.block([[np.eye(dim), rho * np.eye(dim)], [rho * np.eye(dim), np.eye(dim)]])
    z = rng.multivariate_normal(np.zeros(2 * dim), cov, size=n)
    ux = np.clip(norm.cdf(z[:, :dim]), EPS, 1.0 - EPS)
    uy = np.clip(norm.cdf(z[:, dim:]), EPS, 1.0 - EPS)
    return gamma_dist.ppf(ux, a=shape_x, scale=scale_x).astype(np.float32), \
           gamma_dist.ppf(uy, a=shape_y, scale=scale_y).astype(np.float32)

def sample_correlated_bernoulli(n, dim, q, rng):
    x = rng.integers(0, 2, size=(n, dim))
    same = rng.random(size=(n, dim)) < q
    return x.astype(np.float32), np.where(same, x, 1 - x).astype(np.float32)

def sample_correlated_poisson_shared_component(n, dim, lam0, lam1, lam2, rng):
    u = rng.poisson(lam=lam0, size=(n, dim))
    v = rng.poisson(lam=lam1, size=(n, dim))
    w = rng.poisson(lam=lam2, size=(n, dim))
    return (u + v).astype(np.float32), (u + w).astype(np.float32)


# ============================================================
# Task builder & dispatcher
# ============================================================

def build_sync_tasks():
    """Build 24 tasks: 8 per dimension (1d, 3d, 5d), covering low/medium/high MI.

    Each dimension has:
      - 1 beta copula + 1 gamma copula (different rho to avoid identical GT MI)
      - 3 bernoulli (q = 0.6, 0.75, 0.9)
      - 3 poisson  (lam0 = 1, 2, 4)

    dim=5 uses lower rho values for copula to avoid high relative bias.
    """
    tasks = []

    for dim in [1, 3, 5]:
        # Copula: 1 beta + 1 gamma with different rho
        if dim <= 3:
            beta_rho, gamma_rho = 0.5, 0.8
        else:
            beta_rho, gamma_rho = 0.2, 0.5

        tasks.append({"task_name": f"beta-{dim}d-rho{beta_rho}", "family": "beta",
                       "dim": dim, "rho": beta_rho,
                       "alpha_x": 2.0, "beta_x": 5.0, "alpha_y": 5.0, "beta_y": 2.0,
                       "gt_mi": gt_mi_gaussian_copula_diag(dim, beta_rho)})
        tasks.append({"task_name": f"gamma-{dim}d-rho{gamma_rho}", "family": "gamma",
                       "dim": dim, "rho": gamma_rho,
                       "shape_x": 2.0, "scale_x": 1.0, "shape_y": 5.0, "scale_y": 0.5,
                       "gt_mi": gt_mi_gaussian_copula_diag(dim, gamma_rho)})

        # Bernoulli: 3 tasks
        for q in [0.6, 0.75, 0.9]:
            tasks.append({"task_name": f"bernoulli-{dim}d-q{q}", "family": "bernoulli",
                           "dim": dim, "q": q,
                           "gt_mi": gt_mi_bernoulli_independent_dims(dim, q)})

        # Poisson: 3 tasks
        for lam0, lam1, lam2 in [(1.0, 2.0, 2.0), (2.0, 2.0, 2.0), (4.0, 2.0, 2.0)]:
            tasks.append({"task_name": f"poisson-{dim}d-l0_{lam0}-l1_{lam1}-l2_{lam2}",
                           "family": "poisson", "dim": dim,
                           "lam0": lam0, "lam1": lam1, "lam2": lam2,
                           "gt_mi": gt_mi_poisson_shared_component_independent_dims(dim, lam0, lam1, lam2)})

    return tasks

def sample_from_task(task, sample_size, seed):
    rng = np.random.default_rng(seed)
    family, dim = task["family"], task["dim"]
    if family == "beta":
        return sample_gaussian_copula_beta(sample_size, dim, task["rho"],
                                           task["alpha_x"], task["beta_x"], task["alpha_y"], task["beta_y"], rng)
    if family == "gamma":
        return sample_gaussian_copula_gamma(sample_size, dim, task["rho"],
                                            task["shape_x"], task["scale_x"], task["shape_y"], task["scale_y"], rng)
    if family == "bernoulli":
        return sample_correlated_bernoulli(sample_size, dim, task["q"], rng)
    if family == "poisson":
        return sample_correlated_poisson_shared_component(sample_size, dim, task["lam0"], task["lam1"], task["lam2"], rng)
    raise ValueError(f"Unknown family: {family}")


# ============================================================
# Per-method evaluation
# ============================================================

def eval_sync_infoatlas(model, max_dim, softrank_reg, tasks, sample_size, repeats, parallel_bs, seeds, gauss_copula):
    model.eval()
    results = {}
    for task in tasks:
        tn = task["task_name"]
        # Skip tasks whose native dimension exceeds this checkpoint's max_dim
        # (e.g. the 5d tasks on a max_dim=3 model), which InfoAtlas cannot ingest.
        if task["dim"] > max_dim:
            results[tn] = {"gt_mi": task["gt_mi"], "mean_est": float("nan")}
            print(f"  [InfoAtlas] {tn:<40s} skipped (dim {task['dim']} > max_dim {max_dim})")
            continue
        estimates = []
        for start_idx in range(0, repeats, parallel_bs):
            batch_seeds = seeds[start_idx:start_idx + parallel_bs]
            x_list, y_list = [], []
            for seed in batch_seeds:
                X, Y = sample_from_task(task, sample_size, seed)
                x_list.append(torch.from_numpy(X).float().unsqueeze(0))
                y_list.append(torch.from_numpy(Y).float().unsqueeze(0))
            mi_batch = estimate_mi_batch(
                torch.cat(x_list, 0), torch.cat(y_list, 0),
                model=model, max_dim=max_dim, softrank_reg=softrank_reg, gauss_copula=gauss_copula,
            )
            estimates.extend(mi_batch.numpy().tolist())
        results[tn] = {"gt_mi": task["gt_mi"], "mean_est": float(np.mean(estimates))}
        print(f"  [InfoAtlas] {tn:<40s} GT={task['gt_mi']:>10.6f} | Est={results[tn]['mean_est']:>10.6f}")
    return results

def eval_sync_baseline(method_name, tasks, sample_size, repeats, seeds, device):
    results = {}
    for task in tasks:
        tn = task["task_name"]
        estimates = []
        for seed in seeds:
            X, Y = sample_from_task(task, sample_size, seed)
            mi = estimate_mi_baseline(method_name, X, Y, device=device)
            estimates.append(float(mi))
        results[tn] = {"gt_mi": task["gt_mi"], "mean_est": float(np.mean(estimates))}
        print(f"  [{method_name}] {tn:<40s} GT={task['gt_mi']:>10.6f} | Est={results[tn]['mean_est']:>10.6f}")
    return results


# ============================================================
# Main
# ============================================================

def run_sync_evaluation(args):
    output_dir = os.path.join(args.output_dir, "sync")
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
        print(f"[SYNC EVAL] Running method: {method}")
        print("=" * 100)
        t0 = time.perf_counter()
        if method == "InfoAtlas":
            results = eval_sync_infoatlas(model, max_dim, softrank_reg, tasks, args.sample_size,
                                          args.repeats, args.parallel_bs, seeds, gauss_copula)
        else:
            results = eval_sync_baseline(method, tasks, args.sample_size, args.repeats, seeds, dev)
        all_method_results[method] = results
        print(f"[SYNC EVAL] {method} finished in {time.perf_counter() - t0:.2f}s")

    # Build result table (use relative bias = |est - gt| / gt)
    rows = []
    for task in tasks:
        tn = task["task_name"]
        gt = task["gt_mi"]
        row = {"task_name": tn, "family": task["family"], "dim": task["dim"], "gt_mi": gt}
        for method in args.methods:
            r = all_method_results[method][tn]
            est = r["mean_est"]
            row[f"{method}_est"] = est
            row[f"{method}_rel_bias"] = abs(est - gt) / gt if gt > 0 else float("inf")
        rows.append(row)
    df = pd.DataFrame(rows)

    # Print summary
    print("\n" + "=" * 100)
    for method in args.methods:
        mean_rb = df[f"{method}_rel_bias"].replace([np.inf], np.nan).mean()
        print(f"[SYNC EVAL] {method:<15s} mean_rel_bias = {mean_rb:.6f}")
    print("=" * 100)

    # Save CSV
    csv_path = os.path.join(output_dir, "sync_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[SYNC EVAL] CSV saved to {csv_path}")

    # Save plot
    fig, ax = plt.subplots(figsize=(10, 5))
    mean_biases = []
    for m in args.methods:
        mean_biases.append(df[f"{m}_rel_bias"].replace([np.inf], np.nan).mean())
    x = np.arange(len(args.methods))
    ax.bar(x, mean_biases, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(args.methods, rotation=45, ha="right")
    ax.set_ylabel("Mean Relative Bias")
    ax.set_title("Sync Continuous: Mean Relative Bias by Method")
    plt.tight_layout()
    png_path = os.path.join(output_dir, "sync_results.png")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SYNC EVAL] Plot saved to {png_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Evaluation (continuous)")
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

    run_sync_evaluation(args)
