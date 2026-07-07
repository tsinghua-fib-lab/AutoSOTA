"""
CLIP K-Sliced MI evaluation.

Computes convergence of k-sliced MI under different noise levels and
projection dimensions. Supports comparison with baseline MI estimators
on the same projected pairs.

Usage:
    python -m evaluations.evaluate_clip \
        --ckpt_path /path/to/last.ckpt \
        --data_path /path/to/embeddings.npz \
        --methods InfoAtlas KSG MINE
"""
import os
import sys
import time
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from infer import load_ckpt, estimate_mi_batch
from evaluations._baseline_helper import AVAILABLE_BASELINES, estimate_mi_baseline, init_infonet_v1


# ============================================================
# Projection helpers
# ============================================================

def project_xy(X, Y, projection_dim):
    """Random projection of X and Y to projection_dim dimensions."""
    dx, dy = X.shape[1], Y.shape[1]
    proj_x = torch.randn(dx, projection_dim, device="cpu")
    proj_y = torch.randn(dy, projection_dim, device="cpu")
    proj_x = proj_x / (torch.norm(proj_x, dim=0, keepdim=True) + 1e-12)
    proj_y = proj_y / (torch.norm(proj_y, dim=0, keepdim=True) + 1e-12)
    return X @ proj_x, Y @ proj_y


# ============================================================
# InfoAtlas KSMI with per-projection results
# ============================================================

def ksmi_infoatlas(X, Y, projection_dim, model, proj_num, parallel_bs,
                   max_dim, softrank_reg, gauss_copula, device):
    all_results = []
    for start_idx in range(0, proj_num, parallel_bs):
        bs = min(parallel_bs, proj_num - start_idx)
        x_proj_list, y_proj_list = [], []
        for _ in range(bs):
            Xp, Yp = project_xy(X, Y, projection_dim)
            x_proj_list.append(Xp.unsqueeze(0))
            y_proj_list.append(Yp.unsqueeze(0))
        mi_batch = estimate_mi_batch(
            torch.cat(x_proj_list, 0), torch.cat(y_proj_list, 0),
            model=model, max_dim=max_dim, softrank_reg=softrank_reg,
            gauss_copula=gauss_copula, device=device,
        )
        all_results.append(mi_batch.cpu())
    return torch.cat(all_results, 0).numpy()


def ksmi_baseline(method_name, X, Y, projection_dim, proj_num, device):
    results = []
    for _ in range(proj_num):
        Xp, Yp = project_xy(X, Y, projection_dim)
        mi = estimate_mi_baseline(method_name, Xp.numpy(), Yp.numpy(), device=device)
        results.append(float(mi))
    return np.array(results)


# ============================================================
# Main
# ============================================================

def run_clip_evaluation(args):
    output_dir = os.path.join(args.output_dir, "clip")
    os.makedirs(output_dir, exist_ok=True)

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    proj_dims = [5]
    noise_levels = [0, 0.01, 0.02, 0.05, 0.1]

    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    data = np.load(args.data_path)
    X_all = torch.from_numpy(data["image_embeddings"][:args.sample_num]).float()
    Y_all = torch.from_numpy(data["text_embeddings"][:args.sample_num]).float()
    print(f"[data] X.shape = {tuple(X_all.shape)}, Y.shape = {tuple(Y_all.shape)}")

    # Load InfoNet V1 model if needed
    if "InfoNet" in args.methods:
        init_infonet_v1(args.infonet_config_path, args.infonet_ckpt_path, device=dev)

    # Load InfoAtlas model if needed
    model, max_dim, softrank_reg, gauss_copula, device_obj = None, None, None, True, None
    if "InfoAtlas" in args.methods:
        module_wrapper, model, cfg, device_obj = load_ckpt(
            ckpt_path=args.ckpt_path, cfg_path=args.cfg_path, device=dev, verbose=True,
        )
        max_dim = int(cfg.input_dim_x)
        softrank_reg = float(cfg.softrank_reg)
        gauss_copula = bool(cfg.gauss_copula) if hasattr(cfg, "gauss_copula") else True

    # Run evaluation
    # Structure: {method: {k: {noise: [repeat_cumulative_means]}}}
    all_results: Dict[str, Dict[int, Dict[float, List[np.ndarray]]]] = {}

    for method in args.methods:
        print("=" * 100)
        print(f"[CLIP EVAL] Running method: {method}")
        print("=" * 100)
        all_results[method] = {}
        t0 = time.perf_counter()

        for k in proj_dims:
            all_results[method][k] = {}
            for noise_scale in noise_levels:
                curves = []
                for repeat in range(args.num_repeats):
                    # Add noise
                    x_np = X_all.numpy()
                    norms = np.linalg.norm(x_np, axis=1, keepdims=True)
                    noise = np.random.randn(*x_np.shape) * norms * noise_scale
                    X_noisy = X_all + torch.from_numpy(noise).float()

                    if method == "InfoAtlas":
                        mi_per_proj = ksmi_infoatlas(
                            X_noisy, Y_all, k, model, args.proj_num, args.parallel_bs,
                            max_dim, softrank_reg, gauss_copula, device_obj,
                        )
                    else:
                        mi_per_proj = ksmi_baseline(method, X_noisy, Y_all, k, args.proj_num, dev)

                    cumulative_mean = np.cumsum(mi_per_proj) / np.arange(1, len(mi_per_proj) + 1)
                    curves.append(cumulative_mean)

                    mean_mi = float(mi_per_proj.mean())
                    print(f"  [{method}] k={k}, noise={noise_scale}, repeat={repeat+1}/{args.num_repeats}, mean_mi={mean_mi:.6f}")

                all_results[method][k][noise_scale] = curves

        print(f"[CLIP EVAL] {method} finished in {time.perf_counter() - t0:.2f}s")

    # Save CSV per (k, method)
    for k in proj_dims:
        for method in args.methods:
            data_dict = {"projection_index": np.arange(1, args.proj_num + 1)}
            for noise_scale in noise_levels:
                curves = np.array(all_results[method][k][noise_scale])
                noise_str = str(noise_scale).replace(".", "p")
                data_dict[f"noise_{noise_str}_mean"] = np.mean(curves, axis=0)
                data_dict[f"noise_{noise_str}_std"] = np.std(curves, axis=0)
            df = pd.DataFrame(data_dict)
            csv_path = os.path.join(output_dir, f"clip_k{k}_{method}.csv")
            df.to_csv(csv_path, index=False)
            print(f"[CLIP EVAL] CSV saved to {csv_path}")

    # Save comparison plot per k
    for k in proj_dims:
        fig, axes = plt.subplots(1, len(noise_levels), figsize=(4 * len(noise_levels), 4), sharey=True)
        if len(noise_levels) == 1:
            axes = [axes]
        for ni, noise_scale in enumerate(noise_levels):
            ax = axes[ni]
            for method in args.methods:
                curves = np.array(all_results[method][k][noise_scale])
                mean_curve = np.mean(curves, axis=0)
                std_curve = np.std(curves, axis=0)
                x_axis = np.arange(1, args.proj_num + 1)
                ax.plot(x_axis, mean_curve, label=method)
                ax.fill_between(x_axis, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)
            ax.set_title(f"noise={noise_scale}")
            ax.set_xlabel("# Projections")
            if ni == 0:
                ax.set_ylabel("Cumulative Mean MI")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"CLIP KSMI Convergence (k={k})", fontsize=13)
        plt.tight_layout()
        png_path = os.path.join(output_dir, f"clip_convergence_k{k}.png")
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[CLIP EVAL] Plot saved to {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP K-Sliced MI Evaluation")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--cfg_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True, help="Path to embeddings.npz")
    parser.add_argument("--methods", type=str, nargs="+", default=["InfoAtlas"],
                        help=f"Methods to evaluate. Available: InfoAtlas, {', '.join(AVAILABLE_BASELINES)}")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--parallel_bs", type=int, default=40,
                        help="Batch size for InfoAtlas parallel inference")
    parser.add_argument("--sample_num", type=int, default=5000)
    parser.add_argument("--proj_num", type=int, default=25)
    parser.add_argument("--num_repeats", type=int, default=20)
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

    run_clip_evaluation(args)
