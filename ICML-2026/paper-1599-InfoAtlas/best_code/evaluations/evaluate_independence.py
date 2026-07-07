"""
Independence testing evaluation via K-Sliced MI and ROC-AUC.

Supports multiple test methods, dimensions, and num_test in a single run.

Usage:
    python -m evaluations.evaluate_independence \
        --ckpt_path /path/to/last.ckpt \
        --methods InfoAtlas KSG MINE \
        --test_methods test1 test2 test3 \
        --dims 16 64 128 \
        --num_test 20
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
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from infer import load_ckpt, compute_ksmi_mean
from evaluations._baseline_helper import AVAILABLE_BASELINES, estimate_mi_baseline, init_infonet_v1


# ============================================================
# Data generators
# ============================================================

def gen_independent(d, seq_len):
    mean = np.zeros(d)
    sigma = np.eye(d)
    return np.random.multivariate_normal(mean, sigma, seq_len), \
           np.random.multivariate_normal(mean, sigma, seq_len)

def gen_test1(d, seq_len):
    mean = np.zeros(d)
    sigma = np.eye(d)
    X = np.random.multivariate_normal(mean, sigma, seq_len)
    Z = np.random.multivariate_normal(mean, sigma, seq_len)
    tmp = np.dot(np.ones(d), X.T)
    Y = (np.outer(tmp, np.ones(d)) / np.sqrt(d) + Z) / np.sqrt(2)
    return X, Y

def gen_test2(d, seq_len):
    mean = np.zeros(d)
    sigma = np.eye(d)
    X = np.random.multivariate_normal(mean, sigma, seq_len)
    Z = np.random.multivariate_normal(mean, sigma, seq_len)
    tmp1 = np.zeros(d // 2)
    tmp2 = np.ones(d // 2)
    Y = np.zeros((seq_len, d))
    for i in range(d):
        if i <= d // 2:
            Y[:, i] = np.dot(np.concatenate((tmp2, tmp1)).T, X.T) + Z[:, i]
        else:
            Y[:, i] = np.dot(np.concatenate((tmp2, tmp1)).T, X.T) + Z[:, i]
    Y = Y / (d * np.sqrt(2))
    return X, Y

def gen_test3(d, seq_len):
    mean = np.zeros(d)
    sigma = np.eye(d)
    X = np.random.multivariate_normal(mean, sigma, seq_len)
    Z = np.random.multivariate_normal(mean, sigma, seq_len)
    return X, (X + Z) / np.sqrt(2)

GEN_FN = {"test1": gen_test1, "test2": gen_test2, "test3": gen_test3}

TEST_DISPLAY_NAME = {"test1": "Test 1", "test2": "Test 2", "test3": "Test 3"}


# ============================================================
# Per-method AUC computation
# ============================================================

def compute_auc_infoatlas(model, max_dim, softrank_reg, gauss_copula,
                          test_method, d, seq_len, proj_num, num_test, parallel_bs):
    gen_fn = GEN_FN[test_method]
    smi_means = []
    y_labels = np.concatenate((np.ones(num_test // 2), np.zeros(num_test // 2)))

    for j in range(num_test // 2):
        X, Y = gen_fn(d, seq_len)
        smi_mean = compute_ksmi_mean(
            torch.from_numpy(X).float(), torch.from_numpy(Y).float(),
            projection_dim=max_dim, model=model,
            proj_num=proj_num, batchsize=parallel_bs,
            max_dim=max_dim, softrank_reg=softrank_reg, gauss_copula=gauss_copula,
        )
        smi_means.append(smi_mean)

    for j in range(num_test // 2):
        X, Y = gen_independent(d, seq_len)
        smi_mean = compute_ksmi_mean(
            torch.from_numpy(X).float(), torch.from_numpy(Y).float(),
            projection_dim=max_dim, model=model,
            proj_num=proj_num, batchsize=parallel_bs,
            max_dim=max_dim, softrank_reg=softrank_reg, gauss_copula=gauss_copula,
        )
        smi_means.append(smi_mean)

    fpr, tpr, _ = roc_curve(y_labels, np.array(smi_means))
    return auc(fpr, tpr)


def compute_auc_baseline(method_name, test_method, d, seq_len, num_test, device):
    gen_fn = GEN_FN[test_method]
    mi_vals = []
    y_labels = np.concatenate((np.ones(num_test // 2), np.zeros(num_test // 2)))

    for j in range(num_test // 2):
        X, Y = gen_fn(d, seq_len)
        mi = estimate_mi_baseline(method_name, X, Y, device=device)
        mi_vals.append(float(mi))

    for j in range(num_test // 2):
        X, Y = gen_independent(d, seq_len)
        mi = estimate_mi_baseline(method_name, X, Y, device=device)
        mi_vals.append(float(mi))

    fpr, tpr, _ = roc_curve(y_labels, np.array(mi_vals))
    return auc(fpr, tpr)


# ============================================================
# Plotting (publication quality)
# ============================================================

# Method display style
METHOD_STYLE = {
    "InfoAtlas":  {"color": "#E63946", "marker": "o",  "linestyle": "-"},
    "KSG":        {"color": "#457B9D", "marker": "s",  "linestyle": "--"},
    "MINE":       {"color": "#2A9D8F", "marker": "^",  "linestyle": "--"},
    "InfoNCE":    {"color": "#E9C46A", "marker": "D",  "linestyle": "--"},
    "SMILE":      {"color": "#F4A261", "marker": "v",  "linestyle": "--"},
    "MINDE":      {"color": "#264653", "marker": "<",  "linestyle": "--"},
    "DoE":        {"color": "#6A4C93", "marker": ">",  "linestyle": "--"},
    "FLE":        {"color": "#1982C4", "marker": "p",  "linestyle": "--"},
    "FastMI":     {"color": "#8AC926", "marker": "h",  "linestyle": "--"},
    "MIENF":      {"color": "#FF595E", "marker": "H",  "linestyle": "--"},
    "MRE":        {"color": "#6D6875", "marker": "X",  "linestyle": "--"},
    "VCE":        {"color": "#B5838D", "marker": "d",  "linestyle": "--"},
    "InfoNet":    {"color": "#FFCA3A", "marker": "*",  "linestyle": "--"},
}

_DEFAULT_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']


def _get_style(method, idx):
    if method in METHOD_STYLE:
        return METHOD_STYLE[method]
    return {"color": _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)],
            "marker": "o", "linestyle": "--"}


def save_publication_plots(all_results, methods, test_methods, dims, seq_lens, output_dir):
    """
    One figure per test_method.
    Each figure has len(dims) subplots side by side.
    Each subplot: AUC vs sample size, one line per method.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    for test_method in test_methods:
        n_dims = len(dims)
        fig_w = min(4.5 * n_dims, 18)
        fig_h = 3.8
        fig, axes = plt.subplots(1, n_dims, figsize=(fig_w, fig_h), squeeze=False)
        axes = axes[0]

        for col, d in enumerate(dims):
            ax = axes[col]
            for mi, method in enumerate(methods):
                key = (test_method, d)
                if key not in all_results or method not in all_results[key]:
                    continue
                auc_by_sl = all_results[key][method]
                sls = sorted(auc_by_sl.keys())
                aucs = [auc_by_sl[sl] for sl in sls]
                style = _get_style(method, mi)
                ax.plot(sls, aucs, label=method,
                        color=style["color"], marker=style["marker"],
                        linestyle=style["linestyle"], markeredgecolor="white",
                        markeredgewidth=0.4)

            ax.set_xlabel("Sample Size")
            if col == 0:
                ax.set_ylabel("AUC")
            ax.set_title(f"{TEST_DISPLAY_NAME.get(test_method, test_method)},  $d = {d}$")
            ax.set_ylim(-0.02, 1.05)
            ax.grid(True, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # shared legend at the bottom
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(len(methods), 7), frameon=False,
                   bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout(rect=[0, 0.06, 1, 1])
        png_path = os.path.join(output_dir, f"independence_{test_method}.pdf")
        fig.savefig(png_path)
        png_path2 = os.path.join(output_dir, f"independence_{test_method}.png")
        fig.savefig(png_path2)
        plt.close(fig)
        print(f"[INDE] Plot saved to {png_path} and {png_path2}")


# ============================================================
# Main
# ============================================================

def run_independence_evaluation(args):
    output_dir = os.path.join(args.output_dir, "independence_testing")
    os.makedirs(output_dir, exist_ok=True)

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    seq_lens = list(range(50, 850, 50))

    test_methods = args.test_methods
    dims = args.dims

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

    # all_results[(test_method, dim)][method] = {seq_len: auc}
    all_results = {}
    all_rows = []

    for test_method in test_methods:
        for d in dims:
            print("=" * 100)
            print(f"[INDE] test={test_method}, dim={d}, num_test={args.num_test}")
            print("=" * 100)

            combo_key = (test_method, d)
            all_results[combo_key] = {}

            for method in args.methods:
                t0 = time.perf_counter()
                auc_by_seqlen = {}

                for seq_len in seq_lens:
                    if method == "InfoAtlas":
                        roc_auc = compute_auc_infoatlas(
                            model, max_dim, softrank_reg, gauss_copula,
                            test_method, d, seq_len,
                            args.proj_num, args.num_test, args.parallel_bs,
                        )
                    else:
                        roc_auc = compute_auc_baseline(
                            method, test_method, d, seq_len, args.num_test, dev,
                        )
                    auc_by_seqlen[seq_len] = roc_auc
                    print(f"  [{method}] test={test_method} dim={d} seq_len={seq_len} AUC={roc_auc:.4f}")

                    all_rows.append({
                        "test_method": test_method,
                        "dim": d,
                        "seq_len": seq_len,
                        "method": method,
                        "auc": roc_auc,
                    })

                all_results[combo_key][method] = auc_by_seqlen
                elapsed = time.perf_counter() - t0
                mean_auc = np.mean(list(auc_by_seqlen.values()))
                print(f"  [{method}] done in {elapsed:.1f}s, mean AUC={mean_auc:.4f}")

    # ---- Save CSV ----
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(output_dir, "independence_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[INDE] CSV saved to {csv_path}")

    # ---- Print summary table ----
    print("\n" + "=" * 100)
    print(f"{'Method':<15s}", end="")
    for test_method in test_methods:
        for d in dims:
            print(f"  {test_method}_d{d:>4d}", end="")
    print()
    for method in args.methods:
        print(f"{method:<15s}", end="")
        for test_method in test_methods:
            for d in dims:
                key = (test_method, d)
                if key in all_results and method in all_results[key]:
                    mean_auc = np.mean(list(all_results[key][method].values()))
                    print(f"  {mean_auc:>12.4f}", end="")
                else:
                    print(f"  {'N/A':>12s}", end="")
        print()
    print("=" * 100)

    # ---- Save plots ----
    save_publication_plots(all_results, args.methods, test_methods, dims, seq_lens, output_dir)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Independence Testing Evaluation")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--cfg_path", type=str, default=None)
    parser.add_argument("--methods", type=str, nargs="+", default=["InfoAtlas"],
                        help=f"Methods to evaluate. Available: InfoAtlas, {', '.join(AVAILABLE_BASELINES)}")
    parser.add_argument("--test_methods", type=str, nargs="+", default=["test2"],
                        choices=["test1", "test2", "test3"],
                        help="Independence test types to run (default: test2)")
    parser.add_argument("--dims", type=int, nargs="+", default=[16],
                        help="Dimensions to test (default: 16)")
    parser.add_argument("--num_test", type=int, default=10,
                        help="Number of tests per (dependent + independent), must be even")
    parser.add_argument("--proj_num", type=int, default=64)
    parser.add_argument("--parallel_bs", type=int, default=64,
                        help="Batch size for InfoAtlas parallel inference")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--infonet_config_path", type=str, default=None)
    parser.add_argument("--infonet_ckpt_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if "InfoAtlas" in args.methods and args.ckpt_path is None:
        parser.error("--ckpt_path is required when InfoAtlas is in --methods")
    if "InfoNet" in args.methods and (args.infonet_config_path is None or args.infonet_ckpt_path is None):
        parser.error("--infonet_config_path and --infonet_ckpt_path are required when InfoNet is in --methods")

    run_independence_evaluation(args)
