"""
2D Tracking MI-based video segmentation evaluation.

Loads annot.npz from data_root, preprocesses trajectories (filters extreme
points), then computes MI between point trajectories and evaluates
segmentation quality via ROC-AUC.

Usage:
    python -m evaluations.evaluate_2dtrack \
        --ckpt_path /path/to/last.ckpt \
        --data_root data/point_odyssey/val \
        --methods InfoAtlas KSG MINE \
        --video_configs "animal2_s:1700,3400,3700" "ani_s:2000,2100"
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import measure
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from infer import load_ckpt, estimate_mi_batch
from evaluations._baseline_helper import (
    AVAILABLE_BASELINES, estimate_mi_baseline, init_infonet_v1,
)


# ============================================================
# Trajectory preprocessing
# ============================================================

def preprocess_trajectories(annot_path, screen_w=960, screen_h=540, extreme=4000):
    """
    Load annot.npz and filter out points with extreme coordinates.

    Filtering rules (same as the original preprocessing notebook):
      1. Frame 0: remove points outside [0, screen_w] x [0, screen_h].
      2. All frames: remove points outside [-extreme, extreme] on either axis.

    Returns:
        trajs_2d: [frames, n_valid, 2] filtered trajectories (float32)
    """
    gt = np.load(annot_path, allow_pickle=True)
    track2d = gt["trajs_2d"]  # [frames, n_points, 2]
    valid = set(range(track2d.shape[1]))

    for t in range(track2d.shape[0]):
        frame = track2d[t]
        if t == 0:
            bad = np.where(
                (frame[:, 0] < 0) | (frame[:, 0] > screen_w) |
                (frame[:, 1] < 0) | (frame[:, 1] > screen_h)
            )[0]
            valid -= set(bad)

        bad = np.where(
            (frame[:, 0] <= -extreme) | (frame[:, 0] >= extreme) |
            (frame[:, 1] <= -extreme) | (frame[:, 1] >= extreme)
        )[0]
        valid -= set(bad)

    valid_list = sorted(valid)
    trajs_2d = track2d[:, valid_list, :].astype(np.float32)
    return trajs_2d


def get_or_create_trajectories(data_root, video_name):
    """
    Return trajs_2d for a video. If trajs_2d.npy already exists next to
    annot.npz, load it; otherwise preprocess and save.
    """
    video_dir = os.path.join(data_root, video_name)
    traj_path = os.path.join(video_dir, "trajs_2d.npy")

    if os.path.isfile(traj_path):
        trajs_2d = np.load(traj_path)
        print(f"  Loaded cached trajectories: {traj_path}  shape={trajs_2d.shape}")
        return trajs_2d

    # Try both naming conventions
    annot_path = os.path.join(video_dir, "annot.npz")
    if not os.path.isfile(annot_path):
        annot_path = os.path.join(video_dir, "annotations.npz")
    if not os.path.isfile(annot_path):
        raise FileNotFoundError(
            f"Neither annot.npz nor annotations.npz found in {video_dir}"
        )

    print(f"  Preprocessing trajectories from {annot_path} ...")
    trajs_2d = preprocess_trajectories(annot_path)
    np.save(traj_path, trajs_2d)
    print(f"  Saved trajs_2d.npy: shape={trajs_2d.shape}")
    return trajs_2d


# ============================================================
# Instance segmentation mask utilities
# ============================================================

def get_mask_instance_map(mask_path):
    """Read a mask PNG and return per-pixel instance IDs (H, W)."""
    mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask_img is None:
        raise FileNotFoundError(f"Mask image not found: {mask_path}")
    if mask_img.ndim == 2:
        mask_rgb = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB)
    else:
        mask_rgb = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
    H, W, _ = mask_rgb.shape
    instance_map = np.zeros((H, W), dtype=np.int32)
    colors = np.unique(mask_rgb.reshape(-1, 3), axis=0)
    instance_id = 1
    for color in colors:
        color_mask = np.all(mask_rgb == color, axis=-1)
        if not np.any(color_mask):
            continue
        labeled = measure.label(color_mask, connectivity=1)
        for lid in range(1, labeled.max() + 1):
            instance_map[labeled == lid] = instance_id
            instance_id += 1
    return instance_map


def get_point_instance_ids(trajs_2d, instance_map):
    """Assign instance IDs to tracked points based on first-frame positions."""
    H, W = instance_map.shape
    n_points = trajs_2d.shape[1]
    first_frame_xy = np.round(trajs_2d[0]).astype(int)
    point_instance_ids = np.zeros(n_points, dtype=int)
    for pid in range(n_points):
        x, y = first_frame_xy[pid]
        if 0 <= x < W and 0 <= y < H:
            point_instance_ids[pid] = instance_map[y, x]
    return point_instance_ids


# ============================================================
# MI computation: InfoAtlas
# ============================================================

def compute_mi_all_points_infoatlas(trajs_2d, ref_idx, model, max_dim,
                                     softrank_reg, gauss_copula, parallel_bs):
    n_points = trajs_2d.shape[1]
    mi_values = np.zeros(n_points, dtype=np.float32)
    other_indices = [i for i in range(n_points) if i != ref_idx]
    Y_ref = torch.from_numpy(trajs_2d[:, ref_idx, :]).float()

    for start in range(0, len(other_indices), parallel_bs):
        batch_indices = other_indices[start:start + parallel_bs]
        x_list, y_list = [], []
        for i in batch_indices:
            x_list.append(torch.from_numpy(trajs_2d[:, i, :]).float().unsqueeze(0))
            y_list.append(Y_ref.unsqueeze(0))

        mi_batch = estimate_mi_batch(
            torch.cat(x_list, 0), torch.cat(y_list, 0),
            model=model, max_dim=max_dim, softrank_reg=softrank_reg,
            gauss_copula=gauss_copula,
        )
        for j, i in enumerate(batch_indices):
            mi_values[i] = float(mi_batch[j])

    return mi_values


# ============================================================
# MI computation: baselines
# ============================================================

def compute_mi_all_points_baseline(trajs_2d, ref_idx, method_name, device):
    n_points = trajs_2d.shape[1]
    mi_values = np.zeros(n_points, dtype=np.float32)
    Y_ref = trajs_2d[:, ref_idx, :]

    for i in range(n_points):
        if i == ref_idx:
            continue
        X_i = trajs_2d[:, i, :]
        mi = estimate_mi_baseline(method_name, X_i, Y_ref, device=device)
        mi_values[i] = float(mi)

    return mi_values


# ============================================================
# AUC computation
# ============================================================

def compute_auc(point_instance_ids, mi_values, ref_idx):
    anchor_inst = point_instance_ids[ref_idx]
    y_true = (point_instance_ids == anchor_inst).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    mi_clean = mi_values.copy()
    mi_clean[np.isnan(mi_clean)] = 0.0
    mi_clean[np.isinf(mi_clean)] = 0.0
    try:
        return float(roc_auc_score(y_true, mi_clean))
    except Exception:
        return float("nan")


# ============================================================
# Visualization
# ============================================================

def plot_mi_heatmap(image, location, mi_values, ref_point, save_path,
                    method_name="", video_name="", ref_idx=0):
    """Publication-quality MI heatmap overlay on the reference frame."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
    })

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.imshow(image)

    mi_plot = mi_values.copy()
    mi_plot[np.isnan(mi_plot)] = 0.0
    vmin, vmax = np.percentile(mi_plot[mi_plot > 0], [2, 98]) if (mi_plot > 0).any() else (0, 1)

    scatter = ax.scatter(
        location[:, 0], location[:, 1], c=mi_plot,
        cmap="inferno", s=18, alpha=0.85, edgecolors="none",
        vmin=vmin, vmax=vmax, rasterized=True,
    )
    ax.scatter(
        ref_point[0], ref_point[1], facecolors="none", edgecolors="cyan",
        s=220, linewidths=2.0, marker="o", zorder=10, label="Reference point",
    )
    ax.scatter(
        ref_point[0], ref_point[1], c="cyan",
        s=30, marker="o", zorder=11,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.08)
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label("Mutual Information (nats)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.set_axis_off()
    ax.legend(loc="upper right", fontsize=9, frameon=True,
              facecolor="white", edgecolor="gray", framealpha=0.9,
              handletextpad=0.3, borderpad=0.4)

    title = f"{method_name}" if method_name else ""
    if video_name:
        title += f"  ({video_name}, ref={ref_idx})"
    if title:
        ax.set_title(title, fontsize=12, pad=6)

    fig.tight_layout(pad=0.5)
    fig.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)


def generate_visualization_plots(output_dir, data_root, video_configs, methods,
                                  all_method_results_by_video):
    """
    After all methods finish, generate per-ref-point comparison plots and
    per-method MI distribution plots in results/2dtrack/visualization/.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
    })
    vis_dir = os.path.join(output_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)

    for video_name, ref_data in all_method_results_by_video.items():
        # Try to load reference image
        image_path = os.path.join(data_root, video_name, "rgbs", "rgb_00001.jpg")
        if not os.path.isfile(image_path):
            image_path = os.path.join(data_root, video_name, "rgbs", "rgb_00000.jpg")
        image = None
        if os.path.isfile(image_path):
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        for ref_idx, method_mi in ref_data.items():
            available_methods = [m for m in methods if m in method_mi]
            if not available_methods:
                continue

            trajs_2d = method_mi["_trajs_2d"]
            location = trajs_2d[0]  # first-frame positions
            ref_point = location[ref_idx]

            # --- 1) Side-by-side comparison across methods ---
            n_methods = len(available_methods)
            if image is not None and n_methods >= 1:
                fig, axes = plt.subplots(1, n_methods,
                                         figsize=(5.0 * n_methods, 4.5), dpi=300)
                if n_methods == 1:
                    axes = [axes]

                for ax, method in zip(axes, available_methods):
                    mi_vals = method_mi[method].copy()
                    mi_vals[np.isnan(mi_vals)] = 0.0
                    vmin, vmax = (np.percentile(mi_vals[mi_vals > 0], [2, 98])
                                  if (mi_vals > 0).any() else (0, 1))

                    ax.imshow(image)
                    sc = ax.scatter(
                        location[:, 0], location[:, 1], c=mi_vals,
                        cmap="inferno", s=12, alpha=0.85, edgecolors="none",
                        vmin=vmin, vmax=vmax, rasterized=True,
                    )
                    ax.scatter(ref_point[0], ref_point[1], facecolors="none",
                               edgecolors="cyan", s=180, linewidths=1.8,
                               marker="o", zorder=10)
                    ax.scatter(ref_point[0], ref_point[1], c="cyan",
                               s=20, marker="o", zorder=11)
                    ax.set_title(method, fontsize=12)
                    ax.set_axis_off()

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="3%", pad=0.05)
                    fig.colorbar(sc, cax=cax).ax.tick_params(labelsize=8)

                fig.suptitle(f"{video_name}  (ref={ref_idx})", fontsize=13, y=1.02)
                fig.tight_layout(pad=0.8)
                comp_path = os.path.join(vis_dir,
                                         f"comparison_{video_name}_ref{ref_idx}.pdf")
                fig.savefig(comp_path, bbox_inches="tight", dpi=300, facecolor="white")
                plt.close(fig)
                print(f"  [VIS] Comparison plot saved: {comp_path}")

            # --- 2) MI distribution violin/box plot across methods ---
            fig, ax = plt.subplots(figsize=(max(3.5, 1.2 * n_methods), 4), dpi=300)
            mi_data = []
            labels = []
            for method in available_methods:
                mi_vals = method_mi[method].copy()
                mi_vals[np.isnan(mi_vals)] = 0.0
                nonzero = mi_vals[mi_vals != 0]
                mi_data.append(nonzero)
                labels.append(method)

            parts = ax.violinplot(mi_data, positions=range(len(labels)),
                                  showextrema=False, showmedians=False)
            colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
            for pc, color in zip(parts["bodies"], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)

            bp = ax.boxplot(mi_data, positions=range(len(labels)), widths=0.15,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", linewidth=1.5))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.9)

            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=10)
            ax.set_ylabel("MI (nats)", fontsize=11)
            ax.set_title(f"MI Distribution  ({video_name}, ref={ref_idx})", fontsize=12)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout(pad=0.8)
            dist_path = os.path.join(vis_dir,
                                     f"distribution_{video_name}_ref{ref_idx}.pdf")
            fig.savefig(dist_path, bbox_inches="tight", dpi=300, facecolor="white")
            plt.close(fig)
            print(f"  [VIS] Distribution plot saved: {dist_path}")


# ============================================================
# Parse video configs
# ============================================================

def parse_video_configs(video_config_strs):
    """Parse 'video_name:idx1,idx2,...' strings into a dict."""
    configs = {}
    for s in video_config_strs:
        name, idxs_str = s.split(":")
        configs[name.strip()] = [int(x.strip()) for x in idxs_str.split(",")]
    return configs


# ============================================================
# Main
# ============================================================

def run_2dtrack_evaluation(args):
    output_dir = os.path.join(args.output_dir, "2dtrack")
    os.makedirs(output_dir, exist_ok=True)

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    video_configs = parse_video_configs(args.video_configs)

    # Load InfoAtlas model if needed
    model, max_dim, softrank_reg, gauss_copula = None, None, None, True
    if "InfoAtlas" in args.methods:
        module_wrapper, model, cfg, device_obj = load_ckpt(
            ckpt_path=args.ckpt_path, cfg_path=args.cfg_path, device=dev, verbose=True,
        )
        max_dim = int(cfg.input_dim_x)
        softrank_reg = float(cfg.softrank_reg)
        gauss_copula = bool(cfg.gauss_copula) if hasattr(cfg, "gauss_copula") else True

    # Load InfoNet V1 model if needed
    if "InfoNet" in args.methods:
        if args.infonet_config_path is None or args.infonet_ckpt_path is None:
            raise ValueError("--infonet_config_path and --infonet_ckpt_path required for InfoNet")
        init_infonet_v1(args.infonet_config_path, args.infonet_ckpt_path, device=dev)

    # Collect all results
    all_rows = []
    # {video_name: {ref_idx: {"_trajs_2d": ..., method: mi_values, ...}}}
    all_mi_by_video = {}

    for video_name, ref_indices in video_configs.items():
        print("\n" + "=" * 100)
        print(f"[2DTRACK] Processing video: {video_name}, ref points: {ref_indices}")
        print("=" * 100)

        # Get or create trajectories
        try:
            trajs_2d = get_or_create_trajectories(args.data_root, video_name)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        n_points = trajs_2d.shape[1]
        print(f"  Trajectory shape: {trajs_2d.shape}")

        # Load instance mask
        mask_path = os.path.join(args.data_root, video_name, "masks", "mask_00000.png")
        if not os.path.isfile(mask_path):
            print(f"  [SKIP] Mask not found: {mask_path}")
            continue
        instance_map = get_mask_instance_map(mask_path)
        point_instance_ids = get_point_instance_ids(trajs_2d, instance_map)

        # Load reference image for visualization
        image = None
        image_path = os.path.join(args.data_root, video_name, "rgbs", "rgb_00001.jpg")
        if not os.path.isfile(image_path):
            image_path = os.path.join(args.data_root, video_name, "rgbs", "rgb_00000.jpg")
        if os.path.isfile(image_path):
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        all_mi_by_video.setdefault(video_name, {})

        for ref_idx in ref_indices:
            if ref_idx >= n_points:
                print(f"  [SKIP] ref_idx={ref_idx} exceeds n_points={n_points}")
                continue

            all_mi_by_video[video_name].setdefault(ref_idx, {"_trajs_2d": trajs_2d})

            for method in args.methods:
                print(f"  [{method}] Computing MI for ref_idx={ref_idx} ...")
                t0 = time.perf_counter()

                # Check for cached results
                cache_path = os.path.join(output_dir, f"{video_name}_{ref_idx}_{method}.npy")
                if args.load_cached and os.path.isfile(cache_path):
                    mi_values = np.load(cache_path)
                    print(f"    Loaded cached results from {cache_path}")
                else:
                    if method == "InfoAtlas":
                        mi_values = compute_mi_all_points_infoatlas(
                            trajs_2d, ref_idx, model, max_dim,
                            softrank_reg, gauss_copula, args.parallel_bs,
                        )
                    else:
                        mi_values = compute_mi_all_points_baseline(
                            trajs_2d, ref_idx, method, device=dev,
                        )
                    np.save(cache_path, mi_values)

                elapsed = time.perf_counter() - t0
                roc_auc = compute_auc(point_instance_ids, mi_values, ref_idx)
                print(f"    AUC={roc_auc:.4f}, time={elapsed:.2f}s")

                all_mi_by_video[video_name][ref_idx][method] = mi_values

                all_rows.append({
                    "video": video_name,
                    "ref_idx": ref_idx,
                    "method": method,
                    "auc": roc_auc,
                    "time_s": elapsed,
                })

                # Save per-method heatmap
                if image is not None:
                    heatmap_dir = os.path.join(output_dir, "heatmaps")
                    os.makedirs(heatmap_dir, exist_ok=True)
                    heatmap_path = os.path.join(
                        heatmap_dir, f"{method}_{video_name}_ref{ref_idx}.png",
                    )
                    location = trajs_2d[0]
                    plot_mi_heatmap(image, location, mi_values,
                                    location[ref_idx], heatmap_path,
                                    method_name=method, video_name=video_name,
                                    ref_idx=ref_idx)
                    print(f"    Heatmap saved to {heatmap_path}")

    if not all_rows:
        print("\n[2DTRACK] No results generated.")
        return None

    df = pd.DataFrame(all_rows)

    # ---- Summary ----
    print("\n" + "=" * 100)
    print("[2DTRACK] AUC Summary:")
    for method in args.methods:
        sub = df[df["method"] == method]
        if len(sub) > 0:
            mean_auc = sub["auc"].mean()
            std_auc = sub["auc"].std()
            mean_time = sub["time_s"].mean()
            print(f"  {method:<15s} mean_AUC={mean_auc:.4f} +/- {std_auc:.4f}, mean_time={mean_time:.2f}s")
    print("=" * 100)

    # ---- Save CSV ----
    csv_path = os.path.join(output_dir, "2dtrack_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[2DTRACK] CSV saved to {csv_path}")

    # ---- Save AUC bar plot ----
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    methods_present = [m for m in args.methods if m in df["method"].values]
    mean_aucs = [df[df["method"] == m]["auc"].mean() for m in methods_present]
    std_aucs = [df[df["method"] == m]["auc"].std() for m in methods_present]
    x = np.arange(len(methods_present))
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods_present)))
    ax.bar(x, mean_aucs, yerr=std_aucs, color=colors, alpha=0.85,
           capsize=4, edgecolor="gray", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods_present, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("2D Tracking: MI-based Segmentation AUC")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    png_path = os.path.join(output_dir, "2dtrack_auc.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[2DTRACK] AUC bar plot saved to {png_path}")

    # ---- Save AUC vs Time scatter plot ----
    if len(methods_present) > 1:
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
        colors = plt.cm.Set2(np.linspace(0, 1, len(methods_present)))
        for idx, method in enumerate(methods_present):
            sub = df[df["method"] == method]
            mean_auc = sub["auc"].mean()
            std_auc = sub["auc"].std()
            mean_time = sub["time_s"].mean()
            ax.scatter(mean_time, mean_auc, s=160, color=colors[idx],
                       edgecolors="gray", linewidths=0.5, zorder=5)
            ax.errorbar(mean_time, mean_auc, yerr=std_auc, fmt="none",
                        color="gray", alpha=0.5, capsize=4, zorder=3)
            ax.annotate(method, (mean_time, mean_auc),
                        xytext=(0, 10), textcoords="offset points",
                        fontsize=9, ha="center")
        ax.set_xscale("log")
        ax.set_xlabel("Time (s, log scale)")
        ax.set_ylabel("AUC")
        ax.set_title("2D Tracking: AUC vs. Compute Time")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        scatter_path = os.path.join(output_dir, "2dtrack_auc_vs_time.pdf")
        fig.savefig(scatter_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[2DTRACK] Scatter plot saved to {scatter_path}")

    # ---- Generate visualization plots (comparison + distribution) ----
    generate_visualization_plots(output_dir, args.data_root, video_configs,
                                  args.methods, all_mi_by_video)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Tracking MI-based Segmentation Evaluation")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Path to InfoAtlas checkpoint (required if InfoAtlas in --methods)")
    parser.add_argument("--cfg_path", type=str, default=None)
    parser.add_argument("--infonet_config_path", type=str, default=None)
    parser.add_argument("--infonet_ckpt_path", type=str, default=None)
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root path to point_odyssey data (e.g., data/point_odyssey/val)")
    parser.add_argument("--methods", type=str, nargs="+", default=["InfoAtlas"],
                        help=f"Methods to evaluate. Available: InfoAtlas, {', '.join(AVAILABLE_BASELINES)}")
    parser.add_argument("--video_configs", type=str, nargs="+", required=True,
                        help='Video configs as "video_name:idx1,idx2,...". '
                             'Example: "ani_s:2000,2100" "animal2_s:1700,3400"')
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--parallel_bs", type=int, default=64)
    parser.add_argument("--load_cached", action="store_true",
                        help="Load cached MI results if available")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if "InfoAtlas" in args.methods and args.ckpt_path is None:
        parser.error("--ckpt_path is required when InfoAtlas is in --methods")
    if "InfoNet" in args.methods and (args.infonet_config_path is None or args.infonet_ckpt_path is None):
        parser.error("--infonet_config_path and --infonet_ckpt_path are required when InfoNet is in --methods")

    run_2dtrack_evaluation(args)

