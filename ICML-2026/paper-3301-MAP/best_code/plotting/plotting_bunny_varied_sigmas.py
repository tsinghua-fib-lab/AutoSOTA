import argparse
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

# Move to project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

from datasets import BunnyDataset
from trainers import DDPMTrainer
from utils.constraints import MeshConstraintProjector
from utils.metrics import (MMD, compute_jsd_3d, compute_tvd_3d, coverage,
                           ensure_tensor_2d, filter_valid_samples)

def set_paper_style():
    mpl.rcParams.update({
        "figure.figsize": (3.2, 2.4),
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "lines.linewidth": 2.6,
        "lines.markersize": 7,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def _save_pdf_png(fig, output_path, **kwargs):
    fig.savefig(output_path, **kwargs)
    if output_path.lower().endswith(".pdf"):
        fig.savefig(output_path[:-4] + ".png", **kwargs)

def _attach_time_embed_if_needed(denoiser, state_dict, device):
    """Placeholder function - time embedding utilities have been removed."""
    pass
def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args, _ = parser.parse_known_args()

    set_paper_style()

    sigma_list = [1e-4, 5e-4, 1e-3, 5e-3, 5e-2, 1e-1, 5e-1, 1.0]
    num_samples = 1000
    epochs = 200
    hidden_dim = 128
    timesteps = 250
    # random seed used for deterministic sampling and matching checkpoint filenames
    random_seed = args.seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--trials', type=int, default=3, help='Number of trials per sigma')
    # parser.add_argument('--time-embed', choices=['default', 'sinusoidal', 'fourier'], default='default', help='Time embedding module to use')
    # args = parser.parse_args()
    trials = 5
    time_embed_choice = "default"
    time_embed_dim = 32
    time_concat = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bunny_path = "data/stanford-bunny.obj"

    # Create datasets (lifted training data and plain points on mesh)
    dataset = BunnyDataset(
        num_samples=num_samples,
        mean_idx=10500,
        bunny_path=bunny_path,
        mode="heat",
        noise_level=0.0,
        lifted=True,
    )
    data_points = torch.stack([dataset[i] for i in range(len(dataset))])

    dataset_plain = BunnyDataset(
        num_samples=num_samples,
        mean_idx=10500,
        bunny_path=bunny_path,
        mode="heat",
        noise_level=0.0,
        lifted=False,
    )
    data_points_plain = torch.stack(
        [dataset_plain[i] for i in range(len(dataset_plain))]
    )

    true_tensor = filter_valid_samples(
        ensure_tensor_2d(data_points_plain.cpu(), D=3)
    ).cpu()

    projector = MeshConstraintProjector(bunny_path, device)

    from utils.plotting import plot_scores_vs_time

    # Prepare results directory and containers for metrics across sigmas
    outdir = os.path.join("results", "bunny")
    os.makedirs(outdir, exist_ok=True)

    coverage_list = []
    coverage_std_list = []
    mmd_list = []
    mmd_std_list = []
    jsd_list = []
    jsd_std_list = []
    tvd_list = []
    tvd_std_list = []
    scores_list = []

    # Defaults for the plain (non-projected) model; will be overwritten below
    coverage_plain = float("nan")
    coverage_plain_std = float("nan")
    mmd_plain = float("nan")
    mmd_plain_std = float("nan")
    scores_plain = None

    # Loop over sigma values and evaluate the saved lifted DDPM for each sigma
    for sigma in sigma_list:
        print(f"Processing sigma={sigma}")

        # Build trainer for lifted model
        trainer = DDPMTrainer(
            data_points.squeeze(),
            project_x0_sample=True,
            timesteps=timesteps,
            constraints_dict={"bunny": bunny_path},
            projector=projector,
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim,
            time_conditioning=time_embed_choice,
            mesh=True,
            time_concat=True,
        )

        checkpoint_path = f"models/bunny/model_DDPM_epoch_{epochs}_noise_level_{sigma}_time_{time_embed_choice}_seed_{random_seed}.pth"

        if not os.path.exists(checkpoint_path):
            # If checkpoint missing, record NaNs and continue
            print(
                f"Warning: checkpoint not found for sigma={sigma} (tried several paths), inserting NaNs"
            )
            coverage_list.append(float("nan"))
            coverage_std_list.append(float("nan"))
            mmd_list.append(float("nan"))
            mmd_std_list.append(float("nan"))
            continue

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state = checkpoint.get("model_state_dict", checkpoint)
        # Handle possible mismatched time-embedding modules
        has_timeembed_keys = any(
            k.startswith("time_embed_module.") for k in state.keys()
        )
        if (
            not has_timeembed_keys
            and getattr(trainer.denoiser, "time_embed_module", None) is not None
        ):
            trainer.denoiser.time_embed_module = None
        trainer.denoiser.load_state_dict(state)
        trainer.denoiser.eval()

        trial_coverages = []
        trial_mmds = []
        trial_jsds = []
        trial_tvds = []

        for t in range(trials):
            with torch.no_grad():
                samples_lifted, _ = trainer.sample(num_samples=num_samples)
            try:
                samples_lifted = projector.project(torch.tensor(samples_lifted).cpu())[0].cpu()
            except Exception:
                samples_lifted = torch.tensor(samples_lifted)

            # Collect scores only on first trial
            if t == 0:
                try:
                    scores_list.append(list(trainer.scores))
                except Exception:
                    scores_list.append([])

            # Convert and filter samples
            D = data_points_plain.shape[1]
            samples_lifted_tensor = ensure_tensor_2d(samples_lifted, D).cpu()
            samples_lifted_tensor = filter_valid_samples(samples_lifted_tensor).cpu()

            # Compute metrics against the true samples
            try:
                cov = float(coverage(true_tensor, samples_lifted_tensor))
            except Exception:
                cov = float("nan")
            try:
                mmd_val = (
                    float(MMD(samples_lifted_tensor, true_tensor).item())
                    if hasattr(MMD(samples_lifted_tensor, true_tensor), "item")
                    else float(MMD(samples_lifted_tensor, true_tensor))
                )
            except Exception:
                mmd_val = float("nan")
            try:
                jsd_val = float(
                    compute_jsd_3d(
                        samples_lifted_tensor.numpy(), true_tensor.numpy(), bins=50
                    )
                )
            except Exception:
                jsd_val = float("nan")
            try:
                tvd_val = float(
                    compute_tvd_3d(
                        samples_lifted_tensor.numpy(), true_tensor.numpy(), bins=50
                    )
                )
            except Exception:
                tvd_val = float("nan")

            trial_coverages.append(cov)
            trial_mmds.append(mmd_val)
            trial_jsds.append(jsd_val)
            trial_tvds.append(tvd_val)

        coverage_list.append(float(np.nanmean(np.array(trial_coverages))))
        coverage_std_list.append(float(np.nanstd(np.array(trial_coverages))))
        mmd_list.append(float(np.nanmean(np.array(trial_mmds))))
        mmd_std_list.append(float(np.nanstd(np.array(trial_mmds))))
        jsd_list.append(float(np.nanmean(np.array(trial_jsds))))
        jsd_std_list.append(float(np.nanstd(np.array(trial_jsds))))
        tvd_list.append(float(np.nanmean(np.array(trial_tvds))))
        tvd_std_list.append(float(np.nanstd(np.array(trial_tvds))))

    # Evaluate the plain (traditional DDPM) model to get reference lines
    print("Evaluating traditional (non-projected) DDPM model for reference")
    trainer_plain = DDPMTrainer(
        data_points_plain.squeeze(),
        project_x0_sample=True,
        timesteps=timesteps,
        constraints_dict={"bunny": bunny_path},
        projector=projector,
        hidden_dim=hidden_dim,
        time_embed_dim=time_embed_dim,
        time_conditioning=time_embed_choice,
        mesh=True,
        time_concat=True,
    )
    checkpoint_path = f"models/bunny/model_DDPM_NONPROJECT_epoch_{epochs}_noise_level_0.0_time_{time_embed_choice}_seed_{random_seed}.pth"

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_plain = checkpoint.get("model_state_dict", checkpoint)
        has_timeembed_keys_plain = any(
            k.startswith("time_embed_module.") for k in state_plain.keys()
        )
        if (
            not has_timeembed_keys_plain
            and getattr(trainer_plain.denoiser, "time_embed_module", None) is not None
        ):
            trainer_plain.denoiser.time_embed_module = None
        trainer_plain.denoiser.load_state_dict(state_plain)
        trainer_plain.denoiser.eval()

        plain_coverages = []
        plain_mmds = []
        plain_jsds = []
        plain_tvds = []
        for t in range(trials):
            with torch.no_grad():
                samples_plain, norms = trainer_plain.sample(num_samples=num_samples)
            if t == 0:
                try:
                    scores_plain = list(trainer_plain.scores)
                except Exception:
                    scores_plain = None

            samples_plain_tensor = ensure_tensor_2d(
                samples_plain, data_points_plain.shape[1]
            ).cpu()
            samples_plain_tensor = filter_valid_samples(samples_plain_tensor).cpu()
            try:
                plain_coverages.append(
                    float(coverage(true_tensor, samples_plain_tensor))
                )
            except Exception:
                plain_coverages.append(float("nan"))
            try:
                plain_mmds.append(
                    float(MMD(samples_plain_tensor, true_tensor).item())
                    if hasattr(MMD(samples_plain_tensor, true_tensor), "item")
                    else float(MMD(samples_plain_tensor, true_tensor))
                )
            except Exception:
                plain_mmds.append(float("nan"))
            try:
                # compute JSD and TVD for the plain samples (use same bins as lifted computation)
                plain_jsd_val = float(
                    compute_jsd_3d(
                        samples_plain_tensor.numpy(), true_tensor.numpy(), bins=50
                    )
                )
            except Exception:
                plain_jsd_val = float("nan")
            try:
                plain_tvd_val = float(
                    compute_tvd_3d(
                        samples_plain_tensor.numpy(), true_tensor.numpy(), bins=50
                    )
                )
            except Exception:
                plain_tvd_val = float("nan")

            plain_jsds.append(plain_jsd_val)
            plain_tvds.append(plain_tvd_val)

        coverage_plain = float(np.nanmean(np.array(plain_coverages)))
        coverage_plain_std = float(np.nanstd(np.array(plain_coverages)))
        mmd_plain = float(np.nanmean(np.array(plain_mmds)))
        mmd_plain_std = float(np.nanstd(np.array(plain_mmds)))
        jsd_plain = float(np.nanmean(np.array(plain_jsds)))
        jsd_plain_std = float(np.nanstd(np.array(plain_jsds)))
        tvd_plain = float(np.nanmean(np.array(plain_tvds)))
        tvd_plain_std = float(np.nanstd(np.array(plain_tvds)))
    else:
        print(
            f"Warning: plain DDPM checkpoint not found at {checkpoint_path}; leaving plain metrics as NaN"
        )

    # Save computed metrics for later inspection
    with open(os.path.join(outdir, "metrics_varied_sigmas.json"), "w") as f:
        json.dump(
            {
                "sigma_list": sigma_list,
                "coverage_mean": coverage_list,
                "coverage_std": coverage_std_list,
                "MMD_mean": mmd_list,
                "MMD_std": mmd_std_list,
                "JSD_mean": jsd_list,
                "JSD_std": jsd_std_list,
                "TVD_mean": tvd_list,
                "TVD_std": tvd_std_list,
                "plain": {
                    "coverage_mean": coverage_plain,
                    "coverage_std": coverage_plain_std,
                    "MMD_mean": mmd_plain,
                    "MMD_std": mmd_plain_std,
                    "JSD_mean": jsd_plain if "jsd_plain" in locals() else float("nan"),
                    "JSD_std": (
                        jsd_plain_std if "jsd_plain_std" in locals() else float("nan")
                    ),
                    "TVD_mean": tvd_plain if "tvd_plain" in locals() else float("nan"),
                    "TVD_std": (
                        tvd_plain_std if "tvd_plain_std" in locals() else float("nan")
                    ),
                },
            },
            f,
        )

    if len(scores_list) > 0 and scores_plain is not None:
        plot_scores_vs_time(
            scores_list=scores_list,
            scores_plain=scores_plain,
            sigma_list=sigma_list,
            output_path=os.path.join(outdir, "scores_vs_time.pdf"),
        )

        # 'MMD_mean': mmd_list,
        # 'MMD_std': mmd_std_list,
        # 'plain': {
        #     'coverage_mean': coverage_plain,
        #     'coverage_std': coverage_plain_std,
        #     'MMD_mean': mmd_plain,
        #     'MMD_std': mmd_plain_std,
        # }

    # Optional: scores plotting if we collected them
    if len(scores_list) > 0 and scores_plain is not None:
        set_paper_style()
        cmap = plt.cm.viridis
        try:
            norm = mpl.colors.LogNorm(
                vmin=float(min(sigma_list)), vmax=float(max(sigma_list))
            )
        except Exception:
            norm = mpl.colors.Normalize(
                vmin=float(min(sigma_list)), vmax=float(max(sigma_list))
            )

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        fig, ax = plt.subplots(figsize=(3.5, 4.0))
        nan_label_plotted = False
        for i, sigma in enumerate(sigma_list):
            raw = scores_list[i]
            try:
                scores_list_sigma = np.array([float(x) for x in raw])
            except Exception:
                scores_list_sigma = np.array(raw)
            invalid_mask = ~np.isfinite(scores_list_sigma)
            valid_mask = np.isfinite(scores_list_sigma)
            color = cmap(norm(sigma))
            ax.plot(scores_list_sigma, color=color, linewidth=2.0)
            if np.any(valid_mask):
                top_y = np.nanmax(scores_list_sigma[valid_mask]) * 1.1
            else:
                top_y = 1.0
            if np.any(invalid_mask):
                label = "NaN or Inf" if not nan_label_plotted else ""
                ax.plot(
                    np.where(invalid_mask)[0],
                    [top_y] * np.sum(invalid_mask),
                    "x",
                    color=color,
                    markersize=5,
                    label=label,
                )
                nan_label_plotted = True

        try:
            scores_plain_plot = np.array([float(x) for x in scores_plain])
        except Exception:
            scores_plain_plot = np.array(scores_plain)
        invalid_mask_plain = ~np.isfinite(scores_plain_plot)
        valid_mask_plain = np.isfinite(scores_plain_plot)
        if np.any(valid_mask_plain):
            top_y_plain = np.nanmax(scores_plain_plot[valid_mask_plain]) * 1.1
        else:
            top_y_plain = 1.0
        ax.plot(scores_plain_plot, label="DDPM", linestyle="--", color="red", linewidth=2.6)
        if np.any(invalid_mask_plain):
            ax.plot(
                np.where(invalid_mask_plain)[0],
                [top_y_plain] * np.sum(invalid_mask_plain),
                "x",
                color="red",
                markersize=5,
                label="NaN or Inf" if not nan_label_plotted else "",
            )

        ax.set_yscale("log")

        cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.20)
        cbar.set_label("σ", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        # Prefer ticks at powers of ten within the sigma range and label as 10^{exp}
        try:
            min_s = float(min(sigma_list))
            max_s = float(max(sigma_list))
            if min_s > 0 and max_s > 0:
                exp_min = int(np.floor(np.log10(min_s)))
                exp_max = int(np.ceil(np.log10(max_s)))
                exps = np.arange(exp_min, exp_max + 1)
                ticks = (10.0**exps).tolist()
                ticks = [t for t in ticks if t >= min_s and t <= max_s]
                if len(ticks) >= 1:
                    cbar.set_ticks(ticks)
                    cbar.set_ticklabels(
                        [r"$10^{%d}$" % int(np.round(np.log10(t))) for t in ticks]
                    )
                else:
                    cbar.set_ticks([min_s, max_s])
                    cbar.set_ticklabels([f"{min_s:.3g}", f"{max_s:.3g}"])
        except Exception:
            try:
                cbar.set_ticks(sigma_list)
                cbar.set_ticklabels([str(s) for s in sigma_list])
            except Exception:
                pass

        ax.legend(fontsize=9, frameon=False, loc="upper left")
        num_points = len(scores_plain_plot)
        xticks = np.linspace(0, num_points - 1, num=6, dtype=int)
        xtick_labels = [f"{num_points - 1 - x}" for x in xticks]
        xtick_labels[0] = str(timesteps)
        xtick_labels[-1] = "0"
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, fontsize=9)
        ax.set_xlabel(r"$t$ (reversed)", fontsize=11, labelpad=8)
        ax.set_ylabel(
            r"Median $\nabla_x(t) \, \log p_t(x(t))$", fontsize=11
        )
        ax.grid(True)
        fig.subplots_adjust(bottom=0.20)
        _save_pdf_png(fig, os.path.join(outdir, "scores_vs_time.pdf"), bbox_inches="tight")
        plt.close(fig)

    print("Done. Results in", outdir)

    # Combined figure styled like other scripts
    def build_combined_figure_from_results(output_dir,
                                           sigma_list,
                                           coverage_list,
                                           coverage_std_list,
                                           jsd_list,
                                           jsd_std_list,
                                           tvd_list,
                                           tvd_std_list,
                                           coverage_plain,
                                           coverage_plain_std,
                                           jsd_plain,
                                           jsd_plain_std,
                                           tvd_plain,
                                           tvd_plain_std):
        set_paper_style()
        color_map = {
            "OURS": "#1f77b4",  # blue
            "DDPM": "#d62728",   # red
        }
        line_width = 3.0
        marker_size = 7

        # Use consistent proportions for 3-panel figures
        fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), sharex=True)
        metric_data = [
            ("Coverage", np.array(coverage_list), np.array(coverage_std_list), coverage_plain, coverage_plain_std),
            ("JSD", np.array(jsd_list), np.array(jsd_std_list), jsd_plain if 'jsd_plain' in locals() else float('nan'), jsd_plain_std if 'jsd_plain_std' in locals() else float('nan')),
            ("TVD", np.array(tvd_list), np.array(tvd_std_list), tvd_plain if 'tvd_plain' in locals() else float('nan'), tvd_plain_std if 'tvd_plain_std' in locals() else float('nan')),
        ]
        handles = []
        labels = []

        for ax, (label, vals, stds, base_mean, base_std) in zip(axes, metric_data):
            h, = ax.plot(
                sigma_list,
                vals,
                marker="o",
                markersize=marker_size,
                linewidth=line_width,
                color=color_map["OURS"],
                label=r"$p_{\sigma}$ (ours)",
                zorder=2,
            )
            try:
                arr = vals.astype(float)
                std_arr = stds.astype(float)
                min_band = 0.005 if label == "Coverage" else 0.002
                # For positions where std is missing, use min_band, but only draw
                # the fill where the main value is finite (so JSD/TVD still show bands
                # at the sigmas that have data).
                std_arr = np.where(np.isfinite(std_arr), std_arr, min_band)
                mask = np.isfinite(arr)
                if mask.any():
                    sig = np.array(sigma_list)[mask]
                    delta = np.maximum(std_arr[mask], min_band)
                    lower = arr[mask] - delta
                    upper = arr[mask] + delta
                    if label in ("JSD", "TVD"):
                        eps = 1e-12
                        lower = np.maximum(lower, eps)
                        upper = np.maximum(upper, eps)
                    ax.fill_between(sig, lower, upper, color=color_map["OURS"], alpha=0.25, linewidth=0, zorder=1)
            except Exception:
                pass
            if ax is axes[0]:
                handles.append(h)
                labels.append(r"$p_{\sigma}$ (ours)")

            if np.isfinite(base_mean):
                hb = ax.axhline(
                    y=base_mean,
                    color=color_map["DDPM"],
                    linestyle="--",
                    linewidth=line_width,
                    alpha=0.9,
                    label="DDPM (proj.)",
                )
                if ax is axes[0]:
                    handles.append(hb)
                    labels.append("DDPM (proj.)")
                if base_std is not None and np.isfinite(base_std):
                    eps = 1e-12 if label in ("JSD", "TVD") else 0.0
                    low = max(base_mean - base_std, eps)
                    high = max(base_mean + base_std, eps)
                    ax.fill_between(
                        sigma_list,
                        [low] * len(sigma_list),
                        [high] * len(sigma_list),
                        color=color_map["DDPM"],
                        alpha=0.10,
                    )

            ax.set_xscale("log")
            ax.set_xlabel("σ")
            ax.set_ylabel(label)
            try:
                min_sigma = float(min(sigma_list))
                max_sigma = float(max(sigma_list))
                ax.set_xlim(min_sigma, max_sigma)
                from matplotlib.ticker import LogLocator, FuncFormatter
                ax.xaxis.set_major_locator(LogLocator(base=10.0))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
                ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
            except Exception:
                pass
            # Adaptive y-limits with padding
            try:
                values_for_ylim = [vals]
                if np.isfinite(base_mean):
                    values_for_ylim.append(np.array([base_mean]))
                y_all = np.concatenate([v.flatten() for v in values_for_ylim])
                y_all = y_all[np.isfinite(y_all)]
                if label == "Coverage":
                    if y_all.size > 0:
                        ymin = max(0.0, float(y_all.min()) - 0.03)
                        ymax = min(1.0, float(y_all.max()) + 0.03)
                        ax.set_ylim(ymin, ymax)
                    else:
                        ax.set_ylim(0, 1)
                else:
                    if y_all.size > 0:
                        # Use tighter padding for JSD/TVD so bands are easier to see.
                        y_min_val = float(y_all.min())
                        y_max_val = float(y_all.max())
                        span = max(1e-12, y_max_val - y_min_val)
                        pad = span * 0.05 if span > 0 else max(1e-3, 0.01 * max(abs(y_max_val), 1.0))
                        ymin = max(0.0, y_min_val - pad)
                        ymax = y_max_val + pad
                        ax.set_ylim(ymin, ymax)
                    else:
                        ax.set_ylim(bottom=0)
            except Exception:
                pass
            ax.grid(True)

        try:
            ncols = max(1, len(labels))
            fig.legend(handles, labels, loc="lower center", ncol=ncols, frameon=False, bbox_to_anchor=(0.5, -0.14))
        except Exception:
            pass

        fig.tight_layout()
        try:
            fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.92, wspace=0.35)
        except Exception:
            pass
        out_path = os.path.join(output_dir, "combined_metrics.pdf")
        try:
            _save_pdf_png(fig, out_path, bbox_inches="tight")
        except Exception:
            out_path = os.path.join(output_dir, "combined_metrics_fallback.pdf")
            _save_pdf_png(fig, out_path, bbox_inches="tight")
        finally:
            plt.close(fig)

    # Build and save combined figure (Coverage/JSD/TVD)
    build_combined_figure_from_results(
        outdir,
        sigma_list,
        coverage_list,
        coverage_std_list,
        jsd_list,
        jsd_std_list,
        tvd_list,
        tvd_std_list,
        coverage_plain,
        coverage_plain_std,
        jsd_plain if 'jsd_plain' in locals() else float('nan'),
        jsd_plain_std if 'jsd_plain_std' in locals() else float('nan'),
        tvd_plain if 'tvd_plain' in locals() else float('nan'),
        tvd_plain_std if 'tvd_plain_std' in locals() else float('nan'),
    )


if __name__ == "__main__":
    main()
