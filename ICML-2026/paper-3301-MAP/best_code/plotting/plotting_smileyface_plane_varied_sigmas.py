import argparse
import json
import os
import sys

# Move to the project root (one level up from current file)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

import numpy as np
import matplotlib as mpl
import torch
import trimesh
from plotting_smileyface_plane import to_intrinsic_2d_plane

from datasets import *
from trainers import *
from utils.constraints import *
from utils.metrics import *
from utils.metrics import ensure_tensor_2d, filter_valid_samples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_pdf_png(fig, output_path, **kwargs):
    fig.savefig(output_path, **kwargs)
    if output_path.lower().endswith(".pdf"):
        fig.savefig(output_path[:-4] + ".png", **kwargs)


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
A = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(0)  # Normal vector (x-axis)
b = torch.tensor([1.0])  # Offset (good gracious!)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args, _ = parser.parse_known_args()

sigma_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
num_samples = 10000
epochs = 200
hidden_dim = 64
time_embed_dim = 32
timesteps = 250
time_concat = True
# seed used for deterministic runs and to select matching checkpoints
random_seed = args.seed
trials = 10
time_embed_choice = "default"

torch.manual_seed(random_seed)
np.random.seed(random_seed)
set_paper_style()
dataset = SmileyFaceDataset(
    num_samples=num_samples,
    A=A,
    b=b,
    lifted=False,
    noise_level=0.0,
    device=device,
    seed=random_seed,
)
data_points = torch.stack([dataset[i] for i in range(len(dataset))])
D2 = 2
with torch.no_grad():
    data_points_plain_2d = to_intrinsic_2d_plane(data_points.cpu(), A.cpu(), b.cpu())
true_tensor_2d = ensure_tensor_2d(data_points_plain_2d, D2).cpu()
true_tensor_2d = filter_valid_samples(true_tensor_2d).cpu()
trainer = DDPMTrainer(
    data_points.squeeze(),
    project_x0_sample=True,
    timesteps=timesteps,
    constraints_dict={"linear_equality": (A.to(device), b.to(device))},
    hidden_dim=hidden_dim,
    time_concat=time_concat,
    time_embed_dim=time_embed_dim,
    time_conditioning=time_embed_choice,
)

coverage_list = []
JSD_list = []
TVD_list = []
coverage_std_list = []
JSD_std_list = []
TVD_std_list = []
scores_list = []
for sigma in sigma_list:
    # load model for this sigma once
    checkpoint_path = f"models/smileyface_plane/model_DDPM_epoch_{epochs}_noise_level_{sigma}_time_{time_embed_choice}_seed_{random_seed}.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    has_timeembed_keys = any(k.startswith("time_embed_module.") for k in state.keys())
    if (
        not has_timeembed_keys
        and getattr(trainer.denoiser, "time_embed_module", None) is not None
    ):
        print(
            f"Checkpoint {checkpoint_path} has no time_embed_module keys — removing trainer.denoiser.time_embed_module to match checkpoint"
        )
        trainer.denoiser.time_embed_module = None
    trainer.denoiser.load_state_dict(state)
    torch.cuda.empty_cache()
    trainer.denoiser.eval()

    trial_coverages = []
    trial_jsds = []
    trial_tvds = []
    # Run multiple sampling trials with the same model to estimate variability
    for t in range(trials):
        with torch.no_grad():
            samples_lifted, _ = trainer.sample(num_samples=num_samples)

        if t == 0:
            # Collect scores only for the first trial: append the per-time-step
            # score norms (trainer.scores is a list of scalar tensors)
            scores_list.append(list(trainer.scores))

        # If torch.tensor got shadowed upstream, recover here once.
        import importlib

        if not callable(torch.tensor):
            torch = importlib.reload(torch)

        with torch.no_grad():
            ref_dir = torch.tensor([1.0, 0.0, 0.0], device=A.device, dtype=A.dtype)
            samples_lifted_2d = to_intrinsic_2d_plane(
                torch.tensor(samples_lifted), A, b
            )

        samples_lifted_2d = ensure_tensor_2d(samples_lifted_2d, D2).cpu()
        samples_lifted_2d = filter_valid_samples(samples_lifted_2d).cpu()
        if not isinstance(true_tensor_2d, torch.Tensor):
            true_tensor_2d = torch.tensor(true_tensor_2d)

        coverage_val = float(coverage(true_tensor_2d, samples_lifted_2d))
        JSD_val = float(jsd_histogram_2d(true_tensor_2d, samples_lifted_2d, bins=25))

        # compute TVD similarly to JSD (same binning)
        try:
            TVD_val = float(
                tvd_histogram_2d(true_tensor_2d, samples_lifted_2d, bins=25)
            )
        except Exception:
            TVD_val = float("nan")

        trial_coverages.append(coverage_val)
        trial_jsds.append(JSD_val)
        trial_tvds.append(TVD_val)

    # compute mean and std across trials
    cov_arr = np.array(trial_coverages)
    jsd_arr = np.array(trial_jsds)
    tvd_arr = np.array(trial_tvds)
    coverage_list.append(float(np.mean(cov_arr)))
    coverage_std_list.append(float(np.std(cov_arr)))
    JSD_list.append(float(np.mean(jsd_arr)))
    JSD_std_list.append(float(np.std(jsd_arr)))
    TVD_list.append(float(np.mean(tvd_arr)))
    TVD_std_list.append(float(np.std(tvd_arr)))

# Traditional DDPM Score
print("Traditional DDPM Model")
trainer_plain = DDPMTrainer(
    data_points.squeeze(),
    project_x0_sample=False,
    timesteps=timesteps,
    constraints_dict={"linear_equality": (A.to(device), b.to(device))},
    hidden_dim=hidden_dim,
    time_concat=time_concat,
    time_embed_dim=time_embed_dim,
    time_conditioning=time_embed_choice,
)
checkpoint_path = f"models/smileyface_plane/model_DDPM_NONPROJECT_epoch_{epochs}_noise_level_0.0_time_{time_embed_choice}_seed_{random_seed}.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
state_plain = checkpoint.get("model_state_dict", checkpoint)
has_timeembed_keys_plain = any(
    k.startswith("time_embed_module.") for k in state_plain.keys()
)
if (
    not has_timeembed_keys_plain
    and getattr(trainer_plain.denoiser, "time_embed_module", None) is not None
):
    print(
        f"Checkpoint {checkpoint_path} has no time_embed_module keys — removing trainer_plain.denoiser.time_embed_module to match checkpoint"
    )
    trainer_plain.denoiser.time_embed_module = None
trainer_plain.denoiser.load_state_dict(state_plain)
trainer_plain.denoiser.eval()

# Run multiple trials for the plain model too
plain_coverages = []
plain_jsds = []
plain_tvds = []
for t in range(trials):
    with torch.no_grad():
        samples_plain, norms = trainer_plain.sample(num_samples=num_samples)
    if t == 0:
        scores_plain = trainer_plain.scores
        # scores_plain.append(scores_t_plain)
    with torch.no_grad():
        ref_dir = torch.tensor([1.0, 0.0, 0.0], device=A.device, dtype=A.dtype)
        samples_plain_2d = to_intrinsic_2d_plane(torch.tensor(samples_plain), A, b)

    samples_plain_2d = ensure_tensor_2d(samples_plain_2d, D2).cpu()
    samples_plain_2d = filter_valid_samples(samples_plain_2d).cpu()
    plain_coverages.append(float(coverage(true_tensor_2d, samples_plain_2d)))
    plain_jsds.append(
        float(jsd_histogram_2d(true_tensor_2d, samples_plain_2d, bins=50))
    )
    plain_tvds.append(
        float(tvd_histogram_2d(true_tensor_2d, samples_plain_2d, bins=50))
    )

coverage_plain = float(np.mean(np.array(plain_coverages)))
coverage_plain_std = float(np.std(np.array(plain_coverages)))
JSD_plain = float(np.mean(np.array(plain_jsds)))
JSD_plain_std = float(np.std(np.array(plain_jsds)))
TVD_plain = float(np.mean(np.array(plain_tvds)))
TVD_plain_std = float(np.std(np.array(plain_tvds)))

output_dir = "results/smileyface_plane"
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "metrics_varied_sigmas.json"), "w") as f:
    json.dump(
        {
            "sigma_list": sigma_list,
            "coverage_mean": coverage_list,
            "coverage_std": coverage_std_list,
            "JSD_mean": JSD_list,
            "JSD_std": JSD_std_list,
            "TVD_mean": TVD_list,
            "TVD_std": TVD_std_list,
            "plain": {
                "coverage_mean": coverage_plain,
                "coverage_std": coverage_plain_std,
                "JSD_mean": JSD_plain,
                "JSD_std": JSD_plain_std,
                "TVD_mean": TVD_plain,
                "TVD_std": TVD_plain_std,
            },
        },
        f,
    )
import matplotlib as mpl
import matplotlib.pyplot as plt

def build_combined_figure_from_results(output_dir,
                                       sigma_list,
                                       coverage_list,
                                       coverage_std_list,
                                       JSD_list,
                                       JSD_std_list,
                                       TVD_list,
                                       TVD_std_list,
                                       coverage_plain,
                                       coverage_plain_std,
                                       JSD_plain,
                                       JSD_plain_std,
                                       TVD_plain,
                                       TVD_plain_std):
    """Build a combined 1x3 figure for Coverage/JSD/TVD vs sigma
    with our method ($p_{\sigma}$) and DDPM baseline.
    """
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
        ("JSD", np.array(JSD_list), np.array(JSD_std_list), JSD_plain, JSD_plain_std),
        ("TVD", np.array(TVD_list), np.array(TVD_std_list), TVD_plain, TVD_plain_std),
    ]
    handles = []
    labels = []

    for ax, (label, vals, stds, base_mean, base_std) in zip(axes, metric_data):
        # ours curve
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
        # error band (min thickness enforcement similar to sphere script)
        try:
            arr = vals.astype(float)
            std_arr = stds.astype(float)
            mask = np.isfinite(arr) & np.isfinite(std_arr)
            if mask.any():
                sig = np.array(sigma_list)[mask]
                min_band = 0.005 if label == "Coverage" else 0.002
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

        # Collect legend handle once
        if ax is axes[0]:
            handles.append(h)
            labels.append(r"$p_{\sigma}$ (ours)")

        # baseline line and band
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

        # axes cosmetics
        ax.set_xscale("log")
        ax.set_xlabel("σ")
        ax.set_ylabel(label)
        # Set x-limits and ticks to standard increments for clarity
        try:
            min_sigma = float(min(sigma_list))
            max_sigma = float(max(sigma_list))
            ax.set_xlim(min_sigma, max_sigma)
            # Major ticks at powers of 10 only
            from matplotlib.ticker import LogLocator, FuncFormatter
            ax.xaxis.set_major_locator(LogLocator(base=10.0))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
            # Remove minor tick labels entirely to avoid 5s
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
        except Exception:
            pass
        # Adaptive y-limits with small padding to make lines clearer
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
                    ymin = 0.0
                    ymax = max(1e-12, float(y_all.max()) * 1.10)
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
if "scores_list" in globals() and "scores_plain" in globals():
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

    fig, ax = plt.subplots(figsize=(4.5, 4.8))
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

    # Plot plain scores if present
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
    cbar.set_label("σ", fontsize=12)
    cbar.ax.tick_params(labelsize=11)
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

    ax.legend(fontsize=11, frameon=False, loc="upper left")
    ax.set_xlabel(r"$t$ (reversed)", fontsize=13, labelpad=8)
    num_points = len(scores_plain_plot)
    xticks = np.linspace(0, num_points - 1, num=6, dtype=int)
    xtick_labels = [f"{num_points - 1 - x}" for x in xticks]
    xtick_labels[0] = "250"
    xtick_labels[-1] = "0"
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_ylabel(
        r"Median $\nabla_x(t) \, \log p_t(x(t))$", fontsize=13
    )
    ax.grid(True)
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.28, top=0.92)
    _save_pdf_png(fig, os.path.join(output_dir, "scores_vs_time.pdf"), bbox_inches="tight")
    plt.close(fig)

# Build combined figure at the end using computed arrays
build_combined_figure_from_results(
    output_dir,
    sigma_list,
    coverage_list,
    coverage_std_list,
    JSD_list,
    JSD_std_list,
    TVD_list,
    TVD_std_list,
    coverage_plain,
    coverage_plain_std,
    JSD_plain,
    JSD_plain_std,
    TVD_plain,
    TVD_plain_std,
)
