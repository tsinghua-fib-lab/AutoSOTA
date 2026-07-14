"""
Fit large TG models using mouse data.

"""
__date__ = "July - September 2025"

import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lfp_loader import SinglePhaseLoader
from src.plots import stats_to_colors
from src.ssm import estimate_params_ssm_with_loader
from src.stats import get_average_stats_with_loader


def parse_args():
    parser = argparse.ArgumentParser(description="SGD mouse command line parser")

    # Boolean flags
    parser.add_argument("--load", action="store_true", help="Whether to load phi")
    parser.add_argument("--get_stats", action="store_true", help="Whether to compute statistics")
    parser.add_argument("--plot", action="store_true", help="Regenerate all plots from saved files without training")

    # Integer argument
    parser.add_argument("--label_value", type=int, required=True, help="Label value to use")
    parser.add_argument(
        "--n_iter",
        type=int,
        default=None,
        help="If provided, overrides: stats num_batches, optimizer num_steps.",
    )

    return parser.parse_args()


def load_labels(fns, repeat_factor=500):
    labels = jnp.concatenate([jnp.load(fn) for fn in fns], 0)
    return jnp.repeat(labels, repeat_factor)


_REGIONS = [
    'Amy', 'Cg_Cx', 'DL_OFC', 'DL_Str', 'DM_Str', 'IL_Cx', 'LH', 'M1',
    'MD_Thal', 'NAc', 'PrL', 'S1', 'SNR', 'V1', 'VTA', 'aIns', 'dHipp', 'vHipp',
]

def _channel_region(ch):
    for r in _REGIONS:
        if ch == r or ch.startswith(r + '_'):
            return r
    return ch


def _save_labeled_matrix_plot(rgb, channels, F, title, fn):
    """Save a large labeled matplotlib figure of an (R*F)×(R*F) color matrix."""
    n_ch = len(channels)
    region_labels = [_channel_region(ch) for ch in channels]

    # Build contiguous region groups: (name, ch_start, ch_end_exclusive)
    groups = []
    i = 0
    while i < n_ch:
        r = region_labels[i]
        j = i + 1
        while j < n_ch and region_labels[j] == r:
            j += 1
        groups.append((r, i, j))
        i = j

    # Region centers and separator positions in pixel space
    tick_pos = [(s + e) / 2 * F - 0.5 for _, s, e in groups]
    tick_labels = [r for r, _, _ in groups]
    sep_pos = [e * F - 0.5 for _, _, e in groups[:-1]]

    size = 10
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(np.array(rgb), aspect="equal")

    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.tick_params(length=2, pad=2)

    for pos in sep_pos:
        ax.axvline(pos, color='lightgray', linewidth=0.5)
        ax.axhline(pos, color='lightgray', linewidth=0.5)

    ax.set_title(title, fontsize=12)
    fig.savefig(fn, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", fn)


if __name__ == '__main__':
    args = parse_args()
    print("load:", args.load)
    print("get_stats:", args.get_stats)
    print("label_value:", args.label_value)
    print("n_iter:", args.n_iter)

    # Unified override for iterations across modes; fallback to current defaults
    N_STATS = args.n_iter if args.n_iter is not None else 40
    N_TRAIN = args.n_iter if args.n_iter is not None else 500

    FS = 250
    LABEL_WINDOW_DURATION = 2
    key = None # jax.random.PRNGKey(17)

    if key is None:
        key = jax.random.PRNGKey(np.random.randint(2**30))

    
    # Where the LFP data is stored.
    data_fn = os.path.join(ROOT, "data", "lfp_data", "torus_data.npz")
    data_dir = os.path.split(data_fn)[0]
    label_fns = [f"Mouse04_051419_Sleep_{i:02d}.npy" for i in range(1,49)]
    label_fns = [os.path.join(data_dir, i) for i in label_fns]
    labels = load_labels(label_fns, repeat_factor=int(round(FS * LABEL_WINDOW_DURATION)))

    out_dir = os.path.join(ROOT, "data", "mouse_tg")
    phi_out_fn = os.path.join(out_dir, f"sleep_{args.label_value}_sgd_phi.npy")
    phi_out_img_fn = os.path.join(out_dir, f"sleep_{args.label_value}_sgd_phi.png")
    phi_diff_out_img_fn = os.path.join(out_dir, "sleep_diff_sgd_phi.png")
    stats_out_fn = os.path.join(out_dir, f"sleep_{args.label_value}_sgd_stats.npz")
    stats_out_img_fn = os.path.join(out_dir, f"sleep_{args.label_value}_sgd_stats.png")
    stats_diff_out_img_fn = os.path.join(out_dir, "sleep_diff_sgd_stats.png")
    loss_fn = os.path.join(out_dir, f"phi_{args.label_value}_sgd_loss.png")

    F = 30

    if args.plot:
        all_channels = jnp.load(data_fn)["channels"].tolist()

        # Plot phi and loss from saved phi file.
        if os.path.exists(phi_out_fn):
            d = jnp.load(phi_out_fn, allow_pickle=True).item()
            phi = d["phi"]
            losses = jnp.array(d["losses"])
            r_max = jnp.quantile(jnp.abs(phi[..., 0] + 1j * phi[..., 1]), 0.999)
            rgb = stats_to_colors(phi, r_max=r_max, mode="expanded_complex")
            rgb = (255 * rgb.clip(0, 1)).astype(jnp.uint8)
            Image.fromarray(np.array(rgb)).save(phi_out_img_fn)
            print("Saved:", phi_out_img_fn)
            channels = all_channels[: phi.shape[0] // F]
            phi_labeled_fn = phi_out_img_fn.replace(".png", "_labeled.png")
            title = rf"{'Wake' if args.label_value == 0 else 'NREM'} $\phi$"
            _save_labeled_matrix_plot(rgb, channels, F, title, phi_labeled_fn)
            plt.figure()
            plt.plot(losses)
            plt.ylabel("Loss")
            plt.xlabel("Batch")
            plt.savefig(loss_fn)
            plt.close()
            print("Saved:", loss_fn)
        else:
            print("No phi file found:", phi_out_fn)

        # Plot stats from saved stats file.
        if os.path.exists(stats_out_fn):
            stats = jnp.load(stats_out_fn)["stats"]
            rgb = stats_to_colors(stats, mode="expanded_complex")
            rgb = (255 * rgb.clip(0, 1)).astype(jnp.uint8)
            Image.fromarray(np.array(rgb)).save(stats_out_img_fn)
            print("Saved:", stats_out_img_fn)
            channels = all_channels[: stats.shape[0] // F]
            stats_labeled_fn = stats_out_img_fn.replace(".png", "_labeled.png")
            title = rf"{'Wake' if args.label_value == 0 else 'NREM'} $S$"
            _save_labeled_matrix_plot(rgb, channels, F, title, stats_labeled_fn)
        else:
            print("No stats file found:", stats_out_fn)

        # Plot phi diff.
        fn0 = os.path.join(out_dir, "sleep_0_sgd_phi.npy")
        fn1 = os.path.join(out_dir, "sleep_1_sgd_phi.npy")
        if os.path.exists(fn0) and os.path.exists(fn1):
            phi_0 = jnp.load(fn0, allow_pickle=True).item()["phi"]
            phi_1 = jnp.load(fn1, allow_pickle=True).item()["phi"]
            phi_diff = phi_0 - phi_1
            r_max = jnp.quantile(jnp.abs(phi_diff[..., 0] + 1j * phi_diff[..., 1]), 0.999)
            rgb = stats_to_colors(phi_diff, r_max=r_max, mode="expanded_complex")
            rgb = (255 * rgb.clip(0, 1)).astype(jnp.uint8)
            Image.fromarray(np.array(rgb)).save(phi_diff_out_img_fn)
            print("Saved:", phi_diff_out_img_fn)
            channels = all_channels[: phi_diff.shape[0] // F]
            phi_diff_labeled_fn = phi_diff_out_img_fn.replace(".png", "_labeled.png")
            title = rf"$\phi$ Diff. (Wake - NREM)"
            _save_labeled_matrix_plot(rgb, channels, F, title, phi_diff_labeled_fn)
        else:
            print("No phi diff files found:", fn0, fn1)

        # Plot stats diff.
        fn0 = os.path.join(out_dir, "sleep_0_sgd_stats.npz")
        fn1 = os.path.join(out_dir, "sleep_1_sgd_stats.npz")
        if os.path.exists(fn0) and os.path.exists(fn1):
            stats_0 = jnp.load(fn0)["stats"]
            stats_1 = jnp.load(fn1)["stats"]
            stats_diff = stats_0 - stats_1
            r_max = jnp.quantile(jnp.abs(stats_diff[..., 0] + 1j * stats_diff[..., 1]), 0.999)
            rgb = stats_to_colors(stats_diff, r_max=r_max, mode="expanded_complex")
            rgb = (255 * rgb.clip(0, 1)).astype(jnp.uint8)
            Image.fromarray(np.array(rgb)).save(stats_diff_out_img_fn)
            print("Saved:", stats_diff_out_img_fn)
            channels = all_channels[: stats_diff.shape[0] // F]
            stats_diff_labeled_fn = stats_diff_out_img_fn.replace(".png", "_labeled.png")
            title = rf"$S$ Diff. (Wake - NREM)"
            _save_labeled_matrix_plot(rgb, channels, F, title, stats_diff_labeled_fn)
        else:
            print("No stats diff files found:", fn0, fn1)

        quit()

    all_channels = jnp.load(data_fn)["channels"].tolist()
    mouse_data = jnp.load(data_fn)["lfps"].astype(jnp.float16)

    # roi_indices = [0,28,31,37,49] # Amy, MdThal, NAc, PrL, VTA
    roi_indices = jnp.arange(mouse_data.shape[1])
    R = len(roi_indices)

    mouse_data = mouse_data[:,roi_indices]
    print("Mouse data shape:", mouse_data.shape)

    key1, key2 = jax.random.split(key)

    loader = SinglePhaseLoader(
        mouse_data,
        fs=FS,
        freqs=jnp.linspace(1,55,F),
        window_length_s=2,
        batch_size=32,
        key=key1,
        labels=labels,
        label_value=args.label_value,
        shuffle=True,
    )

    # Collect the TG statistics for the sleep state.
    if args.get_stats:
        if args.load:
            d = jnp.load(stats_out_fn)
            prev_stats, prev_counts = d["stats"], d["counts"]
        else:
            prev_stats, prev_counts = 0, 0
        stats = get_average_stats_with_loader(loader, n_iter=N_STATS)

        w = N_STATS / (N_STATS + prev_counts)
        stats = w * stats + (1 - w) * prev_stats
        counts = N_STATS + prev_counts
        print(f"Combining {prev_counts} saved with {N_STATS} new batches.")
        print(f"Diff: {jnp.linalg.norm(stats - prev_stats):.4f}")
        jnp.savez(stats_out_fn, stats=stats, counts=counts)
        print("Saved:", stats_out_fn)

        rgb = stats_to_colors(stats, mode="expanded_complex")
        rgb = (255 * rgb.clip(0,1)).astype(jnp.uint8)
        img = Image.fromarray(np.array(rgb))
        img.save(stats_out_img_fn)
        print("Saved:", stats_out_img_fn)
        quit()


    # Train phi for the sleep state.
    prev_phi = None
    opt_state = None
    prev_losses = None
    if args.load:
        try:
            d = jnp.load(phi_out_fn, allow_pickle=True).item()
            prev_phi = d["phi"]
            opt_state = d["opt_state"]
            prev_losses = d["losses"]
        except Exception as e:
            print("Failed to load:", e)
            quit()

    phi, opt_state, losses = estimate_params_ssm_with_loader(
        key2,
        loader,
        phi=prev_phi,
        batch_size=32,
        n_iter=N_TRAIN,
        alpha=0.99,
        opt_state=opt_state,
        l2_reg=0.1,
        l1_reg=0.0,
        replace=True,
        transition_steps=10000,
        lr=3e-3,
    )

    if prev_phi is not None:
        print(f"Phi diff: {jnp.linalg.norm(phi - prev_phi):.4f}")
        del prev_phi

    losses = jnp.array(losses)
    if prev_losses is not None:
        prev_losses = jnp.array(prev_losses)
        losses = jnp.concatenate([prev_losses, losses], 0)

    jnp.save(phi_out_fn, dict(phi=phi, opt_state=opt_state, losses=losses))
    print("Saved:", phi_out_fn)

    # Plot phi.
    r_max = jnp.quantile(jnp.abs(phi[...,0] + 1j * phi[...,1]), 0.999)
    print("r_max", r_max)
    rgb = stats_to_colors(phi, r_max=r_max, mode="expanded_complex")
    rgb = (255 * rgb.clip(0,1)).astype(jnp.uint8)
    img = Image.fromarray(np.array(rgb))
    img.save(phi_out_img_fn)
    print("Saved:", phi_out_img_fn)

    # Plot phi diff.
    fn0 = os.path.join(out_dir, "sleep_0_sgd_phi.npy")
    fn1 = os.path.join(out_dir, "sleep_1_sgd_phi.npy")
    if os.path.exists(fn0) and os.path.exists(fn1):
        phi_0 = jnp.load(fn0, allow_pickle=True).item()["phi"]
        phi_1 = jnp.load(fn1, allow_pickle=True).item()["phi"]
        phi_diff = phi_0 - phi_1
        r_max = jnp.quantile(jnp.abs(phi_diff[...,0] + 1j * phi_diff[...,1]), 0.999)
        # r_max = jnp.max(jnp.abs(phi_diff[...,0] + 1j * phi_diff[...,1]))
        print("r_max", r_max)
        rgb = stats_to_colors(phi_diff, r_max=r_max, mode="expanded_complex")
        rgb = (255 * rgb.clip(0,1)).astype(jnp.uint8)
        img = Image.fromarray(np.array(rgb))
        img.save(phi_diff_out_img_fn)
        print("Saved:", phi_diff_out_img_fn)


    # Plot stats diff.
    fn0, fn1 = "sleep_0_sgd_stats.npz", "sleep_1_sgd_stats.npz"
   
    if os.path.exists(fn0) and os.path.exists(fn1):
        stats_0 = jnp.load(fn0)["stats"]
        stats_1 = jnp.load(fn1)["stats"]
        stats_diff = stats_0 - stats_1
        r_max = jnp.quantile(jnp.abs(stats_diff[...,0] + 1j * stats_diff[...,1]), 0.999)
        print("r_max", r_max)
        rgb = stats_to_colors(stats_diff, r_max=r_max, mode="expanded_complex")
        rgb = (255 * rgb.clip(0,1)).astype(jnp.uint8)
        img = Image.fromarray(np.array(rgb))
        img.save(stats_diff_out_img_fn)
        print("Saved:", stats_diff_out_img_fn)


    # Plot loss.
    plt.plot(losses)
    plt.ylabel("Loss")
    plt.xlabel("Batch")
    plt.savefig(loss_fn)
    print("Saved:", loss_fn)

###