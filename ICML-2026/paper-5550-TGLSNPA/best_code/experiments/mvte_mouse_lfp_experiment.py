"""
Estimate multivariate transfer entropy on mouse LFP data.

"""
__date__ = "September 2025"

import argparse
import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fit_mv_artg import fit_multivariate_artg_ssm
from src.imputation_model import get_lag_statistics
from src.lfp_loader import BlockPhaseLoader
from src.multivariate_transfer_entropy import estimate_mv_te
from src.plots import stats_to_colors, a1_to_cov_and_relation, a2_to_cov_and_relation



_REGIONS = [
    'Amy', 'Cg_Cx', 'DL_OFC', 'DL_Str', 'DM_Str', 'IL_Cx', 'LH', 'M1',
    'MD_Thal', 'NAc', 'PrL', 'S1', 'SNR', 'V1', 'VTA', 'aIns', 'dHipp', 'vHipp',
]


def _channel_region(ch):
    for r in _REGIONS:
        if ch == r or ch.startswith(r + '_'):
            return r
    return ch


def _save_labeled_te_plot(data, channels, vmin, vmax, cmap, title, fn):
    """Save a large labeled R×R TE plot with region labels and separators."""
    n_ch = len(channels)
    region_labels = [_channel_region(ch) for ch in channels]

    groups = []
    i = 0
    while i < n_ch:
        r = region_labels[i]
        j = i + 1
        while j < n_ch and region_labels[j] == r:
            j += 1
        groups.append((r, i, j))
        i = j

    tick_pos = [(s + e) / 2 - 0.5 for _, s, e in groups]
    tick_labels = [r for r, _, _ in groups]
    sep_pos = [e - 0.5 for _, _, e in groups[:-1]]

    size = 10
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(np.array(data), vmin=vmin, vmax=vmax, cmap=cmap, aspect='equal', interpolation='nearest')
    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.tick_params(length=2, pad=2)
    for pos in sep_pos:
        ax.axvline(pos, color='lightgray', linewidth=0.5)
        ax.axhline(pos, color='lightgray', linewidth=0.5)
    ax.set_title(title, fontsize=12)
    fig.savefig(fn, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved:", fn)


def _censor_te_within_regions(te, channels):
    """Set within-region entries to NaN in-place. te shape: (F, R, R)."""
    region_labels = [_channel_region(ch) for ch in channels]
    for region in _REGIONS:
        idx = [i for i, r in enumerate(region_labels) if r == region]
        if len(idx) > 1:
            te[:, idx[0]:idx[-1] + 1, idx[0]:idx[-1] + 1] = np.nan
    return te


def parse_args():
    parser = argparse.ArgumentParser(description="MVTE mouse command line parser")

    # Boolean flags
    parser.add_argument("--load", action="store_true", help="Whether to load phi")
    parser.add_argument("--get_stats", action="store_true", help="Whether to compute lag statistics")
    parser.add_argument("--train_ar", action="store_true", help="Whether to train AR model")
    parser.add_argument("--estimate_te", action="store_true", help="Whether to estimate transfer entropy")
    parser.add_argument("--plot", action="store_true", help="Regenerate all plots from saved files without training")

    # Integer argument
    parser.add_argument("--label_value", type=int, required=True, help="Label value to use")
    parser.add_argument(
        "--n_iter",
        type=int,
        default=None,
        help="If provided, overrides: stats num_batches, AR num_steps, TE max_num_batches.",
    )

    # Float argumemt
    parser.add_argument("--covar_reg", type=float, default=0.1, help="Add covar_reg * I to the covariance for imputation")

    return parser.parse_args()


def load_labels(fns, repeat_factor=500):
    labels = jnp.concatenate([jnp.load(fn) for fn in fns], 0)
    return jnp.repeat(labels, repeat_factor)


def main(args):
    # Unified override for iterations across modes
    N_STATS = args.n_iter if args.n_iter is not None else 100
    N_TRAIN = args.n_iter if args.n_iter is not None else 500
    N_TE    = args.n_iter if args.n_iter is not None else 100
    
    FS = 250
    LABEL_WINDOW_DURATION = 2
    L = 10
    F = 30
    F1, F2 = 5, 6
    assert F1 * F2 == F
    freqs = jnp.linspace(1,55,F)
    
    key = None # jax.random.PRNGKey(17)
    if key is None:
        key = jax.random.PRNGKey(np.random.randint(2**30))

    # output files
    out_dir = os.path.join(ROOT, "data", "mouse_mvte")
    w_out_fn = os.path.join(out_dir, f"sleep_{args.label_value}_W.npy")
    w_out_img_fn = os.path.join(out_dir, f"sleep_{args.label_value}_W.png")
    w_loss_fn = os.path.join(out_dir, f"sleep_{args.label_value}_w_loss.png")
    stats_out_fn = os.path.join(out_dir, f"sleep_{args.label_value}_artg_stats.npz")
    stats_out_img_fn = os.path.join(out_dir, f"sleep_{args.label_value}_artg_stats.png")
    te_out_fn = os.path.join(out_dir, f"sleep_{args.label_value}_TE.npz")
    te_out_img_fn = os.path.join(out_dir, f"sleep_{args.label_value}_TE.png")
    te_out_img_fn_2 = os.path.join(out_dir, f"sleep_{args.label_value}_TE_2.png")
    te_out_diff_img_fn = os.path.join(out_dir, "sleep_diff_TE.png")
    te_out_diff_img_fn_2 = os.path.join(out_dir, "sleep_diff_TE_2.png")

    if args.plot:
        _data_fn = os.path.join(ROOT, "data", "lfp_data", "torus_data.npz")
        all_channels = jnp.load(_data_fn)["channels"].tolist()
        freqs_np = np.array(freqs)
        idx_4  = int(np.argmin(np.abs(freqs_np - 4.0)))
        idx_18 = int(np.argmin(np.abs(freqs_np - 18.0)))
        idx_30 = int(np.argmin(np.abs(freqs_np - 30.0)))
        idx_45 = int(np.argmin(np.abs(freqs_np - 45.0)))

        # W (AR model) plots.
        if os.path.exists(w_out_fn):
            d = jnp.load(w_out_fn, allow_pickle=True).item()
            W_hat = d["w"]
            losses = jnp.array(d["losses"])
            R = int(np.round(np.sqrt(W_hat.size / (F * L * 4))))
            W_hat = W_hat.reshape(F, L, R, 2, R, 2)
            cov, rel = a1_to_cov_and_relation(W_hat)
            cov = jnp.transpose(cov, (2, 1, 3, 0)).reshape(R * L, F * R)
            rel = jnp.transpose(rel, (2, 1, 3, 0)).reshape(R * L, F * R)
            stats = jnp.concatenate([cov, rel], axis=1)
            r_max = jnp.quantile(jnp.abs(stats), 0.999)
            rgb = stats_to_colors(stats, r_max=r_max, mode="complex")
            rgb = (255 * rgb.clip(0, 1)).astype(jnp.uint8)
            Image.fromarray(np.array(rgb)).save(w_out_img_fn)
            print("Saved:", w_out_img_fn)
            plt.figure()
            plt.plot(losses)
            plt.ylabel("Loss")
            plt.xlabel("Batch")
            plt.savefig(w_loss_fn)
            plt.close()
            print("Saved:", w_loss_fn)
        else:
            print("No W file found:", w_out_fn)

        # Lag statistics plot.
        if os.path.exists(stats_out_fn):
            covars = jnp.load(stats_out_fn)["covars"]
            R = covars.shape[1] // (L * 2)
            covars = covars.reshape(F, R, L, 2, R, L, 2)
            cov, rel = a2_to_cov_and_relation(covars)
            cov = cov.reshape(F * R * L, R * L)
            rel = rel.reshape(F * R * L, R * L)
            stats = jnp.concatenate([cov, rel], axis=1)
            r_max = jnp.quantile(jnp.abs(stats), 0.999)
            rgb = stats_to_colors(stats, r_max=r_max, mode="complex")
            rgb = (255 * rgb.clip(0, 1)).astype(jnp.uint8)
            Image.fromarray(np.array(rgb)).save(stats_out_img_fn)
            print("Saved:", stats_out_img_fn)
        else:
            print("No stats file found:", stats_out_fn)

        # TE plots.
        if os.path.exists(te_out_fn):
            te = jnp.array(jnp.load(te_out_fn)["te"])
            R = te.shape[1]
            te = te.at[:, jnp.arange(R), jnp.arange(R)].set(0.0)
            flat_te = jnp.copy(te).reshape(F * R, R)
            te_max = 0.2
            flat_te = flat_te.clip(0.0, te_max)
            vmax = jnp.max(te)
            rgb = flat_te / vmax
            rgb = (255 * rgb.clip(0, 1)).astype(jnp.uint8)
            Image.fromarray(np.array(255 - rgb)).save(te_out_img_fn)
            print("Saved:", te_out_img_fn)
            for i in range(R):
                for j in range(R):
                    if i == j:
                        continue
                    plt.plot(freqs, te[:, i, j].clip(0, None), alpha=0.6)
            plt.ylabel("Transfer Entropy")
            plt.xlabel("Frequency (Hz)")
            plt.savefig(te_out_img_fn_2)
            plt.close("all")
            print("Saved:", te_out_img_fn_2)

            channels = all_channels[:R]
            te_np = _censor_te_within_regions(np.array(te), channels)
            for idx, hz in [(idx_4, 4), (idx_18, 18), (idx_30, 30), (idx_45, 45)]:
                labeled_fn = os.path.join(out_dir, f"sleep_{args.label_value}_TE_{hz}Hz_labeled.png")
                title = f"{'Wake' if args.label_value == 0 else 'NREM'} TE ({hz} Hz)"
                _save_labeled_te_plot(
                    te_np[idx], channels, 0, 0.0056, 'binary',
                    title, labeled_fn,
                )
        else:
            print("No TE file found:", te_out_fn)

        # TE diff plots.
        fn1 = os.path.join(out_dir, "sleep_0_TE.npz")
        fn2 = os.path.join(out_dir, "sleep_1_TE.npz")
        if os.path.exists(fn1) and os.path.exists(fn2):
            te_0 = jnp.array(jnp.load(fn1)["te"])
            te_1 = jnp.array(jnp.load(fn2)["te"])
            R = te_0.shape[1]
            te_0 = te_0.at[:, jnp.arange(R), jnp.arange(R)].set(0.0)
            te_1 = te_1.at[:, jnp.arange(R), jnp.arange(R)].set(0.0)
            te_diff = te_0 - te_1

            te_diff_2d = te_diff.reshape(F * R, R)
            vmax = jnp.quantile(jnp.abs(te_diff_2d), 0.99)
            normed = np.clip(np.array(te_diff_2d / vmax), -1, 1)
            cmap = plt.get_cmap("bwr")
            rgba_img = cmap((normed + 1) / 2.0)
            rgb_img = (rgba_img[:, :, :3] * 255).astype(np.uint8)
            Image.fromarray(rgb_img).save(te_out_diff_img_fn)
            print("Saved:", te_out_diff_img_fn)

            for i in range(R):
                for j in range(R):
                    if i == j:
                        continue
                    plt.plot(freqs, te_diff[:, i, j], alpha=0.6)
            plt.ylabel("Transfer Entropy Difference")
            plt.xlabel("Frequency (Hz)")
            plt.tight_layout()
            plt.savefig(te_out_diff_img_fn_2)
            plt.close("all")
            print("Saved:", te_out_diff_img_fn_2)

            channels = all_channels[:R]
            te_diff_np = _censor_te_within_regions(np.array(te_diff), channels)
            for idx, hz in [(idx_4, 4), (idx_18, 18), (idx_30, 30), (idx_45, 45)]:
                labeled_fn = os.path.join(out_dir, f"sleep_diff_TE_{hz}Hz_labeled.png")
                title = f"Wake TE - NREM TE ({hz} Hz)"
                _save_labeled_te_plot(
                    te_diff_np[idx], channels, -0.002, 0.002, 'bwr',
                    title, labeled_fn,
                )
        else:
            print("No TE diff files found:", fn1, fn2)

        return

    # Where the LFP data is stored.
    data_fn = os.path.join(ROOT, "data", "lfp_data", "torus_data.npz")
    data_dir = os.path.split(data_fn)[0]
    label_fns = [f"Mouse04_051419_Sleep_{i:02d}.npy" for i in range(1,49)]
    label_fns = [os.path.join(data_dir, i) for i in label_fns]
    labels = load_labels(label_fns, repeat_factor=int(round(FS * LABEL_WINDOW_DURATION)))

    # Load channels and LFPs.
    # all_channels = jnp.load(data_fn)["channels"].tolist()
    lfps = jnp.load(data_fn)["lfps"].astype(jnp.float16)

    roi_indices = [0,28,31,37,49] # Amy, MdThal, NAc, PrL, VTA
    roi_indices = jnp.arange(lfps.shape[1])
    R = len(roi_indices)

    lfps = lfps[:,roi_indices]
    print("Mouse data shape:", lfps.shape)  

    key, sub = jax.random.split(key)
    loader = BlockPhaseLoader(
        lfps,
        fs=FS,
        freqs=freqs,
        window_length_s=2,
        batch_size=64,
        L=L,
        key=sub,
        labels=labels,
        num_batches=N_STATS if args.get_stats else None,
        label_value=args.label_value,
        shuffle=True,
    )

    # Get the lag statistics for the imputation model.
    if args.get_stats:
        prev_means = None
        prev_covars = None
        if args.load:
            d = jnp.load(stats_out_fn)
            prev_means = d["means"]
            prev_covars = d["covars"]
            prev_counts = d["counts"]
            del d

        # Get stats.
        means, covars, counts = get_lag_statistics(loader, L)

        # Add to previously saved stats.
        if prev_means is not None:
            ExxT  = covars + jnp.einsum('fd,fe->fde', means, means)
            prev_ExxT = prev_covars + jnp.einsum('fd,fe->fde', prev_means, prev_means)

            p = counts / (prev_counts + counts)
            ExxT = p * ExxT + (1.0 - p) * prev_ExxT
            means = p * means + (1.0 - p) * prev_means

            print(f"Mean diff: {jnp.linalg.norm(means - prev_means)}")
            print(f"ExxT diff: {jnp.linalg.norm(ExxT - prev_ExxT)}")

            covars = ExxT - jnp.einsum('fd,fe->fde', means, means)
            counts += prev_counts

        jnp.savez(stats_out_fn, means=means.astype(jnp.float16), covars=covars.astype(jnp.float16), counts=counts)
        print("Saved:", stats_out_fn)
        
        # Plot stats.
        covars = jnp.load(stats_out_fn)["covars"]
        covars = covars.reshape(F, R, L, 2, R, L, 2)
        cov, rel = a2_to_cov_and_relation(covars) # both [F, R, L, R, L]
        cov = cov.reshape(F * R*L, R*L)
        rel = rel.reshape(F * R*L, R*L)
        # Top-to-bottom: increasing frequency
        # Left covariance, right relation
        # Intermediate blocks: regions
        # Smallest blocks: lags
        stats = jnp.concatenate([cov, rel], axis=1) # [F*R*L, 2*R*L]
        r_max = jnp.quantile(jnp.abs(stats), 0.999)
        print("r_max:", r_max)
        rgb = stats_to_colors(stats, r_max=r_max, mode="complex")
        print("rgb", rgb.shape)
        rgb = (255 * rgb.clip(0,1)).astype(jnp.uint8)
        img = Image.fromarray(np.array(rgb))
        img.save(stats_out_img_fn)
        print("Saved:", stats_out_img_fn)
        quit()


    # Train the autoregressive model.
    if args.train_ar:
        prev_w = None
        opt_state = None
        prev_losses = None
        if args.load:
            try:
                d = jnp.load(w_out_fn, allow_pickle=True).item()
                prev_w = d["w"]
                opt_state = d["opt_state"]
                prev_losses = d["losses"]
            except:
                print("Failed to load!")
                quit()

        key, sub = jax.random.split(key)
        W_hat, opt_params, losses = fit_multivariate_artg_ssm(
            sub,
            loader,
            F,
            R,
            L,
            W=prev_w,
            opt_state=opt_state,
            lr=3e-3,
            num_steps=N_TRAIN,
        )

        # Concatenate losses.
        losses = jnp.array(losses)
        if prev_losses is not None:
            prev_losses = jnp.array(prev_losses)
            losses = jnp.concatenate([prev_losses, losses], 0)

        # Save the checkpoint.
        jnp.save(w_out_fn, dict(w=W_hat, opt_state=opt_params, losses=losses))
        print("Saved:", w_out_fn)

        # Print the difference from the saved W.
        if prev_w is not None:
            print(f"W diff: {jnp.linalg.norm(W_hat - prev_w):.4f}")
            del prev_w

        # Plot W.
        # Left: covariance, right: relation
        # Sub-blocks: regions by regions
        # Sub-block vertical: lags
        # Sub-block horizontal: frequencies
        W_hat = W_hat.reshape(F, L, R, 2, R, 2)
        cov, rel = a1_to_cov_and_relation(W_hat) # both [F, L, R, R]
        print(cov.shape)
        cov = jnp.transpose(cov, (2,1,3,0)).reshape(R*L,F*R) # [RL, RF]
        rel = jnp.transpose(rel, (2,1,3,0)).reshape(R*L,F*R) # [RL, RF]
        stats = jnp.concatenate([cov, rel], axis=1) # [R*L, 2*R*F]
        r_max = jnp.quantile(jnp.abs(stats), 0.999)
        print("r_max:", r_max)
        rgb = stats_to_colors(stats, r_max=r_max, mode="complex")
        print("rgb", rgb.shape)
        rgb = (255 * rgb.clip(0,1)).astype(jnp.uint8)
        img = Image.fromarray(np.array(rgb))
        img.save(w_out_img_fn)
        print("Saved:", w_out_img_fn)

        # Plot loss.
        plt.plot(losses)
        plt.ylabel("Loss")
        plt.xlabel("Batch")
        plt.savefig(w_loss_fn)
        print(f"Saved:", w_loss_fn)
        quit()


    # Estimate transfer entropy using the imputation and AR models.
    if args.estimate_te:
        # Load the estimated W.
        W_hat = jnp.load(w_out_fn, allow_pickle=True).item()["w"]

        # Load lagged window stats.
        d = jnp.load(stats_out_fn)
        means, covars = d["means"].astype(jnp.float32), d["covars"].astype(jnp.float32)
        covars = covars + args.covar_reg * jnp.eye(R * L * 2)[None] # Add regularization
        del d

        # Load previous TE and its batch count if available.
        prev_te = None
        prev_batches = 0
        if args.load:
            try:
                d = jnp.load(te_out_fn, allow_pickle=True)
                prev_te = d["te"]
                prev_counts = d["counts"]
                del d
            except:
                print("Failed to load!")
                quit()

        # Estimate the transfer entropy.
        counts = N_TE
        key, sub = jax.random.split(key)
        te = estimate_mv_te(
            sub,
            loader,
            W_hat,
            means,
            covars,
            R,
            L,
            F,
            max_num_batches=N_TE,
            show_progress=True,
            K=8,
        )

        # Weighted average with previous TE by number of batches
        if prev_te is not None:
            p = counts / (prev_counts + counts)
            te = p * te + (1.0 - p) * prev_te
            print(f"TE rel. diff: {jnp.linalg.norm(te - prev_te) / jnp.linalg.norm(te):.4f}")
            counts = counts + prev_counts

        print("NaN portion:", jnp.isnan(te).sum() / te.size)
        te = te.at[jnp.isnan(te)].set(0.0)

        # Save TE.
        jnp.savez(te_out_fn, te=te, counts=counts)
        print(f"Saved: {te_out_fn} with {counts} batches.")
        print(te.shape)

        # Plot.
        te = te.at[:,jnp.arange(R),jnp.arange(R)].set(0.0)
        flat_te = jnp.copy(te).reshape(F*R, R) # [FR, R]
        # te_min = 0.001
        te_max = 0.2
        flat_te = flat_te.clip(0.0, te_max)
        # flat_te = (jnp.log(flat_te.clip(te_min, te_max)) - jnp.log(te_min)) / (jnp.log(te_max) - jnp.log(te_min))
        vmax = jnp.max(te)
        print('vmax;', vmax)
        rgb = flat_te / vmax
        rgb = (255 * rgb.clip(0,1)).astype(jnp.uint8)
        img = Image.fromarray(np.array(255 - rgb))
        img.save(te_out_img_fn)
        print("Saved:", te_out_img_fn)

        # Plot all pairs.
        for i in range(R):
            for j in range(R):
                if i == j:
                    continue
                plt.plot(freqs, te[:,i,j].clip(0, None), alpha=0.6)
        plt.ylabel("Transfer Entropy")
        plt.xlabel("Frequency (Hz)")
        plt.savefig(te_out_img_fn_2)
        plt.close("all")
        print("Saved:", te_out_img_fn_2)

    # Plot differences between sleep and wake states.
    fn1, fn2 = os.path.join(out_dir, "sleep_0_TE.npz"), os.path.join(out_dir, "sleep_1_TE.npz")
    if os.path.exists(fn1) and os.path.exists(fn2):
        te_0 = jnp.array(jnp.load(fn1)["te"])
        te_0 = te_0.at[:, jnp.arange(R), jnp.arange(R)].set(0.0)
        te_0 = te_0.reshape(F * R, R)  # [FR, R]

        te_1 = jnp.array(jnp.load(fn2)["te"])
        te_1 = te_1.at[:, jnp.arange(R), jnp.arange(R)].set(0.0)
        te_1 = te_1.reshape(F * R, R)  # [FR, R]

        te_diff = te_0 - te_1

        # Symmetric vmax for diverging colormap
        vmax = jnp.quantile(jnp.abs(te_diff), 0.99)
        print("vmax:", vmax)

        # Normalize to [-1,1]
        normed = np.array(te_diff / vmax)
        normed = np.clip(normed, -1, 1)

        # Map to colormap (returns RGBA in [0,1])
        cmap = plt.get_cmap("bwr")
        rgba_img = cmap((normed + 1) / 2.0)  # rescale [-1,1] → [0,1]

        # Convert to uint8 RGB
        rgb_img = (rgba_img[:, :, :3] * 255).astype(np.uint8)

        img = Image.fromarray(rgb_img)
        img.save(te_out_diff_img_fn)
        print("Saved:", te_out_diff_img_fn)


    fn1, fn2 = os.path.join(out_dir, "sleep_0_TE.npz"), os.path.join(out_dir, "sleep_1_TE.npz")
    if os.path.exists(fn1) and os.path.exists(fn2):
        te_0 = jnp.array(jnp.load(fn1)["te"])
        te_0 = te_0.at[:, jnp.arange(R), jnp.arange(R)].set(0.0)

        te_1 = jnp.array(jnp.load(fn2)["te"])
        te_1 = te_1.at[:, jnp.arange(R), jnp.arange(R)].set(0.0)

        te_diff = te_0 - te_1
        # Plot all pairs.
        for i in range(R):
            for j in range(R):
                if i == j:
                    continue
                plt.plot(freqs, te_diff[:,i,j], alpha=0.6)
        plt.ylabel("Transfer Entropy Difference")
        plt.xlabel("Frequency (Hz)")
        plt.tight_layout()
        plt.savefig(te_out_diff_img_fn_2)
        plt.close("all")
        print("Saved:", te_out_diff_img_fn_2)



if __name__ == "__main__":
    args = parse_args()
    print("load:", args.load)
    print("get_stats:", args.get_stats)
    print("train_ar:", args.train_ar)
    print("estimate_te:", args.estimate_te)
    print("plot:", args.plot)
    print("label_value:", args.label_value)
    print("n_iter:", args.n_iter)
    print("covar_reg:", args.covar_reg)
    main(args)
