"""
Generate response matrix heatmaps with item parameter histograms and
Test Information Function (TIF) to demonstrate benchmark homogeneity.

This addresses Sanmi's comment #12: proving that BoolQ and HellaSwag
underperform because the benchmarks are homogeneous (low item parameter
diversity), not because the IRT method fails.

Usage:
    conda run -n irsl python plot_resmat_heatmap.py

Outputs:
    results/resmat_heatmap_boolq_hellaswag_vs_arc.png  (3-panel)
    results/resmat_heatmap_comparison.png               (6-panel)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.special import expit
from pathlib import Path

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
})

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "results"


def test_information_function(theta_grid, z, d):
    """
    Compute the Test Information Function I(theta) for a 2PL IRT model.

    I(theta) = sum_j d_j^2 * p_j(theta) * (1 - p_j(theta))
    where p_j(theta) = sigmoid(d_j * (theta - z_j))

    Parameters
    ----------
    theta_grid : array of shape (G,)
    z : array of shape (N,) — item difficulties
    d : array of shape (N,) — item discriminations

    Returns
    -------
    array of shape (G,) — total information at each theta
    """
    p = expit(d[None, :] * (theta_grid[:, None] - z[None, :]))
    info_per_item = d[None, :] ** 2 * p * (1 - p)
    return info_per_item.sum(axis=1)


def load_data():
    print("Loading prob matrix...")
    prob_df = pd.read_parquet(DATA_DIR / "4_prob_matrix_calibrated_2pl.parquet")

    train_df = prob_df[prob_df.index.get_level_values("model_split") == "train"]
    bench_names = train_df.columns.get_level_values('bench_name').map(
        lambda b: 'mmlu' if b.startswith('mmlu') else b
    )
    difficulties = train_df.columns.get_level_values('difficulty').to_numpy(dtype=np.float32)
    discriminations = train_df.columns.get_level_values('discrimination').to_numpy(dtype=np.float32)
    idx_df = train_df.index.to_frame().reset_index(drop=True)

    return train_df, bench_names, difficulties, discriminations, idx_df


def get_bench_data(bench, train_df, bench_names, difficulties, discriminations, idx_df):
    mask = bench_names == bench
    bench_data = train_df.iloc[:, mask].to_numpy(dtype=np.float32)
    bench_z = difficulties[mask]
    bench_d = discriminations[mask]

    thetas = idx_df[f'ability_{bench}'].to_numpy(dtype=np.float32)

    row_means = np.nanmean(bench_data, axis=1)
    row_order = np.argsort(row_means)
    bench_data = bench_data[row_order, :]

    col_order = np.argsort(bench_z)
    bench_data = bench_data[:, col_order]

    n_models, n_items = bench_data.shape
    if n_items > 500:
        step = max(1, n_items // 500)
        bench_data = bench_data[:, ::step]

    return bench_data, bench_z, bench_d, mask.sum(), thetas


def plot_panel(fig, gs, col_idx, bench, label, train_df, bench_names,
               difficulties, discriminations, idx_df,
               norm, cmap, z_range, d_range, theta_grid, tif_norm, tif_norm_max):
    data, z, d, n, thetas = get_bench_data(
        bench, train_df, bench_names, difficulties, discriminations, idx_df
    )

    # Row 0: Heatmap
    ax_heat = fig.add_subplot(gs[0, col_idx])
    im = ax_heat.imshow(data, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
    ax_heat.set_title(
        rf'\textbf{{{label}}} ({n} items)'
        '\n'
        rf'$\sigma_z={z.std():.2f},\;\sigma_d={d.std():.2f}$',
        fontsize=14,
    )
    ax_heat.set_xlabel(r'Questions (sorted by $z$)', fontsize=11)
    if col_idx == 0:
        ax_heat.set_ylabel(r'Models (sorted by mean $p_{\mathrm{CC}}$)', fontsize=11)
    ax_heat.tick_params(labelsize=9)

    # Row 1: Histogram of z
    ax_z = fig.add_subplot(gs[1, col_idx])
    ax_z.hist(z, bins=40, color='tab:blue', alpha=0.8, edgecolor='white', linewidth=0.3, range=z_range)
    ax_z.set_xlabel(r'Difficulty $z$', fontsize=11)
    if col_idx == 0:
        ax_z.set_ylabel('Count', fontsize=11)
    ax_z.set_xlim(z_range)
    ax_z.tick_params(labelsize=9)

    # Row 2: Histogram of d
    ax_d = fig.add_subplot(gs[2, col_idx])
    ax_d.hist(d, bins=40, color='tab:orange', alpha=0.8, edgecolor='white', linewidth=0.3, range=d_range)
    ax_d.set_xlabel(r'Discrimination $d$', fontsize=11)
    if col_idx == 0:
        ax_d.set_ylabel('Count', fontsize=11)
    ax_d.set_xlim(d_range)
    ax_d.tick_params(labelsize=9)

    # Row 3: Test Information Function
    ax_tif = fig.add_subplot(gs[3, col_idx])
    ax_tif.fill_between(theta_grid, tif_norm, alpha=0.3, color='tab:green')
    ax_tif.plot(theta_grid, tif_norm, color='tab:green', linewidth=1.5)
    theta_lo, theta_hi = np.percentile(thetas, [5, 95])
    ax_tif.axvspan(theta_lo, theta_hi, alpha=0.15, color='gray',
                   label=rf'Model $\theta$ range (5--95\%)')
    ax_tif.set_xlabel(r'Ability $\theta$', fontsize=11)
    if col_idx == 0:
        ax_tif.set_ylabel(r'$I(\theta)\,/\,N_{\mathrm{items}}$', fontsize=11)
    ax_tif.set_xlim(theta_grid[0], theta_grid[-1])
    ax_tif.set_ylim(0, tif_norm_max * 1.1)
    ax_tif.legend(fontsize=9, loc='upper right')
    ax_tif.tick_params(labelsize=9)

    peak_theta = theta_grid[np.argmax(tif_norm)]
    fwhm_mask = tif_norm >= tif_norm.max() / 2
    fwhm_thetas = theta_grid[fwhm_mask]
    fwhm = fwhm_thetas[-1] - fwhm_thetas[0] if len(fwhm_thetas) > 1 else 0
    print(f"  {label:<15} peak_I/N={tif_norm.max():.3f}  FWHM={fwhm:.2f}  "
          f"model_theta=[{theta_lo:.2f}, {theta_hi:.2f}]")

    return im


def main():
    train_df, bench_names, difficulties, discriminations, idx_df = load_data()

    # ── 3-panel figure ───────────────────────────────────────────────
    panels = [
        ('boolq',         'BoolQ'),
        ('hellaswag',     'HellaSwag'),
        ('arc_challenge', 'ARC-Challenge'),
    ]

    fig = plt.figure(figsize=(18, 13.5))
    gs = gridspec.GridSpec(
        4, 4, figure=fig,
        height_ratios=[4, 1.3, 1.3, 1.8],
        width_ratios=[1, 1, 1, 0.05],
        hspace=0.42, wspace=0.30,
    )
    norm = Normalize(vmin=0, vmax=1)
    cmap = LinearSegmentedColormap.from_list('RedBlue', ['tab:red', 'tab:blue'])

    # Compute shared ranges
    all_z, all_d, all_thetas = [], [], []
    for bench, _ in panels:
        _, z, d, _, thetas = get_bench_data(
            bench, train_df, bench_names, difficulties, discriminations, idx_df
        )
        all_z.append(z); all_d.append(d); all_thetas.append(thetas)
    z_range = (min(a.min() for a in all_z), max(a.max() for a in all_z))
    d_range = (min(a.min() for a in all_d), max(a.max() for a in all_d))

    theta_min = min(t.min() for t in all_thetas) - 1.0
    theta_max = max(t.max() for t in all_thetas) + 1.0
    theta_grid = np.linspace(theta_min, theta_max, 500).astype(np.float32)

    all_tif_norm = []
    for bench, _ in panels:
        _, z, d, n, _ = get_bench_data(
            bench, train_df, bench_names, difficulties, discriminations, idx_df
        )
        tif = test_information_function(theta_grid, z, d)
        all_tif_norm.append(tif / n)
    tif_norm_max = max(t.max() for t in all_tif_norm)

    print("3-panel figure:")
    for col_idx, (bench, label) in enumerate(panels):
        im = plot_panel(
            fig, gs, col_idx, bench, label,
            train_df, bench_names, difficulties, discriminations, idx_df,
            norm, cmap, z_range, d_range, theta_grid,
            all_tif_norm[col_idx], tif_norm_max,
        )

    cbar_ax = fig.add_subplot(gs[0, 3])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax, label=r'$p_{\mathrm{Correct\;Choice}}$',
    )

    out_path = OUT_DIR / "resmat_heatmap_boolq_hellaswag_vs_arc.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"  Saved to {out_path}\n")
    plt.close(fig)

    # ── 6-panel figure ───────────────────────────────────────────────
    panels_6 = [
        ('boolq',         'BoolQ'),
        ('hellaswag',     'HellaSwag'),
        ('arc_challenge', 'ARC-Challenge'),
        ('piqa',          'PIQA'),
        ('csqa',          'CSQA'),
        ('mmlu',          'MMLU'),
    ]

    fig2 = plt.figure(figsize=(18, 27))
    gs2 = gridspec.GridSpec(
        8, 4, figure=fig2,
        height_ratios=[4, 1.3, 1.3, 1.8, 4, 1.3, 1.3, 1.8],
        width_ratios=[1, 1, 1, 0.05],
        hspace=0.45, wspace=0.30,
    )

    all_z6, all_d6, all_thetas6 = [], [], []
    for bench, _ in panels_6:
        _, z, d, _, thetas = get_bench_data(
            bench, train_df, bench_names, difficulties, discriminations, idx_df
        )
        all_z6.append(z); all_d6.append(d); all_thetas6.append(thetas)
    z_range6 = (min(a.min() for a in all_z6), max(a.max() for a in all_z6))
    d_range6 = (min(a.min() for a in all_d6), max(a.max() for a in all_d6))

    theta_min6 = min(t.min() for t in all_thetas6) - 1.0
    theta_max6 = max(t.max() for t in all_thetas6) + 1.0
    theta_grid6 = np.linspace(theta_min6, theta_max6, 500).astype(np.float32)

    all_tif6_norm = []
    for bench, _ in panels_6:
        _, z, d, n, _ = get_bench_data(
            bench, train_df, bench_names, difficulties, discriminations, idx_df
        )
        tif = test_information_function(theta_grid6, z, d)
        all_tif6_norm.append(tif / n)
    tif_norm_max6 = max(t.max() for t in all_tif6_norm)

    print("6-panel figure:")
    for idx, (bench, label) in enumerate(panels_6):
        major_row = idx // 3
        col_idx = idx % 3
        base_row = major_row * 4

        # Create a local gridspec view for this panel group
        local_gs = gridspec.GridSpec(
            4, 1,
            figure=fig2,
            height_ratios=[4, 1.3, 1.3, 1.8],
        )

        data, z, d, n, thetas = get_bench_data(
            bench, train_df, bench_names, difficulties, discriminations, idx_df
        )
        tif_norm = all_tif6_norm[idx]

        ax_heat = fig2.add_subplot(gs2[base_row, col_idx])
        im = ax_heat.imshow(data, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
        ax_heat.set_title(
            rf'\textbf{{{label}}} ({n} items, $\sigma_z={z.std():.2f}$, $\sigma_d={d.std():.2f}$)',
            fontsize=12,
        )
        ax_heat.set_xlabel(r'Questions (sorted by $z$)', fontsize=10)
        if col_idx == 0:
            ax_heat.set_ylabel(r'Models (sorted by $p_{\mathrm{CC}}$)', fontsize=10)
        ax_heat.tick_params(labelsize=8)

        ax_z = fig2.add_subplot(gs2[base_row + 1, col_idx])
        ax_z.hist(z, bins=40, color='tab:blue', alpha=0.8, edgecolor='white', linewidth=0.3, range=z_range6)
        ax_z.set_xlabel(r'Difficulty $z$', fontsize=10)
        if col_idx == 0:
            ax_z.set_ylabel('Count', fontsize=10)
        ax_z.set_xlim(z_range6)
        ax_z.tick_params(labelsize=8)

        ax_d = fig2.add_subplot(gs2[base_row + 2, col_idx])
        ax_d.hist(d, bins=40, color='tab:orange', alpha=0.8, edgecolor='white', linewidth=0.3, range=d_range6)
        ax_d.set_xlabel(r'Discrimination $d$', fontsize=10)
        if col_idx == 0:
            ax_d.set_ylabel('Count', fontsize=10)
        ax_d.set_xlim(d_range6)
        ax_d.tick_params(labelsize=8)

        ax_tif = fig2.add_subplot(gs2[base_row + 3, col_idx])
        ax_tif.fill_between(theta_grid6, tif_norm, alpha=0.3, color='tab:green')
        ax_tif.plot(theta_grid6, tif_norm, color='tab:green', linewidth=1.5)
        theta_lo, theta_hi = np.percentile(thetas, [5, 95])
        ax_tif.axvspan(theta_lo, theta_hi, alpha=0.15, color='gray',
                       label=rf'Model $\theta$ (5--95\%)')
        ax_tif.set_xlabel(r'Ability $\theta$', fontsize=10)
        if col_idx == 0:
            ax_tif.set_ylabel(r'$I(\theta)\,/\,N_{\mathrm{items}}$', fontsize=10)
        ax_tif.set_xlim(theta_grid6[0], theta_grid6[-1])
        ax_tif.set_ylim(0, tif_norm_max6 * 1.1)
        ax_tif.legend(fontsize=8, loc='upper right')
        ax_tif.tick_params(labelsize=8)

        peak_theta = theta_grid6[np.argmax(tif_norm)]
        fwhm_mask = tif_norm >= tif_norm.max() / 2
        fwhm_thetas = theta_grid6[fwhm_mask]
        fwhm = fwhm_thetas[-1] - fwhm_thetas[0] if len(fwhm_thetas) > 1 else 0
        print(f"  {label:<15} peak_I/N={tif_norm.max():.3f}  FWHM={fwhm:.2f}  "
              f"model_theta=[{theta_lo:.2f},{theta_hi:.2f}]")

    cbar_ax2 = fig2.add_subplot(gs2[0, 3])
    fig2.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax2, label=r'$p_{\mathrm{Correct\;Choice}}$',
    )

    out_path2 = OUT_DIR / "resmat_heatmap_comparison.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches='tight')
    print(f"  Saved to {out_path2}")
    plt.close('all')
    print("\nDone.")


if __name__ == "__main__":
    main()
