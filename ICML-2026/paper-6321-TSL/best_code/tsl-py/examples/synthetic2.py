"""Synthetic2 — backbone bimodality at epoch 0.

Reconstructs the figure from Reviewer zSd4's rebuttal (paper response,
section "An illustrative example") on the DGP

    f(x1, x2) = exp(sin(x1) * cos(x2)) + x1,    x1, x2 ~ U[-4, 4],
    y = f(x1, x2)  (noise-free),  n = 5000.

At a single stage, TSL fits ``n_trees`` independent grid tensors on bootstrap
samples (Algorithm 10) and then combines them (Algorithm 11). Because the
two-tensor rank-1 family is not uniquely identified, independently bagged
grids can converge to different but similarly-fitted backbone shapes. The
figure shows that

  * sorting bags by total scale ``lambda+ + lambda-`` separates two distinct
    converged representations on Feature 0;
  * the similarity-filtering step (Algorithm 11) selects a single coherent
    set so the averaged backbone is sharp and stable;
  * all 389 raw backbones span a flat sequential (pale → indigo) fan whose
    extremes match the two clusters of row 1.

Outputs (PDF, flat theme):

  * backbone_bimodal_epoch0.pdf
        3-row x 2-col panel:
          row 1 — bottom-17 vs top-17 bagged grids by lambda+ + lambda-,
                  overlaid with the similarity-filtered combined backbone
                  (black);
          row 2 — kept candidates (top k = ceil((1 - xi) * n_grids) by
                  Algorithm 11 score), the reference grid (orange),
                  and the combined backbone (black);
          row 3 — all ``n_grids`` bagged backbones colored by
                  ``lambda+ + lambda-``, combined backbone (black), shared
                  horizontal colorbar.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as mcm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from tensorsl import TSL
from tensorsl.plot._theme import (
    TOKENS,
    airy,
    axis_label,
    figure_title,
    flat_backbone_cmap,
    flat_canvas,
    flat_legend,
    mix,
    panel_title,
    reserve_title_band,
    setup_fonts,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def make_dataset(n: int, seed: int, x_min: float = -4.0, x_max: float = 4.0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(x_min, x_max, size=n).astype(np.float64)
    x2 = rng.uniform(x_min, x_max, size=n).astype(np.float64)
    X = np.ascontiguousarray(np.column_stack([x1, x2]))
    y = np.exp(np.sin(x1) * np.cos(x2)) + x1
    return X, np.ascontiguousarray(y)


# ---------------------------------------------------------------------------
# Per-grid quantities
# ---------------------------------------------------------------------------


def _lookup_bin_index(splits: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Index into ``backbone_values`` / ``tilt_values`` for each ``x``.

    Matches the Rust ``partition_point(|s| s <= v)`` semantics: returns the
    smallest k such that splits[k] > v, clipped to [0, n_intervals - 1].
    """
    n_intervals = splits.size + 1
    idx = np.searchsorted(splits, x, side="right")
    return np.clip(idx, 0, n_intervals - 1)


def _per_point_backbone_tilt(grid, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-point product backbone b(x) and sum tilt d(x)."""
    n, p = X.shape
    b = np.ones(n, dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    for j in range(p):
        splits = np.asarray(grid.splits[j], dtype=np.float64)
        b_vals = np.asarray(grid.backbone_values[j], dtype=np.float64)
        d_vals = np.asarray(grid.tilt_values[j], dtype=np.float64)
        idx = _lookup_bin_index(splits, X[:, j])
        b *= b_vals[idx]
        d += d_vals[idx]
    return b, d


def _step_xy(splits: np.ndarray, values: np.ndarray, x_min: float, x_max: float):
    """Coordinates for ``ax.step(..., where='post')`` over [x_min, x_max].

    ``splits`` has ``n_intervals - 1`` internal split points; ``values`` has
    ``n_intervals`` per-bin backbone (or tilt) values.
    """
    x = np.concatenate([[x_min], splits, [x_max]])
    y = np.concatenate([values, [values[-1]]])
    return x, y


# ---------------------------------------------------------------------------
# Algorithm 11 — reference (medoid) and similarity scores
# ---------------------------------------------------------------------------


def select_reference_index(lambda_plus: np.ndarray, lambda_minus: np.ndarray) -> int:
    """Reference grid g*: argmin_i sum_{i'} (Δλ+)^2 + (Δλ-)^2 (Algorithm 11)."""
    dx = lambda_plus[:, None] - lambda_plus[None, :]
    dy = lambda_minus[:, None] - lambda_minus[None, :]
    return int(np.argmin((dx * dx + dy * dy).sum(axis=1)))


def similarity_scores(grids, ref_idx: int, X: np.ndarray) -> np.ndarray:
    """Combined similarity scores against the reference grid (Eq. sim_combined).

    score_c = (sim_b + 1) * (sim_d + 1) / 4 in [0, 1], where sim_b and sim_d
    are cosine similarities on the per-data-point product backbone and sum
    tilt vectors.
    """
    b_ref, d_ref = _per_point_backbone_tilt(grids[ref_idx], X)
    norm_b_ref = np.linalg.norm(b_ref) + 1e-300
    norm_d_ref = np.linalg.norm(d_ref) + 1e-300
    n_grids = len(grids)
    scores = np.empty(n_grids, dtype=np.float64)
    for c in range(n_grids):
        b_c, d_c = _per_point_backbone_tilt(grids[c], X)
        sim_b = float(b_ref @ b_c) / (norm_b_ref * (np.linalg.norm(b_c) + 1e-300))
        norm_d_c = np.linalg.norm(d_c)
        if norm_d_ref == 1e-300 or norm_d_c == 0.0:
            sim_d = 1.0
        else:
            sim_d = float(d_ref @ d_c) / (norm_d_ref * (norm_d_c + 1e-300))
        scores[c] = (sim_b + 1.0) * (sim_d + 1.0) / 4.0
    return scores


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _eval_bag_b(grid, x_grid: np.ndarray, j: int) -> np.ndarray:
    """Backbone ``b_j`` of one bag evaluated on ``x_grid`` (piecewise-constant)."""
    splits = np.asarray(grid.splits[j], dtype=np.float64)
    b_vals = np.asarray(grid.backbone_values[j], dtype=np.float64)
    idx = _lookup_bin_index(splits, x_grid)
    return b_vals[idx]


def _combined_backbone_on_grid(grids_subset, x_grid: np.ndarray, j: int) -> np.ndarray:
    """Geometric mean (in log-space) of the per-axis backbones ``b_j`` over
    the kept set, evaluated on ``x_grid``.

    This realises the Algorithm 11 averaging step on the kept candidates,
    using ``x_grid`` as a fine common evaluation grid in place of the union
    grid. Equivalent to ``exp(mean log b_j)`` since the geometric mean of
    ``a_pm = b exp(pm d)`` collapses to ``exp(mean log b)`` for the
    backbone component.
    """
    log_b = np.stack(
        [np.log(np.maximum(_eval_bag_b(g, x_grid, j), 1e-300)) for g in grids_subset]
    )
    return np.exp(log_b.mean(axis=0))


def plot_figure(
    grids,
    X: np.ndarray,
    out_path: Path,
    *,
    xi: float = 0.9,
    x_min: float = -4.0,
    x_max: float = 4.0,
    n_extreme: int = 20,
    n_grid_eval: int = 1000,
) -> None:
    n_grids = len(grids)
    lambda_plus = np.array([g.lambda_plus for g in grids])
    lambda_minus = np.array([g.lambda_minus for g in grids])
    lambda_total = lambda_plus + lambda_minus

    # Reference grid (closest to the (lambda+, lambda-) centroid) + Algorithm 11 scores
    ref_idx = select_reference_index(lambda_plus, lambda_minus)
    lambda_ref = float(lambda_total[ref_idx])
    scores = similarity_scores(grids, ref_idx, X)

    # Trim: keep top k = ceil((1 - xi) * n_grids) by similarity score
    keep_count = int(np.ceil((1.0 - xi) * n_grids))
    kept_idx = np.argsort(-scores)[:keep_count]
    kept_grids = [grids[int(i)] for i in kept_idx]

    # Sort by total lambda for the row-1 split
    order = np.argsort(lambda_total)
    low_idx = order[:n_extreme]
    high_idx = order[-n_extreme:]
    low_max = float(lambda_total[low_idx].max())
    high_min = float(lambda_total[high_idx].min())

    # Combined backbone (geometric mean of b_j across kept bags) on a fine grid
    x_grid = np.linspace(x_min, x_max, n_grid_eval)
    combined_b = [_combined_backbone_on_grid(kept_grids, x_grid, j) for j in range(2)]

    # Flat sequential normalisation for row 3
    norm = Normalize(vmin=lambda_total.min(), vmax=lambda_total.max())
    cmap = flat_backbone_cmap()
    candidate_c = mix(TOKENS["accent"], 0.55)

    disp, mono = setup_fonts()
    fig, axes = plt.subplots(3, 2, figsize=(11, 11))

    for j in range(2):
        # ---- Row 1: two illustrative groups ----
        ax = axes[0, j]
        for i in low_idx:
            xs, ys = _step_xy(
                np.asarray(grids[i].splits[j], dtype=np.float64),
                np.asarray(grids[i].backbone_values[j], dtype=np.float64),
                x_min, x_max,
            )
            ax.step(xs, ys, where="post", color=TOKENS["neg"], lw=1, alpha=0.4)
        for i in high_idx:
            xs, ys = _step_xy(
                np.asarray(grids[i].splits[j], dtype=np.float64),
                np.asarray(grids[i].backbone_values[j], dtype=np.float64),
                x_min, x_max,
            )
            ax.step(xs, ys, where="post", color=TOKENS["pos"], lw=1, alpha=0.4)
        ax.plot(x_grid, combined_b[j], color=TOKENS["ink"], lw=2.2)

        # Empty handles for legend (so colours show as solid swatches)
        ax.plot([], [], color=TOKENS["neg"], lw=2,
                label=rf"Low $\lambda^+\!+\!\lambda^- <$ {low_max:.1f} ({n_extreme})")
        ax.plot([], [], color=TOKENS["pos"], lw=2,
                label=rf"High $\lambda^+\!+\!\lambda^- \geq$ {high_min:.1f} ({n_extreme})")
        ax.plot(
            [], [], color=TOKENS["ink"], lw=2,
            label=rf"Combined ($\xi={xi:.1f}$, $|\mathcal{{K}}|={keep_count}$, $\lambda^\star \approx {lambda_ref:.4f}$)",
        )
        airy(ax, mono)
        panel_title(ax, "Extremes by scale", disp)
        axis_label(ax, mono, xlabel=rf"$x_{{{j + 1}}}$", ylabel=rf"$b_{{{j + 1}}}$")
        ax.set_xlim(x_min, x_max)
        flat_legend(ax, mono, loc="best", fontsize=8)

        # ---- Row 2: combination candidates ----
        ax = axes[1, j]
        for i in kept_idx:
            xs, ys = _step_xy(
                np.asarray(grids[i].splits[j], dtype=np.float64),
                np.asarray(grids[i].backbone_values[j], dtype=np.float64),
                x_min, x_max,
            )
            ax.step(xs, ys, where="post", color=candidate_c, lw=1, alpha=0.45)
        xs_ref, ys_ref = _step_xy(
            np.asarray(grids[ref_idx].splits[j], dtype=np.float64),
            np.asarray(grids[ref_idx].backbone_values[j], dtype=np.float64),
            x_min, x_max,
        )
        ax.step(xs_ref, ys_ref, where="post", color=TOKENS["pos"],
                lw=2.0, label="Reference grid")
        ax.plot(x_grid, combined_b[j], color=TOKENS["ink"], lw=2.2, label="Combined")

        ax.plot([], [], color=candidate_c, lw=2, label=f"Candidates ({keep_count})")
        handles, labels = ax.get_legend_handles_labels()
        ordered = ["Candidates", "Reference", "Combined"]
        items = {k: (h, l) for h, l in zip(handles, labels) for k in ordered if l.startswith(k)}
        flat_legend(ax, mono,
                    [items[k][0] for k in ordered if k in items],
                    [items[k][1] for k in ordered if k in items],
                    loc="best", fontsize=8)
        airy(ax, mono)
        panel_title(ax, "Filtered candidates", disp)
        axis_label(ax, mono, xlabel=rf"$x_{{{j + 1}}}$", ylabel=rf"$b_{{{j + 1}}}$")
        ax.set_xlim(x_min, x_max)

        # ---- Row 3: all n_grids tensors colored by lambda+ + lambda- ----
        ax = axes[2, j]
        for i in range(n_grids):
            xs, ys = _step_xy(
                np.asarray(grids[i].splits[j], dtype=np.float64),
                np.asarray(grids[i].backbone_values[j], dtype=np.float64),
                x_min, x_max,
            )
            ax.step(xs, ys, where="post", color=cmap(norm(lambda_total[i])),
                    lw=1, alpha=0.45)
        ax.plot(x_grid, combined_b[j], color=TOKENS["ink"], lw=2.2, label="Combined")
        airy(ax, mono)
        panel_title(ax, "All bagged grids", disp)
        axis_label(ax, mono, xlabel=rf"$x_{{{j + 1}}}$", ylabel=rf"$b_{{{j + 1}}}$")
        ax.set_xlim(x_min, x_max)
        flat_legend(ax, mono, loc="best", fontsize=8)

    fig.tight_layout(rect=[0, 0.07, 1, reserve_title_band(fig, 1.3)])

    # Shared horizontal colorbar at bottom (row 3 cells)
    sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.25, 0.045, 0.50, 0.015])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.outline.set_edgecolor(TOKENS["border"])
    cbar.outline.set_linewidth(0.9)
    cbar.ax.tick_params(length=0, labelsize=7.5, colors=TOKENS["muted"])
    for lab in cbar.ax.get_xticklabels():
        lab.set_family(mono)
    cbar.set_label(r"$\lambda^+\!+\!\lambda^-$", family=mono, fontsize=8,
                   color=TOKENS["muted"])

    flat_canvas(fig)
    figure_title(fig, "TSL / diagnostics", "Backbone bimodality",
                 badge="stage 1 · epoch 0", badge_color=TOKENS["accent"])

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    print(f"  reference bag = {ref_idx},  lambda+={lambda_plus[ref_idx]:.4f}, "
          f"lambda-={lambda_minus[ref_idx]:.4f},  total={lambda_ref:.4f}")
    print(f"  lambda+ + lambda-: min={lambda_total.min():.3f}, "
          f"max={lambda_total.max():.3f}")
    print(f"  kept |K| = {keep_count} of {n_grids} (xi={xi:.2f})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(
    out: Path, *,
    n: int = 5000,
    n_trees: int = 389,
    n_iter: int = 20,
    split_try: int = 10,
    min_interval_samples: int = 20,
    alpha: float = 1e-2,
    xi: float = 0.9,
    seed: int = 0,
) -> None:
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic data (n={n}, seed={seed}) ...")
    X, y = make_dataset(n=n, seed=seed)

    print(f"Fitting TSL (epochs=1, n_trees={n_trees}, similarity_threshold={1.0 - xi:.2f}) ...")
    model, _ = TSL.fit(
        X, y,
        epochs=1,
        n_trees=n_trees,
        n_iter=n_iter,
        split_try=split_try,
        colsample_bytree=1.0,
        alpha=alpha,
        min_interval_samples=min_interval_samples,
        tilt_tau=0.0,
        tilt_rho=0.0,
        similarity_threshold=1.0 - xi,
        seed=seed,
        verbosity=0,
    )

    sp = model.stage_predictors[0]
    grids = list(sp.grid_tensors)

    print(f"Generating figure ({len(grids)} raw bagged grids) ...")
    plot_figure(
        grids, X,
        out_path=out / "backbone_bimodal_epoch0.pdf",
        xi=xi,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic2 — backbone bimodality at epoch 0")
    parser.add_argument("--out", type=Path, default=Path("/tmp/tsl_examples/synthetic2"))
    parser.add_argument("--n", type=int, default=5000, help="Training sample size")
    parser.add_argument("--n-trees", type=int, default=389,
                        help="Number of bagged grid tensors per stage")
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--split-try", type=int, default=10)
    parser.add_argument("--min-interval-samples", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--xi", type=float, default=0.9,
                        help="Algorithm 11 trim threshold (keep top (1-xi) by similarity)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Fixed seed driving data generation and bagging")
    args = parser.parse_args()
    main(
        out=args.out, n=args.n, n_trees=args.n_trees,
        n_iter=args.n_iter, split_try=args.split_try,
        min_interval_samples=args.min_interval_samples,
        alpha=args.alpha, xi=args.xi, seed=args.seed,
    )
