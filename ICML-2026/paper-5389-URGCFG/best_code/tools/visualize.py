from __future__ import annotations
"""Evaluation and plotting helpers for mechanism visualization tasks."""

import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from config import WRITING_ROOT
from tools.feedback import logger
from tools.utils import loader

# ---------- core save/show helper ----------

def _save_or_show(fig: plt.Figure, filename: str, save_dir: Optional[str] = WRITING_ROOT):
    """
    Save the figure under `save_dir` (defaults to config.WRITING_ROOT), or
    display it interactively if save_dir is None.
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[viz] saved {path}")
    else:
        fig.show()

# ---------- core evaluation helpers ----------

@torch.no_grad()
def eval_mech_1d(mechanism: torch.nn.Module, N: int = 801, device: Optional[torch.device] = None):
    """
    Evaluate mechanism on a 1D grid x in [0,1].
    Returns xs, q (N,), t (N,), v (N or None), k (N or None).
    """
    xs = np.linspace(0.0, 1.0, N, dtype=np.float32)
    X = torch.from_numpy(xs[:, None])  # (N,1)

    if device is None:
        try:
            device = next(mechanism.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    X = X.to(device)

    out = mechanism.compute_mechanism(X, mode="hard")
    q = out["choice"].detach().cpu().numpy().reshape(-1)          # (N,)
    t = out["revenue"].detach().cpu().numpy().reshape(-1)         # (N,)
    v = out.get("v", None)
    k = out.get("kernel", None)
    if v is not None: v = v.detach().cpu().numpy().reshape(-1)
    if k is not None: k = k.detach().cpu().numpy().reshape(-1)
    return xs, q, t, v, k


@torch.no_grad()
def eval_mech_2d(mechanism: torch.nn.Module, N: int = 1000, device: Optional[torch.device] = None):
    """
    Evaluate mechanism on a 2D grid (x1,x2) in [0,1]^2.
    Returns (xs, ys, X, Y, q1, q2, t, v, k)
    """
    xs = np.linspace(0.0, 1.0, N, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, N, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    grid = np.stack([X, Y], axis=-1).reshape(-1, 2)
    T = torch.from_numpy(grid)

    if device is None:
        try:
            device = next(mechanism.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    T = T.to(device)

    out = mechanism.compute_mechanism(T, mode="hard")
    Q = out["choice"].detach().cpu().numpy().reshape(N, N, 2)     # (N,N,2)
    t = out["revenue"].detach().cpu().numpy().reshape(N, N)       # (N,N)
    v = out.get("v", None)
    k = out.get("kernel", None)
    if v is not None: v = v.detach().cpu().numpy().reshape(N, N)
    if k is not None: k = k.detach().cpu().numpy().reshape(N, N)
    q1, q2 = Q[..., 0], Q[..., 1]
    return xs, ys, X, Y, q1, q2, t, v, k

# ---------- math helpers ----------

def bundle_probs_from_marginals(q1: np.ndarray, q2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Independent implementation map: (q1,q2) -> probs over {00,10,01,11}.
    """
    p11 = q1 * q2
    p10 = q1 * (1.0 - q2)
    p01 = (1.0 - q1) * q2
    p00 = (1.0 - q1) * (1.0 - q2)
    return p00, p10, p01, p11

def estimate_cutoff(xs: np.ndarray, q: np.ndarray, level: float = 0.5) -> float:
    """
    First x where q(x) crosses 'level' (linear interpolation). Returns nan if no crossing.
    """
    idx = np.where(q >= level)[0]
    if len(idx) == 0: return float("nan")
    k = idx[0]
    if k == 0: return float(xs[0])
    x0, x1 = xs[k-1], xs[k]
    q0, q1 = q[k-1], q[k]
    if q1 == q0: return float(x1)
    a = (level - q0) / (q1 - q0)
    return float(x0 + a * (x1 - x0))

def theory_step_allocation(xs: np.ndarray, reserve: float = 0.5):
    """
    Posted price theory for x ~ U[0,1]: q*(x)=1{x >= reserve}, t*(x)=reserve*q*(x).
    """
    q_star = (xs >= reserve).astype(float)
    t_star = reserve * q_star
    return q_star, t_star

# ---------- small plotting primitives ----------

def _heatmap(Z, xs, ys, title, clabel=None, filename=None, save_dir: Optional[str] = WRITING_ROOT):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    im = ax.imshow(Z, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect="equal")
    ax.set_title(title); ax.set_xlabel("x1"); ax.set_ylabel("x2")
    if clabel: fig.colorbar(im, ax=ax, label=clabel, shrink=0.73)
    fig.tight_layout()
    if filename is None:
        # derive a safe filename from the title
        filename = title.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".png"
    _save_or_show(fig, filename, save_dir)

def _quiver(X, Y, U, V, title, step=8, filename=None, save_dir: Optional[str] = WRITING_ROOT):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.quiver(X[::step, ::step], Y[::step, ::step], U[::step, ::step], V[::step, ::step],
              angles="xy", scale_units="xy", scale=1.0)
    ax.set_title(title); ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    if filename is None:
        filename = title.lower().replace(" ", "_") + ".png"
    _save_or_show(fig, filename, save_dir)

# ---------- 1D high-level visualization ----------

def plot_mechanism_1d(mechanism: torch.nn.Module,
                      reserve: float = 0.5,
                      N: int = 1201,
                      show_revenue_curve: bool = True,
                      save_dir: Optional[str] = WRITING_ROOT):
    """
    Plots learned allocation/payment vs. posted-price theory, reports reserve & revenue.
    """
    xs, q, t, v, k = eval_mech_1d(mechanism, N=N)
    q_star, t_star = theory_step_allocation(xs, reserve)
    r_hat = estimate_cutoff(xs, q, level=0.5)

    # allocation
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(xs, q, label="learned a(x)", lw=2)
    ax.plot(xs, q_star, "--", label=f"optimal $a^*(x) = 1${{x ≥ {reserve}}}", lw=2)
    if not np.isnan(r_hat):
        ax.axvline(r_hat, color="k", ls=":", label=f"learned price $\hat{{p}}$ ≈ {r_hat:.3f}")
    ax.axvline(reserve, color="gray", ls="--", alpha=0.8, label=f"optimal price $\hat{{p}}$={reserve}")
    ax.set_ylim(-0.05, 1.05); ax.set_xlabel("type x"); ax.set_ylabel("allocation a(x)")
    ax.set_title("Single Good Allocation"); ax.legend(); fig.tight_layout()
    _save_or_show(fig, "1d_allocation.png", save_dir)

    # payment

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xs, t, label="learned t(x)", lw=2)
    ax.plot(xs, t_star, "--", label="theory t*(x)", lw=2)
    ax.axvline(reserve, color="gray", ls="--", alpha=0.8)
    ax.set_xlabel("type x"); ax.set_ylabel("payment t(x)")
    ax.set_title("1D payment: learned vs theory"); ax.legend(); fig.tight_layout()
    _save_or_show(fig, "1d_payment.png", save_dir)

    # optional revenue curve context
    if show_revenue_curve:
        ps = np.linspace(0, 1, 400)
        R = ps * (1 - ps)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ps, R, lw=2, label="Theoretical Profit")
        ax.axvline(reserve, ls="--")
        ax.scatter([reserve], [reserve * (1 - reserve)], zorder=3, color="red", label=f"Posted Price Found")
        ax.set_xlabel("posted price p"); ax.set_ylabel("expected revenue R(p)")
        ax.set_title("Monopoly revenue (Uniform[0,1])"); fig.tight_layout()
        ax.legend()
        _save_or_show(fig, "1d_revenue_curve.png", save_dir)

    # metrics
    rev_learned = float(t.mean())
    rev_theory = float(reserve * (1 - reserve))
    l2_q = float(np.sqrt(np.mean((q - q_star)**2)))
    l2_t = float(np.sqrt(np.mean((t - t_star)**2)))

    logger.info(f"[1D] learned cutoff r̂: {r_hat:.3f}")
    logger.info(f"[1D] expected revenue — learned: {rev_learned:.4f} | theory(opt): {rev_theory:.4f} | gap: {rev_learned - rev_theory:+.4f}")
    logger.info(f"[1D] L2 error: q vs theory = {l2_q:.4f},  t vs theory = {l2_t:.4f}")

# ---------- 2D high-level visualization ----------

def plot_mechanism_2d(mechanism: torch.nn.Module,
                      N: int = 201,
                      tau: float = 1e-3,
                      show_q_heatmaps: bool = True,
                      show_bundle_map: bool = True,
                      show_payment_heatmap: bool = True,
                      show_value_heatmap: bool = True,
                      show_quiver_field: bool = True,
                      quiver_step: int = 10,
                      kernel_label: str = "gross value (kernel)",
                      save_dir: Optional[str] = WRITING_ROOT):
    """
    Visual diagnostics on [0,1]^2:
      - q1, q2 heatmaps
      - implied bundle regions & tie (Maxwell) set from independent implementation
      - payment heatmap
      - optional v(x) and kernel(x,q(x)) heatmaps
      - optional quiver of q(x)
    """
    xs, ys, X, Y, q1, q2, t, v, k = eval_mech_2d(mechanism, N=N)

    if show_q_heatmaps:
        _heatmap(q1, xs, ys, "$a_1(x)$: probability of getting good 1", "$a_1$", filename="a_1_heatmap.png", save_dir=save_dir)
        _heatmap(q2, xs, ys, "$a_2(x)$: probability of getting good 2", "$a_2$", filename="a_2_heatmap.png", save_dir=save_dir)

    if show_bundle_map:
        p00, p10, p01, p11 = bundle_probs_from_marginals(q1, q2)
        stacks = np.stack([p00, p10, p01, p11], axis=-1)  # (N,N,4)
        labels = stacks.argmax(axis=-1)

        # tie curve
        sv = np.sort(stacks, axis=-1)
        gap = sv[..., -1] - sv[..., -2]
        tie = (gap < tau).astype(float)

        names  = {0: "∅ (00)", 1: "{1} (10)", 2: "{2} (01)", 3: "{1,2} (11)"}
        colors = ["#d9d9d9", "#6baed6", "#74c476", "#fd8d3c"]
        cmap   = ListedColormap(colors)
        norm   = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(labels, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]],
                       interpolation="nearest", aspect="equal", cmap=cmap, norm=norm)
        # draw boundaries + tie
        ax.contour(X, Y, labels, levels=[0.5, 1.5, 2.5], colors="k", linewidths=1.0)
        if tie.any():
            ax.contour(X, Y, tie, levels=[0.5], colors="k", linewidths=0.8, linestyles="--")
        ax.set_title("Implied bundle regions (indep. implementation)")
        ax.set_xlabel("x1"); ax.set_ylabel("x2")
        patches = [Patch(facecolor=colors[k], edgecolor='none', label=names[k]) for k in range(4)]
        ax.legend(handles=patches, title="Chosen bundle", loc="upper left", frameon=True)
        fig.tight_layout()
        _save_or_show(fig, "bundle_map.png", save_dir)

    if show_payment_heatmap:
        _heatmap(t, xs, ys, "Payment t(x)", "t(x)", filename="payment_heatmap.png", save_dir=save_dir)

    if show_value_heatmap and (v is not None):
        _heatmap(v, xs, ys, "Indirect utility v(x)", "v(x)", filename="indirect_utility_heatmap.png", save_dir=save_dir)

    if (k is not None):
        _heatmap(k, xs, ys, kernel_label, kernel_label, filename="kernel_heatmap.png", save_dir=save_dir)

    if show_quiver_field:
        _quiver(X, Y, q1, q2, "Allocation vector field q(x)", step=quiver_step, filename="allocation_quiver.png", save_dir=save_dir)


def plot_revenue_fit_1d(mechanism: torch.nn.Module,
                        reserve: float | None = None,
                        N_price_grid: int = 400,
                        price_from: str = "cutoff",
                        N_eval: int = 1201,
                        save_dir: Optional[str] = WRITING_ROOT):
    """
    Show how your posted price fits the revenue curve R(p) for x ~ Uniform[0,1].

    Parameters
    ----------
    mechanism : trained mechanism with compute_mechanism(X)
    reserve   : if provided, use this as the posted price p (overrides price_from)
    N_price_grid : resolution for plotting the revenue curve
    price_from : "cutoff" (estimate from q crossing 0.5) or "argmax" (maximizes empirical revenue curve)
    N_eval    : resolution for evaluating the learned mechanism in 1D

    Notes
    -----
    - Theory for Uniform[0,1]: demand at price p is 1 - F(p) = 1 - p, so R(p) = p*(1-p).
    - If reserve is None and price_from == "cutoff", we use the learned cutoff r̂ from q(x).
    - If price_from == "argmax", we pick p that maximizes p*(1-p) on the same grid (≈ 0.5).
    """
    # 1) Evaluate learned mechanism in 1D (for reserve-from-cutoff & reporting)
    xs, q, t, _, _ = eval_mech_1d(mechanism, N=N_eval)
    r_hat = estimate_cutoff(xs, q, level=0.5)

    # 2) Theoretical revenue curve
    ps = np.linspace(0.0, 1.0, N_price_grid)
    R = ps * (1.0 - ps)  # Uniform[0,1]

    # 3) Choose the posted price to mark
    if reserve is not None:
        p_mark = float(reserve)
        mark_label = f"your posted price p = {p_mark:.3f}"
    else:
        if price_from == "argmax":
            j = int(np.argmax(R))
            p_mark = float(ps[j])
            mark_label = f"argmax R(p) ≈ {p_mark:.3f}"
        else:  # "cutoff"
            p_mark = float(r_hat) if not np.isnan(r_hat) else 0.5
            mark_label = "learned price $\hat{p}$ ≈ "+ f"{p_mark:.3f}"

    R_mark = float(p_mark * (1.0 - p_mark))

    # 4) Plot & save
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(ps, R, lw=2, label="$R(p)=p(1-p)$")
    ax.axvline(0.5, ls="--", color="gray", alpha=0.8, label="optimal price $\hat{p}=0.5$")
    ax.scatter([p_mark], [R_mark], s=50, zorder=3, label=f"{mark_label}", color="red")#\nR(p)={R_mark:.3f}
    ax.set_xlabel("posted price p"); ax.set_ylabel("expected revenue R(p)")
    ax.set_title("Posted Price and Single Good")
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, "revenue_fit.png", save_dir)

    # 5) Print a quick summary vs learned mechanism revenue
    rev_learned = float(t.mean())  # uniform types on [0,1]
    rev_theory_opt = 0.25
    logger.info(f"[Revenue fit] marked price p = {p_mark:.3f}, R(p) = {R_mark:.4f}")
    logger.info(f"[Revenue fit] learned mechanism revenue (mean t): {rev_learned:.4f}")
    logger.info(f"[Revenue fit] theory optimum: {rev_theory_opt:.4f}, gap to marked: {R_mark - rev_theory_opt:+.4f}")
    if not np.isnan(r_hat):
        logger.info(f"[Revenue fit] learned cutoff r̂ from q(x): {r_hat:.3f}")


def plot_profit(sample, dim_to_mechs):
    """
    Plot profit curves for the given mechanism.
    """
    n_items = []
    P = []
    for dim, (mech, mech_date) in sorted(dim_to_mechs.items()):
        n_items.append(dim)
        p = mech_date["profits"].mean().item()
        logger.debug(f"{dim} {p}")
        P.append(p/dim)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(n_items, P, lw=2, label="profits per item")
    ax.set_xlabel("number of items"); ax.set_ylabel("profit")
    ax.set_title("Profit curve")
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, "profit_per_item_curve.png", save_dir)


def make_table(dim_to_mechs, sample, save_dir):
    """
    Create a profit table for the given mechanisms and sample.
    """
    dims = []
    mp = []
    ms = []
    mv = []
    for dim, (mech, mech_data) in sorted(dim_to_mechs.items()):
        profit_per_item = mech_data["profits"].mean().item() / dim
        surplus_per_item = mech_data["kernel"].mean().item() / dim
        v_per_item = mech_data["v"].mean().item() / dim
        dims.append(int(dim))
        mp.append(profit_per_item)
        ms.append(surplus_per_item)
        mv.append(v_per_item)
        
    df = pd.DataFrame({
        "Number of Items": dims,
        "Mean Profit per Item": mp,
        "Mean Surplus per Item": ms,
        "Mean Utility per Item": mv
    })
    
    fig, ax = plt.subplots(figsize=(7, 2))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    _save_or_show(fig, "mech_table.png", save_dir)

    latex_table = df.to_latex(index=False,
                              caption="Optimized Mechanisms", label="tab:results",
                              float_format="%.3f",
                              column_format="|l|r|r|")
    with open(f"{save_dir}/table.tex", "w") as f:
        f.write(latex_table)


if __name__=="__main__":    
    from config import WRITING_ROOT
    data = loader()
    for tag, (sample, models) in data.items():
        logger.info(f"working on {tag}...")
        save_dir = f"{WRITING_ROOT}/plots/{tag}"
        logger.info(f"saving to {save_dir}")
        dim_to_mechs = {}
        for m in models:
            dim = m.full_Y().shape[2]
            dim_to_mechs[dim] = (m, m.compute_mechanism(sample[:, :dim], mode="hard"))
        if 1 in dim_to_mechs:
            plot_revenue_fit_1d(dim_to_mechs[1][0], save_dir=save_dir)
            plot_mechanism_1d(dim_to_mechs[1][0], reserve=0.5, N=1000, show_revenue_curve=True, save_dir=save_dir)
        if 2 in dim_to_mechs:
            plot_mechanism_2d(dim_to_mechs[2][0], N=1000, tau=1e-3,
                      show_q_heatmaps=True,
                      show_bundle_map=True,
                      show_payment_heatmap=True,
                      show_value_heatmap=False,
                      show_quiver_field=False,
                      quiver_step=10,
                      save_dir=save_dir)            
        plot_profit(sample, dim_to_mechs)
        make_table(dim_to_mechs, sample, save_dir)
        # Collect profit per item for each mechanism


def visualize_transport(
    x, y, model, save_dir=WRITING_ROOT,
    n_arrows=200, figsize=(7, 7)
):
    """
    Visualize learned Monge map T(x)=∇f(x) with straight-line transport arrows.
    Each arrow is a single line colored from red (start) → blue (end).
    """
    assert x.shape[1] == 2, "Visualization only implemented for d=2."

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.collections import LineCollection
    import numpy as np
    import os

    # -----------------------------------------------------
    # Prepare directory + filename
    # -----------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    model_name = model.__class__.__name__
    fig_path = os.path.join(save_dir, f"OTplot_{model_name}.png")

    # -----------------------------------------------------
    # Prepare data
    # -----------------------------------------------------
    x_cpu = x.detach().cpu()
    y_cpu = y.detach().cpu()
    Tx = model.transport_X_to_Y(x_cpu).detach().cpu()

    x_np = x_cpu.detach().numpy()
    y_np = y_cpu.detach().numpy()
    Tx_np = Tx.detach().numpy()

    # -----------------------------------------------------
    # Subsample for arrows
    # -----------------------------------------------------
    n = min(n_arrows, x_np.shape[0])
    idx = np.random.choice(x_np.shape[0], n, replace=False)
    xs = x_np[idx]
    Txs = Tx_np[idx]

    # -----------------------------------------------------
    # Build the transport lines
    # -----------------------------------------------------
    # Each line is shape (2,2): [[x1,y1],[x2,y2]]
    lines = np.stack([xs, Txs], axis=1)  # shape (n,2,2)

    # Create gradient colors: 0=red (start), 1=blue (end)
    cmap = cm.get_cmap("coolwarm")  # red→white→blue
    start_color = cmap(0.0)         # red side of coolwarm
    end_color   = cmap(1.0)         # blue side

    # Build a Nx2 array of colors: start red, end blue
    # LineCollection expects one color *per line*, but we can fade using alpha
    # So instead: fade based on distance
    dists = np.sqrt(np.sum((Txs - xs)**2, axis=1))
    if dists.max() > 0:
        d_norm = (dists - dists.min()) / (dists.max() - dists.min())
    else:
        d_norm = np.zeros_like(dists)

    # Blend: color = (1-t)*red + t*blue
    colors = (1 - d_norm)[:, None] * start_color + d_norm[:, None] * end_color
    colors[:, 3] = 0.8  # set alpha

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------
    plt.figure(figsize=figsize)

    # Softer scatter for X, Y, and T(x)
    plt.scatter(x_np[:, 0], x_np[:, 1], s=10, color="#b33939", alpha=0.45, label="source $x$")
    plt.scatter(y_np[:, 0], y_np[:, 1], s=10, color="#3b6ea8", alpha=0.45, label="target $y$")
    plt.scatter(Tx_np[:, 0], Tx_np[:, 1], s=12, color="#f5b041", alpha=0.55, label="$T(x)$")

    ax = plt.gca()
    # lc = LineCollection(lines, colors=colors, linewidth=1.6)
    # ax.add_collection(lc)

    plt.legend()
    plt.title("Dynamic Transport: red → blue flow")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=240)
    plt.close()

    logger.info(f"[✓] Saved dynamic transport figure to: {fig_path}")
    return fig_path
