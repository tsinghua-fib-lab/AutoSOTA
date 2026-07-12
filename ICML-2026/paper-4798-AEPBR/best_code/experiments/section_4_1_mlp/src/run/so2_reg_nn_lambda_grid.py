# Two-dimensional sweep over equivariant / non-equivariant penalties.
import math
import os
import random
import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

# ensure parent src directory is on path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from approx_so2_equiv_nn import ApproxHarmonicInvariantMLP

# ----------------------- deterministic setup -----------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# -------------------------- data --------------------------
def generate_data(n_inner: int = 100, n_outer: int = 100, seed: int = 1):
    rng = np.random.default_rng(seed)
    r_inner, r_outer_min, r_outer_max = 1.0, 2.3, 3.0
    theta_in = 2 * np.pi * rng.random(n_inner)
    rad_in = r_inner * np.sqrt(rng.random(n_inner))
    X_in = np.c_[rad_in * np.cos(theta_in), rad_in * np.sin(theta_in)]
    y_in = np.ones(n_inner)

    theta_out = 0.5 * np.pi * rng.random(n_outer) - 0.25 * np.pi
    rad_out = rng.uniform(r_outer_min, r_outer_max, n_outer)
    X_out = np.c_[rad_out * np.cos(theta_out), rad_out * np.sin(theta_out)]
    y_out = -np.ones(n_outer)

    X = np.vstack([X_in, X_out]).astype(np.float32)
    y = np.hstack([y_in, y_out]).astype(np.int64)
    return X, y


# ------------------------ training loop -------------------------
def train_once(
    model,
    X,
    y,
    lambda_eq: float,
    lambda_ne: float,
    *,
    epochs: int = 20,
    lr: float = 3e-3,
    device: str = DEVICE,
):
    model.to(device)
    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor((y > 0).astype(np.float32), device=device)[:, None]

    opt = optim.Adam(model.parameters(), lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X_t).real
        loss = bce(logits, y_t)

        penalty = model.compute_non_equivariance_penalty()
        loss = loss + lambda_eq * penalty["equivariant_part"]
        loss = loss + lambda_ne * penalty["nonequiv_part"]
        loss.backward()
        opt.step()
    return model.eval()


# -------------------- empirical invariance test -------------------
@torch.no_grad()
def invariance_error(
    model,
    X: np.ndarray,
    *,
    n_rot: int = 30,
    n_batch: int = 256,
    device: str = DEVICE,
):
    """
    Returns mean & max |f(R_alpha * x) - f(x)| over `n_rot` random rotations
    and `n_batch` random data points.
    """
    sel = np.random.choice(len(X), size=n_batch, replace=False)
    pts = torch.tensor(X[sel], device=device)
    base = model(pts).real  # (B,1)

    errs = []
    for _ in range(n_rot):
        alpha = 2 * math.pi * random.random()
        rot = torch.tensor(
            [[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]],
            device=device,
        )
        pts_rot = pts @ rot.T
        out_rot = model(pts_rot).real
        errs.append((out_rot - base).abs())

    errs = torch.stack(errs)  # (n_rot,B,1)
    return errs.mean().item(), errs.max().item()


# -------------------------- plotting --------------------------
def plot_lambda_grid(
    models,
    X,
    y,
    lambda_eq_values,
    lambda_ne_values,
    errors,
    *,
    lim: float = 4.0,
    grid: int = 300,
    n_lvls: int = 8,
):
    rows, cols = len(lambda_eq_values), len(lambda_ne_values)
    xs = np.linspace(-lim, lim, grid)
    xx, yy = np.meshgrid(xs, xs)
    pts = torch.tensor(np.c_[xx.ravel(), yy.ravel()].astype(np.float32)).to(DEVICE)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4.0 * cols, 4.0 * rows),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    cmap = cm.get_cmap("coolwarm")

    os.makedirs("figures/", exist_ok=True)

    for i, lambda_eq in enumerate(lambda_eq_values):
        for j, lambda_ne in enumerate(lambda_ne_values):
            ax = axes[i, j]
            net = models[i][j]
            mean_err, max_err = errors[i][j]

            with torch.no_grad():
                score = net(pts).cpu().real.numpy().reshape(xx.shape)

            # Background region
            ax.contourf(
                xx,
                yy,
                np.sign(score),
                levels=[-1, 0, 1],
                colors=["#ffdddd", "#ddddff"],
                alpha=0.5,
            )

            # Decision boundary
            ax.contour(xx, yy, score, levels=[0], colors="black", linewidths=1.5)

            # Adaptive level sets
            vmin, vmax = np.percentile(score, [5, 95])
            if abs(vmax - vmin) < 1e-2:  # handle near-constant outputs
                vmax = vmin + 1e-2

            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            lvls = np.linspace(vmin, vmax, 2 * n_lvls + 1)[1:-1]
            ax.contour(
                xx,
                yy,
                score,
                levels=lvls,
                linewidths=1.0,
                cmap=cmap,
                norm=norm,
                alpha=0.9,
                zorder=1,
            )

            # Data points
            pos, neg = y > 0, y < 0
            ax.scatter(X[pos, 0], X[pos, 1], c="blue", edgecolors="k", s=20, zorder=10)
            ax.scatter(
                X[neg, 0],
                X[neg, 1],
                c="red",
                edgecolors="k",
                marker="X",
                s=30,
                zorder=10,
            )

            # Styling
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

            # annotate invariance error
            ax.text(
                0.5,
                -0.025,
                "$\mathcal{E}(T)$" + f" = {max_err:.2e}",
                ha="center",
                va="top",
                fontsize=20,
                transform=ax.transAxes,
            )

    # annotate row / column headers once
    for i, lambda_eq in enumerate(lambda_eq_values):
        axes[i, 0].set_ylabel(f"$\lambda_G$={lambda_eq:g}", fontsize=25, labelpad=6)
    for j, lambda_ne in enumerate(lambda_ne_values):
        axes[-1, j].set_xlabel(
            f"$\lambda_\perp$={lambda_ne:g}", fontsize=25, labelpad=50
        )

    plt.savefig(
        "figures/nn_approx_invariance_lambda_grid.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        "figures/nn_approx_invariance_lambda_grid.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# --------------------------- main ---------------------------
if __name__ == "__main__":
    X, y = generate_data(seed=3)

    # Choose sweep ranges (rows = lambda_eq, cols = lambda_ne)
    lambda_eq_values = [1e-1, 1e-2, 1e-3, 0.0]
    lambda_ne_values = [0.0, 1e-3, 1e-2, 1e-1]

    models = [[None for _ in lambda_ne_values] for _ in lambda_eq_values]
    errors = [[None for _ in lambda_ne_values] for _ in lambda_eq_values]

    for i, lambda_eq in enumerate(lambda_eq_values):
        for j, lambda_ne in enumerate(lambda_ne_values):
            net = ApproxHarmonicInvariantMLP(M=4, C=4, hidden_c=8)
            net = train_once(net, X, y, lambda_eq, lambda_ne, epochs=200, device=DEVICE)
            models[i][j] = net.to(DEVICE)

            mean_err, max_err = invariance_error(
                net, X, n_rot=40, n_batch=len(X), device=DEVICE
            )
            errors[i][j] = (mean_err, max_err)
            print(
                f"lambda_G={lambda_eq:8.2e} lambda_perp={lambda_ne:8.2e}, "
                f"mean|delta|={mean_err:.3e} max|delta|={max_err:.3e}"
            )

    plot_lambda_grid(models, X, y, lambda_eq_values, lambda_ne_values, errors)
