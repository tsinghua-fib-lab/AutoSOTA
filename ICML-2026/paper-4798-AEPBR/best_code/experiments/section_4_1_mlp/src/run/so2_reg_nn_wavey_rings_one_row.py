# Sweep showing how λ⊥ interacts with angular "wave" perturbations of the rings.
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import torch.nn as nn
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
def generate_wavey_rings(
    wave_amp: float,
    *,
    n_pos: int = 350,
    n_neg: int = 350,
    r_inner: float = 1.1,
    r_outer: float = 2.20,
    band_inner: float = 0.15,
    band_outer: float = 0.22,
    freq: int = 3,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates two rings where the admissible radius depends on the angle when
    `wave_amp > 0`.  At `wave_amp=0` we recover the usual concentric rings.
    Increasing `wave_amp` causes the rings to "clock" into each other.
    """
    rng = np.random.default_rng(seed)

    theta_pos = rng.uniform(0, 2 * math.pi, n_pos)
    theta_neg = rng.uniform(0, 2 * math.pi, n_neg)

    center_pos = r_inner + wave_amp * np.sin(freq * theta_pos)
    center_neg = r_outer + wave_amp * np.sin(freq * theta_neg)

    rad_pos = center_pos + band_inner * (rng.random(n_pos) * 2.0 - 1.0)
    rad_neg = center_neg + band_outer * (rng.random(n_neg) * 2.0 - 1.0)

    # keep radii positive / finite
    # rad_pos = np.clip(rad_pos, 0.25, r_outer - 0.35)
    rad_neg = np.clip(rad_neg, 0.8, r_outer + 1.2)

    X_pos = np.c_[rad_pos * np.cos(theta_pos), rad_pos * np.sin(theta_pos)]
    X_neg = np.c_[rad_neg * np.cos(theta_neg), rad_neg * np.sin(theta_neg)]

    X = np.vstack([X_pos, X_neg]).astype(np.float32)
    y = np.hstack([np.ones(n_pos), -np.ones(n_neg)]).astype(np.int64)
    return X, y


def split_train_test(
    X: np.ndarray, y: np.ndarray, *, test_ratio: float = 0.2, seed: int = 0
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    split = int((1 - test_ratio) * len(X))
    train_idx, test_idx = perm[:split], perm[split:]
    return {
        "train_X": X[train_idx],
        "train_y": y[train_idx],
        "test_X": X[test_idx],
        "test_y": y[test_idx],
        "full_X": X,
        "full_y": y,
    }


# ------------------------ training utils -------------------------
class SimpleMLP(nn.Module):
    def __init__(self, hidden_dim: int = 64, depth: int = 3):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.GELU()]
        for _ in range(max(0, depth - 2)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_equivariant_model(
    X: np.ndarray,
    y: np.ndarray,
    lambda_eq: float,
    lambda_ne: float,
    *,
    epochs: int = 200,
    lr: float = 3e-3,
    device: str = DEVICE,
):
    model = ApproxHarmonicInvariantMLP(M=4, C=4, hidden_c=8).to(device)
    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor((y > 0).astype(np.float32), device=device)[:, None]

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X_t).real
        loss = loss_fn(logits, y_t)

        penalty = model.compute_non_equivariance_penalty()
        loss = loss + lambda_eq * penalty["equivariant_part"]
        loss = loss + lambda_ne * penalty["nonequiv_part"]
        loss.backward()
        opt.step()
    return model.eval()


def train_mlp_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 200,
    lr: float = 3e-3,
    hidden_dim: int = 96,
    depth: int = 4,
    device: str = DEVICE,
):
    model = SimpleMLP(hidden_dim=hidden_dim, depth=depth).to(device)
    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor((y > 0).astype(np.float32), device=device)[:, None]

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        opt.step()
    return model.eval()


@torch.no_grad()
def compute_accuracy(
    model, X: np.ndarray, y: np.ndarray, *, device: str = DEVICE
) -> float:
    logits = model(torch.tensor(X, device=device))
    if torch.is_complex(logits):
        logits = logits.real
    probs = torch.sigmoid(logits).cpu().numpy().ravel()
    preds = probs > 0.5
    return np.mean(preds == (y > 0))


@torch.no_grad()
def invariance_error(
    model,
    X: np.ndarray,
    *,
    n_rot: int = 30,
    n_batch: int = 256,
    device: str = DEVICE,
) -> Tuple[float, float]:
    """Returns mean & max |f(R_alpha * x) - f(x)|."""
    sel = np.random.choice(len(X), size=min(n_batch, len(X)), replace=False)
    pts = torch.tensor(X[sel], device=device)
    base = model(pts)
    if torch.is_complex(base):
        base = base.real

    errs = []
    for _ in range(n_rot):
        alpha = 2 * math.pi * random.random()
        rot = torch.tensor(
            [[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]],
            device=device,
        )
        out_rot = model(pts @ rot.T)
        if torch.is_complex(out_rot):
            out_rot = out_rot.real
        errs.append((out_rot - base).abs())

    errs = torch.stack(errs)
    return errs.mean().item(), errs.max().item()


# -------------------------- plotting --------------------------
def plot_wave_lambda_grid(
    models_eq: Sequence[Sequence[torch.nn.Module]],
    models_mlp: Sequence[Sequence[torch.nn.Module]],
    datasets: Sequence[Dict[str, np.ndarray]],
    wave_levels: Sequence[float],
    lambda_ne_values: Sequence[float],
    invariances: np.ndarray,
    *,
    lim: float = 3.5,
    grid: int = 300,
    n_lvls: int = 8,
    save_prefix: str = "figures/nn_wavey_rings_lambda_grid",
):
    rows, cols = len(lambda_ne_values), len(wave_levels)
    xs = np.linspace(-lim, lim, grid)
    xx, yy = np.meshgrid(xs, xs)
    pts = torch.tensor(np.c_[xx.ravel(), yy.ravel()].astype(np.float32), device=DEVICE)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4.0 * cols, 4.0 * rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    os.makedirs(Path(save_prefix).parent, exist_ok=True)
    cmap = plt.get_cmap("coolwarm")

    for i, lambda_ne in enumerate(lambda_ne_values):
        for j, wave_amp in enumerate(wave_levels):
            ax = axes[i, j]
            net_eq = models_eq[i][j]
            net_mlp = models_mlp[i][j]
            data = datasets[j]
            X = data["full_X"]
            y = data["full_y"]

            with torch.no_grad():
                score_eq = net_eq(pts).real
                score_eq = score_eq.cpu().numpy().reshape(xx.shape)
                score_mlp = net_mlp(pts)
                if torch.is_complex(score_mlp):
                    score_mlp = score_mlp.real
                score_mlp = score_mlp.cpu().numpy().reshape(xx.shape)

            ax.contourf(
                xx,
                yy,
                np.sign(score_eq),
                levels=[-1, 0, 1],
                colors=["#ffdddd", "#ddddff"],
                alpha=0.45,
            )
            vmin, vmax = np.percentile(score_eq, [5, 95])
            if abs(vmax - vmin) < 1e-2:
                vmax = vmin + 1e-2

            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            lvls = np.linspace(vmin, vmax, 2 * n_lvls + 1)[1:-1]
            ax.contour(
                xx,
                yy,
                score_eq,
                levels=lvls,
                linewidths=1.0,
                cmap=cmap,
                norm=norm,
                alpha=0.9,
                zorder=1,
            )
            ax.contour(
                xx,
                yy,
                score_eq,
                levels=[0],
                colors="#1f77b4",
                linewidths=2.2,
            )
            ax.contour(
                xx,
                yy,
                score_mlp,
                levels=[0],
                colors="#ff7f0e",
                linestyles="--",
                linewidths=2.0,
            )

            pos, neg = y > 0, y < 0
            ax.scatter(X[pos, 0], X[pos, 1], c="blue", edgecolors="k", s=16, zorder=5)
            ax.scatter(
                X[neg, 0],
                X[neg, 1],
                c="red",
                edgecolors="k",
                marker="X",
                s=20,
                zorder=5,
            )

            ax.set_aspect("equal")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xticks([])
            ax.set_yticks([])

            ax.text(
                0.5,
                -0.02,
                "$\\mathcal{E}(T)$" + f" = {invariances[i, j]:.2e}",
                ha="center",
                va="top",
                fontsize=18,
                transform=ax.transAxes,
            )

    def format_lambda(val: float) -> str:
        if val == 0:
            return "0"
        if val >= 1.0:
            return f"{val:.1f}"
        if val >= 0.1:
            txt = f"{val:.2f}"
            return txt.rstrip("0").rstrip(".")
        if val >= 0.01:
            txt = f"{val:.3f}"
            return txt.rstrip("0").rstrip(".")
        return f"{val:.0e}"

    def format_sigma(val: float) -> str:
        if val >= 1.0:
            return f"{val:.2f}"
        if val >= 0.1:
            return f"{val:.2f}".rstrip("0").rstrip(".")
        return f"{val:.3f}".rstrip("0").rstrip(".")

    for i, lambda_ne in enumerate(lambda_ne_values):
        axes[i, 0].set_ylabel(
            f"$\\lambda_\\perp={format_lambda(lambda_ne)}$", fontsize=20, labelpad=20
        )
    for j, wave_amp in enumerate(wave_levels):
        axes[-1, j].set_xlabel(
            f"$\\sigma_\\perp={format_sigma(wave_amp)}$", fontsize=20, labelpad=50
        )

    legend_handles = [
        Line2D([0], [0], color="#1f77b4", lw=2.5),
        Line2D([0], [0], color="#ff7f0e", lw=2.5, linestyle="--"),
    ]
    legend_labels = ["Approx equivariant", "Plain MLP"]
    # fig.legend(legend_handles, legend_labels, loc="upper center", ncol=2, fontsize=16)
    fig.savefig(f"{save_prefix}_one_row.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{save_prefix}_one_row.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------- main ---------------------------
if __name__ == "__main__":
    wave_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    lambda_ne_values = [1.0]
    lambda_eq = 1e-2
    epochs = 200

    datasets = []
    for j, amp in enumerate(wave_levels):
        combo_seed = 1234 + 13 * j

        X, y = generate_wavey_rings(wave_amp=amp, freq=5, seed=combo_seed)
        split = split_train_test(X, y, test_ratio=0.2, seed=999 + j)
        datasets.append(split)

    rows, cols = len(lambda_ne_values), len(wave_levels)
    models_eq = [[None for _ in wave_levels] for _ in lambda_ne_values]
    models_mlp = [[None for _ in wave_levels] for _ in lambda_ne_values]
    acc_eq = np.zeros((rows, cols))
    acc_mlp = np.zeros((rows, cols))
    invariances = np.zeros_like(acc_eq)

    for i, lambda_ne in enumerate(lambda_ne_values):
        for j, amp in enumerate(wave_levels):
            data = datasets[j]
            model = train_equivariant_model(
                data["train_X"],
                data["train_y"],
                lambda_eq,
                lambda_ne,
                epochs=epochs,
                device=DEVICE,
            )
            models_eq[i][j] = model.to(DEVICE)

            acc_eq[i, j] = compute_accuracy(
                model, data["test_X"], data["test_y"], device=DEVICE
            )
            inv_mean, inv_max = invariance_error(
                model,
                data["test_X"],
                n_rot=40,
                n_batch=len(data["test_X"]),
                device=DEVICE,
            )
            invariances[i, j] = inv_max

            mlp_model = train_mlp_model(
                data["train_X"],
                data["train_y"],
                epochs=epochs,
                device=DEVICE,
            )
            models_mlp[i][j] = mlp_model.to(DEVICE)
            acc_mlp[i, j] = compute_accuracy(
                mlp_model, data["test_X"], data["test_y"], device=DEVICE
            )

            print(
                f"[wave={amp:.2f}, lambda_perp={lambda_ne:.1e}] "
                f"acc_eq={acc_eq[i, j]:.3f}  acc_mlp={acc_mlp[i, j]:.3f}  E(T)={inv_max:.2e}"
            )

    plot_wave_lambda_grid(
        models_eq, models_mlp, datasets, wave_levels, lambda_ne_values, invariances
    )
