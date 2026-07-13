"""Plot the learned Monge maps for the cached grid pair and FCOT-Separable models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimal_transport.ot_fc_sep_map import FCOTSeparable
from tools.utils import (
    L22_1d,
    nL22_1d,
    inverse_L22x,
    inverse_nL22x,
)


DIM = 2
RADIUS = 4.0
X_ACCURACY = 2e-3
Y_ACCURACY = 2e-3
NY = int((2 * RADIUS) / Y_ACCURACY) + 1
N_PARAMS = NY * DIM

TITLE_FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 14
FIG_SIZE = (18, 6.5)
LAYOUT_PAD = 1.0

OUTER_LR = 1e-2
TEMP_MIN = 1.0
TEMP_MAX = 60.0
TEMP_WARMUP_ITERS = 2_500
REACTIVATE_EVERY = 50
REACTIVATE_EPS = 1e-3
FULL_REFRESH_EVERY = 300
COARSE_X_FACTOR = 100
COARSE_TOP_K = 4
COARSE_WINDOW = 1

KERNEL_CONFIGS = {
    "L22_1d": {
        "kernel": L22_1d,
        "inverse": inverse_L22x,
        "label": "L22",
    },
    "nL22_1d": {
        "kernel": nL22_1d,
        "inverse": inverse_nL22x,
        "label": "nL22",
    },
}


def _latest_file(directory: Path, pattern: str) -> Path:
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No files matching {pattern} under {directory}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_grid_XY(tmp_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the cached X, Y pair used for the 2D grid experiments."""
    grid_x_dir = tmp_dir / "grid_X"
    grid_y_dir = tmp_dir / "grid_Y"
    if not grid_x_dir.is_dir() or not grid_y_dir.is_dir():
        raise FileNotFoundError("Grid directories tmp/grid_X or tmp/grid_Y are missing.")

    latest_x = _latest_file(grid_x_dir, "*.npz")
    latest_y = _latest_file(grid_y_dir, "*.npz")

    X = torch.load(latest_x, map_location="cpu")["x"].float().clamp(-RADIUS, RADIUS)
    Y_vert = torch.load(latest_y, map_location="cpu")["x"].float().clamp(-RADIUS, RADIUS)
    Y = Y_vert[:, [1, 0]]
    return X, Y


def build_solver(kernel_tag: str) -> FCOTSeparable:
    """Construct the FCOT-Separable solver architecture for the given kernel."""
    config = KERNEL_CONFIGS[kernel_tag]
    solver = FCOTSeparable.initialize_right_architecture(
        dim=DIM,
        radius=RADIUS,
        n_params=N_PARAMS,
        x_accuracy=X_ACCURACY,
        kernel_1d=config["kernel"],
        inverse_kx=config["inverse"],
        outer_lr=OUTER_LR,
        temp_min=TEMP_MIN,
        temp_max=TEMP_MAX,
        temp_warmup_iters=TEMP_WARMUP_ITERS,
        reactivate_every=REACTIVATE_EVERY,
        reactivate_eps=REACTIVATE_EPS,
        full_refresh_every=FULL_REFRESH_EVERY,
        cache_gradients=True,
        coarse_x_factor=COARSE_X_FACTOR,
        coarse_top_k=COARSE_TOP_K,
        coarse_window=COARSE_WINDOW,
    )
    return solver


def load_solver_from_checkpoint(tmp_dir: Path, kernel_tag: str) -> tuple[FCOTSeparable, Path]:
    """Load the trained model checkpoint for the provided kernel tag."""
    pattern = f"FCOTSeparable_dim{DIM}_kernel-{kernel_tag}_*.pt"
    checkpoint = _latest_file(tmp_dir, pattern)
    solver = build_solver(kernel_tag)
    solver.load(checkpoint)
    return solver, checkpoint


def compute_transport(solver: FCOTSeparable, X: torch.Tensor) -> torch.Tensor:
    """Apply the fitted Monge map to the input grid."""
    return solver.transport_X_to_Y(X, selection_mode="soft", snap_to_grid=False).cpu()


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detach tensor, move to CPU, and return NumPy view."""
    return tensor.detach().cpu().numpy()


def _choose_indices(total: int, count: int) -> torch.Tensor:
    """Return up to `count` unique indices sampled from `[0, total)`."""
    if count <= 0 or total == 0:
        return torch.empty(0, dtype=torch.long)
    if count >= total:
        return torch.arange(total, dtype=torch.long)
    return torch.randperm(total)[:count]


def plot_monge_map(
    X: torch.Tensor,
    Y: torch.Tensor,
    T_L22: torch.Tensor,
    T_nL22: torch.Tensor,
    output_path: Path,
    subsample_l22: int,
    subsample_nl22: int,
) -> None:
    """Scatter the input/target data and overlay the learned transports."""
    X_np = _tensor_to_numpy(X)
    Y_np = _tensor_to_numpy(Y)
    T_L22_np = _tensor_to_numpy(T_L22)
    T_nL22_np = _tensor_to_numpy(T_nL22)
    size = 2
    markersize = 4.
    markerscale = 2.
    colors = {
        "X": "blue",
        "Y": "red",
        "T": "yellow",
    }

    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE)
    axes = axes.tolist()

    axes[0].scatter(
        X_np[:, 0], X_np[:, 1], s=size, alpha=0.6,
        color=colors["X"],
    )
    axes[0].scatter(
        Y_np[:, 0], Y_np[:, 1], s=size, alpha=0.6,
        color=colors["Y"],
    )
    axes[0].set_title("X and Y marginals", fontsize=TITLE_FONT_SIZE)
    axes[0].set_aspect("equal", "box")
    legend_handles_0 = [
        Line2D(
            [0], [0], marker="o", color="black", markerfacecolor=colors["X"],
            markeredgecolor="black", linewidth=0, markersize=markersize, label="X"
        ),
        Line2D(
            [0], [0], marker="o", color="black", markerfacecolor=colors["Y"],
            markeredgecolor="black", linewidth=0, markersize=markersize, label="Y"
        ),
    ]
    axes[0].legend(handles=legend_handles_0, fontsize=12, loc="lower left", markerscale=markerscale)
    axes[0].set_xlabel("Dimension 1", fontsize=AXIS_LABEL_FONT_SIZE)
    axes[0].set_ylabel("Dimension 2", fontsize=AXIS_LABEL_FONT_SIZE)

    axes[1].scatter(
        X_np[:, 0], X_np[:, 1], s=size, alpha=0.6,
        color=colors["X"],
    )
    axes[1].scatter(
        Y_np[:, 0], Y_np[:, 1], s=size, alpha=0.6,
        color=colors["Y"],
    )
    axes[1].scatter(
        T_L22_np[:, 0], T_L22_np[:, 1], s=size, alpha=0.6,
        color=colors["T"],
    )
    axes[1].set_title("Monge map for $c(x,y)=||x-y||_2^2$", fontsize=TITLE_FONT_SIZE)
    axes[1].set_aspect("equal", "box")
    axes[1].set_xlabel("Dimension 1", fontsize=AXIS_LABEL_FONT_SIZE)
    axes[1].set_ylabel("Dimension 2", fontsize=AXIS_LABEL_FONT_SIZE)
    indices_l22 = _choose_indices(X.shape[0], subsample_l22)
    for idx in indices_l22.tolist():
        axes[1].plot(
            [X_np[idx, 0], T_L22_np[idx, 0]],
            [X_np[idx, 1], T_L22_np[idx, 1]],
            color="black",
            linewidth=0.5,
            alpha=0.9,
        )
    legend_handles_1 = [
        Line2D(
            [0], [0], marker="o", color="black", markerfacecolor=colors["X"],
            markeredgecolor="black", linewidth=0, markersize=markersize, label="X"
        ),
        Line2D(
            [0], [0], marker="o", color="black", markerfacecolor=colors["Y"],
            markeredgecolor="black", linewidth=0, markersize=markersize, label="Y"
        ),
        Line2D(
            [0], [0], marker="o", color="black", markerfacecolor=colors["T"],
            markeredgecolor="black", linewidth=0, markersize=markersize, label="T(X)"
        ),
        Line2D([0], [0], color="black", linewidth=0.9, label="X--T(X)"),
    ]
    axes[1].legend(handles=legend_handles_1, fontsize=12, loc="lower left", markerscale=markerscale)

    axes[2].scatter(
        X_np[:, 0], X_np[:, 1], s=size, alpha=0.6,
        color=colors["X"],
    )
    axes[2].scatter(
        Y_np[:, 0], Y_np[:, 1], s=size, alpha=0.6,
        color=colors["Y"],
    )
    axes[2].scatter(
        T_nL22_np[:, 0], T_nL22_np[:, 1], s=size, alpha=0.6,
        color=colors["T"],
    )
    axes[2].set_title("Monge map for $c(x,y) = -||x-y||_2^2$", fontsize=TITLE_FONT_SIZE)
    axes[2].set_aspect("equal", "box")
    axes[2].set_xlabel("Dimension 1", fontsize=AXIS_LABEL_FONT_SIZE)
    axes[2].set_ylabel("Dimension 2", fontsize=AXIS_LABEL_FONT_SIZE)
    indices_nl22 = _choose_indices(X.shape[0], subsample_nl22)
    for idx in indices_nl22.tolist():
        axes[2].plot(
            [X_np[idx, 0], T_nL22_np[idx, 0]],
            [X_np[idx, 1], T_nL22_np[idx, 1]],
            color="black",
            linewidth=1,
            alpha=0.6,
        )
    legend_handles_2 = [
        Line2D(
            [0], [0], marker="o", color="black", markerfacecolor=colors["X"],
            markeredgecolor="black", linewidth=0, markersize=markersize, label="X"
        ),
        Line2D(
            [0], [0], marker="o", color="black", markerfacecolor=colors["Y"],
            markeredgecolor="black", linewidth=0, markersize=markersize, label="Y"
        ),
        Line2D(
            [0], [0], marker="o", color="black", markerfacecolor=colors["T"],
            markeredgecolor="black", linewidth=0, markersize=markersize, label="T(X)"
        ),
        Line2D([0], [0], color="black", linewidth=0.9, label="X--T(X)"),
    ]
    axes[2].legend(handles=legend_handles_2, fontsize=12, loc="lower left", markerscale=markerscale)

    fig.tight_layout(pad=LAYOUT_PAD)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved Monge map figure to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Monge map for cached grid + FCOT-Separable models.")
    parser.add_argument("--tmp-dir", type=Path, default=Path("tmp"), help="Directory containing cached data/models.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path for the figure.")
    parser.add_argument("--subsample-l22", type=int, default=2000, help="Number of transports in the L22 panel to connect with black lines.")
    parser.add_argument("--subsample-nl22", type=int, default=100, help="Number of transports in the nL22 panel to connect with black lines.")
    args = parser.parse_args()

    tmp_dir = args.tmp_dir
    if not tmp_dir.is_dir():
        raise FileNotFoundError(f"{tmp_dir} does not exist or is not a directory.")

    X, Y = load_grid_XY(tmp_dir)
    solver_L22, ckpt_L22 = load_solver_from_checkpoint(tmp_dir, "L22_1d")
    solver_nL22, ckpt_nL22 = load_solver_from_checkpoint(tmp_dir, "nL22_1d")
    print(f"Loaded checkpoints:\n  L22: {ckpt_L22}\n  nL22: {ckpt_nL22}")

    T_L22 = compute_transport(solver_L22, X)
    T_nL22 = compute_transport(solver_nL22, X)

    output_path = args.output or tmp_dir / "monge_plots" / "monge_map.png"
    plot_monge_map(
        X,
        Y,
        T_L22,
        T_nL22,
        output_path,
        args.subsample_l22,
        args.subsample_nl22,
    )


if __name__ == "__main__":
    main()
