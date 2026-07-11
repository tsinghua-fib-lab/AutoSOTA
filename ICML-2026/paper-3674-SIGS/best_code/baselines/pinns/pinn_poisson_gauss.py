#!/usr/bin/env python3
"""
PINNs baseline for Poisson-Gauss 2D problems (2c, 3c, 4c).

Solves:  -nabla^2 u = f(x,y)   on [0,1]^2
         u = 0                  on boundary  (homogeneous Dirichlet)

where f is the source term derived from the Poisson-Gauss problem definition.
The PINN has NO knowledge of the solution — it only knows f and the BCs.

Compares against FEM (FEniCS) ground truth.

Usage:
    Step 1: Run fem_poisson_gauss.py to generate FEM solutions
    Step 2: Run this script to train PINNs and compare
"""

import argparse
import math
import time
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ========== Problem definitions ==========

PROBLEMS = {
    "pg2": {
        "name": "Poisson-Gauss 2 centers",
        "centers": [(0.3, 0.8), (0.7, 0.2)],
        "sigma": 0.12,
    },
    "pg3": {
        "name": "Poisson-Gauss 3 centers",
        "centers": [(0.3, 0.8), (0.7, 0.8), (0.5, 0.2)],
        "sigma": 0.12,
    },
    "pg4": {
        "name": "Poisson-Gauss 4 centers",
        "centers": [(0.3, 0.8), (0.7, 0.2), (0.5, 0.2), (0.4, 0.6)],
        "sigma": 0.12,
    },
}


def compute_f_source(xy, centers, sigma):
    """
    Compute the source term f(x,y) for the Poisson-Gauss problem.

    The source is defined as:
        f = -laplacian(u_manufactured)
    where u_manufactured = sin(pi*x)*sin(pi*y) * sum_i G_i
    and G_i = exp(-((x-cx_i)^2 + (y-cy_i)^2) / (2*sigma^2))

    This is a KNOWN forcing function — the PINN receives it as input data,
    just like FEniCS does. The PINN does NOT know the solution.
    """
    xy_ad = xy.clone().requires_grad_(True)
    x, y = xy_ad[:, 0:1], xy_ad[:, 1:2]

    # Manufactured u (only used to derive f, NOT used by the PINN)
    mask = torch.sin(math.pi * x) * torch.sin(math.pi * y)
    gsum = sum(
        torch.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
        for cx, cy in centers
    )
    u_manuf = mask * gsum

    grad_u = torch.autograd.grad(u_manuf, xy_ad, grad_outputs=torch.ones_like(u_manuf),
                                  create_graph=True)[0]
    u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xy_ad, grad_outputs=torch.ones_like(u_x),
                                create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, xy_ad, grad_outputs=torch.ones_like(u_y),
                                create_graph=True)[0][:, 1:2]

    f = -(u_xx + u_yy)
    return f.detach()


# ========== PINN network ==========

class PINN(nn.Module):
    """
    Plain fully-connected network. No hard BC encoding.
    The network directly outputs u(x,y).
    BCs are enforced via soft penalty in the loss.
    """
    def __init__(self, hidden_dim=64, num_layers=5, activation="tanh"):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim)]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, 1))
        self.layers = nn.ModuleList(layers)
        if activation == "tanh":
            self.act = torch.tanh
        elif activation == "sin":
            self.act = torch.sin
        else:
            self.act = torch.tanh

        # Xavier init
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, xy):
        """Network output u(x,y)."""
        h = xy
        for layer in self.layers[:-1]:
            h = self.act(layer(h))
        return self.layers[-1](h)


def sample_boundary(n_per_side, device, dtype=torch.float64):
    """Sample points on the boundary of [0,1]^2."""
    t = torch.linspace(0, 1, n_per_side, device=device, dtype=dtype).unsqueeze(1)
    zero = torch.zeros_like(t)
    one = torch.ones_like(t)
    # bottom (y=0), top (y=1), left (x=0), right (x=1)
    bc_points = torch.cat([
        torch.cat([t, zero], dim=1),     # bottom
        torch.cat([t, one], dim=1),      # top
        torch.cat([zero, t], dim=1),     # left
        torch.cat([one, t], dim=1),      # right
    ], dim=0)
    return bc_points


# ========== Training ==========

def train_pinn(problem_key, epochs=20000, lr=1e-3, n_interior=4096,
               n_bc_per_side=128, hidden_dim=64, num_layers=5, seed=42,
               device="cpu", scheduler_step=5000, scheduler_gamma=0.5,
               lambda_bc=100.0):
    """Train a PINN for the given Poisson-Gauss problem."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    prob = PROBLEMS[problem_key]
    centers = prob["centers"]
    sigma = prob["sigma"]

    model = PINN(hidden_dim=hidden_dim, num_layers=num_layers).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma
    )

    loss_history = []
    best_loss = float("inf")
    best_state = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # --- PDE residual loss ---
        # Sample random interior collocation points
        xy_int = torch.rand(n_interior, 2, device=device, dtype=torch.float64)
        xy_int.requires_grad_(True)

        # PINN prediction and its Laplacian
        u_pred = model(xy_int)

        grad_u = torch.autograd.grad(
            u_pred, xy_int, grad_outputs=torch.ones_like(u_pred),
            create_graph=True
        )[0]
        u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]

        u_xx = torch.autograd.grad(
            u_x, xy_int, grad_outputs=torch.ones_like(u_x),
            create_graph=True
        )[0][:, 0:1]
        u_yy = torch.autograd.grad(
            u_y, xy_int, grad_outputs=torch.ones_like(u_y),
            create_graph=True
        )[0][:, 1:2]

        laplacian_u = u_xx + u_yy

        # Source term f (known, same as what FEniCS receives)
        f = compute_f_source(xy_int.detach(), centers, sigma).to(device)

        # PDE residual: -laplacian(u) = f  =>  laplacian(u) + f = 0
        residual = laplacian_u + f
        loss_pde = torch.mean(residual**2)

        # --- Boundary condition loss ---
        # u = 0 on boundary of [0,1]^2
        xy_bc = sample_boundary(n_bc_per_side, device)
        u_bc = model(xy_bc)
        loss_bc = torch.mean(u_bc**2)

        # Total loss
        loss = loss_pde + lambda_bc * loss_bc
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        loss_pde_val = loss_pde.item()
        loss_bc_val = loss_bc.item()
        loss_history.append(loss_pde_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 2000 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  [{problem_key}] Epoch {epoch:>6d}/{epochs}  "
                  f"PDE={loss_pde_val:.4e}  BC={loss_bc_val:.4e}  "
                  f"total={loss_val:.4e}  ({elapsed:.1f}s)")

    # Restore best
    model.load_state_dict(best_state)
    total_time = time.time() - t0

    return model, loss_history, total_time


# ========== Evaluation ==========

def evaluate_model(model, problem_key, n_eval=200, device="cpu", fem_dir="results_pinns"):
    """Evaluate PINN on a grid and compute rel L2 error vs FEM solution."""
    prob = PROBLEMS[problem_key]

    x = np.linspace(0, 1, n_eval)
    y = np.linspace(0, 1, n_eval)
    X, Y = np.meshgrid(x, y)
    xy_flat = np.stack([X.ravel(), Y.ravel()], axis=1)

    # PINN prediction
    model.eval()
    with torch.no_grad():
        xy_t = torch.tensor(xy_flat, dtype=torch.float64, device=device)
        u_pred_flat = model(xy_t).cpu().numpy().reshape(n_eval, n_eval)

    # Load FEM ground truth
    fem_path = os.path.join(fem_dir, f"fem_{problem_key}.npz")
    if os.path.exists(fem_path):
        fem_data = np.load(fem_path)
        u_fem = fem_data["u_fem"]
        X_fem, Y_fem = fem_data["X"], fem_data["Y"]
        # Check if grids match
        if u_fem.shape == (n_eval, n_eval):
            u_true = u_fem
            gt_label = "FEM (FEniCS)"
        else:
            # Interpolate FEM to our grid
            from scipy.interpolate import RegularGridInterpolator
            x_fem = X_fem[0, :]
            y_fem = Y_fem[:, 0]
            interp = RegularGridInterpolator((y_fem, x_fem), u_fem)
            pts = np.stack([Y.ravel(), X.ravel()], axis=1)
            u_true = interp(pts).reshape(n_eval, n_eval)
            gt_label = "FEM (FEniCS, interpolated)"
        print(f"  [{problem_key}] Loaded FEM ground truth from {fem_path}")
    else:
        print(f"  [{problem_key}] WARNING: FEM file not found at {fem_path}")
        print(f"  [{problem_key}] Run fem_poisson_gauss.py first!")
        print(f"  [{problem_key}] Falling back to manufactured solution")
        centers = prob["centers"]
        sigma = prob["sigma"]
        mask = np.sin(np.pi * X) * np.sin(np.pi * Y)
        gsum = sum(
            np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
            for cx, cy in centers
        )
        u_true = mask * gsum
        gt_label = "Manufactured"

    # Relative L2 error
    rel_l2 = np.linalg.norm(u_pred_flat - u_true) / (np.linalg.norm(u_true) + 1e-12)

    # Max absolute error
    max_err = np.max(np.abs(u_pred_flat - u_true))

    return {
        "X": X, "Y": Y,
        "u_true": u_true,
        "u_pred": u_pred_flat,
        "rel_l2": rel_l2,
        "max_abs_err": max_err,
        "gt_label": gt_label,
    }


# ========== Plotting ==========

def plot_results(results, problem_key, loss_history, save_dir="results_pinns"):
    """Generate comparison plots: FEM, PINN, error, loss curve."""
    os.makedirs(save_dir, exist_ok=True)
    prob = PROBLEMS[problem_key]

    X = results["X"]
    Y = results["Y"]
    u_true = results["u_true"]
    u_pred = results["u_pred"]
    error = np.abs(u_pred - u_true)
    gt_label = results.get("gt_label", "FEM")

    fig = plt.figure(figsize=(18, 4.5))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 1])

    vmin = min(u_true.min(), u_pred.min())
    vmax = max(u_true.max(), u_pred.max())

    # (a) Ground truth (FEM)
    ax0 = fig.add_subplot(gs[0])
    c0 = ax0.contourf(X, Y, u_true, levels=50, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    plt.colorbar(c0, ax=ax0, fraction=0.046)
    ax0.set_title(gt_label)
    ax0.set_xlabel("x"); ax0.set_ylabel("y")
    ax0.set_aspect("equal")

    # (b) PINN prediction
    ax1 = fig.add_subplot(gs[1])
    c1 = ax1.contourf(X, Y, u_pred, levels=50, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    plt.colorbar(c1, ax=ax1, fraction=0.046)
    ax1.set_title("PINN prediction")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    ax1.set_aspect("equal")

    # (c) Absolute error
    ax2 = fig.add_subplot(gs[2])
    c2 = ax2.contourf(X, Y, error, levels=50, cmap="hot_r")
    plt.colorbar(c2, ax=ax2, fraction=0.046)
    ax2.set_title(f"|Error|  (rel L2 = {results['rel_l2']:.4e})")
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    ax2.set_aspect("equal")

    # (d) Loss curve
    ax3 = fig.add_subplot(gs[3])
    ax3.semilogy(loss_history)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("PDE residual loss")
    ax3.set_title("Training loss")
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f"{prob['name']}  |  PINN vs {gt_label}  |  Rel L2 = {results['rel_l2']:.4e}", fontsize=13)
    plt.tight_layout()

    fname = os.path.join(save_dir, f"pinn_{problem_key}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {fname}")
    return fname


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser(description="PINNs for Poisson-Gauss 2D problems")
    parser.add_argument("--problems", nargs="+", default=["pg2", "pg3", "pg4"],
                        choices=["pg2", "pg3", "pg4"],
                        help="Which problems to run")
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-interior", type=int, default=4096,
                        help="Number of interior collocation points per epoch")
    parser.add_argument("--n-bc-per-side", type=int, default=128,
                        help="Number of BC points per side")
    parser.add_argument("--lambda-bc", type=float, default=100.0,
                        help="BC penalty weight")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="results_pinns")
    parser.add_argument("--n-eval", type=int, default=200,
                        help="Grid resolution for evaluation")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Problems: {args.problems}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Hidden: {args.hidden_dim}, "
          f"Layers: {args.num_layers}, Interior pts: {args.n_interior}")
    print(f"BC: {args.n_bc_per_side} pts/side, lambda={args.lambda_bc}")
    print("=" * 60)

    summary = []

    for pk in args.problems:
        print(f"\n{'='*60}")
        print(f"  Training PINN for: {PROBLEMS[pk]['name']}")
        print(f"  Centers: {PROBLEMS[pk]['centers']}")
        print(f"{'='*60}")

        model, loss_history, wall_time = train_pinn(
            pk,
            epochs=args.epochs,
            lr=args.lr,
            n_interior=args.n_interior,
            n_bc_per_side=args.n_bc_per_side,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            seed=args.seed,
            device=device,
            lambda_bc=args.lambda_bc,
        )

        results = evaluate_model(model, pk, n_eval=args.n_eval, device=device,
                                  fem_dir=args.save_dir)
        plot_results(results, pk, loss_history, save_dir=args.save_dir)

        row = {
            "problem": pk,
            "rel_l2": results["rel_l2"],
            "max_abs_err": results["max_abs_err"],
            "wall_time_s": wall_time,
            "final_pde_loss": loss_history[-1],
            "best_pde_loss": min(loss_history),
            "gt_label": results["gt_label"],
        }
        summary.append(row)
        print(f"\n  {pk}: Rel L2 = {results['rel_l2']:.6e} (vs {results['gt_label']}), "
              f"Max |err| = {results['max_abs_err']:.6e}, "
              f"Time = {wall_time:.1f}s")

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY: PINNs on Poisson-Gauss problems")
    print(f"{'='*60}")
    print(f"  {'Problem':<10} {'Rel L2':>12} {'Max |err|':>12} {'Time (s)':>10} {'Ground truth':>15}")
    print(f"  {'-'*62}")
    for r in summary:
        print(f"  {r['problem']:<10} {r['rel_l2']:>12.6e} {r['max_abs_err']:>12.6e} "
              f"{r['wall_time_s']:>10.1f} {r['gt_label']:>15}")

    # Save summary CSV
    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, "pinn_summary.csv")
    with open(csv_path, "w") as f:
        f.write("problem,rel_l2,max_abs_err,wall_time_s,final_pde_loss,best_pde_loss,ground_truth\n")
        for r in summary:
            f.write(f"{r['problem']},{r['rel_l2']:.8e},{r['max_abs_err']:.8e},"
                    f"{r['wall_time_s']:.1f},{r['final_pde_loss']:.8e},{r['best_pde_loss']:.8e},"
                    f"{r['gt_label']}\n")
    print(f"\n  Summary CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
