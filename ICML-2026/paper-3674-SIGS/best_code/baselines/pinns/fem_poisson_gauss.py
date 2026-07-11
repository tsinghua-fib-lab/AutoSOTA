#!/usr/bin/env python3
"""
FEM (FEniCS) solver for Poisson-Gauss 2D problems (2c, 3c, 4c).

Solves:  -nabla^2 u = f(x,y)   on [0,1]^2
         u = 0                  on boundary

where f = -nabla^2 u_manufactured, with:
    u_manufactured = sin(pi*x)*sin(pi*y) * sum_i exp(-((x-cx_i)^2+(y-cy_i)^2)/(2*sigma^2))

Saves u_FEM evaluated on a regular grid as .npz files for PINN comparison.

Run with the fenics-env conda environment:
    export PATH=/home/oroikon/miniconda3/envs/fenics-env/bin:$PATH
    python fem_poisson_gauss.py
"""

import os
import math
import numpy as np
import dolfin as df

# Suppress FEniCS logging
df.set_log_level(df.LogLevel.WARNING)

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


def build_source_expression(centers, sigma):
    """Build the FEniCS Expression for f = -laplacian(u_manufactured)."""
    # u = sin(pi*x)*sin(pi*y) * sum_i G_i
    # where G_i = exp(-((x-cx)^2+(y-cy)^2)/(2*sigma^2))
    # f = -laplacian(u) is computed symbolically via sympy, then converted
    # to a FEniCS Expression string.
    import sympy as sp

    x, y = sp.symbols('x[0] x[1]', real=True)
    pi = sp.pi
    mask = sp.sin(pi * x) * sp.sin(pi * y)
    sig2 = 2 * sigma**2

    gsum = sum(
        sp.exp(-((x - cx)**2 + (y - cy)**2) / sig2)
        for cx, cy in centers
    )

    u_sym = mask * gsum
    lap_u = sp.diff(u_sym, x, 2) + sp.diff(u_sym, y, 2)
    f_sym = -lap_u
    f_sym = sp.simplify(f_sym)

    # Convert to C++ string for FEniCS
    f_code = sp.ccode(f_sym)
    # Replace x[0], x[1] notation (already correct from symbol names)
    return f_code, u_sym


def solve_poisson_fem(problem_key, mesh_n=128, degree=4, eval_n=200, save_dir="results_pinns"):
    """Solve Poisson-Gauss with FEniCS P{degree} elements on mesh_n x mesh_n mesh."""
    prob = PROBLEMS[problem_key]
    centers = prob["centers"]
    sigma = prob["sigma"]

    print(f"  [{problem_key}] Building source term...")
    f_code, u_sym = build_source_expression(centers, sigma)

    print(f"  [{problem_key}] Creating mesh ({mesh_n}x{mesh_n}) and function space (P{degree})...")
    mesh = df.UnitSquareMesh(mesh_n, mesh_n)
    V = df.FunctionSpace(mesh, "Lagrange", degree)

    # Dirichlet BC: u = 0 on boundary
    bc = df.DirichletBC(V, df.Constant(0.0), "on_boundary")

    # Source term
    f_expr = df.Expression(f_code, degree=degree + 2)

    # Variational form: find u in V such that
    #   integral(grad(u) . grad(v) dx) = integral(f * v dx)  for all v in V
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    a = df.inner(df.grad(u), df.grad(v)) * df.dx
    L = f_expr * v * df.dx

    print(f"  [{problem_key}] Solving...")
    u_h = df.Function(V)
    df.solve(a == L, u_h, bc)

    # Evaluate on a regular grid
    print(f"  [{problem_key}] Evaluating on {eval_n}x{eval_n} grid...")
    x_eval = np.linspace(0, 1, eval_n)
    y_eval = np.linspace(0, 1, eval_n)
    X, Y = np.meshgrid(x_eval, y_eval)

    u_fem = np.zeros_like(X)
    for i in range(eval_n):
        for j in range(eval_n):
            try:
                u_fem[i, j] = u_h(df.Point(X[i, j], Y[i, j]))
            except RuntimeError:
                u_fem[i, j] = 0.0  # boundary points

    # Save
    os.makedirs(save_dir, exist_ok=True)
    npz_path = os.path.join(save_dir, f"fem_{problem_key}.npz")
    np.savez(npz_path, X=X, Y=Y, u_fem=u_fem,
             centers=np.array(centers), sigma=sigma,
             mesh_n=mesh_n, degree=degree, eval_n=eval_n)
    print(f"  [{problem_key}] Saved FEM solution to {npz_path}")

    return X, Y, u_fem


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FEM solver for Poisson-Gauss problems")
    parser.add_argument("--problems", nargs="+", default=["pg2", "pg3", "pg4"],
                        choices=["pg2", "pg3", "pg4"])
    parser.add_argument("--mesh-n", type=int, default=128,
                        help="Mesh resolution (default: 128x128)")
    parser.add_argument("--degree", type=int, default=4,
                        help="FE polynomial degree (default: 4, matching paper)")
    parser.add_argument("--eval-n", type=int, default=200,
                        help="Evaluation grid resolution")
    parser.add_argument("--save-dir", type=str, default="results_pinns")
    args = parser.parse_args()

    for pk in args.problems:
        print(f"\n{'='*50}")
        print(f"  FEM solve: {PROBLEMS[pk]['name']}")
        print(f"  Mesh: {args.mesh_n}x{args.mesh_n}, P{args.degree}")
        print(f"{'='*50}")
        X, Y, u_fem = solve_poisson_fem(
            pk, mesh_n=args.mesh_n, degree=args.degree,
            eval_n=args.eval_n, save_dir=args.save_dir
        )
        print(f"  u_fem range: [{u_fem.min():.6f}, {u_fem.max():.6f}]")


if __name__ == "__main__":
    main()
