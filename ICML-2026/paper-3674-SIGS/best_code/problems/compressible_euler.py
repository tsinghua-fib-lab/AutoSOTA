"""Compressible Euler PDE Problem Definition (paper §4.2).

2D steady compressible Euler system with seeded random Fourier-mode manufactured solution.
Use create_compressible_euler_problem(seed=42) to get a reproducible problem dict.
"""

import sympy as sp
import numpy as np


def create_compressible_euler_problem(seed=42, K=2):
    """
    Creates a single, reproducible 2D steady compressible Euler problem
    with seeded random coefficients.

    Args:
        seed: Random seed for reproducible coefficients (default: 42)

    Returns:
        dict: Problem specification with operators, forcing terms, etc.
    """

    # Set seed for reproducibility - always produces the same problem
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # 0. Symbols and unknown fields
    # ------------------------------------------------------------------
    x, y = sp.symbols('x y', real=True)
    rho = sp.Function('rho')(x, y)
    u   = sp.Function('u')(x, y)
    v   = sp.Function('v')(x, y)
    p   = sp.Function('p')(x, y)

    # Parameters
    gamma = sp.Rational(7, 5)   # 1.4 as exact rational
    # K: number of Fourier modes in each direction (user can pass K)
    r     = sp.Rational(1, 2)   # 0.5

    # ------------------------------------------------------------------
    # 1. Generate random coefficients (always the same with fixed seed)
    # ------------------------------------------------------------------
    A = np.random.uniform(-1, 1, (K, K))
    B = np.random.uniform(-1, 1, (K, K))
    C = np.random.uniform(-1, 1, (K, K))
    D = np.random.uniform(-1, 1, (K, K))

    # ------------------------------------------------------------------
    # 2. Build manufactured solution with random coefficients
    # ------------------------------------------------------------------
    # Use sin(pi*i*x) * sin(pi*j*y) templates (zero-mean, orthogonal on unit box)
    rho_star = sp.exp((1 / K**2) * sum(
        A[i-1, j-1] * (i**2 + j**2)**r *
        sp.sin(sp.pi * i * x) * sp.sin(sp.pi * j * y)
        for i in range(1, K + 1) for j in range(1, K + 1)
    ))

    u_star = (sp.pi / K**2) * sum(
        B[i-1, j-1] * (i**2 + j**2)**r *
        sp.sin(sp.pi * i * x) * sp.sin(sp.pi * j * y)
        for i in range(1, K + 1) for j in range(1, K + 1)
    )

    v_star = (sp.pi / K**2) * sum(
        C[i-1, j-1] * (i**2 + j**2)**r *
        sp.sin(sp.pi * i * x) * sp.sin(sp.pi * j * y)
        for i in range(1, K + 1) for j in range(1, K + 1)
    )

    p_star = sp.exp((1 / K**2) * sum(
        D[i-1, j-1] * (i**2 + j**2)**r *
        sp.sin(sp.pi * i * x) * sp.sin(sp.pi * j * y)
        for i in range(1, K + 1) for j in range(1, K + 1)
    ))

    # Total energy
    E = p / (gamma - 1) + sp.Rational(1, 2) * rho * (u**2 + v**2)
    E_star = p_star / (gamma - 1) + sp.Rational(1, 2) * rho_star * (u_star**2 + v_star**2)

    # ------------------------------------------------------------------
    # 3. PDE operators L[rho,u,v,p]
    # ------------------------------------------------------------------
    op_rho = sp.diff(rho * u, x) + sp.diff(rho * v, y)
    op_u   = sp.diff(rho * u**2 + p, x) + sp.diff(rho * u * v, y)
    op_v   = sp.diff(rho * u * v, x) + sp.diff(rho * v**2 + p, y)
    op_E   = sp.diff((E + p) * u, x) + sp.diff((E + p) * v, y)

    # ------------------------------------------------------------------
    # 4. Forcing terms: plug manufactured solution into operators
    # ------------------------------------------------------------------
    subs_star = {
        rho: rho_star,
        u:   u_star,
        v:   v_star,
        p:   p_star,
        E:   E_star,
    }

    f_rho = sp.sympify(op_rho.subs(subs_star))
    f_u   = sp.sympify(op_u.subs(subs_star))
    f_v   = sp.sympify(op_v.subs(subs_star))
    f_E   = sp.sympify(op_E.subs(subs_star))

    # ------------------------------------------------------------------
    # 5. Boundary conditions
    # ------------------------------------------------------------------
    bc_rho = [
        f"rho(x=0, y) = {str(sp.sympify(rho_star.subs(x, 0)))}",
        f"rho(x=1, y) = {str(sp.sympify(rho_star.subs(x, 1)))}",
        f"rho(x, y=0) = {str(sp.sympify(rho_star.subs(y, 0)))}",
        f"rho(x, y=1) = {str(sp.sympify(rho_star.subs(y, 1)))}",
    ]
    bc_u = [
        f"u(x=0, y) = {str(sp.sympify(u_star.subs(x, 0)))}",
        f"u(x=1, y) = {str(sp.sympify(u_star.subs(x, 1)))}",
        f"u(x, y=0) = {str(sp.sympify(u_star.subs(y, 0)))}",
        f"u(x, y=1) = {str(sp.sympify(u_star.subs(y, 1)))}",
    ]
    bc_v = [
        f"v(x=0, y) = {str(sp.sympify(v_star.subs(x, 0)))}",
        f"v(x=1, y) = {str(sp.sympify(v_star.subs(x, 1)))}",
        f"v(x, y=0) = {str(sp.sympify(v_star.subs(y, 0)))}",
        f"v(x, y=1) = {str(sp.sympify(v_star.subs(y, 1)))}",
    ]
    bc_p = [
        f"p(x=0, y) = {str(sp.sympify(p_star.subs(x, 0)))}",
        f"p(x=1, y) = {str(sp.sympify(p_star.subs(x, 1)))}",
        f"p(x, y=0) = {str(sp.sympify(p_star.subs(y, 0)))}",
        f"p(x, y=1) = {str(sp.sympify(p_star.subs(y, 1)))}",
    ]

    # ------------------------------------------------------------------
    # 6. Final problem dict
    # ------------------------------------------------------------------
    problem = {
        "problem_type": "CompressibleEuler2D-System-Seeded",

        # Variables
        "system_variables": ["rho", "u", "v", "p"],

        # Symbolic operators L[rho,u,v,p]
        "operators_symbolic": {
            "rho": op_rho,
            "u":   op_u,
            "v":   op_v,
            "E":   op_E,
        },

        # Forcing terms from manufactured solution
        "F_symbolic": {
            "rho": f_rho,
            "u":   f_u,
            "v":   f_v,
            "E":   f_E,
        },

        # Fixed manufactured solution (for reference / later checks)
        "manufactured_solution": {
            "rho": rho_star,
            "u":   u_star,
            "v":   v_star,
            "p":   p_star,
            "E":   E_star,
        },

        # Domain & symbols
        "symbols": {"x": x, "y": y},
        "mesh": {
            "x": {"start": 0.0, "end": 1.0, "points": 128},
            "y": {"start": 0.0, "end": 1.0, "points": 128},
        },

        # "boundary_conditions": {
        #     "rho": bc_rho,
        #     "u":   bc_u,
        #     "v":   bc_v,
        #     "p":   bc_p,
        # }

        "boundary_conditions" : {
        "type": "periodic",
        "variables": ["rho", "u", "v", "p"],
        "directions": ["x", "y"],
    }
,
        "initial_conditions": [],  # steady problem

        # Parameters (including seed for reproducibility)
        "parameters": {
            "gamma": float(gamma),
            "K": int(K),
            "r": float(r),
            "seed": seed,
        },

        # Store coefficients for reference
        "coefficients": {
            "A": A.tolist(),
            "B": B.tolist(),
            "C": C.tolist(),
            "D": D.tolist(),
        }
    }

    return problem


def build_mesh(problem):
    """Helper function to build mesh from problem dict"""
    mx = problem["mesh"]["x"]
    my = problem["mesh"]["y"]
    X = np.linspace(mx["start"], mx["end"], mx["points"])
    Y = np.linspace(my["start"], my["end"], my["points"])
    X, Y = np.meshgrid(X, Y)
    return X, Y
