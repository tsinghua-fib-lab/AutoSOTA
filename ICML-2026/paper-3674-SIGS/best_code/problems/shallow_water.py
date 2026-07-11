"""Shallow Water PDE Problem Definition

This module defines the 2D shallow water equations with a manufactured solution.
The manufactured solution has radially symmetric structure centered at (x_center, y_center).
"""

import numpy as np
import symengine as sm


def create_shallow_water_problem(decay=0.6, amplitude=1.142, sigma=1.500, 
                                  freq=2.600, phase=0.700, 
                                  x_center=2.4, y_center=-2.4):
    """Create shallow water PDE problem with manufactured solution.

    Args:
        decay: Temporal decay rate (default: 0.6)
        amplitude: Gaussian envelope amplitude (default: 1.142)
        sigma: Gaussian width parameter (default: 1.500)
        freq: Wave frequency (default: 2.600)
        phase: Phase velocity (default: 0.700)
        x_center: Center x-coordinate (default: 2.4)
        y_center: Center y-coordinate (default: -2.4)

    Returns:
        dict: Problem specification with PDE operators, forcing terms, and solution
    """
    x, y, t = sm.symbols("x y t")
    g = 9.81  # Gravitational constant

    # Manufactured solution: height field rho and velocities (ux, uy)
    # Form: exp(-decay*t) * (1 + amplitude*exp(-r²/(sigma*(1+t)))) * cos(freq*r - phase*t)
    rho_expr = sm.sympify(
        f"exp(-{decay}*t)*(1+{amplitude}*exp((-1*((x-{x_center})^2+(y-{y_center})^2))/({sigma}*(1+t))))*cos(({freq}*sqrt(((x-{x_center})^2+(y-{y_center})^2)+1e-8)-{phase}*t))"
    )
    
    # Velocity fields: radially symmetric with same center
    shared_amp = amplitude
    ux_expr = sm.sympify(
        f"exp(-{decay}*t)*(1+{shared_amp}*exp((-1*((x-{x_center})^2+(y-{y_center})^2))/({sigma}*(1+t))))"
        f"*cos(({freq}*sqrt(((x-{x_center})^2+(y-{y_center})^2)+1e-8)-{phase}*t))"
        f"*x*{shared_amp}/(sqrt(((x-{x_center})^2+(y-{y_center})^2)+1e-8))"
    )
    uy_expr = sm.sympify(
        f"exp(-{decay}*t)*(1+{shared_amp}*exp((-1*((x-{x_center})^2+(y-{y_center})^2))/({sigma}*(1+t))))"
        f"*cos(({freq}*sqrt(((x-{x_center})^2+(y-{y_center})^2)+1e-8)-{phase}*t))"
        f"*y*{shared_amp}/(sqrt(((x-{x_center})^2+(y-{y_center})^2)+1e-8))"
    )

    # Conservative shallow water equations
    rho = rho_expr
    ux = ux_expr
    uy = uy_expr
    rho_ux = rho * ux
    rho_uy = rho * uy
    g_rho_sq = 0.5 * g * rho**2

    # Forcing terms (PDE residuals of manufactured solution)
    F1 = sm.diff(rho, t) + sm.diff(rho_ux, x) + sm.diff(rho_uy, y)
    F2 = sm.diff(rho_ux, t) + sm.diff(rho_ux * ux + g_rho_sq, x) + sm.diff(rho_ux * uy, y)
    F3 = sm.diff(rho_uy, t) + sm.diff(rho_ux * uy, x) + sm.diff(rho_uy * uy + g_rho_sq, y)

    return {
        "problem_type": "ShallowWater2D",
        "operators": [
            "rho_t + (rho*ux)_x + (rho*uy)_y",
            "(rho*ux)_t + (rho*ux^2 + 0.5*g*rho^2)_x + (rho*ux*uy)_y",
            "(rho*uy)_t + (rho*ux*uy)_x + (rho*uy^2 + 0.5*g*rho^2)_y",
        ],
        "F": [F1, F2, F3],
        "domain": {
            "x": [-10.0, 10.0],
            "y": [-10.0, 10.0],
            "t": [0.0, 5.0],
        },
        "mesh": {
            "x": {"start": -10.0, "end": 10.0, "points": 32},
            "y": {"start": -10.0, "end": 10.0, "points": 32},
            "t": {"start": 0.0, "end": 5.0, "points": 32},
        },
        "initial_conditions": [
            rho.subs(t, 0),
            (rho_ux).subs(t, 0),
            (rho_uy).subs(t, 0),
        ],
        "parameters": {"g": g},
        "solution": {
            "rho": rho,
            "ux": ux,
            "uy": uy,
        },
    }


def generate_meshes(mesh_config):
    """Generate numpy meshgrids from mesh configuration."""
    axes = [
        np.linspace(details["start"], details["end"], details["points"])
        for details in mesh_config.values()
    ]
    return np.meshgrid(*axes, indexing="ij")


def setup_shallow_symbols_meshes(problem):
    """Create symbolic variables and numerical meshes for the problem."""
    dimensions = list(problem["mesh"].keys())
    symbols = {dim: sm.Symbol(dim) for dim in dimensions}
    meshes = generate_meshes(problem["mesh"])
    return symbols, meshes
