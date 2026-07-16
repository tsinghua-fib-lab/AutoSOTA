"""
Configuration for 2D Kuramoto-Sivashinsky Equation MPC using CasADi + IPOPT
"""

import numpy as np

# ==============================================================================
# Grid and Physical Parameters
# ==============================================================================
N_grid = 128                     # Number of spatial grid points in each dimension
L_domain = 64.0                  # Spatial domain length
dx = L_domain / N_grid           # Spatial step size
x = np.linspace(0, L_domain, N_grid, endpoint=False)  # Spatial grid X
y = np.linspace(0, L_domain, N_grid, endpoint=False)  # Spatial grid Y

# Time step
dt = 0.05

# ==============================================================================
# Control Source Parameters (16 Gaussian sources in 4x4 grid)
# ==============================================================================
# Placing 16 actuators in a 4x4 grid
grid_points = np.linspace(0, L_domain, 4, endpoint=False) + L_domain / 8.0 
centers = []
for cx in grid_points:
    for cy in grid_points:
        centers.append([cx, cy])
centers = np.array(centers)
n_controls = len(centers)
sigma = 1.2                             # Gaussian width

# Control bounds
u_min = -50.0
u_max = 50.0

# ==============================================================================
# MPC Parameters
# ==============================================================================
horizon = 20

# How often to re-solve MPC (every N simulation steps)
mpc_substeps = 1

# Cost weights
Q = 1.0      # State tracking weight
R = 0.01     # Control penalty (smaller = allow more aggressive control)

# Terminal weight (multiplier on Q for terminal state)
terminal_weight = 10.0

# Number of simulation steps for evaluation
T_sim = 400

# ==============================================================================
# Solver Options
# ==============================================================================
ipopt_options = {
    'ipopt.print_level': 0,
    'print_time': 0,
    'ipopt.max_iter': 500,
    'ipopt.tol': 1e-4,
    'ipopt.acceptable_tol': 1e-3,
    'ipopt.acceptable_iter': 5,
    'ipopt.warm_start_init_point': 'yes',
    'ipopt.mu_init': 1e-2,
    'ipopt.mu_strategy': 'adaptive',
}
