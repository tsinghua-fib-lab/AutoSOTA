"""
Configuration for 1D Kuramoto-Sivashinsky Equation MPC using CasADi + IPOPT
"""

import numpy as np

# ==============================================================================
# Grid and Physical Parameters
# ==============================================================================
N_grid = 128                     # Number of spatial grid points
L_domain = 22.0                  # Spatial domain length
dx = L_domain / N_grid           # Spatial step size
x = np.linspace(0, L_domain, N_grid, endpoint=False)  # Spatial grid

# Time step
dt = 0.05

# ==============================================================================
# Control Source Parameters (8 Gaussian sources)
# ==============================================================================
n_controls = 8
centers = np.linspace(0, L_domain, n_controls, endpoint=False)  # Gaussian centers
sigma = 1.0                             # Gaussian width

# Control bounds
u_min = -50.0
u_max = 50.0

# ==============================================================================
# MPC Parameters
# ==============================================================================
# The KS equation is highly chaotic. 
# MPC needs to foresee chaotic deviations effectively.

# Number of horizon steps in MPC optimization
horizon = 20

# How often to re-solve MPC (every N simulation steps)
# mpc_substeps=1 means solve at every time step
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
