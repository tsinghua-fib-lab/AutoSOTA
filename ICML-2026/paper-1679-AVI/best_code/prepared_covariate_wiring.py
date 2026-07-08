"""
Prepared changes for CODE-01: Wire covariate-assisted e-values into real mode runner.

This script documents the exact changes needed. Apply manually for clean git history.

Changes needed:
1. real_world/runner.py: Switch from serpant_algorithm to serpant_algorithm_covariate
2. real_world/runner.py: Build covariate_info dict from model metadata
3. real_world/config.py: Add use_covariates field to RealModeConfig
"""

# --- Change 1: config.py - Add use_covariates field ---
# In RealModeConfig dataclass, add:
#   use_covariates: bool = False
#   theta_update_interval: int = 1

# --- Change 2: runner.py - Build covariate_info ---
# Before calling serpant_algorithm, add:
#   if config.use_covariates:
#       # Extract log param count as covariate
#       param_counts = []
#       for client in env.model_clients:
#           # Try to get from config, default to 1B
#           params = getattr(client.config, num_params, None)
#           if params is None:
#               params = 1e9  # default
#           param_counts.append(params)
#       covariate_info = {X: np.array(param_counts)}
#   else:
#       covariate_info = None

# --- Change 3: runner.py - Switch algorithm ---
# Replace:
#   from core import serpant_algorithm
# With conditional:
#   if config.use_covariates:
#       from core import serpant_algorithm_covariate
#       results = serpant_algorithm_covariate(...)
#   else:
#       from core import serpant_algorithm
#       results = serpant_algorithm(...)

# For now, just log that this is prepared
print("Covariate wiring code prepared. Will apply after iter 1 completes.")
