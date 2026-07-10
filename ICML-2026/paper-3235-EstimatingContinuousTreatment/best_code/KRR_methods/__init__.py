# KRR_methods/__init__.py
"""
CTE estimation package for the Job Corps semi-synthetic experiments.

Submodules:
    - data_jobcorps   : data generation / preprocessing
    - kernels         : kernel functions (tensor-product etc.)
    - algorithms      : estimators and tuning routines
"""

# We intentionally avoid importing submodules here to prevent circular imports.
# Import directly from submodules, for example:
#
#   from KRR_methods.data_jobcorps import load_jobcorps_data, make_Xss
#   from KRR_methods.algorithms.estimators_ours import run_ours_tensor_kernel
#   from KRR_methods.algorithms.estimators_plugin import run_plugin_loocv_on_original_grid
#   from KRR_methods.algorithms.length_selection import tune_length2d_and_beta_loocv_krr_nystrom
