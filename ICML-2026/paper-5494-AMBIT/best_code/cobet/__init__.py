"""
cobet
=====
Adaptive Multiscale Binary Expansion Tests for Independence.

Three test statistics
---------------------
CoBET        – Copula-based Binary Expansion Test (univariate / multivariate)
dCoBET       – multivariate extension with identity or J-weighted matrices
wa-dCoBET    – weight-adaptive dCoBET (10-fold SNR blending, pairwise heatmap)

Baseline comparators
--------------------
HSIC         – Hilbert-Schmidt Independence Criterion  (requires hyppo)
dCov / dCor  – Distance Covariance / Correlation       (requires hyppo)

Quick start
-----------
>>> from cobet import pairwise_heatmap_new_stat
>>> out = pairwise_heatmap_new_stat(n=500, d_coords=10, theta=2, K=4,
...                                  b=0.4, transform_key="logquad")

>>> from cobet import run_plugin_only
>>> results = run_plugin_only(n=500, theta=2, K=4,
...                           b_config={"trigU": [0.05, 0.1]}, d=5)

>>> from cobet import simulate_hsic, simulate_dcorr
>>> hsic_res = simulate_hsic(n=500, theta=2, D=10,
...                           b_config={"linear": [0.1, 0.2]})
"""

# ── Core building blocks ──────────────────────────────────────────────────────
from .cobet import (
    # feature construction
    bits_from_uniform,
    all_nonempty_subsets_indices,
    features_by_u,
    build_AB_features,
    ranks_to_uniforms,
    # weight matrices
    J_numeric_K,
    block_diag,
    kron_block_diag,
    get_weights,
    # test statistic
    compute_full_T,
    plugin_var_tildeT1,
    # simulation runners
    run_plugin_only,
    run_multi_n_and_save,
)

# ── wa-dCoBET ─────────────────────────────────────────────────────────────────
from .wa_dcobet import (
    # pairwise heatmap (main visualization)
    pairwise_heatmap_new_stat,
    # BH-FDR
    bh_fdr_mask,
    # blending helpers (useful for custom pipelines)
    blended_weight_from_10fold,
    Z_for_pair,
    # power simulation
    aggregated_weights_power,
    power_and_selection_one_setting,
    run_full_grid_and_export,
)

# ── Baselines ─────────────────────────────────────────────────────────────────
from .baselines import (
    simulate_hsic,
    run_multi_n_hsic,
    simulate_dcorr,
    run_multi_n_dcorr,
    print_results,
)

# ── DGP utilities ─────────────────────────────────────────────────────────────
from .utils import (
    clayton_copula_sample_nd,
    generate_XY,
    TRANSFORM_MAP_ND,
    transform_trig_uniform_nd,
    transform_expquad_nd,
    transform_linear_nd,
    transform_logquad_nd,
)

__version__ = "0.1.0"
__author__  = "Your Name"
__all__ = [
    # cobet / dCoBET
    "bits_from_uniform",
    "all_nonempty_subsets_indices",
    "features_by_u",
    "build_AB_features",
    "ranks_to_uniforms",
    "J_numeric_K",
    "block_diag",
    "kron_block_diag",
    "get_weights",
    "compute_full_T",
    "plugin_var_tildeT1",
    "run_plugin_only",
    "run_multi_n_and_save",
    # wa-dCoBET
    "pairwise_heatmap_new_stat",
    "bh_fdr_mask",
    "blended_weight_from_10fold",
    "Z_for_pair",
    "aggregated_weights_power",
    "power_and_selection_one_setting",
    "run_full_grid_and_export",
    # baselines
    "simulate_hsic",
    "run_multi_n_hsic",
    "simulate_dcorr",
    "run_multi_n_dcorr",
    "print_results",
    # utils / DGP
    "clayton_copula_sample_nd",
    "generate_XY",
    "TRANSFORM_MAP_ND",
    "transform_trig_uniform_nd",
    "transform_expquad_nd",
    "transform_linear_nd",
    "transform_logquad_nd",
]
