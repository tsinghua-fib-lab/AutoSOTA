"""
Shared configuration for the GWAS pipeline.

Reads GA4GH_Demo/gwas_env.env and exposes:
  USE_CUML              - bool: use cuML (GPU) instead of scikit-learn
  VARIANT_SCALING       - float 0.0-1.0: fraction of variants to use at analysis time
  VARIANT_SCALING_SEED  - int: fixed RNG seed for VARIANT_SCALING subsampling
  HIT_P_THRESHOLD       - float: genome-wide significance threshold (default 5e-8)
  DATA_SIM_SCALING      - float 0.0-1.0: fraction of BIM variants retained at data-sim stage
  DATA_SIM_SCALING_SEED - int: fixed RNG seed for DATA_SIM_SCALING subsampling

Helper functions:
  get_linear_regression(**kwargs)   -> LinearRegression instance
  get_logistic_regression(**kwargs) -> LogisticRegression instance
  apply_variant_scaling(variant_df, G)   -> (filtered_variant_df, filtered_G)
  apply_data_sim_scaling(variant_df, G)  -> (filtered_variant_df, filtered_G)
"""

from __future__ import annotations

import pathlib

import numpy as np

# ---------------------------------------------------------------------------
# Load gwas_env.env (same directory as this file)
# ---------------------------------------------------------------------------
_env_path = pathlib.Path(__file__).parent / "gwas_env.env"
_cfg: dict[str, str] = {}
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                _cfg[_k.strip()] = _v.strip()

USE_CUML: bool = _cfg.get("Use_cuML", "false").lower() == "true"
VARIANT_SCALING: float = float(_cfg.get("Variant_Scaling", "1.0"))
VARIANT_SCALING_SEED: int = 42
HIT_P_THRESHOLD: float = float(_cfg.get("Hit_P_Threshold", "5e-8"))
DATA_SIM_SCALING: float = float(_cfg.get("Data_Sim_Scaling", "1.0"))
DATA_SIM_SCALING_SEED: int = 7  # distinct from VARIANT_SCALING_SEED=42


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------


def get_linear_regression(**kwargs):
    """Return a LinearRegression instance (cuML or sklearn based on Use_cuML)."""
    if USE_CUML:
        from cuml.linear_model import LinearRegression
    else:
        from sklearn.linear_model import LinearRegression
    return LinearRegression(**kwargs)


def get_logistic_regression(**kwargs):
    """Return a LogisticRegression instance (cuML or sklearn based on Use_cuML).

    cuML LogisticRegression does not support penalty=None or penalty='none';
    it uses penalty='none' as a string but the preferred way is to omit
    the penalty kwarg or pass C=1e10 (effectively no regularization).
    This function strips incompatible kwargs when USE_CUML is True.
    """
    if USE_CUML:
        from cuml.linear_model import LogisticRegression

        cuml_kwargs = {
            k: v for k, v in kwargs.items() if k not in ("solver",)
        }  # cuML ignores 'solver'
        # cuML does not accept penalty=None; use a large C to approximate no regularization
        if cuml_kwargs.get("penalty") is None:
            cuml_kwargs.pop("penalty", None)
            cuml_kwargs.setdefault("C", 1e10)
        return LogisticRegression(**cuml_kwargs)
    else:
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(**kwargs)


# ---------------------------------------------------------------------------
# Variant subsampling
# ---------------------------------------------------------------------------


def apply_data_sim_scaling(variant_df, G=None):
    """Deterministically subsample variants to DATA_SIM_SCALING fraction (data-sim stage).

    Applied at the earliest pipeline stage — before PGS overlap in simulate_phenotypes.py
    and before site BED writing in data_bundler.py. Uses a different seed from
    apply_variant_scaling so the two can be applied independently and compound:
    e.g. Data_Sim_Scaling=0.7 then Variant_Scaling=0.6 → 42% of original variants in GWAS.

    Parameters
    ----------
    variant_df : pd.DataFrame
        Variant metadata DataFrame (one row per variant).
    G : xarray Dataset, optional
        Genotype array from read_plink1_bin. When provided, a filtered view is
        also returned via G.isel(variant=chosen_idx).

    Returns
    -------
    (filtered_variant_df, filtered_G)
        filtered_G is None when G was not supplied.
    """
    if DATA_SIM_SCALING >= 1.0:
        return variant_df, G

    n_total = len(variant_df)
    n_keep = max(1, round(n_total * DATA_SIM_SCALING))
    rng = np.random.default_rng(DATA_SIM_SCALING_SEED)
    chosen_idx = np.sort(rng.choice(n_total, size=n_keep, replace=False))

    filtered_df = variant_df.iloc[chosen_idx].reset_index(drop=True)
    filtered_G = G.isel(variant=chosen_idx) if G is not None else None
    return filtered_df, filtered_G


def apply_variant_scaling(variant_df, G=None):
    """Deterministically subsample variants to VARIANT_SCALING fraction.

    Parameters
    ----------
    variant_df : pd.DataFrame
        Full variant metadata DataFrame (one row per variant, matching G order).
    G : xarray Dataset, optional
        Genotype array from read_plink1_bin. When provided, a filtered view is
        also returned via G.isel(variant=chosen_idx).

    Returns
    -------
    (filtered_variant_df, filtered_G)
        filtered_G is None when G was not supplied.
    """
    if VARIANT_SCALING >= 1.0:
        return variant_df, G

    n_total = len(variant_df)
    n_keep = max(1, round(n_total * VARIANT_SCALING))
    rng = np.random.default_rng(VARIANT_SCALING_SEED)
    chosen_idx = np.sort(rng.choice(n_total, size=n_keep, replace=False))

    filtered_df = variant_df.iloc[chosen_idx].reset_index(drop=True)
    filtered_G = G.isel(variant=chosen_idx) if G is not None else None
    return filtered_df, filtered_G
