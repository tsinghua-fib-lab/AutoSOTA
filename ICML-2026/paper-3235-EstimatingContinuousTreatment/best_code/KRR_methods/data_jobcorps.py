# KRR_methods/data_jobcorps.py
from __future__ import annotations

import pathlib
from typing import Tuple, Union, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame]

INT_LIKE_DTYPES = [
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
]


# ----------------------------------------------------------------------
# Paths and data loading
# ----------------------------------------------------------------------

def _default_data_dir() -> pathlib.Path:
    """
    Default directory for empirical data.
    We assume `emp_app.csv` lives in the same directory as this file.
    """
    return pathlib.Path(__file__).resolve().parent


def load_jobcorps_data(data_file, semi_syn_file, h_star_file):
    """
    Loads empirical data, precomputed nuisance components, and ground truth.
    
    Args:
        data_file (Path or str): Path to 'emp_app.csv'
        semi_syn_file (Path or str): Path to 'semi-syn data grf.csv'
        h_star_file (Path or str): Path to 'h_star_grf_empapp.csv'
    """
    data_path = pathlib.Path(data_file)
    semi_path = pathlib.Path(semi_syn_file)
    h_star_path = pathlib.Path(h_star_file)

    print(f"Loading empirical data from: {data_path.name}...")
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found.")

    data = pd.read_csv(data_path, index_col=0)

    # 1. Shuffle once (fixed random_state=20) to match original script
    data = data.sample(frac=1, random_state=20)

    # 2. One-Hot Encoding for categorical (int64) columns
    data_processed = pd.concat(
        [
            data.select_dtypes(exclude="int64"),
            pd.get_dummies(
                data.select_dtypes("int64").astype("category"),
                drop_first=True,
                dtype=float,
            ),
        ],
        axis=1,
    )

    X_emp = data_processed.drop(["d", "y"], axis=1)
    T_emp = data_processed["d"]
    # Y_emp = data["y"]  # Not used for semi-synthetic generation

    # 3. Load Precomputed Components (mu_hat, g)
    print(f"Loading semi-synthetic components from: {semi_path.name}...")
    semi_df = pd.read_csv(semi_path, index_col=0)
    mu_hat = semi_df["mu_hat_grf"].to_numpy()
    g_func = semi_df["g_grf"].to_numpy()

    # 4. Load Ground Truth h*(t)
    print(f"Loading ground truth h*(t) from: {h_star_path.name}...")
    h_star_df = pd.read_csv(h_star_path)
    t_grid = h_star_df["t"].to_numpy()
    h_star = h_star_df["h_star"].to_numpy()

    return X_emp, T_emp, mu_hat, g_func, t_grid, h_star

# ----------------------------------------------------------------------
# RF surrogate for f(x, t) and semi-synthetic generator
# ----------------------------------------------------------------------

def gen_semi_y(mu_hat: np.ndarray, g: np.ndarray, rng: np.random.Generator):
    """
    Generates semi-synthetic outcomes: Y_syn = mu_hat + e * g
    where e is random Rademacher noise {-1, 1}.
    """
    n = len(mu_hat)
    e = rng.choice([-1.0, 1.0], size=n)
    return mu_hat + e * g



def compute_h_star_over_grid(
    X: pd.DataFrame,
    fhat_callable,
    grid: np.ndarray,
) -> np.ndarray:
    """
    Approximate h*(t) = E_X[ fhat(X, t) ] over a grid of t values.
    """
    vals = []
    for t in grid:
        vals.append(np.mean(fhat_callable(X, float(t))))
    return np.asarray(vals)



# ----------------------------------------------------------------------
# Preprocessing (X → Xss)
# ----------------------------------------------------------------------

def make_Xss(Xs: pd.DataFrame) -> pd.DataFrame:
    """
    Map augmented covariates Xs into Xss by column-wise min-max scaling
    and L2 row-normalization (mirrors the original preprocessing).
    """
    X_vals = Xs.to_numpy(dtype=float, copy=True)
    col_min = np.nanmin(X_vals, axis=0)
    col_max = np.nanmax(X_vals, axis=0)
    col_range = col_max - col_min
    safe_range = np.where(col_range == 0.0, 1.0, col_range)

    X_minmax = (X_vals - col_min) / safe_range
    X_minmax = np.nan_to_num(X_minmax, nan=0.0, posinf=0.0, neginf=0.0)

    # In the original code you set X_unit = X_minmax (no extra normalization).
    X_unit = X_minmax

    Xss = pd.DataFrame(X_unit, columns=Xs.columns, index=Xs.index)
    return Xss
