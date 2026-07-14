"""
Generate data for heatmaps comparing methods across dimension (R) and sparsity (p).

Each experiment function sweeps over R and sparsity p, repeats multiple times,
and saves a JSON grid of ROC AUC values for off-diagonal edges.

Outputs:
  - data/synthetic_mvte/*.json
"""
__date__ = "September 2025"


import argparse
import json
import os
import time
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from sklearn.metrics import roc_auc_score
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fit_mv_artg import fit_multivariate_artg_ssm
from src.multivariate_transfer_entropy import (
    draw_random_process,
    sample_process,
    estimate_mv_te,
)


# ------------------------------ Small utilities ------------------------------

def _ensure_parent_dir(path: str) -> None:
    """Create parent directory for `path` if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _offdiag_mask(R: int) -> np.ndarray:
    """Boolean mask for off-diagonal entries in an RxR matrix."""
    m = np.ones((R, R), dtype=bool)
    np.fill_diagonal(m, False)
    return m


# ------------------------------ Data loaders ---------------------------------

class ArrayRandomWindowLoader:
    """
    Iterator that yields random contiguous windows from an array of phase angles.

    Yields arrays with shape (B, L+1, R, 1):
      - B windows per batch
      - Each window covers L history steps plus 1 current step
      - R channels (regions)
      - singleton frequency axis (F=1)
    """
    def __init__(self, data: np.ndarray, batch_size: int, L: int, seed: int = 0):
        self.data = np.asarray(data)
        self.batch_size = int(batch_size)
        self.L = int(L)
        self.T, self.R = self.data.shape
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def __next__(self):
        starts = self.rng.integers(0, self.T - self.L, size=self.batch_size)
        # shape: (B, L+1, R)
        windows = np.stack([self.data[s : s + self.L + 1] for s in starts], axis=0)
        # add freq axis -> (B, L+1, R, 1)
        return jnp.asarray(windows)[..., None]


# -------------------------- Synthetic stats (features) ------------------------

def lag_statistics_from_array(data: np.ndarray, L: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute mean and covariance of lagged cos/sin features from phase data.

    Parameters
    ----------
    data : (T, R) array of phase angles (radians)
    L    : number of lag steps (history length)

    Returns
    -------
    mean  : (1, R*L*2)
    covar : (1, R*L*2, R*L*2)
    """
    data = jnp.asarray(data)         # (T, R)
    T, R = data.shape
    W = T - L
    wins = jnp.stack([data[t : t + L] for t in range(W)], axis=0)  # (W, L, R)
    cosv, sinv = jnp.cos(wins), jnp.sin(wins)
    z = jnp.stack([cosv, sinv], axis=-1)                 # (W, L, R, 2)
    z = jnp.transpose(z, (0, 2, 1, 3)).reshape(W, R * L * 2)  # (W, R*L*2)
    mean = jnp.mean(z, axis=0)                           # (R*L*2,)
    centered = z - mean[None, :]
    cov = centered.T @ centered / W                      # (R*L*2, R*L*2)
    return mean[None, :], cov[None, :, :]


# ---------------------------- Random edge patterns ----------------------------

def _random_offdiag_idx(key: jax.Array, R: int) -> Tuple[jax.Array, jax.Array]:
    """Sample one off-diagonal index (i, j) uniformly."""
    i = jax.random.randint(key, (), 0, R)
    j = jax.random.randint(key, (), 0, R - 1)
    j = j + (j >= i)  # skip diagonal
    return i, j

def draw_binary_pattern(key: jax.Array, R: int, p: float) -> jax.Array:
    """
    Draw an R x R binary adjacency matrix with diag=0 and ensure both 0/1
    appear off-diagonal (avoid degenerate all-zero/all-one cases).
    """
    key1, key2, key3 = jax.random.split(key, 3)
    bern = jax.random.bernoulli(key1, p=p, shape=(R, R))
    pattern = jnp.where(jnp.eye(R, dtype=bool), 0, bern)

    off_mask = ~jnp.eye(R, dtype=bool)
    off_vals = pattern[off_mask]
    all_zero = jnp.all(off_vals == 0)
    all_one = jnp.all(off_vals == 1)

    i1, j1 = _random_offdiag_idx(key2, R)  # to flip to 1 if all_zero
    i0, j0 = _random_offdiag_idx(key3, R)  # to flip to 0 if all_one

    pattern = lax.cond(all_zero,
                       lambda _: pattern.at[i1, j1].set(1),
                       lambda _: pattern,
                       operand=None)
    pattern = lax.cond(all_one,
                       lambda _: pattern.at[i0, j0].set(0),
                       lambda _: pattern,
                       operand=None)
    return pattern.astype(jnp.bool_)  # explicit bool dtype


# ------------------------------ Estimator wrappers ----------------------------

def evaluate_auc_artg_logprob(R: int, sparsity: float, key: jax.Array) -> float:
    """
    ARTG (log-prob based MV-TE) AUC against ground-truth off-diagonal edges.
    """
    L, F, T = 8, 1, 1000
    batch_size = 64
    train_steps = 4000

    key, k1, k2, k3, kfit, kte = jax.random.split(key, 6)
    pattern = draw_binary_pattern(k1, R, sparsity)
    W_true = draw_random_process(k2, pattern, L)
    samples = sample_process(k3, W_true, T)  # (T, R)

    means, covars = lag_statistics_from_array(samples, L)
    covars = covars + 1e-1 * jnp.eye(R * L * 2)[None]

    loader = ArrayRandomWindowLoader(samples, batch_size=batch_size, L=L, seed=0)
    W_hat, _, _ = fit_multivariate_artg_ssm(kfit, loader, F, R, L, lr=3e-3, num_steps=train_steps)

    eval_loader = ArrayRandomWindowLoader(samples, batch_size=batch_size, L=L, seed=1)
    te = estimate_mv_te(
        kte,
        eval_loader,
        W_hat,
        means,
        covars,
        R,
        L,
        F,
        max_num_batches=50,
        show_progress=False,
        K=4,
    )
    te_matrix = np.asarray(te[0])  # (R, R)

    mask = _offdiag_mask(R)
    labels = np.asarray(pattern)[mask]
    preds = te_matrix[mask]
    return roc_auc_score(labels, preds)



def _granger_multivariate_scores(samples: np.ndarray, L: int) -> np.ndarray:
    """
    Return a matrix of multivariate Granger 'strengths' from a fitted VAR(L).

    For each ordered pair (i, j), use statsmodels' `test_causality` F-test and
    report F across pairs.
    """
    R = samples.shape[1]
    model = VAR(samples)
    res = model.fit(L)
    scores = np.zeros((R, R))
    for i in range(R):
        for j in range(R):
            if i == j:
                continue
            # Does i cause j?
            t_i_j = res.test_causality(caused=j, causing=i, kind="f")
            # Does j cause i?
            t_j_i = res.test_causality(caused=i, causing=j, kind="f")
            scores[i, j] = t_i_j.test_statistic
            scores[j, i] = t_j_i.test_statistic
    return scores


def evaluate_auc_granger_multivariate(R: int, sparsity: float, key: jax.Array) -> float:
    """Multivariate Granger AUC (VAR(L), applied to cos-transformed phases)."""
    L, T = 8, 1000
    key1, key2, key3 = jax.random.split(key, 3)
    pattern = draw_binary_pattern(key1, R, sparsity)
    W_true = draw_random_process(key2, pattern, L)
    samples = np.cos(sample_process(key3, W_true, T))

    scores = _granger_multivariate_scores(samples, L)
    mask = _offdiag_mask(R)
    labels = np.asarray(pattern)[mask]
    preds = scores[mask]
    return roc_auc_score(labels, preds)


# ------------------------------ Experiment grids ------------------------------

def test_artg() -> None:
    """
    Grid for ARTG log-prob MV-TE (primary method).

    Saves:
      mvte_auc_artg_logprob_dim_sparsity_grid.json
    """
    out_json = os.path.join(
        ROOT,
        "data",
        "synthetic_mvte",
        "artg.json",
    )
    print("Output JSONs:", out_json)
    os.makedirs(os.path.split(out_json)[0], exist_ok=True)

    key = jax.random.PRNGKey(0)
    dims = [16, 32, 64, 128, 256, 512]
    sparsities = [0.1, 0.3, 0.5, 0.7, 0.9]
    repeats = 3

    performance: Dict[str, Dict[str, Dict[int, Dict[float, float]]]] = {}
    for k in range(repeats):
        rep_key = f"Repeat {k}"
        performance[rep_key] = {"log_prob": {}}
        for R in dims:
            for p in sparsities:
                key, sub = jax.random.split(key)
                auc = evaluate_auc_artg_logprob(R, p, sub)
                print(f"method=log_prob, R={R}, sparsity={p:.1f}, auc={auc:.3f}")
                performance[rep_key]["log_prob"].setdefault(R, {})[p] = auc
                with open(out_json, "w") as f:
                    json.dump(performance, f, indent=4)


def test_granger_mv() -> None:
    """
    Small grid for multivariate Granger (VAR) — narrower R.

    """
    out_json = os.path.join(
        ROOT,
        "data",
        "synthetic_mvte",
        "granger_mv.json",
    )
    print("Output JSONs:", out_json)
    os.makedirs(os.path.split(out_json)[0], exist_ok=True)
    
    key = jax.random.PRNGKey(17)
    dims = [16, 32, 64, 128, 256, 512] # kept small to avoid heavy VAR fitting cost explosion
    sparsities = [0.1, 0.3, 0.5, 0.7, 0.9]
    repeats = 3

    performance: Dict[str, Dict[str, Dict[int, Dict[float, float]]]] = {}
    for k in range(repeats):
        rep_key = f"Repeat {k}"
        performance[rep_key] = {"granger_mv": {}}
        for R in dims:
            for p in sparsities:
                start = time.time()
                key, sub = jax.random.split(key)
                auc = evaluate_auc_granger_multivariate(R, p, sub)
                elapsed = time.time() - start
                print(f"method=granger_mv, R={R}, sparsity={p:.1f}, auc={auc:.3f}, time={elapsed:.2f}s")
                performance[rep_key]["granger_mv"].setdefault(R, {})[p] = auc
                with open(out_json, "w") as f:
                    json.dump(performance, f, indent=4)


# --- CLI ---------------------------------------------------------------------


def _run_cli():
    parser = argparse.ArgumentParser(
        description="Run MVTE/Granger heatmap experiments (dimension × sparsity AUC grids)."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["artg", "granger_mv"],
        help=(
            "artg: ARTG TE (log-prob) grid \n"
            "granger_mv: multivariate VAR Granger "
        ),
    )
    args = parser.parse_args()

    if args.mode == "artg":
        test_artg()
    elif args.mode == "granger_mv":
        test_granger_mv()
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")



# ------------------------------- Entrypoint -----------------------------------

if __name__ == "__main__":
    _run_cli()
