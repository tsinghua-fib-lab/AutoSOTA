"""SyNG-D (forest variant): ForestDiffusion over LSM latents.

Trains a ForestDiffusion model on the fitted latents X = [Z, alpha] and
samples n_reps fake latents; each fake (Z, alpha) is then reconstructed
into an adjacency matrix via the LSM link.
"""
from __future__ import annotations

import numpy as np

from syngler.utils.source import reconstruct_adjacency


DEFAULT_XGB_KW = dict(
    max_depth=7,
    n_estimators=100,
    eta=0.3,
    tree_method="hist",
    reg_lambda=0.0,
    reg_alpha=0.0,
    subsample=1.0,
    n_jobs=-1,
)


def build_forest(model_Z, model_alpha, seed=0, n_t=50, duplicate_K=100, xgb_kw=None):
    """Train a ForestDiffusion model on stacked latents X = [Z | alpha].

    Returns: (fitted ForestFlowModel, X stacked)
    """
    from ForestDiffusion import ForestDiffusionModel as ForestFlowModel

    Z = np.asarray(model_Z)
    alpha = np.asarray(model_alpha).reshape(-1, 1)
    X = np.hstack([Z, alpha]).astype(np.float64)
    y_dummy = np.zeros(X.shape[0], dtype=np.int64)
    if xgb_kw is None:
        xgb_kw = DEFAULT_XGB_KW
    model = ForestFlowModel(
        X, label_y=y_dummy, n_t=n_t, duplicate_K=duplicate_K,
        bin_indexes=[], cat_indexes=[], int_indexes=[],
        diffusion_type="vp", seed=int(seed), **xgb_kw,
    )
    return model, X


def diffuse_latents(model_Z, model_alpha, n_reps, seed=0,
                    n_t=50, duplicate_K=100, xgb_kw=None):
    """Train a forest diffusion on (Z, alpha) and sample `n_reps` new pairs.

    Returns: list of (Z_fake, alpha_fake) tuples.
    """
    r = np.asarray(model_Z).shape[1]
    model, X = build_forest(model_Z, model_alpha, seed=seed,
                            n_t=n_t, duplicate_K=duplicate_K, xgb_kw=xgb_kw)
    n = X.shape[0]
    out = []
    for k in range(n_reps):
        Xy_fake = model.generate(batch_size=n)
        x = Xy_fake[:, :-1]
        out.append((x[:, :r], x[:, r:r + 1].flatten()))
    return out


def generate_graphs(model_Z, model_alpha, n_reps, rho=0.0, seed=0, **kw):
    """SyNG-D (forest): diffuse latents and reconstruct adjacency."""
    for k, (Z, alpha) in enumerate(diffuse_latents(model_Z, model_alpha, n_reps, seed=seed, **kw)):
        yield reconstruct_adjacency(Z, alpha, rho=rho, seed=seed + k + 1)
