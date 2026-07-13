from __future__ import annotations

import functools
import inspect
import json
import logging
import random
import time
from pathlib import Path
from typing import cast

import anndata as ad
import hydra
import numpy as np
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

from cellbridge.cci.pair_extraction import pairs_from_adata
from cellbridge.core.types import Coupling
from cellbridge.data.ann import load_adata
from cellbridge.ot.solvers import (
    get_unbalanced_marginals,
    two_step_unbalanced_fgw_multi,
)
from cellbridge.utils.data import _split_h5ad
from cellbridge.utils.io import get_run_dir, resolve_path

logger = logging.getLogger(__name__)  # Set up a logger for this module
logging.basicConfig(level=logging.INFO)


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="sweep_align_multi_channel",
)
def main(cfg: DictConfig) -> None:
    """Run multi-channel alignment for a sweep of alpha values."""
    t0 = time.perf_counter()
    logger.info("===== Align pipeline start =====")
    logger.info("Config:\n" + OmegaConf.to_yaml(cfg))
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    random.seed(seed)
    logger.info("Global RNG seeds set to %d", seed)
    root = Path(get_original_cwd())
    run_dir = get_run_dir()
    out_art = run_dir / "artifacts"
    out_art.mkdir(parents=True, exist_ok=True)

    mode = cfg.inputs.mode  # "h5ad" or "two_h5ad"
    logger.info(f"Mode: {mode}")

    if mode == "from_h5ad":  # One combined file; split by labels
        h5_path = resolve_path(cfg.inputs.h5ad_path, root)
        logger.info(f"Loading combined h5ad: {h5_path}")
        A = ad.read_h5ad(h5_path)
        logger.info(f"Loaded AnnData shape={A.shape}")
        X_anndata, Y_anndata = _split_h5ad(
            A, cfg.inputs.domain_key, cfg.inputs.start_label, cfg.inputs.end_label
        )
        logger.info(
            "Split shapes X=%s (%s) Y=%s (%s)",
            X_anndata.shape,
            cfg.inputs.start_label,
            Y_anndata.shape,
            cfg.inputs.end_label,
        )

        obsm_key = cfg.rep.get("obsm_key", None)
        if not obsm_key:
            raise ValueError(
                "When inputs.mode='h5ad', set rep.obsm_key to a precomputed embedding "
                "(e.g., X_pca or X_pca_harmony)."
            )
        if obsm_key not in X_anndata.obsm or obsm_key not in Y_anndata.obsm:
            raise KeyError(f"Embedding '{obsm_key}' not found in .obsm for X and/or Y.")

        Zx = X_anndata.obsm[obsm_key]
        Zy = Y_anndata.obsm[obsm_key]
        logger.info(
            "Using existing embedding %s -> Zx=%s Zy=%s", obsm_key, Zx.shape, Zy.shape
        )
    elif mode == "two_h5ad":
        logger.info("Loading separate h5ad files")
        X_anndata = load_adata(str(resolve_path(cfg.inputs.x_path, root)))
        Y_anndata = load_adata(str(resolve_path(cfg.inputs.y_path, root)))
        logger.info("Loaded X=%s Y=%s", X_anndata.shape, Y_anndata.shape)
        obsm_key = cfg.rep.get("obsm_key", None)
        if obsm_key:
            if obsm_key not in X_anndata.obsm or obsm_key not in Y_anndata.obsm:
                raise KeyError(
                    f"Embedding '{obsm_key}' not found in .obsm of both x and y."
                )
            Zx, Zy = X_anndata.obsm[obsm_key], Y_anndata.obsm[obsm_key]
            logger.info(
                "Using existing embedding %s -> Zx=%s Zy=%s",
                obsm_key,
                Zx.shape,
                Zy.shape,
            )
        else:
            logger.info("Instantiating representation encoder")
            enc = instantiate(cfg.rep)
            combo = ad.concat([X_anndata, Y_anndata], join="inner")
            enc.fit(combo)
            Zx = enc.transform(X_anndata)
            Zy = enc.transform(Y_anndata)
            logger.info("Learned embeddings: Zx=%s Zy=%s", Zx.shape, Zy.shape)
    else:
        raise ValueError(
            f"Unknown inputs.mode='{mode}'. Expected 'h5ad' or 'two_h5ad'."
        )

    if cfg.inputs.subsample:
        logger.info("Subsampling to max %d cells", cfg.inputs.subsampling_size)
        rng_x = np.random.Generator(np.random.PCG64(seed))
        rng_y = np.random.Generator(np.random.PCG64(seed))
        logger.info(
            "Created separate RNGs with seeds %d (X) and %d (Y)", seed, seed + 1
        )

        if Zx.shape[0] > cfg.inputs.subsampling_size:
            indices = rng_x.choice(
                Zx.shape[0], size=cfg.inputs.subsampling_size, replace=False
            )
            Zx = Zx[indices]
            X_anndata = X_anndata[indices].copy()
            logger.info("Subsampled X to %s", Zx.shape)
            logger.info(f"First 10 indices: {indices[:10]}")
        if Zy.shape[0] > cfg.inputs.subsampling_size:
            indices = rng_y.choice(
                Zy.shape[0], size=cfg.inputs.subsampling_size, replace=False
            )
            Zy = Zy[indices]
            Y_anndata = Y_anndata[indices].copy()
            logger.info("Subsampled Y to %s", Zy.shape)
            logger.info(f"First 10 indices: {indices[:10]}")

    Zx, Zy = map(np.asarray, (Zx, Zy))

    if cfg.inputs.get("rep_transform", None):
        rep_transform = instantiate(cfg.inputs.rep_transform)
        Zx = rep_transform(Zx)
        Zy = rep_transform(Zy)
        logger.info("Applied representation transform")

    logger.info("Shapes -> Zx=%s Zy=%s", Zx.shape, Zy.shape)

    logger.info("Instantiating cost/scale/solver")
    cost_fn = instantiate(cfg.cost.fn)  # plain function
    pipeline_transform_cost = (
        instantiate(cfg.cost.pipeline_transform)
        if "pipeline_transform" in cfg.cost
        else None
    )
    logger.info(
        "Components -> cost_fn=%s scale_fn=%s solver=%s",
        cfg.cost.fn._target_,
        "None"
        if pipeline_transform_cost is None
        else cfg.cost.pipeline_transform._target_,
        cfg.solver._target_,
    )

    logger.info("Extracting interaction pairs from X")
    pair_mode = cfg.inputs.get("pair_mode", "none")
    pairs = pairs_from_adata(
        X_anndata, mode=pair_mode, random_state=seed
    )  # Assume interaction pairs are the same across snapshots
    logger.info(f"Interaction pairs extracted (mode={pair_mode})")

    clustering_pipeline = instantiate(cfg.cluster)
    logger.info("Clustering X")
    labels_X, S_X, res_X = clustering_pipeline(X_anndata)
    logger.info(
        "X clustering: cells=%s clusters=%s res=%s",
        X_anndata.n_obs,
        S_X.shape[1] if isinstance(S_X, np.ndarray) and S_X.ndim == 2 else "NA",
        res_X,
    )
    logger.info("Clustering Y")
    labels_Y, S_Y, res_Y = clustering_pipeline(Y_anndata)
    logger.info(
        "Y clustering: cells=%s clusters=%s res=%s",
        Y_anndata.n_obs,
        S_Y.shape[1] if isinstance(S_Y, np.ndarray) and S_Y.ndim == 2 else "NA",
        res_Y,
    )

    X_anndata.obs["cluster"] = labels_X
    Y_anndata.obs["cluster"] = labels_Y

    # Keep the soft/hard membership matrices in .obsm for later use
    X_anndata.obsm["S"] = S_X
    Y_anndata.obsm["S"] = S_Y

    logger.info("Instantiating CCI constructor from cfg.cci")
    cci_constructor = instantiate(cfg.cci)
    logger.info("Building CCI matrices via configured constructor")
    CCI_X, _ = cci_constructor(X_anndata, pairs=pairs)
    CCI_Y, _ = cci_constructor(Y_anndata, pairs=pairs)
    logger.info(f"CCI matrices shapes: X={CCI_X.shape}, Y={CCI_Y.shape}")

    np.save(out_art / "labels_X.npy", labels_X)
    np.save(out_art / "labels_Y.npy", labels_Y)

    a = np.ones(Zx.shape[0], dtype=float) / max(1, Zx.shape[0])
    b = np.ones(Zy.shape[0], dtype=float) / max(1, Zy.shape[0])

    cci_transform = instantiate(cfg.cci_transform)
    C1 = cci_transform(CCI_X)
    C2 = cci_transform(CCI_Y)

    np.save(out_art / "CCI_X.npy", C1)
    np.save(out_art / "CCI_Y.npy", C2)

    alphas = cfg.align.get("alphas", [cfg.align.get("alpha", 0.5)])
    logger.info("Sweeping %d alpha values: %s", len(alphas), alphas)
    sweep_index = []
    t_sweep0 = time.perf_counter()

    solver = instantiate(cfg.solver)  # partial callable

    C = cost_fn(Zx, Zy)
    C, D1, D2 = (
        pipeline_transform_cost(C, C1, C2) if pipeline_transform_cost else (C, C1, C2)
    )

    np.save(out_art / "cost_C.npy", C)
    np.save(out_art / "cost_D1.npy", D1)
    np.save(out_art / "cost_D2.npy", D2)

    a = np.ones(Zx.shape[0], dtype=float) / max(1, Zx.shape[0])
    b = np.ones(Zy.shape[0], dtype=float) / max(1, Zy.shape[0])

    logger.info("Finished the cost pipeline transform.")

    # Hydra may wrap solver targets in partials/decorators.
    def _unwrap_callable(fn):
        if isinstance(fn, functools.partial):
            return fn.func
        try:
            return inspect.unwrap(fn)
        except Exception:
            return fn

    if _unwrap_callable(solver) is two_step_unbalanced_fgw_multi:
        logger.info("Using %s solver.", _unwrap_callable(solver).__name__)
        logger.info("First step: unbalanced OT to get new marginals")
        a, b = get_unbalanced_marginals(
            a,
            b,
            C,
            reg=cfg.solver.reg,
            reg_marginals=cfg.solver.reg_marginals,
            numItermax_uot=cfg.solver.numItermax_uot,
            stopThr_uot=cfg.solver.stopThr_uot,
        )
        a /= a.sum()
        b /= b.sum()

        logger.info("New marginals obtained: a=%s b=%s", a, b)
        # Note that the marginals are fixed for all alphas in the sweep.

    G_previous = np.outer(a, b)

    use_warm_start = bool(cfg.align.get("use_previous_initialization", False))

    for alpha in alphas:
        t_align = time.perf_counter()
        logger.info(f"Alpha {alpha:.3f} with {solver}")

        initialization = G_previous if use_warm_start else None
        if initialization is not None:
            row_err = float(np.max(np.abs(initialization.sum(axis=1) - a)))
            col_err = float(np.max(np.abs(initialization.sum(axis=0) - b)))
            # POT enforces strict feasibility of G0 wrt p and q in FGW.
            # If a previous plan drifts numerically, fall back to default init.
            if row_err > 1e-8 or col_err > 1e-8:
                logger.warning(
                    "Warm-start plan violates marginals (row_err=%.3e, col_err=%.3e); "
                    "falling back to default initialization for alpha %.3f",
                    row_err,
                    col_err,
                    alpha,
                )
                initialization = None

        G, artifacts = solver(
            a,
            b,
            C,
            D1,
            D2,
            alpha=alpha,
            initialization=initialization,
        )

        if use_warm_start:
            logger.info("Using current coupling as warm start for next alpha")
            G_previous = G

        logger.info("Number of nonzeros in coupling: %d", np.nonzero(G)[0].size)
        G = Coupling(G)

        dt = time.perf_counter() - t_align
        logger.info(f"Alpha {alpha:.3f}: done in {dt:.2f}s")
        G_arr = cast(np.ndarray, getattr(G, "G", G))
        alpha_dir = out_art / f"alpha_{alpha:.3f}"
        alpha_dir.mkdir(parents=True, exist_ok=True)
        np.save(alpha_dir / "coupling.npy", G_arr)
        (alpha_dir / "artifacts.json").write_text(json.dumps(artifacts))
        (alpha_dir / "summary.txt").write_text(
            f"alpha={alpha}\n"
            f"Zx={Zx.shape} Zy={Zy.shape}\n"
            f"clusters_X={C1.shape[0]} clusters_Y={C2.shape[0]}\n"
        )
        sweep_index.append((alpha, dt, str(alpha_dir)))

    with open(out_art / "sweep_index.tsv", "w") as fh:
        fh.write("alpha\truntime_sec\tdir\n")
        for a_val, runtime, d in sweep_index:
            fh.write(f"{a_val}\t{runtime:.4f}\t{d}\n")
    logger.info("Alpha sweep complete in %.2fs", time.perf_counter() - t_sweep0)

    encoder_name = cfg.rep.get("_target_", cfg.rep.get("obsm_key", "unknown")).split(
        "."
    )[-1]
    (out_art / "summary.txt").write_text(
        (
            "mode={m}\n"
            "Zx={sx} Zy={sy}\n"
            "encoder={enc}\n"
            "embedding_key={emb}\n"
            "alpha_values={alphas}\n"
        ).format(
            m=mode,
            sx=Zx.shape,
            sy=Zy.shape,
            enc=encoder_name,
            emb=cfg.rep.get("obsm_key", "(learned)"),
            alphas=alphas,
        )
    )

    pairs.df.to_csv(out_art / "pairs.csv", index=False)

    logger.info("Artifacts saved (per-alpha coupling, sweep_index.tsv, summary.txt)")
    logger.info(
        "===== Align pipeline complete in %.2fs -> %s =====",
        time.perf_counter() - t0,
        out_art,
    )


if __name__ == "__main__":
    main()
