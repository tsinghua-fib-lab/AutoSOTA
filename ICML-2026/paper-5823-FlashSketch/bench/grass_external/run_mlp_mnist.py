from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from dataclasses import asdict
import logging
import os
import subprocess
import sys
import time

import polars as pl
import torch

import globals as g
from bench.grass_external.config import MLP_MNIST_CONFIG as CONFIG
from io_utils import ensure_dir, write_json, write_parquet
from logging_utils import get_logger
from provenance import ensure_clean_repo, get_git_state, get_tree_hashes, make_run_id, run_dir


_LOGGER = get_logger("bench.grass_external.run_mlp_mnist")
TREE_HASH_PATHS = ("bench", "sketches", "kernels", "data", "external/GraSS")


def _external_repo_dir():
    """Return the external GraSS repo path."""
    path = g.EXTERNAL_GRASS_DIR
    if not path.exists():
        raise FileNotFoundError(f"GraSS repo not found at {path}")
    return path


def _external_git_state(path):
    """Return commit hash and dirty flag for external GraSS repo."""
    commit = None
    dirty = None
    try:
        commit = (
            subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(path), "status", "--porcelain"]
            ).strip()
        )
    except subprocess.SubprocessError:
        pass
    return {"commit": commit, "dirty": dirty}


def _attach_file_logger(log_path):
    """Attach a file handler to the module logger."""
    for handler in _LOGGER.handlers:
        if isinstance(handler, logging.FileHandler):
            return
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    _LOGGER.addHandler(file_handler)


def _apply_env_overrides(env, proj_type, proj_dim, seed):
    """Return env with GraSS overrides applied."""
    env = env.copy()
    env[g.ENV_TQDM_DISABLE] = "1"
    env[g.ENV_FLASH_SKETCH_ROOT] = str(g.REPO_ROOT)
    env[g.ENV_GRASS_DEVICE] = CONFIG.device
    env[g.ENV_GRASS_PROJ_TYPE] = proj_type
    env[g.ENV_GRASS_PROJ_DIM] = str(proj_dim)
    env[g.ENV_GRASS_SEED] = str(seed)
    env[g.ENV_GRASS_VAL_RATIO] = str(CONFIG.val_ratio)
    env[g.ENV_GRASS_BATCH_SIZE] = str(CONFIG.batch_size)
    env[g.ENV_GRASS_PROJ_MAX_BATCH_SIZE] = str(CONFIG.proj_max_batch_size)
    env[g.ENV_GRASS_SJLT_C] = str(CONFIG.sjlt_c)
    env[g.ENV_GRASS_MLP_ACTIVATION] = str(CONFIG.mlp_activation)
    env[g.ENV_GRASS_MLP_DROPOUT_RATE] = str(CONFIG.mlp_dropout_rate)
    if getattr(CONFIG, "sparsity_log_batches", 0) > 0:
        env["GRASS_SPARSITY_LOG"] = "1"
        env["GRASS_SPARSITY_MAX_BATCHES"] = str(CONFIG.sparsity_log_batches)
    if proj_type in (
        "flashsketch",
        "flashsketch_grass",
    ):
        env[g.ENV_GRASS_FLASH_KAPPA] = str(CONFIG.flashsketch_kappa)
        env[g.ENV_GRASS_FLASH_S] = str(CONFIG.flashsketch_s)
        env[g.ENV_GRASS_FLASH_BLOCK_ROWS] = str(CONFIG.flashsketch_block_rows)
        env[g.ENV_GRASS_FLASH_SKIP_ZEROS] = (
            "1" if CONFIG.flashsketch_skip_zeros else "0"
        )
        if CONFIG.flashsketch_seed is not None:
            env[g.ENV_GRASS_FLASH_SEED] = str(CONFIG.flashsketch_seed)
    return env


def _run_score(score_dir, proj_type, proj_dim, seed):
    """Run GraSS score.py for the given projection type."""
    env = _apply_env_overrides(os.environ, proj_type, proj_dim, seed)
    result_path = score_dir / "results" / f"{proj_type}-{proj_dim}.pt"
    if result_path.exists():
        result_path.unlink()
    start = time.perf_counter()
    subprocess.run([sys.executable, "score.py"], cwd=score_dir, env=env, check=True)
    elapsed = time.perf_counter() - start
    if not result_path.exists():
        raise FileNotFoundError(f"Missing GraSS result file at {result_path}")
    result = torch.load(result_path, map_location="cpu", weights_only=False)
    return elapsed, result


def _row_from_result(proj_type, proj_dim, elapsed, result, seed):
    """Build a summary row from a GraSS result."""
    proj_time_ms = result.get("proj_time_ms")
    proj_time_s = result.get("proj_time_s")
    if proj_time_ms is None and proj_time_s is not None:
        proj_time_ms = float(proj_time_s) * 1000.0
    proj_time_mean_ms = result.get("proj_time_mean_ms")
    proj_time_std_ms = result.get("proj_time_std_ms")
    proj_only_time_ms = result.get("proj_only_time_ms")
    proj_only_time_s = result.get("proj_only_time_s")
    if proj_only_time_ms is None and proj_only_time_s is not None:
        proj_only_time_ms = float(proj_only_time_s) * 1000.0
    proj_only_time_mean_ms = result.get("proj_only_time_mean_ms")
    proj_only_time_std_ms = result.get("proj_only_time_std_ms")
    proj_build_time_ms = result.get("proj_build_time_ms")
    proj_build_time_s = result.get("proj_build_time_s")
    if proj_build_time_ms is None and proj_build_time_s is not None:
        proj_build_time_ms = float(proj_build_time_s) * 1000.0
    proj_build_time_mean_ms = result.get("proj_build_time_mean_ms")
    proj_build_time_std_ms = result.get("proj_build_time_std_ms")

    row = {
        "task": g.TASK_GRASS_LDS,
        "dataset": g.DATASET_MNIST,
        "dataset_name": g.DATASET_MNIST,
        "dataset_group": None,
        "model": g.MODEL_MLP,
        "proj_type": proj_type,
        "proj_dim": proj_dim,
        "k": proj_dim,
        "seed": seed,
        "val_ratio": CONFIG.val_ratio,
        "wall_time_s": float(elapsed),
        "lds": float(result.get("lds")) if result.get("lds") is not None else None,
        "lds_mean": (
            float(result.get("lds_mean")) if result.get("lds_mean") is not None else None
        ),
        "lds_std": (
            float(result.get("lds_std")) if result.get("lds_std") is not None else None
        ),
        "proj_time_ms": float(proj_time_ms) if proj_time_ms is not None else None,
        "proj_time_mean_ms": (
            float(proj_time_mean_ms) if proj_time_mean_ms is not None else None
        ),
        "proj_time_std_ms": (
            float(proj_time_std_ms) if proj_time_std_ms is not None else None
        ),
        "proj_only_time_ms": (
            float(proj_only_time_ms) if proj_only_time_ms is not None else None
        ),
        "proj_only_time_mean_ms": (
            float(proj_only_time_mean_ms)
            if proj_only_time_mean_ms is not None
            else None
        ),
        "proj_only_time_std_ms": (
            float(proj_only_time_std_ms) if proj_only_time_std_ms is not None else None
        ),
        "proj_build_time_ms": (
            float(proj_build_time_ms) if proj_build_time_ms is not None else None
        ),
        "proj_build_time_mean_ms": (
            float(proj_build_time_mean_ms)
            if proj_build_time_mean_ms is not None
            else None
        ),
        "proj_build_time_std_ms": (
            float(proj_build_time_std_ms) if proj_build_time_std_ms is not None else None
        ),
        "method": g.METHOD_GRASS,
        "kappa": None,
        "s": None,
        "block_rows": None,
        "sjlt_c": None,
        "mlp_activation": CONFIG.mlp_activation,
        "mlp_dropout_rate": CONFIG.mlp_dropout_rate,
    }
    if proj_type == "flashsketch":
        row["method"] = g.METHOD_FLASH_SKETCH
        row["kappa"] = CONFIG.flashsketch_kappa
        row["s"] = CONFIG.flashsketch_s
        row["block_rows"] = CONFIG.flashsketch_block_rows
    if proj_type == "flashsketch_grass":
        row["method"] = g.METHOD_FLASH_SKETCH
        row["kappa"] = CONFIG.flashsketch_kappa
        row["s"] = CONFIG.flashsketch_s
        row["block_rows"] = CONFIG.flashsketch_block_rows
    if proj_type in ("sjlt_cusparse", "sjlt_cusparse_grass"):
        row["method"] = g.METHOD_SJLT_CUSPARSE
        row["sjlt_c"] = CONFIG.sjlt_c
    if proj_type in ("gaussian_dense_cublas", "gaussian_dense_cublas_grass"):
        row["method"] = g.METHOD_GAUSSIAN_DENSE_CUBLAS
    if proj_type in ("srht_fwht", "srht_fwht_grass"):
        row["method"] = g.METHOD_SRHT_FWHT
    if proj_type == "sjlt_kernel":
        row["method"] = g.METHOD_SJLT_GRASS_KERNEL
        row["sjlt_c"] = CONFIG.sjlt_c
    if proj_type == "sjlt_kernel_grass":
        row["method"] = g.METHOD_SJLT_GRASS_KERNEL
        row["sjlt_c"] = CONFIG.sjlt_c
    if proj_type.startswith("grass"):
        row["method"] = g.METHOD_GRASS
        row["sjlt_c"] = CONFIG.sjlt_c
    if proj_type == "sjlt":
        row["sjlt_c"] = CONFIG.sjlt_c
    return row


def main():
    """Run the external GraSS MLP+MNIST experiment for SJLT vs FlashSketch."""
    if CONFIG.require_clean_repo:
        ensure_clean_repo()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GraSS external runs.")

    grass_repo = _external_repo_dir()
    score_dir = grass_repo / "MLP_MNIST"

    run_id = make_run_id()
    run_root = run_dir(run_id)
    run_path = run_root / "grass_external"
    ensure_dir(run_path)
    _attach_file_logger(run_path / "run.log")

    git_state = get_git_state()
    tree_hashes = get_tree_hashes(TREE_HASH_PATHS)
    grass_state = _external_git_state(grass_repo)

    rows = []
    proj_dims = CONFIG.proj_dims if CONFIG.proj_dims else (CONFIG.proj_dim,)
    repeat_seeds = CONFIG.repeat_seeds or (CONFIG.seed,)
    for seed in repeat_seeds:
        for proj_dim in proj_dims:
            for proj_type in CONFIG.proj_types:
                _LOGGER.info(
                    "Running GraSS MLP_MNIST proj_type=%s proj_dim=%s seed=%s",
                    proj_type,
                    proj_dim,
                    seed,
                )
                elapsed, result = _run_score(score_dir, proj_type, proj_dim, seed)
                rows.append(_row_from_result(proj_type, proj_dim, elapsed, result, seed))

    results_path = run_path / "grass_compare.parquet"
    e2e_results_path = run_root / "results.parquet"
    csv_path = run_path / "grass_compare.csv"
    summary_path = run_path / "summary.json"
    manifest_path = run_root / "manifest.json"

    df = pl.DataFrame(rows)
    write_parquet(results_path, rows)
    write_parquet(e2e_results_path, rows)
    df.write_csv(csv_path)
    write_json(summary_path, {"rows": len(rows)})
    write_json(
        manifest_path,
        {
            "run_id": run_id,
            "git": git_state,
            "tree_hashes": tree_hashes,
            "config": asdict(CONFIG),
            "results_path": str(results_path.relative_to(g.REPO_ROOT)),
            "e2e_results_path": str(e2e_results_path.relative_to(g.REPO_ROOT)),
            "csv_path": str(csv_path.relative_to(g.REPO_ROOT)),
            "summary_path": str(summary_path.relative_to(g.REPO_ROOT)),
            "external_grass": {
                "path": str(grass_repo.relative_to(g.REPO_ROOT)),
                "commit": grass_state.get("commit"),
                "dirty": grass_state.get("dirty"),
            },
        },
    )

if __name__ == "__main__":
    main()
