from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from dataclasses import asdict

import polars as pl
import torch

import globals as g
from bench.e2e.metrics import gram_errors
from data.synthetic import SyntheticDatasetConfig, make_synthetic_matrix
from data.suitesparse.load import SuiteSparseDatasetConfig, load_suitesparse_matrix
from io_utils import ensure_dir, write_json, write_parquet
from logging_utils import get_logger
from provenance import ensure_clean_repo, get_git_state, get_tree_hashes, make_run_id, run_dir
from sketches.flashblockrow import FlashBlockRowConfig, sketch as flashblockrow
from timing_utils import CudaEventTimer, WallClockTimer
from torch_utils import resolve_device
from kernels.flashblockrow.tune_flashblockrow_config import CONFIG


_LOGGER = get_logger("kernels.flashblockrow.tune_flashblockrow")


def _load_dataset(cfg, device):
    """Load a dataset based on its config type."""
    if isinstance(cfg, SyntheticDatasetConfig):
        return make_synthetic_matrix(cfg, device)
    if isinstance(cfg, SuiteSparseDatasetConfig):
        return load_suitesparse_matrix(cfg, device)
    raise ValueError(f"Unknown dataset config type: {type(cfg)}")


def _dataset_id(metadata):
    """Return a stable dataset id string from metadata."""
    if metadata["dataset"] == g.DATASET_SYNTHETIC:
        return f"{metadata['dataset']}:{metadata['name']}"
    if metadata["dataset"] == g.DATASET_SUITESPARSE:
        return f"{metadata['dataset']}:{metadata['group']}/{metadata['name']}"
    return metadata["dataset"]


def _summarize_times(times):
    """Return timing summary statistics."""
    tensor = torch.tensor(times, device="cpu")
    return {
        "time_mean_ms": float(tensor.mean().item()),
        "time_median_ms": float(tensor.median().item()),
        "time_p95_ms": float(tensor.quantile(0.95).item()),
        "time_min_ms": float(tensor.min().item()),
    }


def _pareto_front(group_df):
    """Return Pareto-optimal rows (time/error minimization) for a group."""
    if group_df.is_empty():
        return pl.DataFrame()

    ordered = group_df.sort("time_median_ms")
    best_err = float("inf")
    keep = []
    for row in ordered.iter_rows(named=True):
        err = float(row["err_median"])
        if err < best_err:
            keep.append(row)
            best_err = err
    return pl.DataFrame(keep)


def _pick_knee(pareto_df):
    """Pick a knee point on a Pareto frontier using min normalized time+error."""
    if pareto_df.is_empty():
        return None
    if pareto_df.height == 1:
        return pareto_df.to_dicts()[0]

    t_min = pareto_df["time_median_ms"].min()
    t_max = pareto_df["time_median_ms"].max()
    e_min = pareto_df["err_median"].min()
    e_max = pareto_df["err_median"].max()

    def _norm(val, lo, hi):
        if hi <= lo:
            return 0.0
        return (val - lo) / (hi - lo)

    best = None
    best_score = None
    for row in pareto_df.iter_rows(named=True):
        score = _norm(row["time_median_ms"], t_min, t_max) + _norm(row["err_median"], e_min, e_max)
        if best_score is None or score < best_score:
            best_score = score
            best = row
    return best


def _select_optimal(df, error_tolerance):
    """Select optimal kappa parameters using Pareto fronts and knee selection."""
    if df.is_empty():
        return {}, pl.DataFrame(), pl.DataFrame(), pl.DataFrame()

    agg = df.group_by(
        ["dataset_id", "dataset", "dataset_group", "dataset_name", "k", "block_size", "kappa", "s", "tc"]
    ).agg(
        [
            pl.col("time_median_ms").median().alias("time_median_ms"),
            pl.col("gram_fro_rel").median().alias("err_median"),
            pl.len().alias("seed_count"),
        ]
    )

    best_err = agg.group_by(["dataset_id", "k"]).agg(
        pl.col("err_median").min().alias("best_err")
    )

    candidates = (
        agg.join(best_err, on=["dataset_id", "k"], how="left")
        .with_columns(
            (pl.col("best_err") * (1.0 + error_tolerance)).alias("err_threshold")
        )
        .filter(pl.col("err_median") <= pl.col("err_threshold"))
    )

    pareto_rows = []
    knee_rows = []
    for dataset_id in candidates.select("dataset_id").unique().to_series():
        subset = candidates.filter(pl.col("dataset_id") == dataset_id)
        for k in subset.select("k").unique().to_series():
            group_df = subset.filter(pl.col("k") == k)
            pareto_df = _pareto_front(group_df)
            if pareto_df.is_empty():
                continue
            pareto_rows.extend(pareto_df.to_dicts())
            knee = _pick_knee(pareto_df)
            if knee is not None:
                knee_rows.append(knee)

    pareto_df = pl.DataFrame(pareto_rows) if pareto_rows else pl.DataFrame()
    knee_df = pl.DataFrame(knee_rows) if knee_rows else pl.DataFrame()

    wins = knee_df.group_by(["block_size", "kappa", "s", "tc"]).agg(
        pl.len().alias("wins")
    ) if knee_df.height else pl.DataFrame()

    overall = pareto_df.group_by(["block_size", "kappa", "s", "tc"]).agg(
        [
            pl.col("time_median_ms").median().alias("overall_time_median_ms"),
            pl.col("err_median").median().alias("overall_err_median"),
        ]
    ) if pareto_df.height else pl.DataFrame()

    ranked = wins.join(overall, on=["block_size", "kappa", "s", "tc"], how="left").sort(
        ["wins", "overall_time_median_ms", "overall_err_median"], descending=[True, False, False]
    ) if wins.height else pl.DataFrame()

    optimal = ranked.to_dicts()[0] if ranked.height > 0 else {}
    return optimal, knee_df, ranked, pareto_df


def main():
    """Tune kappa-SJLT kernel parameters and summarize the best configuration."""
    if CONFIG.require_clean_repo:
        ensure_clean_repo()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for kappa tuning.")

    device = resolve_device()
    run_id = make_run_id()
    run_path = run_dir(run_id)
    ensure_dir(run_path)

    git_state = get_git_state()
    tree_hashes = get_tree_hashes(["kernels", "sketches", "data", "bench"])
    max_shared = torch.cuda.get_device_properties(0).shared_memory_per_block

    rows = []

    for dataset_cfg in CONFIG.datasets:
        A, metadata = _load_dataset(dataset_cfg, device)
        A = A.contiguous() if not A.is_contiguous() else A
        d, n = A.shape
        dataset_id = _dataset_id(metadata)

        for k in CONFIG.k_values:
            for block_size in CONFIG.block_size_values:
                for kappa in CONFIG.kappa_values:
                    for s in CONFIG.s_values:
                        for tc in CONFIG.tc_values:
                            shared_bytes = 2 * block_size * tc * 4 + kappa * 4
                            if shared_bytes > max_shared:
                                continue
                            for seed in CONFIG.seeds:
                                cfg = FlashBlockRowConfig(
                                    k=k,
                                    seed=seed,
                                    dtype=g.DTYPE_FP32,
                                    block_size=block_size,
                                    kappa=kappa,
                                    s=s,
                                    tc=tc,
                                    return_contiguous=True,
                                )

                                for _ in range(CONFIG.warmup):
                                    _ = flashblockrow(A, cfg)

                                cuda_times = []
                                for _ in range(CONFIG.repeats):
                                    with CudaEventTimer(True) as cuda_timer:
                                        _ = flashblockrow(A, cfg)
                                    cuda_times.append(cuda_timer.elapsed_ms)

                                for _ in range(CONFIG.warmup):
                                    _ = flashblockrow(A, cfg)

                                wall_times = []
                                for _ in range(CONFIG.repeats):
                                    with WallClockTimer(True) as wall_timer:
                                        _ = flashblockrow(A, cfg)
                                    wall_times.append(wall_timer.elapsed_ms)

                                SA = flashblockrow(A, cfg)
                                errors = gram_errors(A, SA)

                                row = {
                                    "dataset_id": dataset_id,
                                    "dataset": metadata["dataset"],
                                    "dataset_name": metadata.get("name") or "",
                                    "dataset_group": metadata.get("group") or "",
                                    "d": int(d),
                                    "n": int(n),
                                    "k": int(k),
                                    "seed": int(seed),
                                    "block_size": int(block_size),
                                    "kappa": int(kappa),
                                    "s": int(s),
                                    "tc": int(tc),
                                }
                                row.update(_summarize_times(wall_times))
                                row.update({
                                    "time_mean_cuda_ms": float(torch.tensor(cuda_times).mean().item()),
                                    "time_median_cuda_ms": float(torch.tensor(cuda_times).median().item()),
                                    "time_p95_cuda_ms": float(torch.tensor(cuda_times).quantile(0.95).item()),
                                    "time_min_cuda_ms": float(torch.tensor(cuda_times).min().item()),
                                })
                                row.update(errors)
                                rows.append(row)

    results_path = run_path / "results.parquet"
    summary_path = run_path / "summary.json"
    manifest_path = run_path / "manifest.json"

    write_parquet(results_path, rows)

    df = pl.DataFrame(rows) if rows else pl.DataFrame()
    optimal, best_per_k, ranked, pareto_df = _select_optimal(df, CONFIG.error_tolerance)

    summary = {
        "run_id": run_id,
        "git": git_state,
        "tree_hashes": tree_hashes,
        "config": asdict(CONFIG),
        "error_tolerance": CONFIG.error_tolerance,
        "optimal": optimal,
        "best_per_k": best_per_k.to_dicts() if best_per_k.height else [],
        "ranked": ranked.to_dicts() if ranked.height else [],
        "pareto_frontier": pareto_df.to_dicts() if pareto_df.height else [],
    }
    write_json(summary_path, summary)

    manifest = {
        "run_id": run_id,
        "git": git_state,
        "tree_hashes": tree_hashes,
        "config": asdict(CONFIG),
        "results_path": str(results_path.relative_to(g.REPO_ROOT)),
        "summary_path": str(summary_path.relative_to(g.REPO_ROOT)),
    }
    write_json(manifest_path, manifest)

    _LOGGER.info("Kappa tuning complete. Results: %s", results_path.relative_to(g.REPO_ROOT))
    _LOGGER.info("Summary: %s", summary_path.relative_to(g.REPO_ROOT))


if __name__ == "__main__":
    main()
