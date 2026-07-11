# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Data loading utilities for sweep results."""

from pathlib import Path

import polars as pl


def load_config(config_path: Path) -> list:
    """Load a Python config file and return its ``experiments`` list."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load config: {config_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "experiments", [])


def load_sweep_data(sweep_config_path: str = "configs/sweep/top_k.py") -> pl.DataFrame:
    """Load all experiment results from a sweep into a single DataFrame.

    Args:
        sweep_config_path: Path to the sweep config YAML.

    Returns:
        DataFrame with columns: name, value, t, type, g, batch_size, tol, output_dim, top_k, seed, eid
    """
    df = _load_from_logs(sweep_config_path)
    if df is not None:
        return df
    else:
        raise FileNotFoundError(
            f"No logs found for config {sweep_config_path}. Please run the sweep experiments first to generate logs."
        )


def _load_from_logs(sweep_config_path: str) -> pl.DataFrame | None:
    """Load sweep data from .logs directory.

    Args:
        sweep_config_path: Path to the sweep config YAML.

    Returns:
        DataFrame if results found, None otherwise.
    """
    config_path = Path(sweep_config_path)
    if not config_path.exists():
        return None

    loaded = load_config(config_path)

    if isinstance(loaded, list):
        configs = loaded
    else:
        raise ValueError(f"Config at {config_path} must export an 'experiments' list")

    dfs = []
    for exp in configs:
        path = Path(f".logs/{exp.id}/metrics.jsonl")

        if not path.exists():
            continue

        _df = pl.read_ndjson(path)

        # Support both dict and dataclass config fields for backward compatibility
        calibrator_config = exp.calibrator_config
        dataset_config = exp.dataset_config

        if isinstance(calibrator_config, dict):
            batch_size = calibrator_config["batch_size"]
            tol = calibrator_config.get("tol", calibrator_config.get("eps"))
            if tol is None:
                raise KeyError("calibrator_config must define 'tol' or 'eps'")
            output_dim = calibrator_config["output_dim"]
            top_k = dataset_config.get("top_k", 1)
            spread = dataset_config.get("spread", 4.0)
        else:
            batch_size = calibrator_config.batch_size
            tol = getattr(calibrator_config, "tol", None)
            if tol is None:
                tol = getattr(calibrator_config, "eps", None)
            if tol is None:
                raise AttributeError("calibrator_config must define 'tol' or 'eps'")
            output_dim = calibrator_config.output_dim
            top_k = getattr(dataset_config, "top_k", 1)
            spread = getattr(dataset_config, "spread", 4.0)

        # Add experiment parameters as columns
        _df = _df.with_columns(
            pl.lit(batch_size).alias("batch_size"),
            pl.lit(tol).alias("tol"),
            pl.lit(output_dim).alias("output_dim"),
            pl.lit(top_k).alias("top_k"),
            pl.lit(spread).alias("spread"),
            pl.lit(exp.seed).alias("seed"),
            pl.lit(exp.id).alias("eid"),
        )
        dfs.append(_df)

    if not dfs:
        return None

    return pl.concat(dfs, how="diagonal")


def compute_utility_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Compute utility summary statistics per experiment.

    Extracts:
    - best_util: best utility across all grid cells (min value for grid type)
    - ew_final_util: final equally-weighted utility (last t value)
    - oracle_util: oracle utility value

    Args:
        df: Raw metrics DataFrame from load_sweep_data.

    Returns:
        Summary DataFrame with one row per experiment.
    """
    util_df = df.filter(pl.col("name") == "util")

    # Get oracle utility per experiment
    oracle = (
        util_df.filter(pl.col("type") == "oracle").group_by("eid").agg(pl.col("value").first().alias("oracle_util"))
    )

    # Get best grid utility per experiment (minimum across all grid cells)
    best_grid = (
        util_df.filter(pl.col("type") == "grid").group_by("eid").agg(pl.col("value").max().alias("best_grid_util"))
    )

    # Get final equally-weighted utility (at max t)
    ew_final = (
        util_df.filter(pl.col("type") == "ew")
        .group_by("eid")
        .agg(pl.col("value").filter(pl.col("t") == pl.col("t").max()).first().alias("ew_final_util"))
    )

    # Get pre-calibration equally-weighted utility (at min t)
    ew_pre = (
        util_df.filter(pl.col("type") == "ew")
        .group_by("eid")
        .agg(pl.col("value").filter(pl.col("t") == pl.col("t").min()).first().alias("ew_pre_util"))
    )

    # Get experiment params (one row per eid)
    params = df.select("eid", "batch_size", "tol", "output_dim", "top_k", "spread", "seed").unique(subset=["eid"])

    # Join all together
    summary = (
        params.join(oracle, on="eid", how="left")
        .join(best_grid, on="eid", how="left")
        .join(ew_final, on="eid", how="left")
        .join(ew_pre, on="eid", how="left")
    )

    # Compute improvement metrics
    summary = summary.with_columns(
        # Best-grid baseline metrics
        (pl.col("ew_final_util") - pl.col("best_grid_util")).alias("util_diff_bestgrid"),
        (pl.col("oracle_util") - pl.col("ew_final_util")).alias("util_oracle_gap_bestgrid"),
        # EW-pre baseline metrics
        (pl.col("ew_final_util") - pl.col("ew_pre_util")).alias("util_diff_self"),
        (pl.col("oracle_util") - pl.col("ew_final_util")).alias("util_oracle_gap_self"),
    )

    return summary


def compute_mse_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Compute MSE summary statistics per experiment.

    Extracts:
    - mse_first: MSE at t=0 (before calibration)
    - mse_last: MSE at max t (after calibration)

    Args:
        df: Raw metrics DataFrame from load_sweep_data.

    Returns:
        Summary DataFrame with one row per experiment.
    """
    mse_df = df.filter(pl.col("name") == "mse")

    # Get first MSE (t=0)
    mse_first = mse_df.group_by("eid").agg(
        pl.col("value").filter(pl.col("t") == pl.col("t").min()).first().alias("mse_first")
    )

    # Get last MSE (max t)
    mse_last = mse_df.group_by("eid").agg(
        pl.col("value").filter(pl.col("t") == pl.col("t").max()).first().alias("mse_last")
    )

    # Get experiment params (one row per eid)
    params = df.select("eid", "batch_size", "tol", "output_dim", "top_k", "spread", "seed").unique(subset=["eid"])

    # Join all together
    summary = params.join(mse_first, on="eid", how="left").join(mse_last, on="eid", how="left")

    # Compute improvement metrics (negative diff means improvement)
    summary = summary.with_columns(
        (pl.col("mse_last") - pl.col("mse_first")).alias("mse_diff"),
        ((pl.col("mse_last") - pl.col("mse_first")) / pl.col("mse_first") * 100).alias("mse_pct_change"),
    )

    return summary


def compute_full_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Compute both utility and MSE summary statistics per experiment.

    Args:
        df: Raw metrics DataFrame from load_sweep_data.

    Returns:
        Summary DataFrame with utility and MSE metrics per experiment.
    """
    util_summary = compute_utility_summary(df)
    mse_summary = compute_mse_summary(df)

    # Join on eid, keeping all columns from utility and mse-specific from mse
    return util_summary.join(
        mse_summary.select("eid", "mse_first", "mse_last", "mse_diff", "mse_pct_change"),
        on="eid",
        how="left",
    )
