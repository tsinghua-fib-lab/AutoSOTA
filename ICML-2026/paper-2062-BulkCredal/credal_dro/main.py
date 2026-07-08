"""Entrypoint app functions"""

from datetime import datetime
import math
from pathlib import Path
from typing import Optional
from uuid import uuid4, UUID
from joblib import Parallel, delayed
import cvxpy as cp
import numpy as np
import pandas as pd
import scipy as sp
import typer
import warnings
import matplotlib.pyplot as plt
import numpy.linalg as la
import os
import re
import json
import torch
import fcntl

from bayesian_dro.Bayesian_DRO_continuous import main_Bayesian_DRO
from .bayes_conjugates import (
    sample_posterior,
    default_prior_params,
    get_log_partition_constant,
    get_posterior_params,
    derive_analytical_posterior_params,
    posterior_predictive_params,
    sample_posterior_predictive,
)
from .constants import (
    CONTAMINATION_LEVEL,
    IN_SAMPLE_TIME_WINDOW,
    NPL_ETA,
    NUM_LIKELIHOOD_SAMPLES,
    NUM_OBSERVATIONS,
    NUM_POSTERIOR_SAMPLES,
    NUM_REPLICATIONS,
    NUM_TEST_OBSERVATIONS,
    NUM_CERTIFY,
    MAX_PARAMS_OOM,
    ROBAS_NEWSVENDOR_NUM_REPLICATIONS,
    DO_THRESHOLD_CALIBRATION,
    upper_triangular_size,
)
from .dataset import sample_dgp, portfolio_dataset, get_num_time_windows, get_portfolio_returns_df
from .experiments import ExperimentName, get_experiment
from .likelihood import sample_likelihood, reconstruct_covariance_from_triu
from .newsvendor import newsvendor_cost_cvxpy, get_lv_newsvendor_problem, make_newsvendor_a_mat
from .or_wdro import (
    get_or_wdro_newsvendor_problem,
    cheap_robust_mean_estimate,
    robust_sigma_sq_estimate,
)

from .npl import sample_npl
from .optimise import get_kl_bdro_problem, DRO_BAS_MMD
from .california_housing_lp import get_lv_bas_ch_problem, get_erm_lad_problem, get_cvar_lad_problem, get_max_lad_problem, get_erm_ridge_problem, get_wass_lad_problem
from .portfolio import (
    get_kl_portfolio_problem,
    portfolio_objective_cvxpy,
    bdro_portfolio_posterior_samples,
    calibrate_lv_bulk_set,
    make_lv_portfolio_f_spec,
    make_lv_pc_sampler_from_niw,
    solve_lv_portfolio_socp,
    get_lv_portfolio_problem,
)
from .lv_bulk_set import build_score, dkw_select_threshold
from .lv_dro import make_bulk_set_spec, truncated_mean
from .preprocessing import normalise_by_dimension
from .gaussian_kernel import *
from .summarise_and_plot import *
from . import constants as _constants
app = typer.Typer(name="credaldro")
_UUID_CSV_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\.csv$"
)

@app.command(name="setup-kl")
def setup_kl_dro_bas(
    experiment_name: ExperimentName, experiment_dir: Path, batch_size: int, dataset_dir: Path = None, overwrite: bool = False
):
    """Setup an experiment in a new directory"""
    if not experiment_dir.exists() or not overwrite:
        experiment_dir.mkdir(parents=False, exist_ok=False)

    # get the experiment from the name
    experiment = get_experiment(experiment_name, dataset_dir=dataset_dir)

    # write experiment file to JSON
    filepath = experiment_dir / "experiment.json"
    with open(filepath, "w", encoding="utf-8") as json_file:
        json.dump(experiment, json_file, indent=4)

    # for the given batch size, how many batches do we need?
    num_batches = math.ceil(float(len(experiment)) / float(batch_size))-1 # -1 because SLURM array starts at 0

    # setup SLURM file
    with open(
        Path(__file__).parent / "newsvendor_calhousing_template.slurm", "r", encoding="utf-8"
    ) as slurm_file:
        slurm_string = slurm_file.read()
    dgp_string = slurm_string.format(
        experiment_dir=experiment_dir, num_batches=num_batches, batch_size=batch_size
    )
    (experiment_dir / f"{experiment_name}.slurm").write_text(dgp_string)

@app.command(name="setup-lv")
def setup_lv_dro_bas(
    experiment_name: ExperimentName,
    experiment_dir: Path,
    num_batches: int = 0, # 1 batch by default
    dataset_dir: Path = None,
    overwrite: bool = False,
    experiment_data_dir: Optional[Path] = None,
) -> None:
    """Setup an LV-BAS experiment in a new directory (mirrors setup-kl).

    This writes an `experiment.json` file for the given experiment name
    and a SLURM array script which calls `credal batch` with the chosen
    batch size, reusing the `newsvendor_calhousing_template.slurm` template.
    """
    # Follow the same overwrite semantics as setup_kl_dro_bas
    if not experiment_dir.exists() or not overwrite:
        experiment_dir.mkdir(parents=False, exist_ok=False)

    # Get the experiment configuration
    experiment = get_experiment(experiment_name, dataset_dir=dataset_dir)

    # Write experiment file to JSON
    filepath = experiment_dir / "experiment.json"
    with open(filepath, "w", encoding="utf-8") as json_file:
        json.dump(experiment, json_file, indent=4)

    batch_size = math.ceil(float(len(experiment)) / float(num_batches + 1))
    # Setup SLURM file (reuse the KL-BAS template)
    with open(
        Path(__file__).parent / "newsvendor_calhousing_template.slurm",
        "r",
        encoding="utf-8",
    ) as slurm_file:
        slurm_string = slurm_file.read()

    slurm_string = slurm_string.format(
        experiment_dir=experiment_dir,
        num_batches=num_batches,
        batch_size=batch_size,
        experiment_data_dir=(experiment_data_dir or experiment_dir)
    )
    (experiment_dir / f"{experiment_name}.slurm").write_text(slurm_string)


@app.command(name="setup-mmd")
def setup_mmd_dro_bas(
    experiment_name: ExperimentName, experiment_dir: Path, npl_samples_dir: Path, batch_size: int, dataset_dir: Path = None, overwrite: bool = False, njobs: int = -1,
):
    """Setup an experiment in a new directory"""
    if not experiment_dir.exists() or not overwrite:
        experiment_dir.mkdir(parents=False, exist_ok=False)

    # get the experiment from the name
    print("Creating experiment...")
    experiment = get_experiment(experiment_name, dataset_dir=dataset_dir)

    # write experiment file to JSON
    print("Writing experiment to JSON...")
    filepath = experiment_dir / "experiment.json"
    with open(filepath, "w", encoding="utf-8") as json_file:
        json.dump(experiment, json_file, indent=4)

    # for the given batch size, how many batches do we need?
    num_batches = math.ceil(float(len(experiment)) / float(batch_size)) -1  # -1 because SLURM array starts at 0

    # setup SLURM file
    print("Setting up SLURM files for optimization...")
    with open(
        Path(__file__).parent / "mmd_dro_bas_template.slurm", "r", encoding="utf-8"
    ) as slurm_file:
        slurm_string = slurm_file.read()
    slurm_string = slurm_string.format(
        experiment_dir=experiment_dir, njobs=njobs, num_batches=num_batches, batch_size=batch_size
    )
    slurm_string += f" --npl-samples-dir {npl_samples_dir}"
    (experiment_dir / f"{experiment_name}.slurm").write_text(slurm_string)

    # create an ID for each unique posterior setting and save the IDs to a CSV
    if npl_samples_dir.exists():
        print("Using NPL samples from", npl_samples_dir)
    else:
        print("Setting up SLURM files ready for sampling the NPL on GPUs")
        npl_samples_dir.mkdir(parents=False)
        experiment_df = pd.DataFrame(experiment)
        gb = experiment_df.groupby(POSTERIOR_GB_COLS)
        posterior_settings = []
        for group, _ in gb:
            npl_row = dict(zip(POSTERIOR_GB_COLS, group))
            npl_row["npl_uuid"] = str(uuid4())
            posterior_settings.append(npl_row)
        posterior_settings_df = pd.DataFrame(posterior_settings)
        posterior_settings_df = posterior_settings_df.loc[posterior_settings_df["inference"].isin(["npl_mmd", "npl_wlb"])]
        posterior_settings_df.to_csv(npl_samples_dir / "npl_settings.csv", index=False)

        # then create SLURM file ready to sample the NPL on the cluster
        num_npl_batches = len(posterior_settings_df) - 1
        with open(
            Path(__file__).parent / "sample_npl.slurm", "r", encoding="utf-8"
        ) as npl_slurm_file:
            npl_slurm_string = npl_slurm_file.read()
        npl_slurm_string = npl_slurm_string.format(num_npl_batches=num_npl_batches, npl_samples_dir=npl_samples_dir, dataset_dir=dataset_dir)
        (npl_samples_dir / f"sample_npl_{experiment_name}.slurm").write_text(npl_slurm_string)

# =========================
# Real-world experiment CLI
# =========================

@app.command(name="setup_real_world")
def setup_real_world(
    experiment_name: ExperimentName,
    experiment_dir: Path,
    overwrite: bool = False,
    num_batches: int = 0, # 1 batch by default
    experiment_data_dir: Optional[Path] = None,
    dataset_dir: Optional[Path] = None,
    rw_dataset_dir: Optional[Path] = None,
) -> None:
    """
    Setup a real-world (PyTorch) experiment directory:
      experiments.py -> experiment.json -> real_world_experiment.slurm

    - experiment_name: an ExperimentName entry that returns real-world rows
    - experiment_dir: where experiment.json + slurm are written
    - experiment_data_dir: where uuid.csv files are written (defaults to experiment_dir)
    - rw_dataset_dir: root dir for WILDS caches & downloads (defaults from env)
    """
    if not experiment_dir.exists() or not overwrite:
        experiment_dir.mkdir(parents=False, exist_ok=False)

    # get experiment settings from experiments.py
    experiment = get_experiment(experiment_name, dataset_dir=dataset_dir)

    # write experiment.json
    filepath = experiment_dir / "experiment.json"
    with open(filepath, "w", encoding="utf-8") as json_file:
        json.dump(experiment, json_file, indent=4)

    batch_size = math.ceil(float(len(experiment)) / float(num_batches + 1))

    # Resolve real-world dataset root
    if rw_dataset_dir is None:
        rw_dataset_dir = Path(
            os.environ.get(
                "CIVILCOMMENTS_DATASET_DIR"
            )
        )

    # read civilcomments SLURM template
    template_path = Path(__file__).parent / "civilcomments_template.slurm"
    with open(template_path, "r", encoding="utf-8") as slurm_file:
        slurm_string = slurm_file.read()

    slurm_string = slurm_string.format(
        experiment_dir=experiment_dir,
        num_batches=num_batches,
        batch_size=batch_size,
        experiment_data_dir=(experiment_data_dir or experiment_dir),
        rw_dataset_dir=rw_dataset_dir,
    )

    # write a fixed-name slurm file (as requested)
    (experiment_dir / "rw_civilcomments.slurm").write_text(slurm_string)


@app.command(name="batch_real_world")
def batch_real_world(
    experiment_dir: Path,
    batch_id: int,
    batch_size: int,
    only_missing: bool = False,
    experiment_data_dir: Optional[Path] = None,
    rw_dataset_dir: Optional[Path] = None,
) -> None:
    """
    Run a batch slice of experiment.json for real-world experiments.
    Writes one {uuid}.csv per experiment row (like synthetics).
    """
    print(datetime.now(), "Running REAL-WORLD batch from array index", batch_id)
    print()

    # Resolve real-world dataset root
    if rw_dataset_dir is None:
        rw_dataset_dir = Path(
            os.environ.get(
                "CIVILCOMMENTS_DATASET_DIR"
            )
        )

    filepath = experiment_dir / "experiment.json"
    with open(filepath, "r", encoding="utf-8") as json_file:
        experiment = json.load(json_file)

    start = batch_id * batch_size
    batch_experiment = experiment[start : min(start + batch_size, len(experiment))]

    out_dir = experiment_data_dir or experiment_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for params in batch_experiment:
        uuid = params.get("uuid")
        if uuid is None:
            raise ValueError("Each experiment.json row must include a 'uuid' key for real-world runs.")

        csv_path = out_dir / f"{uuid}.csv"
        if only_missing and csv_path.exists():
            continue

        run_real_world(
            experiment_dir=experiment_dir,
            experiment_data_dir=experiment_data_dir,
            rw_dataset_dir=rw_dataset_dir,
            **params,
        )


@app.command(name="run_real_world")
def run_real_world(
    experiment_dir: Path,

    # --- keys typically coming from experiment.json ---
    algorithm: str,
    dataset: str,
    epsilon: float,
    gamma: float,
    uuid: str = str(uuid4()),
    num_replications: int = NUM_REPLICATIONS,

    # keep these for compatibility with existing rw_* experiments
    dgp: str = "S1",
    contamination: float = 0.0,
    dim: int = 10,
    num_observations: int = NUM_OBSERVATIONS,
    num_test_observations: int = NUM_TEST_OBSERVATIONS,
    ignore_dpp: bool = True,
    njobs: int = 1,
    normalise: bool = False,
    verbose: bool = False,

    # --- output location ---
    experiment_data_dir: Optional[Path] = None,

    # --- real-world dataset root (WILDS downloads + caches) ---
    rw_dataset_dir: Optional[Path] = None,

    # --- shared real-world conmsdls (passed to run_real_world_replication) ---
    rw_cal_fraction: float = 0.2,
    rw_split_seed: int = 0,
    rw_cache_dir: Optional[Path] = None,
    rw_device: str = "auto",

    rw_head: str = "linear",
    rw_mlp_hidden: int = 256,

    rw_train_batch_size: int = 256,
    rw_train_num_workers: int = 0,
    rw_epochs: int = 20,
    rw_lr: float = 1e-3,
    rw_weight_decay: float = 1e-4,
    rw_smoothmax_temperature: float = 0.1,
    rw_groupdro_step_size: float = 0.01,
    rw_max_grad_norm: Optional[float] = None,

    rw_text_n_features: int = 2**12,
    rw_text_ngram_min: int = 1,
    rw_text_ngram_max: int = 1,

    require_cache: bool = True,
    do_threshold_calibration: bool = False,
) -> None:
    """
    Run ONE real-world experiment.json row (one uuid), producing uuid.csv with all replications.
    This mirrors the synthetic `run` workflow: one file per uuid + progress tracking.
    """
    from .wilds.runner import run_real_world_replication

    if rw_dataset_dir is None:
        rw_dataset_dir = Path(
            os.environ.get(
                "CIVILCOMMENTS_DATASET_DIR"
            )
        )
    
    if rw_cache_dir is None:
        env_cache = os.environ.get("CIVILCOMMENTS_CACHE_DIR", None)
        if env_cache:
            rw_cache_dir = Path(env_cache)


    print(datetime.now(), "- REAL-WORLD run:", "dataset=", dataset, "algorithm=", algorithm, "uuid=", uuid)

   
    if require_cache:
        # Effective cache root must match runner.py semantics
        cache_root_eff = (Path(rw_cache_dir) if rw_cache_dir is not None else (Path(rw_dataset_dir) / "cache" / "wilds"))
        cache_root_eff.mkdir(parents=True, exist_ok=True)

        # Make the lock key specific enough to avoid accidental collisions
        cal_pct = int(round(100 * float(rw_cal_fraction)))
        dataset_key = str(dataset).lower()
        if dataset_key in ("rw_civilcomments", "civilcomments"):
            # Lock must match the *text* cache identity (embedder is irrelevant for CivilComments).
            lock_name = (
                f".prepare_cache_lock__civilcomments__hash{int(rw_text_n_features)}__ng{int(rw_text_ngram_min)}-{int(rw_text_ngram_max)}"
                f"__split{int(rw_split_seed)}__cal{cal_pct}"
            )
    
        lock_path = cache_root_eff / lock_name
        done_path = lock_path.with_suffix(".done")

        with lock_path.open("w", encoding="utf-8") as _lf:
            fcntl.flock(_lf.fileno(), fcntl.LOCK_EX)
            try:
                # If this exact cache-key has already been prepared, skip entirely.
                if not done_path.exists():
                    prepare_realworld_cache(
                        dataset_dir=Path(rw_dataset_dir),
                        dataset=str(dataset),

                        rw_cache_dir=rw_cache_dir,
                        rw_device=str(rw_device),
                        rw_split_seed=int(rw_split_seed),
                        rw_cal_fraction=float(rw_cal_fraction),

                        # text (use the same batch/workers you use in the actual runs)
                        rw_text_n_features=int(rw_text_n_features),
                        rw_text_ngram_min=int(rw_text_ngram_min),
                        rw_text_ngram_max=int(rw_text_ngram_max),
                        rw_text_batch_size=int(rw_train_batch_size),
                        rw_text_num_workers=int(rw_train_num_workers),

                        # Optional cache knobs
                        prepare_dense_text_features=False,
                    )

                    # Mark as complete *only after* successful preparation.
                    done_path.write_text("ok\n", encoding="utf-8")
            finally:
                fcntl.flock(_lf.fileno(), fcntl.LOCK_UN)


    # Collect replication rows (IMPORTANT: do not include input-parameter columns to avoid overlap with experiment_df join)
    DROP_PARAM_KEYS = {
        # generic inputs in experiment.json
        "algorithm", "dataset", "epsilon", "gamma", "dgp", "contamination",
        "dim", "num_observations", "num_test_observations", "num_replications",
        "ignore_dpp", "njobs", "normalise",
        # real-world knobs that are typically in experiment.json
        "rw_cal_fraction", "rw_split_seed", "rw_cache_dir", "rw_device",
        "rw_head", "rw_mlp_hidden",
        "rw_train_batch_size", "rw_train_num_workers", "rw_epochs", "rw_lr", "rw_weight_decay",
        "rw_smoothmax_temperature", "rw_groupdro_step_size", "rw_max_grad_norm",
        "rw_text_n_features", "rw_text_ngram_min", "rw_text_ngram_max",
    }

    def _json_default(o):
        # numpy scalar -> python scalar
        if isinstance(o, np.generic):
            return o.item()
        # numpy array -> list
        if isinstance(o, np.ndarray):
            return o.tolist()
        # torch tensor -> list
        if torch.is_tensor(o):
            return o.detach().cpu().tolist()
        # Path -> str
        if isinstance(o, Path):
            return str(o)
        # fallback (keeps logging robust)
        return str(o)

    rows = []
    out_dir = experiment_data_dir or experiment_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_filepath = out_dir / f"{uuid}.jsonl"
    print(f"Writing REAL-WORLD JSONL to {jsonl_filepath}")
    with jsonl_filepath.open("w", encoding="utf-8") as jf:
        for r in range(int(num_replications)):
            stats = run_real_world_replication(
                replication=r,
                dataset=dataset,
                algorithm=algorithm,
                epsilon=float(epsilon),
                gamma=float(gamma),
                dataset_dir=Path(rw_dataset_dir),

                rw_cal_fraction=float(rw_cal_fraction),
                rw_split_seed=int(rw_split_seed),
                rw_cache_dir=rw_cache_dir,
                rw_device=str(rw_device),

                rw_head=str(rw_head),
                rw_mlp_hidden=int(rw_mlp_hidden),

                rw_train_batch_size=int(rw_train_batch_size),
                rw_train_num_workers=int(rw_train_num_workers),
                rw_epochs=int(rw_epochs),
                rw_lr=float(rw_lr),
                rw_weight_decay=float(rw_weight_decay),
                rw_smoothmax_temperature=float(rw_smoothmax_temperature),
                rw_groupdro_step_size=float(rw_groupdro_step_size),
                rw_max_grad_norm=rw_max_grad_norm,

                rw_text_n_features=int(rw_text_n_features),
                rw_text_ngram_min=int(rw_text_ngram_min),
                rw_text_ngram_max=int(rw_text_ngram_max),
                verbose=verbose,
                do_threshold_calibration = do_threshold_calibration
            )

            # --- JSONL: write FULL stats per replication (do NOT drop keys) ---
            json_row = dict(stats)
            json_row["uuid"] = uuid
            json_row["replication"] = int(r)
            jf.write(json.dumps(json_row, default=_json_default) + "\n")
            jf.flush()

            row = dict(stats)
            row["uuid"] = uuid
            row["replication"] = int(r)

            for k in DROP_PARAM_KEYS:
                row.pop(k, None)

            rows.append(row)

    df = pd.DataFrame(rows)
    csv_filepath = out_dir / f"{uuid}.csv"
    print(f"Writing REAL-WORLD CSV to {csv_filepath}")
    df.to_csv(csv_filepath, index=False)

    _update_uuid_progress(experiment_dir, experiment_data_dir)


@app.command(name="csv")
def generate_csv(experiment_dir: Path, npl_samples_dir: Optional[Path] = None, experiment_data_dir: Optional[Path] = None):
    """Write a CSV file with all the results"""
    experiment_filepath = experiment_dir / "experiment.json"
    with open(experiment_filepath, "r", encoding="utf-8") as json_file:
        experiment = json.load(json_file)
    experiment_df = pd.DataFrame(experiment).set_index("uuid")
    print("Loading and concatenating", len(experiment_df), "CSV files into a pandas dataframe...")
    result_df = pd.DataFrame()
    result_list = []
    failed_uuid_list = []
    missing_uuid_list = []
    data_dir = experiment_data_dir or experiment_dir
    for uuid in experiment_df.index:
        csv_path = data_dir / f"{uuid}.csv" 
        if csv_path.exists():
            try:
                result_list.append(pd.read_csv(
                    csv_path, index_col=["uuid", "replication"]
                ))
            except pd.errors.ParserError:
                failed_uuid_list.append(uuid)
        else:
            missing_uuid_list.append(uuid)

    print("The following UUIDs did not have a CSV file:")
    print(missing_uuid_list)
    print()
    print("The following UUIDs failed due to a pandas.errors.ParserError:")
    print(failed_uuid_list)
    result_df = pd.concat([result_df] + result_list)

    # Join config columns from experiment.json.
    # Use rsuffix to tolerate per-uuid CSVs that already include some config columns
    # (e.g. california_housing writes algorithm/dataset/epsilon per replication).
    result_df = result_df.join(experiment_df, on="uuid", rsuffix="_cfg")

    # If any columns overlapped, prefer the per-uuid value when present; otherwise fall back to config.
    for c in experiment_df.columns:
        cfg_c = f"{c}_cfg"
        if cfg_c in result_df.columns:
            if c in result_df.columns:
                result_df[c] = result_df[c].fillna(result_df[cfg_c])
            else:
                result_df[c] = result_df[cfg_c]
            result_df = result_df.drop(columns=[cfg_c])

    result_df = result_df.reset_index()
    if npl_samples_dir:
        # load the settings for the NPL sampling
        settings_df = pd.read_csv(npl_samples_dir / "npl_settings.csv")
        # filter df because the empirical method doesn't produce anything and we get an error
        filtered_settings_df = settings_df[settings_df["inference"] == "npl_mmd"]
        # then, for each npl_uuid, load the times taken for each replication
        times_df = pd.concat([pd.read_csv(npl_samples_dir / npl_uuid / f"npl_times_{npl_uuid}.csv") for npl_uuid in filtered_settings_df["npl_uuid"]])
        # now merge the times and the npl_uuids together
        result_df = result_df.merge(settings_df, how="left", on=POSTERIOR_GB_COLS)
        result_df = result_df.merge(times_df, on=["npl_uuid", "replication"], how="left", suffixes=('', '_drop'))
        # finally, replace the incorrect posterior times with the correct ones
        result_df.loc[~(result_df['posterior_time_drop'].isna()), 'posterior_time'] = result_df['posterior_time_drop']
        result_df = result_df.drop("posterior_time_drop", axis=1)
    # save to a CSV file
    print(result_df)
    out_path = data_dir / "results.csv" 
    result_df.to_csv(out_path, index=False)

@app.command(name="summary")
def summary(experiment_dir: Path, experiment_data_dir: Path) -> None:
    RESULTS_CSV = experiment_data_dir / "results.csv"
    raw_df = pd.read_csv(RESULTS_CSV)

    # Detect dataset (assume one dataset per experiment directory).
    dataset_name = ""
    if "dataset" in raw_df.columns:
        ds_vals = raw_df["dataset"].dropna().astype(str).unique()
        if len(ds_vals) == 1:
            dataset_name = str(ds_vals[0])
        elif len(ds_vals) > 1:
            raise ValueError(
                f"results.csv contains multiple datasets: {sorted(ds_vals)}. "
                "Run summary separately per dataset/experiment directory."
            )

    # --- California Housing summary ---
    if dataset_name == "california_housing":
        df_ch = raw_df.copy()
        gap_ratio = df_ch["gap_ratio"].iloc[0] if "gap_ratio" in df_ch.columns else None
        # Coerce key columns to numeric (robust against dtype inference quirks)
        numeric_cols = [
            "epsilon",
            "contamination",
            "gamma_bulk",
            "gap_ratio",
            "mae",
            "rmse",
            "mae_trivial",
            "cvar_trivial_error",
            "outlier_mae",
            "p98_abs_error",
            "cvar_abs_error",
            "validation_time",
            "solve_time",
            "setup_time",
            "likelihood_time",
            "val_mae_raw",
            "val_mae_std",
            "cv_n_folds",
            "cv_n_blocks",
            "cv_lon_bins",
            "cv_lat_bins",
            "ridge_lambda_chosen",
            "wass_rho_chosen",
        ]
        for c in numeric_cols:
            if c in df_ch.columns:
                df_ch[c] = pd.to_numeric(df_ch[c], errors="coerce")

        # Total time (Option A: validation_time now means geo-block CV time)
        if "validation_time" in df_ch.columns and "solve_time" in df_ch.columns:
            df_ch["total_time"] = df_ch["validation_time"].fillna(0.0) + df_ch["solve_time"].fillna(0.0)+ df_ch["likelihood_time"].fillna(0.0)

        plots_dir = Path(experiment_dir) / "california_housing_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        plot_california_housing_mae_vs_cvar_trajectory(df_ch, plots_dir, gap_ratio=gap_ratio)
        
        # -----------------------------
        # Aggregated performance tables
        # -----------------------------
        def _make_perf_table(df_in: pd.DataFrame, *, group_cols: list) -> pd.DataFrame:
            agg_spec = {}

            # add count of replications if available
            if "replication" in df_in.columns:
                agg_spec["n_rep"] = ("replication", "count")

            # performance metrics (report mean/std/median)
            for m in ["mae", "rmse", "mae_trivial","cvar_trivial_error", "p98_abs_error", "cvar_abs_error"]:
                if m in df_in.columns:
                    agg_spec[f"{m}_mean"] = (m, "mean")
                    agg_spec[f"{m}_std"] = (m, "std")
                    agg_spec[f"{m}_median"] = (m, "median")

            # chosen hyperparameters + CV diagnostics + timings (report mean/std/median)
            for m in [
                "epsilon",
                "ridge_lambda_chosen",
                "wass_rho_chosen",
                "val_mae_raw",
                "val_mae_std",
                "cv_n_folds",
                "cv_n_blocks",
                "cv_lon_bins",
                "cv_lat_bins",
                "validation_time",
                "likelihood_time",
                "solve_time",
                "total_time",
            ]:
                if m in df_in.columns:
                    agg_spec[f"{m}_mean"] = (m, "mean")
                    agg_spec[f"{m}_std"] = (m, "std")
                    agg_spec[f"{m}_median"] = (m, "median")

            return (
                df_in.groupby(group_cols, dropna=False)
                .agg(**agg_spec)
                .reset_index()
            )

        has_cal_flag = "calibrate_on_validation" in df_ch.columns
        if has_cal_flag:
            cal_series = df_ch["calibrate_on_validation"].fillna(False).astype(bool)
            df_cal = df_ch[cal_series].copy()
            df_fixed = df_ch[~cal_series].copy()
        else:
            df_cal = df_ch.iloc[0:0].copy()
            df_fixed = df_ch.copy()

        # geo-block CV: no epsilon in grouping
        if df_cal.shape[0] > 0:
            group_cols = ["algorithm", "gamma_bulk", "calibrate_on_validation"]
            for extra in ["ch_geo_split", "ch_geo_axis"]:
                if extra in df_cal.columns:
                    group_cols.append(extra)

            perf_table = _make_perf_table(df_cal, group_cols=group_cols)

            perf_csv = plots_dir / "california_housing_performance_table.csv"
            perf_table.to_csv(perf_csv, index=False)
            print("Saved California Housing performance table (geo-block CV) to", perf_csv)

        # Fixed-epsilon table (if present): keep epsilon in grouping
        if df_fixed.shape[0] > 0:
            group_cols = ["algorithm", "epsilon", "contamination", "gamma_bulk"]
            if "calibrate_on_validation" in df_fixed.columns:
                group_cols.append("calibrate_on_validation")
            for extra in ["ch_geo_axis"]:
                if extra in df_fixed.columns:
                    group_cols.append(extra)

            perf_table_fixed = _make_perf_table(df_fixed, group_cols=group_cols)

            perf_csv_fixed = plots_dir / "california_housing_performance_table_fixed_epsilon.csv"
            perf_table_fixed.to_csv(perf_csv_fixed, index=False)
            print("Saved California Housing performance table (fixed epsilon) to", perf_csv_fixed)

        return

    # Non-CH datasets: use the synthetic portfolio/newsvendor parser
    df, agg_df = prepare_portfolio_syn_results(raw_df, dataset=dataset_name if dataset_name else None)
    if df.empty:
        raise ValueError(f"No rows loaded from {RESULTS_CSV} (df is empty).")

    num_observations_list = df["num_observations"].unique() if "num_observations" in df.columns else []
    multiple_n = len(num_observations_list) > 1

    algo_list = df["algorithm"].unique() if "algorithm" in df.columns else []

    dataset_name = str(df["dataset"].iloc[0]) if "dataset" in df.columns else ""

    # Detect sample-efficiency experiments:
    # it should be sufficient to check whether num_likelihood_samples has multiple values for the first algorithm
    multiple_budget = False
    if dataset_name == "newsvendor" and {"algorithm", "num_likelihood_samples"}.issubset(df.columns):
        first_algo = df["algorithm"].iloc[0]
        mask_lv_bas = df["algorithm"].astype(str) == first_algo
        if mask_lv_bas.any():
            multiple_budget = df.loc[mask_lv_bas, "num_likelihood_samples"].nunique(dropna=True) > 1

    if dataset_name == "newsvendor" and multiple_budget:
        out_dir = Path(experiment_dir) / "sample_efficiency"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Build the joint ε × total_samples summary
        summ_eps_budget = summarise_vs_epsilon_and_total_samples(
            df,
            gamma = 0.0,
            t_scale=LV_NEWSVENDOR_T_SCALE,
            convert_kl_eps_to_lv_scale=True,
        )
        summ_eps_budget_csv = out_dir / "summarise_vs_epsilon_and_total_samples.csv"
        summ_eps_budget.to_csv(summ_eps_budget_csv, index=False)
        print("Saved", summ_eps_budget_csv)

        # 2) Sample-efficiency stats (per algorithm)
        se_df = sample_efficiency_summary(df)
        se_csv = out_dir / "sample_efficiency_summary.csv"
        se_df.to_csv(se_csv, index=False)
        print("Saved", se_csv)

        # 3) Plot sample-efficiency
        _ = plot_sample_efficiency(
            df,
            out_path=out_dir / "sample_efficiency.pdf",
            out_csv=out_dir / "sample_efficiency_plot_data.csv",
            title="Sample efficiency (newsvendor Student-t)",
        )

        # 4) Mean–variance trade-off per algorithm across total_samples
        for algorithm in algo_list:
            _ = plot_frontiers_by_total_samples_per_algorithm(
                df,
                algorithm=algorithm,
                title=f"Mean–variance frontiers ({algorithm})",
                out_path=out_dir / f"{algorithm}_frontiers_by_total_samples.pdf",
            )
        # 5) MSD vs ε per algorithm across total_samples
        for algorithm in algo_list:
            _ = plot_msd_vs_epsilon_by_total_samples_per_algorithm(
                df,
                algorithm=algorithm,
                t_scale=LV_NEWSVENDOR_T_SCALE,
                out_path=out_dir / f"{algorithm}_msd_by_total_samples.pdf",
            )

        # Also keep basic runtime artefacts (useful sanity check)
        runtime_pdf = out_dir / "runtime_by_algorithm.pdf"
        _ = plot_runtime_by_algorithm(df, out_path=runtime_pdf, log_scale=False)

        runtime_table_csv = out_dir / "runtime_table.csv"
        _ = make_and_save_runtime_table(df, out_path=runtime_table_csv)
        return

    multiple_gamma_bulk = False
    if dataset_name == "newsvendor" and {"algorithm", "gamma_bulk"}.issubset(df.columns):
        mask_lv_bas = df["algorithm"].astype(str) == "lv_bas"
        if mask_lv_bas.any():
            gamma_vals = pd.to_numeric(df.loc[mask_lv_bas, "gamma_bulk"], errors="coerce")
            multiple_gamma_bulk = gamma_vals.nunique(dropna=True) > 1

    if dataset_name == "newsvendor" and multiple_gamma_bulk:
        out_dir = Path(experiment_dir) / "gamma_sensitivity"
        out_dir.mkdir(parents=True, exist_ok=True)

        summ_eps_gamma = summarise_vs_epsilon_and_gamma_bulk(
            df,
            gamma=0.0,
            t_scale=LV_NEWSVENDOR_T_SCALE,
            convert_kl_eps_to_lv_scale=True,
        )
        summ_eps_gamma_csv = out_dir / "summarise_vs_epsilon_and_gamma_bulk.csv"
        summ_eps_gamma.to_csv(summ_eps_gamma_csv, index=False)
        print("Saved", summ_eps_gamma_csv)

        _ = plot_frontiers_by_gamma_bulk(
            df,
            algorithm="lv_bas",
            title=r"Mean--variance frontiers for $\mathrm{LV}$ with varying $\gamma$",
            out_path=out_dir / "lv_bas_frontiers_by_gamma_bulk.pdf",
        )
        _ = plot_msd_vs_epsilon_by_gamma_bulk(
            df,
            algorithm="lv_bas",
            t_scale=LV_NEWSVENDOR_T_SCALE,
            out_path=out_dir / "lv_bas_msd_by_gamma_bulk.pdf",
        )
        print("Saved gamma sensitivity frontier and MSD plots.")
        return

    if dataset_name == "newsvendor":
        frontier_pdf = experiment_dir / "lv_newsvendor_syn_frontiers.pdf"
        _ = plot_oos_frontiers(
            agg_df,
            title="Synthetic newsvendor: OOS mean–variance frontiers",
            special_epsilons=[0.01, 0.1, 0.25, 1.0],
            out_path=frontier_pdf,
            include_legend=True,
        )
        _ = plot_msd_vs_epsilon(
            df,
            t_scale=LV_NEWSVENDOR_T_SCALE,
            out_path=experiment_dir / "MSD_vs_epsilon.pdf",
            out_csv=experiment_dir / "MSD_vs_epsilon.csv",
            title="",
            include_legend=True,
        )
        runtime_pdf = experiment_dir / "lv_newsvendor_syn_runtime.pdf"
        _ = plot_runtime_by_algorithm(df, out_path=runtime_pdf, log_scale=False)

        runtime_table_csv = experiment_dir / "lv_newsvendor_syn_runtime_table.csv"
        _ = make_and_save_runtime_table(df, out_path=runtime_table_csv)
        return

    if multiple_n:
        plots_dir = experiment_dir
        plots_dir.mkdir(parents=True, exist_ok=True)

        df_plot = df.copy()
        fig, ax, runtime_vs_nobs = plot_runtime_vs_num_observations(
            df_plot,
            title="Runtime vs number of observations",
            logy=False,
            out_path=plots_dir / "runtime_vs_num_observations.pdf",
            out_csv=plots_dir / "runtime_vs_num_observations.csv",
        )

        frontier_dir = experiment_dir / "frontier_plots_by_n"
        save_frontiers_per_n(df, out_dir=frontier_dir)
        return

    # Default: portfolio-style single-n plots
    frontier_pdf = experiment_dir / "lv_portfolio_syn_frontiers.pdf"
    _ = plot_oos_frontiers(
        agg_df,
        title="Synthetic portfolio: OOS mean–variance frontiers",
        special_epsilons=[0.001, 0.01, 0.1, 1.0],
        out_path=frontier_pdf,
    )

    runtime_pdf = experiment_dir / "lv_portfolio_syn_runtime.pdf"
    _ = plot_runtime_by_algorithm(df, out_path=runtime_pdf, log_scale=False)

    runtime_table_csv = experiment_dir / "lv_portfolio_syn_runtime_table.csv"
    _ = make_and_save_runtime_table(df, out_path=runtime_table_csv)

    gamma_list = [1, 2, 2.5, 3, 4, 5]
    for gamma in gamma_list:
        plot_ce_gamma_vs_epsilon(
            df,
            gamma=gamma,
            out_path=experiment_dir / f"ce_gamma_vs_epsilon_gamma_{gamma}.pdf",
            out_csv=experiment_dir / f"ce_gamma_vs_epsilon_gamma_{gamma}.csv",
            title="Certainty Equivalent vs tolerance",
        )

@app.command(name="summary-realworld")
def summary_realworld(
    experiment_dir: Path,
    experiment_data_dir: Path,
    dataset: Optional[str] = None,
) -> None:
    """
    Summarise real-world (PyTorch) LV-BAS experiments:
      - curves vs ε with panels by γ (Option A)
      - Pareto plots (avg vs tail/worst)
      - Option-B selection based on validation metrics only

    This is intentionally separate from `summary`, which is currently tailored to
    the synthetic portfolio pipeline.
    """
    RESULTS_CSV = experiment_data_dir / "results.csv"
    raw_df = pd.read_csv(RESULTS_CSV)

    _, agg_df = prepare_realworld_results(raw_df, dataset=dataset)

    plots_dir = experiment_dir / "realworld_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    agg_csv = plots_dir / "realworld_agg.csv"
    agg_df.to_csv(agg_csv, index=False)
    print("Saved aggregated real-world summary to", agg_csv)

    selected_df, sel_meta = select_realworld_minimax(agg_df, split="val")
    sel_csv = plots_dir / "minimax_selection.csv"
    selected_df.to_csv(sel_csv, index=False)
    print("Saved Minimax selection table to", sel_csv)
    print("Minimax meta:", sel_meta)

    # Determine task instance columns for per-slice plots
    task_cols = [c for c in ["dataset", "scenario", "eps_true", "corruption", "severity"] if c in agg_df.columns]
    if not task_cols:
        task_records = [dict()]  # single global plot
    else:
        task_records = agg_df[task_cols].drop_duplicates().to_dict("records")

    metric_alias = {
        "avg_acc": "acc_mean",
        "worst_group_acc": "worst_group_acc",
        "p90_loss": "loss_p90",          
        "p95_loss": "loss_p95",
        "max_loss": "loss_max",
        "bulk_coverage": "bulk_coverage",
    }

    candidate_metrics = ["avg_acc", "worst_group_acc", "p90_loss", "p95_loss", "max_loss", "bulk_coverage"]

    for rec in task_records:
        # Safe filename stem
        stem_parts = []
        for k, v in rec.items():
            if pd.isna(v):
                continue
            if isinstance(v, float):
                vv = f"{v:g}".replace(".", "p")
            else:
                vv = str(v)
            vv = vv.replace("/", "_").replace(" ", "_")
            stem_parts.append(f"{k}-{vv}")
        stem = "__".join(stem_parts) if stem_parts else "all"

        if rec.get("dataset") in ("rw_civilcomments", "civilcomments"):
            if "test_worst_group_acc_mean" in agg_df.columns:
                out_pdf = plots_dir / f"{stem}__bulk_ablation_test_worst_group_acc_vs_epsilon_by_gamma.pdf"
                out_csv = plots_dir / f"{stem}__bulk_ablation_test_worst_group_acc_vs_epsilon_by_gamma.csv"
                plot_realworld_bulk_ablation_sweeps(
                    agg_df,
                    metric="worst_group_acc",
                    split="test",
                    dataset=rec.get("dataset"),
                    scenario=rec.get("scenario"),
                    eps_true=rec.get("eps_true"),
                    out_path=out_pdf,
                    out_csv=out_csv,
                )

            if "test_acc_mean_mean" in agg_df.columns:
                out_pdf = plots_dir / f"{stem}__bulk_ablation_test_avg_acc_vs_epsilon_by_gamma.pdf"
                out_csv = plots_dir / f"{stem}__bulk_ablation_test_avg_acc_vs_epsilon_by_gamma.csv"
                plot_realworld_bulk_ablation_sweeps(
                    agg_df,
                    metric="acc_mean",
                    split="test",
                    dataset=rec.get("dataset"),
                    scenario=rec.get("scenario"),
                    eps_true=rec.get("eps_true"),
                    out_path=out_pdf,
                    out_csv=out_csv,
                )

        x_key = metric_alias["avg_acc"] 

        if f"test_{x_key}_mean" in agg_df.columns:
            if "test_worst_group_acc_mean" in agg_df.columns:
                y_key = "worst_group_acc"
                y_label = "worst_group_acc"
            else:
                y_key = None
                y_label = None

            if y_key is not None:
                out_pdf = plots_dir / f"{stem}__pareto_test_{y_label}_vs_avg_acc.pdf"
                out_csv = plots_dir / f"{stem}__pareto_test_{y_label}_vs_avg_acc.csv"
                plot_realworld_pareto(
                    agg_df,
                    x_metric=x_key,     # "acc_mean"
                    y_metric=y_key,     # e.g. "worst_group_acc" or "loss_p95"
                    split="test",
                    dataset=rec.get("dataset"),
                    scenario=rec.get("scenario"),
                    eps_true=rec.get("eps_true"),
                    selected_df=selected_df,
                    out_path=out_pdf,
                    out_csv=out_csv,
                )

@app.command(name="prepare-realworld-cache")
def prepare_realworld_cache(
    dataset_dir: Path,
    dataset: str,
    rw_cache_dir: Optional[Path] = None,
    rw_device: str = "auto",
    rw_split_seed: int = 0,
    rw_cal_fraction: float = 0.2,
    # text
    rw_text_n_features: int = 2**12,
    rw_text_ngram_min: int = 1,
    rw_text_ngram_max: int = 1,
    rw_text_batch_size: int = 256,
    rw_text_num_workers: int = 4,
    prepare_dense_text_features: bool = False,
) -> None:
    from .wilds.runner import prepare_real_world_caches  # adjust import path to your project

    meta = prepare_real_world_caches(
        dataset=dataset,
        dataset_dir=dataset_dir,
        rw_cache_dir=rw_cache_dir,
        rw_device=rw_device,
        rw_split_seed=rw_split_seed,
        rw_cal_fraction=rw_cal_fraction,
        rw_text_n_features=rw_text_n_features,
        rw_text_ngram_min=rw_text_ngram_min,
        rw_text_ngram_max=rw_text_ngram_max,
        rw_text_batch_size=rw_text_batch_size,
        rw_text_num_workers=rw_text_num_workers,
        prepare_dense_text_features=prepare_dense_text_features,
    )
    print("Cache preparation complete:")
    for k, v in meta.items():
        print(f"  {k}: {v}")


@app.command(name="experiment")
def run_experiment(
    experiment_dir: Path,
    dgp: str,
    algorithm: str,
    only_missing: bool = False, #NOTE if the experiment runs out of memory set this to true to true 
    njobs: int = -1,
    dataset_dir: Path = None,
    experiment_data_dir: Optional[Path] = None,
):
    """When using SLURM, this function is called to run an experiment"""
    print(datetime.now(), "- Running algorithm", algorithm, "with DGP", dgp, "from experiment directory", experiment_dir)
    filepath = experiment_dir / "experiment.json"
    with open(filepath, "r", encoding="utf-8") as json_file:
        experiment = json.load(json_file)
    # NOTE if only-missing flag, then only run if the results CSV file doesn't exist
    for params in experiment:
        if params["dgp"] == dgp and params["algorithm"] == algorithm and not ((experiment_dir / params["uuid"]).exists() and only_missing):
            run(experiment_dir, experiment_data_dir=experiment_data_dir, **params, njobs=njobs)


@app.command(name="batch")
def batch(experiment_dir: Path, batch_id: int, batch_size: int, only_missing: bool = False, dataset_dir: Path = None, npl_samples_dir: Optional[Path] = None, experiment_data_dir: Optional[Path] = None):
    print(datetime.now(), "Running batch from array index", batch_id)
    print()
    filepath = experiment_dir / "experiment.json"
    with open(filepath, "r", encoding="utf-8") as json_file:
        experiment = json.load(json_file)
    start = batch_id * batch_size
    batch_experiment = experiment[start: min(start + batch_size, len(experiment))]
    for params in batch_experiment:
        if not ((experiment_dir / params["uuid"]).exists() and only_missing):
            run(experiment_dir, dataset_dir=dataset_dir, npl_samples_dir=npl_samples_dir, experiment_data_dir=experiment_data_dir, **params)

@app.command(name="uuid")
def run_uuid(experiment_dir: Path, uuid: UUID, dataset_dir: Path = None, npl_samples_dir: Optional[Path] = None, njobs: int = -1, verbose: bool = False, experiment_data_dir: Optional[Path] = None) -> None:
    """Run DRO for only one specified uuid parameters"""
    filepath = experiment_dir / "experiment.json"
    with open(filepath, "r", encoding="utf-8") as json_file:
        experiment = json.load(json_file)
    found = False
    for params in experiment:
        if params["uuid"] == str(uuid):
            found = True
            # verbose is already in params; don't pass it twice
            run(experiment_dir, dataset_dir=dataset_dir, npl_samples_dir=npl_samples_dir, experiment_data_dir=experiment_data_dir, **params)
    if not found:
        raise ValueError(f"UUID {uuid} not found in {filepath}")


def _update_uuid_progress(experiment_dir: Path, experiment_data_dir: Optional[Path]) -> None:
    """
    Update a tqdm-style progress file tracking how many UUID jobs are complete.

    - Total = number of entries in experiment.json
    - Completed = number of UUID-named CSVs in experiment_data_dir (or experiment_dir)
    - Progress file is written in the *parent* of experiment_dir, so it can be shared
      and easily inspected.
    """
    try:
        data_dir = experiment_data_dir or experiment_dir

        # Total jobs = number of rows in experiment.json
        experiment_path = experiment_dir / "experiment.json"
        with open(experiment_path, "r", encoding="utf-8") as f:
            experiment = json.load(f)
        total = len(experiment)

        # Completed jobs = number of UUID-style CSVs in the data directory
        completed = 0
        if data_dir.exists():
            for p in data_dir.iterdir():
                if p.is_file() and _UUID_CSV_RE.match(p.name):
                    completed += 1

        # Basic safety
        if total <= 0:
            total = max(completed, 1)

        bar_width = 30
        filled = int(bar_width * min(completed, total) / total)
        bar = "#" * filled + "-" * (bar_width - filled)

        progress_dir = experiment_dir.parent
        progress_dir.mkdir(parents=True, exist_ok=True)
        progress_path = progress_dir / "progress.txt"
        progress_str = f"[{bar}] {completed}/{total} jobs completed\n"

        # append instead of overwrite so `tail -f` sees new lines
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(progress_str)

        print(progress_str.strip())
    except Exception as e:
        print("Warning: failed to update UUID progress:", e)


@app.command(name="run")
def run(
    experiment_dir: Path,
    algorithm: str = "kl_bdro",
    contamination: float = CONTAMINATION_LEVEL,
    contamination_type: Optional[str] = None,
    dataset: str = "newsvendor",
    dataset_dir: Optional[Path] = None,
    dgp: str = "truncated_normal",
    dim: int = 1,
    epsilon: float = 1.0,
    eta: float = NPL_ETA,
    ignore_dpp: bool = False,
    inference: str = "bayes",
    kernel_name: str = "k_jax",
    lengthscale: float = -1.0,
    likelihood: str = "exponential",
    njobs: int = -1,
    normalise: bool = False,
    npl_samples_dir: Optional[Path] = None,
    num_likelihood_samples: int = NUM_LIKELIHOOD_SAMPLES,
    num_observations: int = NUM_OBSERVATIONS,
    num_posterior_samples: int = NUM_POSTERIOR_SAMPLES,
    num_replications: int = NUM_REPLICATIONS,
    num_test_observations: int = NUM_TEST_OBSERVATIONS,
    num_certify_points: int = NUM_CERTIFY,
    posterior: str = "gamma",
    uuid: str = str(uuid4()),
    verbose: bool = False,
    lv_use_pc_bulk_geometry: bool = False,
    experiment_data_dir: Optional[Path] = None,
    gamma_bulk: float = 0.05,
    standardise_y: bool = False,
    calibrate_on_validation: bool = False,
    Ch_gap_ratio: float = 0.0,
    clip_extreme_y: bool = False,
    vary_rho: bool = False,
):
    """Run Newsvendor Misspecified Bayesian DRO"""
    if uuid:
        print(uuid)
    print("DGP:", dgp, " - ALGORITHM:", algorithm, " - NUM LIKELIHOOD SAMPLES:", num_likelihood_samples, " - POSTERIOR:", posterior, "- DATASET:", dataset, "- DIM:", dim)

    # Real-world experiments are handled by `run_real_world` / `batch_real_world`
    if dataset.lower().startswith("rw_") or dataset.lower() in {"waterbirds", "celeba", "civilcomments"}:
        raise ValueError(
            "Real-world datasets must be run via `credaldro run_real_world` "
            "(and on cluster via `credaldro batch_real_world`)."
        )

    if algorithm in ("kl_bdro", "kl_dro_bas", "kl_pp", "kl_empirical") and dataset == "newsvendor":
        problem = get_kl_bdro_problem(
            newsvendor_cost_cvxpy, num_posterior_samples, num_likelihood_samples, dim=dim,
        )
    elif algorithm == "or_wdro" and dataset == "newsvendor":
        problem = get_or_wdro_newsvendor_problem(
            dim=dim, num_observations=num_observations,
        )
    elif algorithm == "kl_pp" and dataset in ("portfolio", "portfolio_synthetic"):
        problem = get_kl_bdro_problem(portfolio_objective_cvxpy, num_posterior_samples, num_likelihood_samples, dim=dim, is_portfolio=True)
    elif algorithm == "kl_empirical" and dataset in ("portfolio", "portfolio_synthetic"):
        problem = get_kl_bdro_problem(portfolio_objective_cvxpy, 1, num_observations, dim=dim, is_portfolio=True)
    elif algorithm in ("kl_bdro", "kl_dro_bas") and dataset in ("portfolio", "portfolio_synthetic") and likelihood == "multivariate_normal":
        problem = get_kl_portfolio_problem(dim, num_posterior_samples)
    elif algorithm == "lv_bas" and dataset in ("portfolio", "portfolio_synthetic"):
        problem = get_lv_portfolio_problem(dim)
    elif algorithm == "lv_bas" and dataset == "newsvendor":
        problem = get_lv_newsvendor_problem(dim=dim, n_trunc=int(num_likelihood_samples*0.5))
    elif dataset == "california_housing" and algorithm == "lv_bas_ch":
        if dim != 8:
            raise ValueError("california_housing expects dim=8 (number of features).")
        problem = get_lv_bas_ch_problem(dim=dim, num_samples=int(num_likelihood_samples)) # placeholder; overwritten for ellipsoid_x_interval_y bulk set
    elif dataset == "california_housing" and algorithm == "erm_lad":
        if dim != 8:
            raise ValueError("california_housing expects dim=8 (number of features).")
        from .dataset import california_housing_split_sizes
        n_train = int(california_housing_split_sizes()["n_train"] + california_housing_split_sizes()["n_select"])
        problem = get_erm_lad_problem(dim=dim, num_samples=n_train)
    elif dataset == "california_housing" and algorithm == "erm_ridge":
        if dim != 8:
            raise ValueError("california_housing expects dim=8 (number of features).")
        from .dataset import california_housing_split_sizes
        n_train = int(california_housing_split_sizes()["n_train"] + california_housing_split_sizes()["n_select"])
        problem = get_erm_ridge_problem(dim=dim, num_samples=n_train)
    elif dataset == "california_housing" and algorithm == "wass_lad":
        if dim != 8:
            raise ValueError("california_housing expects dim=8 (number of features).")
        from .dataset import california_housing_split_sizes
        n_train = int(california_housing_split_sizes()["n_train"] + california_housing_split_sizes()["n_select"])
        problem = get_wass_lad_problem(dim=dim, num_samples=n_train)
    elif dataset == "california_housing" and algorithm == "or_wdro":
        # OR-WDRO (python-dro) is solved outside CVXPY in `run_replication` for California Housing.
        # We create a trivial CVXPY problem here for compatibility with the existing driver logic.
        problem = cp.Problem(cp.Minimize(0), [])
    elif dataset == "california_housing" and algorithm == "chi2_lad":
        # Chi2-DRO (python-dro) is solved outside CVXPY in `run_replication` for California Housing.
        problem = cp.Problem(cp.Minimize(0), [])
    elif dataset == "california_housing" and algorithm == "kl_lad":
        # KL-DRO (python-dro) is solved outside CVXPY in `run_replication` for California Housing.
        problem = cp.Problem(cp.Minimize(0), [])
    elif dataset == "california_housing" and algorithm == "cvar_lad":
        if dim != 8:
            raise ValueError("california_housing expects dim=8 (number of features).")
        from .dataset import california_housing_split_sizes
        n_train = int(california_housing_split_sizes()["n_train"] + california_housing_split_sizes()["n_select"])
        if float(epsilon) <= 0.0:
            problem = get_max_lad_problem(dim=dim, num_samples=n_train)
        else:
            problem = get_cvar_lad_problem(dim=dim, num_samples=n_train)
    elif algorithm in ("dro_bas_mmd", "empirical_mmd"):
        dim_theta = dim
        if algorithm == "dro_bas_mmd":
            n_samples = num_posterior_samples*num_likelihood_samples
        elif algorithm == "empirical_mmd":
            n_samples = num_observations
        if dataset == "newsvendor":
            kdro_class = DRO_BAS_MMD(dim_theta, dim, newsvendor_cost_cvxpy)
            problem = kdro_class.get_newsvendor_problem(n_samples, num_certify_points)
        elif dataset in ("portfolio", "portfolio_synthetic"):
            kdro_class = DRO_BAS_MMD(dim_theta, dim, portfolio_objective_cvxpy)
            problem = kdro_class.get_portfolio_problem(n_samples, num_certify_points)
        else:
            raise ValueError(f"Objective not implemented for dataset '{dataset}'")
    else:
        raise NotImplementedError(f"Algorithm {algorithm} not implemented.")
    # If the number of parameters is small enough, then use Disciplined Parametrized Programming (DPP)
    # to reduce the compilation time in each replication.
    # However, a large number of parameters uses an enormous amout of RAM in the current cvxpy implementation.
    if not ignore_dpp and algorithm in ("kl_bdro", "kl_dro_bas", "dro_bas_mmd", "empirical_mmd", "lv_bas", "or_wdro","lv_bas_ch","erm_lad","cvar_lad"):
        n_parameters = np.sum(np.prod(param.shape) for param in problem.parameters())
        # NOTE whilst DRO-BAS can handle at least 5000 params, BDRO cannot.
        # So, for a fair comparison, we turn off DPP for both DRO-BAS and BDRO.
        if algorithm != "or_wdro" and n_parameters >= cp.settings.PARAM_THRESHOLD:
        # if n_parameters >= 1000:
            ignore_dpp = True
            njobs = 1
    if algorithm == "wass_lad":
        ignore_dpp = True

    # California Housing run_replication does NOT use the passed `problem` object
    # (it creates its own CVXPY problems internally), so we set it to None to avoid
    # pickling errors with joblib Parallel.
    if dataset == "california_housing":
        problem = None
        # Reset njobs since the DPP check above may have set it to 1
        # based on the now-discarded placeholder problem.
        njobs = -1

    params = {
        "algorithm": algorithm,
        "contamination": contamination,
        "contamination_type": contamination_type,
        "dataset": dataset,
        "dataset_dir": dataset_dir,
        "dgp": dgp,
        "dim": dim,
        "epsilon": epsilon,
        "eta": eta,
        "ignore_dpp": ignore_dpp,
        "inference": inference,
        "lengthscale": lengthscale,
        "likelihood": likelihood,
        "normalise": normalise,
        "num_certify_points": num_certify_points,
        "num_likelihood_samples": num_likelihood_samples,
        "num_observations": num_observations,
        "num_posterior_samples": num_posterior_samples,
        "num_replications": num_replications,
        "num_test_observations": num_test_observations,
        "posterior": posterior,
        "uuid": uuid,
        "verbose": verbose,
        "lv_use_pc_bulk_geometry": lv_use_pc_bulk_geometry,
        "gamma_bulk": gamma_bulk,
        "standardise_y": standardise_y,
        "calibrate_on_validation": calibrate_on_validation,
        "Ch_gap_ratio": Ch_gap_ratio,
        "clip_extreme_y": clip_extreme_y,
        "vary_rho": vary_rho
    }

    if inference in ("npl_wlb", "npl_mmd"):
        posterior_df = pd.read_csv(npl_samples_dir / "npl_settings.csv").set_index(POSTERIOR_GB_COLS)
        npl_params = {key: params[key] for key in POSTERIOR_GB_COLS}
        npl_uuid = get_npl_uuid(posterior_df, npl_params)
        params["npl_uuid_dir"] = npl_samples_dir / npl_uuid
    else:
        params["npl_uuid_dir"] = None
    params.pop("num_replications")  # popped because we don't need to pass this to the run_replication method, but it is needed above for getting the npl_uuid

    if njobs == 1:
        all_solve_start = datetime.now()
        list_of_replication_stats = []
        print(all_solve_start, "- Running all replications in series.")
        for j in range(num_replications):
            list_of_replication_stats.append(run_replication(j, problem, **params))
        all_solve_end = datetime.now()
        print(all_solve_end, "- Finished solving all replications in series. Total solve time is", (all_solve_end - all_solve_start).total_seconds())

    else:
        # NOTE we want to do the first replication solve *not* in parallel
        # because it takes a lot of memory to compile a large DPP problem.
        # Subsequent solves don't compile the problem and are solved in parallel
        print("Running replication 0: cvxpy Problem is compiling.")
        first_solve_start = datetime.now()
        replication_stats_0 = run_replication(0, problem, **params)
        print("Compilation + first solve took", (datetime.now() - first_solve_start).total_seconds(), "seconds.")
        all_solve_start = datetime.now()
        print(f"Solving remaining {num_replications - 1} replications in parallel.")
        list_of_replication_stats = Parallel(n_jobs=njobs)(
            delayed(run_replication)(j, problem, **params) for j in range(1, num_replications)
        )
        print("Solving all other replications took", (datetime.now() - all_solve_start).total_seconds(), "seconds.")
        list_of_replication_stats.insert(0, replication_stats_0)
        print(f"Finished running {algorithm} with posterior {posterior} and DGP {dgp}.")

    df = pd.DataFrame(list_of_replication_stats)
    out_dir = experiment_data_dir or experiment_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_filepath = out_dir / f"{uuid}.csv"
    print(f"Writing CSV to {csv_filepath}")
    print()
    df.to_csv(csv_filepath, index=False)
    _update_uuid_progress(experiment_dir, experiment_data_dir)


def run_replication(
    replication: int,
    problem: cp.Problem,
    algorithm: str = "kl_bdro",
    contamination: float = CONTAMINATION_LEVEL,
    contamination_type: Optional[str] = None,
    dataset: str = "newsvendor",
    dataset_dir: Optional[Path] = None,
    dgp: str = "truncated_normal",
    dim: int = 1,
    epsilon: float = 1.0,
    eta: float = NPL_ETA,
    ignore_dpp: bool = False,
    inference: str = "bayes",
    lengthscale: float = -1.0,
    likelihood: str = "exponential",
    normalise: bool = False,
    npl_uuid_dir: Optional[Path] = None,
    num_certify_points: int = NUM_CERTIFY,
    num_likelihood_samples: int = NUM_LIKELIHOOD_SAMPLES,
    num_observations: int = NUM_OBSERVATIONS,
    num_posterior_samples: int = NUM_POSTERIOR_SAMPLES,
    num_test_observations: int = NUM_TEST_OBSERVATIONS,
    posterior: str = "gamma",
    uuid: str = str(uuid4()),
    verbose: bool = False,
    lv_use_pc_bulk_geometry: bool = False,
    gamma_bulk: float = 0.05,
    standardise_y: bool = False,
    calibrate_on_validation: bool = False,
    Ch_gap_ratio: float = 0.0,
    clip_extreme_y: bool = False,
    vary_rho: bool = False,
):
    """Run a single replication of the DRO experiment"""
    #print(f"num_observations: {num_observations}, num_test_observations: {num_test_observations}")
    # 1. generate dataset
    dgp_start = datetime.now()
    generator = np.random.default_rng(seed=replication)
    if dataset == "california_housing":
        # California Housing: continuous frequentist LV-BAS + baselines.
        # We bypass the Bayesian posterior/likelihood pipeline entirely.

        # ---- 1) Load splits and standardise X (train-only)
        ch_geo_axis = 0 # angle of the split (points to the training side. 0: east, 90: north, 180: west, 270: south)
        gap_ratio = Ch_gap_ratio  # gap between train/select/val and test
        bulk_shape = "ellipsoid_x_interval_y"
        #bulk_shape = "ellipsoid" #uncomment this line if you want try ellipsoid bulk
        #bulk_shape = "box" #uncomment this line if you want try box bulk
        from .dataset import (
            CALIFORNIA_HOUSING_D,
            CALIFORNIA_HOUSING_SPLIT_FRACS,
            california_housing_splits,
            calibrate_dkw_ellipsoid_bulk_set,
            calibrate_dkw_box_xi_bulk_set,
            fit_copula_ridge_centre,
            rejection_sample_centre_in_ellipsoid_bulk,
            empirical_cvar,
            in_ellipsoid_bulk,
            california_housing_split_geographic,
            _california_housing_geo_block_folds,
        )
        geo_splits = california_housing_split_geographic(
            axis=ch_geo_axis,
            seed=int(replication),
            standardise_y=standardise_y,
            gap_ratio=gap_ratio,  # gap between train/select/val and test
        )
        X_train = geo_splits["X_train"]
        y_train = geo_splits["y_train"]
        X_select = geo_splits["X_select"]
        y_select = geo_splits["y_select"]
        X_val = geo_splits["X_val"]
        y_val = geo_splits["y_val"]
        X_test = geo_splits["X_test"]
        y_test = geo_splits["y_test"]

        if verbose:
            print(
                f"[california_housing] Geographic split enabled: "
                f"axis={ch_geo_axis}. "
                f"TRAIN/SELECT sizes=({X_train.shape[0]}, {X_select.shape[0]}), TEST size={X_test.shape[0]}"
            )
        d = int(X_train.shape[1])
        if d != CALIFORNIA_HOUSING_D:
            raise ValueError(f"california_housing expects d={CALIFORNIA_HOUSING_D}, got {d}")

        # xi = (x, y) concatenated
        xi_train = np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1)
        xi_select = np.concatenate([X_select, y_select.reshape(-1, 1)], axis=1)

        dgp_time = (datetime.now() - dgp_start).total_seconds()

        if verbose:
            print("California Housing splits:")
            print(f"  X_train:  {X_train.shape}   y_train:  {y_train.shape}")
            print(f"  X_select: {X_select.shape}  y_select: {y_select.shape}")
            print(f"  X_test:   {X_test.shape}    y_test:   {y_test.shape}")

            # Sanity: standardisation uses TRAIN only
            train_mean = X_train.mean(axis=0)
            train_std = X_train.std(axis=0, ddof=0)
            print("  TRAIN X mean (should be ~0):", np.round(train_mean, 4))
            print("  TRAIN X std  (should be ~1):", np.round(train_std, 4))

            if np.any(~np.isfinite(X_train)) or np.any(~np.isfinite(y_train)):
                raise ValueError("Non-finite values detected in TRAIN split.")

        # ---- 2) Calibrate DKW bulk set Xi0 (shape depends on `bulk_shape`) using TRAIN/SELECT
        posterior_start = datetime.now()
        dkw_delta = 0.05  # fixed per spec

        # Normalise bulk_shape (accept a few aliases; default matches the existing ellipsoid in (x,y))
        bulk_shape_key = "ellipsoid" if bulk_shape is None else str(bulk_shape).strip().lower().replace("-", "_")
        if bulk_shape_key in {"ellipsoid_xi", "full_ellipsoid"}:
            bulk_shape_key = "ellipsoid"
        if bulk_shape_key in {"box_xi", "linf", "l_inf"}:
            bulk_shape_key = "box"
        if bulk_shape_key in {"ell_x_interval_y", "ell_x_int_y"}:
            bulk_shape_key = "ellipsoid_x_interval_y"

        # Helper: build the LV-BAS CVXPY problem that matches `bulk_shape_key`.
        # (Used in both geo-block CV and final optimisation; other algorithms ignore this.)
        def _make_lv_bas_problem(*, dim: int, num_samples: int) -> cp.Problem:
            if bulk_shape_key == "ellipsoid":
                return get_lv_bas_ch_problem(dim=int(dim), num_samples=int(num_samples))
            if bulk_shape_key == "box":
                from .california_housing_lp import get_lv_bas_box_xi_problem
                return get_lv_bas_box_xi_problem(dim=int(dim), num_samples=int(num_samples))
            if bulk_shape_key == "ellipsoid_x_interval_y":
                from .california_housing_lp import get_lv_bas_ellipsoid_x_interval_y_problem
                return get_lv_bas_ellipsoid_x_interval_y_problem(dim=int(dim), num_samples=int(num_samples))
            raise ValueError(f"Unknown bulk_shape='{bulk_shape}' (normalised='{bulk_shape_key}').")

        # Helper: set bulk-shape-specific parameters on an LV-BAS problem instance.
        def _set_lv_bas_bulk_params(
            prob: cp.Problem,
            *,
            mu_x: np.ndarray,
            mu_y: float,
            r_x: np.ndarray,
            r_y: float,
            t_hat: float,
            sqrt_Sigma_xi,
            sqrt_Sigma_x,
            t_x,
        ) -> None:
            if bulk_shape_key == "ellipsoid":
                prob.param_dict["mu_x"].value = mu_x
                prob.param_dict["mu_y"].value = float(mu_y)
                prob.param_dict["sqrt_Sigma_xi"].value = sqrt_Sigma_xi
                prob.param_dict["t_hat"].value = float(t_hat)
            elif bulk_shape_key == "box":
                prob.param_dict["mu_x"].value = mu_x
                prob.param_dict["mu_y"].value = float(mu_y)
                prob.param_dict["r_x"].value = r_x
                prob.param_dict["r_y"].value = float(r_y)
            elif bulk_shape_key == "ellipsoid_x_interval_y":
                prob.param_dict["mu_x"].value = mu_x
                prob.param_dict["mu_y"].value = float(mu_y)
                prob.param_dict["sqrt_Sigma_x"].value = sqrt_Sigma_x
                prob.param_dict["t_x"].value = float(t_x)
                prob.param_dict["r_y"].value = float(r_y)
            else:
                raise ValueError(f"Unknown bulk_shape_key='{bulk_shape_key}'.")

        def _rejection_sample_centre_in_bulk(
            *,
            centre: object,
            in_bulk_fn,
            n_accept: int,
            rng: np.random.Generator,
            max_total_draws: int = 5_000_000,
        ):
            n_accept = int(n_accept)
            if n_accept <= 0:
                raise ValueError("n_accept must be positive.")

            accepted_chunks = []
            total_draws = 0
            n_acc = 0
            while total_draws < int(max_total_draws):
                remaining = n_accept - n_acc
                n_draw = max(2 * remaining, 100)
                if n_draw <= 0:
                    break

                # For diagnostics purposes, we allow an empirical centres
                is_empirical = (isinstance(centre, np.ndarray))

                if is_empirical:
                    xi_pool = centre
                    xi_pool = np.asarray(xi_pool, dtype=float)
                    if xi_pool.ndim != 2 or xi_pool.shape[0] < 1 or xi_pool.shape[1] < 2:
                        raise ValueError(
                            "Empirical xi pool must be a non-empty 2D array of shape (N, d+1) with d>=1. "
                            f"Got shape {xi_pool.shape}."
                        )

                    draw_idx = rng.integers(0, xi_pool.shape[0], size=int(n_draw))
                    xi_draw = xi_pool[draw_idx]
                else:
                    xi_draw = centre.sample_xi(int(n_draw), rng)

                total_draws += int(n_draw)

                mask = in_bulk_fn(xi_draw)
                if np.any(mask):
                    accepted_chunks.append(np.asarray(xi_draw[mask], dtype=float))

                    n_acc = int(sum(chunk.shape[0] for chunk in accepted_chunks))
                    if n_acc >= n_accept:
                        break

            if len(accepted_chunks) == 0:
                raise RuntimeError("Rejection sampling failed: zero accepted samples.")

            xi_acc = np.concatenate(accepted_chunks, axis=0)
            if xi_acc.shape[0] < n_accept:
                raise RuntimeError(
                    "Rejection sampling failed: "
                    f"accepted {xi_acc.shape[0]} < n_accept={n_accept} after total_draws={total_draws}."
                )
            if total_draws > n_accept * 2:
                warnings.warn(
                    "Warning: rejection sampling efficiency low: "
                    f"accepted {xi_acc.shape[0]} samples from {total_draws} draws; n_accept = {n_accept}  ",
                    f"({100.0 * float(xi_acc.shape[0]) / float(total_draws):.2f}%).",
                    UserWarning,
                )
            xi_saa = xi_acc[:n_accept]
            accept_rate = float(xi_acc.shape[0]) / float(total_draws) if total_draws > 0 else 0.0
            return xi_saa, int(total_draws), float(accept_rate)

        # ---- Fit the frequentist centre (needed for lv_bas_ch sampling; also needed by some bulk shapes)
        centre = None
        centre_diag = {}

        X_tr = np.concatenate([X_train, X_select], axis=0)
        y_tr = np.concatenate([y_train, y_select], axis=0)

        if algorithm == "lv_bas_ch":
            centre = fit_copula_ridge_centre(X_tr, y_tr, ridge_alpha=0.005, copula_jitter=1e-6)
            centre_diag = {
                "ridge_coef_norm": float(np.linalg.norm(centre.w)),
                "ridge_intercept": float(centre.b),
                "ridge_sigma_y": float(centre.sigma_y),
            }

            if verbose and algorithm == "lv_bas_ch":
                print("Centre P_c (copula + ridge) fit diagnostics:")
                for k, v in centre_diag.items():
                    print(f"  {k:>25s} = {v}")

                if not np.isfinite(centre_diag["ridge_sigma_y"]) or centre_diag["ridge_sigma_y"] <= 0.0:
                    raise ValueError("Non-positive ridge residual sigma_y detected.")

        # ---- Bulk-set calibration (TRAIN stats + SELECT thresholding)
        mu = None
        Sigma = None
        sqrt_Sigma = None
        sqrt_Sigma_x = None
        t_x = None

        if bulk_shape_key == "ellipsoid":
            bulk = calibrate_dkw_ellipsoid_bulk_set(
                xi_train,
                xi_select,
                gamma=float(gamma_bulk),
                delta=float(dkw_delta),
                ridge=1e-8,
            )
            mu = np.asarray(bulk["mu"], dtype=float)
            Sigma = np.asarray(bulk["Sigma"], dtype=float)
            sqrt_Sigma = np.asarray(bulk["sqrt_Sigma"], dtype=float)
            t_hat = float(bulk["t_hat"])
            dkw_info = bulk["dkw_info"]

            mu_x = mu[:d]
            mu_y = float(mu[d])

            std_diag = np.sqrt(np.clip(np.diag(Sigma), 1e-12, np.inf))
            r_x = t_hat * std_diag[:d]
            r_y = float(t_hat * std_diag[d])

            def _in_bulk_xi(xi: np.ndarray) -> np.ndarray:
                xi = np.asarray(xi, dtype=float)
                return in_ellipsoid_bulk(xi, mu, sqrt_Sigma, float(t_hat))
        
        elif bulk_shape_key == "box":
            from .dataset import calibrate_dkw_box_xi_bulk_set

            bulk = calibrate_dkw_box_xi_bulk_set(
                xi_train=xi_train,
                xi_select=xi_select,
                gamma=float(gamma_bulk),
                delta=float(dkw_delta),
                scale_floor=1e-8,
            )
            mu = np.asarray(bulk["mu"], dtype=float).reshape(-1)
            if "q" in bulk:
                q = np.asarray(bulk["q"], dtype=float).reshape(-1)
            elif "scale" in bulk:
                q = np.asarray(bulk["scale"], dtype=float).reshape(-1)
            else:
                raise KeyError("box bulk calibration must return 'q' (or 'scale').")
            t_hat = float(bulk["t_hat"])
            dkw_info = bulk["dkw_info"]

            r = float(t_hat) * q
            mu_x = mu[:d]
            mu_y = float(mu[d])
            r_x = r[:d]
            r_y = float(r[d])

            def _in_bulk_xi(xi: np.ndarray) -> np.ndarray:
                xi = np.asarray(xi, dtype=float)
                return np.all(np.abs(xi - mu.reshape(1, -1)) <= r.reshape(1, -1) + 1e-12, axis=1)

        elif bulk_shape_key == "ellipsoid_x_interval_y":
            from .dataset import calibrate_dkw_ellipsoid_x_interval_y_bulk_set

            bulk = calibrate_dkw_ellipsoid_x_interval_y_bulk_set(
                xi_train=xi_train,
                xi_select=xi_select,
                gamma=float(gamma_bulk),
                delta=float(dkw_delta),
                ridge=1e-8,
            )
            mu_x = np.asarray(bulk["mu_x"], dtype=float).reshape(-1)
            mu_y = float(bulk["mu_y"])
            sqrt_Sigma_x = np.asarray(bulk["sqrt_Sigma_x"], dtype=float)
            t_x = float(bulk["t_x"])
            r_y = float(bulk["r_y"])
            dkw_info_x = bulk["dkw_info_x"]
            dkw_info_y = bulk["dkw_info_y"]
            dkw_info = {**dkw_info_x, **dkw_info_y}

            std_diag_x = np.sqrt(np.clip(np.sum(sqrt_Sigma_x * sqrt_Sigma_x, axis=1), 1e-12, np.inf))
            r_x = float(t_x) * std_diag_x
            t_hat = float(t_x)

            def _in_bulk_xi(xi: np.ndarray) -> np.ndarray:
                xi = np.asarray(xi, dtype=float)
                Xp = xi[:, :d]
                yp = xi[:, d]
                U = np.linalg.solve(sqrt_Sigma_x, (Xp - mu_x.reshape(1, -1)).T).T
                in_x = np.linalg.norm(U, axis=1) <= float(t_x) + 1e-12
                in_y = np.abs(yp - float(mu_y)) <= float(r_y) + 1e-12
                return in_x & in_y

            r_y = float(r_y)

        else:
            raise ValueError(f"Unsupported bulk_shape='{bulk_shape}' (normalised='{bulk_shape_key}').")

        # Diagnostics: bulk coverage on each split (shape-dependent membership)
        train_bulk_rate = float(np.mean(_in_bulk_xi(xi_train)))
        select_bulk_rate = float(np.mean(_in_bulk_xi(xi_select)))

        if verbose:
            print("California Housing DKW bulk calibration:")
            print(f"  bulk_shape = {bulk_shape_key}")
            print(f"  gamma_bulk = {gamma_bulk}   delta_DKW = {dkw_delta}")
            for k, v in dkw_info.items():
                print(f"  {k:>25s} = {v}")
            print(f"  t_hat = {t_hat}")
            print(f"  TRAIN in-bulk rate  = {train_bulk_rate:.4f}")
            print(f"  SELECT in-bulk rate = {select_bulk_rate:.4f}")

            if not np.isfinite(float(t_hat)) or float(t_hat) < 0.0:
                raise ValueError(f"Invalid t_hat={t_hat}")

            if train_bulk_rate < 0.5:
                warnings.warn(
                    "TRAIN in-bulk rate is unexpectedly low (<0.5). "
                    "This suggests a pathological bulk calibration or split issue.",
                    UserWarning,
                )

        posterior_time = (datetime.now() - posterior_start).total_seconds()

        # ========================
        # 3) Likelihood stage (SAA sampling / data for optimisation)
        # ========================
        likelihood_time = 0.0
        setup_time = 0.0
        solve_time = 0.0
        # --- Optional validation-calibration bookkeeping
        validation_time = 0.0
        val_mae_raw = np.nan

        # Hyperparameters actually used (may be tuned on VAL)
        epsilon_chosen = float(epsilon)          # used by lv_bas_ch and cvar_lad
        wass_rho_chosen = np.nan                 # used by wass_lad
        ridge_lambda_chosen = np.nan            # used by erm_ridge
        solver_failed = False
        val_mae_std = np.nan  # CV fold-to-fold std for the chosen hyperparameter (Option A)

        # Geo-block CV metadata (Option A)
        cv_n_folds = np.nan
        cv_n_blocks = np.nan
        cv_lon_bins = np.nan
        cv_lat_bins = np.nan

        def _ca_housing_geo_block_cv_select(
            *,
            algorithm: str,
            X_east: np.ndarray,
            y_east: np.ndarray,
            d: int,
            gamma_bulk: float,
            num_likelihood_samples: int,
            seed: int,
        ) -> dict:
            """
            Geo-block CV inside the EAST region (no WEST leakage).

            - Build geographic blocks by quantile-binning lon/lat (first two columns).
            - Randomly assign blocks to a SMALL number of folds (default 3).
            - For each candidate hyperparameter, fit on fold-train and evaluate MAE on fold-val.
            - Return the hyperparameter with the lowest mean fold MAE.
            """
            X_east = np.asarray(X_east, dtype=float)
            y_east = np.asarray(y_east, dtype=float).reshape(-1)

            if X_east.ndim != 2:
                raise ValueError("X_east must be 2D.")
            if X_east.shape[0] != y_east.shape[0]:
                raise ValueError("X_east and y_east must have the same number of rows.")
            if X_east.shape[1] != int(d):
                raise ValueError(f"X_east must have d={int(d)} columns.")

            # Keep CV inexpensive
            cv_n_folds_target = 3
            cv_lon_bins_local = 6
            cv_lat_bins_local = 6

            cv_cfg = _california_housing_geo_block_folds(
                X=X_east,
                n_folds=int(cv_n_folds_target),
                seed=int(seed),
                n_lon_bins=int(cv_lon_bins_local),
                n_lat_bins=int(cv_lat_bins_local),
            )
            folds = cv_cfg["folds"]
            n_folds_eff = int(cv_cfg["n_folds"])

            # Candidate grids
            eps_grid = np.array([0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20], dtype=float)
            rho_grid = np.array([0.05, 0.10, 0.50, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=float)

            # Grid around a mild size-based heuristic
            #lambda0 = float(1.0 / np.sqrt(max(1, int(X_east.shape[0]))))
            ridge_grid = np.array([1e-2, 1e-1, 0.5, 1.0, 5, 1e1, 50, 1e2], dtype=float)
            chi2_grid = np.asarray(
                getattr(_constants, "CALIFORNIA_HOUSING_CHI2_SET", [0.0, 0.05, 0.10, 0.20, 0.50, 1.0, 5.0, 10.0]),
                dtype=float,
            )
            kl_grid = np.asarray(
                getattr(_constants, "CALIFORNIA_HOUSING_KL_SET", [0.0, 0.01, 0.02, 0.05, 0.10, 0.50, 1.0, 5.0]),
                dtype=float,
            )
            if algorithm in {"lv_bas_ch", "cvar_lad"}:
                grid = eps_grid
            elif algorithm == "erm_ridge":
                grid = ridge_grid
            elif algorithm in {"wass_lad", "or_wdro"}:
                grid = rho_grid
            elif algorithm == "chi2_lad":
                grid = chi2_grid
            elif algorithm == "kl_lad":
                grid = kl_grid
            else:
                raise ValueError(f"Unsupported algorithm for geo-block CV: {algorithm}")

            score_mat = np.full((n_folds_eff, int(grid.size)), np.nan, dtype=float)

            start = datetime.now()
            # Independent RNG for CV (do not disturb the main replication RNG)
            base_seed = int(seed) + 123456

            # MOSEK: keep each solve single-threaded to avoid oversubscription under joblib.

            for fold_id, fold in enumerate(folds):
                tr_idx = fold["train_idx"]
                va_idx = fold["val_idx"]

                X_tr = np.asarray(X_east[tr_idx], dtype=float)
                y_tr = np.asarray(y_east[tr_idx], dtype=float).reshape(-1)
                X_va = np.asarray(X_east[va_idx], dtype=float)
                y_va = np.asarray(y_east[va_idx], dtype=float).reshape(-1)

                if algorithm == "cvar_lad":
                    n_tr = int(X_tr.shape[0])
                    if n_tr < 5:
                        continue

                    prob = get_cvar_lad_problem(dim=d, num_samples=n_tr)
                    prob.param_dict["X_train"].value = X_tr
                    prob.param_dict["y_train"].value = y_tr

                    for j, eps in enumerate(grid):
                        tail_mass = float(eps)
                        if tail_mass <= 0.0 or tail_mass >= 1.0:
                            continue

                        prob.param_dict["cvar_coeff"].value = float(1.0 / (tail_mass * n_tr))

                        try:
                            prob.solve(
                                solver=cp.CLARABEL,
verbose=verbose,
                                warm_start=True,
                            )
                        except Exception:
                            continue

                        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                            continue

                        sol = prob.var_dict["x"].value
                        if sol is None:
                            continue
                        sol = np.asarray(sol, dtype=float).reshape(-1)
                        if sol.size != d + 1 or np.any(~np.isfinite(sol)):
                            continue

                        w_hat = sol[:d]
                        b_hat = float(sol[d])
                        score_mat[fold_id, j] = float(np.mean(np.abs(y_va - (X_va @ w_hat + b_hat))))
                elif algorithm == "erm_ridge":
                    n_tr = int(X_tr.shape[0])
                    if n_tr < 5:
                        continue

                    # Closed-form ridge: (1/n)||y - (Xw+b)||^2 + lam||w||^2, intercept unpenalised.
                    X_aug = np.concatenate([X_tr, np.ones((n_tr, 1), dtype=float)], axis=1)  # (n_tr, d+1)
                    XtX = (X_aug.T @ X_aug) / float(n_tr)
                    Xty = (X_aug.T @ y_tr) / float(n_tr)

                    P = np.eye(d + 1, dtype=float)
                    P[-1, -1] = 0.0  # do not penalise intercept

                    for j, lam in enumerate(grid):
                        lam = float(lam)
                        if lam < 0.0:
                            continue

                        A = XtX + lam * P
                        try:
                            beta = np.linalg.solve(A, Xty)
                        except np.linalg.LinAlgError:
                            beta = np.linalg.lstsq(A, Xty, rcond=None)[0]

                        w_hat = beta[:d]
                        b_hat = float(beta[d])

                        score_mat[fold_id, j] = float(np.mean(np.abs(y_va - (X_va @ w_hat + b_hat))))
                elif algorithm == "wass_lad":
                    n_tr = int(X_tr.shape[0])
                    if n_tr < 5:
                        continue

                    prob = get_wass_lad_problem(dim=d, num_samples=n_tr)
                    prob.param_dict["X_train"].value = X_tr
                    prob.param_dict["y_train"].value = y_tr
                    prob.param_dict["y_transport_coeff"].value = float(np.std(y_tr, ddof=0))
                    for j, rho in enumerate(grid):
                        rho = float(rho)
                        if rho < 0.0:
                            continue

                        prob.param_dict["wass_rho"].value = rho

                        try:
                            prob.solve(
                                solver=cp.CLARABEL,
verbose=verbose,
                                warm_start=True,
                            )
                        except Exception:
                            continue

                        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                            continue

                        sol = prob.var_dict["x"].value
                        if sol is None:
                            continue
                        sol = np.asarray(sol, dtype=float).reshape(-1)
                        if sol.size != d + 1 or np.any(~np.isfinite(sol)):
                            continue

                        w_hat = sol[:d]
                        b_hat = float(sol[d])
                        score_mat[fold_id, j] = float(np.mean(np.abs(y_va - (X_va @ w_hat + b_hat))))
                elif algorithm == "or_wdro":
                    n_tr = int(X_tr.shape[0])
                    if n_tr < 5:
                        continue

                    try:
                        from dro.linear_model.or_wasserstein_dro import ORWDRO
                    except Exception as e:
                        raise ImportError(
                            "OR-WDRO baseline requires the 'dro' package. "
                            "Install it with: pip install dro"
                        ) from e

                    # Fixed OR-WDRO settings (keep eta=0 by default to recover standard WDRO)
                    or_eta = float(getattr(_constants, "CALIFORNIA_HOUSING_OR_WDRO_ETA", 0.0))
                    dual_norm = int(getattr(_constants, "CALIFORNIA_HOUSING_OR_WDRO_DUAL_NORM", 2))

                    if not (0.0 <= or_eta <= 0.5):
                        raise ValueError(f"OR-WDRO requires eta in [0, 0.5], got {or_eta}.")
                    if dual_norm not in (1, 2):
                        raise ValueError(f"OR-WDRO dual_norm must be 1 or 2, got {dual_norm}.")

                    for j, eps in enumerate(grid):
                        eps = float(eps)
                        if eps < 0.0:
                            continue

                        model = ORWDRO(
                            input_dim=int(d),
                            model_type="lad",
                            solver=cp.CLARABEL,
                            eps=float(eps),
                            eta=float(or_eta),
                            dual_norm=int(dual_norm),
                        )

                        try:
                            model.fit(X_tr, y_tr)
                            y_hat = model.predict(X_va)
                        except Exception:
                            continue

                        y_hat = np.asarray(y_hat, dtype=float).reshape(-1)
                        if y_hat.shape[0] != y_va.shape[0] or np.any(~np.isfinite(y_hat)):
                            continue

                        score_mat[fold_id, j] = float(np.mean(np.abs(y_va - y_hat)))
                elif algorithm == "chi2_lad":
                    n_tr = int(X_tr.shape[0])
                    if n_tr < 5:
                        continue

                    try:
                        from dro.linear_model.chi2_dro import Chi2DRO
                    except Exception as e:
                        raise ImportError(
                            "Chi2-DRO baseline requires the 'dro' package. "
                            "Install it with: pip install dro"
                        ) from e

                    for j, eps in enumerate(grid):
                        eps = float(eps)
                        if eps < 0.0:
                            continue

                        model = Chi2DRO(
                            input_dim=int(d),
                            model_type="lad",
                            solver=cp.CLARABEL,
                        )

                        try:
                            model.update({"eps": float(eps)})
                            model.fit(X_tr, y_tr)
                            y_hat = model.predict(X_va)
                        except Exception:
                            continue

                        y_hat = np.asarray(y_hat, dtype=float).reshape(-1)
                        if y_hat.shape[0] != y_va.shape[0] or np.any(~np.isfinite(y_hat)):
                            continue

                        score_mat[fold_id, j] = float(np.mean(np.abs(y_va - y_hat)))
                elif algorithm == "kl_lad":
                    n_tr = int(X_tr.shape[0])
                    if n_tr < 5:
                        continue

                    try:
                        from dro.linear_model.kl_dro import KLDRO
                    except Exception as e:
                        raise ImportError(
                            "KL-DRO baseline requires the 'dro' package. "
                            "Install it with: pip install dro"
                        ) from e

                    for j, eps in enumerate(grid):
                        eps = float(eps)
                        if eps < 0.0:
                            continue

                        model = KLDRO(
                            input_dim=int(d),
                            model_type="lad",
                            solver=cp.CLARABEL,
                        )

                        try:
                            model.update({"eps": float(eps)})
                            model.fit(X_tr, y_tr)
                            y_hat = model.predict(X_va)
                        except Exception:
                            continue

                        y_hat = np.asarray(y_hat, dtype=float).reshape(-1)
                        if y_hat.shape[0] != y_va.shape[0] or np.any(~np.isfinite(y_hat)):
                            continue

                        score_mat[fold_id, j] = float(np.mean(np.abs(y_va - y_hat)))
                elif algorithm == "lv_bas_ch":
                    n_tr = int(X_tr.shape[0])
                    if n_tr < 10:
                        continue

                    rng_fold = np.random.default_rng(seed=base_seed + 1000 * int(fold_id))
                    perm = rng_fold.permutation(n_tr)

                    # Inner TRAIN/SELECT split to calibrate the DKW bulk set
                    n_sel = int(max(2, np.floor(0.2 * n_tr)))
                    n_sel = int(min(n_sel, n_tr - 2))
                    sel_loc = perm[:n_sel]
                    tr_loc = perm[n_sel:]

                    X_tr_in = X_tr[tr_loc]
                    y_tr_in = y_tr[tr_loc]
                    X_sel_in = X_tr[sel_loc]
                    y_sel_in = y_tr[sel_loc]

                    xi_tr_in = np.concatenate([X_tr_in, y_tr_in.reshape(-1, 1)], axis=1)
                    xi_sel_in = np.concatenate([X_sel_in, y_sel_in.reshape(-1, 1)], axis=1)

                    # Fit centre on inner TRAIN+SELECT (needed for LV sampling; also needed by some bulk shapes)
                    X_tr_in_full = np.concatenate([X_tr_in, X_sel_in], axis=0)
                    y_tr_in_full = np.concatenate([y_tr_in, y_sel_in], axis=0)
                    try:
                        centre_cv = fit_copula_ridge_centre(X_tr_in_full, y_tr_in_full, ridge_alpha=0.005, copula_jitter=1e-6)
                    except Exception:
                        continue

                    # Calibrate fold-specific bulk set based on outer `bulk_shape_key`
                    try:
                        if bulk_shape_key == "ellipsoid":
                            bulk_cv = calibrate_dkw_ellipsoid_bulk_set(
                                xi_tr_in,
                                xi_sel_in,
                                gamma=float(gamma_bulk),
                                delta=float(0.05),
                                ridge=1e-8,
                            )
                            mu_cv = np.asarray(bulk_cv["mu"], dtype=float)
                            sqrt_Sigma_cv = np.asarray(bulk_cv["sqrt_Sigma"], dtype=float)
                            t_hat_cv = float(bulk_cv["t_hat"])

                            mu_x_cv = mu_cv[:d]
                            mu_y_cv = float(mu_cv[d])

                            def _in_bulk_cv(xi: np.ndarray) -> np.ndarray:
                                xi = np.asarray(xi, dtype=float)
                                return in_ellipsoid_bulk(xi, mu_cv, sqrt_Sigma_cv, float(t_hat_cv))

                            # Placeholders (not used by the ellipsoid LV-BAS setter)
                            r_x_cv = np.zeros(d, dtype=float)
                            r_y_cv = 0.0
                            sqrt_Sigma_x_cv = None
                            t_x_cv = None
                            r_c_cv = None
                            w_c_cv = None
                            b_c_cv = None
                      
                        elif bulk_shape_key == "box":
                            from .dataset import calibrate_dkw_box_xi_bulk_set

                            bulk_cv = calibrate_dkw_box_xi_bulk_set(
                                xi_train=xi_tr_in,
                                xi_select=xi_sel_in,
                                gamma=float(gamma_bulk),
                                delta=float(0.05),
                                scale_floor=1e-8,
                            )
                            mu_cv = np.asarray(bulk_cv["mu"], dtype=float).reshape(-1)
                            if "q" in bulk_cv:
                                q_cv = np.asarray(bulk_cv["q"], dtype=float).reshape(-1)
                            elif "scale" in bulk_cv:
                                q_cv = np.asarray(bulk_cv["scale"], dtype=float).reshape(-1)
                            else:
                                raise KeyError("box bulk calibration must return 'q' (or 'scale').")
                            t_hat_cv = float(bulk_cv["t_hat"])
                            r_cv = float(t_hat_cv) * q_cv

                            mu_x_cv = mu_cv[:d]
                            mu_y_cv = float(mu_cv[d])
                            r_x_cv = r_cv[:d]
                            r_y_cv = float(r_cv[d])

                            def _in_bulk_cv(xi: np.ndarray) -> np.ndarray:
                                xi = np.asarray(xi, dtype=float)
                                return np.all(np.abs(xi - mu_cv.reshape(1, -1)) <= r_cv.reshape(1, -1) + 1e-12, axis=1)

                            sqrt_Sigma_cv = None
                            sqrt_Sigma_x_cv = None
                            t_x_cv = None
                            r_c_cv = None
                            w_c_cv = None
                            b_c_cv = None

                        elif bulk_shape_key == "ellipsoid_x_interval_y":
                            from .dataset import calibrate_dkw_ellipsoid_x_interval_y_bulk_set

                            bulk_cv = calibrate_dkw_ellipsoid_x_interval_y_bulk_set(
                                xi_train=xi_tr_in,
                                xi_select=xi_sel_in,
                                gamma=float(gamma_bulk),
                                delta=float(0.05),
                                ridge=1e-8,
                            )
                            mu_x_cv = np.asarray(bulk_cv["mu_x"], dtype=float).reshape(-1)
                            mu_y_cv = float(bulk_cv["mu_y"])
                            sqrt_Sigma_x_cv = np.asarray(bulk_cv["sqrt_Sigma_x"], dtype=float)
                            t_x_cv = float(bulk_cv["t_x"])
                            r_y_cv = float(bulk_cv["r_y"])
                            t_hat_cv = float(t_x_cv)

                            std_diag_x_cv = np.sqrt(np.clip(np.sum(sqrt_Sigma_x_cv * sqrt_Sigma_x_cv, axis=1), 1e-12, np.inf))
                            r_x_cv = float(t_x_cv) * std_diag_x_cv

                            def _in_bulk_cv(xi: np.ndarray) -> np.ndarray:
                                xi = np.asarray(xi, dtype=float)
                                Xp = xi[:, :d]
                                yp = xi[:, d]
                                U = np.linalg.solve(sqrt_Sigma_x_cv, (Xp - mu_x_cv.reshape(1, -1)).T).T
                                in_x = np.linalg.norm(U, axis=1) <= float(t_x_cv) + 1e-12
                                in_y = np.abs(yp - float(mu_y_cv)) <= float(r_y_cv) + 1e-12
                                return in_x & in_y

                            sqrt_Sigma_cv = None
                            r_c_cv = None
                            w_c_cv = None
                            b_c_cv = None

                        else:
                            raise ValueError(f"Unsupported bulk_shape_key='{bulk_shape_key}' in geo-block CV.")
                    except Exception:
                        continue

                    # Smaller SAA inside CV to keep it cheap
                    m_cv = int(min(int(num_likelihood_samples), 4000))
                    m_cv = int(max(200, m_cv))

                    try:
                        if bulk_shape_key == "ellipsoid":
                            xi_saa_cv, _, _ = rejection_sample_centre_in_ellipsoid_bulk(
                                centre=centre_cv,
                                mu=mu_cv,
                                sqrt_Sigma=sqrt_Sigma_cv,
                                t_hat=t_hat_cv,
                                n_accept=int(m_cv),
                                rng=rng_fold,
                            )
                        else:
                            xi_saa_cv, _, _ = _rejection_sample_centre_in_bulk(
                                centre=centre_cv,
                                in_bulk_fn=_in_bulk_cv,
                                n_accept=int(m_cv),
                                rng=rng_fold,
                            )
                    except Exception:
                        continue

                    X_saa = np.asarray(xi_saa_cv[:, :d], dtype=float)
                    y_saa = np.asarray(xi_saa_cv[:, d], dtype=float).reshape(-1)

                    prob = _make_lv_bas_problem(dim=d, num_samples=m_cv)
                    prob.param_dict["X_saa"].value = X_saa
                    prob.param_dict["y_saa"].value = y_saa
                    _set_lv_bas_bulk_params(
                        prob,
                        mu_x=np.asarray(mu_x_cv, dtype=float),
                        mu_y=float(mu_y_cv),
                        r_x=np.asarray(r_x_cv, dtype=float).reshape(-1),
                        r_y=float(r_y_cv),
                        t_hat=float(t_hat_cv),
                        sqrt_Sigma_xi=sqrt_Sigma_cv,
                        sqrt_Sigma_x=sqrt_Sigma_x_cv,
                        t_x=t_x_cv,
                    )

                    for j, eps in enumerate(grid):
                        eps_cand = float(eps)
                        if eps_cand < 0.0 or eps_cand >= 1.0:
                            continue

                        prob.param_dict["epsilon"].value = float(eps_cand)
                        prob.param_dict["one_minus_epsilon"].value = float(1.0 - float(eps_cand))

                        try:
                            prob.solve(
                                solver=cp.CLARABEL,
verbose=verbose,
                                warm_start=True,
                            )
                        except Exception:
                            continue

                        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                            continue

                        sol = prob.var_dict["x"].value
                        if sol is None:
                            continue
                        sol = np.asarray(sol, dtype=float).reshape(-1)
                        if sol.size != d + 1 or np.any(~np.isfinite(sol)):
                            continue

                        w_hat = sol[:d]
                        b_hat = float(sol[d])
                        score_mat[fold_id, j] = float(np.mean(np.abs(y_va - (X_va @ w_hat + b_hat))))

            mean_scores = np.nanmean(score_mat, axis=0)
            if not np.any(np.isfinite(mean_scores)):
                raise RuntimeError(f"Geo-block CV failed for {algorithm}: no finite fold scores.")

            best_j = int(np.nanargmin(mean_scores))
            best_param = float(grid[best_j])
            best_mean = float(mean_scores[best_j])
            best_std = float(np.nanstd(score_mat[:, best_j], ddof=1)) if n_folds_eff > 1 else float(
                np.nanstd(score_mat[:, best_j])
            )

            return {
                "validation_time": float((datetime.now() - start).total_seconds()),
                "cv_mae_mean": float(best_mean),
                "cv_mae_std": float(best_std),
                "best_param": float(best_param),
                "cv_n_folds": float(cv_cfg["n_folds"]),
                "cv_n_blocks": float(cv_cfg["n_blocks"]),
                "cv_lon_bins": float(cv_cfg["n_lon_bins"]),
                "cv_lat_bins": float(cv_cfg["n_lat_bins"]),
            }


        if algorithm == "lv_bas_ch":
            # Rejection sample from P_c conditioned on Xi0
            likelihood_start = datetime.now()

            if bulk_shape_key == "ellipsoid":
                xi_saa, total_draws, accept_rate = rejection_sample_centre_in_ellipsoid_bulk(
                    centre=centre,
                    mu=mu,
                    sqrt_Sigma=sqrt_Sigma,
                    t_hat=t_hat,
                    n_accept=int(num_likelihood_samples),
                    rng=generator,
                )
            else:
                xi_saa, total_draws, accept_rate = _rejection_sample_centre_in_bulk(
                    centre=centre,
                    in_bulk_fn=_in_bulk_xi,
                    n_accept=int(num_likelihood_samples),
                    rng=generator,
                )

            likelihood_time = (datetime.now() - likelihood_start).total_seconds()

            X_saa = xi_saa[:, :d]
            y_saa = xi_saa[:, d].reshape(-1)

            if verbose:
                print("LV-BAS-CH rejection sampling diagnostics:")
                print(f"  accepted = {len(y_saa)}  total_draws = {total_draws}  accept_rate = {accept_rate:.6f}")
                if accept_rate < 1e-4:
                    warnings.warn(
                        "Rejection acceptance rate is extremely low (<1e-4). "
                        "This will make the experiment slow and may indicate bulk miscalibration.",
                        UserWarning,
                    )
                # Sanity: accepted points are in bulk
                saa_in_bulk = float(np.mean(_in_bulk_xi(xi_saa)))
                print(f"  SAA in-bulk rate (should be 1.0): {saa_in_bulk:.6f}")

            # ========================
            # 4) Optimisation stage (solve the chosen problem)
            # ========================
            # Ensure the LV-BAS problem matches the chosen bulk shape (rebuild if needed).
            if bulk_shape_key != "ellipsoid":
                problem = _make_lv_bas_problem(dim=d, num_samples=int(num_likelihood_samples))

            # Fixed parameters (do not depend on epsilon)
            problem.param_dict["X_saa"].value = X_saa
            problem.param_dict["y_saa"].value = y_saa

            _set_lv_bas_bulk_params(
                problem,
                mu_x=np.asarray(mu_x, dtype=float),
                mu_y=float(mu_y),
                r_x=np.asarray(r_x, dtype=float).reshape(-1),
                r_y=float(r_y),
                t_hat=float(t_hat),
                sqrt_Sigma_xi=sqrt_Sigma,
                sqrt_Sigma_x=sqrt_Sigma_x,
                t_x=t_x,
            )

            if calibrate_on_validation:
                # Option A: geo-block CV within EAST to choose epsilon, then fit once on full EAST.
                X_east = np.concatenate([X_train, X_select, X_val], axis=0)
                y_east = np.concatenate([y_train, y_select, y_val], axis=0)

                cv_info = _ca_housing_geo_block_cv_select(
                    algorithm="lv_bas_ch",
                    X_east=X_east,
                    y_east=y_east,
                    d=d,
                    gamma_bulk=float(gamma_bulk),
                    num_likelihood_samples=int(num_likelihood_samples),
                    seed=int(replication),
                )

                validation_time = float(cv_info["validation_time"])
                val_mae_raw = float(cv_info["cv_mae_mean"])
                val_mae_std = float(cv_info["cv_mae_std"])
                epsilon_chosen = float(cv_info["best_param"])

                cv_n_folds = float(cv_info["cv_n_folds"])
                cv_n_blocks = float(cv_info["cv_n_blocks"])
                cv_lon_bins = float(cv_info["cv_lon_bins"])
                cv_lat_bins = float(cv_info["cv_lat_bins"])

                problem.param_dict["epsilon"].value = float(epsilon_chosen)
                problem.param_dict["one_minus_epsilon"].value = float(1.0 - float(epsilon_chosen))

                solve_start = datetime.now()
                problem.solve(
                    solver=cp.CLARABEL,
                    verbose=verbose,
                )
                solve_time = (datetime.now() - solve_start).total_seconds()
                setup_time = problem.solver_stats.setup_time

                solution = problem.var_dict["x"].value
            else:
                problem.param_dict["epsilon"].value = float(epsilon)
                problem.param_dict["one_minus_epsilon"].value = float(1.0 - float(epsilon))

                solve_start = datetime.now()
                problem.solve(
                    solver=cp.CLARABEL,
                    verbose=verbose,
                )
                solve_time = (datetime.now() - solve_start).total_seconds()
                setup_time = problem.solver_stats.setup_time

                solution = problem.var_dict["x"].value

            pc_mass_est = float(accept_rate)
            rej_total_draws = int(total_draws)

        elif algorithm == "erm_lad":
            # ========================
            # 4) Optimisation stage (solve the chosen problem)
            # ========================
            # Use TRAIN + SELECT for ERM-LAD
            if problem is None:
                n_tr = np.concatenate([X_train, X_select], axis=0).shape[0]
                problem = get_erm_lad_problem(dim=d, num_samples=n_tr)
            problem.param_dict["X_train"].value = np.concatenate([X_train, X_select], axis=0)
            problem.param_dict["y_train"].value = np.concatenate([y_train, y_select], axis=0)

            solve_start = datetime.now()
            problem.solve(
                solver=cp.CLARABEL,
                verbose=verbose,
            )
            solve_time = (datetime.now() - solve_start).total_seconds()
            setup_time = problem.solver_stats.setup_time

            solution = problem.var_dict["x"].value
            pc_mass_est = np.nan
            rej_total_draws = 0

        elif algorithm == "erm_ridge":

            def _ridge_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
                X = np.asarray(X, dtype=float)
                y = np.asarray(y, dtype=float).reshape(-1)
                n = int(X.shape[0])
                if n <= 0:
                    raise ValueError("Ridge: empty training set.")
                if y.shape[0] != n:
                    raise ValueError("Ridge: X/y size mismatch.")

                X_aug = np.concatenate([X, np.ones((n, 1), dtype=float)], axis=1)  # (n, d+1)

                XtX = (X_aug.T @ X_aug) / float(n)
                Xty = (X_aug.T @ y) / float(n)

                P = np.eye(d + 1, dtype=float)
                P[-1, -1] = 0.0  # do not penalise intercept

                A = XtX + float(lam) * P
                try:
                    beta = np.linalg.solve(A, Xty)
                except np.linalg.LinAlgError:
                    beta = np.linalg.lstsq(A, Xty, rcond=None)[0]
                return np.asarray(beta, dtype=float).reshape(-1)

            if calibrate_on_validation:
                # Option A: geo-block CV within EAST to choose ridge_lambda, then refit on full EAST.
                X_east = np.concatenate([X_train, X_select, X_val], axis=0)
                y_east = np.concatenate([y_train, y_select, y_val], axis=0)

                cv_info = _ca_housing_geo_block_cv_select(
                    algorithm="erm_ridge",
                    X_east=X_east,
                    y_east=y_east,
                    d=d,
                    gamma_bulk=float(gamma_bulk),
                    num_likelihood_samples=int(num_likelihood_samples),
                    seed=int(replication),
                )

                validation_time = float(cv_info["validation_time"])
                val_mae_raw = float(cv_info["cv_mae_mean"])
                val_mae_std = float(cv_info["cv_mae_std"])
                ridge_lambda_chosen = float(cv_info["best_param"])

                cv_n_folds = float(cv_info["cv_n_folds"])
                cv_n_blocks = float(cv_info["cv_n_blocks"])
                cv_lon_bins = float(cv_info["cv_lon_bins"])
                cv_lat_bins = float(cv_info["cv_lat_bins"])

                solve_start = datetime.now()
                solution = _ridge_closed_form(X_east, y_east, ridge_lambda_chosen)
                solve_time = (datetime.now() - solve_start).total_seconds()
                setup_time = 0.0
            else:
                # No CV: fit on TRAIN+SELECT with a mild size-based heuristic.
                X_tr = np.concatenate([X_train, X_select], axis=0)
                y_tr = np.concatenate([y_train, y_select], axis=0)

                ridge_lambda_chosen = float(epsilon)

                solve_start = datetime.now()
                solution = _ridge_closed_form(X_tr, y_tr, ridge_lambda_chosen)
                solve_time = (datetime.now() - solve_start).total_seconds()
                setup_time = 0.0

            # Ridge does not use LV centre mass / rejection sampling
            pc_mass_est = np.nan
            rej_total_draws = 0
        elif algorithm == "wass_lad":

            X_tr = np.concatenate([X_train, X_select], axis=0)
            y_tr = np.concatenate([y_train, y_select], axis=0)
            n_tr = int(X_tr.shape[0])

            # Safety: ensure the passed-in problem is the Wasserstein one.
            if problem is None or "wass_rho" not in problem.param_dict:
                warnings.warn(
                    "[california_housing] Received a 'wass_lad' problem without parameter 'wass_rho'. "
                    "Rebuilding get_wass_lad_problem so the Wasserstein radius is applied.",
                    UserWarning,
                )
                problem = get_wass_lad_problem(dim=d, num_samples=n_tr)

            problem.param_dict["X_train"].value = X_tr
            problem.param_dict["y_train"].value = y_tr
            problem.param_dict["y_transport_coeff"].value = float(np.std(y_tr, ddof=0))
            # MOSEK: keep single-threaded under joblib.

            if calibrate_on_validation:
                X_east = np.concatenate([X_train, X_select, X_val], axis=0)
                y_east = np.concatenate([y_train, y_select, y_val], axis=0)

                cv_info = _ca_housing_geo_block_cv_select(
                    algorithm="wass_lad",
                    X_east=X_east,
                    y_east=y_east,
                    d=d,
                    gamma_bulk=float(gamma_bulk),
                    num_likelihood_samples=int(num_likelihood_samples),
                    seed=int(replication),
                )

                validation_time = float(cv_info["validation_time"])
                val_mae_raw = float(cv_info["cv_mae_mean"])
                val_mae_std = float(cv_info["cv_mae_std"])
                wass_rho_chosen = float(cv_info["best_param"])

                cv_n_folds = float(cv_info["cv_n_folds"])
                cv_n_blocks = float(cv_info["cv_n_blocks"])
                cv_lon_bins = float(cv_info["cv_lon_bins"])
                cv_lat_bins = float(cv_info["cv_lat_bins"])
            else:
                # Interpret the run-level `epsilon` as the Wasserstein radius when not cross-validating.
                wass_rho_chosen = float(epsilon)

            problem.param_dict["wass_rho"].value = float(wass_rho_chosen)

            solve_start = datetime.now()
            problem.solve(
                solver=cp.CLARABEL,
                verbose=verbose,
                warm_start=True,
            )
            solve_time = (datetime.now() - solve_start).total_seconds()
            setup_time = problem.solver_stats.setup_time

            solution = problem.var_dict["x"].value
            pc_mass_est = np.nan
            rej_total_draws = 0

        elif algorithm == "chi2_lad":
            # ========================
            # 4) Optimisation stage (Chi2-DRO baseline via python-dro)
            # ========================
            X_tr = np.concatenate([X_train, X_select], axis=0)
            y_tr = np.concatenate([y_train, y_select], axis=0).reshape(-1)

            if calibrate_on_validation:
                X_east = np.concatenate([X_train, X_select, X_val], axis=0)
                y_east = np.concatenate([y_train, y_select, y_val], axis=0)

                cv_info = _ca_housing_geo_block_cv_select(
                    algorithm="chi2_lad",
                    X_east=X_east,
                    y_east=y_east,
                    d=d,
                    gamma_bulk=float(gamma_bulk),
                    num_likelihood_samples=int(num_likelihood_samples),
                    seed=int(replication),
                )

                validation_time = float(cv_info["validation_time"])
                val_mae_raw = float(cv_info["cv_mae_mean"])
                val_mae_std = float(cv_info["cv_mae_std"])
                epsilon_chosen = float(cv_info["best_param"])

                cv_n_folds = float(cv_info["cv_n_folds"])
                cv_n_blocks = float(cv_info["cv_n_blocks"])
                cv_lon_bins = float(cv_info["cv_lon_bins"])
                cv_lat_bins = float(cv_info["cv_lat_bins"])
            else:
                epsilon_chosen = float(epsilon)

            try:
                from dro.linear_model.chi2_dro import Chi2DRO
            except Exception as e:
                raise ImportError(
                    "Chi2-DRO baseline requires the 'dro' package. "
                    "Install it with: pip install dro"
                ) from e

            solve_start = datetime.now()
            model = Chi2DRO(
                input_dim=int(d),
                model_type="lad",
                solver=cp.CLARABEL,
            )
            model.update({"eps": float(epsilon_chosen)})
            fit_out = model.fit(np.asarray(X_tr, dtype=float), np.asarray(y_tr, dtype=float).reshape(-1))
            solve_time = (datetime.now() - solve_start).total_seconds()
            setup_time = 0.0

            w_hat = None
            b_hat = None
            if isinstance(fit_out, dict):
                if "theta" in fit_out:
                    w_hat = np.asarray(fit_out["theta"], dtype=float).reshape(-1)
                elif "w" in fit_out:
                    w_hat = np.asarray(fit_out["w"], dtype=float).reshape(-1)

                if "b" in fit_out:
                    b_hat = float(np.asarray(fit_out["b"], dtype=float).reshape(-1)[0])
                elif "bias" in fit_out:
                    b_hat = float(np.asarray(fit_out["bias"], dtype=float).reshape(-1)[0])

            if w_hat is None:
                for attr in ("theta", "coef_", "w"):
                    if hasattr(model, attr):
                        w_hat = np.asarray(getattr(model, attr), dtype=float).reshape(-1)
                        break

            if b_hat is None:
                for attr in ("b", "bias", "intercept_", "intercept"):
                    if hasattr(model, attr):
                        b_hat = float(np.asarray(getattr(model, attr), dtype=float).reshape(-1)[0])
                        break

            if w_hat is None or b_hat is None:
                raise RuntimeError(
                    "Chi2-DRO fit did not expose parameters. Expected keys 'theta' and 'b' (or 'bias') "
                    "from model.fit(...) or corresponding attributes on the fitted model."
                )
            if w_hat.size != int(d) or not np.isfinite(b_hat):
                raise RuntimeError(f"Chi2-DRO returned invalid parameter shapes: w_hat.size={w_hat.size}, b_hat={b_hat}.")

            solution = np.concatenate([w_hat, np.array([float(b_hat)], dtype=float)], axis=0)

            pc_mass_est = np.nan
            rej_total_draws = 0

        elif algorithm == "kl_lad":
            # ========================
            # 4) Optimisation stage (KL-DRO baseline via python-dro)
            # ========================
            X_tr = np.concatenate([X_train, X_select], axis=0)
            y_tr = np.concatenate([y_train, y_select], axis=0).reshape(-1)

            if calibrate_on_validation:
                X_east = np.concatenate([X_train, X_select, X_val], axis=0)
                y_east = np.concatenate([y_train, y_select, y_val], axis=0)

                cv_info = _ca_housing_geo_block_cv_select(
                    algorithm="kl_lad",
                    X_east=X_east,
                    y_east=y_east,
                    d=d,
                    gamma_bulk=float(gamma_bulk),
                    num_likelihood_samples=int(num_likelihood_samples),
                    seed=int(replication),
                )

                validation_time = float(cv_info["validation_time"])
                val_mae_raw = float(cv_info["cv_mae_mean"])
                val_mae_std = float(cv_info["cv_mae_std"])
                epsilon_chosen = float(cv_info["best_param"])

                cv_n_folds = float(cv_info["cv_n_folds"])
                cv_n_blocks = float(cv_info["cv_n_blocks"])
                cv_lon_bins = float(cv_info["cv_lon_bins"])
                cv_lat_bins = float(cv_info["cv_lat_bins"])
            else:
                epsilon_chosen = float(epsilon)

            try:
                from dro.linear_model.kl_dro import KLDRO
            except Exception as e:
                raise ImportError(
                    "KL-DRO baseline requires the 'dro' package. "
                    "Install it with: pip install dro"
                ) from e

            solve_start = datetime.now()
            model = KLDRO(
                input_dim=int(d),
                model_type="lad",
                solver=cp.CLARABEL,
            )
            model.update({"eps": float(epsilon_chosen)})
            fit_out = model.fit(np.asarray(X_tr, dtype=float), np.asarray(y_tr, dtype=float).reshape(-1))
            solve_time = (datetime.now() - solve_start).total_seconds()
            setup_time = 0.0

            w_hat = None
            b_hat = None
            if isinstance(fit_out, dict):
                if "theta" in fit_out:
                    w_hat = np.asarray(fit_out["theta"], dtype=float).reshape(-1)
                elif "w" in fit_out:
                    w_hat = np.asarray(fit_out["w"], dtype=float).reshape(-1)

                if "b" in fit_out:
                    b_hat = float(np.asarray(fit_out["b"], dtype=float).reshape(-1)[0])
                elif "bias" in fit_out:
                    b_hat = float(np.asarray(fit_out["bias"], dtype=float).reshape(-1)[0])

            if w_hat is None:
                for attr in ("theta", "coef_", "w"):
                    if hasattr(model, attr):
                        w_hat = np.asarray(getattr(model, attr), dtype=float).reshape(-1)
                        break

            if b_hat is None:
                for attr in ("b", "bias", "intercept_", "intercept"):
                    if hasattr(model, attr):
                        b_hat = float(np.asarray(getattr(model, attr), dtype=float).reshape(-1)[0])
                        break

            if w_hat is None or b_hat is None:
                raise RuntimeError(
                    "KL-DRO fit did not expose parameters. Expected keys 'theta' and 'b' (or 'bias') "
                    "from model.fit(...) or corresponding attributes on the fitted model."
                )
            if w_hat.size != int(d) or not np.isfinite(b_hat):
                raise RuntimeError(f"KL-DRO returned invalid parameter shapes: w_hat.size={w_hat.size}, b_hat={b_hat}.")

            solution = np.concatenate([w_hat, np.array([float(b_hat)], dtype=float)], axis=0)

            pc_mass_est = np.nan
            rej_total_draws = 0

        elif algorithm == "or_wdro":
            # ========================
            # 4) Optimisation stage (OR-WDRO baseline via python-dro)
            # ========================
            # We use the python-dro ORWDRO implementation with LAD loss.
            # The run-level `epsilon` is interpreted as the Wasserstein radius (eps).
            # By default we set eta=0 (recovering standard WDRO); override via constants if desired.

            X_tr = np.concatenate([X_train, X_select], axis=0)
            y_tr = np.concatenate([y_train, y_select], axis=0).reshape(-1)

            if calibrate_on_validation:
                X_east = np.concatenate([X_train, X_select, X_val], axis=0)
                y_east = np.concatenate([y_train, y_select, y_val], axis=0)

                cv_info = _ca_housing_geo_block_cv_select(
                    algorithm="or_wdro",
                    X_east=X_east,
                    y_east=y_east,
                    d=d,
                    gamma_bulk=float(gamma_bulk),
                    num_likelihood_samples=int(num_likelihood_samples),
                    seed=int(replication),
                )

                validation_time = float(cv_info["validation_time"])
                val_mae_raw = float(cv_info["cv_mae_mean"])
                val_mae_std = float(cv_info["cv_mae_std"])
                epsilon_chosen = float(cv_info["best_param"])

                cv_n_folds = float(cv_info["cv_n_folds"])
                cv_n_blocks = float(cv_info["cv_n_blocks"])
                cv_lon_bins = float(cv_info["cv_lon_bins"])
                cv_lat_bins = float(cv_info["cv_lat_bins"])
            else:
                epsilon_chosen = float(epsilon)

            try:
                from dro.linear_model.or_wasserstein_dro import ORWDRO
            except Exception as e:
                raise ImportError(
                    "OR-WDRO baseline requires the 'dro' package. "
                    "Install it with: pip install dro"
                ) from e

            or_eta = float(getattr(_constants, "CALIFORNIA_HOUSING_OR_WDRO_ETA", 0.0))
            dual_norm = int(getattr(_constants, "CALIFORNIA_HOUSING_OR_WDRO_DUAL_NORM", 2))

            if not (0.0 <= or_eta <= 0.5):
                raise ValueError(f"OR-WDRO requires eta in [0, 0.5], got {or_eta}.")
            if dual_norm not in (1, 2):
                raise ValueError(f"OR-WDRO dual_norm must be 1 or 2, got {dual_norm}.")

            solve_start = datetime.now()
            model = ORWDRO(
                input_dim=int(d),
                model_type="lad",
                solver=cp.CLARABEL,
                eps=float(epsilon_chosen),
                eta=float(or_eta),
                dual_norm=int(dual_norm),
            )

            fit_out = model.fit(np.asarray(X_tr, dtype=float), np.asarray(y_tr, dtype=float).reshape(-1))
            solve_time = (datetime.now() - solve_start).total_seconds()
            setup_time = 0.0

            # Extract (w,b) robustly across python-dro versions
            w_hat = None
            b_hat = None
            if isinstance(fit_out, dict):
                if "theta" in fit_out:
                    w_hat = np.asarray(fit_out["theta"], dtype=float).reshape(-1)
                elif "w" in fit_out:
                    w_hat = np.asarray(fit_out["w"], dtype=float).reshape(-1)

                if "b" in fit_out:
                    b_hat = float(np.asarray(fit_out["b"], dtype=float).reshape(-1)[0])
                elif "bias" in fit_out:
                    b_hat = float(np.asarray(fit_out["bias"], dtype=float).reshape(-1)[0])

            if w_hat is None:
                for attr in ("theta", "coef_", "w"):
                    if hasattr(model, attr):
                        w_hat = np.asarray(getattr(model, attr), dtype=float).reshape(-1)
                        break

            if b_hat is None:
                for attr in ("b", "bias", "intercept_", "intercept"):
                    if hasattr(model, attr):
                        b_hat = float(np.asarray(getattr(model, attr), dtype=float).reshape(-1)[0])
                        break

            if w_hat is None or b_hat is None:
                raise RuntimeError(
                    "OR-WDRO fit did not expose parameters. Expected keys 'theta' and 'b' (or 'bias') "
                    "from model.fit(...) or corresponding attributes on the fitted model."
                )
            if w_hat.size != int(d) or not np.isfinite(b_hat):
                raise RuntimeError(f"OR-WDRO returned invalid parameter shapes: w_hat.size={w_hat.size}, b_hat={b_hat}.")

            solution = np.concatenate([w_hat, np.array([float(b_hat)], dtype=float)], axis=0)

            pc_mass_est = np.nan
            rej_total_draws = 0

        elif algorithm == "cvar_lad":
            # ========================
            # 4) Optimisation stage (solve the chosen problem)
            # ========================
            X_tr = np.concatenate([X_train, X_select], axis=0)
            y_tr = np.concatenate([y_train, y_select], axis=0)
            n_tr = int(X_tr.shape[0])

            # BUGFIX (geo-block CV): when epsilon is used as a placeholder (often 0.0),
            # some runners/builders construct the Chebyshev "max LAD" problem instead of CVaR.
            # That problem has NO 'cvar_coeff' parameter, so the CV-chosen epsilon is silently ignored.
            # If we are calibrating on validation, we must ensure we are solving the *CVaR* problem.
            if calibrate_on_validation and (problem is None or "cvar_coeff" not in problem.param_dict):
                warnings.warn(
                    "[california_housing] Received a 'cvar_lad' problem without parameter 'cvar_coeff'. "
                    "This typically means the driver built get_max_lad_problem because epsilon==0. "
                    "Rebuilding get_cvar_lad_problem so the CV-chosen tail mass is actually applied.",
                    UserWarning,
                )
                problem = get_cvar_lad_problem(dim=d, num_samples=n_tr)

            problem.param_dict["X_train"].value = X_tr
            problem.param_dict["y_train"].value = y_tr

            # Interpret epsilon as tail mass (so CVaR^{1-epsilon}).
            if calibrate_on_validation:
                # Option A: geo-block CV within EAST to choose epsilon (tail mass), then fit once on full EAST.
                X_east = np.concatenate([X_train, X_select, X_val], axis=0)
                y_east = np.concatenate([y_train, y_select, y_val], axis=0)

                cv_info = _ca_housing_geo_block_cv_select(
                    algorithm="cvar_lad",
                    X_east=X_east,
                    y_east=y_east,
                    d=d,
                    gamma_bulk=float(gamma_bulk),
                    num_likelihood_samples=int(num_likelihood_samples),
                    seed=int(replication),
                )

                validation_time = float(cv_info["validation_time"])
                val_mae_raw = float(cv_info["cv_mae_mean"])
                val_mae_std = float(cv_info["cv_mae_std"])
                epsilon_chosen = float(cv_info["best_param"])

                cv_n_folds = float(cv_info["cv_n_folds"])
                cv_n_blocks = float(cv_info["cv_n_blocks"])
                cv_lon_bins = float(cv_info["cv_lon_bins"])
                cv_lat_bins = float(cv_info["cv_lat_bins"])

                tail_mass = float(epsilon_chosen)
                if "cvar_coeff" in problem.param_dict and tail_mass > 0.0:
                    problem.param_dict["cvar_coeff"].value = float(1.0 / (tail_mass * n_tr))

                solve_start = datetime.now()
                problem.solve(
                    solver=cp.CLARABEL,
                    verbose=verbose,
                )
                solve_time = (datetime.now() - solve_start).total_seconds()
                setup_time = problem.solver_stats.setup_time

                solution = problem.var_dict["x"].value
            else:
                tail_mass = float(epsilon)
                if "cvar_coeff" in problem.param_dict and tail_mass > 0.0:
                    problem.param_dict["cvar_coeff"].value = float(1.0 / (tail_mass * n_tr))

                solve_start = datetime.now()
                problem.solve(
                    solver=cp.CLARABEL,
                    verbose=verbose,
                )
                solve_time = (datetime.now() - solve_start).total_seconds()
                setup_time = problem.solver_stats.setup_time

                solution = problem.var_dict["x"].value

            pc_mass_est = np.nan
            rej_total_draws = 0

        else:
            raise NotImplementedError(f"Algorithm '{algorithm}' not implemented for california_housing.")

        xi_test = np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1)
        test_bulk_rate = float(np.mean(_in_bulk_xi(xi_test)))
        # Predictions
        if solution is None or np.any(~np.isfinite(solution)):
            raise RuntimeError("Solver returned non-finite solution for california_housing.")

        sol = np.asarray(solution, dtype=float).reshape(-1)
        if sol.size != d + 1:
            raise ValueError(f"Expected solution size {d+1}, got {sol.size}")
        w_hat = sol[:d]
        b_hat = float(sol[d])

        y_tr = np.concatenate([y_train, y_select], axis=0)
        y_pred = X_test @ w_hat + b_hat
        if clip_extreme_y:
            y_pred = np.clip(y_pred, a_min=None, a_max=500001.0)  # cap extreme predictions for stability
            print("Y_clippped extreme predictions at 500001.0")
        #y_pred = np.clip(y_pred, a_min=None, a_max=500001.0)  # cap extreme predictions for stability
        abs_errors = np.abs(y_test - y_pred)
        y_tr_median = np.median(y_tr)
        y_test_median = np.median(y_test)
        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(abs_errors ** 2)))
        p98_abs_error = float(np.quantile(abs_errors, 0.98))
        cvar_abs_error = float(empirical_cvar(abs_errors, tail_mass=0.02))
        trivial_error = np.abs(y_test - y_tr_median)
        cvar_trivial_error = float(empirical_cvar(trivial_error, tail_mass=0.02))
        mae_trivial =float(np.mean(trivial_error))

        if verbose:
            print("California Housing TEST diagnostics:")
            print(f"  MAE   = {mae:.6f}")
            print(f"  RMSE  ={rmse:.6f}")
            print(f"  p98AE = {p98_abs_error:.6f}")
            print(f"  CVaR  = {cvar_abs_error:.6f}")

            if bulk_shape_key == "ellipsoid":
                # Sanity: closed-form sup bound is indeed an upper bound on |residual| over the ellipsoid.
                n_check = min(2000, int(X_test.shape[0]))
                p = d + 1

                # Sample approximately-uniform points in the ellipsoid:
                #   xi = mu + sqrt_Sigma @ u,  where ||u||_2 <= t_hat
                U = generator.normal(size=(n_check, p))
                norms = np.linalg.norm(U, axis=1, keepdims=True)
                U_unit = U / np.maximum(norms, 1e-12)
                radii = (generator.random(size=(n_check, 1)) ** (1.0 / p)) * float(t_hat)
                U_ball = radii * U_unit
                xi_unif = mu.reshape(1, -1) + (sqrt_Sigma @ U_ball.T).T

                X_unif = xi_unif[:, :d]
                y_unif = xi_unif[:, d]
                resid_unif = np.abs(y_unif - (X_unif @ w_hat + b_hat))

                a = np.concatenate([-w_hat, np.array([1.0])], axis=0)
                sup_closed_form = abs(mu_y - (mu_x @ w_hat + b_hat)) + float(t_hat) * float(
                    np.linalg.norm(sqrt_Sigma.T @ a, ord=2)
                )

                max_violation = float(np.max(resid_unif - sup_closed_form))
                print(f"  sup closed-form bound check: max(resid - sup) = {max_violation:.6e}")
                if max_violation > 1e-6:
                    warnings.warn(
                        "Closed-form ellipsoid sup bound appears violated (numerically). "
                        "Check sqrt_Sigma construction and the SOCP constraint.",
                        UserWarning,
                    )

        return {
            "uuid": uuid,
            "replication": replication,
            "algorithm": algorithm,
            "dataset": dataset,
            "epsilon": float(epsilon_chosen),
            "gamma_bulk": float(gamma_bulk),
            "ch_geo_axis": str(ch_geo_axis),
            "gap_ratio": float(gap_ratio),

            "solution": list(sol),

            "dgp_time": dgp_time,
            "posterior_time": posterior_time,
            "likelihood_time": likelihood_time,
            "solve_time": solve_time,
            "setup_time": setup_time,

            "calibrate_on_validation": bool(calibrate_on_validation),
            "validation_time": float(validation_time),
            "val_mae_raw": float(val_mae_raw) if np.isfinite(val_mae_raw) else np.nan,
            "val_mae_std": float(val_mae_std) if np.isfinite(val_mae_std) else np.nan,

            "cv_n_folds": float(cv_n_folds) if np.isfinite(cv_n_folds) else np.nan,
            "cv_n_blocks": float(cv_n_blocks) if np.isfinite(cv_n_blocks) else np.nan,
            "cv_lon_bins": float(cv_lon_bins) if np.isfinite(cv_lon_bins) else np.nan,
            "cv_lat_bins": float(cv_lat_bins) if np.isfinite(cv_lat_bins) else np.nan,

            "ridge_lambda_chosen": float(ridge_lambda_chosen) if np.isfinite(ridge_lambda_chosen) else np.nan,
            "wass_rho_chosen": float(wass_rho_chosen) if np.isfinite(wass_rho_chosen) else np.nan,
            "solver_failed": bool(solver_failed),

            "t_hat": t_hat,
            "mu_x": list(mu_x),
            "mu_y": float(mu_y),
            "r_y": float(r_y),

            "pc_bulk_mass_est": pc_mass_est,
            "rejection_total_draws": rej_total_draws,

            "train_bulk_rate": train_bulk_rate,
            "select_bulk_rate": select_bulk_rate,
            "test_bulk_rate": test_bulk_rate,

            "mae": mae,
            "rmse": rmse,
            "mae_trivial": mae_trivial,
            "p98_abs_error": p98_abs_error,
            "cvar_abs_error": cvar_abs_error,
            "cvar_trivial_error": cvar_trivial_error,

            "log_partition_constant": 0.0,
            "out_of_sample_cost": list(abs_errors),
        }
    if dataset == "newsvendor" or dataset == "portfolio_synthetic":
        # NOTE if contamination is specified, then only contaminate the test samples (not training samples)
        data = sample_dgp(
            dgp, num_observations, dim=dim, contamination=0.0, contamination_type=None, generator=generator
        )
        eval_contamination = 1.0 if contamination_type == "entire_shift" else contamination
        data_eval = sample_dgp(
            dgp,
            num_test_observations,
            dim=dim,
            contamination=eval_contamination,
            contamination_type=contamination_type,
            generator=generator,        
        )
    elif dataset == "portfolio":
        # NOTE shape of data (N, D) where N is number of weeks and D is the number of stocks
        if dgp == "DowJones-crash":
            CRASH_WINDOW_ID = 75    # NOTE use 72 for long term evaluation of crash
            CRASH_OOS_TIME_WINDOW = IN_SAMPLE_TIME_WINDOW  # NOTE use 4 years for long term
            data, data_eval = portfolio_dataset(dgp, CRASH_WINDOW_ID, dataset_dir, out_of_sample_time_window=CRASH_OOS_TIME_WINDOW)
        else:
            data, data_eval = portfolio_dataset(dgp, replication, dataset_dir)
        # NOTE only normalise training data
        if normalise:
            data = normalise_by_dimension(data)
    else:
        raise NotImplementedError(f"Dataset not implemented: {dataset}")
    dgp_time = (datetime.now() - dgp_start).total_seconds()

    # 2. sample from the posterior
    posterior_start = datetime.now()
    log_partition_constant = 0.0
    if algorithm == "or_wdro":
        # OR-WDRO does not use a posterior
        theta_sample = None
        posterior_time = (datetime.now() - posterior_start).total_seconds()
    elif inference == "bayes":
        theta_prior = default_prior_params(posterior, dim=dim)
        theta_posterior = get_posterior_params(posterior, data, theta_prior)
        if algorithm == "kl_dro_bas" and posterior == "pareto":
            raise NotImplementedError(
                "kl_dro_bas is not implemented for posterior='pareto' (Uniform(0,θ) non-regular model). "
                "Use kl_pp or kl_bdro instead."
            )

        if algorithm == "kl_dro_bas":
            assert num_posterior_samples == 1
            log_partition_constant = get_log_partition_constant(posterior, theta_posterior)
            theta_sample = derive_analytical_posterior_params(
                posterior, theta_posterior
            )
        elif algorithm == "kl_bdro" and dataset in ("portfolio", "portfolio_synthetic") and posterior == "normal_inverse_wishart":
            mu_post, _, iota_post, Psi_post = theta_posterior
            theta_sample = bdro_portfolio_posterior_samples(num_posterior_samples, mu_post, iota_post, Psi_post, generator=generator)
        elif algorithm == "kl_pp":
            theta_sample = posterior_predictive_params(posterior, theta_posterior)
        elif algorithm == "lv_bas":
            # LV-BAS, TV-BAS, and LV-Reverse use theta_posterior directly (for NIW-based P_c),
            # so we do not need to draw theta_sample
            theta_sample = None
        else:
            theta_sample = sample_posterior(posterior, theta_posterior, num_posterior_samples, generator=generator)
    elif inference in ("npl_wlb", "npl_mmd"):
        # get the posterior 
        path_to_csv = npl_uuid_dir / f"npl_sample_{replication}.csv"
        theta_sample = pd.read_csv(path_to_csv, index_col=False, header=None).values
        assert num_posterior_samples == theta_sample.shape[0]
        # assert dim == theta_sample.shape[1], f"Dimension dim={dim} not equal to {theta_sample.shape[1]}"
    elif inference == "empirical":
        # empirical does not have a posterior
        theta_sample = np.nan * np.ones(num_posterior_samples)
    else:
        raise ValueError(f"Inference procedure '{inference}' is not supported.")
    assert log_partition_constant >= 0
    # assert log_partition_constant < epsilon
    posterior_time = (datetime.now() - posterior_start).total_seconds()

    # 3. sample from the likelihood
    if algorithm in ("lv_bas", "or_wdro") and dataset == "newsvendor":
        # LV-BAS newsvendor does not use the generic likelihood pipeline.
        # It will build its own posterior-predictive SAA inside the LV-specific
        # branch below, so we skip likelihood sampling here.
        xi = None
        likelihood_time = 0.0
    else:
        likelihood_start = datetime.now()
        if inference == "empirical":
            xi = data
        elif inference == "bayes" and algorithm == "kl_pp":
            xi = sample_posterior_predictive(likelihood, posterior, theta_sample, dim, num_likelihood_samples, generator=generator).reshape((1, num_likelihood_samples, dim))
        elif inference == "bayes" and dataset in ("portfolio", "portfolio_synthetic") and likelihood == "multivariate_normal":
            pass    # no need to sample from likelihood cause we have closed form
        else:
            xi = sample_likelihood(
                likelihood,
                theta_sample,
                dim,
                num_likelihood_samples,
                num_posterior_samples,
                generator=generator,
                inference=inference,
            )
        likelihood_time = (datetime.now() - likelihood_start).total_seconds()


    # 4. run the chosen DRO algorithm
    solve_start = datetime.now()
    solve_time = 0.0
    solution = np.nan
    if (
            (dataset == "portfolio" or dataset == "portfolio_synthetic")
            and algorithm in ("kl_bdro", "kl_dro_bas")
            and likelihood == "multivariate_normal"
    ):
        # if epsilon - log_partition_constant < 0:
        #     # NOTE the optimisation problem is unbounded below
        #     solution = np.inf * np.ones(dim)
        #     solve_time = 0.0
        #     setup_time = 0.0
        # else:
        problem.param_dict["epsilon_minus_constant"].value = np.array([epsilon])
        problem.param_dict["mu_post"].value = theta_sample[0, :dim]
        for i in range(num_posterior_samples):
            # get a PSD covariance from the upper triangular vector
            cov = reconstruct_covariance_from_triu(theta_sample[i, dim:], dim)
            # then take the square root of the covariance and set to parameter value
            problem.param_dict[f"sqrt_cov_post_{i}"].value = sp.linalg.sqrtm(cov)

        problem.solve(solver=cp.CLARABEL, verbose=verbose, ignore_dpp=ignore_dpp)
        solution = problem.var_dict["x"].value
        setup_time = problem.solver_stats.setup_time
    elif (
        (dataset == "portfolio" or dataset == "portfolio_synthetic")
        and algorithm == "lv_bas"
        and likelihood == "multivariate_normal"
    ):
        # LV-BAS portfolio with ellipsoidal bulk set Xi_0 and linear f_x(ξ) = -x^T ξ.
        if inference != "bayes" or posterior != "normal_inverse_wishart":
            raise NotImplementedError(
                "LV-BAS portfolio currently implemented only for "
                "Bayesian NIW model with multivariate_normal likelihood."
            )

        in_sample = data
        n_in = in_sample.shape[0]
        if n_in < 2:
            raise RuntimeError(
                "Not enough in-sample points to form train/selection split "
                f"for LV-BAS (n_in={n_in})."
            )

        # Random split into DKW-train (geometry) and DKW-selection (threshold)
        rng_lv = np.random.default_rng(seed=replication)
        perm = rng_lv.permutation(n_in)
        train_size = n_in // 2
        train_idx = perm[:train_size]
        sel_idx = perm[train_size:]

        train_returns = in_sample[train_idx]
        selection_returns = in_sample[sel_idx]

        gamma_DKW = gamma_bulk
        delta_DKW = 0.05

        def _ellipsoid_score_from_params(mu: np.ndarray, Sigma: np.ndarray):
            """
            s(xi) = || Sigma^{-1/2} (xi - mu) ||_2 computed via Cholesky solve.
            Faster and numerically safer than forming Sigma^{-1}.
            """
            mu = np.asarray(mu, dtype=float).reshape(-1)
            d = mu.size
            Sigma = np.asarray(Sigma, dtype=float).reshape(d, d)
            Sigma = 0.5 * (Sigma + Sigma.T) + 1e-12 * np.eye(d)  # symmetrise + tiny ridge
            L = np.linalg.cholesky(Sigma)

            def score_fn(xi: np.ndarray) -> np.ndarray:
                xi_arr = np.asarray(xi, dtype=float)
                if xi_arr.ndim == 1:
                    xi_arr = xi_arr[None, :]
                D = (xi_arr - mu).T                 # (d, n)
                Z = np.linalg.solve(L, D)           # (d, n) s.t. L Z = D
                return np.sqrt(np.sum(Z * Z, axis=0))

            score_meta = {"type": "ellipsoid", "mu": mu, "Sigma": Sigma}
            return score_fn, score_meta


        if lv_use_pc_bulk_geometry:
            # (A) Geometry and P_c both from NIW posterior on DKW-train
            theta_prior_pc = default_prior_params("normal_inverse_wishart", dim=dim)
            theta_posterior_pc = get_posterior_params(
                "normal_inverse_wishart",
                train_returns,
                theta_prior_pc,
            )

            pp_vec = np.asarray(
                posterior_predictive_params("normal_inverse_wishart", theta_posterior_pc),
                dtype=float
            ).reshape(-1)

            mu_pc = pp_vec[:dim]
            vec_triu_size = upper_triangular_size(dim)
            vec_shape = pp_vec[dim : dim + vec_triu_size]
            Sigma_pc = reconstruct_covariance_from_triu(vec_shape, dim)

            score_fn, score_meta = _ellipsoid_score_from_params(mu_pc, Sigma_pc)

        else:
            # (B) Empirical geometry on DKW-train; P_c from NIW on full in-sample
            score_fn, score_meta = build_score(train_returns, score_type="ellipsoid")

            pc_sampler = make_lv_pc_sampler_from_niw(
                theta_posterior=theta_posterior,
                dim=dim,
                likelihood="multivariate_normal",
                posterior="normal_inverse_wishart",
            )

        # DKW threshold on selection scores (same for both branches)
        audit_scores = score_fn(selection_returns)
        dkw_info = dkw_select_threshold(audit_scores, gamma=gamma_DKW, delta=delta_DKW)

        if verbose:
            mode = "Pc-geometry" if lv_use_pc_bulk_geometry else "empirical-geometry"
            print(f"LV-BAS DKW calibration (portfolio, {mode}):")
            for k, v in dkw_info.items():
                print(f"  {k:>25s} = {v}")

        if not dkw_info.get("exists", False):
            t_hat = float(np.max(audit_scores))
            r = dkw_info["r"]
            warnings.warn(
                f"DKW certificate for gamma={gamma_DKW}, delta={delta_DKW} does not exist; "
                f"using t_hat=max(score) (coverage margin r={r}).",
                UserWarning,
            )
        else:
            t_hat = float(dkw_info["t_hat"])

        # Bulk set spec built from the SAME score_meta used to produce audit_scores
        xi0_spec = make_bulk_set_spec(score_meta, t_hat)

        # Expose these explicitly (keeps downstream code unchanged)
        mu_bulk = np.asarray(score_meta["mu"], dtype=float)
        Sigma_bulk = np.asarray(score_meta["Sigma"], dtype=float)

        if lv_use_pc_bulk_geometry:
            mu_trunc = mu_bulk
        else:
            mu_trunc = truncated_mean(
                sampler_or_density=pc_sampler,
                xi0_spec=xi0_spec,
                n_saa=5000,
                rng=rng_lv,  # use your in-scope rng; if it's called rng, replace rng_lv -> rng
            )

        mu_eff = (1.0 - epsilon) * mu_trunc + epsilon * mu_bulk
        lam = float(epsilon * t_hat)

        # Robust symmetric covariance and its Cholesky / square root
        Sigma_sym = 0.5 * (Sigma_bulk + Sigma_bulk.T)
        Sigma_sym += 1e-8 * np.eye(dim)
        sqrt_cov_bulk = la.cholesky(Sigma_sym)

        # Set LV-BAS problem parameters and solve
        problem.param_dict["mu_eff"].value = mu_eff
        problem.param_dict["sqrt_cov_bulk"].value = sqrt_cov_bulk
        problem.param_dict["lam"].value = lam

        problem.solve(
            solver=cp.CLARABEL,
            verbose=verbose,
        )
        solution = problem.var_dict["x"].value
        setup_time = problem.solver_stats.setup_time
    elif dataset == "newsvendor" and algorithm == "lv_bas":
        # LV-BAS newsvendor with ellipsoidal bulk set Xi_0 and truncated posterior predictive SAA.

        if inference != "bayes":
            raise NotImplementedError(
                "LV-BAS newsvendor currently requires inference='bayes' "
                "(needs posterior predictive centre P_c)."
            )
        if not (0.0 <= float(epsilon) <= 1.0):
            raise ValueError("LV-BAS newsvendor requires epsilon in [0,1].")
        lv_likelihood_start = datetime.now()

        in_sample = data
        n_in = int(in_sample.shape[0])
        if n_in < 2:
            raise RuntimeError(
                "Not enough in-sample points to form train/selection split "
                f"for LV-BAS (n_in={n_in})."
            )

        # Split into DKW-train (geometry) and DKW-selection (threshold)
        rng_lv = np.random.default_rng(seed=replication)
        perm = rng_lv.permutation(n_in)
        train_size = n_in // 2
        train_xi = in_sample[perm[:train_size]]
        selection_xi = in_sample[perm[train_size:]]

        gamma_DKW = gamma_bulk
        delta_DKW = 0.05

        def _ellipsoid_score_from_params(mu: np.ndarray, Sigma: np.ndarray):
            """s(xi) = ||Sigma^{-1/2}(xi - mu)||_2 via Cholesky solve."""
            mu = np.asarray(mu, dtype=float).reshape(-1)
            d = int(mu.size)
            Sigma = np.asarray(Sigma, dtype=float).reshape(d, d)
            Sigma = 0.5 * (Sigma + Sigma.T) + 1e-12 * np.eye(d)
            L = np.linalg.cholesky(Sigma)

            def score_fn(xi_arr: np.ndarray) -> np.ndarray:
                X = np.asarray(xi_arr, dtype=float)
                if X.ndim == 1:
                    X = X[None, :]
                D = (X - mu).T
                Z = np.linalg.solve(L, D)
                return np.sqrt(np.sum(Z * Z, axis=0))

            meta = {"type": "ellipsoid", "mu": mu, "Sigma": Sigma}
            return score_fn, meta

        # Posterior predictive sampler P_c (always used for truncated expectation)
        if lv_use_pc_bulk_geometry:
            # Geometry from posterior predictive (fit mu/cov via Monte Carlo draws),
            # with posterior updated using DKW-train only.
            theta_prior_pc = default_prior_params(posterior, dim=dim)
            theta_posterior_pc = get_posterior_params(posterior, train_xi, theta_prior_pc)
            pp_pc = posterior_predictive_params(posterior, theta_posterior_pc)

            def pc_sampler(n: int) -> np.ndarray:
                draws = sample_posterior_predictive(
                    likelihood, posterior, pp_pc, dim, n, generator=rng_lv
                )
                return np.asarray(draws, dtype=float).reshape(-1, dim)

            # Estimate Gaussian geometry of P_c
            mc_n = int(max(2000, min(20000, 20 * dim * dim)))
            mc = pc_sampler(mc_n)
            mu_pc = mc.mean(axis=0)
            Xc = mc - mu_pc
            Sigma_pc = (Xc.T @ Xc) / max(1, mc.shape[0] - 1)
            Sigma_pc += 1e-8 * np.eye(dim)

            score_fn, score_meta = _ellipsoid_score_from_params(mu_pc, Sigma_pc)
        else:
            # Empirical geometry from DKW-train; P_c from posterior on full in-sample.
            score_fn, score_meta = build_score(train_xi, score_type="ellipsoid")
            pp_full = posterior_predictive_params(posterior, theta_posterior)

            def pc_sampler(n: int) -> np.ndarray:
                draws = sample_posterior_predictive(
                    likelihood, posterior, pp_full, dim, n, generator=rng_lv
                )
                return np.asarray(draws, dtype=float).reshape(-1, dim)

        # DKW threshold on selection scores
        audit_scores = score_fn(selection_xi)
        dkw_info = dkw_select_threshold(audit_scores, gamma=gamma_DKW, delta=delta_DKW)

        if verbose:
            mode = "Pc-geometry" if lv_use_pc_bulk_geometry else "empirical-geometry"
            print(f"LV-BAS DKW calibration (newsvendor, {mode}):")
            for k, v in dkw_info.items():
                print(f"  {k:>25s} = {v}")

        if not dkw_info.get("exists", False) or not np.isfinite(dkw_info.get("t_hat", np.nan)):
            t_hat = float(np.max(audit_scores)) if audit_scores.size else 0.0
            warnings.warn(
                f"DKW certificate for gamma={gamma_DKW}, delta={delta_DKW} does not exist; "
                f"using t_hat=max(score) (r={dkw_info.get('r', np.nan)}).",
                UserWarning,
            )
        else:
            t_hat = float(dkw_info["t_hat"])

        xi0_spec = make_bulk_set_spec(score_meta, t_hat)

        # --- Truncated posterior predictive sampling (exactly N accepted points)
        N = int(num_likelihood_samples * 0.5)
        if N <= 0:
            raise ValueError("num_likelihood_samples must be positive for LV-BAS newsvendor.")

        accepted = []
        n_acc = 0
        total_draws = 0
        draw_cap_factor = 5000
        max_draws = draw_cap_factor * N

        while n_acc < N and total_draws < max_draws:
            remaining = N - n_acc
            batch_size = max(2 * remaining, 256)
            xi_batch = pc_sampler(batch_size)

            mask = score_fn(xi_batch) <= t_hat + 1e-10
            if np.any(mask):
                acc = xi_batch[mask]
                accepted.append(acc)
                n_acc += int(acc.shape[0])
            total_draws += batch_size
        
        if total_draws > int(num_likelihood_samples): 
            warnings.warn(
                f"LV-BAS newsvendor: required {total_draws} posterior predictive draws to get {N} accepted samples; "
                "Pc might have small mass on the bulk set.",
                UserWarning,
            )

        if not accepted:
            raise RuntimeError(
                "LV-BAS newsvendor: could not obtain any accepted posterior predictive samples in Xi_0."
            )

        xi_acc = np.concatenate(accepted, axis=0)
        if xi_acc.shape[0] < N:
            warnings.warn(
                f"LV-BAS newsvendor: accepted only {xi_acc.shape[0]}/{N} points inside Xi_0; "
                "filling remainder by resampling with replacement from the accepted set.",
                UserWarning,
            )
            extra = rng_lv.choice(xi_acc.shape[0], size=N - xi_acc.shape[0], replace=True)
            xi_acc = np.vstack([xi_acc, xi_acc[extra]])

        xi_trunc = xi_acc[:N]

        # --- Worst-case constants for the ellipsoid Xi_0
        a_mat = getattr(problem, "_lv_newsvendor_a_mat", None)
        if a_mat is None:
            a_mat = make_newsvendor_a_mat(dim)

        mu_bulk = np.asarray(score_meta["mu"], dtype=float).reshape(-1)
        Sigma_bulk = np.asarray(score_meta["Sigma"], dtype=float).reshape(dim, dim)
        Sigma_sym = 0.5 * (Sigma_bulk + Sigma_bulk.T) + 1e-12 * np.eye(dim)
        chol = la.cholesky(Sigma_sym)

        # wcs_const[k] = a_k^T mu + t * ||chol(Sigma)^T a_k||_2
        norms = la.norm((chol.T @ a_mat.T).T, axis=1)
        wcs_const = a_mat @ mu_bulk + t_hat * norms

        # Record LV-BAS posterior-predictive / truncation time separately.
        likelihood_time += (datetime.now() - lv_likelihood_start).total_seconds()
        solve_start = datetime.now()

        # Set parameters and solve
        problem.param_dict["epsilon"].value = float(epsilon)

        # Set parameters and solve
        problem.param_dict["epsilon"].value = float(epsilon)
        problem.param_dict["one_minus_epsilon"].value = float(1.0 - float(epsilon))
        problem.param_dict["xi_trunc"].value = xi_trunc
        problem.param_dict["wcs_const"].value = wcs_const

        problem.solve(
            solver=cp.CLARABEL,
            verbose=verbose,
        )
        solution = problem.var_dict["x"].value
        setup_time = problem.solver_stats.setup_time

        
    elif dataset == "newsvendor" and algorithm == "or_wdro":
        # OR-WDRO (Outlier-Robust Wasserstein DRO)
        # epsilon is the outlier fraction vareps in the paper.

        # if not (0.0 <= epsilon < 0.5):
        #     raise ValueError(f"epsilon must be in [0, 0.5), got {epsilon}")
        if vary_rho: #for diagnostics only
            eps_cont = float(contamination)  # fixed contamination level for rho sweep
            # Robust moment centre z0 via coordinate-wise trimmed mean (direct MATLAB port)
            z0 = cheap_robust_mean_estimate(data, eps_cont)

            # Robust sigma^2 estimate (q=2). Conservative trimmed mean of squared distances.
            sigma_sq = robust_sigma_sq_estimate(data, z0, eps_cont)

            # Set CVXPY parameters
            problem.param_dict["Z"].value = data
            problem.param_dict["z0"].value = z0
            problem.param_dict["sigma_sq"].value = float(sigma_sq)
            problem.param_dict["rho"].value = epsilon
            problem.param_dict["inv_one_minus_vareps"].value = float(1.0 / (1.0 - eps_cont))
        else:
            rho_fixed = float(5.0)
            # Robust moment centre z0 via coordinate-wise trimmed mean (direct MATLAB port)
            z0 = cheap_robust_mean_estimate(data, epsilon)

            # Robust sigma^2 estimate (q=2). Conservative trimmed mean of squared distances.
            sigma_sq = robust_sigma_sq_estimate(data, z0, epsilon)

            # Set CVXPY parameters
            problem.param_dict["Z"].value = data
            problem.param_dict["z0"].value = z0
            problem.param_dict["sigma_sq"].value = float(sigma_sq)
            problem.param_dict["rho"].value = rho_fixed
            problem.param_dict["inv_one_minus_vareps"].value = float(1.0 / (1.0 - epsilon))

        problem.solve(cp.CLARABEL, verbose=False, ignore_dpp=ignore_dpp)
        solution = problem.var_dict["x"].value
        setup_time = problem.solver_stats.setup_time

    elif algorithm in ("kl_bdro", "kl_dro_bas", "kl_pp", "kl_empirical"):
        if epsilon - log_partition_constant < 0:
            # NOTE the optimisation problem is unbounded below
            solution = np.inf * np.ones(dim)
            solve_time = 0.0
            setup_time = 0.0
        else:
            # set parameters then solve
            problem.param_dict["epsilon_minus_constant"].value = np.array([epsilon - log_partition_constant])
            xi = xi.reshape((num_posterior_samples, num_likelihood_samples, dim))
            for i in range(num_posterior_samples):
                problem.param_dict[f"xi_{i}"].value = xi[i]
            problem.solve(solver=cp.CLARABEL, verbose=verbose, ignore_dpp=ignore_dpp)
            solution = problem.var_dict["x"].value
            # solve_time = problem.solver_stats.solve_time
            setup_time = problem.solver_stats.setup_time
    elif algorithm in ("dro_bas_mmd", "empirical_mmd"):
        if algorithm == "dro_bas_mmd":
            xi = xi.reshape((num_likelihood_samples*num_posterior_samples,dim))
        elif algorithm == "empirical_mmd":
            xi = data.reshape((num_observations,dim))
        Xcert = np.random.uniform(np.min(xi), np.max(xi), size=[num_certify_points,dim])
        zetai = np.concatenate([xi, Xcert], axis=0)
        l = np.sqrt((1/2)*np.median(distance.cdist(zetai, zetai, 'sqeuclidean')))
        # K = k_jax(zetai, zetai, l)
        K = k_comp(zetai, zetai)
        K_decomp = mat_decomp_jax(K)
        problem.param_dict["Xobs"].value = xi
        problem.param_dict["Xcert"].value = Xcert
        problem.param_dict["K"].value = np.asarray(K)
        problem.param_dict["K_decomposed"].value = np.asarray(K_decomp)
        problem.param_dict["epsilon"].value = np.array([epsilon])
        problem.solve(cp.CLARABEL, verbose=False, ignore_dpp=ignore_dpp)
        solution = problem.var_dict["theta"].value
        # solve_time = problem.solver_stats.solve_time
        setup_time = problem.solver_stats.setup_time
    elif algorithm == "bdro_grid_search":
        solution = main_Bayesian_DRO(xi, epsilon)
        setup_time = 0.0  # can't really measure this easily
    else:
        raise ValueError("Please choose a valid algorithm")
    if solve_time == 0.0:
        solve_time = (datetime.now() - solve_start).total_seconds()
    # evaluate the out-of-sample cost
    if (solution == np.inf).any():
        out_of_sample_cost = np.inf * np.ones(num_test_observations)
    else:
        if dataset == "newsvendor":
            out_of_sample_cost = newsvendor_cost_cvxpy(solution, data_eval.reshape((num_test_observations, dim))).value
        elif dataset in ("portfolio", "portfolio_synthetic"):
            out_of_sample_cost = data_eval @ solution
        else:
            raise NotImplementedError(f"Out-of-sample cost for dataset '{dataset}' not implemented")
        solution = list(solution)

    return {
        "uuid": uuid,
        "replication": replication,
        "gamma_bulk": float(gamma_bulk),
        "solution": solution,
        "dgp_time": dgp_time,
        "likelihood_time": likelihood_time,
        "posterior_time": posterior_time,
        "solve_time": solve_time,
        "setup_time": setup_time,
        "log_partition_constant": log_partition_constant,
        "out_of_sample_cost": list(out_of_sample_cost),
    }


POSTERIOR_GB_COLS = [
    "contamination",
    "dataset",
    "dgp",
    "dim",
    "eta",
    "inference",
    "kernel_name",
    "lengthscale",
    "likelihood",
    "normalise",
    "num_observations",
    "num_posterior_samples",
    "num_replications",
    "posterior"
]

def get_npl_uuid(posterior_settings_df: pd.DataFrame, params: dict) -> str:
    params_tuple = tuple([params[key] for key in POSTERIOR_GB_COLS])
    param_sr = posterior_settings_df.loc[params_tuple]
    return param_sr["npl_uuid"]

@app.command("npl")
def sample_npl_for_experiment(
    npl_samples_dir: Path,
    batch: int,
    dataset_dir: Optional[Path] = None,
):
    """Run a single replication where the seed is given by the replication number"""

    npl_samples_dir.mkdir(parents=False, exist_ok=True)
    posterior_times = []

    npl_df = pd.read_csv(npl_samples_dir / "npl_settings.csv")
    npl_row = npl_df.iloc[batch]
    npl_dir = npl_samples_dir / npl_row["npl_uuid"]
    npl_dir.mkdir()
    dataset = npl_row["dataset"]
    dgp = npl_row["dgp"]
    npl_uuid = npl_row["npl_uuid"]
    for replication in range(npl_row["num_replications"]):
        # 1. load portfolio dataset
        generator = np.random.default_rng(seed=replication)
        if dataset in ("newsvendor", "portfolio_synthetic"):
            # NOTE if contamination is specified, then only contaminate the training samples (not test samples)
            data = sample_dgp(
                dgp, npl_row["num_observations"], dim=npl_row["dim"], contamination=npl_row["contamination"], generator=generator
            )
        elif dataset == "portfolio":
            data, _ = portfolio_dataset(dgp, replication, dataset_dir)
            if npl_row["normalise"]:
                data = normalise_by_dimension(data)
        else:
            raise NotImplementedError(f"Dataset not implemented: {dataset}")

        # 2. sample from the posterior
        print()
        npl_start = datetime.now()
        print(npl_start, "- Starting", dataset, "sample NPL for replication", replication)    
        theta_sample = sample_npl(
            data,
            npl_row["inference"],
            npl_row["likelihood"],
            npl_row["num_posterior_samples"],
            seed=replication,
            lengthscale=npl_row["lengthscale"],
            generator=generator,
            dim=npl_row["dim"],
            kernel_name=npl_row["kernel_name"],
            eta=npl_row["eta"],
        )
        npl_finish = datetime.now()
        total_seconds =  (datetime.now() - npl_start).total_seconds()
        print(npl_finish, "- Finished replication", replication, "in", total_seconds, "seconds.")

        df = pd.DataFrame(theta_sample)
        df.to_csv(npl_dir / f"npl_sample_{replication}.csv", index=False, header=False)
        posterior_times.append({
            "replication": replication,
            "npl_uuid": npl_uuid,
            "posterior_time": total_seconds,
        })
    pd.DataFrame(posterior_times).to_csv(npl_samples_dir / npl_uuid / f"npl_times_{npl_uuid}.csv", index=False)

if __name__ == "__main__":
    app()
