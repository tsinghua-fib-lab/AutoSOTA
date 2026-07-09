"""Consolidated likelihood computation for CPC-LLM.

Computes the likelihood scores that an LLM assigns to particular sequences,
averaged over a set of input seeds, for one or more models.

Functions:
    compute_likelihoods_inmemory: Core in-memory function operating on DataFrames.
    compute_likelihoods_all_models_one_target: File I/O wrapper — all models, one target dataset.
    compute_likelihoods_one_model_all_data: File I/O wrapper — one model, all historical datasets.
"""

import logging
import os

import numpy as np
import pandas as pd
import torch

from omegaconf import DictConfig, OmegaConf

from ..core.model_client import ModelClient
from ..core.model_loading import init_model_client_with_retry
from ..data_contracts import LIK_PREFIX, PARTICLE, SCORE, lik_col
from ..test_functions.finetune_utils import formatting_texts_func_single_seq


CUDA_ERROR = getattr(torch.cuda, "CudaError", RuntimeError)


def _resolve_test_fn_dim(cfg: DictConfig) -> int | None:
    """Read particle dimension from cfg for conditional EOS likelihood scoring."""
    if cfg is not None and cfg.get("test_fn_dim") is not None:
        return int(cfg.test_fn_dim)
    for path in (
        "test_fn.dim",
        "evol_dataset_gen.args.test_function.dim",
    ):
        dim = OmegaConf.select(cfg, path, default=None)
        if dim is not None:
            return int(dim)
    return None


def compute_likelihoods_inmemory(
    target_df: pd.DataFrame,
    input_data_list: list[pd.DataFrame],
    model_name_or_path_list: list[str],
    model_indices: list[int],
    cfg: DictConfig,
    logger: logging.Logger | None = None,
    model_clients: dict[str, ModelClient] | None = None,
) -> pd.DataFrame:
    """Compute likelihoods for target_df under each model, entirely in memory.

    Args:
        target_df: DataFrame with particle/score columns (and possibly existing lik_r* columns).
        input_data_list: List of DataFrames, one per model, containing seed sequences.
        model_name_or_path_list: List of model paths to load for likelihood computation.
        model_indices: List of integer indices identifying each model (for column naming).
        cfg: Hydra config (needs batch_size, generation_config, overwrite_cmp_lik_all).
        logger: Logger instance.
        model_clients: Optional dict mapping model path to pre-loaded ModelClient.
            When provided, skips model loading/unloading to avoid reload churn
            across AR iterations.

    Returns:
        DataFrame with lik_r{i} columns appended for each model index.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    lik_col_names_old = [col for col in target_df.columns if LIK_PREFIX in col]
    lik_col_names_new = [lik_col(m) for m in model_indices]

    logger.info(f"pre selection lik_col_names_old : {lik_col_names_old}")
    logger.info(f"pre selection lik_col_names_new : {lik_col_names_new}")

    if cfg.overwrite_cmp_lik_all:
        logger.info("Overwrite")
        old_not_in_new = [c not in lik_col_names_new for c in lik_col_names_old]
        lik_col_names_old = list(np.array(lik_col_names_old)[old_not_in_new])
        if SCORE in target_df.columns:
            target_df = target_df[[PARTICLE, SCORE] + lik_col_names_old]
        else:
            target_df = target_df[target_df.columns[0:2].tolist() + lik_col_names_old]
    else:
        logger.info("No Overwrite")
        new_not_in_old = [c not in lik_col_names_old for c in lik_col_names_new]
        lik_col_names_new = list(np.array(lik_col_names_new)[new_not_in_old])
        model_indices = list(np.array(model_indices)[new_not_in_old])
        model_name_or_path_list = list(
            np.array(model_name_or_path_list)[new_not_in_old]
        )
        input_data_list = [
            input_data_list[i] for i, keep in enumerate(new_not_in_old) if keep
        ]

    logger.info(f"post selection lik_col_names_old : {lik_col_names_old}")
    logger.info(f"post selection lik_col_names_new : {lik_col_names_new}")
    logger.info(f"post selection model_indices : {model_indices}")
    logger.info(f"post selection model_name_or_path_list : {model_name_or_path_list}")

    # Format target
    target_data = target_df.to_dict("list")
    formatted_targets = formatting_texts_func_single_seq(target_data)
    # logger.info(f"formatted_targets : {formatted_targets}")


    # For each model, compute likelihoods of each target sequence, averaged over seeds
    all_timestep_likelihoods = []
    for i, model_name_or_path in enumerate(model_name_or_path_list):
        # Use cached model client if available, otherwise load fresh
        cached = model_clients is not None and model_name_or_path in model_clients
        if cached:
            model_client = model_clients[model_name_or_path]
            logger.info(f"Using cached ModelClient for {model_name_or_path}")
        else:
            torch.cuda.empty_cache()
            logger.info(f"torch.cuda.is_available() : {torch.cuda.is_available()}")
            model_client = init_model_client_with_retry(
                model_name_or_path,
                cfg.generation_config.max_new_tokens,
                logger,
                temperature=cfg.generation_config.temperature,
            )

        # Format input seeds
        input_df = input_data_list[i]
        input_data = input_df.to_dict("list")
        formatted_inputs = formatting_texts_func_single_seq(input_data)

        # logger.info(f"len(formatted_targets) : {len(formatted_targets)}")

        # logger.info(f"formatted_inputs : {formatted_inputs}")

        # Compute likelihoods
        target_likelihoods = model_client.compute_likelihoods_avg(
            formatted_inputs,
            formatted_targets,
            batch_size=min(cfg.batch_size, len(formatted_targets)),
            logger=logger,
            test_fn_dim=_resolve_test_fn_dim(cfg),
        )
        # logger.info(f"target_likelihoods : {target_likelihoods}")
        all_timestep_likelihoods.append(target_likelihoods)

        # Free model to reclaim GPU memory before loading the next one
        # (skip if using a cached client — caller manages lifecycle)
        if not cached:
            del model_client
            torch.cuda.empty_cache()

    # Assemble result DataFrame — use pandas column assignment instead of np.c_
    # to preserve column dtypes (np.c_ coerces mixed str/float to object)
    if SCORE in target_df.columns:
        keep_cols = [PARTICLE, SCORE] + lik_col_names_old
    else:
        keep_cols = target_df.columns[0:2].tolist() + lik_col_names_old
    target_all_likelihoods_df = target_df[keep_cols].copy()

    if len(lik_col_names_new) > 0 and len(all_timestep_likelihoods) > 0:
        for col_name, liks in zip(lik_col_names_new, all_timestep_likelihoods):
            target_all_likelihoods_df[col_name] = liks
    else:
        logger.info("No new likelihoods to compute, using existing data")

    logger.info(
        f"target_all_likelihoods_df.columns : {target_all_likelihoods_df.columns}"
    )
    return target_all_likelihoods_df


def compute_likelihoods_all_models_one_target(
    cfg: DictConfig, logger: logging.Logger | None = None
) -> None:
    """Compute likelihoods from all models on a single target calibration dataset.

    Loops through all models (and their corresponding input seed prompts) and
    computes the likelihoods output by those models for the most recently
    sampled calibration dataset. Results are saved to disk.

    Args:
        cfg: Hydra config with model_name_or_path_list, input_data_path_list,
            target_data_path, output_dir, output_filename.
        logger: Logger instance.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Here in compute_likelihoods_all_models_one_target")

    n_models = len(cfg.model_name_or_path_list)
    if n_models != len(cfg.input_data_path_list):
        raise ValueError(
            f"Num models ({n_models}) != Num input seed files ({len(cfg.input_data_path_list)})"
        )

    # Load target data
    target_df = pd.read_json(cfg.target_data_path, orient="records", lines=True)

    logger.info(f"target_df : {target_df}")

    model_indices = (
        cfg.model_indices if len(cfg.model_indices) > 0 else list(range(n_models))
    )
    model_name_or_path_list = list(cfg.model_name_or_path_list)
    input_data_path_list = list(cfg.input_data_path_list)

    # Load input DataFrames from file paths
    input_data_list = [
        pd.read_json(fp, orient="records", lines=True) for fp in input_data_path_list
    ]

    output_fp = os.path.join(cfg.output_dir, cfg.output_filename)

    target_all_likelihoods_df = compute_likelihoods_inmemory(
        target_df,
        input_data_list,
        model_name_or_path_list,
        model_indices,
        cfg,
        logger,
    )
    target_all_likelihoods_df.to_json(output_fp, orient="records", lines=True)


def compute_likelihoods_one_model_all_data(
    cfg: DictConfig, logger: logging.Logger | None = None
) -> None:
    """Compute likelihoods from the most recent model on all historical calibration data.

    Backfills likelihood columns: for each previously sampled calibration dataset,
    appends a ``lik_r{n}`` column with the likelihood under the latest model.

    Args:
        cfg: Hydra config with model_name_or_path_list, input_data_path_list,
            prev_target_data_path_list, generation_config, batch_size.
        logger: Logger instance.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if len(cfg.prev_target_data_path_list) == 0:
        logger.info(
            "No previous calibration data given (expected if is first iteration), "
            "skipping 'compute_likelihoods_one_model_all_data'"
        )
        return

    logger.info(f"torch.cuda.is_available()   : {torch.cuda.is_available()}")
    try:
        logger.info(f"torch.cuda.device_count()   : {torch.cuda.device_count()}")
        logger.info(f"torch.cuda.current_device() : {torch.cuda.current_device()}")
    except (RuntimeError, CUDA_ERROR) as e:
        logger.warning(f"Could not get CUDA device info: {e}")

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (RuntimeError, CUDA_ERROR) as e:
        logger.warning(f"Could not clear CUDA cache: {e}")

    model_client = init_model_client_with_retry(
        cfg.model_name_or_path_list[-1],  # Use most recent model
        cfg.generation_config.max_new_tokens,
        logger,
        temperature=cfg.generation_config.temperature,
    )

    # Load most recent input prompt seeds
    input_fp = cfg.input_data_path_list[-1]
    input_df = pd.read_json(input_fp, orient="records", lines=True)
    input_data = input_df.to_dict("list")
    formatted_inputs = formatting_texts_func_single_seq(input_data)

    # Model idx matches the number of previous calibration sets
    # e.g., if 3 prev cal sets {0, 1, 2}, then current model idx is 3
    num_prev_cal = len(cfg.prev_target_data_path_list)

    for i, target_data_path in enumerate(cfg.prev_target_data_path_list):
        # Load target data
        target_df = pd.read_json(target_data_path, orient="records", lines=True)

        if target_df.empty:
            logger.warning(
                f"Skipping empty calibration file (round {i}): {target_data_path}"
            )
            continue

        # Keep only particle/score + existing likelihood columns
        lik_col_names_prev = [lik_col(c) for c in range(num_prev_cal)]
        if SCORE in target_df.columns:
            target_df = target_df[[PARTICLE, SCORE] + lik_col_names_prev]
        else:
            target_df = target_df[target_df.columns[0:2].tolist() + lik_col_names_prev]

        target_data = target_df.to_dict("list")
        formatted_targets = formatting_texts_func_single_seq(target_data)

        # Compute likelihoods
        target_likelihoods = model_client.compute_likelihoods_avg(
            formatted_inputs,
            formatted_targets,
            batch_size=min(cfg.batch_size, len(formatted_targets)),
            logger=logger,
            test_fn_dim=_resolve_test_fn_dim(cfg),
        )

        lik_col_name = [lik_col(num_prev_cal)]

        logger.info(f"target_df.columns : {target_df.columns}")
        logger.info(f"lik_col_name      : {lik_col_name}")

        target_all_likelihoods_df = target_df.copy()
        target_all_likelihoods_df[lik_col_name[0]] = target_likelihoods
        target_all_likelihoods_df.to_json(
            target_data_path, orient="records", lines=True
        )
