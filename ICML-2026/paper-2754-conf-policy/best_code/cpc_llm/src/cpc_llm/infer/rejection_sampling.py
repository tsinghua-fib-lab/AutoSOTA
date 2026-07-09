from __future__ import annotations

import datasets
import functools
import numpy as np
import pandas as pd

import logging
import os

from typing import List
from omegaconf import DictConfig, OmegaConf
from transformers import GenerationConfig

from ..infrastructure.file_handler import LocalOrS3Client
from ..infrastructure.orchestration import (
    run_compute_liks_all_models_and_cal_data,
    run_iterative_generation,
)
from ..calibrate.process_likelihoods import constrain_likelihoods
from ..core.model_client import ModelClient
from ..core.model_loading import (
    init_model_client_with_retry,
    preload_model_clients,
    cleanup_model_clients,
)
from ..core.likelihoods import compute_likelihoods_inmemory
from ..data_contracts import (
    HIGHER_SCORE,
    HIGHER_SCORE_PARTICLE,
    LOWER_SCORE,
    LOWER_SCORE_PARTICLE,
    NUM_PARTICLES_GENERATED,
    PARTICLE,
    SCORE,
    con_lik_col,
    lik_col,
)
from .iterative_generation2 import generate_single_batch
from holo.test_functions.closed_form import Ehrlich, RoughMtFuji
from ..metrics import ARSamplingMetrics, ARSamplingResult, SampleQualityMetrics, compute_sample_quality


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory helpers (used when job_submission_system == "direct")
# ---------------------------------------------------------------------------


def _is_direct_mode(cfg: DictConfig) -> bool:
    return getattr(cfg, "job_submission_system", "slurm") == "direct"


@functools.lru_cache(maxsize=32)
def _read_jsonl_cached(filepath: str) -> pd.DataFrame:
    """Read a JSONL file, caching the result across AR iterations."""
    return pd.read_json(filepath, orient="records", lines=True)


def _load_test_fn(ga_data_dir: str) -> Ehrlich | RoughMtFuji:
    """Load the test function from the ehrlich.jsonl parameter file.

    Args:
        ga_data_dir: Directory containing the ehrlich.jsonl parameter file.

    Returns:
        An Ehrlich or RoughMtFuji test function instance.
    """
    test_fn_fp = os.path.join(ga_data_dir, "ehrlich.jsonl")
    params = pd.read_json(test_fn_fp, orient="records", lines=True).to_dict("records")[
        0
    ]
    if "num_motifs" in params:
        return Ehrlich(
            num_states=params["num_states"],
            dim=params["dim"],
            num_motifs=params["num_motifs"],
            motif_length=params["motif_length"],
            quantization=params["quantization"],
            noise_std=params["noise_std"],
            negate=params["negate"],
            random_seed=params["random_seed"],
        )
    return RoughMtFuji(
        dim=params["dim"],
        additive_term=params["additive_term"],
        random_term_std=params["random_term_std"],
        noise_std=params["noise_std"],
        negate=params["negate"],
        random_seed=params["random_seed"],
    )


# _init_model_client_with_retry is now in core.model_loading; alias for
# backward compatibility with any callers that import the private name.
_init_model_client_with_retry = init_model_client_with_retry


def generate_sample_batch(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    seeds_fp: str,
    ga_data_dir: str,
    model_dir: str,
    output_dir: str,
    temp: float,
    model_idx: int = 0,
    higher_score_particle_field: str = HIGHER_SCORE_PARTICLE,
    lower_score_particle_field: str = LOWER_SCORE_PARTICLE,
    higher_score_field: str = HIGHER_SCORE,
    lower_score_field: str = LOWER_SCORE,
    random_seed: int = 0,
    call_idx: int = 0,
    global_random_seed: int = 0,
    n_target: int | None = None,
    _model_client: ModelClient | None = None,
    _test_fn: Ehrlich | RoughMtFuji | None = None,
    post_policy_control: bool = False, ## Whether calling post CPC (True) or pre CPC (False)
) -> pd.DataFrame | None:
    """Generate samples from a model, with optional early stopping.

    In in-memory mode, generates one batch at a time via generate_single_batch,
    looping up to max_iterations and stopping early when n_target valid samples
    have been collected. In subprocess mode, delegates to run_iterative_generation.

    Args:
        cfg: Pipeline config.
        fs: File handler (local or S3).
        seeds_fp: Path to the JSONL seeds file.
        ga_data_dir: Directory containing test function parameters.
        model_dir: Path to the model to generate from.
        output_dir: Output directory for subprocess mode.
        temp: Generation temperature.
        model_idx: Index of model to generate from.
        higher_score_particle_field: Column name for higher-score particles.
        lower_score_particle_field: Column name for lower-score particles.
        higher_score_field: Column name for higher scores.
        lower_score_field: Column name for lower scores.
        random_seed: Random seed for reproducible seed selection.
        call_idx: Call index for subprocess filename construction.
        global_random_seed: Global random seed for subprocess mode.
        n_target: If provided, stop generation early once this many valid
            samples have been collected (in-memory mode only).
        _model_client: Pre-loaded ModelClient (required for in-memory mode).
        _test_fn: Pre-loaded test function (required for in-memory mode).

    Returns:
        DataFrame with columns: particle, score, num_particles_generated.
        Returns None if generation produced no samples.
    """
    gen_fp = None
    if _is_direct_mode(cfg):
        ## Mirror the SLURM-mode output filename built in run_iterative_generation,
        ## so that direct mode checkpoints (and restarts) the same way.
        output_filename_prefix = (
            f"alpha{cfg.conformal_policy_control.alpha}_postPC{post_policy_control}"
            f"_model{model_idx}_cn{call_idx}_gens_likelihood_"
            f"{cfg.iterative_generation.args.sample_size}sample_"
            f"{cfg.iterative_generation.args.max_iterations}iter"
        )
        gen_fp = os.path.join(
            output_dir,
            f"{output_filename_prefix}_temp{temp}_"
            f"{cfg.generation_sampling_num_return_sequences}seqs.jsonl",
        )
        if not cfg.overwrite_ig and fs.exists(gen_fp):
            logger.info(f"{gen_fp} already exists. Loading instead of regenerating...")
            try:
                gen_df = pd.read_json(gen_fp, orient="records", lines=True)
            except ValueError:
                ## Empty/corrupt checkpoint file: treat as a round with no samples
                gen_df = pd.DataFrame()
            if len(gen_df) == 0:
                return None, gen_fp
            return gen_df, gen_fp

        gen_df = _generate_inmemory(
            cfg,
            seeds_fp,
            _model_client,
            _test_fn,
            temp,
            higher_score_particle_field,
            lower_score_particle_field,
            higher_score_field,
            lower_score_field,
            random_seed,
            n_target=n_target,
        )
        ## Persist generated proposals so a restarted run resumes with the same samples
        gen_df.to_json(gen_fp, orient="records", lines=True)
    else:
        _, iter_gen_outputs_list, _hd = run_iterative_generation(
            cfg,
            fs,
            seeds_fp,
            ga_data_dir,
            model_dir,
            output_dir,
            higher_score_particle_field=higher_score_particle_field,
            lower_score_particle_field=lower_score_particle_field,
            higher_score_field=higher_score_field,
            lower_score_field=lower_score_field,
            temps=[temp],
            model_idx=model_idx,
            call_idx=call_idx,
            post_policy_control=post_policy_control,
            global_random_seed=global_random_seed,
        )
        gen_fp = iter_gen_outputs_list[-1]
        gen_df = pd.read_json(gen_fp, orient="records", lines=True)

    if len(gen_df) == 0:
        return None, gen_fp
    return gen_df, gen_fp


def compute_batch_likelihoods(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    target_df: pd.DataFrame,
    seeds_fp_list: List[str],
    model_dir_list: List[str],
    model_indices: List[int] | None = None,
    target_fp: str | None = None,
    _lik_model_clients: dict[str, ModelClient] | None = None,
) -> pd.DataFrame:
    """Compute likelihoods for a batch of samples under all models.

    Handles both in-memory (direct) and subprocess (SLURM) execution modes.

    Args:
        cfg: Pipeline config.
        fs: File handler (local or S3).
        target_df: DataFrame of generated samples (particle, score, ...).
        seeds_fp_list: List of seed filepaths, one per model.
        model_dir_list: List of model directory paths.
        model_indices: Indices for likelihood column naming (lik_r0, lik_r1, ...).
            Defaults to range(len(model_dir_list)).
        target_fp: Filepath of target data on disk (required for subprocess mode).
        _lik_model_clients: Pre-loaded model clients for in-memory mode.

    Returns:
        DataFrame with lik_r0, lik_r1, ... columns appended.
    """
    if model_indices is None:
        model_indices = list(range(len(model_dir_list)))

    if _is_direct_mode(cfg):
        ## Mirror SLURM-mode checkpointing: if the target file on disk already
        ## contains likelihood columns for all requested models, load it instead
        ## of recomputing.
        if target_fp and not cfg.overwrite_cmp_lik_all and fs.exists(target_fp):
            try:
                target_on_disk = pd.read_json(target_fp, orient="records", lines=True)
            except ValueError:
                target_on_disk = pd.DataFrame()
            if len(target_on_disk) > 0 and all(
                lik_col(m) in target_on_disk.columns for m in model_indices
            ):
                logger.info(
                    f"target_fp {target_fp} already exists with likelihoods computed. "
                    "Skipping likelihoods computation..."
                )
                return target_on_disk

        liks_df = _compute_liks_all_models_inmemory(
            cfg,
            target_df,
            seeds_fp_list,
            model_dir_list,
            model_indices,
            _lik_model_clients=_lik_model_clients,
        )
        ## Persist likelihoods to the target file (as the SLURM subprocess does)
        ## so a restarted run can skip recomputation.
        if target_fp:
            liks_df.to_json(target_fp, orient="records", lines=True)
        return liks_df
    else:
        gen_liks_fp_list, _hd = run_compute_liks_all_models_and_cal_data(
            cfg,
            fs,
            seeds_fp_list=seeds_fp_list,
            prev_cal_data_fp_list=[],
            model_dir_list=model_dir_list,
            target_fp=target_fp,
            model_indices=model_indices,
            temps=[cfg.temperature],
        )
        return pd.read_json(gen_liks_fp_list[-1], orient="records", lines=True)


def _generate_inmemory(
    cfg: DictConfig,
    seeds_fp: str,
    model_client: ModelClient,
    test_fn: Ehrlich | RoughMtFuji,
    temp: float,
    higher_score_particle_field: str,
    lower_score_particle_field: str,
    higher_score_field: str,
    lower_score_field: str,
    random_seed: int,
    n_target: int | None = None,
) -> pd.DataFrame:
    """Read seeds, select, and generate samples entirely in memory.

    Calls generate_single_batch in a loop up to max_iterations times,
    stopping early if n_target valid samples have been collected.
    """
    df = _read_jsonl_cached(seeds_fp)

    if cfg.sanity_check:
        logger.info(
            "Running in sanity check mode... reducing data down to 10 examples."
        )
        df = df.sample(n=min(10, len(df)))

    sample_size = cfg.iterative_generation.args.sample_size
    sampling_method = cfg.iterative_generation.args.sampling_method

    if higher_score_field in df.columns and lower_score_field in df.columns:
        logger.info(f"sample_size : {sample_size}")
        if sampling_method == "best_scoring":
            logger.info("sampling_method : best_scoring")
            df = df.sort_values(by=[lower_score_field], ascending=True)[:sample_size]
        elif sampling_method == "uniform":
            logger.info("sampling_method : uniform")
            df = df.sample(n=min(len(df), sample_size), random_state=random_seed)
        elif sampling_method == "combination":
            half = int(sample_size / 2)
            df = pd.concat(
                [
                    df.sort_values(by=[lower_score_field], ascending=True)[:half],
                    df.sample(n=min(len(df), half), random_state=random_seed),
                ]
            )
        else:
            raise ValueError(f"Unknown sampling method '{sampling_method}'")

        ds = datasets.Dataset.from_pandas(df)
        input_ds = datasets.Dataset.from_list(
            [
                {
                    higher_score_particle_field: ex[lower_score_particle_field],
                    SCORE: ex[lower_score_field],
                }
                for ex in ds
            ]
        )
    else:
        input_ds = datasets.Dataset.from_pandas(df)

    gen_config = GenerationConfig(
        do_sample=True,
        num_beams=1,
        temperature=temp,
        num_return_sequences=cfg.generation_sampling_num_return_sequences,
        max_new_tokens=cfg.iterative_generation.args.generation_config.max_new_tokens,
    )

    gen_cfg = OmegaConf.create(
        {
            "batch_size": cfg.sampling_gen_batch_size,
            "subsample_seeds": cfg.iterative_generation.args.subsample_seeds,
            "permissive_parsing": cfg.iterative_generation.permissive_parsing,
            "higher_score_particle_field": higher_score_particle_field,
            "lower_score_particle_field": lower_score_particle_field,
            "min_rel_len_feasible_particle": cfg.iterative_generation.min_rel_len_feasible_particle,
        }
    )

    max_iterations = cfg.iterative_generation.args.max_iterations
    all_dfs = []
    n_collected = 0
    for _iter in range(max_iterations):
        batch_df = generate_single_batch(
            input_ds, model_client, test_fn, gen_config, gen_cfg, logger
        )
        if len(batch_df) > 0:
            all_dfs.append(batch_df)
            n_collected += len(batch_df)

        if n_target is not None and n_collected >= n_target:
            logger.info(
                f"Early exit at iteration {_iter + 1}/{max_iterations}: "
                f"{n_collected} >= {n_target} target samples"
            )
            break

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame(columns=[PARTICLE, SCORE, NUM_PARTICLES_GENERATED])


def _compute_liks_all_models_inmemory(
    cfg: DictConfig,
    target_df: pd.DataFrame,
    seeds_fp_list: List[str],
    model_dir_list: List[str],
    model_indices: List[int],
    _lik_model_clients: dict[str, ModelClient] | None = None,
) -> pd.DataFrame:
    """Compute likelihoods for all models in memory, without subprocesses."""
    input_data_list = [_read_jsonl_cached(fp) for fp in seeds_fp_list]

    # batch_size lives at the top level of the Hydra subprocess config
    # (compute_liks_all_models_and_cal_data.yaml), not in the pipeline
    # config's args sub-dict.  Default is 10.
    lik_batch_size = getattr(cfg.compute_likelihooods_all_models.args, "batch_size", 10)
    test_fn_dim = OmegaConf.select(
        cfg, "evol_dataset_gen.args.test_function.dim", default=None
    )
    lik_cfg = OmegaConf.create(
        {
            "batch_size": lik_batch_size,
            "generation_config": {
                "temperature": cfg.temperature,
                "max_new_tokens": cfg.compute_likelihooods_all_models.args.generation_config.max_new_tokens,
            },
            "overwrite_cmp_lik_all": cfg.overwrite_cmp_lik_all,
            "test_fn_dim": test_fn_dim,
        }
    )

    return compute_likelihoods_inmemory(
        target_df,
        input_data_list,
        list(model_dir_list),
        list(model_indices),
        lik_cfg,
        logger,
        model_clients=_lik_model_clients,
    )


# ---------------------------------------------------------------------------
# Main AR-sampling function
# ---------------------------------------------------------------------------


def accept_reject_sample_and_get_likelihoods(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    model_dir_list: List[str],
    seeds_fp_list: List[str],
    output_dir: str,
    betas_list: List[float],
    psis_list: List[float],  ## Normalization constants
    n_target: int,
    ga_data_dir: str,
    temps: List[float] = [1.0],
    depth: int = 0,  ## Recursion depth
    higher_score_particle_field: str = HIGHER_SCORE_PARTICLE,
    lower_score_particle_field: str = LOWER_SCORE_PARTICLE,
    higher_score_field: str = HIGHER_SCORE,
    lower_score_field: str = LOWER_SCORE,
    proposal: str = None,  ## Proposal distribution (safe or unconstrained), or None --> means running for filtering
    post_policy_control: bool = False,  ## Whether calling post policy control (True <--> generating risk-controlled actions), or pre control (False <--> Generating proposals)
    safe_prop_mix_weight: float = 1.0,  ## if proposal == "mixture":  weight in (0, 1) to assign to safe proposal
    env_const: float = 1.0,  ## Recalculated envelope constant,
    global_random_seed: int = 0.0,
) -> ARSamplingResult:

    n_models = len(model_dir_list)
    use_inmemory = _is_direct_mode(cfg)

    # accepted = [] ## List containing indicators for whether each considered proposal sample is accepted
    n_accepted = 0

    ## Initialize data frames for storing data for accepted samples
    unconstrained_lik_cols = [lik_col(i) for i in range(n_models)]
    unconstrained_col_names = [PARTICLE, SCORE] + unconstrained_lik_cols
    constrained_lik_cols = [con_lik_col(i) for i in range(n_models)]
    constrained_col_names = [PARTICLE, SCORE] + constrained_lik_cols

    accepted_unconstrained_dfs = []
    accepted_constrained_dfs = []

    # Metrics tracking
    all_proposal_dfs: list[pd.DataFrame] = []
    n_proposals_total = 0
    imh_switched = False
    imh_switch_call_idx: int | None = None

    call_idx = 0

    # ---- Pre-load models for in-memory mode ----
    gen_model_client: ModelClient | dict[str, ModelClient] | None = None
    test_fn = None
    lik_model_clients: dict[str, ModelClient] = {}
    if use_inmemory:
        test_fn = _load_test_fn(ga_data_dir)
        gen_model_client, lik_model_clients = preload_model_clients(
            cfg, model_dir_list, proposal
        )

    if proposal == "unconstrained":
        ## If beta_t >= 1, then using unconstrained policy as proposal

        unconstrained_pre_cpc_call_num_check = True

        if cfg.conformal_policy_control.alpha >= 1.0:
            unconstrained_pre_cpc_call_num_check = (
                True  ## Set to True if running unconstrained
            )

        while (
            n_accepted < n_target
            and unconstrained_pre_cpc_call_num_check
            and call_idx < cfg.conformal_policy_control.accept_reject.max_total_AR_calls
        ):
            ## If pre conformal policy control, running constrained model (ie, not alpha>=1.0), also check that, if the specified number of calls have occurred, if at least 1/4 way to n_target_pre_cpc
            if (
                not post_policy_control
                and cfg.conformal_policy_control.alpha < 1.0
                and call_idx
                >= cfg.conformal_policy_control.accept_reject.n_opt_prop_calls_pre_cpc_quarter_check
                and n_accepted < n_target / 4
            ):
                unconstrained_pre_cpc_call_num_check = False
                break

            accepted_curr = []

            ## Step 1: Generate from latest model
            gen_liks_fp = os.path.join(output_dir, "liks.jsonl")
            random_seed_curr = (
                global_random_seed * 10000 + post_policy_control * 1000 + call_idx
            )
            temps_curr = temps if len(model_dir_list) > 1 else [cfg.temperature_init]
            gen_df, gen_fp = generate_sample_batch(
                cfg,
                fs,
                seeds_fp_list[-1],
                ga_data_dir,
                model_dir_list[-1],
                output_dir,
                temps_curr[-1],
                model_idx=len(model_dir_list) - 1,
                higher_score_particle_field=higher_score_particle_field,
                lower_score_particle_field=lower_score_particle_field,
                higher_score_field=higher_score_field,
                lower_score_field=lower_score_field,
                random_seed=random_seed_curr,
                call_idx=call_idx,
                global_random_seed=global_random_seed,
                _model_client=gen_model_client,
                _test_fn=test_fn,
                post_policy_control=post_policy_control,
            )
            call_idx += 1

            if gen_df is None:
                continue

            ## Step 2: Compute likelihoods under all models
            gen_liks_df = compute_batch_likelihoods(
                cfg,
                fs,
                gen_df,
                seeds_fp_list,
                model_dir_list,
                target_fp=gen_fp,
                _lik_model_clients=lik_model_clients or None,
            )
            gen_liks_df = gen_liks_df[unconstrained_col_names]
            gen_liks_mat = gen_liks_df[unconstrained_lik_cols].to_numpy()

            ## Step 3: Constrain likelihoods
            if cfg.conformal_policy_control.constrain_against == "init":
                constrained_liks_mat = np.zeros(np.shape(gen_liks_mat))
                constrained_liks_mat[:, 0] = gen_liks_mat[:, 0]
                for c in range(1, n_models):
                    constrained_liks_mat[:, c] = constrain_likelihoods(
                        cfg,
                        gen_liks_mat[:, [0, c]],
                        [betas_list[0], betas_list[c]],
                        [psis_list[0], psis_list[c]],
                    )[:, -1]
                safe_liks = constrained_liks_mat[:, 0]
            else:
                constrained_liks_mat = constrain_likelihoods(
                    cfg, gen_liks_mat, betas_list, psis_list
                )
                if constrained_liks_mat.shape[1] > 1:
                    safe_liks = constrained_liks_mat[:, -2]
                else:
                    safe_liks = constrained_liks_mat[:, -1]

            unconstrained_liks = gen_liks_mat[:, -1]
            lik_ratios_unconstrained_over_safe = unconstrained_liks / safe_liks
            constrained_liks_df = pd.concat(
                [
                    gen_liks_df[[PARTICLE, SCORE]],
                    pd.DataFrame(constrained_liks_mat, columns=constrained_lik_cols),
                ],
                axis=1,
            )

            n_prop = len(gen_liks_df)
            n_proposals_total += n_prop

            ## Accept or reject each proposal

            ## Arbitrary way of standardizing random seeds so that is consistent when rerunning from checkpoint (but uses different random seed for each call)
            ar_random_seed = call_idx if not post_policy_control else 1000 + call_idx
            np.random.seed(ar_random_seed)

            ## Initialize MH state from first proposal
            target_lik = min(unconstrained_liks[0], betas_list[-1] * safe_liks[0])
            prop_lik = unconstrained_liks[0]

            batch_acc_probs = []
            for i in range(n_prop):
                u = np.random.uniform()

                ## Initial state for MH sampling
                if n_accepted == 0:
                    ## Keep updating the initial state until first acceptance
                    prev_target_lik = target_lik
                    prev_prop_lik = prop_lik

                ## Current target and proposal likelihoods (up to normalizing constant)
                target_lik = min(unconstrained_liks[i], betas_list[-1] * safe_liks[i])
                prop_lik = unconstrained_liks[i]

                is_imh = post_policy_control and (
                    cfg.conformal_policy_control.ind_metropolis_hastings
                    or call_idx > cfg.conformal_policy_control.num_AR_before_MH
                )
                if is_imh:
                    if not imh_switched:
                        imh_switched = True
                        imh_switch_call_idx = call_idx
                    ## Conditions for running IMH
                    acc_prob = min(
                        1, (target_lik / prev_target_lik) * (prev_prop_lik / prop_lik)
                    )

                else:
                    ## Condition for rejection sampling or until first acceptance of MH sampling
                    acc_prob = min(
                        1, betas_list[-1] / lik_ratios_unconstrained_over_safe[i]
                    )

                batch_acc_probs.append(acc_prob)

                if u < acc_prob:
                    ## Update target and proposal likelihoods for last accepted sample
                    prev_target_lik = target_lik
                    prev_prop_lik = prop_lik

                    accepted_curr.append(True)
                    n_accepted += 1

                    if n_accepted >= n_target:
                        break

                else:
                    accepted_curr.append(False)

            # Collect all proposals from this batch with acc_prob and accepted
            n_evaluated = len(accepted_curr)
            batch_df = gen_liks_df[:n_evaluated].copy()
            batch_df["acc_prob"] = batch_acc_probs[:n_evaluated]
            batch_df["accepted"] = accepted_curr
            all_proposal_dfs.append(batch_df)

            accepted_unconstrained_curr_df = gen_liks_df[:n_evaluated][accepted_curr]
            accepted_constrained_curr_df = constrained_liks_df[:n_evaluated][accepted_curr]

            accepted_unconstrained_dfs.append(accepted_unconstrained_curr_df)
            accepted_constrained_dfs.append(accepted_constrained_curr_df)


            logger.info(f"Accepted {sum(accepted_curr)} samples in current sampling round for {proposal} proposal")
            ## Record samples accepted in current sampling round for tracking acceptance progress
            if sum(accepted_curr) > 0:
                base_output_name = f"alpha{cfg.conformal_policy_control.alpha}_gens_likelihood_{cfg.iterative_generation.args.sample_size}sample_{cfg.iterative_generation.args.max_iterations}iter_temp{temps[-1]}_{cfg.generation_sampling_num_return_sequences}seqs.jsonl"

                curr_accepted_unconstrained_fp = os.path.join(
                    os.path.dirname(gen_liks_fp),
                    f"curr_accepted_uLiks_cn{call_idx}_{proposal}Prop_{base_output_name}",
                )
                curr_accepted_constrained_fp = os.path.join(
                    os.path.dirname(gen_liks_fp),
                    f"curr_accepted_cLiks_cn{call_idx}_{proposal}Prop_{base_output_name}",
                )

                if cfg.overwrite_ig or not fs.exists(curr_accepted_unconstrained_fp):
                    accepted_unconstrained_curr_df.to_json(
                        curr_accepted_unconstrained_fp, orient="records", lines=True
                    )

                if cfg.overwrite_ig or not fs.exists(curr_accepted_constrained_fp):
                    accepted_constrained_curr_df.to_json(
                        curr_accepted_constrained_fp, orient="records", lines=True
                    )

            ## If still no samples accepted after max(20% of planned AR calls, 1), go ahead and switch to IMH sampling
            if n_accepted == 0 and call_idx >= max(0.2 * cfg.conformal_policy_control.num_AR_before_MH, 1):
                call_idx = cfg.conformal_policy_control.num_AR_before_MH
    
    elif proposal == "safe":
        ## Else, beta_t < 1, then using safe policy as proposal

        while (
            n_accepted < n_target
            and call_idx < cfg.conformal_policy_control.accept_reject.max_total_AR_calls
        ):
            accepted_curr = []

            ## Sample using unconstrained model as proposal

            if cfg.conformal_policy_control.constrain_against == "init":
                ## Step 1: Generate from initial safe model
                gen_liks_fp = os.path.join(output_dir, "liks.jsonl")
                random_seed_curr = (
                    global_random_seed * 10000 + post_policy_control * 1000 + call_idx
                )
                gen_df, gen_fp = generate_sample_batch(
                    cfg,
                    fs,
                    seeds_fp_list[0],
                    ga_data_dir,
                    model_dir_list[0],
                    output_dir,
                    cfg.temperature_init,
                    model_idx=0,
                    higher_score_particle_field=higher_score_particle_field,
                    lower_score_particle_field=lower_score_particle_field,
                    higher_score_field=higher_score_field,
                    lower_score_field=lower_score_field,
                    random_seed=random_seed_curr,
                    call_idx=call_idx,
                    global_random_seed=global_random_seed,
                    _model_client=gen_model_client,
                    _test_fn=test_fn,
                    post_policy_control=post_policy_control,
                )
                call_idx += 1

                if gen_df is None:
                    continue

                ## Step 2: Compute likelihoods under all models
                gen_liks_df = compute_batch_likelihoods(
                    cfg,
                    fs,
                    gen_df,
                    seeds_fp_list,
                    model_dir_list,
                    target_fp=gen_fp,
                    _lik_model_clients=lik_model_clients or None,
                )
                gen_liks_df = gen_liks_df[unconstrained_col_names]
                gen_liks_mat = gen_liks_df[unconstrained_lik_cols].to_numpy()

                ## Step 3: Constrain likelihoods
                constrained_liks_mat = np.zeros(np.shape(gen_liks_mat))
                constrained_liks_mat[:, 0] = gen_liks_mat[:, 0]
                for c in range(1, n_models):
                    constrained_liks_mat[:, c] = constrain_likelihoods(
                        cfg,
                        gen_liks_mat[:, [0, c]],
                        [betas_list[0], betas_list[c]],
                        [psis_list[0], psis_list[c]],
                    )[:, -1]

                unconstrained_liks = gen_liks_mat[:, -1]
                safe_liks = constrained_liks_mat[:, 0]
                lik_ratios_unconstrained_over_safe = unconstrained_liks / safe_liks
                constrained_liks_df = pd.concat(
                    [
                        gen_liks_df[[PARTICLE, SCORE]],
                        pd.DataFrame(
                            constrained_liks_mat, columns=constrained_lik_cols
                        ),
                    ],
                    axis=1,
                )

            else:
                ## Shouldn't need this, handled in base case of recursion
                # temps_curr = temps if len(model_dir_list) - 1 > 1 else [cfg.temperature_init] ## If model list after removing one model is not just initial, use temps

                ## Sample using unconstrained model as proposal
                _recursive_result = accept_reject_sample_and_get_likelihoods(
                    cfg,
                    fs,
                    model_dir_list[:-1],
                    seeds_fp_list[:-1],
                    output_dir,
                    betas_list[:-1],
                    psis_list[:-1],  ## Normalization constants
                    n_target,
                    ga_data_dir,
                    temps,
                    depth + 1,
                    higher_score_particle_field=higher_score_particle_field,
                    lower_score_particle_field=lower_score_particle_field,
                    higher_score_field=higher_score_field,
                    lower_score_field=lower_score_field,
                    proposal="safe",
                    global_random_seed=global_random_seed,
                )
                gen_liks_tmin1_df = _recursive_result.unconstrained_df
                gen_liks_tmin1_fp = _recursive_result.unconstrained_fp
                constrained_gen_liks_tmin1_df = _recursive_result.constrained_df
                _constrained_gen_liks_tmin1_fp = _recursive_result.constrained_fp  # noqa: F841
                call_idx += 1

                ## Compute likelihoods under the most recent model
                gen_liks_df = compute_batch_likelihoods(
                    cfg,
                    fs,
                    gen_liks_tmin1_df,
                    [seeds_fp_list[-1]],
                    [model_dir_list[-1]],
                    model_indices=[len(model_dir_list) - 1],
                    target_fp=gen_liks_tmin1_fp,
                    _lik_model_clients=lik_model_clients or None,
                )

                gen_liks_df = gen_liks_df[unconstrained_col_names]
                gen_liks_mat = gen_liks_df[
                    unconstrained_lik_cols
                ].to_numpy()  ## Shape (n_prop, n_models)

                gen_liks_df_t0_safe_and_t_unconstrained_mat = pd.concat(
                    [
                        constrained_gen_liks_tmin1_df.iloc[:, -1],
                        gen_liks_df.iloc[:, -1],
                    ],
                    axis=1,
                ).to_numpy()  ## Double check this

                constrained_liks_mat = constrain_likelihoods(
                    cfg,
                    gen_liks_df_t0_safe_and_t_unconstrained_mat,
                    betas_list[-2:],
                    psis_list[-2:],
                )

                if constrained_liks_mat.shape[1] > 1:
                    ## If is not original safe model, \pi_{\theta_0}
                    lik_ratios_unconstrained_over_safe = (
                        gen_liks_mat[:, -1] / constrained_liks_mat[:, -2]
                    )
                else:
                    ## Else is original safe model, \pi_{\theta_0}, so unconstrained and constrained likelihoods are the same
                    ## (Lik ratios should be == 1, and bound == inf, so should accept everything)
                    lik_ratios_unconstrained_over_safe = (
                        gen_liks_mat[:, -1] / constrained_liks_mat[:, -1]
                    )

                constrained_liks_df_ = pd.DataFrame(
                    constrained_liks_mat, columns=constrained_lik_cols[-2:]
                )
                constrained_liks_df = pd.concat(
                    [constrained_gen_liks_tmin1_df, constrained_liks_df_.iloc[:, -1]],
                    axis=1,
                )

            n_prop = len(gen_liks_df)
            n_proposals_total += n_prop

            ## Arbitrary way of standardizing random seeds so that is consistent when rerunning from checkpoint (but uses different random seed for each call)
            ar_random_seed = call_idx if not post_policy_control else 1000 + call_idx
            np.random.seed(ar_random_seed)

            ## Accept or reject each proposal
            batch_acc_probs = []
            for i in range(n_prop):
                u = np.random.uniform()

                ## Initial state for MH sampling
                if n_accepted == 0:
                    if i == 0:
                        prev_target_lik = min(
                            unconstrained_liks[i], betas_list[-1] * safe_liks[i]
                        )  # / psis_list[-1]
                        prev_prop_lik = safe_liks[i]

                    else:
                        ## Can keep updating the initial state until first acceptance
                        prev_target_lik = target_lik
                        prev_prop_lik = prop_lik

                ## Current target and proposal likelihoods (up to normalizing constant)
                target_lik = min(
                    unconstrained_liks[i], betas_list[-1] * safe_liks[i]
                )  # / psis_list[-1]
                prop_lik = safe_liks[i]

                is_imh = post_policy_control and (
                    cfg.conformal_policy_control.ind_metropolis_hastings
                    or call_idx > cfg.conformal_policy_control.num_AR_before_MH
                )
                if is_imh:
                    if not imh_switched:
                        imh_switched = True
                        imh_switch_call_idx = call_idx
                    ## Conditions for running IMH
                    acc_prob = min(
                        1, (target_lik / prev_target_lik) * (prev_prop_lik / prop_lik)
                    )

                else:
                    ## Condition for rejection sampling or until first acceptance of MH sampling
                    acc_prob = min(
                        1, lik_ratios_unconstrained_over_safe[i] / betas_list[-1]
                    )

                batch_acc_probs.append(acc_prob)

                if u < acc_prob:
                    ## Update target and proposal likelihoods for last accepted sample
                    prev_target_lik = target_lik
                    prev_prop_lik = prop_lik

                    accepted_curr.append(True)
                    n_accepted += 1

                    if n_accepted >= n_target:
                        break

                else:
                    accepted_curr.append(False)

            # Collect all proposals from this batch
            n_evaluated = len(accepted_curr)
            batch_df = gen_liks_df[:n_evaluated].copy()
            batch_df["acc_prob"] = batch_acc_probs[:n_evaluated]
            batch_df["accepted"] = accepted_curr
            all_proposal_dfs.append(batch_df)

            accepted_unconstrained_curr_df = gen_liks_df[:n_evaluated][accepted_curr]
            accepted_constrained_curr_df = constrained_liks_df[:n_evaluated][accepted_curr]

            accepted_unconstrained_dfs.append(accepted_unconstrained_curr_df)
            accepted_constrained_dfs.append(accepted_constrained_curr_df)


            logger.info(f"Accepted {sum(accepted_curr)} samples in current sampling round for {proposal} proposal")
            ## Record samples accepted in current sampling round for tracking acceptance progress
            if sum(accepted_curr) > 0:
                base_output_name = f"alpha{cfg.conformal_policy_control.alpha}_gens_likelihood_{cfg.iterative_generation.args.sample_size}sample_{cfg.iterative_generation.args.max_iterations}iter_temp{temps[-1]}_{cfg.generation_sampling_num_return_sequences}seqs.jsonl"

                curr_accepted_unconstrained_fp = os.path.join(
                    os.path.dirname(gen_liks_fp),
                    f"curr_accepted_uLiks_cn{call_idx}_{proposal}Prop_{base_output_name}",
                )
                curr_accepted_constrained_fp = os.path.join(
                    os.path.dirname(gen_liks_fp),
                    f"curr_accepted_cLiks_cn{call_idx}_{proposal}Prop_{base_output_name}",
                )

                if cfg.overwrite_ig or not fs.exists(curr_accepted_unconstrained_fp):
                    accepted_unconstrained_curr_df.to_json(
                        curr_accepted_unconstrained_fp, orient="records", lines=True
                    )

                if cfg.overwrite_ig or not fs.exists(curr_accepted_constrained_fp):
                    accepted_constrained_curr_df.to_json(
                        curr_accepted_constrained_fp, orient="records", lines=True
                    )
                    
            ## If still no samples accepted after max(20% of planned AR calls, 1) of planned AR calls, go ahead and switch to IMH sampling
            if n_accepted == 0 and call_idx >= max(0.2 * cfg.conformal_policy_control.num_AR_before_MH, 1):
                call_idx = cfg.conformal_policy_control.num_AR_before_MH

    elif proposal == "mixture":
        accepted_curr_dict: dict[str, list[bool]] = {"safe": [], "unconstrained": []}
        mix_acc_probs_dict: dict[str, list[float]] = {"safe": [], "unconstrained": []}

        n_proposed_dict = {"safe": 0, "unconstrained": 0}
        N_prop_dict = {"safe": 0, "unconstrained": 0}

        (
            gen_liks_df_dict,
            gen_liks_fp_dict,
            constrained_liks_df_dict,
            lik_ratios_unconstrained_over_safe_dict,
        ) = {}, {}, {}, {}
        unconstrained_liks_dict, safe_liks_dict = {}, {}

        while (
            n_accepted < n_target
            and call_idx < cfg.conformal_policy_control.accept_reject.max_total_AR_calls
        ):
            for proposal_curr in ["safe", "unconstrained"]:
                ## If have already proposed the number of total proposals, then redraw samples for that policy:
                if n_proposed_dict[proposal_curr] >= N_prop_dict[proposal_curr]:
                    # Select model and seeds based on sub-proposal type
                    mix_model_client = None
                    if use_inmemory and isinstance(gen_model_client, dict):
                        mix_model_client = gen_model_client[proposal_curr]
                    if proposal_curr == "unconstrained":
                        mix_seeds_fp = seeds_fp_list[-1]
                        mix_model_dir = model_dir_list[-1]
                        curr_model_idx = len(model_dir_list) - 1
                        mix_temp = (
                            temps[-1]
                            if len(model_dir_list) > 1
                            else cfg.temperature_init
                        )
                    else:
                        mix_seeds_fp = seeds_fp_list[0]
                        mix_model_dir = model_dir_list[0]
                        curr_model_idx = 0
                        mix_temp = cfg.temperature_init

                    ## Step 1: Generate
                    gen_liks_fp_dict[proposal_curr] = os.path.join(
                        output_dir, "liks.jsonl"
                    )
                    random_seed_curr = (
                        global_random_seed * 10000
                        + post_policy_control * 1000
                        + call_idx
                    )
                    mix_gen_df, gen_fp = generate_sample_batch(
                        cfg,
                        fs,
                        mix_seeds_fp,
                        ga_data_dir,
                        mix_model_dir,
                        output_dir,
                        mix_temp,
                        model_idx=curr_model_idx,
                        higher_score_particle_field=higher_score_particle_field,
                        lower_score_particle_field=lower_score_particle_field,
                        higher_score_field=higher_score_field,
                        lower_score_field=lower_score_field,
                        random_seed=random_seed_curr,
                        call_idx=call_idx,
                        global_random_seed=global_random_seed,
                        _model_client=mix_model_client,
                        _test_fn=test_fn,
                        post_policy_control=post_policy_control,
                    )
                    call_idx += 1

                    if mix_gen_df is None:
                        gen_liks_df_dict[proposal_curr] = None
                        ## If no proposals generated, break out of for loop
                        break

                    ## Step 2: Compute likelihoods under all models
                    gen_liks_df_dict[proposal_curr] = compute_batch_likelihoods(
                        cfg,
                        fs,
                        mix_gen_df,
                        seeds_fp_list,
                        model_dir_list,
                        target_fp=gen_fp,
                        _lik_model_clients=lik_model_clients or None,
                    )
                    gen_liks_df_dict[proposal_curr] = gen_liks_df_dict[proposal_curr][
                        unconstrained_col_names
                    ]
                    mix_liks_mat = gen_liks_df_dict[proposal_curr][
                        unconstrained_lik_cols
                    ].to_numpy()

                    ## Step 3: Constrain likelihoods
                    if cfg.conformal_policy_control.constrain_against == "init":
                        mix_constrained_mat = np.zeros(np.shape(mix_liks_mat))
                        mix_constrained_mat[:, 0] = mix_liks_mat[:, 0]
                        for c in range(1, n_models):
                            mix_constrained_mat[:, c] = constrain_likelihoods(
                                cfg,
                                mix_liks_mat[:, [0, c]],
                                [betas_list[0], betas_list[c]],
                                [psis_list[0], psis_list[c]],
                            )[:, -1]
                        safe_liks_dict[proposal_curr] = mix_constrained_mat[:, 0]
                    else:
                        mix_constrained_mat = constrain_likelihoods(
                            cfg, mix_liks_mat, betas_list, psis_list
                        )
                        if mix_constrained_mat.shape[1] > 1:
                            safe_liks_dict[proposal_curr] = mix_constrained_mat[:, -2]
                        else:
                            safe_liks_dict[proposal_curr] = mix_constrained_mat[:, -1]

                    unconstrained_liks_dict[proposal_curr] = mix_liks_mat[:, -1]
                    lik_ratios_unconstrained_over_safe_dict[proposal_curr] = (
                        unconstrained_liks_dict[proposal_curr]
                        / safe_liks_dict[proposal_curr]
                    )
                    constrained_liks_df_dict[proposal_curr] = pd.concat(
                        [
                            gen_liks_df_dict[proposal_curr][[PARTICLE, SCORE]],
                            pd.DataFrame(
                                mix_constrained_mat, columns=constrained_lik_cols
                            ),
                        ],
                        axis=1,
                    )

                    ## Mixture proposal probabilities for constrained likelihoods
                    constrained_liks_df_dict[proposal_curr].iloc[:, -1] = (
                        safe_prop_mix_weight * safe_liks_dict[proposal_curr]
                        + (1 - safe_prop_mix_weight)
                        * unconstrained_liks_dict[proposal_curr]
                    )

                    N_prop_dict[proposal_curr] = len(
                        gen_liks_df_dict[proposal_curr]
                    )  ## Reset number of available proposals
                    n_proposals_total += N_prop_dict[proposal_curr]
                    n_proposed_dict[proposal_curr] = (
                        0  ## Reset number of used proposals to 0
                    )
                    accepted_curr_dict[
                        proposal_curr
                    ] = []  ## Reset running list of current acceptances
                    mix_acc_probs_dict[proposal_curr] = []

            if gen_liks_df_dict[proposal_curr] is None:
                ## If no proposals were generated, continue to restart the while loop (and try generating proposals again)
                continue

            while (
                n_proposed_dict["safe"] < N_prop_dict["safe"]
                and n_proposed_dict["unconstrained"] < N_prop_dict["unconstrained"]
            ):
                ## Arbitrary way of standardizing random seeds so that is consistent when rerunning from checkpoint (but uses different random seed for each call)
                ar_random_seed = (
                    call_idx if not post_policy_control else 1000 + call_idx
                )
                np.random.seed(ar_random_seed)

                ## Select proposal from the mixture
                u_mix = np.random.uniform()
                if u_mix < safe_prop_mix_weight:
                    #  or (
                    #     n_accepted == 0 and safe_prop_mix_weight > 0.5
                    # ):
                    proposal_curr = "safe"
                else:
                    proposal_curr = "unconstrained"

                lik_ratios_unconstrained_over_safe = (
                    lik_ratios_unconstrained_over_safe_dict[proposal_curr]
                )
                safe_liks = safe_liks_dict[proposal_curr]
                unconstrained_liks = unconstrained_liks_dict[proposal_curr]

                u = np.random.uniform()
                i_curr = n_proposed_dict[proposal_curr]

                ## Initial state for MH sampling
                if n_accepted == 0:
                    if i_curr == 0:
                        if cfg.conformal_policy_control.use_overlap_mix_weight:
                            prev_target_lik = (
                                min(
                                    unconstrained_liks[i_curr],
                                    betas_list[-1] * safe_liks[i_curr],
                                )
                                / psis_list[-1]
                            )
                        else:
                            prev_target_lik = min(
                                unconstrained_liks[i_curr],
                                betas_list[-1] * safe_liks[i_curr],
                            )

                        # prev_prop_lik = safe_liks[i_curr]  #safe_prop_mix_weight * safe_liks[i_curr] + (1 - safe_prop_mix_weight) * unconstrained_liks[i_curr]
                        prev_prop_lik = (
                            safe_prop_mix_weight * safe_liks[i_curr]
                            + (1 - safe_prop_mix_weight) * unconstrained_liks[i_curr]
                        )

                    else:
                        ## Can keep updating the initial state until first acceptance
                        prev_target_lik = target_lik
                        prev_prop_lik = prop_lik

                if cfg.conformal_policy_control.use_overlap_mix_weight:
                    target_lik = (
                        min(
                            unconstrained_liks[i_curr],
                            betas_list[-1] * safe_liks[i_curr],
                        )
                        / psis_list[-1]
                    )
                else:
                    target_lik = min(
                        unconstrained_liks[i_curr], betas_list[-1] * safe_liks[i_curr]
                    )

                prop_lik = (
                    safe_prop_mix_weight * safe_liks[i_curr]
                    + (1 - safe_prop_mix_weight) * unconstrained_liks[i_curr]
                )

                is_imh = post_policy_control and (
                    cfg.conformal_policy_control.ind_metropolis_hastings
                    or call_idx > cfg.conformal_policy_control.num_AR_before_MH
                )
                if is_imh:
                    if not imh_switched:
                        imh_switched = True
                        imh_switch_call_idx = call_idx
                    ## Conditions for running IMH
                    acc_prob = min(
                        1, (target_lik / prev_target_lik) * (prev_prop_lik / prop_lik)
                    )

                else:
                    ## Condition for rejection sampling or until first acceptance of MH sampling
                    acc_prob = min(1, (target_lik / (prop_lik * env_const)))

                mix_acc_probs_dict[proposal_curr].append(acc_prob)
                n_proposed_dict[proposal_curr] += 1

                if u < acc_prob:
                    ## Update states for MH
                    prev_target_lik = target_lik
                    prev_prop_lik = prop_lik

                    accepted_curr_dict[proposal_curr].append(True)
                    n_accepted += 1

                    if n_accepted >= n_target:
                        break

                else:
                    accepted_curr_dict[proposal_curr].append(False)

            # Collect all proposals from mixture sub-batches
            for proposal_curr in ["safe", "unconstrained"]:
                n_evaluated = len(accepted_curr_dict[proposal_curr])
                if n_evaluated > 0 and gen_liks_df_dict[proposal_curr] is not None:
                    batch_df = gen_liks_df_dict[proposal_curr][:n_evaluated].copy()
                    batch_df["acc_prob"] = mix_acc_probs_dict[proposal_curr][
                        :n_evaluated
                    ]
                    batch_df["accepted"] = accepted_curr_dict[proposal_curr]
                    all_proposal_dfs.append(batch_df)


                accepted_unconstrained_curr_df = gen_liks_df_dict[proposal_curr][
                                                    : len(accepted_curr_dict[proposal_curr])
                                                ][accepted_curr_dict[proposal_curr]]
                accepted_constrained_curr_df = constrained_liks_df_dict[proposal_curr][
                                                    : len(accepted_curr_dict[proposal_curr])
                                                ][accepted_curr_dict[proposal_curr]]

                accepted_unconstrained_dfs.append(accepted_unconstrained_curr_df)
                accepted_constrained_dfs.append(accepted_constrained_curr_df)

                logger.info(f"Accepted {sum(accepted_curr_dict[proposal_curr])} samples in current sampling round for {proposal_curr} proposal (in mixture)")

                ## Record samples accepted in current sampling round for tracking acceptance progress
                if sum(accepted_curr_dict[proposal_curr]) > 0:
                    base_output_name = f"alpha{cfg.conformal_policy_control.alpha}_gens_likelihood_{cfg.iterative_generation.args.sample_size}sample_{cfg.iterative_generation.args.max_iterations}iter_temp{temps[-1]}_{cfg.generation_sampling_num_return_sequences}seqs.jsonl"

                    curr_accepted_unconstrained_fp = os.path.join(
                        os.path.dirname(gen_liks_fp_dict[proposal_curr]),
                        f"curr_accepted_uLiks_cn{call_idx}_{proposal_curr}Prop_{base_output_name}",
                    )
                    curr_accepted_constrained_fp = os.path.join(
                        os.path.dirname(gen_liks_fp_dict[proposal_curr]),
                        f"curr_accepted_cLiks_cn{call_idx}_{proposal_curr}Prop_{base_output_name}",
                    )

                    if cfg.overwrite_ig or not fs.exists(curr_accepted_unconstrained_fp):
                        accepted_unconstrained_curr_df.to_json(
                            curr_accepted_unconstrained_fp, orient="records", lines=True
                        )

                    if cfg.overwrite_ig or not fs.exists(curr_accepted_constrained_fp):
                        accepted_constrained_curr_df.to_json(
                            curr_accepted_constrained_fp, orient="records", lines=True
                        )

            ## If still no samples accepted after max(20% of planned AR calls, 1) of planned AR calls, go ahead and switch to IMH sampling
            if n_accepted == 0 and call_idx >= max(0.2 * cfg.conformal_policy_control.num_AR_before_MH, 1):
                call_idx = cfg.conformal_policy_control.num_AR_before_MH

    else:
        raise ValueError(f"Unknown proposal name : {proposal}")

    # ---- Free pre-loaded models to reclaim GPU memory ----
    cleanup_model_clients(lik_model_clients, gen_model_client)

    # ## Save accepted with unconstrained likelihoods
    # t = len(model_dir_list)-1
    # accepted_unconstrained_df = gen_liks_df[accepted]

    if accepted_unconstrained_dfs:
        accepted_unconstrained_df = pd.concat(
            accepted_unconstrained_dfs, ignore_index=True
        )
    else:
        accepted_unconstrained_df = pd.DataFrame(columns=unconstrained_col_names)

    if accepted_constrained_dfs:
        accepted_constrained_df = pd.concat(accepted_constrained_dfs, ignore_index=True)
    else:
        accepted_constrained_df = pd.DataFrame(columns=constrained_col_names)

    base_output_name = f"alpha{cfg.conformal_policy_control.alpha}_gens_likelihood_{cfg.iterative_generation.args.sample_size}sample_{cfg.iterative_generation.args.max_iterations}iter_temp{temps[-1]}_{cfg.generation_sampling_num_return_sequences}seqs.jsonl"

    if proposal == "mixture":
        gen_liks_fp = gen_liks_fp_dict[proposal_curr]

    if depth == 0:
        if post_policy_control:
            ## Output filenames

            u_output_filename = f"accepted_uLiks_{base_output_name}"  # f"accepted_uLiks_{base_output_name}"
            c_output_filename = f"accepted_cLiks_{base_output_name}"  # f"accepted_cLiks_{base_output_name}"

            accepted_unconstrained_gen_liks_fp = os.path.join(
                os.path.dirname(gen_liks_fp), u_output_filename
            )
            accepted_constrained_gen_liks_fp = os.path.join(
                os.path.dirname(gen_liks_fp), c_output_filename
            )

        else:
            if gen_liks_fp is None and not unconstrained_pre_cpc_call_num_check:
                ## If terminating due to exceeding limit on number of unconstrained pre-CPC, return None
                return ARSamplingResult(None, None, None, None, None, None)

            accepted_unconstrained_gen_liks_fp = os.path.join(
                os.path.dirname(gen_liks_fp),
                f"prop_{proposal}_beta{betas_list[-1]}_cn{call_idx}_{base_output_name}",
            )
            accepted_constrained_gen_liks_fp = os.path.join(
                os.path.dirname(gen_liks_fp),
                f"prop_{proposal}_beta{betas_list[-1]}_cn{call_idx}_{base_output_name}",
            )

    else:
        accepted_unconstrained_gen_liks_fp = os.path.join(
            os.path.dirname(gen_liks_fp), f"{depth}u_cn{call_idx}_{base_output_name}"
        )
        accepted_constrained_gen_liks_fp = os.path.join(
            os.path.dirname(gen_liks_fp), f"{depth}c_cn{call_idx}_{base_output_name}"
        )

    if cfg.overwrite_ig or not fs.exists(accepted_unconstrained_gen_liks_fp):
        accepted_unconstrained_df.to_json(
            accepted_unconstrained_gen_liks_fp, orient="records", lines=True
        )

    if cfg.overwrite_ig or not fs.exists(accepted_constrained_gen_liks_fp):
        accepted_constrained_df.to_json(
            accepted_constrained_gen_liks_fp, orient="records", lines=True
        )

    # Persist all proposals and compute quality metrics.
    # Batches are written one at a time and metrics are accumulated incrementally
    # to avoid materialising a single ~n_proposals_total-row DataFrame in memory.
    all_proposals_fp: str | None = None
    sampling_metrics: ARSamplingMetrics | None = None
    if depth == 0 and all_proposal_dfs:
        try:
            all_proposals_fp = os.path.join(
                os.path.dirname(accepted_unconstrained_gen_liks_fp),
                f"all_proposals_{base_output_name}",
            )

            # --- incremental file write ---
            if cfg.overwrite_ig or not fs.exists(all_proposals_fp):
                with fs.open(all_proposals_fp, "w") as _f:
                    for _batch in all_proposal_dfs:
                        if len(_batch) > 0:
                            _f.write(
                                _batch.to_json(
                                    path_or_buf=None, orient="records", lines=True
                                )
                            )

            # --- incremental quality metrics ---
            # Running totals for accepted and rejected subsets.
            _acc = dict(n=0, n_parsable=0, n_feasible=0, score_sum=0.0,
                        min_score=float("inf"), max_score=float("-inf"))
            _rej = dict(n=0, n_parsable=0, n_feasible=0, score_sum=0.0,
                        min_score=float("inf"), max_score=float("-inf"))

            for _batch in all_proposal_dfs:
                _mask = _batch["accepted"].astype(bool)
                for _subset, _stats in [(_batch[_mask], _acc), (_batch[~_mask], _rej)]:
                    if len(_subset) == 0:
                        continue
                    _scores = _subset[SCORE].to_numpy(dtype=float)
                    _finite = np.isfinite(_scores)
                    _stats["n"] += len(_subset)
                    _stats["n_parsable"] += int(np.sum(~np.isnan(_scores)))
                    _stats["n_feasible"] += int(np.sum(_finite))
                    if np.any(_finite):
                        _stats["score_sum"] += float(np.sum(_scores[_finite]))
                        _stats["min_score"] = min(
                            _stats["min_score"], float(np.min(_scores[_finite]))
                        )
                        _stats["max_score"] = max(
                            _stats["max_score"], float(np.max(_scores[_finite]))
                        )

            def _make_quality(s: dict) -> SampleQualityMetrics:
                n, n_feas = s["n"], s["n_feasible"]
                return SampleQualityMetrics(
                    n_samples=n,
                    n_parsable=s["n_parsable"],
                    n_feasible=n_feas,
                    frac_feasible=n_feas / n if n > 0 else 0.0,
                    mean_score=s["score_sum"] / n_feas if n_feas > 0 else float("nan"),
                    min_score=s["min_score"] if n_feas > 0 else float("nan"),
                    max_score=s["max_score"] if n_feas > 0 else float("nan"),
                )

            sampling_metrics = ARSamplingMetrics(
                n_accepted=n_accepted,
                n_calls=call_idx,
                n_proposals_total=n_proposals_total,
                acceptance_rate=n_accepted / max(1, n_proposals_total),
                proposal_type=proposal or "unconstrained",
                ar_to_imh_switch=imh_switched,
                imh_switch_call_idx=imh_switch_call_idx,
                safe_prop_mix_weight=safe_prop_mix_weight
                if proposal == "mixture"
                else None,
                env_const=env_const,
                accepted_quality=_make_quality(_acc),
                rejected_quality=_make_quality(_rej),
            )
        except Exception:
            logger.warning(
                "Failed to write all_proposals or compute sampling metrics "
                "(n_accepted=%d, n_proposals_total=%d); continuing with accepted samples only.",
                n_accepted,
                n_proposals_total,
                exc_info=True,
            )
            all_proposals_fp = None
            sampling_metrics = None

    return ARSamplingResult(
        unconstrained_df=accepted_unconstrained_df,
        unconstrained_fp=accepted_unconstrained_gen_liks_fp,
        constrained_df=accepted_constrained_df,
        constrained_fp=accepted_constrained_gen_liks_fp,
        all_proposals_fp=all_proposals_fp,
        sampling_metrics=sampling_metrics,
    )
