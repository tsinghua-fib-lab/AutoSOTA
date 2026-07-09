import numpy as np
import pandas as pd

import logging
import sys
import os

from typing import List, Optional
from omegaconf import DictConfig

from ..infrastructure.file_handler import LocalOrS3Client
from ..infer.rejection_sampling import generate_sample_batch, compute_batch_likelihoods
from ..core.model_loading import preload_model_clients, cleanup_model_clients
from ..infer.rejection_sampling import _load_test_fn, _is_direct_mode
from .grid import prepare_grid
from .normalization import (
    importance_weighted_monte_carlo_integration,
    iwmci_overlap_est,
)
from .process_likelihoods import (
    mixture_pdf_from_densities_mat,
    constrain_likelihoods,
    check_col_names,
)
from ..data_contracts import (
    HIGHER_SCORE,
    HIGHER_SCORE_PARTICLE,
    LOWER_SCORE,
    LOWER_SCORE_PARTICLE,
    PARTICLE,
    SCORE,
    con_lik_col,
    lik_col,
)
from ..metrics import CPCSearchMetrics, CPCSearchResult

logger = logging.getLogger(__name__)


def cpc_beta_search(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    model_dir_list: List[str],
    seeds_fp_list: List[str],
    prev_cal_data_unconstrained_liks_fp_list: List[
        str
    ],  ## Should contain both cal data and *constrained* likelihoods, up to r{t}
    prev_cal_data_constrained_liks_fp_list: List[
        str
    ],  ## Should contain both cal data and *constrained* likelihoods, up to r{t-1}
    betas_list: List[float],
    psis_list: List[float],  ## Normalization constants
    ga_data_dir: str,
    higher_score_particle_field: str = HIGHER_SCORE_PARTICLE,
    lower_score_particle_field: str = LOWER_SCORE_PARTICLE,
    higher_score_field: str = HIGHER_SCORE,
    lower_score_field: str = LOWER_SCORE,
    global_random_seed: int = 0,
) -> Optional[CPCSearchResult]:
    """Runs conformal policy control.

    Returns:
        A ``CPCSearchResult`` containing the selected beta, psi, constrained
        likelihood DataFrames/filepaths, proposal info, and search metrics.

        Returns ``None`` when no proposal produced any samples this round (e.g.
        every ``gen_df`` was ``None``/empty, which can happen when
        ``cfg.conformal_policy_control.alpha >= 1.0``). Callers should treat a
        ``None`` result as "skip this round and reuse the previous round's model
        and data" rather than crashing.
    """

    ## Load calibration data into one dataframe
    n_cal_sets = len(prev_cal_data_constrained_liks_fp_list)
    n_models = len(model_dir_list)

    if n_cal_sets != len(prev_cal_data_unconstrained_liks_fp_list):
        raise ValueError(
            "Number of unconstrained and constrained cal sets must be the same"
        )

    unconstrained_lik_cols = [lik_col(i) for i in range(n_cal_sets)]
    constrained_lik_cols = [con_lik_col(i) for i in range(n_cal_sets)]

    cal_data_constrained_all = pd.read_json(
        prev_cal_data_constrained_liks_fp_list[0], orient="records", lines=True
    )
    cal_data_unconstrained_all = pd.read_json(
        prev_cal_data_unconstrained_liks_fp_list[0], orient="records", lines=True
    )

    n_cal_per_model = [len(cal_data_constrained_all)]

    for i in range(1, n_cal_sets):
        cal_data_constrained_curr = pd.read_json(
            prev_cal_data_constrained_liks_fp_list[i], orient="records", lines=True
        )
        cal_data_unconstrained_curr = pd.read_json(
            prev_cal_data_unconstrained_liks_fp_list[i], orient="records", lines=True
        )

        if cal_data_constrained_curr.empty or cal_data_unconstrained_curr.empty:
            empty = []
            if cal_data_constrained_curr.empty:
                empty.append(f"constrained={prev_cal_data_constrained_liks_fp_list[i]}")
            if cal_data_unconstrained_curr.empty:
                empty.append(
                    f"unconstrained={prev_cal_data_unconstrained_liks_fp_list[i]}"
                )
            logger.warning(
                f"Skipping empty calibration file at index {i}: {', '.join(empty)}"
            )
            n_cal_per_model.append(0)
            continue

        if len(cal_data_constrained_curr) != len(cal_data_unconstrained_curr):
            raise ValueError(
                "Num samples in constrained and constrained cal sets should be same (ie, same particles)!"
            )

        n_cal_per_model.append(len(cal_data_constrained_curr))

        ## Check that columns are the same
        if cal_data_constrained_all.columns.equals(
            cal_data_constrained_curr.columns
        ) and cal_data_unconstrained_all.columns.equals(
            cal_data_unconstrained_curr.columns
        ):
            cal_data_constrained_all = pd.concat(
                [cal_data_constrained_all, cal_data_constrained_curr], ignore_index=True
            )
            cal_data_unconstrained_all = pd.concat(
                [cal_data_unconstrained_all, cal_data_unconstrained_curr],
                ignore_index=True,
            )

        else:
            ## If columns are not the same, try to subset to currently relevant likelihood columns
            cal_data_constrained_all = cal_data_constrained_all[
                list(cal_data_constrained_all.columns[0:2]) + constrained_lik_cols
            ]
            cal_data_unconstrained_all = cal_data_unconstrained_all[
                list(cal_data_unconstrained_all.columns[0:2]) + unconstrained_lik_cols
            ]
            cal_data_constrained_curr = cal_data_constrained_curr[
                list(cal_data_constrained_curr.columns[0:2]) + constrained_lik_cols
            ]
            cal_data_unconstrained_curr = cal_data_unconstrained_curr[
                list(cal_data_unconstrained_curr.columns[0:2]) + unconstrained_lik_cols
            ]

            ## Try again to concatenate
            if cal_data_constrained_all.columns.equals(
                cal_data_constrained_curr.columns
            ) and cal_data_unconstrained_all.columns.equals(
                cal_data_unconstrained_curr.columns
            ):
                cal_data_constrained_all = pd.concat(
                    [cal_data_constrained_all, cal_data_constrained_curr],
                    ignore_index=True,
                )
                cal_data_unconstrained_all = pd.concat(
                    [cal_data_unconstrained_all, cal_data_unconstrained_curr],
                    ignore_index=True,
                )

            else:
                logger.info(
                    f"cal_data_constrained_all.columns : {cal_data_constrained_all.columns}"
                )
                logger.info(
                    f"cal_data_constrained_curr.columns : {cal_data_constrained_curr.columns}"
                )
                logger.info(
                    f"cal_data_unconstrained_all.columns : {cal_data_unconstrained_all.columns}"
                )
                logger.info(
                    f"cal_data_unconstrained_curr.columns : {cal_data_unconstrained_curr.columns}"
                )
                raise ValueError(
                    f"Error: cal_data_constrained_all.columns. equals(cal_data_constrained_curr.columns) : {cal_data_constrained_all.columns.equals(cal_data_constrained_curr.columns)}"
                )

    check_col_names(cal_data_constrained_all)
    check_col_names(cal_data_unconstrained_all)

    ## Number of columns to keep when rerunning from checkpoint (relevant when not overwriting)
    ## Columns should be ["particle", "score", "lik_r0", ..., "lik_r{n_cal_sets-1}"] (similarly for constrained)
    num_cols_to_keep = 2 + n_models
    cal_data_constrained_all = cal_data_constrained_all.iloc[:, :num_cols_to_keep]
    cal_data_unconstrained_all = cal_data_unconstrained_all.iloc[:, :num_cols_to_keep]

    ## Prep cal data safe & unconstrained liks: Need most recent safe likelihoods (safe at t-1) and current unconstrained likelihoods (unconstrained at t) in loop
    cal_data_t0_safe_and_t_unconstrained_liks = pd.concat(
        [
            cal_data_constrained_all[constrained_lik_cols[0]],
            cal_data_unconstrained_all.iloc[:, -1],
        ],
        axis=1,
    ).to_numpy()  ## Double check this

    ## For cal data: Use constrained likelihoods to compute mixture distribution
    cal_data_constrained_all_liks = cal_data_constrained_all[
        constrained_lik_cols
    ].to_numpy()
    mixture_weights = np.array(n_cal_per_model)
    cal_mixture_constrained_density = mixture_pdf_from_densities_mat(
        cal_data_constrained_all_liks, mixture_weights
    )

    if cfg.conformal_policy_control.alpha >= 1.0:
        policy_names = ["unconstrained"]
        adjusted_alpha = 1.0
    elif cfg.conformal_policy_control.num_starts_beta_search == 2:
        policy_names = ["unconstrained", "safe"]
        adjusted_alpha = (
            cfg.conformal_policy_control.alpha / 2
        )  ## Multistart correction

    else:
        policy_names = ["safe", "unconstrained"]
        adjusted_alpha = (
            cfg.conformal_policy_control.alpha
        )  ## Fixed sequence testing, no multistart correction


    lik_ratios_unconstrained_over_safe_cal_and_prop_dict = {}
    lik_ratios_unconstrained_over_safe_dict = {}

    unconstrained_liks_dict, safe_liks_dict = {}, {}
    prop_mixture_constrained_density_dict = {}

    (
        unconstrained_df_dict,
        unconstrained_gen_liks_fp_dict,
        constrained_liks_df_dict,
        constrained_gen_liks_fp_dict,
    ) = {}, {}, {}, {}
    prop_data_t0_safe_and_t_unconstrained_liks_dict = {}

    ## Pre-load models for in-memory mode
    use_inmemory = _is_direct_mode(cfg)
    gen_model_client, lik_model_clients = None, {}
    test_fn = None
    if use_inmemory:
        test_fn = _load_test_fn(ga_data_dir)
        # Load lik clients once; gen client is selected per proposal below
        _, lik_model_clients = preload_model_clients(
            cfg, model_dir_list, "unconstrained"
        )

    try:
        for i, proposal in enumerate(policy_names):
            if proposal == "safe":
                betas_list_tmp = betas_list + [sys.float_info.min]
                psis_list_tmp = psis_list + [sys.float_info.min]
                gen_model_dir = model_dir_list[0]
                gen_seeds_fp = seeds_fp_list[0]
                gen_temp = cfg.temperature_init
            elif proposal == "unconstrained":
                betas_list_tmp = betas_list + [np.inf]
                psis_list_tmp = psis_list + [1.0]
                gen_model_dir = model_dir_list[-1]
                gen_seeds_fp = seeds_fp_list[-1]
                gen_temp = (
                    cfg.temperature if len(model_dir_list) > 1 else cfg.temperature_init
                )
            else:
                raise ValueError(f"Unrecognized proposal name : {proposal}")

            gen_model_client = lik_model_clients.get(gen_model_dir)
            if use_inmemory and gen_model_client is None:
                raise RuntimeError(f"Model client not pre-loaded for {gen_model_dir}")

            ## Step 1: Generate samples from the proposal model
            gen_df, gen_fp = generate_sample_batch(
                cfg,
                fs,
                gen_seeds_fp,
                ga_data_dir,
                gen_model_dir,
                model_dir_list[-1],  # output_dir
                gen_temp,
                model_idx = 0 if proposal == "safe" else len(model_dir_list) - 1,
                higher_score_particle_field=higher_score_particle_field,
                lower_score_particle_field=lower_score_particle_field,
                higher_score_field=higher_score_field,
                lower_score_field=lower_score_field,
                random_seed=global_random_seed * 10000,
                n_target=cfg.conformal_policy_control.accept_reject.n_target_pre_cpc,
                _model_client=gen_model_client,
                _test_fn=test_fn,
            )


            if gen_df is None and proposal == "unconstrained" and cfg.conformal_policy_control.alpha < 1.0:
                policy_names = ["safe"]
                continue


            if gen_df is None or len(gen_df) == 0:
                continue
            
            logger.info(f"_is_direct_mode(cfg) : {_is_direct_mode(cfg)}")
            logger.info(f"cpc search gen_fp: {gen_fp}")

            ## Step 2: Compute likelihoods under all models
            unconstrained_df = compute_batch_likelihoods(
                cfg,
                fs,
                gen_df,
                seeds_fp_list,
                model_dir_list,
                target_fp=gen_fp,
                _lik_model_clients=lik_model_clients or None,
            )

            unconstrained_lik_cols_prop = [lik_col(j) for j in range(n_models)]
            unconstrained_col_names_prop = [
                PARTICLE,
                SCORE,
            ] + unconstrained_lik_cols_prop
            unconstrained_df = unconstrained_df[unconstrained_col_names_prop]

            ## Step 3: Constrain likelihoods
            gen_liks_mat = unconstrained_df[unconstrained_lik_cols_prop].to_numpy()

            if cfg.conformal_policy_control.constrain_against == "init":
                constrained_liks_mat = np.zeros(np.shape(gen_liks_mat))
                constrained_liks_mat[:, 0] = gen_liks_mat[:, 0]
                for c in range(1, n_models):
                    constrained_liks_mat[:, c] = constrain_likelihoods(
                        cfg,
                        gen_liks_mat[:, [0, c]],
                        [betas_list_tmp[0], betas_list_tmp[c]],
                        [psis_list_tmp[0], psis_list_tmp[c]],
                    )[:, -1]
                safe_liks = constrained_liks_mat[:, 0]
            else:
                constrained_liks_mat = constrain_likelihoods(
                    cfg, gen_liks_mat, betas_list_tmp, psis_list_tmp
                )
                if constrained_liks_mat.shape[1] > 1:
                    safe_liks = constrained_liks_mat[:, -2]
                else:
                    safe_liks = constrained_liks_mat[:, -1]

            unconstrained_liks = gen_liks_mat[:, -1]
            lik_ratios_unconstrained_over_safe = unconstrained_liks / safe_liks

            constrained_lik_cols_prop = [con_lik_col(j) for j in range(n_models)]
            constrained_liks_df = pd.concat(
                [
                    unconstrained_df[[PARTICLE, SCORE]],
                    pd.DataFrame(
                        constrained_liks_mat, columns=constrained_lik_cols_prop
                    ),
                ],
                axis=1,
            )

            ## Restrict to number of columns to keep when rerunning from checkpoint
            unconstrained_df = unconstrained_df.iloc[:, :num_cols_to_keep]
            constrained_liks_df = constrained_liks_df.iloc[:, :num_cols_to_keep]

            ## Construct proposal-specific filepaths for downstream saving
            proposal_fp = os.path.join(
                model_dir_list[-1], f"prop_{proposal}_liks.jsonl"
            )

            unconstrained_df_dict[proposal] = unconstrained_df
            unconstrained_gen_liks_fp_dict[proposal] = proposal_fp
            constrained_liks_df_dict[proposal] = constrained_liks_df
            constrained_gen_liks_fp_dict[proposal] = proposal_fp

            check_col_names(unconstrained_df)
            check_col_names(constrained_liks_df)

            ## Combine proposal and calibration likelihood ratios
            prop_data_t0_safe_and_t_unconstrained_liks = np.column_stack(
                [np.array(safe_liks), np.array(unconstrained_liks)]
            )

            if cfg.conformal_policy_control.constrain_against == "init":
                lik_ratios_unconstrained_over_safe_cal_and_prop = np.concatenate(
                    (
                        np.array(lik_ratios_unconstrained_over_safe),
                        np.array(
                            cal_data_unconstrained_all.iloc[:, -1]
                            / cal_data_constrained_all[con_lik_col(0)]
                        ),
                    )
                )
            else:
                lik_ratios_unconstrained_over_safe_cal_and_prop = np.concatenate(
                    (
                        np.array(lik_ratios_unconstrained_over_safe),
                        np.array(
                            cal_data_unconstrained_all.iloc[:, -1]
                            / cal_data_unconstrained_all.iloc[:, -2]
                        ),
                    )
                )

            prop_data_t0_safe_and_t_unconstrained_liks_dict[proposal] = (
                prop_data_t0_safe_and_t_unconstrained_liks
            )
            lik_ratios_unconstrained_over_safe_cal_and_prop_dict[proposal] = (
                lik_ratios_unconstrained_over_safe_cal_and_prop
            )
            lik_ratios_unconstrained_over_safe_dict[proposal] = (
                lik_ratios_unconstrained_over_safe
            )

            unconstrained_liks_dict[proposal] = unconstrained_liks
            safe_liks_dict[proposal] = safe_liks

            ## Compute mixture weights for past data
            prop_data_constrained_all = constrained_liks_df
            prop_data_constrained_prev_liks = prop_data_constrained_all[
                constrained_lik_cols
            ].to_numpy()
            mixture_weights = np.array(n_cal_per_model)
            prop_mixture_constrained_density = mixture_pdf_from_densities_mat(
                prop_data_constrained_prev_liks, mixture_weights
            )
            prop_mixture_constrained_density_dict[proposal] = (
                prop_mixture_constrained_density
            )
    finally:
        if use_inmemory:
            cleanup_model_clients(lik_model_clients, gen_model_client)

    ## Edge case: no proposal produced any samples this round (every gen_df was
    ## None/empty). This can occur when alpha >= 1.0, where policy_names only
    ## contains "unconstrained" and the alpha < 1.0 fallback above does not apply.
    ## Rather than crashing on the empty dict below, signal the caller to fall
    ## back gracefully (reuse the previous round's model and data).
    if not lik_ratios_unconstrained_over_safe_cal_and_prop_dict:
        logger.warning(
            "CPC beta search produced no proposal samples (all generations were "
            f"empty/None) for alpha={cfg.conformal_policy_control.alpha}; returning "
            "None so the pipeline can reuse the previous round's model and data."
        )
        return None


    lik_ratios_unconstrained_over_safe_cal_and_prop_arr = np.array(
        lik_ratios_unconstrained_over_safe_cal_and_prop_dict[policy_names[0]]
    )
    lik_ratios_unconstrained_over_safe_arr = np.array(
        lik_ratios_unconstrained_over_safe_dict[policy_names[0]]
    )

    if len(policy_names) > 1:
        ## Grid containing betas obtained from both proposals
        lik_ratios_unconstrained_over_safe_cal_and_prop_arr = np.concatenate(
            (
                lik_ratios_unconstrained_over_safe_cal_and_prop_arr,
                lik_ratios_unconstrained_over_safe_cal_and_prop_dict[policy_names[1]],
            )
        )
        lik_ratios_unconstrained_over_safe_arr = np.concatenate(
            (
                lik_ratios_unconstrained_over_safe_arr,
                lik_ratios_unconstrained_over_safe_dict[policy_names[1]],
            )
        )

        G = prepare_grid(
            cfg,
            lik_ratios_unconstrained_over_safe_arr,  # lik_ratios_unconstrained_over_safe,
            n_grid=cfg.conformal_policy_control.args.n_grid,
            proposal="mixed",
        )

    else:
        if cfg.conformal_policy_control.alpha >= 1.0:
            G = [np.inf]
        else:
            if proposal == "unconstrained":
                G = prepare_grid(
                    cfg,
                    lik_ratios_unconstrained_over_safe_cal_and_prop,  # lik_ratios_unconstrained_over_safe,
                    n_grid=cfg.conformal_policy_control.args.n_grid,
                    proposal=proposal,
                )

            elif proposal == "safe":
                G = prepare_grid(
                    cfg,
                    lik_ratios_unconstrained_over_safe_cal_and_prop,  # lik_ratios_unconstrained_over_safe,
                    n_grid=cfg.conformal_policy_control.args.n_grid,
                    proposal=proposal,
                )
            else:
                raise ValueError(f"Unrecognized proposal name : {proposal}")

    ## Search over grid for largest bound that satisfies conformal constraint
    beta_hat_t_curr = (
        sys.float_info.min if adjusted_alpha < 1.0 else np.inf
    )  ## Currently selected beta_t is initially smallest float value
    envelope_const_constrained_over_proposal = 1

    # Metrics tracking
    last_w_test = 0.0
    last_w_infeasible_normalized = 0.0
    last_switch_to_mixture_proposal = False
    last_switch_to_optimized_proposal = False
    last_psi_hat_t = sys.float_info.min
    last_psi_hat_t_unconstrained = 1.0
    last_psi_hat_t_safe = sys.float_info.min
    last_psi_hat_intersection_safe = 1.0
    last_psi_hat_intersection_unconstrained = 0.0
    # last_envelope_const_constrained_over_proposal = 1.0

    G_original = G.copy()
    beta_position_in_G_original = 0
    last_proposal = policy_names[0]

    for i, proposal in enumerate(policy_names):
        unconstrained_df = unconstrained_df_dict[proposal]
        unconstrained_gen_liks_fp = unconstrained_gen_liks_fp_dict[proposal]
        constrained_liks_df = constrained_liks_df_dict[proposal]
        constrained_gen_liks_fp = constrained_gen_liks_fp_dict[proposal]

        prop_data_t0_safe_and_t_unconstrained_liks = (
            prop_data_t0_safe_and_t_unconstrained_liks_dict[proposal]
        )
        lik_ratios_unconstrained_over_safe_cal_and_prop = (
            lik_ratios_unconstrained_over_safe_cal_and_prop_dict[proposal]
        )
        lik_ratios_unconstrained_over_safe = lik_ratios_unconstrained_over_safe_dict[
            proposal
        ]
        unconstrained_liks = unconstrained_liks_dict[proposal]
        safe_liks = safe_liks_dict[proposal]

        prop_mixture_constrained_density = prop_mixture_constrained_density_dict[
            proposal
        ]

        ## Get infeasibility indicators for calibration data
        cal_scores = cal_data_constrained_all[SCORE].to_numpy()
        cal_infeasible_indicators = np.isnan(cal_scores) | np.isinf(cal_scores)

        for b, beta_t in enumerate(G):


            # if cfg.conformal_policy_control.alpha == 0.8 and global_random_seed == 0 and len(model_dir_list) in [20, 21]\
            #     and beta_t >= 3.06441459e-039:
            #     breakpoint()
            ## Estimate normalization constant via IWMCI
            psi_hat_t = importance_weighted_monte_carlo_integration(
                lik_ratios_unconstrained_over_safe, beta_t, proposal
            )

            if "unconstrained" in policy_names:
                psi_hat_t_unconstrained = importance_weighted_monte_carlo_integration(
                    lik_ratios_unconstrained_over_safe_dict["unconstrained"],
                    beta_t,
                    "unconstrained",
                )

                psi_hat_intersection_unconstrained = iwmci_overlap_est(
                    LRs_unconstrained_over_safe=lik_ratios_unconstrained_over_safe_dict[
                        "unconstrained"
                    ],  ## 1D numpy array
                    unconstrained_liks=unconstrained_liks_dict["unconstrained"],
                    safe_liks=safe_liks_dict["unconstrained"],
                    beta_t=beta_t,  ## float
                    psi_t=psi_hat_t_unconstrained,
                    proposal="unconstrained",
                )
            else:
                psi_hat_t_unconstrained = 0.0
                psi_hat_intersection_unconstrained = 0.0

            if "safe" in policy_names:
                psi_hat_t_safe = importance_weighted_monte_carlo_integration(
                    lik_ratios_unconstrained_over_safe_dict["safe"], beta_t, "safe"
                )

                ## Estimated density under both the minimum of the proposal and constrained policies
                psi_hat_intersection_safe = iwmci_overlap_est(
                    LRs_unconstrained_over_safe=lik_ratios_unconstrained_over_safe_dict[
                        "safe"
                    ],  ## 1D numpy array
                    unconstrained_liks=unconstrained_liks_dict["safe"],
                    safe_liks=safe_liks_dict["safe"],
                    beta_t=beta_t,  ## float
                    psi_t=psi_hat_t_safe,
                    proposal="safe",
                )

            else:
                psi_hat_intersection_safe = 0.0  # None
                psi_hat_t_safe = sys.float_info.min

            ## If using mixture proposal, flag for when to swtich to it (from safe proposal)
            switch_to_mixture_proposal = (
                cfg.conformal_policy_control.mixture_proposal
                and (
                    cfg.conformal_policy_control.mixture_proposal_factor
                    * psi_hat_intersection_safe
                    < psi_hat_intersection_unconstrained
                    and cfg.conformal_policy_control.alpha < 1.0
                    and beta_t > cfg.conformal_policy_control.min_beta_mix_prop
                )
            )


            ## If not using mixture proposal, flag for whent o switch to optimized proposal (from safe proposal)
            switch_to_optimized_proposal = (
                not switch_to_mixture_proposal
                and (
                    cfg.conformal_policy_control.optimized_proposal_factor
                    * psi_hat_intersection_safe
                    < psi_hat_intersection_unconstrained
                    and cfg.conformal_policy_control.alpha < 1.0
                    and beta_t > cfg.conformal_policy_control.min_beta_mix_prop
                )
            )

            ## If accept_reject_sample_and_get_likelihoods terminated early for unconstrained model (which can only happen for alpha < 1.0), then just stick to safe proposal
            if (
                "unconstrained" not in unconstrained_df_dict
                or len(unconstrained_df_dict["unconstrained"])
                < cfg.conformal_policy_control.accept_reject.n_target_pre_cpc
            ):
                switch_to_mixture_proposal = False
                switch_to_optimized_proposal = False
                policy_names = ["safe"]

            if switch_to_mixture_proposal:
                ## If using mixture proposal, then include the appropriate number of safe and unconstrained proposals
                if cfg.conformal_policy_control.alpha >= 1.0:
                    safe_prop_mix_weight = 0.0
                elif cfg.conformal_policy_control.use_overlap_mix_weight:
                    # safe_prop_mix_weight = psi_hat_intersection_safe / (
                    #     psi_hat_intersection_safe + psi_hat_intersection_unconstrained
                    # )
                    if isinstance(cfg.conformal_policy_control.min_safe_mix_weight, (int, float)):
                        min_safe_mix_weight = cfg.conformal_policy_control.min_safe_mix_weight
                    else:
                        min_safe_mix_weight = (1 - cfg.conformal_policy_control.alpha) ** 4
                    safe_prop_mix_weight = max(
                        psi_hat_intersection_safe
                        / (psi_hat_intersection_safe + psi_hat_intersection_unconstrained),
                        min_safe_mix_weight, 
                        # cfg.conformal_policy_control.min_safe_mix_weight,
                    )
                else:
                    logger.info(f"Using fixed safe mix weight: 1 / (1 + beta_t) = {1 / (1 + beta_t)}")
                    safe_prop_mix_weight = 1 / (1 + beta_t)

                n_safe_prop_include = int(
                    safe_prop_mix_weight
                    * len(prop_data_t0_safe_and_t_unconstrained_liks_dict["safe"])
                )
                n_unconstrained_prop_include = int(
                    (1 - safe_prop_mix_weight)
                    * len(
                        prop_data_t0_safe_and_t_unconstrained_liks_dict["unconstrained"]
                    )
                )

                if cfg.conformal_policy_control.mixture_proposal_subsample_cpc:
                    ## Subsample proposals from safe and unconstrained proposals, here first for constrained likelihoods
                    constrained_liks_df_safe = constrained_liks_df_dict["safe"].sample(
                        n=n_safe_prop_include
                    )  ## Constrained liks for safe proposal
                    constrained_liks_df_unconstrained = constrained_liks_df_dict[
                        "unconstrained"
                    ].sample(
                        n=n_unconstrained_prop_include
                    )  ## Constrained liks for unconstrained proposal
                    ## Use same indices to get unconstrained likelihoods corresponding to sampled proposal samples
                    unconstrained_liks_df_safe = unconstrained_df_dict["safe"].loc[
                        constrained_liks_df_safe.index
                    ]  ## Unconstrained liks for safe proposal
                    unconstrained_liks_df_unconstrained = unconstrained_df_dict[
                        "unconstrained"
                    ].loc[
                        constrained_liks_df_unconstrained.index
                    ]  ## Unconstrained liks for unconstrained proposal

                    constrained_liks_df = pd.concat(
                        [constrained_liks_df_safe, constrained_liks_df_unconstrained],
                        ignore_index=True,
                    )
                    unconstrained_df = pd.concat(
                        [
                            unconstrained_liks_df_safe,
                            unconstrained_liks_df_unconstrained,
                        ],
                        ignore_index=True,
                    )
                    
                    prop_data_t0_safe_and_t_unconstrained_liks = np.vstack(
                        (
                            prop_data_t0_safe_and_t_unconstrained_liks_dict["safe"].loc[
                                constrained_liks_df_safe.index
                            ],
                            prop_data_t0_safe_and_t_unconstrained_liks_dict[
                                "unconstrained"
                            ].loc[
                                constrained_liks_df_unconstrained.index
                            ],
                        )
                    )

                    prop_mixture_constrained_density = np.concatenate(
                        (
                            prop_mixture_constrained_density_dict["safe"].loc[
                                constrained_liks_df_safe.index
                            ],
                            prop_mixture_constrained_density_dict["unconstrained"].loc[
                                constrained_liks_df_unconstrained.index
                            ],
                        )
                    )
                    safe_liks = np.concatenate(
                        (
                            safe_liks_dict["safe"].loc[
                                constrained_liks_df_safe.index
                            ],
                            safe_liks_dict["unconstrained"].loc[
                                constrained_liks_df_unconstrained.index
                            ],
                        )
                    )
                    unconstrained_liks = np.concatenate(
                        (
                            unconstrained_liks_dict["safe"].loc[
                                constrained_liks_df_safe.index
                            ],
                            unconstrained_liks_dict["unconstrained"].loc[
                                constrained_liks_df_unconstrained.index
                            ],
                        )
                    )
                else:
                    constrained_liks_df = pd.concat(
                        [
                            constrained_liks_df_dict["safe"].iloc[:n_safe_prop_include],
                            constrained_liks_df_dict["unconstrained"].iloc[
                                :n_unconstrained_prop_include
                            ],
                        ],
                        ignore_index=True,
                    )
                    unconstrained_df = pd.concat(
                        [
                            unconstrained_df_dict["safe"].iloc[:n_safe_prop_include],
                            unconstrained_df_dict["unconstrained"].iloc[
                                :n_unconstrained_prop_include
                            ],
                        ],
                        ignore_index=True,
                    )

                    prop_data_t0_safe_and_t_unconstrained_liks = np.vstack(
                        (
                            prop_data_t0_safe_and_t_unconstrained_liks_dict["safe"][
                                :n_safe_prop_include
                            ],
                            prop_data_t0_safe_and_t_unconstrained_liks_dict[
                                "unconstrained"
                            ][:n_unconstrained_prop_include],
                        )
                    )

                    prop_mixture_constrained_density = np.concatenate(
                        (
                            prop_mixture_constrained_density_dict["safe"][
                                :n_safe_prop_include
                            ],
                            prop_mixture_constrained_density_dict["unconstrained"][
                                :n_unconstrained_prop_include
                            ],
                        )
                    )
                    safe_liks = np.concatenate(
                        (
                            safe_liks_dict["safe"][:n_safe_prop_include],
                            safe_liks_dict["unconstrained"][:n_unconstrained_prop_include],
                        )
                    )
                    unconstrained_liks = np.concatenate(
                        (
                            unconstrained_liks_dict["safe"][:n_safe_prop_include],
                            unconstrained_liks_dict["unconstrained"][
                                :n_unconstrained_prop_include
                            ],
                        )
                    )

            else:
                safe_prop_mix_weight = 1.0 if proposal == "safe" else 0.0

            if (
                i == 0
                and len(policy_names) > 1
                and (switch_to_mixture_proposal or switch_to_optimized_proposal)
            ):
                G = G[b:]
                break

            ## Compute constrained likelihoods for cal data on current candidate bound, beta_t
            if cfg.conformal_policy_control.constrain_against == "init":
                test_pt_factor = 1
                cal_constrained_liks_curr = constrain_likelihoods(
                    cfg,
                    cal_data_t0_safe_and_t_unconstrained_liks,
                    [betas_list[0], beta_t],
                    [psis_list[0], psi_hat_t],
                )
                prop_constrained_liks_curr = constrain_likelihoods(
                    cfg,
                    prop_data_t0_safe_and_t_unconstrained_liks,
                    [betas_list[0], beta_t],
                    [psis_list[0], psi_hat_t],
                )
            else:
                test_pt_factor = 2
                cal_constrained_liks_curr = constrain_likelihoods(
                    cfg,
                    cal_data_t0_safe_and_t_unconstrained_liks,
                    [betas_list[-1]] + [beta_t],
                    [psis_list[-1]] + [psi_hat_t],
                )
                prop_constrained_liks_curr = constrain_likelihoods(
                    cfg,
                    prop_data_t0_safe_and_t_unconstrained_liks,
                    [betas_list[-1]] + [beta_t],
                    [psis_list[-1]] + [psi_hat_t],
                )

            ## Compute (unnormalized) CP weights for cal data: current constrained likelihoods over mixture density
            w_cal = (
                cal_constrained_liks_curr[:, -1].flatten()
                / cal_mixture_constrained_density
            )

            ## Compute estimated test point weight as the expectation of the ratio, with probabilities in the expectation given by prop_constrained_liks_curr[:,-1]
            prop_constrained_liks_curr_t = prop_constrained_liks_curr[:, -1].flatten()

            w_test = np.max(
                prop_constrained_liks_curr_t / prop_mixture_constrained_density
            )

            test_pt_factor *= cfg.conformal_policy_control.test_pt_scale_factor  ## Can conservatively scale test point weight to est population max (1.0 is plug in)
            w_test *= test_pt_factor


            ## Concatenate and normalize
            sum_w_cal_test = np.sum(w_cal) + w_test
            w_cal_normalized = w_cal / sum_w_cal_test
            w_test_normalized = w_test / sum_w_cal_test
            w_infeasible_normalized = (
                np.sum(w_cal_normalized[cal_infeasible_indicators]) + w_test_normalized
            )

            # # Track metrics for this grid position
            # last_w_test = w_test_normalized #float(w_test) ## Track normalized test point weight instead of raw test point weight
            # last_switch_to_mixture = switch_to_mixture_proposal
            # last_switch_to_optimized = switch_to_optimized_proposal


            if cfg.conformal_policy_control.randomized_cpc:
                w_infeasible_normalized *= np.random.uniform()

            if (
                w_infeasible_normalized > adjusted_alpha
                or (w_infeasible_normalized <= adjusted_alpha and beta_t == np.inf)
                or adjusted_alpha >= 1.0
            ):
                ## Stopping condition: (1) First uncontrolled risk, return previous beta_t where risk is controlled, (2) Last beta_t (np.inf) and risk is controlled there, (3) Running uncontrolled
                
                # If candidate passes, track metrics for this grid position
                # last_w_test = w_test_normalized #float(w_test) ## Track normalized test point weight instead of raw test point weight
                # last_switch_to_mixture = switch_to_mixture_proposal
                # last_switch_to_optimized = switch_to_optimized_proposal

                ## If running with risk control, return previous beta_t where risk is controlled
                if adjusted_alpha < 1.0:
                    beta_t = beta_hat_t_curr

                # psi_hat_t = importance_weighted_monte_carlo_integration(
                #     lik_ratios_unconstrained_over_safe, beta_t, proposal
                # )
                psi_hat_t = last_psi_hat_t
                psi_hat_t_unconstrained = last_psi_hat_t_unconstrained
                psi_hat_t_safe = last_psi_hat_t_safe
                psi_hat_intersection_safe = last_psi_hat_intersection_safe
                psi_hat_intersection_unconstrained = last_psi_hat_intersection_unconstrained
                proposal = last_proposal
                switch_to_mixture_proposal = last_switch_to_mixture_proposal
                switch_to_optimized_proposal = last_switch_to_optimized_proposal

                if beta_t > sys.float_info.min and cfg.conformal_policy_control.alpha < 1.0:
                    prop_constrained_liks_curr = last_prop_constrained_liks
                    prop_mixture_constrained_density = last_prop_mixture_constrained_density
                    safe_liks = last_safe_liks
                    unconstrained_liks = last_unconstrained_liks
                    unconstrained_liks_dict = last_unconstrained_liks_dict
                    safe_liks_dict = last_safe_liks_dict
                    constrained_liks_df_dict = last_constrained_liks_df_dict
                    unconstrained_df_dict = last_unconstrained_df_dict
                    prop_data_t0_safe_and_t_unconstrained_liks_dict = last_prop_data_t0_safe_and_t_unconstrained_liks_dict
                    prop_mixture_constrained_density_dict = last_prop_mixture_constrained_density_dict
                    constrained_liks_df = last_constrained_liks_df
                    unconstrained_df = last_unconstrained_df
                # envelope_const_constrained_over_proposal = last_envelope_const_constrained_over_proposal

                logger.info(f"Selected beta_t = {beta_t}, psi_hat_t = {psi_hat_t}")
                logger.info(
                    f"cal weights normalized sum : {np.sum(w_cal_normalized[cal_infeasible_indicators])}"
                )
                logger.info(f"w_test_normalized sum : {w_test_normalized}")

                ## Compute constrained likelihoods for cal data on current candidate bound, beta_t

                ## Save proposals with cpc-constrained likelihoods
                constrained_liks_df_beta_hat = pd.concat(
                    [
                        constrained_liks_df.iloc[:, :-1],
                        pd.DataFrame(
                            {con_lik_col(n_cal_sets): prop_constrained_liks_curr[:, -1]}
                        ),
                    ],
                    axis=1,
                )
                constrained_liks_df_beta_hat_fp = os.path.join(
                    os.path.dirname(constrained_gen_liks_fp),
                    f"prop_likBeta{beta_t:.4g}_psiS{psi_hat_intersection_safe:.4g}_psiU{psi_hat_intersection_unconstrained:.4g}_mixProp{switch_to_mixture_proposal}_{os.path.basename(constrained_gen_liks_fp)}",
                )

                if cfg.overwrite_ig or not fs.exists(constrained_liks_df_beta_hat_fp):
                    constrained_liks_df_beta_hat.to_json(
                        constrained_liks_df_beta_hat_fp, orient="records", lines=True
                    )

                ## Estimate new envelope constant for constrained policy over proposal
                # if (
                #     cfg.conformal_policy_control.mixture_proposal
                #     or cfg.conformal_policy_control.mixture_proposal_factor
                #     * psi_hat_intersection_safe
                #     < psi_hat_intersection_unconstrained
                # ):
                # logger.info(f"switch_to_mixture_proposal: {switch_to_mixture_proposal}")
                # logger.info(f"proposal {proposal}")

            

                if switch_to_mixture_proposal:
                    if cfg.conformal_policy_control.alpha >= 1.0:
                        safe_prop_mix_weight = 0.0
                    elif cfg.conformal_policy_control.use_overlap_mix_weight:
                        # safe_prop_mix_weight = psi_hat_intersection_safe / (
                        #     psi_hat_intersection_safe
                        #     + psi_hat_intersection_unconstrained
                        # )
                        if isinstance(cfg.conformal_policy_control.min_safe_mix_weight, (int, float)):
                            min_safe_mix_weight = cfg.conformal_policy_control.min_safe_mix_weight
                        else:
                            min_safe_mix_weight = (1 - cfg.conformal_policy_control.alpha) ** 4
                        safe_prop_mix_weight = max(
                            psi_hat_intersection_safe
                            / (psi_hat_intersection_safe + psi_hat_intersection_unconstrained),
                            min_safe_mix_weight, 
                            # cfg.conformal_policy_control.min_safe_mix_weight,
                        )
                    else:
                        safe_prop_mix_weight = 1 / (1 + beta_t)

                    lik_ratios_constrained_over_proposal = prop_constrained_liks_curr[
                        :, -1
                    ] / (
                        safe_prop_mix_weight * safe_liks
                        + (1 - safe_prop_mix_weight) * unconstrained_liks
                    )

                elif proposal == "unconstrained":
                    lik_ratios_constrained_over_proposal = (
                        prop_constrained_liks_curr[:, -1]
                        / unconstrained_liks_dict[proposal]
                    )

                elif proposal == "safe":
                    lik_ratios_constrained_over_proposal = (
                        prop_constrained_liks_curr[:, -1] / safe_liks_dict[proposal]
                    )

                else:
                    raise ValueError(f"Invalid proposal: {proposal}")

                envelope_const_constrained_over_proposal = (
                    max(lik_ratios_constrained_over_proposal
                    * cfg.conformal_policy_control.accept_reject.env_const_scale_emp_max)
                )

                # if envelope_const_constrained_over_proposal != last_envelope_const_constrained_over_proposal and beta_t > sys.float_info.min:
                #     breakpoint()
                #     raise ValueError(f"Error: Not using proper env constant from last iter")

                if cfg.overwrite_ig or not fs.exists(unconstrained_gen_liks_fp):
                    unconstrained_df.to_json(
                        unconstrained_gen_liks_fp, orient="records", lines=True
                    )

                check_col_names(constrained_liks_df_beta_hat)
                check_col_names(unconstrained_df)

                search_metrics = CPCSearchMetrics(
                    beta_t=float(beta_t),
                    psi_hat_t=float(psi_hat_t),
                    grid_size=len(G_original),
                    grid_position_selected=beta_position_in_G_original, #max(b - 1, 0),
                    risk_margin=float(adjusted_alpha - last_w_infeasible_normalized),
                    w_test=last_w_test,
                    proposal_selected="mixture" if switch_to_mixture_proposal else proposal,
                    switch_to_mixture=switch_to_mixture_proposal,
                    switch_to_optimized=switch_to_optimized_proposal,
                    psi_hat_intersection_safe=float(psi_hat_intersection_safe),
                    psi_hat_intersection_unconstrained=float(
                        psi_hat_intersection_unconstrained
                    ),
                    envelope_const=float(envelope_const_constrained_over_proposal),
                )
                return CPCSearchResult(
                    beta_t=beta_t,
                    psi_hat_t=psi_hat_t,
                    constrained_liks_df=constrained_liks_df_beta_hat,
                    constrained_liks_fp=constrained_liks_df_beta_hat_fp,
                    unconstrained_df=unconstrained_df,
                    unconstrained_liks_fp=unconstrained_gen_liks_fp,
                    proposal="mixture" if switch_to_mixture_proposal else proposal,
                    psi_hat_intersection_safe=psi_hat_intersection_safe,
                    psi_hat_intersection_unconstrained=psi_hat_intersection_unconstrained,
                    envelope_const=envelope_const_constrained_over_proposal,
                    search_metrics=search_metrics,
                )

            else:
                ## beta_t appears safe, record it as the current candidate
                beta_position_in_G_original += 1
                last_w_test = w_test_normalized #float(w_test) ## Track normalized test point weight instead of raw test point weight
                last_w_infeasible_normalized = w_infeasible_normalized

                beta_hat_t_curr = beta_t
                last_psi_hat_t = psi_hat_t
                last_psi_hat_t_unconstrained = psi_hat_t_unconstrained
                last_psi_hat_t_safe = psi_hat_t_safe
                last_psi_hat_intersection_safe = psi_hat_intersection_safe
                last_psi_hat_intersection_unconstrained = psi_hat_intersection_unconstrained
                last_proposal = proposal
                last_switch_to_mixture_proposal = switch_to_mixture_proposal
                last_switch_to_optimized_proposal = switch_to_optimized_proposal
                last_prop_constrained_liks = prop_constrained_liks_curr
                last_prop_mixture_constrained_density = prop_mixture_constrained_density
                last_safe_liks = safe_liks
                last_unconstrained_liks = unconstrained_liks
                last_unconstrained_liks_dict = unconstrained_liks_dict
                last_safe_liks_dict = safe_liks_dict
                last_constrained_liks_df_dict = constrained_liks_df_dict
                last_unconstrained_df_dict = unconstrained_df_dict
                last_prop_data_t0_safe_and_t_unconstrained_liks_dict = prop_data_t0_safe_and_t_unconstrained_liks_dict
                last_prop_mixture_constrained_density_dict = prop_mixture_constrained_density_dict
                last_constrained_liks_df = constrained_liks_df
                last_unconstrained_df = unconstrained_df
                # last_envelope_const_constrained_over_proposal = envelope_const_constrained_over_proposal
                # psi_hat_t_curr = psi_hat_t
                # psi_hat_intersection_safe_curr = psi_hat_intersection_safe
                # psi_hat_intersection_unconstrained_curr = psi_hat_intersection_unconstrained
                # envelope_const_constrained_over_proposal_curr = envelope_const_constrained_over_proposal
                # safe_prop_mix_weight_curr = safe_prop_mix_weight


    ## If does not find a risk-controlling policy:
    logger.info(
        f"WARNING : Conformal Policy Control could not control risk at desired risk level {cfg.conformal_policy_control.alpha}, with the provided safe policy, returning with what safe policy could provide."
    )

    psi_hat_t = importance_weighted_monte_carlo_integration(
        lik_ratios_unconstrained_over_safe, beta_t, proposal
    )

    constrained_liks_df_beta_hat = pd.concat(
        [
            constrained_liks_df.iloc[:, :-1],
            pd.DataFrame({con_lik_col(n_cal_sets): prop_constrained_liks_curr[:, -1]}),
        ],
        axis=1,
    )
    constrained_liks_df_beta_hat_fp = os.path.join(
        os.path.dirname(constrained_gen_liks_fp),
        f"prop_alpha{cfg.conformal_policy_control.alpha}_uncontrolled_beta{beta_t:.3g}_{os.path.basename(constrained_gen_liks_fp)}",
    )
    if cfg.overwrite_ig or not fs.exists(constrained_liks_df_beta_hat_fp):
        constrained_liks_df_beta_hat.to_json(
            constrained_liks_df_beta_hat_fp, orient="records", lines=True
        )

    if cfg.overwrite_ig or not fs.exists(unconstrained_gen_liks_fp):
        unconstrained_df.to_json(
            unconstrained_gen_liks_fp, orient="records", lines=True
        )

    check_col_names(constrained_liks_df_beta_hat)
    check_col_names(unconstrained_df)

    search_metrics = CPCSearchMetrics(
        beta_t=float(beta_t),
        psi_hat_t=float(psi_hat_t),
        grid_size=len(G),
        grid_position_selected=b,
        risk_margin=float("nan"),  # risk not controlled
        w_test=last_w_test,
        proposal_selected=proposal,
        switch_to_mixture=last_switch_to_mixture_proposal,
        switch_to_optimized=last_switch_to_optimized_proposal,
        psi_hat_intersection_safe=float(psi_hat_intersection_safe),
        psi_hat_intersection_unconstrained=float(psi_hat_intersection_unconstrained),
        envelope_const=float(envelope_const_constrained_over_proposal),
    )
    return CPCSearchResult(
        beta_t=beta_t,
        psi_hat_t=psi_hat_t,
        constrained_liks_df=constrained_liks_df_beta_hat,
        constrained_liks_fp=constrained_liks_df_beta_hat_fp,
        unconstrained_df=unconstrained_df,
        unconstrained_liks_fp=unconstrained_gen_liks_fp,
        proposal=proposal,
        psi_hat_intersection_safe=psi_hat_intersection_safe,
        psi_hat_intersection_unconstrained=psi_hat_intersection_unconstrained,
        envelope_const=envelope_const_constrained_over_proposal,
        search_metrics=search_metrics,
    )
