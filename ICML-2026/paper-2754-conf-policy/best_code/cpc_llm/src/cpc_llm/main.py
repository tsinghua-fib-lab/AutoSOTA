from collections.abc import Callable

import hydra
import logging
import numpy as np
import os
import pandas as pd
import sys
import time


from .infrastructure.file_handler import LocalOrS3Client
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


from .calibrate.cpc_search import cpc_beta_search
from .calibrate.process_likelihoods import constrain_likelihoods, check_col_names
from .infer.rejection_sampling import (
    run_iterative_generation,
    accept_reject_sample_and_get_likelihoods,
)
from .infrastructure.orchestration import (
    create_propen_preference_dataset,
    create_propen_sft_dataset,
    generate_ga_dataset,
    run_compute_liks_all_models_and_cal_data,
    train_dpo,
    train_initial_sft,
    train_marge,
    train_sft,
)
from .data.combine_and_split import (
    combine_new_with_old_datasets,
    train_cal_split_gen_outputs,
)
from .data.select import get_seeds_from_training_data
from .data_contracts import (
    CHOSEN,
    HIGHER_SCORE,
    HIGHER_SCORE_PARTICLE,
    LOWER_SCORE,
    LOWER_SCORE_PARTICLE,
    PARTICLE,
    PROMPT,
    SCORE,
    con_lik_col,
    lik_col,
)
from .infer.generation_utils import get_temperatures
from .metrics import DatasetSizes, RoundSummary, StageTiming

logger = logging.getLogger(__name__)


def run_pipeline(cfg: DictConfig, on_round_complete: Callable[[], None] | None = None):
    """Run the CPC-LLM pipeline. Can be called directly with a DictConfig.

    Args:
        cfg: Hydra configuration for the pipeline.
        on_round_complete: Optional callback invoked after each training round
            completes. Used by modal_runner to commit the outputs volume so that
            completed rounds survive client disconnects.
    """

    for random_seed in range(cfg.initial_seed, cfg.last_seed + 1):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        logger.setLevel(cfg.log_level.upper())
        logger.info(OmegaConf.to_yaml(cfg, resolve=True))
        use_s3 = cfg.parent_output_dir is not None
        file_client_args = {"init_s3": use_s3}
        if use_s3:
            file_client_args["anon"] = False
            file_client_args["use_listings_cache"] = False
        file_client = LocalOrS3Client(**file_client_args)

        # Save full config to file!
        cfg_fp = (
            f"{cfg.parent_output_dir}/{cfg.run_name}/config.json"
            if use_s3
            else f"{cfg.local_output_dir}/{cfg.run_name}/config.json"
        )
        if not use_s3:
            os.makedirs(f"{cfg.local_output_dir}/{cfg.run_name}", exist_ok=True)
        if not cfg.overwrite and file_client.exists(cfg_fp):
            logger.info(f"{cfg_fp} already exists. Not overwiting.")
        else:
            cfg_dicts = pd.DataFrame([OmegaConf.to_container(cfg)])
            cfg_dicts.to_json(cfg_fp, orient="records", lines=True)

        ## Get name of policy improvement optimizer(s)
        setting = ""  ## Setting: optimizer and constrain_against
        pi_optimizer_name = ""
        if cfg.num_sft_rounds > 0:
            pi_optimizer_name += "sft"
            setting += f"sft_seed{random_seed}_{cfg.setting_str}"
        elif cfg.num_dpo_rounds > 0:
            pi_optimizer_name += "dpo"
            setting += f"dpo_seed{random_seed}_{cfg.setting_str}"
            # setting += f"dpo_CA{cfg.conformal_policy_control.constrain_against}_{cfg.conformal_policy_control.num_starts_beta_search}_ep{cfg.dpo.args.dpo_config.num_train_epochs}"
        elif cfg.num_marge_rounds > 0:
            pi_optimizer_name += "marge"
            setting += f"marge_seed{random_seed}_{cfg.setting_str}"
        else:
            raise ValueError("Must have at least one optimization round")

        ## Genetic algorithm to gather initial training data
        ga_data_dir = generate_ga_dataset(cfg, file_client)
        logger.info(f"GA dataset dir: {ga_data_dir}")

        ## Running list of all model paths
        all_model_paths = []

        all_model_paths.append(cfg.initial_model)
        logger.info(f"Initial model dir: {all_model_paths[-1]}")

        ## (Abandoned this step, due to issues with unconditional sampling)
        ## Sample conformal calibration sequences unconditionally from GPT model
        # uncon_gen_outputs, hd = run_unconditional_generation(
        #     cfg,
        #     file_client,
        #     # combined_marge_dataset_fp,
        #     ga_data_dir,
        #     ga_data_dir,
        #     model_dir=all_model_paths[-1],
        #     particle_field="particle",
        #     score_field="score",
        #     temps=[1.0],
        # )

        """Initialization: SFT pre-training, where generation has uniformly selected seeds to improve pretrained model"""
        all_prev_sft_datasets = []
        prev_round_outputs_fp = f"{ga_data_dir}/plain_pairs.jsonl"  ## TO DO: Update this to samples from GPT model
        prev_hd = None

        for i in tqdm(
            range(cfg.num_init_sft_rounds), desc="SFT Initialization Iterations"
        ):
            n = cfg.num_labels_after_first_round if i > 0 else None

            if cfg.retrain_initial_sft_each_trial:
                filename_prefix = f"sft_init_seed{random_seed}_r{i}_"
            else:
                filename_prefix = f"sft_init_r{i}_"

            sft_dataset_fp = create_propen_sft_dataset(
                cfg,
                file_client,
                prev_round_outputs_fp,
                filename_prefix=filename_prefix,
                n=n,
                initial_sft=True,
            )

            combined_sft_dataset_fp = combine_new_with_old_datasets(
                cfg, file_client, all_prev_sft_datasets, sft_dataset_fp, random_seed
            )
            logger.info(f"SFT dataset path: {combined_sft_dataset_fp}")
            all_prev_sft_datasets.append(sft_dataset_fp)

            train_from_scratch = all_model_paths[-1] == cfg.initial_model and hasattr(
                cfg, "initial_model_config"
            )

            if cfg.retrain_initial_sft_each_trial:
                initial_sft_run_name = f"sft_init_seed{random_seed}_r{i}"
            else:
                initial_sft_run_name = f"sft_init_r{i}"

            sft_dir = train_initial_sft(
                cfg,
                file_client,
                sft_dataset_fp,  # combined_sft_dataset_fp, ## Only training on most recently generated data
                ga_data_dir,
                initial_sft_run_name=initial_sft_run_name,
                model_dir=all_model_paths[-1],
                train_from_scratch=train_from_scratch,
            )

            if i == 0:
                seeds_fp = ""

            seeds_fp = get_seeds_from_training_data(
                cfg,
                file_client,
                prev_seeds_fp=seeds_fp,
                curr_training_data_fp=sft_dataset_fp,
                output_dir=sft_dir,
                sample_size=cfg.iterative_generation.init_args.sample_size,
                sampling_method=cfg.iterative_generation.init_args.sampling_method,
                pi_optimizer_name=pi_optimizer_name,
                setting=setting
                if i == cfg.num_init_sft_rounds - 1
                else "",  ## Only include setting string if is last SFT round and will use seeds for policy improvement (or if changing overall config)
                # random_seed = cfg.iterative_generation.init_args.seed, ## Use fixed random seed in initial SFT training
                random_seed=random_seed,
                first_iter=i == 0,
            )

            logger.info(f"Trained initial SFT model: {sft_dir}")
            all_model_paths.append(sft_dir)

            if on_round_complete is not None:
                try:
                    on_round_complete()
                except Exception:  # callback is opaque; don't abort pipeline on failure
                    logger.warning("on_round_complete callback failed", exc_info=True)

        if pi_optimizer_name == "dpo":
            higher_score_particle_field = PROMPT
            lower_score_particle_field = CHOSEN
            higher_score_field = "prompt_score"
            lower_score_field = "chosen_score"
        else:
            higher_score_particle_field = HIGHER_SCORE_PARTICLE
            lower_score_particle_field = LOWER_SCORE_PARTICLE
            higher_score_field = HIGHER_SCORE
            lower_score_field = LOWER_SCORE

        ## Generate examples from initial safe policy
        iter_gen_outputs_combined, iter_gen_outputs_list, hd = run_iterative_generation(
            cfg,
            file_client,
            seeds_fp,  # combined_sft_dataset_fp,
            ga_data_dir,
            sft_dir,
            higher_score_particle_field=higher_score_particle_field,
            lower_score_particle_field=lower_score_particle_field,
            higher_score_field=higher_score_field,
            lower_score_field=lower_score_field,
            temps=[cfg.temperature_init],
            first_iter=True,
            setting=setting,
            global_random_seed=random_seed,
        )

        ## Check that initial model is empirically safe
        init_gen_outputs_df = pd.read_json(
            iter_gen_outputs_list[0], orient="records", lines=True
        )
        if SCORE not in init_gen_outputs_df.columns or init_gen_outputs_df.empty:
            logger.warning(
                "No parsable outputs from initial generation — the model may be "
                "too small or untrained to produce valid particles. Skipping "
                "remaining pipeline stages for this seed."
            )
            continue
        init_gen_scores = init_gen_outputs_df[SCORE].to_numpy()
        np.isnan(init_gen_scores) | np.isinf(init_gen_scores)

        ## Initialize lists of models and seeds for policy improvement loop
        pi_model_fp_list = [all_model_paths[-1]]
        pi_seeds_filepaths_list = [seeds_fp]

        # ## Compute likelihoods for all initial generated data
        # gen_liks_fp, hd = run_compute_liks_all_models_and_cal_data(
        #     cfg,
        #     file_client,
        #     seeds_fp_list=pi_seeds_filepaths_list,
        #     prev_cal_data_fp_list=[],
        #     model_dir_list=pi_model_fp_list,
        #     target_fp=iter_gen_outputs_list[-1],
        #     temps=[cfg.temperature],
        # )
        # gen_liks_df = pd.read_json(gen_liks_fp, orient="records", lines=True)

        """Split last batch of generated outputs into training and calibration data"""
        cal_df, cal_unconstrained_output_path, train_df, train_output_path = (
            train_cal_split_gen_outputs(
                cfg,
                file_client,
                iter_gen_outputs_list[0],
                sft_dir,
                first_iter=True,
                setting=setting,
                random_seed=random_seed,
            )
        )  # , sample_num_cal=cfg.num_cal_per_step, sample_num_train=cfg.num_train_per_step)
        prev_round_outputs_fp = train_output_path  ## Hereon, prev_round_outputs_fp will only contain training data
        # cal_data_fp_list.append(cal_output_path)
        logger.info(
            f"cal_r0 (n_cal{i}={len(cal_df)}) output path: {cal_unconstrained_output_path}"
        )
        logger.info(
            f"train_r0 (n_tr{i}={len(train_df)}) output path: {train_output_path}"
        )

        ## Compute likelihoods for all initial generated data
        cal_unconstrained_output_path_list, hd = (
            run_compute_liks_all_models_and_cal_data(
                cfg,
                file_client,
                seeds_fp_list=pi_seeds_filepaths_list,
                prev_cal_data_fp_list=[],
                model_dir_list=pi_model_fp_list,
                target_fp=cal_unconstrained_output_path,
                temps=[cfg.temperature],
            )
        )
        cal_unconstrained_output_path = cal_unconstrained_output_path_list[0]

        ## Keep track of calibration data with *unconstrained* liklihoods
        cal_data_unconstrained_fp_list = [cal_unconstrained_output_path]

        cal_df = pd.read_json(
            cal_unconstrained_output_path_list[0], orient="records", lines=True
        )
        ## Save initial calibration data with constrained likelihoods
        cal_constrained_liks_df = cal_df.copy(deep=True)
        cal_constrained_liks_df = cal_constrained_liks_df[[PARTICLE, SCORE, lik_col(0)]]
        cal_constrained_liks_df = cal_constrained_liks_df.rename(
            columns={lik_col(0): con_lik_col(0)}
        )
        cal_constrained_output_path = os.path.join(
            os.path.dirname(cal_unconstrained_output_path),
            f"constrained_{os.path.basename(cal_unconstrained_output_path)}",
        )
        if cfg.overwrite_split_init or not file_client.exists(
            cal_constrained_output_path
        ):
            cal_constrained_liks_df.to_json(
                cal_constrained_output_path, orient="records", lines=True
            )

        ## Keep track of calibration data with *constrained* liklihoods
        cal_data_constrained_fp_list = [cal_constrained_output_path]

        betas_list = [np.inf]
        psis_list = [1.0]
        proposals_list = ["safe"]
        intersection_psis_safe_list = [1.0]
        intersection_psis_unconstrained_list = [1.0]
        envelope_const_constrained_over_proposal_list = [1.0]

        """SFT Policy Improvement Outer Loop, with Policy Control Inner Loop"""
        for i in tqdm(
            range(1, cfg.num_sft_rounds), desc="SFT Policy Improvement Iterations"
        ):
            t_round_start = time.perf_counter()
            # n = cfg.num_labels_after_first_round if i > 0 else None
            n = cfg.num_labels_after_first_round

            ## TRAINING
            ## Format data
            t0 = time.perf_counter()
            sft_dataset_fp = create_propen_sft_dataset(
                cfg,
                file_client,
                prev_round_outputs_fp,
                filename_prefix=f"{setting}_r{i}",
                n=n,
            )

            combined_sft_dataset_fp = combine_new_with_old_datasets(
                cfg, file_client, all_prev_sft_datasets, sft_dataset_fp, random_seed
            )
            logger.info(f"SFT dataset path: {combined_sft_dataset_fp}")
            all_prev_sft_datasets.append(sft_dataset_fp)

            train_from_scratch = all_model_paths[-1] == cfg.initial_model and hasattr(
                cfg, "initial_model_config"
            )

            ## Train new model
            sft_dir = train_sft(
                cfg,
                file_client,
                combined_sft_dataset_fp,  # sft_dataset_fp, # ## Only training on most recently generated data
                ga_data_dir,
                sft_run_name=f"{cfg.run_name}_alpha{cfg.conformal_policy_control.alpha}_{setting}_r{i}",
                model_dir=all_model_paths[-1],
                train_from_scratch=train_from_scratch,
            )
            training_s = time.perf_counter() - t0

            ## SELECT PROMPTS: Select new prompts/seeds from historical training data
            t0 = time.perf_counter()
            old_seeds_idx = 0 if cfg.select_old_seeds_from == "init" else -1
            seeds_fp = get_seeds_from_training_data(
                cfg,
                file_client,
                prev_seeds_fp=pi_seeds_filepaths_list[old_seeds_idx],
                curr_training_data_fp=sft_dataset_fp,
                output_dir=sft_dir,
                sample_size=cfg.iterative_generation.args.sample_size,
                sampling_method=cfg.iterative_generation.args.sampling_method,
                pi_optimizer_name=pi_optimizer_name,
                setting=setting,
                # random_seed = cfg.iterative_generation.args.seed,
                random_seed=random_seed,
            )
            pi_seeds_filepaths_list.append(seeds_fp)
            seed_selection_s = time.perf_counter() - t0

            ## Add new trained model to list
            logger.info(f"Trained SFT model: {sft_dir}")
            all_model_paths.append(sft_dir)
            pi_model_fp_list.append(sft_dir)

            if cfg.temperature_scaling:
                temps = get_temperatures(cfg, file_client, sft_dir, prev_hd)
            else:
                # temps = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
                temps = [cfg.temperature]
            logger.info(f"temps: {temps}")

            ## Add curr model unconstrained likelihoods to previously collected calibration data
            t0 = time.perf_counter()
            cal_all_liks_fp, hd = run_compute_liks_all_models_and_cal_data(
                cfg,
                file_client,
                seeds_fp_list=pi_seeds_filepaths_list,
                prev_cal_data_fp_list=cal_data_unconstrained_fp_list,
                # data_dir=ga_data_dir,
                model_dir_list=pi_model_fp_list,
                target_fp="",  ## Empty because this should already be updated #cal_output_path,
                # particle_field= "higher_score_particle",
                # score_field= "score",
                temps=[cfg.temperature],
            )
            likelihood_computation_s = time.perf_counter() - t0
            logger.info(f"cal_all_liks_fp : {cal_all_liks_fp}")

            # ## CONFORMAL POLICY CONTROL
            t0 = time.perf_counter()
            cpc_result = cpc_beta_search(
                cfg,
                file_client,
                model_dir_list=pi_model_fp_list,
                seeds_fp_list=pi_seeds_filepaths_list,
                prev_cal_data_unconstrained_liks_fp_list=cal_data_unconstrained_fp_list,  ## Should contain both cal data and *constrained* likelihoods
                prev_cal_data_constrained_liks_fp_list=cal_data_constrained_fp_list,  ## Should contain both cal data and *constrained* likelihoods
                betas_list=betas_list,
                psis_list=psis_list,  ## Normalization constants
                ga_data_dir=ga_data_dir,
                global_random_seed=random_seed,
            )
            cpc_search_s = time.perf_counter() - t0
            beta_t = cpc_result.beta_t
            psi_hat_t = cpc_result.psi_hat_t
            constrained_liks_df_beta_hat = cpc_result.constrained_liks_df
            unconstrained_df = cpc_result.unconstrained_df
            proposal = cpc_result.proposal
            psi_hat_intersection_safe = cpc_result.psi_hat_intersection_safe
            psi_hat_intersection_unconstrained = (
                cpc_result.psi_hat_intersection_unconstrained
            )
            envelope_const_constrained_over_proposal = cpc_result.envelope_const
            ar_sampling_s = 0.0

            betas_list.append(beta_t)
            psis_list.append(psi_hat_t)
            proposals_list.append(proposal)
            intersection_psis_safe_list.append(psi_hat_intersection_safe)
            intersection_psis_unconstrained_list.append(
                psi_hat_intersection_unconstrained
            )
            envelope_const_constrained_over_proposal_list.append(
                envelope_const_constrained_over_proposal
            )

            if constrained_liks_df_beta_hat.iloc[0, 2] != unconstrained_df.iloc[0, 2]:
                ## Sanity check: First likelihood column should be same for safe policy
                raise ValueError(
                    f"constrained_liks_df_beta_hat.iloc[0,2] ({constrained_liks_df_beta_hat.iloc[0, 2]}) != ({unconstrained_df.iloc[0, 2]}) unconstrained_df.iloc[0,2]"
                )

            ## Add constrained likelihoods for the current model to previous calibration data
            for c_i, cal_data_constrained_fp in enumerate(cal_data_constrained_fp_list):
                ## Load currently available constrained (0:t-1) and unconstrained (0:t) likelihoods
                cal_data_constrained_curr = pd.read_json(
                    cal_data_constrained_fp, orient="records", lines=True
                )
                cal_data_unconstrained_curr = pd.read_json(
                    cal_data_unconstrained_fp_list[c_i], orient="records", lines=True
                )

                if cal_data_constrained_curr.empty or cal_data_unconstrained_curr.empty:
                    empty = []
                    if cal_data_constrained_curr.empty:
                        empty.append(f"constrained={cal_data_constrained_fp}")
                    if cal_data_unconstrained_curr.empty:
                        empty.append(
                            f"unconstrained={cal_data_unconstrained_fp_list[c_i]}"
                        )
                    logger.warning(
                        f"Skipping calibration index {c_i}, empty file(s): {', '.join(empty)}"
                    )
                    continue

                num_lik_cols_constrained_curr = (
                    len(cal_data_constrained_curr.columns) - 2
                )  ## Number of steps for which constrained likelihood has been computed

                ## Compute constrained likelihoods for (t), which only requires constrained from (t-1) and unconstrained for (t)
                if cfg.conformal_policy_control.constrain_against == "init":
                    cal_liks_df_t0_safe_and_t_unconstrained_mat = pd.concat(
                        [
                            cal_data_constrained_curr[con_lik_col(0)],
                            cal_data_unconstrained_curr.iloc[:, -1],
                        ],
                        axis=1,
                    ).to_numpy()  ## Double check this
                    cal_constrained_t_curr = constrain_likelihoods(
                        cfg,
                        cal_liks_df_t0_safe_and_t_unconstrained_mat,
                        [betas_list[0], betas_list[-1]],
                        [psis_list[0], psis_list[-1]],
                    )
                else:
                    cal_liks_df_t0_safe_and_t_unconstrained_mat = pd.concat(
                        [
                            cal_data_constrained_curr.iloc[:, -1],
                            cal_data_unconstrained_curr.iloc[:, -1],
                        ],
                        axis=1,
                    ).to_numpy()
                    cal_constrained_t_curr = constrain_likelihoods(
                        cfg,
                        cal_liks_df_t0_safe_and_t_unconstrained_mat,
                        betas_list,
                        psis_list,
                    )

                ## Add recently computed constrained likelihoods for (t) to the previously computed (0:t-1) values
                constrained_liks_df_beta_hat = pd.concat(
                    [
                        cal_data_constrained_curr,
                        pd.DataFrame({con_lik_col(i): cal_constrained_t_curr[:, -1]}),
                    ],
                    axis=1,
                )

                # constrained_liks_df_beta_hat_fp = os.path.join(os.path.dirname(constrained_gen_liks_fp), f"cpc_prop_{constrained_gen_liks_fp}")
                if (
                    cfg.overwrite_ig
                    or not file_client.exists(cal_data_constrained_fp)
                    or num_lik_cols_constrained_curr < i + 1
                ):
                    constrained_liks_df_beta_hat.to_json(
                        cal_data_constrained_fp, orient="records", lines=True
                    )

            ## Compute likelihoods for all initial generated data
            gen_liks_fp, hd = run_compute_liks_all_models_and_cal_data(
                cfg,
                file_client,
                seeds_fp_list=pi_seeds_filepaths_list,
                prev_cal_data_fp_list=[],  ## Empty because not updating previous cal data likelihoods here
                model_dir_list=pi_model_fp_list,
                target_fp=iter_gen_outputs_list[-1],
                temps=[cfg.temperature],
            )

            """Split last batch of generated outputs into training and calibration data"""
            t0 = time.perf_counter()
            cal_df, cal_unconstrained_output_path, train_df, train_output_path = (
                train_cal_split_gen_outputs(
                    cfg,
                    file_client,
                    iter_gen_outputs_list[0],
                    sft_dir,
                    setting=setting,
                    random_seed=random_seed,
                )
            )  # , sample_num_cal=cfg.num_cal_per_step, sample_num_train=cfg.num_train_per_step)
            dataset_split_s = time.perf_counter() - t0
            prev_round_outputs_fp = train_output_path  ## Hereon, prev_round_outputs_fp will only contain training data
            cal_data_unconstrained_fp_list.append(cal_unconstrained_output_path)
            logger.info(
                f"cal_r0 (n_cal{i}={len(cal_df)}) output path: {cal_unconstrained_output_path}"
            )
            logger.info(
                f"train_r0 (n_tr{i}={len(train_df)}) output path: {train_output_path}"
            )

            ## Save new calibration data with constrained likelihoods
            cal_constrained_liks_df = constrained_liks_df_beta_hat.loc[cal_df.index]
            cal_constrained_liks_df = cal_constrained_liks_df.rename(
                columns={lik_col(0): con_lik_col(0)}
            )
            cal_constrained_output_path = os.path.join(
                os.path.dirname(cal_unconstrained_output_path),
                f"cpc_constrained_{os.path.basename(cal_unconstrained_output_path)}",
            )
            if cfg.overwrite_ig or not file_client.exists(cal_constrained_output_path):
                cal_constrained_liks_df.to_json(
                    cal_constrained_output_path, orient="records", lines=True
                )

            ## Keep track of calibration data with *constrained* liklihoods
            cal_data_constrained_fp_list.append(cal_constrained_output_path)

            ## Write round summary
            round_summary = RoundSummary(
                round_idx=i,
                round_type="sft",
                cpc=cpc_result.search_metrics,
                ar_sampling=None,
                dataset_sizes=DatasetSizes(
                    n_cal=len(cal_df),
                    n_train=len(train_df),
                    n_combined=len(cal_df) + len(train_df),
                ),
                temperatures=temps,
                timing=StageTiming(
                    training_s=training_s,
                    seed_selection_s=seed_selection_s,
                    likelihood_computation_s=likelihood_computation_s,
                    cpc_search_s=cpc_search_s,
                    ar_sampling_s=ar_sampling_s,
                    dataset_split_s=dataset_split_s,
                    total_round_s=time.perf_counter() - t_round_start,
                ),
            )
            round_summary_fp = os.path.join(pi_model_fp_list[-1], "round_summary.json")
            if cfg.overwrite_ig or not file_client.exists(round_summary_fp):
                with file_client.open(round_summary_fp, "w") as f:
                    f.write(round_summary.model_dump_json(indent=2))
            logger.info(f"Round {i} summary written to {round_summary_fp}")

            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(round_summary.model_dump(), step=i)
            except ImportError:
                pass

            if on_round_complete is not None:
                try:
                    on_round_complete()
                except Exception:  # callback is opaque; don't abort pipeline on failure
                    logger.warning("on_round_complete callback failed", exc_info=True)

        all_prev_dpo_datasets = []
        """DPO Policy Improvement Outer Loop, with Policy Control Inner Loop"""
        for i in tqdm(
            range(1, cfg.num_dpo_rounds), desc="DPO Policy Improvement Iterations"
        ):
            t_round_start = time.perf_counter()

            ## TRAINING
            ## Format data
            n = cfg.num_labels_after_first_round
            t0 = time.perf_counter()
            dpo_dataset_fp = create_propen_preference_dataset(
                cfg,
                file_client,
                prev_round_outputs_fp,
                filename_prefix=f"alpha{cfg.conformal_policy_control.alpha}_{setting}_r{i}",
                n=n,
            )

            combined_dpo_dataset_fp = combine_new_with_old_datasets(
                cfg, file_client, all_prev_dpo_datasets, dpo_dataset_fp, random_seed
            )
            logger.info(f"DPO dataset path: {combined_dpo_dataset_fp}")
            all_prev_dpo_datasets.append(dpo_dataset_fp)

            train_from_scratch = all_model_paths[-1] == cfg.initial_model and hasattr(
                cfg, "initial_model_config"
            )

            ## Train new model
            dpo_dir = train_dpo(
                cfg,
                file_client,
                data_fp=combined_dpo_dataset_fp,  # dpo_dataset_fp, # ## Only training on most recently generated data
                ga_data_dir=ga_data_dir,
                run_name=f"{cfg.run_name}_alpha{cfg.conformal_policy_control.alpha}_{setting}_r{i}",
                ref_model_path=all_model_paths[-1],
                # train_from_scratch=train_from_scratch,
            )
            training_s = time.perf_counter() - t0

            ## SELECT PROMPTS: Select new prompts/seeds from historical training data
            t0 = time.perf_counter()
            old_seeds_idx = 0 if cfg.select_old_seeds_from == "init" else -1
            seeds_fp = get_seeds_from_training_data(
                cfg,
                file_client,
                prev_seeds_fp=pi_seeds_filepaths_list[old_seeds_idx],
                curr_training_data_fp=dpo_dataset_fp,
                output_dir=dpo_dir,
                sample_size=cfg.iterative_generation.args.sample_size,
                sampling_method=cfg.iterative_generation.args.sampling_method,
                higher_score_particle_field=PROMPT,
                lower_score_particle_field=CHOSEN,
                higher_score_field="prompt_score",
                lower_score_field="chosen_score",
                pi_optimizer_name=pi_optimizer_name,
                setting=setting,
                # random_seed = cfg.iterative_generation.args.seed,
                random_seed=random_seed,
            )
            pi_seeds_filepaths_list.append(seeds_fp)

            seed_selection_s = time.perf_counter() - t0

            ## Add new trained model to list
            logger.info(f"Trained DPO model: {dpo_dir}")
            all_model_paths.append(dpo_dir)
            pi_model_fp_list.append(dpo_dir)

            if cfg.temperature_scaling:
                temps = get_temperatures(cfg, file_client, dpo_dir, prev_hd)
            else:
                # temps = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
                temps = [cfg.temperature]
            logger.info(f"temps: {temps}")

            ## Add curr model unconstrained likelihoods to previously collected calibration data
            t0 = time.perf_counter()
            cal_all_liks_fp, hd = run_compute_liks_all_models_and_cal_data(
                cfg,
                file_client,
                seeds_fp_list=pi_seeds_filepaths_list,
                prev_cal_data_fp_list=cal_data_unconstrained_fp_list,
                # data_dir=ga_data_dir,
                model_dir_list=pi_model_fp_list,
                target_fp="",  ## Empty because this should already be updated #cal_output_path,
                # particle_field= "higher_score_particle",
                # score_field= "score",
                temps=[cfg.temperature],
            )
            likelihood_computation_s = time.perf_counter() - t0
            logger.info(f"cal_all_liks_fp : {cal_all_liks_fp}")

            # ## CONFORMAL POLICY CONTROL
            t0 = time.perf_counter()
            cpc_result = cpc_beta_search(
                cfg,
                file_client,
                model_dir_list=pi_model_fp_list,
                seeds_fp_list=pi_seeds_filepaths_list,
                prev_cal_data_unconstrained_liks_fp_list=cal_data_unconstrained_fp_list,
                prev_cal_data_constrained_liks_fp_list=cal_data_constrained_fp_list,
                betas_list=betas_list,
                psis_list=psis_list,
                ga_data_dir=ga_data_dir,
                higher_score_particle_field=PROMPT,
                lower_score_particle_field=CHOSEN,
                higher_score_field="prompt_score",
                lower_score_field="chosen_score",
                global_random_seed=random_seed,
            )
            cpc_search_s = time.perf_counter() - t0
            beta_t = cpc_result.beta_t
            psi_hat_t = cpc_result.psi_hat_t
            constrained_liks_df_beta_hat = cpc_result.constrained_liks_df
            unconstrained_df = cpc_result.unconstrained_df
            proposal = cpc_result.proposal
            psi_hat_intersection_safe = cpc_result.psi_hat_intersection_safe
            psi_hat_intersection_unconstrained = (
                cpc_result.psi_hat_intersection_unconstrained
            )
            envelope_const_constrained_over_proposal = cpc_result.envelope_const

            ## For now, just dealing with this edge case by continuing to next step with one action
            # n_safe_actions = max(1, n_safe_actions)

            betas_list.append(beta_t)
            psis_list.append(psi_hat_t)
            proposals_list.append(proposal)
            intersection_psis_safe_list.append(psi_hat_intersection_safe)
            intersection_psis_unconstrained_list.append(
                psi_hat_intersection_unconstrained
            )
            envelope_const_constrained_over_proposal_list.append(
                envelope_const_constrained_over_proposal
            )

            logger.info(f"constrained_liks_df_beta_hat.columns: {constrained_liks_df_beta_hat.columns}")
            logger.info(f"unconstrained_df.columns: {unconstrained_df.columns}")

            if constrained_liks_df_beta_hat.iloc[0, 2] != unconstrained_df.iloc[0, 2]:
                ## Sanity check: First likelihood column should be same for safe policy
                raise ValueError(
                    f"constrained_liks_df_beta_hat.iloc[0,2] ({constrained_liks_df_beta_hat.iloc[0, 2]}) != ({unconstrained_df.iloc[0, 2]}) unconstrained_df.iloc[0,2]"
                )

            check_col_names(constrained_liks_df_beta_hat)
            check_col_names(unconstrained_df)

            ## Add constrained likelihoods for the current model to previous calibration data
            for c_i, cal_data_constrained_fp in enumerate(cal_data_constrained_fp_list):
                cal_data_constrained_curr = pd.read_json(
                    cal_data_constrained_fp, orient="records", lines=True
                )
                cal_data_unconstrained_curr = pd.read_json(
                    cal_data_unconstrained_fp_list[c_i], orient="records", lines=True
                )

                if cal_data_constrained_curr.empty or cal_data_unconstrained_curr.empty:
                    empty = []
                    if cal_data_constrained_curr.empty:
                        empty.append(f"constrained={cal_data_constrained_fp}")
                    if cal_data_unconstrained_curr.empty:
                        empty.append(
                            f"unconstrained={cal_data_unconstrained_fp_list[c_i]}"
                        )
                    logger.warning(
                        f"Skipping calibration index {c_i}, empty file(s): {', '.join(empty)}"
                    )
                    continue

                num_lik_cols_constrained_curr = (
                    len(cal_data_constrained_curr.columns) - 2
                )  ## Number of steps for which constrained likelihood has been computed

                # constrained_liks_df_beta_hat_fp = os.path.join(os.path.dirname(constrained_gen_liks_fp), f"cpc_prop_{constrained_gen_liks_fp}")
                if (
                    cfg.overwrite_ig
                    or not file_client.exists(cal_data_constrained_fp)
                    or num_lik_cols_constrained_curr < i + 1
                ):
                    check_col_names(cal_data_constrained_curr)
                    check_col_names(cal_data_unconstrained_curr)

                    if cfg.conformal_policy_control.constrain_against == "init":
                        cal_liks_df_t0_safe_and_t_unconstrained_mat = pd.concat(
                            [
                                cal_data_constrained_curr[con_lik_col(0)],
                                cal_data_unconstrained_curr.iloc[:, -1],
                            ],
                            axis=1,
                        ).to_numpy()  ## Double check this
                    else:
                        cal_liks_df_t0_safe_and_t_unconstrained_mat = pd.concat(
                            [
                                cal_data_constrained_curr.iloc[:, -1],
                                cal_data_unconstrained_curr.iloc[:, -1],
                            ],
                            axis=1,
                        ).to_numpy()

                    ## Compute constrained likelihoods, only starting from most recent safe likelihoods
                    cal_constrained_t_curr = constrain_likelihoods(
                        cfg,
                        cal_liks_df_t0_safe_and_t_unconstrained_mat,
                        [betas_list[0], betas_list[-1]],
                        [psis_list[0], psis_list[-1]],
                    )

                    cal_constrained_liks_df_beta_hat = pd.concat(
                        [
                            cal_data_constrained_curr,
                            pd.DataFrame(
                                {f"con_lik_r{i}": cal_constrained_t_curr[:, -1]}
                            ),
                        ],
                        axis=1,
                    )
                    check_col_names(cal_constrained_liks_df_beta_hat)

                    cal_constrained_liks_df_beta_hat.to_json(
                        cal_data_constrained_fp, orient="records", lines=True
                    )

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
                )
            else:
                logger.info(f"Using fixed safe mix weight: 1 / (1 + beta_t) = {1 / (1 + beta_t)}")
                safe_prop_mix_weight = 1 / (1 + beta_t)

            if (
                cfg.conformal_policy_control.mixture_proposal
                and cfg.conformal_policy_control.mixture_proposal_factor
                * psi_hat_intersection_safe
                < psi_hat_intersection_unconstrained
                and cfg.conformal_policy_control.alpha < 1.0
                and beta_t > cfg.conformal_policy_control.min_beta_mix_prop
            ):
                proposal = "mixture"

            ## Sample with conformal policy control
            t0 = time.perf_counter()
            ar_result = accept_reject_sample_and_get_likelihoods(
                cfg,
                file_client,
                pi_model_fp_list,
                pi_seeds_filepaths_list,
                dpo_dir,
                betas_list,
                psis_list,
                cfg.conformal_policy_control.accept_reject.n_target_post_cpc,
                ga_data_dir,
                higher_score_particle_field=PROMPT,
                lower_score_particle_field=CHOSEN,
                higher_score_field="prompt_score",
                lower_score_field="chosen_score",
                proposal=proposal,
                post_policy_control=True,
                safe_prop_mix_weight=safe_prop_mix_weight,
                env_const=envelope_const_constrained_over_proposal,
                global_random_seed=random_seed,
            )
            ar_sampling_s = time.perf_counter() - t0
            unconstrained_df = ar_result.unconstrained_df
            unconstrained_gen_liks_fp = ar_result.unconstrained_fp
            constrained_liks_df = ar_result.constrained_df

            """Split last batch of generated outputs into training and calibration data"""
            t0 = time.perf_counter()
            cal_df, cal_unconstrained_output_path, train_df, train_output_path = (
                train_cal_split_gen_outputs(
                    cfg,
                    file_client,
                    unconstrained_gen_liks_fp,
                    dpo_dir,
                    setting=setting,
                    random_seed=random_seed,
                )
            )
            dataset_split_s = time.perf_counter() - t0
            prev_round_outputs_fp = train_output_path
            cal_data_unconstrained_fp_list.append(cal_unconstrained_output_path)
            logger.info(
                f"cal_r0 (n_cal{i}={len(cal_df)}) output path: {cal_unconstrained_output_path}"
            )
            logger.info(
                f"train_r0 (n_tr{i}={len(train_df)}) output path: {train_output_path}"
            )

            # cal_constrained_liks_df = constrained_liks_df_beta_hat.loc[cal_df.index]
            cal_constrained_liks_df = constrained_liks_df.loc[cal_df.index]
            cal_constrained_liks_df = cal_constrained_liks_df.rename(
                columns={lik_col(0): con_lik_col(0)}
            )
            cal_constrained_output_path = os.path.join(
                os.path.dirname(cal_unconstrained_output_path),
                f"cpc_constrained_{os.path.basename(cal_unconstrained_output_path)}",
            )
            if cfg.overwrite_ig or not file_client.exists(cal_constrained_output_path):
                cal_constrained_liks_df.to_json(
                    cal_constrained_output_path, orient="records", lines=True
                )

            ## Keep track of calibration data with *constrained* liklihoods
            cal_data_constrained_fp_list.append(cal_constrained_output_path)

            ## Write round summary
            round_summary = RoundSummary(
                round_idx=i,
                round_type="dpo",
                cpc=cpc_result.search_metrics,
                ar_sampling=ar_result.sampling_metrics,
                dataset_sizes=DatasetSizes(
                    n_cal=len(cal_df),
                    n_train=len(train_df),
                    n_combined=len(cal_df) + len(train_df),
                ),
                temperatures=temps,
                timing=StageTiming(
                    training_s=training_s,
                    seed_selection_s=seed_selection_s,
                    likelihood_computation_s=likelihood_computation_s,
                    cpc_search_s=cpc_search_s,
                    ar_sampling_s=ar_sampling_s,
                    dataset_split_s=dataset_split_s,
                    total_round_s=time.perf_counter() - t_round_start,
                ),
            )
            round_summary_fp = os.path.join(pi_model_fp_list[-1], "round_summary.json")
            if cfg.overwrite_ig or not file_client.exists(round_summary_fp):
                with file_client.open(round_summary_fp, "w") as f:
                    f.write(round_summary.model_dump_json(indent=2))
            logger.info(f"Round {i} summary written to {round_summary_fp}")

            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(round_summary.model_dump(), step=i)
            except ImportError:
                pass

            if on_round_complete is not None:
                try:
                    on_round_complete()
                except Exception:  # callback is opaque; don't abort pipeline on failure
                    logger.warning("on_round_complete callback failed", exc_info=True)

        all_prev_marge_datasets = []
        """MARGE Policy Improvement Outer Loop, with Policy Control Inner Loop"""
        for i in tqdm(range(1, cfg.num_marge_rounds), desc="MargE Iterations"):
            t_round_start = time.perf_counter()
            # n = cfg.num_labels_after_first_round if i > 0 else None

            ## TRAINING
            ## Format data
            n = cfg.num_labels_after_first_round
            t0 = time.perf_counter()
            marge_dataset_fp = create_propen_sft_dataset(
                cfg,
                file_client,
                prev_round_outputs_fp,
                filename_prefix=f"alpha{cfg.conformal_policy_control.alpha}_{setting}__r{i}",
                n=n,
            )
            combined_marge_dataset_fp = combine_new_with_old_datasets(
                cfg, file_client, all_prev_marge_datasets, marge_dataset_fp, random_seed
            )
            logger.info(f"MARGE dataset path: {combined_marge_dataset_fp}")
            all_prev_marge_datasets.append(marge_dataset_fp)

            train_from_scratch = all_model_paths[-1] == cfg.initial_model and hasattr(
                cfg, "initial_model_config"
            )

            ## Train new model
            marge_dir = train_marge(
                cfg,
                file_client,
                data_fp=combined_marge_dataset_fp,  # marge_dataset_fp, # ## Only training on most recently generated data
                ga_data_dir=ga_data_dir,
                run_name=f"{cfg.run_name}_alpha{cfg.conformal_policy_control.alpha}_{setting}_r{i}",
                ref_model_path=all_model_paths[-1],
                # train_from_scratch=train_from_scratch,
            )
            training_s = time.perf_counter() - t0

            ## SELECT PROMPTS: Select new prompts/seeds from historical training data
            t0 = time.perf_counter()
            old_seeds_idx = 0 if cfg.select_old_seeds_from == "init" else -1
            seeds_fp = get_seeds_from_training_data(
                cfg,
                file_client,
                prev_seeds_fp=pi_seeds_filepaths_list[old_seeds_idx],
                curr_training_data_fp=marge_dataset_fp,
                output_dir=marge_dir,
                sample_size=cfg.iterative_generation.args.sample_size,
                sampling_method=cfg.iterative_generation.args.sampling_method,
                pi_optimizer_name=pi_optimizer_name,
                setting=setting,
                # random_seed = cfg.iterative_generation.args.seed,
                random_seed=random_seed,
            )
            pi_seeds_filepaths_list.append(seeds_fp)
            seed_selection_s = time.perf_counter() - t0

            ## Add new trained model to list
            logger.info(f"Trained MARGE model: {marge_dir}")
            all_model_paths.append(marge_dir)
            pi_model_fp_list.append(marge_dir)

            if cfg.temperature_scaling:
                temps = get_temperatures(cfg, file_client, marge_dir, prev_hd)
            else:
                # temps = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
                temps = [cfg.temperature]
            logger.info(f"temps: {temps}")

            ## Add curr model unconstrained likelihoods to previously collected calibration data
            t0 = time.perf_counter()
            cal_all_liks_fp, hd = run_compute_liks_all_models_and_cal_data(
                cfg,
                file_client,
                seeds_fp_list=pi_seeds_filepaths_list,
                prev_cal_data_fp_list=cal_data_unconstrained_fp_list,
                # data_dir=ga_data_dir,
                model_dir_list=pi_model_fp_list,
                target_fp="",  ## Empty because this should already be updated #cal_output_path,
                # particle_field= "higher_score_particle",
                # score_field= "score",
                temps=[cfg.temperature],
            )
            likelihood_computation_s = time.perf_counter() - t0
            logger.info(f"cal_all_liks_fp : {cal_all_liks_fp}")

            ### CONFORMAL POLICY CONTROL
            t0 = time.perf_counter()
            cpc_result = cpc_beta_search(
                cfg,
                file_client,
                model_dir_list=pi_model_fp_list,
                seeds_fp_list=pi_seeds_filepaths_list,
                prev_cal_data_unconstrained_liks_fp_list=cal_data_unconstrained_fp_list,  ## Should contain both cal data and *constrained* likelihoods
                prev_cal_data_constrained_liks_fp_list=cal_data_constrained_fp_list,  ## Should contain both cal data and *constrained* likelihoods
                betas_list=betas_list,
                psis_list=psis_list,  ## Normalization constants
                ga_data_dir=ga_data_dir,
                global_random_seed=random_seed,
            )
            cpc_search_s = time.perf_counter() - t0
            beta_t = cpc_result.beta_t
            psi_hat_t = cpc_result.psi_hat_t
            constrained_liks_df_beta_hat = cpc_result.constrained_liks_df
            unconstrained_df = cpc_result.unconstrained_df
            proposal = cpc_result.proposal
            psi_hat_intersection_safe = cpc_result.psi_hat_intersection_safe
            psi_hat_intersection_unconstrained = (
                cpc_result.psi_hat_intersection_unconstrained
            )
            envelope_const_constrained_over_proposal = cpc_result.envelope_const

            ## For now, just dealing with this edge case by continuing to next step with one action
            # n_safe_actions = max(1, n_safe_actions)

            betas_list.append(beta_t)
            psis_list.append(psi_hat_t)
            proposals_list.append(proposal)
            intersection_psis_safe_list.append(psi_hat_intersection_safe)
            intersection_psis_unconstrained_list.append(
                psi_hat_intersection_unconstrained
            )
            envelope_const_constrained_over_proposal_list.append(
                envelope_const_constrained_over_proposal
            )

            if constrained_liks_df_beta_hat.iloc[0, 2] != unconstrained_df.iloc[0, 2]:
                ## Sanity check: First likelihood column should be same for safe policy
                raise ValueError(
                    f"constrained_liks_df_beta_hat.iloc[0,2] ({constrained_liks_df_beta_hat.iloc[0, 2]}) != ({unconstrained_df.iloc[0, 2]}) unconstrained_df.iloc[0,2]"
                )

            check_col_names(constrained_liks_df_beta_hat)
            check_col_names(unconstrained_df)

            ## Add constrained likelihoods for the current model to previous calibration data
            for c_i, cal_data_constrained_fp in enumerate(cal_data_constrained_fp_list):
                cal_data_constrained_curr = pd.read_json(
                    cal_data_constrained_fp, orient="records", lines=True
                )
                cal_data_unconstrained_curr = pd.read_json(
                    cal_data_unconstrained_fp_list[c_i], orient="records", lines=True
                )

                if cal_data_constrained_curr.empty or cal_data_unconstrained_curr.empty:
                    empty = []
                    if cal_data_constrained_curr.empty:
                        empty.append(f"constrained={cal_data_constrained_fp}")
                    if cal_data_unconstrained_curr.empty:
                        empty.append(
                            f"unconstrained={cal_data_unconstrained_fp_list[c_i]}"
                        )
                    logger.warning(
                        f"Skipping calibration index {c_i}, empty file(s): {', '.join(empty)}"
                    )
                    continue

                num_lik_cols_constrained_curr = (
                    len(cal_data_constrained_curr.columns) - 2
                )  ## Number of steps for which constrained likelihood has been computed

                # constrained_liks_df_beta_hat_fp = os.path.join(os.path.dirname(constrained_gen_liks_fp), f"cpc_prop_{constrained_gen_liks_fp}")
                if (
                    cfg.overwrite_ig
                    or not file_client.exists(cal_data_constrained_fp)
                    or num_lik_cols_constrained_curr < i + 1
                ):
                    check_col_names(cal_data_constrained_curr)
                    check_col_names(cal_data_unconstrained_curr)

                    if cfg.conformal_policy_control.constrain_against == "init":
                        cal_liks_df_t0_safe_and_t_unconstrained_mat = pd.concat(
                            [
                                cal_data_constrained_curr[con_lik_col(0)],
                                cal_data_unconstrained_curr.iloc[:, -1],
                            ],
                            axis=1,
                        ).to_numpy()  ## Double check this
                        ## Compute constrained likelihoods, only starting from most recent safe likelihoods
                        cal_constrained_t_curr = constrain_likelihoods(
                            cfg,
                            cal_liks_df_t0_safe_and_t_unconstrained_mat,
                            [betas_list[0], betas_list[-1]],
                            [psis_list[0], psis_list[-1]],
                        )
                    else:
                        cal_liks_df_t0_safe_and_t_unconstrained_mat = pd.concat(
                            [
                                cal_data_constrained_curr.iloc[:, -1],
                                cal_data_unconstrained_curr.iloc[:, -1],
                            ],
                            axis=1,
                        ).to_numpy()
                        cal_constrained_t_curr = constrain_likelihoods(
                            cfg,
                            cal_liks_df_t0_safe_and_t_unconstrained_mat,
                            betas_list,
                            psis_list,
                        )

                    cal_constrained_liks_df_beta_hat = pd.concat(
                        [
                            cal_data_constrained_curr,
                            pd.DataFrame(
                                {f"con_lik_r{i}": cal_constrained_t_curr[:, -1]}
                            ),
                        ],
                        axis=1,
                    )
                    check_col_names(cal_constrained_liks_df_beta_hat)

                    cal_constrained_liks_df_beta_hat.to_json(
                        cal_data_constrained_fp, orient="records", lines=True
                    )

            if cfg.conformal_policy_control.alpha >= 1.0:
                safe_prop_mix_weight = 0.0
            elif cfg.conformal_policy_control.use_overlap_mix_weight:
                if isinstance(cfg.conformal_policy_control.min_safe_mix_weight, (int, float)):
                    min_safe_mix_weight = cfg.conformal_policy_control.min_safe_mix_weight
                else:
                    min_safe_mix_weight = (1 - cfg.conformal_policy_control.alpha) ** 4
                safe_prop_mix_weight = max(
                    psi_hat_intersection_safe
                    / (psi_hat_intersection_safe + psi_hat_intersection_unconstrained),
                    min_safe_mix_weight, 
                )
            else:
                safe_prop_mix_weight = 1 / (1 + beta_t)

            if (
                cfg.conformal_policy_control.mixture_proposal
                and cfg.conformal_policy_control.mixture_proposal_factor
                * psi_hat_intersection_safe
                < psi_hat_intersection_unconstrained
                and cfg.conformal_policy_control.alpha < 1.0
                and beta_t > cfg.conformal_policy_control.min_beta_mix_prop
            ):
                proposal = "mixture"

            ## Sample with conformal policy control
            t0 = time.perf_counter()
            ar_result = accept_reject_sample_and_get_likelihoods(
                cfg,
                file_client,
                pi_model_fp_list,
                pi_seeds_filepaths_list,
                marge_dir,
                betas_list,
                psis_list,
                cfg.conformal_policy_control.accept_reject.n_target_post_cpc,
                ga_data_dir,
                proposal=proposal,
                post_policy_control=True,
                safe_prop_mix_weight=safe_prop_mix_weight,
                env_const=envelope_const_constrained_over_proposal,
                global_random_seed=random_seed,
            )
            ar_sampling_s = time.perf_counter() - t0
            unconstrained_df = ar_result.unconstrained_df
            unconstrained_gen_liks_fp = ar_result.unconstrained_fp
            constrained_liks_df = ar_result.constrained_df

            """Split last batch of generated outputs into training and calibration data"""
            t0 = time.perf_counter()
            cal_df, cal_unconstrained_output_path, train_df, train_output_path = (
                train_cal_split_gen_outputs(
                    cfg,
                    file_client,
                    unconstrained_gen_liks_fp,
                    marge_dir,
                    setting=setting,
                    random_seed=random_seed,
                )
            )  # , sample_num_cal=cfg.num_cal_per_step, sample_num_train=cfg.num_train_per_step)
            # train_cal_split_gen_outputs(cfg, file_client, unconstrained_liks_df_beta_hat_fp, marge_dir)
            dataset_split_s = time.perf_counter() - t0
            prev_round_outputs_fp = train_output_path  ## Hereon, prev_round_outputs_fp will only contain training data
            cal_data_unconstrained_fp_list.append(cal_unconstrained_output_path)
            logger.info(
                f"cal_r0 (n_cal{i}={len(cal_df)}) output path: {cal_unconstrained_output_path}"
            )
            logger.info(
                f"train_r0 (n_tr{i}={len(train_df)}) output path: {train_output_path}"
            )

            ## Save new calibration data with constrained likelihoods
            # cal_constrained_liks_df = constrained_liks_df_beta_hat.loc[cal_df.index]
            cal_constrained_liks_df = constrained_liks_df.loc[cal_df.index]
            cal_constrained_liks_df = cal_constrained_liks_df.rename(
                columns={lik_col(0): con_lik_col(0)}
            )
            cal_constrained_output_path = os.path.join(
                os.path.dirname(cal_unconstrained_output_path),
                f"cpc_constrained_{os.path.basename(cal_unconstrained_output_path)}",
            )
            if cfg.overwrite_ig or not file_client.exists(cal_constrained_output_path):
                cal_constrained_liks_df.to_json(
                    cal_constrained_output_path, orient="records", lines=True
                )

            ## Keep track of calibration data with *constrained* liklihoods
            cal_data_constrained_fp_list.append(cal_constrained_output_path)

            ## Write round summary
            round_summary = RoundSummary(
                round_idx=i,
                round_type="marge",
                cpc=cpc_result.search_metrics,
                ar_sampling=ar_result.sampling_metrics,
                dataset_sizes=DatasetSizes(
                    n_cal=len(cal_df),
                    n_train=len(train_df),
                    n_combined=len(cal_df) + len(train_df),
                ),
                temperatures=temps,
                timing=StageTiming(
                    training_s=training_s,
                    seed_selection_s=seed_selection_s,
                    likelihood_computation_s=likelihood_computation_s,
                    cpc_search_s=cpc_search_s,
                    ar_sampling_s=ar_sampling_s,
                    dataset_split_s=dataset_split_s,
                    total_round_s=time.perf_counter() - t_round_start,
                ),
            )
            round_summary_fp = os.path.join(pi_model_fp_list[-1], "round_summary.json")
            if cfg.overwrite_ig or not file_client.exists(round_summary_fp):
                with file_client.open(round_summary_fp, "w") as f:
                    f.write(round_summary.model_dump_json(indent=2))
            logger.info(f"Round {i} summary written to {round_summary_fp}")

            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(round_summary.model_dump(), step=i)
            except ImportError:
                pass

            if on_round_complete is not None:
                try:
                    on_round_complete()
                except Exception:  # callback is opaque; don't abort pipeline on failure
                    logger.warning("on_round_complete callback failed", exc_info=True)


_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "config")


@hydra.main(config_path=_CONFIG_PATH, config_name="pipeline")
def main(cfg: DictConfig):
    import sys; print("cpc-llm: process started, initializing...", file=sys.stderr, flush=True)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
