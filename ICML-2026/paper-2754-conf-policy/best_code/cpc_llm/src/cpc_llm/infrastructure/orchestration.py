import logging
import os

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, List, Mapping, Optional

from ..data.combine_and_split import combine_datasets
from .file_handler import LocalOrS3Client
from .slurm_utils import (
    submit_cmd_direct,
    submit_cmd_to_slurm,
    wait_for_direct_jobs_to_complete,
    wait_for_slurm_jobs_to_complete,
)

logger = logging.getLogger(__name__)


def _submit_cmd(cfg, py_cmd, dump_dir, blocking=True, **kwargs):
    """Dispatch command execution based on cfg.job_submission_system."""
    system = getattr(cfg, "job_submission_system", "slurm")
    if system == "direct":
        return submit_cmd_direct(py_cmd, dump_dir, blocking=blocking, **kwargs)
    else:
        return submit_cmd_to_slurm(py_cmd, dump_dir, blocking=blocking, **kwargs)


def _wait_for_jobs(cfg, jobs):
    """Wait for jobs to complete based on cfg.job_submission_system."""
    system = getattr(cfg, "job_submission_system", "slurm")
    if system == "direct":
        wait_for_direct_jobs_to_complete(jobs)
    else:
        wait_for_slurm_jobs_to_complete(jobs)


def get_all_strs_from_nested_dict(nested_dict: Dict[str, Any]) -> List[str]:
    outputs = []
    for k, v in nested_dict.items():
        if not isinstance(v, Mapping):
            outputs.append(f"{k}={v}")
        else:
            for o in get_all_strs_from_nested_dict(v):
                outputs.append(f"{k}.{o}")
    return outputs


def model_already_trained(
    cfg: DictConfig, fs: LocalOrS3Client, s3_output_dir: str, local_output_dir: str
) -> Optional[str]:
    model_fp_names = ["model.safetensors.index.json", "model.safetensors"]
    model_dir = s3_output_dir if cfg.parent_output_dir is not None else local_output_dir
    for model_fn in model_fp_names:
        fp = f"{model_dir}/{model_fn}"
        if fs.exists(fp):
            return model_dir
    return None


def _get_launch_prefix(cfg, slurm_cfg) -> str:
    """Build the command prefix for training jobs.

    Uses ``python -m`` for single-GPU direct execution (e.g. Modal, local dev),
    ``torchrun`` otherwise (multi-GPU or SLURM).
    """
    gpus_per_node = (
        slurm_cfg.gpus_per_node
        if hasattr(slurm_cfg, "gpus_per_node")
        else int(slurm_cfg.gres.split(":")[-1])
    )
    system = getattr(cfg, "job_submission_system", "slurm")
    if system == "direct" and gpus_per_node <= 1 and slurm_cfg.nodes == 1:
        return "python -m "
    return (
        f"torchrun --standalone --nnodes={slurm_cfg.nodes} "
        f"--nproc-per-node={gpus_per_node} -m "
    )


def generate_ga_dataset(cfg: DictConfig, fs: LocalOrS3Client) -> str:
    """
    Either run genetic algorithm to generate dataset, or simply return directory containing data.
    """
    python_cmd_str = "python -m cpc_llm.data.synthetic_dataset_generator "
    opts = get_all_strs_from_nested_dict(cfg["evol_dataset_gen"]["args"])
    opts_str = " ".join(opts)
    python_cmd_str += f"{opts_str} "
    num_motifs = cfg["evol_dataset_gen"]["args"]["test_function"]["num_motifs"]
    motif_length = cfg["evol_dataset_gen"]["args"]["test_function"]["motif_length"]
    dim = cfg["evol_dataset_gen"]["args"]["test_function"]["dim"]
    num_particles = cfg["evol_dataset_gen"]["args"]["optimizer"]["num_particles"]
    mutation_prob = cfg["evol_dataset_gen"]["args"]["optimizer"]["mutation_prob"]
    num_opt_steps = cfg["evol_dataset_gen"]["args"]["num_opt_steps"]
    ds_name = f"c{num_motifs}_k{motif_length}_l{dim}_n{num_particles}_pm{mutation_prob}_steps{num_opt_steps}"
    output_dir = (
        f"{cfg.parent_output_dir}/{cfg.run_name}/{ds_name}"
        if cfg.parent_output_dir is not None
        else f"{cfg.local_output_dir}/{cfg.run_name}/{ds_name}"
    )
    if not output_dir.startswith("s3://"):
        os.makedirs(output_dir, exist_ok=True)
    python_cmd_str += f"output_dir={output_dir} "
    output_fp = f"{output_dir}/plain_pairs.jsonl"
    if not cfg.overwrite_ga and fs.exists(output_fp):
        logger.info(f"{output_fp} already exists. Skipping...")
        return output_dir
    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"
    os.makedirs(slurm_dump_dir, exist_ok=True)
    if cfg.run_evol_dataset_gen:
        slurm_kwargs = OmegaConf.to_container(cfg.evol_dataset_gen.slurm_args)
        slurm_kwargs["job_name"] = "ga_seeds"

        _submit_cmd(
            cfg,
            python_cmd_str,
            slurm_dump_dir,
            blocking=True,
            path_to_repo=cfg.path_to_repo,
            **slurm_kwargs,
        )

    return output_dir


def create_propen_sft_dataset(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    source_dataset_fp: str,
    filename_prefix: str = "",
    n: int = None,
    initial_sft: bool = False,  ## Whether is initialization (False : means policy improvement or extrapolation)
    **extra_kwargs,
) -> str:
    python_cmd_str = "python -m cpc_llm.data.synthetic_dataset_formatter "
    if initial_sft:
        opts = get_all_strs_from_nested_dict(
            cfg["propen_dataset_formatting_initial_sft"]["args"]
        )
    else:
        opts = get_all_strs_from_nested_dict(
            cfg["propen_dataset_formatting_sft"]["args"]
        )
    opts_str = " ".join(opts)
    opts_str += (
        f" source_dataset_path={source_dataset_fp} format=dense_neighborhood_pairs "
    )
    if initial_sft:
        pdf_cfg = cfg.propen_dataset_formatting_initial_sft.args
    else:
        pdf_cfg = cfg.propen_dataset_formatting_sft.args

    if initial_sft:
        ## Want same initial SFT dataset for both policy control and no PC
        output_fn = f"{filename_prefix}dense_neighborhood_pairs_xthres{pdf_cfg.dist_x_threshold}_maxinfs{pdf_cfg.max_proportion_infeasible}_{pdf_cfg.n_neighbors}nn.jsonl"
    else:
        ## If not initial SFT, then keep track of whether doing policy control (ie, alpha level)
        output_fn = f"alpha{cfg.conformal_policy_control.alpha}_{filename_prefix}dense_neighborhood_pairs_xthres{pdf_cfg.dist_x_threshold}_maxinfs{pdf_cfg.max_proportion_infeasible}_{pdf_cfg.n_neighbors}nn.jsonl"

    output_fp = (
        f"{cfg.parent_output_dir}/{cfg.run_name}/{output_fn}"
        if cfg.parent_output_dir is not None
        else f"{cfg.local_output_dir}/{cfg.run_name}/{output_fn}"
    )
    if not output_fp.startswith("s3://"):
        os.makedirs(os.path.dirname(output_fp), exist_ok=True)
    opts_str += f"output_path={output_fp} "
    for k, v in extra_kwargs.items():
        opts_str += f"{k}={v} "
    if n is not None:
        opts_str += f"n={n} "
    python_cmd_str += f"{opts_str} "
    overwrite_sft_flag = (
        cfg.overwrite_init_sft_formatter if initial_sft else cfg.overwrite_sft_formatter
    )
    if not overwrite_sft_flag and fs.exists(output_fp):
        logger.info(f"{output_fp} already exists. Skipping...")
        return output_fp
    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"
    os.makedirs(slurm_dump_dir, exist_ok=True)
    if cfg.run_propen_sft_dataset_formatting:
        if initial_sft:
            slurm_kwargs = OmegaConf.to_container(
                cfg.propen_dataset_formatting_initial_sft.slurm_args
            )
            slurm_kwargs["job_name"] = "propen_initial_sft_formatting"
        else:
            slurm_kwargs = OmegaConf.to_container(
                cfg.propen_dataset_formatting_sft.slurm_args
            )
            slurm_kwargs["job_name"] = "propen_sft_formatting"
        _submit_cmd(
            cfg,
            python_cmd_str,
            slurm_dump_dir,
            blocking=True,
            path_to_repo=cfg.path_to_repo,
            **slurm_kwargs,
        )
    return output_fp


def create_propen_preference_dataset(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    source_dataset_fp: str,
    filename_prefix: str = "",
    n: int = None,
) -> str:
    python_cmd_str = "python -m cpc_llm.data.synthetic_dataset_formatter "
    opts = get_all_strs_from_nested_dict(
        cfg["propen_dataset_formatting_preference"]["args"]
    )
    opts_str = " ".join(opts)
    opts_str += (
        f" source_dataset_path={source_dataset_fp} format=dense_preference_pairs "
    )
    pdf_cfg = cfg.propen_dataset_formatting_preference.args
    output_fn = f"{filename_prefix}xthres{pdf_cfg.dist_x_threshold}_maxinfs{pdf_cfg.max_proportion_infeasible}_{pdf_cfg.n_neighbors}nn.jsonl"
    output_fp = (
        f"{cfg.parent_output_dir}/{cfg.run_name}/{output_fn}"
        if cfg.parent_output_dir is not None
        else f"{cfg.local_output_dir}/{cfg.run_name}/{output_fn}"
    )
    if not output_fp.startswith("s3://"):
        os.makedirs(os.path.dirname(output_fp), exist_ok=True)
    opts_str += f"output_path={output_fp} "
    if n is not None:
        opts_str += f"n={n} "

    python_cmd_str += f"{opts_str} "
    if not cfg.overwrite_dpo_formatter and fs.exists(output_fp):
        logger.info(f"{output_fp} already exists. Skipping...")
        return output_fp

    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"
    os.makedirs(slurm_dump_dir, exist_ok=True)
    if cfg.run_propen_dpo_dataset_formatting:
        slurm_kwargs = OmegaConf.to_container(
            cfg.propen_dataset_formatting_preference.slurm_args
        )
        slurm_kwargs["job_name"] = "propen_dpo_formatting"
        _submit_cmd(
            cfg,
            python_cmd_str,
            slurm_dump_dir,
            blocking=True,
            path_to_repo=cfg.path_to_repo,
            **slurm_kwargs,
        )
    return output_fp


def run_iterative_generation(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    data_fp: str,  ## Either path to paired training data (to select seeds from) or path to pre-selected seeds (unpaired)
    data_dir: str,
    model_dir: str,
    output_dir: str = None,
    higher_score_particle_field: str = "higher_score_particle",
    lower_score_particle_field: str = "lower_score_particle",
    lower_score_field: str = "lower_score",
    higher_score_field: str = "higher_score",
    temps: List[float] = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
    return_seeds: bool = False,
    first_iter: bool = False,
    model_idx: int = 0,  ## Time index of model using for (unconstrained) generation
    call_idx: int = 0,  ## Index for this generation has been called, including current, for same model directory
    proportion_of_target_n_accepted: float = None,  ## If being run as submodule of AR-sampling, the proportion of target samples accepted
    post_policy_control: bool = False,
    setting: str = "",
    global_random_seed: int = 0,
    # proposal: str = 'unconstrained',
):
    """
    Runs iterative generation jobs, combines the outputs, and returns the combined output filepath.
    """

    ## Arbitrary way of standardizing seeds for checkpointing, while ensuring they're different across calls within an experiment (assuming no more than 1000 calls of same model at a step)
    random_seed_curr = (
        global_random_seed * 10000 + post_policy_control * 1000 + call_idx
    )

    if output_dir is None:
        output_dir = model_dir

    opt_str = " ".join(
        get_all_strs_from_nested_dict(cfg["iterative_generation"]["args"])
    )
    args = f"{opt_str} data_path={data_fp} model_name_or_path={model_dir} output_dir={output_dir} "
    args += f"test_fn_fp={data_dir}/ehrlich.jsonl "
    args += f"higher_score_particle_field={higher_score_particle_field} "
    args += f"lower_score_particle_field={lower_score_particle_field} "
    args += f"lower_score_field={lower_score_field} "
    args += f"higher_score_field={higher_score_field} "
    args += f"sanity_check={cfg.sanity_check} "
    args += f"seed={random_seed_curr} "
    args += f"permissive_parsing={cfg.iterative_generation.permissive_parsing} "
    args += f"min_rel_len_feasible_particle={cfg.iterative_generation.min_rel_len_feasible_particle} "
    # args += f"first_iter={cfg.first_iter}"

    if first_iter:
        args += f"sample_size={cfg.iterative_generation.init_args.sample_size} "
        args += f"max_iterations={cfg.iterative_generation.init_args.max_iterations} "
        args += f"sampling_method={cfg.iterative_generation.init_args.sampling_method} "
        ## For first_iter==True (initial generation), have the string be different only for each setting (incl. random seed), not for each alpha level
        output_filename_prefix = f"gens_likelihood_{setting}_{cfg.iterative_generation.init_args.sample_size}sample_{cfg.iterative_generation.init_args.max_iterations}iter"
    else:
        args += f"sample_size={cfg.iterative_generation.args.sample_size} "
        args += f"max_iterations={cfg.iterative_generation.args.max_iterations} "
        args += f"sampling_method={cfg.iterative_generation.args.sampling_method} "
        if proportion_of_target_n_accepted is not None:
            output_filename_prefix = f"alpha{cfg.conformal_policy_control.alpha}_postPC{post_policy_control}_model{model_idx}_cn{call_idx}_propAcc{proportion_of_target_n_accepted:.3g}_gens_likelihood_{cfg.iterative_generation.args.sample_size}sample_{cfg.iterative_generation.args.max_iterations}iter"
        else:
            output_filename_prefix = f"alpha{cfg.conformal_policy_control.alpha}_postPC{post_policy_control}_model{model_idx}_cn{call_idx}_gens_likelihood_{cfg.iterative_generation.args.sample_size}sample_{cfg.iterative_generation.args.max_iterations}iter"

    greedy_decoding_gen_args = f"generation_config.do_sample=False generation_config.num_beams=1 batch_size={cfg.greedy_gen_batch_size}"
    temp_sampling_gen_args = [
        "generation_config.do_sample=True generation_config.num_beams=1 "
        + f"+generation_config.temperature={temp} "
        + f"generation_config.num_return_sequences={cfg.generation_sampling_num_return_sequences} "
        + f"batch_size={cfg.sampling_gen_batch_size} "
        for temp in temps
    ]
    if cfg.greedy_decoding:
        all_gen_args = [greedy_decoding_gen_args, *temp_sampling_gen_args]
        output_filenames = [
            f"{output_filename_prefix}_greedy.jsonl",
            *[
                f"{output_filename_prefix}_temp{temp}_{cfg.generation_sampling_num_return_sequences}seqs.jsonl"
                for temp in temps
            ],
        ]
    else:
        all_gen_args = temp_sampling_gen_args

        output_filenames = [
            f"{output_filename_prefix}_temp{temp}_{cfg.generation_sampling_num_return_sequences}seqs.jsonl"
            for temp in temps
        ]

    seeds_filenames = [
        f"seeds_{output_filename}" for output_filename in output_filenames
    ]
    seeds_filepaths = [f"{output_dir}/{seeds_fn}" for seeds_fn in seeds_filenames]

    output_filepaths = [f"{output_dir}/{output_fn}" for output_fn in output_filenames]
    combined_outputs_fp = f"{output_dir}/{output_filename_prefix}.jsonl"
    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"
    os.makedirs(slurm_dump_dir, exist_ok=True)
    hd = None
    if cfg.run_iter_gen:
        all_args = []
        for gen_args, output_fn in zip(all_gen_args, output_filenames):
            ## If first_iter, use overwrite_initg flag rather than overwrite_ig
            overwrite_flag = cfg.overwrite_initg if first_iter else cfg.overwrite_ig
            if not overwrite_flag and fs.exists(f"{output_dir}/{output_fn}"):
                logger.info(f"{output_dir}/{output_fn} already exists. Skipping...")
            else:
                all_args.append(f"{args} {gen_args} output_filename={output_fn}")
        all_python_commands = [
            f"python -m cpc_llm.infer.iterative_generation2 {a}" for a in all_args
        ]
        slurm_kwargs = OmegaConf.to_container(cfg.iterative_generation.slurm_args)
        slurm_kwargs["job_name"] = "iter_gen"
        job_submissions = [
            _submit_cmd(
                cfg,
                py_cmd,
                slurm_dump_dir,
                blocking=False,
                path_to_repo=cfg.path_to_repo,
                **slurm_kwargs,
            )
            for py_cmd in all_python_commands
        ]
        _wait_for_jobs(cfg, job_submissions)
        hd = combine_datasets(cfg, fs, output_filepaths, combined_outputs_fp)

    if return_seeds:
        return combined_outputs_fp, output_filepaths, hd, seeds_filepaths
    else:
        return combined_outputs_fp, output_filepaths, hd


def run_compute_liks_all_models_and_cal_data(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    seeds_fp_list: List[str],
    prev_cal_data_fp_list: List[str],
    # data_dir: str,
    model_dir_list: List[str],
    target_fp: str,
    # particle_field: str = "higher_score_particle",
    # score_field: str = "score",
    model_indices: List[int] = [],
    temps: List[float] = [1.0],
) -> str:
    """
    Runs compute_likelihoods_all_models jobs.
    """
    opt_str = " ".join(
        get_all_strs_from_nested_dict(cfg["compute_likelihooods_all_models"]["args"])
    )

    ## Format lists of strings into a long string that python and hydra can interpret
    seeds_fp_list_str = f"\\['{seeds_fp_list[0]}'"
    model_dir_list_str = f"\\['{model_dir_list[0]}'"
    if len(prev_cal_data_fp_list) > 0:
        prev_cal_data_fp_list_str = f"\\['{prev_cal_data_fp_list[0]}'"
    else:
        prev_cal_data_fp_list_str = "\\["
    for i in range(1, len(seeds_fp_list)):
        seeds_fp_list_str += f",'{seeds_fp_list[i]}'"
        model_dir_list_str += f",'{model_dir_list[i]}'"
        if i < len(prev_cal_data_fp_list):
            prev_cal_data_fp_list_str += f",'{prev_cal_data_fp_list[i]}'"
    seeds_fp_list_str += "\\]"
    model_dir_list_str += "\\]"
    prev_cal_data_fp_list_str += "\\]"

    ## Indices of models, only if not running from beginning with r0
    if len(model_indices) > 0:
        model_indices_str = f"\\['{model_indices[0]}'"
        for i in range(1, len(model_indices)):
            model_indices_str += f",'{model_indices[i]}'"
        model_indices_str += "\\]"
    else:
        model_indices_str = "\\[\\]"
        model_indices = [i for i in range(len(model_dir_list))]

    logger.info(f"target_fp: {target_fp}")
    output_dir = os.path.dirname(target_fp)

    args = f"{opt_str} input_data_path_list={seeds_fp_list_str} target_data_path={target_fp} prev_target_data_path_list={prev_cal_data_fp_list_str} model_name_or_path_list={model_dir_list_str} output_dir={output_dir} "
    # args += f"model_indices={model_indices_str}"

    args += f"model_indices={model_indices_str} overwrite_cmp_lik_all={cfg.overwrite_cmp_lik_all} "

    # args += f"test_fn_fp={data_dir}/ehrlich.jsonl "
    # args += f"particle_field={particle_field} "
    # args += f"score_field={score_field} "
    args += f"generation_config.max_new_tokens={cfg.compute_likelihooods_all_models.args.generation_config.max_new_tokens} "
    args += f"sanity_check={cfg.sanity_check} "
    args += f"test_fn_dim={cfg.evol_dataset_gen.args.test_function.dim} "

    # output_filename_prefix = f"cal_gens_all_likelihoods"
    # greedy_decoding_gen_args = f"generation_config.do_sample=False generation_config.num_beams=1 batch_size={cfg.greedy_gen_batch_size}"
    temp_sampling_gen_args = [
        f"generation_config.temperature={temp} " for temp in temps
    ]

    all_gen_args = temp_sampling_gen_args
    # output_filenames = [f"{output_filename_prefix}_temp{temp}.jsonl" for temp in temps]

    # output_filepaths = [f"{model_dir_list[-1]}/{output_fn}" for output_fn in output_filenames]
    output_filepaths = [target_fp]
    output_filenames = [os.path.basename(target_fp)]
    # combined_outputs_fp = f"{model_dir_list[-1]}/{output_filename_prefix}.jsonl"
    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"
    os.makedirs(slurm_dump_dir, exist_ok=True)
    hd = None
    if cfg.run_compute_liks_all_models_and_cal_data:
        all_python_commands = []

        for gen_args, output_fn in zip(all_gen_args, output_filenames):
            if len(target_fp) > 0:
                ## In this condition, run compute_liks_all_models_one_target
                all_args_one_target = []

                file_exists = fs.exists(target_fp)
                target_df = pd.read_json(target_fp, orient="records", lines=True)

                if file_exists:
                    ## If file exists, check if already has all likelihoods computed for models trying to compute for
                    all_liks_computed = True
                    for m in model_indices:
                        all_liks_computed = (
                            all_liks_computed and f"lik_r{m}" in target_df.columns
                        )

                ## If not overwriting, file exists, and contains updated likelihoods (for most recent model), then don't overwrite
                if not cfg.overwrite_cmp_lik_all and file_exists and all_liks_computed:
                    logger.info(
                        f"target_fp {target_fp} already exists with likelihoods computed. Skipping likelihoods computation..."
                    )
                else:
                    logger.info("Running compute_likelihoods all models...")
                    all_args_one_target.append(
                        f"{args} {gen_args} output_filename={output_fn} "
                    )
                ## Can only run script for updating curr data if provided non-empty filepath
                ## (so, providing empty `target_fp` allows skipping this)
                all_python_commands.extend(
                    [
                        f"export CUDA_LAUNCH_BLOCKING=1 \n python -m cpc_llm.core.compute_liks_all_models_one_target {a}"
                        for a in all_args_one_target
                    ]
                )

            if len(prev_cal_data_fp_list) > 0:
                all_args_prev_cal = []

                ## In this condition, run compute_likelihoods_one_model_all_data

                # file_exists = fs.exists(f"{model_dir_list[-1]}/{output_fn}")
                all_prev_cal_files_exist = True
                all_prev_cal_has_lik_col = True
                for prev_cal_data_fp in prev_cal_data_fp_list:
                    all_prev_cal_files_exist = all_prev_cal_files_exist and fs.exists(
                        prev_cal_data_fp
                    )
                    if not all_prev_cal_files_exist:
                        break
                    cal_curr_df = pd.read_json(
                        prev_cal_data_fp, orient="records", lines=True
                    )

                    ## Check if cal file already has all of the likelihood columns trying to add now
                    for m in model_indices:
                        all_prev_cal_has_lik_col = (
                            all_prev_cal_has_lik_col
                            and f"lik_r{m}" in cal_curr_df.columns
                        )

                ## If not overwriting, all prev cal files exist, and all contain updated likelihoods (for most recent model), then don't overwrite
                if (
                    not cfg.overwrite_cmp_lik_all
                    and all_prev_cal_files_exist
                    and all_prev_cal_has_lik_col
                ):
                    logger.info(
                        f"target_fp {target_fp} already exists with likelihoods computed. Skipping likelihoods computation..."
                    )
                else:
                    logger.info("Running compute_likelihoods all models...")
                    all_args_prev_cal.append(
                        f"{args} {gen_args} output_filename={output_fn}"
                    )

                ## Can only run script for updating prev data with curr model likelihoods if provided paths to prev data
                ## (so, providing empty `prev_cal_data_fp_list` allows skipping this)
                all_python_commands.extend(
                    [
                        f"export CUDA_LAUNCH_BLOCKING=1 \n python -m cpc_llm.core.compute_likelihoods_one_model_all_data {a}"
                        for a in all_args_prev_cal
                    ]
                )

        slurm_kwargs = OmegaConf.to_container(
            cfg.compute_likelihooods_all_models.slurm_args
        )
        slurm_kwargs["job_name"] = "comp_lik_all_models"
        job_submissions = [
            _submit_cmd(
                cfg,
                py_cmd,
                slurm_dump_dir,
                blocking=False,
                path_to_repo=cfg.path_to_repo,
                **slurm_kwargs,
            )
            for py_cmd in all_python_commands
        ]
        _wait_for_jobs(cfg, job_submissions)
        # hd = combine_datasets(cfg, fs, output_filepaths, combined_outputs_fp)
    return output_filepaths, hd


def train_initial_sft(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    data_fp: str,
    ga_data_dir: str,
    initial_sft_run_name: str,
    model_dir: str = "EleutherAI/pythia-2.8b",
    train_from_scratch: bool = False,
) -> str:

    ## Creating/Getting Output Directories
    test_fn_fp = f"{ga_data_dir}/ehrlich.jsonl"  ## Path to Ehrlich function parameters
    os.makedirs(f"{cfg.local_output_dir}/{cfg.run_name}", exist_ok=True)
    output_dir = f"{cfg.local_output_dir}/{cfg.run_name}/{initial_sft_run_name}"
    s3_output_dir = (
        f"{cfg.parent_output_dir}/{cfg.run_name}/{initial_sft_run_name}"
        if cfg.parent_output_dir is not None
        else "null"
    )

    ## Loading args
    args = f"--config-name=pythia-2.8b_edit_pairs data_fp={data_fp} "
    args += " ".join(get_all_strs_from_nested_dict(cfg["initial_sft"]["args"])) + " "
    args += f"test_fn_type=ehrlich test_fn_fp={test_fn_fp} "
    args += f"job_name={initial_sft_run_name} s3_output_dir={s3_output_dir} "
    args += f"model_config.model_name_or_path={model_dir} "
    args += f"sanity_check={cfg.sanity_check} "

    # train from scratch
    if train_from_scratch and hasattr(cfg, "initial_model_config"):
        args += "train_from_scratch=True "
        for k, v in cfg.initial_model_config.items():
            args += f"+init_model_config.{k}={v} "

    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"

    trained_model_dir = model_already_trained(cfg, fs, s3_output_dir, output_dir)
    if trained_model_dir is not None and not cfg.overwrite_initial_sft:
        logger.info(f"Trained model already exists in {trained_model_dir}. Skipping...")
        return trained_model_dir
    else:
        if trained_model_dir is None:
            logger.info("Did not find trained model, Continuing to train...")
        else:
            trained_model_dir = None
            logger.info(
                f"Config says to overwrite model (cfg.overwrite_initial_sft={cfg.overwrite_initial_sft}), Continuing to train..."
            )
    os.makedirs(slurm_dump_dir, exist_ok=True)
    if cfg.run_initial_sft:
        slurm_cfg = cfg.initial_sft.slurm_args
        ## If overwriting Initial SFT, then delete previous checkpoints
        py_cmd = ""
        if cfg.overwrite_initial_sft:
            py_cmd += f"rm -rf {output_dir}/checkpoint-*\n"

        py_cmd += _get_launch_prefix(cfg, slurm_cfg)
        py_cmd += f"cpc_llm.test_functions.finetune_ehrlich {args} training_args.output_dir={output_dir}\n"

        # if S3 was used, capture exit code, clean up local checkpoints, then exit
        if cfg.parent_output_dir is not None:
            py_cmd += "RETURN_CODE=$?\n"
            py_cmd += f"rm -rf {output_dir}/checkpoint-*/model-*.safetensors\n"
            py_cmd += f"rm -rf {output_dir}/model-*.safetensors\n"
            py_cmd += f"rm -rf {output_dir}/checkpoint-*/optimizer.pt\n"
            py_cmd += "exit ${RETURN_CODE}\n"

        slurm_kwargs = OmegaConf.to_container(cfg.initial_sft.slurm_args)
        slurm_kwargs["job_name"] = "initial_sft"
        _submit_cmd(
            cfg,
            py_cmd,
            slurm_dump_dir,
            blocking=True,
            path_to_repo=cfg.path_to_repo,
            **slurm_kwargs,
        )
    return_path = s3_output_dir if cfg.parent_output_dir is not None else output_dir
    return return_path


def train_sft(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    data_fp: str,
    ga_data_dir: str,
    sft_run_name: str,
    model_dir: str = "EleutherAI/pythia-2.8b",
    train_from_scratch: bool = False,
) -> str:

    ## Creating/Getting Output Directories
    test_fn_fp = f"{ga_data_dir}/ehrlich.jsonl"  ## Path to Ehrlich function parameters
    os.makedirs(f"{cfg.local_output_dir}/{cfg.run_name}", exist_ok=True)
    output_dir = f"{cfg.local_output_dir}/{cfg.run_name}/{sft_run_name}"
    s3_output_dir = (
        f"{cfg.parent_output_dir}/{cfg.run_name}/{sft_run_name}"
        if cfg.parent_output_dir is not None
        else "null"
    )

    ## Loading args
    args = f"--config-name=pythia-2.8b_edit_pairs data_fp={data_fp} "
    args += " ".join(get_all_strs_from_nested_dict(cfg["sft"]["args"])) + " "
    args += f"test_fn_type=ehrlich test_fn_fp={test_fn_fp} "
    args += f"job_name={sft_run_name} s3_output_dir={s3_output_dir} "
    args += f"model_config.model_name_or_path={model_dir} "
    args += f"sanity_check={cfg.sanity_check} "

    # train from scratch
    if train_from_scratch and hasattr(cfg, "initial_model_config"):
        args += "train_from_scratch=True "
        for k, v in cfg.initial_model_config.items():
            args += f"+init_model_config.{k}={v} "

    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"

    trained_model_dir = model_already_trained(cfg, fs, s3_output_dir, output_dir)
    if trained_model_dir is not None and not cfg.overwrite_sft:
        logger.info(f"Trained model already exists in {trained_model_dir}. Skipping...")
        return trained_model_dir
    else:
        if trained_model_dir is None:
            logger.info("Did not find trained model, Continuing to train...")
        else:
            trained_model_dir = None
            logger.info(
                f"Config says to overwrite model (cfg.overwrite_sft={cfg.overwrite_sft}), Continuing to train..."
            )
    os.makedirs(slurm_dump_dir, exist_ok=True)
    if cfg.run_sft:
        slurm_cfg = cfg.sft.slurm_args
        ## If overwriting SFT, then delete previous checkpoints
        py_cmd = ""
        if cfg.overwrite_sft:
            py_cmd += f"rm -rf {output_dir}/checkpoint-*\n"

        py_cmd += _get_launch_prefix(cfg, slurm_cfg)
        py_cmd += f"cpc_llm.test_functions.finetune_ehrlich {args} training_args.output_dir={output_dir}\n"

        # if S3 was used, capture exit code, clean up local checkpoints, then exit
        if cfg.parent_output_dir is not None:
            py_cmd += "RETURN_CODE=$?\n"
            py_cmd += f"rm -rf {output_dir}/checkpoint-*/model-*.safetensors\n"
            py_cmd += f"rm -rf {output_dir}/model-*.safetensors\n"
            py_cmd += f"rm -rf {output_dir}/checkpoint-*/optimizer.pt\n"
            py_cmd += "exit ${RETURN_CODE}\n"

        slurm_kwargs = OmegaConf.to_container(cfg.sft.slurm_args)
        slurm_kwargs["job_name"] = "sft"
        _submit_cmd(
            cfg,
            py_cmd,
            slurm_dump_dir,
            blocking=True,
            path_to_repo=cfg.path_to_repo,
            **slurm_kwargs,
        )
    return_path = s3_output_dir if cfg.parent_output_dir is not None else output_dir
    return return_path


def train_dpo(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    ref_model_path: str,
    data_fp: str,
    ga_data_dir: str,
    run_name: str,
) -> str:
    test_fn_fp = f"{ga_data_dir}/ehrlich.jsonl"
    output_dir = f"{cfg.local_output_dir}/{cfg.run_name}/{run_name}"
    s3_output_dir = (
        f"{cfg.parent_output_dir}/{cfg.run_name}/{run_name}"
        if cfg.parent_output_dir is not None
        else "null"
    )
    trained_model_dir = model_already_trained(cfg, fs, s3_output_dir, output_dir)
    if trained_model_dir is not None and not cfg.overwrite_dpo:
        logger.info(f"Trained model already exists in {trained_model_dir}. Skipping...")
        return trained_model_dir
    else:
        if trained_model_dir is None:
            logger.info("Did not find trained model, Continuing to train...")
        else:
            trained_model_dir = None
            logger.info(
                f"Config says to overwrite model (cfg.overwrite_dpo={cfg.overwrite_dpo}), Continuing to train..."
            )

    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"

    os.makedirs(slurm_dump_dir, exist_ok=True)
    args = "--config-name=pythia-2.8b-dpo "
    args += " ".join(get_all_strs_from_nested_dict(cfg["dpo"]["args"])) + " "
    args += f"data_fp={data_fp} "
    args += f"dpo_config.run_name={run_name} "
    args += f"model_config.model_name_or_path={ref_model_path} "
    args += f"test_fn_fp={test_fn_fp} "
    args += f"job_name={run_name} "
    args += f"s3_output_dir={s3_output_dir} "
    args += f"dpo_script_args.sanity_check={cfg.sanity_check} "

    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"
    os.makedirs(slurm_dump_dir, exist_ok=True)
    slurm_cfg = cfg.dpo.slurm_args
    py_cmd = _get_launch_prefix(cfg, slurm_cfg)
    py_cmd += f"cpc_llm.train.dpo {args} dpo_config.output_dir={output_dir}\n"

    # if S3 was used, capture exit code, clean up local checkpoints, then exit
    if cfg.parent_output_dir is not None:
        py_cmd += "RETURN_CODE=$?\n"
        py_cmd += f"rm -rf {output_dir}/checkpoint-*/model-*.safetensors\n"
        py_cmd += f"rm -rf {output_dir}/model-*.safetensors\n"
        py_cmd += f"rm -rf {output_dir}/checkpoint-*/optimizer.pt\n"
        py_cmd += "exit ${RETURN_CODE}\n"

    slurm_kwargs = OmegaConf.to_container(cfg.dpo.slurm_args)
    slurm_kwargs["job_name"] = "dpo"
    _submit_cmd(
        cfg,
        py_cmd,
        slurm_dump_dir,
        blocking=True,
        path_to_repo=cfg.path_to_repo,
        **slurm_kwargs,
    )
    return_path = s3_output_dir if cfg.parent_output_dir is not None else output_dir
    return return_path


def train_marge(
    cfg: DictConfig,
    fs: LocalOrS3Client,
    ref_model_path: str,
    data_fp: str,
    ga_data_dir: str,
    run_name: str,
) -> str:
    test_fn_fp = f"{ga_data_dir}/ehrlich.jsonl"
    output_dir = f"{cfg.local_output_dir}/{cfg.run_name}/{run_name}"
    s3_output_dir = (
        f"{cfg.parent_output_dir}/{cfg.run_name}/{run_name}"
        if cfg.parent_output_dir is not None
        else "null"
    )
    trained_model_dir = model_already_trained(cfg, fs, s3_output_dir, output_dir)
    if trained_model_dir is not None and not cfg.overwrite_marge:
        logger.info(f"Trained model already exists in {trained_model_dir}. Skipping...")
        return trained_model_dir
    else:
        if trained_model_dir is None:
            logger.info("Did not find trained model, Continuing to train...")
        else:
            trained_model_dir = None
            logger.info(
                f"Config says to overwrite model (cfg.overwrite_marge={cfg.overwrite_marge}), Continuing to train..."
            )
    args = "--config-name=pythia-2.8b-marge "
    args += " ".join(get_all_strs_from_nested_dict(cfg["marge"]["args"])) + " "
    args += f"data_fp={data_fp} "
    args += f"marge_config.run_name={run_name} "
    args += f"model_config.model_name_or_path={ref_model_path} "
    args += f"test_fn_fp={test_fn_fp} "
    args += f"job_name={run_name} "
    args += f"s3_output_dir={s3_output_dir} "
    args += f"dpo_script_args.sanity_check={cfg.sanity_check} "

    slurm_dump_dir = f"{cfg.local_output_dir}/slurm_logs"
    os.makedirs(slurm_dump_dir, exist_ok=True)
    slurm_cfg = cfg.marge.slurm_args
    py_cmd = _get_launch_prefix(cfg, slurm_cfg)
    py_cmd += f"cpc_llm.train.marge {args} marge_config.output_dir={output_dir}\n"

    # if S3 was used, capture exit code, clean up local checkpoints, then exit
    if cfg.parent_output_dir is not None:
        py_cmd += "RETURN_CODE=$?\n"
        py_cmd += f"rm -rf {output_dir}/checkpoint-*/model-*.safetensors\n"
        py_cmd += f"rm -rf {output_dir}/model-*.safetensors\n"
        py_cmd += f"rm -rf {output_dir}/checkpoint-*/optimizer.pt\n"
        py_cmd += "exit ${RETURN_CODE}\n"

    slurm_kwargs = OmegaConf.to_container(cfg.marge.slurm_args)
    slurm_kwargs["job_name"] = "marge"
    _submit_cmd(
        cfg,
        py_cmd,
        slurm_dump_dir,
        blocking=True,
        path_to_repo=cfg.path_to_repo,
        **slurm_kwargs,
    )
    return_path = s3_output_dir if cfg.parent_output_dir is not None else output_dir
    return return_path
