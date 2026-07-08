# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluate oracle, uniform, and logging on synthetic simulation."""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any, Dict, Union

import hydra
import pandas as pd

import torch

from experiments.synthetic.function import (
    evaluate_policy,
    initialize_optimal_policy,
    initialize_uniform_policy,
    initialize_noisy_optimal_late_stage_policy,
    initialize_anti_optimal_late_stage_policy,
    setup_data_generation_process,
)
from experiments.synthetic.utils import (
    assert_configuration,
    defaultdict_to_dict,
    format_runtime,
    reset_seed,
)
from omegaconf import DictConfig


def _process(
    n_user: int,
    n_action: int,
    n_latent: int,
    n_output_action: int,
    dim_context: int,
    dim_action_emb: int,
    reward_scaler: Union[int, float],
    n_candidate_action_eval: int,
    device: torch.device,
    base_random_seed: int,
    **kwargs,
):
    reset_seed(base_random_seed)

    env = setup_data_generation_process(
        n_user=n_user,
        n_action=n_action,
        n_latent=n_latent,
        n_output_action=n_output_action,
        dim_context=dim_context,
        dim_action_emb=dim_action_emb,
        reward_scaler=reward_scaler,
        device=device,
        random_seed=base_random_seed,
    )

    uniform_early_stage_policy, uniform_late_stage_policy = initialize_uniform_policy(
        env=env,
        device=device,
        random_seed=base_random_seed,
    )
    optimal_early_stage_policy, optimal_late_stage_policy = initialize_optimal_policy(
        env=env,
        device=device,
        random_seed=base_random_seed,
    )
    noisy_late_stage_policy = initialize_noisy_optimal_late_stage_policy(
        env=env,
        device=device,
        random_seed=base_random_seed,
    )
    anti_late_stage_policy = initialize_anti_optimal_late_stage_policy(
        env=env,
        device=device,
        random_seed=base_random_seed,
    )

    algorithms = [
        "unif_anti",    # uniform x anti-optimal
        "unif_unif",    # uniform x uniform
        "unif_noisy",   # uniform x noisy optimal
        "unif_opt",     # optimal x optimal
        "opt_anti",     # optimal x anti-optimal
        "opt_uniform",  # optimal x uniform
        "opt_noisy",    # optimal x noisy optimal
        "opt_opt",      # optimal x optimal
    ]
    early_stage_policies = [
        uniform_early_stage_policy,
        uniform_early_stage_policy,
        uniform_early_stage_policy,
        uniform_early_stage_policy,
        optimal_early_stage_policy,
        optimal_early_stage_policy,
        optimal_early_stage_policy,
        optimal_early_stage_policy,
    ]
    late_stage_policies = [
        anti_late_stage_policy,
        uniform_late_stage_policy,
        noisy_late_stage_policy,
        optimal_late_stage_policy,
        anti_late_stage_policy,
        uniform_late_stage_policy,
        noisy_late_stage_policy,
        optimal_late_stage_policy,
    ]
    is_deterministic_early_stage_flgs = [False, False, False, False, True, True, True, True]
    is_derterministic_late_stage_flgs = [True, False, False, True, True, False, False, True]

    performance = {}
    for i, algo in enumerate(algorithms):
        performance[algo] = evaluate_policy(
            env=env,
            early_stage_policy=early_stage_policies[i],
            late_stage_policy=late_stage_policies[i],
            n_candidate_action=n_candidate_action_eval,
            is_deterministic_early_stage=is_deterministic_early_stage_flgs[i],
            is_deterministic_late_stage=is_derterministic_late_stage_flgs[i],
        )

    return performance


def process(
    conf: Dict[str, Any],
):
    conf_ = deepcopy(conf)
    conf_["key_param"] = None

    setting = conf["setting"]
    key_param = conf["setting"]
    experiment_name = conf["experiment_name"]

    rootdir = conf["rootdir"]

    if experiment_name == "auto":
        if setting == "n_candidate_action_eval":
            experiment_name = "default"
        else:
            experiment_name = setting

    performance_dict = defaultdict(list)

    if setting != "default":
        key_param_name = conf["setting"]

        for key_param in conf[key_param_name]:
            conf_["key_param"] = key_param
            conf_[key_param_name] = key_param

            print(f"Setting: {setting}, Key Param: {key_param}")

            performance_ = _process(**conf_)
            performance_dict[key_param_name].append(key_param)

            for key, value in performance_.items():
                performance_dict[key].append(value)

    else:
        print(f"Setting: {setting}, Key Param: {'None'}")

        performance_ = _process(**conf_)
        performance_dict["setting"].append("default")

        for key, value in performance_.items():
            performance_dict[key].append(value)

    performance_dict = defaultdict_to_dict(performance_dict)
    df = pd.DataFrame(performance_dict)

    Path(rootdir).mkdir(parents=True, exist_ok=True)
    df_path = f"{rootdir}/reference_performance.csv"
    df.to_csv(df_path, index=False)


@hydra.main(config_path="conf/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg)
    assert_configuration(cfg)
    print(f"The current working directory is: {Path().cwd()}")
    print(f"The original working directory is: {hydra.utils.get_original_cwd()}")
    print()

    conf = {
        "setting": cfg.setting.setting,
        "n_action": cfg.setting.n_action,
        "n_output_action": cfg.setting.n_output_action,
        "n_candidate_action_train": cfg.setting.n_candidate_action_train,
        "n_candidate_action_eval": cfg.setting.n_candidate_action_eval,
        "late_stage_optimality": cfg.setting.late_stage_optimality,
        "n_user": cfg.setting.n_user,
        "n_latent": cfg.setting.n_latent,
        "dim_context": cfg.setting.dim_context,
        "dim_action_emb": cfg.setting.dim_action_emb,
        "reward_scaler": cfg.setting.reward_scaler,
        "device": cfg.setting.device,
        "n_random_seed": cfg.setting.n_random_seed,
        "start_random_seed": cfg.setting.start_random_seed,
        "base_random_seed": cfg.setting.base_random_seed,
        "dim_model_emb": cfg.model.dim_model_emb,
        "n_moe_model": cfg.model.n_moe_model,
        "early_stage_naive_cf_lr": cfg.model.early_stage_naive_cf_lr,
        "late_stage_neural_lr": cfg.model.late_stage_neural_lr,
        "online_vanilla_pg_lr": cfg.model.online_vanilla_pg_lr,
        "online_credit_assigned_pg_lr": cfg.model.online_credit_assigned_pg_lr,
        "online_top1_pg_lr": cfg.model.online_top1_pg_lr,
        "credit_assignment_type": cfg.model.credit_assignment_type,
        "is_vanilla_replacement": cfg.model.is_vanilla_replacement,
        "n_epoch": cfg.model.n_epoch,
        "n_steps_per_epoch": cfg.model.n_steps_per_epoch,
        "n_epochs_per_log": cfg.model.n_epochs_per_log,
        "rootdir": cfg.logs.rootdir,
        "experiment_name": cfg.logs.experiment_name,
        "use_wandb": cfg.logs.use_wandb,
        "early_stage_naive_cf_path": cfg.path.early_stage_naive_cf_path,  # unused
        "late_stage_naive_cf_path": cfg.path.late_stage_naive_cf_path,  # unused
        "dataset_path": cfg.path.dataset_path,
        "early_stage_online_credit_assigned_pg_path": cfg.path.early_stage_online_credit_assigned_pg_path,
        "early_stage_online_vanilla_pg_path": cfg.path.early_stage_online_vanilla_pg_path,
        "early_stage_online_top1_pg_path": cfg.path.early_stage_online_top1_pg_path,
        "early_stage_online_vanilla_pg_replacement_path": cfg.path.early_stage_online_vanilla_pg_replacement_path,
    }
    process(conf)


if __name__ == "__main__":
    start = time()
    main()
    finish = time()
    print(f"Total runtime: {format_runtime(start, finish)}")
