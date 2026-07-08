# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluate Online-learned policies on synthetic simulation."""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any, Dict, Union, Optional

import hydra
import pandas as pd

import torch

from experiments.synthetic.function import (
    evaluate_policy,
    initialize_optimal_policy,
    load_pg_policy,
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
    dim_model_emb: int,
    n_candidate_action_eval: int,
    device: torch.device,
    base_random_seed: int,
    random_seed: int,
    early_stage_online_credit_assigned_pg_path: str,
    early_stage_online_vanilla_pg_path: str,
    early_stage_online_top1_pg_path: str,
    early_stage_online_vanilla_pg_replacement_path: Optional[str] = None,
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

    _, optimal_late_stage_policy = initialize_optimal_policy(
        env=env,
        device=device,
        random_seed=base_random_seed,
    )
    early_stage_credit_assigned_pg_policy = load_pg_policy(
        env=env,
        dim_model_emb=dim_model_emb,
        early_stage_model_path=early_stage_online_credit_assigned_pg_path,
        device=device,
        random_seed=random_seed,
    )
    early_stage_vanilla_pg_policy = load_pg_policy(
        env=env,
        dim_model_emb=dim_model_emb,
        early_stage_model_path=early_stage_online_vanilla_pg_path,
        device=device,
        random_seed=random_seed,
    )
    early_stage_top1_pg_policy = load_pg_policy(
        env=env,
        dim_model_emb=dim_model_emb,
        early_stage_model_path=early_stage_online_top1_pg_path,
        device=device,
        random_seed=random_seed,
    )

    algorithms = [
        "online-credit-assigned",
        "online-vanilla",
        "online-top1",
    ]
    early_stage_policies = [
        early_stage_credit_assigned_pg_policy,
        early_stage_vanilla_pg_policy,
        early_stage_top1_pg_policy,
    ]

    if early_stage_online_vanilla_pg_replacement_path is not None:
        early_stage_vanilla_replacement_policy = load_pg_policy(
            env=env,
            dim_model_emb=dim_model_emb,
            early_stage_model_path=early_stage_online_vanilla_pg_replacement_path,
            device=device,
            random_seed=random_seed,
        )
        algorithms = algorithms + ["online-vanilla-swr"]
        early_stage_policies = early_stage_policies + [early_stage_vanilla_replacement_policy]

    performance = {}
    for i, algo in enumerate(algorithms):
        performance[algo] = evaluate_policy(
            env=env,
            early_stage_policy=early_stage_policies[i],
            late_stage_policy=optimal_late_stage_policy,
            n_candidate_action=n_candidate_action_eval,
            is_deterministic_early_stage=True,
            is_deterministic_late_stage=True,
        )

    return performance


def process(
    conf: Dict[str, Any],
):
    conf_ = deepcopy(conf)
    conf_["key_param"] = None

    setting = conf["setting"]
    key_param_name = conf["setting"]
    experiment_name = conf["experiment_name"]

    rootdir = conf["rootdir"]

    if (
        setting != "n_candidate_action_eval"
        and conf["n_candidate_action_train"] == "auto"
    ):
        conf_["n_candidate_action_train"] = conf["n_candidate_action_eval"]

    if experiment_name == "auto":
        if setting == "n_candidate_action_eval":
            experiment_name = "default"
        else:
            experiment_name = setting

    if setting == "default" or (
        setting == "n_candidate_action_eval"
        and conf["n_candidate_action_train"] != "auto"
    ):
        for random_seed in range(conf["n_random_seed"]):
            conf_["random_seed"] = random_seed + conf["start_random_seed"]
            n_cand_ = conf_['n_candidate_action_train']
            late_stage_ = conf_['late_stage_optimality']
            n_model_ = conf_['n_moe_model']
            n_out_ = conf_['n_output_action']
            seed_ = conf_['random_seed']

            detailed_configs_ = f"n_candidate={n_cand_},late_stage={late_stage_},n_model={n_model_},n_output={n_out_},seed={seed_}"

            if conf["early_stage_online_credit_assigned_pg_path"] == "auto":
                conf_["early_stage_online_credit_assigned_pg_path"] = (
                    f"{rootdir}/online_early_stage/credit={'CA'}/{detailed_configs_}.pt"
                )
            if conf["early_stage_online_vanilla_pg_path"] == "auto":
                conf_["early_stage_online_vanilla_pg_path"] = (
                    f"{rootdir}/online_early_stage/credit={'ALL'}/{detailed_configs_}.pt"
                )
            if conf["early_stage_online_top1_pg_path"] == "auto":
                conf_["early_stage_online_top1_pg_path"] = (
                    f"{rootdir}/online_early_stage/credit={'TOP1'}/{detailed_configs_}.pt"
                )

            if conf["early_stage_online_vanilla_pg_replacement_path"] == "auto" and conf["is_vanilla_replacement"]:
                conf_["early_stage_online_top1_pg_path"] = (
                    f"{rootdir}/online_early_stage/credit={'ALL-SwR'}/{detailed_configs_}.pt"
                )

            if not Path(conf_["early_stage_online_credit_assigned_pg_path"]).exists():
                raise ValueError(
                    "early_stage_online_credit_assigned_pg_path does not exist."
                )
            if not Path(conf_["early_stage_online_vanilla_pg_path"]).exists():
                raise ValueError("early_stage_online_vanilla_pg_path does not exist.")
            if not Path(conf_["early_stage_online_top1_pg_path"]).exists():
                raise ValueError("early_stage_online_top1_pg_path does not exist.")
            if conf["is_vanilla_replacement"] and not Path(conf_["early_stage_online_vanilla_pg_replacement_path"]).exists():
                raise ValueError("early_stage_online_vanilla_pg_replacement_path does not exist.")

    else:
        for key_param in conf[key_param_name]:
            conf_["key_param"] = key_param
            conf_[key_param_name] = key_param

            if (
                setting == "n_candidate_action_eval"
                and conf["n_candidate_action_train"] == "auto"
            ):
                conf_["n_candidate_action_train"] = key_param

            for random_seed in range(conf["n_random_seed"]):
                conf_["random_seed"] = random_seed + conf["start_random_seed"]
                n_cand_ = conf_['n_candidate_action_train']
                late_stage_ = conf_['late_stage_optimality']
                seed_ = conf_['random_seed']

                detailed_configs_ = f"n_candidate={n_cand_},late_stage={late_stage_},n_model={n_model_},n_output={n_out_},seed={seed_}"

                if conf["early_stage_online_credit_assigned_pg_path"] == "auto":
                    conf_["early_stage_online_credit_assigned_pg_path"] = (
                        f"{rootdir}/online_early_stage/credit={'CA'}/{detailed_configs_}.pt"
                    )
                if conf["early_stage_online_vanilla_pg_path"] == "auto":
                    conf_["early_stage_online_vanilla_pg_path"] = (
                        f"{rootdir}/online_early_stage/credit={'ALL'}/{detailed_configs_}.pt"
                    )
                if conf["early_stage_online_top1_pg_path"] == "auto":
                    conf_["early_stage_online_top1_pg_path"] = (
                        f"{rootdir}/online_early_stage/credit={'TOP1'}/{detailed_configs_}.pt"
                    )
                if conf["early_stage_online_vanilla_pg_replacement_path"] == "auto" and conf["is_vanilla_replacement"]:
                    conf_["early_stage_online_top1_pg_path"] = (
                        f"{rootdir}/online_early_stage/credit={'ALL-SwR'}/{detailed_configs_}.pt"
                    )

                if not Path(conf_["early_stage_online_credit_assigned_pg_path"]).exists():
                    raise ValueError(
                        "early_stage_online_credit_assigned_pg_path does not exist."
                    )
                if not Path(conf_["early_stage_online_vanilla_pg_path"]).exists():
                    raise ValueError("early_stage_online_vanilla_pg_path does not exist.")
                if not Path(conf_["early_stage_online_top1_pg_path"]).exists():
                    raise ValueError("early_stage_online_top1_pg_path does not exist.")
                if conf["is_vanilla_replacement"] and not Path(conf_["early_stage_online_vanilla_pg_replacement_path"]).exists():
                    raise ValueError("early_stage_online_vanilla_pg_replacement_path does not exist.")

    performance_dict = defaultdict(list)

    if setting != "default":
        key_param_name = conf["setting"]

        for key_param in conf[key_param_name]:
            conf_["key_param"] = key_param
            conf_[key_param_name] = key_param

            if (
                setting == "n_candidate_action_eval"
                and conf["n_candidate_action_train"] == "auto"
            ):
                conf_["n_candidate_action_train"] = key_param

            for random_seed in range(conf["n_random_seed"]):
                conf_["random_seed"] = random_seed + conf["start_random_seed"]
                n_cand_ = conf_['n_candidate_action_train']
                late_stage_ = conf_['late_stage_optimality']
                n_model_ = conf_['n_moe_model']
                n_out_ = conf_['n_output_action']
                seed_ = conf_['random_seed']

                detailed_configs_ = f"n_candidate={n_cand_},late_stage={late_stage_},n_model={n_model_},n_output={n_out_},seed={seed_}"

                if conf["early_stage_online_credit_assigned_pg_path"] == "auto":
                    conf_["early_stage_online_credit_assigned_pg_path"] = (
                        f"{rootdir}/online_early_stage/credit={'CA'}/{detailed_configs_}.pt"
                    )
                if conf["early_stage_online_vanilla_pg_path"] == "auto":
                    conf_["early_stage_online_vanilla_pg_path"] = (
                        f"{rootdir}/online_early_stage/credit={'ALL'}/{detailed_configs_}.pt"
                    )
                if conf["early_stage_online_top1_pg_path"] == "auto":
                    conf_["early_stage_online_top1_pg_path"] = (
                        f"{rootdir}/online_early_stage/credit={'TOP1'}/{detailed_configs_}.pt"
                    )
                if conf["early_stage_online_vanilla_pg_replacement_path"] == "auto" and conf["is_vanilla_replacement"]:
                    conf_["early_stage_online_top1_pg_path"] = (
                        f"{rootdir}/online_early_stage/credit={'ALL-SwR'}/{detailed_configs_}.pt"
                    )

                print(
                    f"Setting: {setting}, Key Param: {key_param}, Random Seed: {random_seed}/{conf['n_random_seed']}"
                )

                performance_ = _process(**conf_)
                performance_dict[key_param_name].append(key_param)
                performance_dict["random_seed"].append(random_seed)

                for key, value in performance_.items():
                    performance_dict[key].append(value)

    else:
        for random_seed in range(conf["n_random_seed"]):
            conf_["random_seed"] = random_seed + conf["start_random_seed"]
            n_cand_ = conf_['n_candidate_action_train']
            late_stage_ = conf_['late_stage_optimality']
            n_model_ = conf_['n_moe_model']
            n_out_ = conf_['n_output_action']
            seed_ = conf_['random_seed']

            detailed_configs_ = f"n_candidate={n_cand_},late_stage={late_stage_},n_model={n_model_},n_output={n_out_},seed={seed_}"

            if conf["early_stage_online_credit_assigned_pg_path"] == "auto":
                conf_["early_stage_online_credit_assigned_pg_path"] = (
                    f"{rootdir}/online_early_stage/credit={'CA'}/{detailed_configs_}.pt"
                )
            if conf["early_stage_online_vanilla_pg_path"] == "auto":
                conf_["early_stage_online_vanilla_pg_path"] = (
                    f"{rootdir}/online_early_stage/credit={'ALL'}/{detailed_configs_}.pt"
                )
            if conf["early_stage_online_top1_pg_path"] == "auto":
                conf_["early_stage_online_top1_pg_path"] = (
                    f"{rootdir}/online_early_stage/credit={'TOP1'}/{detailed_configs_}.pt"
                )
            if conf["early_stage_online_vanilla_pg_replacement_path"] == "auto" and conf["is_vanilla_replacement"]:
                    conf_["early_stage_online_top1_pg_path"] = (
                        f"{rootdir}/online_early_stage/credit={'ALL-SwR'}/{detailed_configs_}.pt"
                    )

            print(
                f"Setting: {setting}, Key Param: {'None'}, Random Seed: {random_seed}/{conf['n_random_seed']}"
            )

            performance_ = _process(**conf_)
            performance_dict["setting"].append("default")
            performance_dict["random_seed"].append(random_seed)

            for key, value in performance_.items():
                performance_dict[key].append(value)

    performance_dict = defaultdict_to_dict(performance_dict)
    df = pd.DataFrame(performance_dict)

    Path(f"{rootdir}/{experiment_name}").mkdir(parents=True, exist_ok=True)
    df_path = f"{rootdir}/{experiment_name}/online_pg_performance.csv"
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
        "is_vanilla_repalcement": cfg.model.is_vanilla_replacement,
        "n_epoch": cfg.model.n_epoch,
        "n_steps_per_epoch": cfg.model.n_steps_per_epoch,
        "n_epochs_per_log": cfg.model.n_epochs_per_log,
        "rootdir": cfg.logs.rootdir,
        "experiment_name": cfg.logs.experiment_name,
        "use_wandb": cfg.logs.use_wandb,
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
