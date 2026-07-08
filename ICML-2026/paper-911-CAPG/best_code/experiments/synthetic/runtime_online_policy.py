# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Train the policy via Online PG."""

from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any, Dict, Optional, Union

import hydra

import torch

from experiments.synthetic.function import (
    initialize_trainable_policy,
    setup_data_generation_process,
    runtime_online_pg_policy,
)
from experiments.synthetic.utils import (
    assert_configuration,
    format_runtime,
    reset_seed,
)
from omegaconf import DictConfig


def _process(
    n_user: int,
    n_action: int,
    n_latent: int,
    n_output_action: int,
    n_moe_model: int,
    dim_context: int,
    dim_action_emb: int,
    reward_scaler: Union[int, float],
    dim_model_emb: int,
    online_vanilla_pg_lr: Union[int, float],
    online_credit_assigned_pg_lr: Union[int, float],
    online_top1_pg_lr: Union[int, float],
    credit_assignment_type: str,
    is_vanilla_replacement: bool,
    n_epoch: int,
    n_steps_per_epoch: int,
    n_epochs_per_log: int,
    n_candidate_action_train: int,
    n_candidate_action_eval: int,
    device: torch.device,
    base_random_seed: int,
    random_seed: int,
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

    reset_seed(random_seed)

    if credit_assignment_type == "CA":
        early_stage_lr = online_credit_assigned_pg_lr
    elif credit_assignment_type == "ALL":
        early_stage_lr = online_vanilla_pg_lr
    elif credit_assignment_type == "TOP1":
        early_stage_lr = online_top1_pg_lr

    online_early_stage_policy, _ = initialize_trainable_policy(
        env=env,
        dim_model_emb=dim_model_emb,
        n_moe_model=n_moe_model,
        device=device,
        random_seed=random_seed,
    )

    start = time()

    runtime_online_pg_policy(
        env=env,
        early_stage_policy=online_early_stage_policy,
        early_stage_lr=early_stage_lr,
        credit_assignment_type=credit_assignment_type,
        is_vanilla_replacement=is_vanilla_replacement,
        n_epoch=n_epoch,
        n_steps_per_epoch=n_steps_per_epoch,
        n_epochs_per_log=n_epochs_per_log,
        n_candidate_action_train=n_candidate_action_train,
        n_candidate_action_eval=n_candidate_action_eval,
        device=device,
        random_seed=random_seed,
        use_wandb=False,
    )
    
    finish = time()
    return finish - start


def process(
    conf: Dict[str, Any],
):
    conf_ = deepcopy(conf)
    conf_["key_param"] = None

    setting = conf["setting"]
    key_param = conf["setting"]

    if (
        setting != "n_candidate_action_eval"
        and conf["n_candidate_action_train"] == "auto"
    ):
        conf_["n_candidate_action_train"] = conf["n_candidate_action_eval"]

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

            is_vanilla_flg_ = "-SwR" if conf_['credit_assignment_type'] == "ALL" and conf_["is_vanilla_replacement"] else ""
            detailed_configs_ = f"n_candidate={conf_['n_candidate_action_train']},late_stage={conf_['late_stage_optimality']},n_model={conf_['n_moe_model']},n_output={conf_['n_output_action']}"
            online_pg_runtime_dir = Path(f"{conf_['rootdir']}/online_early_stage/training_process/credit={conf_['credit_assignment_type']}{is_vanilla_flg_}/runtime")
            Path(online_pg_runtime_dir).mkdir(parents=True, exist_ok=True)

            runtime_logs = torch.zeros((conf["n_random_seed"]))
            for random_seed in range(conf["n_random_seed"]):
                conf_["random_seed"] = random_seed + conf["start_random_seed"]

                print(
                    f"Setting: {setting}, Key Param: {key_param}, Random Seed: {random_seed}/{conf['n_random_seed']}"
                )
                runtime_logs[random_seed] = _process(**conf_)

            torch.save(runtime_logs, f"{online_pg_runtime_dir}/{detailed_configs_}.pt")

    else:
        is_vanilla_flg_ = "-SwR" if conf_['credit_assignment_type'] == "ALL" and conf_["is_vanilla_replacement"] else ""
        detailed_configs_ = f"n_candidate={conf_['n_candidate_action_train']},late_stage={conf_['late_stage_optimality']},n_model={conf_['n_moe_model']},n_output={conf_['n_output_action']}"
        online_pg_runtime_dir = Path(f"{conf_['rootdir']}/online_early_stage/training_process/credit={conf_['credit_assignment_type']}{is_vanilla_flg_}/runtime")
        Path(online_pg_runtime_dir).mkdir(parents=True, exist_ok=True)

        runtime_logs = torch.zeros((conf["n_random_seed"]))
        for random_seed in range(conf["n_random_seed"]):
            conf_["random_seed"] = random_seed + conf["start_random_seed"]

            print(
                f"Setting: {setting}, Key Param: {'None'}, Random Seed: {random_seed}/{conf['n_random_seed']}"
            )
            runtime_logs[random_seed] = _process(**conf_)

        torch.save(runtime_logs, f"{online_pg_runtime_dir}/{detailed_configs_}.pt")


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
