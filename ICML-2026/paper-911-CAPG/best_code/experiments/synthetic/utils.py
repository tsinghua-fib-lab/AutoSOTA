# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Useful tools for teh experiments."""

import random
from typing import Union

import torch

from omegaconf import DictConfig


def format_runtime(start: Union[int, float], finish: Union[int, float]) -> str:
    runtime = finish - start
    hour = int(runtime // 3600)
    min = int((runtime) // 60 % 60)
    sec = int(runtime % 60)
    return f"{hour}h.{min}m.{sec}s"


def defaultdict_to_dict(d):
    if isinstance(d, dict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


def reset_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def assert_configuration(cfg: DictConfig) -> None:
    setting = cfg.setting.setting
    assert setting in [
        "default",
        "n_candidate_action_eval",
        "n_output_action",
        "n_moe_model",
        "late_stage_optimality",
    ]

    n_candidate_action_eval = cfg.setting.n_candidate_action_eval
    if setting != "n_candidate_action_eval":
        assert n_candidate_action_eval in [5, 10, 20, 50, 100, 200]
    else:
        for value in n_candidate_action_eval:
            assert value in [5, 10, 20, 50, 100, 200]

    n_candidate_action_train = cfg.setting.n_candidate_action_train
    if not n_candidate_action_train == "auto":
        assert n_candidate_action_train == n_candidate_action_eval

    n_output_action = cfg.setting.n_output_action
    assert n_output_action in [1, 5, 10]

    n_moe_model = cfg.model.n_moe_model
    assert n_moe_model in [1, 2, 5, 10]
    assert n_moe_model <= n_candidate_action_train

    late_stage_optimality = cfg.setting.late_stage_optimality
    assert late_stage_optimality in ["anti", "uniform", "noisy", "optimal"]

    credit_assignment_type = cfg.model.credit_assignment_type
    assert credit_assignment_type in ["CA", "ALL", "TOP1"]

    n_user = cfg.setting.n_user
    assert n_user == 1000

    n_latent = cfg.setting.n_latent
    assert n_latent == 1

    dim_context = cfg.setting.dim_context
    assert dim_context in [5, 10]

    dim_action_emb = cfg.setting.dim_action_emb
    assert dim_action_emb in [5, 10]

    reward_scaler = cfg.setting.reward_scaler
    assert reward_scaler >= 1

    device = cfg.setting.device
    assert device in ["cpu", "cuda"]

    n_random_seed = cfg.setting.n_random_seed
    assert n_random_seed >= 1

    start_random_seed = cfg.setting.start_random_seed
    assert start_random_seed >= 0

    base_random_seed = cfg.setting.base_random_seed
    assert base_random_seed >= 0

    dim_model_emb = cfg.model.dim_model_emb
    assert dim_model_emb in [5, 10]

    early_stage_naive_cf_lr = cfg.model.early_stage_naive_cf_lr
    assert early_stage_naive_cf_lr >= 0

    late_stage_neural_lr = cfg.model.late_stage_neural_lr
    assert late_stage_neural_lr >= 0

    online_vanilla_pg_lr = cfg.model.online_vanilla_pg_lr
    assert online_vanilla_pg_lr >= 0

    online_credit_assigned_pg_lr = cfg.model.online_credit_assigned_pg_lr
    assert online_credit_assigned_pg_lr >= 0

    n_epoch = cfg.model.n_epoch
    assert n_epoch >= 1

    n_steps_per_epoch = cfg.model.n_steps_per_epoch
    assert n_steps_per_epoch >= 1

    n_epochs_per_log = cfg.model.n_epochs_per_log
    assert n_epochs_per_log >= 1

    early_stage_naive_cf_path = cfg.path.early_stage_naive_cf_path
    if early_stage_naive_cf_path != "auto":
        assert early_stage_naive_cf_path.endswith(".pt")

    late_stage_naive_cf_path = cfg.path.late_stage_naive_cf_path
    if late_stage_naive_cf_path != "auto":
        assert late_stage_naive_cf_path.endswith(".pt")

    early_stage_online_credit_assigned_pg_path = (
        cfg.path.early_stage_online_credit_assigned_pg_path
    )
    if early_stage_online_credit_assigned_pg_path != "auto":
        assert early_stage_online_credit_assigned_pg_path.endswith(".pt")

    early_stage_online_vanilla_pg_path = cfg.path.early_stage_online_vanilla_pg_path
    if early_stage_online_vanilla_pg_path != "auto":
        assert early_stage_online_vanilla_pg_path.endswith(".pt")
