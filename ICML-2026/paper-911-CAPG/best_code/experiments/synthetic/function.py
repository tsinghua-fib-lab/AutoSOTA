# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Functions called in the experiment pyfiles."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch

from src.dataset import (
    SyntheticDataGenerator,
    VectorialContextSampler,
    VectorialLatentSampler,
    VectorialRewardModel,
)
from src.learner import (
    OnlinePolicyLearner,
)
from src.policy import (
    BaseEarlyStagePolicy,
    BaseLateStagePolicy,
    BaselineEarlyStagePolicy,
    BaselineLateStagePolicy,
    EarlyStageTwoTowerModel,
    LateStageNeuralModel,
    OptimalEarlyStagePolicy,
    OptimalLateStagePolicy,
    UniformEarlyStagePolicy,
    UniformLateStagePolicy,
    OracleSoftmaxLateStagePolicy,
    VectorialActionSet,
)

def setup_data_generation_process(
    n_user: int,
    n_action: int,
    n_latent: int,
    n_output_action: int,
    dim_context: int,
    dim_action_emb: int,
    reward_scaler: Union[int, float],
    device: torch.device,
    random_seed: int,
) -> SyntheticDataGenerator:
    context_sampler = VectorialContextSampler(
        is_discrete=True,
        n_discrete_context=n_user,
        dim_context=dim_context,
        device=device,
        random_seed=random_seed,
    )
    action_set = VectorialActionSet(
        n_action=n_action,
        dim_action_emb=dim_action_emb,
        device=device,
        random_seed=random_seed,
    )
    latent_sampler = VectorialLatentSampler(
        is_discrete=True,
        n_discrete_latent=n_latent,
        dim_context=context_sampler.dim_context,
        dim_action_emb=action_set.dim_action_emb,
        device=device,
        random_seed=random_seed,
    )
    reward_model = VectorialRewardModel(
        context_sampler=context_sampler,
        action_set=action_set,
        n_output_action=n_output_action,
        reward_scaler=reward_scaler,
        device=device,
        random_seed=random_seed,
    )
    datagen = SyntheticDataGenerator(
        action_set=action_set,
        context_sampler=context_sampler,
        latent_sampler=latent_sampler,
        reward_model=reward_model,
    )
    return datagen


def evaluate_policy(
    env: SyntheticDataGenerator,
    early_stage_policy: BaseEarlyStagePolicy,
    late_stage_policy: BaseLateStagePolicy,
    is_deterministic_early_stage: bool,
    is_deterministic_late_stage: bool,
    n_candidate_action: int,
    model_assignment: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    online_performance = env.evaluate_policy_online(
        early_stage_policy=early_stage_policy,
        late_stage_policy=late_stage_policy,
        is_deterministic_early_stage=is_deterministic_early_stage,
        is_deterministic_late_stage=is_deterministic_late_stage,
        n_candidate_action=n_candidate_action,
        n_candidate_per_model=model_assignment,
    ).item()
    return online_performance


def train_online_pg_policy(
    env: SyntheticDataGenerator,
    early_stage_policy: BaselineEarlyStagePolicy,
    early_stage_lr: float,
    late_stage_optimality: str,
    credit_assignment_type: str,
    is_vanilla_replacement: str,
    n_epoch: int,
    n_epochs_per_log: int,
    n_candidate_action_train: int,
    n_candidate_action_eval: int,
    device: torch.device,
    random_seed: int,
    use_wandb: bool,
) -> Tuple[
    BaselineEarlyStagePolicy,
    Dict[str, torch.Tensor],
]:    
    if late_stage_optimality == "optimal":
        _, late_stage_policy = initialize_optimal_policy(
            env=env,
            device=device,
            random_seed=random_seed,
        )
    elif late_stage_optimality == "noisy":
        late_stage_policy = initialize_noisy_optimal_late_stage_policy(
            env=env,
            device=device,
            random_seed=random_seed,
        )
    elif late_stage_optimality == "uniform":
        _, late_stage_policy = initialize_uniform_policy(
            env=env,
            device=device,
            random_seed=random_seed,
        )
    elif late_stage_optimality == "anti":
        late_stage_policy = initialize_anti_optimal_late_stage_policy(
            env=env,
            device=device,
            random_seed=random_seed,
        )

    online_pg_learner = OnlinePolicyLearner(
        early_stage_policy=early_stage_policy,
        target_late_stage_policy=late_stage_policy,
        eval_late_stage_policy=late_stage_policy,
        early_stage_optimizer_kwargs={"lr": early_stage_lr},
        env=env,
        device=device,
        random_seed=random_seed,
    )

    credit_assignment_type_ = credit_assignment_type
    if credit_assignment_type == "ALL" and is_vanilla_replacement:
        credit_assignment_type_ = "ALL-SwR"

    trained_online_pg_early_stage_policy, online_pg_training_logs = (
        online_pg_learner.train_early_stage_policy_online(
            n_epoch=n_epoch,
            n_epochs_per_log=n_epochs_per_log,
            patience=torch.inf,
            make_copy=False,  #
            return_training_logs=True,
            credit_assignment_type=credit_assignment_type,  #
            is_vanilla_replacement=is_vanilla_replacement,
            is_deterministic_early_stage_eval=True,
            is_deterministic_late_stage_eval=(late_stage_optimality != "uniform"),
            n_candidate_action_train=n_candidate_action_train,  #
            n_candidate_action_eval=n_candidate_action_eval,  #
            random_seed=random_seed,
            use_wandb=use_wandb,
            experiment_name=f"Meta-ESR-{credit_assignment_type_}-pos",  # added prefix
        )
    )
    return (
        trained_online_pg_early_stage_policy,
        online_pg_training_logs,
    )

def runtime_online_pg_policy(
    env: SyntheticDataGenerator,
    early_stage_policy: BaselineEarlyStagePolicy,
    early_stage_lr: float,
    credit_assignment_type: str,
    is_vanilla_replacement: bool,
    n_epoch: int,
    n_steps_per_epoch: int,
    n_epochs_per_log: int,
    n_candidate_action_train: int,
    n_candidate_action_eval: int,
    device: torch.device,
    random_seed: int,
    use_wandb: bool,
) -> Tuple[
    BaselineEarlyStagePolicy,
    Dict[str, torch.Tensor],
]:    
    _, late_stage_policy = initialize_optimal_policy(
        env=env,
        device=device,
        random_seed=random_seed,
    )
    online_pg_learner = OnlinePolicyLearner(
        early_stage_policy=early_stage_policy,
        target_late_stage_policy=late_stage_policy,
        eval_late_stage_policy=late_stage_policy,
        early_stage_optimizer_kwargs={"lr": early_stage_lr},
        env=env,
        device=device,
        random_seed=random_seed,
    )

    credit_assignment_type_ = credit_assignment_type
    if credit_assignment_type == "ALL" and is_vanilla_replacement:
        credit_assignment_type_ = "ALL-SwR"

    trained_online_pg_early_stage_policy = (
        online_pg_learner.train_early_stage_policy_online(
            n_epoch=n_epoch,
            n_steps_per_epoch=n_steps_per_epoch,
            n_epochs_per_log=n_epochs_per_log,
            patience=torch.inf,
            make_copy=False,  #
            return_training_logs=False,
            credit_assignment_type=credit_assignment_type,  #
            is_vanilla_replacement=is_vanilla_replacement,
            is_deterministic_early_stage_eval=True,
            is_deterministic_late_stage_eval=True,
            n_candidate_action_train=n_candidate_action_train,  #
            n_candidate_action_eval=n_candidate_action_eval,  #
            random_seed=random_seed,
            use_wandb=use_wandb,
            experiment_name=f"Meta-ESR-{credit_assignment_type_}-runtime",  # added prefix
        )
    )


def save_logs(
    rootdir: str,
    random_seed: int,
    late_stage_optimality: str,
    credit_assignment_type: Optional[str],
    is_vanilla_replacement: bool,
    n_candidate_action_train: Optional[int],
    n_output_action: int,
    n_moe_model: int,
    trained_online_pg_early_stage_policy: Optional[BaseEarlyStagePolicy] = None,
    online_pg_training_logs: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    detailed_configs_ = f"n_candidate={n_candidate_action_train},late_stage={late_stage_optimality},n_model={n_moe_model},n_output={n_output_action},seed={random_seed}"

    online_pg_dir = Path(f"{rootdir}/online_early_stage/credit={credit_assignment_type}")
    online_pg_log_dir = Path(f"{rootdir}/online_early_stage/training_process/credit={credit_assignment_type}")

    online_pg_dir.mkdir(parents=True, exist_ok=True)
    online_pg_log_dir.mkdir(parents=True, exist_ok=True)

    if trained_online_pg_early_stage_policy is not None:
        online_pg_dir = Path(f"{rootdir}/online_early_stage/credit={credit_assignment_type}")
        online_pg_dir.mkdir(parents=True, exist_ok=True)

        online_pg_early_stage_policy_path = online_pg_dir / f"{detailed_configs_}.pt"

        torch.save(
            trained_online_pg_early_stage_policy.base_model.state_dict(),
            online_pg_early_stage_policy_path,
        )

    if online_pg_training_logs is not None:
        for key in online_pg_training_logs.keys():

            if credit_assignment_type == "ALL" and is_vanilla_replacement:
                online_pg_log_dir = Path(f"{rootdir}/online_early_stage/training_process/credit={'ALL-SwR'}/{key}")
            else:
                online_pg_log_dir = Path(f"{rootdir}/online_early_stage/training_process/credit={credit_assignment_type}/{key}")

            online_pg_log_dir.mkdir(parents=True, exist_ok=True)

            online_pg_training_logs_path = (
                online_pg_log_dir / f"{detailed_configs_}.pt"
            )
            torch.save(online_pg_training_logs[key], online_pg_training_logs_path)


def initialize_trainable_policy(
    env: SyntheticDataGenerator,
    dim_model_emb: int,
    n_moe_model: int,
    device: torch.device,
    random_seed: int,
) -> Tuple[BaselineEarlyStagePolicy, BaselineLateStagePolicy]:
    early_stage_base_model = EarlyStageTwoTowerModel(
        n_context=env.context_sampler.n_discrete_context,
        n_action=env.action_set.n_action,
        dim_emb=dim_model_emb,
        n_model=n_moe_model,
    )
    trainable_early_stage_policy = BaselineEarlyStagePolicy(
        base_model=early_stage_base_model,
        action_set=env.action_set,
        device=device,
        random_seed=random_seed,
    )
    late_stage_base_model = LateStageNeuralModel(
        n_context=env.context_sampler.n_discrete_context,
        n_latent=env.latent_sampler.n_discrete_latent,
        n_action=env.action_set.n_action,
        dim_emb=dim_model_emb,
    )
    trainable_late_stage_policy = BaselineLateStagePolicy(
        base_model=late_stage_base_model,
        action_set=env.action_set,
        device=device,
        random_seed=random_seed,
    )
    return trainable_early_stage_policy, trainable_late_stage_policy


def initialize_optimal_policy(
    env: SyntheticDataGenerator,
    device: torch.device,
    random_seed: int,
) -> Tuple[OptimalEarlyStagePolicy, OptimalLateStagePolicy]:
    early_stage_policy_optimal = OptimalEarlyStagePolicy(
        action_set=env.action_set,
        device=device,
        random_seed=random_seed,
    )
    late_stage_policy_optimal = OptimalLateStagePolicy(
        action_set=env.action_set,
        device=device,
        random_seed=random_seed,
    )
    return early_stage_policy_optimal, late_stage_policy_optimal


def initialize_uniform_policy(
    env: SyntheticDataGenerator,
    device: torch.device,
    random_seed: int,
) -> Tuple[UniformEarlyStagePolicy, UniformLateStagePolicy]:
    uniform_early_stage_policy = UniformEarlyStagePolicy(
        action_set=env.action_set,
        device=device,
        random_seed=random_seed,
    )
    uniform_late_stage_policy = UniformLateStagePolicy(
        action_set=env.action_set,
        device=device,
        random_seed=random_seed,
    )
    return uniform_early_stage_policy, uniform_late_stage_policy


def initialize_noisy_optimal_late_stage_policy(
    env: SyntheticDataGenerator,
    device: torch.device,
    random_seed: int,
) -> Tuple[OracleSoftmaxLateStagePolicy]:
    noisy_late_stage_policy = OracleSoftmaxLateStagePolicy(
        action_set=env.action_set,
        inverse_temperature=1.0,
        device=device,
        random_seed=random_seed,
    )
    return noisy_late_stage_policy


def initialize_anti_optimal_late_stage_policy(
    env: SyntheticDataGenerator,
    device: torch.device,
    random_seed: int,
) -> Tuple[OptimalLateStagePolicy]:
    anti_late_stage_policy = OptimalLateStagePolicy(
        action_set=env.action_set,
        is_anti_optimal=True,
        device=device,
        random_seed=random_seed,
    )
    return anti_late_stage_policy


def load_naive_cf_policy(
    env: SyntheticDataGenerator,
    dim_model_emb: int,
    early_stage_naive_cf_path: str,
    device: torch.device,
    random_seed: int,
) -> BaselineEarlyStagePolicy:
    early_stage_naive_cf_policy, _ = initialize_trainable_policy(
        env=env,
        dim_model_emb=dim_model_emb,
        n_moe_model=1,
        device=device,
        random_seed=random_seed,
    )
    early_stage_naive_cf_policy.base_model.load_state_dict(
        torch.load(early_stage_naive_cf_path, map_location=device)
    )
    return early_stage_naive_cf_policy


def load_pg_policy(
    env: SyntheticDataGenerator,
    dim_model_emb: int,
    early_stage_model_path: str,
    device: torch.device,
    random_seed: int,
) -> BaselineEarlyStagePolicy:
    early_stage_pg_policy, _ = initialize_trainable_policy(
        env=env,
        dim_model_emb=dim_model_emb,
        n_moe_model=1,
        device=device,
        random_seed=random_seed,
    )
    early_stage_pg_policy.base_model.load_state_dict(
        torch.load(early_stage_model_path, map_location=device)
    )
    return early_stage_pg_policy


def load_late_stage_cf_policy(
    env: SyntheticDataGenerator,
    dim_model_emb: int,
    late_stage_naive_cf_path: str,
    device: torch.device,
    random_seed: int,
) -> BaselineLateStagePolicy:
    _, late_stage_naive_cf_policy = initialize_trainable_policy(
        env=env,
        dim_model_emb=dim_model_emb,
        device=device,
        random_seed=random_seed,
    )
    late_stage_naive_cf_policy.base_model.load_state_dict(
        torch.load(late_stage_naive_cf_path, map_location=device)
    )
    return late_stage_naive_cf_policy
