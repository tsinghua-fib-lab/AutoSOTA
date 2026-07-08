# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Traning procedure of on-policy learning."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import wandb

import torch

from synthetic.dataset import (
    BaseDataGenerator,
)
from synthetic.policy import (
    BaseEarlyStagePolicy,
    BaseLateStagePolicy,
)
from torch.optim import Adagrad, Optimizer
from tqdm.auto import tqdm

from .base import BasePolicyLearner


@dataclass
class OnlinePolicyLearner(BasePolicyLearner):
    """Training procedure of on-policy learning.

    Input
    ------
    env: BaseDataGenerator
        The data generation environment.

    early_stage_policy: BaseEarlyStagePolicy
        The early stage policy.

    target_late_stage_policy: BaseLateStagePolicy
        The late stage policy used for training.

    eval_late_stage_policy: BaseLateStagePolicy
        The late stage policy used for evaluation.

    is_model_free_target_late_stage_policy: bool, default=False
        Whether the target late stage policy is model-free.

    is_model_free_eval_late_stage_policy: bool, default=False
        Whether the evaluation late stage policy is model-free.

    early_stage_optimizer: Optimizer = Adagrad
        The optimizer to use for the early stage policy.

    early_stage_optimizer_kwargs: Optional[Dict[str, Any]]
        The optimizer kwargs to use for the early stage policy.

    device: torch.device, default=torch.device("cpu")
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    env: BaseDataGenerator
    early_stage_policy: BaseEarlyStagePolicy
    target_late_stage_policy: BaseLateStagePolicy
    eval_late_stage_policy: BaseLateStagePolicy
    is_model_free_target_late_stage_policy: bool = False
    is_model_free_eval_late_stage_policy: bool = False
    early_stage_optimizer: Optimizer = Adagrad
    early_stage_optimizer_kwargs: Optional[Dict[str, Any]] = None
    device: torch.device = torch.device("cpu")
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)

            self.early_stage_policy.base_model = self.early_stage_policy.base_model.to(
                self.device
            )

            if not self.is_model_free_target_late_stage_policy:
                self.target_late_stage_policy.base_model = (
                    self.target_late_stage_policy.base_model.to(self.device)
                )
            if not self.is_model_free_eval_late_stage_policy:
                self.eval_late_stage_policy.base_model = (
                    self.eval_late_stage_policy.base_model.to(self.device)
                )

        if self.early_stage_optimizer_kwargs is None:
            self.early_stage_optimizer_kwargs = {"lr": 1e-2}

        self.n_model = self.early_stage_policy.base_model.n_model

    def _policy_gradient(
        self,
        early_stage_log_prob: torch.Tensor,
        reward: torch.Tensor,
        agg_reward: torch.Tensor,
        position_weight: Optional[torch.Tensor] = None,
        credit_assignment_type: str = "full",
    ):
        """Compute the policy gradient. Note that this is a rough approximation of the true policy gradient.

        Input
        ------
        early_stage_log_prob: torch.Tensor, shape (n_samples, n_output_action) or (n_samples, )
            The log probability of the candidate action (differential).

        reward: torch.Tensor, shape (n_samples, n_output_action)
            The position-wise reward (non-differential).

        agg_reward: torch.Tensor, shape (n_samples, 1)
            The aggregated reward (non-differential).

        position_weight: Optional[torch.Tensor], shape (n_output_action, )
            The position weight for the reward.

        credit_assignment_type: str, default="CA"
            The credit assignment method. Either "CA", "ALL", "TOP1", or "JOINT".
            "CA" is the credit-assigned PG, "ALL" is the vanilla PG, and "TOP1" is the top-1 PG.
            "JOINT" is the vanilla PG, where gradient is taken for the aggregated reward, not indidual reward for each output action.

        Output
        ------
        policy_gradient: torch.Tensor, shape (1, )
            The policy gradient.

        """
        if credit_assignment_type != "JOINT":
            if credit_assignment_type in ["CA", "TOP1"]:
                policy_gradient = -early_stage_log_prob * reward
            else:
                policy_gradient = -early_stage_log_prob.unsqueeze(1) * reward

            if position_weight is None:
                policy_gradient = policy_gradient.mean(dim=-1)
            else:
                policy_gradient = (policy_gradient * position_weight).sum(dim=-1)

        else:
            policy_gradient = -early_stage_log_prob * agg_reward.squeeze()

        return policy_gradient.mean()

    def train_early_stage_policy_online(
        self,
        position_weight: Optional[torch.Tensor] = None,
        credit_assignment_type: str = "CA",
        is_vanilla_replacement: bool = False,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        n_candidate_action_train: int = 20,
        n_epoch: int = 1000,
        n_steps_per_epoch: int = 10,
        n_epochs_per_log: int = 10,
        patience: int = 5,
        batch_size: int = 128,
        make_copy: bool = False,
        return_training_logs: bool = False,
        is_deterministic_early_stage_eval: bool = False,
        is_deterministic_late_stage_eval: bool = False,
        n_candidate_action_eval: int = 10,
        n_candidate_per_model_eval: Optional[List[int]] = None,
        save_path: Optional[Path] = None,
        random_seed: Optional[int] = None,
        use_wandb: bool = False,
        experiment_name: str = "online_pg",
        run_id: str = "0",
    ):
        """Train the base model used in the early stage policy.

        Input
        ------
        position_weight: Optional[torch.Tensor], shape (n_output_action, )
            The position weight for the reward.

        credit_assignment_type: str, default="CA"
            The credit assignment method. Either "CA", "ALL", "TOP1", or "JOINT".
            "CA" is the credit-assigned PG, "ALL" is the vanilla PG, and "TOP1" is the top-1 PG.
            "JOINT" is the vanilla PG, where gradient is taken for the aggregated reward, not indidual reward for each output action.

        is_vanilla_replacement: bool, default=False
            Whether to use the vanilla policy gradient under sampling w/ replacement (SwR) approximation.

        val_ratio: float, default=0.1
            The ratio of validation data.

        test_ratio: float, default=0.1
            The ratio of test data.

        n_epoch: int, default=1000
            The number of epochs to train.

        n_steps_per_epoch: int, default=10
            The number of steps per epoch.

        n_epochs_per_log: int, default=10
            The number of epochs per log.

        patience: int, default=5
            The number of epochs to wait before early stopping.

        batch_size: int, default=128
            The batch size.

        make_copy: bool, default=False
            Whether to make a copy of the base model.

        return_training_logs: bool, default=False
            Whether to return the training logs.

        is_deterministic_early_stage_eval: bool, default=False
            Whether to use deterministic policy for evaluation.

        is_deterministic_late_stage_eval: bool, default=False
            Whether to use deterministic policy for evaluation.

        n_candidate_action_eval: int, default=10
            The number of candidate actions to sample for evaluation.

        n_candidate_per_model_eval: Optional[List[int]]
            The number of candidate actions to sample per model for evaluation.

        save_path: Optional[Path]
            The path to save the model.

        random_seed: Optional[int]
            The random seed to use.

        use_wandb: bool, default=False
            Whether to use wandb to report the training statistics.

        experiment_name: str, default="online_pg"
            Experiment name (project) used for wandb reports.

        """
        if credit_assignment_type not in ["CA", "ALL", "TOP1", "JOINT"]:
            raise ValueError(
                f"credit_assignment_type must be either 'CA', 'ALL', 'TOP1', or 'JOINT', but got {credit_assignment_type}."
            )

        if random_seed is None:
            random_seed = self.random_seed

        self.seed(random_seed)

        if use_wandb:
            wandb_run = wandb.init(
                entity="", 
                project=experiment_name, 
                name=f"{random_seed}", 
                reinit=True,
            )

        if make_copy:
            early_stage_policy = deepcopy(self.early_stage_policy)
        else:
            early_stage_policy = self.early_stage_policy

        optimizer = self.early_stage_optimizer(
            early_stage_policy.base_model.parameters(),
            **self.early_stage_optimizer_kwargs,
        )

        train_losses = torch.zeros((n_epoch + 1,), device=self.device)
        action_probs = torch.zeros((n_epoch + 1,), device=self.device)
        policy_values = torch.zeros(
            (n_epoch // n_epochs_per_log + 1,), device=self.device
        )

        policy_values[0] = self.env.evaluate_policy_online(
            early_stage_policy=early_stage_policy,
            late_stage_policy=self.eval_late_stage_policy,
            is_deterministic_early_stage=is_deterministic_early_stage_eval,
            is_deterministic_late_stage=is_deterministic_late_stage_eval,
            n_candidate_action=n_candidate_action_eval,
            n_candidate_per_model=n_candidate_per_model_eval,
        )

        gradient_overflow_flg = False

        with tqdm(range(n_epoch)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[Train base early stage model: Epoch {i}]")
                pbar.set_postfix(
                    {
                        "train_loss": f"{train_losses[i]:.4g}",
                        "action_prob": f"{action_probs[i]:.4g}",
                        "policy_value": f"{policy_values[i // n_epochs_per_log]:.4g}",
                    },
                )

                for j in range(n_steps_per_epoch):
                    require_grad_model_id_ = None
                    n_candidate_action_ = n_candidate_action_train
                    n_candidate_per_model_ = None

                    context_, context_id_ = (
                        self.env.context_sampler.sample(
                            batch_size
                        )
                    )
                    latent_, latent_id_ = self.env.latent_sampler.sample(
                        batch_size
                    )

                    factual_rewards_ = self.env.retrieve_factual_rewards(
                        context_id=context_id_,
                        latent_id=latent_id_,
                    )

                    with torch.no_grad():
                        candidate_actions_ = early_stage_policy.sample(
                            context=context_,
                            context_id=context_id_,
                            n_candidate_action=n_candidate_action_,
                            n_candidate_per_model=n_candidate_per_model_,
                        )
                        action_ids_ = self.target_late_stage_policy.sample(
                            context=context_,
                            context_id=context_id_,
                            latent=latent_,
                            latent_id=latent_id_,
                            candidate_actions=candidate_actions_,
                            factual_rewards=factual_rewards_,
                            n_output_action=self.env.n_output_action,
                        )

                    reward_, agg_reward_ = self.env.reward_model.sample(
                        context=context_,
                        latent=latent_,
                        actions=action_ids_,
                    )

                    early_stage_prob_, early_stage_log_prob_ = (
                        early_stage_policy.calc_prob_given_actions(
                            context=context_,
                            context_id=context_id_,
                            actions=action_ids_,
                            candidate_actions=candidate_actions_,
                            n_candidate_action=n_candidate_action_,
                            n_candidate_per_model=n_candidate_per_model_,
                            require_grad_model_id=require_grad_model_id_,
                            is_credit_assigned=(credit_assignment_type == "CA"),
                            is_top1=(credit_assignment_type == "TOP1"),
                            is_vanilla_replacement=(credit_assignment_type == "ALL") and is_vanilla_replacement,
                        )
                    )

                    loss_ = self._policy_gradient(
                        early_stage_log_prob=early_stage_log_prob_,
                        reward=reward_,
                        agg_reward=agg_reward_,
                        position_weight=position_weight,
                        credit_assignment_type=credit_assignment_type,
                    )

                    optimizer.zero_grad()

                    try:
                        with torch.autograd.set_detect_anomaly(True):
                            loss_.backward()

                    except Exception as e:
                        print(f"Gradient overflow at epoch {i + 1}", e)
                        gradient_overflow_flg = True
                        train_losses[i + 1] = 0
                        action_probs[i + 1] = 0
                        break

                    optimizer.step()

                    train_losses[i + 1] += loss_ / n_steps_per_epoch
                    action_probs[i + 1] += (
                        early_stage_prob_
                        ** (1 / n_candidate_action_eval)
                    ).mean() / n_steps_per_epoch

                if gradient_overflow_flg:
                    break

                if (i + 1) % n_epochs_per_log == 0:
                    policy_value_ = self.env.evaluate_policy_online(
                        early_stage_policy=early_stage_policy,
                        late_stage_policy=self.eval_late_stage_policy,
                        is_deterministic_early_stage=is_deterministic_early_stage_eval,
                        is_deterministic_late_stage=is_deterministic_late_stage_eval,
                        n_candidate_action=n_candidate_action_eval,
                        n_candidate_per_model=n_candidate_per_model_eval,
                    )
                    policy_values[(i + 1) // n_epochs_per_log] += policy_value_.item()

                if use_wandb:
                    wandb.log(
                        {
                            "train_loss": train_losses[i + 1],
                            "average_prob": action_probs[i + 1],
                            "policy_value": policy_values[(i + 1) // n_epochs_per_log],
                        }
                    )

        if use_wandb:
            wandb_run.finish()

        self.trained_early_stage_policy = early_stage_policy

        if save_path is not None:
            self.save_early_stage_model(save_path)

        if return_training_logs:
            output = (
                early_stage_policy,
                {
                    "train_losses": train_losses,
                    "action_probs": action_probs,
                    "policy_values": policy_values,
                },
            )
        else:
            output = early_stage_policy

        return output

    def train_early_stage_policy_offline(self, **kwargs):
        raise NotImplementedError()
