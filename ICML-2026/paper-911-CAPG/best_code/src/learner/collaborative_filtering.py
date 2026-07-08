# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Training procedure of collaborative filtering."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.dataset import (
    BaseDataGenerator,
    LoggedDataset,
)
from src.policy import (
    BaseEarlyStagePolicy,
    BaseLateStagePolicy,
)
from torch.optim import Adagrad, Optimizer
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from .base import BasePolicyLearner


@dataclass
class CollaborativeFilteringLearner(BasePolicyLearner):
    """Training procedure of collaborative filtering.

    Input
    ------
    early_stage_policy: BaseEarlyStagePolicy
        The early stage policy.

    late_stage_policy: BaseLateStagePolicy
        The late stage policy.

    is_model_free_early_stage_policy: bool = False
        Whether the early stage policy is model-free.

    is_model_free_late_stage_policy: bool = False
        Whether the late stage policy is model-free.

    model_selector: nn.Module
        The model selector (i.e., classifier) of the mixture-of-expert policy.
        Note that this is only used during the training process to weigh the loss among multiple models, and not used in the inference process.

    env: Optional[BaseDataGenerator]
        The data generation environment.

    early_stage_optimizer: Optimizer = Adagrad
        The optimizer to use for the early stage policy.

    late_stage_optimizer: Optimizer = Adagrad
        The optimizer to use for the late stage policy.

    model_selector_optimizer: Optimizer = Adagrad
        The optimizer to use for the model selector of the mixture-of-experts model.

    early_stage_optimizer_kwargs: Optional[Dict[str, Any]]
        The optimizer kwargs to use for the early stage policy.

    late_stage_optimizer_kwargs: Optional[Dict[str, Any]]
        The optimizer kwargs to use for the late stage policy.

    model_selector_optimizer_kwargs: Optional[Dict[str, Any]]
        The optimizer kwargs to use for the model selector of the mixture-of-experts model.

    device: torch.device, default=torch.device("cpu")
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    early_stage_policy: BaseEarlyStagePolicy
    late_stage_policy: BaseLateStagePolicy
    is_model_free_early_stage_policy: bool = False
    is_model_free_late_stage_policy: bool = False
    model_selector: Optional[nn.Module] = None
    env: Optional[BaseDataGenerator] = None
    early_stage_optimizer: Optimizer = Adagrad
    late_stage_optimizer: Optimizer = Adagrad
    model_selector_optimizer: Optimizer = Adagrad
    early_stage_optimizer_kwargs: Optional[Dict[str, Any]] = None
    late_stage_optimizer_kwargs: Optional[Dict[str, Any]] = None
    model_selector_optimizer_kwargs: Optional[Dict[str, Any]] = None
    device: torch.device = torch.device("cpu")
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)

            if not self.is_model_free_early_stage_policy:
                self.early_stage_policy.base_model = (
                    self.early_stage_policy.base_model.to(self.device)
                )
            if not self.is_model_free_late_stage_policy:
                self.late_stage_policy.base_model = (
                    self.late_stage_policy.base_model.to(self.device)
                )

            if self.model_selector is not None:
                self.model_selector = self.model_selector.to(self.device)

        if self.early_stage_optimizer_kwargs is None:
            self.early_stage_optimizer_kwargs = {"lr": 1e-2}

        if self.late_stage_optimizer_kwargs is None:
            self.late_stage_optimizer_kwargs = {"lr": 1e-2}

        if self.model_selector_optimizer_kwargs is None:
            self.model_selector_optimizer_kwargs = {"lr": 1e-2}

        if not self.is_model_free_early_stage_policy:
            self.n_base_model = self.early_stage_policy.base_model.n_model
            if self.n_base_model > 1:
                if self.model_selector is None:
                    raise ValueError("model_selector must be given.")
                else:
                    assert self.model_selector.n_model == self.n_base_model
            else:
                self.model_selector = None

    def _MSELoss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ):
        """Custom MSE loss for the mixture-of-expert model.

        Input
        ------
        preds: torch.Tensor, shape (n_model, batch_size, n_output_action)
            The predictions of the multiple expert models.

        targets: torch.Tensor, shape (batch_size, n_output_action)
            The prediction target.

        weights: Optional[torch.Tensor], shape (batch_size, n_model)
            The weights of each model.

        """
        se = (preds - targets.unsqueeze(0)) ** 2

        if weights is not None:
            if se.ndim == 3:
                se = se * weights.T.unsqueeze(2)
            else:
                se = se * weights.T

            mse = se.sum(dim=0).mean()

        else:
            mse = se.mean()

        return mse

    def _BSELoss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ):
        """Custom BCE loss for the mixture-of-expert model.

        Input
        ------
        preds: torch.Tensor, shape (n_model, batch_size, n_output_action)
            The predictions of the multiple expert models.

        targets: torch.Tensor, shape (batch_size, n_output_action)
            The prediction target.

        weights: Optional[torch.Tensor], shape (batch_size, n_model)
            The weights of each model.

        """
        targets = targets.unsqueeze(0)
        logit = -(targets * preds.log() + (1 - targets) * (1 - preds).log())

        if weights is not None:
            if logit.ndim == 3:
                logit = logit * weights.T.unsqueeze(2)
            else:
                logit = logit * weights.T

            bce = logit.sum(dim=0).mean()

        else:
            bce = logit.mean()

        return bce

    def train_early_stage_policy_offline(
        self,
        dataset: LoggedDataset,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        loss_type: str = "mse",
        n_epoch: int = 1000,
        n_steps_per_epoch: int = 10,
        n_epochs_per_log: int = 10,
        patience: int = 5,
        n_candidate_action: int = 10,
        n_candidate_per_model: Optional[List[int]] = None,
        batch_size: int = 128,
        make_copy: bool = False,
        return_training_logs: bool = False,
        is_deterministic_early_stage_eval: bool = True,
        is_deterministic_late_stage_eval: bool = True,
        save_path: Optional[Path] = None,
        random_seed: Optional[int] = None,
    ):
        """Train the base model used in the early stage policy.

        Input
        ------
        dataset: LoggedDataset
            The training data, which contains the following keys:

            .. code-block:: python

                key: [
                    "context",     # (n_samples, dim_context)
                    "latent",      # (n_samples, dim_hidden, dim_action_emb)
                    "action",      # (n_samples, n_output_action)
                    "reward",      # (n_samples, n_output_action)
                    "agg_reward",  # (n_samples, 1)
                    "context_id",  # (n_samples, ), optional
                    "latent_id",   # (n_samples, ), optional
                    ]

        val_ratio: float, default=0.1
            The ratio of validation data.

        test_ratio: float, default=0.1
            The ratio of test data.

        loss_type: str, default="mse"
            The loss type to use. Either "mse" or "bce".

        n_epoch: int, default=1000
            The number of epochs to train.

        n_steps_per_epoch: int, default=10
            The number of steps per epoch.

        n_epochs_per_log: int, default=10
            The number of epochs per log.

        patience: int, default=5
            The number of epochs to wait before early stopping.

        n_candidate_action: int, default=10
            The number of candidate actions to use for online policy evaluation.

        batch_size: int, default=128
            The batch size.

        make_copy: bool, default=False
            Whether to make a copy of the base model.

        return_training_logs: bool, default=False
            Whether to return the training logs.

        is_deterministic_early_stage_eval: bool, default=True
            Whether to use deterministic early stage policy for evaluation.

        is_deterministic_late_stage_eval: bool, default=True
            Whether to use deterministic late stage policy for evaluation.

        save_path: Optional[Path]
            The path to save the model.

        random_seed: Optional[int]
            The random seed to use.

        """
        if self.is_model_free_early_stage_policy:
            raise ValueError("Cannot train model-free early stage policy.")

        if random_seed is None:
            random_seed = self.random_seed

        self.seed(random_seed)

        if self.model_selector is None:
            if make_copy:
                early_stage_policy = deepcopy(self.early_stage_policy)
                model_selector = None
            else:
                early_stage_policy = self.early_stage_policy
                model_selector = None
        else:
            if make_copy:
                early_stage_policy = deepcopy(self.early_stage_policy)
                model_selector = deepcopy(self.model_selector)
            else:
                early_stage_policy = self.early_stage_policy
                model_selector = self.model_selector

        early_stage_optimizer = self.early_stage_optimizer(
            early_stage_policy.base_model.parameters(),
            **self.early_stage_optimizer_kwargs,
        )

        if model_selector is not None:
            model_selector_optimizer = self.model_selector_optimizer(
                self.model_selector.parameters(),
                **self.model_selector_optimizer_kwargs,
            )

        val_size = int(val_ratio * len(dataset))
        test_size = int(test_ratio * len(dataset))
        train_size = len(dataset) - val_size - test_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        train_losses = torch.zeros((n_epoch + 1,), device=self.device)
        val_losses = torch.zeros((n_epoch // n_epochs_per_log + 1,), device=self.device)
        policy_values = torch.zeros(
            (n_epoch // n_epochs_per_log + 1,), device=self.device
        )

        loss_fn = self._MSELoss if loss_type == "mse" else self._BCELoss

        with tqdm(range(n_epoch)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[Train base early stage model: Epoch {i}]")
                pbar.set_postfix(
                    {
                        "train_loss": f"{train_losses[i]:.4g}",
                        "val_loss": f"{val_losses[i // n_epochs_per_log]:.4g}",
                        "policy_value": f"{policy_values[i // n_epochs_per_log]:.4g}",
                    },
                )

                train_iterator = iter(train_loader)

                for j in range(n_steps_per_epoch):
                    try:
                        batch_ = next(train_iterator)
                    except StopIteration:
                        train_iterator = iter(train_loader)
                        batch_ = next(train_iterator)

                    context_ = batch_["context"].to(self.device)
                    latent_ = batch_["latent"].to(self.device)
                    context_id_ = batch_["context_id"].to(self.device)
                    latent_id_ = batch_["latent_id"].to(self.device)
                    action_ids_ = batch_["action"].to(self.device)
                    rewards_ = batch_["reward"].to(self.device)

                    if model_selector is None:
                        preds_ = early_stage_policy.base_model(
                            context=context_,
                            context_id=context_id_,
                            action_ids=action_ids_,
                        )
                        loss_ = loss_fn(preds_, rewards_)

                        early_stage_optimizer.zero_grad()
                        loss_.backward()
                        early_stage_optimizer.step()

                    else:
                        # freeze model selector and unfreeze base model
                        preds_ = early_stage_policy.base_model(
                            context=context_,
                            context_id=context_id_,
                            action_ids=action_ids_,
                        )

                        with torch.no_grad():
                            weights_ = model_selector(
                                latent=latent_, latent_id=latent_id_
                            )

                        loss_ = loss_fn(preds_, rewards_, weights_)

                        early_stage_optimizer.zero_grad()
                        loss_.backward()
                        early_stage_optimizer.step()

                        # freeze base model and unfreeze model selector
                        with torch.no_grad():
                            preds_ = early_stage_policy.base_model(
                                context=context_,
                                context_id=context_id_,
                                action_ids=action_ids_,
                            )

                        weights_ = model_selector(latent=latent_, latent_id=latent_id_)

                        loss_ = loss_fn(preds_, rewards_, weights_)

                        model_selector_optimizer.zero_grad()
                        loss_.backward()
                        model_selector_optimizer.step()

                    train_losses[i + 1] += loss_ / n_steps_per_epoch

                if (i + 1) % n_epochs_per_log == 0:
                    with torch.no_grad():
                        for batch_ in val_loader:
                            context_ = batch_["context"].to(self.device)
                            latent_ = batch_["latent"].to(self.device)
                            context_id_ = batch_["context_id"].to(self.device)
                            latent_id_ = batch_["latent_id"].to(self.device)
                            action_ids_ = batch_["action"].to(self.device)
                            rewards_ = batch_["reward"].to(self.device)
                            preds_ = early_stage_policy.base_model(
                                context=context_,
                                context_id=context_id_,
                                action_ids=action_ids_,
                            )

                            if model_selector is None:
                                val_loss_ = loss_fn(preds_, rewards_)
                            else:
                                weights_ = model_selector(
                                    latent=latent_, latent_id=latent_id_
                                )
                                val_loss_ = loss_fn(preds_, rewards_, weights_)

                            val_losses[(i + 1) // n_epochs_per_log] += (
                                val_loss_ * len(batch_["context_id"]) / len(val_dataset)
                            )

                    if self.env is not None:
                        policy_value_ = self.env.evaluate_policy_online(
                            early_stage_policy=early_stage_policy,
                            late_stage_policy=self.late_stage_policy,
                            is_deterministic_early_stage=is_deterministic_early_stage_eval,
                            is_deterministic_late_stage=is_deterministic_late_stage_eval,
                            n_candidate_action=n_candidate_action,
                            n_candidate_per_model=n_candidate_per_model,
                        )
                        policy_values[(i + 1) // n_epochs_per_log] += (
                            policy_value_.item()
                        )

                    # early stopping
                    current_epoch = (i + 1) // n_epochs_per_log
                    best_loss_epoch = torch.argmin(
                        val_losses[1 : current_epoch + 1]
                    ).item()

                    if current_epoch - best_loss_epoch > patience:
                        print("early stopping at epoch", i + 1)
                        break

        self.trained_early_stage_policy = early_stage_policy

        if save_path is not None:
            self.save_early_stage_model(save_path)

        if return_training_logs:
            output = (
                early_stage_policy,
                model_selector,
                {
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "policy_values": policy_values,
                },
            )
        else:
            output = (early_stage_policy, model_selector)

        return output

    def train_late_stage_policy_offline(
        self,
        dataset: LoggedDataset,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        loss_type: str = "mse",
        n_epoch: int = 1000,
        n_steps_per_epoch: int = 10,
        n_epochs_per_log: int = 10,
        patience: int = 5,
        n_candidate_action: int = 10,
        n_candidate_per_model: Optional[List[int]] = None,
        batch_size: int = 128,
        make_copy: bool = False,
        return_training_logs: bool = False,
        is_deterministic_early_stage_eval: bool = True,
        is_deterministic_late_stage_eval: bool = True,
        save_path: Optional[Path] = None,
        random_seed: Optional[int] = None,
    ):
        """Train the base model used in the late stage policy.

        Input
        ------
        dataset: LoggedDataset
            The training data, which contains the following keys:

            .. code-block:: python

                key: [
                    "context",     # (n_samples, dim_context)
                    "latent",      # (n_samples, dim_hidden, dim_action_emb)
                    "action",      # (n_samples, n_output_action)
                    "reward",      # (n_samples, n_output_action)
                    "agg_reward",  # (n_samples, 1)
                    "context_id",  # (n_samples, ), optional
                    "latent_id",   # (n_samples, ), optional
                    ]

        val_ratio: float, default=0.1
            The ratio of validation data.

        test_ratio: float, default=0.1
            The ratio of test data.

        loss_type: str, default="mse"
            The loss type to use. Either "mse" or "bce".

        n_epoch: int, default=1000
            The number of epochs to train.

        n_steps_per_epoch: int, default=10
            The number of steps per epoch.

        n_epochs_per_log: int, default=10
            The number of epochs per log.

        patience: int, default=5
            The number of epochs to wait before early stopping.

        n_candidate_action: int, default=10
            The number of candidate actions to use for online policy evaluation.

        n_candidate_per_model: List[int], default=None
            The number of candidate actions to use for each model.

        batch_size: int, default=128
            The batch size.

        make_copy: bool, default=False
            Whether to make a copy of the base model.

        return_training_logs: bool, default=False
            Whether to return the training logs.

        is_deterministic_early_stage_eval: bool, default=True
            Whether to use deterministic early stage policy for evaluation.

        is_deterministic_late_stage_eval: bool, default=True
            Whether to use deterministic late stage policy for evaluation.

        save_path: Optional[Path]
            The path to save the model.

        random_seed: Optional[int]
            The random seed to use.

        """
        if self.is_model_free_late_stage_policy:
            raise ValueError("Cannot train model-free late stage policy.")

        if random_seed is None:
            random_seed = self.random_seed

        self.seed(random_seed)

        if make_copy:
            late_stage_policy = deepcopy(self.late_stage_policy)
        else:
            late_stage_policy = self.late_stage_policy

        optimizer = self.late_stage_optimizer(
            late_stage_policy.base_model.parameters(),
            **self.late_stage_optimizer_kwargs,
        )

        val_size = int(val_ratio * len(dataset))
        test_size = int(test_ratio * len(dataset))
        train_size = len(dataset) - val_size - test_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)

        train_losses = torch.zeros((n_epoch + 1,), device=self.device)
        val_losses = torch.zeros((n_epoch // n_epochs_per_log + 1,), device=self.device)
        policy_values = torch.zeros(
            (n_epoch // n_epochs_per_log + 1,), device=self.device
        )

        loss_fn = nn.MSELoss() if loss_type == "mse" else nn.BCELoss()

        with tqdm(range(n_epoch)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[Train base late stage model: Epoch {i}]")
                pbar.set_postfix(
                    {
                        "train_loss": f"{train_losses[i]:.4g}",
                        "val_loss": f"{val_losses[i // n_epochs_per_log]:.4g}",
                        "policy_value": f"{policy_values[i // n_epochs_per_log]:.4g}",
                    },
                )

                train_iterator = iter(train_loader)

                for j in range(n_steps_per_epoch):
                    try:
                        batch_ = next(train_iterator)
                    except StopIteration:
                        train_iterator = iter(train_loader)
                        batch_ = next(train_iterator)

                    context_ = batch_["context"].to(self.device, non_blocking=True)
                    latent_ = batch_["latent"].to(self.device, non_blocking=True)
                    context_id_ = batch_["context_id"].to(
                        self.device, non_blocking=True
                    )
                    latent_id_ = batch_["latent_id"].to(self.device, non_blocking=True)
                    action_ids_ = batch_["action"].to(self.device, non_blocking=True)
                    rewards_ = batch_["reward"].to(self.device, non_blocking=True)

                    preds_ = late_stage_policy.base_model(
                        context=context_,
                        latent=latent_,
                        context_id=context_id_,
                        latent_id=latent_id_,
                        action_ids=action_ids_,
                    )
                    loss_ = loss_fn(preds_, rewards_)

                    optimizer.zero_grad()
                    loss_.backward()
                    optimizer.step()

                    train_losses[i + 1] += loss_ / n_steps_per_epoch

                if (i + 1) % n_epochs_per_log == 0:
                    with torch.no_grad():
                        for batch_ in val_loader:
                            context_ = batch_["context"].to(
                                self.device, non_blocking=True
                            )
                            latent_ = batch_["latent"].to(
                                self.device, non_blocking=True
                            )
                            context_id_ = batch_["context_id"].to(
                                self.device, non_blocking=True
                            )
                            latent_id_ = batch_["latent_id"].to(
                                self.device, non_blocking=True
                            )
                            action_ids_ = batch_["action"].to(
                                self.device, non_blocking=True
                            )
                            rewards_ = batch_["reward"].to(
                                self.device, non_blocking=True
                            )

                            preds_ = late_stage_policy.base_model(
                                context=context_,
                                latent=latent_,
                                context_id=context_id_,
                                latent_id=latent_id_,
                                action_ids=action_ids_,
                            )
                            val_loss_ = loss_fn(preds_, rewards_)
                            val_losses[(i + 1) // n_epochs_per_log] += (
                                val_loss_ * len(batch_["context_id"]) / len(val_dataset)
                            )

                    if self.env is not None:
                        policy_value_ = self.env.evaluate_policy_online(
                            early_stage_policy=self.early_stage_policy,
                            late_stage_policy=late_stage_policy,
                            is_deterministic_early_stage=is_deterministic_early_stage_eval,
                            is_deterministic_late_stage=is_deterministic_late_stage_eval,
                            n_candidate_action=n_candidate_action,
                            n_candidate_per_model=n_candidate_per_model,
                        )
                        policy_values[(i + 1) // n_epochs_per_log] += (
                            policy_value_.item()
                        )

                    # early stopping
                    current_epoch = (i + 1) // n_epochs_per_log
                    best_loss_epoch = torch.argmin(
                        val_losses[1 : current_epoch + 1]
                    ).item()

                    if current_epoch - best_loss_epoch > patience:
                        print("early stopping at epoch", i + 1)
                        break

        self.trained_late_stage_policy = late_stage_policy

        if save_path is not None:
            self.save_late_stage_model(save_path)

        if return_training_logs:
            output = (
                late_stage_policy,
                {
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "policy_values": policy_values,
                },
            )
        else:
            output = late_stage_policy

        return output

    def train_early_and_late_stage_policies_offline(
        self,
        dataset: LoggedDataset,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        loss_type: str = "mse",
        n_epoch: int = 1000,
        n_steps_per_epoch: int = 10,
        n_epochs_per_log: int = 10,
        patience: int = 5,
        n_candidate_action: int = 10,
        n_actions_per_model: Optional[List[int]] = None,
        batch_size: int = 128,
        make_copy: bool = False,
        return_training_logs: bool = False,
        is_deterministic_early_stage_eval: bool = True,
        is_deterministic_late_stage_eval: bool = True,
        save_path_early_stage: Optional[Path] = None,
        save_path_late_stage: Optional[Path] = None,
        random_seed: Optional[int] = None,
    ):
        """Train the base model used in the late stage policy.

        Input
        ------
        dataset: LoggedDataset
            The training data, which contains the following keys:

            .. code-block:: python

                key: [
                    "context",     # (n_samples, dim_context)
                    "latent",      # (n_samples, dim_hidden, dim_action_emb)
                    "action",      # (n_samples, n_output_action)
                    "reward",      # (n_samples, n_output_action)
                    "agg_reward",  # (n_samples, 1)
                    "context_id",  # (n_samples, ), optional
                    "latent_id",   # (n_samples, ), optional
                    ]

        val_ratio: float, default=0.1
            The ratio of validation data.

        test_ratio: float, default=0.1
            The ratio of test data.

        loss_type: str, default="mse"
            The loss type to use. Either "mse" or "bce".

        n_epoch: int, default=1000
            The number of epochs to train.

        n_steps_per_epoch: int, default=10
            The number of steps per epoch.

        n_epochs_per_log: int, default=10
            The number of epochs per log.

        patience: int, default=5
            The number of epochs to wait before early stopping.

        n_candidate_action: int, default=10
            The number of candidate actions to use for online policy evaluation.

        n_actions_per_model: List[int], default=None
            The number of candidate actions to use for each model.

        batch_size: int, default=128
            The batch size.

        make_copy: bool, default=False
            Whether to make a copy of the base model.

        return_training_logs: bool, default=False
            Whether to return the training logs.

        is_deterministic_early_stage_eval: bool, default=True
            Whether to use deterministic early stage policy for evaluation.

        is_deterministic_late_stage_eval: bool, default=True
            Whether to use deterministic late stage policy for evaluation.

        save_path_eary_stage: Optional[Path]
            The path to save the early stage model.

        save_path_late_stage: Optional[Path]
            The path to save the late stage model.

        random_seed: Optional[int]
            The random seed to use.

        """
        if self.is_model_free_early_stage_policy:
            raise ValueError("Cannot train model-free early stage policy.")
        if self.is_model_free_late_stage_policy:
            raise ValueError("Cannot train model-free late stage policy.")

        if random_seed is None:
            random_seed = self.random_seed

        self.seed(random_seed)

        if self.model_selector is None:
            if make_copy:
                early_stage_policy = deepcopy(self.early_stage_policy)
                late_stage_policy = deepcopy(self.late_stage_policy)
                model_selector = None
            else:
                early_stage_policy = self.early_stage_policy
                late_stage_policy = self.late_stage_policy
                model_selector = None
        else:
            if make_copy:
                early_stage_policy = deepcopy(self.early_stage_policy)
                late_stage_policy = deepcopy(self.late_stage_policy)
                model_selector = deepcopy(self.model_selector)
            else:
                early_stage_policy = self.early_stage_policy
                late_stage_policy = self.late_stage_policy
                model_selector = self.model_selector

        early_stage_optimizer = self.early_stage_optimizer(
            early_stage_policy.base_model.parameters(),
            **self.early_stage_optimizer_kwargs,
        )
        late_stage_optimizer = self.late_stage_optimizer(
            late_stage_policy.base_model.parameters(),
            **self.late_stage_optimizer_kwargs,
        )
        if model_selector is not None:
            model_selector_optimizer = self.model_selector_optimizer(
                model_selector.parameters(), **self.model_selector_optimizer_kwargs
            )

        val_size = int(val_ratio * len(dataset))
        test_size = int(test_ratio * len(dataset))
        train_size = len(dataset) - val_size - test_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)

        early_stage_train_losses = torch.zeros((n_epoch + 1,), device=self.device)
        late_stage_train_losses = torch.zeros((n_epoch + 1,), device=self.device)
        early_stage_val_losses = torch.zeros(
            (n_epoch // n_epochs_per_log + 1,), device=self.device
        )
        late_stage_val_losses = torch.zeros(
            (n_epoch // n_epochs_per_log + 1,), device=self.device
        )
        policy_values = torch.zeros(
            (n_epoch // n_epochs_per_log + 1,), device=self.device
        )

        early_stage_loss_fn = self._MSELoss if loss_type == "mse" else self._BSELoss
        late_stage_loss_fn = nn.MSELoss() if loss_type == "mse" else nn.BCELoss()

        early_stage_early_stopping_flg = False
        late_stage_early_stopping_flg = False

        with tqdm(range(n_epoch)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(
                    f"[Train base early and late stage models: Epoch {i}]"
                )
                pbar.set_postfix(
                    {
                        "early_stage_train_loss": f"{early_stage_train_losses[i]:.4g}",
                        "late_stage_train_loss": f"{late_stage_train_losses[i]:.4g}",
                        "early_stage_val_loss": f"{early_stage_val_losses[i // n_epochs_per_log]:.4g}",
                        "late_stage_val_loss": f"{late_stage_val_losses[i // n_epochs_per_log]:.4g}",
                        "policy_value": f"{policy_values[i // n_epochs_per_log]:.4g}",
                    },
                )

                train_iterator = iter(train_loader)

                for j in range(n_steps_per_epoch):
                    try:
                        batch_ = next(train_iterator)
                    except StopIteration:
                        train_iterator = iter(train_loader)
                        batch_ = next(train_iterator)

                    context_ = batch_["context"].to(self.device, non_blocking=True)
                    latent_ = batch_["latent"].to(self.device, non_blocking=True)
                    context_id_ = batch_["context_id"].to(
                        self.device, non_blocking=True
                    )
                    latent_id_ = batch_["latent_id"].to(self.device, non_blocking=True)
                    action_ids_ = batch_["action"].to(self.device, non_blocking=True)
                    rewards_ = batch_["reward"].to(self.device, non_blocking=True)

                    if not early_stage_early_stopping_flg:
                        if model_selector is None:
                            early_stage_preds_ = early_stage_policy.base_model(
                                context=context_,
                                context_id=context_id_,
                                action_ids=action_ids_,
                            )
                            early_stage_loss_ = early_stage_loss_fn(
                                early_stage_preds_, rewards_
                            )

                            early_stage_optimizer.zero_grad()
                            early_stage_loss_.backward()
                            early_stage_optimizer.step()

                        else:
                            # freeze model selector and unfreeze base model
                            early_stage_preds_ = early_stage_policy.base_model(
                                context=context_,
                                context_id=context_id_,
                                action_ids=action_ids_,
                            )

                            with torch.no_grad():
                                weights_ = model_selector(
                                    latent=latent_, latent_id=latent_id_
                                )

                            early_stage_loss_ = early_stage_loss_fn(
                                early_stage_preds_, rewards_, weights_
                            )

                            early_stage_optimizer.zero_grad()
                            early_stage_loss_.backward()
                            early_stage_optimizer.step()

                            # freeze base model and unfreeze model selector
                            with torch.no_grad():
                                early_stage_preds_ = early_stage_policy.base_model(
                                    context=context_,
                                    context_id=context_id_,
                                    action_ids=action_ids_,
                                )

                            weights_ = model_selector(
                                latent=latent_, latent_id=latent_id_
                            )
                            early_stage_loss_ = early_stage_loss_fn(
                                early_stage_preds_, rewards_, weights_
                            )

                            model_selector_optimizer.zero_grad()
                            early_stage_loss_.backward()
                            model_selector_optimizer.step()

                        early_stage_train_losses[i + 1] += (
                            early_stage_loss_ / n_steps_per_epoch
                        )

                    if not late_stage_early_stopping_flg:
                        late_stage_preds_ = late_stage_policy.base_model(
                            context=context_,
                            context_id=context_id_,
                            latent=latent_,
                            latent_id=latent_id_,
                            action_ids=action_ids_,
                        )

                        late_stage_loss_ = late_stage_loss_fn(
                            late_stage_preds_, rewards_
                        )
                        late_stage_optimizer.zero_grad()
                        late_stage_loss_.backward()
                        late_stage_optimizer.step()

                        late_stage_train_losses[i + 1] += (
                            late_stage_loss_ / n_steps_per_epoch
                        )

                if (i + 1) % n_epochs_per_log == 0:
                    with torch.no_grad():
                        for batch_ in val_loader:
                            context_ = batch_["context"].to(
                                self.device, non_blocking=True
                            )
                            latent_ = batch_["latent"].to(
                                self.device, non_blocking=True
                            )
                            context_id_ = batch_["context_id"].to(
                                self.device, non_blocking=True
                            )
                            latent_id_ = batch_["latent_id"].to(
                                self.device, non_blocking=True
                            )
                            action_ids_ = batch_["action"].to(
                                self.device, non_blocking=True
                            )
                            rewards_ = batch_["reward"].to(
                                self.device, non_blocking=True
                            )

                            if not early_stage_early_stopping_flg:
                                early_stage_preds_ = early_stage_policy.base_model(
                                    context=context_,
                                    context_id=context_id_,
                                    action_ids=action_ids_,
                                )

                                if model_selector is None:
                                    early_stage_val_loss_ = early_stage_loss_fn(
                                        early_stage_preds_, rewards_
                                    )
                                else:
                                    latent_id_ = batch_["latent_id"].to(self.device)
                                    weights_ = model_selector(
                                        latent=latent_, latent_id=latent_id_
                                    )
                                    early_stage_val_loss_ = early_stage_loss_fn(
                                        early_stage_preds_, rewards_, weights_
                                    )

                                early_stage_val_losses[(i + 1) // n_epochs_per_log] += (
                                    early_stage_val_loss_
                                    * len(batch_["context_id"])
                                    / len(val_dataset)
                                )

                            if not late_stage_early_stopping_flg:
                                late_stage_preds_ = late_stage_policy.base_model(
                                    context=context_,
                                    context_id=context_id_,
                                    latent=latent_,
                                    latent_id=latent_id_,
                                    action_ids=action_ids_,
                                )
                                late_stage_loss_val_ = late_stage_loss_fn(
                                    late_stage_preds_, rewards_
                                )
                                late_stage_val_losses[(i + 1) // n_epochs_per_log] += (
                                    late_stage_loss_val_
                                    * len(batch_["context_id"])
                                    / len(val_dataset)
                                )

                    if self.env is not None:
                        policy_value_ = self.env.evaluate_policy_online(
                            early_stage_policy=early_stage_policy,
                            late_stage_policy=late_stage_policy,
                            is_deterministic_early_stage=is_deterministic_early_stage_eval,
                            is_deterministic_late_stage=is_deterministic_late_stage_eval,
                            n_candidate_action=n_candidate_action,
                            n_candidate_per_model=n_actions_per_model,
                        )
                        policy_values[(i + 1) // n_epochs_per_log] += (
                            policy_value_.item()
                        )

                    # early stopping
                    current_epoch = (i + 1) // n_epochs_per_log
                    early_stage_best_loss_epoch = torch.argmin(
                        early_stage_val_losses[1 : current_epoch + 1]
                    ).item()
                    late_stage_best_loss_epoch = torch.argmin(
                        late_stage_val_losses[1 : current_epoch + 1]
                    ).item()

                    if (
                        current_epoch - early_stage_best_loss_epoch
                        > patience
                        and not early_stage_early_stopping_flg
                    ):
                        early_stage_early_stopping_flg = True
                        print(
                            "early stopping for the early stage policy at epoch", i + 1
                        )

                    if (
                        current_epoch - late_stage_best_loss_epoch
                        > patience
                        and not late_stage_early_stopping_flg
                    ):
                        late_stage_early_stopping_flg = True
                        print(
                            "early stopping for the late stage policy at epoch", i + 1
                        )

                    if early_stage_early_stopping_flg and late_stage_early_stopping_flg:
                        break

        self.trained_early_stage_policy = early_stage_policy
        self.trained_late_stage_policy = late_stage_policy

        if save_path_early_stage is not None:
            self.save_early_stage_model(save_path_early_stage)
        if save_path_late_stage is not None:
            self.save_late_stage_model(save_path_late_stage)

        if return_training_logs:
            output = (
                early_stage_policy,
                late_stage_policy,
                model_selector,
                {
                    "early_stage_train_losses": early_stage_train_losses,
                    "late_stage_train_losses": late_stage_train_losses,
                    "early_stage_val_losses": early_stage_val_losses,
                    "late_stage_val_losses": late_stage_val_losses,
                    "policy_values": policy_values,
                },
            )
        else:
            output = (early_stage_policy, late_stage_policy, model_selector)

        return output
