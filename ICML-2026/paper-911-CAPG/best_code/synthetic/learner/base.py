# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Base class of the learner module."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from synthetic.dataset import (
    LoggedDataset,
)


@dataclass
class BasePolicyLearner(ABC):
    """Base class of the policy learner module.

    Input
    ------
    early_stage_policy: BaseEarlyStagePolicy
        The early stage policy.

    late_stage_policy: BaseLateStagePolicy
        The late stage policy.

    device: torch.device, default=torch.device("cpu")
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    @abstractmethod
    def train_early_stage_policy_offline(
        self,
        dataset: LoggedDataset,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        n_epoch: int = 1000,
        n_steps_per_epoch: int = 10,
        n_epochs_per_log: int = 10,
        patience: int = 5,
        batch_size: int = 128,
        make_copy: bool = False,
        return_training_logs: bool = False,
        save_path: Optional[Path] = None,
        random_seed: Optional[int] = None,
        **kwargs,
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

        save_path: Optional[Path]
            The path to save the model.

        random_seed: Optional[int]
            The random seed to use.

        """
        raise NotImplementedError()

    def load_early_stage_model(self, path: Path, is_init: bool = False):
        """Load the early stage model."""
        if is_init:
            self.early_stage_policy.base_model.load_state_dict(
                torch.load(path, map_location=self.device)
            )
            model = self.early_stage_policy.base_model
        else:
            self.trained_early_stage_policy.base_model.load_state_dict(
                torch.load(path, map_location=self.device)
            )
            model = self.trained_early_stage_policy.base_model

        return model

    def load_late_stage_model(self, path: Path, is_init: bool = False):
        """Load the late stage model."""
        if is_init:
            self.late_stage_policy.base_model.load_state_dict(
                torch.load(path, map_location=self.device)
            )
            model = self.late_stage_policy.base_model
        else:
            self.trained_late_stage_policy.base_model.load_state_dict(
                torch.load(path, map_location=self.device)
            )
            model = self.trained_late_stage_policy.base_model

        return model

    def save_early_stage_model(self, path: Path):
        """Save the early stage model."""
        torch.save(
            self.trained_early_stage_policy.base_model.state_dict(),
            path,
        )

    def save_late_stage_model(self, path: Path):
        """Save the early stage model."""
        torch.save(
            self.trained_late_stage_policy.base_model.state_dict(),
            path,
        )

    def seed(self, seed: int):
        """Set the random seed."""
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)


@dataclass
class BaseModelLearner(ABC):
    """Base class of the model learner module.

    Input
    ------
    model: nn.Module
        The model to be trained.

    device: torch.device, default=torch.device("cpu")
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    @abstractmethod
    def train_model_offline(
        self,
        dataset: LoggedDataset,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        n_epoch: int = 1000,
        n_steps_per_epoch: int = 10,
        n_epochs_per_log: int = 10,
        patience: int = 5,
        batch_size: int = 128,
        make_copy: bool = False,
        return_training_logs: bool = False,
        save_path: Optional[Path] = None,
        random_seed: Optional[int] = None,
    ):
        """Train the model.

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

        save_path: Optional[Path]
            The path to save the model.

        random_seed: Optional[int]
            The random seed to use.

        """
        raise NotImplementedError()

    def load_model(self, path: Path, is_init: bool = False):
        """Load the model."""
        if is_init:
            self.model.load_state_dict(
                torch.load(path, map_location=self.device)
            )
            model = self.model
        else:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            model = self.model

        return model

    def save_model(self, path: Path):
        """Save the model."""
        torch.save(self.model.state_dict(), path)

    def seed(self, seed: int):
        """Set the random seed."""
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
