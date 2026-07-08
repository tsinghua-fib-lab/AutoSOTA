# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Base class for data generation functions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from src.policy import (
    BaseEarlyStagePolicy,
    BaseLateStagePolicy,
)

from .dataset import LoggedDataset


@dataclass
class BaseDataGenerator(ABC):
    """Class for the synthetic data generation.

    Input
    ------
    context_sampler: ContextSampler
        The context sampler object.

    latent_sampler: LatentSampler
        The latent sampler object.

    reward_model: RewardModel
        The reward model object.

    """

    @abstractmethod
    def sample_dataset(
        self,
        early_stage_policy: BaseEarlyStagePolicy,
        late_stage_policy: BaseLateStagePolicy,
        is_deterministic_early_stage: bool = False,
        is_deterministic_late_stage: bool = False,
        n_samples: int = 10000,
    ) -> LoggedDataset:
        """Function to sample the dataset.

        Input
        ------
        early_stage_policy: EarlyStagePolicy
            The early stage policy object.

        late_stage_policy: LateStagePolicy
            The late stage policy object.

        is_deterministic_early_stage: bool, default=False
            Whether the early stage policy is deterministic.

        is_deterministic_late_stage: bool, default=False
            Whether the late stage policy is deterministic.

        n_samples: int, default=10000
            The number of samples to generate.

        Output
        ------
        logged_dataset: LoggedDataset
            The logged dataset object.

        """
        raise NotImplementedError()

    @abstractmethod
    def evaluate_policy_online(
        self,
        early_stage_policy: BaseEarlyStagePolicy,
        late_stage_policy: BaseLateStagePolicy,
        is_deterministic_early_stage: bool = False,
        is_deterministic_late_stage: bool = False,
        n_samples: int = 10000,
    ) -> torch.Tensor:
        """Function to evaluate the policy via online rollouts.

        Input
        ------
        early_stage_policy: EarlyStagePolicy
            The early stage policy object.

        late_stage_policy: LateStagePolicy
            The late stage policy object.

        is_deterministic_early_stage: bool, default=False
            Whether the early stage policy is deterministic.

        is_deterministic_late_stage: bool, default=False
            Whether the late stage policy is deterministic.

        n_samples: int, default=10000
            The number of samples to generate.

        Output
        ------
        logged_dataset: LoggedDataset
            The logged dataset object.

        """
        raise NotImplementedError()


@dataclass
class BaseContextSampler(ABC):
    """Base class for context sampler.

    Input
    ------
    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    @abstractmethod
    def sample(
        self,
        n_samples: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample context.

        Input
        ------
        n_samples: int
            The number of samples to generate.

        Output
        ------
        context: Tensor, shape (n_samples, dim_context)
            The sampled context.

        context_ids: Optional[torch.Tensor], shape (n_samples, )
            The sampled context ids. None if the context is not discrete.

        """
        raise NotImplementedError()


@dataclass
class BaseLatentSampler(ABC):
    """Base class for latent sampler.

    Input
    ------
    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    @abstractmethod
    def sample(
        self,
        n_samples: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample latent features.

        Input
        ------
        n_samples: int
            The number of samples to generate.

        Output
        ------
        latent: Tensor, shape (n_samples, dim_action_emb, dim_action_emb)
            The sampled latent features.

        latent_ids: Optional[torch.Tensor], shape (n_samples, )
            The sampled latent features ids. None if the latent features are not discrete.

        """
        raise NotImplementedError()


@dataclass
class BaseRewardModel(ABC):
    """Base class for reward model.

    Input
    ------
    action_set: ActionSet
        The action set object.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    @abstractmethod
    def expected(
        self,
        context: torch.Tensor,
        latent: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate expected reward.

        Input
        ------
        context: Tensor, shape (n_samples, dim_context)
            The context.

        latent: Tensor, shape (n_samples, dim_action_emb, dim_action_emb)
            The latent features.

        action: Tensor, shape (n_samples, n_output_actions)
            The actions.

        Output
        ------
        expected_reward: Tensor, shape (n_samples, n_output_actions)
            The expected reward.

        agg_expected_reward: Tensor, shape (n_samples, 1)
            The expected reward aggregated across ranking.

        """
        raise NotImplementedError()

    @abstractmethod
    def sample(
        self,
        context: torch.Tensor,
        latent: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample reward.

        Input
        ------
        context: Tensor, shape (n_samples, dim_context)
            The context.

        latent: Tensor, shape (n_samples, dim_action_emb, dim_action_emb)
            The latent features.

        action: Tensor, shape (n_samples, n_output_actions)
            The actions.

        Output
        ------
        reward: Tensor, shape (n_samples, n_output_actions)
            The sampled reward.

        agg_reward: Tensor, shape (n_samples, 1)
            The sampled reward aggregated across ranking.

        """
        raise NotImplementedError()
