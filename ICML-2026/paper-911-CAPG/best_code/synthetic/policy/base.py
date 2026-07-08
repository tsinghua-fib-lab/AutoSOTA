# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Base class for the early stage retrieval and late stage ranking policies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class BaseEarlyStagePolicy(ABC):
    """Base class for the early stage retrieval policy.

    Input
    ------
    base_model: nn.Module
        The base model.

    action_set: Optional[BaseActionSet]
        The action set.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    @abstractmethod
    def retrieve_topk(
        self,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        n_candidate_action: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve top-k candidate actions.

        Input
        ------
        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            The context to retrieve actions from. Either context or context_id must be provided.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            The context ids to retrieve actions from. Either context or context_id must be provided.

        n_candidate_action: int
            The number of candidate actions to retrieve.

        Output
        ------
        actions: Tensor, shape (n_samples, n_candidate_action)
            The retrieved actions.

        """
        raise NotImplementedError()

    @abstractmethod
    def sample(
        self,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        n_candidate_action: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Sample k-candidate actions.

        Input
        ------
        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            The context to sample actions from. Either context or context_id must be provided.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            The context ids to sample actions from. Either context or context_id must be provided.

        n_candidate_action: int
            The number of candidate actions to sample.

        Output
        ------
        actions: Tensor, shape (n_samples, n_candidate_action)
            The sampled actions.

        """
        raise NotImplementedError()

    @abstractmethod
    def calc_prob_given_actions(
        self,
        actions: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the probability of the given actions.

        Input
        ------
        actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The actions to calculate the probability.

        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            The context to calculate the probability of the given actions. Either context or context_id must be provided.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            The context ids to calculate the probability of the given actions. Either context or context_id must be provided.

        Output
        ------
        prob: torch.Tensor, shape (n_samples, )
            The joint probability of the given actions (non-differential).

        log_prob: torch.Tensor, shape (n_samples, )
            The log joint probability of the given actions (differential).

        """
        raise NotImplementedError()


@dataclass
class BaseLateStagePolicy(ABC):
    """Base class for the late stage generation policy.

    Input
    ------
    base_model: nn.Module
        The base model.

    action_set: BaseActionSet
        The action set.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    @abstractmethod
    def retrieve_topk(
        self,
        candidate_actions: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        n_output_action: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve top-k candidate actions.

        Input
        ------
        candidate_actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The candidate actions to retrieve from.

        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            The context to retrieve actions from. Either context or context_id must be provided.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            The context ids to retrieve actions from. Either context or context_id must be provided.

        latent: Optional[torch.Tensor], shape (n_samples, dim_hidden, dim_action_emb)
            The latent to retrieve actions from. Either latent or latent_id must be provided.

        latent_id: Optional[torch.Tensor], shape (n_samples, )
            The latent ids to retrieve actions from. Either latent or latent_id must be provided.

        n_output_action: int
            The number of output actions to retrieve.

        Output
        ------
        actions: Tensor, shape (n_samples, n_output_action)
            The retrieved actions.

        """
        raise NotImplementedError()

    @abstractmethod
    def sample(
        self,
        candidate_actions: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        n_output_action: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Sample k-candidate actions.

        Input
        ------
        candidate_actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The candidate actions to sample from.

        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            The context to sample actions from. Either context or context_id must be provided.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            The context ids to sample actions from. Either context or context_id must be provided.

        latent: Optional[torch.Tensor], shape (n_samples, dim_hidden, dim_action_emb)
            The latent to sample actions from. Either latent or latent_id must be provided.

        latent_id: Optional[torch.Tensor], shape (n_samples, )
            The latent ids to sample actions from. Either latent or latent_id must be provided.

        n_output_action: int
            The number of output actions to sample.

        Output
        ------
        actions: Tensor, shape (n_samples, n_output_action)
            The sampled actions.

        """
        raise NotImplementedError()


@dataclass
class BaseJointPolicy(ABC):
    """Base class for the joint (early stage + late stage) policy.

    Input
    ------
    early_stage_policy: BaseEarlyStagePolicy
        The early stage policy.

    late_stage_policy: BaseLateStagePolicy
        The late stage policy.

    action_prob_regressor: nn.Module
        The regression model for estimating the (joint) action choice probability.

    action_set: BaseActionSet
        The action set.

    n_candidate_action: int
        The number of candidate actions.

    n_output_action: int
        The number of output actions.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    @abstractmethod
    def retrieve_topk(
        self,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve top-k candidate actions.

        Input
        ------
        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            The context to retrieve actions from. Either context or context_id must be provided.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            The context ids to retrieve actions from. Either context or context_id must be provided.

        latent: Optional[torch.Tensor], shape (n_samples, dim_hidden, dim_action_emb)
            The latent to retrieve actions from. Either latent or latent_id must be provided.

        latent_id: Optional[torch.Tensor], shape (n_samples, )
            The latent ids to retrieve actions from. Either latent or latent_id must be provided.

        Output
        ------
        actions: Tensor, shape (n_samples, n_candidate_action)
            The retrieved actions.

        """
        raise NotImplementedError()

    @abstractmethod
    def sample(
        self,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Sample k-candidate actions.

        Input
        ------
        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            The context to sample actions from. Either context or context_id must be provided.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            The context ids to sample actions from. Either context or context_id must be provided.

        latent: Optional[torch.Tensor], shape (n_samples, dim_hidden, dim_action_emb)
            The latent to sample actions from. Either latent or latent_id must be provided.

        latent_id: Optional[torch.Tensor], shape (n_samples, )
            The latent ids to sample actions from. Either latent or latent_id must be provided.

        Output
        ------
        actions: Tensor, shape (n_samples, n_candidate_action)
            The sampled actions.

        """
        raise NotImplementedError()


@dataclass
class BaseSingleStagePolicy(ABC):
    """Base class for the single stage policy.

    Input
    ------
    base_model: nn.Module
        The base model.

    action_set: BaseActionSet
        The action set.

    n_output_action: int
        The number of output actions.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    @abstractmethod
    def retrieve_topk(
        self,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve top-k candidate actions.

        Input
        ------
        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            The context to retrieve actions from. Either context or context_id must be provided.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            The context ids to retrieve actions from. Either context or context_id must be provided.

        latent: Optional[torch.Tensor], shape (n_samples, dim_hidden, dim_action_emb)
            The latent to retrieve actions from. Either latent or latent_id must be provided.

        latent_id: Optional[torch.Tensor], shape (n_samples, )
            The latent ids to retrieve actions from. Either latent or latent_id must be provided.

        Output
        ------
        actions: Tensor, shape (n_samples, n_candidate_action)
            The retrieved actions.

        """
        raise NotImplementedError()

    @abstractmethod
    def sample(
        self,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Sample k-candidate actions.

        Input
        ------
        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            The context to sample actions from. Either context or context_id must be provided.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            The context ids to sample actions from. Either context or context_id must be provided.

        latent: Optional[torch.Tensor], shape (n_samples, dim_hidden, dim_action_emb)
            The latent to sample actions from. Either latent or latent_id must be provided.

        latent_id: Optional[torch.Tensor], shape (n_samples, )
            The latent ids to sample actions from. Either latent or latent_id must be provided.

        Output
        ------
        actions: Tensor, shape (n_samples, n_candidate_action)
            The sampled actions.

        """
        raise NotImplementedError()
