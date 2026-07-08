# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Implementation of the uniform early stage retrieval and late stage ranking policies."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .action_set import BaseActionSet
from .base import BaseEarlyStagePolicy, BaseLateStagePolicy, BaseSingleStagePolicy


@dataclass
class UniformEarlyStagePolicy(BaseEarlyStagePolicy):
    """Implementation of the uniform early stage retrieval policy.

    Input
    ------
    base_model: Optional[nn.Module]
        API consistency.

    action_set: Optional[BaseActionSet]
        The action set.

    n_action: Optional[int] = None
        The number of actions.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    base_model: Optional[nn.Module] = None
    action_set: Optional[BaseActionSet] = None
    n_action: Optional[int] = None
    device: Optional[torch.device] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)

        if self.action_set is None:
            if self.n_action is None:
                raise ValueError("n_action must be provided.")
        else:
            if self.n_action is not None:
                assert self.n_action == self.action_set.n_action

            self.n_action = self.action_set.n_action

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
            API consistency.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            The context ids to retrieve actions from. Either context or context_id must be provided.

        n_candidate_action: int
            The number of candidate actions to retrieve.

        Output
        ------
        actions: Tensor, shape (n_samples, n_candidate_action)
            The retrieved actions.

        """
        n_samples = (
            context.shape[0]
            if context is not None
            else context_id.shape[0]
        )
        topk = torch.arange(n_candidate_action, device=self.device).expand(
            n_samples, n_candidate_action
        )
        return topk

    def sample(
        self,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        n_candidate_action: int = 1,
        is_deterministic: bool = False,
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

        is_deterministic: bool
            Whether to sample deterministically.

        Output
        ------
        actions: Tensor, shape (n_samples, n_candidate_action)
            The sampled actions.

        """
        if is_deterministic:
            return self.retrieve_topk(
                context=context,
                context_id=context_id,
                n_candidate_action=n_candidate_action,
            )
        else:
            n_samples = (
                context.shape[0]
                if context is not None
                else context_id.shape[0]
            )
            actions = torch.randint(
                self.n_action,
                (n_samples, n_candidate_action),
                device=self.device,
            )
        return actions

    def calc_prob_given_actions(
        self,
        actions: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        is_deterministic: bool = False,
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

        is_deterministic: bool
            Whether the actions are sampled deterministically.

        Output
        ------
        prob: torch.Tensor, shape (n_samples, )
            The joint probability of the given actions (non-differential).

        log_prob: torch.Tensor, shape (n_samples, )
            The log joint probability of the given actions (non-differential).

        """
        raise NotImplementedError()


@dataclass
class UniformLateStagePolicy(BaseLateStagePolicy):
    """Implementation of the uniform late stage retrieval policy.

    Input
    ------
    base_model: Optional[nn.Module]
        The base model.

    action_set: Optional[BaseActionSet]
        The action set.

    n_action: Optional[int] = None
        The number of actions.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    base_model: Optional[nn.Module] = None
    action_set: Optional[BaseActionSet] = None
    n_action: Optional[int] = None
    device: Optional[torch.device] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)

        if self.action_set is None:
            if self.n_action is None:
                raise ValueError("n_action must be provided.")
        else:
            if self.n_action is not None:
                assert self.n_action == self.action_set.n_action
            self.n_action = self.action_set.n_action

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
        n_samples = (
            context.shape[0]
            if context is not None
            else context_id.shape[0]
        )
        within_candidate_idx = torch.arange(n_output_action, device=self.device).expand(
            n_samples, n_output_action
        )
        topk = torch.gather(candidate_actions, 1, within_candidate_idx)
        return topk

    def sample(
        self,
        candidate_actions: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        n_output_action: int = 1,
        is_deterministic: bool = False,
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

        is_deterministic: bool
            Whether to sample deterministically.

        Output
        ------
        actions: Tensor, shape (n_samples, n_output_action)
            The sampled actions.

        """
        if is_deterministic:
            actions = self.retrieve_topk(
                candidate_actions=candidate_actions,
                context=context,
                context_id=context_id,
                latent=latent,
                latent_id=latent_id,
                n_output_action=n_output_action,
            )
        else:
            n_samples, n_candidate_action = candidate_actions.shape
            within_candidate_idx = torch.randint(
                n_candidate_action, (n_samples, n_output_action), device=self.device
            )
            actions = torch.gather(candidate_actions, 1, within_candidate_idx)

        return actions


@dataclass
class UniformSingleStagePolicy(BaseSingleStagePolicy):
    """Implementation of the uniform single stage retrieval policy.

    Input
    ------
    base_model: Optional[nn.Module]
        The base model. API consistency.

    action_set: Optional[BaseActionSet]
        The action set.

    n_action: Optional[int] = None
        The number of actions.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    base_model: Optional[nn.Module] = None
    action_set: Optional[BaseActionSet] = None
    n_action: Optional[int] = None
    device: Optional[torch.device] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)

        if self.action_set is None:
            if self.n_action is None:
                raise ValueError("n_action must be provided.")
        else:
            if self.n_action is not None:
                assert self.n_action == self.action_set.n_action
            self.n_action = self.action_set.n_action

    def retrieve_topk(
        self,
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
        n_samples = (
            context.shape[0]
            if context is not None
            else context_id.shape[0]
        )
        actions = torch.arange(n_output_action, device=self.device).expand(
            n_samples, n_output_action
        )
        return actions

    def sample(
        self,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        n_output_action: int = 1,
        is_deterministic: bool = False,
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

        n_output_action: int
            The number of output actions to sample.

        is_deterministic: bool
            Whether to sample deterministically.

        Output
        ------
        actions: Tensor, shape (n_samples, n_output_action)
            The sampled actions.

        """
        if is_deterministic:
            actions = self.retrieve_topk(
                context=context,
                context_id=context_id,
                latent=latent,
                latent_id=latent_id,
                n_output_action=n_output_action,
            )
        else:
            n_samples = (
                context.shape[0]
                if context is not None
                else context_id.shape[0]
            )
            actions = torch.randint(
                self.n_action,
                (n_samples, n_output_action),
                device=self.device,
            )

        return actions
