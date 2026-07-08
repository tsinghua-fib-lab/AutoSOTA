# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Implementation of the optimal early and late stage policies."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .action_set import BaseActionSet
from .base import BaseEarlyStagePolicy, BaseLateStagePolicy, BaseSingleStagePolicy


@dataclass
class OptimalEarlyStagePolicy(BaseEarlyStagePolicy):
    """Implementation of the optimal early stage retrieval policy.

    Input
    ------
    base_model: Optional[nn.Module]
        API consistency.

    action_set: Optional[BaseActionSet]
        The action set.

    n_action: Optional[int]
        The number of actions.

    is_anti_optimal: bool, default=False
        Whether the policy is anti optimal.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    base_model: Optional[nn.Module] = None
    action_set: Optional[BaseActionSet] = None
    n_action: Optional[int] = None
    is_anti_optimal: bool = False
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
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        n_candidate_action: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve top-k candidate actions.

        Input
        ------
        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            API consistency.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        n_candidate_action: int
            The number of candidate actions to retrieve.

        Output
        ------
        actions: Tensor, shape (n_samples, n_candidate_action)
            The retrieved actions.

        """
        if self.is_anti_optimal:
            _, topk = torch.topk(-factual_rewards, k=n_candidate_action, dim=1)
        else:
            _, topk = torch.topk(factual_rewards, k=n_candidate_action, dim=1)
        return topk

    def sample(
        self,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        n_candidate_action: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Sample k-candidate actions.

        Input
        ------
        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            API consistency.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        n_candidate_action: int
            The number of candidate actions to sample.

        Output
        ------
        actions: Tensor, shape (n_samples, n_candidate_action)
            The sampled actions.

        """
        actions = self.retrieve_topk(
            factual_rewards=factual_rewards,
            n_candidate_action=n_candidate_action,
        )
        return actions

    def calc_prob_given_actions(
        self,
        actions: torch.Tensor,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the probability of the given actions.

        Input
        ------
        actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The actions to calculate the probability.

        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            The context to calculate the probability of the given actions. Either context or context_id must be provided.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            The context ids to calculate the probability of the given actions. Either context or context_id must be provided.

        Output
        ------
        prob: torch.Tensor, shape (n_samples, )
            The joint probability of the given actions (non-differential).

        log_prob: torch.Tensor, shape (n_samples, )
            The log joint probability of the given actions (non-differential).

        """
        raise NotImplementedError()


@dataclass
class OptimalLateStagePolicy(BaseLateStagePolicy):
    """Implementation of the optimal late stage retrieval policy.

    Input
    ------
    base_model: Optional[nn.Module]
        API consistency.

    action_set: Optional[BaseActionSet]
        The action set.

    n_action: Optional[int]
        The number of actions.

    is_anti_optimal: bool, default=False
        Whether the policy is anti-optimal.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    base_model: Optional[nn.Module] = None
    action_set: Optional[BaseActionSet] = None
    n_action: Optional[int] = None
    is_anti_optimal: bool = False
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
        factual_rewards: torch.Tensor,
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

        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            API consistency.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        latent: Optional[torch.Tensor], shape (n_samples, dim_hidden, dim_action_emb)
            API consistency.

        latent_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        n_output_action: int
            The number of output actions to retrieve.

        Output
        ------
        actions: Tensor, shape (n_samples, n_output_action)
            The retrieved actions.

        """
        factual_rewards = torch.gather(factual_rewards, 1, candidate_actions)

        if self.is_anti_optimal:
            _, within_candidate_idx = torch.topk(-factual_rewards, k=n_output_action, dim=1)
        else:
            _, within_candidate_idx = torch.topk(factual_rewards, k=n_output_action, dim=1)
        topk = torch.gather(candidate_actions, 1, within_candidate_idx)
        return topk

    def sample(
        self,
        candidate_actions: torch.Tensor,
        factual_rewards: torch.Tensor,
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

        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            API consistency.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        latent: Optional[torch.Tensor], shape (n_samples, dim_hidden, dim_action_emb)
            API consistency.

        latent_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        n_output_action: int
            The number of output actions to sample.

        Output
        ------
        actions: Tensor, shape (n_samples, n_output_action)
            The sampled actions.

        """
        actions = self.retrieve_topk(
            candidate_actions=candidate_actions,
            factual_rewards=factual_rewards,
            n_output_action=n_output_action,
        )
        return actions

    def calc_prob_given_actions(
        self,
        actions: torch.Tensor,
        candidate_actions: torch.Tensor,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the probability of the given actions.

        Input
        ------
        actions: torch.Tensor, shape (n_samples, n_output_action)
            The actions to calculate the probability.

        candidate_actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The candidate actions to sample from.

        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            API consistency.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        latent: Optional[torch.Tensor], shape (n_samples, dim_hidden, dim_action_emb)
            API consistency.

        latent_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        Output
        ------
        prob: torch.Tensor, shape (n_samples, )
            The joint probability of the given actions (non-differential).

        log_prob: torch.Tensor, shape (n_samples, )
            The log joint probability of the given actions (non-differential).

        """
        raise NotImplementedError()


@dataclass
class OptimalSingleStagePolicy(BaseSingleStagePolicy):
    """Implementation of the optimal single stage retrieval policy.

    Input
    ------
    base_model: Optional[nn.Module]
        The base model.

    action_set: Optional[BaseActionSet]
        The action set.

    n_action: Optional[int]
        The number of actions.

    is_anti_optimal: bool, default=False.
        Whether the policy is anti-optimal.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    base_model: Optional[nn.Module] = None
    action_set: Optional[BaseActionSet] = None
    n_action: Optional[int] = None
    is_anti_optimal: bool = False
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
        factual_rewards: torch.Tensor,
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
        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            API consistency.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        latent: Optional[torch.Tensor], shape (n_samples, dim_hidden, dim_action_emb)
            API consistency.

        latent_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        n_output_action: int
            The number of output actions to retrieve.

        Output
        ------
        actions: Tensor, shape (n_samples, n_output_action)
            The retrieved actions.

        """
        if self.is_anti_optimal:
            _, topk = torch.topk(-factual_rewards, k=n_output_action, dim=1)
        else:
            _, topk = torch.topk(factual_rewards, k=n_output_action, dim=1)
        return topk

    def sample(
        self,
        factual_rewards: torch.Tensor,
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
        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Optional[torch.Tensor], shape (n_samples, dim_context)
            API consistency.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        latent: Optional[torch.Tensor], shape (n_samples, dim_hidden, dim_action_emb)
            API consistency.

        latent_id: Optional[torch.Tensor], shape (n_samples, )
            API consistency.

        n_output_action: int
            The number of output actions to sample.

        Output
        ------
        actions: Tensor, shape (n_samples, n_output_action)
            The sampled actions.

        """
        actions = self.retrieve_topk(
            factual_rewards=factual_rewards,
            n_output_action=n_output_action,
        )
        return actions


@dataclass
class OracleSoftmaxEarlyStagePolicy(BaseEarlyStagePolicy):
    """Early stage retrieval policy.

    Input
    ------
    base_model: Optional[nn.Module]
        API consistency.

    action_set: Optional[BaseActionSet]
        The action set.

    n_action: Optional[int]
        The number of actions.

    inverse_temperature: float, default=1.0
        Inverse temperature parameter of softmax.

    is_anti_optimal: bool, default=False.
        Whether the policy is anti-optimal.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    base_model: Optional[nn.Module] = None
    action_set: Optional[BaseActionSet] = None
    n_action: Optional[int] = None
    inverse_temperature: float = 1.0
    is_anti_optimal: bool = False
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

        self.gumbel_dist = torch.distributions.Gumbel(loc=0.0, scale=1.0)

    def _topk(
        self,
        logits: torch.Tensor,
        n_candidate_action: int = 1,
    ) -> torch.Tensor:
        """Retrieve top-k candidate actions.

        Input
        ------
        logits: torch.Tensor, shape (n_model, n_samples, n_actions)
            The logits of the actions.

        n_candidate_action: int
            The number of candidate actions to retrieve.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The sampled actions.

        """
        logits = logits * self.inverse_temperature

        if self.is_anti_optimal:
            _, topk = torch.topk(-logits, k=n_candidate_action, dim=1)
        else:
            _, topk = torch.topk(logits, k=n_candidate_action, dim=1)

        return topk

    def retrieve_topk(
        self,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        n_candidate_action: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve top-k candidate actions.

        Input
        ------
        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Optional[Tensor], shape (n_samples, dim_context)
            API consistency.

        context_id: Optional[Tensor], shape (n_samples, )
            API consistency.

        n_candidate_action: int
            The number of candidate actions to retrieve.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The retrieved actions.

        """
        topk = self._topk(
            logits=factual_rewards,
            n_candidate_action=n_candidate_action,
        )
        return topk

    def sample(
        self,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        n_candidate_action: int = 1,
        is_deterministic: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Sample k-candidate actions.

        Input
        ------
        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Optional[Tensor], shape (n_samples, dim_context)
            API consistency.

        context_id: Optional[Tensor], shape (n_samples, )
            API consistency.

        n_candidate_action: int
            The number of candidate actions to sample.

        is_deterministic: bool
            Whether to use deterministic sampling.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The sampled actions.

        """
        if is_deterministic:
            gumbel_noise = torch.zeros_like(factual_rewards).to(self.device)
        else:
            gumbel_noise = self.gumbel_dist.sample(
                factual_rewards.shape
            ).to(self.device)

        gumbel_noise = -gumbel_noise if self.is_anti_optimal else gumbel_noise

        actions = self._topk(
            logits=factual_rewards + gumbel_noise.squeeze(0),
            n_candidate_action=n_candidate_action,
        )
        return actions

    def calc_prob_given_actions(
        self,
        actions: torch.Tensor,
        factual_rewards: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        is_deterministic: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the plackett-luce probability given actions.

        Input
        ------
        actions: Tensor, shape (n_samples, n_candidate_action)
            The actions to calculate the probability.

        factual_rewards: Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Tensor, shape (n_samples, dim_context)
            API consistency.

        context_id: Tensor, shape (n_samples, )
            API consistency.

        is_deterministic: bool
            Whether to use deterministic sampling.

        Output
        ------
        prob: torch.Tensor, shape (n_samples, )
            The joint probability of the sampled actions (used for caluculating the importance weight, non-differential).

        log_prob: torch.Tensor, shape (n_samples, )
            The log joint probability of the sampled actions (used for caluculating the policy gradient, differential).

        """
        raise NotImplementedError()


@dataclass
class OracleSoftmaxLateStagePolicy(BaseLateStagePolicy):
    """Late stage ranking/recommendation policy.

    Input
    ------
    base_model: Optional[nn.Module]
        API consistency.

    action_set: Optional[BaseActionSet]
        The action set.

    n_action: Optional[int]
        The number of actions.

    inverse_temperature: float, default=1.0
        Inverse temperature parameter of softmax.

    is_anti_optimal: bool, default=False.
        Whether the policy is anti-optimal.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    base_model: Optional[nn.Module] = None
    action_set: Optional[BaseActionSet] = None
    n_action: Optional[int] = None
    inverse_temperature: float = 1.0
    is_anti_optimal: bool = False
    device: Optional[torch.device] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            self.base_model = self.base_model.to(self.device)

        if self.action_set is None:
            if self.n_action is None:
                raise ValueError("n_action must be provided.")
        else:
            if self.n_action is not None:
                assert self.n_action == self.action_set.n_action

            self.n_action = self.action_set.n_action

        self.gumbel_dist = torch.distributions.Gumbel(loc=0.0, scale=1.0)

    def _topk(
        self,
        candidate_actions: torch.Tensor,
        logits: torch.Tensor,
        n_output_action: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve top-k candidate actions.

        Input
        ------
        candidate_actions: torch.Tensor, shape (n_samples, n_candidate_action)
            The index of candidate actions to retrieve from.

        logits: torch.Tensor, shape (n_samples, n_candidate_action)
            The logits of the actions.

        n_output_action: int
            The number of output actions to retrieve.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_output_action)
            The sampled actions.

        within_candidate_idx: torch.Tensor, shape (n_samples, n_output_action)
            The index of the sampled actions within the candidate actions.

        """
        logits = logits * self.inverse_temperature

        if self.is_anti_optimal:
            _, within_candidate_idx = torch.topk(-logits, k=n_output_action, dim=1)
        else:
            _, within_candidate_idx = torch.topk(logits, k=n_output_action, dim=1)
        actions = torch.gather(candidate_actions, 1, within_candidate_idx)
        return actions, within_candidate_idx

    def retrieve_topk(
        self,
        candidate_actions: torch.Tensor,
        factual_rewards: torch.Tensor,
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
            The index of candidate actions to retrieve from.

        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Tensor, shape (n_samples, dim_context)
            API consistency.

        context_id: Tensor, shape (n_samples, )
            API consistency.

        latent: Tensor, shape (n_samples, dim_hidden, dim_action_emb)
            API consistency.

        latent_id: Tensor, shape (n_samples, )
            API consistency.

        n_output_action: int
            The number of output actions to retrieve.

        Output
        ------
        actions: torch.Tensor, shape (n_samples, n_output_action)
            The retrieved actions.

        """
        factual_rewards = torch.gather(factual_rewards, 1, candidate_actions)

        topk, _ = self._topk(
            candidate_actions=candidate_actions,
            logits=factual_rewards,
            n_output_action=n_output_action,
        )
        return topk

    def sample(
        self,
        candidate_actions: torch.Tensor,
        factual_rewards: torch.Tensor,
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
            The index of candidate actions to sample from.

        factual_rewards: torch.Tensor, shape (n_samples, n_actions)
            The factual rewards for all actions.

        context: Tensor, shape (n_samples, dim_context)
            API consistency.

        context_id: Tensor, shape (n_samples, )
            API consistency.

        latent: Tensor, shape (n_samples, dim_hidden, dim_action_emb)
            API consistency.

        latent_id: Tensor, shape (n_samples, )
            API consistency.

        n_output_action: int
            The number of output actions to sample.

        is_deterministic: bool
            Whether to use deterministic sampling.

        Output
        ------
        actions: torch.Tensor
            The sampled actions.

        """
        factual_rewards = torch.gather(factual_rewards, 1, candidate_actions)

        if is_deterministic:
            gumbel_noise = torch.zeros_like(factual_rewards).to(self.device)
        else:
            gumbel_noise = self.gumbel_dist.sample(
                factual_rewards.shape
            ).to(self.device)

        gumbel_noise = -gumbel_noise if self.is_anti_optimal else gumbel_noise

        topk, _ = self._topk(
            candidate_actions=candidate_actions,
            logits=factual_rewards + gumbel_noise,
            n_output_action=n_output_action,
        )
        return topk
