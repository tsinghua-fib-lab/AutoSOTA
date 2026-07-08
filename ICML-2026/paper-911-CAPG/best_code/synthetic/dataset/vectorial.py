# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Vectorial data generation functions."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ..policy.action_set import VectorialActionSet
from ..utils import sigmoid

from .base import BaseContextSampler, BaseLatentSampler, BaseRewardModel


@dataclass
class VectorialContextSampler(BaseContextSampler):
    """Vectorial context sampler.

    Input
    ------
    is_discrete: bool
        Whether the latent is discrete or not.

    n_discrete_context: int
        The number of discrete latent variables.

    dim_context: int
        The dimension of the context.

    contexts: Optional[torch.Tensor], shape (n_discrete_context, dim_context)
        The (discrete) context variables.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    is_discrete: bool = False
    n_discrete_context: Optional[int] = None
    dim_context: Optional[int] = None
    contexts: Optional[torch.Tensor] = None
    device: Optional[torch.device] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)

        if self.is_discrete and self.contexts is None:
            if self.n_discrete_context is None:
                self.n_discrete_context = 10
            if self.dim_context is None:
                self.dim_context = 10

            # fix distribution here
            self.contexts = torch.rand(
                self.n_discrete_context,
                self.dim_context,
                device=self.device,
            )
            self.contexts = 2 * self.contexts - 1  # U [-1, 1]
            self.contexts = self.contexts / self.contexts.norm(dim=-1).unsqueeze(-1)

        elif self.is_discrete and self.contexts is not None:
            assert self.contexts.ndim == 2
            self.n_discrete_context, self.dim_context = self.contexts.shape
            self.contexts = self.contexts.to(self.device)

        self.latent_reactiveness_coef = torch.randn(
            self.dim_context,
            device=self.device,
        )
        # shift to increase the reactiveness of the users
        self.latent_reactiveness_bias = 3.0

    def retrieve_embeddings(
        self,
        context_id: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve context embeddings.

        Input
        ------
        context_id: Tensor, shape (n_samples, )
            Context id to retrieve the embeddings.

        Output
        ------
        context: Tensor, shape (n_samples, dim_context)
            The context embeddings.
        """
        return self.contexts[context_id]

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
        context: torch.Tensor, shape (n_samples, dim_context)
            The sampled context.

        context_id: Optional[torch.Tensor], shape (n_samples, )
            The sampled context ids. None if the context is not discrete.

        """
        if self.is_discrete:
            context_id = torch.randint(
                self.n_discrete_context,
                (n_samples,),
                device=self.device,
            )
            context = self.retrieve_embeddings(context_id)

        else:
            context_id = None
            context = torch.rand(
                n_samples,
                self.dim_context,
                device=self.device,
            )
            context = 2 * context - 1  # U [-1, 1]
            context = context / context.norm(dim=-1).unsqueeze(-1)

        return context, context_id

    def retrieve_latent_reactiveness(
        self,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve action-specific reactiveness hyperparam to the latent features.

        Input
        ------
        context: Tensor, shape (n_samples, dim_context)
            The context.

        Output
        ------
        latent_reactiveness: Tensor, shape (n_samples, dim_context)
            The context-specific reward noise.

        """
        logit = (
            context * self.latent_reactiveness_coef.view(1, -1)
        ).sum(dim=-1) + self.latent_reactiveness_bias
        return sigmoid(logit)


@dataclass
class VectorialLatentSampler(BaseLatentSampler):
    """Vectorial latent sampler.

    Input
    ------
    is_discrete: bool
        Whether the latent is discrete or not.

    n_discrete_latent: int
        The number of discrete latent variables.

    dim_context: int
        The dimension of the context.

    dim_action_emb: int
        The dimension of the action embs.

    latents: Optional[torch.Tensor], shape (n_discrete_latent, dim_context, dim_action_emb)
        The (discrete) latent variables.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    is_discrete: bool = False
    n_discrete_latent: Optional[int] = None
    dim_context: Optional[int] = None
    dim_action_emb: Optional[int] = None
    latents: Optional[torch.Tensor] = None
    device: Optional[torch.device] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)

        if self.is_discrete and self.latents is None:
            if self.n_discrete_latent is None:
                self.n_discrete_latent = 10
            if self.dim_context is None:
                self.dim_context = 10
            if self.dim_action_emb is None:
                self.dim_action_emb = 10

            # fix distribution here
            self.latents = torch.randn(
                self.n_discrete_latent,
                self.dim_context,
                self.dim_action_emb,
                device=self.device,
            )
            self.latents = self.latents / self.latents.norm(dim=-1).unsqueeze(-1)

        elif self.is_discrete and self.latents is not None:
            assert self.latents.ndim == 3
            (
                self.n_discrete_latent,
                self.dim_context,
                self.dim_action_emb,
            ) = self.latents.shape
            self.latents = self.latents.to(self.device)

    def retrieve_embeddings(
        self,
        latent_id: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve latent embeddings.

        Input
        ------
        latent_id: Tensor, shape (n_samples, )
            Latent id to retrieve the embeddings.

        Output
        ------
        latent: Tensor, shape (n_samples, dim_context, dim_action_emb)
            The latent embeddings.

        """
        return self.latents[latent_id]

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
        lantent: torch.Tensor, shape (n_samples, dim_context, dim_action_emb)
            The sampled latent feartures.

        latent_id: Optional[torch.Tensor], shape (n_samples, )
            The sampled latent feartures ids. None if the latent features are not discrete.

        """
        if self.is_discrete:
            latent_id = torch.randint(
                self.n_discrete_latent,
                (n_samples,),
                device=self.device,
            )
            latent = self.retrieve_embeddings(latent_id)

        else:
            latent_id = None
            latent = torch.randn(
                n_samples,
                self.dim_context,
                self.dim_action_emb,
                device=self.device,
            )
            # latent = 2 * latent - 1  # U [-1, 1]
            latent = latent / latent.norm(dim=-1).unsqueeze(-1)

        return latent, latent_id


@dataclass
class VectorialRewardModel(BaseRewardModel):
    """Vectorial reward model.

    Input
    ------
    context_sampler: BaseContextSampler
        The context sampler object.

    action_set: ActionSet
        The action set object.

    n_output_action: int
        The number of output actions.

    proj_matrix: Optional[torch.Tensor], shape (dim_context, dim_action_emb)
        The projection matrix of context and action.

    ranking_weight: Optional[torch.Tensor], shape (n_output_action, )
        The ranking weight for each action.

    reward_scaler: int = 1
        The reward scaler.

    reward_std_scaler: int = 1
        The reward std scaler.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    context_sampler: VectorialContextSampler
    action_set: VectorialActionSet
    n_output_action: int = 1
    proj_matrix: Optional[torch.Tensor] = None
    ranking_weight: Optional[torch.Tensor] = None
    reward_scaler: int = 1
    reward_std_scaler: int = 1
    device: Optional[torch.device] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)

        self.dim_context = self.context_sampler.dim_context
        self.dim_action_emb = self.action_set.dim_action_emb
        self.n_action = self.action_set.n_action

        if self.proj_matrix is not None:
            assert self.proj_matrix.ndim == 2
            assert self.proj_matrix.shape[0] == self.dim_context
            assert self.proj_matrix.shape[1] == self.action_set.dim_action_emb

        else:
            # fix distribution here
            self.proj_matrix = torch.randn(
                self.dim_context,
                self.dim_action_emb,
                device=self.device,
            )
            self.proj_matrix = self.proj_matrix / self.proj_matrix.norm(
                dim=-1
            ).unsqueeze(-1)

        self.reward_std_coefs = torch.randn(
            self.dim_context,
            self.action_set.dim_action_emb,
            device=self.device,
        )
        # shift to increase the reward std
        self.reward_std_bias = 1.0

    def _retrieve_reward_std(
        self,
        context: torch.Tensor,
        latent: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve context-action-specific reward std.

        Input
        ------
        context: Tensor, shape (n_samples, dim_context)
            Context to retrieve the reward std.

        latent: Tensor, shape (n_samples, dim_action_hidden, dim_action_emb)
            Latent features to retrieve the reward std.

        actions: Tensor, shape (n_samples, n_output_action)
            Actions to retrieve the embeddings.

        Output
        ------
        reward_noise: Tensor, shape (n_samples, n_actions)
            The context-action-specific reward noise.

        """
        action_emb = self.action_set.retrieve_embeddings(actions)

        context_proj = context @ self.reward_std_coefs
        logit = (context_proj.unsqueeze(1) * action_emb).sum(
            dim=-1
        ) + self.reward_std_bias
        return sigmoid(logit) * self.reward_std_scaler

    def _expected(
        self,
        context: torch.Tensor,
        latent: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Expected reward.

        Input
        ------
        context: Tensor, shape (n_samples, dim_context)
            Context to retrieve the reward std.

        latent: Tensor, shape (n_samples, dim_context, dim_action_emb)
            Latent features.

        actions: Tensor, shape (n_samples, n_output_action)
            Actions to retrieve the embeddings.

        Output
        ------
        expected_reward: Tensor, shape (n_samples, n_output_action)
            The expected reward.

        """
        action_emb = self.action_set.retrieve_embeddings(actions)

        action_latent_reactiveness = self.action_set.retrieve_latent_reactiveness(
            actions
        )  # (n_samples, n_output_action)
        context_latent_reactiveness = self.context_sampler.retrieve_latent_reactiveness(
            context,
        )  # (n_samples, )
        latent_reactiveness = (
            action_latent_reactiveness
            * context_latent_reactiveness.unsqueeze(1)
        ) ** (1 / 2)  # (n_samples, n_output_action)

        M0 = self.proj_matrix.unsqueeze(0).unsqueeze(0)
        Mz = latent.unsqueeze(1)
        eta = latent_reactiveness.unsqueeze(-1).unsqueeze(-1)

        proj_matrix = (1 - eta) * M0 + eta * Mz

        expected_reward = torch.einsum(
            "ic,ijca,ija->ij", context, proj_matrix, action_emb
        )
        expected_reward = expected_reward * self.reward_scaler
        return expected_reward

    def expected(
        self,
        context: torch.Tensor,
        latent: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Expected reward.

        Input
        ------
        context: Tensor, shape (n_samples, dim_context)
            Context to retrieve the reward std.

        latent: Tensor, shape (n_samples, dim_context, dim_action_emb)
            Latent features.

        actions: Tensor, shape (n_samples, n_output_action)
            Actions to retrieve the embeddings.

        Output
        ------
        expected_reward: Tensor, shape (n_samples, n_output_action)
            The expected reward.

        agg_expected_reward: Tensor, shape (n_samples, 1)
            The expected reward aggregated across ranking.

        """
        expected_reward = self._expected(
            context=context,
            latent=latent,
            actions=actions,
        )

        if self.ranking_weight is not None:
            agg_expected_reward = (
                expected_reward * self.ranking_weight.unsqueeze(0)
            ).sum(keepdim=True)
        else:
            agg_expected_reward = expected_reward.mean(dim=-1, keepdim=True)

        return expected_reward, agg_expected_reward

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
            Context to retrieve the reward std.

        latent: Tensor, shape (n_samples, dim_action_hidden, dim_action_emb)
            Latent features.

        actions: Tensor, shape (n_samples, n_output_action)
            Actions to retrieve the embeddings.

        Output
        ------
        reward: Tensor, shape (n_samples, n_output_action)
            The sampled reward.

        agg_reward: Tensor, shape (n_samples, 1)
            The sampled reward aggregated across ranking.

        """
        expected_reward, _ = self.expected(
            context=context, latent=latent, actions=actions
        )

        # positive reward shift (softplus and const)
        expected_reward = torch.log1p(torch.exp(expected_reward)) + 1.0

        reward_std = self._retrieve_reward_std(
            context=context,
            latent=latent,
            actions=actions,
        )
        reward = torch.normal(expected_reward, reward_std * self.reward_std_scaler)

        if self.ranking_weight is not None:
            agg_reward = (reward * self.ranking_weight.unsqueeze(0)).sum(
                keepdim=True
            )
        else:
            agg_reward = reward.mean(dim=-1, keepdim=True)

        return reward, agg_reward

    def all_action_expected_reward(
        self,
        context: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """Expected reward for all actions.

        Input
        ------
        context: Tensor, shape (n_samples, dim_context)
            Context to retrieve the reward std.

        latent: Tensor, shape (n_samples, dim_context, dim_action_emb)
            Latent features.

        Output
        ------
        expected_reward: Tensor, shape (n_samples, n_action)
            The expected reward for all actions.

        """
        n_samples = context.shape[0]
        actions = (
            torch.arange(self.n_action, device=self.device)
            .unsqueeze(0)
            .expand(n_samples, -1)
        )

        expected = self._expected(
            context=context,
            latent=latent,
            actions=actions,
        )
        return expected
