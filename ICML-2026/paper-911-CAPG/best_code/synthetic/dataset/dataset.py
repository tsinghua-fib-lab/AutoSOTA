# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch.utils.data import Dataset


class LoggedDataset(Dataset):
    """Dataset class for logged data.

    Input
    ------
    context: Tensor, shape (n_samples, dim_context)
        Context vector.

    latent: Tensor, shape (n_samples, dim_context, dim_action_emb)
        Latent features.

    action: Tensor, shape (n_samples, n_output_actions)
        Actions.

    reward: Tensor, shape (n_samples, n_output_actions)
        The sampled reward.

    agg_reward: Tensor, shape (n_samples, 1)
        The sampled reward aggregated across ranking.

    factual_rewards: Tensor, shape (n_samples, n_actions)
        The factual rewards for all actions.

    context_id: Optional[Tensor], shape (n_samples, )
        Context id.

    latent_id: Optional[Tensor], shape (n_samples, )
        Latent id.

    """

    def __init__(
        self,
        context: torch.Tensor,
        latent: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        agg_reward: torch.Tensor,
        factual_rewards: Optional[torch.Tensor] = None,
        context_id: Optional[torch.Tensor] = None,
        latent_id: Optional[torch.Tensor] = None,
    ):
        self.context = context
        self.latent = latent
        self.action = action
        self.reward = reward
        self.agg_reward = agg_reward
        self.factual_rewards = factual_rewards
        self.context_id = context_id
        self.latent_id = latent_id

    def __len__(self):
        return len(self.context)

    def __getitem__(self, idx):
        context = self.context[idx]
        latent = self.latent[idx]
        action = self.action[idx]
        reward = self.reward[idx]
        agg_reward = self.agg_reward[idx]

        if self.factual_rewards is not None:
            factual_rewards = self.factual_rewards[idx]
        else:
            factual_rewards = torch.tensor(float("nan"))

        if self.context_id is not None:
            context_id = self.context_id[idx]
        else:
            context_id = torch.tensor(float("nan"))

        if self.latent_id is not None:
            latent_id = self.latent_id[idx]
        else:
            latent_id = torch.tensor(float("nan"))

        output = {
            "context": context,
            "latent": latent,
            "action": action,
            "reward": reward,
            "agg_reward": agg_reward,
            "factual_rewards": factual_rewards,
            "context_id": context_id,
            "latent_id": latent_id,
        }
        return output
