# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Implementation of the action set."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from src.utils import (
    sigmoid,
)


@dataclass
class BaseActionSet(ABC):
    """Base class for action set.

    Input
    ------
    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    @abstractmethod
    def retrieve_embeddings(
        self,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve action embeddings.

        Input
        ------
        actions: Tensor, shape (n_samples, n_actions)
            Actions to retrieve the embeddings.

        Output
        ------
        action_embs: Tensor, shape (n_samples, n_actions, dim_action_emb)
            The action embeddings.

        """
        raise NotImplementedError()
    

@dataclass
class SimpleActionSet(BaseActionSet):
    """Vectorial action set.

    Input
    ------
    n_action: int
        The number of actions.

    dim_action_emb: int
        The dimension of the action.

    action_embs: Optional[torch.Tensor], shape (n_action, dim_action_emb)
        The action embs.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    n_action: int
    device: Optional[torch.device] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        pass

    def retrieve_embeddings(
        self,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """API consistency."""
        pass


@dataclass
class VectorialActionSet(BaseActionSet):
    """Vectorial action set.

    Input
    ------
    n_action: int
        The number of actions.

    dim_action_emb: int
        The dimension of the action.

    action_embs: Optional[torch.Tensor], shape (n_action, dim_action_emb)
        The action embs.

    device: Optional[torch.device]
        The device to use.

    random_seed: Optional[int]
        The random seed to use.

    """

    n_action: Optional[int] = None
    dim_action_emb: Optional[int] = None
    action_embs: Optional[torch.Tensor] = None
    device: Optional[torch.device] = None
    random_seed: Optional[int] = None

    def __post_init__(self):
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)

        if self.action_embs is None:
            if self.n_action is None:
                self.n_action = 1000
            if self.dim_action_emb is None:
                self.dim_action_emb = 10

            # fix distribution here
            self.action_embs = torch.rand(
                self.n_action,
                self.dim_action_emb,
                device=self.device,
            )
            self.action_embs = 2 * self.action_embs - 1  # U [-1, 1]
            self.action_embs = self.action_embs / self.action_embs.norm(
                dim=-1
            ).unsqueeze(-1)

        else:
            assert self.action_embs.ndim == 2
            self.n_action, self.dim_action_emb = self.action_embs.shape
            self.action_embs = self.action_embs.to(self.device)

        self.latent_reactiveness_coefs = torch.torch.randn(
            self.dim_action_emb,
            device=self.device,
        )
        # shift to increase the reactiveness of the users
        self.latent_reactiveness_bias = 3.0

    def retrieve_embeddings(
        self,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve action embeddings.

        Input
        ------
        actions: Tensor, shape (n_samples, n_actions)
            Actions to retrieve the embeddings.

        Output
        ------
        action_embs: Tensor, shape (n_samples, n_actions, dim_action_emb)
            The action embeddings.
        """
        return self.action_embs[actions]

    def retrieve_latent_reactiveness(
        self,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve action-specific reactiveness hyperparam to the latent features.

        Input
        ------
        actions: Tensor, shape (n_samples, n_actions)
            Actions to retrieve the embeddings.

        Output
        ------
        latent_reactiveness: Tensor, shape (n_samples, n_actions)
            The action-specific reward noise.

        """
        action_emb = self.retrieve_embeddings(actions)
        logit = (
            action_emb * self.latent_reactiveness_coefs.view(1, 1, -1)
        ).sum(dim=-1) + self.latent_reactiveness_bias
        return sigmoid(logit)
