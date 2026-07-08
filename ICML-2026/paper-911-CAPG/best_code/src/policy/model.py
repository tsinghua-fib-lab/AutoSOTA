# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Model architectures."""

from typing import Optional

import torch
import torch.nn as nn


class EarlyStageTwoTowerModel(nn.Module):
    """Basic two tower model for the early stage policy.

    Input
    ------
    n_context: int
        The number of contexts.

    n_action: int
        The number of actions.

    dim_emb: int = 10
        The dimension of the embedding.

    n_model: int = 1
        The number of models (mixture-of-experts).

    """

    def __init__(
        self,
        n_context: int,
        n_action: int,
        dim_emb: int = 10,
        n_model: int = 1,
    ):
        super().__init__()
        self.n_model = n_model
        self.n_action = n_action
        self.context_encoder = nn.ModuleList(
            [nn.Embedding(n_context, dim_emb) for _ in range(n_model)]
        )
        self.action_encoder = nn.ModuleList(
            [nn.Embedding(n_action, dim_emb) for _ in range(n_model)]
        )

    def forward(
        self,
        context_id: torch.Tensor,  # (batch_size, )
        action_ids: Optional[
            torch.Tensor
        ] = None,  # (batch_size, ) or (batch_size, n_action)
        require_grad_model_id: Optional[int] = None,
        **kwargs,
    ):
        ### MAY NEED BATCH PROCESSING HERE ###

        if action_ids is None:  # enumerate all actions
            action_ids = torch.arange(self.n_action).unsqueeze(0)

        context_id = context_id.to(self.context_encoder[0].weight.device)
        action_ids = action_ids.to(self.action_encoder[0].weight.device)

        logits = []
        for model_id_ in range(self.n_model):
            if require_grad_model_id is None or model_id_ == require_grad_model_id:
                context_emb_ = self.context_encoder[model_id_](context_id)
                action_emb_ = self.action_encoder[model_id_](action_ids)
            else:
                with torch.no_grad():
                    context_emb_ = self.context_encoder[model_id_](context_id)
                    action_emb_ = self.action_encoder[model_id_](action_ids)

            if action_emb_.ndim == 3:
                context_emb_ = context_emb_.unsqueeze(1)

            logit_ = (context_emb_ * action_emb_).sum(dim=-1)
            logits.append(logit_.unsqueeze(0))

        logits = torch.cat(logits, dim=0)
        return logits  # (n_model, batch_size, ) or (n_model, batch_size, n_action)


class EarlyStageTwoTowerQuantileModel(nn.Module):
    """Basic two tower model for the early stage policy.

    Input
    ------
    n_context: int
        The number of contexts.

    n_action: int
        The number of actions.

    dim_emb: int = 10
        The dimension of the embedding.

    n_bin: int = 11
        The number of models (corresponding to each quantile).

    """

    def __init__(
        self,
        n_context: int,
        n_action: int,
        dim_emb: int = 10,
        n_bin: int = 10,
    ):
        super().__init__()
        self.n_bin = n_bin
        self.bins = torch.linspace(0, 1, n_bin)
        self.n_action = n_action

        self.context_encoder = nn.ModuleList(
            [nn.Embedding(n_context, dim_emb) for _ in range(n_bin)]
        )
        self.action_encoder = nn.ModuleList(
            [nn.Embedding(n_action, dim_emb) for _ in range(n_bin)]
        )

        self.softplus = nn.Softplus()

    def _interp(
        self,
        x: torch.Tensor,
        xp: torch.Tensor,
        fp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differential PyTorch implementation of numpy.interp for 1D linear interpolation.

        Input
        ------
        x: torch.Tensor, shape (n_samples, )
            Points to interpolate at.

        xp: torch.Tensor, shape (n_bin, )
            Known x-coordinates (must be sorted and be monotonically increasing).

        fp: torch.Tensor, shape (n_bin, n_samples)
            Known y-coordinates.

        Output
        ------
        interpolated: torch.Tensor, shape (n_samples, )
            Interpolated values at x.

        """
        n_bin, n_sample = fp.shape
        xp_expanded = xp.unsqueeze(1).expand(-1, n_sample)
        x_expanded = x.unsqueeze(0).expand(n_bin, -1)

        slopes = (fp[1:, :] - fp[:-1, :]) / (xp_expanded[1:, :] - xp_expanded[:-1, :])
        biases = fp[:-1, :] - slopes * xp_expanded[:-1, :]  # (n_bin - 1, n_sample)

        idxs = torch.sum(x_expanded >= xp_expanded, dim=0) - 1
        idxs = torch.clamp(idxs, 0, n_bin - 2)  # clamp to valid indices

        slopes_selected = slopes[idxs, torch.arange(n_sample)]
        biases_selected = biases[idxs, torch.arange(n_sample)]

        interpolated = slopes_selected * x + biases_selected
        return interpolated  # (n_sample, )

    def forward(
        self,
        target_quantile: torch.Tensor,  # (batch_size, ) or (batch_size, n_action) or (batch_size, n_action, n_monte_carlo_sample)
        context_id: torch.Tensor,  # (batch_size, )
        action_ids: Optional[
            torch.Tensor
        ] = None,  # (batch_size, ) or (batch_size, n_action)
        **kwargs,
    ):
        ### MAY NEED BATCH PROCESSING HERE ###
        bins = self.bins.to(self.context_encoder[0].weight.device)

        if action_ids is None:  # enumerate all actions
            action_ids = torch.arange(self.n_action).unsqueeze(0)

        context_id = context_id.to(self.context_encoder[0].weight.device)
        action_ids = action_ids.to(self.action_encoder[0].weight.device)

        # we want to estimate the value with residual fitting to ensure monotonisticity, i.e.,
        # quantile 0: logit0 = inner_product(context_emb0, action_emb0)
        # quantile 1: logit1 = logit0 + softplus(inner_product(context_emb1, action_emb1))
        # quantile 2: logit2 = logit1 + softplus(inner_product(context_emb2, action_emb2))
        # ...
        logits = []
        for model_id_ in range(self.n_bin):
            context_emb_ = self.context_encoder[model_id_](context_id)
            action_emb_ = self.action_encoder[model_id_](action_ids)

            if action_emb_.ndim == 3:
                context_emb_ = context_emb_.unsqueeze(1)

            logit_ = (context_emb_ * action_emb_).sum(dim=-1)

            # ensuring that the residual is non-negative
            if model_id_ > 0:
                logit_ = self.softplus(logit_)

            logits.append(logit_.unsqueeze(0))

        logits = torch.cat(logits, dim=0).reshape(self.n_bin, -1)

        # ensuring monotonicity of quantile values
        logits = logits.cumsum(dim=0)

        # interpolating the quantile values
        if target_quantile.ndim < 3:
            quantile_logits = self._interp(target_quantile.flatten(), bins, logits)
            quantile_logits = quantile_logits.reshape(target_quantile.shape)

        else:
            quantile_logits = []
            for i in range(target_quantile.shape[-1]):
                quantile_logits_ = self._interp(
                    target_quantile[:, :, i].flatten(), bins, logits
                )
                quantile_logits_ = quantile_logits_.reshape(
                    target_quantile.shape[:-1]
                ).unsqueeze(-1)
                quantile_logits.append(quantile_logits_)

            quantile_logits = torch.cat(quantile_logits, dim=-1)

        return quantile_logits  # (batch_size, ) or (batch_size, n_action) or (batch_size, n_action, n_monte_carlo_sample)


class LateStageNeuralModel(nn.Module):
    """Basic neural net model for the late stage policy.

    Input
    ------
    n_context: int
        The number of contexts.

    n_latent: int
        The number of latent variables.

    n_action: int
        The number of actions.

    dim_emb: int
        The dimension of the embedding.

    """

    def __init__(
        self,
        n_context: int,
        n_latent: int,
        n_action: int,
        dim_emb: int = 10,
        dim_hidden1: int = 100,
        dim_hidden2: int = 20,
    ):
        super().__init__()
        self.n_action = n_action
        self.context_encoder = nn.Embedding(n_context, dim_emb)
        self.latent_encoder = nn.Embedding(n_latent, dim_emb)
        self.action_encoder = nn.Embedding(n_action, dim_emb)

        self.fc1 = nn.Linear(dim_emb * 3, dim_hidden1)
        self.fc2 = nn.Linear(dim_hidden1, dim_hidden2)
        self.fc3 = nn.Linear(dim_hidden2, 1)
        self.relu = nn.ReLU()

    def forward(
        self,
        context_id: torch.Tensor,  # (batch_size, )
        latent_id: torch.Tensor,  # (batch_size, )
        action_ids: Optional[
            torch.Tensor
        ] = None,  # (batch_size, ) or (batch_size, n_action)
        **kwargs,
    ):
        ### MAY NEED BATCH PROCESSING HERE ###
        context_id = context_id.to(self.context_encoder.weight.device)
        latent_id = latent_id.to(self.latent_encoder.weight.device)

        if action_ids is None:  # enumerate all actions
            action_ids = torch.arange(self.n_action).unsqueeze(0)

        action_ids = action_ids.to(self.action_encoder.weight.device)

        context_embs = self.context_encoder(context_id)
        latent_embs = self.latent_encoder(latent_id)
        action_embs = self.action_encoder(action_ids)

        if action_embs.ndim == 3:
            n_action = action_embs.shape[1]
            context_embs = context_embs.unsqueeze(1).expand(-1, n_action, -1)
            latent_embs = latent_embs.unsqueeze(1).expand(-1, n_action, -1)

            inputs = torch.cat([context_embs, latent_embs, action_embs], dim=-1)
            inputs = inputs.view(-1, inputs.shape[-1])

            x = self.relu(self.fc1(inputs))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)

            logits = x.view(-1, n_action)

        else:
            inputs = torch.cat([context_embs, latent_embs, action_embs], dim=-1)

            x = self.relu(self.fc1(inputs))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)

            logits = x

        return logits  # (batch_size, ) or (batch_size, n_action)
