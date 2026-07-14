#!/usr/bin/env python3
"""
Description: Implements the LoRAExperts class, a LoRA-based approach for
integrating modality-specific experts into a transformer model. The experts
share a common set of base weights and can be applied to the feedforward layer
(similar to Mixture of Experts, MoE), the attention mechanism, or both, as
seen in Mixture of Transformers (MoT) architectures.
"""

import torch
import torch.nn as nn
from torch import Tensor

from srl.multi_modal_encoder.lora import LoRA


class LoRAExperts(nn.Module):
    def __init__(
        self,
        w: nn.Linear,
        rank: int,
        dim: int,
        n_modalities: int,
        scaling: float = None,
        initialize: bool = True,
        init_type: str = "xavier_uniform",
        init_settings: dict = None,
        out_dim: int = None,
        chunk_size: int = None,
        sequence_modalities: dict = None,
    ) -> None:
        """
        A LoRA-based approach for integrating modality-specific experts into a
        transformer model. The experts share a common set of base weights and
        can be applied to the feedforward layer(similar to Mixture of Experts
        (MoE), the attention mechanism, or both, as seen in Mixture of
        Transformers (MoT) architectures.

        Parameters
        ----------
        w : nn.Linear
            The original projection layer.
        rank : int
            Rank of the LoRA module (low-rank adaptation).
        dim : int
            Input dimension of the weight matrix.
        n_modalities : int
            Number of modality specific experts.
        scaling : float, optional
            Scaling factor for LoRA adjustment, by default equal to rank.
        initialize : bool, optional
            Whether to initialize weights, by default True.
        init_type : str, optional
            Type of initialization: "normal", "kaiming_uniform",
            "xavier_uniform", by default "xavier_uniform".
        init_settings : dict, optional
            Custom initialization settings (e.g., mean/std for "normal").
        out_dim : int, optional
            Output dimension of the LoRA layer, by default equal to dim.
        chunk_size : int, optional
            Chunk size for processing modalities.
            - If None, all modalities are processed in parallel.
            - If 1, they are processed sequentially like a for-loop.
            Has influence on memory consumption and speed, by default None.
        sequence_modalities : dict, optional
            Dictionary specifying tokens per modalities, by default None.
        """
        super().__init__()

        # Store parameters
        self.out_dim = out_dim if out_dim is not None else dim
        self.n_modalities = n_modalities
        self.chunk_size = chunk_size
        self.sequence_modalities = sequence_modalities
        scaling = scaling if scaling is not None else rank

        self.lora_models = nn.ModuleList(
            [
                LoRA(
                    w,
                    rank,
                    dim,
                    initialize,
                    init_type,
                    init_settings,
                    self.out_dim,
                    scaling,
                )
                for _ in range(n_modalities)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass using each expert’s own parameters.
        """
        # x: (batch_size, sequence_length, feature_dim)
        # Permute to (sequence_length, batch_size, feature_dim)
        x = x.permute(1, 0, 2)

        expert_outputs = []
        start_index = 0
        # x has shape (total_sequence_length, batch_size, feature_dim)
        # where total_sequence_length = sum(self.sequence_modality.values())
        for i, expert in enumerate(self.lora_models):
            # Determine number of tokens for this expert from the dict.
            # The dict is built as: {0: seq_len0, 1: seq_len1, ...}
            token_count = self.sequence_modalities[i]

            end_index = start_index + token_count

            # Slice the tokens for this expert: shape (token_count, batch_size,
            # feature_dim)
            expert_input = x[start_index:end_index, :, :]

            # Permute to (batch_size, token_count, feature_dim)
            expert_input = expert_input.transpose(0, 1)
            bs, seq_len, feature_dim = expert_input.shape

            # Flatten the tokens: shape (batch_size * token_count, feature_dim)
            expert_input_flat = expert_input.reshape(bs * seq_len, feature_dim)

            # Process through the expert (LoRA model): expected output shape
            # (batch_size * token_count, out_dim)
            out_i = expert(expert_input_flat)

            # Reshape back to (batch_size, token_count, out_dim)
            out_dim = out_i.shape[-1]
            out_i = out_i.view(bs, seq_len, out_dim)

            # Permute to get final shape (token_count, batch_size, out_dim)
            out_i = out_i.transpose(0, 1)

            expert_outputs.append(out_i)

            # Update start_index for the next expert
            start_index = end_index

        # Concatenate along the sequence dimension
        out = torch.cat(expert_outputs, dim=0)  # (sequence_length, batch_size,
        # out_dim)

        # Permute back to (batch_size, sequence_length, out_dim)
        return out.permute(1, 0, 2)
