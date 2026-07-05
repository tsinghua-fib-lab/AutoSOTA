from __future__ import annotations

import math
from typing import Any
from itertools import chain

from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Dropout,
    LayerNorm,
    Linear,
    Module,
    ReLU,
    Sequential,
)

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeEncoder,
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder
from .base import BaseSearchSpace


class FCResidualBlock(Module):
    r"""Fully connected residual block.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        normalization (str, optional): The type of normalization to use.
            :obj:`layer_norm`, :obj:`batch_norm`, or :obj:`None`.
            (default: :obj:`layer_norm`)
        dropout_prob (float): The dropout probability (default: `0.0`, i.e.,
            no dropout).
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            normalization: str | None = "layer_norm",
            dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.lin1 = Linear(in_channels, out_channels)
        self.lin2 = Linear(out_channels, out_channels)
        self.relu = ReLU()
        self.dropout = Dropout(dropout_prob)

        self.norm1: BatchNorm1d | LayerNorm | None
        self.norm2: BatchNorm1d | LayerNorm | None
        if normalization == "batch_norm":
            self.norm1 = BatchNorm1d(out_channels)
            self.norm2 = BatchNorm1d(out_channels)
        elif normalization == "layer_norm":
            self.norm1 = LayerNorm(out_channels)
            self.norm2 = LayerNorm(out_channels)
        else:
            self.norm1 = self.norm2 = None

        self.shortcut: Linear | None
        if in_channels != out_channels:
            self.shortcut = Linear(in_channels, out_channels)
        else:
            self.shortcut = None

    def reset_parameters(self) -> None:
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.shortcut is not None:
            self.shortcut.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        out = self.lin1(x)
        out = self.norm1(out) if self.norm1 else out
        out = self.relu(out)
        out = self.dropout(out)

        out = self.lin2(out)
        out = self.norm2(out) if self.norm2 else out
        out = self.relu(out)
        out = self.dropout(out)

        if self.shortcut is not None:
            x = self.shortcut(x)

        out = out + x

        return out


class PTNASResNet(Module, BaseSearchSpace):
    r"""pTnas:  Modified from from torch_frame.nn.models.resnet
        block_widths (list[int] | None):each residual block width，
        length must == num_layers. if it == None，then use `channels`。
    """
    blocks_choices = [2, 3]
    channel_choices = [64, 128, 256]

    blocks_choices_large = [2, 3, 4]
    channel_choices_large = [32, 64, 128, 256]

    blocks_choices_xlarge = [2, 3, 4, 5, 6]
    channel_choices_xlarge = [32, 64, 128, 256]

    def __init__(
            self,
            channels: int,
            out_channels: int,
            num_layers: int,
            col_stats: dict[str, dict[StatType, Any]],
            col_names_dict: dict[torch_frame.stype, list[str]],
            stype_encoder_dict: dict[torch_frame.stype, StypeEncoder]
                                | None = None,
            normalization: str | None = "layer_norm",
            dropout_prob: float = 0.2,

            block_widths: list[int] | None = None,  # ← pTnas added
    ) -> None:
        super().__init__()

        if stype_encoder_dict is None:
            stype_encoder_dict = {
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),
            }

        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )

        num_cols = sum(
            [len(col_names) for col_names in col_names_dict.values()])
        in_channels = channels * num_cols

        # ===== pTnas =====
        self.pre_backbone_dim = in_channels  # = channels * num_cols

        if block_widths is not None:
            if len(block_widths) != num_layers:
                raise ValueError(
                    f"`block_widths` length ({len(block_widths)}) "
                    f"must equal `num_layers` ({num_layers})."
                )
            widths = list(block_widths)
        else:
            widths = [channels] * num_layers

        # connect residual blocks：1st: in_channels -> widths[0]，following connected thereby
        blocks = []
        cur_in = in_channels
        for out_dim in widths:
            blocks.append(FCResidualBlock(
                cur_in, out_dim,
                normalization=normalization,
                dropout_prob=dropout_prob,
            ))
            cur_in = out_dim
        self.backbone = Sequential(*blocks)

        # pTnas: decoder fit last block width, here the channels is already the final demision.
        channels = widths[-1]
        # ==========================

        self.decoder = Sequential(
            LayerNorm(channels),
            ReLU(),
            Linear(channels, out_channels),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        for block in self.backbone:
            block.reset_parameters()
        self.decoder[0].reset_parameters()
        self.decoder[-1].reset_parameters()

    def forward(self, tf: TensorFrame) -> Tensor:
        r"""Transforming :class:`TensorFrame` object into output prediction.

        Args:
            tf (TensorFrame): Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        x, _ = self.encoder(tf)

        # Flattening the encoder output
        x = x.view(x.size(0), math.prod(x.shape[1:]))

        x = self.backbone(x)
        out = self.decoder(x)
        return out

    def forward_wo_embedding(self, x: Tensor) -> Tensor:
        """ pTnas
        x: [B, self.pre_backbone_dim]，this is dimension after encoder+flatten
        return: [B, out_channels]
        """
        x = self.backbone(x)
        return self.decoder(x)

    def estimate_capacity(self, include_bias: bool = True) -> int:
        """Head capacity (Linear params only; exclude encoder)."""
        n = 0
        for m in chain(self.backbone.modules(), self.decoder.modules()):
            if isinstance(m, Linear):
                n += m.in_features * m.out_features
                if include_bias and (m.bias is not None):
                    n += m.out_features
        return n

    @staticmethod
    def estimate_capacity_static(block_widths: list[int], in_channels: int = 1,
                                 out_channels: int = 1) -> int:
        """Estimate ResNet backbone+decoder capacity without constructing the full model.

        Args:
            block_widths: per-block widths (length = num_layers)
            in_channels: input dimension (from encoder flatten = channels * num_cols)
            out_channels: output dimension
        """
        n = 0
        cur_in = in_channels
        for out_dim in block_widths:
            # FCResidualBlock: lin1 + lin2 + optional shortcut
            n += cur_in * out_dim + out_dim      # lin1
            n += out_dim * out_dim + out_dim      # lin2
            if cur_in != out_dim:
                n += cur_in * out_dim + out_dim   # shortcut
            cur_in = out_dim
        # decoder: Linear(channels -> out_channels)
        n += cur_in * out_channels + out_channels
        return n

    @staticmethod
    def mutate_architecture(architecture: list[int], mutation_rate: float = 0.3) -> list[int]:
        """
        Mutate an architecture via one of:
        1. change_width
        2. insert_layer
        3. delete_layer

        Args:
            architecture: Original architecture (list of channel sizes)
            mutation_rate: Probability of applying a mutation

        Returns:
            Mutated architecture
        """
        import random

        mutated = architecture.copy()
        if not mutated or random.random() >= mutation_rate:
            return mutated

        min_depth = min(PTNASResNet.blocks_choices_large)
        max_depth = max(PTNASResNet.blocks_choices_large)

        operations = ["change_width"]
        if len(mutated) < max_depth:
            operations.append("insert_layer")
        if len(mutated) > min_depth:
            operations.append("delete_layer")

        operation = random.choice(operations)

        if operation == "change_width":
            idx = random.randrange(len(mutated))
            choices = [c for c in PTNASResNet.channel_choices_large if c != mutated[idx]]
            mutated[idx] = random.choice(choices or PTNASResNet.channel_choices_large)
        elif operation == "insert_layer":
            insert_idx = random.randint(0, len(mutated))
            mutated.insert(insert_idx, random.choice(PTNASResNet.channel_choices_large))
        else:  # delete_layer
            delete_idx = random.randrange(len(mutated))
            del mutated[delete_idx]

        return mutated

    @staticmethod
    def crossover_architectures(parent1: list[int], parent2: list[int]) -> tuple[list[int], list[int]]:
        """
        Crossover two architectures to create two children

        Args:
            parent1: First parent architecture
            parent2: Second parent architecture

        Returns:
            Two child architectures
        """
        import random

        def build_mixed_child(target_len: int) -> list[int]:
            child = []
            for i in range(target_len):
                candidates = []
                if i < len(parent1):
                    candidates.append(parent1[i])
                if i < len(parent2):
                    candidates.append(parent2[i])
                child.append(random.choice(candidates))
            return child

        if len(parent1) != len(parent2):
            target_lengths = [len(parent1), len(parent2)]
            random.shuffle(target_lengths)
            return build_mixed_child(target_lengths[0]), build_mixed_child(target_lengths[1])

        # Single-point crossover
        if len(parent1) <= 1:
            return parent1.copy(), parent2.copy()
        crossover_point = random.randint(1, len(parent1) - 1)

        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        return child1, child2
