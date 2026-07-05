from __future__ import annotations

import random
from itertools import chain
from typing import Any

import torch
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Dropout,
    GELU,
    Identity,
    LayerNorm,
    Linear,
    Module,
    ModuleList,
    MultiheadAttention,
    ReLU,
    Sequential,
    SiLU,
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


BlockSpec = tuple[str, int, str, str, float, str, int]


def build_activation(name: str) -> Module:
    if name == "relu":
        return ReLU()
    if name == "gelu":
        return GELU()
    if name == "silu":
        return SiLU()
    raise ValueError(f"Unsupported activation: {name}")


def reset_multihead_attention(attn: MultiheadAttention) -> None:
    """Version-stable parameter reset for MultiheadAttention."""
    if attn.in_proj_weight is not None:
        torch.nn.init.xavier_uniform_(attn.in_proj_weight)
    else:
        torch.nn.init.xavier_uniform_(attn.q_proj_weight)
        torch.nn.init.xavier_uniform_(attn.k_proj_weight)
        torch.nn.init.xavier_uniform_(attn.v_proj_weight)

    if attn.in_proj_bias is not None:
        torch.nn.init.constant_(attn.in_proj_bias, 0.0)

    attn.out_proj.reset_parameters()

    if attn.bias_k is not None:
        torch.nn.init.xavier_normal_(attn.bias_k)
    if attn.bias_v is not None:
        torch.nn.init.xavier_normal_(attn.bias_v)


class VectorNorm(Module):
    """Normalization over flat vectors [B, D]."""

    def __init__(self, dim: int, normalization: str):
        super().__init__()
        if normalization == "layer_norm":
            self.norm = LayerNorm(dim)
        elif normalization == "batch_norm":
            self.norm = BatchNorm1d(dim)
        else:
            self.norm = Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)


class TokenNorm(Module):
    """Normalization over token tensors [B, F, C]."""

    def __init__(self, dim: int, normalization: str):
        super().__init__()
        self.normalization = normalization
        if normalization == "layer_norm":
            self.norm = LayerNorm(dim)
        elif normalization == "batch_norm":
            self.norm = BatchNorm1d(dim)
        else:
            self.norm = Identity()

    def forward(self, x: Tensor) -> Tensor:
        if self.normalization == "batch_norm":
            batch_size, num_fields, dim = x.shape
            x = x.reshape(batch_size * num_fields, dim)
            x = self.norm(x)
            return x.reshape(batch_size, num_fields, dim)
        return self.norm(x)


class ReadoutNorm(Module):
    def __init__(self, dim: int, normalization: str):
        super().__init__()
        if normalization == "layer_norm":
            self.norm = LayerNorm(dim)
        elif normalization == "batch_norm":
            self.norm = BatchNorm1d(dim)
        else:
            self.norm = Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)


class FlatLinearCore(Module):
    def __init__(self, in_dim: int, out_dim: int, normalization: str, activation: str, dropout_prob: float):
        super().__init__()
        self.linear = Linear(in_dim, out_dim)
        self.norm = VectorNorm(out_dim, normalization)
        self.activation = build_activation(activation)
        self.dropout = Dropout(dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class FlatMLPCore(Module):
    def __init__(self, in_dim: int, out_dim: int, normalization: str, activation: str, dropout_prob: float):
        super().__init__()
        self.linear1 = Linear(in_dim, out_dim)
        self.norm1 = VectorNorm(out_dim, normalization)
        self.activation1 = build_activation(activation)
        self.dropout1 = Dropout(dropout_prob)
        self.linear2 = Linear(out_dim, out_dim)
        self.norm2 = VectorNorm(out_dim, normalization)
        self.activation2 = build_activation(activation)
        self.dropout2 = Dropout(dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.norm2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        return x


class FlatAttentionCore(Module):
    def __init__(
        self,
        in_dim: int,
        num_fields: int,
        token_dim: int,
        num_heads: int,
        normalization: str,
        activation: str,
        dropout_prob: float,
    ):
        super().__init__()
        self.num_fields = num_fields
        self.token_dim = token_dim
        self.out_dim = num_fields * token_dim
        self.input_proj = Linear(in_dim, self.out_dim) if in_dim != self.out_dim else Identity()
        self.norm = TokenNorm(token_dim, normalization)
        self.attn = MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True,
        )
        self.activation = build_activation(activation)
        self.dropout = Dropout(dropout_prob)

    def reset_parameters(self):
        if isinstance(self.input_proj, Linear):
            self.input_proj.reset_parameters()
        reset_multihead_attention(self.attn)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_proj(x)
        x = x.view(x.size(0), self.num_fields, self.token_dim)
        normed = self.norm(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        attn_out = self.activation(attn_out)
        x = x + self.dropout(attn_out)
        return x.reshape(x.size(0), -1)


class FlatTransformerCore(Module):
    def __init__(
        self,
        in_dim: int,
        num_fields: int,
        token_dim: int,
        num_heads: int,
        normalization: str,
        activation: str,
        dropout_prob: float,
    ):
        super().__init__()
        self.num_fields = num_fields
        self.token_dim = token_dim
        self.out_dim = num_fields * token_dim
        self.input_proj = Linear(in_dim, self.out_dim) if in_dim != self.out_dim else Identity()
        self.norm1 = TokenNorm(token_dim, normalization)
        self.attn = MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True,
        )
        self.attn_dropout = Dropout(dropout_prob)
        self.norm2 = TokenNorm(token_dim, normalization)
        self.ffn_up = Linear(token_dim, token_dim * 2)
        self.activation = build_activation(activation)
        self.ffn_dropout = Dropout(dropout_prob)
        self.ffn_down = Linear(token_dim * 2, token_dim)
        self.out_dropout = Dropout(dropout_prob)

    def reset_parameters(self):
        if isinstance(self.input_proj, Linear):
            self.input_proj.reset_parameters()
        reset_multihead_attention(self.attn)
        self.ffn_up.reset_parameters()
        self.ffn_down.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_proj(x)
        x = x.view(x.size(0), self.num_fields, self.token_dim)
        attn_input = self.norm1(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.attn_dropout(attn_out)

        ffn_input = self.norm2(x)
        ffn_out = self.ffn_up(ffn_input)
        ffn_out = self.activation(ffn_out)
        ffn_out = self.ffn_dropout(ffn_out)
        ffn_out = self.ffn_down(ffn_out)
        x = x + self.out_dropout(ffn_out)
        return x.reshape(x.size(0), -1)


class FlatSkipCore(Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = Linear(in_dim, out_dim) if in_dim != out_dim else Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class MixedFlatBlock(Module):
    def __init__(self, in_dim: int, num_fields: int, spec: BlockSpec):
        super().__init__()
        op_type, width, normalization, activation, dropout_prob, connectivity, aux = spec
        self.op_type = op_type
        self.token_width = width
        self.out_dim = num_fields * width

        if op_type == "linear":
            self.core = FlatLinearCore(in_dim, self.out_dim, normalization, activation, dropout_prob)
        elif op_type == "mlp":
            self.core = FlatMLPCore(in_dim, self.out_dim, normalization, activation, dropout_prob)
        elif op_type == "attention":
            self.core = FlatAttentionCore(in_dim, num_fields, width, aux, normalization, activation, dropout_prob)
        elif op_type == "transformer":
            self.core = FlatTransformerCore(
                in_dim,
                num_fields,
                width,
                aux,
                normalization,
                activation,
                dropout_prob,
            )
        elif op_type == "skip":
            self.core = FlatSkipCore(in_dim, self.out_dim)
        else:
            raise ValueError(f"Unsupported block operator: {op_type}")

        if connectivity == "residual" and op_type != "skip":
            self.shortcut = Linear(in_dim, self.out_dim) if in_dim != self.out_dim else Identity()
        else:
            self.shortcut = None

    def reset_parameters(self):
        if hasattr(self.core, "reset_parameters"):
            self.core.reset_parameters()
        if isinstance(self.shortcut, Linear):
            self.shortcut.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x) if self.shortcut is not None else None
        out = self.core(x)
        if residual is not None:
            out = out + residual
        return out


class PTNASBlockMixed(Module, BaseSearchSpace):
    """
    Flat block-level mixed search space.

    Each block spec is:
    (op_type, width, normalization, activation, dropout, connectivity, aux)

    where width denotes the per-field token dimension and the actual flat
    hidden size is num_fields * width.
    """

    blocks_choices = [2, 3]
    channel_choices = [64, 128, 256]

    blocks_choices_large = [2, 3, 4]
    channel_choices_large = [32, 64, 128, 256]
    operator_choices_large = ["linear", "mlp", "attention", "transformer", "skip"]
    normalization_choices_large = ["layer_norm", "batch_norm"]
    activation_choices_large = ["relu", "gelu", "silu"]
    dropout_choices_large = [0.0, 0.1, 0.2]
    connectivity_choices_large = ["plain", "residual"]
    aux_choices_large = [1, 2, 4]

    @staticmethod
    def parse_block_specs(block_specs_like: list[list | tuple]) -> list[BlockSpec]:
        return [tuple(block) for block in block_specs_like]  # type: ignore[list-item]

    @classmethod
    def sanitize_block_spec(cls, spec: BlockSpec) -> BlockSpec:
        op_type, width, normalization, activation, dropout_prob, connectivity, aux = spec

        if op_type != "skip" and normalization == "none":
            normalization = "layer_norm"

        if op_type in {"attention", "transformer"}:
            aux = cls._valid_aux(width, aux if aux > 0 else 1)
            # Attention/transformer already contain internal residual connections.
            connectivity = "plain"
        else:
            aux = 0

        if op_type == "skip":
            normalization = "none"
            activation = "relu"
            dropout_prob = 0.0
            connectivity = "plain"
            aux = 0

        return (op_type, width, normalization, activation, dropout_prob, connectivity, aux)

    @classmethod
    def sanitize_architecture(cls, architecture: list[BlockSpec]) -> list[BlockSpec]:
        return [cls.sanitize_block_spec(spec) for spec in architecture]

    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        stype_encoder_dict: dict[torch_frame.stype, StypeEncoder] | None = None,
        block_specs: list[BlockSpec] | None = None,
        readout_normalization: str = "layer_norm",
        readout_activation: str = "relu",
    ) -> None:
        super().__init__()

        if stype_encoder_dict is None:
            stype_encoder_dict = {
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),
            }

        self.num_fields = sum(len(col_names) for col_names in col_names_dict.values())
        self.encoder_channels = channels

        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )

        if block_specs is None:
            block_specs = [
                ("mlp", channels, "layer_norm", "relu", 0.1, "residual", 0)
                for _ in range(num_layers)
            ]
        if len(block_specs) != num_layers:
            raise ValueError(
                f"`block_specs` length ({len(block_specs)}) must equal `num_layers` ({num_layers})."
            )

        self.block_specs = self.sanitize_architecture(list(block_specs))
        self.backbone = ModuleList()

        in_dim = self.num_fields * channels
        for spec in self.block_specs:
            block = MixedFlatBlock(in_dim, self.num_fields, spec)
            self.backbone.append(block)
            in_dim = block.out_dim

        self.readout = Sequential(
            ReadoutNorm(in_dim, readout_normalization),
            build_activation(readout_activation),
            Linear(in_dim, out_channels),
        )

        self.reset_parameters()

    @classmethod
    def from_space_record(
        cls,
        record: dict[str, Any],
        *,
        channels: int,
        out_channels: int,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        stype_encoder_dict: dict[torch_frame.stype, StypeEncoder] | None = None,
        readout_normalization: str = "layer_norm",
        readout_activation: str = "relu",
    ) -> "PTNASBlockMixed":
        block_specs = cls.parse_block_specs(record["block_specs"])
        return cls(
            channels=channels,
            out_channels=out_channels,
            num_layers=len(block_specs),
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
            block_specs=block_specs,
            readout_normalization=readout_normalization,
            readout_activation=readout_activation,
        )

    @classmethod
    def _valid_aux(cls, width: int, aux: int) -> int:
        valid = [choice for choice in cls.aux_choices_large if width % choice == 0]
        if not valid:
            return 1
        if aux in valid:
            return aux
        return valid[0]

    @classmethod
    def random_block_spec(cls) -> BlockSpec:
        op_type = random.choice(cls.operator_choices_large)
        width = random.choice(cls.channel_choices_large)
        normalization = random.choice(cls.normalization_choices_large)
        activation = random.choice(cls.activation_choices_large)
        dropout_prob = random.choice(cls.dropout_choices_large)
        connectivity = random.choice(cls.connectivity_choices_large)
        if op_type in {"attention", "transformer"}:
            aux = cls._valid_aux(width, random.choice(cls.aux_choices_large))
        else:
            aux = 0
        if op_type == "skip":
            connectivity = "plain"
            aux = 0
        return cls.sanitize_block_spec(
            (op_type, width, normalization, activation, dropout_prob, connectivity, aux)
        )

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        for block in self.backbone:
            if hasattr(block, "reset_parameters"):
                block.reset_parameters()
        for module in self.readout:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, tf: TensorFrame) -> Tensor:
        x, _ = self.encoder(tf)
        x = x.reshape(x.size(0), -1)
        for block in self.backbone:
            x = block(x)
        return self.readout(x)

    def forward_wo_embedding(self, x: Tensor) -> Tensor:
        """
        x: [B, num_fields * channels], i.e., flattened encoder output.
        """
        for block in self.backbone:
            x = block(x)
        return self.readout(x)

    def estimate_capacity(self, include_bias: bool = True) -> int:
        total = 0
        for module in chain(self.backbone.modules(), self.readout.modules()):
            if isinstance(module, Linear):
                total += module.in_features * module.out_features
                if include_bias and module.bias is not None:
                    total += module.out_features
            elif isinstance(module, LayerNorm):
                total += 2 * module.normalized_shape[0]
            elif isinstance(module, BatchNorm1d):
                total += 2 * module.num_features
            elif isinstance(module, MultiheadAttention):
                total += module.in_proj_weight.numel()
                if include_bias and module.in_proj_bias is not None:
                    total += module.in_proj_bias.numel()
                total += module.out_proj.weight.numel()
                if include_bias and module.out_proj.bias is not None:
                    total += module.out_proj.bias.numel()
        return total

    @staticmethod
    def estimate_capacity_static(
        block_specs: list[BlockSpec],
        num_fields: int = 1,
        in_channels: int = 32,
        out_channels: int = 1,
        include_bias: bool = True,
        readout_normalization: str = "layer_norm",
    ) -> int:
        total = 0
        current_dim = num_fields * in_channels
        block_specs = PTNASBlockMixed.sanitize_architecture(block_specs)

        for op_type, width, normalization, _, _, connectivity, _ in block_specs:
            block_dim = num_fields * width
            if op_type == "linear":
                total += current_dim * block_dim + (block_dim if include_bias else 0)
            elif op_type == "mlp":
                total += current_dim * block_dim + (block_dim if include_bias else 0)
                total += block_dim * block_dim + (block_dim if include_bias else 0)
            elif op_type == "attention":
                if current_dim != block_dim:
                    total += current_dim * block_dim + (block_dim if include_bias else 0)
                total += 4 * width * width + (4 * width if include_bias else 0)
            elif op_type == "transformer":
                if current_dim != block_dim:
                    total += current_dim * block_dim + (block_dim if include_bias else 0)
                total += 4 * width * width + (4 * width if include_bias else 0)
                total += width * (2 * width) + (2 * width if include_bias else 0)
                total += (2 * width) * width + (width if include_bias else 0)
            elif op_type == "skip":
                if current_dim != block_dim:
                    total += current_dim * block_dim + (block_dim if include_bias else 0)
            else:
                raise ValueError(f"Unsupported block operator: {op_type}")

            if normalization in {"layer_norm", "batch_norm"} and op_type != "skip":
                if op_type == "linear":
                    total += 2 * block_dim
                elif op_type == "mlp":
                    total += 4 * block_dim
                elif op_type == "attention":
                    total += 2 * width
                elif op_type == "transformer":
                    total += 4 * width

            if connectivity == "residual" and op_type != "skip" and current_dim != block_dim:
                total += current_dim * block_dim + (block_dim if include_bias else 0)

            current_dim = block_dim

        if readout_normalization in {"layer_norm", "batch_norm"}:
            total += 2 * current_dim
        total += current_dim * out_channels + (out_channels if include_bias else 0)
        return total

    @staticmethod
    def mutate_architecture(architecture: list[BlockSpec], mutation_rate: float = 0.3) -> list[BlockSpec]:
        mutated = list(architecture)
        if not mutated or random.random() >= mutation_rate:
            return mutated

        min_depth = min(PTNASBlockMixed.blocks_choices_large)
        max_depth = max(PTNASBlockMixed.blocks_choices_large)

        operations = [
            "change_op",
            "change_width",
            "change_norm",
            "change_activation",
            "change_dropout",
            "change_connectivity",
            "change_aux",
        ]
        if len(mutated) < max_depth:
            operations.append("insert_layer")
        if len(mutated) > min_depth:
            operations.append("delete_layer")

        operation = random.choice(operations)
        idx = random.randrange(len(mutated))
        op_type, width, normalization, activation, dropout_prob, connectivity, aux = mutated[idx]

        if operation == "change_op":
            choices = [choice for choice in PTNASBlockMixed.operator_choices_large if choice != op_type]
            op_type = random.choice(choices)
            if op_type in {"attention", "transformer"}:
                aux = PTNASBlockMixed._valid_aux(width, random.choice(PTNASBlockMixed.aux_choices_large))
            else:
                aux = 0
            if op_type == "skip":
                connectivity = "plain"
            mutated[idx] = (op_type, width, normalization, activation, dropout_prob, connectivity, aux)
        elif operation == "change_width":
            width_choices = [choice for choice in PTNASBlockMixed.channel_choices_large if choice != width]
            width = random.choice(width_choices or PTNASBlockMixed.channel_choices_large)
            if op_type in {"attention", "transformer"}:
                aux = PTNASBlockMixed._valid_aux(width, aux if aux > 0 else 1)
            mutated[idx] = (op_type, width, normalization, activation, dropout_prob, connectivity, aux)
        elif operation == "change_norm":
            choices = [choice for choice in PTNASBlockMixed.normalization_choices_large if choice != normalization]
            normalization = random.choice(choices or PTNASBlockMixed.normalization_choices_large)
            mutated[idx] = (op_type, width, normalization, activation, dropout_prob, connectivity, aux)
        elif operation == "change_activation":
            choices = [choice for choice in PTNASBlockMixed.activation_choices_large if choice != activation]
            activation = random.choice(choices or PTNASBlockMixed.activation_choices_large)
            mutated[idx] = (op_type, width, normalization, activation, dropout_prob, connectivity, aux)
        elif operation == "change_dropout":
            choices = [choice for choice in PTNASBlockMixed.dropout_choices_large if choice != dropout_prob]
            dropout_prob = random.choice(choices or PTNASBlockMixed.dropout_choices_large)
            mutated[idx] = (op_type, width, normalization, activation, dropout_prob, connectivity, aux)
        elif operation == "change_connectivity":
            if op_type == "skip":
                mutated[idx] = (op_type, width, normalization, activation, dropout_prob, "plain", 0)
            else:
                choices = [
                    choice for choice in PTNASBlockMixed.connectivity_choices_large if choice != connectivity
                ]
                connectivity = random.choice(choices or PTNASBlockMixed.connectivity_choices_large)
                mutated[idx] = (op_type, width, normalization, activation, dropout_prob, connectivity, aux)
        elif operation == "change_aux":
            if op_type in {"attention", "transformer"}:
                aux_choices = [choice for choice in PTNASBlockMixed.aux_choices_large if width % choice == 0]
                aux_choices = [choice for choice in aux_choices if choice != aux]
                aux = random.choice(aux_choices or [PTNASBlockMixed._valid_aux(width, 1)])
                mutated[idx] = (op_type, width, normalization, activation, dropout_prob, connectivity, aux)
            else:
                promoted_op = random.choice(["attention", "transformer"])
                aux = PTNASBlockMixed._valid_aux(width, random.choice(PTNASBlockMixed.aux_choices_large))
                mutated[idx] = (promoted_op, width, normalization, activation, dropout_prob, connectivity, aux)
        elif operation == "insert_layer":
            insert_idx = random.randint(0, len(mutated))
            mutated.insert(insert_idx, PTNASBlockMixed.random_block_spec())
        else:
            del mutated[idx]

        return PTNASBlockMixed.sanitize_architecture(mutated)

    @staticmethod
    def crossover_architectures(
        parent1: list[BlockSpec],
        parent2: list[BlockSpec],
    ) -> tuple[list[BlockSpec], list[BlockSpec]]:
        if len(parent1) == len(parent2):
            if len(parent1) <= 1:
                return list(parent1), list(parent2)
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = list(parent1[:crossover_point]) + list(parent2[crossover_point:])
            child2 = list(parent2[:crossover_point]) + list(parent1[crossover_point:])
            return (
                PTNASBlockMixed.sanitize_architecture(child1),
                PTNASBlockMixed.sanitize_architecture(child2),
            )

        def build_child(target_len: int) -> list[BlockSpec]:
            child: list[BlockSpec] = []
            for idx in range(target_len):
                candidates: list[BlockSpec] = []
                if idx < len(parent1):
                    candidates.append(parent1[idx])
                if idx < len(parent2):
                    candidates.append(parent2[idx])
                child.append(random.choice(candidates))
            return child

        target_lengths = [len(parent1), len(parent2)]
        random.shuffle(target_lengths)
        return (
            PTNASBlockMixed.sanitize_architecture(build_child(target_lengths[0])),
            PTNASBlockMixed.sanitize_architecture(build_child(target_lengths[1])),
        )
