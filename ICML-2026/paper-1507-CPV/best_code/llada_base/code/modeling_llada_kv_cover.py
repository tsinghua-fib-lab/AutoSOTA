# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

from __future__ import annotations

import inspect
import logging
import math
import sys
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)
from dataclasses import dataclass, fields
from typing import List, Optional, Tuple, Union
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTN_AVAILABLE = True
except:
    FLEX_ATTN_AVAILABLE = False
import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import AutoModel
from transformers.cache_utils import Cache

try:
    from .configuration_llada_kv import (
        LLaDAConfig,
        StrEnum,
        InitFnType,
        ActivationType,
        BlockType,
        LayerNormType,
        ModelConfig,
        ActivationCheckpointingStrategy,
    )
except ImportError:
    from configuration_llada_kv import (
        LLaDAConfig,
        StrEnum,
        InitFnType,
        ActivationType,
        BlockType,
        LayerNormType,
        ModelConfig,
        ActivationCheckpointingStrategy,
    )

if sys.version_info.minor > 8:
    from collections.abc import MutableMapping
elif sys.version_info.minor == 8:
    from typing import MutableMapping
else:
    raise SystemExit("This script supports Python 3.8 or higher")

__all__ = [
    "LayerNormBase",
    "LayerNorm",
    "RMSLayerNorm",
    "GemmaRMSLayerNorm",
    "RotaryEmbedding",
    "Activation",
    "GELU",
    "ReLU",
    "SwiGLU",
    "LLaDABlock",
    "LLaDASequentialBlock",
    "LLaDAModel",
    "LLaDAOutput",
    "LLaDAGenerateOutput",
]


log = logging.getLogger(__name__)

try:
    _SDPA_HAS_ENABLE_GQA = "enable_gqa" in inspect.signature(F.scaled_dot_product_attention).parameters
except (TypeError, ValueError):
    _SDPA_HAS_ENABLE_GQA = False

@torch.compile()
def scaled_dot_product_attention(
    q,
    k,
    v,
    mask=None,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    enable_gqa: bool = False,
):
    if _SDPA_HAS_ENABLE_GQA:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, enable_gqa=enable_gqa
        )
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)


def _repeat_kv_for_gqa(x: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    num_kv_heads = x.size(1)
    if num_kv_heads == num_q_heads:
        return x
    assert num_q_heads % num_kv_heads == 0
    repeat_factor = num_q_heads // num_kv_heads
    return x.repeat_interleave(repeat_factor, dim=1, output_size=num_q_heads)



def _post_hoc_diagonal_correction_exact(
    attn_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_current: torch.Tensor,
    v_current: torch.Tensor,
    pos: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Exact post-hoc diagonal correction (slower but accurate).

    Use this when you need precise correction and performance is not critical.
    This computes the full attention scores for seed positions.

    NOTE: k_current and v_current are already indexed to seed positions only,
    i.e., they have shape (B, H_kv, S, D) where S = len(pos). For GQA, these
    seed tensors will be expanded to match q's head dimension (H_q) internally.
    """
    B, Hq, L, D = q.shape
    scale = 1.0 / math.sqrt(D)
    S = pos.numel()

    # Extract tensors for seed positions
    Qj = q[:, :, pos, :]  # (B, Hq, S, D)

    Kj_old = k[:, :, pos, :]  # (B, Hkv, S, D)
    Vj_old = v[:, :, pos, :]  # (B, Hkv, S, D)
    Kj_mask = k_current  # (B, Hkv, S, D)
    Vj_mask = v_current  # (B, Hkv, S, D)

    if Kj_old.size(1) != Hq:
        Kj_old = _repeat_kv_for_gqa(Kj_old, Hq)
        Vj_old = _repeat_kv_for_gqa(Vj_old, Hq)
        Kj_mask = _repeat_kv_for_gqa(Kj_mask, Hq)
        Vj_mask = _repeat_kv_for_gqa(Vj_mask, Hq)

    # Compute score difference δ
    s_old = (Qj * Kj_old).sum(dim=-1) * scale
    s_new = (Qj * Kj_mask).sum(dim=-1) * scale
    delta = s_new - s_old

    # Compute α = w_wrong[j, j] (expensive: O(S × L))
    if k.size(1) == Hq:
        scores_j = torch.matmul(Qj, k.transpose(-2, -1)) * scale  # (B, Hq, S, L)
    else:
        # GQA path: compute scores without materializing full K/V repetition.
        Hkv = k.size(1)
        assert Hq % Hkv == 0
        g = Hq // Hkv
        Qj_grouped = Qj.reshape(B, Hkv, g, S, D)
        # (B, Hkv, g, S, L) -> (B, Hq, S, L)
        scores_j = (
            torch.einsum("b h g s d, b h l d -> b h g s l", Qj_grouped, k) * scale
        ).reshape(B, Hq, S, L)

    if attn_mask is not None:
        if attn_mask.dim() == 4:
            mask_j = attn_mask[:, :, pos, :]
        else:
            mask_j = attn_mask[pos, :]
        scores_j = scores_j + mask_j

    weights_j = F.softmax(scores_j, dim=-1)

    pos_expanded = pos.view(1, 1, S, 1).expand(B, Hq, S, 1)
    alpha = weights_j.gather(dim=-1, index=pos_expanded).squeeze(-1)

    # Compute correction
    exp_delta = torch.exp(delta)
    r = 1.0 + alpha * (exp_delta - 1.0)

    out_j_wrong = attn_output[:, :, pos, :]

    alpha_expanded = alpha.unsqueeze(-1)
    exp_delta_expanded = exp_delta.unsqueeze(-1)
    r_expanded = r.unsqueeze(-1)

    correction = alpha_expanded * (exp_delta_expanded * Vj_mask - Vj_old)
    out_j_correct = (out_j_wrong + correction) / r_expanded

    # In-place update
    attn_output[:, :, pos, :] = out_j_correct

    return attn_output

class ModuleType(StrEnum):
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"


def init_weights(
    config: ModelConfig,
    module: Union[nn.Linear, nn.Embedding],
    d: Optional[int] = None,
    layer_id: Optional[int] = None,
    std_factor: float = 1.0,
    type_of_module: Optional[ModuleType] = None,
) -> None:
    """
    Initialize weights of a linear or embedding module.

    :param config: The model config.
    :param module: The linear or embedding submodule to initialize.
    :param d: The effective input dimensionality of the weights. This could be smaller than the actual dimensions
        for fused layers.
    :param layer_id: When set, the standard deviation for the "mitchell" method will be adjusted by
        ``1 / sqrt(2 * (layer_id + 1))``.
    """
    d = d if d is not None else config.d_model
    if config.init_fn == InitFnType.normal:
        std = config.init_std * std_factor
        if config.init_cutoff_factor is not None:
            cutoff_value = config.init_cutoff_factor * std
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.mitchell:
        std = std_factor / math.sqrt(d)
        if layer_id is not None:
            std = std / math.sqrt(2 * (layer_id + 1))
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    elif config.init_fn == InitFnType.kaiming_normal:
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    elif config.init_fn == InitFnType.fan_in:
        std = std_factor / math.sqrt(d)
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.full_megatron:
        if type_of_module is None:
            raise RuntimeError(f"When using the {InitFnType.full_megatron} init, every module must have a type.")

        cutoff_factor = config.init_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        if type_of_module == ModuleType.in_module:
            # for att_proj (same as QKV), ff_proj
            std = config.init_std
        elif type_of_module == ModuleType.out_module:
            # for attn_out, ff_out
            std = config.init_std / math.sqrt(2.0 * config.n_layers)
        elif type_of_module == ModuleType.emb:
            # positional embeddings (wpe)
            # token embeddings (wte)
            std = config.init_std
        elif type_of_module == ModuleType.final_out:
            # final output (ff_out)
            std = config.d_model**-0.5
        else:
            raise RuntimeError(f"Unknown module type '{type_of_module}'")
        nn.init.trunc_normal_(
            module.weight,
            mean=0.0,
            std=std,
            a=-cutoff_factor * std,
            b=cutoff_factor * std,
        )
    else:
        raise NotImplementedError(config.init_fn)

    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)

        if config.init_fn == InitFnType.normal and getattr(module, "_is_residual", False):
            with torch.no_grad():
                module.weight.div_(math.sqrt(2 * config.n_layers))


def ensure_finite_(x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False):
    """
    Modify ``x`` in place to replace ``float("-inf")`` with the minimum value of the dtype when ``check_neg_inf``
    is ``True`` and to replace ``float("inf")`` with the maximum value of the dtype when ``check_pos_inf`` is ``True``.
    """
    if check_neg_inf:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == float("inf"), torch.finfo(x.dtype).max)


def activation_checkpoint_function(cfg: ModelConfig):
    preserve_rng_state = (
        (cfg.attention_dropout == 0.0) and (cfg.embedding_dropout == 0.0) and (cfg.residual_dropout == 0.0)
    )
    from torch.utils.checkpoint import checkpoint

    return partial(
        checkpoint,
        preserve_rng_state=preserve_rng_state,
        use_reentrant=False,
    )


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """
    Cache for attention biases and other things that would normally be stored as buffers.
    We avoid using buffers because we've run into various issues doing so with FSDP.
    In general it appears the way FSDP handles buffers is not well-defined.
    It doesn't shard them but apparently it does synchronize them across processes, which we want to avoid
    since (A) it isn't necessary, and (B) we sometimes have `-inf` in these biases which might get turned into
    NaNs when they're synchronized due to casting or some other issue.
    """


def _non_meta_init_device(config: ModelConfig) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return input
        else:
            return F.dropout(input, self.p, self.training, self.inplace)


class LayerNormBase(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.config = config
        self.eps = eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine or (elementwise_affine is None and self.config.layer_norm_with_affine):
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=config.init_device))
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=config.init_device))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig, size: Optional[int] = None, **kwargs) -> LayerNormBase:
        if config.layer_norm_type == LayerNormType.default:
            return LayerNorm(config, size=size, low_precision=False, **kwargs)
        elif config.layer_norm_type == LayerNormType.low_precision:
            return LayerNorm(config, size=size, low_precision=True, **kwargs)
        elif config.layer_norm_type == LayerNormType.rms:
            return RMSLayerNorm(config, size=size, **kwargs)
        elif config.layer_norm_type == LayerNormType.gemma_rms:
            return GemmaRMSLayerNorm(config, size=size, **kwargs)
        else:
            raise NotImplementedError(f"Unknown LayerNorm type: '{config.layer_norm_type}'")

    def _cast_if_autocast_enabled(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if tensor.device.type == "cuda" and torch.is_autocast_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype())
        elif tensor.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype())
        else:
            return tensor

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)  # type: ignore


class LayerNorm(LayerNormBase):
    """
    The default :class:`LayerNorm` implementation which can optionally run in low precision.
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        low_precision: bool = False,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-05,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)
        self.low_precision = low_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = (
                self._cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
            )
            downcast_bias = self._cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
            with torch.autocast(enabled=False, device_type=module_device.type):
                return F.layer_norm(
                    downcast_x, self.normalized_shape, weight=downcast_weight, bias=downcast_bias, eps=self.eps
                )
        else:
            return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class RMSLayerNorm(LayerNormBase):
    """
    RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return self.weight * x + self.bias
            else:
                return self.weight * x
        else:
            return x


class GemmaRMSLayerNorm(LayerNormBase):
    """
    Gemma RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return x * (1 + self.weight) + self.bias
            else:
                return x * (1 + self.weight)
        else:
            return x


class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    """

    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.__cache = cache
        # Warm up cache.
        self.rope_theta = config.rope_theta
        self.get_rotary_embedding(config.max_sequence_length, _non_meta_init_device(config))

    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            (pos_sin := self.__cache.get("rope_pos_sin")) is not None
            and (pos_cos := self.__cache.get("rope_pos_cos")) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self.__cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self.__cache["rope_pos_cos"] = pos_cos
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            dim = self.config.d_model // self.config.n_heads
            inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim))
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = positions.sin()[None, None, :, :], positions.cos()[None, None, :, :]
        self.__cache["rope_pos_sin"] = pos_sin
        self.__cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                block_end_index: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = q_.shape[-2], k_.shape[-2]
            pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)

            # Build tensor indices instead of using .item()
            if block_end_index is None:
                start = key_len - query_len
                end = key_len
            else:
                # block_end_index is a tensor; keep ops tensor-based
                start = (block_end_index - query_len)
                end = block_end_index

            # Make an index tensor [start, ..., end-1] on the right device/dtype
            idx = torch.arange(start, end, device=q_.device, dtype=torch.long)

            # Use index_select on the sequence dimension (dim=2)
            pos_sin_slice = pos_sin.index_select(2, idx)
            pos_cos_slice = pos_cos.index_select(2, idx)

            q_ = self.apply_rotary_pos_emb(pos_sin_slice, pos_cos_slice, q_)
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)

        return q_.type_as(q), k_.type_as(k)



class Activation(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_multiplier(self) -> float:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig) -> Activation:
        if config.activation_type == ActivationType.gelu:
            return cast(Activation, GELU(approximate="none"))
        elif config.activation_type == ActivationType.relu:
            return cast(Activation, ReLU(inplace=False))
        elif config.activation_type == ActivationType.silu:
            return cast(Activation, SiLU(inplace=False))
        elif config.activation_type == ActivationType.swiglu:
            return SwiGLU(config)
        else:
            raise NotImplementedError(f"Unknown activation: '{config.activation_type}'")


class GELU(nn.GELU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class ReLU(nn.ReLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0

class SiLU(nn.SiLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0

class SwiGLU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5


def causal_attention_bias(seq_len: int, device: torch.device) -> torch.FloatTensor:
    att_bias = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.float),
        diagonal=1,
    )
    att_bias.masked_fill_(att_bias == 1, torch.finfo(att_bias.dtype).min)
    return att_bias.view(1, 1, seq_len, seq_len)  # type: ignore


def get_causal_attention_bias(cache: BufferCache, seq_len: int, device: torch.device) -> torch.Tensor:
    if (causal_bias := cache.get("causal_attention_bias")) is not None and causal_bias.shape[-1] >= seq_len:
        if causal_bias.device != device:
            causal_bias = causal_bias.to(device)
            cache["causal_attention_bias"] = causal_bias
        return causal_bias
    with torch.autocast(device.type, enabled=False):
        causal_bias = causal_attention_bias(seq_len, device)
    cache["causal_attention_bias"] = causal_bias
    return causal_bias


def alibi_attention_bias(seq_len: int, config: ModelConfig, device: torch.device) -> torch.FloatTensor:
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, 1, seq_len)

    # shape: (1, 1, seq_len, seq_len)
    alibi_bias = alibi_bias - torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, seq_len, 1)
    alibi_bias.abs_().mul_(-1)

    # shape: (n_heads,)
    m = torch.arange(1, config.n_heads + 1, dtype=torch.float, device=device)
    m.mul_(config.alibi_bias_max / config.n_heads)

    # shape: (1, n_heads, seq_len, seq_len)
    return alibi_bias * (1.0 / (2 ** m.view(1, config.n_heads, 1, 1)))  # type: ignore


class LLaDABlock(nn.Module):
    """
    A base class for transformer block implementations.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.__cache = cache
        assert config.d_model % config.n_heads == 0

        self._activation_checkpoint_fn = None

        # Dropout.
        self.dropout = Dropout(config.residual_dropout)

        # Layer norms.
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        if config.attention_layer_norm:
            self.k_norm = LayerNormBase.build(
                config,
                size=(config.d_model // config.n_heads) * config.effective_n_kv_heads,
                elementwise_affine=config.attention_layer_norm_with_affine,
            )
            self.q_norm = LayerNormBase.build(config, elementwise_affine=config.attention_layer_norm_with_affine)

        # Activation function.
        self.act = Activation.build(config)
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        # Attention output projection.
        self.attn_out = nn.Linear(
            config.d_model, config.d_model, bias=config.include_bias, device=config.init_device
        )

        # Feed-forward output projection.
        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.ff_out._is_residual = True  # type: ignore

        # Rotary embeddings.
        if self.config.rope:
            self.rotary_emb = RotaryEmbedding(config, self.__cache)

        self.flash_attn_func = None
        if config.flash_attention:
            try:
                from flash_attn import flash_attn_func  # type: ignore

                self.flash_attn_func = flash_attn_func
            except ModuleNotFoundError:
                pass

        # Optional per-position KV override (used by contextual verification code).
        # When set, attention() will replace K/V at these positions with the cached tensors.
        self._seed_kv_positions: Optional[torch.Tensor] = None
        self._seed_kv_key: Optional[torch.Tensor] = None
        self._seed_kv_value: Optional[torch.Tensor] = None
        self._seed_kv_disable_self_attn: bool = False

        # Lightweight seed-KV capture: generation extracts selected seed columns
        # from these references, then clears them immediately.
        self._save_kv_for_seed: bool = False
        self._last_k: Optional[torch.Tensor] = None
        self._last_v: Optional[torch.Tensor] = None

    def reset_parameters(self):
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        init_weights(
            self.config,
            self.attn_out,
            d=self.config.d_model,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )
        init_weights(
            self.config,
            self.ff_out,
            d=self.ff_out.in_features,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        if strategy == ActivationCheckpointingStrategy.fine_grained:
            self._activation_checkpoint_fn = activation_checkpoint_function(self.config)
        else:
            self._activation_checkpoint_fn = None

    @classmethod
    def _cast_attn_bias(cls, bias: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        target_dtype = input_dtype
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if bias.device.type == "cuda" and torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif bias.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            target_dtype = torch.get_autocast_cpu_dtype()
        if bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
            ensure_finite_(bias, check_neg_inf=True, check_pos_inf=False)
        return bias

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        output_attentions: bool = False,
        seed_positions: Optional[torch.Tensor] = None,
        k_current: Optional[torch.Tensor] = None,
        v_current: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Computes scaled dot product attention on query, key and value tensors, using an optional
        attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.

        Returns:
            Tuple of (attention_output, attention_weights, gating_score) where attention_weights is None if output_attentions=False
        """
        if self.flash_attn_func is not None and attn_mask is None and not output_attentions and seed_positions is None:
            r = self.flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p, causal=False
            )
            # Flash attention doesn't return attention weights
            return r.transpose(1, 2), None, None
        else:
            # GQA handling: Prefer SDPA's `enable_gqa=True` to avoid materializing KV repetition.
            assert k.size(1) == v.size(1)
            num_kv_heads = k.size(1)
            num_q_heads = q.size(1)
            enable_gqa = False
            gqa_factor = 1
            if num_q_heads != num_kv_heads:
                assert num_q_heads % num_kv_heads == 0
                gqa_factor = num_q_heads // num_kv_heads
                if not output_attentions and _SDPA_HAS_ENABLE_GQA:
                    enable_gqa = True
                # NOTE: For output_attentions path, we use einsum-based GQA
                # to avoid expensive repeat_interleave memory copies.

            # Modify: MDM set causal to False, and with no attn_mask.
            if output_attentions:
                # Manual attention computation to get attention weights
                B, Hq, L, D = q.shape
                Hkv = k.size(1)
                scale = 1.0 / math.sqrt(D)

                # GQA-aware attention score computation without repeat_interleave
                if gqa_factor > 1:
                    # Reshape Q to (B, Hkv, g, L, D) where g = gqa_factor
                    q_grouped = q.view(B, Hkv, gqa_factor, L, D)
                    # Compute scores: (B, Hkv, g, L, L) -> (B, Hq, L, L)
                    attn_scores = torch.einsum('b h g q d, b h k d -> b h g q k', q_grouped, k) * scale
                    attn_scores = attn_scores.view(B, Hq, L, L)
                else:
                    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

                if attn_mask is not None:
                    attn_scores = attn_scores + attn_mask

                # Seed KV override: for non-seed queries, seed key/value comes from cached K/V (done upstream).
                # For seed queries themselves, use *current* (mask-derived) K/V on the diagonal (self position).
                # NOTE: seed_positions is expected to be pre-validated (all positions valid).
                if (
                    seed_positions is not None
                    and k_current is not None
                    and v_current is not None
                    and L == attn_scores.size(-1)  # Check square attention
                ):
                    pos = seed_positions.to(device=q.device, dtype=torch.long)
                    S = pos.numel()
                    if S > 0:
                        # k_current: (B, Hkv, S, D), need to handle GQA
                        Qj = q[:, :, pos, :]  # (B, Hq, S, D)
                        Kj_old = k[:, :, pos, :]  # (B, Hkv, S, D)

                        if gqa_factor > 1:
                            # GQA-aware dot product without repeat_interleave
                            # Reshape Q: (B, Hq, S, D) -> (B, Hkv, g, S, D)
                            Qj_grouped = Qj.view(B, Hkv, gqa_factor, S, D)
                            # (B, Hkv, g, S, D) * (B, Hkv, 1, S, D) -> sum over D -> (B, Hkv, g, S) -> (B, Hq, S)
                            dot_override = (Qj_grouped * Kj_old.unsqueeze(2)).sum(dim=-1).view(B, Hq, S) * scale
                            dot_current = (Qj_grouped * k_current.unsqueeze(2)).sum(dim=-1).view(B, Hq, S) * scale
                        else:
                            dot_override = (Qj * Kj_old).sum(dim=-1) * scale  # (B, Hq, S)
                            dot_current = (Qj * k_current).sum(dim=-1) * scale

                        attn_scores[:, :, pos, pos] += (dot_current - dot_override)

                attn_weights = F.softmax(attn_scores, dim=-1)

                if dropout_p > 0.0 and self.training:
                    attn_weights = F.dropout(attn_weights, p=dropout_p)

                # GQA-aware output computation without repeat_interleave
                if gqa_factor > 1:
                    # attn_weights: (B, Hq, L, L) -> (B, Hkv, g, L, L)
                    attn_weights_grouped = attn_weights.view(B, Hkv, gqa_factor, L, L)
                    # v: (B, Hkv, L, D)
                    # output: (B, Hkv, g, L, D) -> (B, Hq, L, D)
                    attn_output = torch.einsum('b h g q k, b h k d -> b h g q d', attn_weights_grouped, v)
                    attn_output = attn_output.view(B, Hq, L, D)
                else:
                    attn_output = torch.matmul(attn_weights, v)

                # Apply seed value correction
                if (
                    seed_positions is not None
                    and v_current is not None
                    and L == attn_weights.size(-1)
                ):
                    pos = seed_positions.to(device=q.device, dtype=torch.long)
                    S = pos.numel()
                    if S > 0:
                        self_w = attn_weights[:, :, pos, pos].unsqueeze(-1)  # (B, Hq, S, 1)
                        Vj_old = v[:, :, pos, :]  # (B, Hkv, S, D)

                        if gqa_factor > 1:
                            # GQA-aware delta_v without repeat_interleave
                            # delta_v_kv: (B, Hkv, S, D)
                            delta_v_kv = v_current - Vj_old
                            # Expand via broadcast: self_w (B, Hq, S, 1) -> (B, Hkv, g, S, 1)
                            # delta_v_kv (B, Hkv, S, D) -> (B, Hkv, 1, S, D)
                            # Result: (B, Hkv, g, S, D) -> (B, Hq, S, D)
                            self_w_grouped = self_w.view(B, Hkv, gqa_factor, S, 1)
                            correction = (self_w_grouped * delta_v_kv.unsqueeze(2)).view(B, Hq, S, D)
                            attn_output[:, :, pos, :] += correction
                        else:
                            delta_v = v_current - Vj_old  # (B, Hkv, S, D)
                            attn_output[:, :, pos, :] += self_w * delta_v

                # Calculate gating score
                gating_score = attn_scores.mean().abs()

                return attn_output, attn_weights, gating_score
            else:
                # Fast path: use torch's optimized SDPA, then apply exact post-hoc correction
                # for seed positions if needed
                attn_output = scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=False,
                    enable_gqa=enable_gqa,
                )

                # Apply exact post-hoc diagonal correction for seed positions
                if (
                    seed_positions is not None
                    and k_current is not None
                    and v_current is not None
                    and seed_positions.numel() > 0
                ):
                    attn_output = self._apply_post_hoc_diagonal_correction(
                        attn_output=attn_output,
                        q=q,
                        k=k,
                        v=v,
                        k_current=k_current,
                        v_current=v_current,
                        seed_positions=seed_positions,
                        attn_mask=attn_mask,
                    )

                return attn_output, None, None

    def _apply_post_hoc_diagonal_correction(
        self,
        attn_output: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_current: torch.Tensor,
        v_current: torch.Tensor,
        seed_positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply exact post-hoc diagonal correction for seed positions.

        NOTE: seed_positions is expected to be pre-validated (0 <= pos < L).
        """
        # Assume seed_positions is already validated to avoid GPU sync from filtering
        pos = seed_positions.to(device=q.device, dtype=torch.long)
        return _post_hoc_diagonal_correction_exact(
            attn_output, q, k, v, k_current, v_current, pos, attn_mask
        )

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        replace_position: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]:
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None: #self.q_norm: None, self.k_norm: None
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        # self.config.n_heads: 32
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

        k_current_seeds: Optional[torch.Tensor] = None
        v_current_seeds: Optional[torch.Tensor] = None
        seed_positions_for_attn: Optional[torch.Tensor] = None

        if (
            self._seed_kv_positions is not None
            and self._seed_kv_key is not None
            and self._seed_kv_value is not None
            and self._seed_kv_positions.numel() > 0
        ):
            seed_positions = self._seed_kv_positions
            # 避免不必要的.to()操作 - 只在设备/类型不匹配时转换
            if seed_positions.device != k.device or seed_positions.dtype != torch.long:
                seed_positions = seed_positions.to(device=k.device, dtype=torch.long)

            seed_positions_for_attn = seed_positions
            seed_k = self._seed_kv_key
            seed_v = self._seed_kv_value

            if seed_k.dim() == 3:
                seed_k = seed_k.unsqueeze(0)
                seed_v = seed_v.unsqueeze(0)
            if seed_k.size(0) == 1 and B > 1:
                seed_k = seed_k.expand(B, -1, -1, -1)
                seed_v = seed_v.expand(B, -1, -1, -1)

            # 只在必要时进行类型转换
            if seed_k.dtype != k.dtype or seed_k.device != k.device:
                seed_k = seed_k.to(device=k.device, dtype=k.dtype)
            if seed_v.dtype != v.dtype or seed_v.device != v.device:
                seed_v = seed_v.to(device=v.device, dtype=v.dtype)

            # High-impact optimization:
            # - Only save the *seed* positions' current K/V (for diagonal correction)
            # - Then overwrite k/v in-place (avoid full clone per layer)
            if seed_positions.numel() > 0:
                k_current_seeds = k.index_select(2, seed_positions)
                v_current_seeds = v.index_select(2, seed_positions)
                k.index_copy_(2, seed_positions, seed_k)
                v.index_copy_(2, seed_positions, seed_v)

        if layer_past is not None:
            past_key, past_value = layer_past
            if replace_position is None:
                k = torch.cat((past_key, k), dim=-2)
                v = torch.cat((past_value, v), dim=-2)
            else:
                # k shape is [B, n_kv_h, selected_length, hs]
                # replace_position shape is [B, L], where L contains 0s and 1s
                # past_key shape is [B, n_kv_h, L, hs]
                # Replace selected_length number of 1s in past_key with k

                # Vectorized approach: use masked_scatter_ to avoid Python loop and nonzero
                # For B=1 (common case), this is especially efficient
                B_rp, L_rp = replace_position.shape
                _, n_kv_h, sel_len, hs = k.shape

                # Expand replace_position mask to match past_key shape: (B, n_kv_h, L, hs)
                rp_mask = replace_position.to(dtype=torch.bool)
                rp_mask_expanded = rp_mask.unsqueeze(1).unsqueeze(-1).expand(B_rp, n_kv_h, L_rp, hs)

                # masked_scatter_ replaces True positions with values from k/v in order
                past_key.masked_scatter_(rp_mask_expanded, k)
                past_value.masked_scatter_(rp_mask_expanded, v)

                k = past_key
                v = past_value

        if self._save_kv_for_seed:
            self._last_k = k
            self._last_v = v

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None

        if self.config.rope:
            # Apply rotary embeddings.
            if replace_position is None:
                q, k = self.rotary_emb(q, k)
            else:
                # For batched replace_position, use the maximum position across all batches
                # Keep it tensor-based to avoid host sync from `.any()` / `nonzero()`.
                rp = replace_position.to(dtype=torch.bool)
                seq_idx = torch.arange(rp.size(-1), device=rp.device, dtype=torch.long)
                masked_idx = torch.where(rp, seq_idx, seq_idx.new_full(seq_idx.shape, -1))
                max_idx = masked_idx.max()
                max_replace_pos = torch.where(
                    max_idx >= 0,
                    max_idx + 1,
                    torch.tensor(key_len, device=rp.device, dtype=torch.long),
                )
                q, k = self.rotary_emb(q, k, max_replace_pos)

        k_current_for_attn = None
        v_current_for_attn = None
        if (
            seed_positions_for_attn is not None
            and seed_positions_for_attn.numel() > 0
            and k_current_seeds is not None
            and v_current_seeds is not None
            and key_len == T
        ):
            v_current_for_attn = v_current_seeds
            if self.config.rope:
                # Optimization: Only apply RoPE to seed positions, not the entire k_current
                # This reduces O(L × D) to O(S × D) where S << L
                k_current_seeds_ = k_current_seeds.float() if self.config.rope_full_precision else k_current_seeds
                with torch.autocast(q.device.type, enabled=False):
                    pos_sin, pos_cos = self.rotary_emb.get_rotary_embedding(key_len, q.device)
                    pos_sin = pos_sin.type_as(k_current_seeds_)
                    pos_cos = pos_cos.type_as(k_current_seeds_)
                    # Only extract sin/cos for seed positions
                    pos_sin_seeds = pos_sin[:, :, seed_positions_for_attn, :]
                    pos_cos_seeds = pos_cos[:, :, seed_positions_for_attn, :]
                    k_current_seeds_rope = self.rotary_emb.apply_rotary_pos_emb(pos_sin_seeds, pos_cos_seeds, k_current_seeds_)
                # Create k_current_for_attn with only the seed positions we need
                # Instead of storing the full tensor, we store just the seed positions
                k_current_for_attn = k_current_seeds_rope.type_as(k_current_seeds)
            else:
                k_current_for_attn = k_current_seeds

        if attention_bias is not None:
            # Resize and cast attention bias.
            # The current dtype of the attention bias might not match the dtype that the SDP attn function will
            # run in if AMP is enabled, and this can be a problem if some tokens are masked out due to padding
            # as down-casting the attention bias to the autocast precision will result in -infs, which will
            # cause the SDP attn function to produce NaNs.
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
            )

        # Get the attention scores.
        # shape: (B, nh, T, hs), Optional[(B, nh, T, T)], Optional[scalar]
        att, attn_weights, gating_score = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=False,
            output_attentions=output_attentions,
            seed_positions=seed_positions_for_attn,
            k_current=k_current_for_attn,
            v_current=v_current_for_attn,
        )
        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection.
        return self.attn_out(att), present, attn_weights, gating_score


    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        replace_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]:
        raise NotImplementedError

    @classmethod
    def build(cls, layer_id: int, config: ModelConfig, cache: BufferCache) -> LLaDABlock:
        if config.block_type == BlockType.sequential:
            return LLaDASequentialBlock(layer_id, config, cache)
        elif config.block_type == BlockType.llama:
            return LLaDALlamaBlock(layer_id, config, cache)
        else:
            raise NotImplementedError(f"Unknown block type: '{config.block_type}'")


class LLaDASequentialBlock(LLaDABlock):
    """
    This is a typical transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        # Attention input projection. Projects x -> (q, k, v)
        head_dim = config.d_model // config.n_heads
        self.fused_dims = (
            config.d_model,
            config.effective_n_kv_heads * head_dim,
            config.effective_n_kv_heads * head_dim,
        )
        self.att_proj = nn.Linear(
            config.d_model, sum(self.fused_dims), bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )
        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(
            self.config, self.att_proj, d=self.config.d_model, layer_id=None, type_of_module=ModuleType.in_module
        )
        init_weights(
            self.config, self.ff_proj, d=self.config.d_model, layer_id=None, type_of_module=ModuleType.in_module
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        replace_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        #  - for group query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_kv_heads)
        if self._activation_checkpoint_fn is not None:
            q, k, v = self.att_proj(self._activation_checkpoint_fn(self.attn_norm, x)).split(
                self.fused_dims, dim=-1
            )
        else:
            q, k, v = self.att_proj(self.attn_norm(x)).split(self.fused_dims, dim=-1)

        # Get attention scores.
        if self._activation_checkpoint_fn is not None:
            att, cache, attn_weights, gating_score = self._activation_checkpoint_fn(  # type: ignore
                self.attention, q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache, replace_position=replace_position, output_attentions=output_attentions
            )
        else:
            att, cache, attn_weights, gating_score = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache, replace_position=replace_position, output_attentions=output_attentions)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        x = self.ff_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        return x, cache, attn_weights, gating_score


class LLaDALlamaBlock(LLaDABlock):
    """
    This is a transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). This block is similar to `LLaDASequentialBlock`
    but some operations have slightly different implementations to imitate the
    behavior of Llama.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache

        # Attention input projection. Projects x -> (q, k, v)
        head_dim = config.d_model // config.n_heads
        q_proj_out_dim = config.d_model
        k_proj_out_dim = config.effective_n_kv_heads * head_dim
        v_proj_out_dim = config.effective_n_kv_heads * head_dim
        self.q_proj = nn.Linear(
            config.d_model, q_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )
        self.k_proj = nn.Linear(
            config.d_model, k_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )
        self.v_proj = nn.Linear(
            config.d_model, v_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )

        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )
        # new add
        self.up_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(self.config, self.q_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.k_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.v_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.up_proj, d=self.config.d_model, layer_id=None)  # new add

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        replace_position: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        #  - for group query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_kv_heads)
        x_normed = self.attn_norm(x) #x:torch.Size([2, 168, 4096])
        q = self.q_proj(x_normed) #q:torch.Size([2, 168, 4096])
        k = self.k_proj(x_normed) #k:torch.Size([2, 168, 4096])
        v = self.v_proj(x_normed) #v:torch.Size([2, 168, 4096])
        # attention_bias: None
        # layer_past: None
        # use_cache: False
        # Get attention scores.
        if self._activation_checkpoint_fn is not None:
            att, cache, attn_weights, gating_score = self._activation_checkpoint_fn(  # type: ignore
                self.attention, q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache, replace_position=replace_position, output_attentions=output_attentions
            )
        else:
            att, cache, attn_weights, gating_score = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache, replace_position=replace_position, output_attentions=output_attentions)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        x, x_up = self.ff_proj(x), self.up_proj(x) # new add
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = x * x_up # new add
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        return x, cache, attn_weights, gating_score


class LLaDABlockDiffBlock(LLaDABlock):
    """
    This is a transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). This block is similar to `LLaDASequentialBlock`
    but some operations have slightly different implementations to imitate the
    behavior of Llama.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache

        # Attention input projection. Projects x -> (q, k, v)
        head_dim = config.d_model // config.n_heads
        q_proj_out_dim = config.d_model
        k_proj_out_dim = config.effective_n_kv_heads * head_dim
        v_proj_out_dim = config.effective_n_kv_heads * head_dim
        self.q_proj = nn.Linear(
            config.d_model, q_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )
        self.k_proj = nn.Linear(
            config.d_model, k_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )
        self.v_proj = nn.Linear(
            config.d_model, v_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )

        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )
        # new add
        self.up_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(self.config, self.q_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.k_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.v_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.up_proj, d=self.config.d_model, layer_id=None)  # new add

    def cross_attn_flex(self, qkv, mask=None):
        qkv = rearrange(qkv, 'b s three h d -> b h three s d', h=self.n_heads)
        x = fused_flex_attention(
        qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], mask=mask)
        x = rearrange(x, 'b h s d -> b s (h d)')
        return x

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        replace_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        #  - for group query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_kv_heads)
        x_normed = self.attn_norm(x)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)

        # Get attention scores.
        if self._activation_checkpoint_fn is not None:
            att, cache, attn_weights, gating_score = self._activation_checkpoint_fn(  # type: ignore
                self.attention, q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache, replace_position=replace_position, output_attentions=output_attentions
            )
        else:
            att, cache, attn_weights, gating_score = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache, replace_position=replace_position, output_attentions=output_attentions)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        x, x_up = self.ff_proj(x), self.up_proj(x) # new add
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = x * x_up # new add
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        return x, cache, attn_weights, gating_score


class LLaDAOutput(NamedTuple):
    logits: torch.FloatTensor
    """
    A tensor of shape `(batch_size, seq_len, vocab_size)` representing the log probabilities
    for the next token *before* normalization via (log) softmax.
    """

    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    """
    Attention keys and values from each block.
    """

    hidden_states: Optional[Tuple[torch.Tensor]]
    """
    Hidden states from each block.
    """

    attentions: Optional[Tuple[torch.Tensor]]
    """
    Attention weights from each block.
    A tuple of tensors of shape `(batch_size, num_heads, seq_len, seq_len)`.
    """

    gating_scores: Optional[Tuple[torch.Tensor]] = None
    """
    Gating scores from each block.
    """


@dataclass
class LLaDACausalLMOutput(CausalLMOutputWithPast):
    gating_scores: Optional[Tuple[torch.Tensor]] = None
    """
    Gating scores from each block.
    """


class LLaDAGenerateOutput(NamedTuple):
    token_ids: torch.LongTensor
    """
    The generated token IDs, a tensor of shape `(batch_size, beam_size, max_steps)`.
    These do *not* include the original input IDs.
    """

    scores: torch.FloatTensor
    """
    The scores of the generated sequences, a tensor of shape `(batch_size, beam_size)`.
    """


class LLaDABlockGroup(nn.ModuleList):
    def __init__(self, config: ModelConfig, layer_offset: int, modules: Optional[Iterable[nn.Module]] = None):
        super().__init__(modules)
        self.config = config
        self.layer_offset = layer_offset
        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn = activation_checkpoint_function(self.config)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layers_past: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        replace_position: Optional[torch.Tensor] = None,
        output_attentions_last_only: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]], Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        collect_attentions = output_attentions or output_attentions_last_only
        all_attentions: Optional[List[torch.Tensor]] = [] if collect_attentions else None
        all_gating_scores: Optional[List[torch.Tensor]] = [] if collect_attentions else None
        for local_block_idx, block in enumerate(self):
            layer_past = None if layers_past is None else layers_past[local_block_idx]
            block_idx = local_block_idx + self.layer_offset
            if (
                (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.whole_layer)
                or (
                    self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_two
                    and block_idx % 2 == 0
                )
                or (
                    self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_three
                    and block_idx % 3 == 0
                )
                or (
                    self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_four
                    and block_idx % 4 == 0
                )
            ):
                # shape: (batch_size, seq_len, d_model)
                x, cache, attn_weights, gating_score = self._activation_checkpoint_fn(  # type: ignore
                    block, x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache, output_attentions=collect_attentions, replace_position=replace_position
                )
            else:
                # shape: (batch_size, seq_len, d_model)
                x, cache, attn_weights, gating_score = block(x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache, output_attentions=collect_attentions, replace_position=replace_position)
            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)
            should_keep_attention = output_attentions or (
                output_attentions_last_only and local_block_idx == len(self) - 1
            )
            if should_keep_attention and all_attentions is not None and attn_weights is not None:
                all_attentions.append(attn_weights)
            if should_keep_attention and all_gating_scores is not None and gating_score is not None:
                all_gating_scores.append(gating_score)
        return x, attn_key_values, all_attentions, all_gating_scores



    def reset_parameters(self):
        for block in self:
            block.reset_parameters()

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self.activation_checkpointing_strategy = strategy
        for block in self:
            block.set_activation_checkpointing(strategy)


class LLaDAModel(nn.Module):
    def __init__(self, config: ModelConfig, init_params: bool = True):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()

        # Validate config.
        if self.config.alibi and self.config.flash_attention:
            raise Exception("ALiBi is currently not supported with FlashAttention")

        if self.config.alibi and self.config.rope:
            raise Exception("ALiBi and RoPE are mutually exclusive")

        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise Exception("embedding size should be at least as big as vocab size")
            elif self.config.embedding_size % 128 != 0:
                import warnings

                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning
                )

        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn: Callable = activation_checkpoint_function(self.config)

        if not (
            0 < self.config.block_group_size <= self.config.n_layers
            and self.config.n_layers % self.config.block_group_size == 0
        ):
            raise Exception("n layers must be divisible by block group size")

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # this is super slow so make sure torch won't use it

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.embedding_size or config.vocab_size, config.d_model, device=config.init_device
                ),
                emb_drop=Dropout(config.embedding_dropout),
                ln_f=LayerNorm.build(config),
            )
        )

        blocks = [LLaDABlock.build(i, config, self.__cache) for i in range(config.n_layers)]
        if self.config.block_group_size > 1:
            block_groups = [
                LLaDABlockGroup(config, i, blocks[i : i + config.block_group_size])
                for i in range(0, config.n_layers, config.block_group_size)
            ]
            self.transformer.update({"block_groups": nn.ModuleList(block_groups)})
        else:
            self.transformer.update({"blocks": nn.ModuleList(blocks)})

        if not (self.config.alibi or self.config.rope):
            self.transformer.update(
                {"wpe": nn.Embedding(config.max_sequence_length, config.d_model, device=config.init_device)}
            )
        if not config.weight_tying:
            self.transformer.update(
                {
                    "ff_out": nn.Linear(
                        config.d_model,
                        config.embedding_size or config.vocab_size,
                        bias=config.include_bias,
                        device=config.init_device,
                    )
                }
            )
        # When `init_device="meta"` FSDP will call `reset_parameters()` to initialize weights.
        if init_params and self.config.init_device != "meta":
            self.reset_parameters()
        self.__num_fwd_flops: Optional[int] = None
        # Warm up cache.
        if self.config.alibi:
            get_causal_attention_bias(self.__cache, config.max_sequence_length, _non_meta_init_device(config))
            self.get_alibi_attention_bias(config.max_sequence_length, _non_meta_init_device(config))


    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self.activation_checkpointing_strategy = strategy
        if self.config.block_group_size != 1:
            for block_group in self.transformer.block_groups:
                block_group.set_activation_checkpointing(strategy)
        else:
            for block in self.transformer.blocks:
                block.set_activation_checkpointing(strategy)

    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.wte.weight.device  # type: ignore
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        else:
            return device

    def reset_parameters(self):
        log.info("Initializing model parameters...")
        # Top-level embeddings / linear layers.
        init_weights(
            self.config,
            self.transformer.wte,  # type: ignore
            std_factor=(0.5 * math.sqrt(self.config.d_model)) if self.config.scale_logits else 1.0,
            type_of_module=ModuleType.emb,
        )
        if hasattr(self.transformer, "wpe"):
            init_weights(self.config, self.transformer.wpe, type_of_module=ModuleType.emb)  # type: ignore

        # Top-level layer norm.
        self.transformer.ln_f.reset_parameters()  # type: ignore

        # Output weights.
        if hasattr(self.transformer, "ff_out"):
            init_weights(self.config, self.transformer.ff_out, type_of_module=ModuleType.final_out)  # type: ignore

        # Let the blocks handle themselves.
        if self.config.block_group_size == 1:
            for block in self.transformer.blocks:
                block.reset_parameters()
        else:
            for block_group in self.transformer.block_groups:
                block_group.reset_parameters()

    def get_alibi_attention_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if (alibi_bias := self.__cache.get("alibi_attention_bias")) is not None and alibi_bias.shape[
            -1
        ] >= seq_len:
            if alibi_bias.device != device:
                alibi_bias = alibi_bias.to(device)
                self.__cache["alibi_attention_bias"] = alibi_bias
            return alibi_bias
        with torch.autocast(device.type, enabled=False):
            alibi_bias = alibi_attention_bias(seq_len, self.config, device)
        self.__cache["alibi_attention_bias"] = alibi_bias
        return alibi_bias

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_attentions_last_only: Optional[bool] = None,
        replace_position: Optional[torch.Tensor] = None,
    ) -> LLaDAOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param input_embeddings: A tensor of shape `(batch_size, seq_len, d_model)` with input
            embeddings. When provided, it is treated as the output of the input embedding layer.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            which input IDs are masked. A `1` value in the mask means that
            the corresponding input ID should *not* be ignored. A `0` means
            that the corresponding input ID is masked.

            This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
            library.
        :param attention_bias: A tensor of shape `(batch_size, 1, seq_len, seq_len)`,
            `(1, 1, seq_len, seq_len)`, or `(seq_len, seq_len)`. This is used
            to introduce causal or other biases.

            If the tensor is a bool or byte tensor, a `True` or `1` at `attention_bias[:, :, i, j]`
            indicates that the i-th element in the sequence is allowed to attend to the j-th
            element in the sequence.

            If the tensor is a float tensor, it will just be added to the attention
            scores before the softmax.

            The default is causal, which corresponds to a lower-diagonal byte matrix of ones.
        :param past_key_values: Pre-computed keys and values for each attention block.
            Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        :param use_cache: If `True`, return key and value tensors for each block.
        :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        :param output_attentions: If `True`, return attention weights from each block.
        """
        # Add Basic MDM Model config check
        assert not self.config.alibi, "Alibi length extrapolation is not supported for MDM."
        assert self.config.rope, "Rope must be used in Llama-Encoder for MDM."
        # assert (past_key_values is None and not use_cache), "The kvcache is not suppotred for MDM."

        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        output_attentions = output_attentions if output_attentions is not None else False
        output_attentions_last_only = (
            output_attentions_last_only if output_attentions_last_only is not None else False
        )
        collect_attentions = output_attentions or output_attentions_last_only

        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings  # type: ignore

        if self.config.input_emb_norm:
            x = x * (self.config.d_model**0.5)

        if not (self.config.alibi or self.config.rope):
            # Get positional embeddings.
            # shape: (1, seq_len)
            pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            # shape: (1, seq_len, d_model)
            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = pos_emb + x

        # Add input + positional embeddings and apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        # Transform the attention mask into what the blocks expect.
        if attention_mask is not None and 0.0 in attention_mask:
            # shape: (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
        else:
            attention_mask = None

        # Merge attention mask with attention bias.
        if (
            attention_bias is not None
            or attention_mask is not None
            or self.config.alibi
            # NOTE (epwalsh): we need to initialize the attn bias in order for attn to work properly
            # with key+value cache. Otherwise `F.scaled_dot_product_attention()` doesn't seem to compute
            # scores correctly.
            or past_key_values is not None
        ):
            if attention_bias is None and self.config.alibi:
                attention_bias = get_causal_attention_bias(
                    self.__cache, past_length + seq_len, x.device
                ) + self.get_alibi_attention_bias(past_length + seq_len, x.device)
            elif attention_bias is None:
                attention_bias = get_causal_attention_bias(self.__cache, past_length + seq_len, x.device)
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)

            # Transform to the right shape and data type.
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)

            # Add in the masking bias.
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                # Might get -infs after adding attention mask, since dtype.min + dtype.min = -inf.
                # `F.scaled_dot_product_attention()` doesn't handle -inf like you'd expect, instead
                # it can produce NaNs.
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        all_attentions: Optional[List[torch.Tensor]] = [] if collect_attentions else None
        all_gating_scores: Optional[List[torch.Tensor]] = [] if collect_attentions else None

        # decoder layers
        all_hidden_states = []

        # Apply blocks one-by-one.
        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layer_past = None if past_key_values is None else past_key_values[block_idx]
                if (
                    (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.whole_layer)
                    or (
                        self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_two
                        and block_idx % 2 == 0
                    )
                    or (
                        self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_three
                        and block_idx % 3 == 0
                    )
                    or (
                        self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_four
                        and block_idx % 4 == 0
                    )
                ):
                    # shape: (batch_size, seq_len, d_model)
                    x, cache, attn_weights, gating_score = self._activation_checkpoint_fn(
                        block, x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache, replace_position=replace_position, output_attentions=collect_attentions
                    )
                else:
                    # shape: (batch_size, seq_len, d_model)
                    x, cache, attn_weights, gating_score = block(x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache, replace_position=replace_position, output_attentions=collect_attentions)
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.append(cache)
                should_keep_attention = output_attentions or (
                    output_attentions_last_only and block_idx == self.config.n_layers - 1
                )
                if should_keep_attention and all_attentions is not None and attn_weights is not None:
                    all_attentions.append(attn_weights)
                if should_keep_attention and all_gating_scores is not None and gating_score is not None:
                    all_gating_scores.append(gating_score)
        else:
            for group_idx, block_group in enumerate(self.transformer.block_groups):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layers_past = (
                    None
                    if past_key_values is None
                    else past_key_values[
                        group_idx * self.config.block_group_size : (group_idx + 1) * self.config.block_group_size
                    ]
                )
                x, cache, group_attentions, group_gating_scores = block_group(
                    x,
                    attention_bias=attention_bias,
                    layers_past=layers_past,
                    use_cache=use_cache,
                    output_attentions=collect_attentions,
                    output_attentions_last_only=output_attentions_last_only and group_idx == len(self.transformer.block_groups) - 1,
                )
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.extend(cache)
                if all_attentions is not None and group_attentions is not None:
                    all_attentions.extend(group_attentions)
                if all_gating_scores is not None and group_gating_scores is not None:
                    all_gating_scores.extend(group_gating_scores)

        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            x = x[:, -1, :].unsqueeze(1)

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore
        if output_hidden_states:
            # add final hidden state post-final-layernorm, following HuggingFace's convention
            all_hidden_states.append(x)

        # Get logits.
        # shape: (batch_size, seq_len or 1, vocab_size)
        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore
        else:
            logits = self.transformer.ff_out(x)  # type: ignore
        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))

        return LLaDAOutput(
            logits=logits,
            attn_key_values=attn_key_values,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attentions=tuple(all_attentions) if collect_attentions else None,
            gating_scores=tuple(all_gating_scores) if collect_attentions else None
        )  # type: ignore[arg-type]



def create_model_config_from_pretrained_config(config: LLaDAConfig):
    """
    Utility function
    """

    kwargs = {}
    for field in fields(ModelConfig):
        kwargs[field.name] = getattr(config, field.name)

    model_config = ModelConfig(**kwargs)
    return model_config


class LLaDAModelLM(PreTrainedModel):
    """
    Extremely barebones HF model wrapper.
    """

    config_class = LLaDAConfig
    base_model_prefix = "model"
    _no_split_modules = ["LLaDABlock", "LLaDASequentialBlock", "LLaDALlamaBlock"]

    def __init__(self, config: LLaDAConfig, model: Optional[LLaDAModel] = None, init_params: bool = False):
        super().__init__(config)

        if not model:
            model_config = create_model_config_from_pretrained_config(config)
            # Initialize model (always on CPU to start with so we don't run out of GPU memory).
            model_config.init_device = "cpu"
            self.model = LLaDAModel(model_config, init_params=init_params)
        else:
            self.model = model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        replace_position: Optional[torch.Tensor] = None,  # This is a hack mitigation of an issue in transformers `4.39.x`
        output_attentions_last_only: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if use_cache is None:
            use_cache = self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.forward(
            input_ids=input_ids,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            replace_position=replace_position,
            output_attentions_last_only=output_attentions_last_only,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states
        attentions = outputs.attentions

        loss = None
        if labels is not None:
            import warnings
            warnings.warn("Note that for LLaDA, you cannot calculate the loss here.", UserWarning)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LLaDACausalLMOutput(
            logits=logits,
            past_key_values=outputs.attn_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
            gating_scores=outputs.gating_scores,
        )


    def can_generate(self) -> bool:
        return True

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple]] = None, **kwargs
    ):
        if past_key_values:
            # This is because we want the model to only process the last generated token.
            input_ids = input_ids[:, -1:]
        model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values}

        model_inputs.update(kwargs)
        model_inputs["use_cache"] = kwargs.pop("use_cache", self.config.use_cache)
        return model_inputs


    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.transformer.wte

    def set_input_embeddings(self, value: torch.nn.Module):
        self.model.transformer.wte = value

    def get_output_embeddings(self):
        if self.config.weight_tying:
            return self.model.transformer.wte
        else:
            return self.model.transformer.ff_out

    def set_output_embeddings(self, value: torch.nn.Module):
        if self.config.weight_tying:
            self.model.transformer.wte = value
        else:
            self.model.transformer.ff_out = value

    def tie_weights(self):
        if self.config.weight_tying:
            self.model.transformer.ff_out = self.model.transformer.wte

# Register the model so that it is available for transformer pipelines, auto-loading, etc.
AutoModel.register(LLaDAConfig, LLaDAModelLM)
