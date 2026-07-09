# coding=utf-8
# Code copied from HuggingFace's transformers library
#
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch LLaMA model."""
import math
import warnings
from typing import Optional, Tuple
import logging
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.utils import (
    is_flash_attn_2_available,
    logging,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    is_flash_attn_2_available,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
    repeat_kv,
    apply_rotary_pos_emb,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaDecoderLayer,
)
from ..quantize import get_quantized_layer_cls, get_quantized_func
from ..utils import find_matched_pattern, get_layer_name

logger = logging.get_logger(__name__)


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func  # noqa
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


class LlamaQuantizedMLP(nn.Module):
    def __init__(self, config, qera_config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # fmt: off
        self.gate_proj = get_quantized_layer_cls("linear", q_config=qera_config["gate_proj"])(self.hidden_size, self.intermediate_size, bias=False, q_config=qera_config["gate_proj"])
        self.up_proj = get_quantized_layer_cls("linear", q_config=qera_config["up_proj"])(self.hidden_size, self.intermediate_size, bias=False, q_config=qera_config["up_proj"])
        self.down_proj = get_quantized_layer_cls("linear", q_config=qera_config["down_proj"])(self.intermediate_size, self.hidden_size, bias=False, q_config=qera_config["down_proj"])
        # fmt: on
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [
                    F.linear(x, gate_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )
            up_proj = torch.cat(
                [
                    F.linear(x, up_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaQuantizedAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int, qera_config: dict):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # if layer_idx is None:
        #     logger.warning_once(
        #         f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
        #         "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
        #         "when creating this class."
        #     )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # fmt: off
        self.q_proj = get_quantized_layer_cls("linear", q_config=qera_config["q_proj"])(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias, q_config=qera_config["q_proj"])
        self.k_proj = get_quantized_layer_cls("linear", q_config=qera_config["k_proj"])(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias, q_config=qera_config["k_proj"])
        self.v_proj = get_quantized_layer_cls("linear", q_config=qera_config["v_proj"])(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias, q_config=qera_config["v_proj"])
        self.o_proj = get_quantized_layer_cls("linear", q_config=qera_config["o_proj"])(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias, q_config=qera_config["o_proj"])
        self.qera_config = qera_config
        # fmt: on
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            # *: tensor parallelism, disable this for quantization
            raise ValueError("Quantization is not supported for tensor parallelism")
            # key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            # query_slices = self.q_proj.weight.split(
            #     (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            # )
            # key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            # value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            # query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            # query_states = torch.cat(query_states, dim=-1)

            # key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            # key_states = torch.cat(key_states, dim=-1)

            # value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            # value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # *: matmul of QK^T
        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        query_states = query_states.reshape(bsz * self.num_heads, q_len, self.head_dim)
        key_states = key_states.reshape(bsz * self.num_heads, kv_seq_len, self.head_dim)
        attn_weights = get_quantized_func(
            "matmul", q_config=self.qera_config["matmul_0"]
        )(
            query_states,
            key_states.transpose(1, 2),
            q_config=self.qera_config["matmul_0"],
        ) / math.sqrt(
            self.head_dim
        )
        attn_weights = attn_weights.reshape(bsz, self.num_heads, q_len, kv_seq_len)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        # *: matmul of attention weights and V
        # attn_output = torch.matmul(attn_weights, value_states)
        attn_weights = attn_weights.reshape(bsz * self.num_heads, q_len, kv_seq_len)
        value_states = value_states.reshape(
            bsz * self.num_heads, kv_seq_len, self.head_dim
        )
        attn_output = get_quantized_func(
            "matmul", q_config=self.qera_config["matmul_0"]
        )(attn_weights, value_states, q_config=self.qera_config["matmul_1"])
        attn_output = attn_output.reshape(bsz, self.num_heads, q_len, self.head_dim)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            # *: tensor parallelism, disable this for quantization
            raise ValueError("Quantization is not supported for tensor parallelism")
            # attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            # o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            # attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


LLAMA_QUANTIZED_ATTENTION_CLASSES = {
    "eager": LlamaQuantizedAttention,
    "flash_attention_2": None,
    "sdpa": None,
}


class LlamaQuantizedDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int, qera_config: dict):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config._attn_implementation in ["flash_attention_2", "sdpa"]:
            raise ValueError(
                f"Attention implementation {config._attn_implementation} is not supported for LlamaQuantizedDecoderLayer"
            )
        self.self_attn = LLAMA_QUANTIZED_ATTENTION_CLASSES[config._attn_implementation](
            config=config,
            layer_idx=layer_idx,
            qera_config=qera_config["self_attn"],
        )

        self.mlp = LlamaQuantizedMLP(
            config,
            qera_config=qera_config["mlp"],
        )
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


def build_qera_config_llama(model: LlamaForCausalLM, qera_config: dict):
    parsed_config = {}

    decoder_layer_i: LlamaDecoderLayer
    for i, decoder_layer_i in enumerate(model.model.layers):
        parsed_config[f"model_layer_{i}"] = {"self_attn": {}, "mlp": {}}
        for fc_short_name in ["k_proj", "q_proj", "v_proj", "o_proj"]:
            fc_name = get_layer_name(
                model, getattr(decoder_layer_i.self_attn, fc_short_name)
            )
            matched_entry = find_matched_pattern(fc_name, qera_config.keys())
            assert matched_entry is not None, f"Cannot find matched entry for {fc_name}"
            if isinstance(qera_config[matched_entry], str):
                matched_entry = qera_config[matched_entry]
            parsed_config[f"model_layer_{i}"]["self_attn"][fc_short_name] = deepcopy(
                qera_config[matched_entry]
            )
        for matmul_short_name in ["matmul_0", "matmul_1"]:
            matmul_name = fc_name.replace("o_proj", matmul_short_name)
            matched_entry = find_matched_pattern(matmul_name, qera_config.keys())
            assert (
                matched_entry is not None
            ), f"Cannot find matched entry for {matmul_name}"
            if isinstance(qera_config[matched_entry], str):
                matched_entry = qera_config[matched_entry]
            parsed_config[f"model_layer_{i}"]["self_attn"][matmul_short_name] = (
                deepcopy(qera_config[matched_entry])
            )
        for fc_short_name in ["gate_proj", "up_proj", "down_proj"]:
            fc_name = get_layer_name(model, getattr(decoder_layer_i.mlp, fc_short_name))
            matched_entry = find_matched_pattern(fc_name, qera_config.keys())
            assert matched_entry is not None, f"Cannot find matched entry for {fc_name}"
            if isinstance(qera_config[matched_entry], str):
                matched_entry = qera_config[matched_entry]
            parsed_config[f"model_layer_{i}"]["mlp"][fc_short_name] = deepcopy(
                qera_config[matched_entry]
            )

    return parsed_config


def quantize_llama_model(
    model: LlamaForCausalLM | LlamaForSequenceClassification,
    qera_config: dict,
):
    qera_config = build_qera_config_llama(model, qera_config)

    for layer_id, ori_decoder_layer in enumerate(model.model.layers):
        layer_entry = f"model_layer_{layer_id}"
        layer_qera_config = qera_config[layer_entry]

        # replace the decoder layer with quantized decoder layer
        new_decoder_layer = LlamaQuantizedDecoderLayer(
            model.config, layer_id, layer_qera_config
        )
        ori_rope = ori_decoder_layer.self_attn.rotary_emb
        new_decoder_layer.to(next(iter(ori_decoder_layer.parameters())).dtype)
        new_decoder_layer.self_attn.rotary_emb = ori_rope
        new_decoder_layer.to(next(iter(ori_decoder_layer.parameters())).device)
        new_decoder_layer.load_state_dict(ori_decoder_layer.state_dict(), strict=False)
        model.model.layers[layer_id] = new_decoder_layer
        # remove the original layer
        del ori_decoder_layer

    model._no_split_modules = ["LlamaDecoderLayer", "LlamaQuantizedDecoderLayer"]

    return model


def find_layers_to_register_scale_hook_llama(
    model: LlamaForCausalLM,
) -> list[dict[str, str | list[str]]]:
    """
    return a list of dict, each dict contains the following keys:

    ```
    {
        "target_layer": ...,
        "layers_sharing_scale": [...],
    }
    ```
    """

    assert model.config._attn_implementation == "eager"
    layers_to_register = []

    for decoder_layer in model.model.layers:
        k_name = get_layer_name(model, decoder_layer.self_attn.k_proj)
        q_name = get_layer_name(model, decoder_layer.self_attn.q_proj)
        v_name = get_layer_name(model, decoder_layer.self_attn.v_proj)
        layers_to_register.append(
            dict(target_layer=k_name, layers_sharing_scale=[q_name, v_name]),
        )

        # v_name = get_layer_name(model, decoder_layer.self_attn.v_proj)
        # layers_to_register.append(
        # dict(target_layer=v_name, layers_sharing_scale=[]),
        # )

        o_name = get_layer_name(model, decoder_layer.self_attn.o_proj)
        layers_to_register.append(
            dict(target_layer=o_name, layers_sharing_scale=[]),
        )

        gate_name = get_layer_name(model, decoder_layer.mlp.gate_proj)
        up_name = get_layer_name(model, decoder_layer.mlp.up_proj)
        layers_to_register.append(
            dict(target_layer=gate_name, layers_sharing_scale=[up_name]),
        )

        down_name = get_layer_name(model, decoder_layer.mlp.down_proj)
        layers_to_register.append(
            dict(target_layer=down_name, layers_sharing_scale=[]),
        )

    return layers_to_register


def find_layers_to_approximate_llama(model: LlamaForCausalLM):
    layers_to_approximate = []

    for layer_name, layer in model.named_modules():
        if not isinstance(layer, nn.Linear):
            continue
        if "lm_head" in layer_name:
            continue

        layers_to_approximate.append(layer_name)

    return layers_to_approximate
