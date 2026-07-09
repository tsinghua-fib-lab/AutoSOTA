import logging
from typing import Optional, Tuple
import math
from copy import deepcopy

import torch
from torch import nn
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2RMSNorm,
    Gemma2RotaryEmbedding,
    Gemma2DecoderLayer,
    Gemma2Config,
    ACT2FN,
    Cache,
    apply_rotary_pos_emb,
    repeat_kv,
    Gemma2ForCausalLM,
)

from ..quantize import get_quantized_layer_cls, get_quantized_func
from ..utils import find_matched_pattern, get_layer_name

logger = logging.getLogger(__name__)


class Gemma2QuantizedMLP(nn.Module):
    def __init__(self, config, qera_config: dict):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # fmt: off
        self.gate_proj = get_quantized_layer_cls("linear", q_config=qera_config["gate_proj"])(self.hidden_size, self.intermediate_size, bias=False, q_config=qera_config["gate_proj"])
        self.up_proj = get_quantized_layer_cls("linear", q_config=qera_config["up_proj"])(self.hidden_size, self.intermediate_size, bias=False, q_config=qera_config["up_proj"])
        self.down_proj = get_quantized_layer_cls("linear", q_config=qera_config["down_proj"])(self.intermediate_size, self.hidden_size, bias=False, q_config=qera_config["down_proj"])
        # fmt: on
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Gemma2QuantizedAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Gemma2Config, layer_idx: int, qera_config: dict):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.scaling = config.query_pre_attn_scalar**-0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        # self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        # fmt: off
        self.q_proj = get_quantized_layer_cls("linear", q_config=qera_config["q_proj"])(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias, q_config=qera_config["q_proj"])
        self.k_proj = get_quantized_layer_cls("linear", q_config=qera_config["k_proj"])(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias, q_config=qera_config["k_proj"])
        self.v_proj = get_quantized_layer_cls("linear", q_config=qera_config["v_proj"])(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias, q_config=qera_config["v_proj"])
        self.o_proj = get_quantized_layer_cls("linear", q_config=qera_config["o_proj"])(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias, q_config=qera_config["o_proj"])
        self.qera_config = qera_config
        # fmt: on
        self.rotary_emb = Gemma2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        self.sliding_window = config.sliding_window if not bool(layer_idx % 2) else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

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

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "sliding_window": self.sliding_window,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # *: matmul_0
        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        # fmt: off
        kv_seq_len = key_states.size(2)
        query_states = query_states.reshape(bsz * self.num_heads, q_len, self.head_dim)
        key_states = key_states.reshape(bsz * self.num_heads, kv_seq_len, self.head_dim)
        attn_weights = get_quantized_func("matmul", q_config=self.qera_config["matmul_0"])(
            query_states, key_states.transpose(1, 2), q_config=self.qera_config["matmul_0"]
        ) * self.scaling
        attn_weights = attn_weights.reshape(bsz, self.num_heads, q_len, kv_seq_len)
        # fmt: on

        if self.config.attn_logit_softcapping is not None:
            attn_weights = attn_weights / self.config.attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.config.attn_logit_softcapping

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        # *: matmul_1
        # attn_output = torch.matmul(attn_weights, value_states)
        attn_weights = attn_weights.reshape(bsz * self.num_heads, q_len, kv_seq_len)
        value_states = value_states.reshape(
            bsz * self.num_heads, kv_seq_len, self.head_dim
        )
        attn_output = get_quantized_func(
            "matmul", q_config=self.qera_config["matmul_1"]
        )(attn_weights, value_states, q_config=self.qera_config["matmul_1"])
        attn_output = attn_output.reshape(bsz, self.num_heads, q_len, self.head_dim)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


GEMMA2_ATTENTION_CLASSES = {
    "eager": Gemma2QuantizedAttention,
}


class Gemma2QuantizedDecoderLayer(nn.Module):
    def __init__(self, config: Gemma2Config, layer_idx: int, qera_config: dict):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = GEMMA2_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx, qera_config=qera_config["self_attn"]
        )

        self.mlp = Gemma2QuantizedMLP(config, qera_config=qera_config["mlp"])
        self.input_layernorm = Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.is_sliding = not bool(layer_idx % 2)
        self.pre_feedforward_layernorm = Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.sliding_window = config.sliding_window

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        if (
            self.is_sliding and attention_mask is not None
        ):  # efficient SDPA and no padding
            # Flash-attn is a 2D tensor
            if self.config._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask[:, -self.sliding_window :]
            else:
                min_dtype = torch.finfo(hidden_states.dtype).min
                sliding_window_mask = torch.tril(
                    torch.ones_like(attention_mask, dtype=torch.bool),
                    diagonal=-self.sliding_window,
                )
                attention_mask = torch.where(
                    sliding_window_mask, min_dtype, attention_mask
                )
                if attention_mask.shape[-1] <= 1:  # when decoding
                    attention_mask = attention_mask[:, :, :, -self.sliding_window :]

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
            cache_position=cache_position,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


def build_qera_config_gemma2(model: Gemma2ForCausalLM, qera_config: dict):
    parsed_config = {}

    decoder_layer_i: Gemma2DecoderLayer
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


def quantize_gemma2_model(model: Gemma2ForCausalLM, qera_config: dict):
    qera_config = build_qera_config_gemma2(model, qera_config)

    for layer_id, ori_decoder_layer in enumerate(model.model.layers):
        layer_entry = f"model_layer_{layer_id}"
        layer_qera_config = qera_config[layer_entry]

        # replace the decoder layer with quantized decoder layer
        new_decoder_layer = Gemma2QuantizedDecoderLayer(
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


def find_layers_to_register_scale_hook_gemma2(
    model: Gemma2ForCausalLM,
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


def find_layers_to_approximate_gemma2(model: Gemma2ForCausalLM):
    layers_to_approximate = []

    for layer_name, layer in model.named_modules():
        if not isinstance(layer, nn.Linear):
            continue
        if "lm_head" in layer_name:
            continue

        layers_to_approximate.append(layer_name)

    return layers_to_approximate
