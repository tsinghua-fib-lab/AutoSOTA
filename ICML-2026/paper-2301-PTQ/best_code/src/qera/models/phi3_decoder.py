import logging
from typing import Optional, Tuple
import math
from copy import deepcopy

import torch
from torch import nn

from transformers.models.phi3.modeling_phi3 import (
    Phi3RMSNorm,
    Phi3RotaryEmbedding,
    Phi3LongRoPEScaledRotaryEmbedding,
    Phi3DecoderLayer,
    ACT2FN,
    Cache,
    apply_rotary_pos_emb,
    repeat_kv,
    Phi3ForCausalLM,
    Phi3Config,
)

from ..quantize import get_quantized_layer_cls, get_quantized_func
from ..utils import find_matched_pattern, get_layer_name

logger = logging.getLogger(__name__)


class Phi3QuantizedMLP(nn.Module):
    def __init__(self, config, qera_config: dict):
        super().__init__()

        self.config = config
        # self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        # self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # fmt: off
        self.gate_up_proj = get_quantized_layer_cls("linear", q_config=qera_config["gate_up_proj"])(config.hidden_size, 2 * config.intermediate_size, bias=False, q_config=qera_config["gate_up_proj"])
        self.down_proj = get_quantized_layer_cls("linear", q_config=qera_config["down_proj"])(config.intermediate_size, config.hidden_size, bias=False, q_config=qera_config["down_proj"])
        # fmt: on

        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        up_states = self.gate_up_proj(hidden_states)

        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)

        return self.down_proj(up_states)


class Phi3QuantizedAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Phi3Config, layer_idx: int, qera_config):
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
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        op_size = self.num_heads * self.head_dim + 2 * (
            self.num_key_value_heads * self.head_dim
        )
        # self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        # self.qkv_proj = nn.Linear(self.hidden_size, op_size, bias=False)
        # fmt: off
        self.o_proj = get_quantized_layer_cls("linear", q_config=qera_config["o_proj"])(self.num_heads * self.head_dim, self.hidden_size, bias=False, q_config=qera_config["o_proj"])
        self.qkv_proj = get_quantized_layer_cls("linear", q_config=qera_config["qkv_proj"])(self.hidden_size, op_size, bias=False, q_config=qera_config["qkv_proj"])
        self.qera_config = qera_config
        # fmt: on
        self._init_rope()

    def _init_rope(self):
        if self.rope_scaling is None:
            self.rotary_emb = Phi3RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            if scaling_type == "longrope":
                self.rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(
                    self.head_dim, self.config
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

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
        logger.warning_once(
            "You are not running the flash-attention implementation, expect numerical differences."
        )

        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[
            ..., query_pos : query_pos + self.num_key_value_heads * self.head_dim
        ]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

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
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # *: matmul_0
        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        # fmt: off
        kv_seq_len = key_states.size(2)
        query_states = query_states.reshape(bsz * self.num_heads, q_len, self.head_dim)
        key_states = key_states.reshape(bsz * self.num_heads, kv_seq_len, self.head_dim)
        attn_weights = get_quantized_func("matmul", q_config=self.qera_config["matmul_0"])(
            query_states, key_states.transpose(1, 2), q_config=self.qera_config["matmul_0"]
        ) / math.sqrt(self.head_dim)
        attn_weights = attn_weights.reshape(bsz, self.num_heads, q_len, kv_seq_len)
        # fmt: on

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights += causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(value_states.dtype)
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
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


PHI3_ATTENTION_CLASSES = {"eager": Phi3QuantizedAttention}


class Phi3QuantizedDecoderLayer(nn.Module):
    def __init__(self, config: Phi3Config, layer_idx: int, qera_config: dict):
        super().__init__()

        self.config = config
        self.self_attn = PHI3_ATTENTION_CLASSES[config._attn_implementation](
            config, layer_idx=layer_idx, qera_config=qera_config["self_attn"]
        )

        self.mlp = Phi3QuantizedMLP(config, qera_config=qera_config["mlp"])
        self.input_layernorm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)
        self.post_attention_layernorm = Phi3RMSNorm(
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
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        hidden_states = residual + self.resid_attn_dropout(attn_outputs)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


def build_qera_config_phi3(model: Phi3ForCausalLM, qera_config: dict):
    parsed_config = {}

    decoder_layer_i: Phi3DecoderLayer
    for i, decoder_layer_i in enumerate(model.model.layers):
        parsed_config[f"model_layer_{i}"] = {"self_attn": {}, "mlp": {}}
        for fc_short_name in ["qkv_proj", "o_proj"]:
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
        for fc_short_name in ["gate_up_proj", "down_proj"]:
            fc_name = get_layer_name(model, getattr(decoder_layer_i.mlp, fc_short_name))
            matched_entry = find_matched_pattern(fc_name, qera_config.keys())
            assert matched_entry is not None, f"Cannot find matched entry for {fc_name}"
            if isinstance(qera_config[matched_entry], str):
                matched_entry = qera_config[matched_entry]
            parsed_config[f"model_layer_{i}"]["mlp"][fc_short_name] = deepcopy(
                qera_config[matched_entry]
            )

    return parsed_config


def quantize_phi3_model(model: Phi3ForCausalLM, qera_config: dict):
    qera_config = build_qera_config_phi3(model, qera_config)

    for layer_id, ori_decoder_layer in enumerate(model.model.layers):
        layer_entry = f"model_layer_{layer_id}"
        layer_qera_config = qera_config[layer_entry]

        # replace the decoder layer with quantized decoder layer
        new_decoder_layer = Phi3QuantizedDecoderLayer(
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


def find_layers_to_register_scale_hook_phi3(
    model: Phi3ForCausalLM,
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
        qkv_name = get_layer_name(model, decoder_layer.self_attn.qkv_proj)
        layers_to_register.append(
            dict(target_layer=qkv_name, layers_sharing_scale=[]),
        )
        o_name = get_layer_name(model, decoder_layer.self_attn.o_proj)
        layers_to_register.append(
            dict(target_layer=o_name, layers_sharing_scale=[]),
        )

        gate_up_name = get_layer_name(model, decoder_layer.mlp.gate_up_proj)
        layers_to_register.append(
            dict(target_layer=gate_up_name, layers_sharing_scale=[]),
        )

        down_name = get_layer_name(model, decoder_layer.mlp.down_proj)
        layers_to_register.append(
            dict(target_layer=down_name, layers_sharing_scale=[]),
        )

    return layers_to_register


def find_layers_to_approximate_phi3(model: Phi3ForCausalLM):
    layers_to_approximate = []

    for layer_name, layer in model.named_modules():
        if not isinstance(layer, nn.Linear):
            continue
        if "lm_head" in layer_name:
            continue

        layers_to_approximate.append(layer_name)

    return layers_to_approximate
