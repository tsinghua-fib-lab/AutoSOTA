import copy
import torch
import torch.nn as nn
from typing import Optional, Dict, Union, Tuple, Callable
from transformers import AutoModelForCausalLM
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.cache_utils import DynamicCache


from transformers.modeling_outputs import CausalLMOutputWithPast


from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3DecoderLayer,
    Gemma3RotaryEmbedding,
    Gemma3RMSNorm,
    Gemma3TextScaledWordEmbedding,
)


def split_gemma3(config, split_point: int, torch_dtype=None, device_map=None):
    """
    Split the Gemma3 model into client_side_model and server_side_model.
    """
    # Load base model to copy weights
    base_model = AutoModelForCausalLM.from_pretrained(
        config.name_or_path, torch_dtype=torch_dtype, trust_remote_code=True
    )

    config = base_model.config

    # Create client model
    client_model = ClientGemma3Model(config, split_point)
    # Copy weights
    _copy_gemma3_client_weights(client_model, base_model, split_point)

    # Create server model
    server_model = ServerGemma3Model(config, split_point)
    # Copy weights
    _copy_gemma3_server_weights(server_model, base_model, split_point)

    if device_map:
        client_model = client_model.to(device_map)
        server_model = server_model.to(device_map)

    return client_model, server_model


def _copy_gemma3_client_weights(client_model, base_model, split_point):
    src_model = base_model.model  # Gemma3TextModel

    # Copy embed_tokens
    client_model.embed_tokens.load_state_dict(src_model.embed_tokens.state_dict())

    # Copy Rotary Embeddings (buffers)
    client_model.rotary_emb.load_state_dict(src_model.rotary_emb.state_dict())
    client_model.rotary_emb_local.load_state_dict(
        src_model.rotary_emb_local.state_dict()
    )

    # Copy first split_point layers
    for i in range(split_point):
        client_model.layers[i].load_state_dict(src_model.layers[i].state_dict())


def _copy_gemma3_server_weights(server_model, base_model, split_point):
    src_model = base_model.model

    # Copy Rotary Embeddings (needed on server to re-compute pos embeddings for later layers)
    server_model.rotary_emb.load_state_dict(src_model.rotary_emb.state_dict())
    server_model.rotary_emb_local.load_state_dict(
        src_model.rotary_emb_local.state_dict()
    )

    # Copy remaining layers
    for i in range(split_point, src_model.config.num_hidden_layers):
        server_model.layers[i - split_point].load_state_dict(
            src_model.layers[i].state_dict()
        )

    # Copy norm
    server_model.norm.load_state_dict(src_model.norm.state_dict())

    # Copy lm_head (from Gemma3ForCausalLM)
    server_model.lm_head.load_state_dict(base_model.lm_head.state_dict())


def _bidirectional_window_overlay(
    sliding_window: int,
) -> Callable[[int, int, int, int], bool]:
    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return abs(q_idx - kv_idx) < sliding_window

    return inner_mask


class ClientGemma3Model(nn.Module):
    def __init__(self, config, split_point: int):
        super().__init__()
        self.config = config
        self.split_point = split_point

        if split_point <= 0 or split_point > config.num_hidden_layers:
            raise ValueError(
                f"split_point must be in the range (0, {config.num_hidden_layers}], got {split_point}"
            )

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Embeddings
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=config.hidden_size**0.5,
        )

        # Layers (only up to split_point)
        self.layers = nn.ModuleList(
            [Gemma3DecoderLayer(config, layer_idx) for layer_idx in range(split_point)]
        )

        # Rotary Embeddings
        self.rotary_emb = Gemma3RotaryEmbedding(config=config)
        # Handle local RoPE config logic similarly to original model
        config_local = copy.deepcopy(config)
        config_local.rope_theta = config.rope_local_base_freq
        config_local.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3RotaryEmbedding(config=config_local)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        use_cache: Optional[bool] = False,
    ):
        inputs_embeds = self.embed_tokens(input_ids)

        # Cache position setup
        if past_key_values is None and use_cache:
            past_key_values = DynamicCache(config=self.config)

        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Mask creation logic (simplified from Gemma3TextModel)
        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            sliding_mask_kwargs = mask_kwargs.copy()

            if self.config.use_bidirectional_attention:
                mask_kwargs["or_mask_function"] = lambda *args: torch.tensor(
                    True, dtype=torch.bool
                )
                sliding_mask_kwargs["or_mask_function"] = _bidirectional_window_overlay(
                    self.config.sliding_window
                )

            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(
                    **sliding_mask_kwargs
                ),
            }
        else:
            causal_mask_mapping = attention_mask

        hidden_states = inputs_embeds

        # Calculate position embeddings
        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        # Run client layers
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]

        # Return everything needed for the server to continue
        # Note: We return position_ids so server can re-compute RoPE without transferring large tensors
        return (
            hidden_states,
            causal_mask_mapping,
        )


class ServerGemma3Model(nn.Module):
    def __init__(self, config, split_point: int):
        super().__init__()
        self.config = config
        self.split_point = split_point

        # Remaining layers
        self.layers = nn.ModuleList(
            [
                Gemma3DecoderLayer(config, layer_idx)
                for layer_idx in range(split_point, config.num_hidden_layers)
            ]
        )

        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM Head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Rotary Embeddings (needed to generate inputs for layers)
        self.rotary_emb = Gemma3RotaryEmbedding(config=config)
        config_local = copy.deepcopy(config)
        config_local.rope_theta = config.rope_local_base_freq
        config_local.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3RotaryEmbedding(config=config_local)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Union[Dict, torch.Tensor],
        position_ids: torch.LongTensor = None,
        past_key_values: Optional[DynamicCache] = None,
        use_cache: Optional[bool] = False,
    ):
        # Re-compute position embeddings on server side to save bandwidth
        # (they depend only on hidden_states shape and position_ids)
        position_ids = (
            position_ids
            if position_ids is not None
            else torch.arange(
                hidden_states.shape[1], device=hidden_states.device
            ).unsqueeze(0)
        )
        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        # Prepare cache_position if needed (usually inferred or passed, simplified here)
        # If cache logic is complex across split, passing cache_position might be better.
        # For inference without cache or simple cache, this might suffice.
        if past_key_values is not None:
            past_seen_tokens = past_key_values.get_seq_length()
            # This is an approximation; ideally pass cache_position from client
            cache_position = torch.arange(
                past_seen_tokens - hidden_states.shape[1],
                past_seen_tokens,
                device=hidden_states.device,
            )
        else:
            cache_position = None

        # Run server layers
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                attention_mask=(
                    attention_mask[decoder_layer.attention_type]
                    if isinstance(attention_mask, dict)
                    else attention_mask
                ),
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]

        # Final Norm
        hidden_states = self.norm(hidden_states)

        # LM Head
        logits = self.lm_head(hidden_states)

        # Final Logit Softcapping (Gemma3 specific feature)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        return CausalLMOutputWithPast(
            logits=logits,
        )
