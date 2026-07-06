import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, List
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
    LlamaRMSNorm,
)


def split_llama(config, split_point: int, torch_dtype=None, device_map=None):
    """
    Split the Llama model into client_side_model and server_side_model.
    """
    # Load base model to copy weights
    base_model = AutoModelForCausalLM.from_pretrained(
        config.name_or_path, torch_dtype=torch_dtype, trust_remote_code=True
    )

    config = base_model.config

    # Create client model
    client_model = ClientLlamaModel(config, split_point)
    # Copy weights
    _copy_llama_client_weights(client_model, base_model, split_point)

    # Create server model
    server_model = ServerLlamaModel(config, split_point)
    # Copy weights
    _copy_llama_server_weights(server_model, base_model, split_point)

    if device_map:
        client_model = client_model.to(device_map)
        server_model = server_model.to(device_map)

    return client_model, server_model


def _copy_llama_client_weights(client_model, base_model, split_point):
    src_model = base_model.model  # LlamaModel

    # Copy embed_tokens
    client_model.embed_tokens.load_state_dict(src_model.embed_tokens.state_dict())

    # Copy Rotary Embeddings (buffers)
    client_model.rotary_emb.load_state_dict(src_model.rotary_emb.state_dict())

    # Copy first split_point layers
    for i in range(split_point):
        client_model.layers[i].load_state_dict(src_model.layers[i].state_dict())


def _copy_llama_server_weights(server_model, base_model, split_point):
    src_model = base_model.model

    # Copy Rotary Embeddings (needed on server to re-compute pos embeddings for later layers)
    server_model.rotary_emb.load_state_dict(src_model.rotary_emb.state_dict())

    # Copy remaining layers
    for i in range(split_point, src_model.config.num_hidden_layers):
        server_model.layers[i - split_point].load_state_dict(
            src_model.layers[i].state_dict()
        )

    # Copy norm
    server_model.norm.load_state_dict(src_model.norm.state_dict())

    # Copy lm_head
    server_model.lm_head.load_state_dict(base_model.lm_head.state_dict())


class ClientLlamaModel(nn.Module):
    def __init__(self, config, split_point: int):
        super().__init__()
        self.config = config
        self.split_point = split_point
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if split_point <= 0 or split_point > config.num_hidden_layers:
            raise ValueError(
                f"split_point must be in the range (0, {config.num_hidden_layers}], got {split_point}"
            )

        # Embeddings
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )

        # Rotary Embeddings
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        # Layers (only up to split_point)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(split_point)]
        )

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

        # Create Causal Mask
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # Calculate position embeddings
        # Note: LlamaModel computes this once and passes it to layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Run client layers
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        # Return hidden_states, mask, and position_ids (needed by server)
        return (
            hidden_states,
            causal_mask,
        )


class ServerLlamaModel(nn.Module):
    def __init__(self, config, split_point: int):
        super().__init__()
        self.config = config
        self.split_point = split_point

        # Rotary Embeddings (needed to generate inputs for layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        # Remaining layers
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(split_point, config.num_hidden_layers)
            ]
        )

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.LongTensor = None,
        past_key_values: Optional[DynamicCache] = None,
        use_cache: Optional[bool] = False,
    ):
        # Re-compute position embeddings on server side
        # This is lightweight and avoids transferring large tensors
        position_ids = (
            position_ids
            if position_ids is not None
            else torch.arange(
                hidden_states.shape[1], device=hidden_states.device
            ).unsqueeze(0)
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Prepare cache_position if needed
        cache_position = None
        if past_key_values is not None:
            past_seen_tokens = past_key_values.get_seq_length()
            # Approximate reconstruction of cache_position for slicing if needed
            # Ideally passed from client, but strictly only needed if layers use it for specific masking
            cache_position = torch.arange(
                past_seen_tokens - hidden_states.shape[1],
                past_seen_tokens,
                device=hidden_states.device,
            )

        # Run server layers
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        # Final Norm
        hidden_states = self.norm(hidden_states)

        # LM Head
        logits = self.lm_head(hidden_states)

        return CausalLMOutputWithPast(logits=logits)
