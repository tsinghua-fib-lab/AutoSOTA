"""Patch for Qwen3-VL text model (model.language_model)."""

import torch
from typing import List, Optional, Tuple
from dataclasses import dataclass

from ...base import BasePatch


@dataclass
class TextModelOutputs:
    """Container for captured text model outputs."""
    # Position embeddings (cos, sin) from MRoPE
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    # Hidden states after each decoder layer
    hidden_states_per_layer: List[torch.Tensor] = None
    # Input embeddings before decoder layers
    input_embeds: Optional[torch.Tensor] = None
    # Position IDs used for MRoPE (3, batch, seq_len) for t, h, w
    position_ids: Optional[torch.Tensor] = None
    # Visual position masks used for DeepStack
    visual_pos_masks: Optional[torch.Tensor] = None
    # DeepStack visual embeddings per layer
    deepstack_visual_embeds: Optional[List[torch.Tensor]] = None

    def __post_init__(self):
        if self.hidden_states_per_layer is None:
            self.hidden_states_per_layer = []


class TextPatch(BasePatch):
    """
    Patch model.language_model.forward to capture intermediate outputs.

    Captures:
        - position_embeddings: (cos, sin) from MRoPE
        - hidden_states_per_layer: hidden states after each decoder layer
        - input_embeds: token embeddings before decoder layers
        - position_ids: 3D position IDs (t, h, w)

    Usage:
        with TextPatch(model.model.language_model) as patch:
            output = model.generate(...)
            print(patch.outputs.position_embeddings)
            print(len(patch.outputs.hidden_states_per_layer))
    """

    def __init__(self, text_model, capture_hidden_states: bool = True):
        """
        Args:
            text_model: The language model (model.model.language_model)
            capture_hidden_states: Whether to capture hidden states per layer
        """
        self.text_model = text_model
        self.capture_hidden_states = capture_hidden_states
        self.original_forward = None
        self.outputs = TextModelOutputs()

    def apply(self):
        self.original_forward = self.text_model.forward
        self.text_model.forward = self._patched_forward

    def remove(self):
        if self.original_forward is not None:
            self.text_model.forward = self.original_forward
            self.original_forward = None

    def clear(self):
        self.outputs = TextModelOutputs()

    def _patched_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list] = None,
        **kwargs,
    ):
        """Same as original forward, but captures intermediate outputs."""
        from transformers.cache_utils import DynamicCache
        from transformers.modeling_outputs import BaseModelOutputWithPast
        from transformers.masking_utils import create_causal_mask

        self.clear()

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.text_model.config)

        if inputs_embeds is None:
            inputs_embeds = self.text_model.embed_tokens(input_ids)

        # Capture input embeddings
        self.outputs.input_embeds = inputs_embeds.clone().detach()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device
            )

        # Handle position_ids (3D for t, h, w)
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        # Capture position IDs
        self.outputs.position_ids = position_ids.clone().detach()

        # Capture visual inputs for DeepStack
        if visual_pos_masks is not None:
            self.outputs.visual_pos_masks = visual_pos_masks.clone().detach()
        if deepstack_visual_embeds is not None:
            self.outputs.deepstack_visual_embeds = [
                emb.clone().detach() for emb in deepstack_visual_embeds
            ]

        attention_mask = create_causal_mask(
            config=self.text_model.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds

        # Create position embeddings (MRoPE)
        position_embeddings = self.text_model.rotary_emb(hidden_states, position_ids)

        # Capture position embeddings (cos, sin)
        self.outputs.position_embeddings = (
            position_embeddings[0].clone().detach(),
            position_embeddings[1].clone().detach()
        )

        # Decoder layers
        for layer_idx, decoder_layer in enumerate(self.text_model.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs

            # Capture hidden states per layer
            if self.capture_hidden_states:
                self.outputs.hidden_states_per_layer.append(hidden_states.clone().detach())

            # DeepStack: add visual features to early layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self.text_model._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.text_model.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
