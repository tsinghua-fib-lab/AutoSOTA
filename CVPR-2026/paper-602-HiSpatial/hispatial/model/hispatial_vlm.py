"""HiSpatial Vision-Language Model.

The architecture mirrors PaliGemmaForConditionalGeneration (v4.50) with the
addition of a depth encoder (Conv2dForXYZ) and a combined multi-modal
projector that fuses SigLIP + depth features.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    PaliGemmaConfig,
    PreTrainedModel,
)
from transformers.cache_utils import Cache, HybridCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.utils import ModelOutput

from .depth_encoder import Conv2dForXYZ
from .projector import CombinedMultiModalProjector


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class HiSpatialOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


# ---------------------------------------------------------------------------
# PreTrained base (config, weight init, split policy)
# ---------------------------------------------------------------------------

class HiSpatialPreTrainedModel(PreTrainedModel):
    config_class = PaliGemmaConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = ["CombinedMultiModalProjector"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )
        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class HiSpatialVLM(HiSpatialPreTrainedModel, GenerationMixin):
    """HiSpatial Vision-Language Model.

    Extends PaliGemma with hierarchical 3D spatial understanding by fusing
    RGB image features (SigLIP) with XYZ point cloud features (Conv2dForXYZ)
    through a combined multi-modal projector.
    """

    def __init__(self, config: PaliGemmaConfig):
        super().__init__(config)

        self.vision_tower = AutoModel.from_config(config=config.vision_config)
        self.multi_modal_projector = CombinedMultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size

        language_model = AutoModelForCausalLM.from_config(config=config.text_config)
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]
        self.language_model = language_model

        self.depth_encoder = Conv2dForXYZ(
            in_channels=193,  # 192 xyz positional embedding + 1 mask
            out_channels=config.vision_config.hidden_size,
            kernel_size=config.vision_config.patch_size,
            stride=config.vision_config.patch_size,
            padding="valid",
        )

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load a pretrained PaliGemma backbone and initialise depth components.

        The backbone's original single-stream projector is replaced by a
        ``CombinedMultiModalProjector`` whose SigLIP half is copied from the
        pretrained weights while the depth half is zero-initialised.
        """
        from transformers import PaliGemmaForConditionalGeneration as _OrigPaliGemma

        # 1. Load the original PaliGemma to get its weights (including the
        #    original single-stream projector).
        original = _OrigPaliGemma.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        config = original.config

        # 2. Build our model (random weights for combined projector + depth encoder).
        model = cls(config)

        # 3. Copy all shared weights from the original model.
        model.vision_tower.load_state_dict(original.vision_tower.state_dict())
        model.language_model.load_state_dict(original.language_model.state_dict(), strict=False)

        # 4. Initialise CombinedMultiModalProjector from the original projector.
        model.multi_modal_projector = CombinedMultiModalProjector(config, original.multi_modal_projector)

        del original
        return model

    # -- Embedding helpers (delegate to language model) --------------------

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    # -- Causal mask (copied from PaliGemma v4.50) ---------

    def _update_causal_mask(
        self,
        attention_mask,
        token_type_ids=None,
        past_key_values=None,
        cache_position=None,
        input_tensor=None,
        is_training: bool = None,
    ):
        if self.config.text_config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        is_training = is_training if is_training is not None else self.training
        using_static_cache = isinstance(past_key_values, StaticCache)
        min_dtype = torch.finfo(self.dtype).min

        if input_tensor is None:
            input_tensor = attention_mask

        inputs_lead_dim, sequence_length = input_tensor.shape[:2]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        elif isinstance(past_key_values, HybridCache):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else cache_position[0] + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask

        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=self.dtype,
            device=cache_position.device,
        )
        if sequence_length != 1:
            if is_training:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            else:
                causal_mask[:, :sequence_length] = 0.0

        causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_lead_dim, 1, -1, -1)

        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            if is_training:
                if token_type_ids is None:
                    raise ValueError("Token type ids must be provided during training")
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    token_type_ids[:, None, None, :].to(causal_mask.device) == 0, 0
                )
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

        return causal_mask

    # -- 3-D positional encoding -------------------------------------------

    def get_3d_sincos_pos_embed_from_grid(self, points: torch.Tensor, embed_dim: int) -> torch.Tensor:
        """Generate sin-cos positional encoding for each (x, y, z) point.

        Args:
            points: [B, H, W, 3] point cloud tensor.
            embed_dim: Output embedding dimension, must be divisible by 6.

        Returns:
            [B, H, W, embed_dim] positional embeddings.
        """
        assert embed_dim % 6 == 0, "embed_dim must be divisible by 6"
        B, H, W, _ = points.shape
        points_flat = points.view(B, H * W, 3)

        dim_each = embed_dim // 3
        half_dim = dim_each // 2
        freqs = torch.arange(half_dim, dtype=torch.float32, device=points.device) / half_dim
        freqs = 1.0 / (10000**freqs)

        def encode(coord):
            scaled = coord.unsqueeze(-1) * freqs
            return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)

        pos_embed = torch.cat(
            [encode(points_flat[..., i]) for i in range(3)], dim=-1
        )
        return pos_embed.view(B, H, W, embed_dim)

    def get_3d_sincos_embed_with_mask(self, input: torch.Tensor, embed_dim: int) -> torch.Tensor:
        """Sin-cos encoding on xyz channels with mask appended.

        Args:
            input: [B, H, W, 4] tensor (xyz + mask).
            embed_dim: Embedding dim for xyz, must be divisible by 6.

        Returns:
            [B, H, W, embed_dim + 1] tensor.
        """
        assert input.shape[-1] == 4
        assert embed_dim % 6 == 0
        xyz = input[..., :3]
        mask = input[..., 3:]
        pos_embed = self.get_3d_sincos_pos_embed_from_grid(xyz, embed_dim=embed_dim)
        return torch.cat([pos_embed, mask], dim=-1)

    # -- Image feature extraction (SigLIP + depth) -------------------------

    def get_image_features(self, pixel_values: torch.FloatTensor, xyz_values: torch.Tensor):
        """Fuse SigLIP vision features with depth features from Conv2dForXYZ."""
        siglip_outputs = self.vision_tower(pixel_values)
        siglip_features = siglip_outputs.last_hidden_state

        xyz_values = xyz_values.permute(0, 2, 3, 1)  # -> [B, H, W, 4]
        xyz_embeds = self.get_3d_sincos_embed_with_mask(xyz_values, embed_dim=192)  # [B, H, W, 193]
        xyz_embeds = xyz_embeds.permute(0, 3, 1, 2)  # [B, 193, H, W]

        depth_features = self.depth_encoder(xyz_embeds)  # [B, C, h, w]
        B, C, H, W = depth_features.shape
        depth_features = depth_features.permute(0, 2, 3, 1).reshape(B, H * W, C)

        combined_features = torch.cat([siglip_features, depth_features], dim=-1)
        image_features = self.multi_modal_projector(combined_features)
        image_features = image_features / (self.config.text_config.hidden_size**0.5)
        return image_features

    # -- Forward pass ------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        xyz_values: Optional[torch.Tensor] = None,
        **lm_kwargs,
    ) -> Union[Tuple, HiSpatialOutputWithPast]:

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        is_training = token_type_ids is not None and labels is not None

        # Replace image token id with PAD if OOV to avoid index errors
        if input_ids is not None and self.config.image_token_index >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_index
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0) + 1  # PaliGemma positions are 1-indexed

        # Merge text and image+depth features
        image_features = None
        if pixel_values is not None and xyz_values is not None:
            image_features = self.get_image_features(pixel_values, xyz_values)

            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_index, dtype=torch.long, device=inputs_embeds.device)
                )
            else:
                special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

            if inputs_embeds[special_image_mask].numel() != image_features.numel():
                image_tokens_in_text = special_image_mask.sum(dim=1).sum(dim=0)[0]
                raise ValueError(
                    f"Number of images does not match number of special image tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text but "
                    f"{image_features.shape[0] * image_features.shape[1]} tokens from image embeddings."
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # Mask out pad-token-ids in labels
        if labels is not None and self.pad_token_id in labels:
            labels = torch.where(input_ids == self.pad_token_id, self.config.ignore_index, labels)

        causal_mask = self._update_causal_mask(
            attention_mask, token_type_ids, past_key_values, cache_position, inputs_embeds, is_training
        )

        outputs = self.language_model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1]:].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            loss_fct = nn.CrossEntropyLoss()
            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return HiSpatialOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
        )

    # -- Generation helpers ------------------------------------------------

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        attention_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        xyz_values=None,
        **kwargs,
    ):
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        # PaliGemma positions are 1-indexed
        if model_inputs.get("position_ids") is not None:
            model_inputs["position_ids"] += 1

        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["xyz_values"] = xyz_values

        is_training = token_type_ids is not None and labels is not None
        if cache_position[0] == 0 and isinstance(past_key_values, HybridCache):
            input_tensor = inputs_embeds if inputs_embeds is not None else input_ids
            causal_mask = self._update_causal_mask(
                attention_mask, token_type_ids, past_key_values, cache_position, input_tensor, is_training
            )
            model_inputs["attention_mask"] = causal_mask

        return model_inputs
