"""
FocusUI wrapper around Qwen3-VL.
"""

from typing import Any, List, Tuple, Union, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache, StaticCache
from transformers.utils import TransformersKwargs, auto_docstring, is_torchdynamo_compiling
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast, Qwen3VLForConditionalGeneration, Qwen3VLModelOutputWithPast

from focusui.trainer import rank0_print
from focusui.modeling_patch_scorer import PatchScorerConfig, PatchScorerModel


class FocusUI_QwenVLwithVisionHeadOutputWithPast(Qwen3VLCausalLMOutputWithPast):
    """
    Output class for Qwen3-VL + FocusUI extras, extending the base causal LM output.

    Additional attributes:
    - `lm_loss`, `pointer_loss`, `ps_loss`: optional loss terms.
    - `pointer_scores`: per-sample attention scores from the pointer head.
    - `patch_scores`: patch-level scores from `PatchScorerModel`.
    - `token_keep_mask`: boolean mask over the (possibly shortened) input sequence after token selection.
    - `actual_token_percentage`: ratio of tokens kept after visual token selection.
    """

    def __init__(
        self,
        lm_loss=None,
        pointer_loss=None,
        pointer_scores=None,
        patch_scores=None,
        token_keep_mask=None,
        actual_token_percentage=None,
        ps_loss=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lm_loss = lm_loss
        self.pointer_loss = pointer_loss
        self.pointer_scores = pointer_scores

        self.patch_scores = patch_scores
        self.token_keep_mask = token_keep_mask
        self.actual_token_percentage = actual_token_percentage
        self.ps_loss = ps_loss

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None

    # focus-ui
    patch_scores: Optional[torch.Tensor] = None
    token_keep_mask: Optional[torch.Tensor] = None
    actual_token_percentage: Optional[float] = None
    ps_loss: Optional[torch.FloatTensor] = None


class VisionHead_MultiPatch(nn.Module):
    def __init__(self, d_model, projection_dim, num_attention_heads=8, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model

        # Projection layers for visual (encoder) and target (decoder) representations
        self.projection_enc = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, d_model),
        )
        self.projection_dec = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, d_model),
        )

        # Self-attention on visual tokens to inject local context
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        hidden_state_enc: torch.Tensor,  # [n_enc, d_model]
        hidden_state_dec: torch.Tensor,  # [n_dec, d_model]
        labels: Optional[torch.Tensor] = None,  # [n_dec, n_enc] binary mask per target of patches in bbox
    ):
        enc_input = hidden_state_enc.unsqueeze(0)
        attn_output, _ = self.self_attention(
            query=enc_input,
            key=enc_input,
            value=enc_input,
            need_weights=False
        )
        hidden_state_enc_ctx = self.layer_norm(enc_input + self.dropout(attn_output)).squeeze(0)

        proj_enc = self.projection_enc(hidden_state_enc_ctx)  # [n_enc, d_model]
        proj_dec = self.projection_dec(hidden_state_dec)  # [n_dec, d_model]

        scaling = self.d_model**0.5
        patch_logits = torch.matmul(proj_dec, proj_enc.transpose(0, 1)) / scaling  # [n_dec, n_enc]
        attn_weights = F.softmax(patch_logits, dim=-1)

        loss = None
        if labels is not None:
            epsilon = 1e-8
            labels_float = labels.float()
            target_dist = labels_float / (labels_float.sum(dim=-1, keepdim=True) + epsilon)
            pred_log_probs = F.log_softmax(patch_logits, dim=-1)
            loss = F.kl_div(pred_log_probs, target_dist, reduction="batchmean")

        return attn_weights, loss


class FocusUI_Qwen3VLForConditionalGenerationWithPointer(Qwen3VLForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_patch_pointer_head = VisionHead_MultiPatch(self.config.text_config.hidden_size, self.config.text_config.hidden_size)
        self.pointer_loss_weight = kwargs.get("pointer_loss_weight", 1.0)
        self.lm_loss_weight = kwargs.get("lm_loss_weight", 1.0)
        self.ps_loss_weight = kwargs.get("ps_loss_weight", 1.0)

        self.apply_visual_token_select = kwargs.get("apply_visual_token_select", True)
        self.train_visual_reduct_ratio = kwargs.get("train_visual_reduct_ratio", (0.0, 0.95))
        self.visual_reduct_ratio = kwargs.get("visual_reduct_ratio", 0.5)

        self.patch_scorer_config = kwargs.get("patch_scorer_config", PatchScorerConfig(
            projection_dim=self.config.vision_config.out_hidden_size,
            projection_dropout=0.1,
            text_token_pooling="mean",
        ))
        self.patch_scorer = PatchScorerModel(self.patch_scorer_config)
        self.patch_scorer_early_exit = kwargs.get("patch_scorer_early_exit", False)

        # Cache latest patch_scores for retrieval after generate()
        self._last_patch_scores = None


        self.post_init()

    def reset_loss_weights(self, pointer_loss_weight=None, lm_loss_weight=None, ps_loss_weight=None):
        if pointer_loss_weight is not None:
            self.pointer_loss_weight = pointer_loss_weight
        if lm_loss_weight is not None:
            self.lm_loss_weight = lm_loss_weight
        if ps_loss_weight is not None:
            self.ps_loss_weight = ps_loss_weight

    def reset_focus_ui_options(self, apply_visual_token_select=None, train_visual_reduct_ratio=None, visual_reduct_ratio=None):
        if apply_visual_token_select is not None:
            self.apply_visual_token_select = apply_visual_token_select
        if train_visual_reduct_ratio is not None:
            self.train_visual_reduct_ratio = train_visual_reduct_ratio
        if visual_reduct_ratio is not None:
            self.visual_reduct_ratio = visual_reduct_ratio

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,  # set True for coord-free grounding
        return_dict: Optional[bool] = True,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        # Grounding
        visual_token_indices_of_coordinates: Optional[torch.Tensor] = None,
        multi_patch_labels: Optional[torch.Tensor] = None,
        # Focus-UI
        patch_scores: Optional[torch.Tensor] = None,
        patch_scores_label: Optional[torch.Tensor] = None,
        focus_input_ids: Optional[torch.LongTensor] = None,
        focus_attention_mask: Optional[torch.Tensor] = None,
        # debug
        verbose: bool = False,
        verbose_visual_token_selection_summary: bool = True,
    ) -> Union[Tuple, FocusUI_QwenVLwithVisionHeadOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        ######## Qwen3_VLForConditionalGeneration.forward() ########
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.model.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        ########


        ##### Build Patch Scores #####
        ps_loss = None

        if patch_scores is None and pixel_values is not None:
            instruct_embeds = self.get_input_embeddings()(focus_input_ids) if focus_input_ids is not None else None

            if patch_scores_label is not None:
                if isinstance(patch_scores_label, (list, tuple)):
                    patch_scores_label = patch_scores_label[0]
                if patch_scores_label.dim() == 1:
                    patch_scores_label = patch_scores_label.unsqueeze(0)
            
            if instruct_embeds is not None:
                patch_scorer_results = self.patch_scorer(
                    image_embeds=image_embeds,
                    text_embeds=instruct_embeds,
                    patch_scores_label=patch_scores_label,
                    return_dict=True,
                )

                patch_scores = patch_scorer_results["patch_scores"]
                ps_loss = patch_scorer_results["loss"]

            self._last_patch_scores = patch_scores

        elif patch_scores is not None:
            # If patch_scores is provided externally (e.g., for ablations), cache it for retrieval after `generate()`.
            self._last_patch_scores = patch_scores

        ##### Visual Token Selection #####
        actual_token_percentage = 1.0
        token_keep_mask = None
        image_token_keep_mask = None

        if self.apply_visual_token_select and (patch_scores is not None) and (inputs_embeds.shape[1] != 1):
            # set spatial drop rate
            if self.training:
                visual_reduct_ratio = np.random.uniform(self.train_visual_reduct_ratio[0], self.train_visual_reduct_ratio[1])
                visual_reduct_ratio = np.clip(visual_reduct_ratio, 0.0, 0.99)
            else:
                # during testing, we set a default fixed spatial drop rate
                visual_reduct_ratio = self.visual_reduct_ratio

            (
                input_ids,
                inputs_embeds,
                position_ids,
                attention_mask,
                cache_position,
                labels,
                patch_scores,
                visual_pos_masks,
                deepstack_visual_embeds,
                actual_token_percentage,
                visual_token_percentage,
                token_keep_mask,
                image_token_keep_mask,
            ) = self.visual_token_selection_with_patch_scores(
                visual_reduct_ratio=visual_reduct_ratio,
                patch_scores=patch_scores,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
                labels=labels,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                return_dict=False,
                verbose=verbose,
                verbose_visual_token_selection_summary=verbose_visual_token_selection_summary,
            )
        else:
            actual_token_percentage = 1.0
            token_keep_mask = None
            image_token_keep_mask = None

        if self.patch_scorer_early_exit:
            return FocusUI_QwenVLwithVisionHeadOutputWithPast(
                loss=ps_loss,
                rope_deltas=self.model.rope_deltas,
                patch_scores=patch_scores,
                actual_token_percentage=actual_token_percentage,
                token_keep_mask=token_keep_mask,
                ps_loss=ps_loss,
            )


        language_model_outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outputs = Qwen3VLModelOutputWithPast(
            last_hidden_state=language_model_outputs.last_hidden_state,
            past_key_values=language_model_outputs.past_key_values,
            hidden_states=getattr(language_model_outputs, "hidden_states", None),
            attentions=getattr(language_model_outputs, "attentions", None),
            rope_deltas=self.model.rope_deltas,
        )

        hidden_states = outputs[0]  # shape: (batch_size, seq_len, d_model)

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        lm_loss = None
        if labels is not None and self.lm_loss_weight > 0:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            lm_loss = loss_fct(shift_logits, shift_labels)

        # Pointer supervision using visual encoder outputs and decoder hidden states
        pointer_loss = None
        pointer_scores: List[torch.Tensor] = []

        if visual_token_indices_of_coordinates is not None:
            batch_size = int(input_ids.shape[0])
            image_token_id = self.config.image_token_id

            pointer_losses = []

            for i in range(batch_size):
                dummy_target = False
                input_ids_b = input_ids[i]
                hidden_states_b = hidden_states[i]

                visual_indices = torch.nonzero(input_ids_b == image_token_id, as_tuple=False).squeeze(-1)
                target_mask = input_ids_b == getattr(self.config, "pointer_pad_token_id", self.config.pad_token_id)
                target_indices = torch.nonzero(target_mask, as_tuple=False).squeeze(-1)

                if visual_indices.numel() == 0:
                    raise ValueError(f"No visual tokens found for sample {i}.")
                if target_indices.numel() == 0:
                    # create a dummy single target pointing to some patches; ignored in loss
                    target_indices = torch.tensor([hidden_states_b.shape[0] - 1], device=hidden_states_b.device)
                    sample_labels = torch.zeros((1, visual_indices.numel()), device=hidden_states_b.device, dtype=torch.long)
                    sample_labels[0][: min(4, sample_labels.shape[1])] = 1
                    dummy_target = True
                else:
                    sample_labels = multi_patch_labels[i]

                # Base visual embeddings for this sample come from encoder sequence of image tokens
                visual_embeds_b = inputs_embeds[i][visual_indices]

                # Align label dimension to kept visual tokens after selection if any
                if self.apply_visual_token_select and image_token_keep_mask is not None:
                    sample_labels = sample_labels[:, image_token_keep_mask[i]]
                    # visual_embeds_b = visual_embeds_b[image_token_keep_mask[i]]

                target_hidden_b = hidden_states_b[target_indices]

                # Ensure the number of targets matches between sample and labels
                if sample_labels.shape[0] != target_indices.shape[0]:
                    raise ValueError(
                        f"Sample {i} has mismatched target counts: {sample_labels.shape[0]} labels but found {target_indices.shape[0]} target tokens"
                    )

                attn_scores, loss_v = self.multi_patch_pointer_head(
                    visual_embeds_b,
                    target_hidden_b,
                    labels=sample_labels,
                )

                pointer_scores.append(attn_scores.detach().cpu())
                pointer_losses.append(loss_v * 0.0 if dummy_target else loss_v)

            pointer_loss = torch.stack(pointer_losses).mean()

        total_loss = 0.0
        if pointer_loss is not None:
            total_loss += self.pointer_loss_weight * pointer_loss
        if lm_loss is not None:
            total_loss += self.lm_loss_weight * lm_loss
        if ps_loss is not None:
            total_loss += self.ps_loss_weight * ps_loss

        # rank0_print(f"total_loss: {total_loss}: pointer_loss: {pointer_loss}, lm_loss: {lm_loss}, ps_loss: {ps_loss}")

        if return_dict:
            return FocusUI_QwenVLwithVisionHeadOutputWithPast(
                loss=total_loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=outputs.rope_deltas,
                lm_loss=lm_loss,
                pointer_loss=pointer_loss,
                ps_loss=ps_loss,
                pointer_scores=pointer_scores,
                patch_scores=patch_scores,
                actual_token_percentage=actual_token_percentage,
                token_keep_mask=token_keep_mask,
            )
        else:
            if labels is not None:
                output = (lm_loss, pointer_loss, logits, pointer_scores) + (outputs.past_key_values,)
                return (total_loss,) + output if total_loss is not None else output
            else:
                return outputs

    def generate_with_visual_token_select(self, *args, **kwargs):
        outputs = super().generate(*args, **kwargs)
        return outputs, self._last_patch_scores

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        patch_scores=None,
        patch_scores_label=None,
        focus_input_ids=None,
        focus_attention_mask=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )

        # Qwen3VL position_ids are prepared in forward with rope_deltas
        model_inputs["position_ids"] = None

        if cache_position is not None and cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        # Focus-UI extras
        model_inputs.update(
            {
                "patch_scores": patch_scores,
                "patch_scores_label": patch_scores_label,
                "focus_input_ids": focus_input_ids,
                "focus_attention_mask": focus_attention_mask,
            }
        )

        # Reset cached patch score only at the beginning (prefill) step
        if (past_key_values is None) or (cache_position is not None and int(cache_position[0].item()) == 0):
            self._last_patch_scores = None

        return model_inputs

    def _prepare_patch_scores_per_sample(
        self,
        patch_scores: torch.Tensor,
        n_image_tokens_per_sample: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """Split patch scores into per-sample tensors."""
        if patch_scores.dim() == 1:
            split_sizes = n_image_tokens_per_sample.tolist()
            return list(torch.split(patch_scores.to(device), split_sizes, dim=0))
        elif patch_scores.dim() == 2:
            return [
                patch_scores[b].to(device)[:int(n_image_tokens_per_sample[b].item())]
                for b in range(batch_size)
            ]
        else:
            raise ValueError(f"Unsupported patch_scores ndim: {patch_scores.dim()}")

    def _select_top_patches_with_tiebreaking(
        self,
        patch_scores: torch.Tensor,
        n_keep: int,
        input_ids_sample: torch.Tensor,
    ) -> torch.Tensor:
        """Select top-k patches with deterministic tie-breaking based on input_ids hash."""
        # Generate deterministic seed from input_ids for reproducible tie-breaking
        seed = int(torch.remainder(
            (input_ids_sample.to(torch.int64) * 1315423911).sum(), 2147483647
        ).item())
        gen = torch.Generator()
        gen.manual_seed(seed)

        # Shuffle indices, then select top-k from shuffled scores
        perm = torch.randperm(patch_scores.size(0), generator=gen).to(patch_scores.device)
        scores_shuffled = patch_scores[perm]
        _, top_idx_shuffled = torch.topk(scores_shuffled, n_keep, largest=True)
        return perm[top_idx_shuffled]

    def _find_consecutive_run_ends(
        self,
        drop_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Find the last position of each consecutive run of dropped tokens."""
        if drop_positions.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=drop_positions.device)
        if drop_positions.numel() == 1:
            return drop_positions

        diffs = drop_positions[1:] - drop_positions[:-1]
        run_end_mask = torch.cat([
            diffs != 1,
            torch.tensor([True], device=drop_positions.device, dtype=torch.bool)
        ])
        return drop_positions[run_end_mask]

    def _build_token_keep_mask(
        self,
        image_token_mask: torch.Tensor,
        patch_scores_per_sample: List[torch.Tensor],
        n_image_tokens_per_sample: torch.Tensor,
        visual_reduct_ratio: float,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Build a boolean mask indicating which tokens to keep after selection."""
        batch_size = input_ids.shape[0]
        token_keep_mask = torch.ones_like(input_ids, dtype=torch.bool)

        for b in range(batch_size):
            img_positions = torch.nonzero(image_token_mask[b], as_tuple=False).squeeze(1)
            n_img = int(n_image_tokens_per_sample[b].item())

            if n_img == 0:
                continue

            n_keep = int(n_img * (1.0 - visual_reduct_ratio))

            if n_keep > 0:
                ps = patch_scores_per_sample[b].view(-1)
                assert ps.numel() == n_img, (
                    f"patch_scores length ({ps.numel()}) != image tokens ({n_img}) for sample {b}"
                )

                top_idx = self._select_top_patches_with_tiebreaking(ps, n_keep, input_ids[b])
                keep_positions = img_positions[top_idx.to(input_ids.device)]

                token_keep_mask[b, img_positions] = False
                token_keep_mask[b, keep_positions] = True
            else:
                # Drop all image tokens for this sample
                token_keep_mask[b] = ~image_token_mask[b]

        return token_keep_mask

    def _apply_selection_to_sample(
        self,
        b: int,
        input_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        visual_pos_masks: Optional[torch.Tensor],
        deepstack_visual_embeds: Optional[List[torch.Tensor]],
        image_token_mask: torch.Tensor,
        token_keep_mask: torch.Tensor,
        image_drop_end_token_id: int,
        image_drop_embed: Optional[torch.Tensor],
        orig_seq_len: int,
    ) -> dict:
        """Apply token selection to a single sample, returning selected tensors and metadata."""
        keep_mask_b = token_keep_mask[b]
        img_mask_b = image_token_mask[b]
        drop_mask_b = img_mask_b & (~keep_mask_b)
        img_keep_mask_b = keep_mask_b[img_mask_b]

        # Find last position of each consecutive run of dropped tokens
        drop_positions = torch.nonzero(drop_mask_b, as_tuple=False).squeeze(1)
        run_last_positions = self._find_consecutive_run_ends(drop_positions)

        # Build final selection mask (keep original kept tokens + run-end markers)
        final_keep_mask = keep_mask_b.clone()
        if run_last_positions.numel() > 0:
            final_keep_mask[run_last_positions] = True

        selected_positions = torch.nonzero(final_keep_mask, as_tuple=False).squeeze(1)

        # Identify which selected positions are drop markers
        is_drop_marker = torch.zeros((orig_seq_len,), dtype=torch.bool, device=input_ids.device)
        if run_last_positions.numel() > 0:
            is_drop_marker[run_last_positions] = True
        drop_marker_mask = is_drop_marker[selected_positions]

        # Select and modify tensors
        result = {
            "selected_positions": selected_positions,
            "final_keep_mask": final_keep_mask,
            "image_token_keep_mask": img_keep_mask_b,
            "drop_marker_mask": drop_marker_mask,
        }

        if input_ids is not None:
            ids_kept = input_ids[b, selected_positions].clone()
            if drop_marker_mask.any():
                ids_kept[drop_marker_mask] = image_drop_end_token_id
            result["input_ids"] = ids_kept

        if inputs_embeds is not None:
            embeds_kept = inputs_embeds[b, selected_positions, :].clone()
            if drop_marker_mask.any():
                embeds_kept[drop_marker_mask] = image_drop_embed.expand(
                    int(drop_marker_mask.sum().item()), -1
                )
            result["inputs_embeds"] = embeds_kept

        if attention_mask is not None:
            result["attention_mask"] = attention_mask[b, selected_positions]

        if labels is not None:
            labels_kept = labels[b, selected_positions].clone()
            if drop_marker_mask.any():
                labels_kept[drop_marker_mask] = -100
            result["labels"] = labels_kept

        if visual_pos_masks is not None:
            visual_pos_mask_kept = visual_pos_masks[b, selected_positions].clone()
            if drop_marker_mask.any():
                visual_pos_mask_kept[drop_marker_mask] = 0
            result["visual_pos_masks"] = visual_pos_mask_kept

        if deepstack_visual_embeds is not None:
            result["deepstack_visual_embeds"] = deepstack_visual_embeds[b][img_keep_mask_b, :]

        return result

    def visual_token_selection_with_patch_scores(
        self,
        visual_reduct_ratio: float,
        patch_scores: torch.Tensor,
        input_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
        return_dict: bool = False,
        verbose: bool = False,
        verbose_visual_token_selection_summary: bool = False,
    ):
        """
        Select visual tokens based on patch importance scores to reduce sequence length.

        This method performs token selection by:
        1. Computing which image tokens to keep based on patch_scores
        2. Compressing consecutive dropped tokens into single <image_drop_end> markers
        3. Rebuilding all input tensors with the reduced sequence

        Args:
            visual_reduct_ratio: Fraction of image tokens to drop (0.0 to 1.0)
            patch_scores: Importance scores for each image token [B, N] or [N]
            input_ids: Token IDs [B, L]
            inputs_embeds: Token embeddings [B, L, D]
            position_ids: Position IDs [3, B, L] for RoPE
            attention_mask: Attention mask [B, L]
            cache_position: Cache positions for KV cache
            labels: Training labels [B, L]
            visual_pos_masks: Visual position masks for Qwen3-VL
            deepstack_visual_embeds: Deep-stacked visual embeddings (list of tensors)
            return_dict: Whether to return dict or tuple
            verbose: Enable detailed logging
            verbose_visual_token_selection_summary: Log selection summary

        Returns:
            Selected tensors with reduced sequence length, plus selection statistics
        """
        batch_size, orig_seq_len = input_ids.shape
        device = input_ids.device

        # Initialize outputs for no-selection case
        image_token_mask = (input_ids == self.config.image_token_id).to(device)
        token_keep_mask = torch.ones_like(input_ids, dtype=torch.bool)
        image_token_keep_mask = [token_keep_mask[b][image_token_mask[b]] for b in range(batch_size)]

        n_image_tokens_per_sample = image_token_mask.sum(dim=1)
        n_image_tokens_total = int(n_image_tokens_per_sample.sum().item())

        actual_token_percentage = 1.0
        visual_token_percentage = 1.0

        # Only apply selection if there are image tokens and reduction is requested
        if n_image_tokens_total > 0 and visual_reduct_ratio > 0:
            # Step 1: Prepare per-sample patch scores
            patch_scores_per_sample = self._prepare_patch_scores_per_sample(
                patch_scores, n_image_tokens_per_sample, batch_size, device
            )

            # Step 2: Build token keep mask based on patch scores
            token_keep_mask = self._build_token_keep_mask(
                image_token_mask, patch_scores_per_sample,
                n_image_tokens_per_sample, visual_reduct_ratio, input_ids
            )
            image_token_keep_mask = [token_keep_mask[b][image_token_mask[b]] for b in range(batch_size)]

            # Step 3: Get image drop token embedding
            image_drop_end_token_id = self.config.image_drop_end_token_id

            image_drop_embed = None
            if inputs_embeds is not None:
                image_drop_embed = self.get_input_embeddings()(
                    torch.tensor([int(image_drop_end_token_id)], device=device)
                ).squeeze(0)

            # Step 4: Apply selection to each sample
            sample_results = []
            for b in range(batch_size):
                result = self._apply_selection_to_sample(
                    b, input_ids, inputs_embeds, attention_mask, labels,
                    visual_pos_masks, deepstack_visual_embeds,
                    image_token_mask, token_keep_mask, image_drop_end_token_id,
                    image_drop_embed, orig_seq_len
                )
                sample_results.append(result)

            # Step 5: Stack results into batched tensors
            selected_positions_list = [r["selected_positions"] for r in sample_results]
            image_token_keep_mask = torch.stack([r["image_token_keep_mask"] for r in sample_results], dim=0)
            token_keep_mask = torch.stack([r["final_keep_mask"] for r in sample_results], dim=0)

            if input_ids is not None:
                input_ids = torch.stack([r["input_ids"] for r in sample_results], dim=0)
            if inputs_embeds is not None:
                inputs_embeds = torch.stack([r["inputs_embeds"] for r in sample_results], dim=0)
            if attention_mask is not None:
                attention_mask = torch.stack([r["attention_mask"] for r in sample_results], dim=0)
            if labels is not None:
                labels = torch.stack([r["labels"] for r in sample_results], dim=0)
            if visual_pos_masks is not None:
                visual_pos_masks = torch.stack([r["visual_pos_masks"] for r in sample_results], dim=0)
            if deepstack_visual_embeds is not None:
                deepstack_visual_embeds = [r["deepstack_visual_embeds"] for r in sample_results]

            # Step 6: Update position_ids for the new sequence length
            max_kept_len = max(pos.shape[0] for pos in selected_positions_list)

            if position_ids is not None and position_ids.dim() == 3:
                new_position_ids = torch.zeros(
                    (position_ids.shape[0], batch_size, max_kept_len),
                    dtype=position_ids.dtype, device=device,
                )
                for b in range(batch_size):
                    sel_pos = selected_positions_list[b]
                    kept_len = sel_pos.shape[0]
                    if kept_len > 0:
                        new_position_ids[:, b, -kept_len:] = position_ids[:, b, sel_pos]
                position_ids = new_position_ids

            if cache_position is not None:
                cache_position = torch.arange(max_kept_len, device=device)

            # Align attention_mask to current sequence length
            if attention_mask is not None and attention_mask.ndim == 2:
                target_len = input_ids.shape[1]
                if attention_mask.shape[1] != target_len:
                    if attention_mask.shape[1] > target_len:
                        attention_mask = attention_mask[:, -target_len:]
                    else:
                        pad_len = target_len - attention_mask.shape[1]
                        pad_mask = torch.zeros(
                            (attention_mask.shape[0], pad_len),
                            dtype=attention_mask.dtype, device=device
                        )
                        attention_mask = torch.cat([pad_mask, attention_mask], dim=1)

            # Step 7: Compute statistics
            n_image_tokens_after = int((input_ids == self.config.image_token_id).sum().item())
            n_tokens_after = input_ids.shape[1]
            actual_reduct_ratio = (
                (n_image_tokens_total - n_image_tokens_after) / (orig_seq_len * batch_size)
                if orig_seq_len > 0 else 0
            )
            actual_token_percentage = 1 - actual_reduct_ratio
            visual_token_percentage = 1 - visual_reduct_ratio

            if verbose_visual_token_selection_summary:
                rank0_print(
                    f"Visual token selection - target_ratio={visual_reduct_ratio:.2f}, "
                    f"actual_ratio={actual_reduct_ratio:.2f}: "
                    f"image_tokens={n_image_tokens_total} -> {n_image_tokens_after}, "
                    f"seq_len={orig_seq_len} -> {n_tokens_after}"
                )

        if return_dict:
            return {
                "input_ids": input_ids,
                "inputs_embeds": inputs_embeds,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "labels": labels,
                "patch_scores": patch_scores,
                "visual_pos_masks": visual_pos_masks,
                "deepstack_visual_embeds": deepstack_visual_embeds,
                "visual_token_percentage": visual_token_percentage,
                "actual_token_percentage": actual_token_percentage,
                "token_keep_mask": token_keep_mask,
                "image_token_keep_mask": image_token_keep_mask,
            }
        else:
            return (
                input_ids,
                inputs_embeds,
                position_ids,
                attention_mask,
                cache_position,
                labels,
                patch_scores,
                visual_pos_masks,
                deepstack_visual_embeds,
                actual_token_percentage,
                visual_token_percentage,
                token_keep_mask,
                image_token_keep_mask,
            )

