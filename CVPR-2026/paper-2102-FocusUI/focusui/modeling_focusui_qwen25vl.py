"""
FocusUI wrapper around Qwen2.5-VL.
"""

from typing import Any, List, Tuple, Union, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import (
    is_torchdynamo_compiling,
)
from transformers.cache_utils import Cache, StaticCache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast, Qwen2_5_VLForConditionalGeneration

from focusui.trainer import rank0_print
from focusui.modeling_patch_scorer import PatchScorerConfig, PatchScorerModel

class FocusUI_QwenVLwithVisionHeadOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast):
    """
    Output class for Qwen2_5_VL with pointer head, extending the base output class.
    
    Args:
        lm_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Language modeling loss.
        pointer_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Vision pointer network loss.
        pointer_scores (`List[torch.FloatTensor]`, *optional*):
            Attention scores from the pointer network, one tensor per batch item.
        patch_scores (`torch.Tensor`, *optional*):
            Patch-level scores predicted by `PatchScorerModel` for the image tokens.
        token_keep_mask (`torch.Tensor`, *optional*):
            Boolean mask over the (possibly shortened) input sequence indicating which tokens were kept after
            visual token selection. Shape: `(batch_size, seq_len_after)`.
        actual_token_percentage (`float`, *optional*):
            Ratio of tokens kept after selection.
        ps_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            PatchScorer loss (if `patch_scores_label` was provided).
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Combined loss (weighted sum of lm_loss and pointer_loss).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores from the language modeling head.
        past_key_values, hidden_states, attentions, rope_deltas:
            Same as parent class.
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
        self.ps_loss = ps_loss

        self.pointer_scores = pointer_scores

        self.patch_scores = patch_scores
        self.token_keep_mask = token_keep_mask
        self.actual_token_percentage = actual_token_percentage

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
        
        # Note: We omit additional normalization here because Qwen2VL
        # already normalizes hidden states using RMSNorm.
        self.projection_enc = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, d_model)
        )
        self.projection_dec = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, d_model)
        )

        # Add self-attention layer for visual features
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Layer normalization and residual connection
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(
        self,
        hidden_state_enc,  # shape: [n_enc, d_model] where n_enc can vary with image size
        hidden_state_dec,  # shape: [n_dec, d_model] there can be multiple query in one sample
        labels: Optional[torch.Tensor] = None,  # shape: [n_dec, n_enc], binary mask of patches in bbox
    ):
        
        enc_input = hidden_state_enc.unsqueeze(0)
        attn_output, _ = self.self_attention(
            query=enc_input,
            key=enc_input,
            value=enc_input,
            # attn_mask=attention_mask,
            need_weights=False
        )
        # Residual connection and layer normalization
        hidden_state_enc_ctx = self.layer_norm(enc_input + self.dropout(attn_output))
        # Remove batch dimension
        hidden_state_enc_ctx = hidden_state_enc_ctx.squeeze(0)  # [n_enc, d_model]

        # Apply the projection networks.
        proj_enc = self.projection_enc(hidden_state_enc_ctx)  # [n_enc, d_model]
        proj_dec = self.projection_dec(hidden_state_dec)  # [n_dec, d_model]
        
        # Compute scaled dot-product attention scores.
        # Scaling by sqrt(d_model) is critical regardless of variable n_enc.
        scaling = self.d_model ** 0.5
        patch_logits = torch.matmul(proj_dec, proj_enc.transpose(0, 1)) / scaling  # [n_dec, n_enc]
        
        # Softmax normalization is applied along the encoder dimension.
        attn_weights = F.softmax(patch_logits, dim=-1)

        loss = None
        if labels is not None:
            epsilon = 1e-8
            labels_float = labels.float()
            # Normalize each row to get target probability distribution
            target_dist = labels_float / (labels_float.sum(dim=-1, keepdim=True) + epsilon)

            # Apply log_softmax to logits
            pred_log_probs = F.log_softmax(patch_logits, dim=-1)

            # Use KL divergence as loss
            loss = F.kl_div(pred_log_probs, target_dist, reduction='batchmean')

        return attn_weights, loss


class FocusUI_Qwen2_5_VLForConditionalGenerationWithPointer(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_patch_pointer_head = VisionHead_MultiPatch(self.config.hidden_size, self.config.hidden_size)
        self.pointer_loss_weight = kwargs.get("pointer_loss_weight", 1.0)
        self.lm_loss_weight = kwargs.get("lm_loss_weight", 1.0)
        self.ps_loss_weight = kwargs.get("ps_loss_weight", 1.0)
        
        self.apply_visual_token_select = kwargs.get("apply_visual_token_select", True)
        self.train_visual_reduct_ratio = kwargs.get("train_visual_reduct_ratio", (0.0, 0.95))
        self.visual_reduct_ratio = kwargs.get("visual_reduct_ratio", 0.5)

        self.patch_scorer_config = kwargs.get("patch_scorer_config",
            PatchScorerConfig(
                projection_dim=self.config.vision_config.out_hidden_size,
                projection_dropout=0.1,
                text_token_pooling="mean",
            )
        )
        self.patch_scorer = PatchScorerModel(self.patch_scorer_config)
        self.patch_scorer_early_exit = kwargs.get("patch_scorer_early_exit", False)
        
        # Cache latest patch_scores for retrieval after generate()
        self._last_patch_scores = None
        # Cache latest PatchScorer GPU memory stats
        self._last_ps_mem_stats = None

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
        input_ids: torch.LongTensor = None, # (batch_size, seq_len)
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
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
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        # Grounding
        visual_token_indices_of_coordinates: Optional[torch.Tensor] = None, # shape: (batch_size, n_target); each element is the ground-truth index of the visual token that should be attended to for the corresponding target token
        multi_patch_labels: Optional[torch.Tensor] = None, # shape: list [(n_target, n_visual), ...]; binary mask of patches in bbox
        coordinates: Optional[List[Tuple[float, float]]] = None,
        # Focus-UI
        patch_scores: Optional[torch.Tensor] = None,  # as pred (simulate pred)
        patch_scores_label: Optional[torch.Tensor] = None,  # as gt (compute loss)
        focus_input_ids: Optional[torch.LongTensor] = None,
        focus_attention_mask: Optional[torch.Tensor] = None,
        # debug
        verbose: bool = False,
        verbose_visual_token_selection_summary: bool = False,
    ) -> Union[Tuple, FocusUI_QwenVLwithVisionHeadOutputWithPast]:

        """
        FocusUI Forward Pass:
            1) Vision encoder forward pass (from Qwen2_5_VLForConditionalGeneration.forward())
            2) Build patch scores
            3) Visual token selection
            4) LM decoder forward pass (from Qwen2_5_VLForConditionalGeneration.forward())
            5) Pointer head grounding
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        ######## Vision encoder forward pass ########
        if inputs_embeds is None:
            inputs_embeds = self.model.language_model.embed_tokens(input_ids) # shape: (batch_size, seq_len, d_model)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.model.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.model.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.model.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    delta = delta.to(position_ids.device)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        ########

        ##### Build Patch Scores #####
        ps_loss = None

        if patch_scores is None and pixel_values is not None:
            instruct_embeds = self.model.language_model.embed_tokens(focus_input_ids) if focus_input_ids is not None else None

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
        if self.apply_visual_token_select and (patch_scores is not None) and (inputs_embeds.shape[1] != 1): 
            # set spatial drop rate
            if self.training:
                visual_reduct_ratio = np.random.uniform(self.train_visual_reduct_ratio[0], self.train_visual_reduct_ratio[1])
                visual_reduct_ratio = np.clip(visual_reduct_ratio, 0.0, 0.99) # avoid too many tokens dropped
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
                return_dict=False,
                verbose=verbose,
                verbose_visual_token_selection_summary=verbose_visual_token_selection_summary,
            )

        else:
            actual_token_percentage = 1.0
            visual_token_percentage = 1.0
            token_keep_mask = None
            image_token_keep_mask = None
        
        # for train scorer only
        if self.patch_scorer_early_exit:
            return FocusUI_QwenVLwithVisionHeadOutputWithPast(
                loss=ps_loss,
                rope_deltas=self.model.rope_deltas,
                patch_scores=patch_scores,
                actual_token_percentage=actual_token_percentage,
                token_keep_mask=token_keep_mask,
                ps_loss=ps_loss,
            )

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0] # shape: (batch_size, seq_len, d_model)
        logits = self.lm_head(hidden_states)

        lm_loss = None
        if labels is not None and self.lm_loss_weight > 0:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            lm_loss = loss_fct(shift_logits, shift_labels)

        # If vision supervision is requested, process the action head.
        pointer_loss = None
        pointer_scores = []
        if visual_token_indices_of_coordinates is not None:
            batch_size = input_ids.shape[0]
            pointer_losses = []
            
            # Process each sample individually because the number of visual and target tokens may vary.
            for i in range(batch_size):
                dummy_target = False

                # Get the token ids and corresponding hidden states for sample i.
                token_ids = input_ids[i]          # shape: (seq_length,)
                hs = hidden_states[i]             # shape: (seq_length, d_model)

                # Identify visual tokens indices.
                visual_mask = (token_ids == self.config.image_token_id)
                visual_indices = torch.nonzero(visual_mask, as_tuple=False).squeeze(-1) # shape: (n_visual,)

                # Identify target tokens (the ones that should attend to visual features).
                target_mask = (token_ids == self.config.pointer_pad_token_id)
                target_indices = torch.nonzero(target_mask, as_tuple=False).squeeze(-1)
                
                # If either visual or target tokens are missing, skip this sample.
                if visual_indices.numel() == 0:
                    raise ValueError(f"No visual or target tokens found for sample {i}.")
                if target_indices.numel() == 0:
                    target_indices = torch.tensor([hs.shape[0] - 1]) # take the last token as the dummy target token
                    sample_labels = torch.zeros_like(visual_indices).unsqueeze(0)
                    sample_labels[0][:4] = 1
                    dummy_target = True
                else:
                    # For supervision, we assume that visual_token_indices_of_coordinates[i] is a tensor of shape (n_target,)
                    # where each element is an integer in the range [0, n_visual-1] indicating the ground-truth visual token.
                    sample_labels = multi_patch_labels[i]
                
                # Gather the corresponding hidden state representations.
                visual_embeds = inputs_embeds[i][visual_indices]  # shape: (n_visual, d_model)
                target_hidden = hs[target_indices]  # shape: (n_target, d_model)

                # Calculate loss for multi-patch mode
                # Ensure the number of targets matches between sample and labels
                if sample_labels.shape[0] != target_indices.shape[0]:
                    raise ValueError(f"Sample {i} has mismatched target counts: {sample_labels.shape[0]} labels but found {target_indices.shape[0]} target tokens")
                
                if self.apply_visual_token_select and image_token_keep_mask is not None:
                    sample_labels = sample_labels[:, image_token_keep_mask[i]]

                # Process using VisionHead_MultiPatch
                attn_scores, loss_v = self.multi_patch_pointer_head(
                    visual_embeds,
                    target_hidden,
                    labels=sample_labels,
                )
                
                pointer_scores.append(attn_scores.detach().cpu())
                pointer_losses.append(loss_v * 0.0 if dummy_target else loss_v)
            
            pointer_loss = torch.stack(pointer_losses).mean()

        # Combine the LM loss and vision loss using the provided loss weights.
        total_loss = 0.0

        if pointer_loss is not None:
            total_loss += self.pointer_loss_weight * pointer_loss
        if lm_loss is not None:
            total_loss += self.lm_loss_weight * lm_loss
        if ps_loss is not None:
            total_loss += self.ps_loss_weight * ps_loss
        
        if return_dict:
            return FocusUI_QwenVLwithVisionHeadOutputWithPast(
                lm_loss=lm_loss,
                pointer_loss=pointer_loss,
                pointer_scores=pointer_scores,
                loss=total_loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=self.model.rope_deltas,
                patch_scores=patch_scores,
                actual_token_percentage=actual_token_percentage,
                token_keep_mask=token_keep_mask,
                ps_loss=ps_loss,
            )
        else:
            # When labels are provided, parent's forward returns a tuple with loss as the first element.
            if labels is not None:
                # Replace the LM loss with the combined loss.
                output = (lm_loss, pointer_loss, logits, pointer_scores,) + outputs[1:]
                return (total_loss,) + output if total_loss is not None else output
            else:
                return outputs

    def generate_with_visual_token_select(self, *args, **kwargs):
        """
        Wrap parent generate to attach the last computed patch_scores to the returned
        output object when return_dict_in_generate=True.
        """
        outputs = super().generate(*args, **kwargs)
        return outputs, self._last_patch_scores

    def get_last_patch_scorer_memory_stats(self):
        return self._last_ps_mem_stats

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
        second_per_grid_ts=None,
        patch_scores=None,
        patch_scores_label=None,
        focus_input_ids=None,
        focus_attention_mask=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
        # generate the first token for each sequence. Later use the generated Input ids for continuation.
        if past_key_values is not None:
            if inputs_embeds is not None and input_ids.shape[1] == 0:  # Exception 4
                inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
            elif (
                inputs_embeds is not None  # Exception 1
                or (is_torchdynamo_compiling() or cache_position[-1] >= input_ids.shape[1])  # Exception 3
            ):
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.lm_head.weight.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "cache_position": cache_position,
                "second_per_grid_ts": second_per_grid_ts,
                # Focus-UI
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
            "image_token_keep_mask": keep_mask_b[img_mask_b],
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
            image_drop_end_token_id = getattr(self.config, "image_drop_end_token_id", None)
            if image_drop_end_token_id is None:
                image_drop_end_token_id = (
                    self.config.pad_token_id if self.config.pad_token_id is not None
                    else self.config.bos_token_id
                )

            image_drop_embed = None
            if inputs_embeds is not None:
                image_drop_embed = self.model.language_model.embed_tokens(
                    torch.tensor([int(image_drop_end_token_id)], device=device)
                ).squeeze(0)

            # Step 4: Apply selection to each sample
            sample_results = []
            for b in range(batch_size):
                result = self._apply_selection_to_sample(
                    b, input_ids, inputs_embeds, attention_mask, labels,
                    image_token_mask, token_keep_mask, image_drop_end_token_id,
                    image_drop_embed, orig_seq_len
                )
                sample_results.append(result)

            # Step 5: Stack results into batched tensors
            selected_positions_list = [r["selected_positions"] for r in sample_results]
            image_token_keep_mask = torch.stack([r["image_token_keep_mask"] for r in sample_results])
            token_keep_mask = torch.stack([r["final_keep_mask"] for r in sample_results], dim=0)

            if input_ids is not None:
                input_ids = torch.stack([r["input_ids"] for r in sample_results], dim=0)
            if inputs_embeds is not None:
                inputs_embeds = torch.stack([r["inputs_embeds"] for r in sample_results], dim=0)
            if attention_mask is not None:
                attention_mask = torch.stack([r["attention_mask"] for r in sample_results], dim=0)
            if labels is not None:
                labels = torch.stack([r["labels"] for r in sample_results], dim=0)

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

        # Align attention_mask to current sequence length
        if attention_mask is not None and attention_mask.ndim == 2:
            target_len = input_ids.shape[1]
            if attention_mask.shape[1] > target_len:
                attention_mask = attention_mask[:, -target_len:]

        if return_dict:
            return {
                "input_ids": input_ids,
                "inputs_embeds": inputs_embeds,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "labels": labels,
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
                actual_token_percentage,
                visual_token_percentage,
                token_keep_mask,
                image_token_keep_mask,
            )
