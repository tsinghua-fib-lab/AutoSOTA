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
# Modified from Dream repos: https://github.com/HKUNLP/Dream

import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)

logger = logging.get_logger(__name__)

def get_transfer_index(confidence: torch.Tensor, number_transfer_tokens: int, alg_temp: float) -> torch.Tensor:
    if alg_temp is None or alg_temp == 0:
        _, transfer_index = torch.topk(confidence, number_transfer_tokens)
        return transfer_index
    else:
        confidence = confidence / alg_temp
        confidence = F.softmax(confidence, dim=-1)
        return torch.multinomial(confidence, num_samples=number_transfer_tokens)

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass

class DreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            generation_config = DreamGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.
        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", None)

        # 2. Define model inputs
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        
        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and 
            attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask 
        )
        block_length = kwargs.get("block_length", 32)
        method = kwargs.get("method", "original")
        if method == "original":
            return self._sample(
                input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                generation_tokens_hook_func=generation_tokens_hook_func,
                block_length=block_length,
                method=method
            )

        elif method == "dc_leap":
            return self._generate_with_dc_leap(
                input_ids=input_ids, 
                generation_config=generation_config,
                **kwargs
            )

        
    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func: Optional[callable],
        block_length: Optional[int] = 32,
        method: str = "original"
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp

        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        gen_length = max_length - input_ids.shape[1]
        
        # Handle block configuration
        if block_length is None:
            block_length = gen_length  # Default: single block (original behavior)
        
        assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
        num_blocks = gen_length // block_length
        
        assert steps % num_blocks == 0, f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
        steps_per_block = steps // num_blocks
        timesteps = torch.linspace(1, generation_config.eps, steps_per_block + 1, device=x.device)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        # Initialize cache for the prompt
        past_key_values = None

        # Process each block
        for num_block in range(num_blocks):
            if (x == mask_token_id).sum() == 0:
                break
            
            current_block_start = input_ids.shape[1] + num_block * block_length
            current_block_end = current_block_start + block_length
            
            no_cache_slice = slice(None)
            
            i = 1
            while (x[:, current_block_start:current_block_end] == mask_token_id).any():
                mask_index = (x[:, no_cache_slice] == mask_token_id)

                mask_index[:, current_block_end:] = False
                
                # Prepare attention mask for cached generation
                if attention_mask != "full":
                    # Adjust attention mask for current position
                    current_attention_mask = attention_mask[:, :, :, current_block_start:]
                else:
                    current_attention_mask = attention_mask
                
                model_output = self(x[:, no_cache_slice],
                                    current_attention_mask, 
                                    tok_idx[:, no_cache_slice] if tok_idx is not None else None, 
                                    past_key_values=past_key_values,
                                    use_cache=("cache" in method),
                                    dual_cache=("dual_cache" in method),
                                    replace_position= None
                                    )
                logits = model_output.logits
                logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
                if i == steps_per_block:
                    break
                t = timesteps[i]
                s = timesteps[i + 1]
                mask_logits = logits[mask_index]
                confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=("L2P" not in method))
                # this allows user-defined token control of the intermediate steps
                if generation_tokens_hook_func:
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k)
                    i, x, histories = generation_tokens_hook_func(i, x, x0, confidence, current_block_start, current_block_end, mask_index, histories)
                    continue
                num_mask_token = mask_index.sum() / mask_index.shape[0]
                number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps_per_block - 1 else int(num_mask_token)
                full_confidence = torch.full_like(x[:, no_cache_slice], -torch.inf, device=self.device, dtype=logits.dtype)
                full_confidence[mask_index] = confidence
                
                parallel_transfer = False
                
                if not parallel_transfer and number_transfer_tokens > 0:
                    transfer_index = get_transfer_index(full_confidence, number_transfer_tokens, alg_temp)
                    x_ = torch.zeros_like(x[:, no_cache_slice], device=self.device, dtype=torch.long) + mask_token_id
                    x_[mask_index] = x0.clone()
                    row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                    x[:, no_cache_slice][row_indices,transfer_index] = x_[row_indices,transfer_index]
                
                i += 1

        
        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
            )
        else:
            return x

    @staticmethod
    def _get_top1_info(logits: torch.Tensor):
        probs = F.softmax(logits, dim=-1)
        top1_probs, top1_indices = torch.max(probs, dim=-1)
        return top1_indices, top1_probs

    @staticmethod
    def _verify_and_commit(
        region_logits: torch.Tensor,
        region_x: torch.Tensor,
        mask_id: int,
        commit_thres: float,
        left_boundary_known: bool
    ):
        top1_indices, top1_probs = DreamGenerationMixin._get_top1_info(region_logits)
        is_confident =  (top1_probs > commit_thres)
        if left_boundary_known:
            contiguity_mask = torch.cumprod(is_confident.int(), dim=0).bool()
        else:
            contiguity_mask = torch.zeros_like(is_confident, dtype=torch.bool)
        is_mask = (region_x == mask_id)
        final_commit_mask = contiguity_mask & is_mask
        
        return top1_indices, final_commit_mask

    @staticmethod
    def _get_top1_info(logits: torch.Tensor):
        probs = F.softmax(logits, dim=-1)
        top1_probs, top1_indices = torch.max(probs, dim=-1)
        return top1_indices, top1_probs

    @staticmethod
    def _verify_and_commit(
        region_logits: torch.Tensor,
        region_x: torch.Tensor,
        mask_id: int,
        commit_thres: float,
        left_boundary_known: bool
    ):
        top1_indices, top1_probs = DreamGenerationMixin._get_top1_info(region_logits)
        is_confident =  (top1_probs > commit_thres)
        if left_boundary_known:
            contiguity_mask = torch.cumprod(is_confident.int(), dim=0).bool()
        else:
            contiguity_mask = torch.zeros_like(is_confident, dtype=torch.bool)
        is_mask = (region_x == mask_id)
        final_commit_mask = contiguity_mask & is_mask
        
        return top1_indices, final_commit_mask

    @torch.no_grad()
    def _generate_with_dc_leap(
        self,
        input_ids: torch.Tensor,
        generation_config: DreamGenerationConfig,
        **kwargs
    ) -> torch.Tensor:
        prompt = input_ids
        model = self  
        steps = generation_config.steps
        commit_thres = kwargs.get("commit_thres", 0.7) 
        draft_thres = kwargs.get("draft_thres", 0.98)
        gen_length = generation_config.max_new_tokens if generation_config.max_new_tokens else 256
        block_length = kwargs.get("block_length", 32)
        max_window_size = kwargs.get("max_window_size", 128)
        cfg_scale = kwargs.get("cfg_scale", 0.0)
        temperature = generation_config.temperature
        remasking = kwargs.get("remasking", 'low_confidence')
        mask_id = generation_config.mask_token_id
        device = model.device

        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
        x[:, :prompt.shape[1]] = prompt
        prompt_len = prompt.shape[1]
        
        draft_bank = torch.full((gen_length,), mask_id, dtype=torch.long, device=device)
        verified_end = 0 
        
        while verified_end < gen_length:
            
            l2r_len = max_window_size
            future_len = max_window_size
            
            win_s = verified_end
            win_e = min(verified_end + l2r_len + future_len, gen_length)
            
            abs_win_s = prompt_len + win_s
            abs_win_e = prompt_len + win_e
            
            if abs_win_s >= abs_win_e: break
            
            x_for_prediction = x.clone()
            drafts = draft_bank[win_s:win_e]
            
            target_slice = x_for_prediction[0, abs_win_s:abs_win_e]
            mask_locs = (target_slice == mask_id)
            valid_drafts = (drafts != mask_id)
            fill_locs = mask_locs & valid_drafts
            
            if fill_locs.any():
                x_for_prediction[0, abs_win_s:abs_win_e][fill_locs] = drafts[fill_locs]
            
            leader_end = min(abs_win_s + max_window_size, abs_win_e)
            x_for_prediction[0, abs_win_s:leader_end] = mask_id

            curr_attention_mask = torch.ones_like(x_for_prediction, device=device, dtype=torch.bool)
            curr_attention_mask = torch.logical_and(
                curr_attention_mask.unsqueeze(1).unsqueeze(-2),
                curr_attention_mask.unsqueeze(1).unsqueeze(-1),
            )

            if cfg_scale > 0.:
                pred_model_out = self(x_for_prediction, attention_mask=curr_attention_mask, output_hidden_states=False)
                prediction_logits = pred_model_out.logits
                pred_conditional, pred_unconditional = prediction_logits.chunk(2, dim=0)
                prediction_logits = pred_unconditional + (cfg_scale + 1) * (pred_conditional - pred_unconditional)
            else:
                pred_model_out = self(x_for_prediction, attention_mask=curr_attention_mask, output_hidden_states=False)
                prediction_logits = pred_model_out.logits

            prediction_logits = torch.cat([prediction_logits[:,:1], prediction_logits[:, :-1]], dim=1)
            
            l2r_abs_end = min(prompt_len + verified_end + l2r_len, abs_win_e)
            
            if l2r_abs_end > abs_win_s:
                region_logits = prediction_logits[0, abs_win_s:l2r_abs_end]
                region_x = x[0, abs_win_s:l2r_abs_end]
                
                left_boundary_known = True
                if verified_end > 0:
                    left_boundary_known = (x[0, abs_win_s - 1].item() != mask_id)
                
                top1_tokens, commit_mask = self._verify_and_commit(
                    region_logits, region_x, mask_id, 
                    commit_thres, left_boundary_known
                )

                if not commit_mask.any():
                    is_mask = (region_x == mask_id)
                    if is_mask.any():
                        first_idx = is_mask.nonzero(as_tuple=True)[0][0]
                        commit_mask[first_idx] = True

                if commit_mask.any():
                    update_idx = commit_mask.nonzero(as_tuple=True)[0]
                    x[0, abs_win_s + update_idx] = top1_tokens[update_idx]
                
                    next_mask = (x[0, abs_win_s:l2r_abs_end] == mask_id).nonzero(as_tuple=True)
                    if next_mask[0].numel() > 0:
                        verified_end += next_mask[0][0].item()
                    else:
                        verified_end += (l2r_abs_end - abs_win_s)
            
            draft_logits = prediction_logits[0, abs_win_s:abs_win_e]
            r_idx, r_probs = self._get_top1_info(draft_logits)
            
            draft_candidates_mask = r_probs > draft_thres
            if draft_candidates_mask.any():
                indices = draft_candidates_mask.nonzero(as_tuple=True)[0]
                bank_indices = win_s + indices
                
                valid = bank_indices < gen_length
                if valid.all():
                    draft_bank[bank_indices] = r_idx[indices]
                elif valid.any():
                    draft_bank[bank_indices[valid]] = r_idx[indices][valid]

        final_sequences = x[:, :prompt.shape[1] + gen_length]
        return_dict_in_generate = generation_config.return_dict_in_generate
        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=final_sequences,
                history=None, 
            )
        else:
            return final_sequences
