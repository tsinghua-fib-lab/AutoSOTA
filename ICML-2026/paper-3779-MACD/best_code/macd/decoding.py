from __future__ import annotations

from types import MethodType
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
try:
    from transformers.generation import utils as generation_utils
    from transformers.generation.logits_process import LogitsProcessorList
    from transformers.generation.stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
    try:
        from transformers.generation.utils import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput, SampleOutput
    except ImportError:
        # Transformers >= 5.0 renamed these
        from transformers.generation.utils import GenerateDecoderOnlyOutput as SampleDecoderOnlyOutput
        from transformers.generation.utils import GenerateEncoderDecoderOutput as SampleEncoderDecoderOutput
        from transformers.generation.utils import GenerateOutput as SampleOutput
except Exception:  # pragma: no cover - transformers is a runtime dependency for inference
    generation_utils = None
    LogitsProcessorList = None
    StoppingCriteriaList = None
    validate_stopping_criteria = None
    SampleDecoderOnlyOutput = None
    SampleEncoderDecoderOutput = None
    SampleOutput = Any


_ORIGINAL_VALIDATE_MODEL_KWARGS = (
    generation_utils.GenerationMixin._validate_model_kwargs if generation_utils is not None else None
)
_CD_KWARGS = {
    "use_cd",
    "input_ids_cd",
    "attention_mask_cd",
    "pixel_values_cd",
    "pixel_values_videos_cd",
    "image_grid_thw_cd",
    "video_grid_thw_cd",
    "second_per_grid_ts_cd",
    "mm_token_type_ids_cd",
    "cd_alpha",
    "cd_beta",
}


_SAVED_CD_KWARGS: dict[str, Any] = {}

def relax_cd_kwargs_validation() -> None:
    if generation_utils is None or _ORIGINAL_VALIDATE_MODEL_KWARGS is None:
        raise ImportError("transformers is required to enable MACD contrastive decoding")

    def _validate_model_kwargs_relaxed(self: Any, model_kwargs: dict[str, Any]) -> Any:
        global _SAVED_CD_KWARGS
        _SAVED_CD_KWARGS.clear()
        for key in list(model_kwargs.keys()):
            if key in _CD_KWARGS or (isinstance(key, str) and key.endswith("_cd")):
                _SAVED_CD_KWARGS[key] = model_kwargs.pop(key)
        return _ORIGINAL_VALIDATE_MODEL_KWARGS(self, model_kwargs)

    generation_utils.GenerationMixin._validate_model_kwargs = _validate_model_kwargs_relaxed


def attach_cd_method_if_missing(model: Any) -> None:
    if hasattr(model, "prepare_inputs_for_generation_cd"):
        return

    def prepare_inputs_for_generation_cd(
        self: Any,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool = True,
        pixel_values_cd: torch.Tensor | None = None,
        pixel_values_videos_cd: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        kwargs.pop("generation_config", None)
        kwargs.pop("use_cd", None)
        kwargs.pop("cd_alpha", None)
        kwargs.pop("cd_beta", None)

        if pixel_values_cd is not None:
            kwargs["pixel_values"] = pixel_values_cd
        if pixel_values_videos_cd is not None:
            kwargs["pixel_values_videos"] = pixel_values_videos_cd

        image_grid_cd = kwargs.pop("image_grid_thw_cd", None)
        video_grid_cd = kwargs.pop("video_grid_thw_cd", None)
        if image_grid_cd is not None:
            kwargs["image_grid_thw"] = image_grid_cd
        if video_grid_cd is not None:
            kwargs["video_grid_thw"] = video_grid_cd
        kwargs.pop("second_per_grid_ts", None)
        kwargs.pop("second_per_grid_ts_cd", None)
        # Handle mm_token_type_ids for CD inputs
        mm_type_cd = kwargs.pop("mm_token_type_ids_cd", None)
        if mm_type_cd is not None:
            kwargs["mm_token_type_ids"] = mm_type_cd
        for key in list(kwargs.keys()):
            if isinstance(key, str) and key.endswith("_cd"):
                kwargs.pop(key, None)

        if cache_position is None:
            cache_position = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        model_inputs = self.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
        )
        model_inputs["cache_position"] = cache_position
        model_inputs["position_ids"] = None
        if cache_position.numel() > 0 and int(cache_position[0]) != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
        return model_inputs

    model.prepare_inputs_for_generation_cd = MethodType(prepare_inputs_for_generation_cd, model)


def _drop_cd_keys(model_inputs: dict[str, Any]) -> None:
    blocked = {"generation_config", "return_tensors", "use_cd", "input_ids_cd", "attention_mask_cd", "cd_alpha", "cd_beta"}
    for key in list(model_inputs.keys()):
        if key in blocked or (isinstance(key, str) and key.endswith("_cd")):
            model_inputs.pop(key, None)


def _sample_with_cd(
    self: Any,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList | None = None,
    stopping_criteria: StoppingCriteriaList | None = None,
    generation_config: Any = None,
    synced_gpus: bool = False,
    streamer: Any | None = None,
    **model_kwargs: Any,
) -> SampleOutput | torch.LongTensor:
    # Restore CD kwargs that were saved by _validate_model_kwargs_relaxed
    global _SAVED_CD_KWARGS
    if _SAVED_CD_KWARGS:
        for _k, _v in _SAVED_CD_KWARGS.items():
            if _k not in model_kwargs:
                model_kwargs[_k] = _v
    if LogitsProcessorList is None or StoppingCriteriaList is None:
        raise ImportError("transformers is required to run MACD contrastive decoding")

    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    # Extract generation config values
    if generation_config is None:
        generation_config = self.generation_config
    max_length = getattr(generation_config, "max_length", None)
    pad_token_id = getattr(generation_config, "pad_token_id", None)
    eos_token_id = getattr(generation_config, "eos_token_id", None)
    output_scores = getattr(generation_config, "output_scores", False)
    output_attentions = getattr(generation_config, "output_attentions", False)
    output_hidden_states = getattr(generation_config, "output_hidden_states", False)
    return_dict_in_generate = getattr(generation_config, "return_dict_in_generate", False)

    if max_length is not None:
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

    # Build logits warper from generation config
    from transformers.generation.logits_process import LogitsProcessorList as LPL
    logits_warper = LPL()
    if hasattr(generation_config, "temperature") and generation_config.temperature > 0:
        from transformers.generation.logits_process import TemperatureLogitsWarper
        logits_warper.append(TemperatureLogitsWarper(generation_config.temperature))
    if hasattr(generation_config, "top_k") and generation_config.top_k is not None and generation_config.top_k > 0:
        from transformers.generation.logits_process import TopKLogitsWarper
        logits_warper.append(TopKLogitsWarper(top_k=generation_config.top_k))
    if hasattr(generation_config, "top_p") and generation_config.top_p is not None and generation_config.top_p < 1.0:
        from transformers.generation.logits_process import TopPLogitsWarper
        logits_warper.append(TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=1))

    eos_token_ids = [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
    eos_tensor = torch.tensor(eos_token_ids, device=input_ids.device) if eos_token_ids is not None else None

    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions and self.config.is_encoder_decoder) else None

    unfinished = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    this_peer_finished = False
    use_cd = bool(model_kwargs.get("use_cd")) or model_kwargs.get("pixel_values_videos_cd") is not None
    if model_kwargs.get("cache_position") is None:
        model_kwargs["cache_position"] = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

    input_ids_cd = model_kwargs.pop("input_ids_cd", input_ids.clone())
    attention_mask = model_kwargs.pop("attention_mask", None)
    attention_mask_cd = model_kwargs.pop("attention_mask_cd", attention_mask.clone() if isinstance(attention_mask, torch.Tensor) else None)

    while True:
        if synced_gpus:
            flag = torch.tensor(0.0 if this_peer_finished else 1.0, device=input_ids.device)
            dist.all_reduce(flag, op=dist.ReduceOp.SUM)
            if flag.item() == 0.0:
                break

        model_kwargs.pop("generation_config", None)
        model_kwargs.pop("attention_mask", None)
        model_inputs = self.prepare_inputs_for_generation(input_ids, attention_mask=attention_mask, **model_kwargs)
        _drop_cd_keys(model_inputs)
        outputs = self(**model_inputs, return_dict=True, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        if synced_gpus and this_peer_finished:
            continue
        logits = outputs.logits[:, -1, :]

        if use_cd:
            model_kwargs_cd = dict(model_kwargs)
            model_kwargs_cd.pop("past_key_values", None)
            model_kwargs_cd.pop("attention_mask", None)
            # Reset cache_position for full-sequence forward (no KV cache in CD path)
            model_kwargs_cd["cache_position"] = torch.arange(0, input_ids_cd.shape[1], dtype=torch.long, device=input_ids_cd.device)
            model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids_cd, attention_mask=attention_mask_cd, **model_kwargs_cd)
            _drop_cd_keys(model_inputs_cd)
            outputs_cd = self(
                **model_inputs_cd,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            logits_cd = outputs_cd.logits[:, -1, :]
            alpha = float(model_kwargs.get("cd_alpha", 1.0))
            beta = max(float(model_kwargs.get("cd_beta", 0.1)), 1e-12)
            cutoff = torch.log(torch.tensor(beta, device=logits.device, dtype=logits.dtype)) + logits.max(dim=-1, keepdim=True).values
            cd_logits = ((1.0 + alpha) * logits - alpha * logits_cd).masked_fill(logits < cutoff, -float("inf"))
            next_scores = logits_warper(input_ids, logits_processor(input_ids, cd_logits))
            probs = nn.functional.softmax(next_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_scores = logits_warper(input_ids, logits_processor(input_ids, logits))
            probs = nn.functional.softmax(next_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        if return_dict_in_generate and output_scores:
            scores += (next_scores,)  # type: ignore[operator]
        if eos_tensor is not None:
            if pad_token_id is None:
                raise ValueError("pad_token_id must be set when eos_token_id is set")
            next_tokens = next_tokens * unfinished + pad_token_id * (1 - unfinished)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.cat([attention_mask, attention_mask[:, -1:]], dim=-1)
        if use_cd:
            input_ids_cd = torch.cat([input_ids_cd, next_tokens[:, None]], dim=-1)
            if isinstance(attention_mask_cd, torch.Tensor):
                attention_mask_cd = torch.cat([attention_mask_cd, attention_mask_cd[:, -1:]], dim=-1)

        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder)
        # Remove attention_mask from model_kwargs so our locally tracked versions are used
        model_kwargs.pop("attention_mask", None)
        if model_kwargs.get("cache_position") is not None and int(model_kwargs["cache_position"][0]) != 0:
            model_kwargs["pixel_values"] = None
            model_kwargs["pixel_values_videos"] = None

        if eos_tensor is not None:
            unfinished = unfinished.mul(next_tokens.tile(eos_tensor.shape[0], 1).ne(eos_tensor.unsqueeze(1)).prod(dim=0))
            if unfinished.max() == 0:
                this_peer_finished = True
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True
        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()
    if not return_dict_in_generate:
        return input_ids
    if self.config.is_encoder_decoder:
        return SampleEncoderDecoderOutput(
            sequences=input_ids,
            scores=scores,
            encoder_attentions=None,
            encoder_hidden_states=None,
            decoder_attentions=decoder_attentions,
            cross_attentions=cross_attentions,
            decoder_hidden_states=decoder_hidden_states,
        )
    return SampleDecoderOnlyOutput(
        sequences=input_ids,
        scores=scores,
        attentions=decoder_attentions,
        hidden_states=decoder_hidden_states,
    )


def enable_macd_decoding(model: Any) -> None:
    if generation_utils is None:
        raise ImportError("transformers is required to enable MACD contrastive decoding")

    relax_cd_kwargs_validation()
    attach_cd_method_if_missing(model)
    # transformers >= 5.x only has _sample; transformers 4.x has both sample and _sample
    if hasattr(generation_utils.GenerationMixin, "sample"):
        generation_utils.GenerationMixin.sample = _sample_with_cd
    generation_utils.GenerationMixin._sample = _sample_with_cd
