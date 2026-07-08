from __future__ import annotations

import inspect
from typing import Any, Dict, Tuple

import torch
import transformers

from .decoding import generate_with_saber


def init_saber_model(
    pretrained: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
    low_cpu_mem_usage: bool = True,
    **from_pretrained_kwargs: Any,
) -> transformers.PreTrainedModel:
    """
    Initialize the model for SABER decoding.

    SABER itself is a decoding algorithm; we reuse the default LLaDA model implementation
    provided by this evaluation framework.
    """
    from dllm_eval.models.cover.modeling_llada_kv_cover import LLaDAModelLM

    return LLaDAModelLM.from_pretrained(
        pretrained,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        **from_pretrained_kwargs,
    )


def init_saber_tokenizer(
    pretrained: str,
    *,
    trust_remote_code: bool = True,
    use_fast: bool = True,
    add_bos_token: bool | None = None,
    **tokenizer_kwargs: Any,
) -> transformers.PreTrainedTokenizerBase:
    kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "use_fast": use_fast,
    }
    if add_bos_token is not None:
        kwargs["add_bos_token"] = add_bos_token
    kwargs.update(tokenizer_kwargs)
    return transformers.AutoTokenizer.from_pretrained(pretrained, **kwargs)


def init_saber_engine(
    pretrained: str,
    *,
    device: str | torch.device | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
    low_cpu_mem_usage: bool = True,
    **from_pretrained_kwargs: Any,
) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizerBase]:
    tokenizer = init_saber_tokenizer(pretrained, trust_remote_code=trust_remote_code)
    model = init_saber_model(
        pretrained,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=low_cpu_mem_usage,
        **from_pretrained_kwargs,
    )
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, tokenizer


def _normalize_saber_gen_kwargs(gen_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(generate_with_saber)
    allowed = set(sig.parameters.keys()) - {"model", "prompt"}

    out: Dict[str, Any] = {}
    for k in allowed:
        if k in gen_kwargs and gen_kwargs[k] is not None:
            out[k] = gen_kwargs[k]
    return out


@torch.no_grad()
def generate_saber(
    *,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizerBase,
    prompt: torch.Tensor,
    mask_id: int = 126336,
    **gen_kwargs: Any,
) -> Tuple[torch.Tensor, int]:
    """
    Run SABER decoding. Returns (full_token_ids, actual_steps).
    """
    _ = tokenizer  # kept for a unified signature with other inference helpers
    call_kwargs = _normalize_saber_gen_kwargs({**gen_kwargs, "mask_id": mask_id})
    return generate_with_saber(model, prompt, **call_kwargs)
