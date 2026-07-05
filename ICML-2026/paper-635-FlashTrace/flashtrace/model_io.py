from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_dtype(dtype: str | torch.dtype = "auto") -> str | torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    value = str(dtype).lower()
    if value == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[value]


def load_model_and_tokenizer(
    model_name_or_path: str,
    *,
    device_map: str | dict[str, Any] | None = "auto",
    dtype: str | torch.dtype = "auto",
    trust_remote_code: bool = True,
    **model_kwargs: Any,
):
    """Load a Hugging Face causal LM and matching tokenizer."""

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    # FlashTrace needs attention weights; SDPA returns None for them.
    model_kwargs.setdefault("attn_implementation", "eager")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=_resolve_dtype(dtype),
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        **model_kwargs,
    )
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
