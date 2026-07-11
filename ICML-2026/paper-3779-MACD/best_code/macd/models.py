from __future__ import annotations

from typing import Any

import torch


def load_video_llm(model_path: str, torch_dtype: str = "bfloat16") -> tuple[Any, Any]:
    from transformers import AutoProcessor

    dtype = torch.bfloat16 if torch_dtype == "bfloat16" else "auto"
    model_cls = None
    for name in (
        "AutoModelForImageTextToText",
        "Qwen3VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    ):
        try:
            module = __import__("transformers", fromlist=[name])
            model_cls = getattr(module, name)
            break
        except Exception:
            continue
    if model_cls is None:
        raise ImportError("Could not find a compatible Hugging Face AutoModel class for Video-LLMs.")

    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


def get_model_device(model: Any) -> torch.device:
    return getattr(model, "device", None) or next(model.parameters()).device
