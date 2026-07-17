"""Model utilities for loading transformer models."""

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    FineGrainedFP8Config,
    MistralCommonBackend,
)


def load_model_and_tokenizer(model_name: str, dtype: str | torch.dtype = "auto"):
    """Load the LLM model and tokenizer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_name}")
    print(f"Using device: {device}")

    if isinstance(dtype, torch.dtype):
        torch_dtype = dtype
    else:
        dtype_str = str(dtype).strip().lower() if isinstance(dtype, str) else "auto"
        if dtype_str == "auto":
            torch_dtype = torch.float16 if device == "cuda" else "auto"
        elif dtype_str in {"bf16", "bfloat16"}:
            torch_dtype = torch.bfloat16
        elif dtype_str in {"fp16", "float16"}:
            torch_dtype = torch.float16
        elif dtype_str in {"fp32", "float32"}:
            torch_dtype = torch.float32
        else:
            torch_dtype = "auto"

    print(f"Requested torch_dtype: {torch_dtype}")

    if "ministral-3" in model_name.lower():
        tokenizer = MistralCommonBackend.from_pretrained(model_name)
        model = torch.compile(
            transformers.Mistral3ForConditionalGeneration.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
                quantization_config=FineGrainedFP8Config(dequantize=True),
            )
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.to(device)
    model.eval()

    print("Model loaded successfully!")
    print(f"Model has {sum(p.numel() for p in model.parameters()):_} parameters")
    if hasattr(model, "transformer"):
        print(f"Number of layers: {len(model.transformer.h)}")
    elif hasattr(model, "model"):
        if hasattr(model.model, "layers"):
            print(f"Number of layers: {len(model.model.layers)}")
        elif hasattr(model.model, "language_model"):
            print(f"Number of layers: {len(model.model.language_model.layers)}")
    else:
        raise ValueError("Could not find transformer layers on the loaded model.")

    return model, tokenizer, device
