"""
Loads a Hugging Face large language model (LLM) using the transformers library based on the specified model name.
"""

import torch
from transformers import AutoModelForCausalLM


def load_hf_llm(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )

    return model


def get_model_size_gb(model):
    total_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = model.dtype.itemsize
    print(f"Total parameters: {total_params}, Bytes per parameter: {bytes_per_param}")
    size_gb = total_params * bytes_per_param / (1024**3)
    return size_gb


if __name__ == "__main__":
    model = load_hf_llm("meta-llama/Llama-3.1-8B")
    size_gb = get_model_size_gb(model)
    print(f"Model size (in gigabytes): {size_gb:.2f} GB")
