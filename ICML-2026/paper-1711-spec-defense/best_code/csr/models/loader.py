"""
Model loader utilities for HuggingFace CLIP and OpenCLIP models.

Provides unified functions to load CLIP, FARE, TeCoA, and other 
adversarially trained models from local paths or model hubs.
"""

import os
import torch
from transformers import CLIPProcessor, CLIPModel

from ..config import HF_MODEL_CONFIGS, OPENCLIP_REGISTRY, PathConfig


def load_clip_model(
    model_name: str = "CLIP-L-14",
    device: str = "cuda",
    local_files_only: bool = True,
) -> tuple:
    """
    Load a HuggingFace CLIPModel by registered name.

    Args:
        model_name: Key in HF_MODEL_CONFIGS (e.g., "CLIP-B-16", "CLIP-L-14").
        device: Target device.
        local_files_only: Whether to use only local cache.

    Returns:
        (model, processor) tuple. Model is in eval mode.
    """
    if model_name not in HF_MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(HF_MODEL_CONFIGS.keys())}"
        )

    model_path = HF_MODEL_CONFIGS[model_name]
    model = CLIPModel.from_pretrained(model_path, local_files_only=local_files_only)
    model = model.to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_path, local_files_only=local_files_only)

    for param in model.parameters():
        param.requires_grad = False

    return model, processor


def load_openclip_model(
    model_name: str = "Openai-CLIP-L/14",
    device: str = "cuda",
    paths: PathConfig = None,
) -> tuple:
    """
    Load an OpenCLIP model (Openai, FARE, TeCoA) by registered name.

    Args:
        model_name: Key in OPENCLIP_REGISTRY.
        device: Target device.
        paths: PathConfig instance for resolving weight paths.

    Returns:
        (model, preprocess, tokenizer) tuple.
    """
    import open_clip

    if paths is None:
        paths = PathConfig()

    if model_name not in OPENCLIP_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(OPENCLIP_REGISTRY.keys())}"
        )

    config = OPENCLIP_REGISTRY[model_name]
    arch = config["base_arch"]

    print(f"[ModelLoader] Loading {model_name} (arch={arch})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        arch, pretrained=paths.openclip_pretrained, device=device
    )
    model = model.float().eval()
    tokenizer = open_clip.get_tokenizer(arch)

    # Load custom visual weights if specified
    ckpt_path = config["path"]
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"[ModelLoader] Loading custom weights from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = model.visual.load_state_dict(state_dict, strict=False)
        print(f"[ModelLoader] Weight loading result: {msg}")

    return model, preprocess, tokenizer


# Expose the registries for external modification
MODEL_REGISTRY = OPENCLIP_REGISTRY
