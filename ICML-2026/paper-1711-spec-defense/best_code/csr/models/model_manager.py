"""
Unified Model Manager for loading, caching, and fusing CLIP models.

Provides a single interface to manage multiple models with lazy loading
and built-in fusion / merging capabilities.
"""

import os
import copy
import torch
import torch.nn as nn
from typing import List, Union, Optional
from PIL import Image

from ..config import OPENCLIP_REGISTRY, PathConfig


class UnifiedModelManager:
    """
    Central manager for OpenCLIP-based models.
    
    Features:
      - Lazy loading with automatic caching
      - Linear weight interpolation fusion
      - Encode image/text with gradient control
    """

    def __init__(self, device: str = "cuda", paths: PathConfig = None):
        import open_clip

        self.device = device
        self.paths = paths or PathConfig()
        self.models = {}
        self.tokenizers = {}
        self.preprocesses = {}
        self._open_clip = open_clip  # store reference
        print(f"[ModelManager] Initialized (device={device})")

    def _load_base_model(self, arch_name: str):
        """Load base OpenCLIP architecture with pretrained weights."""
        print(f"[ModelManager] Loading architecture: {arch_name}...")
        model, _, preprocess = self._open_clip.create_model_and_transforms(
            arch_name, pretrained=self.paths.openclip_pretrained, device=self.device
        )
        model = model.float().eval()
        tokenizer = self._open_clip.get_tokenizer(arch_name)
        return model, preprocess, tokenizer

    def get_model(self, model_name: str):
        """Load and cache a model by registry name."""
        if model_name in self.models:
            return self.models[model_name]

        if model_name not in OPENCLIP_REGISTRY:
            raise ValueError(f"Model '{model_name}' not in registry. Available: {list(OPENCLIP_REGISTRY.keys())}")

        config = OPENCLIP_REGISTRY[model_name]
        model, preprocess, tokenizer = self._load_base_model(config["base_arch"])

        ckpt_path = config["path"]
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"[ModelManager] Loading weights for {model_name} from {ckpt_path}...")
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            state_dict = checkpoint.get("state_dict", checkpoint)
            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            msg = model.visual.load_state_dict(state_dict, strict=False)
            print(f"[ModelManager] Weight loading: {msg}")

        self.models[model_name] = model
        self.preprocesses[model_name] = preprocess
        self.tokenizers[model_name] = tokenizer
        return model

    def get_preprocess(self, model_name: str):
        """Get the preprocessing transform for a model."""
        if model_name not in self.preprocesses:
            self.get_model(model_name)
        return self.preprocesses[model_name]

    def get_tokenizer(self, model_name: str):
        """Get the tokenizer for a model."""
        if model_name not in self.tokenizers:
            self.get_model(model_name)
        return self.tokenizers[model_name]

    def get_fusion_model(
        self, model_name_a: str, model_name_b: str, alpha: float = 0.7
    ):
        """
        Create a linearly interpolated fusion model.
        
        Result = A + alpha * (B - A) = (1 - alpha) * A + alpha * B
        """
        fusion_id = f"Fusion_{model_name_a}_{model_name_b}_alpha{alpha}"
        if fusion_id in self.models:
            return self.models[fusion_id]

        print(f"[ModelManager] Creating fusion: {model_name_a} + {alpha} * ({model_name_b} - {model_name_a})")
        model_a = self.get_model(model_name_a)
        model_b = self.get_model(model_name_b)
        fusion_model = copy.deepcopy(model_a)

        with torch.no_grad():
            for param_a, param_b in zip(fusion_model.parameters(), model_b.parameters()):
                param_a.data.lerp_(param_b.data, alpha)

        self.models[fusion_id] = fusion_model
        self.preprocesses[fusion_id] = self.preprocesses[model_name_a]
        self.tokenizers[fusion_id] = self.tokenizers[model_name_a]
        return fusion_model

    def _resolve_model(self, model_or_name):
        """Resolve a model name or nn.Module to (model, config_key)."""
        if isinstance(model_or_name, str):
            if model_or_name not in self.models:
                if model_or_name in OPENCLIP_REGISTRY:
                    return self.get_model(model_or_name), model_or_name
                raise ValueError(f"Unknown model: {model_or_name}")
            return self.models[model_or_name], model_or_name
        elif isinstance(model_or_name, nn.Module):
            return model_or_name, "Openai-CLIP-L/14"
        raise TypeError("Expected model name (str) or nn.Module")

    def encode_image(self, image, model_or_name, with_grad: bool = False):
        """Encode image(s) using the specified model."""
        model, config_key = self._resolve_model(model_or_name)

        if config_key in self.preprocesses:
            preprocess = self.preprocesses[config_key]
        else:
            fallback = "Openai-CLIP-L/14"
            if fallback not in self.preprocesses:
                self.get_model(fallback)
            preprocess = self.preprocesses[fallback]

        if isinstance(image, Image.Image):
            image_tensor = preprocess(image).unsqueeze(0).to(self.device)
        else:
            image_tensor = image.to(self.device)

        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            features = model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def encode_text(self, text_list: List[str], model_or_name, with_grad: bool = False):
        """Encode text using the specified model."""
        model, config_key = self._resolve_model(model_or_name)

        if config_key in self.tokenizers:
            tokenizer = self.tokenizers[config_key]
        else:
            fallback = "Openai-CLIP-L/14"
            if fallback not in self.tokenizers:
                self.get_model(fallback)
            tokenizer = self.tokenizers[fallback]

        text_tokens = tokenizer(text_list).to(self.device)
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            features = model.encode_text(text_tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features
