#!/usr/bin/env python3
"""
SAE and language model utilities for self-interpretation.

Uses SAELens library for loading pretrained Sparse Autoencoders.
"""

import torch


def load_sae(release: str, sae_id: str, device=None):
    """
    Load a pretrained SAE using the SAELens library.

    Args:
        release: SAE release identifier (e.g., "llama_scope_lxr_8x", "goodfire-llama-3.1-8b-instruct")
        sae_id: SAE identifier within the release (e.g., "l19r_8x", "layer_19")
        device: Device to load the model on (optional, SAELens handles this)

    Returns:
        SAE object from SAELens library
    
    Example:
        >>> sae = load_sae("goodfire-llama-3.1-8b-instruct", "layer_19")
        >>> print(f"SAE dimensions: {sae.cfg.d_in} -> {sae.cfg.d_sae}")
    """
    from sae_lens import SAE

    print("📂 Loading SAE from SAELens:")
    print(f"  release: {release}")
    print(f"  sae_id: {sae_id}")

    sae = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )

    print(f"✓ Loaded SAE: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

    return sae


class ObservableLanguageModel:
    """
    Wrapper around language models for self-interpretation.
    
    Handles model loading with device management for both single-GPU
    and multi-GPU setups (device_map="auto").
    """

    def __init__(self, model: str, device: str = "cuda", dtype=None, cache_dir=None):
        """
        Initialize language model wrapper.
        
        Args:
            model: HuggingFace model name (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct")
            device: Device or device_map ("cuda", "cpu", "auto" for multi-GPU)
            dtype: Model dtype (e.g., torch.bfloat16)
            cache_dir: HuggingFace cache directory
        """
        self.model_name = model
        self.dtype = dtype
        self.device_map = device

        from transformers import AutoModelForCausalLM, AutoTokenizer

        cache_kwargs = {}
        if cache_dir is not None:
            cache_kwargs["cache_dir"] = cache_dir

        self._original_model = AutoModelForCausalLM.from_pretrained(
            model, device_map=self.device_map, dtype=dtype, **cache_kwargs
        )
        
        # Resolve actual PyTorch device for tensor operations
        # When device_map="auto", the model is split across GPUs
        if device == "auto":
            embed_device = next(self._original_model.get_input_embeddings().parameters()).device
            self.device = embed_device
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model, **cache_kwargs)
        self.tokenizer.clean_up_tokenization_spaces = False

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.hidden_size = self._attempt_to_infer_hidden_layer_dimensions()

    def _attempt_to_infer_hidden_layer_dimensions(self):
        """Infer hidden layer dimensions from the model config."""
        for attr in ["hidden_size", "d_model", "n_embd"]:
            if hasattr(self._original_model.config, attr):
                return getattr(self._original_model.config, attr)
        raise ValueError("Could not infer hidden size from model config")

    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """Generate text from a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._original_model.generate(
                **inputs,
                max_length=inputs.input_ids.shape[1] + max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return generated_text.strip()
