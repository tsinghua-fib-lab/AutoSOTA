#!/usr/bin/env python3
"""Lightweight inference utilities for trained SelfIE adapters."""

import json
from typing import Optional, Dict, Any
import torch

from selfie_adapters.projection import create_projection_module


class SelfIEAdapter:
    """
    Lightweight loader for inference with trained SelfIE adapters.
    
    This class allows you to load a trained projection module from a checkpoint
    and use it for inference without loading the full training infrastructure
    (language model, optimizer, etc.).
    
    Supports both formats:
    - .safetensors (recommended for HuggingFace, weights + metadata in header)
    - .pt (legacy PyTorch checkpoint format)
    
    Example:
        >>> adapter = SelfIEAdapter("adapter.safetensors")
        >>> soft_tokens = adapter.transform(sae_vectors)
        >>> print(adapter.get_metadata())
    """
    
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        """
        Load a trained SelfIE adapter from a checkpoint file.
        
        Args:
            checkpoint_path: Path to adapter file (.safetensors or .pt)
            device: Device to load projection on (e.g., "cpu", "cuda", "cuda:0").
                   If None, uses "cuda" if available, otherwise "cpu".
        
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint format is invalid or unsupported
        """
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Dispatch based on file extension
        if checkpoint_path.endswith(".safetensors"):
            projection_state, proj_config, model_dim, extra_meta = (
                self._load_safetensors(checkpoint_path)
            )
        else:
            projection_state, proj_config, model_dim, extra_meta = (
                self._load_pt(checkpoint_path)
            )
        
        # Store metadata
        self.model_dim = model_dim
        self.config = proj_config
        self.checkpoint_path = checkpoint_path
        self.checkpoint_format_version = extra_meta.get("checkpoint_format_version", 0)
        self.global_step = extra_meta.get("global_step", None)
        self.best_val_loss = extra_meta.get("best_val_loss", None)
        
        # Recreate projection module with exact training configuration
        print(f"Loading {proj_config['type']} projection (dim={model_dim}) from {checkpoint_path}")
        self.projection = create_projection_module(
            projection_type=proj_config["type"],
            dim=model_dim,
            normalize_input=proj_config["normalize_input"],
            device=self.device,
            init_scale=proj_config.get("init_scale", 30.0),
            low_rank_rank=proj_config.get("low_rank_rank"),
            low_rank_init_factor=proj_config.get("low_rank_init_factor"),
        )
        
        # Load trained weights
        self.projection.load_state_dict(projection_state)
        
        # Set to evaluation mode
        self.projection.eval()
        
        # Verify parameter count if available
        expected_params = extra_meta.get("projection_num_params")
        if expected_params is not None:
            actual = self.projection.num_parameters()
            if expected_params != actual:
                print(f"WARNING: Parameter count mismatch. Expected {expected_params}, got {actual}")
        
        print(f"✓ Loaded adapter with {self.projection.num_parameters():,} parameters")
    
    def _load_safetensors(self, path: str):
        """
        Load adapter from safetensors format.
        
        Returns:
            (projection_state, proj_config, model_dim, extra_meta)
        """
        from safetensors import safe_open
        from safetensors.torch import load_file as safetensors_load_file
        
        # Load tensors (projection weights)
        projection_state = safetensors_load_file(path, device=str(self.device))
        
        # Load metadata from header
        with safe_open(path, framework="pt") as f:
            meta = f.metadata()
        
        if meta is None:
            raise ValueError(
                f"Safetensors file has no metadata header: {path}. "
                f"Expected selfie_adapter_v1 format."
            )
        
        # Parse config from metadata
        if "config_json" in meta:
            config_dict = json.loads(meta["config_json"])
            proj_config = config_dict["projection"]
        else:
            # Fallback: reconstruct proj_config from flat metadata keys
            proj_config = {
                "type": meta.get("projection_type", ""),
                "normalize_input": meta.get("normalize_input", "true").lower() == "true",
                "init_scale": float(meta["init_scale"]) if "init_scale" in meta else 30.0,
                "low_rank_rank": int(meta["low_rank_rank"]) if meta.get("low_rank_rank") else None,
                "low_rank_init_factor": None,
            }
        
        # Get model dimension
        if "model_dim" in meta:
            model_dim = int(meta["model_dim"])
        else:
            model_dim = self._infer_dim_from_state(projection_state, proj_config)
        
        # Extra metadata
        extra_meta = {}
        if "global_step" in meta:
            extra_meta["global_step"] = int(meta["global_step"])
        if "best_val_loss" in meta:
            extra_meta["best_val_loss"] = float(meta["best_val_loss"])
        
        return projection_state, proj_config, model_dim, extra_meta
    
    def _load_pt(self, path: str):
        """
        Load adapter from PyTorch .pt checkpoint format.
        
        Returns:
            (projection_state, proj_config, model_dim, extra_meta)
        """
        # Note: weights_only=False is required because checkpoint contains config dict
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Validate checkpoint format
        if "projection_state" not in checkpoint:
            raise ValueError(
                f"Invalid checkpoint format: missing 'projection_state'. "
                f"Available keys: {list(checkpoint.keys())}"
            )
        
        if "config" not in checkpoint:
            raise ValueError(
                f"Invalid checkpoint format: missing 'config'. "
                f"This checkpoint may be too old or corrupted."
            )
        
        # Extract config
        config_dict = checkpoint["config"]
        if "projection" not in config_dict:
            raise ValueError("Invalid checkpoint: config missing 'projection' section")
        
        proj_config = config_dict["projection"]
        projection_state = checkpoint["projection_state"]
        
        # Get model dimension
        if "model_dim" in checkpoint:
            model_dim = checkpoint["model_dim"]
        else:
            model_dim = self._infer_dim_from_state(projection_state, proj_config)
        
        # Extra metadata
        extra_meta = {
            "checkpoint_format_version": checkpoint.get("checkpoint_format_version", 0),
            "global_step": checkpoint.get("global_step"),
            "best_val_loss": checkpoint.get("best_val_loss"),
            "projection_num_params": checkpoint.get("projection_num_params"),
        }
        
        return projection_state, proj_config, model_dim, extra_meta
    
    def _infer_dim_from_state(self, state_dict: Dict[str, torch.Tensor], config: Dict[str, Any]) -> int:
        """Infer model dimension from state dict (fallback for old checkpoints)."""
        ptype = config["type"]
        
        # Try common parameters
        if "bias" in state_dict:
            return state_dict["bias"].shape[0]
        
        if "scale_direction" in state_dict:
            return state_dict["scale_direction"].shape[0]
        
        # Architecture-specific inference
        if ptype == "full_rank":
            if "weight" in state_dict:
                return state_dict["weight"].shape[0]
        
        elif ptype == "scalar_affine_plus_low_rank":
            if "U" in state_dict:
                return state_dict["U"].shape[0]
        
        elif ptype == "scale_only":
            raise ValueError(
                "Cannot infer model_dim from scale_only projection without additional info."
            )
        
        raise ValueError(
            f"Cannot infer model_dim from {ptype} checkpoint. "
            f"Available state keys: {list(state_dict.keys())}."
        )
    
    @torch.no_grad()
    def transform(self, vectors: torch.Tensor, normalize_input: Optional[bool] = None) -> torch.Tensor:
        """
        Apply trained projection to input vectors.
        
        Args:
            vectors: Input vectors of shape (batch_size, model_dim) or (model_dim,)
            normalize_input: Override normalization behavior:
                - None (default): Use training configuration
                - True: Force L2 normalization of inputs
                - False: Skip normalization
        
        Returns:
            Soft token embeddings of shape (batch_size, model_dim) or (model_dim,)
        """
        # Handle single vector case
        single_vector = vectors.ndim == 1
        if single_vector:
            vectors = vectors.unsqueeze(0)
        
        # Validate shape
        if vectors.shape[-1] != self.model_dim:
            raise ValueError(
                f"Input vectors have dimension {vectors.shape[-1]}, "
                f"but adapter expects {self.model_dim}"
            )
        
        # Move to device and transform
        vectors = vectors.to(self.device)
        vectors_f32 = vectors.float()
        
        # Handle normalization override
        if normalize_input is None:
            soft_tokens = self.projection(vectors_f32)
        else:
            original_normalize = self.projection.normalize_input
            self.projection.normalize_input = normalize_input
            try:
                soft_tokens = self.projection(vectors_f32)
            finally:
                self.projection.normalize_input = original_normalize
        
        # Convert back to input dtype
        soft_tokens = soft_tokens.to(vectors.dtype)
        
        if single_vector:
            soft_tokens = soft_tokens.squeeze(0)
        
        return soft_tokens
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the loaded adapter.
        
        Returns:
            Dictionary with architecture and training information
        """
        metadata = {
            "projection_type": self.config["type"],
            "model_dim": self.model_dim,
            "normalize_input": self.config["normalize_input"],
            "num_parameters": self.projection.num_parameters(),
            "device": str(self.device),
            "checkpoint_format_version": self.checkpoint_format_version,
        }
        
        if self.global_step is not None:
            metadata["global_step"] = self.global_step
        
        if self.best_val_loss is not None:
            metadata["best_val_loss"] = self.best_val_loss
        
        for key in ["init_scale", "low_rank_rank", "low_rank_init_factor"]:
            if key in self.config and self.config[key] is not None:
                metadata[key] = self.config[key]
        
        return metadata
    
    def get_projection_stats(self) -> Dict[str, float]:
        """Get statistics about the projection parameters (scale, bias norm, etc.)."""
        stats = {}
        
        if hasattr(self.projection, "get_scale"):
            stats["scale"] = self.projection.get_scale()
        
        if hasattr(self.projection, "get_bias_norm"):
            stats["bias_norm"] = self.projection.get_bias_norm()
        
        if hasattr(self.projection, "get_weight_norm"):
            stats["weight_norm"] = self.projection.get_weight_norm()
        
        if hasattr(self.projection, "get_low_rank_norm"):
            stats["low_rank_norm"] = self.projection.get_low_rank_norm()
        
        return stats
    
    def __repr__(self) -> str:
        return (
            f"SelfIEAdapter("
            f"type={self.config['type']}, "
            f"dim={self.model_dim}, "
            f"params={self.projection.num_parameters():,}, "
            f"device={self.device})"
        )


def load_adapter(checkpoint_path: str, device: Optional[str] = None) -> SelfIEAdapter:
    """
    Convenience function to load a SelfIE adapter.
    
    Args:
        checkpoint_path: Path to adapter file (.safetensors or .pt)
        device: Device to load on (default: auto-detect)
    
    Returns:
        Loaded SelfIEAdapter instance
    
    Example:
        >>> adapter = load_adapter("goodfire-sae-scalar-affine.safetensors")
        >>> soft_tokens = adapter.transform(vectors)
    """
    return SelfIEAdapter(checkpoint_path, device)
