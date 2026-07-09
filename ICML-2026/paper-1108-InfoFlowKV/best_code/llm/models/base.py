"""Base classes for model patches and utilities."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import torch


class BasePatch(ABC):
    """Base class for all model patches."""

    @abstractmethod
    def apply(self):
        """Apply the patch to the model."""
        pass

    @abstractmethod
    def remove(self):
        """Remove the patch and restore original behavior."""
        pass

    def clear(self):
        """Clear any captured data."""
        pass

    def __enter__(self):
        self.apply()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
        return False


class AttentionCapture(BasePatch):
    """Base class for capturing attention weights."""

    def __init__(self, model, layers_to_capture: Optional[List[int]] = None):
        self.model = model
        self.layers_to_capture = layers_to_capture
        self.captured_attentions = {}
        self._original_forwards = {}

    def clear(self):
        """Clear captured attentions."""
        self.captured_attentions = {}


class ModelConfig:
    """Base model configuration."""

    def __init__(self, model):
        self.model = model
        self.config = model.config
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = getattr(self.config, 'num_key_value_heads', self.num_heads)
        self.head_dim = getattr(self.config, 'head_dim', self.config.hidden_size // self.num_heads)
        self.kv_head_dim = self.head_dim
        self.hidden_size = self.config.hidden_size
        self.rope_theta = getattr(self.config, 'rope_theta', 10000.0)
        self.max_position = getattr(self.config, 'max_position_embeddings', 32768)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dict."""
        return {
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "kv_head_dim": self.kv_head_dim,
            "hidden_size": self.hidden_size,
            "rope_theta": self.rope_theta,
            "max_position": self.max_position,
        }
