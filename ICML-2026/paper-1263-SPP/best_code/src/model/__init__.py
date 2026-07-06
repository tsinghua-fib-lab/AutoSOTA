"""Model wrappers."""

from .base_model import BaseModel
from .quantization import apply_quantization

__all__ = ['BaseModel', 'apply_quantization']
