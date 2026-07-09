"""Patches for Qwen3-VL model components."""

from .visual import VisualPatch
from .attention import AttentionPatch
from .text import TextPatch, TextModelOutputs

__all__ = ["VisualPatch", "AttentionPatch", "TextPatch", "TextModelOutputs"]
