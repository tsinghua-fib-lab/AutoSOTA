"""Patches for Qwen model components."""

from .attention import AttentionPatch, patch_model_attention, unpatch_model_attention

__all__ = ["AttentionPatch", "patch_model_attention", "unpatch_model_attention"]
