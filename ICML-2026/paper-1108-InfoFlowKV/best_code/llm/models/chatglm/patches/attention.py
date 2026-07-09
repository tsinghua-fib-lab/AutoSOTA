"""Attention patches for Qwen model."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List


class AttentionPatch(nn.Module):
    """
    Patch for capturing attention weights during forward pass.
    
    This can be applied to model layers to record attention for later analysis.
    """
    
    def __init__(self, original_attention):
        """
        Args:
            original_attention: The original attention module to wrap
        """
        super().__init__()
        self.original_attention = original_attention
        self.captured_weights = []
        self.capture_enabled = False
        
    def enable_capture(self):
        """Enable attention weight capture."""
        self.capture_enabled = True
        self.captured_weights = []
        
    def disable_capture(self):
        """Disable attention weight capture."""
        self.capture_enabled = False
        
    def get_captured_weights(self) -> List[torch.Tensor]:
        """Get all captured attention weights."""
        return self.captured_weights
    
    def clear_captured_weights(self):
        """Clear captured attention weights."""
        self.captured_weights = []
    
    def forward(self, *args, **kwargs):
        """Forward pass with optional attention capture."""
        # Call original attention
        outputs = self.original_attention(*args, **kwargs)
        
        # Capture attention weights if enabled
        if self.capture_enabled:
            # outputs can be (hidden_states,) or (hidden_states, attention_weights, ...)
            if isinstance(outputs, tuple) and len(outputs) > 1:
                attention_weights = outputs[1]
                if attention_weights is not None:
                    self.captured_weights.append(attention_weights.detach())
        
        return outputs


def patch_model_attention(model, layer_indices: Optional[List[int]] = None):
    """
    Patch model layers to capture attention weights.
    
    Args:
        model: The model to patch
        layer_indices: Indices of layers to patch (None = all layers)
        
    Returns:
        List of AttentionPatch objects
    """
    patches = []
    
    # Get layers
    if hasattr(model, 'transformer'):
        layers = model.transformer.encoder.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        raise ValueError("Cannot find model layers")
    
    # Determine which layers to patch
    if layer_indices is None:
        layer_indices = list(range(len(layers)))
    
    # Patch each layer
    for idx in layer_indices:
        layer = layers[idx]
        if hasattr(layer, 'self_attn'):
            # Create patch
            patch = AttentionPatch(layer.self_attn)
            # Replace attention module
            layer.self_attn = patch
            patches.append(patch)
    
    return patches


def unpatch_model_attention(model, patches: List[AttentionPatch]):
    """
    Remove attention patches and restore original modules.
    
    Args:
        model: The patched model
        patches: List of AttentionPatch objects to remove
    """
    # Get layers
    if hasattr(model, 'transformer'):
        layers = model.transformer.encoder.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        return
    
    # Restore original attention modules
    for layer in layers:
        if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, AttentionPatch):
            layer.self_attn = layer.self_attn.original_attention
