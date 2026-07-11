import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def get_llm_layers_module(model: nn.Module):
    """
    Finds the transformer layers within the model architecture.
    Supports Qwen2.5-VL, LLaVA-1.5/NeXT, and other standard LLM backbones.
    """
    # 1. Qwen2.5-VL / Qwen2-VL / standard Llama-like
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    
    # 2. LLaVA-NeXT / LLaVA-1.6 (HuggingFace implementation)
    if hasattr(model, "language_model") and hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
        return model.language_model.model.layers

    # 3. LLaVA-1.5 (Original LLaVA GitHub implementation)
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers

    # 4. Standard Qwen or generic architectures
    if hasattr(model, "layers"):
        return model.layers
        
    # Fallback for complex nested structures
    try:
        return model.model.language_model.model.layers
    except AttributeError:
        pass

    raise ValueError(f"Unable to locate LLM layers. Unknown architecture: {type(model)}")


class MADRSteeringHook_visual_only:
    """
    Mechanism-Aware Dynamic Rectification Hook (Visual Only).
    
    Core Functions:
    1. Real-time Hallucination Risk Score calculation.
    2. Dynamic Gating Intervention:
       - Low Risk -> No intervention.
       - High Risk -> Activates Visual Lens: Injects visual information to recover recall.
    3. Type and Device Safety: Fully BFloat16/Float16 compatible.
    """
    
    def __init__(self, 
                 vectors: Dict[str, torch.Tensor], 
                 layer_idx: int, 
                 alpha_vis: float, 
                 tau_low: float, 
                 risk_gamma: float):
        
        self.layer_idx = layer_idx
        
        # Preload Visual Vector (Stored on CPU, moved to GPU during forward to save VRAM)
        self.v_vis = vectors.get("visual")
        
        # Strength Coefficient
        self.alpha_vis = alpha_vis   # Visual Lens Strength
        
        # Gating Threshold
        self.tau_low = tau_low       # Low risk line (triggers Visual Lens)
        self.gamma = risk_gamma      # Risk calculation balance factor
        
        self.handle = None

    def _compute_risk(self, h: torch.Tensor) -> torch.Tensor:
        """
        Computes the hallucination risk score.
        Formula: Risk = - (Gamma * Sim(h, V_vis))
        
        Logic:
        - Higher Risk indicates lower visual confidence.
        """
        if self.v_vis is None:
            return torch.zeros(h.shape[:-1], device=h.device, dtype=h.dtype)
            
        target_dtype = h.dtype
        target_device = h.device
        
        # Normalize vectors for cosine similarity
        v_v = F.normalize(self.v_vis.to(device=target_device, dtype=target_dtype), dim=-1)
        h_norm = F.normalize(h, dim=-1)
        
        # Calculate visual confidence projection
        visual_confidence = torch.matmul(h_norm, v_v)
        
        # Synthesize Risk
        risk = - (self.gamma * visual_confidence)
        return risk

    def hook_fn(self, module, input, output):
        """
        PyTorch Forward Hook function.
        """
        # 1. Parse output (HuggingFace outputs are usually tuples)
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
            
        # 2. Calculate risk score (Batch, Seq, 1) for broadcasting
        risk = self._compute_risk(h).unsqueeze(-1)
        
        # 3. Dynamic Gating: Apply Visual Lens if risk exceeds tau_low
        lens_mask = (risk > self.tau_low).float()
        lambda_lens = lens_mask * self.alpha_vis
        
        # 4. Calculate intervention delta
        delta = torch.zeros_like(h)
        target_dtype = h.dtype
        target_device = h.device
        
        if self.v_vis is not None:
            delta += lambda_lens * self.v_vis.to(device=target_device, dtype=target_dtype)
            
        # 5. Apply intervention
        h_new = h + delta
        
        # 6. Reconstruct output format
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    def attach(self, module: nn.Module):
        """Mount hook to the specified module."""
        self.handle = module.register_forward_hook(self.hook_fn)

    def detach(self):
        """Unmount hook."""
        if self.handle:
            self.handle.remove()
            self.handle = None
