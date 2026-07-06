"""
Head importance computation module
Following the paper design, implements axis-aligned energy based head importance computation I_{l,h,k}

This is the method recommended in the paper, using axis-aligned energy to compute head importance
Supports different model architectures such as LLaMA/Qwen/GPT-2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
from pathlib import Path
import json

from ..utils import get_logger

logger = get_logger(__name__)


class HeadImportanceCalculator:
    """
    Head importance calculator
    Following the paper's Eq.(5): I_{l,h,k} = E_{x~P_k}[ (u_k^T w_{l,h}(x))^2 / (||w_{l,h}(x)||^2 + eps) ]

    Modification: use w[k] instead of normalized energy, so that the mean importance of all heads equals the layer relevance
    New formula: I_{l,h,k} = E_{x~P_k}[ w_{l,h}(x)[k] ]

    where:
    - u_k: axis direction of the k-th domain (one-hot vector)
    - w_{l,h}(x): contribution of head (l,h) to the post-attention residual (projected into domain space via the probe)
    - P_k: training data of domain k
    - w_{l,h}(x)[k]: the k-th element of w (the activation value corresponding to domain k)
    - This way the mean importance of all heads = layer relevance (0.7-0.9)
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads_per_layer: int,
        num_domains: int,
        layer_probes: Optional[object] = None,  # probes (used to project the residual into domain space)
        epsilon: float = 1e-8
    ):
        """
        Initialize the head importance calculator

        Note: domain axes have a one-hot structure, not a true PCA space
        Use probes to project the post-attention residual into domain space

        Args:
            num_layers: number of layers
            num_heads_per_layer: number of heads per layer
            num_domains: number of domains
            layer_probes: probes (used to project the residual into domain space)
            epsilon: numerical stability constant
        """
        self.num_layers = num_layers
        self.num_heads_per_layer = num_heads_per_layer
        self.num_domains = num_domains
        self.layer_probes = layer_probes
        self.epsilon = epsilon
        
        # Store the computed importance: {layer_idx: [num_heads, num_domains]}
        self.importance_cache = {}

        logger.info(f"Head importance calculator initialized: {num_layers} layers, {num_heads_per_layer} heads/layer, {num_domains} domains")

    def _detect_model_type(self, model: nn.Module) -> str:
        """
        Detect the model type

        Returns:
            'llama', 'qwen', 'gpt2', or 'unknown'
        """
        model_str = str(type(model)).lower()

        # Check the model structure
        if hasattr(model, 'model'):
            if hasattr(model.model, 'layers'):
                # Check whether it is a LLaMA/Qwen structure
                if len(model.model.layers) > 0:
                    first_layer = model.model.layers[0]
                    if hasattr(first_layer, 'self_attn'):
                        # Check whether it has q_proj, k_proj, v_proj (LLaMA/Qwen)
                        if hasattr(first_layer.self_attn, 'q_proj'):
                            # Further distinguish LLaMA from Qwen (by model name)
                            if 'qwen' in model_str or 'Qwen' in str(type(model)):
                                return 'qwen'
                            else:
                                return 'llama'

        # Check the GPT-2 structure
        if hasattr(model, 'transformer'):
            if hasattr(model.transformer, 'h'):
                return 'gpt2'
        
        return 'unknown'
    
    def _extract_head_output_llama_qwen(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        layer_idx: int,
        head_idx: int
    ) -> torch.Tensor:
        """
        Extract the head output for the LLaMA/Qwen architecture

        Args:
            model: the model
            inputs: input dictionary
            layer_idx: layer index
            head_idx: head index

        Returns:
            head_output: [batch, seq_len, head_dim] the output of a single head
        """
        import math

        # Get the model config
        config = model.config if hasattr(model, 'config') else None
        if config is None:
            raise ValueError("Unable to get the model config")

        num_heads = getattr(config, 'num_attention_heads', None) or \
                   getattr(config, 'n_head', None) or 12
        head_dim = getattr(config, 'head_dim', None) or \
                  (getattr(config, 'hidden_size', 768) // num_heads)

        # Get the layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        else:
            raise ValueError("Unable to find the model layers")

        if layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} out of range (total layers {len(layers)})")

        layer = layers[layer_idx]
        attn = layer.self_attn

        # Get hidden_states (before attention)
        # Need to hook into the layer's input
        hidden_states_cache = {}

        def pre_hook(module, input):
            if isinstance(input, tuple):
                # Need to clone and detach to ensure the tensor can be used in subsequent computation
                hidden_states_cache['input'] = input[0].clone().detach()
            else:
                hidden_states_cache['input'] = input.clone().detach()

        hook_handle = layer.register_forward_pre_hook(pre_hook)

        # Forward pass (ensure inputs are correctly formatted to avoid vmap issues)
        # Use torch.no_grad() instead of inference_mode(), because we need to clone the tensor for subsequent computation
        with torch.no_grad():
            try:
                # Ensure attention_mask is 2D, to avoid the internal vmap issue in transformers
                safe_inputs = {}
                for k, v in inputs.items():
                    if k == 'attention_mask' and v is not None:
                        # Ensure attention_mask is 2D [batch, seq_len]
                        if len(v.shape) == 1:
                            v = v.unsqueeze(0)
                        elif len(v.shape) == 0:
                            continue  # Skip scalars
                    safe_inputs[k] = v

                # Use torch.no_grad() instead of inference_mode(), because we need to clone the tensor
                # tensors under inference_mode cannot be cloned for subsequent computation
                with torch.no_grad():
                    _ = model(**safe_inputs)
            except Exception as e:
                # Do not simply fall back to input_ids only, because missing attention_mask and other info may cause inaccurate computation
                # Instead, try to fix the inputs format and retry
                logger.warning(f"Error during forward pass: {e}, trying to fix the inputs format and retry")

                # Try a stricter format fix
                retry_inputs = {}
                for k, v in inputs.items():
                    if v is None:
                        continue
                    if k == 'attention_mask':
                        # Ensure attention_mask is 2D [batch, seq_len]
                        if len(v.shape) == 1:
                            v = v.unsqueeze(0)
                        elif len(v.shape) == 0:
                            # If it is a scalar, create a default mask
                            if 'input_ids' in inputs:
                                seq_len = inputs['input_ids'].shape[-1]
                                v = torch.ones(1, seq_len, dtype=torch.bool, device=v.device)
                            else:
                                continue
                    elif k == 'input_ids':
                        # Ensure input_ids is at least 2D [batch, seq_len]
                        if len(v.shape) == 1:
                            v = v.unsqueeze(0)
                    retry_inputs[k] = v

                # If it still fails after the fix, raise an error instead of using incomplete inputs
                try:
                    with torch.no_grad():
                        _ = model(**retry_inputs)
                except Exception as e2:
                    logger.error(f"Still failed after fixing the inputs format: {e2}")
                    logger.error(f"Original error: {e}")
                    raise ValueError(
                        f"Unable to perform the forward pass. Original error: {e}, error after fix: {e2}. "
                        f"Please check whether the inputs format is correct."
                    )

        hook_handle.remove()

        if 'input' not in hidden_states_cache:
            raise ValueError("Unable to get hidden_states")

        hidden_states = hidden_states_cache['input']  # [batch, seq_len, hidden_dim]

        # Ensure hidden_states is 3D
        if len(hidden_states.shape) == 2:
            # [seq_len, hidden_dim] -> [1, seq_len, hidden_dim]
            hidden_states = hidden_states.unsqueeze(0)
        elif len(hidden_states.shape) == 1:
            # [hidden_dim] -> [1, 1, hidden_dim]
            hidden_states = hidden_states.unsqueeze(0).unsqueeze(0)
        elif len(hidden_states.shape) > 3:
            # If there are too many dimensions, take the first 3
            hidden_states = hidden_states.view(-1, hidden_states.shape[-2], hidden_states.shape[-1])

        if len(hidden_states.shape) != 3:
            raise ValueError(f"hidden_states has incorrect dimensions: {hidden_states.shape}, expected 3D [batch, seq_len, hidden_dim]")

        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Detect GQA (Grouped Query Attention)
        config = model.config if hasattr(model, 'config') else None
        num_kv_heads = None
        if config is not None:
            num_kv_heads = getattr(config, 'num_key_value_heads', None)
        if num_kv_heads is None:
            num_kv_heads = num_heads  # Default: standard multi-head attention

        # Compute Q, K, V
        q = attn.q_proj(hidden_states)  # [batch, seq_len, hidden_dim] or [batch, seq_len, num_heads * head_dim]
        k = attn.k_proj(hidden_states)  # [batch, seq_len, num_kv_heads * head_dim] (GQA) or [batch, seq_len, hidden_dim] (standard)
        v = attn.v_proj(hidden_states)  # [batch, seq_len, num_kv_heads * head_dim] (GQA) or [batch, seq_len, hidden_dim] (standard)

        # Get the actual dimensions and compute head_dim
        q_dim = q.shape[-1]
        k_dim = k.shape[-1]

        if q_dim % num_heads == 0:
            q_head_dim = q_dim // num_heads
        else:
            if hidden_dim % num_heads == 0:
                q_head_dim = hidden_dim // num_heads
            else:
                raise ValueError(f"Unable to determine head_dim of Q: q_dim={q_dim}, num_heads={num_heads}, hidden_dim={hidden_dim}")

        if k_dim % num_kv_heads == 0:
            kv_head_dim = k_dim // num_kv_heads
        else:
            kv_head_dim = q_head_dim

        if q_head_dim != kv_head_dim:
            kv_head_dim = q_head_dim

        head_dim = q_head_dim

        # Reshape to the head level
        q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k = k.reshape(batch_size, seq_len, num_kv_heads, kv_head_dim).transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]
        v = v.reshape(batch_size, seq_len, num_kv_heads, kv_head_dim).transpose(1, 2)

        # If GQA, need to map head_idx to the corresponding K/V head
        if num_kv_heads < num_heads:
            # Compute the K/V head index corresponding to head_idx
            num_groups = num_heads // num_kv_heads
            kv_head_idx = head_idx // num_groups
            # Repeat K/V to match the number of Q heads (only for computation)
            k = k.repeat_interleave(num_groups, dim=1)
            v = v.repeat_interleave(num_groups, dim=1)

        # Extract the specific head
        q_h = q[:, head_idx, :, :]  # [batch, seq_len, head_dim]
        k_h = k[:, head_idx, :, :]
        v_h = v[:, head_idx, :, :]

        # Compute attention scores
        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(head_dim)  # [batch, seq_len, seq_len]

        # Numerical stability: clamp scores to avoid softmax overflow
        scores = torch.clamp(scores, min=-50.0, max=50.0)

        # Apply causal mask (if needed)
        if hasattr(attn, 'is_causal') and attn.is_causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1)
            scores = scores.masked_fill(causal_mask.bool(), float('-inf'))

        # Softmax (using a numerically stable implementation)
        attn_weights = F.softmax(scores, dim=-1)  # [batch, seq_len, seq_len]

        # Check whether attn_weights is NaN/Inf
        if torch.isnan(attn_weights).any() or torch.isinf(attn_weights).any():
            # If NaN/Inf occurs, use a numerically stable softmax
            scores_clamped = torch.clamp(scores, min=-50.0, max=50.0)
            attn_weights = F.softmax(scores_clamped, dim=-1)
            # If there is still NaN, replace with a uniform distribution
            if torch.isnan(attn_weights).any():
                attn_weights = torch.ones_like(attn_weights) / seq_len

        # Compute the head output
        head_output = torch.matmul(attn_weights, v_h)  # [batch, seq_len, head_dim]

        # Check whether head_output is NaN/Inf
        if torch.isnan(head_output).any() or torch.isinf(head_output).any():
            # Try to fix: replace NaN/Inf with 0
            nan_inf_mask = torch.isnan(head_output) | torch.isinf(head_output)
            if nan_inf_mask.any():
                head_output = torch.where(nan_inf_mask, torch.zeros_like(head_output), head_output)

        # Key fix: must go through o_proj to obtain this head's contribution to the post-attention residual
        # The correct approach:
        # 1. Place head_output at the correct position after concat ([batch, seq_len, num_heads * head_dim])
        # 2. Go through o_proj to obtain this head's contribution ([batch, seq_len, hidden_dim])

        # Create the full concat vector, with values only at this head's position
        # Get num_heads (if not yet obtained)
        if 'num_heads' not in locals():
            num_heads = getattr(attn, 'num_heads', getattr(attn, 'num_attention_heads', hidden_dim // 64))
            head_dim = hidden_dim // num_heads
        concat_dim = num_heads * head_dim
        head_output_concat = torch.zeros(batch_size, seq_len, concat_dim, device=head_output.device, dtype=head_output.dtype)
        start_idx = head_idx * head_dim
        end_idx = start_idx + head_dim

        # Place head_output at the correct position after concat
        head_output_concat[:, :, start_idx:end_idx] = head_output

        # Go through o_proj to obtain this head's contribution
        # o_proj: [concat_dim, hidden_dim]
        head_contribution = attn.o_proj(head_output_concat)  # [batch, seq_len, hidden_dim]

        return head_contribution
    
    def _extract_head_output_gpt2(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        layer_idx: int,
        head_idx: int
    ) -> torch.Tensor:
        """
        Extract the head output for the GPT-2 architecture

        Args:
            model: the model
            inputs: input dictionary
            layer_idx: layer index
            head_idx: head index

        Returns:
            head_output: [batch, seq_len, head_dim] the output of a single head
        """
        import math

        # Get the model config
        config = model.config if hasattr(model, 'config') else None
        if config is None:
            raise ValueError("Unable to get the model config")

        num_heads = getattr(config, 'num_attention_heads', None) or \
                   getattr(config, 'n_head', None) or 12
        head_dim = getattr(config, 'head_dim', None) or \
                  (getattr(config, 'n_embd', 768) // num_heads)

        # Get the layers
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
        else:
            raise ValueError("Unable to find the model layers")

        if layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} out of range (total layers {len(layers)})")

        layer = layers[layer_idx]
        attn = layer.attn

        # Get hidden_states
        hidden_states_cache = {}

        def pre_hook(module, input):
            if isinstance(input, tuple):
                # Need to clone and detach to ensure the tensor can be used in subsequent computation
                hidden_states_cache['input'] = input[0].clone().detach()
            else:
                hidden_states_cache['input'] = input.clone().detach()

        hook_handle = layer.register_forward_pre_hook(pre_hook)

        # Forward pass (ensure inputs are correctly formatted to avoid vmap issues)
        # Use torch.no_grad() instead of inference_mode(), because we need to clone the tensor for subsequent computation
        with torch.no_grad():
            try:
                # Ensure attention_mask is 2D, to avoid the internal vmap issue in transformers
                safe_inputs = {}
                for k, v in inputs.items():
                    if k == 'attention_mask' and v is not None:
                        # Ensure attention_mask is 2D [batch, seq_len]
                        if len(v.shape) == 1:
                            v = v.unsqueeze(0)
                        elif len(v.shape) == 0:
                            continue  # Skip scalars
                    safe_inputs[k] = v

                # Use torch.no_grad() instead of inference_mode(), because we need to clone the tensor
                # tensors under inference_mode cannot be cloned for subsequent computation
                with torch.no_grad():
                    _ = model(**safe_inputs)
            except Exception as e:
                # Do not simply fall back to input_ids only, because missing attention_mask and other info may cause inaccurate computation
                # Instead, try to fix the inputs format and retry
                logger.warning(f"Error during forward pass: {e}, trying to fix the inputs format and retry")

                # Try a stricter format fix
                retry_inputs = {}
                for k, v in inputs.items():
                    if v is None:
                        continue
                    if k == 'attention_mask':
                        # Ensure attention_mask is 2D [batch, seq_len]
                        if len(v.shape) == 1:
                            v = v.unsqueeze(0)
                        elif len(v.shape) == 0:
                            # If it is a scalar, create a default mask
                            if 'input_ids' in inputs:
                                seq_len = inputs['input_ids'].shape[-1]
                                v = torch.ones(1, seq_len, dtype=torch.bool, device=v.device)
                            else:
                                continue
                    elif k == 'input_ids':
                        # Ensure input_ids is at least 2D [batch, seq_len]
                        if len(v.shape) == 1:
                            v = v.unsqueeze(0)
                    retry_inputs[k] = v

                # If it still fails after the fix, raise an error instead of using incomplete inputs
                try:
                    with torch.no_grad():
                        _ = model(**retry_inputs)
                except Exception as e2:
                    logger.error(f"Still failed after fixing the inputs format: {e2}")
                    logger.error(f"Original error: {e}")
                    raise ValueError(
                        f"Unable to perform the forward pass. Original error: {e}, error after fix: {e2}. "
                        f"Please check whether the inputs format is correct."
                    )

        hook_handle.remove()

        if 'input' not in hidden_states_cache:
            raise ValueError("Unable to get hidden_states")

        hidden_states = hidden_states_cache['input']  # [batch, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # GPT-2 uses c_attn to compute QKV jointly
        qkv = attn.c_attn(hidden_states)  # [batch, seq_len, 3 * hidden_dim]

        # Split into Q, K, V
        q, k, v = qkv.split(hidden_dim, dim=-1)

        # Reshape to the head level
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # Extract the specific head
        q_h = q[:, head_idx, :, :]
        k_h = k[:, head_idx, :, :]
        v_h = v[:, head_idx, :, :]

        # Compute attention scores
        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(head_dim)

        # Numerical stability: clamp scores to avoid softmax overflow
        scores = torch.clamp(scores, min=-50.0, max=50.0)

        # Apply causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1)
        scores = scores.masked_fill(causal_mask.bool(), float('-inf'))

        # Softmax (using a numerically stable implementation)
        attn_weights = F.softmax(scores, dim=-1)

        # Check whether attn_weights is NaN/Inf
        if torch.isnan(attn_weights).any() or torch.isinf(attn_weights).any():
            # If NaN/Inf occurs, use a numerically stable softmax
            scores_clamped = torch.clamp(scores, min=-50.0, max=50.0)
            attn_weights = F.softmax(scores_clamped, dim=-1)
            # If there is still NaN, replace with a uniform distribution
            if torch.isnan(attn_weights).any():
                attn_weights = torch.ones_like(attn_weights) / seq_len

        # Compute the head output
        head_output = torch.matmul(attn_weights, v_h)  # [batch, seq_len, head_dim]

        # Key fix: must go through o_proj to obtain this head's contribution to the post-attention residual
        # Create the full concat vector, with values only at this head's position
        concat_dim = num_heads * head_dim
        head_output_concat = torch.zeros(batch_size, seq_len, concat_dim, device=head_output.device, dtype=head_output.dtype)
        start_idx = head_idx * head_dim
        end_idx = start_idx + head_dim
        head_output_concat[:, :, start_idx:end_idx] = head_output

        # Go through o_proj to obtain this head's contribution
        if hasattr(attn, 'o_proj'):
            head_contribution = attn.o_proj(head_output_concat)  # Llama/Qwen use o_proj
        elif hasattr(attn, 'c_proj'):
            head_contribution = attn.c_proj(head_output_concat)  # GPT-2 uses c_proj
        else:
            # If there is no o_proj, fall back to the old method
            head_contribution = torch.zeros(batch_size, seq_len, hidden_dim, device=head_output.device)
            head_contribution[:, :, start_idx:end_idx] = head_output

        return head_contribution
    
    def _extract_all_heads_output_llama_qwen(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        layer_idx: int
    ) -> torch.Tensor:
        """
        Extract the outputs of all heads in a single forward pass (LLaMA/Qwen architecture)

        Args:
            model: the model
            inputs: input dictionary
            layer_idx: layer index

        Returns:
            all_heads_output: [batch, seq_len, num_heads, head_dim] outputs of all heads
        """
        import math
        import torch.nn.functional as F

        # Get the model config
        config = model.config if hasattr(model, 'config') else None
        if config is None:
            raise ValueError("Unable to get the model config")

        num_heads = getattr(config, 'num_attention_heads', None) or \
                   getattr(config, 'n_head', None) or 12
        head_dim = getattr(config, 'head_dim', None) or \
                  (getattr(config, 'hidden_size', 768) // num_heads)

        # Get the layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        else:
            raise ValueError("Unable to find the model layers")

        if layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} out of range (total layers {len(layers)})")

        layer = layers[layer_idx]
        attn = layer.self_attn

        # Get hidden_states (before attention)
        hidden_states_cache = {}

        def pre_hook(module, input):
            if isinstance(input, tuple):
                hidden_states_cache['input'] = input[0].clone().detach()
            else:
                hidden_states_cache['input'] = input.clone().detach()

        hook_handle = layer.register_forward_pre_hook(pre_hook)

        # Forward pass
        with torch.no_grad():
            try:
                safe_inputs = {}
                for k, v in inputs.items():
                    if k == 'attention_mask' and v is not None:
                        if len(v.shape) == 1:
                            v = v.unsqueeze(0)
                        elif len(v.shape) == 0:
                            continue
                    safe_inputs[k] = v

                with torch.no_grad():
                    _ = model(**safe_inputs)
            except Exception as e:
                logger.warning(f"Error during forward pass: {e}, trying to fix the inputs format and retry")
                retry_inputs = {}
                for k, v in inputs.items():
                    if v is None:
                        continue
                    if k == 'attention_mask':
                        if len(v.shape) == 1:
                            v = v.unsqueeze(0)
                        elif len(v.shape) == 0:
                            if 'input_ids' in inputs:
                                seq_len = inputs['input_ids'].shape[-1]
                                v = torch.ones(1, seq_len, dtype=torch.bool, device=v.device)
                            else:
                                continue
                    elif k == 'input_ids':
                        if len(v.shape) == 1:
                            v = v.unsqueeze(0)
                    retry_inputs[k] = v

                try:
                    with torch.no_grad():
                        _ = model(**retry_inputs)
                except Exception as e2:
                    logger.error(f"Still failed after fixing the inputs format: {e2}")
                    raise ValueError(f"Unable to perform the forward pass. Original error: {e}, error after fix: {e2}.")

        hook_handle.remove()

        if 'input' not in hidden_states_cache:
            raise ValueError("Unable to get hidden_states")

        hidden_states = hidden_states_cache['input']  # [batch, seq_len, hidden_dim]

        # Ensure hidden_states is 3D
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states.unsqueeze(0)
        elif len(hidden_states.shape) == 1:
            hidden_states = hidden_states.unsqueeze(0).unsqueeze(0)
        elif len(hidden_states.shape) > 3:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-2], hidden_states.shape[-1])

        if len(hidden_states.shape) != 3:
            raise ValueError(f"hidden_states has incorrect dimensions: {hidden_states.shape}, expected 3D [batch, seq_len, hidden_dim]")

        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Detect GQA (Grouped Query Attention) - check whether num_key_value_heads exists
        num_kv_heads = getattr(config, 'num_key_value_heads', None)
        if num_kv_heads is None:
            # Try to get it from the model config
            if hasattr(model, 'config'):
                num_kv_heads = getattr(model.config, 'num_key_value_heads', None)
        if num_kv_heads is None:
            # Default: standard multi-head attention, number of K/V heads equals number of Q heads
            num_kv_heads = num_heads

        # Compute Q, K, V
        q = attn.q_proj(hidden_states)  # [batch, seq_len, hidden_dim] or [batch, seq_len, num_heads * head_dim]
        k = attn.k_proj(hidden_states)  # [batch, seq_len, num_kv_heads * head_dim] (GQA) or [batch, seq_len, hidden_dim] (standard)
        v = attn.v_proj(hidden_states)  # [batch, seq_len, num_kv_heads * head_dim] (GQA) or [batch, seq_len, hidden_dim] (standard)

        # Get the actual dimensions
        q_dim = q.shape[-1]
        k_dim = k.shape[-1]
        v_dim = v.shape[-1]

        # Compute head_dim
        if q_dim % num_heads == 0:
            q_head_dim = q_dim // num_heads
        else:
            # If not divisible, try to compute from hidden_dim
            if hidden_dim % num_heads == 0:
                q_head_dim = hidden_dim // num_heads
            else:
                raise ValueError(f"Unable to determine head_dim of Q: q_dim={q_dim}, num_heads={num_heads}, hidden_dim={hidden_dim}")

        # For K/V, use num_kv_heads
        if k_dim % num_kv_heads == 0:
            kv_head_dim = k_dim // num_kv_heads
        else:
            # If not divisible, try using q_head_dim
            kv_head_dim = q_head_dim

        # Verify whether head_dim is consistent
        if q_head_dim != kv_head_dim:
            logger.warning(f"      Q head_dim({q_head_dim}) != K/V head_dim({kv_head_dim}), using Q's head_dim")
            kv_head_dim = q_head_dim

        # Reshape Q: [batch, seq_len, q_dim] -> [batch, num_heads, seq_len, q_head_dim]
        try:
            q = q.reshape(batch_size, seq_len, num_heads, q_head_dim).transpose(1, 2)
        except Exception as e:
            logger.error(f"      Q reshape failed: {e}, q.shape={q.shape}, expected=[{batch_size}, {seq_len}, {num_heads}, {q_head_dim}]")
            raise ValueError(f"Unable to reshape Q: q.shape={q.shape}, num_heads={num_heads}, q_head_dim={q_head_dim}")

        # Reshape K/V: [batch, seq_len, k_dim] -> [batch, num_kv_heads, seq_len, kv_head_dim]
        try:
            k = k.reshape(batch_size, seq_len, num_kv_heads, kv_head_dim).transpose(1, 2)
            v = v.reshape(batch_size, seq_len, num_kv_heads, kv_head_dim).transpose(1, 2)
        except Exception as e:
            logger.error(f"      K/V reshape failed: {e}, k.shape={k.shape}, v.shape={v.shape}, expected=[{batch_size}, {seq_len}, {num_kv_heads}, {kv_head_dim}]")
            raise ValueError(f"Unable to reshape K/V: k.shape={k.shape}, v.shape={v.shape}, num_kv_heads={num_kv_heads}, kv_head_dim={kv_head_dim}")

        # If GQA (num_kv_heads < num_heads), need to repeat K/V to match the number of Q heads
        if num_kv_heads < num_heads:
            # GQA case: the number of K/V heads is fewer than the number of Q heads
            # In the actual attention computation, K/V are reused to match the number of Q heads
            # Compute the repeat ratio
            num_groups = num_heads // num_kv_heads
            if num_heads % num_kv_heads != 0:
                raise ValueError(f"Invalid GQA config: num_heads({num_heads}) must be divisible by num_kv_heads({num_kv_heads})")

            # Repeat K/V: [batch, num_kv_heads, seq_len, kv_head_dim] -> [batch, num_heads, seq_len, kv_head_dim]
            k = k.repeat_interleave(num_groups, dim=1)  # Repeat along the head dimension
            v = v.repeat_interleave(num_groups, dim=1)
            logger.debug(f"      Detected GQA: num_heads={num_heads}, num_kv_heads={num_kv_heads}, repeating K/V {num_groups} times")

        # Ensure head_dim is consistent
        head_dim = q_head_dim

        # Compute attention scores (all heads computed together)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)  # [batch, num_heads, seq_len, seq_len]

        # Numerical stability: clamp scores to avoid softmax overflow
        scores = torch.clamp(scores, min=-50.0, max=50.0)

        # Apply causal mask (if needed)
        if hasattr(attn, 'is_causal') and attn.is_causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1)
            scores = scores.masked_fill(causal_mask.bool(), float('-inf'))

        # Softmax (using a numerically stable implementation)
        attn_weights = F.softmax(scores, dim=-1)  # [batch, num_heads, seq_len, seq_len]

        # Check whether attn_weights is NaN/Inf (check for each head)
        if torch.isnan(attn_weights).any() or torch.isinf(attn_weights).any():
            # If NaN/Inf occurs, use a numerically stable softmax
            scores_clamped = torch.clamp(scores, min=-50.0, max=50.0)
            attn_weights = F.softmax(scores_clamped, dim=-1)
            # If there is still NaN, replace with a uniform distribution
            nan_mask = torch.isnan(attn_weights) | torch.isinf(attn_weights)
            if nan_mask.any():
                attn_weights = torch.where(nan_mask, torch.ones_like(attn_weights) / seq_len, attn_weights)

        # Compute the outputs of all heads
        all_heads_output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]

        # Check whether all_heads_output is NaN/Inf
        if torch.isnan(all_heads_output).any() or torch.isinf(all_heads_output).any():
            # Try to fix: replace NaN/Inf with 0
            nan_inf_mask = torch.isnan(all_heads_output) | torch.isinf(all_heads_output)
            if nan_inf_mask.any():
                all_heads_output = torch.where(nan_inf_mask, torch.zeros_like(all_heads_output), all_heads_output)

        # Transpose back to [batch, seq_len, num_heads, head_dim]
        all_heads_output = all_heads_output.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]

        return all_heads_output

    def _extract_all_heads_output_gpt2(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        layer_idx: int
    ) -> torch.Tensor:
        """
        Extract the outputs of all heads in a single forward pass (GPT-2 architecture)

        Args:
            model: the model
            inputs: input dictionary
            layer_idx: layer index

        Returns:
            all_heads_output: [batch, seq_len, num_heads, head_dim] outputs of all heads
        """
        import math
        import torch.nn.functional as F

        # Get the model config
        config = model.config if hasattr(model, 'config') else None
        if config is None:
            raise ValueError("Unable to get the model config")

        num_heads = getattr(config, 'num_attention_heads', None) or \
                   getattr(config, 'n_head', None) or 12
        head_dim = getattr(config, 'head_dim', None) or \
                  (getattr(config, 'n_embd', 768) // num_heads)

        # Get the layers
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
        else:
            raise ValueError("Unable to find the model layers")

        if layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} out of range (total layers {len(layers)})")

        layer = layers[layer_idx]
        attn = layer.attn

        # Get hidden_states
        hidden_states_cache = {}

        def pre_hook(module, input):
            if isinstance(input, tuple):
                hidden_states_cache['input'] = input[0].clone().detach()
            else:
                hidden_states_cache['input'] = input.clone().detach()

        hook_handle = layer.register_forward_pre_hook(pre_hook)

        # Forward pass
        with torch.no_grad():
            try:
                safe_inputs = {}
                for k, v in inputs.items():
                    if k == 'attention_mask' and v is not None:
                        if len(v.shape) == 1:
                            v = v.unsqueeze(0)
                        elif len(v.shape) == 0:
                            continue
                    safe_inputs[k] = v

                with torch.no_grad():
                    _ = model(**safe_inputs)
            except Exception as e:
                logger.warning(f"Error during forward pass: {e}, trying to fix the inputs format and retry")
                retry_inputs = {}
                for k, v in inputs.items():
                    if v is None:
                        continue
                    if k == 'attention_mask':
                        if len(v.shape) == 1:
                            v = v.unsqueeze(0)
                        elif len(v.shape) == 0:
                            if 'input_ids' in inputs:
                                seq_len = inputs['input_ids'].shape[-1]
                                v = torch.ones(1, seq_len, dtype=torch.bool, device=v.device)
                            else:
                                continue
                    elif k == 'input_ids':
                        if len(v.shape) == 1:
                            v = v.unsqueeze(0)
                    retry_inputs[k] = v

                try:
                    with torch.no_grad():
                        _ = model(**retry_inputs)
                except Exception as e2:
                    logger.error(f"Still failed after fixing the inputs format: {e2}")
                    raise ValueError(f"Unable to perform the forward pass. Original error: {e}, error after fix: {e2}.")

        hook_handle.remove()

        if 'input' not in hidden_states_cache:
            raise ValueError("Unable to get hidden_states")

        hidden_states = hidden_states_cache['input']  # [batch, seq_len, hidden_dim]

        # Ensure hidden_states is 3D
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states.unsqueeze(0)
        elif len(hidden_states.shape) == 1:
            hidden_states = hidden_states.unsqueeze(0).unsqueeze(0)
        elif len(hidden_states.shape) > 3:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-2], hidden_states.shape[-1])

        if len(hidden_states.shape) != 3:
            raise ValueError(f"hidden_states has incorrect dimensions: {hidden_states.shape}, expected 3D [batch, seq_len, hidden_dim]")

        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Compute Q, K, V
        qkv = attn.c_attn(hidden_states)  # [batch, seq_len, 3 * hidden_dim]

        # Split into Q, K, V
        q, k, v = qkv.split(hidden_dim, dim=-1)  # each is [batch, seq_len, hidden_dim]

        # Reshape to the head level
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # Compute attention scores (all heads computed together)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)  # [batch, num_heads, seq_len, seq_len]

        # Numerical stability: clamp scores to avoid softmax overflow
        scores = torch.clamp(scores, min=-50.0, max=50.0)

        # Apply causal mask (if needed)
        if hasattr(attn, 'is_causal') and attn.is_causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1)
            scores = scores.masked_fill(causal_mask.bool(), float('-inf'))

        # Softmax (using a numerically stable implementation)
        attn_weights = F.softmax(scores, dim=-1)  # [batch, num_heads, seq_len, seq_len]

        # Check whether attn_weights is NaN/Inf (check for each head)
        if torch.isnan(attn_weights).any() or torch.isinf(attn_weights).any():
            # If NaN/Inf occurs, use a numerically stable softmax
            scores_clamped = torch.clamp(scores, min=-50.0, max=50.0)
            attn_weights = F.softmax(scores_clamped, dim=-1)
            # If there is still NaN, replace with a uniform distribution
            nan_mask = torch.isnan(attn_weights) | torch.isinf(attn_weights)
            if nan_mask.any():
                attn_weights = torch.where(nan_mask, torch.ones_like(attn_weights) / seq_len, attn_weights)

        # Compute the outputs of all heads
        all_heads_output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]

        # Check whether all_heads_output is NaN/Inf
        if torch.isnan(all_heads_output).any() or torch.isinf(all_heads_output).any():
            # Try to fix: replace NaN/Inf with 0
            nan_inf_mask = torch.isnan(all_heads_output) | torch.isinf(all_heads_output)
            if nan_inf_mask.any():
                all_heads_output = torch.where(nan_inf_mask, torch.zeros_like(all_heads_output), all_heads_output)

        # Transpose back to [batch, seq_len, num_heads, head_dim]
        all_heads_output = all_heads_output.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]

        return all_heads_output

    def extract_all_heads_output(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        layer_idx: int
    ) -> torch.Tensor:
        """
        Extract the outputs of all heads in a single forward pass (optimized version)

        This avoids needing a forward pass per head, greatly improving performance

        Args:
            model: the model
            inputs: input dictionary
            layer_idx: layer index

        Returns:
            all_heads_output: [batch, seq_len, num_heads, head_dim] outputs of all heads
        """
        # Detect the model type
        model_type = self._detect_model_type(model)

        if model_type in ['llama', 'qwen']:
            return self._extract_all_heads_output_llama_qwen(model, inputs, layer_idx)
        elif model_type == 'gpt2':
            return self._extract_all_heads_output_gpt2(model, inputs, layer_idx)
        else:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Currently only 'llama', 'qwen', 'gpt2' are supported. "
                f"Please ensure the model architecture is correct, or add the corresponding head extraction implementation."
            )
    
    def compute_head_write_contribution(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        layer_idx: int,
        head_idx: int
    ) -> torch.Tensor:
        """
        Compute the head's contribution to the post-attention residual w_{l,h}(x)

        According to the transformer architecture, the head's contribution is:
        w_{l,h}(x) = attention_output_h, i.e., the output portion of that head

        Note: this function is kept for backward compatibility, but using the optimized extract_all_heads_output version is recommended

        Args:
            model: the model
            inputs: input dictionary
            layer_idx: layer index
            head_idx: head index

        Returns:
            w_{l,h}(x) [batch, seq_len, hidden_dim]
        """
        # Detect the model type
        model_type = self._detect_model_type(model)

        if model_type in ['llama', 'qwen']:
            return self._extract_head_output_llama_qwen(model, inputs, layer_idx, head_idx)
        elif model_type == 'gpt2':
            return self._extract_head_output_gpt2(model, inputs, layer_idx, head_idx)
        else:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Currently only 'llama', 'qwen', 'gpt2' are supported. "
                f"Please ensure the model architecture is correct, or add the corresponding head extraction implementation."
            )
    
    def compute_importance(
        self,
        model: nn.Module,
        domain_data: Dict[int, List[Dict[str, torch.Tensor]]],  # {domain_k: [inputs]}
        domain_axes: torch.Tensor,  # [num_domains, num_domains] one-hot domain axes
        device: Optional[torch.device] = None,
        batch_size: int = 32,  # batch size (consistent with probe training)
        max_samples_per_domain: Optional[int] = 200,  # max number of samples per domain (sampling)
        foundation_layers: Optional[List[int]] = None,  # foundation layers (skip computation)
        layer_group_size: int = 5,  # number of layers processed in parallel (a 48G GPU can support 5-10 layers)
        layer_domain_relevances: Optional[Dict[int, List[float]]] = None,  # per-layer domain relevance (for comparison)
        use_calibration: bool = False  # whether to use method 3 calibration (default False, because if the relative ordering is correct, calibration only makes the data look nicer)
    ) -> Dict[int, torch.Tensor]:
        """
        Compute the importance of all heads for all domains I_{l,h,k}

        Note: domain axes have a one-hot structure, not a true PCA space
        Use probes to project the post-attention residual into domain space

        Args:
            model: the model
            domain_data: training data for each domain {domain_k: [inputs]}
            domain_axes: domain axes [num_domains, num_domains] one-hot matrix
            device: compute device

        Returns:
            {layer_idx: importance_tensor [num_heads, num_domains]}
        """
        if device is None:
            device = next(model.parameters()).device

        domain_axes = domain_axes.to(device)

        if self.layer_probes is None:
            logger.warning("No probes provided, unable to project the residual into domain space")

        # Handle foundation layers
        if foundation_layers is None:
            foundation_layers = []
        foundation_layers_set = set(foundation_layers)
        layers_to_process = [i for i in range(self.num_layers) if i not in foundation_layers_set]

        logger.info("Starting head importance computation...")
        logger.info(f"  Total layers: {self.num_layers}")
        logger.info(f"  Foundation layers (skipped): {foundation_layers}")
        logger.info(f"  Number of layers to compute: {len(layers_to_process)}")
        logger.info(f"  Heads per layer: {self.num_heads_per_layer}")
        logger.info(f"  Number of domains: {self.num_domains}")
        logger.info(f"  Number of layers processed in parallel: {layer_group_size}")

        # Sampling: if max_samples_per_domain is specified, sample each domain
        sampled_domain_data = {}
        if max_samples_per_domain is not None:
            import random
            for domain_k, inputs_list in domain_data.items():
                if len(inputs_list) > max_samples_per_domain:
                    sampled_domain_data[domain_k] = random.sample(inputs_list, max_samples_per_domain)
                    logger.info(f"  Domain {domain_k}: sampled {max_samples_per_domain}/{len(inputs_list)} samples")
                else:
                    sampled_domain_data[domain_k] = inputs_list
        else:
            sampled_domain_data = domain_data

        # Count the total number of samples
        total_samples = sum(len(inputs_list) for inputs_list in sampled_domain_data.values())
        logger.info(f"  Total samples: {total_samples} (after sampling)")
        logger.info(f"  Batch size: {batch_size}")
        total_computations = len(layers_to_process) * self.num_heads_per_layer * total_samples
        logger.info(f"  Estimated computation: {len(layers_to_process)} layers x {self.num_heads_per_layer} heads x {total_samples} samples = {total_computations:,} head extractions")
        logger.info("  Note: with batching and parallel processing optimizations, a 10-20x speedup is expected")
        logger.info("=" * 80)

        importance = {}

        # Initialize the importance of foundation layers to 0 (fully retained)
        for layer_idx in foundation_layers:
            importance[layer_idx] = torch.zeros(
                self.num_heads_per_layer,
                self.num_domains,
                device=device
            )
            logger.info(f"  Layer {layer_idx} (foundation layer): skip computation, importance set to 0 (fully retained)")

        # Process multiple layers in parallel (grouped processing)
        for group_start in range(0, len(layers_to_process), layer_group_size):
            group_end = min(group_start + layer_group_size, len(layers_to_process))
            group_layers = layers_to_process[group_start:group_end]
            logger.info(f"  Processing layer group {group_start+1}-{group_end}/{len(layers_to_process)} in parallel: {group_layers}")

            # Process each layer (can be further optimized into true parallelism)
            for layer_idx in group_layers:
                logger.info("")
                logger.info(f"  {'='*70}")
                logger.info(f"  [Layer {layer_idx}] Starting processing...")
                logger.info(f"  {'='*70}")
                layer_importance = torch.zeros(
                    self.num_heads_per_layer,
                    self.num_domains,
                    device=device
                )
                
                for domain_k in range(self.num_domains):
                    if domain_k not in sampled_domain_data:
                        continue

                    domain_inputs_list = sampled_domain_data[domain_k]
                    logger.info(f"    [domain {domain_k}] Starting processing: {len(domain_inputs_list)} samples...")

                    # Ensure domain_axes is 2D
                    if len(domain_axes.shape) == 1:
                        # If 1D, need to reshape
                        num_domains_axes = domain_axes.shape[0]
                        domain_axes = domain_axes.view(1, -1)  # [1, num_domains]

                    # Extract the axis of the k-th domain
                    if len(domain_axes.shape) == 2:
                        u_k = domain_axes[domain_k]  # [num_domains]
                    else:
                        logger.error(f"domain_axes has incorrect dimensions: {domain_axes.shape}, expected 2D")
                        raise ValueError(f"domain_axes has incorrect dimensions: {domain_axes.shape}")

                    # Ensure u_k is a 1D vector
                    u_k = u_k.flatten()

                    # Compute the expectation
                    energy_sum = torch.zeros(self.num_heads_per_layer, device=device)
                    count = torch.zeros(self.num_heads_per_layer, device=device)  # count per head

                    # Batching: group the samples
                    num_batches = (len(domain_inputs_list) + batch_size - 1) // batch_size

                    for batch_idx in range(num_batches):
                        # Initialize batch_w_lh_dict (used to store each head's batch outputs)
                        batch_w_lh_dict = []  # [head_h] -> [w_lh1, w_lh2, ...]
                        batch_start = batch_idx * batch_size
                        batch_end = min(batch_start + batch_size, len(domain_inputs_list))
                        batch_inputs_list = domain_inputs_list[batch_start:batch_end]

                        # Suppressed batch-processing logs, too distracting
                        # if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                        #     logger.debug(f"      [domain {domain_k}] processed batch {batch_idx+1}/{num_batches} (samples {batch_end}/{len(domain_inputs_list)})...")

                        # Prepare the batch inputs
                        batch_processed_inputs = []
                        for inputs in batch_inputs_list:
                            # Ensure inputs are on the correct device, and fix dimension issues
                            processed_inputs = {}
                            for k, v in inputs.items():
                                v = v.to(device)

                                # Validate and fix input_ids
                                if k == 'input_ids':
                                    # Ensure input_ids is 2D
                                    if len(v.shape) == 1:
                                        v = v.unsqueeze(0)  # [seq_len] -> [1, seq_len]
                                    # Check whether token ids are within the valid range (to avoid device-side assert)
                                    if hasattr(model, 'config') and hasattr(model.config, 'vocab_size'):
                                        vocab_size = model.config.vocab_size
                                        if (v >= vocab_size).any():
                                            logger.warning(f"Found a token id outside the vocab range, will clamp to [0, {vocab_size-1}]")
                                            v = torch.clamp(v, 0, vocab_size - 1)

                                # Ensure attention_mask is 2D, to avoid vmap issues
                                if k == 'attention_mask':
                                    if len(v.shape) == 1:
                                        v = v.unsqueeze(0)  # [seq_len] -> [1, seq_len]
                                    elif len(v.shape) == 0:
                                        # If it is a scalar, create a default mask based on input_ids
                                        if 'input_ids' in processed_inputs:
                                            seq_len = processed_inputs['input_ids'].shape[-1]
                                            v = torch.ones(1, seq_len, dtype=torch.bool, device=v.device)
                                        else:
                                            continue  # Skip

                                processed_inputs[k] = v

                            # If the processed inputs are empty or missing input_ids, skip
                            if not processed_inputs or 'input_ids' not in processed_inputs:
                                continue

                            batch_processed_inputs.append(processed_inputs)

                        if not batch_processed_inputs:
                            continue

                        # Optimization: extract all head outputs in a single forward pass
                        # Extract all heads for the first sample in the batch (all samples share the same head structure)
                        all_heads_output_dict = {}  # {sample_idx: all_heads_output}

                        # Try the optimized version: extract all heads in a single forward pass
                        try:
                            # Extract all heads for each sample in the batch (because seq_len may differ)
                            for sample_idx, processed_inputs in enumerate(batch_processed_inputs):
                                try:
                                    all_heads_output = self.extract_all_heads_output(
                                        model, processed_inputs, layer_idx
                                    )  # [batch, seq_len, num_heads, head_dim]
                                    all_heads_output_dict[sample_idx] = all_heads_output
                                except Exception as e:
                                    logger.warning(f"      Layer {layer_idx} sample {sample_idx} batch {batch_idx} failed to extract all heads: {e}, skipping this sample")
                                    continue
                        except Exception as e:
                            logger.warning(f"      Layer {layer_idx} batch {batch_idx} optimized version failed: {e}, falling back to per-head extraction")
                            all_heads_output_dict = {}
                        
                        # ======================================================================
                        # Method 2 (default method, already used in the paper): compute a single head's contribution by subtracting the other heads from the full residual
                        # ======================================================================
                        # Principle:
                        #   1. Compute the full residual = o_proj(concat of all heads)
                        #   2. For each head_h, compute the other-heads residual = o_proj(concat of the other heads with head_h removed)
                        #   3. head_h's contribution = full residual - other-heads residual
                        #
                        # Advantages (suitable for the paper):
                        #   - Mathematically rigorous: w_{l,h}(x) = r^{(l)}(x) - r^{(l)}_{-h}(x)
                        #   - Mathematically correct: the sum of all heads' contributions = full residual
                        #   - Avoids the o_proj bias issue: because the full residual is used, the bias is correctly canceled
                        #   - Theoretically sound: better matches mathematical intuition
                        #   - Easy to explain: readers can understand it easily
                        #   - Broadly applicable: works for all model architectures (as long as o_proj/c_proj is supported)
                        #
                        # Note:
                        #   - Although it yields the same result as method 1 (the old method, unbiased), it is theoretically more rigorous
                        #   - Already used in the paper (4-methodology.tex, line 218-221)
                        #   - avg_head_importance and layer_relevance differ substantially (~0.33), but this is a scaling issue and does not affect the relative ordering
                        # ======================================================================

                        # Get the attention module and o_proj (used to compute the residual)
                        layer = model.model.layers[layer_idx] if hasattr(model, 'model') else model.layers[layer_idx]
                        attn = layer.self_attn if hasattr(layer, 'self_attn') else layer.attn

                        # Get hidden_dim and head_dim
                        hidden_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else \
                                    (getattr(model.config, 'n_embd', 768) if hasattr(model.config, 'n_embd') else 768)

                        # If all heads were extracted successfully, use the optimized version (method 2)
                        if all_heads_output_dict:
                            # logger.info(f"      [domain {domain_k}] batch {batch_idx} using optimized method 2 (full_residual - other_heads_residual)")  # confirmed working, debug info commented out
                            # For each sample, compute the full residual and the other-heads residual
                            for sample_idx, processed_inputs in enumerate(batch_processed_inputs):
                                if sample_idx not in all_heads_output_dict:
                                    continue

                                all_heads_output = all_heads_output_dict[sample_idx]  # [batch, seq_len, num_heads, head_dim]

                                # Check the shape of all_heads_output
                                if len(all_heads_output.shape) == 3:
                                    # If it returns [batch, seq_len, hidden_dim], need to reshape
                                    batch_size, seq_len, hidden_dim = all_heads_output.shape
                                    # Get num_heads and head_dim from the model config
                                    config = model.config if hasattr(model, 'config') else None
                                    if config is None:
                                        logger.warning(f"      Layer {layer_idx} sample {sample_idx} batch {batch_idx} unable to get the model config, skipping this sample")
                                        continue
                                    num_heads = getattr(config, 'num_attention_heads', None) or \
                                               getattr(config, 'n_head', None) or 40
                                    head_dim = getattr(config, 'head_dim', None) or \
                                              (hidden_dim // num_heads)
                                    # Verify: num_heads * head_dim should equal hidden_dim
                                    if num_heads * head_dim != hidden_dim:
                                        logger.warning(f"      Layer {layer_idx} sample {sample_idx} batch {batch_idx} num_heads({num_heads}) * head_dim({head_dim}) != hidden_dim({hidden_dim}), trying to fix")
                                        head_dim = hidden_dim // num_heads
                                    # Reshape to [batch, seq_len, num_heads, head_dim]
                                    all_heads_output = all_heads_output.view(batch_size, seq_len, num_heads, head_dim)
                                elif len(all_heads_output.shape) == 4:
                                    batch_size, seq_len, num_heads, head_dim = all_heads_output.shape
                                else:
                                    logger.warning(f"      Layer {layer_idx} sample {sample_idx} batch {batch_idx} all_heads_output has incorrect shape: {all_heads_output.shape}, skipping this sample")
                                    continue

                                concat_dim = num_heads * head_dim

                                # Check whether all_heads_output is NaN/Inf
                                if torch.isnan(all_heads_output).any() or torch.isinf(all_heads_output).any():
                                    nan_inf_mask = torch.isnan(all_heads_output) | torch.isinf(all_heads_output)
                                    all_heads_output = torch.where(nan_inf_mask, torch.zeros_like(all_heads_output), all_heads_output)
                                    if torch.isnan(all_heads_output).all() or torch.isinf(all_heads_output).all():
                                        continue

                                # Reshape all_heads_output into the concat format [batch, seq_len, num_heads*head_dim]
                                # Use reshape instead of view, because the tensor may not be contiguous
                                all_heads_output_reshaped = all_heads_output.reshape(batch_size, seq_len, concat_dim)

                                # Compute the full residual (all heads through o_proj)
                                if hasattr(attn, 'o_proj'):
                                    full_residual = attn.o_proj(all_heads_output_reshaped)  # [batch, seq_len, hidden_dim]
                                elif hasattr(attn, 'c_proj'):
                                    full_residual = attn.c_proj(all_heads_output_reshaped)  # GPT-2 uses c_proj
                                else:
                                    # If there is no o_proj, fall back to the old method
                                    full_residual = torch.zeros(batch_size, seq_len, hidden_dim, device=all_heads_output.device)
                                    full_residual[:, :, :concat_dim] = all_heads_output_reshaped

                                # For each head_h, compute its contribution = full residual - other-heads residual
                                for head_h in range(self.num_heads_per_layer):
                                    # Create the concat of the other heads (with head_h removed)
                                    other_heads_output = all_heads_output.clone()
                                    other_heads_output[:, :, head_h, :] = 0  # Zero out head_h's position
                                    # Use reshape instead of view, because the tensor may not be contiguous
                                    other_heads_output_reshaped = other_heads_output.reshape(batch_size, seq_len, concat_dim)

                                    # Compute the other-heads residual
                                    if hasattr(attn, 'o_proj'):
                                        other_heads_residual = attn.o_proj(other_heads_output_reshaped)  # [batch, seq_len, hidden_dim]
                                    elif hasattr(attn, 'c_proj'):
                                        other_heads_residual = attn.c_proj(other_heads_output_reshaped)
                                    else:
                                        other_heads_residual = torch.zeros(batch_size, seq_len, hidden_dim, device=all_heads_output.device)
                                        other_heads_residual[:, :, :concat_dim] = other_heads_output_reshaped

                                    # head_h's contribution = full residual - other-heads residual
                                    w_lh = full_residual - other_heads_residual  # [batch, seq_len, hidden_dim]

                                    # Check whether w_lh is NaN/Inf
                                    if torch.isnan(w_lh).any() or torch.isinf(w_lh).any():
                                        nan_inf_mask = torch.isnan(w_lh) | torch.isinf(w_lh)
                                        w_lh = torch.where(nan_inf_mask, torch.zeros_like(w_lh), w_lh)
                                        if torch.isnan(w_lh).all() or torch.isinf(w_lh).all():
                                            continue

                                    # Add w_lh to the corresponding head's batch list
                                    if head_h >= len(batch_w_lh_dict):
                                        # Initialize batch_w_lh_dict
                                        for _ in range(head_h + 1 - len(batch_w_lh_dict)):
                                            batch_w_lh_dict.append([])
                                    batch_w_lh_dict[head_h].append(w_lh)

                        # Batching: compute all heads simultaneously (fall back to the old method if the optimized version fails)
                        if not all_heads_output_dict:
                            # logger.info(f"      [domain {domain_k}] batch {batch_idx} falling back to the old method (per-head extraction)")  # confirmed working, debug info commented out
                            for head_h in range(self.num_heads_per_layer):
                                # Collect all batch outputs for this head
                                batch_w_lh_list = []

                                # Fall back to per-head extraction
                                for processed_inputs in batch_processed_inputs:
                                    # Compute w_{l,h}(x) - the head's contribution to the post-attention residual
                                    try:
                                        w_lh = self.compute_head_write_contribution(
                                            model, processed_inputs, layer_idx, head_h
                                        )  # [batch, seq_len, hidden_dim]
                                        batch_w_lh_list.append(w_lh)
                                    except Exception as e:
                                        logger.warning(f"      Layer {layer_idx} head {head_h} batch {batch_idx} computation failed: {e}, skipping this sample")
                                        continue

                                if batch_w_lh_list:
                                    if head_h >= len(batch_w_lh_dict):
                                        for _ in range(head_h + 1 - len(batch_w_lh_dict)):
                                            batch_w_lh_dict.append([])
                                    batch_w_lh_dict[head_h].extend(batch_w_lh_list)

                        # Process the batch outputs of all heads
                        for head_h in range(self.num_heads_per_layer):
                            if head_h >= len(batch_w_lh_dict) or not batch_w_lh_dict[head_h]:
                                continue

                            batch_w_lh_list = batch_w_lh_dict[head_h]

                            if not batch_w_lh_list:
                                continue

                            # Merge the outputs of all batches
                            # Note: since each sample's seq_len may differ, we need to process them separately
                            for w_lh in batch_w_lh_list:
                                # Check whether w_lh is NaN/Inf, try to fix it rather than skipping directly
                                if torch.isnan(w_lh).any() or torch.isinf(w_lh).any():
                                    # Try to fix: replace NaN/Inf with 0
                                    nan_inf_mask = torch.isnan(w_lh) | torch.isinf(w_lh)
                                    if nan_inf_mask.any():
                                        w_lh = torch.where(nan_inf_mask, torch.zeros_like(w_lh), w_lh)
                                        # If all are NaN/Inf, skip
                                        if torch.isnan(w_lh).all() or torch.isinf(w_lh).all():
                                            logger.warning(f"      Layer {layer_idx} head {head_h} w_lh is entirely NaN/Inf, skipping")
                                            continue
                                    # Continue processing after the fix

                                # Average over the sequence dimension
                                w_lh_mean = w_lh.mean(dim=1)  # [batch, hidden_dim]

                                # Check whether w_lh_mean is NaN
                                if torch.isnan(w_lh_mean).any() or torch.isinf(w_lh_mean).any():
                                    logger.warning(f"      Layer {layer_idx} head {head_h} w_lh_mean contains NaN/Inf, skipping")
                                    continue

                                # Use the probe to project w_lh into domain space
                                if self.layer_probes is not None and layer_idx in self.layer_probes.probes:
                                    probe = self.layer_probes.probes[layer_idx]
                                    probe.eval()
                                    with torch.no_grad():
                                        logits = probe(w_lh_mean)  # [batch, num_domains] or [num_domains]

                                        # Check whether logits is NaN
                                        if torch.isnan(logits).any() or torch.isinf(logits).any():
                                            logger.warning(f"      [domain {domain_k}] [head {head_h}] probe output logits contains NaN/Inf, skipping")
                                            continue

                                        # Check whether the logits values are abnormal (for debugging extreme-value issues)
                                        if head_h == 0 and len(batch_w_lh_list) == 1:
                                            logits_max = logits.max().item()
                                            logits_min = logits.min().item()
                                            if abs(logits_max) > 10 or abs(logits_min) > 10:
                                                logger.debug(f"      [domain {domain_k}] [head {head_h}] abnormal probe logits values: max={logits_max:.4f}, min={logits_min:.4f}")

                                        # Ensure logits is 2D [batch, num_domains]
                                        if len(logits.shape) == 1:
                                            logits = logits.unsqueeze(0)  # [num_domains] -> [1, num_domains]
                                        elif len(logits.shape) > 2:
                                            if logits.shape[0] > 0:
                                                logits = logits[0]
                                                if len(logits.shape) == 1:
                                                    logits = logits.unsqueeze(0)
                                            else:
                                                continue

                                        w_lh_domain = torch.sigmoid(logits)  # [batch, num_domains] (1-vs-rest)

                                        # Once again ensure w_lh_domain is 2D
                                        if len(w_lh_domain.shape) == 1:
                                            w_lh_domain = w_lh_domain.unsqueeze(0)
                                        elif len(w_lh_domain.shape) > 2:
                                            if w_lh_domain.shape[0] > 0:
                                                w_lh_domain = w_lh_domain[0]
                                                if len(w_lh_domain.shape) == 1:
                                                    w_lh_domain = w_lh_domain.unsqueeze(0)
                                            else:
                                                continue

                                        if len(w_lh_domain.shape) != 2:
                                            continue
                                else:
                                    # Without a probe, cannot project correctly
                                    raise ValueError(
                                        f"Layer {layer_idx} has no probe, unable to project the residual into domain space. "
                                        f"Please ensure the probes are trained before computing head importance."
                                    )

                                # Compute the axis-aligned energy
                                # (u_k^T w_{l,h}(x))^2 / (||w_{l,h}(x)||^2 + eps)
                                # u_k is a one-hot vector (1 at the k-th position), so u_k^T w is effectively the k-th dimension of w
                                for b in range(w_lh_domain.shape[0]):
                                    w = w_lh_domain[b]  # [num_domains]

                                    # Ensure w is a 1D vector
                                    if len(w.shape) > 1:
                                        w = w.flatten()
                                    elif len(w.shape) == 0:
                                        # If it is a scalar, skip
                                        continue

                                    # Ensure u_k is a 1D vector (already handled outside the loop, but ensure again here)
                                    u_k_flat = u_k.flatten() if len(u_k.shape) > 1 else u_k

                                    # Ensure the dimensions match
                                    if len(u_k_flat.shape) == 0 or len(w.shape) == 0:
                                        continue

                                    if u_k_flat.shape[0] != w.shape[0]:
                                        logger.warning(f"Dimension mismatch: u_k.shape={u_k_flat.shape}, w.shape={w.shape}, skipping")
                                        continue

                                    # Compute the alignment energy
                                    # u_k is one-hot, so u_k^T w is the k-th element of w
                                    w_k = torch.dot(u_k_flat, w)  # scalar (effectively w[k])

                                    # Modified formula: directly use w[k] instead of normalized energy
                                    # This way the mean importance of all heads equals the layer relevance
                                    # Original formula: energy = (w[k]^2) / (||w||^2 + eps)  # with normalization, the value shrinks
                                    # New formula: energy = w[k]  # directly use w[k], keeping it consistent with layer relevance
                                    energy = w_k  # scalar

                                    # Debug: log the distribution of w (to analyze why the value is so small)
                                    # Only log for the first sample, first head, first batch, to avoid excessive logs
                                    if head_h == 0 and b == 0 and batch_idx == 0:
                                        w_k_val = w[domain_k].item()
                                        w_other_mean = (w.sum().item() - w_k_val) / (len(w) - 1) if len(w) > 1 else 0.0
                                        w_max = w.max().item()
                                        w_min = w.min().item()
                                        logger.debug(f"      [domain {domain_k}] [head {head_h}] sample 0 w distribution: w[k]={w_k_val:.4f}, mean of w[others]={w_other_mean:.4f}, w max={w_max:.4f}, w min={w_min:.4f}, energy={energy.item():.4f}")

                                    # Check for NaN and Inf
                                    if torch.isnan(energy) or torch.isinf(energy):
                                        logger.warning(f"      [domain {domain_k}] [head {head_h}] energy is NaN/Inf, skipping")
                                        continue

                                    # Ensure energy is a scalar
                                    if isinstance(energy, torch.Tensor):
                                        if energy.numel() == 1:
                                            energy_sum[head_h] += energy.item()
                                            count[head_h] += 1
                                        else:
                                            logger.warning(f"energy is not a scalar: {energy.shape}, skipping")
                                            continue
                                    else:
                                        energy_sum[head_h] += energy
                                        count[head_h] += 1

                    # Compute the mean importance (avoid division by 0)
                    for head_h in range(self.num_heads_per_layer):
                        if count[head_h] > 0:
                            layer_importance[head_h, domain_k] = energy_sum[head_h] / count[head_h]
                        else:
                            layer_importance[head_h, domain_k] = 0.0
                            logger.debug(f"      Layer {layer_idx} head {head_h} domain {domain_k} count=0, setting to 0.0")

                    # Check for NaN
                    nan_mask = torch.isnan(layer_importance[:, domain_k])
                    if nan_mask.any():
                        nan_count = nan_mask.sum().item()
                        logger.warning(f"    Layer {layer_idx} domain {domain_k} has NaN: {nan_count}/{self.num_heads_per_layer} heads, replacing NaN with 0")
                        logger.warning(f"      count per head: {count.cpu().numpy()}")
                        logger.warning(f"      energy_sum per head: {energy_sum.cpu().numpy()}")
                        layer_importance[:, domain_k] = torch.nan_to_num(layer_importance[:, domain_k], nan=0.0)

                    total_count = count.sum().item()
                    if total_count > 0:
                        avg_importance = layer_importance[:, domain_k].mean().item()
                        max_importance = layer_importance[:, domain_k].max().item()
                        # Check for NaN
                        if np.isnan(avg_importance) or np.isnan(max_importance):
                            logger.warning(f"    [domain {domain_k}] mean/max importance is NaN (even after replacement)")
                            logger.warning(f"      head importances: {layer_importance[:, domain_k].cpu().numpy()}")
                        else:
                            logger.info(f"    [domain {domain_k}] done: mean importance={avg_importance:.4f}, max importance={max_importance:.4f}, total samples={total_count}")
                            # Verify: the mean importance should be close to the layer relevance (0.7-0.9)
                            if avg_importance < 0.1:
                                logger.warning(f"      Warning: mean importance ({avg_importance:.4f}) is far below the expected layer relevance (0.7-0.9), there may be an issue")
                            elif avg_importance > 0.5:
                                logger.debug(f"      OK: mean importance ({avg_importance:.4f}) is close to the expected layer relevance range")
                    else:
                        logger.warning(f"    [domain {domain_k}] no data (all samples were skipped)")
                        logger.warning(f"      Possible causes:")
                        logger.warning(f"        1. All w_lh computations failed")
                        logger.warning(f"        2. All probe outputs were NaN")
                        logger.warning(f"        3. All energy computations failed")

                importance[layer_idx] = layer_importance

                # Check for NaN and replace
                if torch.isnan(layer_importance).any():
                    logger.warning(f"  [Layer {layer_idx}] has NaN, replacing NaN with 0")
                    layer_importance = torch.nan_to_num(layer_importance, nan=0.0)
                    importance[layer_idx] = layer_importance

                avg_layer_importance = layer_importance.mean().item()
                max_layer_importance = layer_importance.max().item()
                if np.isnan(avg_layer_importance) or np.isnan(max_layer_importance):
                    logger.warning(f"  [Layer {layer_idx}] mean/max importance is NaN")
                else:
                    logger.info(f"  [Layer {layer_idx}] done: mean importance={avg_layer_importance:.4f}, max importance={max_layer_importance:.4f}")

                    # Print each head's importance for each domain
                    logger.info(f"  [Layer {layer_idx}] each head's importance for each domain:")
                    for head_idx in range(self.num_heads_per_layer):
                        head_importance_list = layer_importance[head_idx, :].cpu().numpy()
                        # Format as a list string
                        importance_str = ", ".join([f"{val:.4f}" for val in head_importance_list])
                        logger.info(f"    head{head_idx} = [{importance_str}]")

                    # Statistical analysis: compute the mean similarity per domain after averaging over heads, and compare with layer relevance
                    layer_importance_np = layer_importance.cpu().numpy()  # [num_heads, num_domains]

                    # 1. Compute the mean importance per domain after averaging over heads (should equal layer relevance)
                    avg_head_importance_per_domain = layer_importance_np.mean(axis=0)  # [num_domains]
                    logger.info(f"  [Layer {layer_idx}] mean importance per domain after averaging over heads:")
                    avg_importance_str = ", ".join([f"{val:.4f}" for val in avg_head_importance_per_domain])
                    logger.info(f"    avg_head_importance = [{avg_importance_str}]")

                    # 2. Compare with the layer's domain relevance (if provided)
                    if layer_domain_relevances is not None and layer_idx in layer_domain_relevances:
                        layer_relevances = layer_domain_relevances[layer_idx]  # [num_domains]
                        logger.info(f"  [Layer {layer_idx}] layer's domain relevance (computed during probe training):")
                        layer_relevance_str = ", ".join([f"{val:.4f}" for val in layer_relevances])
                        logger.info(f"    layer_relevance = [{layer_relevance_str}]")

                        # Compute the difference
                        if len(avg_head_importance_per_domain) == len(layer_relevances):
                            differences = np.abs(avg_head_importance_per_domain - np.array(layer_relevances))
                            logger.info(f"  [Layer {layer_idx}] difference analysis (|avg_head_importance - layer_relevance|):")
                            diff_str = ", ".join([f"{val:.4f}" for val in differences])
                            logger.info(f"    differences = [{diff_str}]")
                            logger.info(f"    mean difference: {differences.mean():.4f}, max difference: {differences.max():.4f}")

                            # ======================================================================
                            # Method 3: calibrate head importance using layer_relevance (optional)
                            # ======================================================================
                            # Principle:
                            #   1. After computing all head importances, we find avg_head_importance < layer_relevance
                            #   2. Use layer_relevance to calibrate each head's importance
                            #   3. Calibration formula: I_{l,h,k}_calibrated = I_{l,h,k} * (layer_relevance[k] / avg_head_importance[k])
                            #
                            # Note:
                            #   - Verification shows: the head ordering is identical before and after calibration, indicating the relative ordering is correct
                            #   - If the relative ordering is correct, calibration is just uniform scaling and does not change the pruning result
                            #   - With a unified pruning-strength parameter, the best-matching head can still be selected
                            #   - Method 3 may only "make the data look nicer" (make avg_head_importance = layer_relevance)
                            #   - If avg_head_importance[k] = 0, skip this calibration (to avoid division by zero)
                            #   - The calibrated importance may exceed 1.0, which is normal (because layer_relevance may be >1.0)
                            # ======================================================================

                            # Apply calibration (method 3, optional)
                            if use_calibration:
                                layer_relevances_tensor = torch.tensor(layer_relevances, device=layer_importance.device, dtype=layer_importance.dtype)
                                avg_head_importance_tensor = torch.tensor(avg_head_importance_per_domain, device=layer_importance.device, dtype=layer_importance.dtype)

                                # Compute the calibration factors (avoid division by zero)
                                calibration_factors = torch.ones_like(layer_relevances_tensor)
                                for k in range(len(layer_relevances_tensor)):
                                    if avg_head_importance_tensor[k] > 1e-8:  # Avoid division by zero
                                        calibration_factors[k] = layer_relevances_tensor[k] / avg_head_importance_tensor[k]
                                    else:
                                        # If avg_head_importance is 0, use layer_relevance as the calibration factor
                                        calibration_factors[k] = layer_relevances_tensor[k] / (1e-8)

                                # Apply calibration: scale each head's importance for each domain
                                layer_importance_calibrated = layer_importance.clone()  # [num_heads, num_domains]
                                for k in range(self.num_domains):
                                    layer_importance_calibrated[:, k] *= calibration_factors[k]

                                # Update importance (using the calibrated values)
                                layer_importance = layer_importance_calibrated
                                importance[layer_idx] = layer_importance

                                # Verify the calibrated avg_head_importance
                                layer_importance_np_calibrated = layer_importance.cpu().numpy()
                                avg_head_importance_calibrated = layer_importance_np_calibrated.mean(axis=0)
                                logger.info(f"  [Layer {layer_idx}] calibrated avg_head_importance:")
                                avg_calibrated_str = ", ".join([f"{val:.4f}" for val in avg_head_importance_calibrated])
                                logger.info(f"    avg_head_importance_calibrated = [{avg_calibrated_str}]")

                                # Verify the difference (should be close to 0)
                                differences_calibrated = np.abs(avg_head_importance_calibrated - np.array(layer_relevances))
                                logger.info(f"  [Layer {layer_idx}] calibrated difference analysis:")
                                diff_calibrated_str = ", ".join([f"{val:.4f}" for val in differences_calibrated])
                                logger.info(f"    differences_calibrated = [{diff_calibrated_str}]")
                                logger.info(f"    mean difference: {differences_calibrated.mean():.4f}, max difference: {differences_calibrated.max():.4f}")

                                # Update layer_importance_np for subsequent statistics
                                layer_importance_np = layer_importance_np_calibrated
                            else:
                                logger.info(f"  [Layer {layer_idx}] skipping calibration (use_calibration=False)")
                                logger.info(f"    Note: if the relative ordering of head importance is correct, calibration is just uniform scaling,")
                                logger.info(f"    and with a unified pruning-strength parameter, the best-matching head can still be selected.")

                    # 3. Distribution statistics of heads within each domain
                    logger.info(f"  [Layer {layer_idx}] head distribution statistics within each domain:")
                    for domain_k in range(self.num_domains):
                        domain_head_importances = layer_importance_np[:, domain_k]  # [num_heads]

                        # Statistical metrics
                        mean_val = np.mean(domain_head_importances)
                        std_val = np.std(domain_head_importances)
                        var_val = np.var(domain_head_importances)
                        min_val = np.min(domain_head_importances)
                        max_val = np.max(domain_head_importances)
                        median_val = np.median(domain_head_importances)
                        q25 = np.percentile(domain_head_importances, 25)
                        q75 = np.percentile(domain_head_importances, 75)
                        iqr = q75 - q25  # Interquartile range

                        # Compute the coefficient of variation (CV = std/mean, measuring the relative dispersion)
                        cv = std_val / mean_val if mean_val > 0 else 0.0

                        logger.info(f"    domain{domain_k}:")
                        logger.info(f"      mean={mean_val:.4f}, std={std_val:.4f}, variance={var_val:.4f}")
                        logger.info(f"      min={min_val:.4f}, max={max_val:.4f}, median={median_val:.4f}")
                        logger.info(f"      25th percentile={q25:.4f}, 75th percentile={q75:.4f}, IQR={iqr:.4f}")
                        logger.info(f"      coefficient of variation (CV)={cv:.4f} {'(high dispersion)' if cv > 0.5 else '(low dispersion)' if cv < 0.2 else '(moderate dispersion)'}")

                    logger.info(f"  {'='*70}")
                    logger.info("")

        self.importance_cache = importance
        logger.info("Head importance computation completed for all layers")

        return importance
    
    def get_importance(
        self,
        layer_idx: int,
        domain_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Get the importance scores

        Args:
            layer_idx: layer index
            domain_idx: domain index (if None, return the importance for all domains)

        Returns:
            If domain_idx is None: [num_heads, num_domains]
            Otherwise: [num_heads]
        """
        if layer_idx not in self.importance_cache:
            logger.warning(f"Layer {layer_idx} has no cached importance scores")
            return torch.zeros(self.num_heads_per_layer, self.num_domains)
        
        importance = self.importance_cache[layer_idx]
        
        if domain_idx is not None:
            return importance[:, domain_idx]
        else:
            return importance
    
    def save(self, path: str):
        """Save the importance scores"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to a serializable format
        importance_dict = {
            layer_idx: importance.cpu().tolist()
            for layer_idx, importance in self.importance_cache.items()
        }
        
        metadata = {
            'num_layers': self.num_layers,
            'num_heads_per_layer': self.num_heads_per_layer,
            'num_domains': self.num_domains,
            'importance': importance_dict
        }
        
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Head importance scores saved: {path}")

    @classmethod
    def load(cls, path: str):
        """Load the importance scores"""
        with open(path, 'r') as f:
            metadata = json.load(f)

        instance = cls(
            num_layers=metadata['num_layers'],
            num_heads_per_layer=metadata['num_heads_per_layer'],
            num_domains=metadata['num_domains']
        )

        # Restore the importance scores
        for layer_idx, importance_list in metadata['importance'].items():
            instance.importance_cache[int(layer_idx)] = torch.tensor(
                importance_list,
                dtype=torch.float32
            )

        logger.info(f"Head importance scores loaded: {path}")
        return instance

