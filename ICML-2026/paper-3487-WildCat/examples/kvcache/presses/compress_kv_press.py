from collections.abc import Callable
import torch

from kvpress import BasePress
import re
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import types

from wildcat.compress_kv import compress as rp_compress

from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS, eager_attention_forward
import math

@dataclass
class CompressKV(BasePress):
    """
    Custom press that subsamples KV cache and applies random weighting vectors
    to the attention mechanism.
    """
    compression_ratio: float = 0.5 # Warning: this is the fraction of points NOT kept
    bin_r: int = 1 # number of KV pairs to keep per bin
    window_size: int = 32
    sink_size: int = 32
        
    def compress(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress KV cache and generate weighting vector.
        
        Args:
            module: The transformer attention layer
            hidden_states: Hidden states with shape (batch_size, seq_len, hidden_dim)
            keys: Key tensors with shape (batch_size, num_kv_heads, seq_len, head_dim)
            values: Value tensors with shape (batch_size, num_kv_heads, seq_len, head_dim)
            attentions: Attention weights (may be None)
            kwargs: Additional keyword arguments
            
        Returns:
            compressed_keys, compressed_values
        """
        window_size = self.window_size
        sink_size = self.sink_size
        layer_idx = module.layer_idx
        bin_r = self.bin_r

        B, H, N, E = keys.shape
        D = values.shape[-1]

        # Store maximum and minimum values
        vmax = torch.amax(values, dim=-2, keepdim=True)
        vmin = torch.amin(values, dim=-2, keepdim=True)

        # Recenter keys.
        kbar = keys.mean(dim = -2, keepdim=True)
        keys = keys - kbar

        # Register value statistics for attention reconstruction
        if not hasattr(module, f"vmax_layer_{layer_idx}"):
            module.register_buffer(f"vmax_layer_{layer_idx}", vmax)
        else:
            module._buffers[f"vmax_layer_{layer_idx}"] = vmax

        if not hasattr(module, f"vmin_layer_{layer_idx}"):
            module.register_buffer(f"vmin_layer_{layer_idx}", vmin)
        else:
            module._buffers[f"vmin_layer_{layer_idx}"] = vmin

        
        target_N_in = (N-sink_size-window_size)
        target_r = int(math.floor(target_N_in * (1-self.compression_ratio)))
        bins = target_r // bin_r
        bin_N = target_N_in // bins
        N_in = bins * bin_N
        window_size = window_size + target_N_in - N_in

        if False:
            # To gain insight about the effective compression ratio
            total_kept = sink_size + window_size + bins*bin_r
            print(f"frac_kept: {total_kept/N}", flush = True)

        # Compression routine
        sink_keys = keys[..., :sink_size, :]
        sink_values = values[..., :sink_size, :]
        sink_ones = torch.ones((*sink_values.shape[:-1], 1), device=sink_values.device, dtype=sink_values.dtype)

        window_keys = keys[..., N-window_size:, :]
        window_values = values[..., N-window_size:, :]
        window_ones = torch.ones((*window_values.shape[:-1], 1), device=window_values.device, dtype=window_values.dtype)

        keys = keys[..., sink_size: N-window_size, :]
        values = values[..., sink_size: N-window_size, :]

        # Reshape keys and values for binning
        keys = keys.reshape(B*H*bins, bin_N, E)
        values = values.reshape(B*H*bins, bin_N, D)


        compressed_keys, compressed_values, weight_vector = rp_compress(
                module,
                hidden_states,
                keys,
                values,
                attentions,
                r=bin_r,
                kwargs=kwargs
            )
        
        compressed_keys = compressed_keys.reshape(B, H, -1, E)
        compressed_values = compressed_values.reshape(B, H, -1, D)
        weight_vector = weight_vector.reshape(B, H, -1, 1)
        
            
        compressed_keys = torch.cat((sink_keys, compressed_keys, window_keys), dim=-2)
        # Add kbar back to retained keys
        compressed_keys = compressed_keys + kbar
        compressed_values = torch.cat((sink_values, compressed_values, window_values), dim=-2)
        weight_vector = torch.cat((sink_ones, weight_vector, window_ones), dim=-2)

        compressed_values = torch.cat((compressed_values, weight_vector), dim=-1)
        
        return compressed_keys, compressed_values


    def forward_hook(self, module, input, kwargs, output):
        """
        Default forward hook called after the forward pass of an attention layer.

        This hook automatically applies compression during the pre-filling phase by:
        0. Patching attention of module to enable weighting
        1. Checking if we're still in pre-filling (not generation) phase
        2. Extracting keys and values from the cache (handling quantization)
        3. Calling the compress method to reduce the cache size
        4. Updating the cache with compressed keys and values

        The hook ensures compression is only applied during pre-filling

        Parameters
        ----------
        module : nn.Module
            The transformer attention layer.
        input : list[torch.Tensor]
            Input tensors to the forward pass of the attention layer. This parameter
            is provided by PyTorch's hook mechanism but not used in the default implementation.
        kwargs : dict
            Keyword arguments passed to the attention layer's forward method, including:
            - hidden_states: Input embeddings to the attention layer
            - past_key_values: The KV cache object being modified
            - cache_position: Position indices indicating where we are in the sequence
            - position_embeddings: RoPE embeddings if applicable
        output : list
            Output from the attention layer's forward pass. Contains:
            - [0]: Hidden states output
            - [1]: Attention weights (may be None)

        Returns
        -------
        list
            The potentially modified output from the forward pass. This
            is the same as the input output, but the underlying cache has been compressed in-place.
        """

        excluded = [] 

        if module.layer_idx in excluded:
            print(f"➡️  Layer(s) {excluded} not compressed")
            return output

         # Patch attentions of model to enable weighting
        if not hasattr(module, "_original_forward"):
            module._original_forward = module.forward
            if getattr(module.config, "model_type", None) == "llama":
                raise NotImplementedError("Llama model not supported yet")
            elif getattr(module.config, "model_type", None) == "qwen2":
                module.forward = types.MethodType(custom_attention_forward, module)

        # Collect kwargs from initial attention call
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        q_len = hidden_states.shape[1]

        # Don't compress after prefill
        if kwargs["cache_position"][-1] > q_len:
            return output
        
        # Collect keys and values from cache
        layer_idx = module.layer_idx
        cache_layer = cache.layers[layer_idx]
        keys = cache_layer.keys
        values = cache_layer.values

        # Compress keys and values
        keys, values = self.compress(
            module, hidden_states, keys, values, output[1], kwargs
        )

        # Write back to cache
        cache_layer.keys = keys
        cache_layer.values = values

        return output
    


def custom_attention_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values,
        cache_position,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if cache_position[-1] > query_states.shape[-2]:
            layer_idx = self.layer_idx
            # Update running max and min for value_states
            self._buffers[f"vmax_layer_{layer_idx}"] = torch.maximum(self._buffers[f"vmax_layer_{layer_idx}"], 
                                                                     torch.amax(value_states, dim=-2, keepdim=True))
            self._buffers[f"vmin_layer_{layer_idx}"] = torch.minimum(self._buffers[f"vmin_layer_{layer_idx}"], 
                                                                     torch.amin(value_states, dim=-2, keepdim=True))
            ###self._buffers[f"vbar_layer_{layer_idx}"] = (self._buffers[f"vmax_layer_{layer_idx}"] + self._buffers[f"vmin_layer_{layer_idx}"])/2

        values_shape = value_states.shape
        if cache_position[-1] > query_states.shape[-2]:
            # After prefilling, concat ones vector to values for weighting
            ones_vector = torch.ones(
                (*values_shape[:-1], 1),
                device=value_states.device,
                dtype=value_states.dtype
            )
            value_states = torch.cat((value_states, ones_vector), dim=-1)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]


        # Adjust attention mask if sequence length has changed due to compression
        cur_seq_len = key_states.shape[-2]
        if attention_mask is not None and attention_mask.shape[-1] != cur_seq_len:
            attention_mask = attention_mask[..., -cur_seq_len:, :]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # main diff with Llama
            **kwargs,
        )

        # TODO: Move magic number to config
        
        if value_states.shape[-1] == self.head_dim + 1:
            # During generation, we added an extra dimension to values for weighting
            # We need to remove this dimension from the output
            # attn_output shape: (batch_size, num_heads, seq_len, head_dim + 1)
            # We divide the output by the last dimension (the weights) to get the final output
            eps = 1e-20
            attn_output = torch.where(attn_output[..., -1:]>eps, attn_output[..., :-1] / attn_output[..., -1:], 0.)
        
        vmax = self._buffers.get(f"vmax_layer_{self.layer_idx}", None).transpose(1, 2)
        vmin = self._buffers.get(f"vmin_layer_{self.layer_idx}", None).transpose(1, 2)
        ###vbar = self._buffers.get(f"vbar_layer_{self.layer_idx}", None).transpose(1, 2)

        if vmax is not None and vmin is not None:
            # Clamp attention output to the running max and min of values
            num_attention_heads = attn_output.shape[2]
            num_kv_heads = key_states.shape[1]
            vmax = vmax.repeat_interleave(num_attention_heads//num_kv_heads, dim=2).expand(attn_output.shape)
            vmin = vmin.repeat_interleave(num_attention_heads//num_kv_heads, dim=2).expand(attn_output.shape)
            ###vbar = vbar.repeat_interleave(num_attention_heads//num_kv_heads, dim=2).expand(attn_output.shape)
            ###attn_output = (attn_output + vbar).clamp(min=vmin, max=vmax) # Disabling vbar centering
            attn_output = (attn_output).clamp(min=vmin, max=vmax)
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
