import torch
from collections.abc import Callable

from kvpress import BasePress

import math
from typing import List, Optional, Tuple, Union, Any, Dict
import torch

import types

from dataclasses import dataclass

from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS, eager_attention_forward

@dataclass
class UniformPress(BasePress):
    """
    Uniform cache compression: after preserving sink and window tokens, uniformly subsample the remaining tokens.
    """
    rng: torch.Generator = torch.Generator('cuda')
    itrs: int = 2 
    block_size: int = 256
    window_size: int = 32
    sink_size: int = 32  

    # Warning: currently not used: itrs determines the compression ratio
    compression_ratio = 0.5

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
        rng = self.rng
        itrs = self.itrs
        block_size = self.block_size
        window_size = self.window_size
        sink_size = self.sink_size
      
        k_compressed = keys[..., sink_size:-window_size, :]
        v_compressed = values[..., sink_size:-window_size, :]

        indices = uniform_sampling(k_compressed, rng, itrs, block_size)

        if indices is None:
            return k_compressed, v_compressed

        k_bw = k_compressed.gather(
            dim=2, index=indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1]))
        v_bw = v_compressed.gather(
            dim=2, index=indices.unsqueeze(-1).expand(-1, -1, -1, values.shape[-1]))

        keys = torch.cat(
            (keys[..., :sink_size, :], k_bw, keys[..., -window_size:, :]), dim=2)
        values = torch.cat(
            (values[..., :sink_size, :], v_bw, values[..., -window_size:, :]), dim=2)

        return keys, values
    
    def forward_hook(self, module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Default forward hook called after the forward pass of an attention layer.

        This hook automatically applies compression during the pre-filling phase by:
        1. Checking if we're still in pre-filling (not generation) phase
        2. Extracting keys and values from the cache (handling quantization)
        3. Calling the compress method to reduce the cache size
        4. Updating the cache with compressed keys and values

        The hook ensures compression is only applied during pre-filling and correctly
        handles both quantized and unquantized caches.

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

        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        q_len = hidden_states.shape[1]

        if not hasattr(module, "_original_forward"):
            #print("patching", module.layer_idx)
            module._original_forward = module.forward
            module.forward = types.MethodType(custom_attention_forward, module)

        # Don't compress after pre-filling
        if kwargs["cache_position"][-1] > q_len:
            return output

        cache_layer = cache.layers[module.layer_idx]
        
        keys = cache_layer.keys
        values = cache_layer.values

        keys, values = self.compress(module, hidden_states, keys, values, output[1], kwargs)
        
        cache_layer.keys = keys
        cache_layer.values = values

        return output

def indexing(key, sort_idx, block_size, value=None):
    indices = sort_idx.unsqueeze(-1).expand(-1, -1, -1, key.shape[-1])
    new_n = math.ceil(sort_idx.shape[-1] / block_size) * block_size
    if new_n < sort_idx.shape[-1]:
        import pdb; pdb.set_trace();
    out_key = torch.nn.functional.pad(key.gather(2, indices), (0,0,0,new_n-sort_idx.shape[-1]), mode='constant', value=0.)
    out_value = None
    if value is not None:
        out_value = torch.nn.functional.pad(value.gather(2, indices), (0,0,0,new_n-sort_idx.shape[-1]), mode='constant', value=0.)
    return out_key, out_value

def uniform_sampling(key, rng, itrs, block_size, sort_idx=None):
    if itrs == 0:
        return sort_idx
    b, h, n, d = key.shape

    if type(block_size) != list:
        block_size = [block_size] * itrs

    for t in range(itrs):
        new_n = math.ceil(n / block_size[t]) * block_size[t]
        if sort_idx is not None:
            key_sorted, _ = indexing(key, sort_idx, block_size[t])
            key_sorted = key_sorted.view(b, h, -1, block_size[t], d)
        else:
            key_sorted = torch.nn.functional.pad(key, (0,0,0,new_n-n), mode='constant', value=0.).view(b, h, -1, block_size[t], d)
        
        kernel_ = torch.exp(torch.einsum('...nd,...sd->...ns', key_sorted, key_sorted)/math.sqrt(d) )
        signs = torch.zeros(kernel_.shape[:4], dtype=torch.int16, device=key.device)
        signs[:, :, :, 0] = 1
        rand_tensor = torch.rand(signs.shape, generator=rng, device=key.device)
        for i in range(1, kernel_.shape[3]):
            samp_prb = 0.5
            signs[:, :, :, i] = 2 * (rand_tensor[:, :, :, i] < samp_prb) - 1
        
        signs = signs.view(b, h, -1)[:, :, :n]
        signs_argsort = torch.argsort(signs, dim=-1, stable=True)
        n = n//2
        if sort_idx is None:
            sort_idx = signs_argsort[:, :, :n]
        else:
            sort_idx = sort_idx.gather(2, signs_argsort[:, :, :n])
    return sort_idx

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

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
            
        cur_seq_len = key_states.shape[-2]
        if attention_mask is not None and attention_mask.shape[-1] != cur_seq_len:
            # Adjust attention mask if sequence length has changed due to compression
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

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
