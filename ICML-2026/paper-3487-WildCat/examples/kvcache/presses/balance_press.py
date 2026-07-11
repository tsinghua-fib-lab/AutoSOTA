import torch
from collections.abc import Callable

from kvpress import BasePress

import math
from typing import List, Optional, Tuple, Union, Any, Dict
import torch

import types

from dataclasses import dataclass

from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS, eager_attention_forward
from transformers.cache_utils import DynamicCache

@dataclass
class BalanceKVPress(BasePress):
    rng: torch.Generator = torch.Generator('cuda')
    gamma: float = 4.
    temp: float = 1.
    beta: float = 0.
    itrs: int = 2 
    block_size: int = 256
    window_size: int = 32
    sink_size: int = 32  

    # Warning: currently not used: itrs determines the compression ratio
    compression_ratio = 0.5
    # def __post_init__(self):
    #     """Initialize after dataclass creation."""
    #     pass # Store weights per layer


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
        gamma = self.gamma
        temp = self.temp
        beta = self.beta
        itrs = self.itrs
        block_size = self.block_size
        window_size = self.window_size
        sink_size = self.sink_size

        k_compressed = keys[..., sink_size:-window_size, :]
        v_compressed = values[..., sink_size:-window_size, :]


        indices, weights = balanced_walk2(
                k_compressed, rng, gamma, temp, beta, itrs, block_size, value=v_compressed)

        if indices is not None:
            k_bw = k_compressed.gather(
                dim=2, index=indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1]))
            v_bw = v_compressed.gather(
                dim=2, index=indices.unsqueeze(-1).expand(-1, -1, -1, values.shape[-1]))

            if weights is not None:
                weights = weights / 2**(itrs)
                weights = weights.unsqueeze(-1)
                v_bw = (v_bw * weights).to(values.dtype)

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


def balanced_walk2(key, rng, gamma_, temp_, beta_, itrs, block_size, value=None, sort_idx=None, query=None):
    if itrs == 0:
        return sort_idx, None
    
    b, h, n, d = key.shape
    if type(gamma_) != list:
        gamma_ = [gamma_] * itrs
    const_denom = 0.025 # change this to 0.00 to change the kernel back

    if type(block_size) != list:
        block_size = [block_size] * itrs
    weight_idx = None
    for t in range(itrs): #write range(1, itrs) to check everything still works
        if sort_idx is not None:
            key_sorted, value_sorted = indexing(key, sort_idx, block_size[t], value)
            key_sorted = key_sorted.view(b, h, -1, block_size[t], d)
            if value is not None:
                weight_idx_padded = torch.nn.functional.pad(weight_idx, (0, math.ceil(n / block_size[t]) * block_size[t] - weight_idx.shape[-1]))
                value_sorted = value_sorted*weight_idx_padded.unsqueeze(-1)
                value_sorted = value_sorted.view(b, h, -1, block_size[t], d)
        else:
            new_n = math.ceil(n / block_size[t]) * block_size[t]
            key_sorted = torch.nn.functional.pad(key, (0,0,0,new_n-n), mode='constant', value=0.).view(b, h, -1, block_size[t], d)
            value_sorted = None
            if value is not None:
                value_sorted = torch.nn.functional.pad(value, (0,0,0,new_n-n), mode='constant', value=0.).view(b, h, -1, block_size[t], d)

        normal_keys = key_sorted - torch.mean(key_sorted, dim=-2, keepdim=True)

        if query is not None:
            query_key_correlation = torch.softmax(torch.einsum('b h n d,b h s m d->b h s n m',query[:,::4,:,:],normal_keys),dim=-1).mean(-2,keepdim=True)
            kernel_ = query_key_correlation*query_key_correlation.transpose(-1,-2)
        else:
            kernel_ = torch.exp(temp_ * torch.einsum('...nd,...sd->...ns', normal_keys, normal_keys)/math.sqrt(d) - beta_)
        if value is not None:
            kernel_ *= (1e-8 + torch.einsum('...nd,...sd->...ns', value_sorted, value_sorted)+const_denom)

        signs = torch.zeros(kernel_.shape[:4], dtype=torch.int16, device=kernel_.device)
        signs[:, :, :, 0] = 1
        rand_tensor = torch.rand(signs.shape, generator=rng, device=key.device)

        for i in range(1, kernel_.shape[3]): 
            partial_inner_prod = (kernel_[:, :, :, i, :] * signs).sum(dim=-1) 
            samp_prb = 0.5 - gamma_[t] * partial_inner_prod
            signs[:, :, :, i] = 2 * (rand_tensor[:, :, :, i] < samp_prb) - 1

        signs = signs.view(b, h, -1)[:, :, :n]

        if signs.shape[-1]==0: # simply to deal with n==0
            sort_idx = signs[:, :, :0]
            weigth_idx = signs[:, :, :0]
            break
        cumsum_neg = (signs == -1).cumsum(dim=-1)
        cumsum_pos = (signs == 1).cumsum(dim=-1)

        c_neg = torch.argmax((cumsum_neg == n//2).to(torch.int64), dim=-1) # Shape (b, h)
        c_pos = torch.argmax((cumsum_pos == n//2).to(torch.int64), dim=-1) # Shape (b, h)
        c = torch.maximum(c_neg, c_pos)

        c = c.to(signs.device)

        weight = signs

        # Create an index tensor `[0, 1, ..., n-1]` for comparison
        indices = torch.arange(signs.shape[2], device=signs.device).view(1, 1, -1)
        # Set all values after `c[a, b]` to `1`
        mask_after_c = indices > c.unsqueeze(-1)  # True for all d > c[a, b]
        weight[mask_after_c] = torch.abs(weight[mask_after_c])  # Set those indices to `1`
        # Identify where `signs[a, b, c[a, b]] == 1`
        mask_flip_needed = (signs.gather(2, c.unsqueeze(-1)) == 1).squeeze(-1)
        # Create mask for all indices `<= c[a, b]`
        mask_before_c = indices <= c.unsqueeze(-1)
        weight[mask_before_c] *= 2
        # Apply flipping only when `signs[a, b, c] == 1`
        flip_mask = mask_before_c & mask_flip_needed.unsqueeze(-1)
        weight[flip_mask] *= -1  # Flip selected values

        

        weight_argsort = torch.argsort(-weight, dim=-1, stable=True)

        n = n//2
        if sort_idx is None:
            sort_idx = weight_argsort[:, :, :n]
            weight_idx = weight.gather(-1, weight_argsort[:, :, :n])
        else:
            sort_idx = sort_idx.gather(2, weight_argsort[:, :, :n])
            weigth_idx_1 = weight.gather(-1, weight_argsort[:, :, :n])
            weight_idx = weight_idx.gather(-1, weight_argsort[:, :, :n])
            weight_idx = weight_idx*weigth_idx_1

    return sort_idx, weight_idx



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



if __name__ == "__main__":
    from transformers import pipeline

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    device = "cuda:1"


    pipe = pipeline("kv-press-text-generation", model=model_name, device=device)

    try:
        with open("example_txt.txt", "r", encoding="utf-8") as f:
            context = f.read()
        with open("example_qstn.txt", "r", encoding="utf-8") as f:
            question = f.read()
    except FileNotFoundError:
        context = """Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It has applications in many fields including computer vision, natural language processing, and robotics."""
        question = "What is machine learning and what are its applications?"



    press = BalanceKV()

    # Pass cache here
    cache = DynamicCache()
    answer = pipe(context, question=question, press=press, cache = cache)["answer"]
    print("Answer:", answer)