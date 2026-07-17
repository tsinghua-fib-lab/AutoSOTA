"""Adapter to make HuggingFace Qwen3 compatible with the quantization pipeline.

This module provides a wrapper that makes HuggingFace Qwen3 models expose
the same interface as the internal Llama implementation, allowing the
quantization pipeline to work seamlessly with Qwen3 models.

Weight Mapping:
    HuggingFace Qwen3          -> Internal Llama Format
    -------------------------------------------------
    self_attn.q_proj           -> attention.wq
    self_attn.k_proj           -> attention.wk
    self_attn.v_proj           -> attention.wv
    self_attn.o_proj           -> attention.wo
    mlp.gate_proj              -> feed_forward.w1
    mlp.up_proj                -> feed_forward.w3
    mlp.down_proj              -> feed_forward.w2
    input_layernorm            -> attention_norm
    post_attention_layernorm   -> ffn_norm
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class Qwen3Args:
    """Qwen3 model arguments - mirrors ModelArgs for Llama."""
    dim: int = 4096
    n_layers: int = 36
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = 151936
    intermediate_size: int = 12288
    norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_batch_size: int = 32
    max_seq_len: int = 2048
    use_scaled_rope: bool = False


class _AttentionAdapter(nn.Module):
    """Maps HuggingFace attention naming to Llama naming.

    Provides submodule aliases that appear in named_modules():
        wq -> q_proj
        wk -> k_proj
        wv -> v_proj
        wo -> o_proj
    """

    def __init__(self, hf_attn, n_heads: int, n_kv_heads: int, head_dim: int, max_batch_size: int, max_seq_len: int):
        super().__init__()
        self.n_local_heads = n_heads
        self.n_local_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads

        # Register as actual submodules so they appear in named_modules()
        # This is critical for the quantization pipeline which uses dict(model.named_modules())
        self.wq = hf_attn.q_proj
        self.wk = hf_attn.k_proj
        self.wv = hf_attn.v_proj
        self.wo = hf_attn.o_proj

        # Initialize KV caches
        cache_shape = (max_batch_size, max_seq_len, n_kv_heads, head_dim)
        self.cache_k = torch.zeros(cache_shape, dtype=torch.bfloat16).cuda()
        self.cache_v = torch.zeros(cache_shape, dtype=torch.bfloat16).cuda()


class _FFNAdapter(nn.Module):
    """Maps HuggingFace MLP naming to Llama naming.

    In Qwen3/Llama SwiGLU:
        gate_proj = w1 (input to silu)
        up_proj = w3 (multiplied with silu output)
        down_proj = w2 (final projection)

    Formula: w2(silu(w1(x)) * w3(x))
    """

    def __init__(self, hf_mlp):
        super().__init__()
        # Register as actual submodules so they appear in named_modules()
        self.w1 = hf_mlp.gate_proj
        self.w2 = hf_mlp.down_proj
        self.w3 = hf_mlp.up_proj


class Qwen3LayerAdapter(nn.Module):
    """Wraps a HuggingFace Qwen3DecoderLayer to expose Llama-like interface.

    Provides:
        .attention.{wq, wk, wv, wo}
        .feed_forward.{w1, w2, w3}
        .attention_norm
        .ffn_norm
    """

    def __init__(self, hf_layer, n_heads: int, n_kv_heads: int, head_dim: int, max_batch_size: int, max_seq_len: int, rotary_emb=None):
        super().__init__()
        # Store HF layer without registering as submodule to avoid named_modules() conflicts
        object.__setattr__(self, '_hf_layer', hf_layer)

        # Store rotary embedding module for computing position embeddings
        # (passed from the main model, needed for layer forward pass)
        object.__setattr__(self, '_rotary_emb', rotary_emb)

        # Create adapters that expose Llama-like naming
        self.attention = _AttentionAdapter(
            hf_layer.self_attn,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )
        self.feed_forward = _FFNAdapter(hf_layer.mlp)

        # Norms are accessed directly (same interface)
        self.attention_norm = hf_layer.input_layernorm
        self.ffn_norm = hf_layer.post_attention_layernorm

    def forward(self, x, start_pos=0, freqs_cis=None, mask=None):
        """Forward pass compatible with Llama interface.

        Args:
            x: Input hidden states, shape (batch, seq_len, dim)
            start_pos: Starting position for RoPE
            freqs_cis: RoPE frequencies (ignored, we compute our own)
            mask: Attention mask (ignored, HF computes its own causal mask)

        Returns:
            Output hidden states, shape (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape

        # Create position IDs (starting from start_pos)
        position_ids = torch.arange(
            start_pos, start_pos + seq_len, device=x.device
        ).unsqueeze(0).expand(batch_size, -1)

        # Compute position embeddings (RoPE cos/sin) using the rotary embedding module
        # HF Qwen3 expects these to be passed to the layer
        position_embeddings = self._rotary_emb(x, position_ids)

        # Create causal attention mask for HF format
        # HF expects 4D mask: (batch, 1, seq_len, seq_len)
        if seq_len > 1:
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1
            )
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        else:
            attention_mask = None

        # Call HuggingFace layer
        outputs = self._hf_layer(
            hidden_states=x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            position_embeddings=position_embeddings,
        )

        # HF returns tuple: (hidden_states, ...) or just hidden_states
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs


class Qwen3ToLlamaAdapter(nn.Module):
    """Wraps a HuggingFace Qwen3 model to provide Llama-compatible interface.

    This adapter allows the quantization pipeline to work with Qwen3 models
    by exposing the same attribute structure as the internal Llama implementation:

        model.layers[i].attention.{wq, wk, wv, wo}
        model.layers[i].feed_forward.{w1, w2, w3}
        model.tok_embeddings
        model.norm
        model.output

    The named_modules() will return paths like:
        layers.0.attention.wq  (actually pointing to self_attn.q_proj)
        layers.0.feed_forward.w1  (actually pointing to mlp.gate_proj)
    """

    def __init__(self, hf_model, override_params: Optional[Dict[str, Any]] = None):
        super().__init__()
        # Store HF model without registering as submodule to avoid named_modules() conflicts
        # (PyTorch deduplicates modules, and we need our adapter paths to take precedence)
        object.__setattr__(self, '_hf_model', hf_model)
        self.config = hf_model.config

        # Build params (equivalent to ModelArgs)
        max_seq_len = 2048
        max_batch_size = 32
        if override_params:
            max_seq_len = override_params.get("max_seq_len", max_seq_len)
            max_batch_size = override_params.get("max_batch_size", max_batch_size)

        self.params = Qwen3Args(
            dim=self.config.hidden_size,
            n_layers=self.config.num_hidden_layers,
            n_heads=self.config.num_attention_heads,
            n_kv_heads=getattr(self.config, 'num_key_value_heads', self.config.num_attention_heads),
            vocab_size=self.config.vocab_size,
            intermediate_size=self.config.intermediate_size,
            norm_eps=self.config.rms_norm_eps,
            rope_theta=getattr(self.config, 'rope_theta', 1000000.0),
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )

        # Compute head dimension
        head_dim = self.params.dim // self.params.n_heads

        # Get the rotary embedding module from the model
        # In HF Qwen3, it's at model.model.rotary_emb
        rotary_emb = hf_model.model.rotary_emb

        # Expose layers with Llama-like interface
        self.n_layers = self.params.n_layers
        self.layers = nn.ModuleList([
            Qwen3LayerAdapter(
                layer,
                n_heads=self.params.n_heads,
                n_kv_heads=self.params.n_kv_heads,
                head_dim=head_dim,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                rotary_emb=rotary_emb,
            )
            for layer in hf_model.model.layers
        ])

        # Other components - expose with Llama naming
        self.tok_embeddings = hf_model.model.embed_tokens
        self.norm = hf_model.model.norm
        self.output = hf_model.lm_head

        # Store vocab size for compatibility
        self.vocab_size = self.params.vocab_size

        # Dummy freqs_cis for compatibility with ActivationCache
        # HuggingFace handles RoPE internally, so we just need a placeholder
        # that can be sliced and passed to layer.forward() (which ignores it)
        self.freqs_cis = torch.zeros(max_seq_len * 2, head_dim // 2, dtype=torch.complex64)

    def resize_kv_caches(self, new_batch_size: int):
        """Resize KV caches for batched computation."""
        for layer in self.layers:
            attn = layer.attention
            old_shape = attn.cache_k.shape
            new_shape = (new_batch_size, old_shape[1], old_shape[2], old_shape[3])
            attn.cache_k = torch.zeros(new_shape, device=attn.cache_k.device, dtype=attn.cache_k.dtype)
            attn.cache_v = torch.zeros(new_shape, device=attn.cache_v.device, dtype=attn.cache_v.dtype)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """Forward pass compatible with Llama interface.

        Args:
            tokens: Input token indices, shape (batch, seq_len)
            start_pos: Starting position for KV cache (used in generation)

        Returns:
            Logits tensor, shape (batch, seq_len, vocab_size)

        Note:
            For quantization, we mainly need this for:
            1. Computing Hessians (activations at each layer)
            2. Evaluation (perplexity computation)
        """
        # Use HuggingFace forward
        # Note: start_pos is ignored here since we use full attention
        # For generation with KV cache, you'd need to handle this differently
        outputs = self._hf_model(
            input_ids=tokens,
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits

    def get_input_embeddings(self):
        """Get token embeddings layer."""
        return self.tok_embeddings

    def get_output_embeddings(self):
        """Get output projection layer."""
        return self.output


class Qwen3TokenizerAdapter:
    """Adapter to make HuggingFace tokenizer compatible with Llama tokenizer interface.

    Provides:
        .encode(text, bos=True, eos=False) -> list[int]
        .decode(tokens) -> str
        .n_words -> int (vocab size)
        .bos_id -> int
        .eos_id -> int
    """

    def __init__(self, hf_tokenizer):
        self._hf_tokenizer = hf_tokenizer
        self.n_words = hf_tokenizer.vocab_size
        self.bos_id = hf_tokenizer.bos_token_id
        self.eos_id = hf_tokenizer.eos_token_id
        # Pad token - use eos if not set
        self.pad_id = hf_tokenizer.pad_token_id if hf_tokenizer.pad_token_id is not None else self.eos_id

    def encode(self, text: str, bos: bool = True, eos: bool = False) -> list:
        """Encode text to token IDs.

        Args:
            text: Input text
            bos: Whether to prepend BOS token
            eos: Whether to append EOS token

        Returns:
            List of token IDs
        """
        tokens = self._hf_tokenizer.encode(text, add_special_tokens=False)
        if bos and self.bos_id is not None:
            tokens = [self.bos_id] + tokens
        if eos and self.eos_id is not None:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, tokens) -> str:
        """Decode token IDs to text."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self._hf_tokenizer.decode(tokens)

    def __call__(self, text, **kwargs):
        """Direct call to underlying tokenizer."""
        return self._hf_tokenizer(text, **kwargs)
