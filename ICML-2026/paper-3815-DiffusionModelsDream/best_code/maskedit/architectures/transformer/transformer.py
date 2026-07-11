from .attention import MultiHeadAttention
from .utils import variance_scaling_for_layers


#from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp

from flax import nnx

Array = jax.Array

class Transformer(nnx.Module):
    """A transformer stack (attention + MLP) with optional context injection."""

    def __init__(
        self,
        num_heads: int,
        num_layers: int,
        attn_size: int,
        C: int,
        D: int,
        dropout_rate: Optional[float] = None,
        widening_factor: int = 4,
        num_hidden_layers: int = 1,
        act: Callable[[Array], Array] = jax.nn.gelu,
        skip_connection_attn: bool = True,
        skip_connection_mlp: bool = True,
        initializer: Optional[nnx.Initializer] = None,
        save_attention_weights: bool = False,
        attention_method: str = "dense",
        *,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn_size = attn_size
        self.dropout_rate = 0.0 if (dropout_rate is None) else float(dropout_rate)
        self.widening_factor = widening_factor
        self.num_hidden_layers = num_hidden_layers
        self.act = act
        self.skip_connection_attn = skip_connection_attn
        self.skip_connection_mlp = skip_connection_mlp
        self.attention_method = attention_method

        if initializer is None:
            initializer = variance_scaling_for_layers(num_layers)
        self.w_init = initializer
        self.b_init = nnx.initializers.zeros_init() #Initalize biases as zeros

        # These will be initialized lazily
        self.layers = []
        self.built = False
        self.dropout = nnx.Dropout(rate=self.dropout_rate, rngs=rngs)
        self.save_attention_weights = save_attention_weights
        self._rngs = rngs  # stash for building submodules
        self.build_layers(C,D)

    def build_layers(self, context_size: int, model_size: int):
        if self.built == True:
            return
        rngs = self._rngs
        layers = []
        for _ in range(self.num_layers):
            ln1 = nnx.LayerNorm(num_features=model_size, rngs=rngs)
            attn = MultiHeadAttention(
                model_size=model_size,
                num_heads=self.num_heads,
                key_size=self.attn_size,
                w_init=self.w_init,
                b_init=self.b_init,
                with_bias=True,
                attention_method=self.attention_method,
                save_attention_weights=self.save_attention_weights,
                rngs=rngs,
            )
            ln2 = nnx.LayerNorm(num_features=model_size, rngs=rngs)

            # Build MLP block: [model -> widen -> ... -> model]
            mlp_layers = []
            for _h in range(self.num_hidden_layers):
                mlp_layers.append(
                    nnx.Linear(model_size, self.widening_factor * model_size, kernel_init=self.w_init, bias_init=self.b_init, rngs=rngs)
                )
                mlp_layers.append(self.act)
            mlp_layers.append(
                nnx.Linear(self.widening_factor * model_size if self.num_hidden_layers > 0 else model_size, model_size, kernel_init=self.w_init, bias_init=self.b_init, rngs=rngs)
            )

            ctx_proj = []
            ctx_proj.append(nnx.Linear(context_size, model_size, kernel_init=self.w_init, bias_init=self.b_init, rngs=rngs))
            ctx_proj.append(self.act)

            layers.append((ln1, attn, ln2, mlp_layers, ctx_proj))
        self.layers = layers
        self.out_ln = nnx.LayerNorm(num_features=model_size, rngs=rngs)
        self.built = True

    def __call__(self, inputs: Array, context: Optional[Array] = None, mask: Array | None = None) -> Array:
        """Applies the transformer to an input sequence.

        Args:
            inputs: [B, T, D]
                Token embeddings for each sequence element.
            context: [B, C], [B, 1, C], or [B, T, C]
                Conditioning features.
            mask: Optional attention mask of shape
                [T, T], [B, T, T], [B, 1, T, T], or [B, H, T, T].

        Returns:
            Array [B, T, D]:
                The transformed sequence.
        """

        h = inputs
        for (ln1, attn, ln2, mlp_layers, ctx_proj) in self.layers:
            # Attention block
            h_norm = ln1(h)
            h_attn = attn(h_norm, mask=mask)
            if self.dropout_rate > 0:
                h_attn = self.dropout(h_attn)
            h = h + h_attn if self.skip_connection_attn else h_attn

            # MLP block
            h_norm = ln2(h)
            z = h_norm
            for layer in mlp_layers:
                z = layer(z)
            if self.dropout_rate > 0:
                z = self.dropout(z)

            if context is not None:

                ctx = context
                if ctx.ndim == 2:
                    ctx = ctx[:, None, :]
                # Project to model size
                ctx_z = ctx_proj[0](ctx)
                # Activation
                ctx_z = ctx_proj[1](ctx_z)
                z = z + ctx_z

            h = h + z if self.skip_connection_mlp else z

        out = self.out_ln(h)
        return out