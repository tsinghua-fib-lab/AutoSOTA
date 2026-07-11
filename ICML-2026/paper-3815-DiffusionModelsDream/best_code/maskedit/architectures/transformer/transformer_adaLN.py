from .attention import MultiHeadAttention
from .utils import variance_scaling_for_layers

from typing import Callable, Optional

import jax
import jax.numpy as jnp

from flax import nnx

Array = jax.Array

class Transformer(nnx.Module):
    """A transformer stack (attention + MLP) with adaLN-Zero context injection."""

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
            # Layer norms now have use_scale=False and use_bias=False
            # The scale/bias (gamma/beta) will be computed from the context.
            ln1 = nnx.LayerNorm(num_features=model_size, use_scale=False, use_bias=False, rngs=rngs)
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
            ln2 = nnx.LayerNorm(num_features=model_size, use_scale=False, use_bias=False, rngs=rngs)

            mlp_layers = []
            for _h in range(self.num_hidden_layers):
                mlp_layers.append(
                    nnx.Linear(model_size, self.widening_factor * model_size, kernel_init=self.w_init, bias_init=self.b_init, rngs=rngs)
                )
                mlp_layers.append(self.act)
            mlp_layers.append(
                nnx.Linear(self.widening_factor * model_size if self.num_hidden_layers > 0 else model_size, model_size, kernel_init=self.w_init, bias_init=self.b_init, rngs=rngs)
            )
            
            # Replacement for ctx_proj
            cond_mlp_layers = []
            cond_mlp_layers.append(nnx.Linear(context_size, model_size, kernel_init=self.w_init, bias_init=self.b_init, rngs=rngs))
            cond_mlp_layers.append(self.act)
            # The final layer is initialized to zeros as per adaLN paper
            zero_init = nnx.initializers.zeros_init()
            cond_mlp_layers.append(nnx.Linear(model_size, 6 * model_size, kernel_init=zero_init, bias_init=zero_init, rngs=rngs))
            
            layers.append((ln1, attn, ln2, mlp_layers, cond_mlp_layers))
        self.layers = layers
        self.out_ln = nnx.LayerNorm(num_features=model_size, rngs=rngs)
        self.built = True

    def __call__(self, inputs: Array, context: Optional[Array] = None, mask: Array | None = None) -> Array:
        """Applies the transformer to an input sequence.

        Args:
            inputs: [B, T, D]
                Token embeddings for each sequence element.
            context: [B, C], [B, 1, C], or [B, T, C]
                Conditioning features, adaLN-Zero.
            mask: Optional attention mask of shape
                [T, T], [B, T, T], [B, 1, T, T], or [B, H, T, T].

        Returns:
            Array [B, T, D]:
                The transformed sequence.
        """

        h = inputs
        for (ln1, attn, ln2, mlp_layers, cond_mlp_layers) in self.layers: # --- CHANGED --- (renamed var)

            # Process context first to get adaLN parameters
            if context is not None:
                ctx = context
                if ctx.ndim == 2:
                    ctx = ctx[:, None, :]
                
                cond_out = ctx
                for layer in cond_mlp_layers:
                    cond_out = layer(cond_out)

                gamma1, beta1, alpha1, gamma2, beta2, alpha2 = jnp.split(cond_out, 6, axis=-1)
                
                # Add 1.0 to gammas for identity initialization
                gamma1 = gamma1 + 1.0
                gamma2 = gamma2 + 1.0
            else:
                # If no context, default to identity
                gamma1, gamma2 = 1.0, 1.0
                beta1, alpha1, beta2, alpha2 = 0.0, 0.0, 0.0, 0.0
        
            # Attention block
            h_norm = ln1(h) * gamma1 + beta1 
            h_attn = attn(h_norm, mask=mask)
            if self.dropout_rate > 0:
                h_attn = self.dropout(h_attn)
            if self.skip_connection_attn:
                h = h + h_attn * alpha1
            else:
                h = h_attn * alpha1
        
            # MLP block
            h_norm = ln2(h) * gamma2 + beta2
            z = h_norm
            for layer in mlp_layers:
                z = layer(z)
            if self.dropout_rate > 0:
                z = self.dropout(z)

            if self.skip_connection_mlp:
                h = h + z * alpha2
            else:
                h = z * alpha2
        
        out = self.out_ln(h)
        return out