from .embedding import GaussianFourierEmbedding
from ..transformer.transformer import Transformer

import jax
import jax.numpy as jnp

from flax import nnx

Array = jax.Array

class Simformer(nnx.Module):
    """Simformer architecture with masking enabled.

    Inputs:
        t: diffusion time
        x: continuous inputs
        topo_mask: boolean mask of topology
        node_ids: integer node identifiers
        condition_mask: boolean mask for conditioning

    Returns:
        score: diffusion model score
    """


    def __init__(
        self,
        diff_eq,
        *,
        num_nodes: int,
        dim_id: int,
        dim_value: int,
        dim_condition: int,
        nlayers: int,
        edge2d: Array,
        rngs: nnx.Rngs,
        num_heads: int,
        time_dim: int,
        dropout_rate,
    ):
        self.edge2d = nnx.BatchStat(edge2d)
        self.diff = diff_eq

        self.dim_id = dim_id
        self.dim_value = dim_value
        self.dim_condition = dim_condition
        self.nlayers = nlayers
        self.model_size = dim_value + dim_id + dim_condition

        self.value_proj = nnx.Linear(1, self.dim_value, rngs=rngs)
        self.time_embed = GaussianFourierEmbedding(time_dim, rngs=rngs)
        self.condition_embedding = nnx.Param(
            jax.random.normal(rngs(), (1, 1, dim_condition)) * 0.5
        )
        self.id_embed = nnx.Embed(num_embeddings=num_nodes, features=dim_id, rngs=rngs)

        self.transformer = Transformer(
            num_heads=num_heads,
            num_layers=nlayers,
            attn_size=(dim_id+dim_value+dim_condition)//2,
            C = time_dim,
            D = dim_id+dim_value+dim_condition,
            dropout_rate=dropout_rate,
            widening_factor=4,#3,
            num_hidden_layers=1,
            act=jax.nn.gelu,
            skip_connection_attn=True,
            skip_connection_mlp=True,
            attention_method="dense",
            rngs=rngs,
        )
        
        self.head = nnx.Linear(self.model_size, 1, rngs=rngs)

    def marginalize(self, rng: Array, edge_mask: Array) -> Array:
        T = edge_mask.shape[0]
        idx = jax.random.choice(rng, jnp.arange(T), shape=(1,), replace=False)
        em = edge_mask.at[idx, :].set(False)
        em = em.at[:, idx].set(False)
        em = em.at[idx, idx].set(True)
        return em

    def __call__(
        self,
        t: Array,               # [B, 1, 1]
        x: Array,               # [B, T, 1]
        topo_mask: Array,       # [B, T, 1] bool
        node_ids: Array,        # [B, T] integer ids
        condition_mask: Array,  # [B, T] or [B, T, 1] bool
    ) -> Array:
                
        B, T, _ = x.shape

        # Shapes
        condition_mask = condition_mask.astype(jnp.bool_).reshape(B, T, 1)

        # Time embedding
        time_embeddings = self.time_embed(t)

        edge_masks = jnp.tile(self.edge2d.value[None, ...], (B, 1, 1)).astype(bool)

        topo_mask_attn = topo_mask @ jnp.swapaxes(topo_mask, -1, -2) # outer product

        combined_masks = jnp.logical_and(edge_masks,topo_mask_attn)

        # Tokenization
        value_embeddings = self.value_proj(x)
        id_embeddings = self.id_embed(node_ids)
        cond_emb = self.condition_embedding.value * condition_mask

        x_encoded = jnp.concatenate([value_embeddings, id_embeddings, cond_emb], axis=-1)  # [B,T,D]

        h = self.transformer(
            x_encoded,
            context=time_embeddings,
            mask=combined_masks
        )

        out = self.head(h)
        score = self.diff.output_scale_fn(t, out)
        return score