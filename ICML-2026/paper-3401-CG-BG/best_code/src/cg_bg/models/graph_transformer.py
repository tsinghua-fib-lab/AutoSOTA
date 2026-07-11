"""
This implementation is adapted from:
https://github.com/noegroup/ScoreMD/blob/main/src/scoremd/models/graph_transformer.py
Which was based on:
https://github.com/microsoft/two-for-one-diffusion/blob/main/models/graph_transformer.py
Original architecture by lucidrains:
https://github.com/lucidrains/graph-transformer-pytorch
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange, repeat


@dataclass
class GraphTransformer(nn.Module):
    t0: float
    t1: float
    rescale_time: bool
    clip_time: bool
    hidden_nf: int
    feature_embedding_dim: int
    max_z: Sequence[int]
    n_layers: int
    use_intrinsic_coords: bool = True
    use_abs_coords: bool = False
    use_distances: bool = False
    dropout: float = 0.0

    def _prepare_input(self, x, features, t, concat=True):

        if self.clip_time:
            t = jnp.clip(t, self.t0, self.t1)

        t_features = [
            t - (self.t1 - self.t0) / 2,
            jnp.cos(2 * jnp.pi * t),
            jnp.sin(2 * jnp.pi * t),
            -jnp.cos(4 * jnp.pi * t),
        ]

        if self.rescale_time:
            if self.t0 - 0.0 >= 1e-6 or self.t1 - 1.0 >= 1e-6:
                # only add these features, if either t0 or t1 is different from 0 or 1.
                # in this case, we allow the model to learn either from the absolute time or from the relative time.
                t01 = (t - self.t0) / (self.t1 - self.t0)
                t_features += [
                    jnp.cos(2 * jnp.pi * t01),
                    jnp.sin(2 * jnp.pi * t01),
                    -jnp.cos(4 * jnp.pi * t01),
                ]

        t = jnp.concatenate(t_features, axis=-1)

        if features is not None:
            features = features.astype(jnp.int32)

        if concat:
            if features is not None:
                return jnp.concatenate([x, features.reshape(features.shape[0], -1), t], axis=-1)
            return jnp.concatenate([x, t], axis=-1)
        return x, features, t

    @staticmethod
    def _reshape_input(x, features, t):

        # Expect x to be reshaped to (BS, atoms*3)
        assert x.ndim in (1, 2, 3), "Input x should be of shape (atoms*3,), (BS, atoms*3) or (BS, atoms, 3)"
        if x.ndim == 1:
            x = x[None, :]  # (atoms*3,) -> (1, atoms*3)
        elif x.ndim == 3:
            x = x.reshape(x.shape[0], -1)  # (BS, atoms, 3) -> (BS, atoms*3)

        original_shape = x.shape

        # Expect features to be reshaped to (BS, atoms, n_features) if exists
        if features is not None:
            assert features.ndim in (1, 2, 3), (
                "Input features should be of shape (atoms,), (atoms, n_features), "
                "(BS, atoms) or (BS, atoms, n_features)"
            )
            if features.ndim == 1:
                features = features[None, :, None]  # (atoms,) -> (1, atoms, 1)
            elif features.ndim == 2:
                if features.shape[1] == 1:
                    features = features[None, :, :]  # (atoms, 1) -> (1, atoms, 1)
                else:
                    features = features[:, :, None]  # (BS, atoms,) -> (BS, atoms, 1)

        # Expect t to be reshaped to (BS, 1)
        t = t.reshape(x.shape[0], 1)

        return x, features, t, original_shape

    def __call__(self, x, features, t, training=True):
        x, features, t, original_shape = self._reshape_input(x, features, t)

        out = self._forward(x, features, t, training)
        return out.reshape(original_shape)  # ensure that we return the same shape as the original input

    @nn.compact
    def _forward(self, x, features, t, training):
        x, features, t = self._prepare_input(x, features, t, concat=False)
        x = x.reshape(x.shape[0], -1, 3)  # This architecture expects the input to be of shape [bs, n_nodes, 3]
        bs, n_nodes, _ = x.shape

        t = jnp.tile(t[:, None, ...], (1, n_nodes, 1))  # shape = (bs, n_nodes, time_features)

        if features is None:
            h = jnp.eye(n_nodes)
            h = jnp.tile(h[None, ...], (bs, 1, 1))  # shape = (bs, n_nodes, n_node_features)
        else:
            assert len(self.max_z) == features.shape[-1], f"len(max_z) = {len(self.max_z)} != {features.shape[-1]}"
            new_features = []
            for i, max_z in enumerate(self.max_z):
                new_features.append(
                    nn.Embed(
                        num_embeddings=max_z,
                        features=self.feature_embedding_dim,
                        name=f"scalar_embedding_{i}",
                    )(features[:, :, i])
                )

            h = jnp.concatenate(new_features, axis=-1)

        edge_attr = self.get_edge_attr(x)
        edge_attr = nn.Dense(self.hidden_nf, name="edge_embedding")(edge_attr)

        if self.use_abs_coords:
            nodes = jnp.concatenate([h, x, t], axis=-1)
        else:
            nodes = jnp.concatenate([h, t], axis=-1)

        nodes = nn.Dense(self.hidden_nf, name="node_embedding")(nodes)
        mask = jnp.ones((bs, n_nodes), dtype=bool)

        nodes, _ = GraphTransformerLucid(depth=self.n_layers, with_feedforwards=True, dropout=self.dropout)(
            nodes, edge_attr, training, mask=mask
        )

        return nn.Dense(x.shape[1] * x.shape[2], name="node_decoder")(nodes.reshape(bs, -1)).reshape(x.shape)

    def get_edge_attr(self, x):
        # x shape: [bs, n_nodes, 3]
        if self.use_distances and not self.use_intrinsic_coords:
            # compute squared distances between nodes
            xa = jnp.expand_dims(x, axis=1)
            xb = jnp.expand_dims(x, axis=2)
            diff = xa - xb
            dist = jnp.sum(diff**2, axis=-1, keepdims=True)
            return dist
        elif self.use_intrinsic_coords and not self.use_distances:
            xa = jnp.expand_dims(x, axis=1)
            xb = jnp.expand_dims(x, axis=2)
            diff = xa - xb
            return diff
        elif self.use_intrinsic_coords and self.use_distances:
            xa = jnp.expand_dims(x, axis=1)
            xb = jnp.expand_dims(x, axis=2)
            diff = xa - xb
            dist = jnp.sum(diff**2, axis=-1, keepdims=True)
            return jnp.concatenate([diff, dist], axis=-1)
        else:
            raise ValueError("Either use_intrinsic_coords or use_distances must be enabled.")


class PreNorm(nn.Module):
    """Apply layer norm to the first argument of the function."""

    fn: Callable

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        x = nn.LayerNorm()(x)
        return self.fn(x, *args, **kwargs)


class GatedResidual(nn.Module):
    @nn.compact
    def __call__(self, x, res):
        gate_input = jnp.concatenate([x, res, x - res], axis=-1)
        gate = nn.Dense(1, use_bias=False)(gate_input)
        gate = nn.sigmoid(gate)
        return x * gate + res * (1 - gate)


class Attention(nn.Module):
    heads: int = 8
    dim_head: int = 64

    @nn.compact
    def __call__(self, nodes, edges, mask=None):
        h = self.heads
        inner_dim = self.dim_head * h

        scale = self.dim_head**-0.5

        q = nn.Dense(inner_dim)(nodes)
        k = nn.Dense(inner_dim)(nodes)
        v = nn.Dense(inner_dim)(nodes)
        e_kv = nn.Dense(inner_dim)(edges)

        q = rearrange(q, "b ... (h d) -> (b h) ... d", h=h)
        k = rearrange(k, "b ... (h d) -> (b h) ... d", h=h)
        v = rearrange(v, "b ... (h d) -> (b h) ... d", h=h)
        e_kv = rearrange(e_kv, "b ... (h d) -> (b h) ... d", h=h)

        ek, ev = e_kv, e_kv

        k = rearrange(k, "b j d -> b () j d")
        v = rearrange(v, "b j d -> b () j d")

        k += ek
        v += ev

        sim = jnp.einsum("b i d, b i j d -> b i j", q, k) * scale

        if mask is not None:
            mask = rearrange(mask, "b i -> b i ()") & rearrange(mask, "b j -> b () j")
            mask = repeat(mask, "b i j -> (b h) i j", h=h)
            sim = jnp.where(mask, sim, -jnp.finfo(sim.dtype).max)

        attn = nn.softmax(sim, axis=-1)
        out = jnp.einsum("b i j, b i j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return nn.Dense(nodes.shape[-1])(out)


@dataclass
class GraphTransformerLucid(nn.Module):
    depth: int
    dim_head: int = 64
    heads: int = 8
    with_feedforwards: bool = True
    norm_edges: bool = False
    dropout: float = 0.0
    ff_mult: int = 4  # This is the multiplier for how many neurons are in the feedforward layer

    @nn.compact
    def __call__(self, nodes, edges, training, mask=None):
        if self.norm_edges:
            edges = nn.LayerNorm()(edges)

        for _ in range(self.depth):
            attn_out = PreNorm(Attention(heads=self.heads, dim_head=self.dim_head))(nodes, edges, mask=mask)
            nodes = GatedResidual()(attn_out, nodes)

            if self.with_feedforwards:
                feed_forward_out = PreNorm(
                    nn.Sequential(
                        [
                            nn.Dense(nodes.shape[-1] * self.ff_mult),
                            nn.gelu,
                            nn.Dense(nodes.shape[-1]),
                            nn.Dropout(self.dropout, deterministic=not training),
                        ]
                    )
                )(nodes)
                nodes = GatedResidual()(feed_forward_out, nodes)

        return nodes, edges
