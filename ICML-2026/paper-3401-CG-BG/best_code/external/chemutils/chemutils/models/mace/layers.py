# MIT License
#
# Copyright (c) 2022 mace-jax
# Copyright (c) 2025 tummfm
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import functools
from typing import Callable, Optional, Tuple, Set, Union

import e3nn_jax as e3nn
import haiku as hk
import jax.nn as jax_nn
import jax.numpy as jnp


class LinearNodeEmbeddingLayer(hk.Module):
    def __init__(self, embed_dim: int, num_species: int, irreps_out: e3nn.Irreps):
        super().__init__()
        self.num_species = num_species
        self.irreps_out = e3nn.Irreps(irreps_out).filter("0e").regroup()
        self.embedding = hk.Embed(num_species, embed_dim)
        self.linear = hk.Linear(self.irreps_out.dim, with_bias=False)

    def __call__(self, node_species: jnp.ndarray) -> e3nn.IrrepsArray:
        embedding = self.embedding(node_species)
        embedding = self.linear(embedding)
        return e3nn.IrrepsArray(self.irreps_out, embedding)


class LinearReadoutLayer(hk.Module):
    def __init__(
        self,
        output_irreps: e3nn.Irreps,
    ):
        super().__init__()
        self.output_irreps = output_irreps

    def __call__(self, x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        # x = [n_nodes, irreps]
        return e3nn.haiku.Linear(self.output_irreps)(x)  # [n_nodes, output_irreps]


class NonLinearReadoutLayer(hk.Module):
    def __init__(
        self,
        hidden_irreps: e3nn.Irreps,
        output_irreps: e3nn.Irreps,
        *,
        activation: Optional[Callable] = None,
        gate: Optional[Callable] = None,
    ):
        super().__init__()
        self.hidden_irreps = hidden_irreps
        self.output_irreps = output_irreps
        self.activation = activation
        self.gate = gate

    def __call__(self, x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        # x = [n_nodes, irreps]
        num_vectors = self.hidden_irreps.filter(
            drop=["0e", "0o"]
        ).num_irreps  # Multiplicity of (l > 0) irreps
        x = e3nn.haiku.Linear(
            (self.hidden_irreps + e3nn.Irreps(f"{num_vectors}x0e")).simplify()
        )(x)
        x = e3nn.gate(x, even_act=self.activation, even_gate_act=self.gate)
        return e3nn.haiku.Linear(self.output_irreps)(x)  # [n_nodes, output_irreps]


class QeqLayer(hk.Module):

    def __init__(self,
                 charge_embed_n_hidden: int = 16,
                 charge_embed_n_layers: int = 1,
                 max_radius: float = 1.0,
                 num_species: int = 100,
                 mlp_n_hidden: int = 32,
                 mlp_n_layers: int = 3,
                 mlp_activation: Callable = None,
                 ):
        super().__init__()

        self.charge_embed_dim = charge_embed_n_hidden
        self.charge_embed_layers = charge_embed_n_layers
        self.max_radius = max_radius

        self.mlp_n_hidden = mlp_n_hidden

        self.mlp_activation = mlp_activation
        self.mlp_n_layers = mlp_n_layers

        self.radius = hk.get_parameter("radius", (num_species,), jnp.float32, init=hk.initializers.Constant(0.0))
        self.hardness = hk.get_parameter("hardness", (num_species,), jnp.float32, init=hk.initializers.Constant(10.0))

        self.chi_shift_scale = ScaleShiftLayer(1.0, False)

        self.charge_embed = hk.get_parameter(
            "charge_embed", (num_species, charge_embed_n_hidden),
            jnp.float32, init=hk.initializers.RandomNormal(1.0)
        )

    def __call__(
        self,
        node_features: e3nn.IrrepsArray,  # [n_edges, 3]
        species: jnp.ndarray,
        charge_fn: Callable,
        exclude_correction: bool = False,
    ) -> e3nn.IrrepsArray:
        # Calculate charges via EEM method.
        chis = e3nn.haiku.MultiLayerPerceptron(
            [self.charge_embed_dim] * self.charge_embed_layers + [1,],
            self.mlp_activation,
            output_activation=False
        )(node_features)
        print(f"Before: {chis.irreps}")
        chis = self.chi_shift_scale(chis).array.squeeze()
        print(f"After: {chis.shape}")


        # Like for 4GNN
        # gammas = jax_nn.softplus(self.radius[species]) / jnp.log(2.0)
        # gammas = 1 + jnp.log(jnp.sqrt(self.radius[species] ** 2 + 1) + self.radius[species]) / 10
        gammas = jax_nn.softplus(self.radius[species]) / jnp.log(2)
        hardness = jax_nn.softplus(self.hardness[species])

        # Charge equilibration
        charges, pot = charge_fn(gammas, chis, hardness)

        w = e3nn.haiku.MultiLayerPerceptron(
            (self.charge_embed_dim,),
            None
        )(
            jnp.concatenate((
                charges[:, jnp.newaxis],
                self.charge_embed[species]), axis=-1
            )
        )

        node_features = e3nn.haiku.MultiLayerPerceptron(
            [self.mlp_n_hidden] * self.mlp_n_layers,
            self.mlp_activation,
            output_activation=False
        )(jnp.concatenate([node_features.array, w], axis=-1))

        return node_features, (charges, pot)



class EquivariantProductBasisLayer(hk.Module):
    def __init__(
        self,
        target_irreps: e3nn.Irreps,
        correlation: int,
        num_species: int,
        symmetric_tensor_product_basis: bool = True,
        off_diagonal: bool = False,
    ) -> None:
        super().__init__()
        self.target_irreps = e3nn.Irreps(target_irreps)
        self.symmetric_contractions = SymmetricContractionLayer(
            keep_irrep_out={ir for _, ir in self.target_irreps},
            correlation=correlation,
            num_species=num_species,
            gradient_normalization="element",  # NOTE: This is to copy mace-torch
            symmetric_tensor_product_basis=symmetric_tensor_product_basis,
            off_diagonal=off_diagonal,
        )

    def __call__(
        self,
        node_feats: e3nn.IrrepsArray,  # [n_nodes, feature * irreps]
        node_specie: jnp.ndarray,  # [n_nodes, ] int
    ) -> e3nn.IrrepsArray:
        node_feats = node_feats.mul_to_axis().remove_zero_chunks()
        node_feats = self.symmetric_contractions(node_feats, node_specie)
        node_feats = node_feats.axis_to_mul()
        return e3nn.haiku.Linear(self.target_irreps)(node_feats)


class InteractionLayer(hk.Module):
    def __init__(
        self,
        *,
        target_irreps: e3nn.Irreps,
        epsilon: float,
        max_ell: int,
        activation: Callable,
    ) -> None:
        super().__init__()
        self.target_irreps = target_irreps
        self.epsilon = epsilon
        self.max_ell = max_ell
        self.activation = activation

    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        radial_embedding: jnp.ndarray,  # [n_edges, radial_embedding_dim]
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
    ) -> Tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]:
        assert node_feats.ndim == 2
        assert vectors.ndim == 2
        assert radial_embedding.ndim == 2

        node_feats = e3nn.haiku.Linear(node_feats.irreps, name="linear_up")(node_feats)

        node_feats = MessagePassingConvolutionLayer(
            self.epsilon, self.target_irreps, self.max_ell, self.activation
        )(vectors, node_feats, radial_embedding, senders, receivers)

        node_feats = e3nn.haiku.Linear(self.target_irreps, name="linear_down")(
            node_feats
        )

        assert node_feats.ndim == 2
        return node_feats  # [n_nodes, target_irreps]


class ScaleShiftLayer(hk.Module):
    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.scale = hk.get_parameter(
            "scale", shape=(), init=hk.initializers.Constant(scale))
        self.shift = hk.get_parameter(
            "shift", shape=(), init=hk.initializers.Constant(shift))

    def __call__(self, x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        return self.scale * x + self.shift

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(scale={self.scale:.6f}, shift={self.shift:.6f})"
        )

class MessagePassingConvolutionLayer(hk.Module):
    def __init__(
        self,
        epsilon: float,
        target_irreps: e3nn.Irreps,
        max_ell: int,
        activation: Callable,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.target_irreps = e3nn.Irreps(target_irreps)
        self.max_ell = max_ell
        self.activation = activation

    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        radial_embedding: jnp.ndarray,  # [n_edges, radial_embedding_dim]
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
    ) -> e3nn.IrrepsArray:
        assert node_feats.ndim == 2

        messages = node_feats[senders]

        messages = e3nn.concatenate(
            [
                messages.filter(self.target_irreps),
                e3nn.tensor_product(
                    messages,
                    e3nn.spherical_harmonics(range(1, self.max_ell + 1), vectors, True),
                    filter_ir_out=self.target_irreps,
                ),
                # e3nn.tensor_product_with_spherical_harmonics(
                #     messages, vectors, self.max_ell
                # ).filter(self.target_irreps),
            ]
        ).regroup()  # [n_edges, irreps]

        # one = e3nn.IrrepsArray.ones("0e", edge_attrs.shape[:-1])
        # messages = e3nn.tensor_product(
        #     messages, e3nn.concatenate([one, edge_attrs.filter(drop="0e")])
        # ).filter(self.target_irreps)

        mix = e3nn.haiku.MultiLayerPerceptron(
            3 * [64] + [messages.irreps.num_irreps],
            self.activation,
            output_activation=False,
        )(
            radial_embedding
        )  # [n_edges, num_irreps]

        messages = messages * mix / node_feats.irreps.num_irreps  # [n_edges, irreps]

        zeros = e3nn.IrrepsArray.zeros(
            messages.irreps, node_feats.shape[:1], messages.dtype
        )
        node_feats = zeros.at[receivers].add(messages)  # [n_nodes, irreps]

        return node_feats * self.epsilon


A025582 = [0, 1, 3, 7, 12, 20, 30, 44, 65, 80, 96, 122, 147, 181, 203, 251, 289]

class SymmetricContractionLayer(hk.Module):
    def __init__(
        self,
        correlation: int,
        keep_irrep_out: Set[e3nn.Irrep],
        num_species: int,
        gradient_normalization: Union[str, float] = None,
        symmetric_tensor_product_basis: bool = True,
        off_diagonal: bool = False,
    ):
        super().__init__()
        self.correlation = correlation

        if gradient_normalization is None:
            gradient_normalization = e3nn.config("gradient_normalization")
        if isinstance(gradient_normalization, str):
            gradient_normalization = {"element": 0.0, "path": 1.0}[
                gradient_normalization
            ]
        self.gradient_normalization = gradient_normalization

        if isinstance(keep_irrep_out, str):
            keep_irrep_out = e3nn.Irreps(keep_irrep_out)
            assert all(mul == 1 for mul, _ in keep_irrep_out)

        self.keep_irrep_out = {e3nn.Irrep(ir) for ir in keep_irrep_out}
        self.num_species = num_species
        self.symmetric_tensor_product_basis = symmetric_tensor_product_basis
        self.off_diagonal = off_diagonal

    def __call__(self, input: e3nn.IrrepsArray, index: jnp.ndarray) -> e3nn.IrrepsArray:
        def fn(input: e3nn.IrrepsArray, index: jnp.ndarray):
            # - This operation is parallel on the feature dimension (but each feature has its own parameters)
            # This operation is an efficient implementation of
            # vmap(lambda w, x: FunctionalLinear(irreps_out)(w, concatenate([x, tensor_product(x, x), tensor_product(x, x, x), ...])))(w, x)
            # up to x power self.correlation
            assert input.ndim == 2  # [num_features, irreps_x.dim]
            assert index.ndim == 0  # int

            out = dict()

            for order in range(self.correlation, 0, -1):  # correlation, ..., 1
                if self.off_diagonal:
                    x_ = jnp.roll(input.array, A025582[order - 1])
                else:
                    x_ = input.array

                if self.symmetric_tensor_product_basis:
                    U = e3nn.reduced_symmetric_tensor_product_basis(
                        input.irreps, order, keep_ir=self.keep_irrep_out
                    )
                else:
                    U = e3nn.reduced_tensor_product_basis(
                        [input.irreps] * order, keep_ir=self.keep_irrep_out
                    )
                # U = U / order  # normalization TODO(mario): put back after testing
                # NOTE(mario): The normalization constants (/order and /mul**0.5)
                # has been numerically checked to be correct.

                # TODO(mario) implement norm_p

                # ((w3 x + w2) x + w1) x
                #  \-----------/
                #       out

                for (mul, ir_out), u in zip(U.irreps, U.list):
                    u = u.astype(x_.dtype)
                    # u: ndarray [(irreps_x.dim)^order, multiplicity, ir_out.dim]

                    w = hk.get_parameter(
                        f"w{order}_{ir_out}",
                        (self.num_species, mul, input.shape[0]),
                        dtype=jnp.float32,
                        init=hk.initializers.RandomNormal(
                            stddev=(mul**-0.5) ** (1.0 - self.gradient_normalization)
                        ),
                    )[
                        index
                    ]  # [multiplicity, num_features]
                    w = (
                        w * (mul**-0.5) ** self.gradient_normalization
                    )  # normalize weights

                    if ir_out not in out:
                        out[ir_out] = (
                            "special",
                            jnp.einsum("...jki,kc,cj->c...i", u, w, x_),
                        )  # [num_features, (irreps_x.dim)^(oder-1), ir_out.dim]
                    else:
                        out[ir_out] += jnp.einsum(
                            "...ki,kc->c...i", u, w
                        )  # [num_features, (irreps_x.dim)^order, ir_out.dim]

                # ((w3 x + w2) x + w1) x
                #  \----------------/
                #         out (in the normal case)

                for ir_out in out:
                    if isinstance(out[ir_out], tuple):
                        out[ir_out] = out[ir_out][1]
                        continue  # already done (special case optimization above)

                    out[ir_out] = jnp.einsum(
                        "c...ji,cj->c...i", out[ir_out], x_
                    )  # [num_features, (irreps_x.dim)^(oder-1), ir_out.dim]

                # ((w3 x + w2) x + w1) x
                #  \-------------------/
                #           out

            # out[irrep_out] : [num_features, ir_out.dim]
            irreps_out = e3nn.Irreps(sorted(out.keys()))
            return e3nn.IrrepsArray.from_list(
                irreps_out,
                [out[ir][:, None, :] for (_, ir) in irreps_out],
                (input.shape[0],),
            )

        # Treat batch indices using vmap
        shape = jnp.broadcast_shapes(input.shape[:-2], index.shape)
        input = input.broadcast_to(shape + input.shape[-2:])
        index = jnp.broadcast_to(index, shape)

        fn_mapped = fn
        for _ in range(input.ndim - 2):
            fn_mapped = hk.vmap(fn_mapped, split_rng=False)

        return fn_mapped(input, index)
