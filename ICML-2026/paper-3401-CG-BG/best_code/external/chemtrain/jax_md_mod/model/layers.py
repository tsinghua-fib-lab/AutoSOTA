# Copyright 2022 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Jax / Haiku implementation of layers to build the DimeNet++ architecture.

The :ref:`dimenet_building_blocks` take components of
:class:`chemtrain.potential.sparse_graph.SparseDirectionalGraph` as input.
Please refer to this class for input descriptions.
"""
import haiku as hk
from jax import nn, ops, numpy as jnp, scipy as jsp
from jax_md import util
from sympy import symbols, utilities

from jax_md_mod.model import dimenet_basis_util
from jax_md_mod.model.dropout import Linear


# util
def high_precision_segment_sum(data, segment_ids, num_segments=None,
                               out_type=util.f32, indices_are_sorted=False,
                               unique_indices=False, bucket_size=None):
    """Implements the jax.ops.segment_sum, but casts input to float64 before
    summation and casts back to a target output type afterwards (float32 by
    default). Used to inprove numerical accuracy of summation.
    """
    data = util.f64(data)
    seg_sum = ops.segment_sum(
        data, segment_ids, num_segments=num_segments,
        indices_are_sorted=indices_are_sorted, unique_indices=unique_indices,
        bucket_size=bucket_size
    )
    return out_type(seg_sum)


# Initializers

class OrthogonalVarianceScalingInit(hk.initializers.Initializer):
    """Initializer scaling variance of uniform orthogonal matrix distribution.

    Generates a weight matrix with variance according to Glorot initialization.
    Based on a random (semi-)orthogonal matrix. Neural networks are expected to
    learn better when features are decorrelated e.g. stated by
    "Reducing overfitting in deep networks by decorrelating representations".

    The approach is adopted from the original DimeNet and the implementation
    is inspired by Haiku's variance scaling initializer.

    Attributes:
        scale: Variance scaling factor
    """
    def __init__(self, scale=2.):
        """Constructs the OrthogonalVarianceScaling Initializer.

        Args:
            scale: Variance scaling factor
        """
        super().__init__()
        self.scale = scale
        self._orth_init = hk.initializers.Orthogonal()

    def __call__(self, shape, dtype=jnp.float32):
        assert len(shape) == 2
        fan_in, fan_out = shape
        # uniformly distributed orthogonal weight matrix
        w_init = self._orth_init(shape, dtype)
        w_var = jnp.var(w_init)
        w_var = jnp.where(w_var < 1.e-7, 1., w_var)  # if var = 0, fix 0 divide
        w_init *= jnp.sqrt(self.scale / (max(1., (fan_in + fan_out)) * w_var))
        return w_init


class RBFFrequencyInitializer(hk.initializers.Initializer):
    """Initializer of the frequencies of the RadialBesselLayer.

     Initializes the  frequency values to its canonical values.
     """

    def __call__(self, shape, dtype):
        return jnp.pi * jnp.arange(1, shape[0] + 1, dtype=dtype)


# DimeNet++ Layers

class SmoothingEnvelope(hk.Module):
    """Smoothing envelope function for radial edge embeddings.

    Smoothing the cut-off enables twice continuous differentiability of the
    model output, including the potential energy.
    The envelope function is 1 at 0 and has a root of multiplicity of 3 at 1 as
    defined in DimeNet. It is applied to scaled radial edge distances
    d_ij / cut_off [0, 1].

    The implementation corresponds to the definition in the DimeNet paper.
    It is different from the original implementation of DimeNet / DimeNet++ that
    define incorrect spherical basis layers as a result (a known issue).
    """

    def __init__(self, p=6, name='Envelope'):
        """Initializes the SmoothingEnvelope layer.

        Args:
            p: Power of the smoothing polynomial
            name: Name of the layer
        """
        super().__init__(name=name)
        self._p = p
        self._a = -(p + 1.) * (p + 2.) / 2.
        self._b = p * (p + 2.)
        self._c = -p * (p + 1.) / 2.

    def __call__(self, distances):
        """Returns the envelope values."""
        envelope_val = (1. + self._a * distances ** self._p
                        + self._b * distances ** (self._p + 1.)
                        + self._c * distances ** (self._p + 2.))
        return jnp.where(distances < 1., envelope_val, 0.)


class RadialBesselLayer(hk.Module):
    """Radial Bessel Function (RBF) representation of pairwise distances.

    Attributes:
        freq_init: RBFFrequencyInitializer
    """
    def __init__(self, cutoff, num_radial=16, envelope_p=6,
                 name='BesselRadial'):
        """Initializes the RBF layer.

        Args:
            cutoff: Radial cut-off distance
            num_radial: Number of radial Bessel embedding functions
            envelope_p: Power of envelope polynomial
            name: Name of RBF layer
        """
        super().__init__(name=name)
        self._inv_cutoff = 1. / cutoff
        self._envelope = SmoothingEnvelope(p=envelope_p)
        self._num_radial = [num_radial]
        self._rbf_scale = jnp.sqrt(2. / cutoff)
        self.freq_init = RBFFrequencyInitializer()

    def __call__(self, distances):
        """Returns the RBF embeddings of edge distances."""
        distances = jnp.expand_dims(distances, -1)  # to broadcast to num_radial
        scaled_distances = distances * self._inv_cutoff
        envelope_vals = self._envelope(scaled_distances)
        frequencies = hk.get_parameter('RBF_Frequencies',
                                       shape=self._num_radial,
                                       init=self.freq_init)
        rbf_vals = (self._rbf_scale * jnp.sin(frequencies * scaled_distances)
                    / distances)
        return envelope_vals * rbf_vals


class SphericalBesselLayer(hk.Module):
    """Spherical Bessel Function (SBF) representation of angular triplets."""
    def __init__(self, r_cutoff, num_spherical, num_radial, envelope_p=6,
                 name='BesselSpherical'):
        """Initializes the SBF layer.

        Args:
            r_cutoff: Radial cut-off
            num_spherical: Number of spherical Bessel embedding functions
            num_radial: Number of radial Bessel embedding functions
            envelope_p: Power of envelope polynomial
            name: Name of SBF layer
        """
        super().__init__(name=name)

        assert num_spherical > 1
        self._envelope = SmoothingEnvelope(p=envelope_p)
        self._inv_cutoff = 1. / r_cutoff
        self._num_radial = num_radial

        bessel_formulars = dimenet_basis_util.bessel_basis(num_spherical,
                                                           num_radial)
        sph_harmonic_formulas = dimenet_basis_util.real_sph_harm(num_spherical)
        self._sph_funcs = []
        self._radual_bessel_funcs = []
        # convert sympy functions: modules overrides them by jax.numpy functions
        x = symbols('x')
        theta = symbols('theta')
        first_sph = utilities.lambdify([theta],  # pyling-ignore
                                       sph_harmonic_formulas[0][0],
                                       modules=[jnp, jsp.special])(0)
        for i in range(num_spherical):
            if i == 0:
                self._sph_funcs.append(lambda a: jnp.zeros_like(a) + first_sph)
            else:
                self._sph_funcs.append(
                    utilities.lambdify([theta], sph_harmonic_formulas[i][0],
                                       modules=[jnp, jsp.special])
                )
            for j in range(num_radial):
                self._radual_bessel_funcs.append(
                    utilities.lambdify([x], bessel_formulars[i][j],
                                       modules=[jnp, jsp.special])
                )

    def __call__(self, pair_distances, angles, angle_mask, expand_to_kj):
        """Returns the SBF embeddings of angular triplets."""

        # initialize distances and envelope values
        scaled_distances = pair_distances * self._inv_cutoff
        envelope_vals = self._envelope(scaled_distances)
        envelope_vals = jnp.expand_dims(envelope_vals, -1)  # broadcast to rbf

        # compute radial bessel envelope for distances kj
        rbf = [radial_bessel(scaled_distances)
               for radial_bessel in self._radual_bessel_funcs]
        rbf = jnp.stack(rbf, axis=1)
        rbf_envelope = rbf * envelope_vals
        rbf_env_expanded = rbf_envelope[expand_to_kj]

        # compute spherical bessel embedding
        sbf = [spherical_bessel(angles) for spherical_bessel in self._sph_funcs]
        sbf = jnp.stack(sbf, axis=1)
        sbf = jnp.repeat(sbf, self._num_radial, axis=1)
        angle_mask = jnp.expand_dims(angle_mask, -1)
        sbf *= angle_mask  # mask non-existing triplets

        return rbf_env_expanded * sbf  # combine radial, spherical and envelope


class ResidualLayer(hk.Module):
    """Residual Layer: 2 activated Linear layers and a skip connection."""
    def __init__(self, layer_size, activation=nn.swish, init_kwargs=None,
                 name='ResLayer'):
        """Initializes the Residual layer.

        Args:
            layer_size: Output size of the Linear layers
            activation: Activation function
            init_kwargs: Dict of initialization kwargs for Linear layers
            name: Name of the Residual layer
        """
        super().__init__(name=name)
        self._layer1 = hk.Sequential([Linear(
            layer_size, name='ResidualSubLayer', **init_kwargs),
            activation])
        self._layer2 = hk.Sequential([Linear(
            layer_size, name='ResidualSubLayer', **init_kwargs),
            activation])

    def __call__(self, inputs, dropout_dict=None):
        """Returns the ouput of the Residual layer."""
        non_linear_part = inputs
        non_linear_part = self._layer1(non_linear_part, dropout_dict)
        non_linear_part = self._layer2(non_linear_part, dropout_dict)
        out = inputs + non_linear_part
        return out


class EmbeddingBlock(hk.Module):
    """Embeddimg block of DimeNet.

    Embeds edges by concattenating RBF embeddings with atom type embeddings of
    both connected atoms. If the network is defined to be kbT-dependent,
    adds a temperature embedding.
    """
    def __init__(self, embed_size, n_species, type_embed_size=None,
                 activation=nn.swish, init_kwargs=None, kbt_dependent=False,
                 name='Embedding'):
        """Initializes an Embedding block.

        Args:
            embed_size: Size of the edge embedding.
            n_species: Number of different atom species the network is supposed
                       to process.
            type_embed_size: Embedding size of atom type embedding. Default None
                             results in embed_size / 2.
            activation: Activation function
            init_kwargs: Dict of initialization kwargs for Linear layers
            kbt_dependent: Boolean, whether network prediction should depend on
                           temperature.
            name: Name of Embedding block
        """
        super().__init__(name=name)

        if type_embed_size is None:
            type_embed_size = int(embed_size / 2)

        embed_init = hk.initializers.RandomUniform(minval=-jnp.sqrt(3),
                                                   maxval=jnp.sqrt(3))
        self._embedding_vect = hk.get_parameter(
            'Embedding_vect', [n_species, type_embed_size], init=embed_init)
        self._kbt_dependent = kbt_dependent
        if kbt_dependent:
            self._kbt_embedding_div = hk.get_parameter(
                'Embedding_kbt_div', [1, type_embed_size], init=embed_init)
            self._kbt_embedding_mul = hk.get_parameter(
                'Embedding_kbt_mul', [1, type_embed_size], init=embed_init)

        # unlike the original DimeNet implementation, there is no activation
        # and bias in RBF_Dense as shown in the network sketch. This is
        # consistent with other Layers processing rbf values in the network
        self._rbf_dense = Linear(embed_size, name='RBF_Dense',
                                 with_bias=False, **init_kwargs)
        self._dense_after_concat = hk.Sequential(
            [Linear(embed_size, name='Concat_Dense', **init_kwargs),
             activation]
        )

    def __call__(self, rbf, species, idx_i, idx_j, dropout_dict=None, **kwargs):
        """Returns output of the Embedding block."""
        transformed_rbf = self._rbf_dense(rbf, dropout_dict)

        type_i = species[idx_i]
        type_j = species[idx_j]

        h_i = self._embedding_vect[type_i]
        h_j = self._embedding_vect[type_j]

        edge_embedding = jnp.concatenate([h_i, h_j, transformed_rbf], axis=-1)
        if self._kbt_dependent:
            assert 'kT' in kwargs, ('If potential should be kbt-dependent. '
                                    '"kT" needs to be provided as kwarg.')
            kbt_embeds_div = jnp.tile(self._kbt_embedding_div,
                                      (h_j.shape[0], 1)) / kwargs['kT']
            kbt_embeds_mul = jnp.tile(self._kbt_embedding_mul,
                                      (h_j.shape[0], 1)) * kwargs['kT']
            edge_embedding = jnp.concatenate([edge_embedding, kbt_embeds_div,
                                              kbt_embeds_mul], axis=-1)
        embedded_messages = self._dense_after_concat(edge_embedding,
                                                     dropout_dict)
        return embedded_messages


class OutputBlock(hk.Module):
    """DimeNet++ Output block.

    Predicts per-atom quantities given RBF embeddings and messages.
    """
    def __init__(self, embed_size, out_embed_size=None, num_dense=3,
                 num_targets=1, activation=nn.swish, init_kwargs=None,
                 name='Output', outscale=False):
        """Initializes an Output block.

        Args:
            embed_size: Size of the edge embedding.
            out_embed_size: Output size of Linear layers after upsampling
            num_dense: Number of dense layers
            num_targets: Number of target quantities to be predicted
            activation: Activation function
            init_kwargs: Dict of initialization kwargs for Linear layers
            name: Name of Output block
            outscale: Scale the output to initially have zero energy
        """
        super().__init__(name=name)
        if out_embed_size is None:
            out_embed_size = int(2 * embed_size)

        self._rbf_dense = Linear(embed_size, with_bias=False,
                                 name='RBF_Dense', **init_kwargs)
        self._upprojection = Linear(out_embed_size, with_bias=False,
                                    name='Upprojection', **init_kwargs)

        # transform summed messages via multiple dense layers before predicting
        # target quantities

        self._outscale = outscale

        self._dense_layers = []
        for _ in range(num_dense):
            self._dense_layers.append(hk.Sequential([
                Linear(out_embed_size, with_bias=True, name='Dense_Series',
                       **init_kwargs), activation])
            )

        self._dense_final = Linear(num_targets, with_bias=False,
                                   name='Final_output', **init_kwargs)

    def __call__(self, messages, rbf, idx_i, n_particles, dropout_dict=None):
        """Returns predicted per-atom quantities."""
        transformed_rbf = self._rbf_dense(rbf, dropout_dict)
        # rbf is masked correctly and transformation only via weights
        # Hence rbf acts as mask
        messages *= transformed_rbf

        # sum incoming messages for each atom: becomes a per-atom quantity
        summed_messages = high_precision_segment_sum(messages, idx_i,
                                                     num_segments=n_particles)

        upsampled_messages = self._upprojection(summed_messages, dropout_dict)
        for dense_layer in self._dense_layers:
            upsampled_messages = dense_layer(upsampled_messages, dropout_dict)

        per_atom_targets = self._dense_final(upsampled_messages)

        if self._outscale:
            out_scale = hk.get_parameter("output_scale", [], init=jnp.zeros)
            per_atom_targets *= out_scale

        return per_atom_targets


class InteractionBlock(hk.Module):
    """DimeNet++ Interaction block.

    Performs directional message-passing based on RBF and SBF embeddings as well
    as messages from the previous message-passing iteration. Updated messages
    are used in the subsequent Output block.
    """
    def __init__(self, embed_size, num_res_before_skip, num_res_after_skip,
                 activation=nn.swish, init_kwargs=None,
                 angle_int_embed_size=None, basis_int_embed_size=8,
                 name='Interaction'):
        """Initializes an Interaction block.

        Args:
            embed_size: Size of the edge embedding.
            num_res_before_skip: Number of Residual blocks before skip
            num_res_after_skip: Number of Residual blocks after skip
            activation: Activation function
            init_kwargs: Dict of initialization kwargs for Linear layers
            angle_int_embed_size: Embedding size of Linear layers for
                                  down-projected triplet interation
            basis_int_embed_size: Embedding size of Linear layers for interation
                                  of RBS/ SBF basis
            name: Name of Interaction block
        """
        super().__init__(name=name)
        if angle_int_embed_size is None:
            angle_int_embed_size = int(embed_size / 2)

        # directional message passing block
        self._rbf1 = Linear(basis_int_embed_size, name='rbf1',
                            with_bias=False, **init_kwargs)
        self._rbf2 = Linear(embed_size, name='rbf2',
                            with_bias=False, **init_kwargs)
        self._sbf1 = Linear(basis_int_embed_size, name='sbf1',
                            with_bias=False, **init_kwargs)
        self._sbf2 = Linear(angle_int_embed_size, name='sbf2',
                            with_bias=False, **init_kwargs)

        self._dense_kj = hk.Sequential([
            Linear(embed_size, name='Dense_kj', **init_kwargs), activation]
        )
        self._down_projection = hk.Sequential([
            Linear(angle_int_embed_size, name='Downprojection',
                   with_bias=False, **init_kwargs), activation]
        )
        self._up_projection = hk.Sequential([
            Linear(embed_size, name='Upprojection', with_bias=False,
                   **init_kwargs), activation]
        )

        # propagation block:
        self._dense_ji = hk.Sequential(
            [Linear(embed_size, name='Dense_ji', **init_kwargs), activation]
        )

        self._res_before_skip = []
        for _ in range(num_res_before_skip):
            self._res_before_skip.append(ResidualLayer(
                embed_size, activation, init_kwargs)
            )
        self._final_before_skip = hk.Sequential(
            [Linear(embed_size, name='FinalBeforeSkip', **init_kwargs),
             activation]
        )

        self._res_after_skip = []
        for _ in range(num_res_after_skip):
            self._res_after_skip.append(ResidualLayer(
                embed_size, activation, init_kwargs)
            )

    def __call__(self, m_input, rbf, sbf, reduce_to_ji, expand_to_kj,
                 dropout_dict=None):
        """Returns messages after interaction via message-passing."""
        # transformed messages for expansion to k -> j
        m_ji_angular = self._dense_kj(m_input, dropout_dict)
        rbf = self._rbf1(rbf, dropout_dict)
        rbf = self._rbf2(rbf, dropout_dict)
        m_ji_angular *= rbf

        m_ji_angular = self._down_projection(m_ji_angular, dropout_dict)
        m_kj = m_ji_angular[expand_to_kj]  # expand to nodes k connecting to j

        sbf = self._sbf1(sbf, dropout_dict)
        sbf = self._sbf2(sbf, dropout_dict)
        # automatic mask: sbf was masked during initial computation.
        # Sbf1 and 2 only weights, no biases
        m_kj *= sbf

        aggregated_m_ji = high_precision_segment_sum(
            m_kj, reduce_to_ji, num_segments=m_input.shape[0])
        propagated_messages = self._up_projection(aggregated_m_ji, dropout_dict)

        # add directional messages to original ones:
        # afterwards only independent edge transformations.
        # masking is lost, but rbf masks in output layer before aggregation
        m_ji = self._dense_ji(m_input, dropout_dict)
        m_combined = m_ji + propagated_messages

        for layer in self._res_before_skip:
            m_combined = layer(m_combined, dropout_dict)
        m_combined = self._final_before_skip(m_combined, dropout_dict)

        m_ji_with_skip = m_combined + m_input

        for layer in self._res_after_skip:
            m_ji_with_skip = layer(m_ji_with_skip, dropout_dict)
        return m_ji_with_skip
