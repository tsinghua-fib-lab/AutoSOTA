# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
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

"""A collection of functions evaluating quantiities of trajectories.
For easiest integration into chemtain, functions should be compatible with
traj_util.quantity_traj. Functions provided to quantity_traj need to take the
state and additional kwargs.
"""
from functools import partial

import jax
import numpy as onp

from jax import grad, vmap, lax, jacrev, jacfwd, numpy as jnp, jit
from jax.scipy.stats import norm

from jax_md import space, util, dataclasses, quantity, simulate, partition

from jax_md_mod.model import sparse_graph
from jax_md_mod import custom_partition

Array = util.Array


def energy_wrapper(energy_fn_template, fixed_energy_params=None):
    """Wrapper around energy_fn to allow energy computation via
    traj_util.quantity_traj.

    Args:
        energy_fn_template: Function creating an energy function when called
            with energy parameters.
        fixed_energy_params: Always use the energy function obtained when
            using the fixed energy params. If not given, the function uses
            dynamially specified parameters.

    """
    def energy(state, neighbor, energy_params, energy_and_force=None, **kwargs):
        if energy_and_force is not None:
            print(f"[Potential] Found precomputed forces.")
            return energy_and_force['energy']

        if fixed_energy_params is None:
            energy_fn = energy_fn_template(energy_params)
        else:
            energy_fn = energy_fn_template(fixed_energy_params)
        return energy_fn(state.position, neighbor=neighbor, **kwargs)
    return energy


def force_wrapper(energy_fn_template, fixed_energy_params=None):
    """Wrapper around energy_fn to allow force computation via
    traj_util.quantity_traj.

    Args:
        energy_fn_template: Function creating an energy function when called
            with energy parameters.
        fixed_energy_params: Always use the energy function obtained when
            using the fixed energy params. If not given, the function uses
            dynamially specified parameters.

    """
    def energy(state, neighbor, energy_params, energy_and_force=None, **kwargs):
        if energy_and_force is not None:
            print(f"[Force] Found precomputed forces.")
            return energy_and_force['force']

        if fixed_energy_params is None:
            energy_fn = energy_fn_template(energy_params)
        else:
            energy_fn = energy_fn_template(fixed_energy_params)
        force_fn = quantity.force(energy_fn)
        return force_fn(state.position, neighbor=neighbor, **kwargs)
    return energy


def energy_force_wrapper(energy_fn_template,
                         fixed_energy_params=None,
                         has_aux=False):
    """Wrapper around energy_fn to allow energy and force computation via
    traj_util.quantity_traj.

    Args:
        energy_fn_template: Function creating an energy function when called
            with energy parameters.
        fixed_energy_params: Always use the energy function obtained when
            using the fixed energy params. If not given, the function uses
            dynamially specified parameters.
        has_aux: Whether the energy function has an auxiliary output. In this
            case, the energy function will be called with the argument
            ``mode="with_aux"`` and should return a tuple with ``(energy, aux)``.

    """
    def energy_and_force_fn(state, neighbor, energy_params, **kwargs):
        if fixed_energy_params is None:
            energy_fn = energy_fn_template(energy_params)
        else:
            energy_fn = energy_fn_template(fixed_energy_params)

        if has_aux:
            kwargs['mode'] = 'with_aux'

        box = kwargs.pop('box', None)
        @partial(jax.value_and_grad, argnums=(0, 1), has_aux=has_aux)
        def energy_and_grads_fn(R, _box):
            if box is not None:
                return energy_fn(R, neighbor=neighbor, box=_box, **kwargs)
            else:
                return energy_fn(R, neighbor=neighbor, **kwargs)

        energy_or_aux, (neg_force, box_grads) = energy_and_grads_fn(state.position, box)
        if has_aux:
            energy, aux = energy_or_aux
        else:
            aux = None
            energy = energy_or_aux

        return {'energy': energy, 'force': -neg_force, 'box_grad': box_grads, 'aux': aux}
    return energy_and_force_fn


def get_aux(aux_key=""):
    """Reads out auxiliary output from the energy function."""
    def snapshot_fn(state, energy_and_force=None, **kwargs):
        assert energy_and_force is not None, f"Need to provide aux for {aux_key}."
        assert aux_key in energy_and_force['aux'].keys(), f"Need to provide aux for {aux_key}."
        print(f"Read out {aux_key} from aux.")

        return energy_and_force['aux'][aux_key]
    return snapshot_fn


def kinetic_energy_wrapper(state, **unused_kwargs):
    """Wrapper around kinetic_energy to allow kinetic energy computation via
    traj_util.quantity_traj.
    """
    return quantity.kinetic_energy(velocity=state.velocity, mass=state.mass)


def total_energy_wrapper(energy_fn_template):
    """Wrapper around energy_fn to allow total energy computation via
    traj_util.quantity_traj.
    """
    def energy(state, neighbor, energy_params, **kwargs):
        energy_fn = energy_fn_template(energy_params)
        pot_energy = energy_fn(state.position, neighbor=neighbor, **kwargs)
        kinetic_energy = quantity.kinetic_energy(state.velocity, state.mass)
        return pot_energy + kinetic_energy
    return energy


def connected(state, neighbor, mask=None, **kwargs):
    """Checks whether the system is connected."""

    return custom_partition.check_connectivity(neighbor, mask=mask)


def temperature(state, mask=None, **kwargs):
    """Temperature function that is consistent with quantity_traj interface."""
    mass = state.mass
    velocity = state.velocity

    if mask is None:
        mask = jnp.ones(velocity.shape[0], dtype=bool)
    else:
        print(f"Masked temperature computation")

    velocity = jnp.where(mask[:, jnp.newaxis], velocity, 0.0)
    mass = jnp.where(mask[:, jnp.newaxis], mass, 1.0)
    dof = jnp.sum(mask) * jnp.shape(velocity)[-1]

    return jnp.sum(velocity ** 2 * mass) / dof


def _dyn_box(reference_box, state, **kwargs):
    """Gets box dynamically from kwargs, if provided, otherwise defaults to
    reference. Ensures that a box is provided and deletes from kwargs.
    """
    # Get the dynamic box if simulation is in NPT ensemble
    if hasattr(state, 'box_position'):
        return simulate.npt_box(state), kwargs
    elif hasattr(state, 'box'):
        return state.box, kwargs

    box = kwargs.pop('box', reference_box)
    assert box is not None, ('If no reference box is given, needs to be '
                             'given as kwarg "box".')
    return box, kwargs

def _dyn_kT(kT, **kwargs):
    """Gets kT dynamically from kwargs, if provided, otherwise defaults to
    reference. Ensures that a kT is provided and deletes from kwargs.
    """
    kT = kwargs.pop('kT', kT)
    assert kT is not None, ('If no reference kT is given, needs to be '
                             'given as kwarg "kT".')
    return kT, kwargs


def volume(state, **kwargs):
    """Returns volume of a single snapshot."""
    dim = state.position.shape[-1]
    box, _ = _dyn_box(None, state, **kwargs)
    volume = quantity.volume(dim, box)
    return volume


def _canonicalized_masses(state):
    if state.mass.ndim == 0:
        masses = jnp.ones(state.position.shape[0]) * state.mass
    else:
        masses = state.mass
    return masses


def density(state, **unused_kwargs):
    """Returns density of a single snapshot of the NPT ensemble."""
    masses = _canonicalized_masses(state)
    total_mass = jnp.sum(masses)
    volume = volume(state)
    return total_mass / volume


# TODO distinct classes and discretization functions don't seem optimal
#  --> possible refactor


@dataclasses.dataclass
class RDFParams:
    """Hyperparameters to initialize the radial distribution function (RDF).

    Attributes:
        reference_rdf: The target rdf; initialize with None if no target available
        rdf_bin_centers: The radial positions of the centers of the rdf bins
        rdf_bin_boundaries: The radial positions of the edges of the rdf bins
        sigma_RDF: Standard deviation of smoothing Gaussian
    """
    reference: Array
    rdf_bin_centers: Array
    rdf_bin_boundaries: Array
    sigma: Array


def rdf_discretization(rdf_cut, nbins=300, rdf_start=0.):
    """Computes dicretization parameters for initialization of RDF function.

    Args:
        rdf_cut: Cut-off length inside which pairs of particles are considered
        nbins: Number of bins in radial direction
        rdf_start: Minimal distance after which particle pairs are considered

    Returns:
        Arrays with radial positions of bin centers, bin edges and the standard
        deviation of the Gaussian smoothing kernel.

    """
    dx_bin = (rdf_cut - rdf_start) / float(nbins)
    rdf_bin_centers = jnp.linspace(rdf_start + dx_bin / 2.,
                                   rdf_cut - dx_bin / 2.,
                                   nbins)
    rdf_bin_boundaries = jnp.linspace(rdf_start, rdf_cut, nbins + 1)
    sigma_rdf = jnp.array(dx_bin)
    return rdf_bin_centers, rdf_bin_boundaries, sigma_rdf


@dataclasses.dataclass
class ADFParams:
    """Hyperparameters to initialize a angular distribution function (ADF).

    Attributes:
        reference_adf: The target adf; initialize with None if no target available
        adf_bin_centers: The positions of the centers of the adf bins over theta
        sigma_ADF: Standard deviation of smoothing Gaussian
        r_outer: Outer radius beyond which particle triplets are not considered
        r_inner: Inner radius below which particle triplets are not considered
    """
    reference: Array
    adf_bin_centers: Array
    sigma: Array
    r_outer: Array
    r_inner: Array


def adf_discretization(nbins=200):
    """Computes dicretization parameters for initialization of ADF function.

    Args:
        nbins: Number of bins discretizing theta

    Returns:
        Arrays containing bin centers in theta direction and the standard
        deviation of the Gaussian smoothing kernel.
    """
    dtheta_bin = jnp.pi / float(nbins)
    adf_bin_centers = jnp.linspace(dtheta_bin / 2.,
                                   jnp.pi - dtheta_bin / 2.,
                                   nbins)
    sigma_adf = util.f32(dtheta_bin)
    return adf_bin_centers, sigma_adf


def dihedral_discretization(nbins=150):
    dbin = 2 * jnp.pi / float(nbins)
    bin_centers = dbin * (jnp.arange(nbins) + 0.5) - jnp.pi
    sigma = util.f32(dbin)
    return bin_centers, sigma


@dataclasses.dataclass
class TCFParams:
    """Hyperparameters to initialize a triplet correlation function (TFC).

    The triplet is defined via the sides x, y, z. Implementation according to
    https://aip.scitation.org/doi/10.1063/1.4898755 and
    https://aip.scitation.org/doi/10.1063/5.0048450.

    Attributes:
        reference_tcf: The target tcf; initialize with None if no target
            available
        sigma_TCF: Standard deviation of smoothing Gaussian
        volume_TCF: Histogram volume element according to
            https://journals.aps.org/pra/abstract/10.1103/PhysRevA.42.849
        tcf_x_bin_centers: The radial positions of the centers of the tcf bins in
            x direction
        tcf_y_bin_centers: The radial positions of the centers of the tcf bins in
            y direction
        tcf_z_bin_centers: The radial positions of the centers of the tcf bins in
            z direction

    """
    reference: Array
    sigma_tcf: Array
    volume: Array
    tcf_x_bin_centers: Array
    tcf_y_bin_centers: Array
    tcf_z_bin_centers: Array


def tcf_discretization(tcf_cut, nbins=30, tcf_start=0.1):
    """Computes dicretization parameters for initialization of TCF function.

    Args:
        tcf_cut: Cut-off length inside which pairs of particles are considered
        nbins: Number of bins in all three radial direction
        tcf_start: Minimal distance after which particle pairs are considered

    Returns:
        Tuple containing standard deviation of the Gaussian smoothing kernel,
        histogram volume array and arrays with radial positions of bin centers
        in x, y, z.

    """
    dx_bin = (tcf_cut - tcf_start) / float(nbins)
    tcf_bin_centers = jnp.linspace(tcf_start + dx_bin / 2.,
                                   tcf_cut - dx_bin / 2.,
                                   nbins)

    tcf_x_binx_centers, tcf_y_bin_centers, tcf_z_bin_centers = jnp.meshgrid(
        tcf_bin_centers, tcf_bin_centers, tcf_bin_centers, sparse=True)

    sigma_tcf = jnp.array(dx_bin)
    # volume for non-linear triplets (sigma / min(x,y,z)->0)
    volume_tcf = (8 * jnp.pi**2 * tcf_x_binx_centers * tcf_y_bin_centers
                  * tcf_z_bin_centers * sigma_tcf**3)

    return (sigma_tcf, volume_tcf, tcf_x_binx_centers, tcf_y_bin_centers,
            tcf_z_bin_centers)


@dataclasses.dataclass
class BondAngleParams:
    reference: Array
    sigma: Array
    bonds: Array
    bin_centers: Array
    bin_boundaries: Array


def init_bond_angle_distribution(displacement_fn, bond_angle_params: BondAngleParams, reference_box=None):
    """Initializes a function computing a dihedral distribution.

        Args:
            displacement_fn: Displacement to compute dihedral angles
            bond_angle_params: Struct describing the dihedral angles and expected
                format of the computed distribution
            reference_box: Unused

        Returns:
            Returns a function that computes a distribution of dihedral angles
            given a simulation state.

        """

    _, sigma, bonds, bin_centers, bin_boundaries = dataclasses.astuple(bond_angle_params)
    bin_size = jnp.diff(bin_boundaries)

    def angle_fn(state, neighbor, **kwargs):

        angles = angular_displacement(
            state.position, displacement_fn, bonds, degrees=True)

        #  Gaussian ensures that discrete integral over distribution is 1
        exp = jnp.exp(
            util.f32(-0.5) * (angles[:, jnp.newaxis] - bin_centers)**2 / sigma)
        gaussian_distances = exp * bin_size / jnp.sqrt(2 * jnp.pi * sigma**2)
        per_bond = util.high_precision_sum(gaussian_distances,
                                                         axis=1)  # sum nbrs
        mean_angle_dist = util.high_precision_sum(per_bond,
                                                 axis=0) / bonds.shape[0]

        return mean_angle_dist

    return angle_fn


@dataclasses.dataclass
class BondDihedralParams:
    reference: Array
    sigma: Array
    bonds: Array
    bin_centers: Array
    bin_boundaries: Array


def init_bond_dihedral_distribution(displacement_fn,
                                    bond_dihedral_params: BondDihedralParams,
                                    smoothing = 'gaussian'):
    """Initializes a function computing a dihedral distribution.

    Args:
        displacement_fn: Displacement to compute dihedral angles
        bond_dihedral_params: Struct describing the dihedral angles and expected
            format of the computed distribution
        reference_box: Unused

    Returns:
        Returns a function that computes a distribution of dihedral angles
        given a simulation state.

    """

    _, sigma, bonds, bin_centers, bin_boundaries = dataclasses.astuple(bond_dihedral_params)
    bin_size = jnp.diff(bin_boundaries)

    def dihedral_fn(state, neighbor, **kwargs):

        dihedrals = dihedral_displacement(
            state.position, displacement_fn, bonds, degrees=False)

        distances = dihedrals[None, :] - bin_centers[:, None]
        if smoothing == 'gaussian':
            # Smooth the bins
            exponent = -0.5 * jnp.square(distances)
            exponent /= sigma ** 2
            kernel = jnp.exp(exponent)
        elif smoothing == 'epanechnikov':
            distances /= bin_size
            kernel = 0.75 * (1 - distances ** 2)
            kernel = jnp.where(kernel >= 0, kernel, 0.0)
        else:
            raise ValueError(f"Smoothing {smoothing} unknown.")

        # Sum over all contributing bonds
        dihedral_distribution = util.high_precision_sum(kernel, axis=1)

        # Normalize the distribution
        dihedral_distribution /= util.high_precision_sum(
            dihedral_distribution * bin_size, axis=0)

        return dihedral_distribution

    return dihedral_fn


def _ideal_gas_density(particle_density, bin_boundaries):
    """Returns bin densities that would correspond to an ideal gas."""
    r_small = bin_boundaries[:-1]
    r_large = bin_boundaries[1:]
    bin_volume = (4. / 3.) * jnp.pi * (r_large**3 - r_small**3)
    bin_weights = bin_volume * particle_density
    return bin_weights


def init_rdf(displacement_fn,
             rdf_params,
             reference_box=None,
             rdf_species=None):
    """Initializes a function that computes the radial distribution function
    (RDF) for a single state.

    Args:
        displacement_fn: Displacement function
        rdf_params: RDFParams defining the hyperparameters of the RDF
        reference_box: Simulation box. Can be provided here for constant boxes
            or on-the-fly as kwarg 'box', e.g. for NPT ensemble
        rdf_species: Array of species pairs for which the RDF should be
            computed. If not provided, compute the RDF for all particles
            irrespectively of their species.

    Returns:
        A function taking a simulation state and returning the instantaneous RDF
    """
    _, bin_centers, bin_boundaries, sigma = dataclasses.astuple(rdf_params)
    distance_metric = space.canonicalize_displacement_or_metric(displacement_fn)
    bin_size = jnp.diff(bin_boundaries)

    def pair_corr_fun(position, box, species):
        """Returns instantaneous pair correlation function while ensuring
        each particle pair contributes exactly 1.
        """
        n_particles = position.shape[0]
        metric = partial(distance_metric, box=box)
        metric = space.map_product(metric)
        dr = metric(position, position)
        # neglect same particles i.e. distance = 0.
        dr = jnp.where(dr > util.f32(1.e-7), dr, util.f32(1.e7))

        #  Gaussian ensures that discrete integral over distribution is 1
        exponent = (dr[:, :, jnp.newaxis] - bin_centers) ** 2.0
        exponent *= -0.5 / sigma ** 2
        exp = jnp.exp(exponent)

        gdist = exp * bin_size / jnp.sqrt(2 * jnp.pi * sigma ** 2)

        if rdf_species is not None:
            # Find the species of each particle
            is_species_i = rdf_species[:, (0,)] == species[None, :]
            is_species_j = rdf_species[:, (1,)] == species[None, :]

            mask = jnp.logical_and(
                is_species_i[:, None, :],
                is_species_j[:, :, None]
            )

            masked_gdist = gdist[None, ...] * mask[:, ..., None]
            # Axes of masked_gdisk are:
            # [rdf_idx, particle_i, particle_j, bins]

            # Sum over neighbors of corresponding species
            mean_pair_corr = util.high_precision_sum(
                masked_gdist, axis=(1, 2)
            )

            # TODO: Improve normalization for per-species RDF
            mean_pair_corr /= jnp.sum(is_species_i, axis=1)[:, None]
            mean_pair_corr *= n_particles / jnp.sum(is_species_j, axis=1, keepdims=True)
        else:
            mean_pair_corr = util.high_precision_sum(
                gdist, axis=(0, 1))  # sum nbrs
            mean_pair_corr /= n_particles
        return mean_pair_corr

    def rdf_compute_fun(state, species=None, **kwargs):
        box, _ = _dyn_box(reference_box, state, **kwargs)

        # Note: we cannot use neighbor list since RDF cutoff and
        # neighbor list cut-off don't coincide in general
        n_particles, spatial_dim = state.position.shape
        total_vol = quantity.volume(spatial_dim, box)
        mean_pair_corr = pair_corr_fun(state.position, box, species)

        # RDF is defined to relate the particle densities to an ideal gas.
        particle_density = n_particles / total_vol
        normalization = _ideal_gas_density(particle_density, bin_boundaries)
        rdf = mean_pair_corr / normalization
        return rdf
    return rdf_compute_fun


def _triplet_pairwise_displacements(position,
                                    neighbor: partition.NeighborList,
                                    displacement_fn,
                                    species = None,
                                    max_triplets: int = None,
                                    return_mask: bool = False):
    """Computes the displacements r_ij and r_kj between triplets of particles.

    For each triplet of particles :math:`(ijk)`, the function computes the
    displacement vectors

    .. math::

        r_{kj} = R_k - R_j\\ \\text{and}\\ r_ij.

    These vectors pointing from the central particle j to the side particles
    i and k.

    Returns:
        Returns a tuple (r_kj, r_ij) that contains the displacement vectors for
        all triplets. The displacement arrays have the shape
        ``r_kj.shape = (N_triplets, 3)``.

    """

    # Compute the indices of the triplet edges
    ij, kj, mask = custom_partition.get_triplet_indices(neighbor)

    if max_triplets is not None:
        ij = ij[:max_triplets, ...]
        kj = kj[:max_triplets, ...]
        mask = mask[:max_triplets]

    # Compute the displacements
    r_ij = vmap(displacement_fn)(position[ij[:, 0]], position[ij[:, 1]])
    r_kj = vmap(displacement_fn)(position[kj[:, 0]], position[kj[:, 1]])

    if return_mask:
        return r_kj, r_ij, mask
    else:
        return r_kj, r_ij


def _triplet_species(neighbor,
                     species,
                     max_triplets = None,
                     return_mask: bool = False):
    """Compute the species of triplets."""
    ij, kj, mask = custom_partition.get_triplet_indices(neighbor)
    if max_triplets is not None:
        ij = ij[:max_triplets]
        kj = kj[:max_triplets]
        mask = mask[:max_triplets]

    si = species[ij[:, 0]]
    sj = species[ij[:, 1]]
    sk = species[kj[:, 0]]

    if return_mask:
        return si, sj, sk, mask
    else:
        return si, sj, sk


def init_adf_nbrs(displacement_fn,
                  adf_params: ADFParams,
                  adf_species: Array = None,
                  smoothing_dr: float = 0.01,
                  r_init: Array = None,
                  nbrs_init: partition.NeighborList = None,
                  max_weight_multiplier: float = 2.):
    """Initializes a function to computes the angular distribution function (ADF).

    Smoothens the histogram in radial direction via a Gaussian kernel
    (compare RDF function).
    In radial direction, triplets are weighted according to a
    Gaussian cumulative distribution function, such that triplets with both
    radii inside the cut-off band are weighted approximately 1 and the weights
    of triplets towards the band edges are smoothly reduced to 0.

    For computational speed-up and reduced memory needs, ``r_init`` and
    ``nbrs_init`` can be provided to estimate the maximum number of triplets.

    Warning:
        Currently, the user does not receive information whether overflow
        occurred.

    Note:
        This function assumes that r_outer is smaller than the neighbor
        list cut-off. If this is not the case, a function computing all pairwise
        distances is necessary.

    Args:
        displacement_fn: Displacement function
        adf_params: Hyperparameters of the ADF
        smoothing_dr: Standard deviation of Gaussian smoothing in radial
            direction
        r_init: Initial positions to estimate maximum number of triplets
        nbrs_init: Initial neighborlist to estimate maximum number of triplets
        max_weight_multiplier: Multiplier to increase maximum number of triplets

    Returns:
        Returns a function that takes a simulation state with neighborlist and
        computes the instantaneous adf.
    """

    _, bin_centers, sigma_theta, r_outer, r_inner = dataclasses.astuple(
        adf_params)
    sigma_theta = util.f32(sigma_theta)
    bin_centers = util.f32(bin_centers)

    def cut_off_weights(r_kj, r_ij, mask):
        """Smoothly constraints triplets to a radial band such that both
        distances are between r_inner and r_outer. The Gaussian cdf is used for
        smoothing. The smoothing width can be controlled by the gaussian
        standard deviation.
        """
        dist_kj = space.distance(r_kj)
        dist_ij = space.distance(r_ij)
        # get d_small and d_large for each triplet
        pair_dist = jnp.column_stack((dist_kj, dist_ij)).sort(axis=1)
        # get inner boundary weight from r_small and outer weight from r_large
        inner_weight = norm.cdf(pair_dist[:, 0], loc=r_inner,
                                scale=smoothing_dr**2)
        outer_weight = 1 - norm.cdf(pair_dist[:, 1], loc=r_outer,
                                    scale=smoothing_dr**2)

        weights = outer_weight * inner_weight
        return weights * mask

    def weighted_adf(angles, weights):
        """Compute weighted ADF contribution of each triplet. For
        differentiability, each triplet contribution is smoothed via a Gaussian.
        """
        exp = jnp.exp(util.f32(-0.5) * (angles[:, jnp.newaxis] - bin_centers)**2
                      / sigma_theta**2)
        gaussians = exp / jnp.sqrt(2 * jnp.pi * sigma_theta**2)
        gaussians *= weights[:, jnp.newaxis]

        unnormed_adf = util.high_precision_sum(gaussians, axis=0)
        adf = unnormed_adf / jnp.trapezoid(unnormed_adf, bin_centers)

        return adf

    # We use initial configuration to estimate the maximum number of triplets
    # inside the cutoff radii.
    #
    if r_init is not None:
        if nbrs_init is None:
            raise ValueError(
                'If we estimate the maximum number of triplets, the initial '
                'neighbor list is a necessary input.'
            )

        r_ij, r_kj, mask = _triplet_pairwise_displacements(
            r_init, nbrs_init, displacement_fn, return_mask=True)

        weights = cut_off_weights(r_kj, r_ij, mask)

        max_weights = min([
            int(jnp.sum(weights > 1.e-6) * max_weight_multiplier),
            mask.size
        ])
        max_triplets = min([
            int(jnp.sum(mask) * max_weight_multiplier),
            mask.size
        ])

        print(f"[ADF] Estimates {max_triplets} max. triplets in neighbor list "
              f"and {max_weights} max. triplets in cutoff-shell.")

    else:
        max_weights = None
        max_triplets = None

    def adf_fn(state, neighbor, species=None, **kwargs):
        """Returns ADF for a single snapshot. Allows changing the box
        on-the-fly via the 'box' kwarg.
        """
        dyn_displacement = partial(displacement_fn, **kwargs)  # box kwarg

        r_kj, r_ij, mask = _triplet_pairwise_displacements(
            state.position, neighbor, dyn_displacement,
            max_triplets=max_triplets, return_mask=True
        )

        weights = cut_off_weights(r_kj, r_ij, mask)

        if species is not None:
            # Compute a second mask depending on the species of the triplets
            si, sj, sk = _triplet_species(neighbor, species, max_triplets)
            selection = si[None, :] == adf_species[:, (0,)]
            selection = jnp.logical_and(
                selection, sj[None, :] == adf_species[:, (1,)])
            selection = jnp.logical_and(
                selection, sk[None, :] == adf_species[:, (2,)])
        else:
            selection = None

        if max_triplets is not None:
            # Prune triplets by returning the most important weights
            non_zero_weights = weights > 1.e-6
            _, sorting_idxs = lax.top_k(weights, max_weights)

            weights = weights[sorting_idxs]
            r_ij = r_ij[sorting_idxs]
            r_kj = r_kj[sorting_idxs]
            mask = mask[sorting_idxs]

            if selection is not None:
                selection = selection[:, sorting_idxs]

            # TODO check for overflow

        if selection is not None:
            weights = jnp.einsum('sn,n->sn', selection, weights)
            _adf_fn = vmap(weighted_adf, in_axes=(None, 0))
        else:
            _adf_fn = weighted_adf

        # ensure differentiability of tanh
        r_ij_safe, r_kj_safe = sparse_graph.safe_angle_mask(r_ij, r_kj, mask)
        angles = vmap(sparse_graph.angle)(r_ij_safe, r_kj_safe)

        return _adf_fn(angles, weights)
    return adf_fn


def init_tcf_nbrs(displacement_fn,
                  tcf_params: TCFParams,
                  reference_box: Array = None,
                  nbrs_init: partition.NeighborList = None,
                  batch_size: int = 1000,
                  max_weight_multiplier: float = 1.2,
                  tcf_species: Array = None):
    """Initializes a function to compute the triplet correlation function (TCF).

    This function assumes that the neighbor list cutoff matches the TCF cutoff.

    Args:
        displacement_fn: Displacement function
        tcf_params: TCFParams defining the hyperparameters of the TCF
        nbrs_init: Initial neighborlist to estimate maximum number of triplets
        max_weight_multiplier: Multiplier for estimate of number of triplets
        batch_size: Batch size for more efficient binning of triplets
        reference_box: Simulation box. Can be provided here for constant boxes
            or on-the-fly as kwarg ``'box'``, e.g., for NPT ensemble

    Returns:
        A function that takes a simulation state with neighborlist and returns
        the instantaneous tcf.
    """
    if tcf_species is not None:
        raise NotImplementedError("Species-dependent TCF not yet implemented.")

    (_, sigma, volume, x_bin_centers, y_bin_centers,
     z_bin_centers) = dataclasses.astuple(tcf_params)
    nbins = x_bin_centers.shape[1]

    # We use the initial configuration to estimate the maximum number of
    # non-zero weights to speed up the computation and improve the memory
    # footprint
    if nbrs_init is None:
        raise NotImplementedError('nbrs_init currently needs to be provided.')

    r_ij, r_kj, mask = _triplet_pairwise_displacements(
        nbrs_init.reference_position, nbrs_init, displacement_fn,
        return_mask=True
    )

    max_triplets = int(jnp.count_nonzero(mask > 1.e-6) * max_weight_multiplier)

    # Increase the maximum number of triplets to enable simple batching
    rem = jnp.remainder(max_triplets, batch_size)
    max_triplets = max_triplets + (batch_size - rem)

    def gaussian_3d_bins(exp, inputs):
        triplet_distances, triplet_mask = inputs
        batch_exp = jnp.exp(util.f32(-0.5) * (
                (triplet_distances[:, 0, jnp.newaxis, jnp.newaxis, jnp.newaxis]
                 - x_bin_centers)**2 / sigma**2
                + (triplet_distances[:, 1, jnp.newaxis, jnp.newaxis,
                   jnp.newaxis] - y_bin_centers)**2 / sigma**2
                + (triplet_distances[:, 2, jnp.newaxis, jnp.newaxis,
                   jnp.newaxis] - z_bin_centers)**2 / sigma**2
        ))
        batch_exp *= triplet_mask[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
        batch_exp = jnp.sum(batch_exp, axis=0)
        exp += batch_exp
        return exp, 0

    def triplet_corr_fun(r_kj, r_ij, triplet_mask):
        """Returns instantaneous triplet correlation function while ensuring
        each particle pair contributes exactly 1.
        """
        # Close the triplet triangle
        r_ki = r_kj - r_ij

        dist_kj = space.distance(r_kj)
        dist_ij = space.distance(r_ij)
        dist_ki = space.distance(r_ki)

        histogram = jnp.zeros((nbins, nbins, nbins))
        distances = jnp.stack((dist_kj, dist_ij, dist_ki), axis=1)
        distances = jnp.reshape(distances, (-1, batch_size, 3))
        triplet_mask = jnp.reshape(triplet_mask, (-1, batch_size))

        # scan over per-batch computations for computational efficiency
        histogram = lax.scan(gaussian_3d_bins, histogram,
                             (distances, triplet_mask))[0]
        return histogram / volume / jnp.sqrt((2 * jnp.pi)**3)

    def tcf_fn(state, neighbor, **kwargs):
        """Returns TCF for a single snapshot. Allows changing the box
        on-the-fly via the 'box' kwarg.
        """
        dyn_displacement = partial(displacement_fn, **kwargs)  # box kwarg

        r_kj, r_ij, triplet_mask = _triplet_pairwise_displacements(
            state.position, neighbor, dyn_displacement,
            max_triplets=max_triplets, return_mask=True)

        box, _ = _dyn_box(reference_box, state, **kwargs)
        n_particles, spatial_dim = state.position.shape
        total_vol = quantity.volume(spatial_dim, box)
        particle_density = n_particles / total_vol

        tcf = triplet_corr_fun(r_kj, r_ij, triplet_mask)
        return tcf / n_particles / particle_density ** 2
    return tcf_fn


def _nearest_tetrahedral_nbrs(displacement_fn, position, nbrs):
    """Returns the displacement vectors r_ij of the 4 nearest neighbors."""
    neighbor_displacement = space.map_neighbor(displacement_fn)
    n_particles, _ = nbrs.idx.shape
    neighbor_mask = nbrs.idx != n_particles
    r_neigh = position[nbrs.idx]
    # R_ij = R_i - R_j; i = central atom
    displacements = neighbor_displacement(position, r_neigh)
    distances = space.distance(displacements)
    jnp.where(neighbor_mask, distances, 1.e7)  # mask non-existing neighbors
    _, nearest_idxs = lax.top_k(-1 * distances, 4)  # 4 nearest neighbor indices
    nearest_displ = jnp.take_along_axis(
        displacements, jnp.expand_dims(nearest_idxs, -1), axis=1)
    return nearest_displ


def init_tetrahedral_order_parameter(displacement_fn):
    """Initializes a function that computes the tetrahedral order parameter q
    for a single state.

    Args:
        displacement_fn: Displacement function

    Returns:
        A function that takes a simulation state with neighborlist and returns
        the instantaneous q value.
    """
    @partial(vmap, in_axes=(None, 0, 0))
    def _masked_inner(nn_disp, j, k):
        mask = k > j

        r_ij = nn_disp[:, j]
        r_ik = nn_disp[:, k]

        psi_ijk = vmap(quantity.cosine_angle_between_two_vectors)(r_ij, r_ik)
        summand = jnp.square(psi_ijk + (1. / 3.))

        return mask * summand


    def q_fn(state, neighbor, **kwargs):
        dyn_displacement = partial(displacement_fn, **kwargs)
        nearest_dispacements = _nearest_tetrahedral_nbrs(
            dyn_displacement, state.position, neighbor)

        all_j, all_k = jnp.meshgrid(jnp.arange(3), jnp.arange(4))
        masked_angles = _masked_inner(
            nearest_dispacements, all_j.ravel(), all_k.ravel())
        summed_angles = jnp.sum(masked_angles, axis=0)

        q = 1 - (3. / 8.) * jnp.mean(summed_angles)
        return q
    return q_fn


def init_local_structure_index(displacement_fn,
                               r_cut: float = 0.37,
                               reference_box = None,
                               r_init = None,
                               max_pairs_multiplier: float = 3.0):
    """Initializes function to compute the local structure index (LSI).

    The LSI measures the gap between the first and second solvation shell
    [#dobouedijon2015]_.

    Args:
        displacement_fn: Function to compute the particle distances
        r_cut: Cutoff of second solvation shell
        reference_box: Reference box to compute particle distances. Necessary
            if no dynamic box is provided.
        r_init: Initial coordinates to estimate the number of particles in the
            shell.
        max_pairs_multiplier: Multiplies the estimated maximum number of
           particles in a shell.


    References:
        .. [#dobouedijon2015] E. Duboué-Dijon und D. Laage,
           „Characterization of the Local Structure in Liquid Water by Various
           Order Parameters“, J. Phys. Chem. B, Bd. 119, Nr. 26, S. 8406–8418,
           Juli 2015, doi: 10.1021/acs.jpcb.5b02936.

    """

    # Estimate the number of pairs inside the first two solvation shells to
    # speed up the computation
    distance_metric = space.canonicalize_displacement_or_metric(
        displacement_fn)

    def _estimate_max_pairs():
        num_atoms = r_init.shape[0]
        if r_init is not None:
            metric = space.map_product(distance_metric)
            distances = metric(r_init, r_init)
            mp = jnp.max(jnp.sum(distances < r_cut, axis=1))
            mp = int(mp * max_pairs_multiplier)
        else:
            mp = num_atoms

        print(f"[LSI] Consider {mp} number of pairs.")
        return mp

    max_pairs = _estimate_max_pairs()

    @vmap
    def _single_lsi(dist):
        # Speed up the computation by only considering a subset of all
        # particles.
        # We seek to include the closest particles, so we have to select k
        # maxima of the negative distance.
        # Additionally, the calculation of the lsi requires sorting
        # these closest particles.
        _, selected = lax.top_k(-dist, max_pairs)
        dr = lax.sort(dist[selected])[1:]
        mask = (dr < util.f32(r_cut))[:-1]

        # Mask out particles that are not close to any other particles
        count = jnp.sum(mask)
        mask /= jnp.where(count > 0, count, 1)

        # Compute variance between the particle distance increments
        delta = jnp.diff(dr)
        lsi = jnp.sum(mask * jnp.square(delta - jnp.sum(mask * delta)))

        # Set the LSI to zero for isolated particles
        return (count > 0) * lsi

    def lsi_fn(state, **kwargs):
        box, _ = _dyn_box(reference_box, state, **kwargs)
        # Incorporate the dynamic box and compute the distance between all
        # pairs of the particles

        dyn_metric = partial(distance_metric, box=box)
        distance_fn = space.map_product(dyn_metric)
        distances = distance_fn(state.position, state.position)

        single_lsi = _single_lsi(distances)

        return jnp.mean(single_lsi)
    return lsi_fn



def rigid_body_alignment(positions, reference_positions, weights=None, eps=1e-5):
    """Computes the optimal rigid body motion aligning the positions to
    a reference structure.

    The weighted RMSD between a current positions $p$ and reference positions
    $q$ is defined as

    .. math ::

        \\mathrm{RMSD} = \\sqrt{\\frac{\\sum_{i=1}^n w_i || (Rp + t )- q||^2}{\\sum_{i=1}^n w_i}},

    where $R$ and $t$ define a rigid body motion that minimizes the
    RMSD [#hornung2017]_.
    
    Args:
        positions: Positions to be aligned

    Returns:
        Aligned positions

    References:
        .. [#hornung2017] O. Sorkine-Hornung und M. Rabinovich,
           „Least-Squares Rigid Motion Using SVD“.
           https://igl.ethz.ch/projects/ARAP/svd_rot.pdf

    """
    if weights is None:
        weights = jnp.ones((positions.shape[0],), dtype=positions.dtype)
    weights /= jnp.sum(weights)

    # Shift the positions to their weighted center of mass
    pbar = jnp.sum(weights[:, None] * positions, axis=0)
    X = positions - pbar

    qbar = jnp.sum(weights[:, jnp.newaxis] * reference_positions, axis=0)
    Y = reference_positions - qbar

    # Weighted covariance: X^T W Y
    cov = jnp.einsum('ni,n,nj->ij', X, weights, Y)

    U, _, Vh = jnp.linalg.svd(cov, full_matrices=False, compute_uv=True)
    V = Vh.T

    # Proper rotation (avoid reflection)
    d = jnp.linalg.det(V @ U.T)
    sig = jnp.concatenate([
        jnp.ones((positions.shape[1] - 1,), dtype=positions.dtype),
        jnp.array([d], dtype=positions.dtype)
    ])

    # Compute the optional rigid body rotation
    R = V @ jnp.diag(sig) @ U.T
    t = qbar - R @ pbar

    # Apply the rigid body motion. Rotate around the center of mass and then
    # translate to the reference center of mass.
    p_opt = jnp.einsum('ij,nj->ni', R, X)
    p_opt += qbar[jnp.newaxis, :]

    # Compute the mean squared deviation after alignment
    msd = jnp.sum(weights[:, jnp.newaxis] * jnp.square(p_opt - reference_positions))

    # Compute the com distance and the rotation angle

    d2 = jnp.sum((pbar - qbar) ** 2)
    d2 = jnp.where(d2 > eps, d2, eps)
    # Save sqrt operation for gradient stability
    d = jnp.sqrt(d2)

    # Convert to rotation angle.
    cos_theta = (jnp.trace(R) - 1) / 2
    sin_theta = jnp.sum((R - R.T) ** 2)
    sin_theta = jnp.where(sin_theta > eps, sin_theta, eps)
    sin_theta = jnp.sqrt(sin_theta) / (2 * jnp.sqrt(2))

    theta = jnp.arctan2(sin_theta, cos_theta)

    return (R, t), p_opt, (msd, d, theta)



def init_rmsd(reference_positions,
              displacement_fn,
              reference_box,
              idx=None,
              weights=None):
    """Initializes the root mean squared distance from a reference structure.

    The RMSD is a common measure in the analysis of
    macrostructures [#sargsyan2017]_.



    Args:
        reference_positions: Reference positions including all atoms.
        displacement_fn: Function to compute displacement between particles.
        reference_box: Reference box of the reference structure.
        idx: Indices selecting only the structure of interest, e.g. for
            a protein in a solvent.
        weights: Weight the rmsd, e.g., with masses of the particles.

    References:
        .. [#sargsyan2017] K. Sargsyan, C. Grauffel, und C. Lim,
           „How Molecular Size Impacts RMSD Applications in Molecular Dynamics
           Simulations“, J. Chem. Theory Comput., Bd. 13, Nr. 4, S. 1518–1524,
           Apr. 2017, doi: 10.1021/acs.jctc.7b00028.


    """
    if idx is None:
        idx = onp.arange(reference_positions.shape[0])

    def rmsd_fn(state, **kwargs):
        box, _ = _dyn_box(reference_box, state, **kwargs)
        dyn_displacement = partial(displacement_fn, box=box)

        ref_p = state.position[idx[0]]
        ref_q = reference_positions[idx[0]]
        
        # Compute the displacements with respect to the first atoms to deal with
        # different kinds of boundary conditions

        p = vmap(dyn_displacement, in_axes=(None, 0))(ref_p, state.position[idx])
        q = vmap(partial(displacement_fn, box=reference_box),
                in_axes=(None, 0))(ref_q, reference_positions[idx])

        *_, (msd, *_) = rigid_body_alignment(p, q, weights=weights)
        rmsd = jnp.sqrt(msd)

        return rmsd

    return rmsd_fn


def init_rigid_body_alignment(displacement_fn, shift_fn, reference_positions, weights=None):
    """Initializes a function that aligns a structure to a reference structure.

    The aligned structure minimizes the (weighted) root mean squared distance
    to the reference structure under rotations and translations, i.e.,
    rigig body motions.

    Args:
        displacement_fn: Displacement function
        reference_position: Reference positions including all atoms.
        weights: Weight the rmsd, e.g., with masses of the particles.

    Returns:
        Returns a function to compute optimally aligned positions.

    """
    if weights is None:
        weights = jnp.ones(reference_positions.shape[0], dtype=reference_positions.dtype)

    def align_fn(positions, **kwargs):
        ref_p = positions[0]
        ref_q = reference_positions[0]
        
        # Compute the displacements with respect to the first atoms to deal with
        # different kinds of boundary conditions

        p = vmap(
            partial(displacement_fn, **kwargs), in_axes=(None, 0)
        )(ref_p, positions)
        q = vmap(
            partial(displacement_fn, **kwargs), in_axes=(None, 0)
        )(ref_q, reference_positions)

        _, p_opt, _ = rigid_body_alignment(p, q, weights=weights)
        p_opt = vmap(
            partial(shift_fn, **kwargs), in_axes=(None, 0)
        )(ref_p, p_opt)

        return p_opt
    return align_fn


def init_velocity_autocorrelation(num_lags):
    """Returns the velocity autocorrelation function (VACF).

    Args:
        num_lags: Number of time lags to compute VACF values for. The time lag
        is implicitly defined by the dime difference between two adjacent
        states in the trajectory.

    Returns:
        An array containing the value of the VACF for each considered time lag.
    """

    # TODO this quadratic-scaling implementation of autocorrelation is not
    #  optimal. Long-term this should be using FFT if efficiency is critical
    @partial(vmap, in_axes=(None, 0))
    def _vel_correlation(vel, lag):
        # Assume that array is of shape (Frames, Particles, Dimension).
        # We roll around the frame axis to create a lag and average over all particles
        lagged_vel = jnp.roll(vel, axis=0, shift=lag)
        corr = jnp.mean(jnp.sum(vel * lagged_vel, axis=-1), axis=-1)
        # Since the first elementwise products are now (v_(n-lag) * v_0), etc. we have to mask them out
        mask = jnp.arange(vel.shape[0]) >= lag
        avg_corr = jnp.sum(mask * corr) / jnp.sum(mask)
        return avg_corr

    @jit
    def vac_fn(state, **kwargs):
        del kwargs
        # Due to a broadcasting error, it is necessary to compute the velocity from momentum
        if state.mass.ndim == 1:
            velocity = state.momentum / state.mass[:, None, None]
        else:
            velocity = state.velocity
        lag_array = jnp.arange(num_lags)

        vacf = lax.map(partial(_vel_correlation, velocity), lag_array)
        return vacf

    return vac_fn


def self_diffusion_green_kubo(traj_state, time_step, t_cut):
    """Green-Kubo formulation to compute self-diffusion D via the velocity
    autocorrelation function (VACF).

    .. math::

       D = \\frac{1}{dim} Int_0^{t_cut} VACF(\\tau) d\\tau

    Args:
        traj_state: TrajectoryState containing a finely resolved trajectory.
        time_step: Time lag between 2 adjacent states in the trajetcory. The
            simulation time step, in the usual case where every state is
            retained.
        t_cut: Cut-off time: Biggest time-difference to consider in the VACF.

    Returns:
        A tuple (D, VACF). Estimate of self-diffusion D and VACF that can be
        used for additional post-processing / analysis.
    """
    num_lags = int(t_cut / time_step)
    vel_autocorr = velocity_autocorrelation(traj_state, num_lags)
    dim = traj_state.trajectory.velocity.shape[-1]
    diffusion = jnp.trapezoid(vel_autocorr, dx=time_step) / dim
    return diffusion, vel_autocorr


def init_bond_length(displacement_fn, bonds, average=False):
    """Initializes a function that computes bond lengths for given atom pairs.

    Args:
        displacement_fn: Displacement function
        bonds: (n, 2) array defining IDs of bonded particles
        average: If False, returns per-pair bond lengths.
                 If True, returns scalar average over all pairs

    Returns:
        A function that takes a simulation state and returns bond lengths
    """
    metric = vmap(space.canonicalize_displacement_or_metric(displacement_fn))

    def bond_length(state, **kwargs):
        dyn_metric = partial(metric, **kwargs)
        r1 = state.position[bonds[:, 0]]
        r2 = state.position[bonds[:, 1]]
        distances = dyn_metric(r1, r2)
        if average:
            return jnp.mean(distances)
        else:
            return distances
    return bond_length


def _bond_length(bonds, positions, displacement_fn):
    """Computes bond lengths for given atom position vector and bonds."""

    def pairwise_bond_length(bond, pos):
        bond_displacement = displacement_fn(pos[bond[0]], pos[bond[1]])
        bond_distance = space.distance(bond_displacement)
        return bond_distance

    batched_pair_boond_length = vmap(pairwise_bond_length, (0, None))

    if positions.ndim == 3:
        distances = vmap(batched_pair_boond_length, (None, 0))(bonds, positions)
        return distances
    elif positions.ndim == 2:
        distances = vmap(pairwise_bond_length, (0, None))(bonds, positions)
        return distances
    else:
        raise ValueError('Positions must be either of shape Ntimestep x Natoms'
                         ' x spatial_dim or N_atoms x spatial_dim')


def estimate_bond_constants(positions, bonds, displacement_fn):
    """Calculates the equlibrium harmonic bond constants from given positions.

    Can be used to estimate the bond constants from an atomistic
    simulation to be used as a coarse-grained prior.

    Args:
        positions: Position vector of size [Ntimestep x Natoms x spatial_dim]
                   or [N_atoms x spatial_dim]
        bonds: (n_bonds, 2) array defining IDs of bonded particles
        displacement_fn: Displacement function

    Returns:
        Tuple (eq_distances, eq_variances) of harmonic bond coefficients.
    """
    distances = _bond_length(bonds, positions, displacement_fn)
    eq_distances = jnp.mean(distances, axis=0)
    eq_variances = jnp.var(distances, axis=0)
    return eq_distances, eq_variances


def angular_displacement(positions, displacement_fn, angle_idxs, degrees=True):
    """Computes the dihedral angle for all quadruple of atoms given in idxs.

    Args:
        positions: Positions of atoms in box
        displacement_fn: Displacement function
        dihedral_idxs: (n, 4) array defining IDs of quadruple particles
        degrees: If False, returns angles in rads.
                 If True, returns angles in degrees.

    Returns:
        An array (n,) of the dihedral angles.
    """
    p0 = positions[angle_idxs[:, 0]]
    p1 = positions[angle_idxs[:, 1]]
    p2 = positions[angle_idxs[:, 2]]

    b0 = -1. * vmap(displacement_fn)(p1, p0)
    b1 = vmap(displacement_fn)(p2, p1)

    cos = vmap(quantity.cosine_angle_between_two_vectors)(b0, b1)

    if degrees:
        return jnp.degrees(jnp.arccos(cos))
    else:
        return jnp.arccos(cos)


def dihedral_displacement(positions, displacement_fn, dihedral_idxs,
                          degrees=True):
    """Computes the dihedral angle for all quadruple of atoms given in idxs.

    Args:
        positions: Positions of atoms in box
        displacement_fn: Displacement function
        dihedral_idxs: (n, 4) array defining IDs of quadruple particles
        degrees: If False, returns angles in rads.
                 If True, returns angles in degrees.

    Returns:
        An array (n,) of the dihedral angles.
    """
    p0 = positions[dihedral_idxs[:, 0]]
    p1 = positions[dihedral_idxs[:, 1]]
    p2 = positions[dihedral_idxs[:, 2]]
    p3 = positions[dihedral_idxs[:, 3]]

    b0 = -1. * vmap(displacement_fn)(p1, p0)
    b1 = vmap(displacement_fn)(p2, p1)
    b2 = vmap(displacement_fn)(p3, p2)

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= jnp.linalg.norm(b1, axis=1)[:, None]

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - jnp.sum(b0 * b1, axis=1)[:, None] * b1
    w = b2 - jnp.sum(b2 * b1, axis=1)[:, None] * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = jnp.sum(v * w, axis=1)
    cross = vmap(jnp.cross)(b1, v)
    y = jnp.sum(cross * w, axis=1)

    if degrees:
        return jnp.degrees(jnp.arctan2(y, x))
    else:
        return jnp.arctan2(y, x)


def kinetic_energy_tensor(state):
    """Computes the kinetic energy tensor of a single snapshot.

    Args:
        state: Jax_md simulation state

    Returns:
        Kinetic energy tensor
    """
    average_velocity = jnp.mean(state.velocity, axis=0)
    thermal_excitation_velocity = state.velocity - average_velocity
    diadic_velocity_product = vmap(lambda v: jnp.outer(v, v))
    velocity_tensors = diadic_velocity_product(
        jnp.sqrt(state.mass) * thermal_excitation_velocity)
    return util.high_precision_sum(velocity_tensors, axis=0)


def virial_potential_part(energy_fn,
                          state,
                          nbrs,
                          reference_box,
                          energy_and_force=None,
                          fractional_coordinates=True,
                          **kwargs):
    """Interaction part of the virial pressure tensor for a single snaphot
    based on the formulation of Chen at al. (2020). See
    init_virial_stress_tensor. for details."""
    position = state.position  # in unit box if fractional coordinates used
    box, kwargs = _dyn_box(reference_box, state, **kwargs)

    if energy_and_force is None:
        energy_fn_ = lambda pos, neighbor, box: energy_fn(
            pos, neighbor=neighbor, box=box, **kwargs)  # for grad
        negative_forces, box_gradient = grad(energy_fn_, argnums=[0, 2])(
            position, nbrs, box)
    else:
        print(f"[Virial] Found precomputed forces.")
        box_gradient = energy_and_force['box_grad']

    box_contribution = jnp.dot(box_gradient, box.T)
    return box_contribution


def init_virial_stress_tensor(energy_fn_template, reference_box=None,
                              include_kinetic=True, pressure_tensor=False):
    """Initializes a function that computes the virial stress tensor for a
    single state.

    This function is applicable to arbitrary many-body interactions without
    explicit volume dependence, e.g., that only model interactions between
    images with minimum distance. It does not respect the volume dependence
    of, e.g., long-range electrostatic correction in periodic systems.

    Chen et al. "TensorAlloy: An automatic atomistic neural network program
    for alloys". Computer Physics Communications 250 (2020): 107057

    Args:
        energy_fn_template: A function that takes energy parameters as input
            and returns an energy function
        reference_box: The transformation T of general periodic boundary
            conditions. If None, box_tensor needs to be provided as ``'box'``
            during function call, e.g. for the NPT ensemble.
        include_kinetic: Whether kinetic part of stress tensor should be added.
        pressure_tensor: If False (default), returns the stress tensor. If True,
             returns the pressure tensor, i.e. the negative stress tensor.

    Returns:
        A function that takes a simulation state with neighbor list,
        energy_params and box (if applicable) and returns the instantaneous
        virial stress tensor.
    """
    if pressure_tensor:
        pressure_sign = -1.
    else:
        pressure_sign = 1.

    def virial_stress_tensor_neighborlist(state, neighbor, energy_params,
                                          **kwargs):
        # Note: this workaround with the energy_template was needed to keep
        #       the function jitable when changing energy_params on-the-fly
        # TODO function to transform box to box-tensor
        box, kwargs = _dyn_box(reference_box, state, **kwargs)
        energy_fn = energy_fn_template(energy_params)
        virial_tensor = virial_potential_part(
            energy_fn, state, neighbor, box, **kwargs)
        spatial_dim = state.position.shape[-1]
        volume = quantity.volume(spatial_dim, box)
        if include_kinetic:
            kinetic_tensor = -1 * kinetic_energy_tensor(state)
            return pressure_sign * (kinetic_tensor + virial_tensor) / volume
        else:
            return pressure_sign * virial_tensor / volume

    return virial_stress_tensor_neighborlist


def init_pressure(energy_fn_template, reference_box=None, include_kinetic=True):
    """Initializes a function that computes the pressure for a single state.

    This function is applicable to arbitrary many-body interactions, even
    under periodic boundary conditions. See `init_virial_stress_tensor`
    for details.

    Args:
        energy_fn_template: A function that takes energy parameters as input
                            and returns an energy function
        ref_box_tensor: The transformation T of general periodic boundary
                        conditions. If None, box_tensor needs to be provided as
                        'box' during function call, e.g. for NPT ensemble.
        include_kinetic: Whether kinetic part of stress tensor should be added.

    Returns:
        A function that takes a simulation state with neighbor list,
        energy_params and box (if applicable) and returns the instantaneous
        pressure.
    """
    # pressure is negative hydrostatic stress
    stress_tensor_fn = init_virial_stress_tensor(
        energy_fn_template, reference_box, include_kinetic=include_kinetic,
        pressure_tensor=True
    )

    def pressure_neighborlist(state, neighbor, energy_params, **kwargs):
        pressure_tensor = stress_tensor_fn(state, neighbor, energy_params,
                                           **kwargs)
        return jnp.trace(pressure_tensor) / 3.
    return pressure_neighborlist


def energy_under_strain(epsilon, energy_fn, box_tensor, state, neighbor,
                        **kwargs):
    """Potential energy of a state after applying linear strain epsilon."""
    # Note: When computing the gradient, we deal with infinitesimally
    #       small strains. Linear strain theory is therefore valid and
    #       additionally tan(gamma) = gamma. These assumptions are used
    #       computing the box after applying the stain.
    strained_box = jnp.dot(box_tensor, jnp.eye(box_tensor.shape[0]) + epsilon)
    energy = energy_fn(state.position, neighbor=neighbor, box=strained_box,
                       **kwargs)
    return energy


def init_sigma_born(energy_fn_template, reference_box=None):
    """Initialiizes a function that computes the Born contribution to the
    stress tensor.

    sigma^B_ij = d U / d epsilon_ij

    Can also be computed to compute the stress tensor at kbT = 0, when called
    on the state of minimum energy. This function requires that `energy_fn`
    takes a `box` keyword argument, usually alongside `periodic_general`
    boundary conditions.

    Args:
        energy_fn_template: A function that takes energy parameters as input
                            and returns an energy function
        ref_box_tensor: The transformation T of general periodic boundary
                        conditions. If None, box_tensor needs to be provided as
                        'box' during function call, e.g. for the NPT ensemble.

    Returns:
        A function that takes a simulation state with neighbor list,
        energy_params and box (if applicable) and returns the instantaneous
        Born contribution to the stress tensor.
    """
    def sigma_born(state, neighbor, energy_params, **kwargs):
        box, kwargs = _dyn_box(reference_box, state, **kwargs)
        spatial_dim = state.position.shape[-1]
        volume = quantity.volume(spatial_dim, box)
        epsilon0 = jnp.zeros((spatial_dim, spatial_dim))

        energy_fn = energy_fn_template(energy_params)
        sigma_b = jacrev(energy_under_strain)(
            epsilon0, energy_fn, box, state, neighbor, **kwargs)
        return sigma_b / volume
    return sigma_born


def init_stiffness_tensor_stress_fluctuation(energy_fn_template, reference_box):
    """Initializes all functions necessary to compute the elastic stiffness
    tensor via the stress fluctuation method in the NVT ensemble.

    The provided functions compute all necessary instantaneous properties
    necessary to compute the elastic stiffness tensor via the stress fluctuation
    method. However, for compatibility with DiffTRe, (weighted) ensemble
    averages need to be computed manually and given to the stiffness_tensor_fn
    for final computation of the stiffness tensor. For an example usage see
    the diamond notebook. The implementation follows the formulation derived by
    Van Workum et al., "Isothermal stress and elasticity tensors for ions and
    point dipoles using Ewald summations", PHYSICAL REVIEW E 71, 061102 (2005).

     # TODO provide sample usage

    Args:
        energy_fn_template: A function that takes energy parameters as input
                            and returns an energy function
        box_tensor: The transformation T of general periodic boundary
                    conditions. As the stress-fluctuation method is only
                    applicable to the NVT ensemble, the box_tensor needs to be
                    provided here as a constant, not on-the-fly.
        kbt: Temperature in units of the Boltzmann constant
        n_particles: Number of particles in the box

    Returns: A tuple of 3 functions:
        born_term_fn: A function computing the Born contribution to the
                      stiffness tensor for a single snapshot
        sigma_born: A function computing the Born contribution to the stress
                    tensor for a single snapshot
        sigma_tensor_prod: A function computing sigma^B_ij * sigma^B_kl given
                           a trajectory of sigma^B_ij
        stiffness_tensor_fn: A function taking ensemble averages of C^B_ijkl,
                             sigma^B_ij and sigma^B_ij * sigma^B_kl and
                             returning the resulting stiffness tensor.
    """
    # TODO this function simplifies a lot if split between per-snapshot
    #  and per-trajectory functions
    # spatial_dim = reference_box.shape[-1]
    # volume = quantity.volume(spatial_dim, reference_box)
    # epsilon0 = jnp.zeros((spatial_dim, spatial_dim))

    def born_term_fn(state, neighbor, energy_params, **kwargs):
        """Born contribution to the stiffness tensor:
        C^B_ijkl = d^2 U / d epsilon_ij d epsilon_kl
        """
        # check if box is passed in dynamic kwargs and use it if provided, else use reference box
        box, kwargs = _dyn_box(reference_box, state, **kwargs)
        spatial_dim = state.position.shape[-1]
        volume = quantity.volume(spatial_dim, box)
        epsilon0 = jnp.zeros((spatial_dim, spatial_dim))

        energy_fn = energy_fn_template(energy_params)
        born_stiffness_contribution = jax.hessian(energy_under_strain)(
            epsilon0, energy_fn, box, state, neighbor, **kwargs
        )

        return born_stiffness_contribution / volume

    return born_term_fn
