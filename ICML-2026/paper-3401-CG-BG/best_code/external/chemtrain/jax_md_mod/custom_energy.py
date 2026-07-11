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

"""Custom definition of some potential energy functions."""
from functools import partial
from typing import Callable, Any

from jax import vmap
import jax.numpy as jnp
from jax_md import space, partition, util, energy, smap

from jax_md_mod import custom_interpolate, custom_quantity
from jax_md_mod.model import sparse_graph

# Types
f32 = util.f32
f64 = util.f64
Array = util.Array

PyTree = Any
Box = space.Box
DisplacementFn = space.DisplacementFn
DisplacementOrMetricFn = space.DisplacementOrMetricFn

NeighborFn = partition.NeighborFn
NeighborList = partition.NeighborList


def stillinger_weber_energy(dr,
                            d_vect,
                            mask=None,
                            a=7.049556277,
                            b=0.6022245584,
                            p=4,
                            lam=21.0,
                            epsilon=2.16826,
                            gamma=1.2,
                            sigma=2.0951,
                            cutoff=1.8*2.0951,
                            three_body_strength=1.0):
    """Computes the stiling weber potential.

    The Stillinger-Weber (SW) potential [#Stillinger]_ is commonly used to model
    silicon and similar systems. This function uses the default SW parameters
    from the original paper. The SW potential was originally proposed to
    model diamond in the diamond crystal phase and the liquid phase, and is
    known to give unphysical amorphous configurations [#Holender]_ [#Barkema]_.
    For this reason,
    we provide a three_body_strength parameter. Changing this number to $1.5$
    or $2.0$ has been known to produce more physical amorphous phase, preventing
    most atoms from having more than four nearest neighbors. Note that this
    function currently assumes nearest-image-convention.


    References:
       .. [#Stillinger] Stillinger, Frank H., and Thomas A. Weber.
          "Computer simulation of local order in condensed phases of silicon."
          Physical review B 31.8 (1985): 5262.
       .. [#Holender] Holender, J. M., and G. J. Morgan.
          "Generation of a large structure (105 atoms) of amorphous Si using
          molecular dynamics."
          Journal of Physics: Condensed Matter 3.38 (1991): 7241.
       .. [#Barkema] Barkema, G. T., and Normand Mousseau.
          "Event-based relaxation of continuous disordered systems."
          Physical review letters 77.21 (1996): 4358.

    Args:
        dr: A ndarray of pairwise distances between particles
        d_vect: An ndarray of pairwise displacements between particles
        a: A scalar that determines the scale of two-body term
        b: Factor for radial power term
        p: Power in radial interaction
        lam: A scalar that determines the scale of the three-body term
        epsilon: A scalar that sets the energy scale
        gamma: Exponential scale in three-body term
        sigma: A scalar that sets the length scale
        cutoff: Cut-off value defined as sigma * a
        three_body_strength: A scalar that determines the relative strength
                             of the angular interaction
        mask: ndarray of size dr masking non-existing neighbors in neighborlist
              (if applicable)
    Returns:
        The Stilinger-Weber energy for a snapshot.
    """

    # initialize
    if mask is None:
        n_particels = dr.shape[0]
        mask = jnp.ones([n_particels, n_particels])
        angle_mask = jnp.ones([n_particels, n_particels, n_particels])
    else:  # for neighborlist input
        max_neighbors = mask.shape[-1]
        angle_mask1 = jnp.tile(jnp.expand_dims(mask, 1), [1, max_neighbors, 1])
        angle_mask2 = jnp.tile(jnp.expand_dims(mask, -1), [1, 1, max_neighbors])
        angle_mask = angle_mask1 * angle_mask2
    sw_radial_interaction = partial(energy._sw_radial_interaction, sigma=sigma,
                                    p=p, b=b, cutoff=cutoff)
    sw_angle_interaction = partial(energy._sw_angle_interaction, gamma=gamma,
                                   sigma=sigma, cutoff=cutoff)
    sw_three_body_term = vmap(vmap(vmap(sw_angle_interaction, (0, None)),
                                   (None, 0)), 0)

    # compute SW energy
    radial_interactions = sw_radial_interaction(dr) * mask
    angular_interactions = sw_three_body_term(d_vect, d_vect) * angle_mask
    first_term = a * jnp.sum(radial_interactions) / 2.0
    second_term = lam * jnp.sum(angular_interactions) / 2.0
    return epsilon * (first_term + three_body_strength * second_term)


def stillinger_weber_pair(displacement,
                          a=7.049556277,
                          b=0.6022245584,
                          p=4,
                          lam=21.0,
                          epsilon=2.16826,
                          gamma=1.2,
                          sigma=2.0951,
                          cutoff=1.8*2.0951,
                          three_body_strength=1.0):
    """Convenience wrapper to compute stilinger-weber energy over a system with
    variable parameters.
    """

    def compute_fn(pos, **dynamic_kwargs):
        d = partial(displacement, **dynamic_kwargs)
        dvect = space.map_product(d)(pos, pos)  # N x N x3 displacement matrix
        dr = space.distance(dvect)  # N x N distances
        return stillinger_weber_energy(dr, dvect, None, a, b, p, lam, epsilon,
                                       gamma, sigma, cutoff,
                                       three_body_strength)
    return compute_fn


def stillinger_weber_neighborlist(displacement,
                                  box_size=None,
                                  a=7.049556277,
                                  b=0.6022245584,
                                  p=4,
                                  lam=21.0,
                                  epsilon=2.16826,
                                  gamma=1.2,
                                  sigma=2.0951,
                                  cutoff=1.8*2.0951,
                                  three_body_strength=1.0,
                                  dr_threshold=0.1,
                                  capacity_multiplier=1.25,
                                  initialize_neighbor_list=True):
    """Convenience wrapper to compute stilinger-weber energy using a neighbor
    list.
    """

    def energy_fn(pos, neighbor, **dynamic_kwargs):
        d = partial(displacement, **dynamic_kwargs)
        n_particles = pos.shape[0]
        mask = neighbor.idx != n_particles
        r_neigh = pos[neighbor.idx]
        dvect = space.map_neighbor(d)(pos, r_neigh)
        dr = space.distance(dvect)
        return stillinger_weber_energy(dr, dvect, mask, a, b, p, lam, epsilon,
                                       gamma, sigma, cutoff,
                                       three_body_strength)

    if initialize_neighbor_list:
        assert box_size is not None
        neighbor_fn = partition.neighbor_list(
            displacement, box_size, cutoff, dr_threshold,
            capacity_multiplier=capacity_multiplier)
        return neighbor_fn, energy_fn

    return energy_fn


def harmonic_angle(displacement_or_metric: DisplacementOrMetricFn,
                   angle_idxs: Array,
                   eq_mean: Array = None,
                   eq_variance: Array = None,
                   kbt: [float, Array] = None,
                   th0: Array = None,
                   kth: Array = None,
                   ):
    """Harmonic Angle interaction.

    The variance of the angle is used to determine the force constant.
    https://manual.gromacs.org/documentation/2019/reference-manual/functions/bonded-interactions.html

    Args:
        displacement_or_metric: Displacement function
        angle_idxs: Indices of particles (i, j, k)
        eq_mean: Equilibrium angle in degrees
        eq_variance: Angle Variance
        kbt: kbT

    Returns:
        Harmonic angle potential energy function.
    """

    angle_mask = jnp.ones([angle_idxs.shape[0], 1])

    if th0 is None:
        th0 = eq_mean
    if kth is None:
        kbt = jnp.array(kbt, dtype=f32)
        kth = kbt / eq_variance

    harmonic_fn = partial(energy.simple_spring, length=th0, epsilon=kth)

    def energy_fn(pos, **unused_kwargs):
        angles = sparse_graph.angle_triplets(pos, displacement_or_metric,
                                             angle_idxs, angle_mask)
        return jnp.sum(harmonic_fn(jnp.rad2deg(angles)))

    return energy_fn


def dihedral_energy(angle,
                    phase_angle: Array,
                    force_constant: Array,
                    n: [int, Array]):
    """Energy of dihedral angles.

    https://manual.gromacs.org/documentation/2019/reference-manual/functions/bonded-interactions.html
    """
    # Alternatively to varying the phase shift angle from 0 to + 180, we
    # allow choosing a negative force constant, which will have the same effect.
    cos_angle = jnp.cos(n * angle - phase_angle)
    energies = jnp.abs(force_constant) + force_constant * cos_angle
    return jnp.sum(energies)


def periodic_dihedral(displacement_or_metric: DisplacementOrMetricFn,
                      dihedral_idxs: Array,
                      phase_angle: Array,
                      force_constant: Array,
                      multiplicity: [float, Array]):
    """Peridoc dihedral angle interaction.

    https://manual.gromacs.org/documentation/2019/reference-manual/functions/bonded-interactions.html

    Args:
        displacement_or_metric: Displacement function
        dihedral_idxs: Indices of particles (i, j, k, l) building the dihedrals
        phase_angle: Dihedral phase angle in degrees.
        force_constant: Force constant
        multiplicity: Dihedral multiplicity

    Returns:
        Peridoc dihedral potential energy function.
    """

    multiplicity = jnp.array(multiplicity, dtype=f32)
    phase_angle = jnp.deg2rad(phase_angle)

    def energy_fn(pos, **unused_kwargs):
        dihedral_angles = custom_quantity.dihedral_displacement(
            pos, displacement_or_metric, dihedral_idxs, degrees=False)
        per_angle_u = vmap(dihedral_energy)(dihedral_angles, phase_angle,
                                            force_constant, multiplicity)
        return jnp.sum(per_angle_u)

    return energy_fn


def truncated_lennard_jones(dr: Array,
                            sigma: Array = 1.,
                            epsilon: Array = 1.,
                            exp: Array = 12.,
                            **unused_dynamic_kwargs) -> Array:
    """Lennard Jones interaction truncated and shifted at the minimum.

    Args:
      dr: An ndarray of pairwise distances between particles.
      sigma: Repulsion length scale
      epsilon: Interaction energy scale
      exp: Exponent specifying interaction stiffness

    Returns:
      Array of energies
    """
    del unused_dynamic_kwargs
    dr = jnp.where(dr > 1.e-7, dr, 1.e7)  # save masks dividing by 0
    r_min = 2.0 ** (2. / exp) * sigma
    dr = jnp.where(dr > r_min, r_min, dr) # truncate at the minimum
    idr = (sigma / dr)
    pot_energy = 4 * epsilon * (idr ** exp - idr ** (exp / 2.))
    pot_energy += epsilon # shift
    return pot_energy


def truncated_lennard_jones_neighborlist(
        displacement_or_metric: DisplacementOrMetricFn,
        box_size: Box = None,
        species: Array = None,
        sigma: Array = 1.0,
        epsilon: Array = 1.0,
        exp: [int, Array] = 12.,
        dr_threshold: float = 0.2,
        per_particle: bool = False,
        capacity_multiplier: float = 1.25,
        initialize_neighbor_list: bool = True):
    """Convenience wrapper to compute generic repulsion energy over a system
    with neighborlist.

    Provides option not to initialize neighborlist. This is useful if energy
    function needs to be initialized within a jitted function.
    """
    if isinstance(sigma, tuple):
        sigma = (sigma[0], jnp.array(sigma[1], f32))
    if isinstance(epsilon, tuple):
        epsilon = (epsilon[0], jnp.array(epsilon[1], f32))

    energy_fn = smap.pair_neighbor_list(
      truncated_lennard_jones,
      space.canonicalize_displacement_or_metric(displacement_or_metric),
      species=species,
      sigma=sigma,
      epsilon=epsilon,
      exp=exp,
      reduce_axis=(1,) if per_particle else None)

    if initialize_neighbor_list:
        assert box_size is not None
        neighbor_fn = partition.neighbor_list(
            displacement_or_metric, box_size, r_cutoff, dr_threshold,
            capacity_multiplier=capacity_multiplier
        )
        return neighbor_fn, energy_fn

    return energy_fn


def generic_repulsion(dr: Array,
                      sigma: Array = 1.,
                      epsilon: Array = 1.,
                      exp: Array = 12.,
                      **unused_dynamic_kwargs) -> Array:
    """
    Repulsive interaction between soft sphere particles:
    U = epsilon * (sigma / r)**exp.

    Args:
      dr: An ndarray of pairwise distances between particles.
      sigma: Repulsion length scale
      epsilon: Interaction energy scale
      exp: Exponent specifying interaction stiffness

    Returns:
      Array of energies
    """
    del unused_dynamic_kwargs
    dr = jnp.where(dr > 1.e-7, dr, 1.e7)  # save masks dividing by 0
    idr = (sigma / dr)
    pot_energy = epsilon * idr ** exp
    return pot_energy


def generic_repulsion_pair(
        displacement_or_metric: DisplacementOrMetricFn,
        species: Array = None,
        sigma: Array = 1.0,
        epsilon: Array = 1.0,
        exp: Array = 12.,
        r_onset: Array = 2.0,
        r_cutoff: Array = 2.5,
        per_particle: bool = False):
    """Convenience wrapper to compute generic repulsion energy over a system."""
    sigma = jnp.array(sigma, dtype=f32)
    epsilon = jnp.array(epsilon, dtype=f32)
    exp = jnp.array(exp, dtype=f32)
    r_onset = jnp.array(r_onset, dtype=f32)
    r_cutoff = jnp.array(r_cutoff, dtype=f32)

    return smap.pair(
        energy.multiplicative_isotropic_cutoff(generic_repulsion, r_onset,
                                               r_cutoff),
        space.canonicalize_displacement_or_metric(displacement_or_metric),
        species=species,
        sigma=sigma,
        epsilon=epsilon,
        exp=exp,
        reduce_axis=(1,) if per_particle else None)


def generic_repulsion_neighborlist(
        displacement_or_metric: DisplacementOrMetricFn,
        box_size: Box = None,
        species: Array = None,
        sigma: Array = 1.0,
        epsilon: Array = 1.0,
        exp: [int, Array] = 12.,
        r_onset: Array = 0.9,
        r_cutoff: Array = 1.,
        dr_threshold: float = 0.2,
        per_particle: bool = False,
        capacity_multiplier: float = 1.25,
        initialize_neighbor_list: bool = True):
    """Convenience wrapper to compute generic repulsion energy over a system
    with neighborlist.

    Provides option not to initialize neighborlist. This is useful if energy
    function needs to be initialized within a jitted function.
    """
    if isinstance(sigma, tuple):
        sigma = (sigma[0], jnp.array(sigma[1], f32))
    if isinstance(epsilon, tuple):
        epsilon = (epsilon[0], jnp.array(epsilon[1], f32))
    exp = jnp.array(exp, dtype=f32)
    r_onset = jnp.array(r_onset, dtype=f32)
    r_cutoff = jnp.array(r_cutoff, dtype=f32)

    energy_fn = smap.pair_neighbor_list(
      energy.multiplicative_isotropic_cutoff(generic_repulsion, r_onset,
                                             r_cutoff),
      space.canonicalize_displacement_or_metric(displacement_or_metric),
      species=species,
      sigma=sigma,
      epsilon=epsilon,
      exp=exp,
      reduce_axis=(1,) if per_particle else None)

    if initialize_neighbor_list:
        assert box_size is not None
        neighbor_fn = partition.neighbor_list(
            displacement_or_metric, box_size, r_cutoff, dr_threshold,
            capacity_multiplier=capacity_multiplier
        )
        return neighbor_fn, energy_fn

    return energy_fn


def generic_repulsion_nonbond(displacement_or_metric: DisplacementOrMetricFn,
                              pair_idxs: Array,
                              sigma: Array = 1.,
                              epsilon: Array = 1.,
                              exp: Array = 12.) -> Callable[[Array], Array]:
    """Convenience wrapper to compute repulsive part of Lennard Jones energy of
    particles via connection idxs.

    Args:
        displacement_or_metric: Displacement_fn
        pair_idxs: Set of pair indices (i, j) defining repulsion pairs
        sigma: sigma
        epsilon: epsilon
        exp: LJ exponent

    Returns:
        Pairwise nonbonded repulsion potential energy function.
    """
    sigma = jnp.array(sigma, f32)
    epsilon = jnp.array(epsilon, f32)
    exp = jnp.array(exp, dtype=f32)

    return smap.bond(
            generic_repulsion,
            space.canonicalize_displacement_or_metric(displacement_or_metric),
            pair_idxs,
            ignore_unused_parameters=True,
            sigma=sigma,
            epsilon=epsilon,
            exp=exp)


def lorentz_berthelot(idxs, species, sigma_dict, epsilon_dict):
    """Applies the Lorentz-Berthelot rule to a indices and species array.

    The Lorentz-Berthelot rules [#wikipedia]_ calculate the $\\sigma$ and
    $\\epsilon$ epsilon values from a given dictonary.

    .. math::

        \\sigma_{ij} = \\frac{\\sigma_{ii} + \\sigma_{jj}}{2}
        \\epsilon_{ij} = \\sqrt{\\epsilon_{ii} * \\epsilon_{jj}}

    References:

       .. [#wikipedia] https://en.wikipedia.org/wiki/Combining_rules

    """
    pairs = species[idxs]
    u, inv = jnp.unique(pairs, return_inverse=True)

    sigma = jnp.array([sigma_dict[x] for x in u])[inv].reshape(pairs.shape)
    sigma = jnp.sum(sigma, axis=1) * 0.5

    epsilon = jnp.array([epsilon_dict[x] for x in u])[inv].reshape(pairs.shape)
    epsilon = jnp.sqrt(jnp.prod(epsilon, axis=1))
    return sigma, epsilon


def lennard_jones_nonbond(displacement_or_metric: DisplacementOrMetricFn,
                          pair_idxs: Array,
                          sigma: Array = 1.,
                          epsilon: Array = 1.) -> Callable[[Array], Array]:
    """Convenience wrapper to compute lennard jones energy of nonbonded
    particles.

    Args:
        displacement_or_metric: Displacement_fn
        pair_idxs: Set of pair indices (i, j) defining repulsion pairs
        sigma: sigma
        epsilon: epsilon

    Returns:
        Pairwise nonbonded repulsion potential energy function.
    """
    sigma = jnp.array(sigma, f32)
    epsilon = jnp.array(epsilon, f32)
    return smap.bond(
            energy.lennard_jones,
            space.canonicalize_displacement_or_metric(displacement_or_metric),
            pair_idxs,
            ignore_unused_parameters=True,
            sigma=sigma,
            epsilon=epsilon)


def customn_lennard_jones_neighbor_list(
        displacement_or_metric: DisplacementOrMetricFn,
        box_size: Box,
        species: Array = None,
        sigma: Array = 1.0,
        epsilon: Array = 1.0,
        r_onset: float = 2.0,
        r_cutoff: float = 2.5,
        dr_threshold: float = 0.5,
        per_particle: bool = False,
        capacity_multiplier: float = 1.25,
        initialize_neighbor_list: bool = True,
        fractional: bool = True,
        disable_cell_list: bool = False):
    """Convenience wrapper to compute lennard-jones using a neighbor list.
     Different implementation of the cutoff to disentable with energy_params.
     Option not to initialize neighbor list to allow jitable building of
     energy function for varying sigma and epsilon."""
    if isinstance(sigma, tuple):
        sigma = (sigma[0], jnp.array(sigma[1], f32))
    if isinstance(epsilon, tuple):
        epsilon = (epsilon[0], jnp.array(epsilon[1], f32))
    r_onset = jnp.array(r_onset, f32)
    r_cutoff = jnp.array(r_cutoff, f32)
    dr_threshold = jnp.array(dr_threshold, f32)

    energy_fn = smap.pair_neighbor_list(
        energy.multiplicative_isotropic_cutoff(energy.lennard_jones, r_onset,
                                               r_cutoff),
        space.canonicalize_displacement_or_metric(displacement_or_metric),
        species=species,
        sigma=sigma,
        epsilon=epsilon,
        reduce_axis=(1,) if per_particle else None)

    if initialize_neighbor_list:
        neighbor_fn = partition.neighbor_list(
            displacement_or_metric, box_size, r_cutoff, dr_threshold,
            capacity_multiplier=capacity_multiplier,
            fractional_coordinates=fractional,
            disable_cell_list=disable_cell_list)
        return neighbor_fn, energy_fn
    return energy_fn


def tabulated(dr: Array, spline: Callable[[Array], Array], **unused_kwargs
              ) -> Array:
    """
    Tabulated radial potential between particles given a spline function.

    Args:
        dr: An ndarray of pairwise distances between particles
        spline: A function computing the spline values at a given pairwise
                distance.

    Returns:
        Array of energies
    """

    return spline(dr)


def tabulated_pair(displacement_or_metric: DisplacementOrMetricFn,
                   x_vals: Array,
                   y_vals: Array,
                   degree: int = 3,
                   r_onset: Array = 0.9,
                   r_cutoff: Array = 1.,
                   species: Array = None,
                   per_particle: bool = False) -> Callable[[Array], Array]:
    """Convenience wrapper to compute tabulated energy over a system."""
    x_vals = jnp.array(x_vals, f32)
    y_vals = jnp.array(y_vals, f32)
    r_onset = jnp.array(r_onset, f32)
    r_cutoff = jnp.array(r_cutoff, f32)

    spline = custom_interpolate.MonotonicInterpolate(x_vals, y_vals)

    tabulated_partial = partial(tabulated, spline=spline)

    return smap.pair(
      energy.multiplicative_isotropic_cutoff(tabulated_partial, r_onset,
                                             r_cutoff),
      space.canonicalize_displacement_or_metric(displacement_or_metric),
      species=species,
      reduce_axis=(1,) if per_particle else None)


def tabulated_neighbor_list(displacement_or_metric: DisplacementOrMetricFn,
                            x_vals: Array,
                            y_vals: Array,
                            box_size: Box,
                            degree: int = 3,
                            r_onset: Array = 0.9,
                            r_cutoff: Array = 1.,
                            dr_threshold: Array = 0.2,
                            species: Array = None,
                            capacity_multiplier: float = 1.25,
                            initialize_neighbor_list: bool = True,
                            per_particle: bool = False,
                            fractional=True):
    """
    Convenience wrapper to compute tabulated energy using a neighbor list.

    Provides option not to initialize neighborlist. This is useful if energy
    function needs to be initialized within a jitted function.
    """

    x_vals = jnp.array(x_vals, f32)
    y_vals = jnp.array(y_vals, f32)
    box_size = jnp.array(box_size, f32)
    r_onset = jnp.array(r_onset, f32)
    r_cutoff = jnp.array(r_cutoff, f32)
    dr_threshold = jnp.array(dr_threshold, f32)

    # Note: cannot provide the spline parameters via kwargs because only
    #       per-particle parameters are supported
    spline = custom_interpolate.MonotonicInterpolate(x_vals, y_vals)

    tabulated_partial = partial(tabulated, spline=spline)

    energy_fn = smap.pair_neighbor_list(
      energy.multiplicative_isotropic_cutoff(tabulated_partial, r_onset,
                                             r_cutoff),
      space.canonicalize_displacement_or_metric(displacement_or_metric),
      species=species,
      reduce_axis=(1,) if per_particle else None)

    if initialize_neighbor_list:
        neighbor_fn = partition.neighbor_list(
            displacement_or_metric, box_size, r_cutoff, dr_threshold,
            capacity_multiplier=capacity_multiplier,
            fractional_coordinates=fractional)
        return neighbor_fn, energy_fn
    return energy_fn
