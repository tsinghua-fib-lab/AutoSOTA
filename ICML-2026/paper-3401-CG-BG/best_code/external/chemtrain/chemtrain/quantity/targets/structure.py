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

"""Initializes structural targets."""

__all__ = (
    "init_radial_distribution_target",
    "init_angular_distribution_target",
    "init_dihedral_distribution_target",
    "init_tetrahedral_order_coefficient",
    "initialize_local_structure_index",
    "initialize_rmsd"
)

from scipy import interpolate
from jax_md import util
import jax.numpy as jnp

from typing import Union, List

from jax_md_mod import custom_quantity
from chemtrain.typing import ArrayLike
from chemtrain.quantity import observables

from.util import target_quantity, TargetInit

def init_radial_distribution_target(target: Union[ArrayLike, List[ArrayLike]],
                                    rdf_start: float = 0.0,
                                    rdf_cut: float = 1.5,
                                    nbins: int = 300,
                                    gamma: float = 1.0,
                                    rdf_species: ArrayLike = None,
                                    ) -> TargetInit:
    """Initializes a RDF target.

    Args:
        target: Target rdf with r-values and distribution.
        rdf_start: Start of the target distribution.
        rdf_cut: End of the target distribution.
        nbins: Number of bins of the target distribution.
        gamma: Scaling factor of the loss contribution.
        rdf_species: Compute the RDF between selected species
    """

    bin_cent, bin_bound, sigma = custom_quantity.rdf_discretization(
        rdf_cut, nbins, rdf_start)

    if rdf_species is None:
        rdf_spline = interpolate.interp1d(
            target[..., 0], target[..., 1], kind='cubic')
        reference_rdf = util.f32(rdf_spline(bin_cent))
    else:
        # Load multiple targets but transfer them to the same format
        reference_rdf = jnp.zeros((len(target), bin_cent.size))
        for idx in range(len(target)):
            rdf_spline = interpolate.interp1d(
                target[idx][:, 0], target[idx][:, 1], kind='cubic')
            reference_rdf = reference_rdf.at[idx, :].set(
                util.f32(rdf_spline(bin_cent))
            )

    rdf_struct = custom_quantity.RDFParams(
        reference_rdf, bin_cent, bin_bound, sigma)

    @target_quantity(['displacement_fn'], ['reference_box'])
    def initialize(key, compute_fns, init_args):

        compute_fn = custom_quantity.init_rdf(
            rdf_params=rdf_struct, **init_args, rdf_species=rdf_species
        )

        target_dict = {
            'target': reference_rdf, 'gamma': gamma,
            'traj_fn': observables.init_traj_mean_fn(key)
        }

        return target_dict, compute_fn

    return initialize


def init_angular_distribution_target(target: Union[ArrayLike, List[ArrayLike]],
                                     r_outer: float = 0.318,
                                     r_inner: float = 0.0,
                                     nbins: int = 150,
                                     gamma: float = 1.0,
                                     adf_species: ArrayLike = None,
                                     ) -> TargetInit:
    """Initializes an ADF target.

    An ADF is a probability distribution of angles between particles in a
    solvation shell specified via `r_outer` and `r_inner`. Triplet contributions
    with distances not in `r_inner` and `r_outer` are smoothly masked out.

    Args:
        target: Target rdf with r-values and distribution.
        r_outer: Maximum pairwise distance between particles of a triplet.
        r_inner: Minimum pairwise distance between particles of a triplet.
        nbins: Number of bins of the target distribution.
        gamma: Scaling factor of the loss contribution.
        adf_species: Compute the ADF for triplets with different species
    """

    adf_bin_centers, sigma_adf = custom_quantity.adf_discretization(nbins)

    if adf_species is None:
        adf_spline = interpolate.interp1d(
            target[:, 0], target[:, 1], kind='cubic')
        reference_adf = util.f32(adf_spline(adf_bin_centers))
    else:
        # Load multiple targets but transfer them to the same format
        reference_adf = jnp.zeros((len(target), adf_bin_centers.size))
        for idx in range(len(target)):
            adf_spline = interpolate.interp1d(
                target[idx][:, 0], target[idx][:, 1], kind='cubic')
            reference_adf = reference_adf.at[idx, :].set(
                util.f32(adf_spline(adf_bin_centers))
            )

    adf_struct = custom_quantity.ADFParams(
        reference_adf, adf_bin_centers, sigma_adf, r_outer, r_inner)

    @target_quantity(['displacement_fn'], ['r_init', 'nbrs_init'])
    def initialize(key, compute_fns, init_args):
        del compute_fns

        compute_fn = custom_quantity.init_adf_nbrs(
            adf_params=adf_struct, smoothing_dr=sigma_adf, **init_args,
            adf_species=adf_species
        )

        target_dict = {
            'target': reference_adf, 'gamma': gamma,
            'traj_fn': observables.init_traj_mean_fn(key)
        }

        return target_dict, compute_fn

    return initialize


def init_dihedral_distribution_target(target: ArrayLike,
                                      bonds: ArrayLike,
                                      nbins: int = 150,
                                      degree: bool = True,
                                      smoothing: str = 'gaussian',
                                      gamma: float = 1.0,
                                      approximate_average: bool = False
                                      ) -> TargetInit:
    """Initializes reference and compute function for a dihedral angle dist.

    Args:
        key: Target key
        target: Target distribution. Array of bin centers and corresponding
            probabilities.
        bonds: Indices of the sites enclosing the dihedral angle.
        degree: Whether the reference distribution is given in degree.
        smoothing: Method to smooth the predicted distribution.
        nbins: The number of bins of the target distribution.
        gamma: Scaling factor in the loss function.
        approximate_average: Use a linearized approximation of the ensemble
            average in the reweighting context.

    Returns:
        Returns a target dict and a compute function to be passed to the
        trainers.

    """

    bin_centers, sigma = custom_quantity.dihedral_discretization(nbins)
    bin_boundaries = 0.5 * (bin_centers[0:-1] + bin_centers[1:])
    bin_boundaries = jnp.concatenate(
        (jnp.asarray([-jnp.pi]), bin_boundaries, jnp.asarray([jnp.pi])))

    dihedrals, probs = target
    # Rescale to rad
    if degree:
        dihedrals *= jnp.pi / 180.
        probs /= jnp.pi / 180.

    # Interpolate and ensure normalization of distribution
    dihedral_spline = interpolate.interp1d(dihedrals, probs, kind='cubic')
    dihedral_dist = util.f32(dihedral_spline(bin_centers))
    dihedral_dist /= util.high_precision_sum(dihedral_dist) * 2.0 * jnp.pi

    dihedral_struct = custom_quantity.BondDihedralParams(
        reference=dihedral_dist, sigma=sigma, bonds=bonds,
        bin_centers=bin_centers, bin_boundaries=bin_boundaries
    )

    @target_quantity(['displacement_fn'], [])
    def initialize(key, compute_fns, init_args):
        del compute_fns

        compute_fn = custom_quantity.init_bond_dihedral_distribution(
            bond_dihedral_params=dihedral_struct, smoothing=smoothing,
            **init_args)

        if approximate_average:
            traj_fn = observables.init_linear_traj_mean_fn(key)
        else:
            traj_fn = observables.init_traj_mean_fn(key)

        target_dict = {
            'target': dihedral_dist, 'gamma': gamma,
            'traj_fn': traj_fn
        }

        return target_dict, compute_fn
    return initialize


def init_tetrahedral_order_coefficient(target: ArrayLike = None,
                                       gamma: float = 1.0,
                                       linearized: bool = False):
    """Initializes the compute function of the tetrahedral order parameter q.

    Args:
        target: Target order parameter. If `target=None`, the parameter is only
            computed but not present as a target.
        gamma: Coefficient of the target in the loss function.

    """

    @target_quantity(['displacement_fn'], [])
    def initialize(key, compute_fns, init_args):
        if target is None:
            target_dict = None
        else:
            if linearized:
                target_dict = {
                    'target': target, 'gamma': gamma,
                    'traj_fn': observables.init_traj_mean_fn(key)
                }
            else:
                target_dict = {
                    'target': target, 'gamma': gamma,
                    'traj_fn': observables.init_linear_traj_mean_fn(key)
                }

        compute_fns = custom_quantity.init_tetrahedral_order_parameter(
            displacement_fn=init_args['displacement_fn'])

        return target_dict, compute_fns
    return initialize

def initialize_local_structure_index(target: ArrayLike = None,
                                     r_cut: float = 0.37,
                                     gamma: float = 1.0,
                                     linearized: bool = False):
    """Initializes the local structure index (LSI).

    Args:
        target: Target order parameter. If `target=None`, the parameter is only
            computed but not present as a target.
        r_cut: Cutoff of the second solvation shell.
        gamma: Coefficient of the target in the loss function.

    """

    @target_quantity(['displacement_fn'], ['reference_box', 'r_init'])
    def initialize(key, compute_fns, init_args):
        if target is None:
            target_dict = None
        else:
            if linearized:
                target_dict = {
                    'target': target, 'gamma': gamma,
                    'traj_fn': observables.init_linear_traj_mean_fn(key)
                }
            else:
                target_dict = {
                    'target': target, 'gamma': gamma,
                    'traj_fn': observables.init_traj_mean_fn(key)
                }

        compute_fns = custom_quantity.init_local_structure_index(
            r_cut=r_cut, **init_args)

        return target_dict, compute_fns
    return initialize


def initialize_rmsd(reference_positions: ArrayLike,
                    idx: ArrayLike = None,
                    weights: ArrayLike = None,
                    target: ArrayLike = None,
                    gamma: float = 1.0):
    """Initializes the rmsd to a reference structure.

    Args:
        reference_positions: Coordinates of the reference system.
        idx: Indices selecting the relevant particles.
        weights: Non-normalized weights for a weighted rmsd computation.
        target: Target order parameter. If `target=None`, the parameter is only
            computed but not present as a target.
        gamma: Coefficient of the target in the loss function.

    """

    @target_quantity(['displacement_fn', 'reference_box'], [])
    def initialize(key, compute_fns, init_args):
        if target is not None:
            raise NotImplementedError("A target is currently not supported.")

        compute_fns = custom_quantity.init_rmsd(
            reference_positions, weights=weights, idx=idx, **init_args)

        return None, compute_fns
    return initialize
