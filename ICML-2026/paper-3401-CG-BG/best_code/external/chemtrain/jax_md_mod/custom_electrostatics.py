# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
# Copyright 2022 Google LLC
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

"""Custom definition of some electrostatic interactions."""

import functools
from typing import Union, Iterable, Callable, Optional, Any

import jax
import jax.numpy as jnp
import jax.scipy as jsp

import numpy as onp

from jax_md import energy, smap, space, util as md_util, quantity, partition
from jax_md._energy import electrostatics


def shielded_self(charge, radii):
    """Self-interaction potential of gaussian charge."""
    return jnp.sum(charge * charge / (2 * radii * jnp.sqrt(jnp.pi)))


def shielded_interaction(dr, charge, alpha, alpha_max=None):
    r"""Gaussian (shielded) charge interaction.

    The shielded interaction between gaussian charges is given by

    .. math::
        V(r) = q_1 q_2 \frac{\operatorname{erf}(\alpha r) - \operatorname{erf}(\alpha_\mathrm{max} r)}{r},

    where $q_1$ and $q_2$ are the charges of the particles, r is the
    distance, $\\alpha = 1 / \\sqrt{2(\\gamma_1^2 + \\gamma_2^2)}$ depends on
    the width of the charges, and $\alpha_{max}$ defines the strength of
    shielding, e.g., for Ewald summation or the PME method.

    Args:
        dr: Distance between particles
        charge: Charge of the particles
        alpha: Shielding parameter for pairs of charges.
        alpha_max: Shielding of interaction.

    Returns:
        Returns the potential energy.
    """

    # Safety: Avoid division by zero
    mask = dr > 1e-7
    dr = jnp.where(mask, dr, 1e-7)

    if alpha_max is not None:
        erfdiff = jsp.special.erf(alpha * dr) - jsp.special.erf(alpha_max * dr)
        pot = mask * charge * erfdiff / dr
    else:
        pot = mask * charge * jsp.special.erf(alpha * dr) / dr

    return pot


def shielded_interaction_neighbor_list(displacement_fn: space.DisplacementFn,
                                       r_onset: float,
                                       r_cutoff: float,
                                       box: jnp.ndarray = None,
                                       alpha: float = 4.5,
                                       grid: Union[int, Iterable[int]] = None,
                                       method: str = "direct",
                                       fractional_coordinates: bool = True
                                       ) -> Callable[[...], jnp.ndarray]:
    """Gaussian (shielded) charge interaction.

    Applies the shielded interaction between gaussian charges using a
    neighborlist. The total interaction consists of a direct pairwise
    contribution in :func:`shielded_interaction`, a self interaction
    in :func:`shielded_self`, and a (optionally) reciprocal space contribution
    set by the ``method`` argument.

    Args:
        displacement_fn: Displacement function.
        r_onset: Onset of the real space interaction truncation.
        r_cutoff: Cutoff of the real space interaction.
        box: Simulation box, required by the reciprocal space methods.
        alpha: Shielding parameter for pairs of charges. Controls the
            tradeoff between reciprocal and real space contributions.
        grid: Grid dimensions for reciprocal space, can be an integer for
            an equal number of grid points in each dimension or a list.
        method: Method to compute the reciprocal space contribution.
            If "direct", all interactions are computed in real space.
            If "ewald", the Ewald summation is used, if "pme", the PME method
            is used (more efficient).
        fractional_coordinates: Whether positions are given in fractional
            coordinates.

    Returns:
        Returns a function to compute the total electrostatic energy of a
        system of Gaussian charges.

    """

    def energy_fn(position: jnp.ndarray,
                  neighbor: partition.NeighborList,
                  charge: jnp.ndarray,
                  radii: jnp.ndarray,
                  chi: jnp.ndarray = None,
                  idmp: jnp.ndarray = None,
                  equilibrate: bool = True,
                  precondition: bool = False,
                  **dynamic_kwargs):

        if method == "direct":
            _energy_fn = smap.pair_neighbor_list(
                energy.multiplicative_isotropic_cutoff(
                    shielded_interaction, r_onset, r_cutoff
                ),
                space.metric(displacement_fn),
                charge=(lambda q1, q2: q1 * q2, charge),
                alpha=(lambda s1, s2: 1 / jnp.sqrt(2 * (s1 ** 2 + s2 ** 2)), radii),
            )
            pot = 0.0
        elif method in ["ewald", "pme"]:
            _box = dynamic_kwargs.get("box", box)

            assert _box is not None, "Box must be provided for reciprocal space calculation."

            if method == "ewald":
                recip_fn = lambda pos, charge, **kwargs: custom_coulomb_recip_ewald(
                    charge, _box, alpha, grid=grid,
                    fractional_coordinates=fractional_coordinates
                )(pos, **kwargs)
            else:
                recip_fn = lambda pos, charge, **kwargs: custom_coulomb_recip_pme(
                    charge, _box, grid=grid,
                    fractional_coordinates=fractional_coordinates, alpha=alpha
                )(pos, **kwargs)

            _energy_fn = smap.pair_neighbor_list(
                energy.multiplicative_isotropic_cutoff(
                    shielded_interaction, r_onset, r_cutoff
                ),
                space.metric(displacement_fn),
                charge=(lambda q1, q2: q1 * q2, charge),
                alpha=(lambda s1, s2: 1 / jnp.sqrt(2 * (s1 ** 2 + s2 ** 2)), radii),
                alpha_max=alpha
            )

            pot = 0.0
            if not precondition:
                pot += recip_fn(position, charge, **dynamic_kwargs)
                pot -= shielded_self(charge, 1 / (2 * alpha)) # Correct for the self-interaction added in reciprocal space

        else:
            raise NotImplementedError(f"Unknown method {method}")

        pot += shielded_self(charge, radii)
        pot += _energy_fn(position, neighbor)

        # Add electronegativity and hardness terms
        if chi is not None and idmp is not None:
            if equilibrate:
                pot += core_interaction(charge, chi, idmp)
            else:
                pot += core_interaction(charge, chi, idmp)

        return pot

    return energy_fn


def core_interaction(charge, chi, idmp):
    """Interaction between partial charge and atom core."""
    return jnp.sum((chi + charge * idmp / 2) * charge)


def structure_factor(g, R, q=1, mask=None):
    """Computes the complex structure factor.

    Adapted from: https://github.com/jax-md/jax-md/blob/main/jax_md/_energy/electrostatics.py
    """
    if mask is None:
        mask = jnp.ones(R.shape[0], dtype=bool)

    if isinstance(q, jnp.ndarray):
        q = q[None, :]
    return md_util.high_precision_sum(
        q * jnp.exp(1j * jnp.einsum('id,jd->ij', g, R)) * mask,
        axis=1
    )


def custom_coulomb_recip_ewald(charge: jnp.ndarray,
                               box: jnp.ndarray,
                               alpha: float,
                               grid: Union[int, Iterable[int]],
                               fractional_coordinates=False
                               ) -> Callable[[jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray]:
    """Ewald summation for Coulomb interactions.

    Modified implementation of :func:`jax_md.energy.coulomb_recip_ewald`
    to specify wavevectors via grid dimensions.
    Adapted from https://github.com/jax-md/jax-md/blob/main/jax_md/_energy/electrostatics.py.

    Args:
        charge: Charges of the particles.
        box: Simulation box, required if fractional coordinates are used.
        alpha: Shielding parameter.
        grid: Grid dimensions for reciprocal space, can be an integer for
            an equal number of grid points in each dimension or a list.
        fractional_coordinates: Whether positions are given in fractional
            coordinates.

    Returns:
        Returns a function to compute the reciprocal part of the coulomb
        interactions using the Ewald summation.

    """

    def energy_fn(position, mask=None, **kwargs):
        n_particles, dim = position.shape

        _box = kwargs.get("box", box)
        if mask is None:
            mask = jnp.ones(n_particles, dtype=bool)

        # Create a grid of reciprocal vectors
        if jnp.isscalar(_box) or jnp.shape(_box) == ():
            _box = jnp.eye(dim) * _box
        elif jnp.ndim(box) == 1:
            _box = jnp.diag(_box)
        else:
            assert jnp.shape(_box) == (dim, dim)

        volume = quantity.volume(dim, _box)
        _invbox = jnp.linalg.inv(_box)

        if fractional_coordinates:
            position = space.transform(_box, position)

        # Non-homogeneous grid dimension
        if isinstance(grid, int):
            _grid = [grid] * dim
        else:
            _grid = grid

        # Create a grid with the specified dimensions but omit all-zero wave vectors
        m = jnp.meshgrid(*(jnp.arange(g) for g in _grid), indexing='ij')
        g = jnp.stack([m[i].ravel() for i in range(dim)], axis=-1)[1:, :]

        # Inverse box gives reciprocal lattice vectors as rows
        g = 2 * jnp.pi * jnp.einsum('ji,nj->ni', _invbox, g)
        g2 = jnp.sum(g ** 2, axis=-1)

        # Compute the structure factors
        S = structure_factor(g, position, charge, mask=mask)
        S2 = jnp.real(jnp.conj(S) * S)

        # Double the sum due to purely positive wave vectors
        pot = jnp.exp(-g2 / (4 * alpha ** 2)) / g2 * S2
        pot = 4 * jnp.pi / volume * jnp.sum(pot)

        return pot
    return energy_fn


def custom_coulomb_recip_pme(charge: jnp.ndarray,
                             box: jnp.ndarray,
                             grid: Union[int, Iterable[int]],
                             fractional_coordinates: bool = False,
                             alpha: float = 0.34
                             ) -> Callable[[jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray]:
    """Particle Mesh Ewald method for Coulomb interactions.

    Adapted implementation from https://github.com/jax-md/jax-md/blob/main/jax_md/_energy/electrostatics.py.

    Args:
        charge: Charges of the particles.
        box: Simulation box, required if fractional coordinates are used.
        grid: Grid dimensions for reciprocal space, can be an integer for
            an equal number of grid points in each dimension or a list.
        fractional_coordinates: Whether positions are given in fractional
        alpha: Shielding parameter of the screening charges.

    Returns:
        Returns a function to compute the reciprocal part of the coulomb
        interactions using the SPME method.

    """

    _ibox = space.inverse(box)
    _grid = grid

    def energy_fn(R, **kwargs):
        q = kwargs.pop('charge', charge)
        _box = kwargs.pop('box', box)
        ibox = space.inverse(kwargs['box']) if 'box' in kwargs else _ibox

        dim = R.shape[-1]
        if isinstance(_grid, int):
            grid_dimensions = [_grid] * dim
        else:
            grid_dimensions = _grid

        grid_dimensions = onp.asarray(grid_dimensions)

        grid = electrostatics.map_charges_to_grid(R, q, ibox, grid_dimensions,
                                                  fractional_coordinates)
        Fgrid = jnp.fft.fftn(grid)

        mx, my, mz = jnp.meshgrid(*[jnp.fft.fftfreq(g) for g in grid_dimensions])
        if jnp.isscalar(_box):
            m_2 = (mx**2 + my**2 + mz**2) * (grid_dimensions[0] * ibox)**2
            V = _box**dim
        else:
            m = (ibox[None, None, None, 0] * mx[:, :, :, None] * grid_dimensions[0] +
                 ibox[None, None, None, 1] * my[:, :, :, None] * grid_dimensions[1] +
                 ibox[None, None, None, 2] * mz[:, :, :, None] * grid_dimensions[2])
            m_2 = jnp.sum(m**2, axis=-1)
            V = jnp.linalg.det(_box)
        mask = m_2 != 0

        exp_m = 1 / (2 * jnp.pi * V) * jnp.exp(-jnp.pi**2 * m_2 / alpha**2) / m_2
        return md_util.high_precision_sum(
            mask * exp_m * electrostatics.B(mx, my, mz) * jnp.abs(Fgrid)**2)
    return energy_fn


def charge_eq_energy_neighborlist(displacement: space.DisplacementFn,
                                  r_onset: float,
                                  r_cutoff: float,
                                  solver: str = "direct",
                                  method: str = "direct",
                                  grid: Union[int, Iterable[int]] = None,
                                  alpha: float = 4.5,
                                  fractional_coordinates: bool = True,
                                  box: jnp.ndarray = None):
    """Charge equilibration energy function.

    Distributes charges globally to minimize the electrostatic energy and
    core-interaction energy respecting charge conservation [#Rappe1991]_.

    Args:
        displacement: Displacement function.
        r_onset: Onset of the real space interaction truncation.
        r_cutoff: Cutoff of the real space interaction.
        solver: Method to solve the linear system. Can be "direct" or "CG".
        method: Method to compute long-range electrostatic interactions.
            See :func:`shielded_interaction_neighbor_list`.
        grid: Grid dimensions for reciprocal space.
            See :func:`shielded_interaction_neighbor_list`.
        alpha: Shielding parameter used by the long-range method.
            See :func:`shielded_interaction_neighbor_list`.
        fractional_coordinates: Whether positions are given in fractional
            coordinates.
        box: Simulation box, required by the reciprocal space methods.

    Returns:
        Returns a function to compute the total electrostatic energy of a
        system of Gaussian charges, given that the charges minimize the
        electrostatic energy and the core-interaction energy.

    References:
       .. [#Rappe1991] Rappe, A. K.; Goddard, W. A. I. Charge Equilibration for
           Molecular Dynamics Simulations. J. Phys. Chem. **1991**, 95 (8),
           3358–3363. https://doi.org/10.1021/j100161a070.

    """

    total_energy_fn = shielded_interaction_neighbor_list(
        displacement, r_onset, r_cutoff, method=method, grid=grid,
        alpha=alpha, fractional_coordinates=fractional_coordinates, box=box
    )

    def energy_fn(position, neighbor, radii=None, chi=None, idmp=None, mask=None, total_charge=None, charge=None, **dynamic_kwargs):
        if mask is None:
            mask = jnp.ones(position.shape[0], dtype=bool)
        if total_charge is None:
            total_charge = 0.0
            print(f"No total charge specified. Total charge will be set to {total_charge}")
        else:
            print(f"Total charge specified: {total_charge}")

        # Evaluate for precomputed charges
        if charge is not None:
            return total_energy_fn(
            position, neighbor, charge=charge, radii=radii, chi=chi, idmp=idmp,
            equilibrate=False, **dynamic_kwargs
        )

        n_particles = mask.size

        charge = jnp.zeros(n_particles)

        if solver == "direct":
            # Count number of particles
            A = jnp.zeros((n_particles + 1, n_particles + 1))

            # Set last row (charge neutrality) to mask
            A = A.at[-1, :-1].set(mask)

            # Set row for muliplier to mask
            A = A.at[:-1, -1].set(mask)

            # Set diagonal entries to hessian. As the charges minimize the
            # energy, the gradient of the coloumb interactions depend on the
            # position only explicitly but not through the charges.

            A = A.at[:-1, :-1].set(
                jax.hessian(total_energy_fn, argnums=2)(
                    position, neighbor, charge, radii,
                    chi, idmp, **dynamic_kwargs
                )
            )

            # Charge neutrality constraint (for now)
            b = jnp.concatenate((-chi, jnp.full((1,), total_charge))).reshape((-1, 1))

            # Solve the linear system with lagrange multipliers
            charges = jsp.linalg.solve(A, b, assume_a="sym")[:-1, 0]
        else:
            raise ValueError(f"Unknown method {solver} to equilibrate charges.")

        charges = jnp.where(mask, charges, 0.0)

        # Do not include core-interaction energy
        qeq_energy = total_energy_fn(
            position, neighbor, charge=charges, radii=radii, chi=chi, idmp=idmp,
            equilibrate=False, **dynamic_kwargs
        )

        return qeq_energy, charges

    return energy_fn
