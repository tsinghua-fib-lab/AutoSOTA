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
from typing import Callable

from jax_md import simulate
from jax import random, numpy as jnp, jit
from jax_md import quantity, util
from jax_md.simulate import Array, ShiftFn, Simulator

static_cast = util.static_cast


def nvt_langevin_gsd(energy_or_force_fn: Callable[..., Array],
                     shift_fn: ShiftFn,
                     dt: float,
                     kT: float,
                     gamma: float=0.1,
                     zero_velocity: bool=False) -> Simulator:
  """Simulation in the NVT ensemble using the GSD/BAOB Langevin thermostat.

  Our implementation follows [#kieninger2022]

  Args:
    energy_or_force_fn: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`. Both
      `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    kT: Floating point number specifying the temperature in units of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
    gamma: A float specifying the friction coefficient between the particles
      and the solvent.
    zero_velocity: A boolean specifying whether velocities for the particles
      should be initialized to zero.
  Returns:
    See above.

  References:
    .. [#kiening2022] Stefanie Kieninger and Bettina G. Keller
       Journal of Chemical Theory and Computation 2022 18 (10), 5792-5798
       DOI: 10.1021/acs.jctc.2c00585

  """
  force_fn = quantity.canonicalize_force(energy_or_force_fn)

  @jit
  def init_fn(key, R, mass=1.0, mask=None, **kwargs):
    _kT = kwargs.pop('kT', kT)
    key, split = random.split(key)
    state = simulate.NVTLangevinState(R, None, None, mass, key)
    state = simulate.canonicalize_mass(state)
    if mask is None:
        mask = jnp.ones_like(R[:, 0], dtype=bool)

    masked_mass = jnp.where(mask[:, jnp.newaxis], state.mass, 1.0)
    momentum = jnp.sqrt(masked_mass * kT) * random.normal(split, R.shape, dtype=R.dtype)
    momentum -= jnp.mean(mask[:, jnp.newaxis] * momentum) / jnp.mean(mask) * mask[:, jnp.newaxis]
    if zero_velocity:
        state = state.set(momentum=jnp.zeros_like(state.momentum))

    return state.set(momentum=momentum)

  @jit
  def step_fn(state, mask=None, **kwargs):
    _dt = kwargs.get('dt', dt)
    _kT = kwargs.get('kT', kT)
    _gamma = kwargs.get('gamma', gamma)

    if mask is None:
      mask = jnp.ones_like(state.position[:, 0], dtype=bool)

    masked_mass = jnp.where(mask[:, jnp.newaxis], state.mass, 1.0)

    # Friction coefficients
    c1 = jnp.exp(-_gamma * _dt)
    c2 = jnp.sqrt((1 - jnp.square(c1)) * masked_mass * _kT)

    force = force_fn(state.position, mask=mask, **kwargs)
    state = state.set(momentum=state.momentum + _dt * force)

    # Sample the noise
    key, split = random.split(state.rng)
    state = state.set(rng=key)

    # Compute the full momentum update
    dp = (1 - c1) * state.momentum + c2 * random.normal(split, state.momentum.shape)

    # Perform the position update with half-updated momentum
    dq = jnp.where(
      mask[:, jnp.newaxis],
      state.momentum / masked_mass - 0.5 * dp / masked_mass, 0.0)

    state = state.set(position=shift_fn(state.position, dq * _dt))

    # Perform the remaining momentum update
    state = state.set(momentum=mask[:, jnp.newaxis] * (state.momentum - dp))

    return state

  return init_fn, step_fn
