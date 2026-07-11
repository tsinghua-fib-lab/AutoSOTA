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

"""Utility functions to sample from ensembles. """
import functools
from functools import partial
from typing import Any, Dict, Callable, Mapping, Tuple, Union, Protocol, Optional

import jax
import numpy as onp

import chex

from jax import lax, jit, vmap, numpy as jnp, random, tree_util

from jax_md_mod import custom_quantity
from jax_md import util as jax_md_util
from jax_md.partition import NeighborList

from chemtrain import util
from chemtrain.ensemble import evaluation

Array = jax_md_util.Array

@partial(chex.dataclass, frozen=True)
class SimulatorState:
    """A tuple of simulator state and neighbor list state.

    Args:
        sim_state: ``jax_md`` simulator state
        nbrs: ``jax_md`` neighbor list state

    """
    sim_state: evaluation.State
    nbrs: NeighborList


@partial(chex.dataclass, frozen=True)
class TimingClass:
    """A dataclass containing run-times for the simulation.

    Attributes:
        t_equilib_start: Starting time of all printouts that will be dumped
            for equilibration
        t_production_start: Starting time of all runs that result in a printout
        t_production_end: Generation time of all printouts
        timesteps_per_printout: Number of simulation timesteps to run forward
            from each starting time
        time_step: Simulation time step
    """
    t_equilib_start: Array
    t_production_start: Array
    t_production_end: Array
    timesteps_per_printout: int
    time_step: float


@partial(chex.dataclass, frozen=True)
class TrajectoryState:
    """A dataclass storing information of a generated trajectory.

    Attributes:
        sim_state: Last simulation state, a tuple of last state and nbrs
        trajectory: Generated trajectory
        overflow: True if neighbor list overflowed during trajectory generation
        dynamic_kwargs: Additional information passed to the simulator and
            energy function, e.g., species, thermostat / barostat targets
        static_kwargs: Same as ``dynamic_kwargs`` but constant for the
            trajectory.
        aux: Dict of auxilary per-snapshot quantities as defined by quantities
            in trajectory generator.
        key: PRNGKey of the trajectory state.
        energy_params: Energy parameters used to generate the trajectory.
        entropy_diff: Entropy difference estimated for the trajectory, e.g.,
            via DiffTRe optimization
        free_energy_diff: Free energy difference estimated for the trajectory,
            e.g., via DiffTRe optimization
    """
    sim_state: SimulatorState
    trajectory: evaluation.State
    overflow: Array = False
    dynamic_kwargs: Dict[str, Array] = None
    static_kwargs: Dict[str, Array] = None
    aux: Dict[str, Any] = None
    key: Array = None
    energy_params: Any = None
    entropy_diff: Array = 0.0
    free_energy_diff: Array = 0.0

    @property
    def thermostat_kbt(self):
        """Target thermostat kbT at time of respective snapshots. """
        return self.dynamic_kwargs.get('kT', None)

    @property
    def barostat_press(self):
        """Target barostat pressure at time of respective snapshots. """
        return self.dynamic_kwargs.get('pressure', None)

    @property
    def reference_nbrs(self):
        """Returns a single neighbor list."""
        if self.sim_state.nbrs is None:
            return

        if self.sim_state.nbrs.reference_position.ndim > 2:
            return util.tree_get_single(self.sim_state.nbrs)
        else:
            return self.sim_state.nbrs


def process_printouts(time_step, total_time, t_equilib, print_every, t_start=0):
    """Initializes a dataclass containing information for the simulator
    on simulation time and saving states.

    This function is not jitable as array sizes depend on input values.

    Args:
        time_step: Time step size
        total_time: Total simulation run length
        t_equilib: Equilibration run length (can be 0)
        print_every: Time after which a state is saved
        t_start: Starting time. Only relevant for time-dependent
            thermostat/barostat.

    Returns:
        A class containing information for the simulator
        on which states to save.
    """
    assert total_time > 0., 'Total time needs to be positive.'
    assert t_equilib >= 0., 'Equilibration time needs to be non-negative.'
    assert total_time > t_equilib, ('Total time needs to exceed equilibration '
                                    'time, otherwise no trajectory will be '
                                    'sampled.')
    
    timesteps_per_printout = int(print_every / time_step)
    
    # Calculate number of frames for each phase
    n_dumped = int(t_equilib / print_every)
    n_production = int((total_time - t_equilib) / print_every)
    
    # Generate time arrays (automatically empty if n=0)
    equilibration_t_start = jnp.arange(n_dumped) * print_every + t_start
    production_t_start = (jnp.arange(n_production) * print_every
                          + t_equilib + t_start)
    production_t_end = production_t_start + print_every
    
    timings = TimingClass(t_equilib_start=equilibration_t_start,
                          t_production_start=production_t_start,
                          t_production_end=production_t_end,
                          timesteps_per_printout=timesteps_per_printout,
                          time_step=time_step)
    return timings


def initialize_simulator_template(init_simulator_fn,
                                  shift_fn: Callable,
                                  nbrs: NeighborList = None,
                                  init_with_PRNGKey: bool = True,
                                  extra_simulator_kwargs: Mapping = None):
    """Initializes the simulator template and reference state.

    Args:
        init_simulator_fn: Function returning a ``(init_fn, apply_fn)`` tuple
            when provided with a potential energy function.
        shift_fn: Function shifting positions back into the simulation box
            after a simulation update step.
        nbrs: Neighbor list to allocate new neighbor list based on particle
            positions.
        init_with_PRNGKey: Whether simulator init function takes an PRNGKey,
            should be set to False e.g. for the Gradient Descend energy
            minimization routine.
        extra_simulator_kwargs: Additional arguments when initializing the
            simulator state.

    Returns:
        Returns a function to initialize the simulator state and the
        corresponding simulator template.

    """
    if extra_simulator_kwargs is None:
        extra_simulator_kwargs = {}

    simulator_template = functools.partial(
        init_simulator_fn, shift_fn=shift_fn, **extra_simulator_kwargs)

    def init_reference_state(key, r_init, energy_or_force_fn,
                             init_sim_kwargs=None, init_nbrs_kwargs=None):
        if init_nbrs_kwargs is None:
            init_nbrs_kwargs = {}
        if init_sim_kwargs is None:
            init_sim_kwargs = {}

        init_simulator, _ = simulator_template(energy_or_force_fn)

        def _single_init_fn(key, r_init):
            if r_init.ndim > 2:
                # Initialize vectorized by calling function recursively until
                # only single conformation is left
                splits = random.split(key, r_init.shape[0])
                return vmap(_single_init_fn, in_axes=0)(splits, r_init)

            nonlocal nbrs
            if nbrs is not None:
                nbrs = nbrs.update(r_init, **init_nbrs_kwargs)

            if init_with_PRNGKey:
                init_state = init_simulator(key, r_init, **init_sim_kwargs)
            else:
                init_state = init_simulator(r_init, **init_sim_kwargs)

            return SimulatorState(sim_state=init_state, nbrs=nbrs)

        # Check no overflow during neighborlist update
        init_state = _single_init_fn(key, r_init)

        assert not onp.any(init_state.nbrs.did_buffer_overflow), (
            "Overflow during initialization of trajectories. Provided a "
            "neighbor list with more capacity."
        )

        return init_state

    return init_reference_state, simulator_template


def reduced_energy_wrapper(reduced_energy_fn):

    @functools.wraps(reduced_energy_fn)
    def wrapped(*args, exchange_kwargs, **kwargs):
        return lax.map(lambda k: reduced_energy_fn(*args, **{**kwargs, **k}), exchange_kwargs)

    return wrapped


def nvt_reduced_energy_wrapper(energy_fn_template):
    """Reduced energy function for NVT ensemble."""

    @reduced_energy_wrapper
    def reduced_energy(state, neighbor=None, kT=None, energy_params=None, **kwargs):
        energy_fn = energy_fn_template(energy_params)

        if kT is None:
            raise ValueError('kT needs to be provided for NVT ensemble.')

        return energy_fn(state.position, neighbor=neighbor, **kwargs) / kT
    
    return reduced_energy


def initialize_replica_exchange(energy_fn_template,
                                simulator_template,
                                timings: TimingClass,
                                exchange_every: int,
                                quantities=None,
                                vmap_batch=10,
                                reduced_energy_wrapper: Callable = nvt_reduced_energy_wrapper,
                                comm=None):
    """Initializes the replica exchange simulator template and reference state.

    Args:
        simulator_template: Simulator template initialized via
            ``initialize_simulator_template``.
        energy_fn_template: Energy function template to initialize the energy
            function for the simulator template.
        init_with_PRNGKey: Whether simulator init function takes an PRNGKey,
            should be set to False e.g. for the Gradient Descend energy
            minimization routine.
        reduced_energy_wrapper: Function to wrap the energy function to compute
            the reduced energy for replica exchange. By default, the reduced
            energy for the NVT ensemble (U/kT) is computed.

            
    Returns: Returns a function to initialize the replica exchange simulator state and
        the corresponding replica exchange simulator template.

    """
    if quantities is None:
        quantities = {}

    def constant_fn(_, c):
        return c

    # Required for replica exchange decision
    quantities["reduced_energy"] = reduced_energy_wrapper(energy_fn_template)

    def candidate_pairs(key, n_replicas):
        """Generates candidate pairs for replica exchange."""
        # Generate random pairs of adjacent replicas
        n_pairs = n_replicas // 2

        even_candidates = jnp.stack([
            2 * jnp.arange(n_pairs),
            2 * jnp.arange(n_pairs) + 1,
        ])
        odd_candidates = even_candidates + 1

        candidates = jnp.where(
            random.bernoulli(key), even_candidates, odd_candidates
        )

        invalid = jnp.logical_or(
            candidates[0] >= n_replicas,candidates[1] >= n_replicas
        )

        return candidates, invalid


    def simulate_to_exchange(state, ipt, *, params, apply_kwargs, exchange_kwargs):
        key, sim_states = state
        dynamic_kwargs, times = ipt
        
        energy_fn = energy_fn_template(params)
        _, apply_fn = simulator_template(energy_fn)

        # Run simulation to the next exchange point and compute the trajectory

        @vmap
        def wrapped_run_to_next_printout_fn(sim_state, kwargs):
            _kwargs = {
                **apply_kwargs,
                **{k: partial(constant_fn, c=v) for k, v in kwargs.items()},
            }

            sim_state, trajectory = lax.scan(
                run_to_next_printout_neighbors(apply_fn, timings, _kwargs),
                sim_state, xs=times.t_production_start
            )

            overflow = sim_state.nbrs.did_buffer_overflow

            traj_state = TrajectoryState(
                sim_state=sim_state, trajectory=trajectory,
                overflow=overflow, dynamic_kwargs=dynamic_kwargs,
                static_kwargs=kwargs, energy_params=params
            )

            aux_trajectory = quantity_traj(
                traj_state, quantities, params, vmap_batch)
            
            return traj_state.replace(aux=aux_trajectory)

        if util.use_mpi():
            # Communicate the sim_states to the correct devices for the next
            # exchange step
            ipt, dim = util.mpi_tree_slice((sim_states, exchange_kwargs))
        else:
            dim = 1
            ipt = (sim_states, exchange_kwargs)

        res = wrapped_run_to_next_printout_fn(*ipt)

        if util.use_mpi():
            trajectories = util.mpi_tree_gather(res, dim=dim)
        else:
            trajectories = res

        n_replicas = sim_states.sim_state.position.shape[0]
        # Compute the reduced energy for the exchange decision
        key, split1, split2 = random.split(key, 3)
        (ca, cb), invalid = candidate_pairs(split1, n_replicas)

        # Always sanitize indices before any gather/scatter.
        ca_safe = jnp.where(invalid, n_replicas, ca)
        cb_safe = jnp.where(invalid, n_replicas, cb)

        splits = random.split(split2, n_replicas // 2)
        accept, diff = exchange_step(splits, ca_safe, cb_safe, trajectories)
        accept = jnp.where(invalid, False, accept)

        perm = jnp.arange(n_replicas)
        perm = perm.at[ca_safe].set(jnp.where(accept, cb_safe, ca_safe))
        perm = perm.at[cb_safe].set(jnp.where(accept, ca_safe, cb_safe))

        new_sim_states = util.tree_map(
            lambda x: x[perm], trajectories.sim_state
        )

        new_state = (key, new_sim_states)
        out = (trajectories.replace(sim_state=new_sim_states), diff, accept)
        return new_state, out


    @functools.partial(jax.vmap, in_axes=(0, 0, 0, None))
    def exchange_step(key, idx_a, idx_b, trajs):
        """Compute propability (or acceptance) of exchange"""

        # Exchange the kwargs for the trajs
        # `reduced_energy` has shape (n_replicas, n_snapshots, n_states).
        # Use the last snapshot as the exchange configuration.
        u_aa = trajs.aux["reduced_energy"][idx_a, -1, idx_a]
        u_bb = trajs.aux["reduced_energy"][idx_b, -1, idx_b]
        u_ab = trajs.aux["reduced_energy"][idx_a, -1, idx_b]
        u_ba = trajs.aux["reduced_energy"][idx_b, -1, idx_a]

        diff = u_aa + u_bb - u_ab - u_ba

        u = jnp.log(random.uniform(key, shape=()))
        accept = diff > u

        return accept, diff

        
    @jax.jit
    def run_replica_exchange(key, params, sim_states, **kwargs):
        # Improve backwards-compatibility
        if isinstance(sim_states, tuple):
            sim_states = SimulatorState(
                sim_state=sim_states[0], nbrs=sim_states[1])

        multiple_trajs = sim_states.sim_state.position.ndim > 2
        assert multiple_trajs, (
            'Replica exchange requires multiple replicas, but only a single '
            'reference state was provided.'
        )

        n_replicas = sim_states.sim_state.position.shape[0]
        apply_kwargs, static_kwargs, dynamic_kwargs = canonicalize_state_kwargs(
            kwargs, timings.t_production_end, n_replicas)
        
        assert jax.tree.reduce(
            lambda x, y: x & (y.shape[0] == n_replicas), static_kwargs, initializer=True
        ), (
            'All static kwargs need to have the same leading dimension as the number of replicas.'
        )

        # TODO: Run equilibration

        # Split time axis into exchange windows so scan iterates over windows.
        dynamic_kwargs = util.tree_vmap_split(dynamic_kwargs, exchange_every)
        timings = timings.replace(
            t_production_start=jnp.reshape(timings.t_production_start, (-1, exchange_every)),
            t_production_end=jnp.reshape(timings.t_production_end, (-1, exchange_every))
        )
                
        scan_fn = partial(
            simulate_to_exchange,
            params=params,
            apply_kwargs=apply_kwargs,
            exchange_kwargs=static_kwargs,
        )
        _, (trajectories, diffs, accepts) = lax.scan(
            scan_fn, (key, sim_states), xs=(dynamic_kwargs, timings)
        )

        trajectories = util.tree_combine(trajectories)

        return trajectories, diffs, accepts

    return run_replica_exchange



def run_to_next_printout_neighbors(apply_fn,
                                   timings: TimingClass,
                                   state_kwargs: Dict[str, Callable]):
    """Initializes a function to run a simulation to the next printout state.

    Run simulation forward to each printout point and return state.
    Updates the neighbor list after

    Args:
        apply_fn: Apply function from initialization of simulator
        neighbor_fn: Neighbor function
        timings: Instance of TimingClass containing information
            about which states to retain and simulation time
        state_kwargs: Kwargs to supply ``'kT'`` and/or ``'pressure'``
            via time-dependent functions to allow for non-equilibrium MD

    Returns:
        A function that takes the current simulation state, runs the
        simulation forward to the next printout state and returns it.
    """

    def do_step(state: SimulatorState, t):
        # Read out the (dynamic) state kwargs at the correct times
        apply_kwargs = {
            key: kwarg_fn(t) for key, kwarg_fn in state_kwargs.items()
        }

        # Step the simulator and update the neighbor list to new positions
        nbrs = util.neighbor_update(state.nbrs, state.sim_state, **apply_kwargs)
        new_state = apply_fn(
            state.sim_state, neighbor=nbrs, **apply_kwargs)

        # Cast back to the input types
        new_state = SimulatorState(sim_state=new_state, nbrs=nbrs)
        new_state = tree_util.tree_map(
            lambda x, y: jnp.asarray(x, dtype=y.dtype), new_state, state
        )

        return new_state, t

    def run_small_simulation(start_state: SimulatorState, t_start=0.):
        times = jnp.arange(timings.timesteps_per_printout) * timings.time_step
        times += t_start

        printout_state, _ = lax.scan(do_step, start_state, xs=times)

        return printout_state, printout_state.sim_state

    return run_small_simulation


def init_simulation_fn(run_to_printout_fn,
                       timings: TimingClass,
                       vmap_batch_size: int = 1,
                       devices = None):
    """Runs a simulation with frequently saved states.

    Args:
        run_to_printout_fn: Function to run simulation to next printout state
        timings: Instance of TimingClass containing information
            about which states to retain and simulation time.
        vmap_batch_size: If multiple simulation states provided, run
            multiple trajectories vectorized via vmap.
        devices: If multiple devices provided, run simulations in parallel via
            ``shmap``.

    Returns:
        Returns the final state after the simulation and the subsampled
        simulator states at the defined printout times.

    """

    assert devices is None, (
        "Parallel simulation not yet implemented."
    )

    @vmap
    def vectorized_simulation(sim_state):
        # Optionally: Perform equilibration
        if timings.t_equilib_start.size > 0:
            sim_state, _ = lax.scan(
                run_to_printout_fn, sim_state, xs=timings.t_equilib_start)

        sim_state, trajectories = lax.scan(
            run_to_printout_fn, sim_state, xs=timings.t_production_start)

        return sim_state, trajectories

    @jit
    def simulation_fn(sim_state: SimulatorState):
        # Add batch dimension
        sim_state = util.tree_vmap_split(sim_state, vmap_batch_size)

        sim_state, trajectories = lax.map(vectorized_simulation, sim_state)

        # Assert no buffer overflowed for neighbor list computation
        overflow = jnp.any(sim_state.nbrs.did_buffer_overflow)

        # Restore the original shape
        sim_state = util.tree_combine(sim_state)
        trajectories = util.tree_combine(trajectories)

        return sim_state, trajectories, overflow

    return simulation_fn


def canonicalize_state_kwargs(state_kwargs: Dict[str, Union[Callable, Array]],
                              t_snapshots: Array,
                              n_trajs: int = 1
                              ) -> Tuple[Dict[str, Callable],
                                         Dict[str, Array],
                                         Dict[str, Array]]:
    """Converts kwargs to the simulator to time-dependent functions.

    Converts constant kwargs to the simulator, such as ``'kT'`` and
    ``'pressure'``, to constant functions over time and deletes all ``None``
    kwargs.

    Additionally, returns the values of the kwargs at tge production printout
    times.

    Args:
        state_kwargs: Dictionary of constant (array) or dynamic (function)
            properties of the statepoint.
        t_snapshots: Array of times corresponding to the subsampled simulation
            states.
        n_trajs: Number of trajectories to run. The kwargs at the corresponding
            printout times are tiled accordingly.

    Returns:
        Returns a tuple of dictionaries. The first dictionary contains the
        time-dependent functions passed to the simulator.
        The second dictionary contains the values of the kwargs at the
        sampled printout times.

    """
    def constant_fn(_, c):
        return c

    def replicate(value):
        # Add a new dimension to replicate the statepoint values for all
        # simulated trajectories.
        # Then, flatten the replicated values as done for the trajectories
        value = jnp.expand_dims(value, axis=0)
        value = jnp.repeat(value, n_trajs, axis=0)
        return value

    # Convert the kwargs to time-dependent functions
    canonical_kwargs = {}

    dynamic_statepoint_vals = {}
    static_statepoint_vals = {}
    for key, kwarg in state_kwargs.items():
        if kwarg is None:
            continue
        if not callable(kwarg):
            static_statepoint_vals[key] = kwarg
            kwarg = partial(constant_fn, c=kwarg)
        else:
            dynamic_statepoint_vals[key] = kwarg(t_snapshots)

        canonical_kwargs[key] = kwarg

    return canonical_kwargs, static_statepoint_vals, dynamic_statepoint_vals


class GenerateFn(Protocol):
    """Function generating a new trajectory state."""

    @staticmethod
    def __call__(params: Any, sim_state: Any, **kwargs) -> TrajectoryState:
        """Computes a new trajectory state.

        The function continues the trajectories from the last simulator state,
        with a potential model defined by the energy params.

        Args:
            params: Energy params to initialize the energy function.
            sim_state: Initial simulation states. Multiple states can be
                provided, with pytree leaves concatenated along the first axes,
                to run multiple trajectories in parallel.
            **kwargs: Properties defining the (time-dependent) statepoint.
                E.g., the temperature (`kT`) or pressure (`pressure`) for
                NVT and NPT ensembles.

        Returns:
            Returns a new TrajectoryState.
        """


def trajectory_generator_init(simulator_template, energy_fn_template,
                              ref_timings=None, quantities=None, vmap_batch=10,
                              vmap_sim_batch=10) -> GenerateFn:
    """Initializes a function to computes a trajectory.

    Args:
        simulator_template: Function returning new simulator given
            current energy function
        energy_fn_template: Energy function template
        ref_timings: Instance of TimingClass containing information about the
            times states need to be retained
        quantities: Quantities dict to compute and store auxilary quantities
            alongside trajectory. This is particularly helpful for
            storing energy and pressure in a reweighting context.
        vmap_batch: Batch size for computation of auxillary quantities.

    The trajectory generation consists of the following steps:

    1. Evaluation of the dynamic kwargs at the specified simulation times.
       If the kwargs are constant, they are converted to constant functions.
    2. Running multiple short simulations, saving only the sim-states at
       the specified printout times.
    3. Computing auxilliary quantities for each of the saved simulation
       states.

    Returns:
        A function taking energy params and the current traj_state (including
        neighbor list) that runs the simulation forward generating the
        next TrajectoryState.
    """
    if quantities is None:
        quantities = {}

    # temperature is inexpensive and generally useful: compute it by default
    quantities['kT'] = custom_quantity.temperature

    def generate_reference_trajectory(params, sim_state, combine=True, **kwargs):
        """
        Returns a new TrajectoryState with auxilary variables.

        Args:
            params: Energy function parameters
            sim_state: Initial simulation state(s). Mulriple states can be
                provided to run multiple trajectories in parallel.
            **kwargs: Kwargs to supply ``'kT'`` and/or ``'pressure'`` to change
                these thermostat/barostat values on the fly. Can be constant
                or function of t.

        Returns:
            TrajectoryState object containing the newly generated trajectory
        """
        # Improve backwards-compatibility
        if isinstance(sim_state, tuple):
            sim_state = SimulatorState(
                sim_state=sim_state[0], nbrs=sim_state[1])

        # Canonicalize for single simulated trajectory
        multiple_trajs = sim_state.sim_state.position.ndim > 2
        nonlocal vmap_sim_batch
        if not multiple_trajs:
            sim_state = tree_util.tree_map(
                partial(jnp.expand_dims, axis=0), sim_state
            )
            vmap_sim_batch = 1

        # Set up the simulated (dynamic) state
        timings = kwargs.pop('timings', ref_timings)
        assert timings is not None

        n_trajs = sim_state.sim_state.position.shape[0]
        apply_kwargs, static_kwargs, dynamic_kwargs = canonicalize_state_kwargs(
            kwargs, timings.t_production_end, n_trajs)

        # With the energy function available, we can now initialize the
        # concrete simulator
        energy_fn = energy_fn_template(params)
        _, apply_fn = simulator_template(energy_fn)
        run_to_printout_fn = run_to_next_printout_neighbors(
            apply_fn, timings, apply_kwargs)
        simulation_fn = init_simulation_fn(
            run_to_printout_fn, timings, vmap_sim_batch)

        # After simulation, combine trajectories to large one for simple
        # processing, e.g., in relative entropy matching
        new_sim_state, trajectories, overflow = simulation_fn(sim_state)

        if combine or not multiple_trajs:
            trajectories = util.tree_combine(trajectories)
            dynamic_kwargs = util.tree_combine(dynamic_kwargs)

        # Restore the original state of the simulator
        if not multiple_trajs:
            new_sim_state = tree_util.tree_map(
                partial(jnp.squeeze, axis=0), new_sim_state
            )

        traj_state = TrajectoryState(
            sim_state=new_sim_state, trajectory=trajectories,
            overflow=overflow, dynamic_kwargs=dynamic_kwargs,
            static_kwargs=static_kwargs, energy_params=params)

        # Compute auxillary quantities on a trajectory
        aux_trajectory = quantity_traj(
            traj_state, quantities, params, vmap_batch)
        return traj_state.replace(aux=aux_trajectory)

    return generate_reference_trajectory


def quantity_traj(traj_state, quantities, energy_params=None, batch_size=1):
    """Computes quantities of interest for all states in a trajectory.

    Arbitrary quantity functions can be provided via the quantities-dict.
    The quantities dict provides the function to compute the quantity on
    a single snapshot. The resulting quantity trajectory will be saved in
    a dict under the same key as the input quantity function.

    Note:
        This version exists for backward-compatibility. Consider using the
        more flexible version :func:`quantity_map` instead.

    Example usage:
        .. code-block:: python

            def custom_compute_fn(state, neighbor=None, **kwargs):
                ...
                return quantity_snapshot


            quantities = {
                'energy': custom_quantity.energy_wrapper(energy_template_fn),
                'custom_quantity': custom_compute_fn
            }

            quantity_trajs = quantity_traj(traj_state, quantities, energy_params)
            custom_quantity = quantity_trajs['custom_quantity']


    Args:
        traj_state: Trajectory state from the trajectory generator
        quantities: The quantity dict containing for each target quantity
            the snapshot compute function
        energy_params: Energy params for energy_fn_template to initialize
            the current energy_fn
        batch_size: Number of batches for vmap

    Returns:
        A dict of quantity trajectories saved under the same key as the
        input quantity function.
    """
    return evaluation.quantity_multimap(
        traj_state.trajectory, quantities=quantities,
        nbrs=traj_state.reference_nbrs, state_kwargs=traj_state.dynamic_kwargs,
        constant_state_kwargs=traj_state.static_kwargs,
        energy_params=energy_params, batch_size=batch_size
    )
