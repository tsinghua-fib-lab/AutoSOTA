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

import jax
import jax.numpy as jnp
from jax import tree_util, random

import optax

import pytest

import jax_md_mod
from jax_md import space, energy, partition, simulate

from chemtrain.data import preprocessing
from chemtrain.trainers import RelativeEntropy
from chemtrain import ensemble, quantity

class TestRelativeEntropy:

    @pytest.fixture
    def setup_problem(self, datafiles):

        box = jnp.asarray([1.0, 1.0, 1.0])
        kT = 2.56

        all_positions = preprocessing.get_dataset(
            datafiles / "positions_ethane.npy")


        displacement_fn, shift_fn = space.periodic_general(
            box, fractional_coordinates=True)

        # Scale the position data into fractional coordinates
        position_dataset = preprocessing.scale_dataset_fractional(
            all_positions, box)

        # Weights for the mapping
        masses = jnp.asarray([15.035, 1.011, 1.011, 1.011])

        weights = jnp.asarray([
            [1, 0.0000, 0, 0, 0, 0.000, 0.000, 0.000],
            [0.0000, 1, 0.000, 0.000, 0.000, 0, 0, 0]
        ])

        position_dataset = preprocessing.map_dataset(
            position_dataset, displacement_fn, shift_fn, weights
        )

        # Setup Model #########

        r_init = position_dataset[0, ...]

        displacement_fn, shift_fn = space.periodic_general(
            box, fractional_coordinates=True)
        neighbor_fn = partition.neighbor_list(
            displacement_fn, box, 1.0, fractional_coordinates=True,
            disable_cell_list=True)

        nbrs_init = neighbor_fn.allocate(r_init)

        init_params = {
            "log_b0": jnp.log(0.11),
            "log_kb": jnp.log(1000.0)
        }

        def energy_fn_template(energy_params):
            harmonic_energy_fn = energy.simple_spring_bond(
                displacement_fn, bond=jnp.asarray([[0, 1]]),
                length=jnp.exp(energy_params["log_b0"]),
                epsilon=jnp.exp(energy_params["log_kb"]),
                alpha=2.0
            )

            return harmonic_energy_fn

        # Compute analytical solution
        disp = jax.vmap(displacement_fn)(position_dataset[:, 0, :],
                                         position_dataset[:, 1, :])
        dist_CC = jnp.sqrt(jnp.sum(disp ** 2, axis=-1))

        # Analytical solution
        b0 = jnp.mean(dist_CC)
        kb = kT / jnp.var(dist_CC)

        print(f"Estimated potential parameters are {kb :.1f} kJ/mol/nm^2 "
              f"and {b0 :.3f} nm")

        # ## Setup Optimizer

        epochs = 150
        initial_lr = 0.1
        lr_decay = 0.15

        lrd = int(position_dataset.shape[0] / epochs)
        lr_schedule = optax.exponential_decay(initial_lr, lrd, lr_decay)
        optimizer = optax.chain(
            optax.scale_by_adam(),
            optax.scale_by_schedule(lr_schedule),
            # Flips the sign of the update for gradient descend
            optax.scale_by_learning_rate(1.0),
        )
        # -

        # Setup simulation

        timings = ensemble.sampling.process_printouts(
            time_step=0.002, total_time=1e3, t_equilib=1e2,
            print_every=0.1, t_start=0.0
        )

        init_ref_state, sim_template = ensemble.sampling.initialize_simulator_template(
            simulate.nvt_langevin, shift_fn=shift_fn, nbrs=nbrs_init,
            init_with_PRNGKey=True,
            extra_simulator_kwargs={"kT": kT, "gamma": 1.0, "dt": 0.002}
        )

        cg_masses = masses[0]

        reference_state = init_ref_state(
            random.PRNGKey(11), r_init,
            energy_or_force_fn=energy_fn_template(init_params),
            init_sim_kwargs={"mass": cg_masses, "neighbor": nbrs_init}
        )

        # Setup relative entropy algorithm
        relative_entropy = RelativeEntropy(
            init_params=init_params, optimizer=optimizer,
            reweight_ratio=1.1, sim_batch_size=1,
            energy_fn_template=energy_fn_template,
        )

        state_kwargs = {"kT": kT}

        relative_entropy.add_statepoint(
            position_dataset, energy_fn_template,
            sim_template, neighbor_fn, timings,
            state_kwargs, reference_state,
        )

        # Sets up step size adaption for all statepoints
        relative_entropy.init_step_size_adaption(0.25)

        return relative_entropy, b0, kb, epochs


    @pytest.mark.test_trainers
    def test_training(self, setup_problem):
        relative_entropy, b0, kb, epochs = setup_problem


        relative_entropy.train(epochs, checkpoint_freq=1000)

        # Compare for convergences and if agrees sufficiently well
        # with the analytical solution.

        pred_parameters = tree_util.tree_map(jnp.exp, relative_entropy.params)
        grad_norm = relative_entropy.gradient_norm_history

        assert grad_norm[0] > grad_norm[-1], (
            f"Trainer did not converge, final grad norm {grad_norm[-1]} > "
            f"grad norm in first epoch {grad_norm[0]}."
        )

        b0_err = jnp.abs(b0 - pred_parameters["log_b0"])
        kb_err = jnp.abs(kb - pred_parameters["log_kb"])

        assert b0_err / b0 < 5e-2, (
            f"Trainer could not reproduce b0 up to tol of 5e-2, analytical "
            f"solution is {b0}, predicted value is "
            f"{pred_parameters['log_b0']} (AE: {b0_err})."
        )

        assert kb_err / kb < 5e-2, (
            f"Trainer could not reproduce kb up to tol of 5e-2, analytical "
            f"solution is {kb}, predicted value is "
            f"{pred_parameters['log_kb']} (AE: {kb_err})."
        )
