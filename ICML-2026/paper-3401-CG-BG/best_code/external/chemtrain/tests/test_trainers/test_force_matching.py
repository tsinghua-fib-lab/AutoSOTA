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

from pathlib import Path

import jax
import jax.numpy as jnp
from jax import tree_util

import matplotlib.pyplot as plt

import numpy as onp

import optax

import pytest

from jax_md_mod import custom_partition
from jax_md import space, energy, partition

from chemtrain.data import preprocessing
from chemtrain.trainers import ForceMatching


class TestForceMatching:

    @pytest.fixture
    def setup_problem(self, datafiles):

        train_ratio = 0.5

        box = jnp.asarray([1.0, 1.0, 1.0])

        all_forces = preprocessing.get_dataset(
            datafiles / "forces_ethane.npy")
        all_positions = preprocessing.get_dataset(
            datafiles / "positions_ethane.npy")


        displacement_fn, shift_fn = space.periodic_general(
            box, fractional_coordinates=True)

        # Scale the position data into fractional coordinates
        position_dataset = preprocessing.scale_dataset_fractional(
            all_positions, box)

        # Weights for the mapping
        weights = jnp.asarray([
            [1, 0.0000, 0, 0, 0, 0.000, 0.000, 0.000],
            [0.0000, 1, 0.000, 0.000, 0.000, 0, 0, 0]
        ])

        position_dataset, force_dataset = preprocessing.map_dataset(
            position_dataset, displacement_fn, shift_fn, weights, weights,
            all_forces,
        )

        # Setup Model #########

        r_init = position_dataset[0, ...]

        displacement_fn, shift_fn = space.periodic_general(
            box, fractional_coordinates=True)
        neighbor_fn = custom_partition.masked_neighbor_list(
            displacement_fn,  r_cutoff=1.0)

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
        disp_dir = disp / dist_CC[:, None]
        force_proj = jnp.einsum('ijk, i...k->ij', force_dataset, disp_dir)

        # Least squares solution
        lhs = jnp.stack((dist_CC, jnp.ones_like(dist_CC)), axis=-1)
        rhs = -force_proj[:, (0,)]

        kb, c = jnp.linalg.lstsq(lhs, rhs, rcond=None)[0]
        b0 = -c / kb

        print(f"Estimated potential parameters are {kb[0] :.1f} kJ/mol/nm^2 "
              f"and {b0[0] :.3f} nm")

        # ## Setup Optimizer

        batch_per_device = 10
        batch = batch_per_device * jax.device_count()
        epochs = 25
        initial_lr = 0.1
        lr_decay = 0.1

        lrd = int(position_dataset.shape[0] / batch * epochs)
        lr_schedule = optax.exponential_decay(initial_lr, lrd, lr_decay)
        optimizer = optax.chain(
            optax.scale_by_adam(),
            optax.scale_by_schedule(lr_schedule),
            # Flips the sign of the update for gradient descend
            optax.scale_by_learning_rate(1.0),
        )
        # -

        # ## Setup Force Matching

        force_matching = ForceMatching(
            init_params=init_params, energy_fn_template=energy_fn_template,
            nbrs_init=nbrs_init, optimizer=optimizer,
            batch_per_device=batch_per_device
        )

        force_matching.set_datasets({
            "R": position_dataset[::50, :, :],
            "F": force_dataset[::50, :, :],
        }, train_ratio=train_ratio)

        return force_matching, b0, kb, epochs


    @pytest.mark.test_trainers
    def test_training(self, setup_problem):
        force_matching, b0, kb, epochs = setup_problem


        force_matching.train(epochs, checkpoint_freq=1000)

        # Compare for convergences and if agrees sufficiently well
        # with the analytical solution.

        pred_parameters = tree_util.tree_map(jnp.exp, force_matching.params)
        losses = force_matching.train_losses

        assert losses[0] > losses[-1], (
            f"Trainer did not converge, final loss {losses[-1]} > loss after "
            f"first epoch {losses[0]}."
        )

        b0_err = jnp.abs(b0[0] - pred_parameters["log_b0"])
        kb_err = jnp.abs(kb[0] - pred_parameters["log_kb"])

        assert b0_err / b0[0] < 5e-2, (
            f"Trainer could not reproduce b0 up to tol of 5e-2, analytical "
            f"solution is {b0[0]}, predicted value is "
            f"{pred_parameters['log_b0']} (AE: {b0_err})."
        )

        assert kb_err / kb[0] < 5e-2, (
            f"Trainer could not reproduce kb up to tol of 5e-2, analytical "
            f"solution is {kb[0]}, predicted value is "
            f"{pred_parameters['log_kb']} (AE: {kb_err})."
        )

        force_matching.evaluate_mae_testset()
