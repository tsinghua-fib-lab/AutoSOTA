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
import copy
from functools import partial

import numpy as onp

import jax.numpy as jnp
import jax
from jax import random

import pytest

import optax

from chemtrain import quantity, trainers, ensemble

from jax_md import energy, space, simulate, partition

from collections import namedtuple

class TestDifftre:

    @pytest.fixture
    def setup_trainer(self, tmpdir):
        box = 1.0

        def radial_distribution(r, r_0=0.35, b=250.0, kbt=2.56):
            b /= kbt
            norm = onp.sqrt(onp.pi / (2 * b)) * (1 + b * r_0 ** 2) / b
            g_r = box ** 3 / (16 * onp.pi) * onp.exp(-0.5 * b * (r - r_0) ** 2) / norm
            return g_r

        # We now want to learn the parameters of this harmonic bond based on a reference
        # radial distribution function.

        r = onp.linspace(0.0, box, 100)
        target = onp.vstack((r, radial_distribution(r))).T

        # We first need to define an appropriate potential model.

        def energy_fn_template(params):
            energy_fn = energy.simple_spring_bond(
                displacement_fn,
                jnp.asarray([[0, 1]]),
                length=params["r_0"],
                epsilon=100 * params["scaled_b"],
                alpha=2.0
            )
            return energy_fn

        init_params = {"r_0": 0.3, "scaled_b": 1.5}

        # Secondly, we need a routine to simulate the positions of the particles.

        r_init = jnp.asarray([[0.0, 0.0, 0.0], [0.11, 0.09, 0.12]])
        displacement_fn, shift_fn = space.periodic_general(box)

        dt = 0.01
        timings = ensemble.sampling.process_printouts(dt, 1100, 100, 1.0)

        simulator_template = partial(
            simulate.nvt_langevin, shift_fn=shift_fn,
            dt=dt, kT=2.56, gamma=0.5, mass=10.0)

        neighbor_fn = partition.neighbor_list(displacement_fn, box, 0.5)

        simulator_init, _ = simulator_template(energy_fn_template(init_params))
        simulator_init_state = simulator_init(jax.random.PRNGKey(0), r_init)
        nbrs_init = neighbor_fn.allocate(r_init)

        system = {
            'displacement_fn': displacement_fn,
            'reference_box': box
        }
        # -

        # There are multiple classical approaches that enable the inversion of a
        # radial distribution function into a pair-potential.
        # However, they are not applicable to general models, e.g., neural networks.
        # Thus, DiffTRe enables gradient based training, which we are going to set up in
        # the next step.


        lr_schedule = optax.exponential_decay(-0.05, 300, 0.1)
        optimizer = optax.chain(
            optax.scale_by_rms(0.9),
            optax.scale_by_schedule(lr_schedule)
        )
        # -

        # Finally, we have to specify the training targets, which is in our case the
        # radial distribution function.
        # Since we only have two particles in a box, we approximate the distribution
        # with slightly coarser bins.

        # +
        target_builder = quantity.targets.TargetBuilder()

        target_builder['rdf'] = quantity.targets.init_radial_distribution_target(
            target, rdf_start=0.00, rdf_cut=1.0, nbins=50)
        r_eval = onp.linspace(0, 1, 50)

        targets, compute_fns = target_builder.build(system)

        # Check whether kwargs can be passed dynamically
        def dynamical_observable_fn(state, weights=None, **kwargs):
            del state, weights
            assert 'kT' in kwargs
            return True

        targets['test_dynamic_statepoint'] = {
            'traj_fn': dynamical_observable_fn, 'target': None
        }

        # We now created a numerical representation of the system and can run the trainer.
        state_kwargs = {"kT": 2.56}
        reference_state = ensemble.sampling.SimulatorState(
            sim_state=simulator_init_state, nbrs=nbrs_init
        )
        reference_state = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x[None, ...], 2, axis=0), reference_state
        )

        trainer = trainers.Difftre(
            init_params, optimizer, reweight_ratio=0.99,
            checkpoint_path=tmpdir / "Checkpoints"
        )

        trainer.add_statepoint(
          energy_fn_template, simulator_template, neighbor_fn, timings, state_kwargs,
          compute_fns, reference_state, targets=targets, vmap_batch=2,
          resample_simstates=True)

        return trainer, radial_distribution, r_eval


    @pytest.mark.test_trainers
    def test_training(self, setup_trainer):
        trainer, ref_rdf_fn, r = setup_trainer

        trainer.train(300)
        last_epoch = len(trainer.predictions[0]) - 1

        preds = trainer.predict(key=0)
        assert 'rdf' in preds.keys()

        error = onp.sum((trainer.predictions[0][last_epoch]['rdf'] - ref_rdf_fn(r)) ** 2)
        error /= r.size

        print(f"Remaining training MSE error is {error : .2e}")

        assert error < 2e-3

    @pytest.mark.test_trainers
    def test_checkpoint(self, setup_trainer):
        trainer, ref_rdf_fn, r = setup_trainer

        trainer.train(300, checkpoint_freq=50)
        print(list(trainer.checkpoint_path.iterdir()))

        # Save the predictions and the epochs
        preds = copy.deepcopy(trainer.predictions)
        epoch = copy.deepcopy(trainer._epoch)
        losses = copy.deepcopy(trainer.batch_losses)

        # Load a previous checkpoint
        checkpoint_path = trainer.checkpoint_path / f"epoch00050.pkl"
        trainer.restore(checkpoint_path)

        last_epoch = len(trainer.predictions[0]) - 1

        trainer.train(300 - trainer._epoch)
        # Assert that the predictions are still available
        assert trainer._epoch == epoch, (
            "Epochs do not match after restoring checkpoint"
        )
        assert onp.isclose(trainer.batch_losses, losses).all(), (
            "Losses deviate."
        )
        assert onp.isclose(
            trainer.predictions[0][last_epoch]['rdf'],
            preds[0][last_epoch]['rdf']
        ).all(), (
            "Predictions deviate after restoring checkpoint."
        )

    @pytest.mark.parametrize("bucket", (True, False))
    @pytest.mark.parametrize("statepoints", (5, 10, 21, 25))
    def test_bucketing(self, mocker, bucket, statepoints):
        batch_size = 5

        key = random.PRNGKey(11)
        key, split = random.split(key)
        neffs = random.uniform(split, (statepoints,))

        difftre_trainer = trainers.DifftreParallel.__new__(trainers.DifftreParallel)
        difftre_trainer.reweight_ratio = 0.5
        difftre_trainer.batch_size = batch_size
        difftre_trainer.key = key
        difftre_trainer._bucket_recompute = bucket
        difftre_trainer.traj_states = namedtuple(
            "TrajectoryStates", ["trajectory"]
        )(
            namedtuple(
                "Trajectory", ["position"]
            )(onp.arange(statepoints).reshape((-1, 1)))
        )
        difftre_trainer.weights = lambda _, t: (None, neffs[t.trajectory.position[:, 0]])
        difftre_trainer.state = namedtuple("State", ["params"])(None)

        samples = []
        for idx in difftre_trainer._get_batch():
            samples.append(idx)

            print(f"Selected idx {idx} with recompute {neffs[idx] > 0.5} "
                  f"has unique {onp.unique(neffs[idx] > 0.5)}")

            # Assert that all batches except for the last are bucketed
            if bucket and len(samples) < statepoints // batch_size:
                assert onp.unique(neffs[idx] > 0.5).size == 1

        # Assert that no traj states are selected in duplication
        samples = onp.concatenate(samples, axis=0)
        assert onp.unique(samples).size == samples.size
