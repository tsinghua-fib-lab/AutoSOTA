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


"""This file contains several Trainer classes as a quickstart for users."""
import functools
import os
import pickle
import time
import warnings
from os import PathLike
from typing import Any, Mapping, Dict, Callable, Optional

import jax.tree_util
import numpy as onp
from jax import numpy as jnp, tree_util, jit, random
from jax_sgmc.data import numpy_loader
from jax_md_mod import custom_quantity

from chemtrain import (util)
from chemtrain import config as chemtrain_config
from chemtrain.learn import (
    force_matching, max_likelihood, difftre, property_prediction
)
from chemtrain.quantity import property_prediction
from chemtrain.trainers import base as tt
from chemtrain.ensemble import sampling, reweighting
from chemtrain.data import data_loaders


try:
    from jax.typing import ArrayLike
except:
    ArrayLike = Any
from optax import GradientTransformationExtraArgs
from jax_md.partition import NeighborFn, NeighborList
from chemtrain.typing import EnergyFnTemplate, TrajFn

class PropertyPrediction(tt.DataParallelTrainer):
    """Trainer for direct prediction of molecular properties."""
    def __init__(self, error_fn, prediction_model, init_params, optimizer,
                 graph_dataset, targets, batch=1, batch_per_device=None, batch_cache=10,
                 train_ratio=0.7, val_ratio=0.1, test_error_fn=None,
                 shuffle=False, convergence_criterion="window_median",
                 checkpoint_folder="Checkpoints"):
        # TODO documentation

        # TODO build graph on-the-fly as memory moving might be bottleneck here
        model = property_prediction.init_model(prediction_model)
        checkpoint_path = "output/property_prediction/" + str(checkpoint_folder)
        loss_fn = property_prediction.init_loss_fn(error_fn)

        super().__init__(
            loss_fn, model, init_params, optimizer, checkpoint_path,
            batch=batch, batch_cache=batch_cache, batch_per_device=batch_per_device,
            convergence_criterion=convergence_criterion
        )

        dataset_dict, _ = property_prediction.build_dataset(targets, graph_dataset)
        self.set_datasets(
            dataset_dict, train_ratio=train_ratio, val_ratio=val_ratio,
            shuffle=shuffle
        )

        self.test_error_fn = test_error_fn

    def predict(self, single_observation):
        """Prediction for a single input graph using the current param state."""
        batched_observation = tree_util.tree_map(
            functools.partial(jnp.expand_dims, axis=0), single_observation
        )
        batched_prediction = self.batched_model(
            self.best_inference_params, batched_observation)
        single_prediction = tree_util.tree_map(
            functools.partial(jnp.squeeze, axis=0), batched_prediction
        )
        return single_prediction

    def evaluate_testset_error(self, best_params=True):
        assert "testing" in self._batch_states.keys(), (
            "No test set available. Check train and val ratios."
        )
        assert self._test_fn is not None, (
            "`test_error_fn` is necessary during initialization."
        )

        params = (self.best_inference_params_replicated
                  if best_params else self.state.params)

        error = self.evaluate("testing", self._test_fn, params=params)

        print(f"Error on test set: {error}")
        return error


class ForceMatching(tt.DataParallelTrainer):
    """Parametrizes potential models via the Force Matching method.

    The Force Matching method can be used to learn atomistic [#Ercolessi1994]_
    and coarse-grained [#Noid2008]_ models from first-principle or atomistic
    reference data.

    Args:
        init_params: Initial energy parameters.
        energy_fn_template: Function that takes energy parameters and returns
            an energy function.
        nbrs_init: Initial neighbor list. The neighbor list must be large enough
            to not overflow for any sample of the dataset.
        optimizer: Optimizer from optax.
        gammas: Coefficients for the individual targets in the weighted loss.
        weights_keys: Dictionary to entries of the dataset that contain a
            per-sample weight for the total loss.
        additional_targets: Additional snapshot targets to train on. Forces
            and energy are derived automatically from the energy_fn_template.
        feature_extract_fns: Features to extract from the data, passed to
            all snapshot functions as keyword arguments.
        energy_fn_has_aux: Energy function has an auxiliary output. The
            energy function will be called with argument ``mode="with_aux"``
            and should return a tuple ``(pot, aux)``.
        batch: Global batch size across MPI ranks and local devices.
        batch_per_device: Legacy local batch size per device (per rank).
        batch_cache: Number of batches to load into the device memories.
        full_checkpoint: Save the whole trainer instead of only some statistics.
        disable_shmap: Use ``pmap`` instead of ``shmap`` for parallelization.
        penalty_fn: Penalty depending only on the parameters.
        convergence_criterion: Check convergence via
            :class:`base.EarlyStopping`.
        checkpoint_path: Path to the folder to store checkpoints.
        log_file: Path to file where to log training progress.

    Warning:
        Currently neighborlist overflow is not checked.
        Make sure to build nbrs_init large enough.

    References:
        .. [#Ercolessi1994] Ercolessi, F.; Adams, J. B. Interatomic Potentials
           from First-Principles Calculations: The Force-Matching Method.
           Europhys. Lett. 1994, 26 (8), 583–588.
           https://doi.org/10.1209/0295-5075/26/8/005.
        .. [#Noid2008] Noid, W. G.; Chu, J.-W.; Ayton, G. S.; Krishna, V.;
           Izvekov, S.; Voth, G. A.; Das, A.; Andersen, H. C. The Multiscale
           Coarse-Graining Method. I. A Rigorous Bridge between Atomistic and
           Coarse-Grained Models. J Chem Phys 2008, 128 (24), 244114.
           https://doi.org/10.1063/1.2938860.

    """
    def __init__(self,
                 init_params,
                 optimizer,
                 energy_fn_template: EnergyFnTemplate,
                 nbrs_init: NeighborList,
                 gammas: Dict[str, float] = None,
                 error_fns: Dict[str, Callable] = None,
                 weights_keys: Dict[str, str] = None,
                 additional_targets: Dict[str, Dict] = None,
                 feature_extract_fns: Dict[str, Callable] = None,
                 energy_fn_has_aux: bool = False,
                 batch: Optional[int] = None,
                 batch_per_device: Optional[int] = None,
                 batch_cache: int = 10,
                 full_checkpoint: bool = False,
                 disable_shmap: bool = False,
                 penalty_fn: Callable = None,
                 convergence_criterion: str = "window_median",
                 log_file: str = "force_matching.log",
                 checkpoint_path: PathLike = "checkpoints"):
        # Add additional trainable targets
        if gammas is None:
            gammas = {}

        # This feature extractor enables to evaluate the energy function
        # only once for all computations involving the energy and forces.
        feature_fns = {
            "energy_and_force": custom_quantity.energy_force_wrapper(
                energy_fn_template, has_aux=energy_fn_has_aux
            )
        }

        # These are common quantities to train on. The energy function is not
        # necessary, since forces and energy are pre-extracted
        quantities = {
            "F": custom_quantity.force_wrapper(None),
            "U": custom_quantity.energy_wrapper(None)
        }

        if additional_targets is not None:
            quantities.update(additional_targets)
        if feature_extract_fns is not None:
            feature_fns.update(feature_extract_fns)

        model = force_matching.init_model(
            nbrs_init, quantities, feature_extract_fns=feature_fns
        )

        loss_fn = force_matching.init_loss_fn(
            error_fns=error_fns, gammas=gammas, weights_keys=weights_keys)

        super().__init__(loss_fn, model, init_params, optimizer,
                         checkpoint_path, batch=batch, batch_cache=batch_cache,
                         batch_per_device=batch_per_device,
                         disable_shmap=disable_shmap, penalty_fn=penalty_fn,
                         convergence_criterion=convergence_criterion,
                         full_checkpoint=full_checkpoint,
                         log_file=log_file,
                         energy_fn_template=energy_fn_template)

        self._nbrs_init = nbrs_init

    def evaluate_mae_testset(self):
        """Prints the Mean Absolute Error for every target on the test set."""
        mae_loss_fn = force_matching.init_loss_fn(
                max_likelihood.mae_loss, individual=True
            )

        _, maes = self.evaluate(
            "testing", mae_loss_fn, params=self.best_inference_params
        )

        for key, mae_value in maes.items():
            print(f"{key}: MAE = {mae_value:.4f}")


class DifftreParallel(tt.MLETrainerTemplate):
    """Trainer class for parametrizing potentials via the DiffTRe method.

    This method performs simulations and updates for multiple statepoints in
    parallel using vmap.

    Args:
        init_params: Initial energy parameters
        optimizer: Optimizer from optax
        energy_fn_template: Function that takes energy parameters and
            initializes a new energy function.
        simulator_template: Function that takes an energy function and
            returns a simulator function.
        neighbor_fn: Neighbor function. Must be of
            :func:`jax_md_mod.custom_partition.masked_neighbor_list` if the
            statepoints have a different number of atoms.
        timings: Instance of TimingClass containing information about the
            trajectory length and which states to retain
        state_kwargs: Properties defining the thermodynamic state. Must at least
            contain the temperature 'kT'. For a non-exhaustive list, see
            :class:`chemtrain.ensemble.templates.StatePoint`.
        quantities: Dict containing for each observable specified by the key a
            corresponding function to compute it for each snapshot using
            :func:`ensemble.sampling.quantity_traj`.
        targets: Dict containing the same keys as quantities and containing
            another dict providing 'gamma' and 'target' for each observable.
        observables: Optional dictionary providing the observable functions
            for the targets.
        initial_trajstates: Initial trajectory states of the statepoints.
            It is usually simpler to let the trainer generate the initial
            trajectory states by providing `reference_states`.
        reference_states: Initial simulator states from which DiffTRe can
            compute the initial trajectory states.
        num_runs_init: Number of runs to perform for the initial trajectory
            states. This number can be increased to ensure a better
            equilibration when starting from less favourable initial states.
        reweight_ratio: Ratio of reference samples required for n_eff to
            surpass to allow re-use of previous reference trajectory state.
            If trajectories should not be re-used, a value > 1 can be
            specified.
        allowed_reduction: Allowed reduction of the effective sample size
            through a parameter update.
        step_size_scale: Initial step size scale for the step size adaption.
        interior_points: Number of interior points to use for the step size
            adaption.
        sim_batch_size: Number of state-points to be processed as a single
            batch. Gradients will be averaged over the batch before stepping the
            optimizer.
        full_checkpoint: If True, the whole trainer state is saved, otherwise
            only important parameters are stored as a dictionary.
        target_loss_fns: Dictionary of loss functions to use for each target.
        loss_fn: Custom loss function to use for the training.
        vmap_batch: Number of samples to process simultaneously when computing
            instantaneous quantities for a trajectory.
        bucket_recompute: Groups together statepoints that need a recomputation.
        resample_simstates: Resample the sim states from all trajectories
            instead of simulating independent chains.
        convergence_criterion: Either 'max_loss' or 'ave_loss'.
            If 'max_loss', stops if the maximum loss across all batches in
            the epoch is smaller than convergence_thresh. 'ave_loss'
            evaluates the average loss across the batch. For a single state
            point, both are equivalent. A criterion based on the rolling
            standatd deviation 'std' might be implemented in the future.
        checkpoint_path: Name of folders to store checkpoints in.
        log_dir: Path to the log file where to store training progress.

    Attributes:
        batch_losses: List of losses for each batch in each epoch.
        epoch_losses: List of losses for each epoch.
        step_size_history: List of step sizes for each batched update.
        gradient_norm_history: List of gradient norms for each batched update.
        batch_gradient_norms: List of gradient norms for each batch.
        predictions: Dictionary containing the predictions for each statepoint
            at each epoch.
        early_stop: Instance of EarlyStopping to check for convergence.

    """

    def __init__(self,
                 key: jax.Array,
                 init_params: Any,
                 optimizer: GradientTransformationExtraArgs,
                 energy_fn_template: EnergyFnTemplate,
                 simulator_template: Callable,
                 neighbor_fn: NeighborFn,
                 timings: sampling.TimingClass,
                 state_kwargs: Dict[str, ArrayLike],
                 quantities: Dict[str, Dict],
                 targets: Dict[str, Any],
                 observables: Dict[str, TrajFn],
                 initial_trajstates = None,
                 reference_states = None,
                 num_runs_init: int = 1,
                 reweight_ratio: ArrayLike = 0.9,
                 allowed_reduction: ArrayLike = 0.95,
                 step_size_scale: float = 1e-4,
                 interior_points: int = 100,
                 sim_batch_size: int = 1,
                 full_checkpoint: bool = False,
                 target_loss_fns: Dict[str, Callable] = None,
                 loss_fn=None,
                 vmap_batch: int = 10,
                 bucket_recompute: bool = True,
                 resample_simstates: bool = False,
                 convergence_criterion: str = "window_median",
                 checkpoint_path: os.PathLike = "Checkpoints",
                 log_dir: os.PathLike = None):
        init_state = util.TrainerState(params=init_params,
                                       opt_state=optimizer.init(init_params))

        # Optional: Initialized by calling trainer.init_step_size_adaption
        # after all statepoints to be considered have been set up.
        self._recompute = False

        gen_init_traj, *reweight_fns = reweighting.init_pot_reweight_propagation_fns(
            energy_fn_template, simulator_template, neighbor_fn, timings,
            state_kwargs, reweight_ratio, False,
            vmap_batch, safe_propagation=False,
            entropy_approximation=False,
            resample_simstates=resample_simstates
        )

        self._bucket_recompute = bucket_recompute

        # TODO: Parallelize over multiple devices
        if target_loss_fns is None:
            target_loss_fns = {}

        if loss_fn is None:
            loss_fn = difftre.init_default_loss_fn(observables, target_loss_fns)

        batched_model, batched_propagation, batched_weights = difftre.init_difftre_gradient_and_propagation(
            reweight_fns, loss_fn, quantities, energy_fn_template,
            wrapped=False, batched=True
        )

        self.reweight_ratio = reweight_ratio

        self.key = key
        self.batch_size = sim_batch_size
        self.statepoints = state_kwargs

        self.model = jax.jit(jax.value_and_grad(batched_model, argnums=0, has_aux=True))
        self.propagate = jax.jit(batched_propagation)
        self.weights = jax.jit(batched_weights)

        self.targets = targets

        if initial_trajstates is not None:
            self.traj_states = initial_trajstates
        else:
            n_statepoints = targets[list(targets.keys())[0]]["target"].shape[0]

            self.key, split = random.split(key)
            self.traj_states = util.batch_map(
                lambda ops: gen_init_traj(
                    ops[0][0], init_params, ops[0][1],
                    num_runs=num_runs_init, **ops[1]
                ),
                (
                    (random.split(split, n_statepoints), reference_states),
                    state_kwargs
                ), batch_size=sim_batch_size
            )


        if allowed_reduction is not None:
            self._adaptive_step_size = difftre.init_step_size_adaption(
                lambda *args: (None, jnp.min(batched_weights(*args)[1])),
                allowed_reduction, step_size_scale=step_size_scale,
                interior_points=interior_points
            )
        else:
            self._adaptive_step_size = lambda *args: (1.0, None)

        super().__init__(
            init_state=init_state,
            optimizer=optimizer,
            checkpoint_path=checkpoint_path,
            full_checkpoint=full_checkpoint,
            log_file=log_dir
        )

        self.batch_losses = self.checkpoint("batch_losses", [])
        self.batch_gradient_norms = self.checkpoint("batch_gradient_norms", [])
        self.epoch_losses = self.checkpoint("epoch_losses", [])
        self.step_size_history = self.checkpoint("step_size_history", [])
        self.gradient_norm_history = self.checkpoint("gradient_norm_history", [])
        self.predictions = self.checkpoint("predictions", {})

        # Initial trajstates should be set by now
        for key in range(self.n_statepoints):
            self.predictions[key] = {}

        self.early_stop = tt.EarlyStopping(
            self.params, convergence_criterion)

    @property
    def params(self):
        """Current energy parameters."""
        return self.state.params

    @params.setter
    def params(self, loaded_params):
        """Replaces the current energy parameters."""
        self.state = self.state.replace(params=loaded_params)

    @property
    def n_statepoints(self):
        return self.traj_states.trajectory.position.shape[0]

    def _get_batch(self):
        """Returns the next batch of statepoints to be processed."""
        self.key, key = random.split(self.key)
        num_statepoints = self.traj_states.trajectory.position.shape[0]
        mask = jnp.ones(num_statepoints)

        for i in range(num_statepoints // self.batch_size):
            key, split = random.split(key)

            # If bucketing is no longer possible or disabled, return back
            # a random batch of statepoints. Otherwise, return a full batch of
            # statepoints that either need or don't need reweighting.
            if not self._bucket_recompute or 2 * self.batch_size > jnp.sum(mask):
                batches = random.choice(
                    split, num_statepoints, (self.batch_size,),
                    replace=False, p=mask
                )
            else:
                # Compute the effective sample size for twice as many samples
                candidates = random.choice(
                    split, num_statepoints, (2 * self.batch_size,),
                    replace=False, p=mask
                )

                trajstates = util.tree_take(self.traj_states, candidates,
                                            on_cpu=False)

                # Compute the effective sample sizes
                _, n_eff = self.weights(self.params, trajstates)
                min_n_eff = self.traj_states.trajectory.position.shape[
                                1] * self.reweight_ratio

                recompute = n_eff < min_n_eff

                # Select samples only from the largest class. At least one
                # of the conditions should be fulfilled:
                # a) At least BS samples must be recomputed
                # b) At least BS samples do not need a recomputation
                key, split = random.split(key)
                if jnp.sum(recompute) > self.batch_size:
                    select = jnp.float32(recompute)
                else:
                    select = jnp.float32(~recompute)

                # Select from the class at random. The not selected samples
                # should remain in the pool.
                batches = random.choice(
                    split, candidates, (self.batch_size,),
                    replace=False, p=select
                )

            # Mark the samples as drawn
            mask = mask.at[batches].set(0.0)
            yield batches

    def _update(self, batch):
        """Computes gradient averaged over the sim_batch by propagating
        respective state points. Additionally saves predictions and loss
        for postprocessing."""

        # Select the relevant trajstates and targets

        trajstates = util.tree_take(self.traj_states, batch, on_cpu=False)
        targets = util.tree_take(self.targets, batch, on_cpu=False)
        statepoints = util.tree_take(self.statepoints, batch, on_cpu=False)

        # Compute the effective sample sizes and print
        _, n_eff = self.weights(self.params, trajstates)
        min_n_eff = self.traj_states.trajectory.position.shape[1] * self.reweight_ratio

        ## Determine if recompute is necessary #################################

        print(f"[DifftreParallel] Effective sample sizes (limit: {min_n_eff})")
        for b, eff in zip(batch, n_eff):
            info = "-> recompute" if eff < min_n_eff else ""
            print(f"\t[Statepoint {b}] Effective sample size: {eff:.2f} {info}")


        if onp.any(n_eff < min_n_eff):
            print(f"[DifftreParallel] Recomputing trajectories...")
            start = time.time()
            trajstates = self.propagate(self.params, trajstates, statepoints)
            print(f"[DifftreParallel] Recomputed trajectories in {(time.time() - start) / 60.:.2f} min")

            # Save the recomputed trajectories
            self.traj_states = util.tree_put(self.traj_states, batch, trajstates, on_cpu=False)

        ## Compute the loss ####################################################

        print(f"[DifftreParallel] Computing loss...")
        start = time.time()
        (loss, state_point_predictions), grad = self.model(
            self.params, trajstates, statepoints, targets
        )
        batch_norm = util.tree_norm(grad)
        self.batch_gradient_norms.append(onp.asarray(batch_norm))
        print(f"[DifftreParallel] Computed loss {loss} in {(time.time() - start) / 60.:.2f} min")

        ## Optimize the step size ##############################################

        proposal = self._optimizer_step(grad)
        # Perform stepsize optimization
        start = time.time()
        alpha, residual = self._adaptive_step_size(self.params, grad, proposal, trajstates)
        print(
            f"[Step Size] Found optimal step size for {alpha} with residual "
            f"{residual} in {(time.time() - start):.1f} s", flush=True)

        self._step_optimizer(grad, alpha=alpha)

        ## Save the predictions for the respective batches #####################
        print(f"[DifftreParallel] Predictions:")
        for idx, b in enumerate(batch):
            self.predictions[int(b)][self._epoch] = {
                key: onp.asarray(val[idx])
                for key, val in state_point_predictions.items()
            }

            # Print scalar predictions
            print(f"\t[Statepoint {b}]")
            for key, value in state_point_predictions.items():
                if jnp.shape(value[idx]) == ():
                    target = ""
                    if key in targets:
                        target = f"(target: {targets[key]['target'][idx]})"

                    print(f"\t\t{key} = {value[idx]} {target}")

        # Save the loss and gradient norm
        self.batch_losses.append(onp.asarray(loss))
        self.step_size_history.append(onp.asarray(alpha))

    def predict(self, batch):
        """Predict for a batch of statepoints."""

        # Select the relevant trajstates and targets

        trajstates = util.tree_take(self.traj_states, batch, on_cpu=False)
        targets = util.tree_take(self.targets, batch, on_cpu=False)
        statepoints = util.tree_take(self.statepoints, batch, on_cpu=False)

        # Compute the effective sample sizes and print
        _, n_eff = self.weights(self.params, trajstates)
        min_n_eff = self.traj_states.trajectory.position.shape[
                        1] * self.reweight_ratio

        ## Determine if recompute is necessary #################################

        print(
            f"[DifftreParallel] Effective sample sizes (limit: {min_n_eff})")
        for b, eff in zip(batch, n_eff):
            info = "-> recompute" if eff < min_n_eff else ""
            print(
                f"\t[Statepoint {b}] Effective sample size: {eff:.2f} {info}")

        if onp.any(n_eff < min_n_eff):
            print(f"[DifftreParallel] Recomputing trajectories...")
            start = time.time()
            trajstates = self.propagate(self.params, trajstates,
                                        statepoints)
            print(
                f"[DifftreParallel] Recomputed trajectories in {(time.time() - start) / 60.:.2f} min")

            # Save the recomputed trajectories
            self.traj_states = util.tree_put(self.traj_states, batch,
                                             trajstates, on_cpu=False)

        ## Compute the loss ####################################################

        print(f"[DifftreParallel] Start predictions...")
        for idx, b in enumerate(batch):
            print(f"\t[Statepoint {b}]")
            for key, val in statepoints.items():
                if not jnp.isscalar(val[idx]): continue
                print(f"\t\t{key} = {val[idx]}")

        (_, state_point_predictions), _ = self.model(
            self.params, trajstates, statepoints, targets
        )

        return state_point_predictions

    def _evaluate_convergence(self, *args, thresh=None, **kwargs):
        # sim_batch_size = -1 means all statepoints are processed in one batch.
        batches_per_epoch = self.n_statepoints // self.batch_size

        last_losses = jnp.array(self.batch_losses[-batches_per_epoch:])
        epoch_loss = jnp.mean(last_losses)
        duration = self.update_times[self._epoch]
        self.epoch_losses.append(epoch_loss)
        self.gradient_norm_history.append(
            onp.mean(self.batch_gradient_norms[-batches_per_epoch:])
        )

        print(
            f"\n[DiffTRe] Epoch {self._epoch}"
            f"\n\tEpoch loss = {epoch_loss:.5f}"
            f"\n\tGradient norm: {self.gradient_norm_history[-1]}"
            f"\n\tElapsed time = {duration:.3f} min")

        self._converged = self.early_stop.early_stopping(
            epoch_loss, thresh, self.params)

    @property
    def best_params(self):
        """Returns the best parameters according to the early stopping criterion."""
        return self.early_stop.best_params

    def move_to_device(self):
        """Transforms the trainer states to JAX arrays."""
        super().move_to_device()
        self.early_stop.move_to_device()


class Difftre(tt.PropagationBase):
    """Trainer class for parametrizing potentials via the DiffTRe method.

    The Differentiable Trajectory Reweighting (DiffTRe) method [#Thaler2021]_
    is a method to compute the gradients of ensemble averages without
    differentiating through the simulation. Therefore, the method can
    efficiently train potential models on macroscopic observables.

    The trainer initialization only sets the initial trainer state
    as well as checkpointing and save-functionality. For training,
    target state points with respective simulations need to be added
    via :func:`Difftre.add_statepoint`.

    Args:
        init_params: Initial energy parameters
        optimizer: Optimizer from optax
        reweight_ratio: Ratio of reference samples required for n_eff to
            surpass to allow re-use of previous reference trajectory state.
            If trajectories should not be re-used, a value > 1 can be
            specified.
        sim_batch_size: Number of state-points to be processed as a single
            batch. Gradients will be averaged over the batch before stepping the
            optimizer.
        energy_fn_template: Function that takes energy parameters and
            initializes a new energy function. Here, the energy_fn_template
            is only a reference that will be saved alongside the trainer.
            Each state point requires its own due to the dependence on the
            box size via the displacement function, which can vary between
            state points.
        convergence_criterion: Either 'max_loss' or 'ave_loss'.
            If 'max_loss', stops if the maximum loss across all batches in
            the epoch is smaller than convergence_thresh. 'ave_loss'
            evaluates the average loss across the batch. For a single state
            point, both are equivalent. A criterion based on the rolling
            standatd deviation 'std' might be implemented in the future.
        checkpoint_folder: Name of folders to store ckeckpoints in.

    Attributes:
        weight_fn: Dictionary containing the reweighting functions for each
            statepoint.
        batch_losses: List of losses for each batch in each epoch.
        epoch_losses: List of losses for each epoch.
        step_size_history: List of step sizes for each batched update.
        gradient_norm_history: List of gradient norms for each batched update.
        predictions: Dictionary containing the predictions for each statepoint
            at each epoch.
        early_stop: Instance of EarlyStopping to check for convergence.

    Examples:

        .. code-block :: python

            trainer = trainers.Difftre(init_params, optimizer)

            # Add all statepoints
            trainer.add_statepoint(energy_fn_template, simulator_template,
                                   neighbor_fn, timings, statepoint_dict,
                                   compute_fns, reference_state, targets)
            ...

            # Optionally initialize the step size adaption
            trainer.init_step_size_adaption(allowed_reduction=0.5)

            trainer.train(num_updates)

    References:
        .. [#Thaler2021] Thaler, S.; Zavadlav, J. Learning Neural Network
           Potentials from Experimental Data via Differentiable Trajectory
           Reweighting. Nat Commun **2021**, 12 (1), 6884.
           https://doi.org/10.1038/s41467-021-27241-4.

    """

    def __init__(self,
                 init_params: Any,
                 optimizer: GradientTransformationExtraArgs,
                 reweight_ratio: ArrayLike = 1.0,
                 adaptive_step_size_threshold: float = 1e-4,
                 sim_batch_size: int = 1,
                 energy_fn_template: EnergyFnTemplate = None,
                 full_checkpoint: bool = False,
                 convergence_criterion: str = "window_median",
                 checkpoint_path: os.PathLike = "Checkpoints",
                 log_dir: os.PathLike = None):
        init_state = util.TrainerState(params=init_params,
                                       opt_state=optimizer.init(init_params))

        # Optional: Initialized by calling trainer.init_step_size_adaption
        # after all statepoints to be considered have been set up.
        self._recompute = False

        self._adaptive_step_size_threshold = adaptive_step_size_threshold

        self.state_dicts = {}
        self.weight_fn = {}
        self.targets = {}
        super().__init__(
            init_trainer_state=init_state, optimizer=optimizer,
            checkpoint_path=checkpoint_path, reweight_ratio=reweight_ratio,
            sim_batch_size=sim_batch_size, full_checkpoint=full_checkpoint,
            energy_fn_template=energy_fn_template, log_dir=log_dir)

        self.batch_losses = self.checkpoint("batch_losses", [])
        self.epoch_losses = self.checkpoint("epoch_losses", [])
        self.step_size_history = self.checkpoint("step_size_history", [])
        self.predictions = self.checkpoint("predictions", {})

        self.early_stop = tt.EarlyStopping(self.params,
                                        convergence_criterion)

    def add_statepoint(self,
                       energy_fn_template: EnergyFnTemplate,
                       simulator_template: Callable,
                       neighbor_fn: NeighborFn,
                       timings: sampling.TimingClass,
                       state_kwargs: Dict[str, ArrayLike],
                       quantities: Dict[str, Dict],
                       reference_state,
                       targets: Dict[str, Any] = None,
                       observables: Dict[str, TrajFn] = None,
                       target_loss_fns: Dict[str, Callable] = None,
                       loss_fn = None,
                       vmap_batch: int = 10,
                       initialize_traj: bool = True,
                       set_key: str = None,
                       resample_simstates: bool = False,
                       allowed_reduction: ArrayLike = None,
                       adaption_kwargs: Dict = None
                       ):
        """
        Adds a state point to the pool of simulations with respective targets.

        Each statepoints initializes a new gradient and propagation function via
        :func:`chemtrain.learn.difftre.init_difftre_gradient_and_propagation`.

        Args:
            energy_fn_template: Function that takes energy parameters and
                initializes a new energy function.
            simulator_template: Function that takes an energy function and
                returns a simulator function.
            neighbor_fn: Neighbor function
            timings: Instance of TimingClass containing information
                about the trajectory length and which states to retain
            state_kwargs: Properties defining the thermodynamic state. Must
                at least contain the temperature 'kT'. For a non-exhaustive
                list, see :class:`chemtrain.ensemble.templates.StatePoint`.
            quantities: Dict containing for each observable specified by the
                key a corresponding function to compute it for each snapshot
                using :func:`ensemble.sampling.quantity_traj`.
            reference_state: Tuple of initial simulation state and neighbor list
            targets: Dict containing the same keys as quantities and containing
                another dict providing 'gamma' and 'target' for each observable.
                Targets are only necessary when using the 'independent_loss_fn'.
            observables: Optional dictionary providing the observable functions
                for the targets. This is only necessary when the observable
                functions are not already contained in the targets dict.
            target_loss_fns: Optional dictionary providing the loss functions
                for the individual targets. This is only necessary when the
                loss functions are not already contained in the targets dict
                or should be different from the MSE loss.
            loss_fn: Custom loss function taking the trajectory of quantities
                and weights and returning the loss and predictions;
                By default, initializes an independent MSE loss, which computes
                reweighting averages from snapshot-based observables.
                In many applications, the default loss function will be
                sufficient. For a description, see
                :func:`chemtrain.learn.difftre.init_default_loss_fn`.
            vmap_batch: Batch size of vmapping of per-snapshot energy for weight
                computation.
            initialize_traj: True, if an initial trajectory should be generated.
                Should only be set to False if a checkpoint is loaded before
                starting any training.
            set_key: Specify a key in order to restart from same statepoint.
                By default, uses the index of the sequance statepoints are
                added, i.e. self.trajectory_states[0] for the first added
                statepoint.
                Can be used for changing the timings of the simulation during
                training.
            resample_simstates: Resample the sim states from all trajectories
                instead of simulating independent chains.
            allowed_reduction: Allowed reduction of the effective sample size
                for the given statepoint.
            adaption_kwargs: Additional keyword arguments for the step size
                line search. For a description, see
                :func:`chemtrain.learn.difftre.init_step_size_adaption`.

        """

        # init simulation, reweighting functions and initial trajectory
        (key, *reweight_fns) = self._init_statepoint(
            reference_state,
            energy_fn_template,
            simulator_template,
            neighbor_fn,
            timings,
            state_kwargs,
            set_key,
            vmap_batch,
            initialize_traj,
            safe_propagation=False,
            entropy_approximation=False,
            resample_simstates=resample_simstates
        )

        # For backwards compatibility and ease of use for a single statepoint
        if observables is None:
            observables = {
                key: target["traj_fn"] for key, target in targets.items()
            }
        if target_loss_fns is None:
            target_loss_fns = {
                key: target["loss_fn"] for key, target in targets.items()
                if "loss_fn" in target
            }

        # Enables a greater flexibility by sorting out data from frunctions
        targets = {
            key: {k: v for k, v in target.items() if k in ["gamma", "target"]}
            for key, target in targets.items() if target.get("target") is not None
        }

        # build loss function for current state point
        if loss_fn is None:
            loss_fn = difftre.init_default_loss_fn(observables, target_loss_fns)
        else:
            print("Using custom loss function. Ignoring 'target' dict.")

        difftre_grad_and_propagation = difftre.init_difftre_gradient_and_propagation(
            reweight_fns, loss_fn, quantities, energy_fn_template
        )

        self.grad_fns[key] = difftre_grad_and_propagation
        self.predictions[key] = {}  # init saving predictions for this point
        self.weight_fn[key] = jax.jit(reweight_fns[0])
        self.state_dicts[key] = state_kwargs
        self.targets[key] = targets

        if allowed_reduction is not None:
            if adaption_kwargs is None:
                adaption_kwargs = {}

            self._adaptive_step_size[key] = difftre.init_step_size_adaption(
                self.weight_fn[key], allowed_reduction, **adaption_kwargs
            )

        # Reset loss measures if new state point es added since loss values
        # are not necessarily comparable
        self.early_stop.reset_convergence_losses()

    def predict(self, *, key: int):
        """Get predictions for a specific statepoint.

        This method predicts the target quantities for a specific
        statepoint. If necessary, the statepoint performs a trajectory
        regeneration.

        Args:
            key: The key of the statepoint to predict.

        Returns:
            Returns a dictionary containing the predicted observables
            given the current parameter values.

        """
        traj_state = self.trajectory_states[key]
        try:
            traj_state.overflow
        except:
            start = time.time()
            traj_state = traj_state()
            compute_time = (time.time() - start) / 60.

            print(
                f"Delayed initialization of trajectory state in {compute_time :.2f} min.")

        grad_fn = self.grad_fns[key]
        (new_traj_state, *_, state_point_predictions) = grad_fn(
            self.params, traj_state, self.state_dicts[key], self.targets[key],
            recompute=self._recompute
        )

        self.trajectory_states[key] = new_traj_state
        return state_point_predictions

    def _update(self, batch):
        """Computes gradient averaged over the sim_batch by propagating
        respective state points. Additionally saves predictions and loss
        for postprocessing."""
        # TODO parallelization? Maybe lift batch requirement and only
        #  sync sporadically?
        # https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices
        # https://github.com/mpi4jax/mpi4jax
        # TODO split gradient and loss computation from stepping optimizer for
        #  building hybrid trainers?

        # TODO is there good way to reuse this function in BaseClass?

        # Note: in principle, we could move all the use of instance attributes
        # into difftre_grad_and_propagation, which would increase re-usability
        # with relative_entropy. However, this would probably stop all
        # parallelization efforts

        losses = 0.0
        grads = None


        for sim_key in batch:
            traj_state = self.trajectory_states[sim_key]
            try:
                traj_state.overflow
            except:
                start = time.time()
                traj_state = traj_state()
                compute_time = (time.time() - start) / 60.

                print(f"Delayed initialization of trajectory state in {compute_time :.2f} min.")

            grad_fn = self.grad_fns[sim_key]
            (new_traj_state, loss_val, curr_grad,
             state_point_predictions) = grad_fn(
                self.params, traj_state,
                self.state_dicts[sim_key], self.targets[sim_key],
                recompute=self._recompute
            )

            self.trajectory_states[sim_key] = new_traj_state
            self.predictions[sim_key][self._epoch] = tree_util.tree_map(
                onp.asarray, state_point_predictions)

            losses += loss_val
            if grads is None:
                grads = curr_grad
            else:
                grads = util.tree_sum(grads, curr_grad)

            # Print scalar predictions and statepoint measurements
            self._print_measured_statepoint(sim_key=sim_key)
            last_predictions = self.predictions[sim_key][self._epoch]
            for quantity, value in last_predictions.items():
                if value.ndim == 0:
                    if quantity in self.targets[sim_key]:
                        target = f"({self.targets[sim_key][quantity]['target']})"
                    else:
                        target = ""
                    print(f"\tPredicted {quantity}: {value} {target}")

            if jnp.isnan(loss_val):
                warnings.warn(f"Loss of state point {sim_key} in epoch "
                              f"{self._epoch} is NaN. This was likely caused by"
                              f" divergence of the optimization or a bad model "
                              f"setup causing a NaN trajectory.")
                self._diverged = True  # ends training
                break

        self.batch_losses.append(onp.asarray(losses / len(batch)))
        batch_grad = tree_util.tree_map(lambda x: x / len(batch), grads)

        step_size = 1.0
        recompute = False
        proposal = self._optimizer_step(batch_grad)
        for sim_key in batch:
            if sim_key not in self._adaptive_step_size: continue

            alpha, residual = self._adaptive_step_size[sim_key](
                self.params, batch_grad, proposal, self.trajectory_states[sim_key]
            )

            recompute |= alpha < self._adaptive_step_size_threshold

            print(f"[Step Size] Found optimal step size for {alpha} for statepoint {sim_key} with residual "
                  f"{residual}", flush=True)

            if alpha < step_size:
                step_size = alpha

        # self._recompute = recompute
        self._step_optimizer(batch_grad, alpha=step_size)

        batch_norm = util.tree_norm(batch_grad)
        self.gradient_norm_history.append(onp.asarray(batch_norm))
        self.step_size_history.append(onp.asarray(step_size))


    def _evaluate_convergence(self, *args, thresh=None, **kwargs):
        # sim_batch_size = -1 means all statepoints are processed in one batch.
        if self.sim_batch_size < 0:
            batches_per_epoch = 1
        else:
            batches_per_epoch = self.n_statepoints // self.sim_batch_size

        last_losses = jnp.array(self.batch_losses[-batches_per_epoch:])
        epoch_loss = jnp.mean(last_losses)
        duration = self.update_times[self._epoch]
        self.epoch_losses.append(epoch_loss)

        print(
            f"\n[DiffTRe] Epoch {self._epoch}"
            f"\n\tEpoch loss = {epoch_loss:.5f}"
            f"\n\tGradient norm: {self.gradient_norm_history[-1]}"
            f"\n\tElapsed time = {duration:.3f} min")

        self._converged = self.early_stop.early_stopping(
            epoch_loss, thresh, self.params)

    @property
    def best_params(self):
        """Returns the best parameters according to the early stopping criterion."""
        return self.early_stop.best_params

    def move_to_device(self):
        """Transforms the trainer states to JAX arrays."""
        super().move_to_device()
        self.early_stop.move_to_device()


class RelativeEntropy(tt.PropagationBase):
    """Trainer for relative entropy minimization.

    The Relative Entropy Minimization procedure coarse-graines potential
    models by minimizing the relative entropy between the atomistic reference
    and coarse-grained target canonical distributions [#Shell2008]_
    [#Thaler2022]_.

    The relative entropy algorithm currently assume a NVT ensemble.

    Args:
        init_params: Initial energy parameters.
        optimizer: Optimizer from optax.
        reweight_ratio: Ratio of reference samples required for n_eff to
            surpass to allow re-use of previous reference trajectory state.
            If trajectories should not be re-used, a value > 1 can be specified.
        sim_batch_size: Number of state-points to be processed as a single
            batch. Gradients will be averaged over the batch before stepping the
            optimizer.
        energy_fn_template: Function that takes energy parameters and
            initializes an new energy function. Here, the ``energy_fn_template``
            is only a reference that will be saved alongside the trainer.
            Each state point requires its own due to the dependence on the box
            size via the displacement function, which can vary between state points.
        convergence_criterion: Either ``'max_loss'`` or ``'ave_loss'``.
            If ``'max_loss'``, stops if the gradient norm cross all batches in
            the epoch is smaller than convergence_thresh.
            ``'ave_loss'`` evaluates  the average gradient norm across the batch.
            For a single state point, both are equivalent.
        checkpoint_path: Path to the folder to store ckeckpoints in.
        full_checkpoint: Save the whole trainer instead of only the inference
            data.

    Attributes:
        data_states: Dictionary containing the dataloader states for each
            state points.
        delta_re: Dictionary containing the improvement of the relative entropy
            with respect to the initial potential.
        step_size_history: List of step size scales for each batched update.
        gradient_norm_history: List of gradient norms for each batched update.
        weight_fn: Dictionary containing the reweighting functions for each
            statepoint.
        early_stop: Instance of EarlyStopping to check for convergence.

    References:
        .. [#Shell2008] Shell, M. S. The Relative Entropy Is Fundamental to
           Multiscale and Inverse Thermodynamic Problems. J. Chem. Phys. 2008,
           129 (14), 144108. https://doi.org/10.1063/1.2992060.
        .. [#Thaler2022] Thaler, S.; Stupp, M.; Zavadlav, J. Deep Coarse-Grained
           Potentials via Relative Entropy Minimization. The Journal of Chemical
           Physics 2022, 157 (24), 244103. https://doi.org/10.1063/5.0124538.

    """
    def __init__(self,
                 init_params,
                 optimizer,
                 reweight_ratio: float = 0.9,
                 sim_batch_size: int = 1,
                 energy_fn_template: EnergyFnTemplate = None,
                 convergence_criterion: str = "window_median",
                 checkpoint_path: os.PathLike = "Checkpoints",
                 full_checkpoint: bool = False):
        init_trainer_state = util.TrainerState(
            params=init_params, opt_state=optimizer.init(init_params))
        super().__init__(init_trainer_state, optimizer, checkpoint_path,
                         reweight_ratio, sim_batch_size, energy_fn_template,
                         full_checkpoint)

        # in addition to the standard trajectory state, we also need to keep
        # track of dataloader states for reference snapshots
        self.data_states = {}
        self.delta_re = self.checkpoint("delta_re", {})
        self.step_size_history = self.checkpoint("step_size_history", [])
        self.gradient_norm_history = self.checkpoint("gradient_norm_history", [])

        self.early_stop = tt.EarlyStopping(self.params, convergence_criterion)

    def _set_dataset(self, key, reference_data, reference_batch_size,
                     batch_cache=1):
        """Set dataset and loader corresponding to current state point."""
        reference_loader = numpy_loader.NumpyDataLoader(
            R=reference_data, copy=False)
        init_ref_batch, get_ref_batch, _ = data_loaders.init_batch_functions(
            data_loader=reference_loader, mb_size=reference_batch_size,
            cache_size=batch_cache,
            prefetch=chemtrain_config.read("async_dataloading", True),
        )
        init_reference_batch_state = init_ref_batch(shuffle=True)
        self.data_states[key] = init_reference_batch_state
        return get_ref_batch

    def add_statepoint(self,
                       reference_data: ArrayLike,
                       energy_fn_template: EnergyFnTemplate,
                       simulator_template: Callable,
                       neighbor_fn: NeighborFn,
                       timings: sampling.TimingClass,
                       state_kwargs: Dict[str, ArrayLike],
                       reference_state,
                       reference_batch_size: int = None,
                       batch_cache: int = 1,
                       initialize_traj: bool = True,
                       set_key: str = None,
                       vmap_batch: int = 10,
                       resample_simstates: bool = False,
                       allowed_reduction: float = None,
                       adaption_kwargs: Dict = None):
        """
        Adds a state point to the pool of simulations.

        The gradient of the relative entropy is computed via the gradient
        function initialized by
        :func:`chemtrain.learn.difftre.init_rel_entropy_gradient_and_propagation`.

        As each reference dataset / trajectory corresponds to a single
        state point, we initialize the dataloader together with the
        simulation.

        Currently only supports NVT simulations.

        Args:
            reference_data: De-correlated reference trajectory
            energy_fn_template: Function that takes energy parameters and
                initializes an new energy function.
            simulator_template: Function that takes an energy function and
                returns a simulator function.
            neighbor_fn: Neighbor function
            timings: Instance of TimingClass containing information
                about the trajectory length and which states to retain
            state_kwargs: Properties defining the thermodynamic state. Must
                at least contain the temperature 'kT'.
            reference_state: Tuple of initial simulation state and neighbor list
            reference_batch_size: Batch size of dataloader for reference
                trajectory. If None, will use the same number of snapshots as
                generated via the optimizer.
            batch_cache: Number of reference batches to cache in order to
                minimize host-device communication. Make sure the cached data
                size does not exceed the full dataset size.
            initialize_traj: True, if an initial trajectory should be generated.
                Should only be set to False if a checkpoint is loaded before
                starting any training.
            set_key: Specify a key in order to restart from same statepoint.
                By default, uses the index of the sequance statepoints are
                added, i.e. ``self.trajectory_states[0]`` for the first added
                statepoint. Can be used for changing the timings of the
                simulation during training.
            vmap_batch: Batch size of vmapping of per-snapshot energy and
                gradient calculation.
            allowed_reduction: Allowed reduction of the effective sample size
                for the given statepoint.
            adaption_kwargs: Additional keyword arguments for the step size
                line search. For a description, see
                :func:`chemtrain.learn.difftre.init_step_size_adaption`.
        """
        if reference_batch_size is None:
            print("No reference batch size provided. Using number of generated "
                  "CG snapshots by default.")
            states_per_traj = jnp.size(timings.t_production_start)
            if reference_state.sim_state.position.ndim > 2:
                n_trajectories = reference_state.sim_state.position.shape[0]
                reference_batch_size = n_trajectories * states_per_traj
            else:
                reference_batch_size = states_per_traj

        (key, *reweight_fns) = self._init_statepoint(reference_state,
                                                     energy_fn_template,
                                                     simulator_template,
                                                     neighbor_fn,
                                                     timings,
                                                     state_kwargs,
                                                     set_key,
                                                     vmap_batch,
                                                     initialize_traj,
                                                     entropy_approximation=False,
                                                     resample_simstates=resample_simstates,
                                                     safe_propagation=False)

        reference_dataloader = self._set_dataset(key,
                                                 reference_data,
                                                 reference_batch_size,
                                                 batch_cache)

        propagation_and_grad = difftre.init_rel_entropy_gradient_and_propagation(
            reference_dataloader, reweight_fns, energy_fn_template,
            state_kwargs["kT"], vmap_batch
        )

        self.grad_fns[key] = propagation_and_grad
        self.delta_re[key] = []
        self.weight_fn[key] = jax.jit(reweight_fns[0])

        if allowed_reduction is not None:
            if adaption_kwargs is None:
                adaption_kwargs = {}

            self._adaptive_step_size[key] = difftre.init_step_size_adaption(
                self.weights_fn[key], allowed_reduction, **adaption_kwargs
            )

    def _update(self, batch):
        """Updates the potential using the gradient from relative entropy."""
        grads = []
        for sim_key in batch:
            grad_fn = self.grad_fns[sim_key]

            self.trajectory_states[sim_key], delta_re, curr_grad, \
            self.data_states[sim_key] = grad_fn(self.params,
                                                self.trajectory_states[sim_key],
                                                self.data_states[sim_key])
            grads.append(curr_grad)
            self.delta_re[sim_key].append(delta_re)


        batch_grad = util.tree_mean(grads)

        step_size = 1.0
        proposal = self._optimizer_step(batch_grad)
        for sim_key in batch:
            if sim_key not in self._adaptive_step_size: continue

            alpha, residual = self._adaptive_step_size[sim_key](
                self.params, batch_grad, proposal,
                self.trajectory_states[sim_key]
            )

            if alpha < step_size:
                step_size = alpha

        print(f"[Step Size] Found optimal step size {step_size} with residual "
              f"{residual}", flush=True)

        self._step_optimizer(batch_grad, alpha=step_size)

        batch_norm = util.tree_norm(batch_grad)
        self.gradient_norm_history.append(onp.asarray(batch_norm))
        self.step_size_history.append(onp.asarray(step_size))


    def _evaluate_convergence(self, *args, thresh=None, **kwargs):
        curr_grad_norm = self.gradient_norm_history[-1]
        # Mean loss from last simbatch
        mean_delta_re = onp.mean(
            [delta_re[-1] for delta_re in self.delta_re.values()]
        )
        duration = self.update_times[self._epoch]

        print(
            f"\n[RE] Epoch {self._epoch}"
            f"\n\tMean Delta RE loss = {mean_delta_re:.5f}"
            f"\n\tGradient norm: {curr_grad_norm}"
            f"\n\tElapsed time = {duration:.3f} min")

        self._print_measured_statepoint()

        self._converged = self.early_stop.early_stopping(
            curr_grad_norm, thresh, save_best_params=False)


class SGMCForceMatching(tt.ProbabilisticFMTrainerTemplate):
    """Trainer for stochastic gradient Markov-chain Monte Carlo training
    based on force-matching.

    init_samples: A list, possibly of size 1, of sets of initial MCMC samples,
     where each spawns a dedicated MCMC chain,
    """
    def __init__(self, sgmc_solver, init_samples, val_dataloader=None,
                 energy_fn_template=None):
        # TODO: Where does alias.py get checkpoint_path info?
        super().__init__(None, energy_fn_template)
        self._params = [init_sample["params"] for init_sample in init_samples]
        self.sgmcmc_run_fn = sgmc_solver
        self.init_samples = init_samples

        # TODO use val dataloader to compute posterior predictive p value or
        #  other convergence metric. In ProbabilisticFMTrainerTemplate??

        # TODO also use test_set?

    def train(self, iterations):
        """Training of any trainer should start by calling train."""
        self.results = self.sgmcmc_run_fn(*self.init_samples,
                                          iterations=iterations)

    @property
    def params(self):
        """Get the sampled parameters from all chains."""
        if len(self.results) == 1:  # single chain
            return self.results[0]["samples"]["variables"]["params"]
        else:
            params = []
            for chain in self.results:
                params.append(chain["samples"]["variables"]["params"])
            stacked_params = util.tree_stack(params)
            return util.tree_combine(stacked_params)

    @params.setter
    def params(self, loaded_params):
        raise NotImplementedError("Setting params seems not meaningful in"
                                  " the case of SG-MCMC samplers.")

    @property
    def list_of_params(self):
        """A list of the sampled parameters."""
        return util.tree_unstack(self.params)

    def save_trainer(self, save_path):
        """Save the trainer to a file."""
        raise NotImplementedError("Saving the trainer currently does not work"
                                  " for SGMCMC.")


class EnsembleOfModels(tt.ProbabilisticFMTrainerTemplate):
    """Train an ensemble of models by starting optimization from different
    initial parameter sets, for use in uncertainty quantification applications.

    Example:

        .. code-block:: python

           trainer_list = []
           for i in range(4):
               trainer_list.append(trainers.ForceMatching(...))
           trainer_ensemble = trainers.EnsembleOfModels(trainer_list)

           trainer_ensemble.train(*args, **kwargs)
           trained_params = trainer_ensemble.list_of_params

    """
    def __init__(self, trainers, ref_energy_fn_template=None):
        super().__init__(None, ref_energy_fn_template)
        self.trainers = trainers

    def train(self, *args, **kwargs):
        for i, trainer in enumerate(self.trainers):
            print(f"---------Starting trainer {i}-----------")
            trainer.train(*args, **kwargs)
        print("Finished training all models.")

    @property
    def params(self):
        return util.tree_stack(self.list_of_params)

    @params.setter
    def params(self, loaded_params):
        for i, params in enumerate(loaded_params):
            self.trainers[i].params = params

    @property
    def list_of_params(self):
        params = []
        for trainer in self.trainers:
            if hasattr(trainer, "best_params"):
                params.append(trainer.best_params)
            else:
                params.append(trainer.params)
        return params


class InterleaveTrainers(tt.TrainerInterface):
    """Interleaves updates to train models using multiple algorithms.

    This special trainer allows to train models simultaneously with different
    algorithms.

    Example:

        .. code-block::

            # First initialize the base-trainers, e.g.
            fm_trainer = trainers.ForceMatching(...)

            difftre_trainer = trainers.Difftre(...)
            difftre_trainer.add_statepoint(...)

            # Now combine the trainers. The trainers are executed in the
            # order in which they are added

            trainer = trainers.InterleaveTrainers('checkpoint_folder',
                                                  energy_fn_template,
                                                  full_checkpoint=False)

            # Force matching should run 10 epochs before difftre runs 2 epochs
            trainer.add_trainer(fm_trainer, num_updates=10, name='Force Matching')
            trainer.add_trainer(difftre_trainer, num_updates=2, name='DiffTRe')

            trainer.train(100, checkpoint_frequency=10)

    Args:
        sequential: Start the next trainer directly with the optimized
            parameters of the previous trainer. In the non-sequential case,
            the trainers start their epoch on the same parameter set and
            the final update is a weighted sum of both updates.
        checkpoint_base_path: Location to store checkpoints of the trainers.
        reference_energy_fn_template: Energy function template to optionally
            return an energy function with current parameters.
        full_checkpoint: Store the complete trainer or important properties
            only.

    """


    def __init__(self,
                 sequential = True,
                 checkpoint_base_path = "checkpoints",
                 reference_energy_fn_template=None,
                 full_checkpoint=False):
        super().__init__(checkpoint_base_path, reference_energy_fn_template,
                         full_checkpoint)
        self.sequential = sequential
        self._trainers = []
        self._epoch = 0

    def add_trainer(self, trainer, num_updates: int = 1, name: str = "trainer",
                    weight: float = 1.0, **trainer_kwargs):
        """Adds a trainer to the combined training.

        The trainers are executed in the order they are added to this instance.
        It is possible to specify how many epochs each trainer should train
        before the next trainer starts again.

        Args:
            trainer: Trainer to add to the chain.
            num_updates: Consecutive updates of the trainer in one epoch of the
                interleaved trainer.
            name: Display name of the trainer.
            weight: Weight for the interpolated update of the parameters.
            trainer_kwargs: Additional arguments for the training method
                of the trainer.

        """
        self._trainers.append(
            {"trainer": trainer, "num_updates": num_updates, "name": name,
             "kwargs": trainer_kwargs, "weight": weight}
        )

    @property
    def params(self):
        return self._trainers[-1]["trainer"].params

    @params.setter
    def params(self, params):
        for trainer in self._trainers:
            trainer["trainer"].params = params

    @property
    def _all_params(self):
        return [t["trainer"].params for t in self._trainers]

    @property
    def _all_weights(self):
        return [t["weight"] for t in self._trainers]

    def _init_interpolated_update(self):
        weights = jnp.asarray(self._all_weights)
        weights /= jnp.sum(weights)
        @jit
        def update(parameters):
            # Scale the parameters
            structure = tree_util.tree_structure(parameters[0])
            leaves = [tree_util.tree_leaves(t) for t in parameters]
            concat = [jnp.concatenate(l) for l in zip(*leaves)]
            summed = [jnp.sum(weights * l, axis=0) for l in concat]
            return tree_util.tree_unflatten(structure, summed)
        return update

    def train(self, epochs, checkpoint_frequency=None):
        """Train model with combined algorithms.

        Args:
            epochs: Number of epochs, where one epoch can contain multiple
                epochs for each added trainer.
            checkpoint_frequency: Save a checkpoint in the given frequency.

        """
        interpolated_update = self._init_interpolated_update()
        self._converged = False
        start_epoch = self._epoch
        end_epoch = start_epoch + epochs
        for e in range(start_epoch, end_epoch):
            start = time.time()
            for t, trainer in enumerate(self._trainers):
                print(f"---------Starting trainer {trainer['name']} for {trainer['num_updates']} updates -----------")
                trainer["trainer"].train(trainer["num_updates"], **trainer["kwargs"])

                next = (t + 1) % len(self._trainers)

                if self.sequential:
                    # Pass updated parameters to the next trainer
                    self._trainers[next]["trainer"].params = trainer["trainer"].params
            if not self.sequential:
                # Update the parameters of all trainers with a weighted sum of
                # the individual parameters
                self.params = interpolated_update(self.params)

            duration = (time.time() - start) / 60.
            self._epoch += 1
            print(f"Finished epoch {e} for all trainers in {duration : .2f} minutes.")
            self._dump_checkpoint_occasionally(frequency=checkpoint_frequency)

    def move_to_device(self):
        for trainer in self._trainers:
            trainer["trainer"].move_to_device()

    def save_trainer(self, save_path, format=".pkl"):
        data = {}
        for t, trainer in enumerate(self._trainers):
            number = str(t + 1).rjust(3, "0")
            key = "trainer_{0}_{1}".format(trainer["name"], number)
            data[key] = trainer["trainer"].save_trainer(None, format="none")

        if format == ".pkl":
            with open(save_path, "wb") as pickle_file:
                pickle.dump(data, pickle_file)
        elif format == "none":
            return data
