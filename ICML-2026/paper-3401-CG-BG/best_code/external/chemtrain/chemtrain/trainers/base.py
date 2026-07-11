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

"""Abstract templates for trainers, defining common functionality and
requirements."""
import abc
import copy
import dataclasses
import functools
import logging
import pathlib
import sys
import time
import warnings
from abc import abstractmethod
from os import PathLike
import inspect
from typing import Callable, Dict, Any, Optional
from types import ModuleType

import cloudpickle as pickle
import jax
import numpy as onp
from jax import (
    numpy as jnp, random, device_count, jit, device_get,
    tree_util
)
from jax.tree_util import tree_map
from jax_sgmc import data

from chemtrain import util
from chemtrain import config as chemtrain_config
from chemtrain.data import data_loaders
from chemtrain.learn import max_likelihood, difftre
from chemtrain.ensemble import sampling, reweighting
from chemtrain.typing import EnergyFnTemplate
from chemtrain.util import format_not_recognized_error

from jax_md_mod.model import dropout


class CaptureStdout:
    """Capture stdout and writes to file.

    This context manager writes messages to stdout and a file.

    Args:
        file: Path to file where to write the stdout.

    """
    def __init__(self, file=None):
        self.files = []

        # Only write to files on root process
        if file is not None and util.is_root():
            self.files = [file]

    def write(self, message):
        for f in self.out:
            f.write(message)
            f.flush()

    def flush(self):
        for f in self.out:
            f.flush()

    def __enter__(self):
        try:
            self.files = [open(file_path, "w") for file_path in self.files]
            self.out = (sys.stdout, *self.files)
            sys.stdout = self
        except Exception as e:
            self.__exit__(None, None, None)
            raise e
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.out[0]
        for f in self.files:
            try:
                f.close()
            except Exception as e:
                print(f"Error closing file: {e}")


@dataclasses.dataclass
class CheckpointAttr:
    name: str
    object: Any


class TrainerInterface(metaclass=abc.ABCMeta):
    """Abstract class defining the user interface of trainers as well as
    checkpointing functionality.
    """

    def __init__(self,
                 checkpoint_path,
                 reference_energy_fn_template=None,
                 full_checkpoint=True):
        """A reference energy_fn_template can be provided, but is not mandatory
        due to the dependence of the template on the box via the displacement
        function.
        """
        checkpoint_path = pathlib.Path(checkpoint_path)
        checkpoint_path.mkdir(exist_ok=True, parents=True)

        self._statistics: Dict[str, str] = {}
        self._full_checkpoint = full_checkpoint
        self.checkpoint_path = checkpoint_path
        self._epoch = self.checkpoint("epoch", 0)
        self.reference_energy_fn_template = reference_energy_fn_template

    @property
    def energy_fn(self):
        """Returns the energy function for the current parameters."""
        if self.reference_energy_fn_template is None:
            raise ValueError("Cannot construct energy_fn as no reference "
                             "energy_fn_template was provided during "
                             "initialization.")
        return self.reference_energy_fn_template(self.params)

    def _dump_checkpoint_occasionally(self, *args, checkpoint_frequency=None, **kwargs):
        """Dumps a checkpoint during training, from which training can
        be resumed.
        """
        if not self.is_root(): return

        assert self.checkpoint_path is not None
        if checkpoint_frequency is not None:
            pathlib.Path(self.checkpoint_path).mkdir(parents=True,
                                                     exist_ok=True)
            if self._epoch % checkpoint_frequency == 0:  # checkpoint model
                epoch = str(self._epoch).rjust(5, "0")
                file_path = (
                    pathlib.Path(self.checkpoint_path) / f"epoch{epoch}.pkl")
                self.save_trainer(file_path)
                print(f"[{type(self).__name__}] Checkpoint created sucessfully at: {str(file_path)}")

    def save_trainer(self, save_path, format=".pkl"):
        """Saves whole trainer, e.g. for production after training."""
        if self._full_checkpoint:
            data = self
        else:
            data = {
                name: self.__getattribute__(key)
                for key, name in self._statistics.items()
            }

        if format == ".pkl":
            # Only root process saves trainer
            if not self.is_root():
                return

            leaves, treedef = tree_util.tree_flatten(data)
            leaves = [
                onp.asarray(leaf) if isinstance(leaf, jnp.ndarray) else leaf
                for leaf in leaves
            ]
            with open(save_path, "wb") as pickle_file:
                pickle.dump(tree_util.tree_unflatten(treedef, leaves), pickle_file)
        elif format == "none":
            return data

    def save_energy_params(self, file_path, save_format=".hdf5", best=False):
        """Saves energy parameters.

        Args:
            file_path: Path to the file where to save the energy parameters.
                Currently, only saving to pickle files (``"*.pkl"``) is
                supported.
            save_format: Format in which to save the energy parameters.
            best: If True, tries to save the best parameters, e.g., on the
                validation loss. If no criterion to determine the best params
                was specified, saves the latest parameters instead.

        """
        if not self.is_root():
            return

        if best:
            try:
                params = self.best_params
            except AttributeError:
                warnings.warn(
                    f"Saving best params is not possible, saving the last "
                    f"paramters.")
                params = self.params
        else:
            params = self.params

        if save_format == ".hdf5":
            raise NotImplementedError
        elif save_format == ".pkl":
            with open(file_path, "wb") as pickle_file:
                pickle.dump(device_get(params), pickle_file)
        else:
            format_not_recognized_error(save_format)

    def load_energy_params(self, file_path):
        """Loads energy parameters.

        Args:
            file_path: Path to the file containing the energy parameters.
                Currently, only loading from pickle files (``"*.pkl"``) is
                supported.

        """
        if file_path.endswith(".hdf5"):
            raise NotImplementedError
        elif file_path.endswith(".pkl"):
            with open(file_path, "rb") as pickle_file:
                params = pickle.load(pickle_file)
        else:
            format_not_recognized_error(file_path[-4:])
        self.params = tree_map(jnp.array, params)  # move state on device

    @property
    @abc.abstractmethod
    def params(self):
        """Short-cut for parameters. Depends on specific trainer."""

    @params.setter
    @abc.abstractmethod
    def params(self, loaded_params):
        raise NotImplementedError()

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        """Training of any trainer should start by calling train."""

    @abc.abstractmethod
    def move_to_device(self):
        """Move all attributes that are expected to be on device to device to
         avoid TracerExceptions after loading trainers from disk, i.e.
         loading numpy rather than device arrays.
         """

    def checkpoint(self, name, object):
        """Marks attribute to be saved in a partial checkpoint.

        The marked attribute is saved to a checkpoint dictionary under
        the specified name.

        Args:
            name: Name of the statistic in the saved dictionary
            object: Object to initialize the attribute

        Returns:
            Returns the original object wrapped as a CheckpointNode.

        """
        return CheckpointAttr(name, object)

    def __setattr__(self, key, value):
        if isinstance(value, CheckpointAttr):
            # The wrapper class is only to identify attributes to be checkpointed.
            # We now track these attributes in a dictionary and remove the wrapper,
            # which is no longer needed.
            if value.name in self._statistics.values():
                for duplicate_key, duplicate_value in self._statistics.items():
                    if duplicate_key == key:
                        warnings.warn(f"[{self.__class__.__name__}] Attribute {duplicate_key} is marked for checkpoining twice.")
                        continue

                    if duplicate_value == value.name:
                        raise ValueError(
                            f"Duplicate checkpoint name found for attribute {key}. "
                            f"Name '{value.name}' is already used for attribute {duplicate_key}."
                        )

            self._statistics[key] = value.name

            object.__setattr__(self, key, value.object)
        else:
            object.__setattr__(self, key, value)

    @staticmethod
    def use_mpi():
        """Checks whether MPI is being used."""
        return util.use_mpi()

    @staticmethod
    def is_root():
        """Checks whether current process is root process."""
        return util.is_root()

    @staticmethod
    def get_mpi() -> ModuleType | None:
        """Returns the MPI interface."""
        return util.get_mpi()


    def restore(self, checkpoint):
        """Restores the trainer from a checkpoint.

        Args:
            checkpoint: Checkpoint to restore from. Can be a path to a file or
                a dictionary containing the trainer state.

        """
        with open(checkpoint, "rb") as f:
            checkpoint = pickle.load(f)

        # Restore all attributes that were marked as checkpointable
        restored = []
        for attr, key in self._statistics.items():
            object.__setattr__(self, attr, checkpoint[key])
            restored.append(attr)

        # Finish this epoch
        self._epoch += 1

        # Write summary
        print(f"[{self.__class__.__name__}] Attributes Restored:")
        print(f", ".join(restored))

        unchanged = [
            attr for attr in self.__dict__.keys() if attr not in restored
        ]
        print(f"[{self.__class__.__name__}] Attributes Unchanged:")
        print(f", ".join(unchanged))


class MLETrainerTemplate(TrainerInterface):
    """Abstract class implementing common properties and methods of single
    point estimate Trainers using optax optimizers.

    Args:
        optimizer: Optax optimizer
        init_state: Initial state of optimizer and model
        checkpoint_path: Path to folder where checkpoints are saved
        full_checkpoint: Whether to save the full trainer with pickle or only
            a subset of attributes.
        log_file: Write loggs of Trainer to the file specified by path.
        reference_energy_fn_template: Function returning a concrete energy
            function for the current parameters

    The MLE trainer performs a sequence of task before and after each training,
    epoch and batch update. It is possible to add custom tasks to the trainer
    via :func:`MLETrainerTemplate.add_task`.

    Attributes:
        update_times: Computation time of each update
        gradient_norm_history: Norms of the gradient for each update

    """

    def __init__(self,
                 optimizer,
                 init_state: util.TrainerState,
                 checkpoint_path: PathLike,
                 full_checkpoint: bool = True,
                 log_file: PathLike = None,
                 reference_energy_fn_template: EnergyFnTemplate = None):
        super().__init__(
            checkpoint_path, reference_energy_fn_template, full_checkpoint)
        self.optimizer = optimizer

        self.state = self.checkpoint("trainer_state", init_state)
        self.update_times = self.checkpoint("update_times", [])
        self.gradient_norm_history = self.checkpoint("gradient_norm_history", [])
        self._converged = self.checkpoint("converged", False)
        self._diverged = self.checkpoint("diverged", False)

        self.log_file = log_file
        self._tasks = {}

        # Add standard tasks
        self.add_task("pre_epoch", self._update_times_start)
        self.add_task("post_epoch", self._update_times_end)
        self.add_task("post_epoch", self._evaluate_convergence)
        self.add_task("post_epoch", self._dump_checkpoint_occasionally)

        # Dropout only if params indicate necessity
        if dropout.dropout_is_used(self.params):
            self.add_task("post_batch", self._update_dropout)

        # Note: make sure not to use such a construct during training as an
        # if-statement based on params forces the python part to wait for the
        # completion of the batch, hence losing the advantage of asynchronous
        # dispatch, which can become the bottleneck in high-throughput learning.

        self.release_fns = {}

    def add_task(self, trigger, fn_or_method):
        """Adds a tasks to perform regularly during training.

        Args:
            trigger: The trigger at which the task is executed. Can be
                ``"pre/post_training/epoch/batch"``.
            fn_or_method: The function or method to be executed.

        Example:

            The following code adds a task printing a specific energy parameter
            after each epoch.

            .. code-block:: python

                def print_parameter(trainer, *args, **kwargs):
                   print(f"Parameter after epoch {trainer._epoch}: "
                         f"{trainer.state.params["parameter"]}")

                trainer.add_task("post_epoch", print_parameter)

        """

        valid_triggers = [
            "pre_training",
            "pre_epoch",
            "pre_batch",
            "post_batch",
            "post_epoch",
            "post_training"
        ]

        assert trigger in valid_triggers, (
            f"Provided trigger {trigger} is invalid, can only be "
            f"{valid_triggers}."
        )

        if trigger not in self._tasks:
            self._tasks[trigger] = []

        self._tasks[trigger].append(fn_or_method)

        return fn_or_method

    def _execute_tasks(self, trigger, *args, **kwargs):
        """Executes a dynamical set of tasks."""
        if trigger not in self._tasks.keys():
            return

        for fn_or_method in self._tasks[trigger]:
            if inspect.ismethod(fn_or_method):
                fn_or_method(*args, **kwargs)
            else:
                fn_or_method(self, *args, **kwargs)

    def print_training_tasks(self):
        """Prints the tasks performed by the trainer."""

        print("Preparation:")
        if "pre_training" in self._tasks:
            for task in self._tasks["pre_training"]:
                print(f" - {task}")
        else:
            print("<no preparation tasks>")

        print("\nFor every EPOCH\n===============")
        if "pre_epoch" in self._tasks:
            for task in self._tasks["pre_epoch"]:
                print(f" - {task}")
        else:
            print("<no pre-epoch tasks>")

        print(f"\n\tFor every BATCH in EPOCH\n\t----------------")
        if "pre_batch" in self._tasks:
            for task in self._tasks["pre_batch"]:
                print(f"\t - {task}")
        else:
            print("\t<no pre-batch tasks>")

        print(f"\n\tUPDATE\n")

        if "post_batch" in self._tasks:
            for task in self._tasks["post_batch"]:
                print(f" - {task}")
        else:
            print("\t<no post-batch tasks>")

        print("")

        if "post_epoch" in self._tasks:
            for task in self._tasks["post_epoch"]:
                print(f" - {task}")
        else:
            print("<no post_epoch tasks>")

        print("\nPostprocessing:")
        if "post_training" in self._tasks:
            for task in self._tasks["post_training"]:
                print(f" - {task}")
        else:
            print("<no postprocessing tasks>")

    def _optimizer_step(self, curr_grad):
        """Wrapper around step_optimizer that is useful whenever the
        update of the optimizer can be done outside jit-compiled functions.

        Returns:
            Returns the parameters after an update of the optimizer, but
            without updating the internal states.
        """
        new_params, _ = max_likelihood.step_optimizer(
            self.params, self.state.opt_state, curr_grad, self.optimizer)
        return new_params

    def _step_optimizer(self, curr_grad, alpha=1.0):
        """Wrapper around step_optimizer that is useful whenever the
        update of the optimizer can be done outside of jit-compiled functions.
        """
        new_params, new_opt_state = max_likelihood.step_optimizer(
            self.params, self.state.opt_state, curr_grad, self.optimizer)

        # Do an optimized update
        new_params = tree_map(
            lambda old, new: old * (1 - alpha) + new * alpha,
            self.params, new_params
        )
        self.state = self.state.replace(params=new_params,
                                        opt_state=new_opt_state)

    def train(self, max_epochs, thresh=None, checkpoint_freq=None):
        """Trains for a maximum number of epochs, checkpoints after a
        specified number of epochs and ends training if a convergence
        criterion is met. This function can be called multiple times to extend
        training.

        This function only implements the training sceleton by splitting the
        training into epochs and batches as well as providing checkpointing and
        ending of training if the convergence criterion is met. The specifics
        of dataloading, parameter updating and convergence criterion evaluation
        needs to be implemented in ``_get_batch()``, ``_update()`` and
        ``_evaluate_convergence()``, respectively, depending on the exact trainer
        details to be implemented.

        Args:
            max_epochs: Maximum number of epochs for which training is
                continued. Training will end sooner if convergence criterion is
                met.
            thresh: Threshold of the early stopping convergence criterion. If
                None, no early stopping is applied. Definition of thresh depends
                on specific convergence criterion. See :class:`EarlyStopping`.
            checkpoint_freq: Number of epochs after which a checkpoint is saved.
                By default, do not save checkpoints.
        """
        self._converged = False
        start_epoch = self._epoch
        end_epoch = start_epoch + max_epochs

        with CaptureStdout(self.log_file):
            self._execute_tasks("pre_training")
            for _ in range(start_epoch, end_epoch):
                try:
                    self._execute_tasks("pre_epoch")
                    for batch in self._get_batch():
                        self._execute_tasks("pre_batch", batch)
                        self._update(batch)
                        self._execute_tasks("post_batch", batch)
                    self._execute_tasks("post_epoch",
                                        checkpoint_frequency=checkpoint_freq,
                                        convergence_thresh=thresh)
                    self._epoch += 1
                except RuntimeError as err:
                    # In case the simulation diverges, break the optimization
                    # and checkpoint the last state such that an analysis can
                    # be performed.
                    self._diverged = True
                    if self.checkpoint_path is not None:
                        path = (self.checkpoint_path / f"epoch{self._epoch - 1}_error_state.pkl")
                        self.save_trainer(save_path=path)
                    print(f"Training has been unsuccessful due to the following"
                          f" error: {err}")
                    break

                if self._converged:
                    break
            else:
                if thresh is not None:
                    print("Maximum number of epochs reached without convergence.")
            self._execute_tasks("post_training")

    def _update_dropout(self, batch):
        """Updates params, while keeping track of Dropout."""
        # TODO refactor this as this needs to wait for when
        #  params will again be available, slowing down re-loading
        #  of batches. We could set dropout key as kwarg and keep
        #  track of keys in this class. Also refactor dropout in
        #  DimeNet taking advantage of haiku RNG key management and
        #  built-in dropout in MLP
        params = dropout.next_dropout_params(self.params)
        self.params = params

    def _update_times_start(self, *args, **kwargs):
        self.update_times.append(time.time())

    def _update_times_end(self, *args, **kwargs):
        self.update_times[self._epoch] -= time.time()
        self.update_times[self._epoch] /= -60.

    @abc.abstractmethod
    def _get_batch(self):
        """A generator that returns the next batch that will be provided to the
        _update function. The length of the generator should correspond to the
        number of batches per epoch.
        """

    @abc.abstractmethod
    def _update(self, batch):
        """Uses the current batch to updates self.state via the training scheme
        implemented in the specific trainer. Can additionally save auxilary
        optimization results, such as losses and observables, that can be
        used by _evaluate_convergence and for post-processing.
        """

    @abc.abstractmethod
    def _evaluate_convergence(self, duration, thresh, *args, **kwargs):
        """Checks whether a convergence criterion has been met. Can also be
        used to print callbacks, such as time per epoch and loss vales.
        """

    def move_to_device(self):
        """Converts all arrays of the trainer state to JAX arrays."""
        self.state = tree_map(jnp.array, self.state)  # move on device

    def _release_data_references(self):
        for release in self.release_fns.values():
            release()
        self.release_fns = {}


class PropagationBase(MLETrainerTemplate):
    """Trainer base class for shared functionality whenever (multiple)
    simulations are run during training. Can be used as a template to
    build other trainers. Currently used for DiffTRe and relative entropy.

    We only save the latest generated trajectory for each state point.
    While accumulating trajectories would enable more frequent reweighting,
    this effect is likely minor as past trajectories become exponentially
    less useful with changing potential. Additionally, saving long trajectories
    for each statepoint would increase memory requirements over the course of
    the optimization.
    """
    def __init__(self, init_trainer_state, optimizer, checkpoint_path,
                 reweight_ratio=0.9, sim_batch_size=1, energy_fn_template=None,
                 full_checkpoint=True, key=None, log_dir=None,):
        super().__init__(optimizer, init_trainer_state, checkpoint_path,
                         full_checkpoint, log_dir, energy_fn_template)
        self.sim_batch_size = sim_batch_size
        self.reweight_ratio = reweight_ratio

        if key is None:
            self.key = random.PRNGKey(0)

        # store for each state point corresponding traj_state and grad_fn
        # save in distinct dicts as grad_fns need to be deleted for checkpoint
        self.grad_fns, self.statepoints = {}, {}
        self.n_statepoints = 0

        self.trajectory_states = self.checkpoint("trajectory_states", {})
        self.shuffle_key = self.checkpoint("key", random.PRNGKey(0))

        self.weight_fn = {}
        self._adaptive_step_size = {}

    def _init_statepoint(self, reference_state, energy_fn_template,
                         simulator_template, neighbor_fn, timings, state_kwargs,
                         set_key=None, energy_batch_size=10,
                         initialize_traj=True,
                         safe_propagation=True, entropy_approximation=False,
                         resample_simstates=False, num_init_runs=2):
        """Initializes the simulation and reweighting functions as well
        as the initial trajectory for a statepoint."""

        # TODO ref pressure only used in print and to have barostat values.
        #  Reevaluate this parameter of barostat values not used in reweighting
        # TODO document ref_press accordingly

        # TODO: Extend this function to allow for multiple statepoints to be
        #       added. Requires batch argument to be set here.

        assert "kT" in state_kwargs, (
            "Reweighting requires at least the temperature to be specified in "
            "the state_kwargs. "
        )

        # Backwards compatibility
        if isinstance(reference_state, tuple):
            warnings.warn(
                "Passing the reference state as tuple of simulator state and "
                "neighbors is deprecated. "
                "Use trajectory.traj_util.SimulatorState instead.",
                DeprecationWarning
            )
            reference_state = sampling.SimulatorState(
                sim_state=reference_state[0], nbrs=reference_state[1])

        if set_key is not None:
            key = set_key
            if set_key not in self.statepoints.keys():
                self.n_statepoints += 1
        else:
            key = self.n_statepoints
            self.n_statepoints += 1

        self.statepoints[key] = state_kwargs
        npt_ensemble = util.is_npt_ensemble(reference_state.sim_state)
        if npt_ensemble:
            assert "pressure" in state_kwargs, (
                "Reweighting in the NPT ensemble requires the pressure to be "
                "defined in the state_kwargs."
            )

        gen_init_traj, *reweight_fns = reweighting.init_pot_reweight_propagation_fns(
            energy_fn_template, simulator_template, neighbor_fn, timings,
            state_kwargs, self.reweight_ratio, npt_ensemble,
            energy_batch_size, safe_propagation=safe_propagation,
            entropy_approximation=entropy_approximation,
            resample_simstates=resample_simstates
        )

        self.key, split = random.split(self.key)

        # To get the correct timings, first compile before evaluation
        start = time.time()
        init_traj_fn = gen_init_traj.lower(split, self.params, reference_state,
                                           num_runs=num_init_runs, **state_kwargs)
        init_traj_fn = init_traj_fn.compile()
        compile_time = (time.time() - start) / 60.
        print(
            f"[Propagation] Time for trajectory compilation {key}: "
            f"{compile_time} mins"
        )

        if initialize_traj:
            start = time.time()
            init_traj = init_traj_fn(split, self.params, reference_state, **state_kwargs)
            run_time = (time.time() - start) / 60. / num_init_runs

            assert not init_traj.overflow, "[Propagation] Neighborlist buffer overflowed."
            assert not onp.any(onp.isnan(init_traj.trajectory.position)), "[Propagation] Initial simulation produced NaNs."

            print(
                f"[Propagation] Time for trajectory simulation {key}: "
                f"{run_time} mins"
            )

            self.trajectory_states[key] = init_traj
        else:
            self.trajectory_states[key] = functools.partial(init_traj_fn, split, self.params, reference_state, **state_kwargs)
            print("Not initializing the initial trajectory is only valid if "
                  "a checkpoint is loaded. In this case, please be use to add "
                  "state points in the same sequence, otherwise loaded "
                  "trajectories will not match its respective simulations.")

        return key, *reweight_fns

    @abstractmethod
    def add_statepoint(self, *args, **kwargs):
        """User interface to add additional state point to train model on."""
        raise NotImplementedError()

    @property
    def params(self):
        """Current energy parameters."""
        return self.state.params

    @params.setter
    def params(self, loaded_params):
        """Replaces the current energy parameters."""
        self.state = self.state.replace(params=loaded_params)

    def get_sim_state(self, key):
        """Gets the simulator state of a statepoint."""
        return self.trajectory_states[key].sim_state

    def _get_batch(self):
        """Helper function to re-shuffle simulations and split into batches."""
        self.shuffle_key, used_key = random.split(self.shuffle_key, 2)
        shuffled_indices = random.permutation(used_key, self.n_statepoints)
        if self.sim_batch_size == 1:
            batch_list = jnp.split(shuffled_indices, shuffled_indices.size)
        elif self.sim_batch_size == -1:
            batch_list = jnp.split(shuffled_indices, 1)
        else:
            raise NotImplementedError("Only batch_size = 1 or -1 implemented.")

        return (batch.tolist() for batch in batch_list)

    def _print_measured_statepoint(self, sim_key=None):
        """Print meausured kbT (and pressure for npt ensemble) for all
        statepoints to ensure the simulation is indeed carried out at the
        prescribed state point.
        """
        if sim_key is None:
            for sim_key in self.trajectory_states.keys():
                self._print_measured_statepoint(sim_key)
        else:
            traj = self.trajectory_states[sim_key]
            print(f"[Statepoint {sim_key}]")
            statepoint = self.statepoints[sim_key]
            measured_kbt = jnp.mean(traj.aux["kT"])
            if "pressure" in statepoint:  # NPT
                measured_press = jnp.mean(traj.aux["pressure"])
                press_print = (f"\n\tpress = {measured_press:.2f} ref_press = "
                               f"{statepoint['pressure']:.2f}")
            else:
                press_print = ""
            print(f"\tkT = {measured_kbt:.3f} ref_kT = "
                  f"{statepoint['kT']:.3f}" + press_print)

    def train(self, max_epochs, thresh=None, checkpoint_freq=None):
        assert self.n_statepoints > 0, ("Add at least 1 state point via "
                                        "'add_statepoint' to start training.")
        super().train(max_epochs, thresh=thresh,
                      checkpoint_freq=checkpoint_freq)

    @abstractmethod
    def _update(self, batch):
        """Implementation of gradient computation, stepping of the optimizer
        and logging of auxiliary results. Takes batch of simulation indices
        as input.
        """

    def init_step_size_adaption(self,
                                allowed_reduction: float = 0.5,
                                interior_points: int = 10,
                                step_size_scale: float = 1e-7
                                ) -> None:
        """Initializes a line search to tune the step size in each iteration.

        The line search optimizes step size to limit the decrease in the
        effective sample size (ESS) via the algorithm
        :func:`chemtrain.learn.difftre.init_step_size_adaption`.

        Args:
            allowed_reduction: Target reduction of the effective sample size
            interior_points: Number of interiour points
            step_size_scale: Accuracy of the found optimal interpolation
                coefficient

        Returns:
            Returns the interpolation coefficient :math:`\\alpha`.

        """

        for key, weight_fn in self.weight_fn.items():
            self._adaptive_step_size[key] = difftre.init_step_size_adaption(
                weight_fn, allowed_reduction, interior_points, step_size_scale
            )


class DataParallelTrainer(MLETrainerTemplate):
    """Trainer for parallelized MLE training based on a dataset.

    This trainer implements methods for MLE training on a dataset, where
    parallelization can simply be accomplished by pmapping over batched data.
    As pmap requires constant batch dimensions, data with unequal number of
    atoms needs to be padded and to be compatible with this trainer.
    """

    _train_loader: data.DataLoader
    _val_loader: data.DataLoader
    _test_loader: data.DataLoader

    def __init__(
        self,
        loss_fn,
        model,
        init_params,
        optimizer,
        checkpoint_path,
        batch: Optional[int] = None,
        batch_cache: int = 1,
        full_checkpoint=True,
        penalty_fn=None,
        energy_fn_template=None,
        convergence_criterion="window_median",
        log_file=None,
        disable_shmap: bool = False,
        *,
        batch_per_device: Optional[int] = None,
    ):

        self._disable_shmap = disable_shmap
        self.batched_model = model
        if disable_shmap:
            self._update_fn = max_likelihood.pmap_update_fn(
                self.batched_model, loss_fn, optimizer, penalty_fn)
            self._evaluate_fn = None
        else:
            # shmap performs better, but some replication rules are missing
            self._update_fn = max_likelihood.shmap_update_fn(
                self.batched_model, loss_fn, optimizer, penalty_fn)
            self._evaluate_fn = max_likelihood.shmap_loss_fn(
                self.batched_model, loss_fn, penalty_fn)

        self._loss_fn = loss_fn
        self.batch_cache = batch_cache

        local_device_count = int(device_count())
        mpi_world_size = 1
        if util.use_mpi():
            comm = util.get_communicator()
            if comm is not None:
                mpi_world_size = int(comm.Get_size())

        if batch is None and batch_per_device is None:
            raise ValueError(
                "`batch` (global) must be provided (or legacy `batch_per_device`)."
            )
        if batch is not None and batch_per_device is not None:
            raise ValueError("Provide only one of `batch` or `batch_per_device`.")

        if batch_per_device is not None:
            # Legacy convention: batch_per_device is local to a rank.
            local_batch_size = int(batch_per_device) * local_device_count
            global_batch_size = local_batch_size * mpi_world_size
        else:
            global_batch_size = int(batch)
            if global_batch_size % mpi_world_size != 0:
                raise ValueError(
                    f"Global batch {global_batch_size} must be divisible by MPI world size {mpi_world_size}."
                )
            local_batch_size = global_batch_size // mpi_world_size

            if local_batch_size % local_device_count != 0:
                raise ValueError(
                    f"Per-rank batch {local_batch_size} must be divisible by local device_count {local_device_count}."
                )
            batch_per_device = local_batch_size // local_device_count

        self.global_batch_size = int(global_batch_size)
        self.local_batch_size = int(local_batch_size)
        self.batch_per_device = int(batch_per_device)

        if optimizer is None:
            print(f"No optimizer specified")
            opt_state = None
        else:
            opt_state = optimizer.init(init_params)  # initialize optimizer state
        init_state = util.TrainerState(params=init_params, opt_state=opt_state)

        super().__init__(
            optimizer=optimizer, init_state=init_state,
            checkpoint_path=checkpoint_path,
            full_checkpoint=full_checkpoint,
            log_file=log_file,
            reference_energy_fn_template=energy_fn_template)

        self.train_batch_losses = self.checkpoint("train_batch_losses", [])
        self.train_losses = self.checkpoint("train_losses", [])
        self.val_losses = self.checkpoint("val_losses", [])
        self.train_target_losses = self.checkpoint("train_target_losses", {})
        self.val_target_losses = self.checkpoint("val_target_losses", {})

        self._batch_states: Dict[str, Any] = {}
        self._batches_per_epoch: Dict[str, int] = {}
        self._get_batch_fns: Dict[str, Callable] = {}

        self._early_stop = EarlyStopping(self.params, convergence_criterion)

    def reset_convergence_losses(self):
        """Resets early stopping convergence monitoring."""
        self._early_stop.reset_convergence_losses()

    def limit_batches_per_epoch(self, max_batches: int = 1):
        """Limits the number of batches per epoch.

        Args:
            max_batches: Maximum number of batches to use within one epoch.

        """

        assert self._batches_per_epoch["training"] >= max_batches, (
            "The number of batches per epoch is smaller than the requested "
            "maximum."
        )

        self._batches_per_epoch["training"] = max_batches

    def set_datasets(self, dataset, train_ratio=0.7, val_ratio=0.1, shuffle=False,
                     include_all=True):
        """Sets the datasets for training, testing and validation.

        Args:
            dataset: Dictionary containing input and target data as numpy
                arrays.
            train_ratio: Percentage of dataset to use for training.
            val_ratio: Percentage of dataset to use for validation.
            shuffle: Whether to shuffle data before splitting into
                train-val-test.
            include_all: Compute the loss for all samples of the splits by
                padding the last batch and masking out double samples.
                Not applied to the training split.

        """
        # release all references before allocating new data to avoid memory leak
        self._release_data_references()

        # Initialize the data loaders
        loaders = data_loaders.init_dataloaders(
            dataset, train_ratio, val_ratio, shuffle=shuffle)

        self.set_loader(loaders.train_loader, stage="training")
        self.set_loader(loaders.val_loader, stage="validation", include_all=include_all)
        self.set_loader(loaders.test_loader, stage="testing", include_all=include_all)

    def set_dataset(self, dataset, stage="testing", shuffle=False, include_all=False, **kwargs):
        """Sets the dataset for a single stage, e.g., training.

        Args:
            dataset: Dictionary containing input and target data as numpy
                arrays.
            stage: Stage for which to set the dataset. Can be ``"training"``,
                ``"validation"``, or ``"testing"``.
            shuffle: Whether the data should be shuffled.
            include_all: Compute the loss for all samples of the split by
                padding the last batch and masking out double samples.
                Not applied to the training split.

        """
        # Will only return one data loader
        loaders = data_loaders.init_dataloaders(
            dataset, train_ratio=1.0, val_ratio=0.0, shuffle=shuffle
        )

        self.set_loader(loaders.train_loader, stage=stage, include_all=include_all, **kwargs)

    def set_loader(self,
                   data_loader,
                   stage="training",
                   include_all=False,
                   batch_size=None,
                   rng_seed=None,
                   **kwargs):
        """Sets a data loader for a specific stage, e.g., training.

        If the dataset consists of numpy arrays, it is simpler to use
        :func:`set_dataset` or :func:`set_datasets` to set the data loaders.

        Args:
            data_loader: The data loader to set.
            stage: The stage for which to set the data loader. Can be
                ``"training"``, ``"validation"``, or ``"testing"``.
            include_all: Compute the loss for all samples of the split by
                padding the last batch and masking out double samples.
                Not applied to the training split.
            rng_seed: Seed to include random keys in the reference data.
                The keys are refreshed whenever a new batch is drawn.
            batch_size: Overwrites the default GLOBAL batch size.

        """
        if stage in self.release_fns.keys():
            self.release_fns[stage]()

        observation_count = data_loader.static_information["observation_count"]

        assert observation_count > 0

        # Overwrite batch size for splits
        if batch_size is not None:
            print(f"Chose custom batch size {batch_size} for split {stage}")

        if batch_size is None:
            batch_size = self.global_batch_size
        if batch_size > observation_count:
            batch_size = observation_count

        mpi_world_size = 1
        if util.use_mpi():
            comm = util.get_communicator()
            if comm is not None:
                mpi_world_size = comm.Get_size()

        local_device_count = int(device_count())
        divisor = int(mpi_world_size) * local_device_count
        if divisor <= 0:
            raise RuntimeError("Invalid divisor for batch sharding.")

        # Ensure GLOBAL batch is divisible by MPI world size and local device_count.
        if onp.mod(batch_size, divisor) != 0:
            new_batch_size = int(batch_size - onp.mod(batch_size, divisor))
            if new_batch_size <= 0:
                raise ValueError(
                    f"Global batch size {batch_size} too small for divisor {divisor}."
                )
            warnings.warn(
                f"Global batch size {batch_size} is not divisible by {divisor}; "
                f"using {new_batch_size} instead."
            )
            batch_size = new_batch_size

        global_batch_size = int(batch_size)
        local_batch_size = int(global_batch_size // int(mpi_world_size))

        if global_batch_size != self.global_batch_size:
            logging.info(
                f"Global batch size for stage {stage} changed to {global_batch_size} "
                f"from {self.global_batch_size}."
            )

        if include_all:
            assert stage != "training", (f"Including all samples not supported "
                                         f"for the training split.")

            # Increase the number of observations to make them divisible by
            # the batch size
            observation_count += onp.mod(
                global_batch_size - onp.mod(observation_count, global_batch_size), global_batch_size
            )

        if onp.mod(observation_count, global_batch_size) != 0:
            warnings.warn(
                f"Global batch size {global_batch_size} does not divide the number of "
                f"observations {observation_count}. "
                f"Trainer will skip {observation_count % global_batch_size} samples "
                f"for state {stage}"
            )

        # Initialize the access functions
        batch_fns = data_loaders.init_batch_functions(
            data_loader,
            mb_size=local_batch_size,
            cache_size=self.batch_cache,
            prefetch=chemtrain_config.read("async_dataloading", True),
            use_mpi=util.use_mpi(),
        )
        init_train_state, get_train_batch, release = batch_fns

        train_batch_state = init_train_state(
            shuffle=True, in_epochs=include_all, rng_seed=rng_seed, **kwargs
        )

        self._get_batch_fns[stage] = get_train_batch
        self._batch_states[stage] = train_batch_state
        self._batches_per_epoch[stage] = observation_count // global_batch_size
        self.release_fns[stage] = release

    def _get_batch_stage(self, stage, information=False):
        for _ in range(self._batches_per_epoch[stage]):
            self._batch_states[stage], train_batch = self._get_batch_fns[stage](
                self._batch_states[stage], information=information)
            yield train_batch


    def set_batches_per_epoch(self, stage="training", max_batches: int = 1):
        """Limits the number of updates within an epoch.

        Useful, e.g., when the validation loss should be computed more
        regularly.

        Args:
            stage: Key to the stage that should be limited. Currently,
                stage other than ``"training"`` is not supported to avoid
                wrongful computations of the validation loss.
            max_batches: Maximum number of batches, i.e., number of batched
                optimizer updates.

        """
        if stage != "training":
            raise NotImplementedError("Only training stage implemented.")

        self._batches_per_epoch[stage] = min([
            self._batches_per_epoch[stage], max_batches
        ])


    def _get_batch(self):
        return self._get_batch_stage("training")

    def _update(self, batch):
        """Function to iterate, optimizing parameters and saving
        training and validation loss values.
        """
        params, opt_state, train_loss, curr_grad, per_target_losses = self._update_fn(
            self.state.params, self.state.opt_state, batch, per_target=True)

        # Save the statistics
        for key, val in per_target_losses.items():
            if key not in self.train_target_losses.keys():
                self.train_target_losses[key] = []

            self.train_target_losses[key].append(onp.asarray(val))

        self.state = self.state.replace(params=params, opt_state=opt_state)
        self.train_batch_losses.append(onp.asarray(train_loss))

        self.gradient_norm_history.append(util.tree_norm(curr_grad))

    def predict(self, dataset, params=None, batch_size=10):
        """Computes predictions for a dataset.

        Args:
            dataset: Dictionary containing input data as numpy arrays. Can be,
                e.g., the whole testing split.
            params: Parameters for the model. If None, uses the current
                parameters.
            batch_size: Batch size for predictions.

        Returns:
            Returns all predictions of the model for the provided inputs.

        """

        # Set random to False to prevent shuffling of results by shuffling
        # inputs
        self.set_dataset(dataset, "predict", include_all=True, random=False)

        if params is None:
            params = self.params

        if self._disable_shmap:
            raise NotImplementedError("Pmapped predictions not implemented.")
        else:
            shmapped_model = max_likelihood.shmap_model(self.batched_model)

        all_predictions = None
        for batch_with_info in self._get_batch_stage("predict", information=True):
            # Compute the total loss and the individual contributions per
            # target
            batch, batch_info = batch_with_info

            # Only get valid samples by masking with numpy
            predictions = shmapped_model(params, batch)
            mask = jnp.asarray(batch_info.mask, dtype=jnp.bool_)
            if util.use_mpi():
                # shmap_model gathers predictions across ranks under MPI.
                # Gather mask as well to keep shapes consistent.
                mask = util.mpi_tree_gather(mask)

            mask_np = onp.asarray(mask)
            predictions = tree_util.tree_map(
                lambda x: x[mask_np, ...],
                device_get(predictions)
            )

            if all_predictions is None:
                all_predictions = predictions
            else:
                all_predictions = util.tree_map(
                    lambda *leaves: onp.concatenate(leaves, axis=0),
                    all_predictions, predictions
                )

        return all_predictions

    def evaluate(self, stage = "validation", loss_fn = None, params=None):
        """Computes the loss on the whole dataset.

        Args:
            stage: Stage for which to evaluate the loss. Can be ``"testing"``,
                ``"validation"``, or ``"training"``.
            loss_fn: Loss function to evaluate. If None, evaluates the loss
                function used for training.
            params: Parameters for the model. If None, uses the current
                parameters.

        Returns:
            Returns the total loss and the loss for each individual target.

        """

        assert stage in self._batch_states, (
            f"A dataloader (dataset) is required to evaluate the loss on "
            f"stage {stage}."
        )

        if params is None:
            params = self.params

        # Option to define a new loss function, e.g., for MAE error
        if loss_fn is None:
            loss_fn = self._evaluate_fn
        elif not self._disable_shmap:
            loss_fn = max_likelihood.shmap_loss_fn(self.batched_model, loss_fn)
        else:
            raise NotImplementedError

        total_loss, per_target_losses = 0.0, {}
        total_samples, valid_samples = 0, 0
        for batch_with_info in self._get_batch_stage(stage, information=True):
            # Compute the total loss and the individual contributions per
            # target
            batch, batch_info = batch_with_info

            # Compute a correction factor
            valid_samples += onp.sum(batch_info.mask)
            total_samples += batch_info.batch_size

            val_loss, per_target_loss = loss_fn(
                params, batch, mask=batch_info.mask, per_target=True)

            total_loss += onp.asarray(val_loss)
            for key, val in per_target_loss.items():
                if key not in per_target_losses:
                    per_target_losses[key] = []

                per_target_losses[key].append(onp.asarray(val))

        # The correction factor accounts for including invalid (masked) samples
        # in the mean of the split
        if util.use_mpi():
            comm = util.get_communicator()
            MPI = util.get_mpi()
            if comm is not None and MPI is not None:
                valid_samples = comm.allreduce(valid_samples, op=MPI.SUM)
                total_samples = comm.allreduce(total_samples, op=MPI.SUM)
        scale_factor =  total_samples / valid_samples

        total_loss /= self._batches_per_epoch[stage]
        total_loss *= scale_factor

        per_target_losses = {
            key: sum(val) / len(val) * scale_factor
            for key, val in per_target_losses.items()
        }

        return total_loss, per_target_losses

    def _evaluate_convergence(self, *args, thresh=None, **kwargs):
        """Prints progress, saves best obtained params and signals converged if
        validation loss improvement over the last epoch is less than the thesh.
        """
        batches_per_epoch = self._batches_per_epoch["training"]
        mean_train_loss = sum(
            self.train_batch_losses[-batches_per_epoch:]
        ) / batches_per_epoch
        self.train_losses.append(mean_train_loss)
        duration = self.update_times[self._epoch]

        start = time.time()
        if "validation" in self._batch_states:
            val_loss, val_target_losses = self.evaluate("validation")

            self.val_losses.append(onp.asarray(val_loss))

            for key, val in val_target_losses.items():
                if key not in self.val_target_losses:
                    self.val_target_losses[key] = []

                self.val_target_losses[key].append(val)

            self._converged = self._early_stop.early_stopping(
                val_loss, thresh, self.params)
        else:
            val_loss = None

        # Add the time for the validation
        total_duration = (time.time() - start) / 60. + duration

        log_str = (
            f"[Epoch {self._epoch}]:\n"
            f"\tAverage train loss: {mean_train_loss:.5f}\n"
            f"\tAverage val loss: {val_loss}\n"
            f"\tGradient norm: {self.gradient_norm_history[-1]}\n"
            f"\tElapsed time = {total_duration:.3f} min (total), {duration:.3f} min (training)\n"
            f"\tPer-target losses:\n"
        )

        for key in self.train_target_losses:
            train_batches_per_epoch = self._batches_per_epoch["training"]

            mean_train_loss = sum(
                self.train_target_losses[key][-train_batches_per_epoch:]
            ) / train_batches_per_epoch

            try:
                target_val_loss = self.val_target_losses[key][-1]
            except IndexError or KeyError:
                target_val_loss = "N.A."

            log_str += (
                f"\t\t{key} | train loss: {mean_train_loss} | "
                f"val loss: {target_val_loss}\n"
            )

        print(log_str)

    def update_with_samples(self, **sample):
        """A single params update step, where a batch is taken from the training
        set and samples of the batch are substituted by the provided samples.

        This function is useful in an active learning context to retrain
        specifically on newly labeled datapoints. The number of provided samples
        must not exceed the trainer batch size.

        Args:
            sample: Kwargs storing data samples to supply to
                ``self._build_dataset`` to build samples in the correct
                pytree. Analogous usage as update_dataset, but the dataset
                only consists of a few observations.
        """
        n_samples = util.tree_multiplicity(sample)
        assert n_samples <= self.local_batch_size, (
            "Number of provided samples must not exceed per-rank batch size."
        )
        batch = next(self._get_batch())
        updated_batch = util.tree_set(batch, sample, n_samples)
        self._update_with_dropout(updated_batch)


    @property
    def params(self):
        """Current energy parameters."""
        single_params = self.state.params
        return single_params

    @params.setter
    def params(self, loaded_params):
        self.state = self.state.replace(params=loaded_params)

    @property
    def best_params(self):
        """Returns the best parameters based on the validation loss.

        If training was performed with early stopping, return the best
        parameters to this criterion instead.
        """
        #  if no validation data given, _early_stop.best_params are simply
        #  init_params
        if "validation" in self._batch_states.keys() is None:
            return self.params
        else:
            return self._early_stop.best_params

    @property
    def best_inference_params(self):
        """Returns best model params irrespective whether dropout is used."""
        if dropout.dropout_is_used(self.best_params):
            # all nodes present during inference
            params, _ = dropout.split_dropout_params(self.best_params)
        else:
            params = self.best_params
        return params

    @property
    def best_inference_params_replicated(self):
        """Returns the best inference params replicated on every device."""
        inference_params = self.best_inference_params
        return util.tree_replicate(inference_params)

    def move_to_device(self):
        """Transforms all arrays of the trainer state to JAX arrays."""
        super().move_to_device()
        self._early_stop.move_to_device()


class ProbabilisticFMTrainerTemplate(TrainerInterface):
    """Trainer template for methods that result in multiple parameter sets for
    Monte-Carlo-style uncertainty quantification, based on a force-matching
    formulation.
    """
    def __init__(self, checkpoint_path, energy_fn_template,
                 val_dataloader=None):
        super().__init__(checkpoint_path, energy_fn_template)
        self.results = []

        # TODO use val_loader for some metrics that are interesting for MCMC
        #  and SG-MCMC

    def move_to_device(self):
        params = []
        for param_set in self.params:
            params.append(tree_map(jnp.array, param_set))  # move on device
        self.params = params

    @property
    @abc.abstractmethod
    def list_of_params(self):
        """ Returns a list containing n single model parameter sets, where n
        is the number of samples. This provides a more intuitive parameter
        interface that self.params, which returns a large set of parameters,
        where n is the leading axis of each leaf. Self.params is most useful,
        if parameter sets are mapped via map or vmap in a postprocessing step.
        """


class MCMCForceMatchingTemplate(ProbabilisticFMTrainerTemplate):
    """Initializes log_posterior function to be used for MCMC with blackjax,
    including batch-wise evaluation of the likelihood and re-materialization.
    """
    def __init__(self, init_state, kernel, checkpoint_path, val_loader=None,
                 ref_energy_fn_template=None):
        super().__init__(checkpoint_path, ref_energy_fn_template, val_loader)
        self.kernel = jit(kernel)
        self.state = init_state

    def train(self, num_samples, checkpoint_freq=None, init_samples=None,
              rng_key=random.PRNGKey(0)):
        if init_samples is not None:
            # TODO implement multiple chains
            raise NotImplementedError

        for i in range(num_samples):
            start_time = time.time()
            rng_key, consumed_key = random.split(rng_key)
            self.state, info = self.kernel(consumed_key, self.state)
            self.results.append(self.state)
            print(f"Time for sample {i}: {(time.time() - start_time) / 60.}"
                  f" min.", info)
            self._epoch += 1
            self._dump_checkpoint_occasionally(frequency=checkpoint_freq)

    @property
    def list_of_params(self):
        """Returns a list of sampled parameters."""
        return [state.position["params"] for state in self.results]

    @property
    def params(self):
        """Concatenates the sampled parameters along the first dimension."""
        return util.tree_stack(self.list_of_params)

    @params.setter
    def params(self, loaded_params):
        raise NotImplementedError("Setting params seems not meaningful for MCMC"
                                  " samplers.")


class EarlyStopping:
    """A class that saves the best parameter obtained so far based on the
    validation loss and determines whether the optimization can be stopped based
    on some stopping criterion.

    The following criteria are implemented:

    - ``"window_median"``: 2 windows are placed at the end of the loss
      history. Stops when the median of the latter window of size "thresh"
      exceeds the median of the prior window of the same size.

    - ``"PQ"``: Stops when the PQ criterion exceeds thresh

    - ``"max_loss"``: Stops when the loss decreased below the maximum allowed
      loss specified via thresh.

    Args:
        criterion: Convergence criterion to employ
        pq_window_size: Window size for PQ method

    Attributes:
        best_loss: Loss of the best performing parameters
        best_params: Parameters with best performance

    """
    def __init__(self, params, criterion, pq_window_size=5):
        self.criterion = criterion

        # own loss history that can be reset on the fly if needed.
        self._epoch_losses = []
        self.best_loss = 1.e16
        self.best_params = copy.copy(params)  # move on device, if loaded

        self.pq_window_size = pq_window_size

    def _is_converged(self, thresh):
        converged = False
        if thresh is not None:  # otherwise no early stopping used
            if self.criterion == "window_median":
                window_size = thresh
                if len(self._epoch_losses) >= 2 * window_size:
                    prior_window = onp.array(
                        self._epoch_losses[-2 * window_size:-window_size])
                    latter_window = onp.array(self._epoch_losses[-window_size:])
                    converged = (onp.median(latter_window)
                                 > onp.median(prior_window))

            elif self.criterion == "PQ":
                if len(self._epoch_losses) >= self.pq_window_size:
                    best_loss = min(self._epoch_losses)
                    loss_window = self._epoch_losses[-self.pq_window_size:]
                    gen_loss = 100. * (loss_window[-1] / best_loss - 1.)
                    window_average = sum(loss_window) / self.pq_window_size
                    window_min = min(loss_window)
                    progress = 1000. * (window_average / window_min - 1.)
                    pq = gen_loss / progress
                    converged = pq > thresh

            elif self.criterion == "max_loss":
                converged = self._epoch_losses[-1] < thresh
            else:
                raise ValueError(f"Convergence criterion {self.criterion} "
                                 f"unknown. Select 'max_loss', 'ave_loss' or "
                                 f"'std'.")
        return converged

    def early_stopping(self, curr_epoch_loss, thresh, params=None,
                       save_best_params=True):
        """Estimates whether the convergence criterion was met and keeps track
        of the best parameters obtained so far.

        Args:
            curr_epoch_loss: Validation loss of the most recent epoch
            thresh: Convergence threshold. Specific definition depends on the
                selected convergence criterion.
            params: Optimization parameters to save in case of being best. Make
                sure to supply non-device-replicated params, i.e.
                ``self.params.``
            save_best_params: If best params are supposed to be tracked

        Returns:
            True if the convergence criterion was met, else False.
        """
        self._epoch_losses.append(curr_epoch_loss)

        if save_best_params:
            assert params is not None, ("If best params are saved, they need to"
                                        " be provided in early_stopping.")
            improvement = self.best_loss - curr_epoch_loss
            if improvement > 0.:
                self.best_loss = curr_epoch_loss
                self.best_params = copy.copy(params)

        return self._is_converged(thresh)

    def reset_convergence_losses(self):
        """Resets loss history used for convergence estimation, e.g., to avoid
        early stopping when loss increases due to on-the-fly changes in the
        dataset or the loss function.
        """
        self._epoch_losses = []
        self.best_loss = 1.e16
        self.best_params = None

    def move_to_device(self):
        """Moves best_params to device to use them after loading trainer."""
        self.best_params = tree_map(jnp.array, self.best_params)
