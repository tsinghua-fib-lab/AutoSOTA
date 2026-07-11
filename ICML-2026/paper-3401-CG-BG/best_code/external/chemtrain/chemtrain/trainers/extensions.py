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

"""Simple extensions, e.g., to log trainer statistics to MLops frameworks."""
import importlib

from chemtrain.trainers import trainers, base

def wandb_log_difftre(run, trainer: trainers.Difftre, plot_fns=None):
    """Logs DiffTRe training statistics to Weights & Biases.

    Args:
        run: Active W&B run
        trainer: Trainer to log to W&B
        plot_fns: Dictionary with functions to plot the selected predictions
            for selected statepoints

    Example usage:

        After initiating the trainer and the run, add a W&B tracking task
        to the trainer via:

        .. code::

           def plot_fn(some_prediction):
               fig = plt.figure()
               ...
               return fig

           plot_fns = {
               0: {'some_prediction': plot_fn}
           }

           wandb_log_difftre(run, difftre_trainer, plot_fn)

    """
    wandb = importlib.import_module("wandb")

    if plot_fns is None:
        plot_fns = {}

    def log_fn(trainer: trainers.Difftre, *args, **kwargs):
        plots = {}

        assert issubclass(type(trainer), trainers.Difftre), (
            f"Supports only DiffTRe trainer."
        )

        for statepoint_key, statepoint_fns in plot_fns.items():
            recent_predictions = trainer.predictions[statepoint_key][trainer._epoch]

            plots[statepoint_key] = {
                pred_key: wandb.Image(
                    plot_fn(recent_predictions[pred_key])
                ) for pred_key, plot_fn in statepoint_fns.items()
            }

        run.log(
            data={
                "Epoch loss": trainer.epoch_losses[-1],
                "Gradient norm": trainer.gradient_norm_history[-1],
                "Elapsed time": trainer.update_times[-1],
                "Predictions": plots
            },
            commit=True
        )

    trainer.add_task("post_epoch", log_fn)


def wandb_log_data_parallel(run, trainer: base.DataParallelTrainer):
    """Logs DataParallel training statistics to Weights & Biases.

    Args:
        run: Active W&B run
        trainer: Trainer to log to W&B

    """
    wandb = importlib.import_module("wandb")

    def get_validation_loss(key):
        try:
            return trainer.val_target_losses[key][-1]
        except IndexError:
            return "N.A."

    def log_fn(trainer: base.DataParallelTrainer, *args, **kwargs):
        assert issubclass(type(trainer), base.DataParallelTrainer), (
            f"Supports only DataParallalTrainer trainers."
        )

        duration = trainer.update_times[trainer._epoch]

        statistics = {
            "training": trainer.train_batch_losses[-1],
            "validation": trainer.val_losses[-1],
            "gradient_norm": trainer.gradient_norm_history[-1],
            "duration": duration,
            "targets": {
                key: {
                    "training": trainer.train_target_losses[key][-1],
                    "validation": get_validation_loss(key),
                } for key in trainer.train_target_losses.keys()
            }
        }

        run.log(data=statistics, commit=True)

    trainer.add_task("post_epoch", log_fn)
