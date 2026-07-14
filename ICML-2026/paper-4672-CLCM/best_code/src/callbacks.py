import numpy as np
import logging

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def _is_class_il_setting(setting: str) -> bool:
    """Check if the setting is a Class-IL (class-incremental learning) setting."""
    class_il_settings = [
        "classIL5task", "classIL2task",  # Legacy names
        "ClassILMNIST5Task", "ClassILCIFAR5Task", "ClassILTinyImageNet10Task",  # New names
    ]
    return setting in class_il_settings


class TrainingCallback:
    """
    Base class for creating training callbacks
    from: clementchadebec github
    """

    def on_init_end(self, training_config, **kwargs):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """

    def on_train_begin(self, training_config, **kwargs):
        """
        Event called at the beginning of training.
        """

    def on_train_end(self, training_config, **kwargs):
        """
        Event called at the end of training.
        """

    def on_task_begin(self, training_config, **kwargs):
        """
        Event called at the beginning of a task.
        """

    def on_task_end(self, training_config, **kwargs):
        """
        Event called at the end of a task.
        """

    def on_epoch_begin(self, training_config, **kwargs):
        """
        Event called at the beginning of an epoch.
        """

    def on_epoch_end(self, training_config, **kwargs):
        """
        Event called at the end of an epoch.
        """

    def on_train_step_begin(self, training_config, **kwargs):
        """
        Event called at the beginning of a training step.
        """

    def on_train_step_end(self, training_config, **kwargs):
        """
        Event called at the end of a training step.
        """

    def on_test_step_begin(self, training_config, **kwargs):
        """
        Event called at the beginning of a test step.
        """

    def on_test_step_end(self, training_config, **kwargs):
        """
        Event called at the end of a test step.
        """

    def on_test(self, training_config, **kwargs):
        """
        Event called after an test phase.
        """

    def on_prediction_step(self, training_config, **kwargs):
        """
        Event called after a prediction phase.
        """

    def on_save(self, training_config, **kwargs):
        """
        Event called after a checkpoint save.
        """

    def on_log(self, training_config, logs, **kwargs):
        """
        Event called after logging the last logs.
        """

    def __repr__(self) -> str:
        return self.__class__.__name__


class CallbackHandler:
    """
    Class to handle list of Callback.
    """

    def __init__(self, callbacks, model):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warning(
                f"You are adding a {cb_class} to the callbacks but there one is already used."
                f" The current list of callbacks is\n: {self.callback_list}"
            )
        self.callbacks.append(cb)

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def on_init_end(self, training_config, **kwargs):
        self.call_event("on_init_end", training_config, **kwargs)

    def on_train_step_begin(self, training_config, **kwargs):
        self.call_event("on_train_step_begin", training_config, **kwargs)

    def on_train_step_end(self, training_config, **kwargs):
        self.call_event("on_train_step_end", training_config, **kwargs)

    def on_test_step_begin(self, training_config, **kwargs):
        self.call_event("on_test_step_begin", training_config, **kwargs)

    def on_test_step_end(self, training_config, **kwargs):
        self.call_event("on_test_step_end", training_config, **kwargs)

    def on_task_begin(self, training_config, **kwargs):
        self.call_event("on_task_begin", training_config, **kwargs)

    def on_task_end(self, training_config, **kwargs):
        self.call_event("on_task_end", training_config, **kwargs)

    def on_train_begin(self, training_config, **kwargs):
        self.call_event("on_train_begin", training_config, **kwargs)

    def on_train_end(self, training_config, **kwargs):
        self.call_event("on_train_end", training_config, **kwargs)

    def on_epoch_begin(self, training_config, **kwargs):
        self.call_event("on_epoch_begin", training_config, **kwargs)

    def on_epoch_end(self, training_config, **kwargs):
        self.call_event("on_epoch_end", training_config, **kwargs)

    def on_test(self, training_config, **kwargs):
        self.call_event("on_test", training_config, **kwargs)

    def on_save(self, training_config, **kwargs):
        self.call_event("on_save", training_config, **kwargs)

    def on_log(self, training_config, logs, **kwargs):
        self.call_event("on_log", training_config, logs=logs, **kwargs)

    def on_prediction_step(self, training_config, **kwargs):
        self.call_event("on_prediction_step", training_config, **kwargs)

    def call_event(self, event, training_config, **kwargs):
        for callback in self.callbacks:
            getattr(callback, event)(
                training_config,
                model=self.model,
                **kwargs,
            )


class MetricConsolePrinterCallback(TrainingCallback):
    """
    A :class:`TrainingCallback` printing the training logs in the console.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def on_log(self, training_config, logs, **kwargs):
        epoch = kwargs.pop("epoch", None)
        if logger is not None:
            epoch_train_loss = logs.epoch_train_loss
        
            if _is_class_il_setting(training_config.setting):
                task_losses = logs.get('task_losses', [])
                task_accuracies = logs.get('task_accuracies', [])
                cumulative_accuracy = logs.get('cumulative_accuracy', 0.0)
                
                loss_str = ", ".join([f"T{i}: {loss:.4f}" for i, loss in enumerate(task_losses)])
                acc_str = ", ".join([f"T{i}: {acc:.2f}" for i, acc in enumerate(task_accuracies)])
                
                self.logger.info(
                    f"\t\t\t\tEpoch: {epoch:03}, Train loss: {epoch_train_loss:.4f} & test losses: {loss_str} & accuracy: {acc_str}, Full: {cumulative_accuracy:.2f}"
                )
            else:
                # Standard logging for other settings
                epoch_test_loss = logs.epoch_test_loss
                accuracy = logs.accuracy
                if epoch_test_loss is not None:
                    self.logger.info(
                        f"\t\t\t\tEpoch: {epoch:03}, Train loss: {epoch_train_loss:.4f} & test loss: {epoch_test_loss:.4f} & accuracy: {accuracy:.2f}"
                    )


class ProgressBarCallback(TrainingCallback):
    """
    A :class:`TrainingCallback` printing the training progress bar.
    """

    def __init__(self):
        self.train_progress_bar = None
        self.test_progress_bar = None

    def on_train_step_begin(self, training_config, **kwargs):
        epoch = kwargs.pop("epoch", None)
        train_loader = kwargs.pop("train_loader", None)
        if train_loader is not None:
            self.train_progress_bar = tqdm(
                total=len(train_loader),
                unit="batch",
                desc=f"Training epoch {epoch:03}/{training_config.epochs}",
            )

    def on_test_step_begin(self, training_config, **kwargs):
        epoch = kwargs.pop("epoch", None)
        test_loader = kwargs.pop("test_loader", None)
        if test_loader is not None:
            self.test_progress_bar = tqdm(
                total=len(test_loader),
                unit="batch",
                desc=f"Testing  epoch {epoch:03}/{training_config.epochs}",
            )

    def on_train_step_end(self, training_config, **kwargs):
        if self.train_progress_bar is not None:
            self.train_progress_bar.update(1)

    def on_test_step_end(self, training_config, **kwargs):
        if self.test_progress_bar is not None:
            self.test_progress_bar.update(1)

    def on_epoch_end(self, training_config, **kwags):
        if self.train_progress_bar is not None:
            self.train_progress_bar.close()

        if self.test_progress_bar is not None:
            self.test_progress_bar.close()
