import copy
import datetime
import json
import torch
import os
import sys
import logging

import wandb
from omegaconf import OmegaConf
from typing import Dict, Optional

from networks.network_interface import FisherInterface
from src.callbacks import (
    CallbackHandler,
    MetricConsolePrinterCallback,
    ProgressBarCallback,
    TrainingCallback,
)
from src.utils import dotdict
from torch.utils.data import Subset, TensorDataset

logger = logging.getLogger(__name__)


class TrainerInterface:
    def __init__(self, model, config, callbacks=None):
        self.model = model
        self.config = config
        self.device = config.device
        self.save = config.save
        self.callbacks = callbacks
        self.setting = config.setting

        self._set_device(self.device)
        self.model.to(self.device)

        self._prepare_training()

        self.callback_handler.on_train_begin(training_config=self.config)

        config_details = "\n".join(
            [f" - {key}: {value}" for key, value in config.items()]
        )
        logger.info(msg=f"Training:\n{config_details}\n - model: {self.model.name}\n")

    def _set_device(self, device: str):
        self.device = torch.device(device)
        # torch.set_default_device(self.device)  # disabled for PyTorch 2.1 compat

    def _save_model(self):
        if not os.path.exists(self.training_dir):
            os.makedirs(self.training_dir)

        torch.save(self.model.state_dict(), os.path.join(self.training_dir, "model.pt"))

        with open(os.path.join(self.training_dir, "config.json"), "w") as fp:
            json.dump(self.config, fp)

        self.callback_handler.on_save(self.config)

    def _setup_logger(self):
        # Create a logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Create file handler which logs even debug messages
        fh = logging.FileHandler(os.path.join(self.training_dir, "training.log"))
        fh.setLevel(logging.INFO)

        # Create console handler with a higher log level
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter("%(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

    def _prepare_training(self):
        self._set_seed(self.config.seed)
        self._set_optimizer()
        self._set_scheduler()
        self._set_output_dir()
        self._setup_logger()
        self._setup_callbacks()

    def _set_seed(self, seed: int):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _set_optimizer(self):
        if self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config["lr"],
            )
        elif self.config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.config["lr"]
            )
        else:
            raise NotImplementedError

    def _set_scheduler(self):
        if self.config.scheduler == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config.lr, gamma=self.config.gamma
            )
        elif self.config.scheduler == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.gamma,
                patience=self.config.patience,
                verbose=True,
            )
        elif self.config.scheduler == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs
            )
        elif self.config.scheduler is None:
            pass
        else:
            raise NotImplementedError

    def _set_output_dir(self):
        self.output_dir = self.config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        self._training_signature = (
            str(datetime.datetime.now())[5:19].replace(" ", "_").replace(":", "-")
        )

        training_dir = os.path.join(
            self.config.output_dir,
            f"{self.model.name}_lr{self.config.lr}_{self._training_signature}",
        )

        self.training_dir = training_dir

        if not os.path.exists(training_dir):
            os.makedirs(training_dir, exist_ok=True)

    def _setup_callbacks(self):
        if self.callbacks is None:
            self.callbacks = [TrainingCallback()]

        self.callback_handler = CallbackHandler(
            callbacks=self.callbacks, model=self.model
        )

        self.callback_handler.add_callback(ProgressBarCallback())
        self.callback_handler.add_callback(MetricConsolePrinterCallback())

    def test_step(self, epoch):
        raise NotImplementedError("test_step must be implemented in a subclass.")

    def train(self):
        raise NotImplementedError("train must be implemented in a subclass.")

    def _train_step(self, epoch: int):
        self.callback_handler.on_train_step_begin(
            training_config=self.config,
            train_loader=self.train_loader,
            epoch=epoch,
        )

        self.model.train()

        epoch_loss = 0

        for X, y in self.train_loader:
            X = X.to(self.device)
            y = y.to(self.device)

            y_hat = self.model(X)
            loss = self.model.calculate_loss(y_hat, y)

            self.optimizer.zero_grad()
            self.model.backward(y)
            self.optimizer.step()

            epoch_loss += loss.item()

            if epoch_loss != epoch_loss:
                raise ArithmeticError("NaN detected in train loss")

            self.callback_handler.on_train_step_end(training_config=self.config)

        epoch_loss /= len(self.train_loader)

        return epoch_loss

    @torch.no_grad()
    def _test_step(self, epoch, task_id):
        self.callback_handler.on_test_step_begin(
            training_config=self.config,
            test_loader=self.test_loader,
            epoch=epoch,
        )

        self.model.eval()

        epoch_loss = 0
        total = 0
        correct = 0

        if _is_task_il_setting(self.setting):
            self.model.task_id = task_id

        for X, y in self.test_loader:
            X = X.to(self.device)
            y = y.to(self.device)

            y_hat = self.model(X)

            loss = self.model.calculate_loss(y_hat, y)

            epoch_loss += loss.item()
            total += y.size(0)
            correct += (y_hat.argmax(dim=1) == y.argmax(dim=1)).sum().item()

            if epoch_loss != epoch_loss:
                raise ArithmeticError("NaN detected in test loss")

            self.callback_handler.on_test_step_end(training_config=self.config)

        epoch_loss /= len(self.test_loader)
        accuracy = 100 * correct / total

        return epoch_loss, accuracy


class Trainer(TrainerInterface):
    def __init__(self, model, train_loader, test_loader, config, callbacks=None):
        super().__init__(model, config, callbacks)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self):
        self.callback_handler.on_train_begin(training_config=self.config)
        metrics = dotdict()

        for epoch in range(1, self.config.epochs + 1):
            self.callback_handler.on_epoch_begin(
                training_config=self.config,
                epoch=epoch,
                train_loader=self.train_loader,
                test_loader=self.test_loader,
            )

            epoch_train_loss = self._train_step(epoch)
            metrics.epoch_train_loss = epoch_train_loss

            if self.test_loader is not None:
                epoch_test_loss, accuracy = self._test_step(epoch)
                metrics.epoch_test_loss = epoch_test_loss
                metrics.accuracy = accuracy

            self.callback_handler.on_epoch_end(training_config=self.config)
            self.callback_handler.on_log(
                self.config,
                metrics,
                logger=logger,
                epoch=epoch,
            )

        if self.save:
            self._save_model()


def _is_class_il_setting(setting: str) -> bool:
    """Check if the setting is a Class-IL (class-incremental learning) setting."""
    return "classil" in setting.lower()

def _is_task_il_setting(setting: str) -> bool:
    """Check if the setting is a Task-IL (task-incremental learning) setting.""" 
    return "taskil" in setting.lower()




class TrainerCL(TrainerInterface):
    def __init__(self, model, tasks_dataloaders, config, callbacks=None):
        super().__init__(model, config, callbacks)
        self.tasks_dataloaders = tasks_dataloaders

        # Peak model tracking
        self.use_peak = config.get("peak", False)
        self.best_cumulative_accuracy = -float("inf")
        self.best_model_state = None
        self.peak_epoch = 0

    def _least_square_initialization(self, dataloader, task_id, weight_decay=1e-4):
        """Least-square optimal initialization for new classifier weights."""
        self.model.eval()
        classes_per_task = self.config.classes_per_task
        new_start = task_id * classes_per_task
        new_end = (task_id + 1) * classes_per_task

        features_list = []
        labels_list = []

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                features = x
                for layer in self.model.layers[:-1]:
                    features = layer(features)
                features_list.append(features)
                labels_list.append(y.argmax(dim=1))

        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        N, d = features.shape
        features_ext = torch.cat([features, torch.ones(N, 1, device=features.device)], dim=1)

        num_new_classes = new_end - new_start
        targets = torch.zeros(N, num_new_classes, device=features.device)
        for i, label in enumerate(labels):
            if new_start <= label < new_end:
                targets[i, label - new_start] = 1.0

        mask = (labels >= new_start) & (labels < new_end)
        features_new = features_ext[mask]
        targets_new = targets[mask]

        ZtZ = features_new.T @ features_new
        ZtY = features_new.T @ targets_new

        reg = weight_decay * features_new.shape[0] * torch.eye(d + 1, device=features.device)
        W_ls = torch.linalg.solve(ZtZ + reg, ZtY)

        with torch.no_grad():
            for c_idx, c in enumerate(range(new_start, new_end)):
                self.model.layers[-1]._weights[c] = W_ls[:d, c_idx]
                self.model.layers[-1]._bias[c] = W_ls[d, c_idx]

        logger.info(f"Applied least-square initialization for task {task_id} classes [{new_start}, {new_end})")


    def _least_square_initialization_taskil(self, dataloader, task_id, classes_per_task=None, weight_decay=1e-4):
        """
        Least-square optimal initialization for new classifier weights in Task-IL.

        For Task-IL, we initialize the specific output head for the new task
        (classes_per_task neurons per task).
        """
        self.model.eval()
        if classes_per_task is None:
            classes_per_task = self.config.classes_per_task
        new_start = task_id * classes_per_task
        new_end = (task_id + 1) * classes_per_task

        features_list = []
        labels_list = []

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                # Get features from penultimate layer
                features = x
                for layer in self.model.layers[:-1]:
                    features = layer(features)
                features_list.append(features)
                labels_list.append(y.argmax(dim=1))  # Binary: 0 or 1

        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        N, d = features.shape
        features_ext = torch.cat([features, torch.ones(N, 1, device=features.device)], dim=1)

        # Create binary targets (0 or 1 for this task)
        num_new_classes = classes_per_task
        targets = torch.zeros(N, num_new_classes, device=features.device)
        for i, label in enumerate(labels):
            targets[i, label] = 1.0

        # Solve least squares
        ZtZ = features_ext.T @ features_ext
        ZtY = features_ext.T @ targets

        reg = weight_decay * N * torch.eye(d + 1, device=features.device)
        W_ls = torch.linalg.solve(ZtZ + reg, ZtY) / 2

        # Initialize the task-specific output neurons
        with torch.no_grad():
            for c_idx in range(num_new_classes):
                global_idx = new_start + c_idx
                self.model.layers[-1]._weights[global_idx] = W_ls[:d, c_idx]
                self.model.layers[-1]._bias[global_idx] = W_ls[d, c_idx]

        logger.info(f"Applied Task-IL least-square initialization for task {task_id} heads [{new_start}, {new_end})")

    @torch.no_grad()
    def _evaluate_taskil(self, current_task_id):
        """
        Evaluate network on Task-IL setting.

        In Task-IL, we evaluate each task separately using only that task's
        output head (classes_per_task classes). The task identity is known at test time.

        Args:
            current_task_id: Number of tasks seen so far (0-indexed, evaluate tasks 0 to current_task_id)

        Returns:
            dict with per-task accuracies and average accuracy
        """
        self.model.eval()
        results = {}
        total_correct = 0
        total_samples = 0
        classes_per_task = self.config.classes_per_task

        for task_id in range(current_task_id + 1):
            # Set network to evaluate on this task
            self.model.task_id = task_id

            correct = 0
            task_total = 0

            # Get the test loader for this specific task
            _, test_loader = self.tasks_dataloaders[task_id]

            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)

                # Forward pass - network masks to current task's outputs
                y_hat = self.model(x)  # Shape: (batch, classes_per_task) due to task mask

                # y is one-hot (batch, classes_per_task), get class index
                labels = y.argmax(dim=1)  # 0 to classes_per_task-1
                preds = y_hat.argmax(dim=1)  # 0 to classes_per_task-1

                correct += (preds == labels).sum().item()
                task_total += x.size(0)

            accuracy = correct / task_total if task_total > 0 else 0.0
            results[f'task_{task_id}'] = accuracy * 100  # Convert to percentage
            total_correct += correct
            total_samples += task_total

        # Average accuracy across tasks (standard Task-IL metric)
        results['average'] = sum(results[f'task_{t}'] for t in range(current_task_id + 1)) / (current_task_id + 1)
        results['overall'] = (total_correct / total_samples * 100) if total_samples > 0 else 0.0

        return results

    def _test_step_taskil(self, epoch, current_task_id):
        """
        Test step for Task-IL setting.

        Evaluates all seen tasks and returns average accuracy.
        Also stores per-task accuracies in self.current_task_accuracies.

        Returns:
            tuple: (loss, average_accuracy)
        """
        eval_results = self._evaluate_taskil(current_task_id)

        # Store per-task accuracies for logging
        self.current_task_accuracies = [eval_results[f'task_{t}'] for t in range(current_task_id + 1)]
        self.current_task_losses = [0.0] * (current_task_id + 1)  # Loss not computed in Task-IL eval

        # Return average accuracy as the main metric
        return 0.0, eval_results['average']

    def train(self):
        self.callback_handler.on_train_begin(training_config=self.config)
        metrics = dotdict()

        for task_id, (train_loader, test_loader) in enumerate(self.tasks_dataloaders):
            logger.info(f"Starting Task {task_id + 1}/{len(self.tasks_dataloaders)}")

            self.train_loader = train_loader
            self.test_loader = test_loader

            self.callback_handler.on_task_begin(
                training_config=self.config, task_id=task_id + 1
            )

            self.model.task_id = task_id

            if task_id == 0:
                self.test_loader_first_task = test_loader
            else:
                # Apply least-square initialization for new task classifier weights 
                # Only apply if Class IL
                if _is_class_il_setting(self.config.setting):
                    self._least_square_initialization(train_loader, task_id)
                else:
                    self._least_square_initialization_taskil(train_loader, task_id, self.config.classes_per_task)
                if hasattr(self.model, '_first_task'):
                    self.model._first_task = False

            # Reset peak tracking for current task
            if self.use_peak and task_id > 0:
                self.best_cumulative_accuracy = -float("inf")
                self.best_model_state = None
                self.peak_epoch = 0

            for epoch in range(1, self.config.epochs + 1):
                self.callback_handler.on_epoch_begin(
                    training_config=self.config,
                    epoch=epoch,
                    train_loader=self.train_loader,
                    test_loader=self.test_loader,
                )

                epoch_train_loss = self._train_step(epoch)
                metrics.epoch_train_loss = epoch_train_loss

                if self.test_loader is not None:
                    if _is_class_il_setting(self.config.setting):
                        epoch_test_loss, accuracy = self._test_step(epoch, task_id)
                        # Get per-task metrics from instance variables
                        task_losses = getattr(self, "current_task_losses", [])
                        task_accuracies = getattr(self, "current_task_accuracies", [])
                        metrics.task_losses = task_losses
                        metrics.task_accuracies = task_accuracies
                        metrics.cumulative_accuracy = accuracy

                        # Track peak model for class-IL settings (after first task)
                        if self.use_peak and task_id > 0:
                            if accuracy > self.best_cumulative_accuracy:
                                self.best_cumulative_accuracy = accuracy
                                self.best_model_state = copy.deepcopy(
                                    self.model.state_dict()
                                )
                                self.peak_epoch = epoch
                                logger.info(f"New peak model saved at epoch {epoch} with cumulative accuracy: {accuracy:.2f}%")
                    elif _is_task_il_setting(self.config.setting):
                        epoch_test_loss, accuracy = self._test_step(epoch, task_id)
                        # Get per-task metrics from instance variables (set by _test_step_taskil)
                        task_losses = getattr(self, 'current_task_losses', [])
                        task_accuracies = getattr(self, 'current_task_accuracies', [])
                        metrics.task_losses = task_losses
                        metrics.task_accuracies = task_accuracies
                        metrics.avg_accuracy = accuracy  # Average accuracy across seen tasks
                    else:
                        epoch_test_loss, accuracy = self._test_step(epoch, task_id)

                    metrics.epoch_test_loss = epoch_test_loss
                    metrics.accuracy = accuracy

                self.callback_handler.on_epoch_end(training_config=self.config)
                self.callback_handler.on_log(
                    self.config,
                    metrics,
                    logger=logger,
                    epoch=epoch,
                )

            # Restore peak model before completing task (if applicable)
            if self.use_peak and task_id > 0 and self.best_model_state is not None:
                logger.info(
                    f"Restoring peak model from epoch {self.peak_epoch} with cumulative accuracy {self.best_cumulative_accuracy:.2f}%"
                )
                self.model.load_state_dict(self.best_model_state)

            # Test on all seen tasks
            self._test_seen_tasks(task_id)
            self.callback_handler.on_task_end(
                training_config=self.config, task_id=task_id + 1
            )

            if isinstance(self.model, FisherInterface):
                self.model.complete_task(train_loader)
            self._set_optimizer()
            self._set_scheduler()
            
        if self.save:
            self._save_model()

    def _test_seen_tasks(self, current_task_id):
        if _is_class_il_setting(self.config.setting):
            logger.info(f"Testing on all seen classes up to Task {current_task_id + 1}")


            _, self.test_loader = self.tasks_dataloaders[current_task_id]
            epoch_test_loss, accuracy = self._test_step(0, current_task_id)

            logger.info(
                f"Seen Classes - Loss: {epoch_test_loss:.4f}, Accuracy: {accuracy:.4f}"
            )
            return

        if _is_task_il_setting(self.config.setting):
            logger.info(f"Testing on all tasks (Task-IL) up to Task {current_task_id + 1}")

            # Evaluate all seen tasks at once using Task-IL evaluation
            _, avg_accuracy = self._test_step_taskil(0, current_task_id)

            # Log per-task results
            for task_id, task_acc in enumerate(self.current_task_accuracies):
                logger.info(
                    f"Task {task_id + 1} - Accuracy: {task_acc:.4f}"
                )

            logger.info(f"Average Task-IL Accuracy: {avg_accuracy:.4f}")
            return

        # Fallback for other settings (Domain-IL, etc.)
        for task_id in range(current_task_id + 1):
            logger.info(f"Testing on Task {task_id + 1}/{current_task_id + 1}")
            _, self.test_loader = self.tasks_dataloaders[task_id]

            self.callback_handler.on_test_step_begin(
                training_config=self.config,
                test_loader=self.test_loader,
                epoch=task_id,
            )

            epoch_test_loss, accuracy = self._test_step(0, task_id)


            logger.info(
                f"Task {task_id + 1} - Loss: {epoch_test_loss:.4f}, Accuracy: {accuracy:.4f}"
            )

    def _test_step(self, epoch, task_id):
        if _is_class_il_setting(self.config.setting):
            return self._test_step_classil(epoch, task_id)
        elif _is_task_il_setting(self.config.setting):
            return self._test_step_taskil(epoch, task_id)
        else:
            return super()._test_step(epoch, task_id)

    def _test_step_classil(self, epoch, current_task_id):
        """
        Test step for Class-IL using task-restricted evaluation (notebook style).

        Computes:
        - Combined accuracy: task-restricted eval over all seen classes [0, seen_classes_end]
        - Per-task accuracies: task-restricted eval for each task's class range
        """
        classes_per_task = self.config.classes_per_task
        seen_classes_end = (current_task_id + 1) * classes_per_task - 1  # inclusive

        # Get cumulative test loader
        _, cumulative_test_loader = self.tasks_dataloaders[current_task_id]

        # Combined accuracy using task-restricted evaluation (notebook style)
        combined_acc, _ = self._evaluate_task_restricted(
            cumulative_test_loader, class_start=0, class_end=seen_classes_end
        )
        # Convert to percentage
        cumulative_accuracy = combined_acc * 100

        # Get per-task accuracies
        self.current_task_losses, self.current_task_accuracies = self._test_individual_tasks_classil(current_task_id)

        # Loss is not computed in task-restricted eval, set to 0
        cumulative_loss = 0.0

        return cumulative_loss, cumulative_accuracy

    @torch.no_grad()
    def _evaluate_task_restricted(self, test_loader, class_start, class_end):
        """
        Evaluate using task-restricted output slice (notebook-style evaluation).

        This matches the notebook's evaluate() function:
        - Filters samples to those with labels in [class_start, class_end]
        - Computes predictions using only the output slice for those classes
        - Returns accuracy as a fraction (0-1), not percentage

        Args:
            test_loader: DataLoader with test samples
            class_start: First class index (inclusive)
            class_end: Last class index (inclusive)

        Returns:
            accuracy: float in [0, 1]
            total: number of samples evaluated
        """
        self.model.eval()
        correct = 0
        total = 0

        for X, y in test_loader:
            X = X.to(self.device)
            y = y.to(self.device)

            # Get true class labels from one-hot encoding
            labels = y.argmax(dim=1)

            # Mask for samples in the specified class range
            mask = (labels >= class_start) & (labels <= class_end)
            if mask.sum() == 0:
                continue

            X_masked = X[mask]
            labels_masked = labels[mask]

            # Forward pass
            y_hat = self.model(X_masked)

            # Predict using only the task-specific output slice
            preds = y_hat[:, class_start:class_end+1].argmax(dim=1) + class_start

            correct += (preds == labels_masked).sum().item()
            total += mask.sum().item()

        accuracy = correct / total if total > 0 else 0.0
        return accuracy, total

    @torch.no_grad()
    def _test_individual_tasks_classil(self, current_task_id):
        """
        Test individual tasks for Class IL using task-restricted evaluation.

        This replicates the notebook's per-task accuracy computation:
        - For each task, only consider samples from that task's classes
        - Compute accuracy using only that task's output neurons
        """
        task_losses = []
        task_accuracies = []


        classes_per_task = self.config.classes_per_task


        # Use the cumulative test loader (has all seen classes)
        _, cumulative_test_loader = self.tasks_dataloaders[current_task_id]


        for task_id in range(current_task_id + 1):
            task_start = task_id * classes_per_task
            task_end = task_start + classes_per_task - 1  # inclusive

            # Use task-restricted evaluation (notebook style)
            acc, total = self._evaluate_task_restricted(
                cumulative_test_loader, task_start, task_end
            )

            # Convert to percentage for consistency with rest of trainer
            task_accuracies.append(acc * 100)
            task_losses.append(0.0)  # Loss not computed in task-restricted eval

        return task_losses, task_accuracies

    def _get_task_subset_for_testing(self, task_id, current_task_id):
        """Get test data for specific task with current network output size."""
        # Get indices for the specific task
        task_indices = self.task_test_indices[task_id]
        task_subset = Subset(self.test_dataset, task_indices)

        # Process with current network's output size
        num_classes_so_far = (current_task_id + 1) * self.classes_per_task

        test_data, test_targets = self._process_data(
            task_subset,
            self.tasks[task_id],
            lambda t: t.item() if torch.is_tensor(t) else t
        )

        return TensorDataset(
            test_data.float(), self._one_hot_encode(test_targets, num_classes_so_far)
        )


class WandBTrainerCL(TrainerCL):
    def __init__(self, model, tasks_dataloaders, config):
        super().__init__(model, tasks_dataloaders, config)
        self.task_accuracies = []
        self.global_step = 0  # Initialize a global step counter

    def _log_metrics(
        self, metrics: Dict[str, float], step: int, task_id: Optional[int] = None
    ):
        """Log metrics to WandB."""
        if wandb.run is not None:
            if task_id is not None:
                # Prefix keys with the task id.
                metrics = {f"task_{task_id}/{k}": v for k, v in metrics.items()}
            wandb.log(metrics, step=step)

    def _train_step(self, epoch: int) -> float:
        """Single training step with WandB logging."""
        # Perform training step (callbacks still receive the local epoch if needed)
        epoch_loss = super()._train_step(epoch)
        # Log training loss using the global step counter
        self._log_metrics({"train/loss": epoch_loss}, self.global_step)
        self.global_step += 1  # Increment the global step after each training epoch
        return epoch_loss

    def _test_step(self, epoch: int, task_id: int = None) -> tuple:
        """Single test step with WandB logging."""
        epoch_loss, accuracy = super()._test_step(epoch, task_id)
        self._log_metrics({
            "test/loss": epoch_loss,
            "test/accuracy": accuracy
        }, self.global_step)
        return epoch_loss, accuracy

    def _test_seen_tasks(self, current_task_id: int):
        """Test on all seen tasks with WandB logging."""
        if _is_class_il_setting(self.config.setting):
            # For Class IL, use task-restricted evaluation (notebook style)
            logger.info(f"Testing on all seen classes up to Task {current_task_id + 1}")


            _, self.test_loader = self.tasks_dataloaders[current_task_id]
            epoch_test_loss, combined_accuracy = self._test_step(self.global_step, current_task_id)

            # Get per-task accuracies computed by _test_step_classil
            task_accuracies = self.current_task_accuracies  # Already in percentage

            # Log combined accuracy (task-restricted over all seen classes)
            # Note: test/loss is 0 for class-IL (loss not computed in task-restricted eval)
            self._log_metrics({
                "test/combined_accuracy": combined_accuracy,
            }, step=self.global_step)

            # Log per-task accuracies (task-restricted per task)
            for task_id, task_acc in enumerate(task_accuracies):
                self._log_metrics({
                    "accuracy": task_acc
                }, step=self.global_step, task_id=task_id)

            # Store combined accuracy AND per-task accuracies for final metrics
            # Format: [combined_accuracy, task_0_acc, task_1_acc, ...]
            self.task_accuracies.append({
                'combined': combined_accuracy,
                'per_task': task_accuracies.copy()
            })

            # Log aggregated metrics
            avg_task_accuracy = sum(task_accuracies) / len(task_accuracies)
            self._log_metrics({
                "metrics/avg_task_accuracy": avg_task_accuracy,
                "metrics/combined_accuracy": combined_accuracy,
            }, step=self.global_step)

            return

        if _is_task_il_setting(self.config.setting):
            # Task-IL: evaluate all tasks at once using the new Task-IL evaluation
            logger.info(f"Testing on all tasks (Task-IL) up to Task {current_task_id + 1}")

            # Use the Task-IL specific test step
            _, avg_accuracy = self._test_step_taskil(0, current_task_id)

            # Get per-task accuracies computed by _test_step_taskil
            task_accuracies = self.current_task_accuracies  # Already in percentage

            # Log average accuracy
            self._log_metrics({
                "test/avg_accuracy": avg_accuracy,
            }, step=self.global_step)

            # Log per-task accuracies
            for task_id, task_acc in enumerate(task_accuracies):
                self._log_metrics({
                    "accuracy": task_acc
                }, step=self.global_step, task_id=task_id)
                logger.info(f"Task {task_id + 1} - Accuracy: {task_acc:.4f}")

            # Store accuracies for final metrics (as list for Task-IL)
            self.task_accuracies.append(task_accuracies.copy())

            # Log aggregated metrics
            self._log_metrics({
                "metrics/avg_accuracy": avg_accuracy,
                "metrics/forgetting": max(task_accuracies) - min(task_accuracies) if len(task_accuracies) > 1 else 0.0
            }, step=self.global_step)

            logger.info(f"Average Task-IL Accuracy: {avg_accuracy:.4f}")
            return

        # Fallback for other settings (Domain-IL, etc.)
        task_accuracies = []
        for task_id in range(current_task_id + 1):
            logger.info(f"Testing on Task {task_id + 1}/{current_task_id + 1}")
            self.test_loader = self.tasks_dataloaders[task_id][1]

            self.callback_handler.on_test_step_begin(
                training_config=self.config,
                test_loader=self.test_loader,
                epoch=task_id,  # used for callbacks; not for logging step
            )

            # Use the current global_step for testing logging
            epoch_test_loss, accuracy = self._test_step(self.global_step, task_id)
            task_accuracies.append(accuracy)

            # Log per-task test metrics using the current global step
            self._log_metrics(
                {"loss": epoch_test_loss, "accuracy": accuracy},
                step=self.global_step,
                task_id=task_id,
            )

        # Log aggregated metrics for all seen tasks using the current global step
        avg_accuracy = sum(task_accuracies) / len(task_accuracies)
        self._log_metrics(
            {
                "metrics/avg_accuracy": avg_accuracy,
                "metrics/forgetting": max(task_accuracies) - min(task_accuracies),
            },
            step=self.global_step,
        )
        self.task_accuracies.append(task_accuracies)

    def train(self):
        """Training loop with WandB logging."""
        # Convert OmegaConf to dict for wandb.
        if wandb.run is not None:
            config_dict = OmegaConf.to_container(self.config, resolve=True)
            wandb.config.update(config_dict)

        super().train()

        # Log final metrics.
        if wandb.run is not None and self.task_accuracies:
            final_snapshot = self.task_accuracies[-1]

            # Handle both Class-IL (dict with 'combined' and 'per_task') and Task-IL (list)
            if isinstance(final_snapshot, dict):
                # Class-IL: use combined accuracy as the main metric
                final_avg_accuracy = final_snapshot['combined']
                final_task_accs = final_snapshot['per_task']

                # Compute forgetting using per-task accuracies
                forgetting_per_task = []
                for task_id in range(len(final_task_accs) - 1):  # No forgetting for last task
                    max_acc = max(
                        snapshot['per_task'][task_id]
                        for snapshot in self.task_accuracies
                        if isinstance(snapshot, dict) and task_id < len(snapshot['per_task'])
                    )
                    forgetting = max_acc - final_task_accs[task_id]
                    forgetting_per_task.append(forgetting)

                avg_forgetting = sum(forgetting_per_task) / len(forgetting_per_task) if forgetting_per_task else 0.0

                wandb.run.summary.update({
                    "final_avg_accuracy": final_avg_accuracy,  # This is the combined accuracy (~46%)
                    "final_avg_task_accuracy": sum(final_task_accs) / len(final_task_accs),  # Avg of per-task (~97%)
                    "final_avg_forgetting": avg_forgetting,
                    **{f"final_task_{i}_accuracy": acc for i, acc in enumerate(final_task_accs)}
                })
            else:
                # Task-IL: use average of task accuracies
                final_task_accs = final_snapshot
                final_avg_accuracy = sum(final_task_accs) / len(final_task_accs)

                forgetting_per_task = []
                for task_id in range(len(final_task_accs) - 1):
                    max_acc = max(
                        snapshot[task_id]
                        for snapshot in self.task_accuracies
                        if task_id < len(snapshot)
                    )
                    forgetting = max_acc - final_task_accs[task_id]
                    forgetting_per_task.append(forgetting)

                avg_forgetting = sum(forgetting_per_task) / len(forgetting_per_task) if forgetting_per_task else 0.0

                wandb.run.summary.update({
                    "final_avg_accuracy": final_avg_accuracy,
                    "final_avg_forgetting": avg_forgetting,
                    **{f"final_task_{i}_accuracy": acc for i, acc in enumerate(final_task_accs)}
                })

