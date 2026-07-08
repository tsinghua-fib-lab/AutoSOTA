from abc import ABC, abstractmethod
import logging
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Tuple
from torch import optim, nn
from torch.optim import lr_scheduler
from torchmetrics import MetricCollection

from models.base_model import BaseModel
from utils.utils import get_device, set_seed, setup_exception_logging

LOGGER = logging.getLogger(__name__)

BEST_EPOCH_FILENAME = 'best_epoch.txt'


class BaseTrainer(ABC):
    def __init__(self,
                 experiment_dir: str,
                 n_epochs: int,
                 val_every: int = 1,
                 save_every: int = 1,
                 early_stopping_patience: int = None,
                 seed: int = None,
                 gpu_id: int = 0):
        super().__init__()
        # parameters
        self._experiment_dir = experiment_dir
        self._seed = seed
        self._gpu_id = gpu_id
        self.device = get_device(self._gpu_id)

        self._n_epochs = n_epochs
        self._val_every = val_every
        self._save_every = save_every
        self._early_stopping_patience = early_stopping_patience

        # member variables
        self._datasets = None
        self._loaders = None
        self._model: BaseModel = None
        self._optimizer: optim.Optimizer = None
        self._lr_scheduler: lr_scheduler._LRScheduler = None
        self._grad_clip_param = None
        self._loss: nn.Module = None
        self._train_metrics: MetricCollection = None
        self._val_metrics: MetricCollection = None
        self._initialized = False
        self._current_best_negative_delt_coverage = None

        if self._seed:
            set_seed(self._seed)
        else:
            LOGGER.info(f"No seed set in trainer!")
        setup_exception_logging()
        LOGGER.info(f'Logging experiment data to directory: {self._experiment_dir}.')
        self._setup()

    def _setup(self):
        pass

    def _initialize(self):
        self._create_datasets()
        self._create_dataloaders()
        self._model = self._create_model()
        self._create_loss()
        self._create_metrics()

        self._model.to(device=self.device)
        self._train_metrics.to(device=self.device)
        self._val_metrics.to(device=self.device)

        self._create_optimizer_and_scheduler(self._model)

        self._train_step = 0
        self._initialized = True

    def run_experiment(self) -> None:
        self.train()

    @abstractmethod
    def _create_datasets(self) -> None:
        pass

    @abstractmethod
    def _create_dataloaders(self) -> None:
        pass

    @abstractmethod
    def _create_model(self) -> BaseModel:
        pass

    @abstractmethod
    def _create_optimizer_and_scheduler(
            self, model: nn.Module
    ):
        pass

    @abstractmethod
    def _create_loss(self) -> None:
        """Create loss for optimization. 
        """
        pass

    @abstractmethod
    def _create_metrics(self) -> None:
        """Create a list of metrics for training and validation.
        The first entry in val_metric is used for early stopping.
        """
        # metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy()])
        # self._train_metrics = metrics.clone(prefix='train_')
        # self._val_metrics = metrics.clone(prefix='val_')
        pass

    def _reset_metrics(self) -> None:
        if self._train_metrics is not None:
            self._train_metrics.reset()
        if self._val_metrics is not None:
            self._val_metrics.reset()

    @abstractmethod
    def _train_epoch(self, epoch: int) -> None:
        pass

    @abstractmethod
    def _val_epoch(self, epoch: int, trained_model: nn.Module) -> float:
        """Implementation of one validation epoch.

        Args:
            epoch (int): Epoch number
            trained_model (nn.Module): Model to validate

        Returns:
            float: Metric value used for early stopping
        """
        pass

    def _val_lower_is_better(self) -> bool:
        """Return the value for the first validation metric in the metric collection."""
        # index 1 is the Metrics class
        return not next(iter(self._val_metrics.items()))[1].higher_is_better

    def _final_hook(self, *args, **kwargs):
        pass

    def train(self) -> Dict[str, Any]:
        """Train for n_epochs using early-stopping, epoch counter starts with 1.

        Returns:
            Dict[str, Any]: the final results
        """
        self._initialize()

        best_epoch = 0
        self._current_best_negative_delt_coverage = 100  # (must be at least alphas * 1)
        lower_is_better = self._val_lower_is_better()
        best_val_score = float('inf') if lower_is_better else -float('inf')

        self._train_step = 0

        # validate untrained model as baseline
        self._model.train(False)
        best_val_score, _ = self._val_epoch(0, self._model)
        
        # save initialized/untrained model
        self._model.save(self._experiment_dir, BaseModel.model_save_name(0))

        for epoch in range(1, self._n_epochs + 1):
            self._model.train(True)
            self._train_epoch(epoch)

            if self._lr_scheduler is not None:
                self._lr_scheduler.step(epoch=epoch)

            model_saved = False
            if self._save_every > 0 and epoch % self._save_every == 0:
                self._model.save(self._experiment_dir,
                                 BaseModel.model_save_name(epoch))
                model_saved = True

            if self._val_every > 0 and epoch % self._val_every == 0:
                self._model.train(False)
                val_score, reset_score = self._val_epoch(epoch, self._model)
                assert isinstance(val_score, float)

                if reset_score or (lower_is_better and val_score < best_val_score) or \
                        (not lower_is_better and val_score > best_val_score):
                    if reset_score:
                        LOGGER.info(
                            f"New best val score (Because of Better Negative Coverage): {self._current_best_negative_delt_coverage} PI-Score:"
                            f" {val_score} {'<' if lower_is_better else '>'} {best_val_score} (old best val score)"
                        )
                    else:
                        LOGGER.info(
                            f"New best val score (Better Width):"
                            f" {val_score} {'<' if lower_is_better else '>'} {best_val_score} (old best val score)"
                        )
                    best_epoch = epoch
                    best_val_score = val_score
                    if not model_saved:
                        self._model.save(self._experiment_dir, BaseModel.model_save_name(epoch))

                if self._early_stopping_patience:
                    if ((lower_is_better and val_score >= best_val_score)
                            or (not lower_is_better and val_score <= best_val_score)) \
                            and epoch > best_epoch + self._early_stopping_patience:
                        LOGGER.info(
                            'Early stopping patience exhausted. '
                            f'Best val score {best_val_score} in epoch {best_epoch}.'
                        )
                        break

        final_results = {
            'best_epoch': best_epoch,
            'best_val_score': best_val_score
        }
        LOGGER.info(
            f"Final results: \n{pd.Series(final_results)}"
        )

        if best_epoch > 0:
            with open(Path(self._experiment_dir) / BEST_EPOCH_FILENAME, 'w') as fp:
                fp.write(str(best_epoch))

        self._model.train(False)
        self._model.load_state((self._experiment_dir / BaseModel.model_save_name(best_epoch)), device=self.device)
        self._final_hook(final_results=final_results)

        return final_results
