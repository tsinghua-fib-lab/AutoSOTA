from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseTrainer(ABC):
    def __init__(
        self,
        dataset_id: str,
        fold: int | str,
        dataset_dir: Path,
        preprocessed_dataset_dir: Path,
    ) -> None:
        self.dataset_id = dataset_id
        self.fold = fold
        self.dataset_dir = dataset_dir
        self.preprocessed_dataset_dir = preprocessed_dataset_dir

    @abstractmethod
    def fit(self) -> None:
        """Run the trainer's fit loop."""

    @abstractmethod
    def get_architecture(self):
        """Return the model architecture used by this trainer."""

    @abstractmethod
    def get_loss(self):
        """Return the loss function used by this trainer."""

    @abstractmethod
    def get_train_dataset(self):
        """Return the training dataset instance used by this trainer."""

    @abstractmethod
    def get_train_augmentation_pipeline(self):
        """Return the augmentation pipeline used for training samples."""

    @abstractmethod
    def get_val_dataset(self):
        """Return the validation dataset instance used by this trainer."""

    @abstractmethod
    def get_val_augmentation_pipeline(self):
        """Return the augmentation pipeline used for validation samples."""

    @abstractmethod
    def get_train_dataloader(self):
        """Return the training dataloader used by this trainer."""

    @abstractmethod
    def get_val_dataloader(self):
        """Return the validation dataloader used by this trainer."""

    @abstractmethod
    def get_optimizer(self):
        """Return the optimizer used by this trainer."""

    @abstractmethod
    def get_scheduler(self):
        """Return the scheduler used by this trainer."""

    @abstractmethod
    def train_step(self, batch, batch_idx: int):
        """Run one training step."""

    @abstractmethod
    def validate_step(self, batch, batch_idx: int):
        """Run one validation step."""
