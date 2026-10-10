from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from meisenmeister.architectures import get_architecture_class
from meisenmeister.data_augmentations import (
    Compose3D,
    RandomShiftWithinMargin3D,
    build_default_mri_torchio_pipeline,
)
from meisenmeister.training.trainers.mm_trainer import mmTrainer


class _WarmupPolyLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_lr: float,
        max_lr: float,
        num_epochs: int,
        warmup_epochs: int,
        poly_exp: float,
        last_epoch: int = -1,
    ) -> None:
        if num_epochs < 1:
            raise ValueError(f"num_epochs must be at least 1, got {num_epochs}")
        if warmup_epochs < 1:
            raise ValueError(f"warmup_epochs must be at least 1, got {warmup_epochs}")
        if warmup_epochs > num_epochs:
            raise ValueError("warmup_epochs must be less than or equal to num_epochs")
        if initial_lr <= 0.0 or max_lr <= 0.0:
            raise ValueError("initial_lr and max_lr must be positive")
        if poly_exp <= 0.0:
            raise ValueError(f"poly_exp must be positive, got {poly_exp}")

        self.initial_lr = float(initial_lr)
        self.max_lr = float(max_lr)
        self.num_epochs = int(num_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.poly_exp = float(poly_exp)
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        current_epoch = self.last_epoch + 1
        if current_epoch <= self.warmup_epochs:
            warmup_progress = current_epoch / float(self.warmup_epochs)
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * warmup_progress
        else:
            decay_steps = max(1, self.num_epochs - self.warmup_epochs)
            progress = (current_epoch - self.warmup_epochs) / float(decay_steps)
            progress = min(max(progress, 0.0), 1.0)
            lr = self.max_lr * ((1.0 - progress) ** self.poly_exp)
        return [lr for _ in self.optimizer.param_groups]


class mmTrainer_NNUNetEncoder(mmTrainer):
    ARCHITECTURE_NAME = "ResidualEncoderClsNetwork"

    def __init__(
        self,
        dataset_id,
        fold,
        dataset_dir,
        preprocessed_dataset_dir,
        results_dir,
        architecture_name: str | None = None,
        num_workers: int | None = None,
        continue_training: bool = False,
        weights_path=None,
        experiment_postfix: str | None = None,
        compile_enabled: bool = True,
        grad_cam_enabled: bool = False,
    ) -> None:
        super().__init__(
            dataset_id=dataset_id,
            fold=fold,
            dataset_dir=dataset_dir,
            preprocessed_dataset_dir=preprocessed_dataset_dir,
            results_dir=results_dir,
            architecture_name=architecture_name or self.ARCHITECTURE_NAME,
            num_workers=num_workers,
            continue_training=continue_training,
            weights_path=weights_path,
            experiment_postfix=experiment_postfix,
            compile_enabled=compile_enabled,
            grad_cam_enabled=grad_cam_enabled,
        )

    def fit(self) -> None:
        self._validate_target_shape()
        super().fit()

    def _validate_target_shape(self) -> None:
        plans = self.get_preprocessing_plans()
        target_shape = plans.get("target_shape")
        if target_shape is None:
            raise ValueError(
                "mmPlans.json must define target_shape for mmTrainer_NNUNetEncoder"
            )

        architecture_class = get_architecture_class(self.architecture_name)
        required_divisibility = getattr(
            architecture_class,
            "ENCODER_INPUT_DIVISIBILITY",
            None,
        )
        if required_divisibility is None:
            return

        resolved_target_shape = [int(axis) for axis in target_shape]
        incompatible_axes = [
            axis
            for axis, divisor in zip(
                resolved_target_shape,
                required_divisibility,
                strict=True,
            )
            if axis % int(divisor) != 0
        ]
        if incompatible_axes:
            raise ValueError(
                f"{self.architecture_name} requires target_shape divisible by "
                f"{list(required_divisibility)}, got {resolved_target_shape}"
            )


class mmTrainer_NNUNetEncoder_Finetune(mmTrainer_NNUNetEncoder):
    def __init__(
        self,
        dataset_id,
        fold,
        dataset_dir: Path,
        preprocessed_dataset_dir: Path,
        results_dir: Path,
        architecture_name: str | None = None,
        num_workers: int | None = None,
        continue_training: bool = False,
        weights_path=None,
        experiment_postfix: str | None = None,
        compile_enabled: bool = True,
        grad_cam_enabled: bool = False,
    ) -> None:
        super().__init__(
            dataset_id=dataset_id,
            fold=fold,
            dataset_dir=dataset_dir,
            preprocessed_dataset_dir=preprocessed_dataset_dir,
            results_dir=results_dir,
            architecture_name=architecture_name,
            num_workers=num_workers,
            continue_training=continue_training,
            weights_path=weights_path,
            experiment_postfix=experiment_postfix,
            compile_enabled=compile_enabled,
            grad_cam_enabled=grad_cam_enabled,
        )
        self.num_epochs = 100
        self.initial_lr = 1e-5
        self.max_lr = 1e-3
        self.warmup_epochs = 10
        self.weight_decay = 3e-5
        self.poly_exp = 0.9
        self._optimizer = None
        self._scheduler = None
        self.batch_size = 1

    def get_optimizer(self):
        if self._optimizer is None:
            architecture = self.get_architecture()
            parameters = list(architecture.parameters())
            if not parameters:
                raise ValueError(
                    f"Architecture '{architecture.__class__.__name__}' has no trainable parameters"
                )
            self._optimizer = torch.optim.SGD(
                parameters,
                lr=self.initial_lr,
                weight_decay=self.weight_decay,
                momentum=0.99,
                nesterov=True,
            )
        return self._optimizer

    def get_scheduler(self):
        if self._scheduler is None:
            optimizer = self.get_optimizer()
            self._scheduler = _WarmupPolyLRScheduler(
                optimizer=optimizer,
                initial_lr=self.initial_lr,
                max_lr=self.max_lr,
                num_epochs=self.num_epochs,
                warmup_epochs=self.warmup_epochs,
                poly_exp=self.poly_exp,
            )
        return self._scheduler


class mmTrainer_NNUNetEncoder_Finetune_ClassBalanced(mmTrainer_NNUNetEncoder_Finetune):
    MIN_CLASS_WEIGHT = 1e-3

    def get_loss(self):
        if self._loss is None:
            labels = [
                int(sample["label"]) for sample in self.get_train_dataset().samples
            ]
            if not labels:
                raise ValueError("Cannot compute class-balanced loss without labels")

            num_classes = max(2, max(labels) + 1)
            class_counts = torch.bincount(
                torch.tensor(labels, dtype=torch.long),
                minlength=num_classes,
            )
            safe_class_counts = class_counts.clamp_min(1)
            class_weights = class_counts.sum().to(torch.float32) / safe_class_counts
            class_weights = class_weights / class_weights.mean()
            class_weights = class_weights.clamp_min(self.MIN_CLASS_WEIGHT)
            self._loss = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        return self._loss


class mmTrainer_NNUNetEncoder_Finetune_TorchIO(mmTrainer_NNUNetEncoder_Finetune):
    def get_train_augmentation_pipeline(self):
        if self._train_augmentation_pipeline is None:
            plans = self.get_preprocessing_plans()
            max_shift_voxels = tuple(
                int(margin_axis / spacing_axis)
                for margin_axis, spacing_axis in zip(
                    plans["margin_mm"],
                    plans["target_spacing"],
                    strict=True,
                )
            )
            self._train_augmentation_pipeline = Compose3D(
                [
                    RandomShiftWithinMargin3D(
                        probability=0.5,
                        max_shift_voxels=max_shift_voxels,
                    ),
                    build_default_mri_torchio_pipeline(),
                ]
            )
        return self._train_augmentation_pipeline
