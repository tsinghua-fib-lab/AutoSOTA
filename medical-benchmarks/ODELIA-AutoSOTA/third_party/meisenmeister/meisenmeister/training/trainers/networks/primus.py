from __future__ import annotations

from pathlib import Path

import torch

from meisenmeister.training.trainers.mm_trainer import mmTrainer


class mmTrainer_PrimusM(mmTrainer):
    ARCHITECTURE_NAME = "PrimusMClsNetwork"
    PATCH_EMBED_SIZE = (8, 8, 8)
    ADAMW_BETAS = (0.9, 0.98)

    def __init__(
        self,
        dataset_id,
        fold,
        dataset_dir: Path,
        preprocessed_dataset_dir: Path,
        results_dir: Path,
        architecture_name: str | None = None,
        num_epochs: int = 100,
        batch_size: int = 2,
        num_workers: int | None = None,
        shuffle: bool = True,
        initial_lr: float = 3e-4,
        weight_decay: float = 5e-2,
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
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            initial_lr=initial_lr,
            weight_decay=weight_decay,
            continue_training=continue_training,
            weights_path=weights_path,
            experiment_postfix=experiment_postfix,
            compile_enabled=compile_enabled,
            grad_cam_enabled=grad_cam_enabled,
        )

    def fit(self) -> None:
        self._validate_target_shape()
        super().fit()

    def get_architecture_kwargs(self) -> dict:
        return {
            "input_shape": self._get_target_shape(),
            "patch_embed_size": self.PATCH_EMBED_SIZE,
        }

    def get_optimizer(self):
        if self._optimizer is None:
            architecture = self.get_architecture()
            parameters = list(architecture.parameters())
            if not parameters:
                raise ValueError(
                    f"Architecture '{architecture.__class__.__name__}' has no trainable parameters"
                )
            self._optimizer = torch.optim.AdamW(
                parameters,
                lr=self.initial_lr,
                weight_decay=self.weight_decay,
                betas=self.ADAMW_BETAS,
            )
        return self._optimizer

    def _get_target_shape(self) -> tuple[int, int, int]:
        plans = self.get_preprocessing_plans()
        target_shape = plans.get("target_shape")
        if target_shape is None:
            raise ValueError(
                "mmPlans.json must define target_shape for mmTrainer_PrimusM"
            )
        resolved_target_shape = tuple(int(axis) for axis in target_shape)
        if len(resolved_target_shape) != 3:
            raise ValueError(
                f"mmTrainer_PrimusM requires a 3D target_shape, got {target_shape}"
            )
        return resolved_target_shape

    def _validate_target_shape(self) -> None:
        resolved_target_shape = self._get_target_shape()
        incompatible_axes = [
            axis
            for axis, divisor in zip(
                resolved_target_shape,
                self.PATCH_EMBED_SIZE,
                strict=True,
            )
            if axis % int(divisor) != 0
        ]
        if incompatible_axes:
            raise ValueError(
                f"{self.architecture_name} requires target_shape divisible by "
                f"{list(self.PATCH_EMBED_SIZE)}, got {list(resolved_target_shape)}"
            )


class mmTrainer_PrimusM_bs4(mmTrainer_PrimusM):
    def __init__(
        self,
        dataset_id,
        fold,
        dataset_dir: Path,
        preprocessed_dataset_dir: Path,
        results_dir: Path,
        architecture_name: str | None = None,
        num_epochs: int = 100,
        batch_size: int = 4,
        num_workers: int | None = None,
        shuffle: bool = True,
        initial_lr: float = 3e-4,
        weight_decay: float = 5e-2,
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
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            initial_lr=initial_lr,
            weight_decay=weight_decay,
            continue_training=continue_training,
            weights_path=weights_path,
            experiment_postfix=experiment_postfix,
            compile_enabled=compile_enabled,
            grad_cam_enabled=grad_cam_enabled,
        )
