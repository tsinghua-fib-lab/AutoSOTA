from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from meisenmeister.architectures import get_architecture_class
from meisenmeister.data_augmentations import (
    Compose3D,
    Contrast3D,
    FlipAxes3D,
    GaussianNoise3D,
    MultiplicativeBrightness3D,
    RandomRotation3D,
    RandomScaling3D,
    RandomShiftWithinMargin3D,
    RemoveMargin3D,
)
from meisenmeister.dataloading import MeisenmeisterROIDataset
from meisenmeister.training.base_trainer import BaseTrainer
from meisenmeister.training.splits import get_fold_sample_ids
from meisenmeister.utils.training import (
    aggregate_epoch_metrics,
    aggregate_validation_classification_metrics,
    append_history,
    autocast_context,
    build_dataloader_kwargs,
    build_experiment_paths,
    build_trainer_config,
    compute_ema,
    configure_training_performance,
    create_empty_history,
    create_grad_scaler,
    ensure_portable_inference_metadata,
    format_metric,
    is_amp_enabled,
    load_resume_checkpoint,
    log_message,
    maybe_compile_model,
    prepare_output_dir,
    resolve_amp_dtype,
    resolve_compile_enabled,
    resolve_num_workers,
    restore_rng_state,
    run_final_validation_evaluation,
    save_checkpoint,
    save_training_curves,
    should_update_best,
    unwrap_model,
    validate_resume_state,
)


class mmTrainer(BaseTrainer):
    ARCHITECTURE_NAME = "ResNet3D18"
    EMA_ALPHA = 0.1
    BEST_SCORE_TOLERANCE = 1e-12
    FINAL_EVAL_N_BOOTSTRAP = 2000
    FINAL_EVAL_CONFIDENCE_LEVEL = 0.95
    FINAL_EVAL_BOOTSTRAP_SEED = 0

    def __init__(
        self,
        dataset_id: str,
        fold: int | str,
        dataset_dir: Path,
        preprocessed_dataset_dir: Path,
        results_dir: Path,
        architecture_name: str | None = None,
        num_epochs: int = 100,
        batch_size: int = 2,
        num_workers: int | None = None,
        shuffle: bool = True,
        initial_lr: float = 3e-4,
        weight_decay: float = 3e-5,
        continue_training: bool = False,
        weights_path: Path | None = None,
        experiment_postfix: str | None = None,
        compile_enabled: bool = True,
        grad_cam_enabled: bool = False,
    ) -> None:
        super().__init__(dataset_id, fold, dataset_dir, preprocessed_dataset_dir)
        self.results_dir = results_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = resolve_num_workers(num_workers)
        self.shuffle = shuffle
        self.architecture_name = architecture_name or self.ARCHITECTURE_NAME
        self.initial_lr = initial_lr
        self.weight_decay = weight_decay
        self.continue_training = continue_training
        self.weights_path = weights_path
        self.experiment_postfix = experiment_postfix
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        configure_training_performance(self.device)
        self.amp_enabled = is_amp_enabled(self.device)
        self.amp_dtype = resolve_amp_dtype(self.device)
        self.grad_scaler = create_grad_scaler(self.device, self.amp_dtype)
        self.compile_enabled, self.compile_status_message = resolve_compile_enabled(
            self.device,
            enabled=compile_enabled,
        )
        self.grad_cam_enabled = grad_cam_enabled
        self.compile_applied = False
        self.split_sample_ids = get_fold_sample_ids(preprocessed_dataset_dir, fold)
        experiment_paths = build_experiment_paths(
            results_dir=self.results_dir,
            dataset_name=self.dataset_dir.name,
            trainer_name=self.__class__.__name__,
            architecture_name=self.architecture_name,
            experiment_postfix=self.experiment_postfix,
            fold=self.fold,
        )
        self.experiment_dir = experiment_paths["experiment_dir"]
        self.fold_dir = experiment_paths["fold_dir"]
        self.log_path = experiment_paths["log_path"]
        self.last_checkpoint_path = experiment_paths["last_checkpoint_path"]
        self.best_checkpoint_path = experiment_paths["best_checkpoint_path"]
        self.eval_last_path = experiment_paths["eval_last_path"]
        self.eval_best_path = experiment_paths["eval_best_path"]
        self.grad_cam_last_dir = experiment_paths["grad_cam_last_dir"]
        self.grad_cam_best_dir = experiment_paths["grad_cam_best_dir"]
        self.plot_path = experiment_paths["plot_path"]
        self._train_dataset = None
        self._val_dataset = None
        self._train_dataloader = None
        self._val_dataloader = None
        self._architecture = None
        self._loss = None
        self._optimizer = None
        self._scheduler = None
        self._train_augmentation_pipeline = None
        self._val_augmentation_pipeline = None
        self._plans = None
        self._history = create_empty_history()
        self._best_state = {
            "epoch": None,
            "ema_val_balanced_accuracy": None,
            "val_loss": None,
        }
        self._resume_state = None

    def fit(self) -> None:
        ensure_portable_inference_metadata(
            dataset_dir=self.dataset_dir,
            preprocessed_dataset_dir=self.preprocessed_dataset_dir,
            experiment_dir=self.experiment_dir,
        )
        overwrite_warning = prepare_output_dir(
            fold_dir=self.fold_dir,
            log_path=self.log_path,
            continue_training=self.continue_training,
        )
        if overwrite_warning is not None:
            log_message(overwrite_warning, self.log_path)

        if self.continue_training:
            self._resume_state = load_resume_checkpoint(
                last_checkpoint_path=self.last_checkpoint_path,
                best_checkpoint_path=self.best_checkpoint_path,
                log_fn=lambda message: log_message(message, self.log_path),
            )
            validate_resume_state(
                self._resume_state,
                dataset_id=self.dataset_id,
                dataset_name=self.dataset_dir.name,
                fold=self.fold,
                trainer_name=self.__class__.__name__,
                architecture_name=self.architecture_name,
                experiment_postfix=self.experiment_postfix,
                architecture_kwargs=self.get_architecture_kwargs(),
            )

        train_dataset = self.get_train_dataset()
        val_dataset = self.get_val_dataset()
        train_dataloader = self.get_train_dataloader()
        val_dataloader = self.get_val_dataloader()
        architecture = self.get_architecture()
        loss_fn = self.get_loss()

        start_epoch = 1
        if self._resume_state is not None:
            architecture.load_state_dict(self._resume_state["model_state_dict"])
            start_epoch = int(self._resume_state["last_completed_epoch"]) + 1
            log_message(
                f"Resuming training from epoch {start_epoch} in {self.fold_dir}",
                self.log_path,
            )
        elif self.weights_path is not None:
            architecture.load_initial_weights(
                path=self.weights_path,
                device=self.device,
            )
            log_message(
                f"Initialized model weights from {self.weights_path}",
                self.log_path,
            )

        log_message(
            "Preparing model for training performance optimizations.",
            self.log_path,
        )
        architecture, self.compile_applied, compile_message = maybe_compile_model(
            architecture,
            device=self.device,
            enabled=self.compile_enabled,
        )
        self._architecture = architecture
        self.compile_status_message = compile_message
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler()

        if self._resume_state is not None:
            optimizer.load_state_dict(self._resume_state["optimizer_state_dict"])
            scheduler.load_state_dict(self._resume_state["scheduler_state_dict"])
            grad_scaler_state_dict = self._resume_state.get("grad_scaler_state_dict")
            if self.grad_scaler is not None and grad_scaler_state_dict is not None:
                self.grad_scaler.load_state_dict(grad_scaler_state_dict)
            self._history = self._resume_state["history"]
            self._best_state = self._resume_state["best_state"]
            restore_rng_state(self._resume_state.get("rng_state"))

        log_message(f"Trainer: {self.__class__.__name__}", self.log_path)
        log_message(f"Dataset id: {self.dataset_id}", self.log_path)
        log_message(f"Fold: {self.fold}", self.log_path)
        log_message(f"Dataset: {self.dataset_dir.name}", self.log_path)
        log_message(f"Results dir: {self.fold_dir}", self.log_path)
        log_message(f"Train samples: {len(train_dataset)}", self.log_path)
        log_message(f"Val samples: {len(val_dataset)}", self.log_path)
        log_message(f"Batch size: {self.batch_size}", self.log_path)
        log_message(f"Architecture: {architecture.__class__.__name__}", self.log_path)
        log_message(f"Loss: {loss_fn.__class__.__name__}", self.log_path)
        log_message(f"Optimizer: {optimizer.__class__.__name__}", self.log_path)
        log_message(f"Scheduler: {scheduler.__class__.__name__}", self.log_path)
        log_message(f"Epochs: {self.num_epochs}", self.log_path)
        log_message(f"Initial LR: {self.initial_lr}", self.log_path)
        log_message(f"Weight decay: {self.weight_decay}", self.log_path)
        log_message(f"AMP enabled: {self.amp_enabled}", self.log_path)
        log_message(f"AMP dtype: {self.amp_dtype}", self.log_path)
        log_message(
            f"Torch compile enabled: {self.compile_enabled}",
            self.log_path,
        )
        log_message(
            f"Torch compile applied: {self.compile_applied}",
            self.log_path,
        )
        log_message(
            f"Torch compile status: {self.compile_status_message}",
            self.log_path,
        )
        log_message(f"DataLoader workers: {self.num_workers}", self.log_path)
        log_message(
            f"DataLoader persistent_workers: {self.num_workers > 0}", self.log_path
        )
        log_message(
            f"DataLoader prefetch_factor: {2 if self.num_workers > 0 else None}",
            self.log_path,
        )

        if start_epoch > self.num_epochs:
            log_message(
                f"Training already completed through epoch {start_epoch - 1}; nothing to do.",
                self.log_path,
            )
            run_final_validation_evaluation(
                self,
                output_path=self.eval_last_path,
                log_path=self.log_path,
                n_bootstrap=self.FINAL_EVAL_N_BOOTSTRAP,
                confidence_level=self.FINAL_EVAL_CONFIDENCE_LEVEL,
                seed=self.FINAL_EVAL_BOOTSTRAP_SEED,
                log_fn=log_message,
                grad_cam_output_dir=(
                    self.grad_cam_last_dir if self.grad_cam_enabled else None
                ),
                grad_cam_checkpoint_kind="last",
            )
            log_message("DONE", self.log_path)
            return

        for epoch_idx in range(start_epoch, self.num_epochs + 1):
            epoch_start_time = time.perf_counter()
            log_message(
                f"Epoch {epoch_idx}/{self.num_epochs} - starting training loop",
                self.log_path,
            )

            architecture.train()
            train_metrics = []
            for batch_idx, batch in enumerate(train_dataloader, start=1):
                train_metrics.append(self.train_step(batch, batch_idx))

            log_message(
                f"Epoch {epoch_idx}/{self.num_epochs} - starting validation loop",
                self.log_path,
            )
            architecture.eval()
            val_metrics = []
            for batch_idx, batch in enumerate(val_dataloader, start=1):
                val_metrics.append(self.validate_step(batch, batch_idx))

            train_loss, train_accuracy = aggregate_epoch_metrics(train_metrics)
            val_loss, val_accuracy = aggregate_epoch_metrics(val_metrics)
            val_balanced_accuracy, val_macro_auc, macro_auc_defined = (
                aggregate_validation_classification_metrics(val_metrics)
            )
            ema_val_balanced_accuracy = compute_ema(
                self._history["ema_val_balanced_accuracy"],
                val_balanced_accuracy,
                alpha=self.EMA_ALPHA,
            )
            current_lr = float(optimizer.param_groups[0]["lr"])
            epoch_time_sec = float(time.perf_counter() - epoch_start_time)

            append_history(
                self._history,
                epoch=epoch_idx,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                val_balanced_accuracy=val_balanced_accuracy,
                val_macro_auc=val_macro_auc,
                ema_val_balanced_accuracy=ema_val_balanced_accuracy,
                lr=current_lr,
                epoch_time_sec=epoch_time_sec,
            )

            if not macro_auc_defined:
                log_message(
                    f"Epoch {epoch_idx}/{self.num_epochs} - val_macro_auc undefined for this epoch because validation targets do not span enough classes.",
                    self.log_path,
                )

            log_message(
                f"Epoch {epoch_idx}/{self.num_epochs} "
                f"- train_loss: {train_loss:.4f} "
                f"- train_acc: {train_accuracy:.4f} "
                f"- val_loss: {val_loss:.4f} "
                f"- val_acc: {val_accuracy:.4f} "
                f"- val_bal_acc: {val_balanced_accuracy:.4f} "
                f"- val_macro_auc: {format_metric(val_macro_auc)} "
                f"- ema_val_bal_acc: {ema_val_balanced_accuracy:.4f} "
                f"- epoch_time_sec: {epoch_time_sec:.4f} "
                f"- lr: {current_lr:.6f}",
                self.log_path,
            )

            if should_update_best(
                self._best_state,
                ema_val_balanced_accuracy,
                val_loss,
                tolerance=self.BEST_SCORE_TOLERANCE,
            ):
                self._best_state = {
                    "epoch": epoch_idx,
                    "ema_val_balanced_accuracy": ema_val_balanced_accuracy,
                    "val_loss": val_loss,
                }
                save_checkpoint(
                    path=self.best_checkpoint_path,
                    epoch_idx=epoch_idx,
                    trainer_config=build_trainer_config(
                        dataset_id=self.dataset_id,
                        dataset_name=self.dataset_dir.name,
                        fold=self.fold,
                        trainer_name=self.__class__.__name__,
                        architecture_name=self.architecture_name,
                        experiment_postfix=self.experiment_postfix,
                        source_weights_path=(
                            None
                            if self.weights_path is None
                            else str(self.weights_path)
                        ),
                        results_dir=self.results_dir,
                        experiment_dir=self.experiment_dir,
                        fold_dir=self.fold_dir,
                        num_epochs=self.num_epochs,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        shuffle=self.shuffle,
                        initial_lr=self.initial_lr,
                        weight_decay=self.weight_decay,
                        device=self.device,
                        architecture=unwrap_model(self.get_architecture()),
                    ),
                    history=self._history,
                    best_state=self._best_state,
                    model_state_dict=unwrap_model(self.get_architecture()).state_dict(),
                    optimizer_state_dict=self.get_optimizer().state_dict(),
                    scheduler_state_dict=self.get_scheduler().state_dict(),
                    grad_scaler_state_dict=(
                        None
                        if self.grad_scaler is None
                        else self.grad_scaler.state_dict()
                    ),
                )
                log_message(
                    f"Saved new best model at epoch {epoch_idx} with ema_val_bal_acc={ema_val_balanced_accuracy:.4f} and val_loss={val_loss:.4f}",
                    self.log_path,
                )

            scheduler.step()
            save_checkpoint(
                path=self.last_checkpoint_path,
                epoch_idx=epoch_idx,
                trainer_config=build_trainer_config(
                    dataset_id=self.dataset_id,
                    dataset_name=self.dataset_dir.name,
                    fold=self.fold,
                    trainer_name=self.__class__.__name__,
                    architecture_name=self.architecture_name,
                    experiment_postfix=self.experiment_postfix,
                    source_weights_path=(
                        None if self.weights_path is None else str(self.weights_path)
                    ),
                    results_dir=self.results_dir,
                    experiment_dir=self.experiment_dir,
                    fold_dir=self.fold_dir,
                    num_epochs=self.num_epochs,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=self.shuffle,
                    initial_lr=self.initial_lr,
                    weight_decay=self.weight_decay,
                    device=self.device,
                    architecture=unwrap_model(self.get_architecture()),
                ),
                history=self._history,
                best_state=self._best_state,
                model_state_dict=unwrap_model(self.get_architecture()).state_dict(),
                optimizer_state_dict=self.get_optimizer().state_dict(),
                scheduler_state_dict=self.get_scheduler().state_dict(),
                grad_scaler_state_dict=(
                    None if self.grad_scaler is None else self.grad_scaler.state_dict()
                ),
            )
            save_training_curves(self._history, self.plot_path)

        run_final_validation_evaluation(
            self,
            output_path=self.eval_last_path,
            log_path=self.log_path,
            n_bootstrap=self.FINAL_EVAL_N_BOOTSTRAP,
            confidence_level=self.FINAL_EVAL_CONFIDENCE_LEVEL,
            seed=self.FINAL_EVAL_BOOTSTRAP_SEED,
            log_fn=log_message,
            grad_cam_output_dir=(
                self.grad_cam_last_dir if self.grad_cam_enabled else None
            ),
            grad_cam_checkpoint_kind="last",
        )
        log_message("DONE", self.log_path)

    def ensure_grad_cam_available(self) -> None:
        architecture = unwrap_model(self.get_architecture())
        try:
            target_layer = architecture.get_grad_cam_target_layer()
        except NotImplementedError as error:
            raise ValueError(str(error)) from error
        if not isinstance(target_layer, nn.Module):
            raise TypeError("Grad-CAM target layer must be a torch.nn.Module instance")

    def get_architecture(self):
        if self._architecture is None:
            architecture_class = get_architecture_class(self.architecture_name)
            in_channels = int(self.get_train_dataset()[0]["image"].shape[0])
            labels = [
                sample["label"]
                for sample in (
                    self.get_train_dataset().samples + self.get_val_dataset().samples
                )
            ]
            num_classes = max(2, max(labels) + 1)
            self._architecture = architecture_class(
                in_channels=in_channels,
                num_classes=num_classes,
                **self.get_architecture_kwargs(),
            ).to(self.device)
        return self._architecture

    def get_architecture_kwargs(self) -> dict:
        return {}

    def get_loss(self):
        if self._loss is None:
            self._loss = nn.CrossEntropyLoss()
        return self._loss

    def get_train_dataset(self):
        if self._train_dataset is None:
            self._train_dataset = MeisenmeisterROIDataset(
                self.preprocessed_dataset_dir,
                allowed_sample_ids=set(self.split_sample_ids["train"]),
                augmentation_pipeline=self.get_train_augmentation_pipeline(),
            )
        return self._train_dataset

    def get_val_dataset(self):
        if self._val_dataset is None:
            self._val_dataset = MeisenmeisterROIDataset(
                self.preprocessed_dataset_dir,
                allowed_sample_ids=set(self.split_sample_ids["val"]),
                augmentation_pipeline=self.get_val_augmentation_pipeline(),
            )
        return self._val_dataset

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
                    RandomRotation3D(
                        probability=0.3,
                        max_rotation_degrees=(15.0, 15.0, 15.0),
                    ),
                    RandomScaling3D(
                        probability=0.2,
                        scaling=(0.7, 1.4),
                    ),
                    MultiplicativeBrightness3D(
                        probability=0.15,
                        multiplier_range=(0.75, 1.25),
                        p_per_channel=1.0,
                        synchronize_channels=True,
                    ),
                    Contrast3D(
                        probability=0.15,
                        contrast_range=(0.75, 1.25),
                        preserve_range=True,
                        p_per_channel=1.0,
                        synchronize_channels=True,
                    ),
                    GaussianNoise3D(
                        probability=0.1,
                        noise_variance=(0.0, 0.1),
                        p_per_channel=1.0,
                        synchronize_channels=True,
                    ),
                    FlipAxes3D(probability=1.0, axes=(0, 1, 2)),
                ]
            )
        return self._train_augmentation_pipeline

    def get_val_augmentation_pipeline(self):
        if self._val_augmentation_pipeline is None:
            plans = self.get_preprocessing_plans()
            margin_voxels = tuple(
                int(margin_axis / spacing_axis)
                for margin_axis, spacing_axis in zip(
                    plans["margin_mm"],
                    plans["target_spacing"],
                    strict=True,
                )
            )
            self._val_augmentation_pipeline = Compose3D(
                [RemoveMargin3D(margin_voxels=margin_voxels)]
            )
        return self._val_augmentation_pipeline

    def get_preprocessing_plans(self) -> dict:
        if self._plans is None:
            plans_path = self.preprocessed_dataset_dir / "mmPlans.json"
            with plans_path.open("r", encoding="utf-8") as file:
                self._plans = json.load(file)
        return self._plans

    def get_train_dataloader(self):
        if self._train_dataloader is None:
            self._train_dataloader = DataLoader(
                self.get_train_dataset(),
                **build_dataloader_kwargs(
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    num_workers=self.num_workers,
                ),
            )
        return self._train_dataloader

    def get_val_dataloader(self):
        if self._val_dataloader is None:
            self._val_dataloader = DataLoader(
                self.get_val_dataset(),
                **build_dataloader_kwargs(
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                ),
            )
        return self._val_dataloader

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
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.num_epochs,
            )
        return self._scheduler

    def train_step(self, batch, batch_idx: int):
        del batch_idx
        architecture = self.get_architecture()
        optimizer = self.get_optimizer()
        loss_fn = self.get_loss()
        images = batch["image"].to(self.device, dtype=torch.float32, non_blocking=True)
        labels = batch["label"].to(self.device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(self.device, self.amp_dtype):
            logits = architecture(images)
            loss = loss_fn(logits, labels)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        predictions = logits.argmax(dim=1)
        return {
            "loss_sum": loss.detach() * labels.shape[0],
            "num_samples": int(labels.shape[0]),
            "num_correct": (predictions == labels).sum().detach(),
        }

    def validate_step(self, batch, batch_idx: int):
        del batch_idx
        architecture = self.get_architecture()
        loss_fn = self.get_loss()
        images = batch["image"].to(self.device, dtype=torch.float32, non_blocking=True)
        labels = batch["label"].to(self.device, dtype=torch.long, non_blocking=True)

        with torch.inference_mode():
            with autocast_context(self.device, self.amp_dtype):
                logits = architecture(images)
                loss = loss_fn(logits, labels)
            predictions = logits.argmax(dim=1)
            probabilities = torch.softmax(logits.float(), dim=1)

        return {
            "loss_sum": loss.detach() * labels.shape[0],
            "num_samples": int(labels.shape[0]),
            "num_correct": (predictions == labels).sum().detach(),
            "labels": labels.detach().cpu(),
            "predictions": predictions.detach().cpu(),
            "probabilities": probabilities.detach().cpu(),
            "sample_ids": list(batch["sample_id"]),
            "case_ids": list(batch["case_id"]),
            "roi_names": list(batch["roi_name"]),
        }
