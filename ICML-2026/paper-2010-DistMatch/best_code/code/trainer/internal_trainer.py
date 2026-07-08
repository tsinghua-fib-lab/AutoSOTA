from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Any

import torch
import wandb
from torch import nn
from torchmetrics import MetricCollection
from tqdm import tqdm

from models.base_model import BaseModel
from trainer.basetrainer import BaseTrainer, LOGGER

_TRAIN_STEP_PREFIX = "train_step/"
_TRAIN_EP_PREFIX = "train_epoch/"
_VAL_EP_PREFIX = "val/"


class ModelInternalTrainer(BaseTrainer):
    """
    Trainer which is called from within the model as a form of subroutine
    """

    def __init__(
        self,
        trainer_config,
        experiment_config,
        model,
        get_data_loader: Callable,
        move_batch_to_device: Callable,
        map_to_model_in: Callable,
        loss_func: Callable,
        map_to_loss_in: Callable,
        train_metrics: MetricCollection,
        val_metrics: MetricCollection,
        map_to_metrics_in: Callable,
    ):
        self._trainer_config = trainer_config
        self._experiment_config = experiment_config
        super().__init__(
            experiment_dir=experiment_config.experiment_dir,
            gpu_id=experiment_config.gpu_id,
            seed=self._experiment_config.seed,
            n_epochs=trainer_config.n_epochs,
            val_every=trainer_config.val_every,
            save_every=trainer_config.save_every,
            early_stopping_patience=trainer_config.early_stopping_patience,
        )
        self._model = model
        self._get_data_loader = get_data_loader
        self._move_batch_to_device = move_batch_to_device
        self._map_to_model_in = map_to_model_in
        self._loss_func = loss_func
        self._map_to_loss_in = map_to_loss_in
        self._train_metrics = train_metrics.clone()
        self._val_metrics = val_metrics.clone()
        self._map_to_metrics_in = map_to_metrics_in

    def _setup(self):
        # Setup Wandb Metrics
        wandb.define_metric(f"{_TRAIN_STEP_PREFIX}train_step", summary="none")
        wandb.define_metric(f"epoch", summary="none")
        # wandb.define_metric(f"{_VAL_EP_PREFIX}loss", step_metric="epoch", objective='minimize', summary='best')
        wandb.define_metric(
            f"{_VAL_EP_PREFIX}loss", step_metric="epoch", summary="best"
        )
        wandb.define_metric(
            f"{_TRAIN_STEP_PREFIX}*",
            step_metric=f"{_TRAIN_STEP_PREFIX}train_step",
            summary="none",
        )
        wandb.define_metric(f"{_TRAIN_EP_PREFIX}*", step_metric="epoch", summary="none")
        wandb.define_metric(f"{_VAL_EP_PREFIX}*", step_metric="epoch", summary="none")

    def _create_dataloaders(self):
        if hasattr(self._trainer_config, "num_worker"):
            num_worker = self._trainer_config.num_worker
            train_loader, val_loader = self._get_data_loader(num_worker)
            LOGGER.info(f"Init Dataloader with {num_worker}.")
        else:
            train_loader, val_loader = self._get_data_loader()
            LOGGER.info(f"Init Dataloader.")

        self._loaders = dict(train=train_loader, val=val_loader)

    def _create_model(self) -> BaseModel:
        return self._model

    def _create_optimizer_and_scheduler(self, model: nn.Module):
        LOGGER.info("Create optimizer and scheduler.")
        self._optimizer = self._trainer_config.optim(params=model.parameters())
        if self._trainer_config.lr_scheduler:
            LOGGER.info("LR Scheduler is used!")
            self._lr_scheduler = self._trainer_config.lr_scheduler(
                optimizer=self._optimizer
            )
        self._grad_clip_param = (
            self._trainer_config.grad_clip_param
            if hasattr(self._trainer_config, "grad_clip_param")
            else None
        )
        if self._grad_clip_param is not None:
            LOGGER.info("Gradient Clipping is used!")

    def _train_epoch(self, epoch: int) -> None:
        # setup logging
        filter_nan_metrics = True

        # training loop (iterations per epoch)
        pbar = tqdm(self._loaders["train"], desc=f"Train epoch {epoch}")

        loss_log = defaultdict(list)
        for batch_idx, batch_data in enumerate(pbar):
            batch_data = self._move_batch_to_device(
                batch_data=batch_data, device=self.device
            )
            model_in = self._map_to_model_in(batch_data)  # Gen model Input
            # batch_info = BatchInfo('train', epoch, batch_idx, len(batch_data[0]), pbar.total, self._train_step)
            # forward pass
            model_out = self._model(val=False, **model_in)
            loss_in = self._map_to_loss_in(model_out, batch_data)
            # loss_in = tuple(map(lambda x: x.to(self.device), loss_in))  # Move all elements to device
            loss_total, loss_individual = self._loss_func(**loss_in)

            # backward pass
            self._optimizer.zero_grad()
            loss_total.backward()
            if self._grad_clip_param is not None:
                torch.nn.utils.clip_grad_norm_(
                    parameters=self._model.parameters(), **self._grad_clip_param
                )
            self._optimizer.step()
            self._train_step += 1

            # Save Loss
            loss_log["loss"].append(loss_total.item())
            for key, val in loss_individual.items():
                loss_log[key].append(val.item())

            # Metrics
            with torch.no_grad():
                metric_in = self._map_to_metrics_in(model_out, batch_data)
                # metric_in = tuple(map(lambda x: x.to(self._metric_device), metric_in))  # Move all elements to CPU
                metric_vals = self._train_metrics(**metric_in)

            # log step
            if filter_nan_metrics:
                metric_vals = {
                    key: val for key, val in metric_vals.items() if not torch.isnan(val)
                }
            step_log_dict = dict(
                train_step=self._train_step,
                loss=loss_total,
                **loss_individual,
                **metric_vals,
            )
            wandb.log(
                {
                    f"{_TRAIN_STEP_PREFIX}{key}": val
                    for key, val in step_log_dict.items()
                }
            )

        # log epoch
        metric_vals = self._train_metrics.compute()
        loss_vals = {
            key: torch.tensor(vals).mean().item() for key, vals in loss_log.items()
        }

        log_dict = {"last_train_step": self._train_step, **loss_vals, **metric_vals}
        if self._lr_scheduler is not None:
            log_dict["learning_rate"] = self._lr_scheduler.get_last_lr()[-1]
        wandb.log(
            dict(
                **{f"{_TRAIN_EP_PREFIX}{key}": val for key, val in log_dict.items()},
                epoch=epoch,
            )
        )
        # LOGGER.info(f'Train epoch \n{pd.Series(convert_dict_to_python_types(log_dict), dtype=float)}')
        self._reset_metrics()

    def _val_epoch(self, epoch: int, trained_model: nn.Module) -> float:
        pbar = tqdm(self._loaders["val"], desc=f"Val epoch {epoch}")

        loss_log = defaultdict(list)
        for batch_idx, batch_data in enumerate(pbar):
            batch_data = self._move_batch_to_device(
                batch_data=batch_data, device=self.device
            )
            model_in = self._map_to_model_in(batch_data)  # Gen model Input
            # batch_info = BatchInfo('val', epoch, batch_idx, len(batch_data[0]), pbar.total, self._train_step)
            with torch.no_grad():
                model_out = trained_model(val=True, **model_in)
                loss_in = self._map_to_loss_in(model_out, batch_data)
                # loss_in = tuple(map(lambda x: x.to(self.device), loss_in))  # Move all elements to device
                loss_total, loss_individual = self._loss_func(**loss_in)

                # Save Loss
                loss_log["loss"].append(loss_total.item())
                for key, val in loss_individual.items():
                    loss_log[key].append(val.item())

                metric_in = self._map_to_metrics_in(model_out, batch_data)
                # metric_in = tuple(map(lambda x: x.to(self._metric_device), metric_in))  # Move all elements to CPU
                m_val = self._val_metrics(**metric_in)

        # compute mean train_metrics over dataset
        metric_vals = self._val_metrics.compute()
        # log epoch
        loss_vals = {
            key: torch.tensor(vals).mean().item() for key, vals in loss_log.items()
        }
        log_dict = {**loss_vals, **metric_vals}
        wandb.log(
            dict(
                **{f"{_VAL_EP_PREFIX}{key}": val for key, val in log_dict.items()},
                epoch=epoch,
            )
        )
        # LOGGER.info(f'Val epoch \n{pd.Series(convert_dict_to_python_types(log_dict), dtype=float)}')

        reset_old_score = False
        if (
            hasattr(self._trainer_config, "model_selection")
            and self._trainer_config.model_selection == "threshold-pi"
        ):
            negative_delta_coverage = metric_vals["MissCoverage"].item()
            negative_delta_coverage = max(
                0, negative_delta_coverage
            )  # avoid any floating point problems
            # Reset old score
            if (
                self._current_best_negative_delt_coverage > 0
                and negative_delta_coverage <= self._current_best_negative_delt_coverage
            ):
                val_score = metric_vals["PIWidth"].item()
                self._current_best_negative_delt_coverage = negative_delta_coverage
                reset_old_score = True
            # Compare with old ones the PI Widht
            elif negative_delta_coverage <= self._current_best_negative_delt_coverage:
                val_score = metric_vals["PIWidth"].item()
            else:
                val_score = float("inf")
        else:
            # val_score is first metric in self._metrics
            first_metric = next(iter(self._val_metrics.items()))[0]

            if first_metric in metric_vals:
                val_score = metric_vals[first_metric].item()
            else:
                val_score = next(iter(metric_vals.values())).item()

        self._reset_metrics()
        return val_score, reset_old_score

    def _final_hook(self, final_results: Dict[str, Any], *args, **kwargs):
        wandb.run.summary.update(final_results)

    def _create_datasets(self):
        pass  # Not used for internal trainer

    def _create_loss(self) -> None:
        pass  # Not used for internal trainer

    def _create_metrics(self) -> None:
        pass  # Not used for internal trainer


@dataclass
class BatchInfo:
    mode: str
    epoch: int
    batch_idx: int
    batch_size: int
    total_batches: int
    overall_train_step: int

    @property
    def is_first_batch(self):
        return self.batch_idx == 0

    @property
    def is_last_batch(self):
        return self.batch_idx == self.total_batches - 1
