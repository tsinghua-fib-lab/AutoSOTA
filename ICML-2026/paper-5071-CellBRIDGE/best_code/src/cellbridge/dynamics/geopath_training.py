from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Literal

import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

try:
    import wandb  # noqa: F401

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from cellbridge.data.flow_matching_loaders import (
    create_train_val_datasets,
    make_eval_dataloader,
    make_train_dataloader,
)
from cellbridge.dynamics.geopath import GeoPathBridge
from cellbridge.dynamics.metrics import MetricConfig, build_metric_evaluator

log = logging.getLogger(__name__)


class GeoPathLitModule(pl.LightningModule):
    """Lightning module for GeoPath bridge training."""

    def __init__(
        self,
        geopath_net: nn.Module,
        alpha: float,
        optim_cfg: DictConfig,
        metric_evaluator,
        t_sampler: Literal["uniform", "beta"] = "uniform",
        beta_a: float = 2.0,
        beta_b: float = 2.0,
    ) -> None:
        super().__init__()
        self.geopath_net = geopath_net
        self.bridge = GeoPathBridge(geopath_net, alpha=alpha, trainable=True)
        self.optim_cfg = optim_cfg
        self.metric_evaluator = metric_evaluator
        self.t_sampler = t_sampler
        self.beta_a = beta_a
        self.beta_b = beta_b

    def _sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.t_sampler == "uniform":
            return torch.rand(batch_size, device=device)
        dist = torch.distributions.Beta(self.beta_a, self.beta_b)
        return dist.sample((batch_size,)).to(device)

    def _apply_metric(self, vec: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        if self.metric_evaluator is None:
            return vec.pow(2).mean(dim=-1)
        xt_flat = xt.reshape(xt.shape[0], -1)
        return self.metric_evaluator(vec, xt_flat)

    def training_step(self, batch, batch_idx):
        """Compute one GeoPath training loss."""
        xi = batch["xi"].to(self.device)
        yj = batch["yj"].to(self.device)
        t = self._sample_t(xi.shape[0], xi.device)
        xt, vt = self.bridge.sample(xi, yj, t, detach_outputs=False)
        loss = self._apply_metric(vt, xt).mean()
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Compute one GeoPath validation loss."""
        xi = batch["xi"].to(self.device)
        yj = batch["yj"].to(self.device)
        t = self._sample_t(xi.shape[0], xi.device)
        xt, vt = self.bridge.sample(xi, yj, t, detach_outputs=False)
        loss = self._apply_metric(vt, xt).mean()
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure the GeoPath optimizer."""
        opt_type = self.optim_cfg.get("type", "adam").lower()
        lr = self.optim_cfg.get("lr", 1e-3)
        weight_decay = self.optim_cfg.get("weight_decay", 0.0)
        betas = tuple(self.optim_cfg.get("betas", [0.9, 0.999]))
        if opt_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.geopath_net.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
            )
        else:
            optimizer = torch.optim.Adam(
                self.geopath_net.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
            )
        return optimizer


class GeoPathTrainer:
    """Train and save a GeoPath bridge model."""

    def __init__(
        self,
        cfg: DictConfig,
        X=None,
        Y=None,
        G=None,
        logger: logging.Logger | None = None,
    ):
        self.cfg = cfg
        self.logger = logger or log
        try:
            self.root = Path(get_original_cwd())
        except Exception:
            self.root = Path.cwd()
        pl.seed_everything(int(cfg.seed))
        self.X = X
        self.Y = Y
        self.G = G

    def _to_abs(self, p: Path | str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (self.root / p)

    def run(self) -> Path:
        """Run GeoPath training and return the trained network."""
        self.logger.info("Config:\n%s", OmegaConf.to_yaml(self.cfg))

        metric_raw = OmegaConf.select(self.cfg, "metric", default={"type": "l2"})
        metric_cfg = MetricConfig(**OmegaConf.to_container(metric_raw, resolve=True))
        metric_evaluator = build_metric_evaluator(metric_cfg, self.X, self.Y)
        if metric_evaluator is None:
            self.logger.info("Metric: L2")
        else:
            self.logger.info("Metric: %s", metric_cfg.type)

        train_ds, val_ds = create_train_val_datasets(
            X=self.X, Y=self.Y, coupling=self.G, **self.cfg.dataset
        )
        steps_per_epoch = math.ceil(len(train_ds) / int(self.cfg.dataloader.batch_size))

        train_loader = make_train_dataloader(
            train_ds,
            steps_per_epoch=steps_per_epoch,
            **self.cfg.dataloader,
            pin_memory=True,
        )
        val_loader = make_eval_dataloader(
            val_ds, **self.cfg.dataloader, pin_memory=True
        )

        D = int(self.X.shape[1]) if self.X.ndim > 1 else 1
        geopath_net = instantiate(self.cfg.model)
        geopath_net = geopath_net(data_dim=D)
        module = GeoPathLitModule(
            geopath_net=geopath_net,
            alpha=self.cfg.alpha,
            optim_cfg=self.cfg.optimizer,
            metric_evaluator=metric_evaluator,
            t_sampler=self.cfg.get("t_sampler", "uniform"),
            beta_a=self.cfg.get("beta_a", 2.0),
            beta_b=self.cfg.get("beta_b", 2.0),
        )

        trainer = instantiate(self.cfg.trainer)

        # Log hyperparameters to wandb if wandb logging is enabled
        if trainer.logger and hasattr(trainer.logger, "experiment") and WANDB_AVAILABLE:
            try:
                # Convert config to a dictionary that wandb can handle
                config_dict = OmegaConf.to_container(self.cfg, resolve=True)
                trainer.logger.experiment.config.update(config_dict)
                self.logger.info("Logged configuration to wandb")
            except Exception as e:
                self.logger.warning(f"Failed to log config to wandb: {e}")
        elif not WANDB_AVAILABLE:
            self.logger.warning("wandb not available for config logging")

        trainer.fit(module, train_loader, val_loader)

        out_ckpt = self._to_abs(self.cfg.outputs.out_dir) / "geopath.ckpt"
        out_ckpt.parent.mkdir(parents=True, exist_ok=True)

        trainer.save_checkpoint(str(out_ckpt))
        self.logger.info("Saved GeoPath checkpoint to %s", out_ckpt)

        # Load the best model from the best checkpoint
        has_callback = (
            hasattr(trainer, "checkpoint_callback")
            and trainer.checkpoint_callback is not None
        )
        if has_callback:
            best_checkpoint = trainer.checkpoint_callback.best_model_path
            if best_checkpoint:
                self.logger.info("Loading best model from %s", best_checkpoint)
                checkpoint = torch.load(best_checkpoint, map_location=module.device)
                module.load_state_dict(checkpoint["state_dict"])

        return module.geopath_net
