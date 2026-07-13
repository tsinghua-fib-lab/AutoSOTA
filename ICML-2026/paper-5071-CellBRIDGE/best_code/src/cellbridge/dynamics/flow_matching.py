from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.utils import get_original_cwd, instantiate

from cellbridge.dynamics.metrics import MetricEvaluator

try:
    import wandb  # noqa: F401

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from omegaconf import DictConfig, OmegaConf

from cellbridge.data.flow_matching_loaders import (
    create_train_val_datasets,
    make_eval_dataloader,
    make_train_dataloader,
)
from cellbridge.dynamics.geopath import GeoPathBridge, load_or_train_geopath
from cellbridge.dynamics.metrics import MetricConfig, build_metric_evaluator
from cellbridge.dynamics.models import ScoreMLP, VelocityMLP
from cellbridge.utils.io import resolve_path

log = logging.getLogger(__name__)


class FlowTrainer:
    """Train a flow matching model from config."""

    def __init__(self, cfg: DictConfig, *, logger: logging.Logger | None = None):
        self.cfg = cfg
        self.logger = logger or log

        try:
            self.root = Path(get_original_cwd())
        except Exception:
            self.root = Path.cwd()

        pl.seed_everything(int(cfg.seed))

        self.X: np.ndarray | None = None
        self.Y: np.ndarray | None = None
        self.G: np.ndarray | None = None

    def run(self) -> Path:
        """Run training and return the checkpoint path."""
        self.logger.info("Config:\n%s", OmegaConf.to_yaml(self.cfg))
        self._load_XY()
        self._load_coupling()

        assert self.X is not None, "X is not loaded"
        assert self.Y is not None, "Y is not loaded"
        assert self.G is not None, "G is not loaded"

        metric_cfg_raw = OmegaConf.select(self.cfg.geopath, "metric")
        if metric_cfg_raw is None:
            metric_cfg = MetricConfig()
        else:
            metric_cfg = MetricConfig(
                **OmegaConf.to_container(metric_cfg_raw, resolve=True)
            )
        self.logger.info("Metric type requested: %s", metric_cfg.type)

        metric_evaluator = build_metric_evaluator(metric_cfg, self.X, self.Y)
        if metric_evaluator is None:
            self.logger.info("Using default L2 metric.")
        else:
            self.logger.info(
                "Metric evaluator (%s) initialized on %d samples",
                metric_cfg.type,
                self.X.shape[0] + self.Y.shape[0],
            )

        train_ds, val_ds = create_train_val_datasets(
            X=self.X, Y=self.Y, coupling=self.G, **self.cfg.dataset
        )

        training_mode = str(getattr(self.cfg, "training_mode", "cfm")).lower()
        if training_mode not in {"cfm", "sf2m"}:
            raise ValueError(
                f"Unsupported training_mode='{training_mode}'. "
                "Expected 'cfm' or 'sf2m'."
            )
        self.logger.info("Training mode: %s", training_mode)
        sf2m_cfg_raw = OmegaConf.select(self.cfg, "sf2m", default=None)
        if sf2m_cfg_raw is None:
            sf2m_cfg = {}
        else:
            sf2m_cfg = OmegaConf.to_container(sf2m_cfg_raw, resolve=True)

        steps_per_epoch = math.ceil(len(train_ds) / int(self.cfg.dataloader.batch_size))
        total_steps = steps_per_epoch * int(self.cfg.trainer.max_epochs)

        # warmup_steps: set on cfg (if present) for modules that read from cfg
        if OmegaConf.select(self.cfg, "lit.optim_cfg") is not None:
            self.cfg.lit.optim_cfg.warmup_steps = (
                total_steps // self.cfg.lit.optim_cfg.warmup_divisor
            )

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

        # model & lit module
        velocity_model = instantiate(self.cfg.model)(in_dim=D)
        score_model_cfg = OmegaConf.select(self.cfg, "score_model", default=None)
        score_model: ScoreMLP | VelocityMLP | None = None
        if score_model_cfg is None and training_mode == "sf2m":
            score_model_cfg = self.cfg.model
        if score_model_cfg is not None:
            score_model = instantiate(score_model_cfg)(in_dim=D)
        self._maybe_restore_weights(velocity_model, score_model)

        geopath_bridge: nn.Module | None = None
        geopath_opt_cfg = None
        geopath_cfg = OmegaConf.select(self.cfg, "geopath", default=None)
        if (
            training_mode == "sf2m"
            and geopath_cfg is not None
            and geopath_cfg.get("enabled", False)
        ):
            self.logger.warning(
                "GeoPath bridge is currently disabled in 'sf2m' training mode."
            )
            geopath_cfg = None

        if geopath_cfg is not None and geopath_cfg.get("enabled", False):
            if geopath_cfg.get("model", None) is None:
                raise ValueError("geopath.model must be provided when enabled.")
            self.logger.info(
                "GeoPath enabled (trainable=%s, alpha=%.3f)",
                geopath_cfg.get("train_inline", False)
                or geopath_cfg.get("trainable", False),
                geopath_cfg.get("alpha", 1.0),
            )

            # Load or train geopath network
            geopath_net = load_or_train_geopath(
                geopath_cfg=geopath_cfg,
                data_dim=D,
                artifacts_folder=self._to_abs(self.cfg.coupling_folder),
                X=self.X,
                Y=self.Y,
                G=self.G,
                logger=self.logger,
                save_trained_checkpoint=False,
            )

            geopath_bridge = GeoPathBridge(
                geopath_net,
                alpha=geopath_cfg.get("alpha", 1.0),
                trainable=False,
            )

        else:
            self.logger.info("GeoPath disabled.")

        flow_lit_factory = instantiate(self.cfg.lit)
        flow_lit = flow_lit_factory(
            velocity_model=velocity_model,
            score_model=score_model,
            total_steps=total_steps,
            geopath_bridge=geopath_bridge,
            metric_evaluator=metric_evaluator,
            geopath_opt_cfg=geopath_opt_cfg,
            training_mode=training_mode,
            sigma_bridge=float(sf2m_cfg.get("sigma_bridge", 1.0)),
            sigma_sample=float(sf2m_cfg.get("sigma_sample", 1.0)),
            sf2m_t_epsilon=float(sf2m_cfg.get("t_epsilon", 0)),
        )

        # trainer (Hydra-configured: callbacks, loggers, etc.)
        trainer = instantiate(self.cfg.trainer)

        # Log configuration to wandb if wandb logging is enabled
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

        # Add periodic sampling callback if configured
        if OmegaConf.select(self.cfg, "periodic_sampling.enabled", default=False):
            self.logger.info("Adding periodic sampling callback")
            try:
                from cellbridge.dynamics.callbacks import (
                    create_sampling_callback_from_config,
                )

                sampling_callback = create_sampling_callback_from_config(
                    cfg=self.cfg,
                    callback_cfg=self.cfg.periodic_sampling,
                    root_path=self.root,
                )
                trainer.callbacks.append(sampling_callback)
                self.logger.info("Periodic sampling callback added successfully")
            except Exception as e:
                self.logger.warning(f"Failed to add sampling callback: {e}")

        # resume training (true resume with optimizer state)
        resume_ckpt = OmegaConf.select(self.cfg, "resume.ckpt")
        resume_enabled = bool(
            OmegaConf.select(self.cfg, "resume.enabled", default=False)
        )
        ckpt_path = resume_ckpt if resume_enabled and resume_ckpt else None

        self.logger.info(
            "Start training: steps/epoch=%d total_steps=%d resume=%s",
            steps_per_epoch,
            total_steps,
            bool(ckpt_path),
        )
        trainer.fit(
            flow_lit,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=ckpt_path,
        )

        # final checkpoint path (in Hydra run dir)
        out_ckpt = self._get_run_dir() / "model.ckpt"
        trainer.save_checkpoint(out_ckpt)
        self.logger.info("Saved final checkpoint to %s", out_ckpt)
        return out_ckpt

    def _to_abs(self, p: Path | str) -> Path:
        return resolve_path(p, self.root)

    def _load_XY(self) -> None:
        cfg = self.cfg
        if cfg.inputs.mode == "from_h5ad":
            A = ad.read_h5ad(self._to_abs(cfg.inputs.h5ad_path))
            X = A[A.obs[cfg.inputs.domain_key] == cfg.inputs.start_label].obsm[
                cfg.rep.obsm_key
            ]
            Y = A[A.obs[cfg.inputs.domain_key] == cfg.inputs.end_label].obsm[
                cfg.rep.obsm_key
            ]
        else:
            Ax = ad.read_h5ad(self._to_abs(cfg.inputs.start_path))
            Ay = ad.read_h5ad(self._to_abs(cfg.inputs.end_path))
            X = Ax.obsm[cfg.rep.obsm_key]
            Y = Ay.obsm[cfg.rep.obsm_key]
        self.X = np.asarray(X, dtype=np.float32)
        self.Y = np.asarray(Y, dtype=np.float32)

        # Apply representation transform if specified
        if cfg.inputs.get("rep_transform", None):
            rep_transform = instantiate(cfg.inputs.rep_transform)
            self.X = rep_transform(self.X)
            self.Y = rep_transform(self.Y)
            self.logger.info("Applied representation transform to training data")

        self.logger.info("Loaded X %s, Y %s", self.X.shape, self.Y.shape)  # type: ignore[union-attr]

    def _load_coupling(self) -> None:
        # prefer explicit path; else folder/coupling.npy
        path = OmegaConf.select(self.cfg, "inputs.coupling_path")
        if path is None:
            folder = self._to_abs(self.cfg.coupling_folder)
            path = folder / "coupling.npy"
        else:
            path = self._to_abs(path)
        self.G = np.load(path)
        assert self.G is not None, "G is not loaded"
        self.logger.info("Loaded coupling %s", self.G.shape)

    def _maybe_restore_weights(
        self, velocity_model: VelocityMLP, score_model: ScoreMLP | VelocityMLP | None
    ) -> None:
        # "restore" = initialize weights from a checkpoint (read-only), not resume
        enabled = bool(OmegaConf.select(self.cfg, "restore.enabled", default=True))
        ckpt = OmegaConf.select(self.cfg, "restore.ckpt")
        if not (enabled and ckpt):
            return
        ckpt = str(self._to_abs(ckpt))
        self.logger.info(
            "Restoring weights from %s (strict=%s)",
            ckpt,
            OmegaConf.select(self.cfg, "restore.strict", default=False),
        )
        try:
            lit = FlowMatchingLitModule.load_from_checkpoint(
                ckpt,
                strict=bool(
                    OmegaConf.select(self.cfg, "restore.strict", default=False)
                ),
            )
            # best-effort: copy underlying net weights
            velocity_model.load_state_dict(lit.net.state_dict(), strict=False)
            if score_model is not None and hasattr(lit, "score_net"):
                lit_score = lit.score_net
                if lit_score is not None:
                    score_model.load_state_dict(lit_score.state_dict(), strict=False)
        except Exception as e:
            self.logger.warning("Restore failed (%s). Continuing with fresh init.", e)

    def _get_run_dir(self) -> Path:
        # Prefer Hydra output dir if available
        try:
            from hydra.core.hydra_config import HydraConfig

            return Path(HydraConfig.get().runtime.output_dir)
        except Exception:
            return Path.cwd()


class WarmupCosine(optim.lr_scheduler._LRScheduler):
    """Linear warmup followed by cosine decay."""

    def __init__(self, optimizer, warmup_steps: int, max_steps: int):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self._step = 0
        super().__init__(optimizer)

    def get_lr(self):
        """Return learning rates for the current scheduler step."""
        self._step += 1
        lrs = []
        for base_lr in self.base_lrs:
            if self._step <= self.warmup_steps:
                lrs.append(base_lr * self._step / max(1, self.warmup_steps))
            else:
                progress = (self._step - self.warmup_steps) / max(
                    1, self.max_steps - self.warmup_steps
                )
                lrs.append(
                    0.5
                    * base_lr
                    * (1 + torch.cos(torch.tensor(math.pi * progress))).item()
                )
        return lrs


class FlowMatchingLitModule(pl.LightningModule):
    """Lightning module for CFM and SF2M training."""

    def __init__(
        self,
        velocity_model: VelocityMLP,
        optim_cfg: DictConfig,
        total_steps: int,
        val_t_samples: int = 16,
        val_t_sampler: Literal["uniform", "beta"] = "uniform",
        val_beta_a: float = 2.0,
        val_beta_b: float = 2.0,
        score_model: VelocityMLP | None = None,
        geopath_bridge: nn.Module | None = None,
        metric_evaluator: MetricEvaluator | None = None,
        geopath_opt_cfg: dict | None = None,
        training_mode: str = "cfm",
        sigma_bridge: float = 1.0,
        sigma_sample: float = 1.0,
        sf2m_t_epsilon: float = 1e-3,
    ) -> None:
        super().__init__()
        # store hyper-parameters but skip heavy modules
        self.save_hyperparameters()
        self.velocity_net = velocity_model
        self.net = self.velocity_net  # Backwards compatibility
        self.score_net = score_model
        self.total_steps = int(total_steps)  # used by the LR scheduler
        self.val_t_samples = int(max(1, val_t_samples))
        self.val_t_sampler = val_t_sampler
        self.val_beta_a = float(val_beta_a)
        self.val_beta_b = float(val_beta_b)
        self.geopath_bridge = geopath_bridge
        self.metric_evaluator = metric_evaluator
        self.geopath_opt_cfg = geopath_opt_cfg or {}
        if self.metric_evaluator is not None:
            self.log("metric/enabled", 1.0)
        if self.geopath_opt_cfg:
            self.log("geopath/optimizer_enabled", 1.0)
        if self.metric_evaluator is not None:
            self.metric_evaluator.eval()
            for p in self.metric_evaluator.parameters():
                p.requires_grad = False

        # for Lightning's example tracing
        self.example_input_array = (
            torch.randn(8, getattr(self.velocity_net, "in_dim", 8)),
            torch.rand(8),
        )

        # running accumulator for exact validation over edges
        self._val_weighted_sum = 0

        self.optim_cfg = optim_cfg
        self.training_mode = str(training_mode).lower()
        if self.training_mode not in {"cfm", "sf2m"}:
            raise ValueError(
                f"Unsupported training mode '{self.training_mode}' in Lightning module."
            )
        if self.training_mode == "sf2m" and self.score_net is None:
            raise ValueError("score_model must be provided for 'sf2m' training mode.")
        self.sigma_bridge = float(sigma_bridge)
        self.sigma_sample = float(sigma_sample)
        self._sf2m_eps = 1e-8
        self.sf2m_t_epsilon = float(sf2m_t_epsilon)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity for positions and times."""
        return self.velocity_net(x, t)

    @staticmethod
    def _per_example_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # MSE averaged over feature dim, per example
        return ((pred - target) ** 2).mean(dim=-1)

    def training_step(self, batch, batch_idx: int):
        """Compute one flow matching training step."""
        t: torch.Tensor = batch["t"].to(self.device).view(-1)
        if self.training_mode == "sf2m":
            eps = self.sf2m_t_epsilon
            t = t.clamp(eps, 1.0 - eps)
        if self.training_mode == "sf2m":
            assert self.score_net is not None, "Score network required for sf2m mode."
            xi: torch.Tensor = batch["xi"].to(self.device)
            yj: torch.Tensor = batch["yj"].to(self.device)
            xt, u_flow, score_target, lambda_t, eps = self._sample_sf2m_conditionals(
                xi, yj, t
            )
            v_pred = self.velocity_net(xt, t)
            s_pred = self.score_net(xt, t)
            flow_mse = (v_pred - u_flow).pow(2).mean(dim=-1)
            score_mse = (s_pred - score_target).pow(2).mean(dim=-1)
            score_residual = lambda_t * s_pred + eps
            score_loss = score_residual.pow(2).mean(dim=-1)
            flow_loss = flow_mse
            loss = (flow_loss + score_loss).mean()
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/flow_loss", flow_loss.mean(), on_step=True, prog_bar=False)
            self.log(
                "train/score_loss", score_loss.mean(), on_step=True, prog_bar=False
            )
            with torch.no_grad():
                self.log(
                    "train/velocity_norm",
                    v_pred.norm(dim=-1).mean(),
                    on_step=True,
                    prog_bar=False,
                )
                self.log(
                    "train/score_norm",
                    s_pred.norm(dim=-1).mean(),
                    on_step=True,
                    prog_bar=False,
                )
                self.log(
                    "train/flow_mse",
                    flow_mse.mean(),
                    on_step=True,
                    prog_bar=False,
                )
                self.log(
                    "train/score_mse",
                    score_mse.mean(),
                    on_step=True,
                    prog_bar=False,
                )
                self.log(
                    "train/target_flow_norm",
                    u_flow.norm(dim=-1).mean(),
                    on_step=True,
                    prog_bar=False,
                )
                self.log(
                    "train/target_score_norm",
                    score_target.norm(dim=-1).mean(),
                    on_step=True,
                    prog_bar=False,
                )
            return loss

        # Deterministic CFM path
        geopath_trainable = self.geopath_bridge is not None and getattr(
            self.geopath_bridge, "trainable", False
        )

        if self.geopath_bridge is not None:
            detach = not geopath_trainable
            xi: torch.Tensor = batch["xi"].to(self.device)
            yj: torch.Tensor = batch["yj"].to(self.device)
            x_t, v_star = self.geopath_bridge.sample(xi, yj, t, detach_outputs=detach)
            if geopath_trainable:
                self.log("train/geopath_active", 1.0, on_step=True, prog_bar=False)
        else:
            x_t = batch["x_t"].to(self.device)
            v_star = batch["v"].to(self.device)

        pred = self.velocity_net(x_t, t)
        residual = pred - v_star
        per = self._apply_metric(residual, x_t)

        loss = per.mean()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        with torch.no_grad():
            self.log(
                "train/pred_norm",
                pred.norm(dim=-1).mean(),
                on_step=True,
                prog_bar=False,
            )
            self.log(
                "train/tgt_norm",
                v_star.norm(dim=-1).mean(),
                on_step=True,
                prog_bar=False,
            )
        return loss

    def _sample_sf2m_conditionals(
        self,
        xi: torch.Tensor,
        yj: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        t_col = t.view(-1, 1)
        one_minus_t = 1.0 - t_col
        sigma_sq = self.sigma_bridge**2
        mu_t = one_minus_t * xi + t_col * yj
        var_t = sigma_sq * t_col * one_minus_t
        if noise is None:
            noise = torch.randn_like(xi)
        std_t = torch.sqrt(var_t + self._sf2m_eps)
        xt = mu_t + std_t * noise
        denom = t_col * one_minus_t + self._sf2m_eps
        coeff = (1.0 - 2.0 * t_col) / denom
        u_flow = coeff * (xt - mu_t) + (yj - xi)
        score_target = -noise / (std_t + self._sf2m_eps)
        lambda_t = (
            2.0
            * torch.sqrt(denom + self._sf2m_eps)
            / (self.sigma_bridge + self._sf2m_eps)
        )
        return xt, u_flow, score_target, lambda_t, noise

    def on_validation_epoch_start(self) -> None:
        """Reset validation loss accumulator."""
        self._val_weighted_sum = 0.0

    def _sample_t(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        """Sample t either from a uniform distribution or a beta distribution"""
        if self.val_t_sampler == "uniform":
            t = torch.rand(shape, device=device)
        else:
            dist = torch.distributions.Beta(self.val_beta_a, self.val_beta_b)
            t = dist.sample(shape).to(device)
        if self.training_mode == "sf2m":
            eps = self.sf2m_t_epsilon
            return t.clamp(eps, 1.0 - eps)
        return t

    def validation_step(self, batch, batch_idx: int):
        """Accumulate weighted validation loss."""
        xi: torch.Tensor = batch["xi"].to(self.device)  # (B, D)
        yj: torch.Tensor = batch["yj"].to(self.device)  # (B, D)
        p: torch.Tensor = (
            batch["p"].to(self.device).float()
        )  # (B,), sums to 1 over entire val set (not per batch)

        if self.training_mode == "sf2m":
            assert self.score_net is not None, "Score network required for sf2m mode."
            B, D = xi.shape
            T = self.val_t_samples
            t = self._sample_t(torch.Size([T, B, 1]), device=xi.device)
            xi_expanded = xi.unsqueeze(0).expand(T, B, D).reshape(T * B, D)
            yj_expanded = yj.unsqueeze(0).expand(T, B, D).reshape(T * B, D)
            t_flat = t.reshape(T * B)
            xt, u_flow, score_target, lambda_t, eps = self._sample_sf2m_conditionals(
                xi_expanded, yj_expanded, t_flat, noise=torch.randn_like(xi_expanded)
            )
            v_pred = self.velocity_net(xt, t_flat)
            s_pred = self.score_net(xt, t_flat)
            flow_loss = (v_pred - u_flow).pow(2).mean(dim=-1)
            score_loss = (lambda_t * s_pred + eps).pow(2).mean(dim=-1)
            per_pair = (flow_loss + score_loss).reshape(T, B).mean(dim=0)
            contrib = (p * per_pair).sum()
            self._val_weighted_sum += float(contrib.detach().cpu().item())
            self.log(
                "val/batch_contrib",
                contrib,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            return contrib

        B, D = xi.shape
        # Monte-Carlo average over t per pair
        T = self.val_t_samples
        # (T, B, 1)
        t = self._sample_t(torch.Size([T, B, 1]), device=xi.device)
        # (T, B, D)
        xi_expanded = xi.unsqueeze(0).expand(T, B, D).reshape(T * B, D)
        yj_expanded = yj.unsqueeze(0).expand(T, B, D).reshape(T * B, D)
        t_flat = t.reshape(T * B)
        if self.geopath_bridge is not None:
            detach = not getattr(self.geopath_bridge, "trainable", False)
            xt_corr, v_corr = self.geopath_bridge.sample(
                xi_expanded, yj_expanded, t_flat, detach_outputs=detach
            )
            xt = xt_corr.reshape(T, B, D)
            v_star = v_corr.reshape(T, B, D)
        else:
            xt = (1.0 - t) * xi.unsqueeze(0) + t * yj.unsqueeze(0)
            v_star = (yj - xi).unsqueeze(0).expand(T, B, D)
        # run model on (T*B) then reshape
        pred = self.velocity_net(xt.reshape(T * B, D), t.reshape(T * B)).reshape(
            T, B, D
        )
        residual = pred - v_star
        per = self._apply_metric(residual, xt)  # (T, B)
        per_pair = per.mean(dim=0)  # (B,)  ≈ E_t[MSE]
        contrib = (
            p * per_pair
        ).sum()  # Multiply by the importance weight p (which corresponds to the
        # coupling)

        # Accumulate exact sum over edges across all batches.
        # Use CPU float to avoid any potential DDP sync overhead.
        self._val_weighted_sum += float(contrib.detach().cpu().item())
        self.log(
            "val/batch_contrib", contrib, on_step=False, on_epoch=True, prog_bar=False
        )

        # Return contrib if you want Lightning to aggregate externally as well
        return contrib

    def on_validation_epoch_end(self) -> None:
        """Log accumulated validation loss."""
        val_loss = torch.tensor(self._val_weighted_sum, device=self.device)
        self.log("val/loss", val_loss, prog_bar=True)

    def _apply_metric(self, residual: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        if self.metric_evaluator is None:
            return residual.pow(2).mean(dim=-1)
        if residual.ndim == 3:
            T, B, D = residual.shape
            residual_flat = residual.reshape(T * B, D)
            xt_flat = xt.reshape(T * B, -1)
            per = self.metric_evaluator(residual_flat, xt_flat)
            return per.view(T, B)
        xt_flat = xt.reshape(xt.shape[0], -1)
        return self.metric_evaluator(residual, xt_flat)

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        flow_params = list(self.velocity_net.parameters())
        if self.score_net is not None and self.training_mode == "sf2m":
            flow_params += list(self.score_net.parameters())
        flow_opt = optim.AdamW(
            flow_params,
            lr=self.optim_cfg.lr,
            weight_decay=self.optim_cfg.wd,
            betas=self.optim_cfg.betas,
        )

        optimizers = [flow_opt]
        schedulers = []

        if self.optim_cfg.scheduler != "none":
            if self.optim_cfg.scheduler == "cosine":
                sched = WarmupCosine(
                    flow_opt,
                    warmup_steps=self.optim_cfg.warmup_steps,
                    max_steps=self.total_steps,
                )
            else:
                sched = optim.lr_scheduler.StepLR(
                    flow_opt,
                    step_size=self.optim_cfg.step_size,
                    gamma=self.optim_cfg.gamma,
                )
            schedulers.append({"scheduler": sched, "interval": "step"})

        geopath_opt = self._build_geopath_optimizer()
        if geopath_opt is not None:
            optimizers.append(geopath_opt)

        if not schedulers:
            if len(optimizers) == 1:
                return optimizers[0]
            return optimizers
        if len(optimizers) == 1:
            return {"optimizer": optimizers[0], "lr_scheduler": schedulers[0]}
        return optimizers, schedulers

    def _build_geopath_optimizer(self):
        if (
            self.geopath_bridge is None
            or not self.geopath_opt_cfg
            or not getattr(self.geopath_bridge, "trainable", False)
        ):
            return None
        params = [p for p in self.geopath_bridge.parameters() if p.requires_grad]
        if not params:
            return None
        opt_type = self.geopath_opt_cfg.get("type", "adam").lower()
        lr = self.geopath_opt_cfg.get("lr", 1e-3)
        weight_decay = self.geopath_opt_cfg.get("weight_decay", 0.0)
        betas = tuple(self.geopath_opt_cfg.get("betas", [0.9, 0.999]))
        if opt_type == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
        return optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas)
