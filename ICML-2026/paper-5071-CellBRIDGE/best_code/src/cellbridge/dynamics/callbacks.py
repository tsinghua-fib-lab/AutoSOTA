"""
PyTorch Lightning callbacks for flow training with periodic sampling and evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from cellbridge.dynamics.sampling import sample_from_velocity_model
from cellbridge.utils.io import resolve_path

log = logging.getLogger(__name__)


class PeriodicSamplingCallback(pl.Callback):
    """
    PyTorch Lightning callback that periodically performs sampling and evaluation
    during training. This allows monitoring training progress via metrics computed
    on pushforward samples from the current model.
    """

    def __init__(
        self,
        eval_every_n_epochs: int = 25,
        sampling_steps: int = 100,
        t_final: float = 1.0,
        method: str = "rk4",
        sampling_mode: str = "ode",
        sigma_sample: float = 0.0,
        X_start: torch.Tensor | None = None,
        X_eval: torch.Tensor | None = None,
        metrics_cfg: DictConfig | None = None,
        subsample_size: int | None = None,
        device: str = "auto",
        log_prefix: str = "sampling",
    ):
        """
        Initialize the periodic sampling callback.

        Args:
            eval_every_n_epochs: Frequency of evaluation (every N training epochs)
            sampling_steps: Number of ODE integration steps for pushforward
            t_final: Final time for ODE integration (usually 1.0)
            method: Integration method for deterministic pushforward ("euler", "rk4")
            sampling_mode: "ode" for deterministic or "sde" for stochastic sampling
            sigma_sample: Diffusion scale used for stochastic sampling
            X_start: Starting distribution samples for pushforward
            X_eval: Target/evaluation distribution samples for metrics
            metrics_cfg: Configuration for metrics computation
            subsample_size: Optionally subsample data for faster evaluation
            device: Device for computation ("auto", "cuda", "cpu")
            log_prefix: Prefix for logged metric names
        """
        super().__init__()
        self.eval_every_n_epochs = eval_every_n_epochs
        self.sampling_steps = sampling_steps
        self.t_final = t_final
        self.method = method
        self.sampling_mode = sampling_mode
        self.sigma_sample = float(sigma_sample)
        self.subsample_size = subsample_size
        self.log_prefix = log_prefix

        self.X_start = X_start
        self.X_eval = X_eval

        self.metrics_cfg = metrics_cfg
        self.metric_collection = None
        if metrics_cfg is not None:
            self.metric_collection = instantiate(metrics_cfg)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.logger = log

    def set_data(
        self,
        X_start: torch.Tensor | np.ndarray,
        X_eval: torch.Tensor | np.ndarray,
    ):
        """
        Set the evaluation data for the callback.

        Args:
            X_start: Starting distribution samples
            X_eval: Target/evaluation distribution samples
        """
        if isinstance(X_start, np.ndarray):
            X_start = torch.from_numpy(X_start).float()
        if isinstance(X_eval, np.ndarray):
            X_eval = torch.from_numpy(X_eval).float()

        if self.subsample_size is not None:
            if X_start.shape[0] > self.subsample_size:
                indices = torch.randperm(X_start.shape[0])[: self.subsample_size]
                X_start = X_start[indices]
                self.logger.info(
                    f"Subsampled X_start from {X_start.shape[0]} to "
                    f"{self.subsample_size}"
                )

            if X_eval.shape[0] > self.subsample_size:
                indices = torch.randperm(X_eval.shape[0])[: self.subsample_size]
                X_eval = X_eval[indices]
                self.logger.info(
                    f"Subsampled X_eval from {X_eval.shape[0]} to {self.subsample_size}"
                )

        self.X_start = X_start.to(self.device)
        self.X_eval = X_eval.to(self.device)

        self.logger.info(
            f"Callback data set: X_start {self.X_start.shape}, "
            f"X_eval {self.X_eval.shape}"
        )

    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Called at the start of training to evaluate the initial model."""
        if self.X_start is None or self.X_eval is None:
            self.logger.warning(
                "Sampling callback data not set. Call set_data() first."
            )
            return

        if self.metric_collection is None:
            self.logger.warning("No metrics configured for sampling callback.")
            return

        self.logger.info("Running initial sampling evaluation at training start")

        try:
            metrics = self._evaluate_current_model(pl_module)
            self._log_metrics(trainer, metrics, 0)  # Log at epoch 0
        except Exception as e:
            self.logger.error(f"Error during initial sampling: {e}")

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Called after each training epoch."""
        if (trainer.current_epoch + 1) % self.eval_every_n_epochs != 0:
            return

        if self.X_start is None or self.X_eval is None:
            self.logger.warning(
                "Sampling callback data not set. Call set_data() first."
            )
            return

        if self.metric_collection is None:
            self.logger.warning("No metrics configured for sampling callback.")
            return

        self.logger.info(
            f"Running periodic sampling evaluation at epoch {trainer.current_epoch + 1}"
        )

        try:
            metrics = self._evaluate_current_model(pl_module)
            self._log_metrics(trainer, metrics, trainer.current_epoch + 1)
        except Exception as e:
            self.logger.error(f"Error during periodic sampling: {e}")

    def _evaluate_current_model(
        self, pl_module: pl.LightningModule
    ) -> dict[str, float]:
        """
        Evaluate the current model by performing pushforward sampling and
        computing metrics.

        Args:
            pl_module: The PyTorch Lightning module being trained

        Returns:
            Dictionary of computed metrics
        """
        was_training = pl_module.training
        pl_module.eval()
        self.logger.info("Evaluating current model via pushforward sampling")
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Sampling steps: {self.sampling_steps}")
        self.logger.info(f"t_final: {self.t_final}")
        self.logger.info(f"Sampling mode: {self.sampling_mode}")

        with torch.no_grad():
            # FlowMatchingLitModule exposes the velocity network as `.net`.
            velocity_model = getattr(pl_module, "net", pl_module)
            score_model = getattr(pl_module, "score_net", None)

            X_push = sample_from_velocity_model(
                velocity_model=velocity_model,
                X_start=self.X_start,  # type: ignore[arg-type]
                steps=self.sampling_steps,
                t_final=self.t_final,
                method=self.method,
                score_model=score_model,
                sampling_mode=self.sampling_mode,
                sigma_sample=self.sigma_sample,
            )

            metrics = self.metric_collection(  # type: ignore[operator]
                X_push.cpu().numpy(),
                self.X_eval.cpu().numpy(),  # type: ignore[union-attr]
            )

        if was_training:
            pl_module.train()

        combined_metrics = {}
        for key, value in metrics.items():
            combined_metrics[f"pushforward_{key}"] = value

        return combined_metrics

    def _log_metrics(
        self, trainer: pl.Trainer, metrics: dict[str, float], epoch: int
    ) -> None:
        """
        Log metrics to the trainer's logger(s).

        Args:
            trainer: PyTorch Lightning trainer
            metrics: Dictionary of metrics to log
            epoch: Current training epoch
        """
        for metric_name, metric_value in metrics.items():
            full_name = f"{self.log_prefix}/{metric_name}"
            trainer.logger.log_metrics({full_name: metric_value}, step=epoch)  # type: ignore[union-attr]

        self.logger.info(f"Logged {len(metrics)} sampling metrics at epoch {epoch}")


def create_sampling_callback_from_config(
    cfg: DictConfig,
    callback_cfg: DictConfig,
    root_path: Path | None = None,
) -> PeriodicSamplingCallback:
    """
    Factory function to create a PeriodicSamplingCallback from configuration.
    This function loads the evaluation data and sets up the callback.

    Args:
        cfg: Main configuration (used for data loading)
        callback_cfg: Callback-specific configuration
        root_path: Root path for resolving relative paths

    Returns:
        Configured PeriodicSamplingCallback
    """
    import anndata as ad

    logger = logging.getLogger(__name__)

    def _to_abs(p: Path | str) -> Path:
        return resolve_path(p, root_path)

    logger.info("Loading evaluation data for sampling callback")

    if cfg.inputs.mode == "from_h5ad":
        A = ad.read_h5ad(_to_abs(cfg.inputs.h5ad_path))
        X_start = A[A.obs[cfg.inputs.domain_key] == cfg.inputs.start_label].obsm[
            cfg.rep.obsm_key
        ]
        eval_label = callback_cfg.get("eval_label", None)
        if eval_label is None:
            eval_label = getattr(cfg.inputs, "eval_label", cfg.inputs.end_label)
        X_eval = A[A.obs[cfg.inputs.domain_key] == eval_label].obsm[cfg.rep.obsm_key]
    else:
        A_start = ad.read_h5ad(_to_abs(cfg.inputs.start_path))
        eval_path = callback_cfg.get("eval_path", None)
        if eval_path is None:
            eval_path = getattr(cfg.inputs, "eval_path", cfg.inputs.end_path)
        A_eval = ad.read_h5ad(_to_abs(eval_path))
        X_start = A_start.obsm[cfg.rep.obsm_key]
        X_eval = A_eval.obsm[cfg.rep.obsm_key]

    X_start = np.asarray(X_start, dtype=np.float32)
    X_eval = np.asarray(X_eval, dtype=np.float32)

    if cfg.inputs.get("rep_transform", None):
        rep_transform = instantiate(cfg.inputs.rep_transform)
        X_start = rep_transform(X_start)
        X_eval = rep_transform(X_eval)
        logger.info("Applied representation transform to callback data")

    sf2m_cfg = getattr(cfg, "sf2m", {})
    if hasattr(sf2m_cfg, "get"):
        default_sigma_sample = float(sf2m_cfg.get("sigma_sample", 0.0))
    else:
        default_sigma_sample = float(getattr(sf2m_cfg, "sigma_sample", 0.0))

    callback = PeriodicSamplingCallback(
        eval_every_n_epochs=callback_cfg.get("eval_every_n_epochs", 25),
        sampling_steps=callback_cfg.get("sampling_steps", 100),
        t_final=callback_cfg.get("t_final", getattr(cfg.inputs, "t_final", 1.0)),
        method=callback_cfg.get("method", "rk4"),
        sampling_mode=callback_cfg.get("sampling_mode", "ode"),
        sigma_sample=callback_cfg.get("sigma_sample", default_sigma_sample),
        metrics_cfg=callback_cfg.get("metrics", None),
        subsample_size=callback_cfg.get("subsample_size", None),
        device=callback_cfg.get("device", "auto"),
        log_prefix=callback_cfg.get("log_prefix", "sampling"),
    )

    callback.set_data(X_start, X_eval)

    logger.info(
        f"Created sampling callback with "
        f"eval_every_n_epochs={callback.eval_every_n_epochs}"
    )

    return callback
