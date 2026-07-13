"""Simulation model trainers (NPE, FMPE).

These models are trained once per task on simulation data and serve
as the base posterior estimators for calibration methods.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from lampe.utils import GDStep
from torch import Tensor, nn

from posteriors import DirectFlowMatchingPosterior, DirectPosteriorEstimator, Estimator
from flow_matching.torch_flow import FlowMatching
from training.base import BaseTrainer, TrainerConfig
from training.registry import register_trainer


@register_trainer("npe", category="simulation")
class NPETrainer(BaseTrainer):
    """Trainer for Neural Posterior Estimation (NPE).

    NPE learns p(theta|x) directly from simulation data using
    normalizing flows.
    """

    def __init__(
        self,
        config: TrainerConfig,
        model_path: Path,
        task_config: Dict[str, Any],
        logger: Optional[Any] = None,
    ):
        super().__init__(config, model_path, task_config, logger)

        # Extract NPE-specific config
        npe_config = task_config.get("npe", {})
        self.npe_params = npe_config.get("params", {})
        self.task_name = task_config.get("name", "unknown")

        # Shapes will be set during build_model
        self._theta_shape = None
        self._x_shape = None

    def build_model(
        self,
        theta_shape: Tuple[int, ...] = None,
        x_shape: Tuple[int, ...] = None,
        **kwargs
    ) -> Estimator:
        """Build NPE estimator."""
        self._theta_shape = theta_shape
        self._x_shape = x_shape

        estimator = Estimator(
            self.task_name,
            theta_shape,
            x_shape,
            **self.npe_params,
        )
        return estimator

    def train(
        self,
        data: Tuple[Tensor, Tensor],
        device: torch.device,
        logname: str = "",
        trial: Optional[Any] = None,
        **kwargs,
    ) -> DirectPosteriorEstimator:
        """Train NPE model.

        Args:
            data: (theta, x) training data
            device: Compute device
            logname: Name for logging
            trial: Optional Optuna trial for HPO pruning support

        Returns:
            Trained DirectPosteriorEstimator
        """
        from tqdm.auto import trange

        from training.base import _OptunaPruned

        theta, x = data

        # Build model
        model = self.build_model(theta_shape=theta.shape[1:], x_shape=x.shape[1:])
        model.set_scales(theta, x, self.config.rescale)
        model.to(device)

        # Check for existing checkpoint
        if self.config.load:
            loaded = self._try_load_checkpoint(model, device)
            if loaded:
                return self.create_posterior(model)

        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        step = GDStep(optimizer, clip=5.0)

        train_loader, val_loader = self.prepare_data(data)

        self._print_training_config()

        try:
            with self.timer.time_operation("training", "npe"):
                for epoch in (pbar := trange(self.config.epochs, desc="Training NPE")):
                    # Training
                    model.train()
                    train_losses = []
                    for thb, xb in train_loader:
                        loss = model.compute_loss(thb.to(device), xb.to(device))
                        train_losses.append(step(loss).item())

                    # Validation
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for thb, xb in val_loader:
                            val_losses.append(model.compute_loss(thb.to(device), xb.to(device)).item())

                    train_loss = sum(train_losses) / len(train_losses)
                    val_loss = sum(val_losses) / len(val_losses)
                    pbar.set_postfix(train=train_loss, val=val_loss)

                    # Update state tracking
                    improved = self.state.update(train_loss, val_loss)
                    if improved:
                        self.state.best_state_dict = deepcopy(model.state_dict())

                    # Logging
                    if self.logger is not None:
                        self.logger.log_metrics(
                            {"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch},
                            log_name=logname,
                        )

                    # Early stopping
                    if self.state.should_stop(self.config.max_patience):
                        pbar.write(f"\tEarly stopping at epoch {epoch + 1}")
                        pbar.write(f"\tBest validation loss: {self.state.best_val_loss:.4f}")
                        break

                    # Optuna pruning support
                    if trial is not None:
                        trial.report(val_loss, epoch)
                        if trial.should_prune():
                            pbar.write(f"\tOptuna pruned at epoch {epoch + 1}")
                            raise _OptunaPruned()

        except _OptunaPruned:
            import optuna

            raise optuna.TrialPruned()

        # Restore best model
        if self.state.best_state_dict is not None:
            model.load_state_dict(self.state.best_state_dict, strict=False)

        self.state.mark_completed()
        model.cpu()

        # Save
        if self.config.save:
            self._save_checkpoint(model, logname)

        return self.create_posterior(model)

    def create_posterior(self, model: Estimator) -> DirectPosteriorEstimator:
        """Wrap model in DirectPosteriorEstimator."""
        return DirectPosteriorEstimator("npe", model)


@register_trainer("fmpe", category="simulation")
class FMPETrainer(BaseTrainer):
    """Trainer for Flow Matching Posterior Estimation (FMPE).

    FMPE uses flow matching to learn p(theta|x) from simulation data.
    """

    def __init__(
        self,
        config: TrainerConfig,
        model_path: Path,
        task_config: Dict[str, Any],
        logger: Optional[Any] = None,
    ):
        super().__init__(config, model_path, task_config, logger)

        # Extract FMPE-specific config
        fmpe_config = task_config.get("fmpe", {})
        self.fmpe_model_config = fmpe_config.get("config", {})

    def build_model(
        self,
        theta_shape: Tuple[int, ...] = None,
        x_shape: Tuple[int, ...] = None,
        **kwargs
    ) -> FlowMatching:
        """Build FMPE flow matching model."""
        cfg = self.fmpe_model_config

        model = FlowMatching(
            cfg.get("probability_path", "ot2"),
            cfg.get("prior", "power"),
            cfg.get("base_dist", "gaussian"),
            theta_shape,
            cond_dim=x_shape,
            **cfg.get("params", {}),
        )
        return model

    def train(
        self,
        data: Tuple[Tensor, Tensor],
        device: torch.device,
        logname: str = "",
        trial: Optional[Any] = None,
        **kwargs,
    ) -> DirectFlowMatchingPosterior:
        """Train FMPE model.

        Args:
            data: (theta, x) training data
            device: Compute device
            logname: Name for logging
            trial: Optional Optuna trial for HPO pruning support

        Returns:
            Trained DirectFlowMatchingPosterior
        """
        from tqdm.auto import trange

        from training.base import _OptunaPruned

        theta, x = data

        # Build model
        model = self.build_model(theta_shape=theta.shape[1:], x_shape=x.shape[1:])
        model.set_scales(theta, x, self.config.rescale)
        model.to(device)

        # Check for existing checkpoint
        if self.config.load:
            loaded = self._try_load_checkpoint(model, device)
            if loaded:
                return self.create_posterior(model)

        # Setup training
        train_loader, val_loader = self.prepare_data(data)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.epochs)

        self._print_training_config()

        try:
            with self.timer.time_operation("training", "fmpe"):
                for epoch in (pbar := trange(self.config.epochs, desc="Training FMPE")):
                    # Training
                    model.train()
                    train_losses = []
                    for target, cond in train_loader:
                        optimizer.zero_grad()
                        loss = model.compute_loss(target.to(device), cond.to(device))
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        train_losses.append(loss.item())

                    scheduler.step()

                    # Validation
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for target, cond in val_loader:
                            loss = model.compute_loss(target.to(device), cond.to(device))
                            val_losses.append(loss.item())

                    train_loss = sum(train_losses) / len(train_losses)
                    val_loss = sum(val_losses) / len(val_losses)
                    pbar.set_postfix(train=train_loss, val=val_loss)

                    # Update state tracking
                    improved = self.state.update(train_loss, val_loss)
                    if improved:
                        self.state.best_state_dict = deepcopy(model.state_dict())

                    # Logging
                    if self.logger is not None:
                        self.logger.log_metrics(
                            {"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch},
                            log_name=logname,
                        )

                    # Early stopping
                    if self.state.should_stop(self.config.max_patience):
                        pbar.write(f"\tEarly stopping at epoch {epoch + 1}")
                        pbar.write(f"\tBest validation loss: {self.state.best_val_loss:.4f}")
                        break

                    # Optuna pruning support
                    if trial is not None:
                        trial.report(val_loss, epoch)
                        if trial.should_prune():
                            pbar.write(f"\tOptuna pruned at epoch {epoch + 1}")
                            raise _OptunaPruned()

        except _OptunaPruned:
            import optuna

            raise optuna.TrialPruned()

        # Restore best model
        if self.state.best_state_dict is not None:
            model.load_state_dict(self.state.best_state_dict, strict=False)

        self.state.mark_completed()
        model.cpu()

        # Save
        if self.config.save:
            self._save_checkpoint(model, logname)

        return self.create_posterior(model)

    def create_posterior(self, model: FlowMatching) -> DirectFlowMatchingPosterior:
        """Wrap model in DirectFlowMatchingPosterior."""
        return DirectFlowMatchingPosterior("fmpe", model)
