"""Calibration method trainers (fm_post_transform, etc.).

These methods use calibration data to improve posterior estimation
under model misspecification.
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from tqdm.auto import trange

from posteriors import (
    DirectFlowMatchingPosterior,
    DirectPosteriorEstimator,
    DualFlowPosteriorEstimator,
    Estimator,
    FlowMatchingPosterior,
)
from flow_matching.torch_flow import FlowMatching
from simulator import get_simulator
from training.base import BaseTrainer, TrainerConfig
from training.registry import register_trainer
from utils.misc import train_val_split_n


def _create_flow_from_config(cfg: Dict, dim: torch.Size, cond_dim: torch.Size) -> FlowMatching:
    """Create a FlowMatching model from a config dict."""
    return FlowMatching(
        cfg.get("probability_path", "ot2"),
        cfg.get("prior", "power"),
        cfg.get("base_dist", "gaussian"),
        dim,
        cond_dim=cond_dim,
        **cfg.get("params", {}),
    )



@register_trainer("fm_post_transform", category="calibration")
class FMPostTransformTrainer(BaseTrainer):
    """Trainer for Flow Matching Post Transform.

    Jointly trains flow matching models for theta and x conditioned on y.
    """

    def __init__(
        self,
        config: TrainerConfig,
        model_path: Path,
        task_config: Dict[str, Any],
        logger: Optional[Any] = None,
    ):
        super().__init__(config, model_path, task_config, logger)

        method_config = task_config.get("fm_post_transform", {})
        self.flow_config = method_config.get("config", {})
        self.npe_type = self.flow_config.get("npe", "fmpe")
        training_params = method_config.get("training_params", {})
        self.velocity_reg = float(training_params.get("velocity_reg", 0.0))

    def build_model(self, **kwargs) -> Tuple[FlowMatching, FlowMatching]:
        """Build theta and x flow models."""
        theta_shape = kwargs.get("theta_shape")
        x_shape = kwargs.get("x_shape")
        y_shape = kwargs.get("y_shape")

        flow_theta_cfg = self.flow_config.get("flow_theta", {})
        flow_x_cfg = self.flow_config.get("flow_x", {})

        flow_theta = _create_flow_from_config(flow_theta_cfg, theta_shape, y_shape)

        flow_x = _create_flow_from_config(flow_x_cfg, x_shape, y_shape)

        return flow_theta, flow_x

    def _load_npe(self, device: torch.device):
        """Load base NPE/FMPE model."""
        if self.npe_type == "fmpe":
            config_path = self.model_path / "npe_fmpe.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config_npe = json.load(f)
            else:
                config_npe = self.task_config.get("fmpe", {}).get("config", {})

            # Need to get theta and x shapes from data
            flow_matching = _create_flow_from_config(config_npe, self._theta_shape, self._x_shape)

            weights_path = self.model_path / "npe_fmpe.pth"
            if weights_path.exists():
                flow_matching.load_state_dict(
                    torch.load(weights_path, map_location=device, weights_only=True),
                    strict=False,
                )

            flow_matching.rescale_name = self.config.rescale
            return DirectFlowMatchingPosterior("npe_fmpe", flow_matching)
        else:
            npe_path = self.model_path / "npe.pkl"
            with open(npe_path, "rb") as f:
                npe = pickle.load(f)
            return DirectPosteriorEstimator("npe", npe)

    def _load_simulator(self):
        """Load simulator for generating training data."""
        sim_path = self.model_path / "simulator.json"
        if sim_path.exists():
            with open(sim_path, "r") as f:
                config = json.load(f)
            return get_simulator(config["name"], **config["params"])
        return None

    def train(
        self,
        data: Tuple[Tensor, Tensor, Tensor],
        device: torch.device,
        logname: str = "",
        **kwargs,
    ) -> DualFlowPosteriorEstimator:
        """Train fm_post_transform."""
        theta, x, y = data
        self._theta_shape = theta.shape[1:]
        self._x_shape = x.shape[1:]

        # Load NPE
        npe = self._load_npe(device)
        npe.to(device)

        # Load simulator for data augmentation
        simulator = self._load_simulator()
        simulate = None
        if simulator is not None:
            try:
                simulate = simulator.get_simulator(misspecified=True)
            except NotImplementedError:
                pass

        # Build flow models
        flow_theta, flow_x = self.build_model(
            theta_shape=theta.shape[1:],
            x_shape=x.shape[1:],
            y_shape=y.shape[1:],
        )

        # Check for existing checkpoints
        if self.config.load:
            theta_path = self.model_path / f"{logname}_theta.pth"
            x_path = self.model_path / f"{logname}_x.pth"
            if theta_path.exists() and x_path.exists():
                flow_theta.load_state_dict(torch.load(theta_path, map_location=device, weights_only=True))
                flow_x.load_state_dict(torch.load(x_path, map_location=device, weights_only=True))
                return DualFlowPosteriorEstimator(
                    "fm_post_transform",
                    base_dist=npe,
                    flow_theta=flow_theta,
                    flow_x=flow_x,
                )

        flow_theta.to(device)
        flow_x.to(device)

        # Set scales
        flow_theta.set_scales(theta, y, self.config.rescale)
        flow_x.set_scales(x, y, self.config.rescale)

        self._print_training_config()

        # Training setup
        train_loader, val_loader = train_val_split_n(
            theta, x, y, train_size=self.config.train_size, batch_size=self.config.batch_size
        )

        optimizer = torch.optim.Adam(
            list(flow_theta.parameters()) + list(flow_x.parameters()),
            lr=self.config.lr,
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.epochs)

        best_loss = float("inf")
        best_theta_state = flow_theta.state_dict()
        best_x_state = flow_x.state_dict()
        patience = 0
        lambda_ = 0.5
        train_losses_per_epoch = []
        val_losses_per_epoch = []

        with self.timer.time_operation("training", "fm_post_transform"):
            for epoch in (pbar := trange(self.config.epochs, desc="Training FM Post Transform")):
                flow_theta.train()
                flow_x.train()
                train_losses = []

                for thb, xb, yb in train_loader:
                    if simulate is not None:
                        xb = simulate(thb)
                    thb, xb, yb = thb.to(device), xb.to(device), yb.to(device)

                    # Sample x|y and theta|x for theta flow source
                    with torch.no_grad():
                        source_x = flow_x.sample_base(yb).to(device)
                        with self.timer.time_operation("training", "sample_x_flow"):
                            x_pred, _ = flow_x.sample(source_x, yb, device, only_last=True, return_nfe=True)

                        with self.timer.time_operation("training", "sample_npe"):
                            source_theta = npe.sample(x_pred, 1, device, show_progress_bars=False).squeeze(0)

                    # Theta flow loss (custom source from NPE)
                    loss_theta = flow_theta.compute_loss_with_source(
                        source_theta, thb, yb, velocity_reg=self.velocity_reg)

                    # X flow loss (standard base distribution)
                    loss_x = flow_x.compute_loss(xb, yb)

                    total_loss = lambda_ * loss_theta + (1 - lambda_) * loss_x

                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(flow_theta.parameters(), 5.0)
                    torch.nn.utils.clip_grad_norm_(flow_x.parameters(), 5.0)
                    optimizer.step()

                    train_losses.append(total_loss.item())

                scheduler.step()

                # Validation
                flow_theta.eval()
                flow_x.eval()
                val_losses = []

                with torch.no_grad():
                    for thb, xb, yb in val_loader:
                        if simulate is not None:
                            xb = simulate(thb)
                        thb, xb, yb = thb.to(device), xb.to(device), yb.to(device)

                        source_x = flow_x.sample_base(yb).to(device)
                        x_pred, _ = flow_x.sample(source_x, yb, device, only_last=True, return_nfe=True)
                        source_theta = npe.sample(x_pred, 1, device, show_progress_bars=False).squeeze(0)

                        loss_theta = flow_theta.compute_loss_with_source(
                            source_theta, thb, yb, velocity_reg=self.velocity_reg)
                        loss_x = flow_x.compute_loss(xb, yb)

                        val_losses.append((lambda_ * loss_theta + (1 - lambda_) * loss_x).item())

                train_loss = sum(train_losses) / len(train_losses)
                val_loss = sum(val_losses) / len(val_losses)
                train_losses_per_epoch.append(train_loss)
                val_losses_per_epoch.append(val_loss)
                pbar.set_postfix(train=train_loss, val=val_loss)

                if self.logger:
                    self.logger.log_metrics(
                        {"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch},
                        log_name=logname,
                    )

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_theta_state = deepcopy(flow_theta.state_dict())
                    best_x_state = deepcopy(flow_x.state_dict())
                    patience = 0
                else:
                    patience += 1

                if self.config.max_patience is not None and patience >= self.config.max_patience:
                    pbar.write(f"\tEarly stopping at epoch {epoch + 1}")
                    break

        self.state.loss_history = {"train": train_losses_per_epoch, "val": val_losses_per_epoch}

        flow_theta.load_state_dict(best_theta_state)
        flow_x.load_state_dict(best_x_state)

        if self.config.save:
            torch.save(best_theta_state, self.model_path / f"{logname}_theta.pth")
            torch.save(best_x_state, self.model_path / f"{logname}_x.pth")

        flow_theta.cpu()
        flow_x.cpu()
        npe.cpu()

        return DualFlowPosteriorEstimator(
            "fm_post_transform",
            base_dist=npe,
            flow_theta=flow_theta,
            flow_x=flow_x,
        )

    def create_posterior(self, model) -> DualFlowPosteriorEstimator:
        raise NotImplementedError()


@register_trainer("fm_post_transform_only_ut", category="calibration")
class FMPostTransformOnlyUtTrainer(FMPostTransformTrainer):
    """FM Post Transform with pretrained x flow (only train theta flow)."""

    def train(
        self,
        data: Tuple[Tensor, Tensor, Tensor],
        device: torch.device,
        logname: str = "",
        **kwargs,
    ) -> DualFlowPosteriorEstimator:
        """Train only theta flow, use pretrained x flow."""
        theta, x, y = data
        self._theta_shape = theta.shape[1:]
        self._x_shape = x.shape[1:]
        ncal = x.shape[0]

        # Load NPE
        npe = self._load_npe(device)
        npe.to(device)

        # Load or train x flow
        simulator = self._load_simulator()
        simulate = None
        if simulator is not None:
            try:
                simulate = simulator.get_simulator(misspecified=True)
            except NotImplementedError:
                pass

        # Get pretrained x flow
        flow_x = self._get_or_train_flow_x(theta, x, y, simulate, device, logname)
        flow_x.to(device)

        # Build theta flow
        flow_theta_cfg = self.flow_config.get("flow_theta", {})
        flow_theta = _create_flow_from_config(flow_theta_cfg, theta.shape[1:], y.shape[1:])

        if self.config.load:
            theta_path = self.model_path / f"{logname}_theta.pth"
            if theta_path.exists():
                flow_theta.load_state_dict(torch.load(theta_path, map_location=device, weights_only=True))
                return DualFlowPosteriorEstimator(
                    "fm_post_transform_only_ut",
                    base_dist=npe,
                    flow_theta=flow_theta,
                    flow_x=flow_x,
                )

        flow_theta.to(device)
        flow_theta.set_scales(theta, y, self.config.rescale)

        self._print_training_config()

        # Train only theta flow
        train_loader, val_loader = train_val_split_n(
            theta, x, y, train_size=self.config.train_size, batch_size=self.config.batch_size
        )

        optimizer = torch.optim.Adam(flow_theta.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.epochs)

        best_loss = float("inf")
        best_theta_state = flow_theta.state_dict()
        patience = 0
        train_losses_per_epoch = []
        val_losses_per_epoch = []

        with self.timer.time_operation("training", "fm_post_transform_only_ut"):
            for epoch in (pbar := trange(self.config.epochs, desc="Training FM Post Transform (theta only)")):
                flow_theta.train()
                train_losses = []

                for thb, xb, yb in train_loader:
                    if simulate is not None:
                        xb = simulate(thb)
                    thb, xb, yb = thb.to(device), xb.to(device), yb.to(device)

                    with torch.no_grad():
                        source_x = flow_x.sample_base(yb).to(device)
                        x_pred, _ = flow_x.sample(source_x, yb, device, only_last=True, return_nfe=True)
                        source_theta = npe.sample(x_pred, 1, device, show_progress_bars=False).squeeze(0)

                    loss = flow_theta.compute_loss_with_source(source_theta, thb, yb)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(flow_theta.parameters(), 5.0)
                    optimizer.step()

                    train_losses.append(loss.item())

                scheduler.step()

                flow_theta.eval()
                val_losses = []
                with torch.no_grad():
                    for thb, xb, yb in val_loader:
                        if simulate is not None:
                            xb = simulate(thb)
                        thb, xb, yb = thb.to(device), xb.to(device), yb.to(device)

                        source_x = flow_x.sample_base(yb).to(device)
                        x_pred, _ = flow_x.sample(source_x, yb, device, only_last=True, return_nfe=True)
                        source_theta = npe.sample(x_pred, 1, device, show_progress_bars=False).squeeze(0)

                        loss = flow_theta.compute_loss_with_source(source_theta, thb, yb)
                        val_losses.append(loss.item())

                train_loss = sum(train_losses) / len(train_losses)
                val_loss = sum(val_losses) / len(val_losses)
                train_losses_per_epoch.append(train_loss)
                val_losses_per_epoch.append(val_loss)
                pbar.set_postfix(train=train_loss, val=val_loss)

                if self.logger:
                    self.logger.log_metrics(
                        {"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch},
                        log_name=logname,
                    )

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_theta_state = deepcopy(flow_theta.state_dict())
                    patience = 0
                else:
                    patience += 1

                if self.config.max_patience is not None and patience >= self.config.max_patience:
                    pbar.write(f"\tEarly stopping at epoch {epoch + 1}")
                    break

        self.state.loss_history = {"train": train_losses_per_epoch, "val": val_losses_per_epoch}

        flow_theta.load_state_dict(best_theta_state)

        if self.config.save:
            torch.save(best_theta_state, self.model_path / f"{logname}_theta.pth")

        flow_theta.cpu()
        flow_x.cpu()
        npe.cpu()

        return DualFlowPosteriorEstimator(
            "fm_post_transform_only_ut",
            base_dist=npe,
            flow_theta=flow_theta,
            flow_x=flow_x,
        )

    def _get_or_train_flow_x(
        self,
        theta: Tensor,
        x: Tensor,
        y: Tensor,
        simulate: Optional[Callable],
        device: torch.device,
        logname: str,
    ) -> FlowMatching:
        """Get or train x flow."""
        ncal = x.shape[0]
        flow_x_cfg = self.flow_config.get("flow_x", {})
        space = flow_x_cfg.get("space", "data")
        saved_path = self.model_path / f"flow_x_y_{space}_{ncal}.pth"

        # Prepare training data
        if simulate is not None:
            n_repeat = 10
            perm = torch.randperm(x.shape[0] * n_repeat)
            theta_repeated = theta.repeat_interleave(n_repeat, dim=0)[perm]
            x_train = simulate(theta_repeated)
            y_train = y.repeat_interleave(n_repeat, dim=0)[perm]
            epochs_x = self.config.epochs // 10
        else:
            x_train = x
            y_train = y
            epochs_x = self.config.epochs

        flow_x = _create_flow_from_config(flow_x_cfg, x_train.shape[1:], y.shape[1:])

        if saved_path.exists():
            print(f"\tLoading pretrained x flow from {saved_path}")
            flow_x.load_state_dict(torch.load(saved_path, map_location=device, weights_only=True))
            flow_x.rescale_name = self.config.rescale
            return flow_x

        # Train x flow
        print(f"\tTraining x flow")
        flow_x.to(device)
        flow_x.set_scales(x_train, y_train, self.config.rescale)

        # Print training configuration
        print(f"  Rescaling: {self.config.rescale}")
        print(f"  Epochs: {epochs_x}, LR: {self.config.lr}, Batch size: {self.config.batch_size}")
        if self.config.max_patience is None:
            print(f"  Early stopping: disabled (returning last epoch model)")
        else:
            print(f"  Early stopping: max_patience={self.config.max_patience}")

        from utils.misc import train_val_split
        train_loader, val_loader = train_val_split(
            x_train, y_train, self.config.train_size, self.config.batch_size
        )

        optimizer = torch.optim.Adam(flow_x.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs_x)

        best_loss = float("inf")
        best_state = flow_x.state_dict()
        patience = 0

        for epoch in (pbar := trange(epochs_x, desc="Training x flow")):
            flow_x.train()
            train_losses = []
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = flow_x.compute_loss(xb.to(device), yb.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(flow_x.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            scheduler.step()

            flow_x.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    loss = flow_x.compute_loss(xb.to(device), yb.to(device))
                    val_losses.append(loss.item())

            train_loss = sum(train_losses) / len(train_losses)
            val_loss = sum(val_losses) / len(val_losses)
            pbar.set_postfix(train=train_loss, val=val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = deepcopy(flow_x.state_dict())
                patience = 0
            else:
                patience += 1

            if self.config.max_patience is not None and patience >= self.config.max_patience:
                break

        flow_x.load_state_dict(best_state)
        torch.save(best_state, saved_path)

        return flow_x


@register_trainer("fm_post_transform_plug_y", category="calibration")
class FMPostTransformPlugYTrainer(FMPostTransformTrainer):
    """FM Post Transform without x flow (plug y directly)."""

    def train(
        self,
        data: Tuple[Tensor, Tensor, Tensor],
        device: torch.device,
        logname: str = "",
        **kwargs,
    ) -> DualFlowPosteriorEstimator:
        """Train theta flow only, use y directly instead of x."""
        theta, x, y = data
        self._theta_shape = theta.shape[1:]
        self._x_shape = x.shape[1:]

        # Load NPE
        npe = self._load_npe(device)
        npe.to(device)

        # Build theta flow
        flow_theta_cfg = self.flow_config.get("flow_theta", {})
        flow_theta = _create_flow_from_config(flow_theta_cfg, theta.shape[1:], y.shape[1:])

        if self.config.load:
            theta_path = self.model_path / f"{logname}_theta.pth"
            if theta_path.exists():
                flow_theta.load_state_dict(torch.load(theta_path, map_location=device, weights_only=True))
                return DualFlowPosteriorEstimator(
                    "fm_post_transform_plug_y",
                    base_dist=npe,
                    flow_theta=flow_theta,
                    flow_x=None,
                )

        flow_theta.to(device)
        flow_theta.set_scales(theta, y, self.config.rescale)

        self._print_training_config()

        train_loader, val_loader = train_val_split_n(
            theta, x, y, train_size=self.config.train_size, batch_size=self.config.batch_size
        )

        optimizer = torch.optim.Adam(flow_theta.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.epochs)

        best_loss = float("inf")
        best_theta_state = flow_theta.state_dict()
        patience = 0
        train_losses_per_epoch = []
        val_losses_per_epoch = []

        with self.timer.time_operation("training", "fm_post_transform_plug_y"):
            for epoch in (pbar := trange(self.config.epochs, desc="Training FM Post Transform (plug y)")):
                flow_theta.train()
                train_losses = []

                for thb, xb, yb in train_loader:
                    thb, yb = thb.to(device), yb.to(device)

                    # Use y directly as conditioning to NPE for source
                    with torch.no_grad():
                        source_theta = npe.sample(yb, 1, device, show_progress_bars=False).squeeze(0)

                    loss = flow_theta.compute_loss_with_source(source_theta, thb, yb)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(flow_theta.parameters(), 5.0)
                    optimizer.step()

                    train_losses.append(loss.item())

                scheduler.step()

                flow_theta.eval()
                val_losses = []
                with torch.no_grad():
                    for thb, xb, yb in val_loader:
                        thb, yb = thb.to(device), yb.to(device)
                        source_theta = npe.sample(yb, 1, device, show_progress_bars=False).squeeze(0)

                        loss = flow_theta.compute_loss_with_source(source_theta, thb, yb)
                        val_losses.append(loss.item())

                train_loss = sum(train_losses) / len(train_losses)
                val_loss = sum(val_losses) / len(val_losses)
                train_losses_per_epoch.append(train_loss)
                val_losses_per_epoch.append(val_loss)
                pbar.set_postfix(train=train_loss, val=val_loss)

                if self.logger:
                    self.logger.log_metrics(
                        {"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch},
                        log_name=logname,
                    )

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_theta_state = deepcopy(flow_theta.state_dict())
                    patience = 0
                else:
                    patience += 1

                if self.config.max_patience is not None and patience >= self.config.max_patience:
                    pbar.write(f"\tEarly stopping at epoch {epoch + 1}")
                    break

        self.state.loss_history = {"train": train_losses_per_epoch, "val": val_losses_per_epoch}

        flow_theta.load_state_dict(best_theta_state)

        if self.config.save:
            torch.save(best_theta_state, self.model_path / f"{logname}_theta.pth")

        flow_theta.cpu()
        npe.cpu()

        return DualFlowPosteriorEstimator(
            "fm_post_transform_plug_y",
            base_dist=npe,
            flow_theta=flow_theta,
            flow_x=None,
        )


@register_trainer("fmcpe_fm", category="calibration")
class FMCPEFMTrainer(FMPostTransformOnlyUtTrainer):
    """Flow Matching CPE with flow matching for x."""

    def train(
        self,
        data: Tuple[Tensor, Tensor, Tensor],
        device: torch.device,
        logname: str = "",
        **kwargs,
    ) -> FlowMatchingPosterior:
        """Train FMCPE-FM."""
        theta, x, y = data
        self._theta_shape = theta.shape[1:]
        self._x_shape = x.shape[1:]

        # Load simulator
        simulator = self._load_simulator()
        simulate = None
        if simulator is not None:
            try:
                simulate = simulator.get_simulator(misspecified=True)
            except NotImplementedError:
                pass

        # Get or train x flow
        flow_x = self._get_or_train_flow_x(theta, x, y, simulate, device, logname)
        flow_x.to(device)

        # Load FMPE config and model
        config_path = self.model_path / "npe_fmpe.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config_npe = json.load(f)
        else:
            config_npe = self.task_config.get("fmpe", {}).get("config", {})

        fmpe_flow = _create_flow_from_config(config_npe, theta.shape[1:], y.shape[1:])

        weights_path = self.model_path / "npe_fmpe.pth"
        if weights_path.exists():
            fmpe_flow.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))

        fmpe_flow.rescale_name = self.config.rescale
        fmpe = DirectFlowMatchingPosterior("npe_fmpe", fmpe_flow)

        # Save x flow
        ncal = x.shape[0]
        torch.save(flow_x.state_dict(), self.model_path / f"flow_x_y_{ncal}.pth")

        flow_x.cpu()

        return FlowMatchingPosterior(
            "fmcpe_fm",
            posterior=fmpe,
            flow=flow_x,
            theta_dim=theta.shape[1:],
            obs_dim=x.shape[1:],
        )


@register_trainer("fm_post_transform_split", category="calibration")
class FMPostTransformSplitTrainer(FMPostTransformTrainer):
    """FM Post Transform with split calibration data.

    Trains T_X on first half of calibration data, T_Theta on second half.
    Addresses the cross-fitting concern (Reviewer xKrR Q3).
    """

    def train(
        self,
        data: Tuple[Tensor, Tensor, Tensor],
        device: torch.device,
        logname: str = "",
        **kwargs,
    ) -> DualFlowPosteriorEstimator:
        """Train fm_post_transform with split calibration data."""
        theta, x, y = data
        self._theta_shape = theta.shape[1:]
        self._x_shape = x.shape[1:]

        # Load NPE
        npe = self._load_npe(device)
        npe.to(device)

        # Load simulator for data augmentation
        simulator = self._load_simulator()
        simulate = None
        if simulator is not None:
            try:
                simulate = simulator.get_simulator(misspecified=True)
            except NotImplementedError:
                pass

        # Build flow models
        flow_theta, flow_x = self.build_model(
            theta_shape=theta.shape[1:],
            x_shape=x.shape[1:],
            y_shape=y.shape[1:],
        )

        # Check for existing checkpoints
        if self.config.load:
            theta_path = self.model_path / f"{logname}_theta.pth"
            x_path = self.model_path / f"{logname}_x.pth"
            if theta_path.exists() and x_path.exists():
                flow_theta.load_state_dict(torch.load(theta_path, map_location=device, weights_only=True))
                flow_x.load_state_dict(torch.load(x_path, map_location=device, weights_only=True))
                return DualFlowPosteriorEstimator(
                    "fm_post_transform_split",
                    base_dist=npe,
                    flow_theta=flow_theta,
                    flow_x=flow_x,
                )

        # Split calibration data into two disjoint halves
        n = theta.shape[0]
        n_half = n // 2
        perm = torch.randperm(n)
        idx_1, idx_2 = perm[:n_half], perm[n_half:]

        theta_1, x_1, y_1 = theta[idx_1], x[idx_1], y[idx_1]
        theta_2, x_2, y_2 = theta[idx_2], x[idx_2], y[idx_2]

        print(f"  Split calibration: {n} -> {n_half} (T_X) + {n - n_half} (T_Theta)")

        self._print_training_config()

        # =====================================================================
        # Phase A: Train T_X on half 1
        # =====================================================================
        flow_x.to(device)

        # Handle simulator augmentation for x flow
        if simulate is not None:
            n_repeat = 10
            aug_perm = torch.randperm(x_1.shape[0] * n_repeat)
            theta_1_rep = theta_1.repeat_interleave(n_repeat, dim=0)[aug_perm]
            x_train = simulate(theta_1_rep)
            y_train = y_1.repeat_interleave(n_repeat, dim=0)[aug_perm]
            epochs_x = self.config.epochs // 10
        else:
            x_train = x_1
            y_train = y_1
            epochs_x = self.config.epochs

        flow_x.set_scales(x_train, y_train, self.config.rescale)

        from utils.misc import train_val_split
        train_loader_x, val_loader_x = train_val_split(
            x_train, y_train, self.config.train_size, self.config.batch_size
        )

        optimizer_x = torch.optim.Adam(flow_x.parameters(), lr=self.config.lr)
        scheduler_x = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_x, epochs_x)

        best_x_loss = float("inf")
        best_x_state = flow_x.state_dict()
        patience_x = 0

        has_val_x = len(val_loader_x.dataset) > 0

        with self.timer.time_operation("training", "fm_post_transform_split_x"):
            for epoch in (pbar := trange(epochs_x, desc="[Split] Training T_X")):
                flow_x.train()
                train_losses = []
                for xb, yb in train_loader_x:
                    optimizer_x.zero_grad()
                    loss = flow_x.compute_loss(xb.to(device), yb.to(device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(flow_x.parameters(), 5.0)
                    optimizer_x.step()
                    train_losses.append(loss.item())

                scheduler_x.step()
                train_loss = sum(train_losses) / len(train_losses)

                if has_val_x:
                    flow_x.eval()
                    val_losses = []
                    with torch.no_grad():
                        for xb, yb in val_loader_x:
                            loss = flow_x.compute_loss(xb.to(device), yb.to(device))
                            val_losses.append(loss.item())
                    val_loss = sum(val_losses) / len(val_losses)
                else:
                    val_loss = train_loss

                pbar.set_postfix(train=train_loss, val=val_loss)

                if self.logger:
                    self.logger.log_metrics(
                        {"train_loss_x": train_loss, "val_loss_x": val_loss, "epoch": epoch},
                        log_name=f"{logname}_phase_x",
                    )

                if val_loss < best_x_loss:
                    best_x_loss = val_loss
                    best_x_state = deepcopy(flow_x.state_dict())
                    patience_x = 0
                else:
                    patience_x += 1

                if self.config.max_patience is not None and patience_x >= self.config.max_patience:
                    pbar.write(f"\tEarly stopping T_X at epoch {epoch + 1}")
                    break

        flow_x.load_state_dict(best_x_state)
        flow_x.eval()

        # =====================================================================
        # Phase B: Train T_Theta on half 2 with frozen T_X
        # =====================================================================
        flow_theta.to(device)
        flow_theta.set_scales(theta_2, y_2, self.config.rescale)

        train_loader_theta, val_loader_theta = train_val_split_n(
            theta_2, x_2, y_2, train_size=self.config.train_size, batch_size=self.config.batch_size
        )

        optimizer_theta = torch.optim.Adam(flow_theta.parameters(), lr=self.config.lr)
        scheduler_theta = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_theta, self.config.epochs)

        best_theta_loss = float("inf")
        best_theta_state = flow_theta.state_dict()
        patience_theta = 0
        train_losses_per_epoch = []
        val_losses_per_epoch = []

        has_val_theta = len(val_loader_theta.dataset) > 0

        with self.timer.time_operation("training", "fm_post_transform_split_theta"):
            for epoch in (pbar := trange(self.config.epochs, desc="[Split] Training T_Theta")):
                flow_theta.train()
                train_losses = []

                for thb, xb, yb in train_loader_theta:
                    if simulate is not None:
                        xb = simulate(thb)
                    thb, xb, yb = thb.to(device), xb.to(device), yb.to(device)

                    with torch.no_grad():
                        source_x = flow_x.sample_base(yb).to(device)
                        x_pred, _ = flow_x.sample(source_x, yb, device, only_last=True, return_nfe=True)
                        source_theta = npe.sample(x_pred, 1, device, show_progress_bars=False).squeeze(0)

                    loss = flow_theta.compute_loss_with_source(
                        source_theta, thb, yb, velocity_reg=self.velocity_reg)

                    optimizer_theta.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(flow_theta.parameters(), 5.0)
                    optimizer_theta.step()

                    train_losses.append(loss.item())

                scheduler_theta.step()
                train_loss = sum(train_losses) / len(train_losses)

                if has_val_theta:
                    flow_theta.eval()
                    val_losses = []
                    with torch.no_grad():
                        for thb, xb, yb in val_loader_theta:
                            if simulate is not None:
                                xb = simulate(thb)
                            thb, xb, yb = thb.to(device), xb.to(device), yb.to(device)

                            source_x = flow_x.sample_base(yb).to(device)
                            x_pred, _ = flow_x.sample(source_x, yb, device, only_last=True, return_nfe=True)
                            source_theta = npe.sample(x_pred, 1, device, show_progress_bars=False).squeeze(0)

                            loss = flow_theta.compute_loss_with_source(
                                source_theta, thb, yb, velocity_reg=self.velocity_reg)
                            val_losses.append(loss.item())
                    val_loss = sum(val_losses) / len(val_losses)
                else:
                    val_loss = train_loss

                train_losses_per_epoch.append(train_loss)
                val_losses_per_epoch.append(val_loss)
                pbar.set_postfix(train=train_loss, val=val_loss)

                if self.logger:
                    self.logger.log_metrics(
                        {"train_loss_theta": train_loss, "val_loss_theta": val_loss, "epoch": epoch},
                        log_name=f"{logname}_phase_theta",
                    )

                if val_loss < best_theta_loss:
                    best_theta_loss = val_loss
                    best_theta_state = deepcopy(flow_theta.state_dict())
                    patience_theta = 0
                else:
                    patience_theta += 1

                if self.config.max_patience is not None and patience_theta >= self.config.max_patience:
                    pbar.write(f"\tEarly stopping T_Theta at epoch {epoch + 1}")
                    break

        self.state.loss_history = {"train": train_losses_per_epoch, "val": val_losses_per_epoch}

        flow_theta.load_state_dict(best_theta_state)

        if self.config.save:
            torch.save(best_theta_state, self.model_path / f"{logname}_theta.pth")
            torch.save(best_x_state, self.model_path / f"{logname}_x.pth")

        flow_theta.cpu()
        flow_x.cpu()
        npe.cpu()

        return DualFlowPosteriorEstimator(
            "fm_post_transform_split",
            base_dist=npe,
            flow_theta=flow_theta,
            flow_x=flow_x,
        )
