"""Baseline method trainers (DPE, MF-NPE, FMPE-cal, NPE-RS, RoPE).

These are comparison baselines that use different approaches to handle
model misspecification.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from lampe.utils import GDStep
from torch import Tensor, nn
from tqdm.auto import trange

from posteriors import (
    DirectFlowMatchingPosterior,
    DirectPosteriorEstimator,
    Estimator,
    NPEPFNPosterior,
    RopePosterior,
)
from flow_matching.torch_flow import FlowMatching
from training.base import BaseTrainer, TrainerConfig
from training.registry import register_trainer
from utils.metrics import MMD
from utils.misc import train_val_split_n


@register_trainer("dpe", category="baseline")
class DPETrainer(BaseTrainer):
    """Trainer for Direct Posterior Estimation (DPE).

    DPE trains NPE directly on (theta, y) pairs from calibration data,
    ignoring the misspecified simulator.
    """

    def __init__(
        self,
        config: TrainerConfig,
        model_path: Path,
        task_config: Dict[str, Any],
        logger: Optional[Any] = None,
    ):
        super().__init__(config, model_path, task_config, logger)
        self.task_name = task_config.get("name", "unknown")

        # Use NPE params for DPE
        npe_config = task_config.get("npe", task_config.get("dpe", {}))
        self.npe_params = npe_config.get("params", {})

    def build_model(
        self,
        theta_shape: Tuple[int, ...] = None,
        y_shape: Tuple[int, ...] = None,
        **kwargs
    ) -> Estimator:
        """Build DPE estimator (same architecture as NPE)."""
        return Estimator(
            self.task_name,
            theta_shape,
            y_shape,
            **self.npe_params,
        )

    def train(
        self,
        data: Tuple[Tensor, Tensor, Tensor],
        device: torch.device,
        logname: str = "",
        **kwargs,
    ) -> DirectPosteriorEstimator:
        """Train DPE on (theta_cal, y_cal)."""
        from tqdm.auto import trange

        theta_cal, _, y_cal = data  # Ignore x_cal

        # Build and setup model
        model = self.build_model(theta_shape=theta_cal.shape[1:], y_shape=y_cal.shape[1:])
        model.set_scales(theta_cal, y_cal, self.config.rescale)
        model.to(device)

        if self.config.load:
            loaded = self._try_load_checkpoint(model, device)
            if loaded:
                return self.create_posterior(model)

        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        step = GDStep(optimizer, clip=5.0)

        from utils.misc import train_val_split
        train_loader, val_loader = train_val_split(
            theta_cal, y_cal, self.config.train_size, self.config.batch_size
        )

        self._print_training_config()

        best_loss = float("inf")
        best_state_dict = model.state_dict()
        patience = 0
        train_losses_per_epoch = []
        val_losses_per_epoch = []

        with self.timer.time_operation("training", "dpe"):
            for epoch in (pbar := trange(self.config.epochs, desc="Training DPE")):
                model.train()
                train_losses = []
                for thb, yb in train_loader:
                    loss = model.compute_loss(thb.to(device), yb.to(device))
                    train_losses.append(step(loss).item())

                model.eval()
                val_losses = []
                with torch.no_grad():
                    for thb, yb in val_loader:
                        val_losses.append(model.compute_loss(thb.to(device), yb.to(device)).item())

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

                patience += 1
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience = 0
                    best_state_dict = deepcopy(model.state_dict())

                if self.config.max_patience is not None and patience >= self.config.max_patience:
                    pbar.write(f"\tEarly stopping at epoch {epoch + 1}")
                    break

        self.state.loss_history = {"train": train_losses_per_epoch, "val": val_losses_per_epoch}

        model.load_state_dict(best_state_dict, strict=False)
        model.cpu()

        if self.config.save:
            self._save_checkpoint(model, logname)

        return self.create_posterior(model)

    def create_posterior(self, model: Estimator) -> DirectPosteriorEstimator:
        return DirectPosteriorEstimator("dpe", model)


@register_trainer("mf_npe", category="baseline")
class MFNPETrainer(BaseTrainer):
    """Trainer for Misspecification-Finetuned NPE (MF-NPE).

    MF-NPE starts from a pretrained NPE (trained on simulation data)
    and finetunes on calibration data (theta_cal, y_cal).
    """

    def __init__(
        self,
        config: TrainerConfig,
        model_path: Path,
        task_config: Dict[str, Any],
        logger: Optional[Any] = None,
    ):
        super().__init__(config, model_path, task_config, logger)
        self.task_name = task_config.get("name", "unknown")
        npe_config = task_config.get("npe", {})
        self.npe_params = npe_config.get("params", {})

    def build_model(
        self,
        theta_shape: Tuple[int, ...] = None,
        y_shape: Tuple[int, ...] = None,
        **kwargs
    ) -> Estimator:
        return Estimator(
            self.task_name,
            theta_shape,
            y_shape,
            **self.npe_params,
        )

    def train(
        self,
        data: Tuple[Tensor, Tensor, Tensor],
        device: torch.device,
        logname: str = "",
        npe: Any = None,
        **kwargs,
    ) -> DirectPosteriorEstimator:
        """Train MF-NPE starting from pretrained NPE.

        Args:
            data: (theta_cal, x_cal, y_cal) - calibration data
            device: torch device
            logname: name for logging
            npe: Pretrained NPE model or posterior (optional)
        """
        from tqdm.auto import trange

        theta_cal, _, y_cal = data

        model = self.build_model(theta_shape=theta_cal.shape[1:], y_shape=y_cal.shape[1:])

        # Load pretrained NPE weights - try multiple sources
        loaded_weights = False

        # First, try to use the passed npe model
        if npe is not None:
            print(f"\tLoading pretrained NPE from provided model")
            # Handle different model types (posterior wrapper or direct estimator)
            if hasattr(npe, 'model'):
                # It's a posterior wrapper
                model.load_state_dict(npe.model.state_dict(), strict=False)
            elif hasattr(npe, 'state_dict'):
                # It's a direct model
                model.load_state_dict(npe.state_dict(), strict=False)
            else:
                print(f"\tWARNING: Could not extract weights from provided npe model")
            loaded_weights = True

        # Fallback: try loading from file
        if not loaded_weights:
            pretrained_path = self.model_path / "npe.pth"
            if pretrained_path.exists():
                print(f"\tLoading pretrained NPE from {pretrained_path}")
                state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
                model.load_state_dict(state_dict, strict=False)
                loaded_weights = True

        if not loaded_weights:
            print(f"\tWARNING: No pretrained NPE found - training MF-NPE from scratch")

        model.set_scales(theta_cal, y_cal, self.config.rescale)
        model.to(device)

        if self.config.load:
            loaded = self._try_load_checkpoint(model, device)
            if loaded:
                return self.create_posterior(model)

        # Training (same as DPE but starting from pretrained)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        step = GDStep(optimizer, clip=5.0)

        from utils.misc import train_val_split
        train_loader, val_loader = train_val_split(
            theta_cal, y_cal, self.config.train_size, self.config.batch_size
        )

        self._print_training_config()

        best_loss = float("inf")
        best_state_dict = model.state_dict()
        patience = 0
        train_losses_per_epoch = []
        val_losses_per_epoch = []

        with self.timer.time_operation("training", "mf_npe"):
            for epoch in (pbar := trange(self.config.epochs, desc="Training MF-NPE")):
                model.train()
                train_losses = []
                for thb, yb in train_loader:
                    loss = model.compute_loss(thb.to(device), yb.to(device))
                    train_losses.append(step(loss).item())

                model.eval()
                val_losses = []
                with torch.no_grad():
                    for thb, yb in val_loader:
                        val_losses.append(model.compute_loss(thb.to(device), yb.to(device)).item())

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

                patience += 1
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience = 0
                    best_state_dict = deepcopy(model.state_dict())

                if self.config.max_patience is not None and patience >= self.config.max_patience:
                    pbar.write(f"\tEarly stopping at epoch {epoch + 1}")
                    break

        self.state.loss_history = {"train": train_losses_per_epoch, "val": val_losses_per_epoch}

        model.load_state_dict(best_state_dict, strict=False)
        model.cpu()

        if self.config.save:
            self._save_checkpoint(model, logname)

        return self.create_posterior(model)

    def create_posterior(self, model: Estimator) -> DirectPosteriorEstimator:
        return DirectPosteriorEstimator("mf_npe", model)


@register_trainer("fmpe_cal", category="baseline")
class FMPECalTrainer(BaseTrainer):
    """Trainer for FMPE on calibration data.

    Trains FMPE directly on (theta_cal, y_cal) as a baseline.
    """

    def __init__(
        self,
        config: TrainerConfig,
        model_path: Path,
        task_config: Dict[str, Any],
        logger: Optional[Any] = None,
    ):
        super().__init__(config, model_path, task_config, logger)
        fmpe_config = task_config.get("fmpe", {})
        self.fmpe_model_config = fmpe_config.get("config", {})

    def build_model(
        self,
        theta_shape: Tuple[int, ...] = None,
        y_shape: Tuple[int, ...] = None,
        **kwargs
    ) -> FlowMatching:
        cfg = self.fmpe_model_config
        return FlowMatching(
            cfg.get("probability_path", "ot2"),
            cfg.get("prior", "power"),
            cfg.get("base_dist", "gaussian"),
            theta_shape,
            cond_dim=y_shape,
            **cfg.get("params", {}),
        )

    def train(
        self,
        data: Tuple[Tensor, Tensor, Tensor],
        device: torch.device,
        logname: str = "",
        **kwargs,
    ) -> DirectFlowMatchingPosterior:
        from tqdm.auto import trange

        theta_cal, _, y_cal = data

        model = self.build_model(theta_shape=theta_cal.shape[1:], y_shape=y_cal.shape[1:])
        model.set_scales(theta_cal, y_cal, self.config.rescale)
        model.to(device)

        if self.config.load:
            loaded = self._try_load_checkpoint(model, device)
            if loaded:
                return self.create_posterior(model)

        from utils.misc import train_val_split
        train_loader, val_loader = train_val_split(
            theta_cal, y_cal, self.config.train_size, self.config.batch_size
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.epochs)

        self._print_training_config()

        best_loss = float("inf")
        best_state_dict = model.state_dict()
        patience = 0

        with self.timer.time_operation("training", "fmpe_cal"):
            for epoch in (pbar := trange(self.config.epochs, desc="Training FMPE-cal")):
                model.train()
                train_losses = []
                for thb, yb in train_loader:
                    optimizer.zero_grad()
                    loss = model.compute_loss(thb.to(device), yb.to(device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_losses.append(loss.item())

                scheduler.step()

                model.eval()
                val_losses = []
                with torch.no_grad():
                    for thb, yb in val_loader:
                        loss = model.compute_loss(thb.to(device), yb.to(device))
                        val_losses.append(loss.item())

                train_loss = sum(train_losses) / len(train_losses)
                val_loss = sum(val_losses) / len(val_losses)
                pbar.set_postfix(train=train_loss, val=val_loss)

                if self.logger:
                    self.logger.log_metrics(
                        {"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch},
                        log_name=logname,
                    )

                patience += 1
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience = 0
                    best_state_dict = deepcopy(model.state_dict())

                if self.config.max_patience is not None and patience >= self.config.max_patience:
                    pbar.write(f"\tEarly stopping at epoch {epoch + 1}")
                    break

        model.load_state_dict(best_state_dict, strict=False)
        model.cpu()

        if self.config.save:
            self._save_checkpoint(model, logname)

        return self.create_posterior(model)

    def create_posterior(self, model: FlowMatching) -> DirectFlowMatchingPosterior:
        return DirectFlowMatchingPosterior("fmpe_cal", model)


@register_trainer("npe_rs", category="baseline")
class NPERSTrainer(BaseTrainer):
    """Trainer for NPE with Robust Specification (NPE-RS).

    NPE-RS trains on simulation data while adding an MMD penalty
    to align embeddings of simulated and real observations.
    """

    def __init__(
        self,
        config: TrainerConfig,
        model_path: Path,
        task_config: Dict[str, Any],
        logger: Optional[Any] = None,
    ):
        super().__init__(config, model_path, task_config, logger)
        self.task_name = task_config.get("name", "unknown")

        npe_rs_config = task_config.get("npe_rs", task_config.get("npe", {}))
        self.npe_params = npe_rs_config.get("params", task_config.get("npe", {}).get("params", {}))

        training_config = npe_rs_config.get("training", {})
        self.mmd_weight = training_config.get("mmd_weight", 10.0)

    def build_model(
        self,
        theta_shape: Tuple[int, ...] = None,
        x_shape: Tuple[int, ...] = None,
        **kwargs
    ) -> Estimator:
        return Estimator(
            self.task_name,
            theta_shape,
            x_shape,
            **self.npe_params,
        )

    def train(
        self,
        data: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        device: torch.device,
        logname: str = "",
        **kwargs,
    ) -> DirectPosteriorEstimator:
        """Train NPE-RS.

        Args:
            data: (theta_sim, x_sim, theta_cal, x_cal, y_cal)
        """
        from tqdm.auto import trange

        theta_sim, x_sim, _, _, y_cal = data

        model = self.build_model(theta_shape=theta_sim.shape[1:], x_shape=x_sim.shape[1:])
        model.set_scales(theta_sim, x_sim, self.config.rescale)
        model.to(device)

        if self.config.load:
            loaded = self._try_load_checkpoint(model, device)
            if loaded:
                return self.create_posterior(model)

        y_cal = y_cal.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)

        from utils.misc import train_val_split
        train_loader, val_loader = train_val_split(
            theta_sim, x_sim, self.config.train_size, self.config.batch_size
        )

        self._print_training_config()

        best_loss = float("inf")
        best_state_dict = model.state_dict()
        patience = 0

        with self.timer.time_operation("training", "npe_rs"):
            for epoch in (pbar := trange(self.config.epochs, desc="Training NPE-RS")):
                model.train()
                train_losses = []
                mmd_losses = []

                for thb, xb in train_loader:
                    thb, xb = thb.to(device), xb.to(device)

                    # Sample from y_cal
                    batch_size = xb.shape[0]
                    y_idx = torch.randint(0, y_cal.shape[0], (batch_size,))
                    yb = y_cal[y_idx]

                    # Compute embeddings (need rescaled data for embedding comparison)
                    with torch.no_grad():
                        _, xb_rescaled = model._auto_rescale(thb, xb)
                        _, yb_rescaled = model._auto_rescale(thb, yb)

                    embed_x = model.embedding_net(xb_rescaled)
                    embed_y = model.embedding_net(yb_rescaled)

                    # MMD loss
                    mmd_loss = MMD(embed_x, embed_y, kernel="rbf")

                    # NPE loss (with automatic rescaling)
                    npe_loss = model.compute_loss(thb, xb)

                    # Combined loss
                    total_loss = npe_loss + self.mmd_weight * mmd_loss

                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

                    train_losses.append(total_loss.item())
                    mmd_losses.append(mmd_loss.item())

                model.eval()
                val_losses = []
                with torch.no_grad():
                    for thb, xb in val_loader:
                        val_losses.append(model.compute_loss(thb.to(device), xb.to(device)).item())

                train_loss = sum(train_losses) / len(train_losses)
                val_loss = sum(val_losses) / len(val_losses)
                mmd_loss_avg = sum(mmd_losses) / len(mmd_losses)
                pbar.set_postfix(train=train_loss, val=val_loss, mmd=mmd_loss_avg)

                if self.logger:
                    self.logger.log_metrics(
                        {
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "mmd_loss": mmd_loss_avg,
                            "epoch": epoch,
                        },
                        log_name=logname,
                    )

                patience += 1
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience = 0
                    best_state_dict = deepcopy(model.state_dict())

                if self.config.max_patience is not None and patience >= self.config.max_patience:
                    pbar.write(f"\tEarly stopping at epoch {epoch + 1}")
                    break

        model.load_state_dict(best_state_dict, strict=False)
        model.cpu()

        if self.config.save:
            self._save_checkpoint(model, logname)

        return self.create_posterior(model)

    def create_posterior(self, model: Estimator) -> DirectPosteriorEstimator:
        return DirectPosteriorEstimator("npe_rs", model)


@register_trainer("rope", category="baseline")
class RoPETrainer(BaseTrainer):
    """Trainer for Robust Posterior Estimation (RoPE).

    RoPE finetunes an embedding network to align representations of
    misspecified observations (y) with well-specified simulations (x).
    """

    def __init__(
        self,
        config: TrainerConfig,
        model_path: Path,
        task_config: Dict[str, Any],
        logger: Optional[Any] = None,
    ):
        super().__init__(config, model_path, task_config, logger)

    def build_model(self, **kwargs) -> nn.Module:
        raise NotImplementedError("RoPETrainer requires npe= kwarg in train()")

    def train(
        self,
        data: Tuple[Tensor, Tensor, Tensor],
        device: torch.device,
        logname: str = "",
        npe: Any = None,
        **kwargs,
    ) -> RopePosterior:
        """Train RoPE by finetuning embedding network.

        Args:
            data: (theta_cal, x_cal, y_cal) - calibration data
            device: torch device
            logname: name for logging
            npe: Pretrained NPE model or posterior wrapper (required)
        """
        theta, x, y = data

        if npe is None:
            raise ValueError("RoPETrainer requires a pretrained NPE model via npe= kwarg")

        # Extract the raw Estimator from a posterior wrapper if needed
        if isinstance(npe, DirectPosteriorEstimator):
            npe_model = npe.posterior
        elif isinstance(npe, Estimator):
            npe_model = npe
        else:
            raise TypeError(f"Expected DirectPosteriorEstimator or Estimator, got {type(npe)}")

        npe_model.set_scales(theta, x, self.config.rescale)

        embedding_net_original = npe_model.embedding_net

        # Handle Identity embedding network
        if isinstance(embedding_net_original, nn.Identity):
            print("\tEmbedding network is Identity - no fine-tuning required")
            return RopePosterior(
                name="rope",
                theta_dim=theta.shape[-1],
                obs_dim=x.shape[-1],
                embedding_net=embedding_net_original,
                fine_tuned_embedding_net=embedding_net_original,
                flow_model=npe_model,
            )

        embedding_net_finetuned = deepcopy(embedding_net_original)
        checkpoint_path = self.model_path / f"{logname}.pth"

        # Try loading existing checkpoint
        if self.config.load and checkpoint_path.exists():
            print(f"\tLoading fine-tuned embedding from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
            embedding_net_finetuned.load_state_dict(state_dict)
            return RopePosterior(
                name="rope",
                theta_dim=theta.shape[-1],
                obs_dim=x.shape[-1],
                embedding_net=embedding_net_original,
                fine_tuned_embedding_net=embedding_net_finetuned,
                flow_model=npe_model,
            )

        print(f"\tTraining RoPE embedding network")

        # Setup training
        embedding_net_original.to(device).requires_grad_(False)
        embedding_net_finetuned.to(device).train()
        npe_model.to(device)

        optimizer = torch.optim.Adam(embedding_net_finetuned.parameters(), lr=self.config.lr)
        train_loader, val_loader = train_val_split_n(
            theta, x, y, train_size=self.config.train_size, batch_size=self.config.batch_size
        )

        best_val_loss = float("inf")
        best_state_dict = deepcopy(embedding_net_finetuned.state_dict())
        patience = 0

        with self.timer.time_operation("training", "rope"):
            for epoch in (pbar := trange(self.config.epochs, desc="Training RoPE")):
                # Training
                embedding_net_finetuned.train()
                train_losses = []
                for theta_batch, x_batch, y_batch in train_loader:
                    theta_batch = theta_batch.to(device)
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    with torch.no_grad():
                        _, x_rescaled = npe_model._auto_rescale(theta_batch, x_batch)
                        _, y_rescaled = npe_model._auto_rescale(theta_batch, y_batch)
                        embedding_original = embedding_net_original(x_rescaled)

                    embedding_finetuned = embedding_net_finetuned(y_rescaled)
                    loss = (embedding_finetuned - embedding_original).pow(2).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(embedding_net_finetuned.parameters(), 5.0)
                    optimizer.step()

                    train_losses.append(loss.item())

                # Validation
                embedding_net_finetuned.eval()
                val_losses = []
                with torch.no_grad():
                    for theta_batch, x_batch, y_batch in val_loader:
                        theta_batch = theta_batch.to(device)
                        x_batch = x_batch.to(device)
                        y_batch = y_batch.to(device)

                        _, x_rescaled = npe_model._auto_rescale(theta_batch, x_batch)
                        _, y_rescaled = npe_model._auto_rescale(theta_batch, y_batch)

                        embedding_original = embedding_net_original(x_rescaled)
                        embedding_finetuned = embedding_net_finetuned(y_rescaled)

                        loss = (embedding_finetuned - embedding_original).pow(2).mean()
                        val_losses.append(loss.item())

                train_loss = sum(train_losses) / len(train_losses)
                val_loss = sum(val_losses) / len(val_losses)
                pbar.set_postfix(train=train_loss, val=val_loss)

                if self.logger:
                    self.logger.log_metrics(
                        {"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch},
                        log_name=logname,
                    )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = deepcopy(embedding_net_finetuned.state_dict())
                    patience = 0
                else:
                    patience += 1

                if self.config.max_patience is not None and patience >= self.config.max_patience:
                    pbar.write(f"\tEarly stopping at epoch {epoch + 1}")
                    break

        embedding_net_finetuned.load_state_dict(best_state_dict)

        if self.config.save:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state_dict, checkpoint_path)
            print(f"\tSaved fine-tuned embedding to {checkpoint_path}")

        embedding_net_original.cpu()
        embedding_net_finetuned.cpu()
        npe_model.cpu()

        return RopePosterior(
            name="rope",
            theta_dim=theta.shape[-1],
            obs_dim=x.shape[-1],
            embedding_net=embedding_net_original,
            fine_tuned_embedding_net=embedding_net_finetuned,
            flow_model=npe_model,
        )

    def create_posterior(self, model: nn.Module) -> RopePosterior:
        raise NotImplementedError()


@register_trainer("npe_pfn", category="baseline")
class NPEPFNTrainer(BaseTrainer):
    """NPE-PFN trainer (no training, just stores calibration data).

    NPE-PFN uses in-context learning via TabPFN instead of gradient descent.
    The calibration set is stored as context and used at inference time.
    """

    def __init__(
        self,
        config: TrainerConfig,
        model_path: Path,
        task_config: Dict[str, Any],
        logger: Optional[Any] = None,
    ):
        super().__init__(config, model_path, task_config, logger)
        npe_pfn_config = task_config.get("npe_pfn", {})
        # Valid filter types: 'latest_filtering', 'random_filtering', 'standardized_euclidean_filtering'
        self.filter_type = npe_pfn_config.get("filter_type", "latest_filtering")
        self.filter_context_size = npe_pfn_config.get("filter_context_size", 10_000)
        self.embedding_net_config = npe_pfn_config.get("embedding_net", None)
        self.x_shape = npe_pfn_config.get("x_shape", None)

    def build_model(
        self,
        theta_shape: Tuple[int, ...] = None,
        y_shape: Tuple[int, ...] = None,
        **kwargs
    ):
        """Build TabPFN_Based_NPE_PFN model."""
        from npe_pfn import TabPFN_Based_NPE_PFN

        embedding_net = None
        x_shape = None
        if self.embedding_net_config:
            from utils.networks import get_embedding_network

            embedding_net = get_embedding_network(**self.embedding_net_config)
            if self.x_shape is not None:
                x_shape = torch.Size(self.x_shape) if not isinstance(self.x_shape, torch.Size) else self.x_shape

        return TabPFN_Based_NPE_PFN(
            prior=None,  # No prior rejection sampling
            filter_type=self.filter_type,
            filter_context_size=self.filter_context_size,
            embedding_net=embedding_net,
            x_shape=x_shape,
        )

    def train(
        self,
        data: Tuple[Tensor, Tensor, Tensor],
        device: torch.device,
        logname: str = "",
        **kwargs,
    ) -> NPEPFNPosterior:
        """Store calibration data in NPE_PFN (no gradient training).

        Args:
            data: (theta_cal, x_cal, y_cal) - uses y_cal (noisy obs), ignores x_cal
            device: torch device (not used, NPE_PFN runs on CPU)
            logname: name for logging
        """
        theta_cal, _, y_cal = data  # Use y_cal (noisy obs), ignore x_cal

        # Flatten y if multi-dimensional (e.g., images)
        if y_cal.dim() > 2:
            y_cal = y_cal.reshape(y_cal.shape[0], -1)

        print(f"  NPE-PFN: Storing {theta_cal.shape[0]} calibration samples as context")
        print(f"  Filter type: {self.filter_type}, Context size limit: {self.filter_context_size}")

        with self.timer.time_operation("training", "npe_pfn"):
            model = self.build_model()
            model.append_simulations(theta_cal, y_cal)  # Store calibration data

        return self.create_posterior(model)

    def create_posterior(self, model) -> NPEPFNPosterior:
        return NPEPFNPosterior("npe_pfn", model)
