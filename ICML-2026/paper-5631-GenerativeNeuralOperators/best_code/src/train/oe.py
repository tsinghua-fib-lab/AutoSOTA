import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Optional

from omegaconf import DictConfig
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger

from data import create_dataloaders
from train.utils import (
    finish_wandb,
    get_dataset_data_name_lower,
    make_checkpoint_callback,
    make_run_name,
    make_trainer,
    make_wandb_logger,
    optional_ckpt_path,
    should_share_xy_normalizer,
    split_dataset_result,
    strip_name_keys,
    StripMetadataModule,
)
from train.metrics import compute_nrmse, compute_psrmse_three_bands, compute_vrmse


class OperatorEncoder(StripMetadataModule):
    """
    OperatorEncoder mirrors the old ridge-based feature extractor structure, but *does not* solve a last-layer
    regression (ridge). Instead, it learns per-example last-layer parameters `w` from the
    *output* `y` via a learned embedder (e.g. `models.diffusion.embedding.FnoEmbedding1d`).

    Reconstruction is:
      features = backbone(x)
      w = output_embedder(y)  -> reshape to [B, C_feat, C_y]
      y_hat = features @ w
    """

    def __init__(
        self,
        model: nn.Module,
        output_embedder: nn.Module,
        optimizer_config: DictConfig,
        scheduler_config: Optional[DictConfig] = None,
        x_normalizer=None,
        y_normalizer=None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["model", "output_embedder", "x_normalizer", "y_normalizer"]
        )

        self.model = model
        self.output_embedder = output_embedder
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.criterion = nn.MSELoss()
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _move_normalizers_to_device(self) -> None:
        """Ensure normalizer stats are on the same device as the module."""
        for n in (
            getattr(self, "x_normalizer", None),
            getattr(self, "y_normalizer", None),
        ):
            if n is None:
                continue
            to_fn = getattr(n, "to", None)
            if callable(to_fn):
                n.to(self.device)

    def on_fit_start(self) -> None:
        self._move_normalizers_to_device()

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        """
        Shared train/val logic.

        Logs (consistent with `Autoencoder`):
          - {stage}_loss: reconstruction loss
          - {stage}_recon_loss: reconstruction MSE on flattened outputs
        """
        self._move_normalizers_to_device()
        inputs, targets = batch
        features = self(inputs)

        X_flat, Y_flat = self._reshape_io(features, targets)
        C_feat = int(features.shape[1])
        weight_dim = int(Y_flat.shape[-1])

        w = self._encode_weights(
            targets, num_feat_channels=C_feat, weight_dim=weight_dim
        )
        preds_flat = torch.matmul(X_flat, w)

        recon_loss = self.criterion(preds_flat, Y_flat)
        total_loss = recon_loss

        sync = stage in {"val", "test"}
        self.log(f"{stage}_loss", total_loss, prog_bar=True, sync_dist=sync)
        if stage in {"val", "test"}:
            preds = self._reshape_back(preds_flat, features, target_channels=weight_dim)
            if self.y_normalizer is not None:
                preds_original = self.y_normalizer.decode(preds)
                targets_original = self.y_normalizer.decode(targets)
            else:
                preds_original = preds
                targets_original = targets
            mse_original = self.criterion(preds_original, targets_original)
            self.log(f"{stage}_recon_loss", mse_original, prog_bar=True, sync_dist=sync)
            rmse_original = torch.sqrt(mse_original + 1e-12)
            self.log(
                f"{stage}_recon_rmse", rmse_original, prog_bar=True, sync_dist=sync
            )
            nrmse_original = compute_nrmse(preds_original, targets_original)
            self.log(
                f"{stage}_recon_nrmse", nrmse_original, prog_bar=False, sync_dist=sync
            )
            vrmse_original = compute_vrmse(preds_original, targets_original)
            self.log(
                f"{stage}_recon_vrmse", vrmse_original, prog_bar=False, sync_dist=sync
            )
            with torch.no_grad():
                ps = compute_psrmse_three_bands(
                    y_samples=preds_original.detach().to("cpu").unsqueeze(0),
                    y_true=targets_original.detach().to("cpu"),
                )
            self.log(
                f"{stage}_recon_psrmse_low",
                ps["psrmse_low"],
                prog_bar=False,
                sync_dist=sync,
            )
            self.log(
                f"{stage}_recon_psrmse_mid",
                ps["psrmse_mid"],
                prog_bar=False,
                sync_dist=sync,
            )
            self.log(
                f"{stage}_recon_psrmse_high",
                ps["psrmse_high"],
                prog_bar=False,
                sync_dist=sync,
            )

        return total_loss, preds_flat, weight_dim, features

    def _reshape_io(self, features: torch.Tensor, targets: torch.Tensor = None):
        """Handles dimension permutation for the projection y = Features * w."""
        B, C = features.shape[0], features.shape[1]

        if features.ndim == 4:  # Spatial: (B, C, H, W)
            X = features.permute(0, 2, 3, 1).reshape(B, -1, C)
            if targets is not None:
                if targets.ndim != 4:
                    raise ValueError(
                        f"Expected 4D targets for 4D features, got targets.shape={tuple(targets.shape)}"
                    )
                Cy = targets.shape[1]
                Y = targets.permute(0, 2, 3, 1).reshape(B, -1, Cy)
                return X, Y
            return X
        X = features.permute(0, 2, 1)  # (B, L, C)
        Y = targets.permute(0, 2, 1) if targets is not None else None  # (B, L, Cy)
        return X, Y

    def _encode_weights(
        self,
        targets: torch.Tensor,
        num_feat_channels: int,
        weight_dim: int,
    ) -> torch.Tensor:
        """
        Encodes last-layer weights from targets using the learned output embedder.
        Returns weights with shape [B, C_feat, weight_dim].

        Applies a deterministic saturating function to the embedder output:
        z → z / sqrt(1 + z²/B²) where B = 1.
        This replaces the KL penalty and mimics the range of a standard Gaussian distribution.
        """
        emb = self.output_embedder(targets)  # expected (B, E)
        if emb.ndim != 2:
            raise ValueError(
                f"output_embedder(targets) must return (B, E), got {tuple(emb.shape)}"
            )
        B_sat = 1.0
        emb = emb / torch.sqrt(1.0 + (emb**2) / (B_sat**2))

        B, E = emb.shape
        expected = int(num_feat_channels) * int(weight_dim)
        if E != expected:
            raise ValueError(
                f"output_embedder output dim mismatch: got E={E}, expected C_feat*weight_dim={expected} "
                f"(C_feat={num_feat_channels}, weight_dim={weight_dim})."
            )
        return emb.view(B, num_feat_channels, weight_dim)

    @torch.no_grad()
    def encode_weights(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Convenience API for DLL precomputation:
          features = model(x)
          w = encode from y with correct dims inferred from features/y.
        """
        features = self.model(x)
        _, Y_flat = self._reshape_io(features, y)
        C_feat = int(features.shape[1])
        weight_dim = int(Y_flat.shape[-1])
        return self._encode_weights(y, num_feat_channels=C_feat, weight_dim=weight_dim)

    def _reshape_back(self, preds_flat, original_features, target_channels: int):
        """Restores flattened predictions to original spatial/temporal shape."""
        if original_features.ndim == 4:
            B, _, H, W = original_features.shape
            return preds_flat.view(B, H, W, target_channels).permute(0, 3, 1, 2)
        return preds_flat.permute(0, 2, 1)

    def training_step(self, batch, batch_idx):
        total_loss, preds_flat, weight_dim, features = self._shared_step(batch, "train")
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, preds_flat, weight_dim, features = self._shared_step(batch, "val")

        if batch_idx == 0:
            preds = self._reshape_back(preds_flat, features, target_channels=weight_dim)
            inputs, targets = batch
            self._log_recon_plots(targets, preds, "val")

        return total_loss

    def test_step(self, batch, batch_idx):
        total_loss, preds_flat, weight_dim, features = self._shared_step(batch, "test")
        if batch_idx == 0:
            preds = self._reshape_back(preds_flat, features, target_channels=weight_dim)
            _inputs, targets = batch
            self._log_recon_plots(targets, preds, "test")
        return total_loss

    def _log_recon_plots(
        self, targets: torch.Tensor, predictions: torch.Tensor, mode: str
    ) -> None:
        """
        Log target / reconstruction / error for 3 random samples (val/test only).
        Matches `Regression._log_plots` style, but without variance (not applicable).
        """
        if not isinstance(self.logger, WandbLogger):
            return
        if getattr(self.trainer, "global_rank", 0) != 0:
            return
        if self.y_normalizer is not None:
            targets_plot = self.y_normalizer.decode(targets)
            preds_plot = self.y_normalizer.decode(predictions)
        else:
            targets_plot = targets
            preds_plot = predictions

        err_plot = (preds_plot - targets_plot).pow(2)

        B = int(targets_plot.shape[0])
        num_samples = min(3, B)
        indices = torch.randperm(B)[:num_samples].cpu()

        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        if num_samples == 1:
            axes = axes[None, :]
        is_1d = False
        if targets_plot.ndim == 3:
            is_1d = True  # [B, C, L]
        elif targets_plot.ndim == 4 and (
            targets_plot.shape[2] == 1 or targets_plot.shape[3] == 1
        ):
            is_1d = True  # [B, C, 1, W] or [B, C, H, 1]

        for i, idx in enumerate(indices):
            idx = int(idx.item())
            plot_items = [
                (axes[i, 0], targets_plot, f"{mode} Target {i}"),
                (axes[i, 1], preds_plot, f"{mode} Recon {i}"),
                (axes[i, 2], err_plot, f"{mode} (Recon - Target)^2 {i}"),
            ]

            for ax, tensor, title in plot_items:
                data = tensor[idx, 0].detach().cpu().numpy()
                if is_1d:
                    ax.plot(data.flatten())
                    ax.set_title(title)
                    ax.grid(True, which="major", alpha=0.3, linewidth=0.8)
                    ax.tick_params(axis="both", which="both", labelsize=9)
                else:
                    im = ax.imshow(data, cmap="viridis", origin="lower")
                    ax.set_title(title)
                    plt.colorbar(im, ax=ax)
                    ax.tick_params(axis="both", which="both", labelsize=9)

        plt.tight_layout()
        self.logger.log_image(key=f"{mode}_recon_samples", images=[fig])
        plt.close(fig)

    def configure_optimizers(self):
        params = list(self.model.parameters()) + list(self.output_embedder.parameters())
        optimizer = instantiate(self.optimizer_config, params=params)

        if (
            self.scheduler_config is None
            or getattr(self.scheduler_config, "scheduler", None) is None
        ):
            return optimizer

        scheduler = instantiate(self.scheduler_config.scheduler, optimizer=optimizer)
        lr_sched = {
            "scheduler": scheduler,
            "interval": str(getattr(self.scheduler_config, "interval", "epoch")),
            "frequency": int(getattr(self.scheduler_config, "frequency", 1)),
            "strict": bool(getattr(self.scheduler_config, "strict", True)),
            "name": str(getattr(self.scheduler_config, "name", "lr")),
        }
        monitor = getattr(self.scheduler_config, "monitor", None)
        if monitor is not None:
            lr_sched["monitor"] = str(monitor)

        return {"optimizer": optimizer, "lr_scheduler": lr_sched}


def train_oe(cfg: DictConfig, return_model: bool = False) -> OperatorEncoder | None:
    """Train OperatorEncoder.

    Args:
        cfg: Configuration dictionary
        return_model: If True, return the trained operator encoder model

    Returns:
        Trained OperatorEncoder if return_model=True, None otherwise
    """
    if "seed" in cfg:
        seed_everything(int(cfg.seed), workers=True)

    dataset_result = instantiate(cfg.dataset)
    train_set, val_set, test_set, _, _, _, _ = split_dataset_result(dataset_result)
    oe_cfg = cfg.operatorencoder
    ckpt_path = optional_ckpt_path(getattr(oe_cfg, "pretrained_ckpt", None))

    if ckpt_path is not None:
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"operatorencoder.pretrained_ckpt was provided but file not found: {ckpt_path}"
            )
        backbone = instantiate(strip_name_keys(oe_cfg.model))
        out_embed = instantiate(strip_name_keys(oe_cfg.output_embedder))
        oe_module = OperatorEncoder.load_from_checkpoint(
            str(ckpt_path),
            model=backbone,
            output_embedder=out_embed,
            optimizer_config=oe_cfg.optimizer,
            scheduler_config=getattr(oe_cfg, "scheduler", None),
            map_location="cpu",
            weights_only=False,
        )
        print(f"Loaded pretrained operator encoder from: {ckpt_path}")
        if return_model:
            return oe_module
        return

    data_name_lower = get_dataset_data_name_lower(cfg)
    share_xy = should_share_xy_normalizer(
        data_name_lower=data_name_lower, training_cfg=cfg.training_oe
    )

    train_loader, val_loader, test_loader, xn, yn = create_dataloaders(
        train_set,
        val_set,
        test_set,
        batch_size=int(cfg.training_oe.batch_size),
        num_workers=int(cfg.training_oe.num_workers),
        normalization_mode=str(cfg.training_oe.normalization_mode),
        share_xy_normalizer=share_xy,
    )

    backbone = instantiate(strip_name_keys(oe_cfg.model))
    out_embed = instantiate(strip_name_keys(oe_cfg.output_embedder))
    oe_module = OperatorEncoder(
        model=backbone,
        output_embedder=out_embed,
        optimizer_config=oe_cfg.optimizer,
        scheduler_config=getattr(oe_cfg, "scheduler", None),
        x_normalizer=xn,
        y_normalizer=yn,
    )

    dataset = getattr(cfg, "dataset", None)
    base_name = getattr(dataset, "data_name", "Experiment") if dataset else "Experiment"
    project_name = f"{base_name}_AE"
    logging_cfg = getattr(cfg, "logging", None)

    model_name = getattr(oe_cfg.model, "model_name", "OperatorEncoder")
    run_name = make_run_name(str(model_name), "OE")

    wandb_logger = make_wandb_logger(
        cfg,
        project_name=str(project_name),
        run_name=str(run_name),
        log_model=(
            getattr(logging_cfg, "log_model", "best_and_last")
            if logging_cfg
            else "best_and_last"
        ),
        save_dir=(
            getattr(logging_cfg, "save_dir", None)
            or getattr(getattr(oe_cfg, "logging", None), "save_dir", None)
        ),
    )

    checkpoint_callback = make_checkpoint_callback(
        project_name=str(project_name),
        run_name=str(run_name),
        filename_base="OE",
        filename="OE-best-{epoch:02d}",
        monitor="val_recon_rmse",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer = make_trainer(
        training_cfg=cfg.training_oe,
        max_epochs=int(cfg.training_oe.epochs),
        train_loader_len=len(train_loader),
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(oe_module, train_loader, val_loader)
    ckpt_path = getattr(checkpoint_callback, "best_model_path", None) or "best"
    try:
        trainer.test(
            oe_module, dataloaders=test_loader, ckpt_path=ckpt_path, weights_only=False
        )
    except TypeError:
        trainer.test(oe_module, dataloaders=test_loader, ckpt_path=ckpt_path)

    finish_wandb(wandb_logger)

    if return_model:
        return oe_module
