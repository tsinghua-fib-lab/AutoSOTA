import lightning.pytorch as pl
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger

from data import create_dataloaders
from train.metrics import compute_nrmse, compute_psrmse_three_bands, compute_vrmse
from train.utils import (
    finish_wandb,
    make_trainer,
    make_checkpoint_callback,
    make_run_name,
    make_wandb_logger,
    get_dataset_data_name_lower,
    load_submodule_state_dict_from_lightning_ckpt,
    optional_ckpt_path,
    should_share_xy_normalizer,
    split_dataset_result,
    strip_name_keys,
)


class Autoencoder(pl.LightningModule):
    """
    Variational Autoencoder (VAE) LightningModule.

    Supports both VAE mode (with reparameterization and KL penalty) and
    deterministic mode (for backward compatibility with LDM).

    When double_z=True (VAE mode):
    - Encoder outputs mean and logvar: [B, 2*C_z, L_z]
    - Reparameterization: z = mean + std * epsilon, where epsilon ~ N(0,1)
    - Loss = Reconstruction Loss + kl_weight * KL Divergence

    When double_z=False (deterministic mode):
    - Encoder outputs latent directly: [B, C_z, L_z]
    - Loss = Reconstruction Loss only
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        optimizer_config: DictConfig,
        scheduler_config: DictConfig | None = None,
        x_normalizer=None,
        y_normalizer=None,
        reconstruct: str = "x",
        kl_weight: float = 1e-6,
        double_z: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["encoder", "decoder", "x_normalizer", "y_normalizer"]
        )
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.criterion = nn.MSELoss()
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer
        self.reconstruct = str(reconstruct).lower()
        if self.reconstruct not in {"x", "y"}:
            raise ValueError(f"reconstruct must be 'x' or 'y', got {reconstruct!r}")
        self.kl_weight = float(kl_weight)
        self.double_z = bool(double_z)
        encoder_double_z = getattr(encoder, "double_z", False)
        if self.double_z and not encoder_double_z:
            raise ValueError(
                f"Autoencoder double_z=True but encoder.double_z={encoder_double_z}. "
                "Set encoder double_z=True for VAE mode."
            )

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

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mean + std * epsilon, where epsilon ~ N(0,1)

        Args:
            mean: [B, C_z, L_z]
            logvar: [B, C_z, L_z]

        Returns:
            z: [B, C_z, L_z]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def kl_divergence(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence: KL(N(mean, var) || N(0, 1))

        Formula: -0.5 * sum(1 + logvar - mean^2 - exp(logvar))

        Args:
            mean: [B, C_z, L_z]
            logvar: [B, C_z, L_z]

        Returns:
            kl: scalar tensor
        """
        kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        return kl.mean()

    def encode(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Encode input to latent space.

        Args:
            x: Input tensor [B, C, L]
            deterministic: If True, return mean only (no sampling). If False and double_z=True, sample.

        Returns:
            z: Latent tensor [B, C_z, L_z]
        """
        encoder_out = self.encoder(x)  # [B, 2*C_z, L_z] if double_z else [B, C_z, L_z]

        if self.double_z:
            z_channels = encoder_out.shape[1] // 2
            mean = encoder_out[:, :z_channels, :]
            logvar = encoder_out[:, z_channels:, :]

            if deterministic:
                z = mean
            else:
                z = self.reparameterize(mean, logvar)
        else:
            z = encoder_out

        return z

    def encode_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent deterministically (returns mean for VAE, or direct output for deterministic AE)."""
        return self.encode(x, deterministic=True)

    def forward(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Forward pass: encode then decode.

        Args:
            x: Input tensor [B, C, L]
            deterministic: If True, use mean only (no sampling)

        Returns:
            x_hat: Reconstructed tensor [B, C, L]
        """
        z = self.encode(x, deterministic=deterministic)
        x_hat = self.decoder(z)
        return x_hat

    def _shared_step(self, batch, stage: str):
        self._move_normalizers_to_device()
        x, y = batch
        inp = x if self.reconstruct == "x" else y
        encoder_out = self.encoder(inp)

        if self.double_z:
            z_channels = encoder_out.shape[1] // 2
            mean = encoder_out[:, :z_channels, :]
            logvar = encoder_out[:, z_channels:, :]
            z = self.reparameterize(mean, logvar)
            kl_loss = self.kl_divergence(mean, logvar)
        else:
            z = encoder_out
            kl_loss = torch.tensor(0.0, device=z.device)
        x_hat = self.decoder(z)
        recon_loss = self.criterion(x_hat, inp)
        total_loss = recon_loss + self.kl_weight * kl_loss
        sync = stage in {"val", "test"}
        self.log(f"{stage}_loss", total_loss, prog_bar=True, sync_dist=sync)
        if self.double_z:
            self.log(f"{stage}_kl_loss", kl_loss, prog_bar=False, sync_dist=sync)
        if stage in {"val", "test"}:
            normalizer = (
                self.x_normalizer if self.reconstruct == "x" else self.y_normalizer
            )
            if normalizer is not None:
                inp_original = normalizer.decode(inp)
                x_hat_original = normalizer.decode(x_hat)
            else:
                inp_original = inp
                x_hat_original = x_hat

            mse_original = self.criterion(x_hat_original, inp_original)
            self.log(f"{stage}_recon_loss", mse_original, prog_bar=True, sync_dist=sync)
            rmse_original = torch.sqrt(mse_original + 1e-12)
            self.log(
                f"{stage}_recon_rmse", rmse_original, prog_bar=True, sync_dist=sync
            )
            nrmse_original = compute_nrmse(x_hat_original, inp_original)
            self.log(
                f"{stage}_recon_nrmse", nrmse_original, prog_bar=False, sync_dist=sync
            )
            vrmse_original = compute_vrmse(x_hat_original, inp_original)
            self.log(
                f"{stage}_recon_vrmse", vrmse_original, prog_bar=False, sync_dist=sync
            )
            with torch.no_grad():
                ps = compute_psrmse_three_bands(
                    y_samples=x_hat_original.detach().to("cpu").unsqueeze(0),
                    y_true=inp_original.detach().to("cpu"),
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

        return total_loss, inp, x_hat

    def training_step(self, batch, batch_idx):
        loss, x, x_hat = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, x, x_hat = self._shared_step(batch, "val")
        if batch_idx == 0:
            self._log_recon_plots(x, x_hat, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss, x, x_hat = self._shared_step(batch, "test")
        if batch_idx == 0:
            self._log_recon_plots(x, x_hat, "test")
        return loss

    def _log_recon_plots(
        self, target: torch.Tensor, recon: torch.Tensor, mode: str
    ) -> None:
        """
        Log target / reconstruction / error for 3 random samples (val/test only).
        Matches `Regression._log_plots` style, but without variance (not applicable).
        """
        if not isinstance(self.logger, WandbLogger):
            return
        if getattr(self.trainer, "global_rank", 0) != 0:
            return

        normalizer = self.x_normalizer if self.reconstruct == "x" else self.y_normalizer
        if normalizer is not None:
            target_plot = normalizer.decode(target)
            recon_plot = normalizer.decode(recon)
        else:
            target_plot = target
            recon_plot = recon

        err_plot = (recon_plot - target_plot).pow(2)

        B = int(target_plot.shape[0])
        num_samples = min(3, B)
        indices = torch.randperm(B)[:num_samples].cpu()

        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        if num_samples == 1:
            axes = axes[None, :]
        is_1d = False
        if target_plot.ndim == 3:
            is_1d = True  # [B, C, L]
        elif target_plot.ndim == 4 and (
            target_plot.shape[2] == 1 or target_plot.shape[3] == 1
        ):
            is_1d = True  # [B, C, 1, W] or [B, C, H, 1]

        for i, idx in enumerate(indices):
            idx = int(idx.item())
            plot_items = [
                (axes[i, 0], target_plot, f"{mode} Target {i}"),
                (axes[i, 1], recon_plot, f"{mode} Recon {i}"),
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
        optimizer = instantiate(self.optimizer_config, params=self.parameters())

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


def train_ae(cfg: DictConfig, return_model: bool = False) -> Autoencoder | None:
    """Standalone autoencoder training task.

    Uses:
      - cfg.dataset
      - cfg.autoencoder (architecture, optimizer, training schedule, pretrained_ckpt)

    Args:
        cfg: Configuration dictionary
        return_model: If True, return the trained autoencoder model

    Returns:
        Trained Autoencoder if return_model=True, None otherwise
    """
    if "seed" in cfg:
        seed_everything(int(cfg.seed), workers=True)

    dataset_result = instantiate(cfg.dataset)
    train_set, val_set, test_set, _, _, _, _ = split_dataset_result(dataset_result)

    ae_cfg = cfg.autoencoder
    encoder = instantiate(strip_name_keys(ae_cfg.encoder))
    decoder = instantiate(strip_name_keys(ae_cfg.decoder))

    autoencoder = Autoencoder(
        encoder=encoder,
        decoder=decoder,
        optimizer_config=ae_cfg.optimizer,
        scheduler_config=getattr(ae_cfg, "scheduler", None),
        reconstruct=str(getattr(ae_cfg, "reconstruct", "x")),
        kl_weight=float(getattr(ae_cfg, "kl_weight", 1e-6)),
        double_z=bool(getattr(ae_cfg, "double_z", True)),
    )

    ckpt_path = optional_ckpt_path(getattr(ae_cfg, "pretrained_ckpt", None))
    if ckpt_path is not None:
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"autoencoder.pretrained_ckpt was provided but file not found: {ckpt_path}"
            )
        missing, unexpected, pfx = load_submodule_state_dict_from_lightning_ckpt(
            autoencoder,
            ckpt_path,
            prefixes=["autoencoder.", ""],
            map_location="cpu",
            strict=True,
        )
        if missing or unexpected:
            print(
                f"[WARN] Loaded pretrained autoencoder from: {ckpt_path} (prefix={pfx}) with "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
        else:
            print(f"Loaded pretrained autoencoder from: {ckpt_path} (prefix={pfx})")
        return autoencoder if return_model else None
    data_name_lower = get_dataset_data_name_lower(cfg)
    share_xy = should_share_xy_normalizer(
        data_name_lower=data_name_lower, training_cfg=ae_cfg.training
    )

    ae_train_loader, ae_val_loader, ae_test_loader, xn, yn = create_dataloaders(
        train_set,
        val_set,
        test_set,
        batch_size=int(ae_cfg.training.batch_size),
        num_workers=int(ae_cfg.training.num_workers),
        normalization_mode=str(ae_cfg.training.normalization_mode),
        share_xy_normalizer=share_xy,
    )
    autoencoder.x_normalizer = xn
    autoencoder.y_normalizer = yn

    dataset = getattr(cfg, "dataset", None)
    base_name = getattr(dataset, "data_name", "Experiment") if dataset else "Experiment"
    logging_cfg = getattr(cfg, "logging", None)
    ae_project_name = base_name + "_AE"
    ae_model_name = getattr(cfg, "ae_model_name", "AE1d")
    ae_run_name = make_run_name(str(ae_model_name), "AE")

    ae_logger = make_wandb_logger(
        cfg,
        project_name=str(ae_project_name),
        run_name=str(ae_run_name),
        log_model=(
            getattr(logging_cfg, "log_model", "best_and_last")
            if logging_cfg is not None
            else "best_and_last"
        ),
        save_dir=(
            getattr(logging_cfg, "save_dir", None) if logging_cfg is not None else None
        ),
    )

    ae_checkpoint_callback = make_checkpoint_callback(
        project_name=str(ae_project_name),
        run_name=str(ae_run_name),
        filename_base=str(ae_model_name),
        filename=f"{ae_model_name}-best-{{epoch:02d}}",
        monitor="val_recon_rmse",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer = make_trainer(
        training_cfg=ae_cfg.training,
        max_epochs=int(ae_cfg.training.epochs),
        train_loader_len=len(ae_train_loader),
        logger=ae_logger,
        callbacks=[ae_checkpoint_callback],
    )

    trainer.fit(autoencoder, ae_train_loader, ae_val_loader)
    ckpt_path = getattr(ae_checkpoint_callback, "best_model_path", None) or "best"
    try:
        trainer.test(
            autoencoder,
            dataloaders=ae_test_loader,
            ckpt_path=ckpt_path,
            weights_only=False,
        )
    except TypeError:
        trainer.test(autoencoder, dataloaders=ae_test_loader, ckpt_path=ckpt_path)

    finish_wandb(ae_logger)

    if return_model:
        return autoencoder
