import torch
import torch.nn as nn
from omegaconf import DictConfig
from lightning.pytorch import seed_everything
from hydra.utils import instantiate
from typing import Any, Optional

from data import create_dataloaders
from train.utils import (
    finish_wandb,
    get_dataset_data_name_lower,
    load_submodule_state_dict_from_lightning_ckpt,
    make_checkpoint_callback,
    make_run_name,
    make_trainer,
    make_wandb_logger,
    optional_ckpt_path,
    should_share_xy_normalizer,
    split_dataset_result,
    strip_name_keys,
)
from train.ae import Autoencoder, train_ae
from train.diffusion.dm import DiffusionModel


class _DeterministicLatentEncoder(nn.Module):
    """Wrap an encoder that may output (mean, logvar) channel-wise (double_z)."""

    def __init__(self, encoder: nn.Module, *, double_z: bool):
        super().__init__()
        self.encoder = encoder
        self.double_z = bool(double_z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if self.double_z:
            c = z.shape[1] // 2
            z = z[:, :c, ...]
        return z


class LDM(DiffusionModel):
    """
    Latent Diffusion Model.
    Diffusion modeling happens in the latent space (z) of a frozen Autoencoder.
    Encodes and decodes on-the-fly during training (no precomputation).

    The target autoencoder (for y) is frozen.

    Conditioning (x -> z_x) can be:
      - shared: use the frozen target AE encoder
      - trainable embedder: pass `cond_embedder` (learned jointly with LDM)
    """

    def __init__(
        self,
        autoencoder: Autoencoder,
        model: nn.Module,
        optimizer_config: DictConfig,
        scheduler_config: DictConfig | None = None,
        T: int = 100,
        samples_per_example: int = 1,
        ema_decay: float = 0.9999,
        cond_embedder: Optional[nn.Module] = None,
        x_normalizer: Any = None,
        y_normalizer: Any = None,
        test_samples_per_example: int | None = None,
        rollout_k_chunk: int = 8,
        rollout_traj_chunk: int | None = None,
        stochastic_n_chunk: int = 32,
        stochastic_k_chunk: int = 4,
        viz_num_trajectories_default: int = 3,
        viz_horizon_percentiles_default: tuple = (1, 20, 40, 60, 80, 100),
        eval_mode: str | None = None,
    ):
        super().__init__(
            model=model,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            T=T,
            samples_per_example=samples_per_example,
            ema_decay=ema_decay,
            x_normalizer=x_normalizer,
            y_normalizer=y_normalizer,
            test_samples_per_example=test_samples_per_example,
            rollout_k_chunk=rollout_k_chunk,
            rollout_traj_chunk=rollout_traj_chunk,
            stochastic_n_chunk=stochastic_n_chunk,
            stochastic_k_chunk=stochastic_k_chunk,
            viz_num_trajectories_default=viz_num_trajectories_default,
            viz_horizon_percentiles_default=viz_horizon_percentiles_default,
            eval_mode=eval_mode,
        )
        self.autoencoder = autoencoder
        self.autoencoder.eval()
        self.autoencoder.requires_grad_(False)
        self.cond_embedder = cond_embedder
        self.save_hyperparameters(
            ignore=[
                "autoencoder",
                "cond_embedder",
                "model",
                "optimizer_config",
                "scheduler_config",
                "x_normalizer",
                "y_normalizer",
            ]
        )

    def _encode_conditioning(self, x: torch.Tensor) -> torch.Tensor:
        """Encode conditioning signal (x) to latent z_x."""
        if self.cond_embedder is not None:
            return self.cond_embedder(x)
        with torch.no_grad():
            return self.autoencoder.encode_deterministic(x)

    def _encode_target(self, y: torch.Tensor) -> torch.Tensor:
        """Encode target signal (y) to latent using autoencoder."""
        with torch.no_grad():
            return self.autoencoder.encode_deterministic(y)

    def training_step(self, batch, batch_idx):
        if len(batch) != 2:
            raise ValueError(
                f"Expected batch format (x, y) for LDM, got len(batch)={len(batch)}"
            )

        x, y = batch
        z_y = self._encode_target(y)

        z_x = self._encode_conditioning(x)

        diffusion_loss = self.compute_diffusion_loss(z_x, z_y)
        self.log("train_loss", diffusion_loss, prog_bar=True)
        return diffusion_loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 2:
            x, y = batch
            z_y = self._encode_target(y)
        elif len(batch) == 4:
            x, y, z_x, z_y = batch
        else:
            raise ValueError(
                f"Unexpected batch format for LDM: len(batch)={len(batch)}"
            )

        if len(batch) == 2:
            z_x = self._encode_conditioning(x)

        diffusion_loss = self.compute_diffusion_loss(z_x, z_y)
        self.log("val_loss", diffusion_loss, prog_bar=True, sync_dist=True)

        return diffusion_loss

    def test_step(self, batch, batch_idx):
        """
        Override DiffusionModel.test_step:
        - LDM trains/tests in latent space (z), but dataloaders yield raw (x, y).
        - We must encode to latents before computing DM loss.
        """
        if len(batch) == 2:
            x, y = batch
            z_x = self._encode_conditioning(x)
            z_y = self._encode_target(y)
        elif len(batch) == 4:
            x, y, z_x, z_y = batch
        else:
            raise ValueError(
                f"Unexpected batch format for LDM: len(batch)={len(batch)}"
            )

        loss = self.compute_diffusion_loss(z_x, z_y, use_ema=False)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer for flow model (+ optional cond_embedder)."""
        params = list(self.model.parameters())
        if self.cond_embedder is not None:
            params += list(self.cond_embedder.parameters())

        optimizer = instantiate(self.optimizer_config, params=params)

        if not self.scheduler_config or not getattr(
            self.scheduler_config, "scheduler", None
        ):
            return optimizer

        scheduler = instantiate(self.scheduler_config.scheduler, optimizer=optimizer)
        lr_sched = {
            "scheduler": scheduler,
            "interval": self.scheduler_config.get("interval", "epoch"),
            "frequency": self.scheduler_config.get("frequency", 1),
            "monitor": self.scheduler_config.get("monitor", "val_loss"),
            "strict": self.scheduler_config.get("strict", True),
            "name": self.scheduler_config.get("name", "lr"),
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_sched}

    @torch.inference_mode()
    def generate_y_samples(
        self,
        x: torch.Tensor,
        num_samples: int = 1,
        use_ema: bool | None = None,
        *,
        num_steps: int | None = None,
        solver: str = "euler",
    ) -> torch.Tensor:
        """Samples z via Flow, then decodes to y.

        Args:
            x: Raw conditioning signal (not latent)
            num_samples: Number of samples to generate per input
            use_ema: Whether to use EMA model for sampling (defaults to None, which uses EMA if available and not training)
            num_steps: Number of ODE steps (defaults to self.T)
            solver: ODE solver ("euler" or "heun")

        Returns:
            Generated samples in signal space: [K, B, ...]
        """
        K = int(num_samples)
        B = x.shape[0]
        z_x = self._encode_conditioning(x)
        z_in = z_x.repeat_interleave(K, dim=0) if K > 1 else z_x
        z_gen = self.sample(
            cond=z_in, num_steps=num_steps, solver=solver, use_ema=use_ema
        )
        y_gen = self.autoencoder.decoder(z_gen)
        if K > 1:
            y_gen = y_gen.view(B, K, *y_gen.shape[1:]).transpose(0, 1)
        else:
            y_gen = y_gen.unsqueeze(0)

        return y_gen


def train_ldm(cfg: DictConfig) -> None:
    if "seed" in cfg:
        seed_everything(int(cfg.seed), workers=True)
    ldm_training_cfg = getattr(cfg, "training_ldm", None) or getattr(
        cfg, "training", None
    )
    if ldm_training_cfg is None:
        raise ValueError(
            "Missing training config: expected `training_ldm` (preferred) or legacy `training`."
        )
    ae_training_cfg = getattr(cfg, "training_ae", None) or getattr(
        getattr(cfg, "autoencoder", None), "training", None
    )
    if ae_training_cfg is None:
        raise ValueError(
            "Missing AE training config: expected `training_ae` (preferred)."
        )
    ae_config = DictConfig(
        {
            "seed": getattr(cfg, "seed", None),
            "dataset": cfg.dataset,
            "autoencoder": cfg.autoencoder,
            "training_ae": ae_training_cfg,  # Add for scheduler interpolation
            "logging": getattr(cfg, "logging", None),
        }
    )
    ae_config.autoencoder.training = ae_training_cfg
    eval_mode = DiffusionModel._normalize_eval_mode(
        getattr(getattr(cfg, "evaluation", None), "mode", None)
    )
    if (
        eval_mode == "stochastic"
        and getattr(ae_config.autoencoder, "reconstruct", None) is None
    ):
        ae_config.autoencoder.reconstruct = "y"
    autoencoder = train_ae(ae_config, return_model=True)
    if autoencoder is None:
        raise RuntimeError(
            "train_ae(..., return_model=True) returned None for the target autoencoder. "
            "If you're using `autoencoder.pretrained_ckpt`, note that `train_ae` must return the loaded model."
        )
    cond_embedder: Optional[nn.Module] = None
    cond_cfg = getattr(cfg, "cond", None)
    cond_mode = str(getattr(cond_cfg, "mode", "shared_ae")).strip().lower()
    if cond_mode == "shared_ae":
        cond_embedder = None
    elif cond_mode == "learned_encoder":
        if cond_cfg is None or getattr(cond_cfg, "encoder", None) is None:
            raise ValueError(
                "cond.mode='learned_encoder' requires `cond.encoder` config (with encoder.z_channels)."
            )
        cond_encoder = instantiate(strip_name_keys(cond_cfg.encoder))
        ckpt_path = optional_ckpt_path(getattr(cond_cfg, "pretrained_ckpt", None))
        if ckpt_path is not None:
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"cond.pretrained_ckpt not found: {ckpt_path}")
            missing, unexpected, used_prefix = (
                load_submodule_state_dict_from_lightning_ckpt(
                    cond_encoder,
                    ckpt_path,
                    prefixes=["cond.encoder.", "autoencoder.encoder.", "encoder.", ""],
                    map_location="cpu",
                    strict=False,
                )
            )
            print(
                f"[INFO] Loaded condition encoder from {ckpt_path} (prefix={used_prefix!r}) "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
        double_z = bool(
            getattr(cond_cfg, "double_z", getattr(cond_encoder, "double_z", False))
        )
        cond_embedder = _DeterministicLatentEncoder(cond_encoder, double_z=double_z)
    else:
        raise ValueError(
            f"Unknown cond.mode={cond_mode!r}. Expected 'shared_ae' or 'learned_encoder'."
        )
    dataset_result = instantiate(cfg.dataset)
    (
        train_set,
        val_set,
        test_set,
        val_trjs,
        test_trjs,
        val_stochastic,
        test_stochastic,
    ) = split_dataset_result(dataset_result)
    data_name_lower = get_dataset_data_name_lower(cfg)
    share_xy = should_share_xy_normalizer(
        data_name_lower=data_name_lower, training_cfg=ldm_training_cfg
    )

    ldm_train_loader, ldm_val_loader, ldm_test_loader, xn, yn = create_dataloaders(
        train_set,
        val_set,
        test_set,
        batch_size=int(ldm_training_cfg.batch_size),
        num_workers=int(ldm_training_cfg.num_workers),
        normalization_mode=str(ldm_training_cfg.normalization_mode),
        share_xy_normalizer=share_xy,
    )
    z_ch = int(cfg.autoencoder.encoder.z_channels)
    if cond_embedder is not None:
        c_ch = int(cond_cfg.encoder.z_channels)
    else:
        c_ch = z_ch

    backbone_kwargs = {
        "input_channels": z_ch,
        "output_channels": z_ch,
        "c_dim": c_ch,
    }
    viz_pcts = getattr(cfg.ldm, "viz_horizon_percentiles", (1, 20, 40, 60, 80, 100))

    ldm_module = LDM(
        autoencoder=autoencoder,
        model=instantiate(strip_name_keys(cfg.ldm.model), **backbone_kwargs),
        optimizer_config=cfg.ldm.optimizer,
        scheduler_config=getattr(cfg.ldm, "scheduler", None),
        T=int(cfg.ldm.T),
        samples_per_example=int(cfg.ldm.samples_per_example),
        ema_decay=getattr(cfg.ldm, "ema_decay", 0.9999),
        cond_embedder=cond_embedder,
        x_normalizer=xn,
        y_normalizer=yn,
        test_samples_per_example=getattr(cfg.ldm, "test_samples_per_example", None),
        rollout_k_chunk=int(getattr(cfg.ldm, "rollout_k_chunk", 8)),
        rollout_traj_chunk=getattr(cfg.ldm, "rollout_traj_chunk", None),
        stochastic_n_chunk=int(getattr(cfg.ldm, "stochastic_n_chunk", 32)),
        stochastic_k_chunk=int(getattr(cfg.ldm, "stochastic_k_chunk", 4)),
        viz_num_trajectories_default=int(getattr(cfg.ldm, "viz_num_trajectories", 3)),
        viz_horizon_percentiles_default=viz_pcts,
        eval_mode=getattr(getattr(cfg, "evaluation", None), "mode", None),
    )
    if val_trjs is not None:
        ldm_module.val_trajectories = val_trjs
    if test_trjs is not None:
        ldm_module.test_trajectories = test_trjs
    if val_stochastic is not None:
        ldm_module.val_stochastic = val_stochastic
    if test_stochastic is not None:
        ldm_module.test_stochastic = test_stochastic

    run_name = make_run_name(getattr(cfg.ldm.model, "model_name", "LDM"), "LDM")
    dataset = getattr(cfg, "dataset", None)
    data_name = getattr(dataset, "data_name", "Exp") if dataset else "Exp"
    project_name = f"{data_name}_GEN"
    wandb_logger = make_wandb_logger(cfg, project_name=project_name, run_name=run_name)
    eval_mode = DiffusionModel._normalize_eval_mode(
        getattr(getattr(cfg, "evaluation", None), "mode", None)
    )
    if eval_mode == "stochastic":
        if val_stochastic is None or test_stochastic is None:
            raise ValueError(
                "evaluation.mode='stochastic' requires dataset extras: val_stochastic and test_stochastic."
            )
        monitor_metric = "val_stochastic_ed"
    elif eval_mode == "rollout":
        if val_trjs is None or test_trjs is None:
            raise ValueError(
                "evaluation.mode='rollout' requires dataset extras: val_trjs and test_trjs."
            )
        monitor_metric = "val_rollout_nrmse"
    else:
        monitor_metric = "val_loss"
    ldm_cb = make_checkpoint_callback(
        project_name=project_name,
        run_name=run_name,
        filename_base="LDM",
        filename="LDM-best-{epoch:02d}",
        monitor=monitor_metric,
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer = make_trainer(
        training_cfg=ldm_training_cfg,
        max_epochs=int(ldm_training_cfg.epochs),
        train_loader_len=len(ldm_train_loader),
        logger=wandb_logger,
        callbacks=[ldm_cb],
    )

    trainer.fit(ldm_module, ldm_train_loader, ldm_val_loader)
    ckpt_path = getattr(ldm_cb, "best_model_path", None) or "best"
    try:
        trainer.test(
            ldm_module,
            dataloaders=ldm_test_loader,
            ckpt_path=ckpt_path,
            weights_only=False,
        )
    except TypeError:
        trainer.test(ldm_module, dataloaders=ldm_test_loader, ckpt_path=ckpt_path)
    finish_wandb(wandb_logger)
