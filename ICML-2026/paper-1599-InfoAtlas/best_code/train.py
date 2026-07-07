"""
InfoAtlas training entry point (PyTorch Lightning + Hydra).

This script is released for transparency: it contains the full model, optimizer,
learning-rate schedule, checkpointing and validation loop used to pretrain
InfoAtlas. The *synthetic data-generation pipeline* is not part of this release.

To train from scratch, provide your own data generator by implementing
``generate_training_episode`` below (or by making a ``gen_data`` package importable
that exposes ``gen_train_dataset_joint_xy_mixed``). It must return a tensor of
preprocessed paired samples of shape ``[1, seq_len, 2 * max_dim]`` (X and Y
concatenated along the last axis), ready to be fed to the model.

Inference and evaluation with the released checkpoints do NOT need this script;
see ``infer.py`` and ``evaluations/``.
"""

import os
import warnings
import multiprocessing as mp

import matplotlib.pyplot as plt
import torch

from lightning.pytorch.callbacks import ModelCheckpoint

from infonet.decoder import Decoder
from infonet.encoder import Encoder
from infonet.infonet import InfoNet
from infonet.query import Query_Gen_transformer

from preprocessing import whiten_blocks

from torch.optim import Adam
import lightning
from lightning.pytorch.loggers import TensorBoardLogger
import hydra
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import StepLR, LambdaLR

from evaluation import evaluate_bmi

warnings.filterwarnings("ignore")
plt.switch_backend("agg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
# Data generation hook (withheld in this release)
# ------------------------------------------------------------------
# InfoAtlas is pretrained purely on synthetic paired samples drawn from a large
# family of dependence structures. That generation pipeline is not part of this
# release. Plug in your own generator here. The optional import below lets the
# original pipeline work transparently if a `gen_data` package is present.
try:
    from gen_data.train_data import gen_train_dataset_joint_xy_mixed as _generate_episode
except Exception:  # pragma: no cover - gen_data is intentionally not shipped
    _generate_episode = None


def generate_training_episode(seq_len, max_dim, softrank_reg, device="cpu"):
    """Return one preprocessed training episode of shape [1, seq_len, 2*max_dim].

    Replace this with your own synthetic / real data generator. It should yield a
    batch of paired samples (X, Y concatenated on the last axis) after the same
    soft-rank copula preprocessing used at inference (see ``preprocessing.py``).
    """
    if _generate_episode is None:
        raise NotImplementedError(
            "Training-data generation is not included in this release. Implement "
            "generate_training_episode() to return a tensor of shape "
            "[1, seq_len, 2*max_dim], or make a `gen_data` package importable."
        )
    return _generate_episode(
        batchsize=1,
        seq_len=seq_len,
        max_dim=max_dim,
        regularization_strength=softrank_reg,
        device=device,
        gauss_copula=True,
    )


class InfoNetDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, total_epoch, device="cpu", max_retry=1000):
        super().__init__()
        self.total_epoch = total_epoch
        self.seq_len = cfg.seq_len
        self.dim = cfg.input_dim_x
        self.softrank_reg = cfg.softrank_reg
        # Optional per-side whitening (config: whiten=eig). Applied AFTER the
        # soft-rank copula transform, so training preprocessing matches inference.
        self.whiten = str(cfg.get("whiten", "none"))
        self.whiten_eps = float(cfg.get("whiten_eps", 1e-3))
        self.device = device
        self.max_retry = max_retry
        print(f"init data set with device {self.device}, and total epoch {total_epoch}")

    def __len__(self):
        return self.total_epoch

    def _is_valid_tensor(self, x):
        if x is None:
            return False
        if not torch.is_tensor(x):
            return False
        if x.numel() == 0:
            return False
        return torch.isfinite(x).all().item()

    @torch.no_grad()
    def __getitem__(self, idx):
        for retry in range(self.max_retry):
            res = generate_training_episode(
                seq_len=self.seq_len,
                max_dim=self.dim,
                softrank_reg=self.softrank_reg,
                device=self.device,
            )
            if self._is_valid_tensor(res):
                if self.whiten == "eig":
                    res = whiten_blocks(res, max_dim=self.dim, eps_floor=self.whiten_eps)
                return res

        raise RuntimeError(
            f"InfoNetDataset failed to generate a finite sample after {self.max_retry} retries at idx={idx}"
        )


class LightningWrapper(lightning.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder = Encoder(
            input_dim_x=cfg.input_dim_x,
            input_dim_y=cfg.input_dim_y,
            latent_num=cfg.latent_num,
            latent_dim=cfg.latent_dim,
            cross_attn_heads=int(cfg.get("cross_attn_heads", 8)),
            self_attn_heads=int(cfg.get("self_attn_heads", 16)),
            num_self_attn_per_block=int(cfg.get("num_self_attn_per_block", 8)),
            num_self_attn_blocks=int(cfg.get("num_self_attn_blocks", 2)),
        )

        self.decoder = Decoder(
            q_dim=cfg.decoder_query_dim,
            latent_dim=cfg.latent_dim,
        )

        self.query_gen = Query_Gen_transformer(
            input_dim_x=cfg.input_dim_x,
            input_dim_y=cfg.input_dim_y,
            dim=cfg.decoder_query_dim,
        )

        hypogen_kwargs = {}
        for k in [
            "weight_dim", "enc_dec_dim", "opt_block_dim", "opt_mid_dim",
            "num_opt_mlp_layer", "num_enc_dec_layer", "num_layers",
            "weight_split_dim", "nhead", "lr_scheme_method", "use_compile",
            "replicate_blocks", "target_layers_num", "ablation",
        ]:
            if hasattr(cfg, k):
                hypogen_kwargs[k] = getattr(cfg, k)

        self.model = InfoNet(
            encoder=self.encoder,
            decoder=self.decoder,
            query_gen=self.query_gen,
            decoder_query_dim=cfg.decoder_query_dim,
            input_dim_x=cfg.input_dim_x,
            input_dim_y=cfg.input_dim_y,
            targetnet_hiddim=cfg.targetnet_hiddim,
            **hypogen_kwargs,
        )

        # Stamp the preprocessing flags onto the model so the validation loop
        # (evaluate_bmi) applies the SAME whitening the data is trained with.
        self.model._whiten = str(cfg.get("whiten", "none"))
        self.model._whiten_eps = float(cfg.get("whiten_eps", 1e-3))

    def forward(self, x):
        return self.model(x.squeeze(1), early_sup=False)

    def training_step(self, batch, batch_idx):
        mi_lb = self(batch)
        loss = -torch.mean(mi_lb)
        self.logger.experiment.add_scalars(
            "train_loss",
            {f"x-dim{self.cfg.input_dim_x}-y-dim{self.cfg.input_dim_y}": loss},
            global_step=self.global_step,
        )
        self.log("loss", loss.item(), on_step=True, prog_bar=True, sync_dist=True, batch_size=self.cfg.batchsize)
        return loss

    def validation_step(self, batch, batch_idx):
        return None

    def on_validation_epoch_end(self):
        if not self.trainer.is_global_zero:
            return

        print("=============== begin validation/evaluation")

        bmi_mean_bias = evaluate_bmi(
            module=self.model,
            max_dim=self.cfg.input_dim_x,
            number_test=0,
            training_step=self.global_step + 1,
            softrank_reg=self.cfg.softrank_reg,
            log_dir=self.logger.log_dir,
            n_samples_to_use=5,
            data_root=self.cfg.bmi_data_root,
            sample_sizes=[100, 250, 500, 1000, 5000],
        )

        self.logger.experiment.add_scalars(
            "bmi_mean_bias",
            {"bmi": bmi_mean_bias},
            global_step=self.global_step,
        )

    def test_step(self, batch, batch_idx):
        print("=============== begin test")
        return torch.tensor(0.0)

    def on_load_checkpoint(self, checkpoint):
        if "optimizer_states" in checkpoint:
            for state in checkpoint["optimizer_states"]:
                for param_group in state["param_groups"]:
                    param_group["lr"] = self.cfg.learning_rate

    def on_save_checkpoint(self, checkpoint):
        checkpoint["cfg"] = OmegaConf.to_container(self.cfg, resolve=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.cfg.learning_rate)

        # Optional linear warmup before the usual StepLR-style exponential decay.
        # warmup_steps absent / 0 -> behaviour is identical to a plain StepLR.
        warmup_steps = int(self.cfg.get("warmup_steps", 0))
        decay_step = self.cfg.lr_decay_step
        gamma = 0.9

        if warmup_steps > 0:
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step + 1) / float(warmup_steps)
                # gamma ** floor(step / decay_step): same shape as StepLR
                return gamma ** (step // decay_step)

            scheduler = LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = StepLR(optimizer, step_size=decay_step, gamma=gamma)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def train_dataloader(self):
        dataset = InfoNetDataset(
            cfg=self.cfg,
            total_epoch=self.cfg.num_per_epoch * self.cfg.gpu_card * self.cfg.batchsize * 1000,
            device="cpu",
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.batchsize,
            shuffle=False,
            num_workers=10,
            pin_memory=True,
            prefetch_factor=3,
            persistent_workers=True,
        )
        return dataloader

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            InfoNetDataset(cfg=self.cfg, total_epoch=1),
            batch_size=1, num_workers=0, pin_memory=False,
        )

    def val_dataloader(self):
        dataset = InfoNetDataset(cfg=self.cfg, total_epoch=1, device="cpu")
        return torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False)


@hydra.main(config_path="config", config_name="default_5d", version_base="1.1")
def main(cfg):
    torch.set_float32_matmul_precision("medium")
    module = LightningWrapper(cfg)

    log_dir = cfg.get("log_dir", "logs")
    logger = TensorBoardLogger(
        log_dir,
        name=cfg.name,
        version=cfg.version,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename="infoatlas-{step:08d}",
        every_n_train_steps=2500,
        save_top_k=-1,
        save_last=True,
    )

    trainer = lightning.Trainer(
        max_epochs=500000,
        accelerator="auto",
        devices=cfg.gpu_card,
        num_nodes=cfg.get("num_nodes", 1),
        logger=logger,
        strategy="ddp_find_unused_parameters_true",
        gradient_clip_val=1.0,
        val_check_interval=2500,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        use_distributed_sampler=False,
        callbacks=[checkpoint_callback],
    )

    ckpt_path = cfg.get("resume_ckpt", None)
    weights_only_resume = cfg.get("weights_only_resume", False)

    if ckpt_path and weights_only_resume:
        # Warm restart: load only model weights, reset optimizer / LR / global step.
        print(f"[warm-restart] loading weights only from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = module.load_state_dict(state_dict, strict=False)
        print(f"[warm-restart] loaded. missing={len(missing)} unexpected={len(unexpected)}")
        trainer.fit(module)
    elif ckpt_path:
        trainer.fit(module, ckpt_path=ckpt_path)
    else:
        trainer.fit(module)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
