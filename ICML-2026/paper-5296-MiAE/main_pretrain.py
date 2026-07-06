"""MiAE pretraining entry point.

Pretrain MiAE on the Foldseek-clustered AlphaFold Database subset with the
composite ESM3 reconstruction objective and high masking ratio (default 0.9).
Config: ``configs/pretrain.yaml``.

Example usage::

    # Single-GPU
    python pretrain.py

    # Multi-GPU (4 GPUs, effective batch 4096 with accumulate_grad=32)
    torchrun --nproc_per_node=4 pretrain.py trainer.devices=4

    # Resume from checkpoint
    python pretrain.py train.ckpt_path=<path/to/ckpt>
"""
import logging
import hydra

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from tedbench.model import MiAE


torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True

OmegaConf.register_new_resolver("eval", eval)

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="./configs", config_name="pretrain")
def main(cfg):
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed, workers=True)

    model = MiAE(cfg)
    if torch.cuda.is_available() and cfg.train.compile:
        model = torch.compile(model)

    datamodule = hydra.utils.instantiate(cfg.datamodule)

    logger = []
    if cfg.wandb:
        wandb_logger = pl.loggers.WandbLogger(
            project="TEDBench", config=OmegaConf.to_container(cfg, resolve=True)
        )
        logger.append(wandb_logger)
    logger.append(pl.loggers.CSVLogger(cfg.logs.path, name="csv_logs"))

    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.ModelCheckpoint(
            monitor="val/rmsd",
            dirpath=cfg.logs.path,
            filename="model",
            mode="min",
        ),
    ]

    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    if cfg.train.ckpt_path is not None:
        trainer.fit(model, datamodule, ckpt_path=cfg.train.ckpt_path, weights_only=False)
    else:
        trainer.fit(model, datamodule)

    trainer.save_checkpoint(f"{cfg.logs.path}/model-last.ckpt")
    trainer.validate(model, datamodule)


if __name__ == "__main__":
    main()
