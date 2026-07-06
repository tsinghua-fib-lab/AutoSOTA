"""End-to-end supervised training or fine-tuning on TEDBench.

When ``pretrained_model_path`` is ``null`` in the config, trains a MiAEEncoder
from scratch (supervised baseline, Table 6a settings).  When a pretrained MiAE
checkpoint is supplied, fine-tunes it end-to-end with layer-wise LR decay
(Table 6b settings).

Config: ``configs/finetune_ted.yaml``.

Example usage::

    # Supervised from scratch (MiAE-B)
    python main_finetune_ted.py

    # Fine-tune from pretrained checkpoint (.ckpt, local HF dir, or HF repo ID)
    python main_finetune_ted.py pretrained_model_path=logs/pretrain/.../model.ckpt
    python main_finetune_ted.py pretrained_model_path=username/miae-b-pretrained

    # Use MiAE-L with sequence input
    python main_finetune_ted.py model.name=miae_l model.use_seq_input=true
"""
import logging
from pathlib import Path

import hydra

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from tedbench.model import MiAE, MiAEClassifier
from tedbench.utils.io import load_from_hf


torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True

OmegaConf.register_new_resolver("eval", eval)

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="./configs", config_name="finetune_ted")
def main(cfg):
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed, workers=True)

    model = MiAEClassifier(cfg)
    if cfg.pretrained_model_path is not None:
        p = cfg.pretrained_model_path
        if Path(p).is_dir() or not Path(p).suffix:
            pretrained_model = load_from_hf(p).state_dict()
        else:
            pretrained_model = MiAE.load_from_checkpoint(
                p, map_location=torch.device("cpu"), weights_only=False
            ).state_dict()
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in pretrained_model
                and pretrained_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del pretrained_model[k]

        msg = model.load_state_dict(pretrained_model, strict=False)
        assert set(msg.missing_keys) == {"model.head.weight", "model.head.bias"}

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
            monitor="val/acc",
            dirpath=cfg.logs.path,
            filename="model",
            mode="max",
        ),
    ]

    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    if cfg.train.ckpt_path is not None:
        trainer.fit(model, datamodule, ckpt_path=cfg.train.ckpt_path, weights_only=False)
    else:
        trainer.fit(model, datamodule)

    trainer.save_checkpoint(f"{cfg.logs.path}/model-last.ckpt")
    trainer.test(model, datamodule, ckpt_path="best", weights_only=False)


if __name__ == "__main__":
    main()
