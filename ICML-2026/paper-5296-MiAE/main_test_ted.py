"""Test a trained MiAEClassifier checkpoint on the TEDBench test split or CATH 4.4.

Config: ``configs/test_ted.yaml``.

``pretrained_model_path`` may point to a Lightning ``.ckpt`` file, a local
HuggingFace-format directory, or a remote HuggingFace repo ID — the latter two
are produced / uploaded by ``convert_checkpoint.py``.

Example usage::

    # Lightning checkpoint
    python main_test_ted.py pretrained_model_path=<path/to/ckpt>

    # Local HuggingFace directory
    python main_test_ted.py pretrained_model_path=<path/to/hf_dir>

    # Remote HuggingFace repo
    python main_test_ted.py pretrained_model_path=username/miae-b-tedbench
"""
import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from tedbench.model import MiAEClassifier
from tedbench.utils.io import load_from_hf

torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True

OmegaConf.register_new_resolver("eval", eval)

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="./configs", config_name="test_ted")
def main(cfg):
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed, workers=True)

    p = cfg.pretrained_model_path
    # Local HF directory or remote repo ID (no file extension); .ckpt → Lightning
    if Path(p).is_dir() or not Path(p).suffix:
        model = load_from_hf(p)
    else:
        model = MiAEClassifier.load_from_checkpoint(p, weights_only=False)

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup("test")

    logger = [pl.loggers.CSVLogger(cfg.logs.path, name="csv_logs")]

    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
