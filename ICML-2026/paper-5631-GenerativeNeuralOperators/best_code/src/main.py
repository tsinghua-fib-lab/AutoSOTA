import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from train.regression import train_regression
from train.diffusion.dm import train_dm
from train.diffusion.dll import train_dll
from train.diffusion.ldm import train_ldm
from train.ae import train_ae
from train.oe import train_oe
from train.dropout import train_dropout
from train.pno import train_pno
from data.generate_dataset import generate_dataset

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra-driven entrypoint."""
    torch.set_float32_matmul_precision("high")

    log.info("Composed config:\n" + OmegaConf.to_yaml(cfg))

    run = getattr(cfg, "run", None)

    if run == "regression":
        train_regression(cfg)
    elif run == "dropout":
        train_dropout(cfg)
    elif run == "pno":
        train_pno(cfg)
    elif run == "dm":
        train_dm(cfg)
    elif run == "dll":
        train_dll(cfg)
    elif run == "ldm":
        train_ldm(cfg)
    elif run == "ae":
        train_ae(cfg)
    elif run in ("operatorencoder", "oe"):
        train_oe(cfg)
    elif run == "generate_dataset":
        generate_dataset(cfg)
    else:
        raise ValueError(
            "Unknown or missing run type "
            f"'{run}'. Expected one of: 'regression', 'dropout', 'pno', 'dm', 'dll', 'ldm', 'ae', 'oe', 'operatorencoder', 'generate_dataset'."
        )


if __name__ == "__main__":
    main()
