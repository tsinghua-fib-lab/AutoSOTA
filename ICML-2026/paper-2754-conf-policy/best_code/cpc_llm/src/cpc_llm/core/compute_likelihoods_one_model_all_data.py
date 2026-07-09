"""Backward-compatible entry point. Use core.likelihoods instead."""

import hydra
import logging
import pprint

from omegaconf import DictConfig, OmegaConf

from .likelihoods import compute_likelihoods_one_model_all_data

__all__ = ["compute_likelihoods_one_model_all_data"]


@hydra.main(
    config_path="../../../config", config_name="compute_liks_all_models_and_cal_data"
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=cfg.log_level.upper(), force=True)
    logging.info(
        f"Running {__file__} with the following arguments:\n{pprint.pformat(OmegaConf.to_container(cfg))}"
    )
    logger = logging.getLogger(__file__)
    logger.info("Running compute_likelihoods_one_model_all_data")
    compute_likelihoods_one_model_all_data(cfg, logger)


if __name__ == "__main__":
    main()
