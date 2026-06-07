import torch
from hydra.utils import instantiate

def fetch_model(cfg, logger):
    model = instantiate(cfg.model.instance)
    logger.info(f'Loading model from {cfg.model.name}.')

    return model