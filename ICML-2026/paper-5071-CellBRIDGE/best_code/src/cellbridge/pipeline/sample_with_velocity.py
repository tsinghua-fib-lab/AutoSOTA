import logging

import hydra
from omegaconf import DictConfig

from cellbridge.dynamics.sampling import VelocitySampler

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../../../conf", config_name="sampling_velocity"
)
def main(cfg: DictConfig) -> None:
    """Sample pushforwards from a trained velocity model."""
    logging.basicConfig(level=logging.INFO)
    sampler = VelocitySampler(cfg, logger=logger)
    sampler.run()


if __name__ == "__main__":
    main()
