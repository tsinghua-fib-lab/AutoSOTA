import logging

import hydra
from omegaconf import DictConfig

from cellbridge.dynamics.sampling import MarginalSampler

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../../../conf", config_name="sampling_marginal"
)
def main(cfg: DictConfig) -> None:
    """Sample marginal interpolation artifacts."""
    logging.basicConfig(level=logging.INFO)
    sampler = MarginalSampler(cfg, logger=logger)
    sampler.run()


if __name__ == "__main__":
    main()
