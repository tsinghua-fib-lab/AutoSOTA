import logging

import hydra
from omegaconf import DictConfig

from cellbridge.dynamics.flow_matching import FlowTrainer


@hydra.main(version_base=None, config_path="../../../conf", config_name="flow_matching")
def main(cfg: DictConfig) -> None:
    """Train the flow matching model."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("train_flow")
    trainer = FlowTrainer(cfg, logger=logger)
    trainer.run()


if __name__ == "__main__":
    main()
