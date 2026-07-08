from pathlib import Path
import hydra
import logging
from omegaconf import DictConfig, OmegaConf

from sweep.run_handler import RunHandler
from hydra.core.hydra_config import HydraConfig
LOGGER = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./../configuration', config_name='default_enbpi.yaml')
def run(cfg: DictConfig):
    # add hydra config
    hydra_config = OmegaConf.create({'hydra': HydraConfig.get()})
    cfg = OmegaConf.merge(hydra_config, cfg)
    LOGGER.info(f'Starting experiment with config: \n{OmegaConf.to_yaml(cfg)}')
    # absolute path to run script
    script_path = f"{cfg.config.experiment_data.base_proj_dir}code/main.py"
    run_handler = RunHandler(cfg, script_path)
    run_handler.run()


if __name__ == '__main__':
    run()
