import logging
from pathlib import Path

import hydra
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from loader.generator import DataGenerator
from models.forcast.forcast_service import ForcastService
from utils.utils import set_seed

LOGGER = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./../configuration', config_name='default_fclstm_train.yaml')
def my_app(cfg: DictConfig):
    cfg = cfg.config
    _setup(cfg)
    fc_persist_dir = f"{cfg.experiment_data.model_dir}/fc"

    datasets = DataGenerator.get_data(cfg.dataset, cfg.task, replace_base_dir=cfg.experiment_data.data_dir)
    alphas = [cfg.task.alpha] if isinstance(cfg.task.alpha, float) else cfg.task.alpha

    # Prepare (Create, Train,..) underlying forcast models
    fc_service = _init_fc(cfg, datasets, fc_persist_dir, )
    _ = fc_service.prepare(datasets, alphas)
    return


def _setup(config):
    config.experiment_data.experiment_dir = Path().cwd()
    set_seed(config.experiment_data.seed)
    LOGGER.info('Init wandb.')
    exp_data = config.experiment_data
    if hasattr(exp_data, "offline"):
        if isinstance(exp_data.offline, bool):
            mode = 'offline' if exp_data.offline else 'online'
        else:
            mode = exp_data.offline
    else:
        mode = 'online'
    wandb.init(project=exp_data.project_name, name=HydraConfig.get().job.name,  #dir=Path.cwd(),
               entity=exp_data.project_entity if hasattr(exp_data, "project_entity") else None,  # Backward Compatible if attr not existing
               config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True), mode=mode,
               tags=config.wandb.tags, notes=config.wandb.notes, group=config.wandb.group,
               settings=wandb.Settings(start_method="fork", _service_wait=240))


def _init_fc(config, datasets, fc_persist_dir) -> ForcastService:
    LOGGER.info('Initialize forcast service.')
    return ForcastService(lambda: hydra.utils.instantiate(config.model_fc, no_x_features=datasets[0].no_x_features,
                                                          alpha=config.task.alpha, train_only=True),
                          data_config=config.dataset, task_config=config.task, model_config=config.model_fc,
                          persist_dir=fc_persist_dir, save_new_reg_bak=True, trainer_config=config.trainer,
                          experiment_config=config.experiment_data)


if __name__ == "__main__":
    my_app()
