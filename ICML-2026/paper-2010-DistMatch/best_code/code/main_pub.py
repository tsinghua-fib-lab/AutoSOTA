import logging

import hydra
import wandb
from omegaconf import DictConfig

from loader.generator import DataGenerator
from main_utils import _init_fc, _init_uc, _setup, Evaluator

LOGGER = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./../configuration', config_name='default_config_pub.yaml')
def my_app(cfg: DictConfig):
    cfg = cfg.config
    _setup(cfg)
    fc_persist_dir = f"{cfg.experiment_data.model_dir}/fc"
    uc_persist_dir = f"{cfg.experiment_data.model_dir}/uc"

    datasets_memory = DataGenerator.get_data(cfg.dataset_mem, cfg.task, replace_base_dir=cfg.experiment_data.data_dir)
    if cfg.task.global_norm:
        if hasattr(datasets_memory[0], "static_normalize_props"):
            static_norm_param = datasets_memory[0].static_normalize_props
        else:
            static_norm_param = None
        datasets_eval = DataGenerator.get_data(cfg.dataset_eval, cfg.task,
                                               replace_base_dir=cfg.experiment_data.data_dir,
                                               X_norm_param=datasets_memory[0].X_normalize_props,
                                               Y_norm_param=datasets_memory[0].Y_normalize_props,
                                               hydro_static_norm_param=static_norm_param)
    else:
        assert "Only global norm works with PUB because no normalization for PUB available"

    alphas = [cfg.task.alpha] if isinstance(cfg.task.alpha, float) else cfg.task.alpha

    # FC needs all datasets (and combine all predictins)
    # UC needs to get trained on complement calib data
    # Evaluation on Leave out data but mixing dataa from complement data

    # Prepare (Create, Train,..) underlying forcast models#
    fc_datasets = datasets_eval + datasets_memory
    fc_service = _init_fc(fc_conf=cfg.model_fc, data_conf=cfg.dataset_eval, task_conf=cfg.task, trainer_conf=cfg.trainer,
                          experiment_conf=cfg.experiment_data, datasets=fc_datasets, fc_persist_dir=fc_persist_dir)
    fc_datasets = fc_service.prepare(fc_datasets, alphas)

    # Calibrate/Train UC
    uc_service = _init_uc(uc_conf=cfg.model_uc, data_conf=cfg.dataset_mem, task_conf=cfg.task, fc_service=fc_service,
                          datasets=datasets_memory, uc_persist_dir=uc_persist_dir, fc_state_dim=fc_service.fc_state_dim,
                          record_attention=cfg.evaluation['att_plot_vega'] or cfg.evaluation['att_hist_matplot'])
    uc_service.prepare(datasets_memory, alphas, experiment_config=cfg.experiment_data, calib_trainer_config=cfg.trainer)

    # Evaluate
    if cfg.experiment_data.evaluate:
        Evaluator.evaluate(uc_service, datasets_eval, alphas, cfg.evaluation, mix_mem_data=datasets_memory)
    else:
        LOGGER.info("Skip Evaluation")
    wandb.finish()


if __name__ == "__main__":
    my_app()
