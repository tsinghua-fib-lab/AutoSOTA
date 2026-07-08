import os
import logging

import hydra
import wandb
from omegaconf import DictConfig

from loader.generator import DataGenerator
from main_utils import _init_fc, _init_uc, _setup, Evaluator

os.environ['WANDB_MODE'] = 'disabled'
LOGGER = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./../configuration', config_name='default_config.yaml')
def my_app(cfg: DictConfig):
    cfg = cfg.config
    _setup(cfg)
    fc_persist_dir = f"{cfg.experiment_data.model_dir}/fc"
    uc_persist_dir = f"{cfg.experiment_data.model_dir}/uc"

    datasets = DataGenerator.get_data(cfg.dataset, cfg.task, replace_base_dir=cfg.experiment_data.data_dir)
    alphas = [cfg.task.alpha] if isinstance(cfg.task.alpha, float) else cfg.task.alpha

    for d in datasets:
        print(f"Calib size: {d.no_calib_steps}")
    if cfg.dataset.add_config is not None and cfg.dataset.add_config.get('subset_before_prepare', False):
        datasets = list(filter(lambda d: d.ts_id in cfg.dataset.add_config['eval_subset'], datasets))
    # Prepare (Create, Train,..) underlying forcast models
    fc_service = _init_fc(fc_conf=cfg.model_fc, data_conf=cfg.dataset, task_conf=cfg.task, trainer_conf=cfg.trainer,
                          experiment_conf=cfg.experiment_data, datasets=datasets, fc_persist_dir=fc_persist_dir)
    datasets = fc_service.prepare(datasets, alphas)

    # Calibrate/Train UC
    uc_service = _init_uc(uc_conf=cfg.model_uc, data_conf=cfg.dataset, task_conf=cfg.task, fc_service=fc_service,
                          datasets=datasets, uc_persist_dir=uc_persist_dir, fc_state_dim=fc_service.fc_state_dim,
                          record_attention=cfg.evaluation['att_plot_vega'] or cfg.evaluation['att_hist_matplot'])
    uc_service.prepare(datasets, alphas, experiment_config=cfg.experiment_data, calib_trainer_config=cfg.trainer)

    # Evaluate
    if cfg.experiment_data.evaluate:
        if cfg.dataset.add_config is not None and 'eval_subset' in cfg.dataset.add_config:
            eval_subset = cfg.dataset.add_config['eval_subset']
            LOGGER.info(f"Evaluate only Subset of Dataset: {eval_subset}")
        else:
            LOGGER.info("Evaluate full dataset!")
            eval_subset = None
        Evaluator.evaluate(uc_service, datasets, alphas, cfg.evaluation, mix_mem_data=None,
                           evaluation_subset=eval_subset)
    else:
        model = cfg.model_uc._target_.split(".")[-1]
        if model in ['EnbPIModel']:
            Evaluator.evaluate_sota_on_validation(uc_service, datasets, alphas, no_calib=True)
        elif model in ['AdaptiveCI', 'NexCP', 'SPICModel', 'EpsSelectionPIStat', 'Bluecat', 'DefaultConformal', 'DefaultConformalPlusRecent']:
            Evaluator.evaluate_sota_on_validation(uc_service, datasets, alphas, no_calib=False)
        elif model == 'EpsPredictionHopfield' and cfg.model_uc.use_adaptiveci:
            Evaluator.evaluate_sota_on_validation(uc_service, datasets, alphas, no_calib=False, prefix="extraval")
        else:
            LOGGER.info("Skip Evaluation")
    wandb.finish()


if __name__ == "__main__":
    my_app()
