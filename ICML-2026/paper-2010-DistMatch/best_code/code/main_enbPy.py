import logging
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from loader.generator import DataGenerator
from utils.utils import set_seed

# Workaround for old dependencies of skygarden
import sys
import six
import sklearn.ensemble._forest as skforest
import sklearn.tree as sktree

LOGGER = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path='./../configuration', config_name='default_enbpi.yaml')
def my_app(cfg: DictConfig):
    # Workaround for old dependencies of skygarden
    sys.modules['sklearn.externals.six'] = six
    sys.modules['sklearn.ensemble.forest'] = skforest
    sys.modules['sklearn.tree.tree'] = sktree

    cfg = cfg.config
    cfg.experiment_data.experiment_dir = Path().cwd()
    set_seed(cfg.experiment_data.seed)
    _setup_wandb(cfg)
    datasets = DataGenerator.get_data_legacy(cfg.dataset, cfg.task)
    wandb.define_metric(f"alpha", summary='none')
    wandb.define_metric(f"Eval/mean_coverage", step_metric='alpha', summary='mean')
    wandb.define_metric(f"Eval/mean_pi_width", step_metric='alpha', summary='mean')
    overall_cov_mean = defaultdict(list)
    overall_with_mean = defaultdict(list)
    for dataset, dataset_info in datasets:
        ts_id = dataset_info['ts_id']
        prefix_per_ts = f"Eval_{ts_id}/"
        wandb.define_metric(f"{prefix_per_ts}mean_coverage", step_metric='alpha', summary='none')
        wandb.define_metric(f"{prefix_per_ts}mean_pi_width", step_metric='alpha', summary='none')
        alphas = cfg.task.alpha
        for alpha in alphas:
            LOGGER.info(f"Start run with data {ts_id} and alpha {alpha}")
            model = hydra.utils.instantiate(
                cfg.model_uc,
                alpha=alpha,
                dataset_name=cfg.dataset.dataset_type
            )
            model.fit_and_eval(dataset['train'].X, dataset['test'].X, dataset['train'].Y, dataset['test'].Y)
            cov_mean, width_mean, coverage, width = model.get_results()
            overall_cov_mean[alpha].append(cov_mean)
            overall_with_mean[alpha].append(width_mean)
            wandb.log({
                'alpha': alpha,
                f'{prefix_per_ts}mean_coverage': cov_mean,
                f'{prefix_per_ts}mean_pi_width': width_mean,
                #"intervals": wandb.Table(columns=["coverage", "width"], data=list(zip(coverage, width)))
            })
    for alpha, cov_means in overall_cov_mean.items():
        wandb.log({
            'alpha': alpha,
            'Eval/mean_coverage': np.mean(cov_means),
            'Eval/mean_pi_width': np.mean(overall_with_mean[alpha]),
        })
    wandb.finish()


def _setup_wandb(config):
    exp_data = config.experiment_data
    wandb.init(project=exp_data.project_name, name=HydraConfig.get().job.name, #dir=Path.cwd(),
               config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
               tags=config.wandb.tags, notes=config.wandb.notes, group=config.wandb.group)


if __name__ == "__main__":
    my_app()
