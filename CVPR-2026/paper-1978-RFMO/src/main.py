import os
import argparse
import logging
from distutils.util import strtobool
from typing import Optional

try:
    import wandb
except ImportError:
    wandb = None
import torch
import numpy as np

from utils import load_config, init_logging, fix_seed
from dataloaders import get_dataloader
from predictor import get_predictor
from trainer import Trainer


torch.multiprocessing.set_sharing_strategy('file_system')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(
        config: dict,
        n_runs: int,
        use_wandb: bool=False,
        project_name: Optional[str]=None,
        exp_name: Optional[str]=None,
        test_only: bool=False,
    ):
    config['trainer']['save_dir'] = os.path.join(CURRENT_DIR, '..', config['trainer']['save_dir'])

    init_logging(os.path.join(config['trainer']['save_dir'], 'log.txt'))
    fix_seed(config['seed'])

    logger = logging.getLogger(__name__)
    logger.info(f'Loaded config: {config}')

    train_loader = get_dataloader(config['dataset']['train'])
    val_loader = get_dataloader(config['dataset']['val'])
    test_loader = get_dataloader(config['dataset']['test'])

    if use_wandb:
        assert n_runs == 1, 'n_runs must be 1 when use_wandb is True'
        wandb.init(
            project=project_name,
            name=exp_name,
            config=config
        )

    def run(use_log: bool=True):
        predictor = get_predictor(config['predictor'])
        trainer = Trainer(
            config=config['trainer'],
            predictor=predictor,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            use_log=use_log,
            use_wandb=use_wandb,
        )
        if not test_only:
            trainer.train()

        trainer.test()
        return trainer.get_metrics()

    if n_runs == 1:
        run()
    else:
        metrics_per_run: dict[str, list[float]] = {}

        for i in range(n_runs):
            logger.info(f'Run {i+1}/{n_runs}')
            results = run(use_log=False)
            torch.cuda.empty_cache()
            for metric_name in results:
                if metric_name not in metrics_per_run:
                    metrics_per_run[metric_name] = []
                metrics_per_run[metric_name].append(results[metric_name])
                logger.info(f'{metric_name}: {results[metric_name]:.4f}')

        logger.info('-'*5 + f' Final Results: {config["dataset"]["train"]["dataset_name"]} ' + '-'*5)
        for metric_name, values in metrics_per_run.items():
            logger.info(f'{metric_name}: {np.mean(values):.4f} ± {np.std(values):.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--use_wandb', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--project_name', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--test-only', type=lambda x: bool(strtobool(x)), default=False)
    ## Trainer
    parser.add_argument('--trainer.device', type=str, default='cpu')
    parser.add_argument('--trainer.batch_size', type=int, default=None)
    parser.add_argument('--trainer.n_epochs', type=int, default=None)
    parser.add_argument('--trainer.split_val', type=lambda x: bool(strtobool(x)), default=None)
    parser.add_argument('--predictor.num_samples_to_infer', type=int, default=None)

    args = parser.parse_args()

    config = load_config(args.config, args)
    main(
        config=config,
        n_runs=args.n_runs,
        use_wandb=args.use_wandb,
        project_name=args.project_name,
        exp_name=args.exp_name,
        test_only=args.test_only,
    )
