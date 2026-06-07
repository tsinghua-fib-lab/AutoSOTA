#!/usr/bin/env python3
"""
In-context evaluation entry point for processed MASS datasets.

This script loads a trained MASS/Iris checkpoint, builds evaluation datasets
from the preprocessed ``*_image.npy`` / ``*_gt.npy`` layout, selects reference
examples according to ``--reference-mode`` and ``--ensemble-size``, then reports
Dice, ASD, and HD95 metrics.
"""

import os
import sys
import argparse
import logging
import inspect
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union
import json
import time

import models
import data
import metrics
from training.utils import verify_registry_integrity

verify_registry_integrity()

from utils.distributed import set_seed
from utils.registry import get_model, get_dataset
from training.evaluator import Evaluator

import warnings
warnings.filterwarnings("ignore")

def load_config_from_checkpoint(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Configuration dictionary
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'config' in checkpoint:
        return checkpoint['config']

    config_path = checkpoint_path.parent / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)

    config_yaml_path = checkpoint_path.parent / 'config.yaml'
    if config_yaml_path.exists():
        import yaml
        with open(config_yaml_path, 'r') as f:
            return yaml.safe_load(f)

    raise RuntimeError("No configuration found in checkpoint or checkpoint directory")


class EvaluationManager:
    """
    Manager class for model evaluation with optional prediction saving.
    """

    def __init__(self, args):
        """
        Initialize evaluation manager.

        Args:
            args: Command line arguments
        """
        self.args = args

        self.config = self._load_config()
        self._update_config_for_eval()
        self.device = self._setup_device()
        self._setup_logging()
        self.model = self._load_model()
        self._setup_data()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from checkpoint."""
        return load_config_from_checkpoint(self.args.checkpoint)

    def _update_config_for_eval(self):
        """Update configuration for evaluation."""
        self.config['mode'] = 'eval'

        if self.args.gpus:
            self.config['distributed'] = {
                'gpus': list(map(int, self.args.gpus.split(','))),
                'world_size': 1,
                'rank': 0,
                'proc_idx': 0,
                'gpus_per_node': len(self.args.gpus.split(','))
            }

        if self.args.save_predictions:
            checkpoint_dir = Path(self.args.checkpoint).parent
            # Prediction volumes are kept next to the checkpoint so each eval
            # run is traceable to the weights that produced it.
            self.predictions_dir = checkpoint_dir / f'predictions_{self.args.dataset if self.args.dataset else "all"}_{time.strftime("%Y%m%d_%H%M%S")}'
            self.predictions_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Predictions will be saved to: {self.predictions_dir}")

        if self.args.batch_size:
            if 'data' not in self.config:
                self.config['data'] = {}
            if 'loader' not in self.config['data']:
                self.config['data']['loader'] = {}
            self.config['data']['loader']['batch_size'] = self.args.batch_size

        if self.args.data_root:
            for split_name in ('train', 'val', 'incontext'):
                if split_name in self.config.get('data', {}):
                    self.config['data'][split_name]['data_root'] = self.args.data_root

        self.config.setdefault('incontext_evaluation', {})['ensemble_size'] = self.args.ensemble_size

        if self.args.disable_amp:
            self.config.setdefault('amp', {})['enabled'] = False
            self.config['amp']['dtype'] = 'float32'

        self.config.setdefault('evaluation', {})['calculate_surface_metrics'] = not self.args.skip_surface_metrics

    def _setup_device(self) -> torch.device:
        """Setup computing device."""
        if self.args.gpus:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpus
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')

        logging.info(f"Using device: {device}")
        return device

    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.args.verbose else logging.INFO

        checkpoint_dir = Path(self.args.checkpoint).parent
        dataset_suffix = self.args.dataset if self.args.dataset else "all"
        log_file = checkpoint_dir / f'evaluate_{dataset_suffix}_{time.strftime("%Y%m%d_%H%M%S")}.log'

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file)
            ]
        )

        logging.info(f"Evaluation log will be saved to: {log_file}")

    def _load_model(self) -> nn.Module:
        """Load model from checkpoint."""
        model_config = self.config.get('model', {}).copy()
        model_name = model_config.pop('type', 'iris')

        model_cls = get_model(model_name)
        model = model_cls(**model_config).to(self.device)

        checkpoint_path = Path(self.args.checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Checkpoints may contain both regular and EMA weights.
        if self.args.use_ema and 'ema_model' in checkpoint:
            state_dict = checkpoint['ema_model']
            logging.info("Loading EMA model weights")
        else:
            state_dict = checkpoint['model']
            logging.info("Loading regular model weights")

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        model.eval()

        logging.info(f"Model loaded from: {checkpoint_path}")
        logging.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

        return model

    def _setup_data(self):
        """Setup data loaders for in-context evaluation."""
        data_config = self.config.get('data', {})
        loader_config = data_config.get('loader', {})

        if self.args.dataset:
            eval_datasets = [self.args.dataset]
        else:
            incontext_config = data_config.get('incontext', {})
            val_config = data_config.get('val', {})
            eval_datasets = incontext_config.get('datasets', val_config.get('datasets', []))

        if not eval_datasets:
            raise ValueError("No datasets specified for evaluation")

        self.data_loaders = {}

        dataset_config = data_config.get('incontext', {}).copy()
        dataset_type = dataset_config.pop('type', 'MetaUniversalDataset')
        dataset_config.pop('datasets', None)
        dataset_cls = get_dataset(dataset_type)
        dataset_signature = inspect.signature(dataset_cls.__init__)
        accepts_extra_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in dataset_signature.parameters.values()
        )
        dataset_kwargs = set(dataset_signature.parameters) - {'self'}

        if self.args.data_root:
            dataset_config['data_root'] = self.args.data_root
        elif 'data_root' not in dataset_config:
            train_data_root = data_config.get('train', {}).get('data_root')
            val_data_root = data_config.get('val', {}).get('data_root')
            data_root = train_data_root or val_data_root
            if data_root is not None:
                dataset_config['data_root'] = data_root

        dataset_config.setdefault('training_size', self.config.get('training_size', [128, 128, 128]))
        # Processed arrays are [D, H, W], so target_spacing is stored as (z, y, x).
        dataset_config.setdefault('spacing', self.config.get('target_spacing', [1.5, 1.5, 1.5]))
        dataset_config['mode'] = 'test_incontext'
        # These two CLI options control how references are selected and how many
        # examples are averaged into each class embedding.
        dataset_config['reference_mode'] = self.args.reference_mode
        dataset_config['num_references_per_class'] = self.args.ensemble_size

        for dataset_name in eval_datasets:
            single_dataset_params = dict(dataset_config)
            single_dataset_params['datasets'] = [dataset_name]
            if not accepts_extra_kwargs:
                single_dataset_params = {
                    key: value
                    for key, value in single_dataset_params.items()
                    if key in dataset_kwargs
                }

            dataset = dataset_cls(**single_dataset_params)

            self.data_loaders[dataset_name] = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=loader_config.get('num_workers', 4),
                pin_memory=loader_config.get('pin_memory', True)
            )

            logging.info(f"Created evaluation loader for: {dataset_name}")

    def evaluate(self):
        """Run evaluation on all datasets."""
        all_results = {}

        for dataset_name, data_loader in self.data_loaders.items():
            logging.info(f"\nEvaluating on dataset: {dataset_name}")

            save_dir = None
            if self.args.save_predictions:
                save_dir = self.predictions_dir / dataset_name
                save_dir.mkdir(exist_ok=True)

            evaluator = Evaluator(
                self.model,
                data_loader,
                self.config,
                device=self.device,
                # The release evaluator always uses in-context segmentation.
                incontext=True,
                save_predictions=self.args.save_predictions,
                save_dir=save_dir
            )

            dice_means, asd_means, hd_means = evaluator.run()

            results = {
                'dice_means': dice_means.tolist(),
                'asd_means': asd_means.tolist(),
                'hd_means': hd_means.tolist(),
                'dice_overall': float(np.mean(dice_means)),
                'asd_overall': float(np.mean(asd_means)),
                'hd_overall': float(np.mean(hd_means)),
                'num_samples': len(data_loader)
            }

            all_results[dataset_name] = results

            self._log_results(dataset_name, results)

        self._save_results(all_results)

        return all_results

    def _log_results(self, dataset_name, results):
        """Log evaluation results."""
        logging.info(f"\nResults for {dataset_name}:")
        logging.info(f"  Overall Dice: {results['dice_overall']:.4f}")
        logging.info(f"  Overall ASD: {results['asd_overall']:.2f}")
        logging.info(f"  Overall HD95: {results['hd_overall']:.2f}")

        if 'dice_means' in results and results['dice_means']:
            logging.info(f"  Per-class Dice: {[f'{d:.4f}' for d in results['dice_means']]}")

        if 'num_samples' in results:
            logging.info(f"  Number of samples: {results['num_samples']}")

    def _save_results(self, all_results):
        """Save all evaluation results to JSON file."""
        checkpoint_dir = Path(self.args.checkpoint).parent
        dataset_suffix = self.args.dataset if self.args.dataset else "all"
        results_file = checkpoint_dir / f'evaluation_results_{dataset_suffix}_{time.strftime("%Y%m%d_%H%M%S")}.json'

        results_with_metadata = {
            'checkpoint': str(self.args.checkpoint),
            'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'reference_mode': self.args.reference_mode,
            'ensemble_size': self.args.ensemble_size,
            'use_ema': self.args.use_ema,
            'results': all_results
        }

        with open(results_file, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)

        logging.info(f"\nEvaluation results saved to: {results_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained Iris model')

    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint file')
    parser.add_argument('--dataset', type=str, default=None, help='Specific dataset to evaluate (default: all validation datasets from config)')
    parser.add_argument('--gpus', type=str, default='0', help='GPU IDs to use (e.g., "0,1,2")')
    parser.add_argument('--use-ema', action='store_true', help='Use EMA model weights if available')
    parser.add_argument('--save-predictions', action='store_true', help='Save predictions as nii.gz files')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for evaluation (default: from config)')
    parser.add_argument('--data-root', type=str, default=None, help='Override data root stored in the checkpoint config')
    parser.add_argument('--reference-mode', choices=['random', 'fixed'], default='random', help='Reference selection mode')
    parser.add_argument('--ensemble-size', type=int, default=1, help='Number of reference examples per class')
    parser.add_argument('--disable-amp', action='store_true', help='Disable AMP during evaluation')
    parser.add_argument('--skip-surface-metrics', action='store_true', help='Skip surface distance metrics calculation (faster evaluation)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--seed', type=int, default=46, help='Random seed')

    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    set_seed(args.seed)
    evaluator = EvaluationManager(args)
    results = evaluator.evaluate()

    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    for dataset_name, dataset_results in results.items():
        print(f"\n{dataset_name}:")
        print(f"  Dice: {dataset_results['dice_overall']:.4f}")
        print(f"  ASD:  {dataset_results['asd_overall']:.2f} mm")
        print(f"  HD95: {dataset_results['hd_overall']:.2f} mm")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
