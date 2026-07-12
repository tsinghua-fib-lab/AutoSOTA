"""
BioFlowMatching Training Script with Hydra

Uses Hydra for configuration management - a mature, well-tested framework.

Usage:
    # defalt (config/biofm_config.yaml)
    python train_biofm_hydra.py
    
    # target path and config name
    python train_biofm_hydra.py --config-path=/path/to/config/dir --config-name=my_config
    python train_biofm_hydra.py --config-path=../other_configs --config-name=experiment1
    
    # Shortened syntax (using -cp and -cn)
    python train_biofm_hydra.py -cp /path/to/config/dir -cn my_config
    
    # Override config values (Hydra syntax)
    python train_biofm_hydra.py model.lr=1e-4 model.d_model=256
    python train_biofm_hydra.py -cp ./configs -cn exp1 model.lr=1e-4
    
    # Switch model type (using _target_)
    python train_biofm_hydra.py model._target_=models.FlowMatching.TrajectoryFlowMatching.biotfm_TimeMoE.BioFlowMatchingTimeMoEModule
    
    # Test only mode
    python train_biofm_hydra.py test_only=true ckpt_path=path/to/checkpoint.ckpt
    
    # Multirun (hyperparameter sweep)
    python train_biofm_hydra.py --multirun model.lr=1e-4,1e-5,1e-6

Installation:
    pip install hydra-core omegaconf
"""

import collections
import os
import sys
from pathlib import Path
from typing import Optional
import typing

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, ListConfig
import omegaconf
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
torch.serialization.add_safe_globals([DictConfig, ListConfig, omegaconf.base.ContainerMetadata, typing.Any, 
                                      dict, collections.defaultdict, omegaconf.nodes.AnyNode, omegaconf.base.Metadata, list, int])
# Add project root to path
sys.path.extend(['./', '../'])

from dataset.BioFMSeries import create_bioflowmatching_datamodule_from_bio
from dataset.get_train_val_test import get_train_val_test_splits, get_train_val_test_splits_ood


def create_model(cfg: DictConfig):
    """
    Create model from config using Hydra's instantiate.
    
    The model class is specified by the _target_ field in the config.
    All other fields in model config are passed as constructor arguments.
    
    Example config:
        model:
          _target_: models.FlowMatching.TrajectoryFlowMatching.biotfm.BioFlowMatchingModule
          d_model: 128
          lr: 1e-4
          ...
    """
    model = instantiate(cfg.model, _recursive_=False)
    return model

from dataset.BioSeries import BioSeriesDataModule
def create_datamodule(cfg: DictConfig):
    """Create data module from config."""
    data_cfg = cfg.data
    
    # Handle relative paths - convert to absolute using original cwd
    data_dir = data_cfg.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(hydra.utils.get_original_cwd(), data_dir)
    
    print("Loading datasets...")

    # config parameters
    val_ratio = data_cfg.get('val_ratio', 0.1)
    test_ratio = data_cfg.get('test_ratio', 0.2)
    seed = cfg.get('seed', 42)
    load_name = data_cfg.get('load_name', 'SysBio-Traj_index.csv')
    input_window = data_cfg.get('input_window', 96)
    output_window = data_cfg.get('output_window', 256)
    stride = data_cfg.get('stride', 1)
    transform_type = data_cfg.get('transform_type', 'minmax')
    condition_names = list(data_cfg.get('condition_names', ['bound_min', 'bound_max']))
    # Select appropriate splitting function based on OOD setting
    get_splits_func = get_train_val_test_splits if not data_cfg.get('OOD', False) else get_train_val_test_splits_ood
    if data_cfg.get('OOD', False): print("Using out-of-distribution data for training and evaluation.")
    else: print("Using standard data for training and evaluation.")

    if data_cfg.get('armd_train', False): 
        assert data_cfg.get('OOD', False) == True, "ARMD dataset must be used with OOD setting."
    assert not (data_cfg.get('armd_train', False) and data_cfg.get('train_AR_dataset', True)), \
        "ARMD dataset must be used with armd_train=False and train_AR_dataset=False."

    train_dataset, val_dataset, test_dataset = get_splits_func(
        data_dir=data_dir,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        load_name=load_name,
        input_window=input_window,
        output_window=output_window,
        stride=stride,
        variable_independent=True,
        transform_type=transform_type,
        condition_names=condition_names,
        armd_train=data_cfg.get('armd_train', False),
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    if data_cfg.get('data_type', 'BioTFM') == 'BioTFM':
        print("Creating BioFlowMatching DataModule...")
        datamodule = create_bioflowmatching_datamodule_from_bio(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            seq_length=data_cfg.get('input_window', 96) + data_cfg.get('output_window', 256) - 1,
            input_length=data_cfg.get('input_window', 96),
            pred_length=data_cfg.get('output_window', 256),
            stride=data_cfg.get('stride', 1),
            batch_size=data_cfg.get('batch_size', 256),
            num_workers=data_cfg.get('num_workers', 4),
            condition_names=list(data_cfg.get('condition_names', ['bound_min', 'bound_max'])),
            normalize=data_cfg.get('normalize', True),
            train_AR_dataset=data_cfg.get('train_AR_dataset', True),
            train_ARMD_dataset=data_cfg.get('armd_train', False),
        )
    elif data_cfg.get('data_type', 'BioTFM') == 'PointTrajectory':
        datamodule = BioSeriesDataModule(train_dataset, val_dataset, test_dataset, 
                            batch_size=data_cfg.get('batch_size', 256),
                            num_workers=data_cfg.get('num_workers', 4),
        )
    else:
        assert False, f"Unknown data_type: {data_cfg.get('data_type', 'BioTFM')}"
    
    return datamodule


from utils.train_ema import EMACallback
def create_callbacks(cfg: DictConfig) -> list:
    """Create training callbacks."""
    callbacks = []
    
    train_cfg = cfg.get('training', {})
    ckpt_cfg = cfg.get('checkpoint', {})
    
    # Checkpoint directory - use Hydra's output dir
    ckpt_dir = ckpt_cfg.get('save_dir', './checkpoints/')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='{epoch}-{val/mse:.4f}',
        monitor=ckpt_cfg.get('monitor', 'val/mse'),
        mode=ckpt_cfg.get('mode', 'min'),
        save_top_k=ckpt_cfg.get('save_top_k', 3),
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if train_cfg.get('early_stopping', True):
        early_stop = EarlyStopping(
            monitor=train_cfg.get('monitor', 'val/mse'),
            mode=train_cfg.get('mode', 'min'),
            patience=train_cfg.get('patience', 20),
            verbose=True,
        )
        callbacks.append(early_stop)
    
    if train_cfg.get('use_ema', False):
        print('Using EMA')
        callbacks.append(EMACallback(
            decay=train_cfg.get('ema_decay', 0.995),
            update_after_step=train_cfg.get('ema_update_after_step', 50),
            update_every=train_cfg.get('ema_update_every', 10),
        ))
    
    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))
    
    return callbacks


def create_logger(cfg: DictConfig):
    """Create WandB logger with full config."""
    log_cfg = cfg.get('logging', {})
    exp_name = cfg.get('experiment_name', 'biofm')
    
    if not log_cfg.get('wandb', False):
        raise NotImplementedError("Only WandB logging is implemented.")
    
    # Convert OmegaConf to plain dict for wandb
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    logger = WandbLogger(
        project=log_cfg.get('wandb_project', 'biofm'),
        name=exp_name,
        save_dir=log_cfg.get('log_dir', './logs/'),
        config=config_dict,  # Upload full config to wandb
    )
    
    return logger


def create_trainer(cfg: DictConfig, callbacks: list, logger) -> pl.Trainer:
    """Create PyTorch Lightning Trainer."""
    train_cfg = cfg.get('training', {})
    hw_cfg = cfg.get('hardware', {})
    
    return pl.Trainer(
        max_epochs=train_cfg.get('max_epochs', 200),
        check_val_every_n_epoch=train_cfg.get('check_val_every_n_epoch', 5),
        gradient_clip_val=train_cfg.get('gradient_clip_val', 1.0) if cfg.data.data_type == 'BioTFM' else None,
        accumulate_grad_batches=train_cfg.get('accumulate_grad_batches', 1),
        accelerator=hw_cfg.get('accelerator', 'gpu'),
        devices=list(hw_cfg.get('devices', [0])),
        precision=hw_cfg.get('precision', 32),
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        enable_progress_bar=True,
        limit_val_batches=train_cfg.get('limit_val_batches', 1.0),
        limit_test_batches=train_cfg.get('limit_test_batches', 1.0),
    )

from hydra.core.hydra_config import HydraConfig
import pandas as pd
def train(cfg: DictConfig) -> None:
    """Main training function."""
    
    # Print config
    print("=" * 60)
    print("Configuration:")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    
    # Get directories
    original_cwd = hydra.utils.get_original_cwd()
    print(f"Original working directory: {original_cwd}")
    print(f"Hydra output directory: {os.getcwd()}")
    
    # Set seed
    pl.seed_everything(cfg.get('seed', 42))
    
    # Create components
    print("\nCreating model...")
    model = create_model(cfg)

    if isinstance(model, (dict, omegaconf.dictconfig.DictConfig)): model = model['model']
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nCreating data module...")
    datamodule = create_datamodule(cfg)
    
    print("\nCreating trainer...")
    callbacks = create_callbacks(cfg)
    logger = create_logger(cfg)
    trainer = create_trainer(cfg, callbacks, logger)
    
    print("=" * 60)
    
    # Test only mode
    if cfg.get('test_only', False):
        if cfg.get('test_without_ckpt', False):
            print("\nTesting without checkpoint (zero-shot mode).")
            test_metrics = trainer.test(model, datamodule=datamodule)
            save_path = os.path.join(cfg.checkpoint.save_dir, "final_test_results_only_test.csv")
            df = pd.DataFrame(test_metrics)
            df.to_csv(save_path, index=False)
            print(f"\nFinal results saved to: {save_path}")

            print("\n" + "=" * 60)
            print("Done!")
            print("=" * 60)
            return

        ckpt_path = cfg.get('ckpt_path')
        if ckpt_path is None:
            ckpt_dir = cfg.checkpoint.get('save_dir', './checkpoints/')
            ckpt_path = os.path.join(ckpt_dir, 'last.ckpt')
        
        # Handle relative checkpoint path
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(original_cwd, ckpt_path)
        
        print(f"\nTesting from checkpoint: {ckpt_path}")
        # ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # with torch.serialization.safe_globals([torch._C._nn.gelu]):

        all_state = torch.load(ckpt_path, map_location='cpu')
        if 'ema_state_dict' in all_state:
            print("Loading EMA weights for testing.")
            all_state['state_dict'] = all_state['ema_state_dict']
            # save it temporarily
            temp_path = ckpt_path.replace('.ckpt', '_ema.ckpt')
            torch.save(all_state, temp_path)
            ckpt_path = temp_path
        test_metrics = trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=True)
        # trainer.validate(model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=True)
        save_path = os.path.join(cfg.checkpoint.save_dir, "final_test_results_only_test.csv")
    else:
        # === resume training ===
        resume_ckpt = cfg.training.get('resume_from_checkpoint', None)
        
        if resume_ckpt is not None:
            if not os.path.isabs(resume_ckpt):
                resume_ckpt = os.path.join(original_cwd, resume_ckpt)
            
            if os.path.exists(resume_ckpt):
                print(f"\nResuming training from checkpoint: {resume_ckpt}")
            else:
                raise FileNotFoundError(f"Checkpoint not found: {resume_ckpt}")
        
        # Training (ckpt_path=None -> train from scratch)
        print("\nStarting training...")
        trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)
        
        # Test with best checkpoint
        print("\nTesting with best checkpoint...")
        test_metrics = trainer.test(model, datamodule=datamodule, ckpt_path='best', weights_only=True)

        save_path = os.path.join(cfg.checkpoint.save_dir, "final_test_results.csv")
    df = pd.DataFrame(test_metrics)
    df.to_csv(save_path, index=False)
    print(f"\nFinal results saved to: {save_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

DEFAULT_CONFIG_PATH = "config"
DEFAULT_CONFIG_NAME = "biofm_config"


@hydra.main(version_base=None, config_path=DEFAULT_CONFIG_PATH, config_name=DEFAULT_CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """
    Main entry point with Hydra.
    
    Config path can be overridden via command line:
        python train_biofm_hydra.py --config-path=/absolute/path/to/config/dir --config-name=my_config
        python train_biofm_hydra.py -cp ./relative/path -cn config_name
    
    Note:
        - config-path (-cp): Directory containing config files (not the file itself)
        - config-name (-cn): Name of the config file without .yaml extension
    """
    train(cfg)


if __name__ == "__main__":
    main()
