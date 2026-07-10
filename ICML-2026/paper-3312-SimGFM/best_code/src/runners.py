"""Training and sampling/testing runner functions."""
import pathlib

from omegaconf import DictConfig
from pytorch_lightning import Trainer

from src.checkpoint_utils import collect_checkpoints_from_directory
from src.datasets.abstract_dataset import AbstractDataModule
from src.graph_discrete_flow_model import GraphDiscreteFlowModel


def is_train_mode(cfg: DictConfig) -> bool:
    """Check if running in training mode."""
    return cfg.general.test_only is None


def run_training(
    cfg: DictConfig,
    model: GraphDiscreteFlowModel,
    datamodule: AbstractDataModule,
    trainer: Trainer
) -> None:
    """Run model training."""
    print("Starting training...")
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)


def run_sampling_single(
    cfg: DictConfig,
    model: GraphDiscreteFlowModel,
    datamodule: AbstractDataModule,
    trainer: Trainer
) -> None:
    """Run sampling/testing on a single checkpoint."""
    ckpt_path = cfg.general.test_only
    print(f"Testing with checkpoint: {ckpt_path}")
    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


def run_sampling_all(
    cfg: DictConfig,
    model: GraphDiscreteFlowModel,
    datamodule: AbstractDataModule,
    trainer: Trainer
) -> None:
    """Run sampling/testing on all checkpoints in the same directory.
    
    All checkpoints in the same directory share the same model config,
    so we only need to load the config once from the test_only checkpoint
    and reuse the same model for all checkpoints.
    """
    main_ckpt_path = cfg.general.test_only
    directory = pathlib.Path(main_ckpt_path).parent
    
    print(f"Evaluating all checkpoints in directory: {directory}")
    print(f"Model config already loaded from: {main_ckpt_path}")
    print("All checkpoints will use the same model config (already loaded in main.py)")
    
    # Collect and test all checkpoints
    checkpoints = collect_checkpoints_from_directory(str(directory))
    
    for epoch, ckpt_path in checkpoints:
        print(f"\n{'='*80}")
        print(f"Testing checkpoint (epoch {epoch}): {ckpt_path}")
        print('='*80)
        
        model.log_step = epoch
        trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)

