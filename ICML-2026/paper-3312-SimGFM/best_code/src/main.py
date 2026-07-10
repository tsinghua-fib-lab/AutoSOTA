"""Main entry point for training and testing the graph discrete flow model."""
import warnings

import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from src.checkpoint_utils import validate_checkpoint_path, validate_train_mode, load_model_config_from_checkpoint
from src.factories import ModelContextBuilder
from src.runners import is_train_mode, run_training, run_sampling_single, run_sampling_all
from src.graph_discrete_flow_model import GraphDiscreteFlowModel
from src.ema import TorchEMACallback
from src.wandb_resume_callback import WandbResumeCallback

torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True

warnings.filterwarnings("ignore", category=PossibleUserWarning)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.train.seed)
    
    # Load model config from checkpoint if resuming or testing
    ckpt_path_to_load = cfg.general.resume if cfg.general.resume else cfg.general.test_only
    if ckpt_path_to_load:
        print(f"Attempting to load model config from checkpoint: {ckpt_path_to_load}")
        model_config = load_model_config_from_checkpoint(ckpt_path_to_load)
        if model_config is not None:
            # Override cfg.model with checkpoint config
            saved_config_omega = OmegaConf.create(model_config)
            cfg.model = OmegaConf.merge(cfg.model, saved_config_omega)
            print("Model config has been overridden with checkpoint values")
        else:
            print("Using current config from yaml files")
    
    # Build model and datamodule
    model_context_builder = ModelContextBuilder(cfg)
    model_kwargs = model_context_builder.build()
    datamodule = model_context_builder.datamodule
    utils.create_folders(cfg)
    model = GraphDiscreteFlowModel(cfg=cfg, **model_kwargs)
    print("Model created")
    
    # Setup callbacks
    callbacks = [WandbResumeCallback()]  # Auto-save/restore wandb run ID
    
    # Prefer model.ema_decay; fall back to train.ema_decay for backward compatibility
    ema_decay = cfg.model.ema_decay
    if ema_decay > 0:
        print("Initializing EMA callback")
        ema_callback = TorchEMACallback(decay=ema_decay)
        callbacks.append(ema_callback)
    
    # Add ModelCheckpoint callback for training mode
    if is_train_mode(cfg):
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.model.model_weights_dir,
            filename="epoch_{epoch}",
            save_top_k=-1,
            every_n_epochs=cfg.general.check_val_every_n_epochs,
            save_on_train_epoch_end=True,  # Save after validation
            auto_insert_metric_name=False,  # Prevent adding metric names to filename
        )
        callbacks.append(checkpoint_callback)
        print(f"ModelCheckpoint callback added: saving to {cfg.model.model_weights_dir} every {cfg.general.check_val_every_n_epochs} epochs")
    
    # Setup trainer
    name = cfg.general.name
    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    wandb_logger = utils.make_wandb_logger(cfg)
    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
        accelerator="gpu" if use_gpu else "cpu",
        devices=cfg.general.gpus if use_gpu else 1,
        max_epochs=cfg.train.n_epochs,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=name == "debug",
        enable_progress_bar=False,
        callbacks=callbacks,
        log_every_n_steps=50 if name != "debug" else 1,
        logger=[wandb_logger],
        num_sanity_val_steps=0
    )
    
    # Determine mode and run appropriate workflow
    if is_train_mode(cfg):
        # Training mode
        model_weights_dir = cfg.model.model_weights_dir
        validate_train_mode(model_weights_dir, cfg.general.resume)
        run_training(cfg, model, datamodule, trainer)
    else:
        # Sampling/testing mode
        ckpt_path = cfg.general.test_only
        validate_checkpoint_path(ckpt_path)
        
        if cfg.general.evaluate_all_checkpoints:
            run_sampling_all(cfg, model, datamodule, trainer)
        else:
            run_sampling_single(cfg, model, datamodule, trainer)
    
    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
