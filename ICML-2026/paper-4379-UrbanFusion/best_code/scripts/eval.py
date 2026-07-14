#!/usr/bin/env python3
"""
Description: PyTorch Lightning-based evaluation pipeline with Hydra
configuration. This script loads a pre-trained model checkpoint and evaluates
it on a specified test dataset. It dynamically initializes the dataset, model,
trainer, and loggers based on the provided configuration. Additionally, it
logs hyperparameters, records evaluation metrics, and ensures consistency
across different evaluation runs.
"""

import os
import types
from typing import Any, Dict, List, Tuple

import hydra
import rootutils
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from srl.utils.logging import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# -------------------------------------------------------------------------- #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import
#        utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root
# dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# -------------------------------------------------------------------------- #

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls
    the behavior during failure. Useful for multiruns, saving info about the
    crash, etc.

    Parameters
    ----------
    cfg : DictConfig
        A configuration dictionary composed by Hydra.

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        A tuple containing:
        - A dictionary with metrics.
        - A dictionary with instantiated objects (datamodule, model, trainer,
          loggers, etc.).
    """

    # Check if checkpoint path is provided
    assert cfg.ckpt_path

    # Instantiate modules
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # Test model
    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(
    # model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path
    # )

    metric_dict = trainer.callback_metrics

    if cfg.get("save_attention", False):

        def custom_attention_forward(self, x: torch.Tensor) -> torch.Tensor:
            B, N, C = x.shape
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
            # Use non-fused attention
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            # Save attention weights for later inspection
            self.attn_weights = attn
            attn = self.attn_drop(attn)
            x = attn @ v
            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        # Model is a timm ViT with blocks attribute
        for blk in model.model.blocks:
            blk.attn.forward = types.MethodType(
                custom_attention_forward, blk.attn
            )

        def make_custom_predict_step(mask_indices: list):
            """
            Returns a custom predict_step function that uses the given
            mask_indices.
            """

            def custom_predict_step(
                self, batch: dict, batch_idx: int, dataloader_idx: int = 0
            ):
                # Remove any unwanted keys
                if "gsv_img" in batch:
                    del batch["gsv_img"]

                # Process the batch (e.g., tokenization and moving tensors
                # to device)
                inputs, _ = self._process_batch(batch)

                # Forward pass with the provided mask_indices
                reps = self(
                    inputs,
                    mask_indices=mask_indices,
                    return_representations=False,
                    return_modality_tokens=False,
                )

                # Collect attention scores from each transformer block
                attention = [
                    blk.attn.attn_weights for blk in self.model.blocks
                ]

                # Return both predictions and attention
                return {"predictions": reps, "attention": attention}

            return custom_predict_step

        # Define your different mask cases.
        mask_cases = (
            [
                [i] for i in range(len(trainer.datamodule.modalities))
            ]  # Each modality separately
            + [[]]  # No mask
            + [list(range(1, 5))]  # A specific range
        )

        # Loop over the different mask cases
        for mask_indices in mask_cases:
            # Override the predict_step method with our custom version that
            # uses mask_indices
            model.predict_step = types.MethodType(
                make_custom_predict_step(mask_indices), model
            )

            # Call predict to get outputs from each batch
            predictions_list = trainer.predict(
                model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path
            )

            # Extract attention scores from each batch.
            # (The structure of each element in predictions_list is assumed to
            # be a dict.)
            attention_scores = [
                batch_output["attention"]
                for batch_output in predictions_list
                if "attention" in batch_output
            ]

            # Create a name string from the mask_indices for the filename.
            if mask_indices:
                mask_name = "_".join(map(str, mask_indices))
            else:
                mask_name = "none"

            # Define the save path.
            # For example, using a directory from config (defaulting to
            # current directory)
            save_dir = cfg.get("attention_save_path", ".")
            filename = f"attention_scores_mask_{mask_name}.pth"
            save_path = os.path.join(save_dir, filename)

            # Save the attention scores to disk.
            torch.save(attention_scores, save_path)
            print(
                f"Saved attention scores for mask indices {mask_indices} "
                f"to {save_path}"
            )

    return metric_dict, object_dict


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="eval.yaml"
)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for evaluation.

    Parameters
    ----------
    cfg : DictConfig
        A configuration dictionary composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
