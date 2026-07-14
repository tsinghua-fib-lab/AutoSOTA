#!/usr/bin/env python3
"""
Description: This script defines a PyTorch Lightning trainer for
a UrbanFusion model. It handles model training, validation. The
trainer processes different modalities, applies masking strategies, and
optimizes the model using optimizer and learning rate scheduler.
"""

import os
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from transformers import AutoTokenizer, CLIPTokenizer

from scripts.eval_utils.eval_metrics import (
    clip_accuray_router,
    compute_r_squared_router,
    count_correct_predictions_router,
    get_correct_count_data_structure,
    r_squared_reduction,
    update_correct_count,
    update_r_squared,
)
from scripts.eval_utils.eval_representation_space import (
    plot_milan_representations,
    plot_synthetic_representations,
)
from scripts.loss.clip_loss import CLIPLoss
from scripts.loss.multi_modal_clip_loss import MultiModalCLIPLoss
from srl.encoders.POI_encoder.text_encoders import TextTransformer

# Metrics to log during training, validation, and testing
# Needed for correct logging if of CSVLogger is used with DDP.
metrics_train_val_test = {
    "train_loss": 0,
    "val_loss": 0,
    "val_accuracy": 0,
    "val_r2_metric": 0,
    "test_loss": 0,
    "test_accuracy": 0,
    "test_r2_metric": 0,
}


class LightningMultiModalModel(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        loss: nn.Module,
        compile: bool = False,
        val_metrics: list = [],
        test_metrics: list = [],
        r2_metric_distance: str = "cosine",
        top_k_accuracy: int = 5,
        plot_representation_space: dict = {},
        modality_reconstruction: bool = False,
        cartesian_p_final: float = 0.1,
        queue_size: int = 0,
        modality_dropout: float = 0.0,
        raw_representations: bool = False,
    ) -> None:
        """
        Initializes the LightningMultiModalModel class.

        Parameters
        ----------
        model : nn.Module
            Multi-modal transformer model.
        optimizer : torch.optim.Optimizer
            Optimizer for the model.
        scheduler : torch.optim.lr_scheduler.LambdaLR
            Learning rate scheduler for the optimizer.
        loss : nn.Module
            Loss function for the model.
        compile : bool, optional
            Whether to compile the model, by default False.
            Only used when stage is 'fit'. Use for speedup in training
            whith high-performance GPUs.
        val_metrics : list, optional
            Validation metrics to compute, by default [].
            Supported metrics are "accuracy" and "r2_metric".
        test_metrics : list, optional
            Test metrics to compute, by default [].
            Supported metrics are "accuracy" and "r2_metric".
        r2_metric_distance : str, optional
                Distance metric to use for R2-like metric, by default "cosine".
                Only used when val_metrics or test_metrics contain "r2_metric".
        top_k_accuracy : int, optional
            The value of k for top-k accuracy, by default 5.
        plot_representation_space : dict, optional
            Configuration for plotting the representation space, by default
            {}. It should contain the following keys:
            - "plot_val": bool, whether to plot the representation space
              during validation.
            - "plot_test": bool, whether to plot the representation space
                during testing.
            - "val_n_epochs": int, number of epochs between validation plots.
            - "test_n_epochs": int, number of epochs between test plots.
            - "mask_indices": list, indices of modalities to mask during
                representation space plotting.
            - "token_index": int, index of the token to use for plotting.
            - "title": str, title for the plot.
            - "save_plots": bool, whether to save the plots.
            - "show_plots": bool, whether to show the plots.
            - "save_dir": str, directory to save the plots.
        modality_reconstruction : bool, optional
            Whether to reconstruct modalities during training, by default
            False. If True, the model will reconstruct the masked modalities
            during training.
        cartesian_p_final : float, optional
            Final probability for the Cartesian product of modalities, by
            default 0.1.
        queue_size : int, optional
            Size of the queue for storing past samples, by default 0.
        modality_dropout : float, optional
            Dropout rate for the modality inputs, by default 0.0.
        raw_representations : bool, optional
            Whether to return raw representations from the model, by default
            False.
        """
        super().__init__()

        # Store parameters
        self.model = model
        self.loss = loss
        self.torch_compile = compile
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.r2_metric_distance = r2_metric_distance
        self.top_k_accuracy = top_k_accuracy
        self.plot_representation_space = plot_representation_space
        self.modality_reconstruction = modality_reconstruction
        self.cartesian_p_final = cartesian_p_final
        self.contrastive_dim = self.model.head_contrastive_dim
        self.queue_size = queue_size
        self.modality_dropout = modality_dropout
        self.raw_representations = raw_representations

        # Check if loss function is compatible with model
        assert not (
            model.only_cls is False and isinstance(self.loss, CLIPLoss)
        ), (
            "Error: CLIPLoss cannot be used when model.only_cls is False. "
            "Use MultiModalCLIPLoss instead."
        )
        assert not (
            model.only_cls is True
            and isinstance(self.loss, MultiModalCLIPLoss)
        ), (
            "Error: MultiModalCLIPLoss cannot be used when model.only_cls is"
            " True. Use CLIPLoss instead."
        )

        # Optimizer and scheduler configurations
        self.optimizer = optimizer(self.model.parameters())
        self.scheduler = scheduler(self.optimizer)

        # Get the text tokenizer
        text_encoder = None
        if hasattr(self.model, "encoders"):
            for encoder in self.model.encoders:
                if hasattr(encoder, "clip_text_model"):
                    text_encoder = encoder
                    break
                if isinstance(encoder, TextTransformer):
                    text_encoder = encoder
                    break

        if text_encoder is not None and hasattr(text_encoder, "model_name"):
            model_name = text_encoder.model_name
        else:
            model_name = "openai/clip-vit-base-patch32"

        if model_name == "openai/clip-vit-base-patch32":
            self.tokenizer = CLIPTokenizer.from_pretrained("/models/hf/hub/models--openai--clip-vit-large-patch14/snapshots/abc123")
        elif model_name == "BAAI/bge-small-en-v1.5":
            self.tokenizer = AutoTokenizer.from_pretrained("/models/hf/hub/models--BAAI--bge-small-en-v1.5/snapshots/abc123")
        elif model_name == "openai/clip-vit-large-patch14":
            self.tokenizer = CLIPTokenizer.from_pretrained("/models/hf/hub/models--openai--clip-vit-large-patch14/snapshots/abc123")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Check if model returns modality tokens
        if self.loss.label_smoothing_type in ["fixed", "learned"]:
            self.return_modality_tokens = True
        else:
            self.return_modality_tokens = False

        # Check if model output is already normalized
        if self.model.normalize_embedding:
            self.normalize_model_output = False

        if queue_size > 0:
            self._initialize_queue(queue_size)

    def _process_batch(self, batch: dict) -> list:
        """
        Processes a batch dictionary by moving each tensor to the device and
        tokenizing text. Extracts the batch size from the first tensor.

        Parameters
        ----------
        batch : dict
            Dictionary containing modalities (e.g., 'image', 'numeric',
            'text').

        Returns
        -------
        processed : list
            A list with processed elements in the same order as they appear in
            the input dict.
        batch_size : int
            Batch size of the input batch.
        """
        processed = []
        batch_size = None
        for key, value in batch.items():
            # For text, tokenize first and move all tensors to the device.
            if key.lower() == "text":
                tokenized = self.tokenizer(
                    value, padding=True, return_tensors="pt"
                ).to(self.device)
                processed.append(tokenized)
                if batch_size is None:
                    batch_size = tokenized.input_ids.size(0)
            elif key.lower() == "coords_original":
                pass
            elif key.lower() == "gsv_img":
                pass
            else:
                processed.append(value.to(self.device))
                if batch_size is None:
                    batch_size = value.size(0)

        return processed, batch_size

    def _generate_mask_indices(self, num_modalities: int = 3) -> list:
        """
        Generates the mask indices on rank 0 and broadcasts them to all ranks.
        Returns the same mask indices across all ranks.

        Parameters
        ----------
        num_modalities : int, optional
            Number of modalities, by default 3.

        Returns
        -------
        list
            List of mask indices.
        """

        # Check if distributed is initialized.
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            # Only rank 0 generates the random masks.
            if rank == 0:
                all_indices = np.arange(num_modalities)
                num_to_mask = np.random.randint(1, num_modalities)
                mask_indices_0 = np.random.choice(
                    all_indices, size=num_to_mask, replace=False
                ).tolist()
                mask_indices_1 = np.setdiff1d(
                    all_indices, mask_indices_0
                ).tolist()
                mask = (mask_indices_0, mask_indices_1)
            else:
                mask = None
            # Broadcast the mask object from rank 0 to all other ranks.
            mask_list = [mask]
            torch.distributed.broadcast_object_list(mask_list, src=0)
            return mask_list[0]
        else:
            # If not in a distributed setting
            all_indices = np.arange(num_modalities)
            num_to_mask = np.random.randint(1, num_modalities)
            mask_indices_0 = np.random.choice(
                all_indices, size=num_to_mask, replace=False
            ).tolist()
            mask_indices_1 = np.setdiff1d(all_indices, mask_indices_0).tolist()
            return (mask_indices_0, mask_indices_1)

    def _validation_masks(self, num_modalities: int) -> dict:
        """
        Returns a dictionary with three predefined pairs of mask indices.
        Dataset specific for having no randomness in validation and testing.

        Parameters
        ----------
        num_modalities : int
            Number of modalities.

        Returns
        -------
        dict
            Dictionary with three predefined pairs of mask indices.
        """
        self.len_mask_dict = len(self.trainer.datamodule.modalities)
        masks = self._generate_two_mask_combinations(num_modalities)
        return masks

    @staticmethod
    def _generate_two_mask_combinations(n: int) -> list:
        """
        Generates all unique combinations of two masks for n modalities.

        This function ensures that each combination is unique and does not
        repeat the same combination in reverse order.

        Parameters
        ----------
        n : int
            Number of modalities.

        Returns
        -------
        list
            List of unique combinations of two masks.
        """
        modalities = set(range(n))
        seen = set()
        result = {}
        idx = 0

        for r in range(1, n // 2 + 1):
            for combo in combinations(modalities, r):
                mask1 = set(combo)
                mask2 = modalities - mask1

                # Ensure canonical representation to avoid duplicates
                canonical = frozenset([frozenset(mask1), frozenset(mask2)])
                if canonical not in seen:
                    seen.add(canonical)
                    result[idx] = [sorted(mask1), sorted(mask2)]
                    idx += 1

        return result

    def forward(
        self,
        inputs: list,
        mask_indices: list = None,
        return_representations: bool = False,
        return_modality_tokens: bool = False,
        return_backbone_features: bool = False,
        modality_dropout: float = 0.0,
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        inputs : list
            List of input tensors of different modalities.
        mask_indices : list, optional
            List of indices to mask, by default None.
        return_modality_tokens : bool, optional
            Whether to return modality tokens, by default False.
            Used for CLIP loss if label_smoothing_type is "fixed"
            or "learned".

        Returns
        -------
        torch.Tensor
            Output tensor of the model.
        """
        return self.model(
            inputs,
            mask_indices=mask_indices,
            return_representations=return_representations,
            return_modality_tokens=return_modality_tokens,
            return_backbone_features=return_backbone_features,
            modality_dropout=modality_dropout,
        )

    def training_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """
        Training step of the model.

        Parameters
        ----------
        batch : dict
            Dictionary containing modalities (e.g., 'image', 'numeric',
            'text').
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            Loss tensor.
        """

        # Tokenize text and move tensors to device
        inputs, batch_size = self._process_batch(batch)

        if self.trainer.datamodule.ablation_masking is not None:

            # Detect which modality indices are actually present (not all‑NaN)
            present = []
            for i, x in enumerate(inputs):
                if isinstance(x, torch.Tensor):
                    if not torch.isnan(x).all():
                        present.append(i)
                else:
                    present.append(i)

            # Randomly partition the *present* modalities into two keep‐lists
            n = len(present)
            k = np.random.randint(1, n)  # how many present go to view0
            keep0 = np.random.choice(present, size=k, replace=False).tolist()
            keep1 = [i for i in present if i not in keep0]

            # 3) now invert to get the actual mask_indices:
            # every modality not in keep0 is masked in view0, and vice versa
            all_idx = list(range(len(inputs)))
            mask0 = [i for i in all_idx if i not in keep0]
            mask1 = [i for i in all_idx if i not in keep1]

        else:
            # Generate  mask indices
            mask0, mask1 = self._generate_mask_indices(
                num_modalities=len(inputs)
            )

        # Forward pass for siamese model and compute loss
        loss = self._compute_loss(
            inputs,
            mask0,
            mask1,
            dataloader_idx,
            modality_dropout=self.modality_dropout,
        )

        # Log loss
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        return loss

    def on_train_epoch_end(self):
        """
        Called at the end of the training epoch.
        """
        enc = self.model
        self._log_memory("train")
        max_epochs = self.trainer.max_epochs

        epoch = self.current_epoch + 1
        if max_epochs is None or epoch >= max_epochs:
            p_t = self.cartesian_p_final
        else:
            frac = epoch / float(max_epochs - 1)
            p_t = 1.0 - frac * (1.0 - self.cartesian_p_final)
        for m in enc.modules():
            if isinstance(m, nn.Dropout2d):
                m.p = p_t

    def on_validation_start(self) -> None:
        """
        Initializes the correct count and r_squared_sum data structures.
        Used during validation for tracking accuracy and R2 like metric
        across batches.
        """
        if (
            self.trainer.datamodule.dataset_name == "PlacePulse2"
            or self.trainer.datamodule.dataset_name == "SyntheticPID"
        ):
            self.total_samples = [0, 0]
        else:
            self.total_samples = 0
        if "accuracy" in self.val_metrics:
            if (
                self.trainer.datamodule.dataset_name == "PlacePulse2"
                or self.trainer.datamodule.dataset_name == "SyntheticPID"
            ):
                self.correct_count0 = get_correct_count_data_structure(
                    self.model.only_cls
                )
                self.correct_count1 = get_correct_count_data_structure(
                    self.model.only_cls
                )
                self.correct_count = [self.correct_count0, self.correct_count1]
            else:
                self.correct_count = get_correct_count_data_structure(
                    self.model.only_cls
                )
        if "r2_metric" in self.val_metrics:
            if (
                self.trainer.datamodule.dataset_name == "PlacePulse2"
                or self.trainer.datamodule.dataset_name == "SyntheticPID"
            ):
                self.r_squared_sum0 = get_correct_count_data_structure(
                    self.model.only_cls
                )
                self.r_squared_sum1 = get_correct_count_data_structure(
                    self.model.only_cls
                )
                self.r_squared_sum = [self.r_squared_sum0, self.r_squared_sum1]
            else:
                self.r_squared_sum = get_correct_count_data_structure(
                    self.model.only_cls
                )

    def validation_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """
        validation_step step of the model.

        Parameters
        ----------
        batch : dict
            Dictionary containing modalities (e.g., 'image', 'numeric',
            'text').
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            Loss tensor.
        """

        # Tokenize text and move tensors to device
        inputs, batch_size = self._process_batch(batch)

        # Generate  mask indices
        mask_dict = self._validation_masks(num_modalities=len(inputs))

        # Calculate loss across different mask pairs
        loss = 0
        for mask0, mask1 in mask_dict.values():
            # Forward pass for siamese model and compute loss
            loss += self._compute_loss(inputs, mask0, mask1, dataloader_idx)

        # Average loss across different mask pairs
        loss /= len(mask_dict)

        if (
            self.trainer.datamodule.dataset_name == "PlacePulse2"
            or self.trainer.datamodule.dataset_name == "SyntheticPID"
        ):
            self.log(
                f"val_loss_dataloader_{dataloader_idx}",
                loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )
        else:
            # Log loss
            self.log(
                "val_loss",
                loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )

        return loss

    def on_validation_epoch_end(self) -> None:
        """
        Calculates the accuracy and R2-like metric at the end of the
        validation epoch. Optionally plots the representation space after PCA
        and t-SNE.
        """
        if (
            self.trainer.datamodule.dataset_name == "PlacePulse2"
            or self.trainer.datamodule.dataset_name == "SyntheticPID"
        ):
            num_batches0 = len(self.trainer.datamodule.val_dataloader()[0])
            num_batches1 = len(self.trainer.datamodule.val_dataloader()[1])
            num_batches = [num_batches0, num_batches1]
        else:
            num_batches = sum(self.trainer.num_val_batches)

        self._log_epoch_metrics(
            metrics=set(self.val_metrics),
            prefix="val",
            num_batches=num_batches,
        )
        self._log_memory("val")
        representations = None

        # Plot the representation space if requested
        if (
            self.plot_representation_space["plot_val"]
            and self.current_epoch
            % self.plot_representation_space["val_n_epochs"]
            == 0
        ):
            if self.trainer.datamodule.dataset_name == "SyntheticMultiModal":

                representations, concatenated_modalities = (
                    self.create_representations(
                        self.trainer.datamodule.val_dataloader(),
                        mask_indices=self.plot_representation_space[
                            "mask_indices"
                        ],
                        return_representations=True,
                    )
                )
                text = concatenated_modalities["text"]
                images = concatenated_modalities["image"]
                plot_synthetic_representations(
                    representations=representations,
                    images=images,
                    text_data=text,
                    token_index=self.plot_representation_space["token_index"],
                    epoch=self.current_epoch,
                    stage="validation",
                    only_cls=self.model.only_cls,
                    title=self.plot_representation_space["title"],
                    save_plots=self.plot_representation_space["save_plots"],
                    show_plots=self.plot_representation_space["show_plots"],
                    save_dir=self.plot_representation_space["save_dir"],
                )
            elif self.trainer.datamodule.dataset_name == "Milan":
                representations, concatenated_modalities = (
                    self.create_representations(
                        self.trainer.datamodule.val_dataloader(),
                        mask_indices=self.plot_representation_space[
                            "mask_indices"
                        ],
                        return_representations=True,
                    )
                )
                coordinates = concatenated_modalities["coordinates"]
                plot_milan_representations(
                    representations=representations,
                    modality_coordinates=coordinates,
                    token_index=self.plot_representation_space["token_index"],
                    epoch=self.current_epoch,
                    stage="validation",
                    only_cls=self.model.only_cls,
                    title=self.plot_representation_space["title"],
                    save_plots=self.plot_representation_space["save_plots"],
                    show_plots=self.plot_representation_space["show_plots"],
                    save_dir=self.plot_representation_space["save_dir"],
                )
            else:
                print(
                    "Plotting the representation space is only supported for "
                    "SyntheticMultiModalDataset."
                )
            if self.plot_representation_space["save_representations_val"]:
                if (
                    self.trainer.datamodule.dataset_name == "PlacePulse2"
                    or self.trainer.datamodule.dataset_name == "SyntheticPID"
                ):
                    val_loaders = self.trainer.datamodule.val_dataloader()
                    if not isinstance(val_loaders, list):
                        val_loaders = [val_loaders]

                    for dl_idx, val_loader in enumerate(val_loaders):
                        for i in range(
                            len(self.trainer.datamodule.modalities)
                        ):
                            representations, concatenated_modalities = (
                                self.create_representations(
                                    val_loader,
                                    mask_indices=[i],
                                    return_representations=True,
                                    not_modality_list=[
                                        "coords",
                                        "SVI",
                                        "sentinel2",
                                        "OSM",
                                        "text",
                                    ],
                                )
                            )
                            if not os.path.exists(
                                self.plot_representation_space["save_dir"]
                            ):
                                os.makedirs(
                                    self.plot_representation_space["save_dir"]
                                )
                            torch.save(
                                representations,
                                f"{self.plot_representation_space['save_dir']}/"
                                f"representations_val_epoch_{self.current_epoch}_"
                                f"masked_modality_{i}_dl_{dl_idx}.pt",
                            )
                            save_path = (
                                f"{self.plot_representation_space['save_dir']}"
                                f"/modalities_val_dl_{dl_idx}.pt"
                            )
                            concatenated_modalities_save = (
                                concatenated_modalities.get("gsv_img", None)
                            )
                            if (
                                concatenated_modalities_save is not None
                                and not os.path.exists(save_path)
                            ):
                                torch.save(
                                    concatenated_modalities_save, save_path
                                )

                else:
                    # if representations is None:
                    for i in range(len(self.trainer.datamodule.modalities)):
                        for boolean in [True, False]:
                            (
                                representations,
                                concatenated_modalities,
                            ) = self.create_representations(
                                self.trainer.datamodule.val_dataloader(),
                                mask_indices=[i],
                                return_representations=boolean,
                            )
                            torch.save(
                                representations,
                                (
                                    f"{self.plot_representation_space['save_dir']}/"
                                    f"representations_val_epoch"
                                    f"_{self.current_epoch}_"
                                    f"rep_{str(boolean)}_"
                                    f"masked_modality_{i}.pt"
                                ),
                            )
                            save_path = (
                                f"{self.plot_representation_space['save_dir']}/"
                                f"modalities_val.pt"
                            )
                            concatenated_modalities_save = (
                                concatenated_modalities["gsv_img"]
                            )
                            if not os.path.exists(save_path):
                                torch.save(
                                    concatenated_modalities_save, save_path
                                )

    def on_test_start(self) -> None:
        """
        Initializes the correct count and r_squared_sum data structures, used
        during testing for tracking accuracy and R2 like metric across batches.
        """
        if (
            self.trainer.datamodule.dataset_name == "PlacePulse2"
            or self.trainer.datamodule.dataset_name == "SyntheticPID"
        ):
            self.total_samples = [0, 0]
        else:
            self.total_samples = 0
        if "accuracy" in self.test_metrics:
            if (
                self.trainer.datamodule.dataset_name == "PlacePulse2"
                or self.trainer.datamodule.dataset_name == "SyntheticPID"
            ):
                self.correct_count0 = get_correct_count_data_structure(
                    self.model.only_cls
                )
                self.correct_count1 = get_correct_count_data_structure(
                    self.model.only_cls
                )
                self.correct_count = [self.correct_count0, self.correct_count1]
            else:
                self.correct_count = get_correct_count_data_structure(
                    self.model.only_cls
                )
        if "r2_metric" in self.test_metrics:
            if (
                self.trainer.datamodule.dataset_name == "PlacePulse2"
                or self.trainer.datamodule.dataset_name == "SyntheticPID"
            ):
                self.r_squared_sum0 = get_correct_count_data_structure(
                    self.model.only_cls
                )
                self.r_squared_sum1 = get_correct_count_data_structure(
                    self.model.only_cls
                )
                self.r_squared_sum = [self.r_squared_sum0, self.r_squared_sum1]
            else:
                self.r_squared_sum = get_correct_count_data_structure(
                    self.model.only_cls
                )

        if type(self.trainer.logger).__name__ == "CSVLogger":
            if self.model.only_cls:
                # This call logs a dummy row, ensuring the CSV header contains
                # all these keys. Needed for correct logging if DDP is used.
                self.log_dict(
                    metrics_train_val_test, logger=True, sync_dist=True
                )
            else:
                transformed_metrics = {}
                for key, value in metrics_train_val_test.items():
                    if "accuracy" in key or "r2_metric" in key:
                        for i in range(self.model.number_tokens):
                            transformed_metrics[f"{key}_{i}"] = value
                    else:
                        transformed_metrics[key] = value
                self.log_dict(transformed_metrics, logger=True, sync_dist=True)

    def test_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """
        Test step of the model.

        Parameters
        ----------
        batch : dict
            Dictionary containing modalities (e.g., 'image', 'numeric',
            'text').
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            Loss tensor.
        """

        # Tokenize text and move tensors to device
        inputs, batch_size = self._process_batch(batch)

        # Generate  mask indices
        mask_dict = self._validation_masks(num_modalities=len(inputs))

        loss = 0
        for mask0, mask1 in mask_dict.values():
            # Forward pass for siamese model and compute loss
            loss += self._compute_loss(inputs, mask0, mask1, dataloader_idx)

        loss /= len(mask_dict)

        if (
            self.trainer.datamodule.dataset_name == "PlacePulse2"
            or self.trainer.datamodule.dataset_name == "SyntheticPID"
        ):
            self.log(
                f"test_loss_dataloader_{dataloader_idx}",
                loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )

        else:
            # Log loss
            self.log(
                "test_loss",
                loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )

        return loss

    def on_test_epoch_end(self) -> None:
        """
        Calculates the accuracy and R2-like metric at the end of the test
        epoch. Optionally plots the representation space after PCA and t-SNE.
        """
        if (
            self.trainer.datamodule.dataset_name == "PlacePulse2"
            or self.trainer.datamodule.dataset_name == "SyntheticPID"
        ):
            num_batches0 = len(self.trainer.datamodule.test_dataloader()[0])
            num_batches1 = len(self.trainer.datamodule.test_dataloader()[1])
            num_batches = [num_batches0, num_batches1]
        else:
            num_batches = sum(self.trainer.num_val_batches)
        self._log_epoch_metrics(
            metrics=set(self.test_metrics),
            prefix="test",
            num_batches=num_batches,
        )
        self._log_memory("test")

        representations = None

        # Plot the representation space if requested
        if self.plot_representation_space["plot_test"]:
            if self.trainer.datamodule.dataset_name == "SyntheticMultiModal":

                representations, concatenated_modalities = (
                    self.create_representations(
                        self.trainer.datamodule.test_dataloader(),
                        mask_indices=self.plot_representation_space[
                            "mask_indices"
                        ],
                        return_representations=False,
                    )
                )
                text = concatenated_modalities["text"]
                images = concatenated_modalities["image"]
                plot_synthetic_representations(
                    representations=representations,
                    images=images,
                    text_data=text,
                    token_index=self.plot_representation_space["token_index"],
                    epoch=self.current_epoch,
                    stage="test",
                    only_cls=self.model.only_cls,
                    title=self.plot_representation_space["title"],
                    save_plots=self.plot_representation_space["save_plots"],
                    show_plots=self.plot_representation_space["show_plots"],
                    save_dir=self.plot_representation_space["save_dir"],
                )

            elif self.trainer.datamodule.dataset_name == "Milan":
                representations, concatenated_modalities = (
                    self.create_representations(
                        self.trainer.datamodule.test_dataloader(),
                        mask_indices=self.plot_representation_space[
                            "mask_indices"
                        ],
                        return_representations=True,
                    )
                )
                coordinates = concatenated_modalities["coordinates"]
                plot_milan_representations(
                    representations=representations,
                    modality_coordinates=coordinates,
                    token_index=self.plot_representation_space["token_index"],
                    epoch=self.current_epoch,
                    stage="test",
                    only_cls=self.model.only_cls,
                    title=self.plot_representation_space["title"],
                    save_plots=self.plot_representation_space["save_plots"],
                    show_plots=self.plot_representation_space["show_plots"],
                    save_dir=self.plot_representation_space["save_dir"],
                )
            else:
                print(
                    "Plotting the representation space is only supported for "
                    "SyntheticMultiModalDataset."
                )
            if self.plot_representation_space["save_representations_test"]:

                mask_cases = (
                    [
                        [i]
                        for i in range(len(self.trainer.datamodule.modalities))
                    ]
                    + [[]]
                    + [
                        list(c)
                        for r in range(
                            2, len(self.trainer.datamodule.modalities)
                        )
                        for c in combinations(
                            range(len(self.trainer.datamodule.modalities)), r
                        )
                    ]
                )
                if (
                    self.trainer.datamodule.dataset_name == "PlacePulse2"
                    or self.trainer.datamodule.dataset_name == "SyntheticPID"
                ):
                    test_loaders = self.trainer.datamodule.test_dataloader()
                    if not isinstance(test_loaders, list):
                        test_loaders = [test_loaders]

                    for dl_idx, val_loader in enumerate(test_loaders):
                        for mask_indices in mask_cases:
                            representations, concatenated_modalities = (
                                self.create_representations(
                                    val_loader,
                                    mask_indices=mask_indices,
                                    return_representations=True,
                                    not_modality_list=[
                                        "coords",
                                        "SVI",
                                        "sentinel2",
                                        "OSM",
                                        "text",
                                    ],
                                )
                            )
                            mask_name = (
                                "none"
                                if not mask_indices
                                else "_".join(map(str, mask_indices))
                            )
                            if not os.path.exists(
                                self.plot_representation_space["save_dir"]
                            ):
                                os.makedirs(
                                    self.plot_representation_space["save_dir"]
                                )
                            torch.save(
                                representations,
                                f"{self.plot_representation_space['save_dir']}"
                                f"/representations_test_epoch_"
                                f"{self.current_epoch}_"
                                f"masked_modality_{mask_name}_dl_{dl_idx}.pt",
                            )
                            save_path = (
                                f"{self.plot_representation_space['save_dir']}"
                                f"/modalities_test_dl_{dl_idx}.pt"
                            )
                            (concatenated_modalities_save) = (
                                concatenated_modalities.get("gsv_img", None)
                            )
                            if (
                                concatenated_modalities_save is not None
                                and not os.path.exists(save_path)
                            ):
                                torch.save(
                                    concatenated_modalities_save, save_path
                                )

                else:
                    for mask_indices in mask_cases:
                        for boolean in [True, False]:
                            (
                                representations,
                                concatenated_modalities,
                            ) = self.create_representations(
                                self.trainer.datamodule.test_dataloader(),
                                mask_indices=mask_indices,
                                return_representations=boolean,
                            )

                            mask_name = (
                                "none"
                                if not mask_indices
                                else "_".join(map(str, mask_indices))
                            )

                            torch.save(
                                representations,
                                f"{self.plot_representation_space['save_dir']}/"
                                f"representations_test_epoch"
                                f"_{self.current_epoch}_"
                                f"rep_{str(boolean)}_"
                                f"masked_modality_{mask_name}.pt",
                            )

                            save_path = (
                                f"{self.plot_representation_space['save_dir']}"
                                f"/modalities_test.pt"
                            )
                            # Save the concatenated modalities if it exists
                            if "gsv_img" in concatenated_modalities:
                                (concatenated_modalities_save) = (
                                    concatenated_modalities["gsv_img"]
                                )
                                if not os.path.exists(save_path):
                                    torch.save(
                                        concatenated_modalities_save, save_path
                                    )

        if self.trainer.datamodule.coordinate_predictions is not None:
            for dl_idx, val_loader in enumerate(
                self.trainer.datamodule.coordinate_dataloaders
            ):
                representations, concatenated_modalities = (
                    self.create_representations(
                        val_loader,
                        mask_indices=list(range(1, 5)),
                        return_representations=True,
                        not_modality_list=[
                            "coords",
                            "SVI",
                            "sentinel2",
                            "OSM",
                            "text",
                        ],
                    )
                )
                dataset_name = val_loader.dataset.dataset_name
                mask_name = "_".join(map(str, list(range(1, 5))))
                if not os.path.exists(
                    self.plot_representation_space["save_dir"]
                ):
                    os.makedirs(self.plot_representation_space["save_dir"])
                torch.save(
                    representations,
                    f"{self.plot_representation_space['save_dir']}/"
                    f"representations_test_epoch_{self.current_epoch}_"
                    f"masked_modality_{mask_name}_dl_{dataset_name}.pt",
                )
                save_path = (
                    f"{self.plot_representation_space['save_dir']}"
                    f"/modalities_test_dl_{dataset_name}.pt"
                )
                concatenated_modalities_save = concatenated_modalities.get(
                    "gsv_img", None
                )
                if (
                    concatenated_modalities_save is not None
                    and not os.path.exists(save_path)
                ):
                    torch.save(concatenated_modalities_save, save_path)

    def predict_step(
        self,
        batch: dict,
        batch_idx: int,
        mask_indices: list = None,
        return_representations: bool = False,
        return_modality_tokens: bool = False,
        return_filenames: bool = True,
        return_backbone_features: bool = False,
    ) -> torch.Tensor:
        """
        Predict step of the model.

        Parameters
        ----------
        batch : dict
            Dictionary containing modalities (e.g., 'image', 'numeric',
            'text').
        batch_idx : int
            Index of the batch.
        mask_indices : list, optional
            List of indices to mask, by default None.
        return_representations : bool, optional
            Whether to return representations, by default False.

        Returns
        -------
        torch.Tensor
            Output tensor of the model.
        """

        if "gsv_img" in batch:
            del batch["gsv_img"]

        # Tokenize text and move tensors to device
        inputs, _ = self._process_batch(batch)

        reps = self(
            inputs,
            mask_indices=mask_indices,
            return_representations=return_representations,
            return_modality_tokens=return_modality_tokens,
            return_backbone_features=return_backbone_features,
        )
        if isinstance(reps, tuple):
            reps = tuple(
                self._gather_output(r, sync_grads=False) for r in reps
            )
        elif isinstance(reps, list):
            reps = [self._gather_output(r, sync_grads=False) for r in reps]
        elif isinstance(reps, dict):
            reps = {
                k: self._gather_output(v, sync_grads=False)
                for k, v in reps.items()
            }
        else:
            reps = self._gather_output(reps, sync_grads=False)
        return reps

    def predict_modalities_step(
        self,
        batch: dict,
        batch_idx: int,
        mask_indices: list = None,
        return_representations: bool = False,
        return_modality_tokens: bool = False,
        return_filenames: bool = True,
        return_backbone_features: bool = False,
    ) -> dict:
        """
        Returns a single concatenated tensor of unmasked modalities.
        Always masks modality at index 0 (e.g., coordinates), and any
        additional indices passed.

        Parameters
        ----------
        batch : dict
            Dictionary of modalities.
        batch_idx : int
            Index of the batch.
        mask_indices : list, optional
            Additional modality indices to mask (in addition to index 0).

        Returns
        -------
        torch.Tensor
            Concatenated tensor of unmasked modalities: shape [batch_size,
            total_features].
        """
        if mask_indices is None:
            mask_indices = []
        full_mask = set([0] + mask_indices)

        # Process and tokenize all modalities
        inputs, _ = self._process_batch(batch)

        # Filter and collect unmasked modalities
        unmasked = [
            tensor for i, tensor in enumerate(inputs) if i not in full_mask
        ]
        print(full_mask)
        # Concatenate along feature dimension
        if not unmasked:
            # return something that can be concatenated
            return torch.zeros(
                (inputs[0].shape[0], 0), device=inputs[0].device
            )

        print(torch.cat(unmasked, dim=1).shape)
        print("")
        return torch.cat(unmasked, dim=1)

    def create_representations(
        self,
        data_loader: torch.utils.data.DataLoader,
        mask_indices: list = None,
        return_representations: bool = False,
        not_modality_list: list = [],
    ) -> tuple:
        """
        Create representations and collect modality inputs from an arbitrary
        number of modalities.

        Each batch is expected to be a dict (e.g., {"image": tensor, "text":
        list, "numeric": tensor, ...}).

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader yielding batches as dictionaries.
        mask_indices : list, optional
            Mask indices to pass to the model, by default None.
        return_representations : bool, optional
            Whether to instruct the model to return representations only,
            by default False.
        not_modality_list : list, optional
            List of modalities to exclude from concatenation, by default [].

        Returns
        -------
        tuple
            A tuple where the first element is a tensor of concatenated
            representations and the second element is a dict mapping modality
            names to their concatenated inputs.
        """
        all_representations = []
        modality_inputs = {}

        # Only if this attribute exists in the dataset
        if hasattr(data_loader.dataset, "return_filename"):
            # Set it to True to return filenames
            data_loader.dataset.return_filename = True

        # Iterate over the dataloader
        for batch_idx, batch in enumerate(data_loader):
            # For every modality in the batch, store its data.
            for modality, value in batch.items():
                if modality not in modality_inputs:
                    modality_inputs[modality] = []
                modality_inputs[modality].append(value)

            # Get representations using your predict_step.
            if self.raw_representations:
                reps = self.predict_modalities_step(
                    batch, batch_idx, mask_indices, return_representations
                )
            else:
                reps = self.predict_step(
                    batch, batch_idx, mask_indices, return_representations
                )
            all_representations.append(reps)

        # Concatenate all representations along the batch dimension
        representations_cat = (
            torch.cat(all_representations, dim=0)
            if all_representations
            else None
        )

        # For each modality, if the values are tensors, concatenate them;
        # otherwise, keep them as a list.
        concatenated_modalities = {}
        for modality, values in modality_inputs.items():
            # Only add modalities that are in modality list
            if modality in not_modality_list:
                continue
            else:
                if isinstance(values[0], torch.Tensor):
                    concatenated_modalities[modality] = torch.cat(
                        values, dim=0
                    )
                else:
                    concatenated_modalities[modality] = values

        if hasattr(data_loader.dataset, "return_filename"):
            # Reset it to False to not return filenames in the future
            data_loader.dataset.return_filename = False

        return representations_cat, concatenated_modalities

    def _gather_output(
        self, tensor: torch.Tensor, sync_grads: bool = True
    ) -> torch.Tensor:
        """
        Gathers a tensor from all GPUs (DDP) into a single (larger) tensor,
        so that the contrastive loss can be computed on the global batch.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to gather.
        sync_grads : bool, optional
            Whether to synchronize gradients, by default True.

        Returns
        -------
        torch.Tensor
            Gathered tensor from all GPUs.
        """
        # If we are not actually in distributed mode, or world_size==1,
        # just return the original tensor.
        if (self.trainer.num_devices * self.trainer.num_nodes) <= 1:
            return tensor

        # In Lightning, self.all_gather returns a tensor of shape:
        #     [world_size, local_batch_size, embedding_dim]
        # so we flatten out the first dimension to get
        # [global_batch_size, embedding_dim].
        gathered = self.all_gather(tensor, sync_grads=sync_grads)
        if self.model.only_cls:
            return gathered.reshape(-1, tensor.shape[-1])
        else:
            return gathered.reshape(
                gathered.shape[0] * gathered.shape[1],
                gathered.shape[2],
                gathered.shape[3],
            )

    def _compute_loss(
        self,
        inputs: list,
        mask0: list,
        mask1: list,
        dataloader_idx: int = 0,
        modality_dropout: float = 0.0,
    ) -> torch.Tensor:
        """
        Computes the loss based on the selected loss type.
        Performs the forward pass for the siamese model and
        and computes the contrastive loss. If requested, it also
        computes the accuracy and r2_metric.

        Parameters
        ----------
        inputs : list
            List of input tensors of different modalities.
        mask0 : list
            List of indices to mask for output0.
        mask1 : list
            List of indices to mask for output1.

        Returns
        -------
        torch.Tensor
            Loss tensor.
        """

        # Get the current stage
        stage = self.trainer.state.stage

        # Forward pass for siamese model
        output0 = self(
            inputs,
            mask_indices=mask0,
            return_modality_tokens=self.return_modality_tokens,
            return_backbone_features=self.modality_reconstruction,
            modality_dropout=modality_dropout,
        )
        output1 = self(
            inputs,
            mask_indices=mask1,
            return_modality_tokens=self.return_modality_tokens,
            return_backbone_features=self.modality_reconstruction,
            modality_dropout=modality_dropout,
        )

        if self.return_modality_tokens:
            if self.modality_reconstruction:
                output0, tokens0, modalities0, rec0 = output0
                output1, tokens1, _, rec1 = output1
                modalities0 = self._gather_output(
                    modalities0, sync_grads=False
                )
                rec0 = self._gather_output(rec0, sync_grads=True)
                rec1 = self._gather_output(rec1, sync_grads=True)
            else:
                output0, tokens0 = output0
                output1, tokens1 = output1
            # Gather (if using DDP) so we compute contrastive loss globally
            # For input similarity, no gradients are needed
            tokens0 = self._gather_output(tokens0, sync_grads=False)
            tokens1 = self._gather_output(tokens1, sync_grads=False)
            # Compute similarity matrices
            token0_cosine_sim = tokens0 @ tokens0.T
            token1_cosine_sim = tokens1 @ tokens1.T
            # Gather the pooled outputs as well
            # Sync grads for the contrastive loss
            output0 = self._gather_output(output0, sync_grads=True)
            output1 = self._gather_output(output1, sync_grads=True)
            if self.queue_size > 0:
                with torch.no_grad():
                    queue_output = self.get_queue().to(self.device)
                    queue_input = torch.cat([output0, output1], dim=0)
                    self._dequeue_and_enqueue(queue_input)
            else:
                queue_output = None
            # Compute contrastive loss
            if self.modality_reconstruction:
                loss = self.loss(
                    output0,
                    output1,
                    token0_cosine_sim,
                    token1_cosine_sim,
                    rec_gt=modalities0,
                    rec0=rec0,
                    rec1=rec1,
                    queue_output=queue_output,
                )
            else:
                loss = self.loss(
                    output0,
                    output1,
                    token0_cosine_sim,
                    token1_cosine_sim,
                    queue_output=queue_output,
                )
        else:
            # Gather pooled outputs for global Loss
            if self.modality_reconstruction:
                output0, modalities0, rec0 = output0
                output1, _, rec1 = output1
                modalities0 = self._gather_output(
                    modalities0, sync_grads=False
                )
                rec0 = self._gather_output(rec0, sync_grads=True)
                rec1 = self._gather_output(rec1, sync_grads=True)
            output0 = self._gather_output(output0, sync_grads=True)
            output1 = self._gather_output(output1, sync_grads=True)
            if self.queue_size > 0:
                with torch.no_grad():
                    queue_output = self.get_queue().to(self.device)
                    queue_input = torch.cat([output0, output1], dim=0)
                    self._dequeue_and_enqueue(queue_input)
            else:
                queue_output = None
            # Compute contrastive loss
            if self.modality_reconstruction:
                loss = self.loss(
                    output0,
                    output1,
                    rec_gt=modalities0,
                    rec0=rec0,
                    rec1=rec1,
                    queue_output=queue_output,
                )
            else:
                loss = self.loss(output0, output1, queue_output=queue_output)

        # Define desired metrics
        desired_metrics = {"accuracy", "r2_metric"}

        # Get the current stage metrics
        if stage == "validate":
            stage_metrics = set(self.val_metrics)
        elif stage == "test":
            stage_metrics = set(self.test_metrics)
        else:
            stage_metrics = set()

        # Compute accuracy and r2_metric if required
        if stage in ("validate", "test") and (desired_metrics & stage_metrics):
            if (
                self.trainer.datamodule.dataset_name == "PlacePulse2"
                or self.trainer.datamodule.dataset_name == "SyntheticPID"
            ):
                # Calculate accuracy
                correct, samples, similarity_matrix = (
                    count_correct_predictions_router(
                        output0,
                        output1,
                        only_cls=self.model.only_cls,
                        normalize=self.normalize_model_output,
                        top_k=self.top_k_accuracy,
                    )
                )

                # Compute r2_metric if required
                if "r2_metric" in stage_metrics:
                    r_squared_metric = compute_r_squared_router(
                        similarity_matrix,
                        output0,
                        output1,
                        distance=self.r2_metric_distance,
                        cls_only=self.model.only_cls,
                    )

                # Update accuracy metric if requested
                if "accuracy" in stage_metrics:
                    self.correct_count[dataloader_idx] = update_correct_count(
                        self.correct_count[dataloader_idx],
                        correct,
                        only_cls=self.model.only_cls,
                    )

                # Update r2_metric if requested
                if "r2_metric" in stage_metrics:
                    self.r_squared_sum[dataloader_idx] = update_r_squared(
                        self.r_squared_sum[dataloader_idx],
                        r_squared_metric,
                        only_cls=self.model.only_cls,
                    )

                # Update total sample count
                self.total_samples[dataloader_idx] += samples

            else:
                # Calculate accuracy
                correct, samples, similarity_matrix = (
                    count_correct_predictions_router(
                        output0,
                        output1,
                        only_cls=self.model.only_cls,
                        normalize=self.normalize_model_output,
                        top_k=self.top_k_accuracy,
                    )
                )

                # Compute r2_metric if required
                if "r2_metric" in stage_metrics:
                    r_squared_metric = compute_r_squared_router(
                        similarity_matrix,
                        output0,
                        output1,
                        distance=self.r2_metric_distance,
                        cls_only=self.model.only_cls,
                    )

                # Update accuracy metric if requested
                if "accuracy" in stage_metrics:
                    self.correct_count = update_correct_count(
                        self.correct_count,
                        correct,
                        only_cls=self.model.only_cls,
                    )

                # Update r2_metric if requested
                if "r2_metric" in stage_metrics:
                    self.r_squared_sum = update_r_squared(
                        self.r_squared_sum,
                        r_squared_metric,
                        only_cls=self.model.only_cls,
                    )

                # Update total sample count
                self.total_samples += samples

        return loss

    def _log_epoch_metrics(
        self, metrics: set, prefix: str, num_batches: int
    ) -> None:
        """
        Logs the epoch metrics for a given stage.

        Parameters
        ----------
        metrics: set
            The set of metric names to log (e.g. {"accuracy", "r2_metric"}).
        prefix: str
            The log prefix, e.g. "val" or "test".
        num_batches: int
            The total number of batches for the stage.
        """
        if (
            self.trainer.datamodule.dataset_name == "PlacePulse2"
            or self.trainer.datamodule.dataset_name == "SyntheticPID"
        ):
            num_dataloaders = len(self.trainer.datamodule.val_dataloader())
            for idx in range(num_dataloaders):
                if self.total_samples[idx] == 0:
                    continue

                metric_prefix = f"{prefix}_dataloader_{idx}"

                if "accuracy" in metrics:
                    accuracy = clip_accuray_router(
                        self.correct_count[idx],
                        self.total_samples[idx],
                        print_metrics=False,
                    )
                    if self.model.only_cls:
                        self.log(
                            f"{metric_prefix}_accuracy",
                            accuracy,
                            prog_bar=False,
                            logger=True,
                            sync_dist=True,
                        )
                    else:  # Log accuracy for all tokens
                        for i, acc in enumerate(accuracy):
                            self.log(
                                f"{metric_prefix}_accuracy_{i}",
                                acc,
                                prog_bar=False,
                                logger=True,
                                sync_dist=True,
                            )

                if "r2_metric" in metrics:

                    r_squared_metric = r_squared_reduction(
                        self.r_squared_sum[idx],
                        num_batches[idx],
                        only_cls=self.model.only_cls,
                        print_results=False,
                    )
                    if self.model.only_cls:
                        self.log(
                            f"{metric_prefix}_r2_metric",
                            r_squared_metric.item(),
                            prog_bar=False,
                            logger=True,
                            sync_dist=True,
                        )
                    else:
                        for i, r2 in enumerate(r_squared_metric):
                            self.log(
                                f"{metric_prefix}_r2_metric_{i}",
                                r2.item(),
                                prog_bar=False,
                                logger=True,
                                sync_dist=True,
                            )
            return  # skip the original below since PlacePulse2 is handled

        else:
            # Check if there are any samples
            if self.total_samples == 0:
                return

            if "accuracy" in metrics:
                accuracy = clip_accuray_router(
                    self.correct_count, self.total_samples, print_metrics=False
                )
                if self.model.only_cls:
                    self.log(
                        f"{prefix}_accuracy",
                        accuracy,
                        prog_bar=False,
                        logger=True,
                        sync_dist=True,
                    )
                else:  # Log accuracy for all tokens
                    for i, acc in enumerate(accuracy):
                        self.log(
                            f"{prefix}_accuracy_{i}",
                            acc,
                            prog_bar=False,
                            logger=True,
                            sync_dist=True,
                        )

            if "r2_metric" in metrics:
                r_squared_metric = r_squared_reduction(
                    self.r_squared_sum,
                    num_batches,  # * self.len_mask_dict,
                    only_cls=self.model.only_cls,
                    print_results=False,
                )
                if self.model.only_cls:
                    self.log(
                        f"{prefix}_r2_metric",
                        r_squared_metric.item(),
                        prog_bar=False,
                        logger=True,
                        sync_dist=True,
                    )
                else:
                    for i, r2 in enumerate(r_squared_metric):
                        self.log(
                            f"{prefix}_r2_metric_{i}",
                            r2.item(),
                            prog_bar=False,
                            logger=True,
                            sync_dist=True,
                        )

    def _log_memory(self, stage: str):
        """
        Call this inside training/validation/test hooks to log
        CPU+GPU usage.
        """
        # CPU usage in MB

        if torch.cuda.is_available():
            self.log(
                f"{stage}/gpu_mem_allocated_MB",
                torch.cuda.memory_allocated() / 1e6,
            )
            self.log(
                f"{stage}/gpu_mem_reserved_MB",
                torch.cuda.memory_reserved() / 1e6,
            )

    def setup(self, stage: str) -> None:
        """
        Lightning hook that is called at the beginning of fit (train +
        validate), validate,
        test, or predict.

        Used to perform optional compilation of the model.

        Parameters
        ----------
        stage : str
            Stage of the training process
        """
        if self.torch_compile and stage == "fit":
            self.model = torch.compile(self.model)

    def _initialize_queue(self, queue_size):
        self.queue_size = queue_size
        self.register_buffer(
            "queue", torch.randn(self.contrastive_dim, self.queue_size)
        )
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, inputs: torch.Tensor):
        """Update GPS queue

        Args:
            gps (torch.Tensor): GPS tensor of shape (batch_size, 2)
        """
        batch_size = inputs.shape[0]
        gps_ptr = int(self.queue_ptr)
        if self.queue_size % batch_size == 0:
            self.queue[:, gps_ptr : gps_ptr + batch_size] = inputs.t()
            gps_ptr = (gps_ptr + batch_size) % self.queue_size
            self.queue_ptr[0] = gps_ptr
        else:
            return

    def get_queue(self):
        return self.queue.t()

    def configure_optimizers(self) -> dict:
        """
        Configures the optimizer and learning rate scheduler.

        Returns
        -------
        dict
            Dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = self.optimizer
        scheduler = self.scheduler

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
