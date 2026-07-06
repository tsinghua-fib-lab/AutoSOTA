from typing import Any, Optional

import torch
import lightning.pytorch as pl


class FFNGreedySearchCallback(pl.Callback):
    """Greedy search evaluation on specific validation batches."""

    def __init__(self, track_batch_indices=[10, 100]):
        super().__init__()
        self.track_batch_indices = track_batch_indices
        self.validation_step_outputs = []

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.correct = 0
        self.total = 0
        self.validation_step_outputs = []

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if batch_idx not in self.track_batch_indices:
            return

        # FP_inputs, a_inputs, a_targets, rxn_class = batch
        FP_inputs, a_inputs, a_targets = batch
        batch_size = a_inputs.shape[0]

        generated_seqs = a_inputs.clone()  # [batch_size, seq_len]

        allowed_steps = (5 - a_inputs.sum(dim=1)).int()  # [batch_size]
        max_steps = allowed_steps.max().item()

        active_seqs = torch.ones(batch_size, dtype=torch.bool, device=a_inputs.device)

        for step in range(max_steps):
            if not active_seqs.any():
                break

            with torch.no_grad():
                logits = pl_module(FP_inputs[active_seqs], generated_seqs[active_seqs])

                logits_masked = torch.where(
                    generated_seqs[active_seqs] == 1,
                    torch.tensor(-1e6, device=logits.device),
                    logits,
                )

                pred_indices = torch.argmax(logits_masked, dim=-1)  # [active_batch_size]

                active_indices = torch.where(active_seqs)[0]
                for idx, pred_idx in zip(active_indices, pred_indices):
                    if pred_idx == 0:  # Stop token
                        active_seqs[idx] = False
                    else:
                        generated_seqs[idx, pred_idx] = 1

                active_seqs = active_seqs & (step + 1 < allowed_steps)

        self.validation_step_outputs.append({"search_preds": generated_seqs, "targets": a_targets})

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self.validation_step_outputs:
            return

        all_preds = torch.cat([x["search_preds"] for x in self.validation_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.validation_step_outputs])

        correct = (all_preds == all_targets).all(dim=1).sum().item()
        total = len(all_targets)

        greedy_accuracy = correct / total if total > 0 else 0
        pl_module.log(
            "val_greedy_exactmatch_accuracy",
            greedy_accuracy,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            logger=True,
        )


class GNNGreedySearchCallback(pl.Callback):
    """
    PyTorch Lightning callback for greedy search evaluation on specific validation batches.
    """

    def __init__(self, track_batch_indices=[10, 100]):
        super().__init__()
        self.track_batch_indices = track_batch_indices
        self.validation_step_outputs = []

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.correct = 0
        self.total = 0
        self.validation_step_outputs = []

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if batch_idx not in self.track_batch_indices:
            return

        a_inputs, bmg, V_d, X_d, targets, *_ = batch
        batch_size = a_inputs.shape[0]

        generated_seqs = a_inputs.clone()
        allowed_steps = (5 - generated_seqs.sum(dim=1)).int()
        max_steps = allowed_steps.max().item()

        active_seqs = torch.ones(batch_size, dtype=torch.bool, device=a_inputs.device)

        for step in range(max_steps):
            if not active_seqs.any():
                break

            with torch.no_grad():
                logits = pl_module(generated_seqs, bmg, V_d, X_d)
                logits_masked = torch.where(
                    generated_seqs == 1, torch.tensor(-1e6, device=logits.device), logits
                )
                pred_indices = torch.argmax(logits_masked, dim=-1)  # [batch_size]

                # Update sequences for only active sequences
                active_indices = torch.where(active_seqs)[0]
                for idx, pred_idx in zip(active_indices, pred_indices[active_seqs]):
                    if pred_idx == 0:  # Stop token
                        active_seqs[idx] = False
                    else:
                        generated_seqs[idx, pred_idx] = 1

                # Update active_seqs based on allowed steps
                active_seqs = active_seqs & (step + 1 < allowed_steps)

        self.validation_step_outputs.append({"search_preds": generated_seqs, "targets": targets})

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self.validation_step_outputs:
            return

        all_preds = torch.cat([x["search_preds"] for x in self.validation_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.validation_step_outputs])

        correct = (all_preds == all_targets).all(dim=1).sum().item()
        total = len(all_targets)

        greedy_accuracy = correct / total if total > 0 else 0
        pl_module.log("val_greedy_exactmatch_accuracy", greedy_accuracy, sync_dist=True)
