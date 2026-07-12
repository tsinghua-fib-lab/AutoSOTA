from __future__ import annotations

import os
from typing import Optional

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from progressive_cramming.inference.generation import generate_from_compression
from progressive_cramming.train.embedding_init import create_compression_embedding, prepare_embedding_init
from progressive_cramming.train.loss import (
    compute_hybrid_cross_entropy_and_alignment_loss,
    token_argmax_match_rate_with_prefix,
)
from progressive_cramming.train.optimization import build_optimizer_and_scheduler
from progressive_cramming.utils.launch import freeze_model_parameters, get_device, set_launch_seed


class BaseTrainer:
    """Base class for compression trainers. Subclasses must implement train method."""

    def __init__(
        self,
        model=None,
        processing_class=None,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        model_reconstructor=None,
    ) -> None:
        self.model = model
        # Optional second model used by the compression-head / progressive trainers when
        # --separate_reconstructor_model is set (or a dual checkpoint is loaded). None means the
        # second forward reuses `self.model` (legacy single-model path).
        self.model_reconstructor = model_reconstructor
        self.processing_class = processing_class
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator

        # Acceleration
        ddp_kwargs = None
        if args.ddp_find_unused_parameters:
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=bool(args.ddp_find_unused_parameters))
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision="no",
            kwargs_handlers=[ddp_kwargs] if ddp_kwargs is not None else None,
        )

        # Tensorboard logging
        log_dir = self.args.logging_dir
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir and self.accelerator.is_main_process else None
        self.global_step = 0

    def train(self) -> str | None:
        """Run training. Subclasses must override. Returns artifact path or output_dir or None."""
        raise NotImplementedError("Subclasses must implement 'train' method!")

    def _initialize_run(self):
        """Seed RNG, move model to device, freeze its parameters, prepare embedding-init helpers."""
        set_launch_seed(self.args.random_seed)
        device = get_device()
        model = self.model.to(device)
        freeze_model_parameters(model)
        init_method, mvn_dist, pca_components, pca_mean, loaded_embeddings = self._prepare_embedding_init(model)
        return model, device, init_method, mvn_dist, pca_components, pca_mean, loaded_embeddings

    def compute_loss(
        self,
        logits: torch.Tensor,  # [batch, compression + sequence, vocabulary]
        input_ids: torch.Tensor,  # [batch, sequence]
        attention_mask: torch.Tensor,  # [batch, sequence]
        num_compression_tokens: int,
        compression_hidden_states: tuple[torch.Tensor, ...] | None = None,
        target_hidden_states: tuple[torch.Tensor, ...] | None = None,
        prefix_len: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute CE + (optionally) activation-alignment loss from already-computed logits.

        ``prefix_len`` skips the uncompressed prefix positions so loss is computed only over the
        continuation (the tokens that follow the prefix); 0 leaves behaviour unchanged.
        """
        loss, alignment_loss = compute_hybrid_cross_entropy_and_alignment_loss(
            logits=logits,
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_compression_tokens=num_compression_tokens,
            target_hidden_states=target_hidden_states,
            compression_hidden_states=compression_hidden_states,
            num_alignment_layers=self.args.num_alignment_layers,
            inverted_alignment=self.args.inverted_alignment,
            loss_type=self.args.loss_type.lower(),
            hybrid_alpha=self.args.hybrid_alpha,
            leading_token_loss_weight=self.args.leading_token_loss_weight,
            leading_token_loss_count=self.args.leading_token_loss_count,
            prefix_len=prefix_len,
        )
        return loss, alignment_loss

    @torch.no_grad()
    def compute_convergence(
        self,
        logits: torch.Tensor,  # [batch, compression + sequence, vocabulary]
        input_ids: torch.Tensor,  # [batch, sequence]
        attention_mask: torch.Tensor,  # [batch, sequence]
        num_compression_tokens: int,
        prefix_len: int = 0,
    ) -> torch.Tensor:
        """Per-sample token-level argmax match rate in [0, 1] (over the post-prefix continuation)."""
        return token_argmax_match_rate_with_prefix(
            logits,
            input_ids,
            attention_mask,
            num_compression_tokens,
            prefix_len=prefix_len,
        )

    @torch.no_grad()
    def compute_hidden_states(
        self,
        model,
        token_embeddings: torch.Tensor,  # [batch, sequence, hidden]
        attention_mask: torch.Tensor,  # [batch, sequence]
    ) -> tuple[torch.Tensor, ...]:
        """Forward the model once and return its per-layer hidden states."""
        outputs = model(
            inputs_embeds=token_embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return outputs.hidden_states

    @torch.no_grad()
    def generate_diagnostics(
        self,
        model,
        input_ids: torch.Tensor,
        united_token_embeddings: torch.Tensor,
        num_compression_tokens: int,
    ) -> tuple[list[str] | None, list[str] | None]:
        """Optionally produce greedy text from the current compression embedding (debug only)."""
        if self.global_step % 100 != 0 or not self.args.generate_in_compute_loss:
            return None, None
        generated_text = generate_from_compression(
            model,
            self.processing_class,
            united_token_embeddings[:, :num_compression_tokens],
            max_new_tokens=self.args.max_sequence_length,
            num_return_sequences=1,
        )
        ground_truth_text = self.processing_class.batch_decode(input_ids, skip_special_tokens=True)
        return generated_text, ground_truth_text

    # ------------------------------------------------------------------
    # Backward-compatibility wrappers for trainers that haven't migrated to
    # the new function-based API yet (low_dim, progressive, prefix_tuning,
    # compression_head). FullCrammingTrainer calls the underlying functions
    # directly.
    # ------------------------------------------------------------------

    def _prepare_embedding_init(self, model):
        """Wrapper around train.embedding_init.prepare_embedding_init."""
        return prepare_embedding_init(self.args, model)

    @staticmethod
    def _init_compression_tokens(
        batch_size: int,
        num_compression_tokens: int,
        hidden_size: int,
        init_method: str,
        mvn_dist,
        token_embeddings: torch.Tensor | None = None,
        single_compression_token_embeddings_initialization: torch.Tensor | None = None,
        pca_components: torch.Tensor | None = None,
        pca_mean: torch.Tensor | None = None,
        loaded_embeddings: torch.Tensor | None = None,
    ) -> torch.nn.Parameter:
        """Wrapper around train.embedding_init.create_compression_embedding."""
        return create_compression_embedding(
            batch_size=batch_size,
            num_compression_tokens=num_compression_tokens,
            hidden_size=hidden_size,
            init_method=init_method,
            mvn_dist=mvn_dist,
            token_embeddings=token_embeddings,
            single_compression_token_embeddings_initialization=single_compression_token_embeddings_initialization,
            pca_components=pca_components,
            pca_mean=pca_mean,
            loaded_embeddings=loaded_embeddings,
        )

    def _build_optimizer_and_scheduler(self, parameters, num_training_steps=None, num_processes=1):
        """Wrapper around train.optimization.build_optimizer_and_scheduler."""
        return build_optimizer_and_scheduler(self.args, parameters, num_training_steps, num_processes)

    def _build_low_dim_projection_module(self, hidden_size: int, device: torch.device) -> torch.nn.Linear:
        """Construct ``nn.Linear(low_dim_size -> hidden_size)`` for the low-dim projection."""
        projection = torch.nn.Linear(self.args.low_dim_size, hidden_size).to(device)

        checkpoint_path = self.args.low_dim_projection_checkpoint
        if checkpoint_path is not None:
            if not os.path.exists(checkpoint_path):
                raise ValueError(f"low_dim_projection_checkpoint does not exist: {checkpoint_path}!")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(checkpoint, dict):
                state = checkpoint.get("low_dim_projection") or checkpoint.get("state_dict") or checkpoint
            else:
                state = checkpoint
            projection.load_state_dict(state)
            print(f"Loaded low-dim projection state from {checkpoint_path} (low_dim_size={self.args.low_dim_size})")

        if not self.args.low_dim_projection_train:
            for parameter in projection.parameters():
                parameter.requires_grad = False
        return projection

    def forward_and_compute_loss(
        self,
        model,
        input_ids: torch.Tensor,  # [batch, sequence]
        token_embeddings: torch.Tensor,  # [batch, sequence, hidden]
        attention_mask: torch.Tensor,  # [batch, sequence]
        united_token_embeddings: torch.Tensor,  # [batch, compression + prefix + sequence, hidden]
        united_attention_mask: torch.Tensor,  # [batch, compression + prefix + sequence]
        num_compression_tokens: int,
        target_hidden_states: tuple[torch.Tensor, ...] | None = None,
        prefix_len: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, list[str] | None, list[str] | None]:
        """Forward + loss + convergence + diagnostics, returning a 5-tuple.

        ``prefix_len`` is the number of uncompressed prefix tokens inserted between the compression
        tokens and the continuation in ``united_token_embeddings``; loss and convergence are then
        computed only over the continuation. ``input_ids``/``attention_mask``/``token_embeddings``
        describe the continuation only.
        """
        # Compute vanilla model hidden states if alignment is required
        loss_type = self.args.loss_type.lower()
        if loss_type != "cross_entropy" and target_hidden_states is None:
            target_hidden_states = self.compute_hidden_states(model, token_embeddings, attention_mask)

        # Forward pass extra kwargs
        forward_extra_kwargs = {}
        if loss_type != "cross_entropy":
            forward_extra_kwargs["output_hidden_states"] = True
        if self.args.fix_position_ids:
            if prefix_len > 0:
                raise NotImplementedError("fix_position_ids is not supported together with progressive_prefix_len > 0.")
            position_ids = torch.arange(
                -num_compression_tokens,
                token_embeddings.size(1),
                device=token_embeddings.device,
            )
            position_ids[:num_compression_tokens] = 0
            position_ids = position_ids.repeat(token_embeddings.size(0), 1)
            forward_extra_kwargs["position_ids"] = position_ids

        # Compression forward pass and loss calculation
        outputs = model(
            inputs_embeds=united_token_embeddings,
            attention_mask=united_attention_mask,
            **forward_extra_kwargs,
        )
        loss, alignment_loss = self.compute_loss(
            outputs.logits,
            input_ids,
            attention_mask,
            num_compression_tokens,
            compression_hidden_states=outputs.hidden_states,
            target_hidden_states=target_hidden_states,
            prefix_len=prefix_len,
        )
        convergence_per_sample = self.compute_convergence(
            outputs.logits, input_ids, attention_mask, num_compression_tokens, prefix_len=prefix_len
        )
        generated_text, ground_truth_text = self.generate_diagnostics(
            model, input_ids, united_token_embeddings, num_compression_tokens
        )
        return loss, alignment_loss, convergence_per_sample, generated_text, ground_truth_text

    def _create_dataloader(self) -> DataLoader:
        """Build the training DataLoader from the trainer args."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def _log_step(
        self,
        loss: torch.Tensor,
        alignment_loss: torch.Tensor | None,
        convergence_per_sample: torch.Tensor,
        compression_token_embeddings: torch.Tensor | None,
        lr_scheduler,
        generated_text: Optional[list[str]] = None,
        ground_truth_text: Optional[list[str]] = None,
        *,
        embedding_namespace: str = "compression_token_embeddings",
        leaf_grad_params: Optional[list[torch.Tensor]] = None,
    ):
        """Write per-step scalars/text to TensorBoard.

        ``compression_token_embeddings`` may be a non-leaf tensor (e.g. the
        materialized ``projection(z)`` output in LowDim parametrizations) — we
        only use it for mean/std stats. Pass ``leaf_grad_params`` (e.g.
        ``parametrization.parameters`` + ``shared_parameters``) to log a real
        grad-norm; accessing ``.grad`` on a non-leaf tensor produces a
        UserWarning and yields ``None`` anyway, so we deliberately skip it.
        """
        if self.writer is None:
            return

        self.writer.add_scalar("train/loss", loss.item(), self.global_step)
        if alignment_loss is not None:
            self.writer.add_scalar("train/alignment_loss", alignment_loss.item(), self.global_step)
        self.writer.add_scalar("train/convergence", convergence_per_sample.mean().item(), self.global_step)

        if compression_token_embeddings is not None:
            with torch.no_grad():
                self.writer.add_scalar(
                    f"{embedding_namespace}/mean", compression_token_embeddings.mean().item(), self.global_step
                )
                self.writer.add_scalar(
                    f"{embedding_namespace}/std", compression_token_embeddings.std().item(), self.global_step
                )

        grad_norm_source = (
            leaf_grad_params
            if leaf_grad_params is not None
            else (
                [compression_token_embeddings]
                if compression_token_embeddings is not None and compression_token_embeddings.is_leaf
                else None
            )
        )
        if grad_norm_source:
            total_sq = 0.0
            for p in grad_norm_source:
                if p.grad is not None:
                    total_sq += float(p.grad.detach().norm(2).item() ** 2)
            self.writer.add_scalar(f"{embedding_namespace}/grad_norm", total_sq**0.5, self.global_step)

        if lr_scheduler is not None:
            self.writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], self.global_step)
        if generated_text:
            self.writer.add_text("train/generated_text", " | ".join(generated_text), self.global_step)
        if ground_truth_text:
            self.writer.add_text("train/ground_truth_text", " | ".join(ground_truth_text), self.global_step)

        flush_steps = getattr(self.args, "logging_flush_steps", 100)
        if flush_steps and self.global_step % flush_steps == 0:
            self.writer.flush()
        self.global_step += 1

    def _save_artifacts(
        self,
        rows: list[dict],
        *,
        tensor: torch.Tensor | None,
        tensor_filename: str,
        subdir_name: str,
    ) -> str | None:
        """Persist the tensor (if any) and a HuggingFace Dataset of rows under output_dir."""
        output_dir = self.args.output_dir
        if not output_dir or len(rows) == 0:
            return None
        os.makedirs(output_dir, exist_ok=True)
        if tensor is not None:
            torch.save(tensor, os.path.join(output_dir, tensor_filename))
        save_path = os.path.join(output_dir, subdir_name)
        Dataset.from_list(rows).save_to_disk(save_path)
        return save_path
