from dataclasses import dataclass

import torch
from tqdm.auto import tqdm

from progressive_cramming.analysis import ConvergedSamplesGuard, ConvergenceTracker
from progressive_cramming.analysis.information_gain import compute_information_gain
from progressive_cramming.train.embedding_init import create_compression_embedding
from progressive_cramming.train.inputs import (
    build_compression_attention_mask,
    build_united_input,
)
from progressive_cramming.train.parametrization import build_parametrization
from progressive_cramming.train.trainers.base import BaseTrainer


@dataclass
class _RunContext:
    """Run-level constants prepared once before the dataloader loop."""

    model: torch.nn.Module
    device: torch.device
    init_method: str
    mvn_dist: object
    pca_components: torch.Tensor | None
    pca_mean: torch.Tensor | None
    loaded_embeddings: torch.Tensor | None
    single_compression_token_init: torch.Tensor | None
    num_compression_tokens: int
    hidden_size: int


@dataclass
class _BatchInputs:
    """Per-batch tensors prepared once before optimization."""

    input_ids: torch.Tensor
    token_embeddings: torch.Tensor
    attention_mask: torch.Tensor
    target_hidden_states: tuple[torch.Tensor, ...] | None
    compression_attention_mask: torch.Tensor
    batch_size: int


@dataclass
class _BatchOptimizationResult:
    """Outputs of the per-batch optimization loop, consumed by row collection."""

    parametrization: object
    convergence_tracker: ConvergenceTracker
    last_loss: float
    last_convergence_per_sample: torch.Tensor
    initialization_embeddings: torch.Tensor


class FullCrammingTrainer(BaseTrainer):
    """Trainer for full cramming: per-sample compression tokens, optional alignment + CE loss."""

    def train(self) -> str | None:
        """Run full cramming training. Returns save path or None."""
        ctx = self._build_run_context()

        collected_rows: list[dict] = []
        sample_id_counter = 0
        # Each batch produces its own compression-embedding tensor; we concatenate
        # them along the sample dim so the saved tensor has one row per instance
        # (otherwise multi-batch runs would only persist the last batch).
        per_batch_compression: list[torch.Tensor] = []
        # Last batch's parametrization — exposed to subclasses via
        # `_on_training_complete` (e.g. LowDim saves its projection weights here).
        final_parametrization = None

        dataloader = self._create_dataloader()
        for batch in tqdm(dataloader):
            inputs = self._prepare_batch_inputs(batch, ctx)
            result = self._optimize_compression(inputs, ctx)
            rows, batch_compression_cpu = self._collect_batch_rows(
                inputs=inputs,
                result=result,
                ctx=ctx,
                sample_id_counter=sample_id_counter,
            )
            collected_rows.extend(rows)
            sample_id_counter += len(rows)
            if batch_compression_cpu is not None:
                per_batch_compression.append(batch_compression_cpu)
            final_parametrization = result.parametrization

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        self._on_training_complete(final_parametrization, ctx)

        final_compression_cpu = torch.cat(per_batch_compression, dim=0) if per_batch_compression else None

        return self._save_artifacts(
            collected_rows,
            tensor=final_compression_cpu,
            tensor_filename="compression_embeddings.pt",
            subdir_name="compressed_prefixes",
        )

    # ------------------------------------------------------------------
    # Subclass extension points.
    # ------------------------------------------------------------------

    def _extra_parametrization_kwargs(self, ctx: "_RunContext") -> dict:
        """Hook for subclasses to inject extra ``build_parametrization`` kwargs.

        ``LowDimTrainer`` uses this to enable the low-dim projection branch
        without copy-pasting the entire optimization loop. ``ctx`` is passed
        so subclasses can read ``hidden_size`` / ``device`` etc. without
        having to reach into ``self.model``.
        """
        return {}

    def _on_training_complete(self, parametrization, ctx: "_RunContext") -> None:
        """Hook called once after all batches finish training.

        ``parametrization`` is the parametrization object produced by the
        *last* batch (or ``None`` if no batches were processed). Subclasses
        override this to persist auxiliary state (e.g. low-dim projection
        weights).
        """
        pass

    def _build_run_context(self) -> _RunContext:
        """Seed RNG, freeze model, prepare embedding-init helpers and constants."""
        (
            model,
            device,
            init_method,
            mvn_dist,
            pca_components,
            pca_mean,
            loaded_embeddings,
        ) = self._initialize_run()
        num_compression_tokens = self.args.number_of_mem_tokens
        hidden_size = model.config.hidden_size
        single_compression_token_init = self._init_single_compressed(
            init_method=init_method,
            mvn_dist=mvn_dist,
            pca_components=pca_components,
            pca_mean=pca_mean,
            loaded_embeddings=loaded_embeddings,
            num_compression_tokens=num_compression_tokens,
            hidden_size=hidden_size,
        )
        return _RunContext(
            model=model,
            device=device,
            init_method=init_method,
            mvn_dist=mvn_dist,
            pca_components=pca_components,
            pca_mean=pca_mean,
            loaded_embeddings=loaded_embeddings,
            single_compression_token_init=single_compression_token_init,
            num_compression_tokens=num_compression_tokens,
            hidden_size=hidden_size,
        )

    def _init_single_compressed(
        self,
        *,
        init_method: str,
        mvn_dist,
        pca_components,
        pca_mean,
        loaded_embeddings,
        num_compression_tokens: int,
        hidden_size: int,
    ) -> torch.Tensor | None:
        """Pre-compute the shared seed embedding for single_*-init methods, or return None."""
        if not init_method.startswith("single_"):
            return None
        seed = create_compression_embedding(
            batch_size=1,
            num_compression_tokens=num_compression_tokens,
            hidden_size=hidden_size,
            init_method=init_method,
            mvn_dist=mvn_dist,
            token_embeddings=None,
            single_compression_token_embeddings_initialization=None,
            pca_components=pca_components,
            pca_mean=pca_mean,
            loaded_embeddings=loaded_embeddings,
        )
        return seed.data.detach().clone()

    def _prepare_batch_inputs(self, batch, ctx: _RunContext) -> _BatchInputs:
        """Move batch to device, compute token embeddings, target hidden states and compression mask."""
        ctx.model.eval()
        input_ids = batch.input_ids.squeeze(1).to(ctx.device)  # [batch, sequence]
        attention_mask = batch.attention_mask.squeeze(1).to(ctx.device)  # [batch, sequence]
        batch_size = input_ids.shape[0]
        with torch.no_grad():
            token_embeddings = ctx.model.get_input_embeddings()(input_ids)  # [batch, sequence, hidden]
        if self.args.loss_type.lower() != "cross_entropy":
            target_hidden_states = self.compute_hidden_states(ctx.model, token_embeddings, attention_mask)
        else:
            target_hidden_states = None
        compression_attention_mask = build_compression_attention_mask(
            batch_size,
            ctx.num_compression_tokens,
            dtype=attention_mask.dtype,
            device=ctx.device,
        )
        return _BatchInputs(
            input_ids=input_ids,
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
            target_hidden_states=target_hidden_states,
            compression_attention_mask=compression_attention_mask,
            batch_size=batch_size,
        )

    def _optimize_compression(self, inputs: _BatchInputs, ctx: _RunContext) -> _BatchOptimizationResult:
        """Build parametrization + optimizer + tracker, run the optimization loop, collect results."""
        parametrization = build_parametrization(
            init_method=ctx.init_method,
            batch_size=inputs.batch_size,
            num_compression_tokens=ctx.num_compression_tokens,
            hidden_size=ctx.hidden_size,
            device=ctx.device,
            init_helper=lambda: self._init_compression_token_embeddings(inputs, ctx),
            pca_components=ctx.pca_components,
            pca_mean=ctx.pca_mean,
            **self._extra_parametrization_kwargs(ctx),
        )
        optimizer, lr_scheduler = self._build_optimizer_and_scheduler(
            # Single optimizer covers both the batch-level parameters and any
            # parametrization-shared modules (e.g. a low-dim Linear projection).
            list(parametrization.parameters) + list(parametrization.shared_parameters),
            self.args.max_optimization_steps_per_sample,
            1,
        )
        guard = ConvergedSamplesGuard(parametrization.optimizable_tensor)
        tracker = ConvergenceTracker(
            max_optimization_steps=self.args.max_optimization_steps_per_sample,
            batch_size=inputs.batch_size,
            convergence_threshold=self.args.full_cramming_convergence_threshold,
        )

        last_loss, last_convergence_per_sample = self._run_optimization_loop(
            ctx=ctx,
            inputs=inputs,
            parametrization=parametrization,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            guard=guard,
            tracker=tracker,
        )

        return _BatchOptimizationResult(
            parametrization=parametrization,
            convergence_tracker=tracker,
            last_loss=last_loss,
            last_convergence_per_sample=last_convergence_per_sample,
            initialization_embeddings=parametrization.initialization_snapshot(),
        )

    def _init_compression_token_embeddings(self, inputs: _BatchInputs, ctx: _RunContext) -> torch.nn.Parameter:
        """Sample a fresh compression-token embedding for the current batch."""
        return create_compression_embedding(
            batch_size=inputs.batch_size,
            num_compression_tokens=ctx.num_compression_tokens,
            hidden_size=ctx.hidden_size,
            init_method=ctx.init_method,
            mvn_dist=ctx.mvn_dist,
            token_embeddings=inputs.token_embeddings,
            single_compression_token_embeddings_initialization=ctx.single_compression_token_init,
            pca_components=ctx.pca_components,
            pca_mean=ctx.pca_mean,
            loaded_embeddings=ctx.loaded_embeddings,
        )

    def _run_optimization_loop(
        self,
        *,
        ctx: _RunContext,
        inputs: _BatchInputs,
        parametrization,
        optimizer,
        lr_scheduler,
        guard: ConvergedSamplesGuard,
        tracker: ConvergenceTracker,
    ) -> tuple[float, torch.Tensor]:
        """Run gradient steps until full convergence or step budget is exhausted."""
        progress_bar = tqdm(
            range(self.args.max_optimization_steps_per_sample),
            total=self.args.max_optimization_steps_per_sample,
        )
        progress_bar.set_description("Training")

        last_loss: float = float("nan")
        last_convergence_per_sample: torch.Tensor = torch.zeros(inputs.batch_size)

        for step_i in progress_bar:
            # Combine compression token embeddings and sequence token embeddings
            compression_token_embeddings = parametrization.materialize()
            united_token_embeddings, united_attention_mask = build_united_input(
                compression_token_embeddings,
                inputs.compression_attention_mask,
                inputs.token_embeddings,
                inputs.attention_mask,
            )
            # Compute logits
            forward_extra_kwargs = {}
            if self.args.fix_position_ids:
                position_ids = torch.arange(
                    -ctx.num_compression_tokens,
                    inputs.token_embeddings.size(1),
                    device=inputs.token_embeddings.device,
                )
                position_ids[: ctx.num_compression_tokens] = 0
                position_ids = position_ids.repeat(inputs.token_embeddings.size(0), 1)
                forward_extra_kwargs["position_ids"] = position_ids

            compression_outputs = ctx.model(
                inputs_embeds=united_token_embeddings,
                attention_mask=united_attention_mask,
                output_hidden_states=inputs.target_hidden_states is not None,
                **forward_extra_kwargs,
            )

            # Compute loss
            loss, alignment_loss = self.compute_loss(
                compression_outputs.logits,
                inputs.input_ids,
                inputs.attention_mask,
                ctx.num_compression_tokens,
                compression_hidden_states=compression_outputs.hidden_states,
                target_hidden_states=inputs.target_hidden_states,
            )

            # Compute convergence
            convergence_per_sample = self.compute_convergence(
                compression_outputs.logits,
                inputs.input_ids,
                inputs.attention_mask,
                ctx.num_compression_tokens,
            )

            # Generate diagnostics
            generated_text, ground_truth_text = self.generate_diagnostics(
                ctx.model,
                inputs.input_ids,
                united_token_embeddings,
                ctx.num_compression_tokens,
            )

            # Plan B (measure-then-decide): decide convergence on the CURRENT (live,
            # just-forwarded) embedding *before* stepping. If the whole batch already
            # meets the threshold, stop here -- otherwise an extra optimizer step
            # drifts the saved embedding off the converged point and any
            # early-position argmax flip cascades catastrophically under
            # autoregressive generation. (Mirrors the same fix in
            # ProgressiveCrammingTrainer's stage loop.)
            last_loss = float(loss.item())
            last_convergence_per_sample = convergence_per_sample.detach().cpu()

            if tracker.update(step_i, convergence_per_sample):
                print(f"Early stopping: whole batch compression converged in {step_i} steps!")
                with torch.no_grad():
                    self._log_progress(
                        progress_bar=progress_bar,
                        loss=loss,
                        alignment_loss=alignment_loss,
                        convergence_per_sample=convergence_per_sample,
                        lr_scheduler=lr_scheduler,
                        compression_token_embeddings=compression_token_embeddings,
                        parametrization=parametrization,
                        generated_text=generated_text,
                        ground_truth_text=ground_truth_text,
                    )
                break

            # Back propagation and parameters update
            loss.backward()
            guard.before_step(tracker.fully_converged)
            optimizer.step()
            guard.after_step(tracker.fully_converged)

            with torch.no_grad():
                self._log_progress(
                    progress_bar=progress_bar,
                    loss=loss,
                    alignment_loss=alignment_loss,
                    convergence_per_sample=convergence_per_sample,
                    lr_scheduler=lr_scheduler,
                    compression_token_embeddings=compression_token_embeddings,
                    parametrization=parametrization,
                    generated_text=generated_text,
                    ground_truth_text=ground_truth_text,
                )

            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()

        return last_loss, last_convergence_per_sample

    def _log_progress(
        self,
        *,
        progress_bar,
        loss,
        alignment_loss,
        convergence_per_sample,
        lr_scheduler,
        compression_token_embeddings,
        parametrization,
        generated_text,
        ground_truth_text,
    ) -> None:
        """Update the progress bar postfix and forward scalars to TensorBoard."""
        progress_bar.update(1)
        progress_bar.set_postfix(
            loss=loss.item(),
            loss_alignment=(alignment_loss.item() if alignment_loss is not None else None),
            convergece_per_sample=convergence_per_sample.mean().item(),
            lr=lr_scheduler.get_last_lr()[0],
        )
        self._log_step(
            loss,
            alignment_loss,
            convergence_per_sample,
            compression_token_embeddings,
            lr_scheduler,
            generated_text,
            ground_truth_text,
            leaf_grad_params=list(parametrization.parameters) + list(parametrization.shared_parameters),
        )

    def _collect_batch_rows(
        self,
        *,
        inputs: _BatchInputs,
        result: _BatchOptimizationResult,
        ctx: _RunContext,
        sample_id_counter: int,
    ) -> tuple[list[dict], torch.Tensor]:
        """Compute Information Gain and assemble per-sample row dicts for this batch."""
        with torch.no_grad():
            final_compression_tokens = result.parametrization.materialize()
            compression_token_embeddings_cpu = final_compression_tokens.detach().cpu()
            pca_coefficients_to_save = result.parametrization.serialize_extras()

            per_sample_info_gain = compute_information_gain(
                model=ctx.model,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                token_embeddings=inputs.token_embeddings,
                compression_token_embeddings=final_compression_tokens,
                compression_attention_mask=inputs.compression_attention_mask,
            )

            steps_below_1_0 = result.convergence_tracker.steps_below(1.0)
            steps_below_0_99 = result.convergence_tracker.steps_below(0.99)
            steps_below_0_95 = result.convergence_tracker.steps_below(0.95)

            rows: list[dict] = [
                self._build_sample_row(
                    sample_index=j,
                    sample_id=sample_id_counter + j,
                    inputs=inputs,
                    ctx=ctx,
                    compression_token_embeddings_cpu=compression_token_embeddings_cpu,
                    initialization_embeddings=result.initialization_embeddings,
                    pca_coefficients_to_save=pca_coefficients_to_save,
                    last_loss=result.last_loss,
                    last_convergence_per_sample=result.last_convergence_per_sample,
                    steps_below_1_0=steps_below_1_0,
                    steps_below_0_99=steps_below_0_99,
                    steps_below_0_95=steps_below_0_95,
                    per_sample_info_gain=per_sample_info_gain,
                )
                for j in range(inputs.batch_size)
            ]

        return rows, compression_token_embeddings_cpu

    def _build_sample_row(
        self,
        *,
        sample_index: int,
        sample_id: int,
        inputs: _BatchInputs,
        ctx: _RunContext,
        compression_token_embeddings_cpu: torch.Tensor,
        initialization_embeddings: torch.Tensor,
        pca_coefficients_to_save: list | None,
        last_loss: float,
        last_convergence_per_sample: torch.Tensor,
        steps_below_1_0: torch.Tensor,
        steps_below_0_99: torch.Tensor,
        steps_below_0_95: torch.Tensor,
        per_sample_info_gain: list[float],
    ) -> dict:
        """Assemble the per-sample row dict (schema consumed by downstream eval scripts)."""
        sample_attention_mask = inputs.attention_mask[sample_index].bool()
        sample_input_ids = inputs.input_ids[sample_index][sample_attention_mask]
        sample_text = self.processing_class.decode(sample_input_ids, skip_special_tokens=True)
        return {
            "sample_id": sample_id,
            "text": sample_text,
            "input_ids": sample_input_ids.detach().cpu().numpy().tolist(),
            "embedding": compression_token_embeddings_cpu[sample_index].to(torch.float32).numpy().tolist(),
            "pca_coefficients": (pca_coefficients_to_save[sample_index] if pca_coefficients_to_save is not None else None),
            "initialization_embedding": initialization_embeddings[sample_index].to(torch.float32).numpy().tolist(),
            "final_loss": last_loss,
            "final_convergence": last_convergence_per_sample[sample_index].item(),
            "convergence_after_steps": int(steps_below_1_0[sample_index].item()),
            "convergence_0.99_after_steps": int(steps_below_0_99[sample_index].item()),
            "convergence_0.95_after_steps": int(steps_below_0_95[sample_index].item()),
            "compression_tokens_mean": float(compression_token_embeddings_cpu[sample_index].mean().item()),
            "compression_tokens_std": float(compression_token_embeddings_cpu[sample_index].std().item()),
            "num_input_tokens": int(sample_attention_mask.sum().item()),
            "num_compression_tokens": int(ctx.num_compression_tokens),
            "hidden_size": ctx.hidden_size,
            "fix_position_ids": self.args.fix_position_ids,
            "loss_type": self.args.loss_type,
            "hybrid_alpha": self.args.hybrid_alpha,
            "dtype": self.args.dtype,
            "embedding_init_method": self.args.embedding_init_method,
            "num_alignment_layers": self.args.num_alignment_layers,
            "model_checkpoint": self.args.model_checkpoint,
            "max_optimization_steps_per_sample": self.args.max_optimization_steps_per_sample,
            "information_gain_bits": float(per_sample_info_gain[sample_index]),
        }
