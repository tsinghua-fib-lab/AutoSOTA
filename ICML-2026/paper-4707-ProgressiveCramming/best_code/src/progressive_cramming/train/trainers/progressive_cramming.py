"""Progressive cramming trainer: progressively grow the target prefix until reconstruction fails."""

import copy
import os
from dataclasses import dataclass, field

import torch
from tqdm.auto import tqdm

from progressive_cramming.analysis import ProgressiveSampleStateMachine
from progressive_cramming.analysis.information_gain import (
    compute_information_gain,
    compute_prefix_surprisal_bits_per_token,
)
from progressive_cramming.train.inputs import build_compression_attention_mask, build_united_input
from progressive_cramming.train.parametrization import build_per_sample_parametrization
from progressive_cramming.train.trainers.base import BaseTrainer
from progressive_cramming.utils.launch import freeze_model_parameters


@dataclass
class _RunContext:
    """Run-level constants prepared once before the dataloader loop."""

    model: torch.nn.Module
    # Model used for all decode-side forwards (target-hidden, CE/alignment loss, info-gain).
    # In dual-checkpoint mode this is the reconstructor; otherwise it is ``model`` itself.
    # The compressor (``model``) is then used ONLY to seed ``compression_head_forward`` init.
    decode_model: torch.nn.Module
    device: torch.device
    init_method: str
    mvn_dist: object
    pca_components: torch.Tensor | None
    pca_mean: torch.Tensor | None
    loaded_embeddings: torch.Tensor | None
    num_compression_tokens: int
    threshold: float
    step_increment: int
    min_len: int
    max_stages_cap: int
    # Number of leading real (uncompressed) prefix tokens the model attends to before the crammed
    # continuation. 0 = no prefix (original behaviour). The prefix is never compressed into [mem];
    # loss/convergence/info-gain are measured only over the post-prefix continuation.
    prefix_len: int = 0
    # In ``--low_dim_projection_global`` mode the projection's ``nn.Linear``,
    # its AdamW state, and its LR scheduler all live for the entire run (as
    # in the pre-refactor implementation in commit a0d39f6). The same
    # ``nn.Linear`` object is handed to every batch's parametrization via
    # ``projection_module`` so that gradient steps and momentum/variance
    # accumulate continuously. All three fields are ``None`` when the
    # projection is per-batch (the default).
    shared_projection: torch.nn.Linear | None = None
    shared_optimizer: object = None
    shared_scheduler: object = None
    # Geometric growth: when True, double the prefix length on each converged stage
    # and bisect the gap once a stage fails (instead of growing by step_increment).
    geometric_growth: bool = False
    # Back-off strategy after the ramp brackets the horizon: "bisect" or "linear"
    # (restore the last converged checkpoint and grow +1 token per stage until failure).
    geometric_backoff: str = "bisect"
    # Artifact-based initialization: map from target sample index (in the new
    # experiment's dataloader order) to the converged embedding tensor loaded from
    # a prior experiment's progressive_prefixes dataset at a specific stage.
    # ``None`` when no artifact init is configured.
    artifact_init_embeddings: dict[int, torch.Tensor] = field(default_factory=dict)
    # The seq_len to resume progressive growth from for artifact-initialized
    # samples (artifact's ``stage_seq_len + 1``). 0 = not set.
    artifact_init_start_seq_len: int = 0


@dataclass
class _BatchContext:
    """Per-batch tensors and per-sample optimization state."""

    input_ids: torch.Tensor  # [batch, full_sequence]
    attention_mask: torch.Tensor  # [batch, full_sequence]
    full_token_embeddings: torch.Tensor  # [batch, full_sequence, hidden]
    target_hidden_states_full: tuple[torch.Tensor, ...]
    compression_attention_mask: torch.Tensor  # [batch, compression]
    batch_size: int
    hidden_size: int  # Always the model's hidden size; the parametrization
    # handles low-dim coefficient space internally.
    max_len: int
    parametrization: object
    per_sample_optimizers: list
    per_sample_schedulers: list
    initialization_embeddings: torch.Tensor
    # Optimizer for parametrization-shared parameters (e.g. low-dim Linear
    # projection weights). ``None`` when the parametrization has no shared
    # parameters (Direct / pretrained-PCA mode).
    shared_optimizer: object
    shared_scheduler: object
    # Fixed uncompressed prefix (``progressive_prefix_len`` tokens). The crammable tensors above
    # (input_ids/attention_mask/full_token_embeddings/target_hidden_states_full/max_len) are the
    # *continuation* only -- already re-based past the prefix. ``prefix_len == 0`` => no prefix and
    # the prefix tensors are ``None`` (identical to the original trainer).
    prefix_len: int = 0
    prefix_token_embeddings: torch.Tensor | None = None  # [batch, prefix, hidden]
    prefix_attention_mask: torch.Tensor | None = None  # [batch, prefix]
    prefix_surprisal_bits_per_token: list[float] | None = None  # per-sample base-LM prefix surprisal


@dataclass
class _StageContext:
    """Per-stage sliced tensors (one stage = one fixed seq_len target)."""

    seq_len: int
    stage_index: int
    input_ids: torch.Tensor  # [batch, seq_len]
    attention_mask: torch.Tensor  # [batch, seq_len]
    inputs_embeds: torch.Tensor  # [batch, seq_len, hidden]
    target_hidden_states: list[torch.Tensor]


class ProgressiveCrammingTrainer(BaseTrainer):
    """Trainer for progressive cramming: grow target prefix token-by-token until reconstruction fails."""

    def train(self) -> str | None:
        """Run progressive training. Returns save path or None."""
        ctx = self._build_run_context()
        collected_rows: list[dict] = []
        sample_id_counter = 0
        # The last batch's parametrization is exposed to ``_on_training_complete``
        # so the projection weights can be persisted in non-global mode (in
        # global mode we save ``ctx.shared_projection`` directly).
        final_parametrization = None

        dataloader = self._create_dataloader()
        for batch in tqdm(dataloader):
            batch_ctx = self._setup_batch(batch, ctx, sample_id_counter=sample_id_counter)
            collected_rows.extend(self._run_progressive_stages(batch_ctx, ctx, sample_id_counter))
            sample_id_counter += batch_ctx.batch_size
            final_parametrization = batch_ctx.parametrization

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        self._on_training_complete(final_parametrization, ctx)

        return self._save_artifacts(
            collected_rows,
            tensor=None,
            tensor_filename="compression_embeddings.pt",
            subdir_name="progressive_prefixes",
        )

    def _on_training_complete(self, parametrization, ctx: _RunContext) -> None:
        """Hook called once after all batches finish. Persists low-dim projection if any.

        Two cases:
        * ``--low_dim_projection_global``: ``ctx.shared_projection`` is the one
          ``nn.Linear`` that lived through the run — save its state_dict.
        * non-global: ``parametrization`` is the *last* batch's parametrization
          (per-batch projection), so its ``shared_state_dict()`` is the most
          recently trained weights.
        """
        if not self.args.low_dim_projection or not self.args.low_dim_projection_train:
            return
        if ctx.shared_projection is not None:
            shared_state = ctx.shared_projection.state_dict()
        elif parametrization is not None:
            shared_state = parametrization.shared_state_dict()
        else:
            return
        if shared_state is None:
            return
        output_dir = self.args.output_dir
        if not output_dir:
            return
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "low_dim_projection.pt")
        torch.save(
            {
                "low_dim_projection": shared_state,
                "low_dim_size": self.args.low_dim_size,
                "hidden_size": ctx.model.config.hidden_size,
            },
            save_path,
        )
        print(f"Saved low-dimensional projection weights to {save_path}")

    # ------------------------------------------------------------------
    # Run-level setup (once per train()).
    # ------------------------------------------------------------------

    def _build_run_context(self) -> _RunContext:
        """Seed RNG, freeze model, prepare embedding-init helpers.

        In ``--low_dim_projection_global`` mode we additionally construct the
        run-shared ``nn.Linear`` projection, its AdamW optimizer, and its LR
        scheduler — exactly once, before the data loop — so that they live
        across all batches (matching the pre-refactor implementation in
        a0d39f6, where global mode created these three objects above
        ``for batch in dataloader:``).
        """
        model, device, init_method, mvn_dist, pca_components, pca_mean, loaded_embeddings = self._initialize_run()

        prefix_len = int(getattr(self.args, "progressive_prefix_len", 0) or 0)
        if prefix_len < 0:
            raise ValueError(f"progressive_prefix_len must be >= 0, got {prefix_len}!")
        if prefix_len > 0 and init_method == "compression_head_forward":
            raise ValueError(
                "progressive_prefix_len > 0 is not compatible with embedding_init_method="
                "'compression_head_forward' (the head seeds the [mem] token from the sequence, which "
                "conflicts with a separate uncompressed prefix)."
            )

        # Dual checkpoint support: when a sibling reconstructor model is provided, the
        # compressor (`model`) is used ONLY to produce the initial compression-embedding
        # value (compression_head_forward init); the inner optimization loop drives
        # gradients through `decode_model` (the reconstructor) so the compression token
        # is co-adapted to whichever model will decode it at downstream inference.
        model_reconstructor = getattr(self, "model_reconstructor", None)
        if model_reconstructor is not None:
            print("[two-model] progressive cramming dual mode: compressor=init, reconstructor=decode")
            model_reconstructor = model_reconstructor.to(device)
            freeze_model_parameters(model_reconstructor)
            model_reconstructor.eval()
        decode_model = model_reconstructor if model_reconstructor is not None else model

        shared_projection: torch.nn.Linear | None = None
        shared_optimizer = None
        shared_scheduler = None
        if self.args.low_dim_projection and self.args.low_dim_projection_global:
            shared_projection = self._build_low_dim_projection_module(
                hidden_size=model.config.hidden_size,
                device=device,
            )
            if self.args.low_dim_projection_train:
                shared_optimizer, shared_scheduler = self._build_optimizer_and_scheduler(
                    list(shared_projection.parameters()),
                    num_training_steps=self.args.max_optimization_steps_per_sample,
                )

        artifact_init_embeddings: dict[int, torch.Tensor] = {}
        artifact_init_start_seq_len = 0
        if self.args.progressive_init_from_artifact:
            artifact_init_embeddings, artifact_init_start_seq_len = self._load_artifact_init()

        return _RunContext(
            model=model,
            decode_model=decode_model,
            device=device,
            init_method=init_method,
            mvn_dist=mvn_dist,
            pca_components=pca_components,
            pca_mean=pca_mean,
            loaded_embeddings=loaded_embeddings,
            num_compression_tokens=self.args.number_of_mem_tokens,
            threshold=self.args.progressive_convergence_threshold,
            step_increment=self.args.progressive_step,
            min_len=self.args.progressive_min_seq_len,
            max_stages_cap=self.args.progressive_max_stages,
            prefix_len=int(getattr(self.args, "progressive_prefix_len", 0) or 0),
            shared_projection=shared_projection,
            shared_optimizer=shared_optimizer,
            shared_scheduler=shared_scheduler,
            geometric_growth=bool(self.args.progressive_geometric_growth),
            geometric_backoff=str(self.args.progressive_geometric_backoff),
            artifact_init_embeddings=artifact_init_embeddings,
            artifact_init_start_seq_len=artifact_init_start_seq_len,
        )

    def _load_artifact_init(self) -> tuple[dict[int, torch.Tensor], int]:
        """Load converged embeddings from a prior experiment's progressive_prefixes dataset.

        Returns ``(artifact_embeddings, start_seq_len)`` where
        ``artifact_embeddings`` maps target sample index → embedding tensor
        ``[num_compression_tokens, hidden_size]`` and ``start_seq_len`` is the
        stage's ``seq_len + 1`` (the first length the resumed sample should
        attempt).
        """
        from datasets import Dataset

        artifact_path = self.args.progressive_init_from_artifact
        stage = self.args.progressive_init_from_stage
        if stage is None:
            raise ValueError("--progressive_init_from_stage is required when --progressive_init_from_artifact is set")

        # Accept either the experiment root or the progressive_prefixes subdir.
        pp_subdir = os.path.join(artifact_path, "progressive_prefixes")
        dataset_path = pp_subdir if os.path.isdir(pp_subdir) else artifact_path

        ds = Dataset.load_from_disk(dataset_path)

        sample_ids_str = self.args.progressive_init_sample_ids or "0"
        source_sample_ids = [int(x.strip()) for x in sample_ids_str.split(",")]

        artifact_embeddings: dict[int, torch.Tensor] = {}
        start_seq_len = 0

        for target_idx, source_id in enumerate(source_sample_ids):
            # Efficient lookup: filter to the exact (sample_id, stage_index) pair.
            matching = ds.filter(
                lambda row: row["sample_id"] == source_id and row["stage_index"] == stage,
                num_proc=1,
            )
            if len(matching) == 0:
                raise ValueError(
                    f"No row found for sample_id={source_id}, stage_index={stage} in {dataset_path}. "
                    f"Available sample_ids: {sorted(set(ds['sample_id']))[:10]}..."
                )
            row = matching[0]
            emb_flat = torch.tensor(row["embedding"], dtype=torch.float32)
            hidden_size = row["hidden_size"]
            num_compression_tokens = row["num_compression_tokens"]
            emb = emb_flat.reshape(num_compression_tokens, hidden_size)
            artifact_embeddings[target_idx] = emb
            start_seq_len = max(start_seq_len, row["stage_seq_len"] + 1)
            print(
                f"[artifact init] target sample {target_idx} ← source sample_id={source_id} "
                f"stage={stage} seq_len={row['stage_seq_len']} "
                f"embedding shape={tuple(emb.shape)}"
            )

        print(f"[artifact init] loaded {len(artifact_embeddings)} embedding(s), resume from seq_len={start_seq_len}")
        return artifact_embeddings, start_seq_len

    # ------------------------------------------------------------------
    # Per-batch setup.
    # ------------------------------------------------------------------

    def _setup_batch(self, batch, ctx: _RunContext, *, sample_id_counter: int = 0) -> _BatchContext:
        """Move batch to device, compute target hidden states, build per-sample parametrization + optimizers."""
        input_ids = batch.input_ids.squeeze(1).to(ctx.device)  # [batch, sequence]
        attention_mask = batch.attention_mask.squeeze(1).to(ctx.device)  # [batch, sequence]
        batch_size = input_ids.shape[0]
        with torch.no_grad():
            # Embeddings + target hidden states must come from the model that will decode the
            # prefix (the reconstructor in dual mode, the compressor otherwise).
            full_token_embeddings = ctx.decode_model.get_input_embeddings()(input_ids)  # [batch, sequence, hidden]
        target_hidden_states_full = self.compute_hidden_states(ctx.decode_model, full_token_embeddings, attention_mask)
        hidden_size = full_token_embeddings.shape[-1]  # Always model's hidden size now.

        # ``compression_head_forward`` seeds each sample's compression embedding by running the
        # trained compression head over its prefix; computed here where the model + batch are in scope.
        ch_forward_init = None
        if ctx.init_method == "compression_head_forward":
            ch_forward_init = self._compute_compression_head_init(ctx.model, input_ids, attention_mask)

        # Fixed uncompressed prefix: peel the first ``prefix_len`` tokens off as a real, visible
        # context block the model attends to but never crams, then re-base every "crammable" tensor
        # to the continuation (positions ``prefix_len:``). Progressive growth and loss then operate
        # only on the continuation; ``prefix_len == 0`` leaves everything untouched.
        prefix_token_embeddings = None
        prefix_attention_mask = None
        prefix_surprisal_bits_per_token = None
        if ctx.prefix_len > 0:
            prefix_len = ctx.prefix_len
            prefix_input_ids = input_ids[:, :prefix_len]
            prefix_attention_mask = attention_mask[:, :prefix_len]
            prefix_token_embeddings = full_token_embeddings[:, :prefix_len, :]
            prefix_surprisal_bits_per_token = compute_prefix_surprisal_bits_per_token(
                model=ctx.decode_model,
                prefix_input_ids=prefix_input_ids,
                prefix_attention_mask=prefix_attention_mask,
            )
            input_ids = input_ids[:, prefix_len:]
            attention_mask = attention_mask[:, prefix_len:]
            full_token_embeddings = full_token_embeddings[:, prefix_len:, :]
            target_hidden_states_full = tuple(h[:, prefix_len:] for h in target_hidden_states_full)

            if self.writer is not None and prefix_surprisal_bits_per_token:
                mean_surprisal = sum(prefix_surprisal_bits_per_token) / len(prefix_surprisal_bits_per_token)
                self.writer.add_scalar("prefix/surprisal_bits_per_token", mean_surprisal, self.global_step)

        parametrization = self._build_parametrization(
            ctx, batch_size, hidden_size, ch_forward_init=ch_forward_init, sample_id_counter=sample_id_counter
        )
        per_sample_optimizers, per_sample_schedulers = self._build_per_sample_optimizers(parametrization, ctx)
        # Global mode: reuse the run-level optimizer/scheduler — they own the
        # AdamW state and LR-curve position accumulated across all previous
        # batches. Non-global mode: create a fresh per-batch pair.
        if ctx.shared_projection is not None:
            shared_optimizer, shared_scheduler = ctx.shared_optimizer, ctx.shared_scheduler
        else:
            shared_optimizer, shared_scheduler = self._build_shared_optimizer(parametrization)
        compression_attention_mask = build_compression_attention_mask(
            batch_size,
            ctx.num_compression_tokens,
            device=ctx.device,
            dtype=attention_mask.dtype,
        )

        per_sample_lengths = attention_mask.sum(dim=1).tolist()
        max_len = int(max(per_sample_lengths)) if per_sample_lengths else attention_mask.shape[1]

        return _BatchContext(
            input_ids=input_ids,
            attention_mask=attention_mask,
            full_token_embeddings=full_token_embeddings,
            target_hidden_states_full=target_hidden_states_full,
            compression_attention_mask=compression_attention_mask,
            batch_size=batch_size,
            hidden_size=hidden_size,
            max_len=max_len,
            parametrization=parametrization,
            per_sample_optimizers=per_sample_optimizers,
            per_sample_schedulers=per_sample_schedulers,
            initialization_embeddings=parametrization.initialization_snapshot(),
            shared_optimizer=shared_optimizer,
            shared_scheduler=shared_scheduler,
            prefix_len=ctx.prefix_len,
            prefix_token_embeddings=prefix_token_embeddings,
            prefix_attention_mask=prefix_attention_mask,
            prefix_surprisal_bits_per_token=prefix_surprisal_bits_per_token,
        )

    def _compute_compression_head_init(self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run the compression head over each sample's full prefix to seed its compression embedding.

        Returns a ``[batch, num_compression_tokens, hidden]`` float32 tensor. Requires a
        ``LlamaForCausalLMCompressionHead`` (i.e. a model exposing ``compression_head`` and returning
        ``compression_embeds``); used by ``--embedding_init_method compression_head_forward`` to evaluate a
        trained compression head with the progressive trainer.
        """
        if not hasattr(model, "compression_head"):
            raise ValueError(
                "embedding_init_method='compression_head_forward' requires a compression-head model "
                "(LlamaForCausalLMCompressionHead); the loaded model has no 'compression_head'. "
                "Point --model_checkpoint at a compression-head checkpoint."
            )
        lengths = attention_mask.sum(dim=1).to(torch.long).clamp_min(1)
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prefix_lengths=lengths,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
        embeds = getattr(out, "compression_embeds", None)
        if embeds is None:
            raise RuntimeError("compression_head_forward init: model returned no compression_embeds.")
        num_queries = embeds.shape[1]
        if num_queries != self.args.number_of_mem_tokens:
            raise ValueError(
                f"compression head produced {num_queries} query embedding(s) but "
                f"--number_of_mem_tokens={self.args.number_of_mem_tokens}. Set --number_of_mem_tokens {num_queries}."
            )
        return embeds.detach().to(torch.float32)

    def _build_parametrization(
        self, ctx: _RunContext, batch_size: int, hidden_size: int, ch_forward_init=None, sample_id_counter: int = 0
    ):
        """Construct the per-sample parametrization (Direct, PretrainedPCA, or low-dim projected).

        Low-dim semantics — the trainer always builds the ``nn.Linear`` itself
        and hands it to the parametrization via ``projection_module``:
        * ``--low_dim_projection_global``: reuse ``ctx.shared_projection`` (one
          Linear for the whole run; AdamW/scheduler in ``ctx`` accumulate
          state across batches).
        * ``--low_dim_projection`` alone: build a **fresh** Linear here per
          batch (warm-started from ``--low_dim_projection_checkpoint`` if
          given, frozen when ``--low_dim_projection_train`` is False). This
          matches the pre-refactor behaviour (commit a0d39f6) where the
          projection constructor was invoked inside the data loop.
        """
        # Coefficient init lives in low-dim space when low-dim is on; otherwise
        # in model hidden space.
        init_dim = self.args.low_dim_size if self.args.low_dim_projection else hidden_size

        if ctx.init_method == "compression_head_forward" and self.args.low_dim_projection:
            raise ValueError("embedding_init_method='compression_head_forward' is not compatible with --low_dim_projection.")

        def _init_helper():
            if ctx.init_method == "compression_head_forward":
                assert ch_forward_init is not None, "compression_head_forward init tensor was not precomputed"
                return torch.nn.Parameter(ch_forward_init.to(device=ctx.device, dtype=torch.float32))
            init_tensor = self._init_compression_tokens(
                batch_size,
                ctx.num_compression_tokens,
                init_dim,
                ctx.init_method,
                ctx.mvn_dist,
                pca_components=ctx.pca_components,
                pca_mean=ctx.pca_mean,
                loaded_embeddings=ctx.loaded_embeddings,
            )
            # Override specific samples with embeddings loaded from a prior experiment's artifacts.
            if ctx.artifact_init_embeddings:
                if self.args.low_dim_projection:
                    raise ValueError(
                        "--progressive_init_from_artifact with --low_dim_projection is not supported: "
                        "the artifact stores materialized (hidden-dim) embeddings but the target "
                        "parametrization expects low-dim coefficients."
                    )
                for target_idx, emb in ctx.artifact_init_embeddings.items():
                    sample_idx_in_batch = target_idx - sample_id_counter
                    if 0 <= sample_idx_in_batch < batch_size:
                        init_tensor.data[sample_idx_in_batch] = emb.to(init_tensor.device, init_tensor.dtype)
            return init_tensor

        projection_module = None
        if self.args.low_dim_projection:
            projection_module = ctx.shared_projection or self._build_low_dim_projection_module(
                hidden_size=hidden_size,
                device=ctx.device,
            )

        return build_per_sample_parametrization(
            init_method=ctx.init_method,
            batch_size=batch_size,
            num_compression_tokens=ctx.num_compression_tokens,
            hidden_size=hidden_size,
            device=ctx.device,
            init_helper=_init_helper,
            pca_components=ctx.pca_components,
            pca_mean=ctx.pca_mean,
            low_dim_train=self.args.low_dim_projection,
            low_dim_size=self.args.low_dim_size,
            train_projection=self.args.low_dim_projection_train,
            projection_module=projection_module,
        )

    def _build_shared_optimizer(self, parametrization):
        """One optimizer for ``parametrization.shared_parameters`` (e.g. low-dim projection)."""
        if not parametrization.shared_parameters:
            return None, None
        return self._build_optimizer_and_scheduler(
            list(parametrization.shared_parameters),
            num_training_steps=self.args.max_optimization_steps_per_sample,
        )

    def _build_per_sample_optimizers(self, parametrization, ctx: _RunContext):
        """One optimizer/scheduler per sample. PCA path uses constant LR; direct path uses cosine over the full per-sample step budget."""
        per_sample_optimizers, per_sample_schedulers = [], []
        for parameter in parametrization.parameters:
            if ctx.init_method == "pretrained_pca":
                optimizer, scheduler = self._build_optimizer_and_scheduler([parameter])
            else:
                optimizer, scheduler = self._build_optimizer_and_scheduler(
                    [parameter],
                    num_training_steps=self.args.max_optimization_steps_per_sample,
                )
            per_sample_optimizers.append(optimizer)
            per_sample_schedulers.append(scheduler)
        return per_sample_optimizers, per_sample_schedulers

    # ------------------------------------------------------------------
    # Stage progression.
    # ------------------------------------------------------------------

    def _run_progressive_stages(self, batch_ctx: _BatchContext, ctx: _RunContext, sample_id_counter: int) -> list[dict]:
        """Outer stage-while loop: grow seq_len, run a stage, save rows, repeat until cap / all-skipped / max-len."""
        state = ProgressiveSampleStateMachine(batch_ctx.batch_size, ctx.threshold)
        if ctx.geometric_growth:
            return self._run_geometric_stages(batch_ctx, ctx, state, sample_id_counter)

        # When resuming from an artifact, start from the artifact's stage_seq_len + 1
        # for batches that contain artifact-initialized samples.
        effective_min = ctx.min_len
        if ctx.artifact_init_start_seq_len > 0:
            batch_has_artifact_sample = any(
                sample_id_counter <= target_idx < sample_id_counter + batch_ctx.batch_size
                for target_idx in ctx.artifact_init_embeddings
            )
            if batch_has_artifact_sample:
                effective_min = max(effective_min, ctx.artifact_init_start_seq_len)
                print(
                    f"[artifact init] batch starting at sample_id={sample_id_counter}: "
                    f"resuming from seq_len={effective_min}"
                )

        seq_len = min(effective_min, batch_ctx.max_len)
        stage_index = 0
        rows: list[dict] = []

        while True:
            stage_ctx = self._setup_stage(batch_ctx, seq_len, stage_index)
            last_loss, last_convergence = self._run_stage_loop(
                batch_ctx, stage_ctx, ctx, state, self.args.max_optimization_steps_per_token
            )
            state.mark_skipped_if_not_converged(seq_len)
            rows.extend(
                self._collect_stage_rows(batch_ctx, stage_ctx, ctx, state, last_loss, last_convergence, sample_id_counter)
            )

            stage_index += 1
            if seq_len >= batch_ctx.max_len:
                break
            if ctx.max_stages_cap and stage_index >= ctx.max_stages_cap:
                break
            if state.all_skipped:
                print("All samples skipped. Stopping at seq_len =", seq_len)
                break
            seq_len = min(seq_len + ctx.step_increment, batch_ctx.max_len)

        return rows

    def _run_geometric_stages(
        self, batch_ctx: _BatchContext, ctx: _RunContext, state: ProgressiveSampleStateMachine, sample_id_counter: int
    ) -> list[dict]:
        """Find the compression horizon by geometric growth + back-off (batch_size 1).

        Instead of growing the prefix a fixed ``progressive_step`` per converged
        stage, double the prefix length each time a stage converges (a
        geometrically growing number of added tokens). The first stage that fails
        brackets the horizon between the largest converged length (``lo``) and the
        smallest failed length (``hi``). ``ctx.geometric_backoff`` then pins the
        horizon inside that bracket:

        * ``"bisect"``: bisect ``(lo, hi)`` until the two are within
          ``progressive_step``, restoring the last converged state before each probe
          (~log2(gap) probes).
        * ``"linear"``: restore the last converged checkpoint once, then grow the
          prefix +1 token per stage (each warm-started from the previous converged
          stage) until a stage fails -- the exact horizon, mirroring Delta=1 cramming
          in the horizon neighborhood at the cost of ``horizon - lo`` probes.

        Back-off probes warm-start from the last *converged* state, not the
        preceding *failed* probe: every time a probe converges we snapshot the
        embedding + per-sample optimizer (Adam moments) + LR-scheduler position, and
        restore that snapshot before backing off. Otherwise a back-off target would
        inherit an embedding tuned for a too-long prefix and a cosine LR already
        decayed by the wasted failed-probe steps, which can make a sub-horizon prefix
        spuriously fail and under-report the horizon.

        ``lo`` only ever advances on a converged stage, so the reported prefix is
        always fully reconstructed (floor: ``progressive_min_seq_len`` -- a short
        prefix that converges trivially), preserving progressive cramming's
        convergence guarantee. The horizon is reached in ~log2(horizon) +
        log2(gap) stages rather than one per token; every stage uses the per-token
        budget and the cumulative per-sample budget bounds the whole search (so
        "Steps to Converge" stays comparable to the fixed-step arms). Every probe
        is recorded as a stage row; downstream ``converged_prefix_len`` already
        ignores non-converged stages.
        """
        if batch_ctx.batch_size != 1:
            raise ValueError("--progressive_geometric_growth requires per_device_train_batch_size=1.")
        if ctx.geometric_backoff not in ("bisect", "linear"):
            raise ValueError(f"--progressive_geometric_backoff must be 'bisect' or 'linear', got {ctx.geometric_backoff!r}.")

        rows: list[dict] = []
        max_len = batch_ctx.max_len
        min_len = max(1, min(ctx.min_len, max_len))
        step = max(1, ctx.step_increment)
        stage_index = 0
        lo: int | None = None  # largest length proven to converge
        hi: int | None = None  # smallest length proven to fail
        best_state: dict | None = None  # optimization state at the largest converged length (lo)

        def probe(seq_len: int, phase: str) -> bool:
            """Run one stage at ``seq_len``; record its row; update lo/hi; return convergence.

            On convergence the length is the new ``lo`` (probes only ever grow lo),
            so we snapshot the optimization state to warm-start later back-off probes.
            """
            nonlocal stage_index, lo, hi, best_state
            seq_len = int(max(min_len, min(seq_len, max_len)))
            stage_ctx = self._setup_stage(batch_ctx, seq_len, stage_index)
            last_loss, last_convergence = self._run_stage_loop(
                batch_ctx, stage_ctx, ctx, state, self.args.max_optimization_steps_per_token
            )
            rows.extend(
                self._collect_stage_rows(batch_ctx, stage_ctx, ctx, state, last_loss, last_convergence, sample_id_counter)
            )
            match_ratio = float(last_convergence[0].item()) if last_convergence is not None else float("nan")
            converged = bool(last_convergence is not None and match_ratio >= ctx.threshold)
            if converged:
                lo = seq_len if lo is None else max(lo, seq_len)
                best_state = self._snapshot_optimization_state(batch_ctx)
            else:
                hi = seq_len if hi is None else min(hi, seq_len)
            steps_done = int(state.steps_taken[0]) if hasattr(state, "steps_taken") else -1
            print(
                f"[geometric s{sample_id_counter}] stage {stage_index:>2} {phase:<6} "
                f"seq_len={seq_len:<5} match={match_ratio:.4f} "
                f"{'CONVERGED' if converged else 'failed   '} "
                f"bracket=(lo={lo}, hi={hi}) steps={steps_done}/{self.args.max_optimization_steps_per_sample}",
                flush=True,
            )
            if self.writer is not None:
                self.writer.add_scalar(f"geometric/seq_len/sample_{sample_id_counter}", seq_len, stage_index)
                self.writer.add_scalar(f"geometric/match_ratio/sample_{sample_id_counter}", match_ratio, stage_index)
            stage_index += 1
            return converged

        def exhausted() -> bool:
            """Stop the search once the per-sample step budget is spent or the stage cap is hit."""
            return state.skipped[0] or bool(ctx.max_stages_cap and stage_index >= ctx.max_stages_cap)

        # --- Geometric ramp up: double the prefix length until a stage fails. ---
        print(
            f"[geometric s{sample_id_counter}] ramp up from seq_len={min_len} " f"(doubling, max_len={max_len}, step={step})",
            flush=True,
        )
        length = min_len
        while True:
            converged = probe(length, phase="ramp")
            if exhausted():
                print(f"[geometric s{sample_id_counter}] budget exhausted during ramp; horizon>={lo}", flush=True)
                return rows
            if not converged:
                break  # (lo, hi) now brackets the horizon -> bisect
            if length >= max_len:
                print(f"[geometric s{sample_id_counter}] converged at full prefix (seq_len={lo}); done", flush=True)
                return rows  # converged at the full prefix; nothing left to grow
            nxt = min(length * 2, max_len)
            if nxt <= length:
                nxt = min(length + step, max_len)
            length = nxt

        # --- Geometric back-off: pin the horizon inside the bracket (lo, hi). ---
        if lo is None or hi is None:
            return rows  # never bracketed (e.g. even the shortest prefix failed, or never failed)

        if ctx.geometric_backoff == "linear":
            # Restore the last converged checkpoint, then grow +1 token per stage (each
            # warm-started from the previous converged stage) until a stage fails. The
            # first failure pins the horizon at the last converged length (lo).
            print(
                f"[geometric s{sample_id_counter}] bracketed horizon in (lo={lo}, hi={hi}); "
                f"linear +1 back-off from the converged checkpoint",
                flush=True,
            )
            if best_state is not None:
                self._restore_optimization_state(batch_ctx, best_state)
            while lo + 1 < hi:
                if not probe(lo + 1, phase="linear"):
                    break  # first failure -> horizon is the last converged length (lo)
                if exhausted():
                    print(
                        f"[geometric s{sample_id_counter}] budget exhausted during linear back-off; horizon>={lo}",
                        flush=True,
                    )
                    return rows
            print(f"[geometric s{sample_id_counter}] pinned horizon: converged_prefix_len={lo} (hi={hi})", flush=True)
            return rows

        # Default ("bisect"): restore the last converged state and bisect (lo, hi).
        print(
            f"[geometric s{sample_id_counter}] bracketed horizon in (lo={lo}, hi={hi}); " f"bisecting down to step={step}",
            flush=True,
        )
        while hi - lo > step:
            mid = (lo + hi) // 2
            if mid <= lo:
                break
            # Warm-start the shorter bisection target from the last converged state
            # (embedding + Adam moments + LR position), not the preceding failed probe.
            if best_state is not None:
                self._restore_optimization_state(batch_ctx, best_state)
            probe(mid, phase="bisect")
            if exhausted():
                print(f"[geometric s{sample_id_counter}] budget exhausted during bisection; horizon>={lo}", flush=True)
                return rows

        print(f"[geometric s{sample_id_counter}] pinned horizon: converged_prefix_len={lo} (hi={hi})", flush=True)
        return rows

    def _snapshot_optimization_state(self, batch_ctx: _BatchContext) -> dict:
        """Capture the per-sample embedding params + optimizer/scheduler state (and shared, if any).

        Used by geometric back-off to warm-start each bisection probe from the last
        *converged* stage. We clone the parameter tensors (the optimizers keep
        referencing the live ``nn.Parameter`` objects, so ``param.copy_`` on restore
        is enough) and deep-copy the optimizer/scheduler ``state_dict``s so later
        steps do not mutate the snapshot.
        """
        return {
            "params": [p.detach().clone() for p in batch_ctx.parametrization.parameters],
            "shared_params": [p.detach().clone() for p in batch_ctx.parametrization.shared_parameters],
            "optimizers": [copy.deepcopy(o.state_dict()) for o in batch_ctx.per_sample_optimizers],
            "schedulers": [copy.deepcopy(s.state_dict()) if s is not None else None for s in batch_ctx.per_sample_schedulers],
            "shared_optimizer": (
                copy.deepcopy(batch_ctx.shared_optimizer.state_dict()) if batch_ctx.shared_optimizer is not None else None
            ),
            "shared_scheduler": (
                copy.deepcopy(batch_ctx.shared_scheduler.state_dict()) if batch_ctx.shared_scheduler is not None else None
            ),
        }

    def _restore_optimization_state(self, batch_ctx: _BatchContext, snapshot: dict) -> None:
        """Restore the embedding params + optimizer/scheduler state captured by ``_snapshot_optimization_state``.

        ``optimizer.load_state_dict`` / ``scheduler.load_state_dict`` rewind the Adam
        moments and the cosine-LR step count to the converged anchor; copying the
        saved tensors into the live parameters keeps those optimizer param references
        valid.
        """
        with torch.no_grad():
            for param, saved in zip(batch_ctx.parametrization.parameters, snapshot["params"]):
                param.copy_(saved)
            for param, saved in zip(batch_ctx.parametrization.shared_parameters, snapshot["shared_params"]):
                param.copy_(saved)
        for optimizer, saved in zip(batch_ctx.per_sample_optimizers, snapshot["optimizers"]):
            optimizer.load_state_dict(saved)
        for scheduler, saved in zip(batch_ctx.per_sample_schedulers, snapshot["schedulers"]):
            if scheduler is not None and saved is not None:
                scheduler.load_state_dict(saved)
        if batch_ctx.shared_optimizer is not None and snapshot["shared_optimizer"] is not None:
            batch_ctx.shared_optimizer.load_state_dict(snapshot["shared_optimizer"])
        if batch_ctx.shared_scheduler is not None and snapshot["shared_scheduler"] is not None:
            batch_ctx.shared_scheduler.load_state_dict(snapshot["shared_scheduler"])

    def _setup_stage(self, batch_ctx: _BatchContext, seq_len: int, stage_index: int) -> _StageContext:
        """Slice batch tensors to the current stage's seq_len."""
        return _StageContext(
            seq_len=seq_len,
            stage_index=stage_index,
            input_ids=batch_ctx.input_ids[:, :seq_len],
            attention_mask=batch_ctx.attention_mask[:, :seq_len],
            inputs_embeds=batch_ctx.full_token_embeddings[:, :seq_len, :],
            target_hidden_states=[h[:, :seq_len] for h in batch_ctx.target_hidden_states_full],
        )

    def _run_stage_loop(
        self,
        batch_ctx: _BatchContext,
        stage_ctx: _StageContext,
        ctx: _RunContext,
        state: ProgressiveSampleStateMachine,
        max_steps_this_stage: int,
    ):
        """One stage with optional scheduler-reset retry on non-convergence. Returns (last_loss, last_convergence)."""
        state.reset_stage()
        scheduler_reset_used = False
        last_loss, last_convergence = None, None

        while True:
            last_loss, last_convergence, converged = self._run_steps(
                batch_ctx, stage_ctx, ctx, state, retry=scheduler_reset_used, max_steps=max_steps_this_stage
            )
            if converged:
                return last_loss, last_convergence
            if not self.args.progressive_reset_lr_scheduler_on_non_convergence or scheduler_reset_used:
                return last_loss, last_convergence
            print(f"Not converged at seq_len={stage_ctx.seq_len}, resetting LR schedulers for non-converged samples...")
            self._reset_per_sample_optimizers(batch_ctx, ctx, state, max_steps_this_stage)
            scheduler_reset_used = True

    def _reset_per_sample_optimizers(
        self, batch_ctx: _BatchContext, ctx: _RunContext, state: ProgressiveSampleStateMachine, max_steps: int
    ) -> None:
        """Rebuild optimizers/schedulers for samples still active in the current stage."""
        for j in range(batch_ctx.batch_size):
            if not state.is_active(j):
                continue
            parameter = batch_ctx.parametrization.parameters[j]
            if ctx.init_method == "pretrained_pca":
                optimizer, scheduler = self._build_optimizer_and_scheduler([parameter])
            else:
                optimizer, scheduler = self._build_optimizer_and_scheduler(
                    [parameter],
                    num_training_steps=max_steps,
                )
            batch_ctx.per_sample_optimizers[j] = optimizer
            batch_ctx.per_sample_schedulers[j] = scheduler

    def _run_steps(
        self,
        batch_ctx: _BatchContext,
        stage_ctx: _StageContext,
        ctx: _RunContext,
        state: ProgressiveSampleStateMachine,
        retry: bool,
        max_steps: int,
    ):
        """Inner step loop within a stage. Returns (last_loss, last_convergence, converged_bool)."""
        progress_bar = tqdm(
            range(max_steps),
            total=max_steps,
            leave=False,
        )
        progress_bar.set_description(f"Stage L={stage_ctx.seq_len}" + (" (retry)" if retry else ""))

        last_loss, last_convergence = None, None

        for _ in progress_bar:
            # `materialize()` returns the hidden-size embedding already projected
            # through the parametrization (low-dim Linear if applicable).
            compression_token_embeddings = batch_ctx.parametrization.materialize().clone()

            united_token_embeddings, united_attention_mask = build_united_input(
                compression_token_embeddings,
                batch_ctx.compression_attention_mask,
                stage_ctx.inputs_embeds,
                stage_ctx.attention_mask,
                prefix_token_embeddings=batch_ctx.prefix_token_embeddings,
                prefix_attention_mask=batch_ctx.prefix_attention_mask,
            )
            (
                loss,
                alignment_loss,
                convergence_per_sample,
                generated_text,
                ground_truth_text,
            ) = self.forward_and_compute_loss(
                ctx.decode_model,
                stage_ctx.input_ids,
                stage_ctx.inputs_embeds,
                stage_ctx.attention_mask,
                united_token_embeddings,
                united_attention_mask,
                ctx.num_compression_tokens,
                target_hidden_states=stage_ctx.target_hidden_states,
                prefix_len=batch_ctx.prefix_len,
            )
            last_loss = float(loss.item())
            last_convergence = convergence_per_sample.detach().cpu()

            # Plan B (measure-then-decide): decide convergence on the CURRENT (live,
            # just-forwarded) embedding *before* stepping. ``state.update`` marks any
            # newly-converged sample inactive, so ``_step_per_sample_optimizers`` below
            # skips it and its live embedding stays exactly the one we just measured.
            # This keeps the embedding saved by ``_collect_stage_rows`` consistent with
            # the recorded ``final_convergence``/``final_loss`` -- previously the optimizer
            # took one extra step after the converged forward, so the saved embedding was
            # one optimizer step past the metrics recorded for it.
            if state.update(convergence_per_sample):
                return last_loss, last_convergence, True

            loss.backward()

            self._log_progress(
                progress_bar=progress_bar,
                loss=loss,
                alignment_loss=alignment_loss,
                convergence_per_sample=convergence_per_sample,
                batch_ctx=batch_ctx,
                state=state,
                compression_token_embeddings=compression_token_embeddings,
                generated_text=generated_text,
                ground_truth_text=ground_truth_text,
            )

            self._step_per_sample_optimizers(batch_ctx, state)
            self._step_shared_optimizer(batch_ctx)

            # Hard cap on cumulative per-sample steps across stages within this batch.
            # An exhausted sample is permanently skipped; if that drains the batch,
            # exit the stage as "converged" so the outer retry path doesn't fire
            # (a budget-exhausted sample can't make progress on retry). The optimizer
            # just stepped, so re-measure the live embedding to keep the returned metrics
            # consistent with the embedding ``_collect_stage_rows`` will save.
            if state.mark_exhausted(self.args.max_optimization_steps_per_sample):
                last_loss, last_convergence = self._measure_live(batch_ctx, stage_ctx, ctx)
                return last_loss, last_convergence, True

        # Max steps for this stage reached without all samples converging. The optimizer
        # stepped on the final iteration, so re-measure the live (post-step) embedding to
        # keep the returned metrics consistent with what ``_collect_stage_rows`` saves.
        last_loss, last_convergence = self._measure_live(batch_ctx, stage_ctx, ctx)
        return last_loss, last_convergence, False

    @torch.no_grad()
    def _measure_live(self, batch_ctx: "_BatchContext", stage_ctx: "_StageContext", ctx: "_RunContext"):
        """Forward the current live embedding; return ``(loss, convergence_per_sample_cpu)``.

        Used only on non-converged stage exits (per-sample budget exhausted, or max steps
        reached) where the optimizer has stepped past the last measured embedding. Re-measuring
        the live embedding keeps the recorded ``final_loss``/``final_convergence`` consistent
        with the post-step embedding that ``_collect_stage_rows`` materializes. Converged exits
        return before stepping, so they need no re-measure.
        """
        compression_token_embeddings = batch_ctx.parametrization.materialize().clone()
        united_token_embeddings, united_attention_mask = build_united_input(
            compression_token_embeddings,
            batch_ctx.compression_attention_mask,
            stage_ctx.inputs_embeds,
            stage_ctx.attention_mask,
            prefix_token_embeddings=batch_ctx.prefix_token_embeddings,
            prefix_attention_mask=batch_ctx.prefix_attention_mask,
        )
        loss, _alignment_loss, convergence_per_sample, _generated_text, _ground_truth_text = self.forward_and_compute_loss(
            ctx.decode_model,
            stage_ctx.input_ids,
            stage_ctx.inputs_embeds,
            stage_ctx.attention_mask,
            united_token_embeddings,
            united_attention_mask,
            ctx.num_compression_tokens,
            target_hidden_states=stage_ctx.target_hidden_states,
            prefix_len=batch_ctx.prefix_len,
        )
        return float(loss.item()), convergence_per_sample.detach().cpu()

    def _step_per_sample_optimizers(self, batch_ctx: _BatchContext, state: ProgressiveSampleStateMachine) -> None:
        """Step active samples' optimizers; zero grads for all (active + skipped + already-converged-in-stage)."""
        for j in range(batch_ctx.batch_size):
            if state.is_active(j):
                batch_ctx.per_sample_optimizers[j].step()
                if batch_ctx.per_sample_schedulers[j] is not None:
                    batch_ctx.per_sample_schedulers[j].step()
                state.increment_steps(j)
            batch_ctx.per_sample_optimizers[j].zero_grad(set_to_none=True)

    def _step_shared_optimizer(self, batch_ctx: _BatchContext) -> None:
        """Step the optimizer for parametrization-shared parameters (e.g. low-dim Linear)."""
        if batch_ctx.shared_optimizer is None:
            return
        batch_ctx.shared_optimizer.step()
        batch_ctx.shared_optimizer.zero_grad(set_to_none=True)
        if batch_ctx.shared_scheduler is not None:
            batch_ctx.shared_scheduler.step()

    def _log_progress(
        self,
        *,
        progress_bar,
        loss: torch.Tensor,
        alignment_loss: torch.Tensor | None,
        convergence_per_sample: torch.Tensor,
        batch_ctx: _BatchContext,
        state: ProgressiveSampleStateMachine,
        compression_token_embeddings: torch.Tensor,
        generated_text,
        ground_truth_text,
    ) -> None:
        """Update tqdm postfix and forward scalars to TensorBoard."""
        grad_norms = [
            param.grad.norm(2).item() if param.grad is not None else 0.0 for param in batch_ctx.parametrization.parameters
        ]
        grad_norm = sum(grad_norms) / len(grad_norms)

        active_scheduler = None
        for j in range(batch_ctx.batch_size):
            if state.is_active(j):
                active_scheduler = batch_ctx.per_sample_schedulers[j]
                break
        log_lr = active_scheduler.get_last_lr()[0] if active_scheduler is not None else self.args.learning_rate

        progress_bar.update(1)
        progress_bar.set_postfix(
            loss=loss.item(),
            convergece_per_sample=convergence_per_sample.mean().item(),
            compr_t_mean=compression_token_embeddings.mean().item(),
            compr_t_std=compression_token_embeddings.std().item(),
            grad=grad_norm,
            lr=log_lr,
        )

        self._log_step(
            loss,
            alignment_loss,
            convergence_per_sample,
            compression_token_embeddings,
            active_scheduler,
            generated_text,
            ground_truth_text,
            leaf_grad_params=list(batch_ctx.parametrization.parameters) + list(batch_ctx.parametrization.shared_parameters),
        )

    # ------------------------------------------------------------------
    # Stage row collection.
    # ------------------------------------------------------------------

    def _collect_stage_rows(
        self,
        batch_ctx: _BatchContext,
        stage_ctx: _StageContext,
        ctx: _RunContext,
        state: ProgressiveSampleStateMachine,
        last_loss,
        last_convergence,
        sample_id_counter: int,
    ) -> list[dict]:
        """Compute Information Gain for the stage and assemble per-sample row dicts."""
        with torch.no_grad():
            # `materialize()` already lifts coefficients to hidden_size when a
            # projection is in use, so there is no separate "after projection"
            # tensor to keep track of in this trainer anymore.
            compression_token_embeddings = batch_ctx.parametrization.materialize()  # [batch, compression, hidden]
            comp_tokens_cpu = compression_token_embeddings.detach().cpu()
            # Per-sample parametrization extras: PCA / low-dim coefficients
            # (None for vanilla Direct parametrization).
            pca_coefficients_to_save = batch_ctx.parametrization.serialize_extras()

            per_sample_info_gain = compute_information_gain(
                model=ctx.decode_model,
                input_ids=stage_ctx.input_ids,
                attention_mask=stage_ctx.attention_mask,
                token_embeddings=stage_ctx.inputs_embeds,
                compression_token_embeddings=compression_token_embeddings,
                compression_attention_mask=batch_ctx.compression_attention_mask,
                prefix_token_embeddings=batch_ctx.prefix_token_embeddings,
                prefix_attention_mask=batch_ctx.prefix_attention_mask,
            )

            embeddings_dir = self._prepare_embeddings_dir()
            tokenizer = self.processing_class
            rows = []
            for j in range(batch_ctx.batch_size):
                rows.append(
                    self._build_sample_row(
                        sample_index=j,
                        sample_id=sample_id_counter + j,
                        batch_ctx=batch_ctx,
                        stage_ctx=stage_ctx,
                        ctx=ctx,
                        state=state,
                        tokenizer=tokenizer,
                        comp_tokens_cpu=comp_tokens_cpu,
                        comp_tokens_gpu=compression_token_embeddings,
                        pca_coefficients_to_save=pca_coefficients_to_save,
                        last_loss=last_loss,
                        last_convergence=last_convergence,
                        per_sample_info_gain=per_sample_info_gain,
                        embeddings_dir=embeddings_dir,
                    )
                )
            return rows

    def _prepare_embeddings_dir(self) -> str | None:
        """Create artifacts/.../embeddings/ subdir used for periodic stage snapshots."""
        if not self.args.output_dir:
            return None
        embeddings_dir = os.path.join(self.args.output_dir, "embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)
        return embeddings_dir

    def _build_sample_row(
        self,
        *,
        sample_index: int,
        sample_id: int,
        batch_ctx: _BatchContext,
        stage_ctx: _StageContext,
        ctx: _RunContext,
        state: ProgressiveSampleStateMachine,
        tokenizer,
        comp_tokens_cpu: torch.Tensor,
        comp_tokens_gpu: torch.Tensor,
        pca_coefficients_to_save,
        last_loss,
        last_convergence,
        per_sample_info_gain: list[float],
        embeddings_dir: str | None,
    ) -> dict:
        """One sample's row dict for this stage. Schema preserved for downstream eval scripts.

        Note: ``orig_embedding`` used to refer to the pre-projection low-dim
        coefficient tensor in the legacy implementation. Now the parametrization
        encapsulates the projection inside ``materialize()`` and ``serialize_extras()``
        already returns the low-dim coefficients via ``pca_coefficients_to_save``.
        We keep ``orig_embedding`` in the schema (set equal to ``embedding``) so
        downstream eval scripts that ``remove_columns([..., "orig_embedding"])``
        continue to work without modification.
        """
        sample_attention_mask = stage_ctx.attention_mask[sample_index].bool()
        sample_input_ids = stage_ctx.input_ids[sample_index][sample_attention_mask]
        sample_text = tokenizer.decode(sample_input_ids.tolist(), skip_special_tokens=True) if tokenizer is not None else ""

        if embeddings_dir is not None and stage_ctx.stage_index % 500 == 0:
            self._dump_stage_embedding(
                embeddings_dir=embeddings_dir,
                sample_id=sample_id,
                stage_index=stage_ctx.stage_index,
                comp_tokens=comp_tokens_gpu[sample_index],
                initialization_embedding=batch_ctx.initialization_embeddings[sample_index],
                shared_state=batch_ctx.parametrization.shared_state_dict(),
            )

        sample_pca_coefficients = pca_coefficients_to_save[sample_index] if pca_coefficients_to_save is not None else None
        embedding_as_list = comp_tokens_cpu[sample_index].to(torch.float32).numpy().tolist()
        return {
            "sample_id": int(sample_id),
            "stage_index": int(stage_ctx.stage_index),
            "stage_seq_len": int(stage_ctx.seq_len),
            "text": sample_text,
            "embedding": embedding_as_list,
            # Back-compat with legacy schema: kept identical to ``embedding`` —
            # the low-dim coefficients (the only meaningful "pre-projection"
            # tensor) now live under ``pca_coefficients_to_save``.
            "orig_embedding": embedding_as_list,
            "pca_coefficients_to_save": sample_pca_coefficients,
            "initialization_embedding": batch_ctx.initialization_embeddings[sample_index].to(torch.float32).numpy().tolist(),
            "final_loss": float(last_loss) if last_loss is not None else None,
            "final_convergence": float(last_convergence[sample_index].item()) if last_convergence is not None else None,
            "num_input_tokens": int(sample_attention_mask.sum().item()),
            "num_compression_tokens": int(ctx.num_compression_tokens),
            "hidden_size": int(comp_tokens_cpu.shape[-1]),
            "loss_type": self.args.loss_type,
            "dtype": self.args.dtype,
            "model_checkpoint": self.args.model_checkpoint,
            "max_optimization_steps_per_sample": int(self.args.max_optimization_steps_per_sample),
            "convergence_threshold": float(ctx.threshold),
            "steps_taken": int(state.steps_taken[sample_index]),
            "information_gain_bits": float(per_sample_info_gain[sample_index]),
            "progressive_prefix_len": int(batch_ctx.prefix_len),
            "prefix_surprisal_bits_per_token": (
                float(batch_ctx.prefix_surprisal_bits_per_token[sample_index])
                if batch_ctx.prefix_surprisal_bits_per_token is not None
                else None
            ),
        }

    def _dump_stage_embedding(
        self,
        *,
        embeddings_dir: str,
        sample_id: int,
        stage_index: int,
        comp_tokens: torch.Tensor,
        initialization_embedding: torch.Tensor,
        shared_state: dict | None,
    ) -> None:
        """Persist bf16 snapshots of compression / init embeddings (+ shared low-dim projection state) for the current stage."""
        embedding_path = os.path.join(embeddings_dir, f"embedding_sample_{sample_id}_stage_{stage_index}.pt")
        init_embedding_path = os.path.join(
            embeddings_dir, f"initialization_embedding_sample_{sample_id}_stage_{stage_index}.pt"
        )

        torch.save(comp_tokens.to(torch.bfloat16).detach().cpu(), embedding_path)
        torch.save(initialization_embedding.to(torch.bfloat16), init_embedding_path)
        if shared_state is not None:
            low_dim_proj_path = os.path.join(embeddings_dir, f"low_dim_proj_sample_{sample_id}_stage_{stage_index}.pt")
            torch.save(shared_state, low_dim_proj_path)
