from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from transformers import SchedulerType, TrainingArguments


def _parse_cli_dict(value: Any) -> dict[str, Any] | None:
    """Parse CLI dict-like values: JSON string, ``key=value`` pairs, or ``None``."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise TypeError(f"Expected dict or str for dict-like CLI arg, got {type(value)}")

    s = value.strip()
    if s == "" or s.lower() in {"none", "null"}:
        return None

    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        parsed = None

    if parsed is not None:
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected a JSON object for dict-like CLI arg, got {type(parsed)}")
        return parsed

    result: dict[str, Any] = {}
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Invalid dict-like CLI arg chunk (expected key=value): {chunk!r}")
        key, raw_val = chunk.split("=", 1)
        key = key.strip()
        raw_val = raw_val.strip()
        if key == "":
            raise ValueError(f"Invalid dict-like CLI arg chunk (empty key): {chunk!r}")
        try:
            val = json.loads(raw_val)
        except json.JSONDecodeError:
            val = raw_val
        result[key] = val
    return result


@dataclass
class MyTrainingArguments(TrainingArguments):
    """Training arguments for tokens compression experiments."""

    # --- Model & data ----------------------------------------------------
    model_checkpoint: str = field(
        default="HuggingFaceTB/SmolLM2-135M",
        metadata={"help": "HuggingFace location for a model and a tokenizer."},
    )
    dataset_name: str = field(
        default="mrsndmn/pg19",
        metadata={"help": "Dataset name to use for training (e.g., 'mrsndmn/pg19')."},
    )
    max_sequence_length: int = field(
        default=128,
        metadata={"help": "Max sequence length for compressing in training."},
    )
    limit_dataset_items: int | None = field(default=1)
    offset_dataset_items: int | None = field(
        default=None,
        metadata={"help": "Offset for dataset items selection (applied before limit_dataset_items)."},
    )
    no_bos_token: bool = field(
        default=False,
        metadata={"help": "Disable BOS token insertion during dataset tokenization."},
    )

    # --- Compression embedding shape & initialization --------------------
    number_of_mem_tokens: int = field(
        default=1,
        metadata={"help": "Number of trainable [mem] tokens for a single sample."},
    )
    embedding_init_method: str = field(
        default="random",
        metadata={
            "help": 'Initialization method for compression embeddings: "random", "mvnormal", "pretrained_pca", "load_from_disk".'
        },
    )
    embedding_init_path: str = field(
        default="",
        metadata={
            "help": (
                "Path to file containing initial compression embeddings (when embedding_init_method=load_from_disk). "
                "If empty, embeddings will be generated using load_from_disk_embedding_init_method and saved. "
                "File should contain a tensor of shape [num_tokens, hidden_size] or [1, num_tokens, hidden_size] "
                "or [batch_size, num_tokens, hidden_size]."
            )
        },
    )
    load_from_disk_embedding_init_method: str = field(
        default="random",
        metadata={
            "help": (
                "Initialization method to use when generating embeddings for load_from_disk "
                "(when embedding_init_path is empty)."
            )
        },
    )
    pretrained_pca_num_components: int = field(
        default=16,
        metadata={"help": "Number of PCA components to use when embedding_init_method=pretrained_pca."},
    )
    pretrained_pca_path: str = field(
        default="",
        metadata={
            "help": "Path to progressive_prefixes dataset for PCA initialization (when embedding_init_method=pretrained_pca)."
        },
    )
    fix_position_ids: bool = field(
        default=False,
        metadata={"help": "Whether position_ids should be adjusted relative to compression embeddings."},
    )

    # --- Loss / activation alignment -----------------
    loss_type: str = field(
        default="l2",
        metadata={"help": "Loss type for activation alignment: l2, l1, or cosine."},
    )
    hybrid_alpha: float | None = field(
        default=None,
        metadata={"help": "Multiplier for the alignment loss; hybrid loss is enabled when specified."},
    )
    num_alignment_layers: int = field(default=0, metadata={"help": "Number of transformer layers to align (0 = all)."})
    inverted_alignment: bool = field(
        default=False,
        metadata={"help": "Direction of taking layers: True = depth-to-shallow, False = shallow-to-depth."},
    )
    leading_token_loss_weight: float = field(
        default=1.0,
        metadata={
            "help": (
                "Loss weight applied to the first leading_token_loss_count positions of the target "
                "sequence in the next-token CE loss. Use 3.0 with count=2 to reproduce the '2 leading' "
                "Table 18 setup. Default 1.0 = uniform weighting."
            )
        },
    )
    leading_token_loss_count: int = field(
        default=0,
        metadata={
            "help": (
                "Number of leading target-sequence positions to upweight in the CE loss "
                "(see leading_token_loss_weight). 0 disables upweighting."
            )
        },
    )

    # --- Low-dimensional projection ------------------
    low_dim_train: bool = field(default=False, metadata={"help": "Run the LowDim full-cramming trainer."})
    low_dim_projection: bool = field(
        default=False,
        metadata={"help": "Enable low-dim projection reparameterization e = W z + b."},
    )
    low_dim_projection_global: bool = field(
        default=False,
        metadata={"help": "Share one projection across the dataset (vs fresh per batch)."},
    )
    low_dim_size: int = field(
        default=32,
        metadata={"help": "Dimension k of the low-dim subspace."},
    )
    low_dim_projection_checkpoint: str | None = field(
        default=None,
        metadata={"help": "Path to checkpoint file to load low-dimensional projection state from."},
    )
    low_dim_projection_train: bool = field(
        default=True,
        metadata={"help": "Whether to optimize the low-dim projection (False to freeze it)."},
    )

    # --- Progressive training ------------------------
    progressive_train: bool = field(default=False, metadata={"help": "Run the progressive cramming trainer."})
    progressive_min_seq_len: int = field(
        default=1,
        metadata={"help": "Starting effective sequence length for progressive_train."},
    )
    progressive_step: int = field(
        default=1,
        metadata={"help": "Step size to increase effective sequence length between stages."},
    )
    progressive_convergence_threshold: float = field(
        default=1.0,
        metadata={"help": "Mean token-level match ratio required to mark a stage as converged."},
    )
    progressive_max_stages: int = field(
        default=0,
        metadata={"help": "Optional cap on number of progressive stages (0 = no cap)."},
    )
    progressive_prefix_len: int = field(
        default=0,
        metadata={
            "help": "Number of leading real (uncompressed) prefix tokens the model attends to before "
            "the crammed continuation in progressive training. The prefix is fed as real token "
            "embeddings (visible context) but is never compressed into the [mem] token; loss, "
            "convergence, and information gain are computed only over the post-prefix continuation. "
            "0 = no prefix (default, identical to the original trainer)."
        },
    )
    progressive_reset_lr_scheduler_on_non_convergence: bool = field(
        default=False,
        metadata={"help": "Reset LR scheduler and continue training when convergence fails (once per stage)."},
    )
    progressive_geometric_growth: bool = field(
        default=False,
        metadata={
            "help": (
                "Progressive_train only. Instead of growing the prefix a fixed progressive_step "
                "per converged stage, double the prefix length each time a stage converges (a "
                "geometrically growing number of added tokens), then bisect the gap once a stage "
                "fails to pin the exact largest converged prefix. Every stage is warm-started from "
                "the previous converged embedding, so the reported prefix is always fully "
                "reconstructed; the horizon is found in ~log2 stages instead of one per token. "
                "Requires per_device_train_batch_size=1."
            )
        },
    )
    progressive_geometric_backoff: str = field(
        default="bisect",
        metadata={
            "help": (
                "progressive_geometric_growth only. Back-off strategy once the doubling ramp "
                "brackets the horizon between the largest converged length (lo) and the smallest "
                "failed length (hi). 'bisect': restore the last converged state and bisect the "
                "(lo, hi) gap, O(log gap) probes. 'linear': restore the last converged checkpoint "
                "and grow the prefix +1 token per stage (each warm-started from the previous "
                "converged stage) until a stage fails -- the exact horizon, mirroring Delta=1 "
                "cramming in the horizon neighborhood at the cost of (horizon - lo) probes. "
                "Choices: bisect, linear."
            )
        },
    )
    progressive_init_from_artifact: str | None = field(
        default=None,
        metadata={
            "help": "Path to an experiment artifact directory (or its progressive_prefixes/ subdir) "
            "to load initial embeddings from. Requires --progressive_init_from_stage."
        },
    )
    progressive_init_from_stage: int | None = field(
        default=None,
        metadata={
            "help": "Stage index in the artifact to load the converged embedding from. "
            "The new experiment resumes progressive growth from the artifact's stage_seq_len + 1."
        },
    )
    progressive_init_sample_ids: str | None = field(
        default=None,
        metadata={
            "help": "Comma-separated source sample IDs to load from the artifact (default: '0'). "
            "The i-th listed ID initializes the i-th sample in the new experiment's dataloader order."
        },
    )
    save_progressive_artifacts: bool = field(
        default=True,
        metadata={"help": "Whether to persist intermediate compression tokens for each stage."},
    )
    max_optimization_steps_per_sample: int = field(
        default=1_000,
        metadata={"help": "Max optimization steps for training a single sample."},
    )
    full_cramming_convergence_threshold: float = field(
        default=1.0,
        metadata={
            "help": (
                "Teacher-forcing accuracy threshold for early stopping in FullCrammingTrainer. "
                "Default 1.0 = stop only at exact reconstruction; set to 0.99 to reproduce the "
                "reference paper's 99%% protocol (Table 18 ablation)."
            )
        },
    )
    max_optimization_steps_per_token: int = field(
        default=1_000,
        metadata={"help": "Max optimization steps for training one newly-added token (progressive only)."},
    )

    # --- Compression head training (single-pass compression-embed prediction) ---
    train_compression_head: bool = field(
        default=False,
        metadata={"help": "Train a compression head (no per-sample embedding optimization)."},
    )
    compression_head_distill_alpha: float = field(
        default=1.0,
        metadata={"help": "Weight for the compressed-sequence (after) loss term in the compression-head objective."},
    )
    compression_head_distill_beta: float = field(
        default=0.0,
        metadata={
            "help": "Weight for the base-LM loss term (uncompressed full-sequence loss). "
            "Default 0.0 keeps the base loss out of the objective so training optimizes the "
            "compression objective only; set > 0 to also train on the plain next-token loss."
        },
    )
    compression_head_freeze_base_model: bool = field(
        default=True,
        metadata={"help": "Freeze base LM parameters and train only compression head parameters."},
    )
    separate_reconstructor_model: bool = field(
        default=False,
        metadata={
            "help": (
                "Compression-head training: when True, build a second copy of the base LM (no CH adapter) "
                "from the same checkpoint and route the second (reconstruction) forward through it while the "
                "first (compression) forward stays on the CH model. Both copies are optimised jointly. "
                "Has no effect outside the compression-head trainer."
            )
        },
    )
    compression_head_kind: str = field(
        default="mlp",
        metadata={
            "help": "Compression-head architecture: 'mlp' (Linear-GELU-Linear on the last prefix hidden state) "
            "or 'qformer' (learnable queries cross-attending to the prefix hidden states)."
        },
    )
    compression_head_num_queries: int = field(
        default=1,
        metadata={"help": "Q-Former: number of learnable query tokens (= number of compression embeddings produced)."},
    )
    compression_head_num_layers: int = field(
        default=2,
        metadata={"help": "Q-Former: number of cross-attention blocks."},
    )
    compression_head_num_heads: int = field(
        default=8,
        metadata={"help": "Q-Former: number of attention heads per block."},
    )
    compression_head_query_proj_factor: int = field(
        default=1,
        metadata={
            "help": "Q-Former wide-init: keep the learnable query in dimension factor*H and "
            "project down to H (more optimizer degrees of freedom). 1 disables it."
        },
    )

    # --- Other trainer modes --------------------------------------------
    train_prefix_tuning: bool = field(default=False, metadata={"help": "Run the PEFT prefix tuning trainer."})
    noop_train: bool = field(default=False, metadata={"help": "Run the no-op trainer."})
    noop_convergence_threshold: float = field(
        default=1.0,
        metadata={"help": "Mean token-level match ratio required to mark a stage as converged (no-op trainer)."},
    )
    max_tokens_in_distribution: int = field(
        default=1,
        metadata={"help": "Number of top tokens to keep in the distribution target (no-op trainer)."},
    )

    # --- Diagnostics ----------------------------------------------------
    generate_in_compute_loss: bool = field(default=False)
    random_seed: int | None = field(default=42, metadata={"help": "Random seed for reproducibility (None to skip)."})

    # --- Precision ------------------------------------------------------
    dtype: str = field(
        default="bf16",
        metadata={"help": "Torch dtype: auto, float32|fp32, bfloat16|bf16, float16|fp16."},
    )
    attn_implementation: str = field(
        default="sdpa",
        metadata={
            "help": (
                "Attention backend passed to from_pretrained: 'sdpa' (default, portable, built into "
                "PyTorch), 'eager', or 'flash_attention_2' (requires the flash-attn package and a "
                "compatible GPU). The research runs used flash_attention_2; 'sdpa' reproduces the same "
                "results without the extra dependency."
            )
        },
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Enable gradient checkpointing (trades memory for extra compute)."},
    )

    # --- TrainingArguments overrides ------------------------------------
    optim: str = field(default="adamw_torch")
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device accelerator core/CPU for training."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    dataloader_drop_last: bool = field(
        default=True,
        metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."},
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={"help": "Number of dataloader subprocesses; 0 = main process."},
    )
    learning_rate: float = field(default=0.01, metadata={"help": "The initial learning rate for an optimizer."})
    adam_beta1: float = 0.9
    adam_beta2: float = 0.9
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay for the optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    lr_scheduler_type: SchedulerType | str = field(
        default="cosine_with_min_lr",
        metadata={"help": "The scheduler type to use."},
    )
    # NOTE: Kept as `str` for CLI parsing; converted to `dict` in __post_init__.
    lr_scheduler_kwargs: str = field(
        default_factory=lambda: {"min_lr": 1e-3},
        metadata={
            "help": (
                "Additional keyword arguments to pass to the learning rate scheduler. "
                "Pass as JSON, e.g. --lr_scheduler_kwargs '{\"min_lr\": 0.0001}', "
                "or as key=value pairs, e.g. --lr_scheduler_kwargs 'min_lr=0.0001'."
            )
        },
    )
    ddp_find_unused_parameters: bool | None = field(
        default=False,
        metadata={"help": "Value of `find_unused_parameters` for `DistributedDataParallel`."},
    )
    load_best_model_at_end: bool = field(
        default=False,
        metadata={"help": "Whether to load the best model found during training at the end of training."},
    )

    def __post_init__(self):
        # Convert CLI-friendly forms into a real dict early, before base TrainingArguments validation.
        self.lr_scheduler_kwargs = _parse_cli_dict(getattr(self, "lr_scheduler_kwargs", None))  # type: ignore[assignment]
        super().__post_init__()
