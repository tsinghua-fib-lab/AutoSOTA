"""Helpers for the public demo notebook (Colab) and the gallery-build script.

This module wraps the package's training/inference primitives into three small,
self-contained functions so the demo notebook and ``scripts/build_demo_gallery.py``
share exactly one cramming implementation:

* :func:`load_frozen_model`  -- load a frozen causal LM + tokenizer (T4-friendly).
* :func:`cram_text`          -- *full* cramming of one arbitrary text into a single
  (or a few) learnable memory embedding(s), with an optional per-step callback and
  embedding-trajectory capture for the live PCA visualization.
* :func:`reconstruct_text`   -- greedily decode the original text back from a
  compression embedding.

Everything here is plain full cramming (fixed-length span, optimise until the
frozen model reconstructs it). Progressive / low-dim variants live in
``progressive_cramming.train`` and ``run_training``; the demo uses those directly
where it needs them (side-by-side TC vs PC), and uses :func:`cram_text` for the
interactive "compress your own text" path where per-step control matters.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from progressive_cramming.analysis.information_gain import compute_information_gain
from progressive_cramming.inference.generation import generate_from_compression
from progressive_cramming.train.embedding_init import create_compression_embedding
from progressive_cramming.train.inputs import build_compression_attention_mask, build_united_input
from progressive_cramming.train.loss import (
    next_token_cross_entropy_loss_with_prefix,
    token_argmax_match_rate_with_prefix,
)

# The demo's default model: the ungated Llama-3.2-1B mirror (no HF license gate),
# small enough to cram a 64-token span on a Colab T4 free-tier GPU.
DEFAULT_MODEL_CHECKPOINT = "unsloth/Llama-3.2-1B"

# Paper recipe for a small model (mirrors examples/quickstart.py): a scaled-down
# uniform init plus a high LR with a short warmup + cosine decay converge fast.
DEFAULT_INIT_METHOD = "random0.02"
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_WARMUP_STEPS = 100
DEFAULT_MIN_LR = 1e-3


_DTYPE_ALIASES = {
    "auto": "auto",
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def resolve_dtype(dtype: str | torch.dtype):
    """Map a dtype name (``"bf16"``, ``"float32"``, ...) to a ``torch.dtype`` (or ``"auto"``)."""
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(f"Unsupported dtype {dtype!r}; expected one of {sorted(_DTYPE_ALIASES)}.")
    return _DTYPE_ALIASES[key]


def pick_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve the compute device: explicit arg, else CUDA if available, else CPU."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_frozen_model(
    checkpoint: str = DEFAULT_MODEL_CHECKPOINT,
    *,
    dtype: str | torch.dtype = "float16",
    device: str | torch.device | None = None,
    attn_implementation: str = "eager",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a causal LM + tokenizer with all weights frozen (cramming optimises only the embedding).

    ``attn_implementation="eager"`` is the portable default (no flash-attn dependency).
    Default ``dtype="float16"`` runs natively on a Colab T4 (Turing has no hardware
    bf16); the compression embedding itself is always kept in float32 for stable
    optimisation regardless of the model dtype. Switch to ``"float32"`` if you hit
    instability, or ``"bfloat16"`` on an Ampere+ GPU.
    """
    device = pick_device(device)
    torch_dtype = resolve_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


@dataclass
class CrammingResult:
    """Everything the demo needs after cramming one text into a memory embedding.

    ``embedding`` is the converged compression embedding ``[num_mem_tokens, hidden]``
    (float32, CPU) -- this is what gets saved/uploaded and later fed to
    :func:`reconstruct_text`. ``trajectory`` (if captured) holds the flattened
    embedding ``[num_snapshots, num_mem_tokens * hidden]`` at the steps in
    ``trajectory_steps``, for the live PCA visualization.
    """

    text: str
    input_ids: list[int]
    num_tokens: int  # n_cram: how many tokens were crammed into the embedding
    num_mem_tokens: int
    hidden_size: int
    embedding: torch.Tensor  # [num_mem_tokens, hidden] float32 cpu
    converged: bool
    final_convergence: float  # teacher-forced token match rate in [0, 1]; 1.0 = exact
    final_loss: float
    steps_taken: int
    information_gain_bits: float
    elapsed_s: float
    model_checkpoint: str
    history: dict = field(default_factory=dict)  # {"step": [...], "loss": [...], "convergence": [...]}
    trajectory: torch.Tensor | None = None  # [num_snapshots, num_mem_tokens*hidden] float32 cpu
    trajectory_steps: list[int] = field(default_factory=list)

    def training_config(self) -> dict:
        """A small JSON-serializable record of how this embedding was produced."""
        return {
            "model_checkpoint": self.model_checkpoint,
            "num_mem_tokens": self.num_mem_tokens,
            "num_tokens": self.num_tokens,
            "hidden_size": self.hidden_size,
            "final_convergence": self.final_convergence,
            "steps_taken": self.steps_taken,
            "information_gain_bits": self.information_gain_bits,
        }


def cram_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    *,
    num_mem_tokens: int = 1,
    max_seq_len: int = 64,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_steps: int = 2000,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    min_lr: float = DEFAULT_MIN_LR,
    weight_decay: float = 0.01,
    init_method: str = DEFAULT_INIT_METHOD,
    convergence_threshold: float = 1.0,
    seed: int | None = 42,
    capture_every: int = 0,
    on_step: Callable[[dict], None] | None = None,
    add_special_tokens: bool = True,
) -> CrammingResult:
    """Full-cram one ``text`` into ``num_mem_tokens`` learnable memory embedding(s).

    Optimises a fresh compression embedding (the frozen model is never updated) until
    the model reconstructs every token teacher-forced (match rate >= ``convergence_threshold``)
    or ``max_steps`` is reached. The text is truncated to ``max_seq_len`` tokens.

    Parameters that drive the live visualization:
      * ``capture_every`` -- if > 0, snapshot the (flattened) embedding every N steps
        into ``result.trajectory`` for a PCA plot of the optimisation path. Step 0 and
        the final step are always captured.
      * ``on_step`` -- called every step with a dict
        ``{"step", "loss", "convergence", "lr", "captured", "embedding"}``; the
        notebook uses it to redraw loss/convergence curves and the PCA path live.
        ``embedding`` is the live ``[num_mem, hidden]`` tensor (detached, on device).

    Returns a :class:`CrammingResult`.
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = next(model.parameters()).device
    model.eval()

    # ---- Tokenize the target span (the continuation the model must reconstruct). ----
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_len,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)  # [1, L]
    attention_mask = enc["attention_mask"].to(device)  # [1, L]
    num_tokens = int(attention_mask.sum().item())
    hidden_size = model.config.hidden_size

    with torch.no_grad():
        token_embeddings = model.get_input_embeddings()(input_ids)  # [1, L, hidden] (model dtype)

    # ---- The single trainable parameter: the compression embedding (kept in float32). ----
    # ``.to(device)`` on a requires_grad-tensor returns a non-leaf in PyTorch 2.12+,
    # which AdamW rejects ("can't optimize a non-leaf Tensor"); detach() + re-enable
    # grad rebuilds a fresh leaf at the right device/dtype.
    compression = (
        create_compression_embedding(
            batch_size=1,
            num_compression_tokens=num_mem_tokens,
            hidden_size=hidden_size,
            init_method=init_method,
        )
        .to(device)
        .detach()
        .requires_grad_(True)
    )  # [1, num_mem, hidden], float32, leaf
    compression_attention_mask = build_compression_attention_mask(
        1, num_mem_tokens, dtype=attention_mask.dtype, device=device
    )

    optimizer = torch.optim.AdamW(
        [compression], lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.9)
    )
    from transformers import get_scheduler

    lr_scheduler = get_scheduler(
        name="cosine_with_min_lr",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": min_lr},
    )

    history: dict[str, list] = {"step": [], "loss": [], "convergence": []}
    snapshots: list[torch.Tensor] = []
    trajectory_steps: list[int] = []

    def _snapshot(step: int) -> None:
        snapshots.append(compression.detach().to(torch.float32).reshape(-1).cpu())
        trajectory_steps.append(step)

    converged = False
    final_convergence = 0.0
    final_loss = float("nan")
    steps_taken = max_steps
    start = time.perf_counter()

    for step in range(max_steps):
        # Direct parametrization: the embedding *is* the parameter. build_united_input
        # casts it to the model dtype (e.g. bf16) before the forward; gradients flow
        # back to the float32 leaf parameter.
        united_embeddings, united_attention_mask = build_united_input(
            compression,
            compression_attention_mask,
            token_embeddings,
            attention_mask,
        )
        logits = model(inputs_embeds=united_embeddings, attention_mask=united_attention_mask).logits
        loss = next_token_cross_entropy_loss_with_prefix(
            logits, input_ids, attention_mask, num_mem_tokens
        )
        with torch.no_grad():
            convergence = token_argmax_match_rate_with_prefix(
                logits, input_ids, attention_mask, num_mem_tokens
            )[0].item()

        final_loss = float(loss.item())
        final_convergence = float(convergence)

        captured = capture_every > 0 and (step % capture_every == 0)
        if captured or step == 0:
            _snapshot(step)
            captured = True

        history["step"].append(step)
        history["loss"].append(final_loss)
        history["convergence"].append(final_convergence)

        if on_step is not None:
            on_step(
                {
                    "step": step,
                    "loss": final_loss,
                    "convergence": final_convergence,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "captured": captured,
                    "embedding": compression.detach(),
                }
            )

        # Measure-then-decide: convergence was computed on the *current* embedding,
        # before this step's update -- so the embedding we return matches the metric.
        if convergence >= convergence_threshold:
            converged = True
            steps_taken = step
            break

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step()

    elapsed_s = time.perf_counter() - start

    # Always capture the final embedding for the trajectory's endpoint.
    if not trajectory_steps or trajectory_steps[-1] != steps_taken:
        _snapshot(steps_taken)

    with torch.no_grad():
        information_gain_bits = compute_information_gain(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_embeddings=token_embeddings,
            compression_token_embeddings=compression.detach(),
            compression_attention_mask=compression_attention_mask,
        )[0]

    embedding = compression.detach().to(torch.float32).cpu()[0]  # [num_mem, hidden]
    trajectory = torch.stack(snapshots, dim=0) if capture_every > 0 and snapshots else None

    return CrammingResult(
        text=tokenizer.decode(input_ids[0][attention_mask[0].bool()], skip_special_tokens=True),
        input_ids=input_ids[0].detach().cpu().tolist(),
        num_tokens=num_tokens,
        num_mem_tokens=num_mem_tokens,
        hidden_size=hidden_size,
        embedding=embedding,
        converged=converged,
        final_convergence=final_convergence,
        final_loss=final_loss,
        steps_taken=steps_taken,
        information_gain_bits=float(information_gain_bits),
        elapsed_s=elapsed_s,
        model_checkpoint=getattr(model.config, "_name_or_path", "") or "",
        history=history,
        trajectory=trajectory,
        trajectory_steps=trajectory_steps,
    )


@dataclass
class ProgressiveStage:
    """One progressive-cramming stage: optimise the embedding to reconstruct a prefix of length ``seq_len``."""

    stage_index: int
    seq_len: int
    converged: bool
    final_convergence: float
    steps_taken: int
    information_gain_bits: float
    embedding: torch.Tensor  # [num_mem_tokens, hidden] float32 cpu -- the embedding at this stage


@dataclass
class ProgressiveResult:
    """Result of progressively cramming one text: the per-stage trace and the compression horizon.

    The **compression horizon** is the longest prefix (in tokens) a single embedding
    reconstructs exactly -- ``horizon``. ``embedding`` is the embedding at that horizon
    (the last stage that converged), i.e. the one to feed :func:`reconstruct_text`.
    """

    text: str
    input_ids: list[int]
    num_tokens: int
    num_mem_tokens: int
    hidden_size: int
    horizon: int  # compression horizon: longest exactly-reconstructed prefix, in tokens
    embedding: torch.Tensor  # [num_mem_tokens, hidden] float32 cpu, at the horizon
    total_steps: int
    elapsed_s: float
    model_checkpoint: str
    stages: list[ProgressiveStage] = field(default_factory=list)
    trajectory: torch.Tensor | None = None  # [num_snapshots, num_mem_tokens*hidden] float32 cpu
    trajectory_steps: list[int] = field(default_factory=list)  # global step per snapshot
    trajectory_seq_len: list[int] = field(default_factory=list)  # stage seq_len per snapshot

    def training_config(self) -> dict:
        return {
            "method": "progressive_cramming",
            "model_checkpoint": self.model_checkpoint,
            "num_mem_tokens": self.num_mem_tokens,
            "num_tokens": self.num_tokens,
            "hidden_size": self.hidden_size,
            "horizon": self.horizon,
            "total_steps": self.total_steps,
            "num_stages": len(self.stages),
        }


def progressive_cram_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    *,
    num_mem_tokens: int = 1,
    max_seq_len: int = 64,
    min_seq_len: int = 1,
    step: int = 4,
    max_steps_per_token: int = 300,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    warmup_steps: int = 50,
    min_lr: float = DEFAULT_MIN_LR,
    weight_decay: float = 0.01,
    init_method: str = DEFAULT_INIT_METHOD,
    convergence_threshold: float = 1.0,
    seed: int | None = 42,
    capture_every: int = 0,
    on_step: Callable[[dict], None] | None = None,
) -> ProgressiveResult:
    """Progressively cram ``text``: grow the target prefix stage by stage, advancing only when the
    current prefix reconstructs **exactly**, to find the per-text *compression horizon*.

    This is a compact, transparent rendering of :class:`ProgressiveCrammingTrainer`
    for a single text and a direct (non-projected) embedding: one warm-started
    embedding is carried across stages (see Fig. 1.1), each stage runs up to
    ``max_steps_per_token`` steps. Stops at the first prefix that fails to converge
    (the horizon stalls there) or when the full span is reached.
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = next(model.parameters()).device
    model.eval()

    enc = tokenizer(text, truncation=True, max_length=max_seq_len, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)  # [1, L]
    attention_mask = enc["attention_mask"].to(device)  # [1, L]
    num_tokens = int(attention_mask.sum().item())
    hidden_size = model.config.hidden_size

    with torch.no_grad():
        token_embeddings = model.get_input_embeddings()(input_ids)  # [1, L, hidden]

    # Build the compression embedding on-device as a leaf tensor. ``.to(device)`` on
    # an already-requires_grad tensor produces a non-leaf in PyTorch 2.12+, which
    # AdamW refuses ("can't optimize a non-leaf Tensor"); ``detach().requires_grad_(True)``
    # forces a fresh leaf at the right device/dtype.
    compression = (
        create_compression_embedding(
            batch_size=1, num_compression_tokens=num_mem_tokens,
            hidden_size=hidden_size, init_method=init_method,
        )
        .to(device)
        .detach()
        .requires_grad_(True)
    )
    compression_attention_mask = build_compression_attention_mask(
        1, num_mem_tokens, dtype=attention_mask.dtype, device=device
    )

    # One optimizer/scheduler carried across all stages; the cosine budget spans the
    # worst case of every stage using its full per-token budget.
    n_stages_max = (max(0, num_tokens - min_seq_len) + step - 1) // step + 1
    total_budget = max(max_steps_per_token * n_stages_max, max_steps_per_token)
    optimizer = torch.optim.AdamW([compression], lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.9))
    from transformers import get_scheduler

    lr_scheduler = get_scheduler(
        name="cosine_with_min_lr",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_budget,
        scheduler_specific_kwargs={"min_lr": min_lr},
    )

    stages: list[ProgressiveStage] = []
    snapshots: list[torch.Tensor] = []
    trajectory_steps: list[int] = []
    trajectory_seq_len: list[int] = []
    horizon = 0
    horizon_embedding = compression.detach().to(torch.float32).cpu()[0]
    global_step = 0
    start = time.perf_counter()

    seq_len = min(min_seq_len, num_tokens)
    stage_index = 0
    while True:
        stage_ids = input_ids[:, :seq_len]
        stage_mask = attention_mask[:, :seq_len]
        stage_token_embeddings = token_embeddings[:, :seq_len]

        converged = False
        stage_convergence = 0.0
        steps_this_stage = 0
        for _ in range(max_steps_per_token):
            united_embeddings, united_attention_mask = build_united_input(
                compression, compression_attention_mask, stage_token_embeddings, stage_mask
            )
            logits = model(inputs_embeds=united_embeddings, attention_mask=united_attention_mask).logits
            loss = next_token_cross_entropy_loss_with_prefix(logits, stage_ids, stage_mask, num_mem_tokens)
            with torch.no_grad():
                stage_convergence = token_argmax_match_rate_with_prefix(
                    logits, stage_ids, stage_mask, num_mem_tokens
                )[0].item()

            if capture_every > 0 and (global_step % capture_every == 0):
                snapshots.append(compression.detach().to(torch.float32).reshape(-1).cpu())
                trajectory_steps.append(global_step)
                trajectory_seq_len.append(seq_len)
            if on_step is not None:
                on_step(
                    {
                        "global_step": global_step,
                        "stage_index": stage_index,
                        "seq_len": seq_len,
                        "loss": float(loss.item()),
                        "convergence": float(stage_convergence),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "embedding": compression.detach(),
                        # Full tokenised input so live-visualisation callbacks can
                        # score accuracy on any prefix without re-tokenising.
                        "input_ids": input_ids[0].detach().cpu().tolist(),
                        "num_mem_tokens": num_mem_tokens,
                        "hidden_size": hidden_size,
                    }
                )

            # Measure-then-decide: convergence is on the current embedding, before stepping.
            if stage_convergence >= convergence_threshold:
                converged = True
                break

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            global_step += 1
            steps_this_stage += 1

        with torch.no_grad():
            info_gain = compute_information_gain(
                model=model,
                input_ids=stage_ids,
                attention_mask=stage_mask,
                token_embeddings=stage_token_embeddings,
                compression_token_embeddings=compression.detach(),
                compression_attention_mask=compression_attention_mask,
            )[0]

        stage_embedding = compression.detach().to(torch.float32).cpu()[0]
        stages.append(
            ProgressiveStage(
                stage_index=stage_index,
                seq_len=seq_len,
                converged=converged,
                final_convergence=float(stage_convergence),
                steps_taken=steps_this_stage,
                information_gain_bits=float(info_gain),
                embedding=stage_embedding,
            )
        )

        if converged:
            horizon = seq_len
            horizon_embedding = stage_embedding
            if seq_len >= num_tokens:
                break
            seq_len = min(seq_len + step, num_tokens)
            stage_index += 1
        else:
            # First prefix that fails to reconstruct exactly: the horizon stalls here.
            break

    elapsed_s = time.perf_counter() - start
    trajectory = torch.stack(snapshots, dim=0) if capture_every > 0 and snapshots else None

    return ProgressiveResult(
        text=tokenizer.decode(input_ids[0][attention_mask[0].bool()], skip_special_tokens=True),
        input_ids=input_ids[0].detach().cpu().tolist(),
        num_tokens=num_tokens,
        num_mem_tokens=num_mem_tokens,
        hidden_size=hidden_size,
        horizon=horizon,
        embedding=horizon_embedding,
        total_steps=global_step,
        elapsed_s=elapsed_s,
        model_checkpoint=getattr(model.config, "_name_or_path", "") or "",
        stages=stages,
        trajectory=trajectory,
        trajectory_steps=trajectory_steps,
        trajectory_seq_len=trajectory_seq_len,
    )


@torch.no_grad()
def reconstruct_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    embedding: torch.Tensor,
    *,
    max_new_tokens: int = 64,
) -> str:
    """Greedily decode text from a compression ``embedding`` (``[num_mem, hidden]`` or ``[1, num_mem, hidden]``).

    This is the reconstruction side of cramming: feed only the memory embedding to the
    frozen model and let it generate. For an embedding that reached ``final_convergence == 1.0``
    teacher-forced, greedy decoding reproduces the original span.
    """
    if embedding.dim() == 2:
        embedding = embedding.unsqueeze(0)  # [1, num_mem, hidden]
    device = next(model.parameters()).device
    embedding = embedding.to(device)
    out = generate_from_compression(
        model,
        tokenizer,
        embedding,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
    )
    # generate_from_compression returns a (texts, texts_or_ids) tuple; the first
    # element is always the decoded texts.
    texts = out[0] if isinstance(out, tuple) else out
    return texts[0] if isinstance(texts, (list, tuple)) else texts
