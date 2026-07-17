"""Precompute unquantized model statistics for reuse across quantization runs.

All outputs are rate-independent: the same precomputed data can be reused
with different target rates, quantization methods, or Qronos/residual settings.

Usage:
    # 1. Run precompute pass (one-time, needs full model on GPU)
    python -m quant_layerwise.precompute --model_name 3-8B --output_dir /data/precomputed/3-8B

    # 2. Use in quantization pipeline (needs only 1 model copy on GPU)
    cfg = PipelineConfig(..., precomputed_dir="/data/precomputed/3-8B")
    run_pipeline(cfg)

Saves:
    block_outputs/   — hidden states at each block boundary (model dtype, ~2GB each)
    hessians/        — per-module Σ_X = E[X X^T] (float64, ~134MB each for dim=4096)
    attention_importance/ — per-token attention weights p_j (float64, ~2MB each)
    meta.json        — calibration config, model info, checksums
"""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from quant_layerwise.data import get_calibration_data, split_dataset
from quant_layerwise.hessian_runtime import (
    ActivationCache,
    RuntimeHessian,
    compute_attention_importance,
)
from quant_layerwise.names import get_hess_name
from quant_layerwise.storage import safe_stem


# ---------------------------------------------------------------------------
# Auto-resolve: canonical cache path under $QUANT_BUCKET
# ---------------------------------------------------------------------------

def default_precomputed_dir(
    model_name: str,
    *,
    calib_dataset: str = "redpajama",
    calib_seed: int = 42,
    seqlen: int = 2048,
) -> Path:
    """Return the canonical cache path for precomputed data.

    Layout: $QUANT_BUCKET/precomputed/{model_name}/{dataset}_s{seqlen}_seed{seed}/
    Uses all available calibration samples (no nsamples in path).
    """
    import os

    bucket = os.environ.get("QUANT_BUCKET", None)
    if not bucket:
        raise RuntimeError("QUANT_BUCKET environment variable is not set")
    # Sanitize dataset name for filesystem (mix specs have colons/commas)
    ds_tag = calib_dataset.replace(":", "").replace(",", "_").replace(" ", "")
    tag = f"{ds_tag}_s{seqlen}_seed{calib_seed}"
    return Path(bucket) / "precomputed" / model_name / tag


def compute_token_hash(tokens: torch.Tensor, nsamples: int) -> str:
    """Hash first/last token IDs + count for staleness detection."""
    flat = tokens[:nsamples].reshape(-1)
    return hashlib.sha256(
        f"{int(flat.shape[0])}|{flat[:8].tolist()}|{flat[-8:].tolist()}".encode()
    ).hexdigest()[:16]


def resolve_precomputed(
    model_name: str,
    *,
    calib_dataset: str = "redpajama",
    calib_seed: int = 42,
    seqlen: int = 2048,
    calib_stride: int | None = None,
    nsamples: int | None = None,
    hessian_batch_size: int = 1,
    local_rank: int = 0,
) -> "PrecomputedUnquantData":
    """Find or create precomputed data for *model_name*.

    1. Check canonical cache path under $QUANT_BUCKET.
    2. If found (meta.json exists), load and return.
    3. If not found, run the precompute pass automatically, then return.

    This means precomputation happens exactly once per (model, calib config).
    """
    out_dir = default_precomputed_dir(
        model_name,
        calib_dataset=calib_dataset,
        calib_seed=calib_seed,
        seqlen=seqlen,
    )
    # Append stride/nsamples to cache dir name when they affect the data
    if calib_stride is not None and calib_stride != seqlen:
        out_dir = out_dir.parent / f"{out_dir.name}_stride{calib_stride}"
    if nsamples is not None and "," in calib_dataset:
        # Mix mode: nsamples affects composition, include in cache key
        out_dir = out_dir.parent / f"{out_dir.name}_n{nsamples}"

    if (out_dir / "meta.json").exists():
        print(f"[precomputed] found cached data at {out_dir}", flush=True)
        return PrecomputedUnquantData(out_dir)

    print(f"[precomputed] no cached data for {model_name}, running precompute pass...", flush=True)
    precompute_batch_size = hessian_batch_size
    cfg = PrecomputeConfig(
        model_name=model_name,
        output_dir=str(out_dir),
        calib_dataset=calib_dataset,
        calib_seed=calib_seed,
        seqlen=seqlen,
        calib_stride=calib_stride,
        nsamples=nsamples,
        hessian_batch_size=precompute_batch_size,
    )
    run_precompute(cfg, local_rank=local_rank)
    return PrecomputedUnquantData(out_dir)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PrecomputeConfig:
    """Configuration for the precompute pass."""

    model_name: str
    output_dir: str

    # Calibration settings — must match the quantization run that uses this data.
    calib_dataset: str = "redpajama"
    calib_seed: int = 42
    seqlen: int = 2048
    calib_stride: int | None = None  # Stride for overlapping windows (None = seqlen)
    nsamples: int | None = None  # Max calibration samples (None = use all)
    # Number of calibration sequences to use for hessians / block outputs.
    # 0 = use all available samples from the calibration dataset.
    hessian_nsamples: int = 0
    hessian_batch_size: int = 1

    # Weight types to precompute hessians for.
    weight_types: Sequence[str] = ("wq", "wk", "wv", "wo", "w1", "w3", "w2")

    # Save attention importance (needed for attn_weighted_qkv).
    save_attention_importance: bool = True

    # sum_mode for attention importance (False = normalized by causal count).
    attn_importance_sum_mode: bool = False


# ---------------------------------------------------------------------------
# Precompute pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_precompute(cfg: PrecomputeConfig, *, local_rank: int = 0):
    """Run the precompute pass over the unquantized model.

    Loads the full model on GPU, streams through all transformer blocks once,
    and saves block outputs, per-module Hessians, and attention importance
    to *cfg.output_dir*.

    Multi-GPU safe: all ranks participate in forward passes (required for TP
    collectives) but only rank 0 writes files.  Hessians for RowParallel
    modules (wo, w2) use per_rank=False so the full H is gathered to rank 0.

    The model is freed after this function returns.
    """
    import torch.distributed as _dist_mod
    from quant_layerwise.pipeline import load_model_and_tokenizer

    _dist_ok = _dist_mod is not None and _dist_mod.is_available() and _dist_mod.is_initialized()
    _rank = int(_dist_mod.get_rank()) if _dist_ok else 0
    _world_size = int(_dist_mod.get_world_size()) if _dist_ok else 1
    _is_rank0 = _rank == 0

    try:
        from fairscale.nn.model_parallel.layers import RowParallelLinear as _RowParallelLinear
    except Exception:
        _RowParallelLinear = None

    out_dir = Path(cfg.output_dir)
    if _is_rank0:
        for subdir in ("block_outputs", "hessians", "attention_importance"):
            (out_dir / subdir).mkdir(parents=True, exist_ok=True)
    # Barrier so non-rank-0 sees the directories.
    if _dist_ok and _world_size > 1:
        _dist_mod.barrier()

    # ------------------------------------------------------------------
    # Load model + calibration data
    # ------------------------------------------------------------------
    print(f"[precompute] loading model {cfg.model_name}", flush=True)
    model, tokenizer = load_model_and_tokenizer(cfg.model_name, local_rank=local_rank, max_seq_len=cfg.seqlen)
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    n_layers = len(model.layers)

    print("[precompute] loading calibration data", flush=True)
    calib_tokens = get_calibration_data(
        tokenizer,
        dataset=cfg.calib_dataset,
        nsamples=cfg.nsamples,
        seqlen=cfg.seqlen,
        seed=cfg.calib_seed,
    )
    train_tokens = split_dataset(calib_tokens, cfg.seqlen, stride=cfg.calib_stride)

    # 0 means "use all available samples"
    nsamples = cfg.hessian_nsamples if cfg.hessian_nsamples > 0 else train_tokens.shape[0]

    print("[precompute] creating activation cache", flush=True)
    cache = ActivationCache(
        model=model,
        dataset=train_tokens,
        seqlen=cfg.seqlen,
        nsamples=nsamples,
        device=device,
        batch_size=cfg.hessian_batch_size,
    )
    print(f"[precompute] cache: {cache.nsamples} samples, seqlen={cfg.seqlen}", flush=True)

    # Save initial embeddings (= input to block 0).
    # Hidden states are replicated across TP ranks; rank 0 saves.
    if _is_rank0:
        torch.save(cache._cached_h, out_dir / "block_outputs" / "block_000.pt")

    # ------------------------------------------------------------------
    # Main loop: one pass through all blocks
    # ------------------------------------------------------------------
    _all_modules = dict(model.named_modules())

    for block_id in range(n_layers):
        t0 = time.time()
        layer = model.layers[block_id]

        # --- Attention importance (computed BEFORE hessian hooks to avoid
        #     interference — _compute_attn_probs calls wq/wk directly) ---
        if cfg.save_attention_importance:
            p_j = compute_attention_importance(
                model, cache, block_id,
                batch_size=cfg.hessian_batch_size,
                sum_mode=cfg.attn_importance_sum_mode,
            )
            if _is_rank0:
                torch.save(p_j, out_dir / "attention_importance" / f"layer_{block_id:03d}.pt")

        # --- Hessians: hook ALL weight modules, single forward pass ---
        # For RowParallel modules (wo, w2) in multi-GPU: use per_rank=False
        # so that inputs are gathered and the full Hessian is computed on rank 0.
        # For ColumnParallel modules: input is replicated, so per_rank=True gives
        # identical results on all ranks.
        accs: Dict[str, RuntimeHessian] = {}
        for w in cfg.weight_types:
            module_name = get_hess_name(block_id, w)
            module = _all_modules[module_name]
            is_row_parallel = (
                _RowParallelLinear is not None
                and isinstance(module, _RowParallelLinear)
                and _world_size > 1
            )
            accs[module_name] = RuntimeHessian(
                module,
                per_rank=False if is_row_parallel else None,
                dtype=torch.float64,
            )

        # Forward pass: accumulates all hessians AND captures block output.
        # Update cache in-place — each batch is copied to GPU before we
        # overwrite, so this is safe and avoids 2x CPU memory (74GB/rank).
        for i in range(0, cache.nsamples, cfg.hessian_batch_size):
            bs = min(cfg.hessian_batch_size, cache.nsamples - i)
            h = cache.get_cached_activations_batch(i, cfg.hessian_batch_size)
            h_out = layer(h, start_pos=0, freqs_cis=cache._freqs_cis, mask=cache._mask)
            cache._cached_h[i:i+bs] = h_out.to(cache.dtype).cpu()
            del h, h_out

        # Save hessians (rank 0 only).
        for module_name, acc in accs.items():
            if acc._is_main:
                hess_result = acc.get(normalize=True, normalize_by="tokens")
                if _is_rank0:
                    torch.save(hess_result.H.cpu(), out_dir / "hessians" / f"{safe_stem(module_name)}.pt")
                del hess_result
            acc.close()
        del accs
        cache.current_block_idx = block_id + 1

        # Save block output (= input to block block_id + 1).
        if _is_rank0:
            torch.save(
                cache._cached_h,
                out_dir / "block_outputs" / f"block_{block_id + 1:03d}.pt",
            )

        # Barrier: all ranks must finish I/O before any rank starts the next
        # block's forward pass (which uses RowParallel all-reduces).
        if _world_size > 1:
            _dist_mod.barrier()

        dt = time.time() - t0
        print(f"[precompute] block {block_id}/{n_layers-1}  ({dt:.1f}s)", flush=True)
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save metadata (rank 0) + barrier so all ranks wait
    # ------------------------------------------------------------------
    if _is_rank0:
        # Token hash: detect stale precomputed data from different tokenization.
        # Uses first/last 8 token IDs + total count — fast and collision-resistant.
        _flat_tokens = train_tokens[:cache.nsamples].reshape(-1)
        _token_hash = hashlib.sha256(
            f"{int(_flat_tokens.shape[0])}|"
            f"{_flat_tokens[:8].tolist()}|"
            f"{_flat_tokens[-8:].tolist()}".encode()
        ).hexdigest()[:16]
        meta = {
            "model_name": cfg.model_name,
            "n_layers": n_layers,
            "calib_dataset": cfg.calib_dataset,
            "calib_seed": cfg.calib_seed,
            "seqlen": cfg.seqlen,
            "hessian_nsamples": cfg.hessian_nsamples,
            "nsamples": cache.nsamples,
            "model_dtype": str(model_dtype),
            "weight_types": list(cfg.weight_types),
            "has_attention_importance": cfg.save_attention_importance,
            "attn_importance_sum_mode": cfg.attn_importance_sum_mode,
            "token_hash": _token_hash,
            "created_at": time.time(),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[precompute] done → {out_dir}", flush=True)

    # Barrier: ensure meta.json is written before any rank tries to load it.
    if _dist_ok and _world_size > 1:
        _dist_mod.barrier()

    # Free the model to reclaim GPU memory for the main pipeline.
    del model, cache
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    # Force glibc to return freed memory to OS so the next model load
    # doesn't push into swap and stall NCCL collectives.
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Loader: provides precomputed data to the quantization pipeline
# ---------------------------------------------------------------------------

class PrecomputedUnquantData:
    """Loads precomputed unquantized model statistics from disk.

    Provides hessians, block outputs, and attention importance without
    needing the unquantized model on GPU.

    For Qronos/residual cross-covariance (which depends on the quantized
    model), the unquant model can be kept on CPU and individual layers
    moved to GPU temporarily via :meth:`layer_on_device`.
    """

    def __init__(self, precomputed_dir: str | Path):
        self.dir = Path(precomputed_dir)
        if not (self.dir / "meta.json").exists():
            raise FileNotFoundError(f"No meta.json in {self.dir}")
        with open(self.dir / "meta.json") as f:
            self.meta: Dict[str, Any] = json.load(f)
        self._model_cpu: Optional[torch.nn.Module] = None

    # -- Properties --------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self.meta["model_name"]

    @property
    def n_layers(self) -> int:
        return self.meta["n_layers"]

    @property
    def nsamples(self) -> int:
        return self.meta["nsamples"]

    @property
    def seqlen(self) -> int:
        return self.meta["seqlen"]

    # -- CPU model for layer replay ----------------------------------------

    def set_model_cpu(self, model: torch.nn.Module):
        """Attach a CPU-resident unquantized model for layer replay."""
        self._model_cpu = model

    @property
    def model_cpu(self) -> torch.nn.Module:
        if self._model_cpu is None:
            raise RuntimeError(
                "CPU model not set.  Call set_model_cpu() first, or load "
                "with PrecomputedUnquantData.load_with_model()."
            )
        return self._model_cpu

    @property
    def has_model_cpu(self) -> bool:
        return self._model_cpu is not None

    # -- Data loaders ------------------------------------------------------

    def load_hessian(self, module_name: str, *, device: str | torch.device = "cpu") -> torch.Tensor:
        """Load precomputed Σ_X = E[X X^T] for *module_name*."""
        path = self.dir / "hessians" / f"{safe_stem(module_name)}.pt"
        if not path.exists():
            raise FileNotFoundError(f"No precomputed hessian for {module_name}: {path}")
        return torch.load(path, map_location=device)

    def has_hessian(self, module_name: str) -> bool:
        return (self.dir / "hessians" / f"{safe_stem(module_name)}.pt").exists()

    def load_block_input(self, block_id: int) -> torch.Tensor:
        """Load cached hidden states that are the input to block *block_id*.

        Returns: (nsamples, seqlen, hidden_dim) tensor on CPU.
        """
        path = self.dir / "block_outputs" / f"block_{block_id:03d}.pt"
        if not path.exists():
            raise FileNotFoundError(f"No precomputed block output: {path}")
        return torch.load(path, map_location="cpu")

    def load_attention_importance(self, layer_id: int) -> torch.Tensor:
        """Load per-token attention importance p_j for *layer_id*.

        Returns: (nsamples, seqlen) float64 tensor on CPU.
        """
        path = self.dir / "attention_importance" / f"layer_{layer_id:03d}.pt"
        if not path.exists():
            raise FileNotFoundError(f"No precomputed attention importance: {path}")
        return torch.load(path, map_location="cpu")

    def has_attention_importance(self, layer_id: int) -> bool:
        return (self.dir / "attention_importance" / f"layer_{layer_id:03d}.pt").exists()

    # -- Single-layer GPU replay -------------------------------------------

    @contextmanager
    def layer_on_device(self, block_id: int, device: torch.device):
        """Context manager: move one unquant layer to *device*, yield it, move back.

        Usage::

            with precomputed.layer_on_device(layer_id, device) as layer_u:
                # layer_u is on GPU, run forward passes
                ...
            # layer_u is back on CPU, GPU memory freed
        """
        layer = self.model_cpu.layers[block_id]
        layer.to(device)
        try:
            yield layer
        finally:
            layer.to("cpu")
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _relative_module_name(module_name: str) -> str:
    """'layers.5.attention.wq' → 'attention.wq'"""
    parts = module_name.split(".")
    return ".".join(parts[2:])


def _resize_layer_kv(layer: torch.nn.Module, batch_size: int):
    """Resize a single layer's KV caches to *batch_size*.

    precomputed.model_cpu layers are built with max_batch_size=32 (the model
    default).  When replayed with a larger batch, the caches must be resized.
    """
    attn = getattr(layer, "attention", None)
    if attn is None:
        return
    cache_k = getattr(attn, "cache_k", None)
    if cache_k is None or cache_k.shape[0] == batch_size:
        return
    _, sl, nkv, hd = cache_k.shape
    attn.cache_k = torch.zeros(batch_size, sl, nkv, hd, device=cache_k.device, dtype=cache_k.dtype)
    attn.cache_v = torch.zeros(batch_size, sl, nkv, hd, device=cache_k.device, dtype=attn.cache_v.dtype)


# ---------------------------------------------------------------------------
# Replay functions: compute cross-covariance stats using precomputed data
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_qronos_stats_precomputed(
    precomputed: PrecomputedUnquantData,
    model_quant: torch.nn.Module,
    cache_quant: ActivationCache,
    layer_id: int,
    module_name: str,
    *,
    normalize: bool = True,
    normalize_by: str = "tokens",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
    batch_size: int = 1,
    token_weights: Optional[torch.Tensor] = None,
    layer_unquant_on_device: Optional[torch.nn.Module] = None,
    gather_row_parallel: bool = False,
):
    """Compute Qronos stats (Σ_X, Σ_X̂, Σ_{X,X̂}) using precomputed block
    outputs and single-layer replay.

    If *layer_unquant_on_device* is provided, it is used directly (caller
    manages GPU placement).  Otherwise the layer is moved to GPU from the
    CPU model in *precomputed* and moved back when done.
    """
    from quant_layerwise.hessian_runtime import DualActivationCapture
    from quant_layerwise.qronos_stats import QronosStatsAccumulator

    # Load precomputed block input (CPU).  Pin memory for faster DMA transfers.
    h_unquant_all = precomputed.load_block_input(layer_id)
    if torch.cuda.is_available() and not h_unquant_all.is_pinned():
        h_unquant_all = h_unquant_all.pin_memory()
    # Use min of precomputed and quant cache sample counts to avoid shape mismatch.
    # Precomputed cache may have been generated with different nsamples.
    nsamples = min(h_unquant_all.shape[0], cache_quant.nsamples)

    # Resolve modules.
    layer_quant = model_quant.layers[layer_id]
    module_quant = dict(model_quant.named_modules())[module_name]
    device = module_quant.weight.device
    rel_name = _relative_module_name(module_name)

    # Place unquant layer on device if not already.
    own_layer = layer_unquant_on_device is None
    if own_layer:
        layer_unquant = precomputed.model_cpu.layers[layer_id]
        layer_unquant.to(device)
        _resize_layer_kv(layer_unquant, batch_size)
    else:
        layer_unquant = layer_unquant_on_device

    module_unquant = dict(layer_unquant.named_modules())[rel_name]
    # Use weight.shape[1] for local input dim (correct for RowParallel in multi-GPU,
    # where in_features is the full dim but weight is sharded along columns).
    n_features_local = int(module_unquant.weight.shape[1])
    if gather_row_parallel:
        import torch.distributed as _d
        n_features = n_features_local * _d.get_world_size()
    else:
        n_features = n_features_local

    # In precomputed mode only one unquant layer is on GPU at a time,
    # so even w2 accumulators (3 × 28672² × 8 = 18GB) fit on 96GB GPUs.
    acc_device = device

    # Always accumulate σ_X from scratch (no precomputed reuse).
    # Precomputed σ_X comes from run_precompute's model instance, which can produce
    # subtly different values than the pipeline's model (CUDA matmul non-determinism
    # across separate forward passes). Accumulating all three matrices (σ_X, σ_X̂, σ_XX̂)
    # from the same forward pass ensures consistency.
    precomputed_sigma_x = None
    if False and token_weights is None and precomputed.has_hessian(module_name):
        _h = precomputed.load_hessian(module_name, device=acc_device)
        if _h.shape[0] == n_features:
            precomputed_sigma_x = _h
            if verbose:
                print(f"[qronos-precomputed] reusing precomputed σ_X for {module_name} (skipping σ_X accumulation)", flush=True)
        else:
            if verbose:
                print(f"[qronos-precomputed] precomputed σ_X shape {tuple(_h.shape)} != accumulator dim {n_features}, will reaccumulate", flush=True)
            del _h

    acc = QronosStatsAccumulator(n_features, device=acc_device, dtype=dtype,
                                  precomputed_sigma_x=precomputed_sigma_x)

    freqs_cis = cache_quant._freqs_cis
    mask = cache_quant._mask

    # Determine fast-path mode based on weight type.
    # Instead of running the full TransformerBlock forward (attention + FFN)
    # and capturing module inputs via hooks, compute module inputs directly:
    #   wq/wk/wv → attention_norm(block_input)           [norm only]
    #   w1/w3    → ffn_norm(block_input + attention(...)) [attention only]
    #   wo       → hook on wo, run attention only         [attention + hook]
    #   w2       → compute silu(w1(·))*w3(·) directly    [attention + partial FFN]
    # Qwen3 has HF-style forward (no start_pos/freqs_cis/mask args on attention).
    # Always use full-forward + hooks (matches on-the-fly compute_qronos_stats_cached exactly).
    _FAST_NORM = False
    _FAST_ATTN = False
    _FAST_ATTN_HOOK = False
    _FAST_FFN = False

    capture = DualActivationCapture(module_unquant, module_quant, dtype=dtype)

    try:
        if _FAST_NORM:
            _fast_str = "norm-only"
        elif _FAST_ATTN:
            _fast_str = "attn-only"
        elif _FAST_ATTN_HOOK:
            _fast_str = "attn+hook"
        elif _FAST_FFN:
            _fast_str = "attn+partial-ffn"
        else:
            _fast_str = "full-forward"
        if verbose:
            print(
                f"[qronos-precomputed] module={module_name}  block={layer_id}  "
                f"nsamples={nsamples}  batch_size={batch_size}  acc_device={acc_device}  acc_dtype={dtype}  "
                f"path={_fast_str}",
                flush=True,
            )

        import time as _time
        _t_qronos_start = _time.monotonic()

        # Double-buffered prefetching: overlap next batch CPU→GPU with current compute.
        _prefetch_stream = torch.cuda.Stream(device) if device.type == "cuda" else None

        def _prefetch(idx):
            """Prefetch batch starting at *idx* on the copy stream."""
            if idx >= nsamples:
                return None, None
            _end = min(idx + batch_size, nsamples)
            if _prefetch_stream is not None:
                with torch.cuda.stream(_prefetch_stream):
                    _hu = h_unquant_all[idx:_end].to(device=device, non_blocking=True)
                    _hq = cache_quant.get_cached_activations_batch(idx, batch_size)
                return _hu, _hq
            else:
                return (h_unquant_all[idx:_end].to(device=device),
                        cache_quant.get_cached_activations_batch(idx, batch_size))

        # Kick off first prefetch.
        _next_hu, _next_hq = _prefetch(0)

        for i in range(0, nsamples, batch_size):
            end = min(i + batch_size, nsamples)
            # Wait for current batch transfer to finish.
            if _prefetch_stream is not None:
                _prefetch_stream.synchronize()
            h_u, h_q = _next_hu, _next_hq

            # Diagnostic: verify precomputed block inputs match quant cache at layer 0
            if i == 0 and layer_id == 0:
                _diff = (h_u.float() - h_q.float()).abs().max().item()
                _scale = max(h_q.float().abs().max().item(), 1e-30)
                print(f"[precompute-diag] layer 0 batch 0: h_u vs h_q diff={_diff:.6e} rel={_diff/_scale:.6e}", flush=True)

            if _FAST_NORM:
                # wq/wk/wv: input = attention_norm(block_input)
                X = layer_unquant.attention_norm(h_u).detach().reshape(-1, n_features_local).to(dtype)
                X_hat = layer_quant.attention_norm(h_q).detach().reshape(-1, n_features_local).to(dtype)
            elif _FAST_ATTN:
                # w1/w3: input = ffn_norm(block_input + attention(attention_norm(block_input)))
                attn_u = layer_unquant.attention(
                    layer_unquant.attention_norm(h_u), start_pos=0, freqs_cis=freqs_cis, mask=mask)
                X = layer_unquant.ffn_norm(h_u + attn_u).detach().reshape(-1, n_features_local).to(dtype)
                del attn_u
                attn_q = layer_quant.attention(
                    layer_quant.attention_norm(h_q), start_pos=0, freqs_cis=freqs_cis, mask=mask)
                X_hat = layer_quant.ffn_norm(h_q + attn_q).detach().reshape(-1, n_features_local).to(dtype)
                del attn_q
            elif _FAST_ATTN_HOOK:
                # wo: run attention only (hooks capture wo input), skip FFN
                _ = layer_unquant.attention(
                    layer_unquant.attention_norm(h_u), start_pos=0, freqs_cis=freqs_cis, mask=mask)
                _ = layer_quant.attention(
                    layer_quant.attention_norm(h_q), start_pos=0, freqs_cis=freqs_cis, mask=mask)
                X, X_hat = capture.get_activations()
            elif _FAST_FFN:
                # w2: input = silu(w1(ffn_in)) * w3(ffn_in)
                import torch.nn.functional as _F
                attn_u = layer_unquant.attention(
                    layer_unquant.attention_norm(h_u), start_pos=0, freqs_cis=freqs_cis, mask=mask)
                ffn_in_u = layer_unquant.ffn_norm(h_u + attn_u)
                del attn_u
                X = (_F.silu(layer_unquant.feed_forward.w1(ffn_in_u))
                     * layer_unquant.feed_forward.w3(ffn_in_u)).detach().reshape(-1, n_features_local).to(dtype)
                del ffn_in_u
                attn_q = layer_quant.attention(
                    layer_quant.attention_norm(h_q), start_pos=0, freqs_cis=freqs_cis, mask=mask)
                ffn_in_q = layer_quant.ffn_norm(h_q + attn_q)
                del attn_q
                X_hat = (_F.silu(layer_quant.feed_forward.w1(ffn_in_q))
                         * layer_quant.feed_forward.w3(ffn_in_q)).detach().reshape(-1, n_features_local).to(dtype)
                del ffn_in_q
            else:
                # Fallback: full block forward with hooks
                _ = layer_unquant(h_u, start_pos=0, freqs_cis=freqs_cis, mask=mask)
                _ = layer_quant(h_q, start_pos=0, freqs_cis=freqs_cis, mask=mask)
                X, X_hat = capture.get_activations()

            # Free input batch before gather/accumulate to reduce peak memory.
            del h_u, h_q

            if X is not None and X_hat is not None:
                # For RowParallel: both unquant and quant inputs are sharded.
                # Gather to full dimension for correct cross-covariance.
                if gather_row_parallel:
                    from quant_layerwise.hessian_runtime import _all_gather_features
                    X = _all_gather_features(X)
                    X_hat = _all_gather_features(X_hat)
                w = None
                if token_weights is not None:
                    actual_batch = end - i
                    w = token_weights[i : i + actual_batch].to(X.device)
                acc.accumulate(X, X_hat, weights=w)

            # Prefetch next batch AFTER accumulate — avoids holding two batches
            # on GPU during the memory-heavy all_gather.
            _next_hu, _next_hq = _prefetch(i + batch_size)

        del _next_hu, _next_hq, _prefetch_stream
        _t_qronos_elapsed = _time.monotonic() - _t_qronos_start
        if verbose:
            print(f"[qronos-precomputed] {module_name} accumulation took {_t_qronos_elapsed:.1f}s (dtype={dtype}, path={_fast_str})", flush=True)

        stats = acc.get(normalize=normalize, normalize_by=normalize_by)
        # Ensure all ranks have bitwise-identical sigma matrices.
        # CUDA matmul non-determinism across GPUs can cause O(1e-13)
        # divergence which breaks the T/Gamma rescaler (gamma must be
        # identical across ranks).
        if gather_row_parallel:
            stats.broadcast_from_rank0()
        return stats
    finally:
        if capture is not None:
            capture.close()
        if own_layer:
            layer_unquant.to("cpu")
            torch.cuda.empty_cache()


@torch.no_grad()
def compute_residual_stats_precomputed(
    precomputed: PrecomputedUnquantData,
    model_quant: torch.nn.Module,
    cache_quant: ActivationCache,
    layer_id: int,
    weight_type: str,
    *,
    normalize: bool = True,
    normalize_by: str = "tokens",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
    batch_size: int = 1,
    layer_unquant_on_device: Optional[torch.nn.Module] = None,
    gather_row_parallel: bool = False,
):
    """Compute residual stats Σ_{ΔR,X̂} using precomputed block outputs."""
    from quant_layerwise.qronos_stats import ResidualStatsAccumulator

    h_unquant_all = precomputed.load_block_input(layer_id)
    if torch.cuda.is_available() and not h_unquant_all.is_pinned():
        h_unquant_all = h_unquant_all.pin_memory()
    # Use min of precomputed and quant cache sample counts to avoid shape mismatch.
    # Precomputed cache may have been generated with different nsamples.
    nsamples = min(h_unquant_all.shape[0], cache_quant.nsamples)

    layer_quant = model_quant.layers[layer_id]

    if weight_type == "wo":
        module = layer_quant.attention.wo
    elif weight_type == "w2":
        module = layer_quant.feed_forward.w2
    else:
        raise ValueError(f"Unsupported weight_type: {weight_type}")

    out_features = int(module.weight.shape[0])
    in_features_local = int(module.weight.shape[1])
    import torch.distributed as _d
    if gather_row_parallel and _d is not None and _d.is_initialized() and _d.get_world_size() > 1:
        in_features = in_features_local * _d.get_world_size()
    else:
        in_features = in_features_local
    device = module.weight.device

    own_layer = layer_unquant_on_device is None
    if own_layer:
        layer_unquant = precomputed.model_cpu.layers[layer_id]
        layer_unquant.to(device)
        _resize_layer_kv(layer_unquant, batch_size)
    else:
        layer_unquant = layer_unquant_on_device

    acc = ResidualStatsAccumulator(out_features, in_features, device=device, dtype=dtype)

    freqs_cis = cache_quant._freqs_cis
    mask = cache_quant._mask

    # Optimization: load R_u (unquant block output) directly from precomputed
    # data instead of running the unquant model forward.  R_u is available as
    # the input to the *next* block: block_output[layer_id+1].
    # For wo: R_u = block input (x), captured by pre-hook on the block.
    # For w2: R_u = h after attention (input to ffn_norm), captured by
    #         pre-hook on ffn_norm.
    # Note: for wo, R_u is the block *input* (skip connection for attention),
    # which equals h_unquant_all.  For w2, R_u is h_u + attention_output
    # (skip connection for FFN).  We can load R_u for wo directly (= block input),
    # but for w2 we need the intermediate which requires running attention.
    #
    # For the quant model, we need R_q and X̂ which require a forward pass.
    # Optimization: run only the quant model forward (not both).
    # For the unquant model:
    #   wo: R_u = block_input = h_unquant_all (no forward needed)
    #   w2: R_u = h_u + attention_u (need attention forward, skip FFN)

    # Set up hooks for quant model only:
    # - R_q: captured from block pre-hook (wo) or ffn_norm pre-hook (w2)
    # - X̂: captured from wo or w2 pre-hook
    # We use a simplified capture that only hooks the quant model + computes R_u directly.

    # For R_q, we need the quant model's residual stream at the right point.
    # Instead of full ResidualCapture, we hook only the quant layer.
    _handles = []
    _R_q = [None]
    _X_hat = [None]
    _resid_dtype = dtype

    if weight_type == "wo":
        # R_q = block input to quant layer (same as h_q)
        def _hook_rq(mod, inputs):
            _R_q[0] = inputs[0].detach().to(_resid_dtype)
        _handles.append(layer_quant.register_forward_pre_hook(_hook_rq))
        # X̂ = input to wo in quant model
        def _hook_xhat(mod, inputs):
            x = inputs[0]
            _X_hat[0] = x.detach().reshape(-1, x.shape[-1]).to(_resid_dtype)
        _handles.append(layer_quant.attention.wo.register_forward_pre_hook(_hook_xhat))
    elif weight_type == "w2":
        # R_q = input to ffn_norm in quant layer (= h_q + attn_q)
        def _hook_rq(mod, inputs):
            _R_q[0] = inputs[0].detach().to(_resid_dtype)
        _handles.append(layer_quant.ffn_norm.register_forward_pre_hook(_hook_rq))
        # X̂ = input to w2 in quant model
        def _hook_xhat(mod, inputs):
            x = inputs[0]
            _X_hat[0] = x.detach().reshape(-1, x.shape[-1]).to(_resid_dtype)
        _handles.append(layer_quant.feed_forward.w2.register_forward_pre_hook(_hook_xhat))

    try:
        if verbose:
            print(
                f"[residual-precomputed] weight={weight_type}  block={layer_id}  "
                f"nsamples={nsamples}  batch_size={batch_size}  (full-forward both models)",
                flush=True,
            )

        import time as _time
        _t_resid_start = _time.monotonic()

        # Double-buffered prefetching for residual accumulation.
        _prefetch_stream_r = torch.cuda.Stream(device) if device.type == "cuda" else None

        def _prefetch_r(idx):
            if idx >= nsamples:
                return None, None
            _end = min(idx + batch_size, nsamples)
            if _prefetch_stream_r is not None:
                with torch.cuda.stream(_prefetch_stream_r):
                    _hu = h_unquant_all[idx:_end].to(device=device, non_blocking=True)
                    _hq = cache_quant.get_cached_activations_batch(idx, batch_size)
                return _hu, _hq
            else:
                return (h_unquant_all[idx:_end].to(device=device),
                        cache_quant.get_cached_activations_batch(idx, batch_size))

        _next_hu_r, _next_hq_r = _prefetch_r(0)

        for i in range(0, nsamples, batch_size):
            if _prefetch_stream_r is not None:
                _prefetch_stream_r.synchronize()
            h_u, h_q = _next_hu_r, _next_hq_r

            # Run full block forward on BOTH models (matches on-the-fly ResidualCapture).
            # Hooks capture R_u, R_q, and X̂ during the forward passes.
            _R_u_buf = [None]
            if weight_type == "wo":
                _h_ru = layer_unquant.register_forward_pre_hook(
                    lambda _m, inp, _b=_R_u_buf: _b.__setitem__(0, inp[0].detach().to(_resid_dtype)))
            else:  # w2
                _h_ru = layer_unquant.ffn_norm.register_forward_pre_hook(
                    lambda _m, inp, _b=_R_u_buf: _b.__setitem__(0, inp[0].detach().to(_resid_dtype)))
            _ = layer_unquant(h_u, start_pos=0, freqs_cis=freqs_cis, mask=mask)
            _h_ru.remove()
            R_u = _R_u_buf[0]

            # Run quant model forward (hooks capture R_q and X̂)
            _ = layer_quant(h_q, start_pos=0, freqs_cis=freqs_cis, mask=mask)

            # Free input batch before gather/accumulate to reduce peak memory.
            del h_u, h_q

            R_q = _R_q[0]
            X_h = _X_hat[0]
            _R_q[0] = None
            _X_hat[0] = None

            if R_u is not None and R_q is not None and X_h is not None:
                # For RowParallel: X_h from TP'd quantized model is sharded.
                if gather_row_parallel:
                    from quant_layerwise.hessian_runtime import _all_gather_features
                    X_h = _all_gather_features(X_h)
                acc.accumulate(R_u, R_q, X_h)

            # Prefetch next batch AFTER accumulate — avoids holding two batches
            # on GPU during the memory-heavy all_gather.
            _next_hu_r, _next_hq_r = _prefetch_r(i + batch_size)

        del _next_hu_r, _next_hq_r, _prefetch_stream_r
        _t_resid_elapsed = _time.monotonic() - _t_resid_start
        if verbose:
            print(f"[residual-precomputed] {weight_type} accumulation took {_t_resid_elapsed:.1f}s", flush=True)

        stats = acc.get(normalize=normalize, normalize_by=normalize_by)
        if gather_row_parallel:
            stats.broadcast_from_rank0()
        return stats
    finally:
        for h in _handles:
            h.remove()
        if own_layer:
            layer_unquant.to("cpu")
            torch.cuda.empty_cache()


@torch.no_grad()
def compute_attention_importance_precomputed(
    precomputed: PrecomputedUnquantData,
    cache_quant: ActivationCache,
    layer_id: int,
    *,
    batch_size: int = 1,
    sum_mode: bool = False,
    layer_unquant_on_device: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """Compute attention importance via single-layer replay.

    Use this when the precomputed attention importance was saved with a
    different sum_mode than needed, or was not saved at all.
    """
    h_unquant_all = precomputed.load_block_input(layer_id)
    if torch.cuda.is_available() and not h_unquant_all.is_pinned():
        h_unquant_all = h_unquant_all.pin_memory()
    # Use min of precomputed and quant cache sample counts to avoid shape mismatch.
    # Precomputed cache may have been generated with different nsamples.
    nsamples = min(h_unquant_all.shape[0], cache_quant.nsamples)
    device = cache_quant.device
    seqlen = cache_quant.seqlen

    own_layer = layer_unquant_on_device is None
    if own_layer:
        layer_unquant = precomputed.model_cpu.layers[layer_id]
        layer_unquant.to(device)
        _resize_layer_kv(layer_unquant, batch_size)
    else:
        layer_unquant = layer_unquant_on_device

    is_qwen3 = hasattr(layer_unquant, "_hf_layer")

    all_weights = []
    try:
        for i in range(0, nsamples, batch_size):
            end = min(i + batch_size, nsamples)
            h = h_unquant_all[i:end].to(device=device, non_blocking=True)

            if is_qwen3:
                from quant_layerwise.hessian_runtime import _compute_attn_probs_qwen3
                probs, n_heads = _compute_attn_probs_qwen3(layer_unquant, h, seqlen)
            else:
                from quant_layerwise.hessian_runtime import _compute_attn_probs_llama
                # _compute_attn_probs_llama expects a cache-like object for freqs_cis/mask
                probs, n_heads = _compute_attn_probs_llama(
                    layer_unquant, h, cache_quant,
                )

            col_sum = probs.sum(dim=(1, 2))
            if sum_mode:
                p_j = col_sum / n_heads
            else:
                count_j = torch.arange(seqlen, 0, -1, device=col_sum.device, dtype=torch.float32)
                p_j = col_sum / (n_heads * count_j.unsqueeze(0))
            all_weights.append(p_j.to(torch.float64).cpu())

        return torch.cat(all_weights, dim=0)
    finally:
        if own_layer:
            layer_unquant.to("cpu")
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Precompute reference activations for adaptive search (precomputed mode)
# ---------------------------------------------------------------------------

@torch.no_grad()
def precompute_wo_in_ref(
    precomputed: PrecomputedUnquantData,
    cache_quant: ActivationCache,
    layer_id: int,
    batch_size: int,
    max_samples: Optional[int] = None,
    *,
    layer_unquant_on_device: Optional[torch.nn.Module] = None,
) -> Tuple[list, list]:
    """Precompute wo input reference activations using precomputed block outputs.

    Equivalent to _precompute_wo_in_ref but uses disk-loaded block inputs
    instead of cache_unquant.

    Returns (ref_acts, ref_sq_per_batch) for use as precomputed_ref
    in _compute_wo_in_rel_mse.
    """
    h_unquant_all = precomputed.load_block_input(layer_id)
    if torch.cuda.is_available() and not h_unquant_all.is_pinned():
        h_unquant_all = h_unquant_all.pin_memory()
    device = cache_quant.device
    freqs_cis = cache_quant._freqs_cis
    mask = cache_quant._mask

    own_layer = layer_unquant_on_device is None
    if own_layer:
        layer_unquant = precomputed.model_cpu.layers[layer_id]
        layer_unquant.to(device)
        _resize_layer_kv(layer_unquant, batch_size)
    else:
        layer_unquant = layer_unquant_on_device

    refs = []
    ref_sq = []
    n = h_unquant_all.shape[0]
    if max_samples is not None:
        n = min(n, max_samples)
    try:
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            h_u = h_unquant_all[i:end].to(device=device, non_blocking=True)
            captured = []
            hnd = layer_unquant.attention.wo.register_forward_pre_hook(
                lambda _m, inp, _c=captured: _c.append(inp[0].detach()))
            _ = layer_unquant(h_u, start_pos=0, freqs_cis=freqs_cis, mask=mask)
            hnd.remove()
            wo_ref = captured[0].float()
            refs.append(wo_ref.cpu())
            ref_sq.append(wo_ref.pow(2).sum().item())
        return refs, ref_sq
    finally:
        if own_layer:
            layer_unquant.to("cpu")
            torch.cuda.empty_cache()


@torch.no_grad()
def precompute_next_qkv_in_ref(
    precomputed: PrecomputedUnquantData,
    cache_quant: ActivationCache,
    layer_id: int,
    batch_size: int,
    max_samples: Optional[int] = None,
    *,
    layer_unquant_on_device: Optional[torch.nn.Module] = None,
) -> Tuple[list, list]:
    """Precompute next-layer QKV input reference activations using precomputed block outputs.

    Returns (ref_acts, ref_sq_per_batch) for use as precomputed_ref
    in _compute_next_qkv_in_rel_mse.
    """
    h_unquant_all = precomputed.load_block_input(layer_id)
    if torch.cuda.is_available() and not h_unquant_all.is_pinned():
        h_unquant_all = h_unquant_all.pin_memory()
    device = cache_quant.device
    freqs_cis = cache_quant._freqs_cis
    mask = cache_quant._mask

    own_layer = layer_unquant_on_device is None
    if own_layer:
        layer_unquant = precomputed.model_cpu.layers[layer_id]
        layer_unquant.to(device)
        _resize_layer_kv(layer_unquant, batch_size)
    else:
        layer_unquant = layer_unquant_on_device

    # Also need next layer's attention_norm on device
    next_id = layer_id + 1
    next_norm_u = precomputed.model_cpu.layers[next_id].attention_norm
    next_norm_u.to(device)

    refs = []
    ref_sq = []
    n = h_unquant_all.shape[0]
    if max_samples is not None:
        n = min(n, max_samples)
    try:
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            h_u = h_unquant_all[i:end].to(device=device, non_blocking=True)
            out_u = layer_unquant(h_u, start_pos=0, freqs_cis=freqs_cis, mask=mask)
            ref = next_norm_u(out_u).float()
            refs.append(ref.cpu())
            ref_sq.append(ref.pow(2).sum().item())
        return refs, ref_sq
    finally:
        next_norm_u.to("cpu")
        if own_layer:
            layer_unquant.to("cpu")
            torch.cuda.empty_cache()


@torch.no_grad()
def precompute_w2_in_ref(
    precomputed: PrecomputedUnquantData,
    cache_quant: ActivationCache,
    layer_id: int,
    batch_size: int,
    max_samples: Optional[int] = None,
    *,
    layer_unquant_on_device: Optional[torch.nn.Module] = None,
) -> Tuple[list, list]:
    """Precompute w2 input reference activations using precomputed block outputs.

    Returns (ref_acts, ref_sq_per_batch) for use as precomputed_ref
    in _compute_w2_in_rel_mse.
    """
    h_unquant_all = precomputed.load_block_input(layer_id)
    if torch.cuda.is_available() and not h_unquant_all.is_pinned():
        h_unquant_all = h_unquant_all.pin_memory()
    device = cache_quant.device
    freqs_cis = cache_quant._freqs_cis
    mask = cache_quant._mask

    own_layer = layer_unquant_on_device is None
    if own_layer:
        layer_unquant = precomputed.model_cpu.layers[layer_id]
        layer_unquant.to(device)
        _resize_layer_kv(layer_unquant, batch_size)
    else:
        layer_unquant = layer_unquant_on_device

    refs = []
    ref_sq = []
    n = h_unquant_all.shape[0]
    if max_samples is not None:
        n = min(n, max_samples)
    try:
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            h_u = h_unquant_all[i:end].to(device=device, non_blocking=True)
            captured = []
            hnd = layer_unquant.feed_forward.w2.register_forward_pre_hook(
                lambda _m, inp, _c=captured: _c.append(inp[0].detach()))
            _ = layer_unquant(h_u, start_pos=0, freqs_cis=freqs_cis, mask=mask)
            hnd.remove()
            w2_ref = captured[0].float()
            refs.append(w2_ref.cpu())
            ref_sq.append(w2_ref.pow(2).sum().item())
        return refs, ref_sq
    finally:
        if own_layer:
            layer_unquant.to("cpu")
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Precompute unquantized model statistics for quantization.",
    )
    parser.add_argument("--model_name", required=True, help="Model name (e.g. '3-8B', 'qwen3-8B')")
    parser.add_argument("--output_dir", required=True, help="Directory to save precomputed data")
    parser.add_argument("--calib_dataset", default="redpajama")
    parser.add_argument("--calib_seed", type=int, default=42)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--hessian_nsamples", type=int, default=0,
                        help="Number of calibration samples (0 = all available)")
    parser.add_argument("--hessian_batch_size", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no_attention_importance", action="store_true")

    args = parser.parse_args()

    cfg = PrecomputeConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        calib_dataset=args.calib_dataset,
        calib_seed=args.calib_seed,
        seqlen=args.seqlen,
        hessian_nsamples=args.hessian_nsamples,
        hessian_batch_size=args.hessian_batch_size,
        save_attention_importance=not args.no_attention_importance,
    )
    run_precompute(cfg, local_rank=args.local_rank)
