from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import torch.distributed as dist
except Exception:  # pragma: no cover
    dist = None

try:
    from fairscale.nn.model_parallel.layers import ColumnParallelLinear, RowParallelLinear
except Exception:  # pragma: no cover
    ColumnParallelLinear = None
    RowParallelLinear = None



def _resize_kv_caches(model: torch.nn.Module, new_batch_size: int):
    """Resize KV caches in all attention layers to match new_batch_size.

    The model pre-allocates KV caches with max_batch_size (default 32).
    Supports both upsizing (for larger batches) and downsizing (to free memory
    when processing layers that need smaller batches, e.g. w2).
    """
    for layer in model.layers:
        attn = layer.attention
        old_cache_k = attn.cache_k
        old_cache_v = attn.cache_v

        # Only resize if needed
        if old_cache_k.shape[0] == new_batch_size:
            continue

        # Get cache dimensions
        _, max_seq_len, n_kv_heads, head_dim = old_cache_k.shape

        # Create new caches
        attn.cache_k = torch.zeros(
            (new_batch_size, max_seq_len, n_kv_heads, head_dim),
            device=old_cache_k.device,
            dtype=old_cache_k.dtype,
        )
        attn.cache_v = torch.zeros(
            (new_batch_size, max_seq_len, n_kv_heads, head_dim),
            device=old_cache_v.device,
            dtype=old_cache_v.dtype,
        )

        # Free old caches
        del old_cache_k, old_cache_v


class ActivationCache:
    """Cache hidden states at transformer block boundaries to avoid quadratic recomputation.

    Instead of running the full model for each layer's Hessian computation,
    we cache activations and only run the target block.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: torch.Tensor,
        seqlen: int,
        nsamples: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
        batch_size: int = 1,
    ):
        """Initialize cache by computing embeddings for all samples.

        Args:
            model: Transformer model
            dataset: Token IDs shaped (total_len,) or (nsamples, seqlen)
            seqlen: Sequence length
            nsamples: Number of samples to cache
            device: Device for cached tensors
            dtype: Data type for cached tensors (default: auto-detect from model)
            batch_size: Batch size for processing (used to resize KV caches if needed)
        """
        self.model = model
        self.seqlen = seqlen
        self.nsamples = nsamples
        self.device = device
        # Auto-detect dtype from model if not specified
        if dtype is None:
            dtype = next(model.parameters()).dtype
        self.dtype = dtype

        # Resize KV caches if batch_size exceeds model's max_batch_size
        if batch_size > 1:
            _resize_kv_caches(model, batch_size)

        # Current block index - activations are valid as input to this block
        self.current_block_idx = 0

        # Reshape dataset if needed
        if dataset.ndim == 1:
            total_len = int(dataset.shape[0])
            nseq = total_len // seqlen
            dataset = dataset[: nseq * seqlen].reshape(nseq, seqlen)

        use = min(nsamples, dataset.shape[0])
        self.nsamples = use

        # Compute and cache embeddings for all samples
        # Shape: [nsamples, seqlen, dim]
        self._cached_h = self._compute_embeddings(dataset[:use])

        # Precompute freqs_cis and mask (reused for all blocks)
        self._freqs_cis = model.freqs_cis[:seqlen].to(device)
        self._mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([
                torch.zeros((seqlen, 0), device=device),  # start_pos=0
                mask
            ])
            self._mask = mask.to(self.dtype)

    @torch.no_grad()
    def _compute_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compute token embeddings for all samples."""
        chunk = max(1, 2 * 1024**3 // (tokens.shape[1] * 8192 * 2))  # ~2GB per chunk
        nsamples = tokens.shape[0]
        result = None
        for i in range(0, nsamples, chunk):
            end = min(i + chunk, nsamples)
            h = self.model.tok_embeddings(tokens[i:end].to(self.device))
            h_cpu = h.to(self.dtype).cpu()
            if result is None:
                # Pre-allocate full output after first chunk reveals the shape.
                # pin_memory=True enables async DMA for later CPU→GPU transfers.
                result = torch.empty(
                    nsamples, h_cpu.shape[1], h_cpu.shape[2],
                    dtype=h_cpu.dtype, device="cpu",
                    pin_memory=torch.cuda.is_available(),
                )
            result[i:end] = h_cpu
            del h, h_cpu
        return result

    def get_cached_activations(self, sample_idx: int) -> torch.Tensor:
        """Get cached activations for a specific sample.

        Returns tensor of shape [1, seqlen, dim] on the target device.
        """
        return self._cached_h[sample_idx:sample_idx+1].to(
            device=self.device, dtype=self.dtype, non_blocking=True)

    def get_cached_activations_batch(self, start_idx: int, batch_size: int) -> torch.Tensor:
        """Get cached activations for a batch of samples.

        Args:
            start_idx: Starting sample index
            batch_size: Number of samples to retrieve

        Returns:
            Tensor of shape [batch_size, seqlen, dim] on the target device.
        """
        end_idx = min(start_idx + batch_size, self.nsamples)
        return self._cached_h[start_idx:end_idx].to(
            device=self.device, dtype=self.dtype, non_blocking=True)

    @torch.no_grad()
    def advance_through_block(self, block_idx: int, batch_size: int = 1):
        """Advance cached activations through a transformer block.

        Should be called after all weights in block_idx have been quantized.
        Updates the cache to hold activations that are input to block_idx+1.

        Args:
            block_idx: Which block to advance through
            batch_size: Number of samples to process in parallel (default: 1)
        """
        if block_idx != self.current_block_idx:
            raise ValueError(
                f"Cannot advance through block {block_idx}, "
                f"cache is at block {self.current_block_idx}"
            )

        layer = self.model.layers[block_idx]

        # Update cache in-place — each batch is copied to GPU before we
        # overwrite, so this is safe and avoids 2x CPU memory.
        for i in range(0, self.nsamples, batch_size):
            bs = min(batch_size, self.nsamples - i)
            h = self.get_cached_activations_batch(i, batch_size)
            # Run the full block on the batch
            h_out = layer(h, start_pos=0, freqs_cis=self._freqs_cis, mask=self._mask)
            self._cached_h[i:i+bs] = h_out.to(self.dtype).cpu()
            del h, h_out
        self.current_block_idx = block_idx + 1


def _all_gather_features(X: torch.Tensor) -> torch.Tensor:
    """All-gather tensor along last dimension (feature dim) across ranks.

    Used to reconstruct full input from RowParallel shards:
    (batch, n_local) on each rank → (batch, n_full) on all ranks.

    Uses a single contiguous buffer to avoid memory fragmentation from
    allocating ws separate tensors + cat.
    """
    if dist is None or not dist.is_initialized():
        return X
    ws = dist.get_world_size()
    if ws <= 1:
        return X
    # Allocate one contiguous buffer for all shards, then split into views.
    # This avoids ws+1 separate allocations (ws shards + cat output).
    shape = list(X.shape)
    shape[-1] *= ws
    buf = torch.empty(shape, dtype=X.dtype, device=X.device)
    shards = list(buf.chunk(ws, dim=-1))
    dist.all_gather(shards, X.contiguous())
    return buf


def is_linear(module: torch.nn.Module) -> bool:
    if ColumnParallelLinear is not None and isinstance(module, ColumnParallelLinear):
        return True
    if RowParallelLinear is not None and isinstance(module, RowParallelLinear):
        return True
    return isinstance(module, torch.nn.Linear)


@dataclass
class HessianResult:
    H: torch.Tensor
    nseq: int
    ntokens: int


class RuntimeHessian:
    """Hook-based Hessian accumulator.

    When per_rank=True (default when world_size > 1), each rank computes
    its own local Hessian from whatever input it sees.  For ColumnParallel
    layers the input is replicated so all ranks get the same H.  For
    RowParallel layers each rank gets H for its local input shard — exactly
    the right Hessian for quantizing that rank's weight shard.

    When per_rank=False (legacy), RowParallel inputs are gathered to
    dst_rank which builds the full H.  Only dst_rank can call get().
    """

    def __init__(
        self,
        module: torch.nn.Module,
        *,
        dst_rank: int = 0,
        per_rank: Optional[bool] = None,
        dtype: torch.dtype = torch.float64,
    ):
        if not is_linear(module):
            raise TypeError(f"RuntimeHessian expects a linear module, got {type(module)}")

        self.module = module
        self.dst_rank = int(dst_rank)
        self.dtype = dtype

        self._dist_enabled = bool(dist is not None and dist.is_available() and dist.is_initialized())
        self._world_size = dist.get_world_size() if self._dist_enabled else 1
        self._rank = dist.get_rank() if self._dist_enabled else 0

        # per_rank mode: each rank accumulates its own local H (no gathering).
        # Default: True when world_size > 1, False otherwise (backward compat).
        if per_rank is None:
            per_rank = self._world_size > 1
        self._per_rank = per_rank

        # n = local input features (what this rank's weight shard sees)
        n = int(module.weight.shape[1])
        self.n = n

        self._do_gather = False
        if not self._per_rank:
            # Legacy mode: gather RowParallel inputs to dst_rank
            if RowParallelLinear is not None and isinstance(module, RowParallelLinear) and self._world_size > 1:
                self._do_gather = True
                # Full input dim when gathering
                n = int(module.in_features)
                self.n = n

        if self._per_rank:
            # Every rank accumulates
            self._is_main = True
        else:
            self._is_main = (self._rank == self.dst_rank)

        if self._is_main:
            self.H = torch.zeros((self.n, self.n), device=module.weight.device, dtype=self.dtype)
            self.nseq = 0
            self.ntokens = 0
        else:
            self.H = None
            self.nseq = 0
            self.ntokens = 0

        # Register hook.
        self._handle = module.register_forward_pre_hook(self._hook)

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _maybe_gather(self, X: torch.Tensor) -> Optional[torch.Tensor]:
        """Gather RowParallelLinear inputs to dst_rank so we can form full X."""
        if not self._do_gather:
            return X

        assert dist is not None and dist.is_initialized(), "distributed must be initialized for gather"
        if self._rank == self.dst_rank:
            tensor_list = [torch.zeros_like(X) for _ in range(self._world_size)]
            dist.gather(X, tensor_list, dst=self.dst_rank)
            X = torch.cat(tensor_list, dim=-1)
            return X
        else:
            dist.gather(X, None, dst=self.dst_rank)
            return None

    def _hook(self, _module: torch.nn.Module, inputs):
        X = inputs[0]
        X = self._maybe_gather(X)
        if X is None:
            return

        if not self._is_main:
            return

        self.nseq += int(X.shape[0])

        X = X.detach().reshape(-1, X.shape[-1]).to(self.dtype)
        self.ntokens += int(X.shape[0])

        self.H.addmm_(X.T, X)

    def get(
        self,
        *,
        normalize: bool = True,
        normalize_by: str = "tokens",
        eps: float = 1e-12,
    ) -> HessianResult:
        if not self._is_main:
            raise RuntimeError("Only dst_rank process can call get().")
        if self.nseq <= 0:
            raise RuntimeError("No samples were accumulated; did the model run forward?")

        H = self.H
        if normalize:
            nb = str(normalize_by).strip().lower()
            if nb in ("seq", "sequences", "batch"):
                denom = max(self.nseq, 1)
            elif nb in ("token", "tokens"):
                denom = max(self.ntokens, 1)
            else:
                raise ValueError(f"normalize_by must be 'seq' or 'tokens', got {normalize_by!r}")
            H = H / float(denom)

        # Add a tiny ridge for numerical stability.
        n = H.shape[0]
        diag = torch.arange(n, device=H.device)
        H = H.clone()
        H[diag, diag] += eps

        return HessianResult(H=H, nseq=int(self.nseq), ntokens=int(self.ntokens))


@torch.no_grad()
def compute_module_hessian(
    model: torch.nn.Module,
    dataset: torch.Tensor,
    module_name: str,
    *,
    seqlen: int,
    nsamples: int,
    dst_rank: int = 0,
    normalize: bool = True,
    normalize_by: str = "tokens",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
):
    """Compute Hessian for one module by running `nsamples` sequences through the model.

    Args:
        model: Transformer
        dataset: token IDs shaped (total_len,) on CPU, or (nsamples, seqlen) on GPU.
        module_name: name in model.named_modules()
        seqlen: sequence length
        nsamples: number of sequences/batches (batch size 1)
        dst_rank: for distributed gathering
        normalize: return average vs sum
    """
    mods = dict(model.named_modules())
    if module_name not in mods:
        raise KeyError(f"Module '{module_name}' not found in model")

    module = mods[module_name]

    acc = RuntimeHessian(module, dst_rank=dst_rank, per_rank=None, dtype=dtype)

    try:
        # Ensure dataset shaped (nsamples, seqlen)
        if dataset.ndim == 1:
            total_len = int(dataset.shape[0])
            nseq = total_len // seqlen
            if nseq <= 0:
                raise ValueError("Dataset too short for requested seqlen")
            dataset2 = dataset[: nseq * seqlen].reshape(nseq, seqlen)
        else:
            dataset2 = dataset

        nseq = int(dataset2.shape[0])
        use = min(nsamples, nseq)

        if verbose:
            print(f"[hessian] module={module_name}  seqlen={seqlen}  nsamples={use}")

        # Run forward passes (batch size 1)
        for i in range(use):
            batch = dataset2[i : i + 1].to(module.weight.device)
            _ = model(batch, start_pos=0)

        out = acc.get(normalize=normalize, normalize_by=normalize_by)
        return out.H, out.nseq, out.ntokens

    finally:
        acc.close()


@torch.no_grad()
def compute_module_hessian_cached(
    model: torch.nn.Module,
    cache: ActivationCache,
    layer_id: int,
    module_name: str,
    *,
    dst_rank: int = 0,
    per_rank: bool = None,
    normalize: bool = True,
    normalize_by: str = "tokens",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
    batch_size: int = 1,
):
    """Compute Hessian using cached activations - only runs ONE transformer block.

    This is O(1) blocks per sample instead of O(N) blocks, making the overall
    pipeline O(N) instead of O(N^2).

    Args:
        model: Transformer model
        cache: ActivationCache with hidden states at current block boundary
        layer_id: Which transformer block (0-indexed)
        module_name: Full module name (e.g., "layers.0.attention.wq")
        dst_rank: For distributed gathering
        normalize: Return average vs sum
        normalize_by: "seq" or "tokens"
        dtype: Data type for Hessian computation
        verbose: Print progress
        batch_size: Number of samples to process in parallel (default: 1)

    Returns:
        (H, nseq, ntokens) tuple
    """
    if layer_id != cache.current_block_idx:
        raise ValueError(
            f"Cache is at block {cache.current_block_idx}, "
            f"but requested layer_id={layer_id}"
        )

    mods = dict(model.named_modules())
    if module_name not in mods:
        raise KeyError(f"Module '{module_name}' not found in model")

    module = mods[module_name]
    layer = model.layers[layer_id]

    acc = RuntimeHessian(module, dst_rank=dst_rank, per_rank=per_rank, dtype=dtype)

    try:
        if verbose:
            print(f"[hessian-cached] module={module_name}  block={layer_id}  nsamples={cache.nsamples}  batch_size={batch_size}  per_rank={acc._per_rank}")

        # Run forward passes through ONLY this block (not the full model)
        # Process in batches for better GPU utilization
        for i in range(0, cache.nsamples, batch_size):
            h = cache.get_cached_activations_batch(i, batch_size)
            # Run the transformer block - this triggers the hook on the target module
            _ = layer(h, start_pos=0, freqs_cis=cache._freqs_cis, mask=cache._mask)

        # When per_rank=False for RowParallel, only dst_rank has the full Hessian.
        # Broadcast it to all ranks so everyone can use it for quantization.
        if not acc._per_rank and acc._do_gather and acc._world_size > 1:
            if acc._is_main:
                out = acc.get(normalize=normalize, normalize_by=normalize_by)
                H = out.H
            else:
                # Allocate empty H matching full dimension
                H = torch.zeros((acc.n, acc.n), device=module.weight.device, dtype=dtype)
            dist.broadcast(H, src=dst_rank)
            nseq = torch.tensor([acc.nseq], device=module.weight.device)
            ntokens = torch.tensor([acc.ntokens], device=module.weight.device)
            dist.broadcast(nseq, src=dst_rank)
            dist.broadcast(ntokens, src=dst_rank)
            return H, int(nseq.item()), int(ntokens.item())
        else:
            out = acc.get(normalize=normalize, normalize_by=normalize_by)
            return out.H, out.nseq, out.ntokens

    finally:
        acc.close()


# ==============================================================================
# Qronos Statistics Computation
# ==============================================================================

class DualActivationCapture:
    """Capture activations from both unquantized and quantized models."""

    def __init__(
        self,
        module_unquant: torch.nn.Module,
        module_quant: torch.nn.Module,
        dtype: torch.dtype = torch.float64,
    ):
        self.dtype = dtype
        self.X_unquant: Optional[torch.Tensor] = None
        self.X_quant: Optional[torch.Tensor] = None

        # Register hooks
        self._handle_unquant = module_unquant.register_forward_pre_hook(self._hook_unquant)
        self._handle_quant = module_quant.register_forward_pre_hook(self._hook_quant)

    def _hook_unquant(self, _module, inputs):
        X = inputs[0]
        self.X_unquant = X.detach().reshape(-1, X.shape[-1]).to(self.dtype)

    def _hook_quant(self, _module, inputs):
        X = inputs[0]
        self.X_quant = X.detach().reshape(-1, X.shape[-1]).to(self.dtype)

    def close(self):
        if self._handle_unquant is not None:
            self._handle_unquant.remove()
            self._handle_unquant = None
        if self._handle_quant is not None:
            self._handle_quant.remove()
            self._handle_quant = None

    def get_activations(self):
        """Return (X_unquant, X_quant) and clear."""
        X, X_hat = self.X_unquant, self.X_quant
        self.X_unquant = None
        self.X_quant = None
        return X, X_hat


class ResidualCapture:
    """Capture residual stream values (h_in for wo, h_mid for w2) during forward pass."""

    def __init__(
        self,
        layer_unquant: torch.nn.Module,
        layer_quant: torch.nn.Module,
        weight_type: str,  # "wo" or "w2"
        dtype: torch.dtype = torch.float64,
    ):
        self.weight_type = weight_type
        self.dtype = dtype

        # Residual values (skip connection inputs)
        self.R_unquant: Optional[torch.Tensor] = None
        self.R_quant: Optional[torch.Tensor] = None
        # Input to the target layer (wo or w2)
        self.X_hat: Optional[torch.Tensor] = None

        self._handles = []

        if weight_type == "wo":
            # For wo: R is input to block (x), X̂ is input to wo
            # Capture block input as residual
            self._handles.append(
                layer_unquant.register_forward_pre_hook(self._hook_residual_unquant)
            )
            self._handles.append(
                layer_quant.register_forward_pre_hook(self._hook_residual_quant)
            )
            # Capture wo input
            self._handles.append(
                layer_quant.attention.wo.register_forward_pre_hook(self._hook_xhat)
            )
        elif weight_type == "w2":
            # For w2: R is h after attention (before FFN), X̂ is input to w2
            # We need to capture h between attention and FFN
            # Hook on ffn_norm input to get h (the residual)
            self._handles.append(
                layer_unquant.ffn_norm.register_forward_pre_hook(self._hook_residual_unquant)
            )
            self._handles.append(
                layer_quant.ffn_norm.register_forward_pre_hook(self._hook_residual_quant)
            )
            # Capture w2 input
            self._handles.append(
                layer_quant.feed_forward.w2.register_forward_pre_hook(self._hook_xhat)
            )
        else:
            raise ValueError(f"Unsupported weight_type: {weight_type}, must be 'wo' or 'w2'")

    def _hook_residual_unquant(self, _module, inputs):
        x = inputs[0]
        self.R_unquant = x.detach().to(self.dtype)

    def _hook_residual_quant(self, _module, inputs):
        x = inputs[0]
        self.R_quant = x.detach().to(self.dtype)

    def _hook_xhat(self, _module, inputs):
        x = inputs[0]
        self.X_hat = x.detach().reshape(-1, x.shape[-1]).to(self.dtype)

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def get_and_clear(self):
        """Return (R_unquant, R_quant, X_hat) and clear."""
        R_u, R_q, X_h = self.R_unquant, self.R_quant, self.X_hat
        self.R_unquant = None
        self.R_quant = None
        self.X_hat = None
        return R_u, R_q, X_h


@torch.no_grad()
def compute_residual_stats_cached(
    model_unquant: torch.nn.Module,
    model_quant: torch.nn.Module,
    cache_unquant: "ActivationCache",
    cache_quant: "ActivationCache",
    layer_id: int,
    weight_type: str,  # "wo" or "w2"
    *,
    normalize: bool = True,
    normalize_by: str = "tokens",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
    batch_size: int = 1,
    gather_row_parallel: bool = False,
):
    """Compute residual stream compensation statistics using cached activations.

    For wo/w2 layers that output to the residual stream, computes:
    - Σ_{ΔR,X̂} = E[(R - R̂) X̂^T]

    where:
    - R is the residual from unquantized model
    - R̂ is the residual from quantized model
    - X̂ is the input to wo/w2 from the quantized model

    Args:
        model_unquant: Original (unquantized) transformer model
        model_quant: Quantized transformer model
        cache_unquant: Activation cache for unquantized model
        cache_quant: Activation cache for quantized model
        layer_id: Which transformer block (0-indexed)
        weight_type: "wo" or "w2"
        normalize: Whether to normalize by count
        normalize_by: "seq" or "tokens"
        dtype: Data type for computation
        verbose: Print progress
        batch_size: Batch size for processing

    Returns:
        ResidualStats object with Σ_{ΔR,X̂}
    """
    from quant_layerwise.qronos_stats import ResidualStatsAccumulator

    if layer_id != cache_unquant.current_block_idx:
        raise ValueError(
            f"Unquant cache is at block {cache_unquant.current_block_idx}, "
            f"but requested layer_id={layer_id}"
        )
    if layer_id != cache_quant.current_block_idx:
        raise ValueError(
            f"Quant cache is at block {cache_quant.current_block_idx}, "
            f"but requested layer_id={layer_id}"
        )

    layer_unquant = model_unquant.layers[layer_id]
    layer_quant = model_quant.layers[layer_id]

    # Get dimensions
    if weight_type == "wo":
        module = layer_quant.attention.wo
    elif weight_type == "w2":
        module = layer_quant.feed_forward.w2
    else:
        raise ValueError(f"Unsupported weight_type: {weight_type}")

    out_features = module.weight.shape[0]  # output dim (hidden_dim for residual)
    in_features_local = module.weight.shape[1]   # input dim (local for RowParallel)
    if gather_row_parallel and dist is not None and dist.is_initialized() and dist.get_world_size() > 1:
        in_features = in_features_local * dist.get_world_size()
    else:
        in_features = in_features_local
    device = module.weight.device

    # Create accumulator
    acc = ResidualStatsAccumulator(out_features, in_features, device=device, dtype=dtype)

    # Create residual capture hooks
    capture = ResidualCapture(layer_unquant, layer_quant, weight_type, dtype=dtype)

    try:
        if verbose:
            print(f"[residual-stats] weight={weight_type}  block={layer_id}  nsamples={cache_unquant.nsamples}  batch_size={batch_size}  gather_row_parallel={gather_row_parallel}")

        for i in range(0, cache_unquant.nsamples, batch_size):
            # Get cached activations (input to block)
            h_unquant = cache_unquant.get_cached_activations_batch(i, batch_size)
            h_quant = cache_quant.get_cached_activations_batch(i, batch_size)

            # Run both blocks - this triggers the hooks
            _ = layer_unquant(h_unquant, start_pos=0, freqs_cis=cache_unquant._freqs_cis, mask=cache_unquant._mask)
            _ = layer_quant(h_quant, start_pos=0, freqs_cis=cache_quant._freqs_cis, mask=cache_quant._mask)

            # Get captured values
            R_unquant, R_quant, X_hat = capture.get_and_clear()

            if R_unquant is not None and R_quant is not None and X_hat is not None:
                # For RowParallel: gather sharded X_hat to full dimension
                if gather_row_parallel:
                    X_hat = _all_gather_features(X_hat)
                acc.accumulate(R_unquant, R_quant, X_hat)

        stats = acc.get(normalize=normalize, normalize_by=normalize_by)
        if gather_row_parallel:
            stats.broadcast_from_rank0()
        return stats

    finally:
        capture.close()


@torch.no_grad()
def compute_qronos_stats_cached(
    model_unquant: torch.nn.Module,
    model_quant: torch.nn.Module,
    cache_unquant: "ActivationCache",
    cache_quant: "ActivationCache",
    layer_id: int,
    module_name: str,
    *,
    normalize: bool = True,
    normalize_by: str = "tokens",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
    batch_size: int = 1,
    token_weights: Optional[torch.Tensor] = None,
    gather_row_parallel: bool = False,
):
    """Compute Qronos statistics using cached activations.

    Runs both unquantized and quantized transformer blocks and computes:
    - Σ_X̂ = E[X̂ X̂^T]
    - Σ_XX̂ = E[X X̂^T]

    Args:
        model_unquant: Original (unquantized) transformer model
        model_quant: Quantized transformer model (with previously quantized layers applied)
        cache_unquant: Activation cache for unquantized model
        cache_quant: Activation cache for quantized model
        layer_id: Which transformer block (0-indexed)
        module_name: Full module name (e.g., "layers.0.attention.wq")
        normalize: Whether to normalize by count
        normalize_by: "seq" or "tokens"
        dtype: Data type for computation
        verbose: Print progress
        batch_size: Batch size for processing
        token_weights: Optional per-token importance weights, shape (nsamples, seqlen).
                       When provided, computes attention-weighted covariance matrices.

    Returns:
        QronosStats object with Σ_X̂ and Σ_XX̂
    """
    from quant_layerwise.qronos_stats import QronosStatsAccumulator

    if layer_id != cache_unquant.current_block_idx:
        raise ValueError(
            f"Unquant cache is at block {cache_unquant.current_block_idx}, "
            f"but requested layer_id={layer_id}"
        )
    if layer_id != cache_quant.current_block_idx:
        raise ValueError(
            f"Quant cache is at block {cache_quant.current_block_idx}, "
            f"but requested layer_id={layer_id}"
        )
    if cache_unquant.nsamples != cache_quant.nsamples:
        raise ValueError(
            f"Cache sample count mismatch: {cache_unquant.nsamples} vs {cache_quant.nsamples}"
        )

    mods_unquant = dict(model_unquant.named_modules())
    mods_quant = dict(model_quant.named_modules())

    if module_name not in mods_unquant:
        raise KeyError(f"Module '{module_name}' not found in unquant model")
    if module_name not in mods_quant:
        raise KeyError(f"Module '{module_name}' not found in quant model")

    module_unquant = mods_unquant[module_name]
    module_quant = mods_quant[module_name]
    layer_unquant = model_unquant.layers[layer_id]
    layer_quant = model_quant.layers[layer_id]

    # Use weight.shape[1] for local input dim (correct for RowParallel in multi-GPU,
    # where in_features is the full dim but weight is sharded along columns).
    # When gather_row_parallel=True, use full input dim for the accumulator.
    n_features_local = module_unquant.weight.shape[1]
    if gather_row_parallel and dist is not None and dist.is_initialized() and dist.get_world_size() > 1:
        n_features = n_features_local * dist.get_world_size()
    else:
        n_features = n_features_local
    device = module_unquant.weight.device

    # In precomputed mode only one unquant layer is on GPU at a time,
    # so even w2 accumulators (3 × 28672² × 8 = 18GB) fit on 96GB GPUs.
    acc_device = device
    acc = QronosStatsAccumulator(n_features, device=acc_device, dtype=dtype)

    # Create dual capture hooks
    capture = DualActivationCapture(module_unquant, module_quant, dtype=dtype)

    try:
        if verbose:
            print(f"[qronos-cached] module={module_name}  block={layer_id}  nsamples={cache_unquant.nsamples}  batch_size={batch_size}  gather_row_parallel={gather_row_parallel}")

        for i in range(0, cache_unquant.nsamples, batch_size):
            # Get cached activations
            h_unquant = cache_unquant.get_cached_activations_batch(i, batch_size)
            h_quant = cache_quant.get_cached_activations_batch(i, batch_size)

            # Run both blocks - this triggers the hooks
            _ = layer_unquant(h_unquant, start_pos=0, freqs_cis=cache_unquant._freqs_cis, mask=cache_unquant._mask)
            _ = layer_quant(h_quant, start_pos=0, freqs_cis=cache_quant._freqs_cis, mask=cache_quant._mask)

            # Get captured activations and accumulate
            X, X_hat = capture.get_activations()
            if X is not None and X_hat is not None:
                # For RowParallel: gather sharded inputs to full dimension
                if gather_row_parallel:
                    X = _all_gather_features(X)
                    X_hat = _all_gather_features(X_hat)
                w = None
                if token_weights is not None:
                    actual_batch = min(batch_size, cache_unquant.nsamples - i)
                    w = token_weights[i:i+actual_batch].to(X.device)  # (batch, seqlen) on GPU
                acc.accumulate(X, X_hat, weights=w)

        stats = acc.get(normalize=normalize, normalize_by=normalize_by)
        if gather_row_parallel:
            stats.broadcast_from_rank0()
        return stats

    finally:
        capture.close()


# ==============================================================================
# Attention Importance Computation (for attention-weighted QKV calibration)
# ==============================================================================


def _compute_attn_probs_llama(layer, h, cache_unquant):
    """Compute attention probabilities for a Llama-style layer.

    Returns (probs, n_heads) where probs is (bs, n_heads, seqlen, seqlen).
    """
    from parallel.model import apply_rotary_emb, repeat_kv

    attn = layer.attention
    freqs_cis = cache_unquant._freqs_cis
    mask = cache_unquant._mask
    bsz, seqlen, _ = h.shape

    x = layer.attention_norm(h)
    xq = attn.wq(x).view(bsz, seqlen, attn.n_local_heads, attn.head_dim)
    xk = attn.wk(x).view(bsz, seqlen, attn.n_local_kv_heads, attn.head_dim)
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    keys = repeat_kv(xk, attn.n_rep)  # GQA expansion
    xq = xq.transpose(1, 2)           # (bs, n_heads, seqlen, head_dim)
    keys = keys.transpose(1, 2)

    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(attn.head_dim)
    if mask is not None:
        scores = scores + mask
    probs = F.softmax(scores.float(), dim=-1)
    return probs, attn.n_local_heads


def _compute_attn_probs_qwen3(layer, h, seqlen):
    """Compute attention probabilities for a Qwen3-style layer.

    Manually computes Q, K projections, applies HF-style RoPE (cos/sin),
    expands KV heads for GQA, and computes softmax attention scores.
    This avoids relying on output_attentions=True which is unsupported by SDPA.

    Returns (probs, n_heads) where probs is (bs, n_heads, seqlen, seqlen).
    """
    bsz = h.shape[0]
    device = h.device

    attn = layer.attention
    head_dim = attn.head_dim
    n_kv_heads = attn.n_local_kv_heads
    n_heads = attn.wq.weight.shape[0] // head_dim
    n_rep = n_heads // n_kv_heads

    x = layer.attention_norm(h)

    # Q, K projections and reshape to heads
    xq = attn.wq(x).view(bsz, seqlen, n_heads, head_dim).transpose(1, 2)
    xk = attn.wk(x).view(bsz, seqlen, n_kv_heads, head_dim).transpose(1, 2)
    # xq: (bs, n_heads, seqlen, head_dim), xk: (bs, n_kv_heads, seqlen, head_dim)

    # RoPE (HF-style cos/sin)
    position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
    cos, sin = layer._rotary_emb(x, position_ids)
    cos = cos.unsqueeze(1)  # (bs, 1, seqlen, head_dim)
    sin = sin.unsqueeze(1)

    def _rotate_half(t):
        t1 = t[..., : t.shape[-1] // 2]
        t2 = t[..., t.shape[-1] // 2 :]
        return torch.cat((-t2, t1), dim=-1)

    xq = (xq * cos) + (_rotate_half(xq) * sin)
    xk = (xk * cos) + (_rotate_half(xk) * sin)

    # GQA expansion: (bs, n_kv_heads, seqlen, head_dim) -> (bs, n_heads, seqlen, head_dim)
    if n_rep > 1:
        xk = xk.unsqueeze(2).expand(bsz, n_kv_heads, n_rep, seqlen, head_dim)
        xk = xk.reshape(bsz, n_heads, seqlen, head_dim)

    # Attention scores + causal mask + softmax
    scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(head_dim)
    causal_mask = torch.triu(
        torch.full((seqlen, seqlen), float("-inf"), device=device, dtype=scores.dtype),
        diagonal=1,
    )
    scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
    probs = F.softmax(scores.float(), dim=-1)
    return probs, n_heads


@torch.no_grad()
def compute_attention_importance(
    model_unquant: torch.nn.Module,
    cache_unquant: "ActivationCache",
    layer_id: int,
    batch_size: int = 1,
    sum_mode: bool = False,
) -> torch.Tensor:
    """Compute per-token attention importance weights p_j for a layer.

    Default (sum_mode=False):
        p_j = sum_{h,i} attn_prob[h,i,j] / (n_heads * count[j])
        where count[j] = seqlen - j (causal: number of queries that attend to key j).

    Sum mode (sum_mode=True):
        p_j = sum_{h,i} attn_prob[h,i,j] / n_heads
        Skips count[j] normalization so tokens attending more queries get higher weight.

    Returns: (nsamples, seqlen) tensor of p_j values on CPU, dtype float64.
    """
    layer = model_unquant.layers[layer_id]
    seqlen = cache_unquant.seqlen
    is_qwen3 = hasattr(layer, '_hf_layer')

    all_weights = []
    for i in range(0, cache_unquant.nsamples, batch_size):
        h = cache_unquant.get_cached_activations_batch(i, batch_size)

        if is_qwen3:
            probs, n_heads = _compute_attn_probs_qwen3(layer, h, seqlen)
        else:
            probs, n_heads = _compute_attn_probs_llama(layer, h, cache_unquant)

        # probs: (bs, n_heads, seqlen, seqlen)
        col_sum = probs.sum(dim=(1, 2))  # (bs, seqlen)
        if sum_mode:
            # p_j = sum_{h,i} probs[h,i,j] / n_heads
            p_j = col_sum / n_heads
        else:
            # p_j = sum_{h,i} probs[h,i,j] / (n_heads * count[j])
            count_j = torch.arange(seqlen, 0, -1, device=col_sum.device, dtype=torch.float32)
            p_j = col_sum / (n_heads * count_j.unsqueeze(0))
        all_weights.append(p_j.to(torch.float64).cpu())

    return torch.cat(all_weights, dim=0)  # (nsamples, seqlen)
