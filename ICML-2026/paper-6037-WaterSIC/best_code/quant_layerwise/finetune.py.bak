"""End-to-end finetuning of ZSIC continuous parameters (t_vec, g_vec).

Given a saved ZSIC quantized checkpoint (run_dir with manifest.json),
this module builds a finetunable model where integer codes Z are frozen
and only t_vec/g_vec are trainable.  The dequantized weight is:

    W_hat = diag(t) @ (Z * alpha_base) @ diag(g)

which is fully differentiable w.r.t. t and g (no STE needed).

Memory strategy:
  - Z codes (~26 GB for 3-8B) stay on CPU, loaded to GPU per-layer in forward.
  - Gradient checkpointing on every TransformerBlock to keep activation memory O(1 block).
  - KV cache is not allocated (training mode, no autoregressive generation).
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from quant_layerwise.storage import LayerArtifact, RunManifest


# ---------------------------------------------------------------------------
# QuantizedLinear: a differentiable drop-in for nn.Linear
# ---------------------------------------------------------------------------

class QuantizedLinear(nn.Module):
    """Linear layer backed by frozen integer codes + trainable rescalers.

    Forward:
        W_hat = t[:, None] * Z_alpha[None, :] * g[None, :]
        out = x @ W_hat.T   (i.e. F.linear)

    Z_alpha (= Z * alpha_base, precomputed) is a frozen buffer on CPU in bf16.
    t_vec and g_vec are Parameters on GPU.
    """

    def __init__(
        self,
        Z: torch.Tensor,           # (a_live, n_live) int codes
        alpha_base: torch.Tensor,   # (n_live,)
        t_vec: torch.Tensor,        # (a_live,)
        g_vec: torch.Tensor,        # (n_live,)
        *,
        dead_indices: list | None = None,
        n_original: int | None = None,
        dead_row_indices: list | None = None,
        a_original: int | None = None,
        tp_mode: str = "none",      # "none", "column", or "row" for tensor parallel
    ):
        super().__init__()
        # Pre-expand Z_alpha to full dimensions (with zeros at dead positions)
        # so that _dequant never needs boolean/fancy indexing at runtime.
        # This avoids a PyTorch 2.9 CUDA misaligned-address bug triggered by
        # repeated boolean mask assignments on bf16 tensors under gradient
        # checkpointing.
        Z_alpha = Z.float() * alpha_base.float().unsqueeze(0)  # (a_live, n_live) fp32

        has_dead_cols = dead_indices and len(dead_indices) > 0 and n_original is not None
        has_dead_rows = dead_row_indices and len(dead_row_indices) > 0 and a_original is not None

        # Expand dead columns into Z_alpha (zeros at dead col positions)
        if has_dead_cols:
            live_col = torch.ones(n_original, dtype=torch.bool)
            for idx in dead_indices:
                live_col[idx] = False
            Z_full = torch.zeros(Z_alpha.shape[0], n_original, dtype=Z_alpha.dtype)
            Z_full[:, live_col] = Z_alpha
            Z_alpha = Z_full
            # g_vec: expand to full size with zeros at dead positions
            g_full = torch.zeros(n_original, dtype=g_vec.dtype)
            g_full[live_col] = g_vec
            g_vec = g_full
        self.n_original = n_original

        # Expand dead rows into Z_alpha (zeros at dead row positions)
        if has_dead_rows:
            live_row = torch.ones(a_original, dtype=torch.bool)
            for idx in dead_row_indices:
                live_row[idx] = False
            Z_full = torch.zeros(a_original, Z_alpha.shape[1], dtype=Z_alpha.dtype)
            Z_full[live_row, :] = Z_alpha
            Z_alpha = Z_full
            # t_vec: expand to full size with zeros at dead positions
            t_full = torch.zeros(a_original, dtype=t_vec.dtype)
            t_full[live_row] = t_vec
            t_vec = t_full
        self.a_original = a_original

        self.register_buffer("Z_alpha", Z_alpha.to(torch.bfloat16).contiguous(), persistent=False)

        # Trainable continuous parameters — on GPU
        self.t_vec = nn.Parameter(t_vec.float())
        self.g_vec = nn.Parameter(g_vec.float())

        # Tensor parallel mode: "column" for wq/wk/wv/w1/w3, "row" for wo/w2
        self.tp_mode = tp_mode

        # Masks for zeroing dead-position gradients (no grad should flow there)
        if has_dead_rows:
            self.register_buffer("_dead_row_mask", ~live_row, persistent=False)
        else:
            self._dead_row_mask = None
        if has_dead_cols:
            self.register_buffer("_dead_col_mask", ~live_col, persistent=False)
        else:
            self._dead_col_mask = None

    @staticmethod
    def _fp16_ste(x: torch.Tensor) -> torch.Tensor:
        """Round to float16 precision with straight-through estimator.

        Forward: returns x rounded to fp16 precision (in float32 container).
        Backward: gradient passes through as if no rounding happened.
        """
        rounded = x.detach().to(torch.float16).to(torch.float32)
        return x + (rounded - x).detach()

    def _dequant(self, device: torch.device) -> torch.Tensor:
        """Dequantize to full weight on the given device (differentiable w.r.t. t/g).

        Always uses float32 for the matmul to avoid cuBLAS bf16 gemm bugs
        (CUBLAS_STATUS_EXECUTION_FAILED after ~130 gradient checkpoint steps).
        Dead dimensions are already expanded in Z_alpha/t_vec/g_vec (zeros at dead
        positions), so no runtime boolean indexing is needed.

        t_vec/g_vec are rounded to fp16 precision via STE so the optimizer
        adapts to the deployed (fp16-stored) parameter precision.
        """
        Z_alpha = self.Z_alpha.to(device=device, dtype=torch.float32)
        t = self._fp16_ste(self.t_vec)
        g = self._fp16_ste(self.g_vec)
        return t.unsqueeze(1) * Z_alpha * g.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._dequant(x.device)
        # Tensor parallel communication (matches FairScale ColumnParallel/RowParallel)
        if self.tp_mode == "column":
            # ColumnParallel: identity in fwd, all-reduce input grad in bwd
            from fairscale.nn.model_parallel.mappings import copy_to_model_parallel_region
            x = copy_to_model_parallel_region(x)
        if self.training:
            # Float32 matmul for stable gradients during training
            out = F.linear(x.float(), W).to(x.dtype)
        else:
            # bf16 matmul to match standard eval_ppl precision
            out = F.linear(x, W.to(x.dtype))
        if self.tp_mode == "row":
            # RowParallel: all-reduce output in fwd, identity in bwd
            from fairscale.nn.model_parallel.mappings import reduce_from_model_parallel_region
            out = reduce_from_model_parallel_region(out)
        return out


# ---------------------------------------------------------------------------
# Build finetunable model from a ZSIC run directory
# ---------------------------------------------------------------------------

def _load_artifact(run_dir: Path, manifest: RunManifest, module_name: str, rank: int = 0) -> LayerArtifact:
    relpath = manifest.artifact_relpath_for_rank(module_name, rank=rank)
    return LayerArtifact.load(run_dir / relpath, map_location="cpu")


def _delete_kv_caches(model: nn.Module):
    """Delete KV caches from Attention layers to free GPU memory."""
    for layer in model.layers:
        if hasattr(layer.attention, "cache_k"):
            del layer.attention.cache_k
        if hasattr(layer.attention, "cache_v"):
            del layer.attention.cache_v


def _get_tp_mode(module_name: str, world_size: int) -> str:
    """Determine tensor parallel mode from module name.

    ColumnParallel (wq, wk, wv, w1, w3): input replicated, output sharded.
    RowParallel (wo, w2): input sharded, output all-reduced.
    """
    if world_size <= 1:
        return "none"
    weight_type = module_name.split(".")[-1]  # e.g. "wq", "wo", "w2"
    if weight_type in ("wo", "w2"):
        return "row"
    return "column"


def build_finetunable_model(
    model_name: str,
    run_dir: str | Path,
    *,
    device: str = "cuda:0",
    max_seq_len: int = 2048,
    master_port: int = 29500,
) -> tuple[nn.Module, nn.Module, RunManifest, object]:
    """Build student (quantized) and teacher (unquantized) models.

    Returns (student, teacher, manifest, tokenizer).
    Student: QuantizedLinear layers with trainable t_vec/g_vec.
    Teacher: frozen unquantized model for KL distillation.

    Supports both single-GPU (world_size=1) and multi-GPU tensor parallel
    (world_size>1, launched via torchrun).
    """
    from quant_layerwise.pipeline import (
        load_model_and_tokenizer,
        ensure_single_process_distributed,
    )

    run_dir = Path(run_dir)
    manifest = RunManifest.load(run_dir / "manifest.json")
    dist_world_size = manifest.world_size

    if dist_world_size == 1:
        # Single-GPU: set up minimal distributed env
        local_rank = int(device.split(":")[-1]) if ":" in device else 0
        ensure_single_process_distributed(local_rank=local_rank, master_port=master_port)
        dist_rank = 0
    else:
        # Multi-GPU: torchrun handles distributed init
        import os as _os
        local_rank = int(_os.environ.get("LOCAL_RANK", 0))
        dist_rank = int(_os.environ.get("RANK", 0))
        env_ws = int(_os.environ.get("WORLD_SIZE", 1))
        if env_ws != dist_world_size:
            raise RuntimeError(
                f"Manifest world_size={dist_world_size} but WORLD_SIZE env={env_ws}. "
                f"Launch with: torchrun --nproc_per_node={dist_world_size} ..."
            )
        device = f"cuda:{local_rank}"

    # Teacher: unquantized model, frozen
    teacher, tokenizer = load_model_and_tokenizer(
        model_name, local_rank=local_rank, max_seq_len=max_seq_len, device=device,
    )
    _delete_kv_caches(teacher)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    print("[finetune] teacher loaded (frozen)", flush=True)

    # Student: same architecture, will replace linears with QuantizedLinear
    student, _ = load_model_and_tokenizer(
        model_name, local_rank=local_rank, max_seq_len=max_seq_len, device=device,
    )
    _delete_kv_caches(student)

    # Replace quantized linear layers with QuantizedLinear
    n_replaced = 0
    for module_name in manifest.artifacts:
        art = _load_artifact(run_dir, manifest, module_name, rank=dist_rank)

        if art.method.lower() not in ("zsic", "sic"):
            from quant_layerwise.partial_model import apply_layer_artifact
            apply_layer_artifact(student, art)
            continue

        payload = art.payload
        if not payload.get("apply_tgamma", False):
            from quant_layerwise.partial_model import apply_layer_artifact
            apply_layer_artifact(student, art)
            continue

        tp_mode = _get_tp_mode(module_name, dist_world_size)
        ql = QuantizedLinear(
            Z=payload["Z"],
            alpha_base=payload["alpha_base"],
            t_vec=payload["t_vec"],
            g_vec=payload["g_vec"],
            dead_indices=payload.get("dead_indices", None),
            n_original=payload.get("n_original", None),
            dead_row_indices=payload.get("dead_row_indices", None),
            a_original=payload.get("a_original", None),
            tp_mode=tp_mode,
        )

        # Move everything to GPU — Z_alpha (~13 GB for 8B, ~35 GB for 70B/4-way) + params.
        # Keeping Z_alpha on CPU caused 65K+ transient GPU allocations per epoch
        # (252 modules × 2 fwd/recompute under grad checkpoint) which fragmented
        # the CUDA allocator and eventually triggered misaligned-address errors.
        ql.to(device)
        ql.t_vec = nn.Parameter(ql.t_vec.data)
        ql.g_vec = nn.Parameter(ql.g_vec.data)

        _replace_module(student, module_name, ql)
        n_replaced += 1

    z_alpha_gb = sum(
        m.Z_alpha.nbytes for m in student.modules() if isinstance(m, QuantizedLinear)
    ) / 1e9
    print(f"[finetune] replaced {n_replaced} layers with QuantizedLinear "
          f"(Z_alpha on GPU: {z_alpha_gb:.1f} GB)", flush=True)

    # For Qwen3: sync adapter modules into HF layer so forward pass uses them
    _sync_qwen3_adapters(student)

    # Freeze everything except QuantizedLinear t_vec/g_vec
    for name, param in student.named_parameters():
        if "t_vec" in name or "g_vec" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in student.parameters())
    print(f"[finetune] trainable: {n_trainable:,} / {n_total:,} params", flush=True)

    torch.cuda.empty_cache()
    return student, teacher, manifest, tokenizer


def _replace_module(model: nn.Module, dotted_name: str, new_module: nn.Module):
    """Replace model.a.b.c with new_module."""
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


# Mapping from adapter attribute to HF layer attribute
_QWEN3_ADAPTER_TO_HF = {
    "wq": "q_proj", "wk": "k_proj", "wv": "v_proj", "wo": "o_proj",
    "w1": "gate_proj", "w2": "down_proj", "w3": "up_proj",
}


def _sync_qwen3_adapters(model: nn.Module):
    """Sync Qwen3 adapter modules into HF layer internals.

    The adapter exposes attention.wq/wk/wv/wo and feed_forward.w1/w2/w3,
    but the forward pass goes through _hf_layer.self_attn.q_proj etc.
    After replacing adapter attributes with QuantizedLinear, we must
    also update the HF layer references so the forward path uses them.
    """
    for layer in model.layers:
        if not hasattr(layer, '_hf_layer'):
            continue
        hf_layer = layer._hf_layer
        # Attention
        for adapter_name, hf_name in _QWEN3_ADAPTER_TO_HF.items():
            if adapter_name.startswith("w") and adapter_name[1] in "qkvo":
                mod = getattr(layer.attention, adapter_name, None)
                if mod is not None:
                    setattr(hf_layer.self_attn, hf_name, mod)
            else:
                mod = getattr(layer.feed_forward, adapter_name, None)
                if mod is not None:
                    setattr(hf_layer.mlp, hf_name, mod)


# ---------------------------------------------------------------------------
# Training forward pass (bypasses @torch.inference_mode on Transformer.forward)
# ---------------------------------------------------------------------------

def train_forward(
    model: nn.Module,
    tokens: torch.Tensor,
    *,
    use_gradient_checkpointing: bool = True,
) -> torch.Tensor:
    """Training forward pass that bypasses @torch.inference_mode.

    Args:
        model: Transformer model with QuantizedLinear layers.
        tokens: (batch, seqlen) input token ids.
        use_gradient_checkpointing: checkpoint each TransformerBlock.

    Returns:
        logits: (batch, seqlen, vocab_size) in float32.
    """
    device = tokens.device
    _bsz, seqlen = tokens.shape

    h = model.tok_embeddings(tokens)
    freqs_cis = model.freqs_cis.to(device)[:seqlen]

    # Causal mask (explicit, for manual attention path)
    mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1).to(h.dtype)

    for layer in model.layers:
        # Qwen3 layers have their own forward (handles RoPE/mask internally via HF)
        is_qwen3 = hasattr(layer, '_hf_layer')
        if use_gradient_checkpointing:
            if is_qwen3:
                h = grad_checkpoint(
                    _qwen3_block_forward, layer, h,
                    use_reentrant=False,
                )
            else:
                h = grad_checkpoint(
                    _block_forward, layer, h, freqs_cis, mask,
                    use_reentrant=False,
                )
        else:
            if is_qwen3:
                h = _qwen3_block_forward(layer, h)
            else:
                h = _block_forward(layer, h, freqs_cis, mask)

    h = model.norm(h)
    logits = model.output(h).float()
    return logits


def _qwen3_block_forward(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Native forward through a Qwen3 block (avoids HF internals).

    Mirrors _block_forward for Llama but handles Qwen3-specific details:
    QK-norm, HF-style RoPE (cos/sin from rotary_emb module), and GQA.
    This avoids CUDA misaligned-address errors that occur when HF's internal
    attention ops are recomputed under gradient checkpointing.
    """
    bsz, seqlen, _ = x.shape
    device = x.device

    attn = block.attention
    head_dim = attn.head_dim
    n_heads = attn.n_local_heads
    n_kv_heads = attn.n_local_kv_heads
    n_rep = attn.n_rep

    # ----- Attention -----
    xn = block.attention_norm(x)

    xq = attn.wq(xn).view(bsz, seqlen, n_heads, head_dim)
    xk = attn.wk(xn).view(bsz, seqlen, n_kv_heads, head_dim)
    xv = attn.wv(xn).view(bsz, seqlen, n_kv_heads, head_dim)

    # QK-norm (Qwen3-specific: RMSNorm applied per-head before RoPE)
    hf_attn = block._hf_layer.self_attn
    if hasattr(hf_attn, "q_norm") and hf_attn.q_norm is not None:
        xq = hf_attn.q_norm(xq)
        xk = hf_attn.k_norm(xk)

    # RoPE (HF-style cos/sin)
    position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
    cos, sin = block._rotary_emb(x, position_ids)
    # cos/sin: (bs, seqlen, head_dim) → broadcast over heads
    cos = cos.unsqueeze(2)  # (bs, seqlen, 1, head_dim)
    sin = sin.unsqueeze(2)

    xq_r1 = xq[..., : head_dim // 2]
    xq_r2 = xq[..., head_dim // 2 :]
    xq = (xq * cos) + (torch.cat((-xq_r2, xq_r1), dim=-1) * sin)

    xk_r1 = xk[..., : head_dim // 2]
    xk_r2 = xk[..., head_dim // 2 :]
    xk = (xk * cos) + (torch.cat((-xk_r2, xk_r1), dim=-1) * sin)

    # GQA repeat
    if n_rep > 1:
        xk = xk[:, :, :, None, :].expand(bsz, seqlen, n_kv_heads, n_rep, head_dim)
        xk = xk.reshape(bsz, seqlen, n_heads, head_dim)
        xv = xv[:, :, :, None, :].expand(bsz, seqlen, n_kv_heads, n_rep, head_dim)
        xv = xv.reshape(bsz, seqlen, n_heads, head_dim)

    # Manual attention (avoids SDPA kernel issues under grad checkpoint)
    xq = xq.transpose(1, 2)       # (bs, n_heads, seqlen, head_dim)
    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2)

    scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(head_dim)
    # Causal mask
    causal = torch.full((seqlen, seqlen), float("-inf"), device=device)
    causal = torch.triu(causal, diagonal=1).to(scores.dtype)
    scores = scores + causal
    if block.training:
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    else:
        scores = F.softmax(scores, dim=-1)
    attn_out = torch.matmul(scores, xv)
    attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    attn_out = attn.wo(attn_out)

    h = x + attn_out

    # ----- FFN (SwiGLU) -----
    hn = block.ffn_norm(h)
    ff = block.feed_forward
    out = h + ff.w2(F.silu(ff.w1(hn)) * ff.w3(hn))
    return out


def _block_forward(
    block: nn.Module,
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Forward through one TransformerBlock WITHOUT KV cache."""
    attn = block.attention
    ff = block.feed_forward

    # ----- Attention (no KV cache) -----
    xn = block.attention_norm(x)
    bsz, seqlen, _ = xn.shape

    xq = attn.wq(xn)
    xk = attn.wk(xn)
    xv = attn.wv(xn)

    xq = xq.view(bsz, seqlen, attn.n_local_heads, attn.head_dim)
    xk = xk.view(bsz, seqlen, attn.n_local_kv_heads, attn.head_dim)
    xv = xv.view(bsz, seqlen, attn.n_local_kv_heads, attn.head_dim)

    # RoPE
    from parallel.model import apply_rotary_emb, repeat_kv
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    # GQA repeat
    keys = repeat_kv(xk, attn.n_rep)
    values = repeat_kv(xv, attn.n_rep)

    # Manual attention (avoids SDPA/Flash kernel issues under grad checkpoint)
    xq = xq.transpose(1, 2)       # (bs, n_heads, seqlen, head_dim)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)

    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(attn.head_dim)
    if mask is not None:
        scores = scores + mask
    if block.training:
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    else:
        scores = F.softmax(scores, dim=-1)
    attn_out = torch.matmul(scores, values)
    attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    attn_out = attn.wo(attn_out)

    h = x + attn_out

    # ----- FFN -----
    out = h + ff(block.ffn_norm(h))
    return out


# ---------------------------------------------------------------------------
# Teacher hidden-state precomputation
# ---------------------------------------------------------------------------

def _forward_to_hidden(
    model: nn.Module,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Forward through all layers + norm, returning hidden states before output head."""
    device = tokens.device
    _bsz, seqlen = tokens.shape

    h = model.tok_embeddings(tokens)
    freqs_cis = model.freqs_cis.to(device)[:seqlen]

    mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1).to(h.dtype)

    for layer in model.layers:
        is_qwen3 = hasattr(layer, '_hf_layer')
        if is_qwen3:
            h = _qwen3_block_forward(layer, h)
        else:
            h = _block_forward(layer, h, freqs_cis, mask)

    return model.norm(h)


@torch.no_grad()
def _precompute_teacher_hidden(
    teacher: nn.Module,
    data: torch.Tensor,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    """Run teacher forward once on all data, return hidden states before output head.

    The output head (norm + vocab projection) is identical between student and teacher
    (not quantized), so we can apply it cheaply at training time via student.output().

    Returns (nseq, seqlen-1, dim) in bfloat16 on CPU.
    """
    teacher.eval()
    all_hidden = []
    n_seqs = data.shape[0]

    print(f"[finetune] precomputing teacher hidden states for {n_seqs} seqs...", flush=True)
    for i in range(0, n_seqs, batch_size):
        batch = data[i : i + batch_size].to(device)
        inp = batch[:, :-1]
        h = _forward_to_hidden(teacher, inp)
        all_hidden.append(h.to(dtype=torch.bfloat16, device="cpu"))

    result = torch.cat(all_hidden, dim=0)
    gb = result.element_size() * result.numel() / 1e9
    print(f"[finetune] teacher hidden: {result.shape}, {gb:.1f} GB on CPU", flush=True)
    return result


# ---------------------------------------------------------------------------
# Save finetuned t_vec / g_vec back into artifacts
# ---------------------------------------------------------------------------

def save_finetuned(
    model: nn.Module,
    manifest: RunManifest,
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    dist_rank: int = 0,
    dist_world_size: int = 1,
):
    """Save finetuned model with parallel per-rank shard writes.

    Strategy (avoids the 30-min NCCL-barrier timeout caused by a single-threaded
    rank-0 `shutil.copytree` on multi-GB run dirs like 70B's ~17 GB / 2240 shards):

      1. Rank 0 creates the output directory and copies only the *top-level*
         files (manifest.json, eval.json, etc.) — fast, a handful of small files.
      2. All ranks barrier.
      3. Each rank writes the artifacts assigned to it by the manifest:
           - If the module has a corresponding trained `QuantizedLinear`,
             load the source shard, update `t_vec`/`g_vec` from the model,
             and save to output_dir.
           - Otherwise (rare — non-finetuned modules from a partial run),
             straight-copy the source shard.
         File I/O is parallel across ranks, so total wall-time scales ~1/N.
      4. All ranks barrier.

    The old single-rank copytree approach is preserved for the single-GPU
    (dist_world_size == 1) path, where there's no NCCL barrier to time out
    and copytree is the simplest correct implementation.
    """
    import torch.distributed as _dist

    run_dir = Path(run_dir)
    output_dir = Path(output_dir)

    # Collect trained QuantizedLinear modules (used by all ranks below)
    ql_modules: dict[str, QuantizedLinear] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, QuantizedLinear):
            ql_modules[name] = mod

    # ── Rank 0: create output dir + copy top-level files (fast) ─────────
    if dist_rank == 0:
        if output_dir.exists():
            print(f"[finetune] WARNING: output dir exists, overwriting: {output_dir}", flush=True)
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        # Ensure the layers/ subdirs exist before other ranks try to write into them
        (output_dir / "layers").mkdir(exist_ok=True)
        # Copy any non-`layers/` content (manifest.json, eval.json, etc.)
        n_top = 0
        for f in run_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, output_dir / f.name)
                n_top += 1
            elif f.is_dir() and f.name != "layers":
                shutil.copytree(f, output_dir / f.name)
        print(f"[finetune] copied {n_top} top-level files (run_dir → output_dir)", flush=True)

    if dist_world_size > 1 and _dist.is_initialized():
        _dist.barrier()

    # ── Each rank: write ITS shards, updating t_vec/g_vec for finetuned linears ──
    n_updated = 0
    n_copied = 0
    for module_name in manifest.artifacts:
        relpath = manifest.artifact_relpath_for_rank(module_name, rank=dist_rank)
        src_path = run_dir / relpath
        dst_path = output_dir / relpath
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if module_name not in ql_modules:
            # No finetuned linear for this module — straight copy of the source shard
            shutil.copy2(src_path, dst_path)
            n_copied += 1
            continue

        ql = ql_modules[module_name]
        art = LayerArtifact.load(src_path, map_location="cpu")

        # Update t_vec and g_vec with trained values.
        # QuantizedLinear stores expanded t_vec/g_vec (with zeros at dead positions),
        # but artifacts expect live-only sizes. Extract live elements.
        t_save = ql.t_vec.data.cpu()
        g_save = ql.g_vec.data.cpu()
        if ql._dead_row_mask is not None:
            t_save = t_save[~ql._dead_row_mask.cpu()]
        if ql._dead_col_mask is not None:
            g_save = g_save[~ql._dead_col_mask.cpu()]
        art.payload["t_vec"] = t_save.to(torch.float16)
        art.payload["g_vec"] = g_save.to(torch.float16)

        art.save(dst_path)
        n_updated += 1

    if dist_world_size > 1 and _dist.is_initialized():
        _dist.barrier()

    print(
        f"[finetune] rank {dist_rank}: updated {n_updated} + copied {n_copied} "
        f"artifacts → {output_dir}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def finetune(
    student: nn.Module,
    teacher: nn.Module,
    tokenizer,
    manifest: RunManifest,
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda:0",
    seqlen: int = 2048,
    batch_size: int = 8,
    lr: float = 5e-4,
    epochs: int = 10,
    log_interval: int = 10,
    max_steps: int | None = None,
    gradient_checkpointing: bool = True,
    dataset: str = "wikitext2",
    nsamples: int | None = None,
    min_lr: float = 5e-6,
    eval_ppl_each_epoch: bool = False,
    eval_datasets: list[tuple[str, int | None]] | None = None,
    calib_stride: int | None = None,
    dist_rank: int = 0,
    dist_world_size: int = 1,
):
    """Finetune via KL distillation with cosine annealing LR schedule.

    Runs for the specified number of epochs with LR decaying from lr to min_lr.
    Uses all data for training (no validation split).
    """
    from quant_layerwise.data import get_calibration_data, split_dataset

    # Load data
    token_ids = get_calibration_data(
        tokenizer, dataset=dataset, nsamples=nsamples, seqlen=seqlen,
    )
    data = split_dataset(token_ids, seqlen, stride=calib_stride)  # (nseq, seqlen)

    # Shuffle
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(data.shape[0], generator=g)
    data = data[perm]

    # Precompute teacher hidden states (one-time cost, saves ~50% per-step compute)
    teacher_hidden = _precompute_teacher_hidden(teacher, data, device, batch_size)

    # Free teacher GPU memory (output head is shared with student, not quantized)
    teacher.cpu()
    torch.cuda.empty_cache()
    print("[finetune] teacher offloaded to CPU, GPU memory freed", flush=True)
    print(f"[finetune] train: {data.shape[0]} seqs (seqlen={seqlen})", flush=True)

    # Optimizer + CosineAnnealing scheduler
    trainable = [p for p in student.parameters() if p.requires_grad]
    # foreach=False: use simple per-parameter update loop instead of the
    # multi-tensor CUDA kernel.  The foreach/fused kernels in PyTorch 2.9
    # hit a misaligned-address error with the many small t_vec/g_vec params
    # (504 tensors of varying non-power-of-2 sizes).
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0, foreach=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=min_lr,
    )

    # Disable Flash and memory-efficient SDPA backends — they produce
    # CUDA errors (misaligned address / cuBLAS execution failed) during
    # gradient checkpoint recomputation.  Math-mode SDPA is slower but safe.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    student.train()
    step = 0

    for epoch in range(epochs):
        # Reshuffle data each epoch
        train_perm = torch.randperm(data.shape[0])
        data_epoch = data[train_perm]
        hidden_epoch = teacher_hidden[train_perm]

        epoch_loss = 0.0
        epoch_batches = 0

        for i in range(0, data_epoch.shape[0], batch_size):
            batch = data_epoch[i : i + batch_size].to(device)
            if batch.shape[0] == 0:
                continue

            inp = batch[:, :-1]

            # Teacher logits from precomputed hidden (just one matmul)
            with torch.no_grad():
                h_t = hidden_epoch[i : i + batch_size].to(device=device)
                teacher_logits = student.output(h_t.to(student.output.weight.dtype)).float()
                teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

            # Student logits (with grad)
            student_logits = train_forward(
                student, inp, use_gradient_checkpointing=gradient_checkpointing,
            )
            student_log_probs = F.log_softmax(student_logits, dim=-1)

            loss = F.kl_div(
                student_log_probs.view(-1, student_logits.size(-1)),
                teacher_log_probs.view(-1, teacher_logits.size(-1)),
                log_target=True,
                reduction="batchmean",
            )

            del student_logits, teacher_logits, teacher_log_probs

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            del loss, student_log_probs  # free remaining graph refs
            epoch_batches += 1
            step += 1

            if step % log_interval == 0:
                avg = epoch_loss / epoch_batches
                cur_lr = optimizer.param_groups[0]["lr"]
                alloc_gb = torch.cuda.memory_allocated() / 1e9
                reserved_gb = torch.cuda.memory_reserved() / 1e9
                print(
                    f"[finetune] epoch {epoch} step {step} | "
                    f"train KL {avg:.6f} | lr {cur_lr:.2e} | "
                    f"GPU {alloc_gb:.1f}/{reserved_gb:.1f} GB",
                    flush=True,
                )

            if max_steps is not None and step >= max_steps:
                break

        # End-of-epoch summary
        avg_train_kl = epoch_loss / max(epoch_batches, 1)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[finetune] epoch {epoch} end | train KL {avg_train_kl:.6f} | lr {cur_lr:.2e}",
            flush=True,
        )

        # Step scheduler
        scheduler.step()

        # Optionally eval test PPL
        if eval_ppl_each_epoch:
            epoch_ppl = evaluate(student, tokenizer, device=device, seqlen=seqlen, batch_size=batch_size)
            print(f"[finetune] epoch {epoch} end | wikitext2 test PPL: {epoch_ppl:.4f}", flush=True)
            if eval_datasets:
                for eval_ds, eval_ns in eval_datasets:
                    ds_ppl = evaluate(student, tokenizer, device=device, seqlen=seqlen,
                                      batch_size=batch_size, dataset=eval_ds, nsamples=eval_ns)
                    print(f"[finetune] epoch {epoch} end | {eval_ds} test PPL: {ds_ppl:.4f}", flush=True)
            student.train()

        if max_steps is not None and step >= max_steps:
            break

    print(f"[finetune] training done, {step} total steps", flush=True)

    # Eval on WikiText-2 test
    test_ppl = evaluate(student, tokenizer, device=device, seqlen=seqlen, batch_size=batch_size)
    print(f"[finetune] wikitext2 test PPL: {test_ppl:.4f}", flush=True)
    if eval_datasets:
        for eval_ds, eval_ns in eval_datasets:
            ds_ppl = evaluate(student, tokenizer, device=device, seqlen=seqlen,
                              batch_size=batch_size, dataset=eval_ds, nsamples=eval_ns)
            print(f"[finetune] {eval_ds} test PPL: {ds_ppl:.4f}", flush=True)

    # Save
    save_finetuned(student, manifest, run_dir, output_dir,
                    dist_rank=dist_rank, dist_world_size=dist_world_size)
    return step


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    tokenizer,
    *,
    device: str = "cuda:0",
    seqlen: int = 2048,
    batch_size: int = 4,
    dataset: str = "wikitext2",
    nsamples: int | None = None,
) -> float:
    """Evaluate perplexity on a dataset.

    Args:
        dataset: "wikitext2", "c4", "redpajama", "redpajama_sample"
        nsamples: Max sequences to evaluate (None = all available)

    Returns:
        Perplexity (float).
    """
    from quant_layerwise.data import split_dataset

    if dataset == "wikitext2":
        from quant_layerwise.data import get_wikitext2
        token_ids = get_wikitext2(tokenizer, split="test")
    elif dataset == "c4":
        # Standard C4 eval: 256 random windows from validation split (GPTQ/QuIP protocol)
        from quant_layerwise.data import get_c4_val
        n = nsamples if nsamples is not None else 256
        token_ids = get_c4_val(tokenizer, nsamples=n, seqlen=seqlen, seed=0)
    else:
        from quant_layerwise.data import get_calibration_data
        token_ids = get_calibration_data(tokenizer, dataset=dataset, nsamples=nsamples, seqlen=seqlen)

    data = split_dataset(token_ids, seqlen)  # (nseq, seqlen)
    if nsamples is not None and dataset not in ("c4",) and data.shape[0] > nsamples:
        data = data[:nsamples]
    print(f"[eval] {dataset} test: {data.shape[0]} sequences of length {seqlen}", flush=True)

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, data.shape[0], batch_size):
        batch = data[i : i + batch_size].to(device)
        inp = batch[:, :-1]
        target = batch[:, 1:]

        logits = train_forward(model, inp, use_gradient_checkpointing=False)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target.reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += target.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return ppl
