"""Layerwise quantization pipeline.

Goal
----
Quantize a model one layer (weight matrix) at a time.
For each new layer:
  1) run the *partially quantized* model on a calibration set
  2) compute the Hessian/covariance statistic for the target module
  3) quantize that module's weight with GPTQ or ZSIC
  4) save the artifact, apply it to the model
  5) repeat

Artifacts are always saved to disk so you can resume.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

try:
    import torch.distributed as _dist
except Exception:
    _dist = None

from quant_layerwise.bucket import get_bucket_path, model_reg
from quant_layerwise.data import get_calibration_data, split_dataset
from quant_layerwise.hessian_runtime import (
    compute_module_hessian_cached,
    ActivationCache,
)
from quant_layerwise.methods.gptq import GPTQConfig, compress_gptq_wrapper as compress_gptq
from quant_layerwise.methods.zsic import ZSICConfig, compress_zsic, find_dead_dimensions
from quant_layerwise.methods import zsic as _zsic_module  # for _nccl_counter access
from quant_layerwise.partial_model import apply_layer_artifact
from quant_layerwise.storage import LayerArtifact, RunManifest, safe_stem
from quant_layerwise.names import get_hess_name, get_weight_name
from quant_layerwise.rate_control import RateControlConfig, RateController
from quant_layerwise.precompute import (
    PrecomputedUnquantData,
    compute_attention_importance_precomputed,
    resolve_precomputed,
    precompute_wo_in_ref,
    precompute_w2_in_ref,
)


# Per-weight NCCL op counter for pipeline-level collectives.
# Combined with _zsic_module._nccl_counter at the per-weight barrier to detect desyncs.
_pipeline_nccl_counter = 0


def _pipeline_collective(kind: str, numel: int = 0):
    """Increment pipeline NCCL counter for diagnostics."""
    global _pipeline_nccl_counter
    _pipeline_nccl_counter += 1


def _check_nccl_counters_at_barrier(module_name: str):
    """Print NCCL op counters from this rank at the per-weight barrier.

    Each rank prints its zsic + pipeline counter.  Compare across ranks in logs
    to detect which weight introduced a collective-count desync.  No extra NCCL
    ops are issued (just a print), so this cannot itself cause a hang.
    """
    global _pipeline_nccl_counter
    if _dist is None or not _dist.is_available() or not _dist.is_initialized():
        return
    if _dist.get_world_size() <= 1:
        return

    rank = _dist.get_rank()
    zsic_cnt = getattr(_zsic_module, "_nccl_counter", 0)
    total = _pipeline_nccl_counter + zsic_cnt
    print(f"[nccl-cnt] rank={rank} {module_name} zsic={zsic_cnt} pipe={_pipeline_nccl_counter} total={total}",
          flush=True)

    # Reset counters for the next weight
    _pipeline_nccl_counter = 0
    _zsic_module._nccl_counter = 0


def _infer_is_llama2(model_name: str) -> bool:
    return str(model_name).startswith("2-")


def _is_qwen3(model_name: str) -> bool:
    return str(model_name).lower().startswith("qwen3")


def ensure_single_process_distributed(*, local_rank: int, master_port: int = 29500):
    """Allow calling `parallel.start.start(...)` without torchrun, *if* WORLD_SIZE=1.

    This is only safe when:
      * you are running a single process per job
      * your ckpt_dir contains exactly one `.pth` shard

    If your checkpoints are sharded (len(checkpoints)>1), you still need torchrun
    with WORLD_SIZE equal to the shard count.
    """
    # Use direct assignment (not setdefault) to ensure each spawned process
    # gets its own unique port, even if parent process set these vars
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(int(master_port))
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    # IMPORTANT: Always set LOCAL_RANK=0 because each process is rank 0 of its own
    # single-process distributed group. The parallel/start.py silences output for
    # LOCAL_RANK>0, which would hide logs from processes on GPU>0.
    os.environ["LOCAL_RANK"] = "0"
    print(f"[dist] MASTER_PORT={master_port}, GPU={local_rank}", flush=True)


def _load_qwen3(
    model_name: str,
    *,
    local_rank: int = 0,
    max_seq_len: int | None = None,
):
    """Load Qwen3 model from HuggingFace.

    Args:
        model_name: Name of the model (e.g., "qwen3-8B")
        local_rank: GPU device to use
        max_seq_len: Override max sequence length for KV cache

    Returns:
        model, tokenizer (with Llama-compatible interface)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from parallel.qwen3_adapter import Qwen3ToLlamaAdapter, Qwen3TokenizerAdapter

    # Get HuggingFace path from registry
    hf_path = model_reg[model_name]

    # Check for local cache in QUANT_BUCKET
    # Try multiple possible folder names: Qwen3-8B, qwen3-8B, Qwen_Qwen3-8B
    bucket = get_bucket_path()
    local_candidates = [
        bucket / "Qwen3-8B",           # Most common manual download name
        bucket / model_name,            # e.g., qwen3-8B
        bucket / hf_path.replace("/", "_"),  # e.g., Qwen_Qwen3-8B
    ]
    local_path = None
    for candidate in local_candidates:
        if candidate.exists():
            local_path = candidate
            break

    if local_path is not None:
        hf_path = str(local_path)
        print(f"[qwen3] loading from local: {hf_path}", flush=True)
    else:
        print(f"[qwen3] loading from HuggingFace: {hf_path}", flush=True)
        print(f"[qwen3] tip: to cache locally, run: huggingface-cli download {model_reg[model_name]} --local-dir $QUANT_BUCKET/Qwen3-8B", flush=True)

    # Load tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
    tokenizer = Qwen3TokenizerAdapter(hf_tokenizer)

    # Load model
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_path,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{local_rank}",
        trust_remote_code=True,
    )

    # Wrap in adapter
    override_params = {}
    if max_seq_len is not None:
        override_params["max_seq_len"] = max_seq_len

    model = Qwen3ToLlamaAdapter(hf_model, override_params)

    return model, tokenizer


def load_model_and_tokenizer(
    model_name: str,
    *,
    local_rank: int = 0,
    max_seq_len: int | None = None,
    device: str | None = None,
):
    """Load your Transformer + tokenizer using your existing `parallel.start.start`.

    Args:
        model_name: Name of the model to load (e.g., "3.2-1B", "3-8B", "qwen3-8B")
        local_rank: GPU device to use
        max_seq_len: Override max sequence length for KV cache and RoPE.
                     If None, uses model default (typically 2048).
                     Set to 4096 for longer context evaluation.

    Returns:
        model, tokenizer
    """
    # Handle Qwen3 models via HuggingFace
    if _is_qwen3(model_name):
        return _load_qwen3(model_name, local_rank=local_rank, max_seq_len=max_seq_len)

    # Llama models via existing loading code
    from parallel.start import start
    from parallel.config import no_q_config

    ckpt_dir = get_bucket_path() / model_reg[model_name]
    is_llama2 = _infer_is_llama2(model_name)

    override_params = {}
    if max_seq_len is not None:
        override_params["max_seq_len"] = max_seq_len

    kwargs = {}
    if device is not None:
        kwargs["device"] = device
    model, tokenizer = start(str(ckpt_dir), is_llama2, no_q_config, override_params=override_params, **kwargs)
    return model, tokenizer


@dataclass(frozen=True)
class PipelineConfig:
    model_name: str
    method: str  # "gptq" | "zsic"

    # Layer order list of tuples (layer_id, weight) using your naming.
    layers: Sequence[Tuple[int, str]]

    # Calibration
    calib_dataset: str = "redpajama"  # "redpajama" or "wikitext2"
    calib_seed: int = 42  # Seed for RedPajama shuffling
    seqlen: int = 2048
    nsamples: Optional[int] = None  # Max calibration samples (None = use all available)
    calib_stride: Optional[int] = None  # Stride for overlapping calibration windows (None = seqlen, no overlap)
    hessian_batch_size: int = 1  # Batch size for Hessian computation (higher = faster but more memory)
    w2_batch_size: Optional[int] = None  # Override batch size for w2 (large input dim after RowParallel resharding)
    replay_batch_size: Optional[int] = None  # Override batch size for Qronos/residual single-layer replay (default: hessian_batch_size)


    # Output
    run_root: str = "quant_runs"  # relative to QUANT_BUCKET
    run_id: str = ""
    resume: bool = True

    # Method-specific configs
    gptq: Optional[GPTQConfig] = None
    zsic: Optional[ZSICConfig] = None

    # Optional: global rate control (mainly for ZSIC).
    rate_control: Optional[RateControlConfig] = None

    # Skip quantization for specific (layer_id, weight) pairs - store in full precision
    # Format: list of tuples like [(0, "wq"), (0, "wk"), (1, "wq"), (1, "wk")]
    skip_quantize_layers: Sequence[Tuple[int, str]] = ()

    # Zero out specific dimensions in weight matrices after quantization.
    # This is useful for removing outlier dimensions that cause activation explosion
    # but have minimal impact on model performance.
    # For w1/w3: zeros rows (output dimensions)
    # For w2: zeros columns (input dimensions, matching the zeroed w1/w3 outputs)
    # Format: dict mapping "layer_id.weight" -> list of dimension indices
    # Example: {"16.w1": [2271, 1875], "16.w3": [2271, 1875], "16.w2": [2271, 1875]}
    zero_out_rows: Optional[Dict[str, Sequence[int]]] = None

    # Qronos: compute and save Σ_X̂ and Σ_XX̂ statistics for each layer
    # This requires maintaining both unquantized and quantized model copies
    qronos: bool = False

    # Qronos layer range: only apply Qronos targeting (W*) to layers in [min, max)
    # Stats are still computed for all layers. Use None for no limit.
    # Example: qronos_layer_min=14 means only layers 14+ use Qronos targeting
    qronos_layer_min: Optional[int] = None
    qronos_layer_max: Optional[int] = None

    # Collect Qronos stats for diagnostics even when qronos=False
    # This allows plotting activation MSE without applying Qronos targeting
    collect_qronos_stats: bool = False

    # Save Qronos stats (Σ_X, Σ_X̂, Σ_{X,X̂}) to disk as .pkl files
    # Off by default — pass --save_qronos_stats to enable
    save_qronos_stats: bool = False

    # Plot activation MSE at the end of the run (requires collect_qronos_stats or qronos)
    plot_activation_mse: bool = False

    # Residual stream compensation for wo/w2 layers
    # When enabled, modifies the quantization target to account for residual stream error:
    # ŷ = (W Σ_{X,X̂} + Σ_{ΔR,X̂}) (L̂^T)^{-1}  where Σ_{ΔR,X̂} = E[(R - R̂)X̂^T]
    # Requires both unquant and quant models (automatically loads unquant model if needed)
    # Can be used with or without qronos=True
    residual_compensation: bool = False
    # Skip residual compensation on the first N layers (0 = apply to all)
    # e.g., rescomp_skip_prefix=8 means skip layers 0-7, apply to layers 8+
    rescomp_skip_prefix: int = 0

    # Attention-weighted QKV calibration (BOS fix)
    # Weights Qronos covariance matrices by per-token attention importance p_j.
    # Only for QKV layers. WO/FFN stay unweighted. Requires qronos=True.
    attn_weighted_qkv: bool = False
    # Mixing: Sig = (1-eps)*Sig_weighted + eps*Sig_unweighted
    # eps=0: pure weighted, eps=1: pure unweighted (no effect)
    attn_weighted_qkv_eps: float = 0.01
    # Which weights to apply attention weighting to (subset of QKV)
    # Default: all QKV. Can restrict to e.g. ("wk", "wv") for ablation.
    attn_weighted_weights: Sequence[str] = ("wq", "wk", "wv")
    # Joint adaptive eps: find single eps for all wq/wk/wv that minimizes wo_in relMSE
    # via forward pass through the attention layer.
    attn_weighted_adapt_eps_joint: bool = False
    # Adaptive qronos/standard blending for QKV (minimize wo_in relMSE)
    # Mixes Sig_hX = (1-eps)*Sigma_Xhat + eps*Sigma_X (eps=0: pure qronos, eps=1: pure standard)
    qronos_adapt: bool = False
    # Adaptive qronos/standard blending for w1/w3 jointly (minimize w2 input relMSE)
    w1w3_qronos_adapt: bool = False
    # Fraction of calibration samples to use during golden-section search for adaptive eps.
    # 1.0 = use all samples (default). E.g., 0.25 = use 25% of samples during search.
    adapt_search_sample_ratio: float = 1.0
    # Number of golden-section steps for q_eps (qronos blend) in all adapt searches.
    # Default 10. Reduce to 6 for ~40% fewer search evals with minimal quality loss.
    coord_adapt_q_eps_steps: int = 10
    # Number of golden-section steps for a_eps (attnw blend) in coord-adapt.
    # Default 10 matches q_eps steps. Set to 0 to skip a_eps search entirely
    # (uses a_eps=0), which saves ~40 quantize_one_layer calls per attention layer.
    coord_adapt_a_eps_steps: int = 10

    # Precomputed unquantized statistics directory (from run_precompute).
    # When set, hessians are loaded from disk instead of computed on-the-fly,
    # and the unquantized model is loaded to CPU (not GPU) for Qronos/residual
    # layer replay.  This halves peak GPU memory and allows reuse across runs
    # with different rates.
    precomputed_dir: Optional[str] = None


def default_run_id(cfg: PipelineConfig) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    # Include target rate in the naming
    if cfg.gptq is not None:
        rate = cfg.gptq.target_rate
    elif cfg.zsic is not None:
        rate = cfg.zsic.target_rate_bits
    else:
        rate = 0
    base = f"{cfg.model_name}.{cfg.method}.r{rate:.2f}.{ts}"
    return base


def get_run_dir(cfg: PipelineConfig) -> Path:
    root = Path(cfg.run_root)
    if not root.is_absolute():
        root = get_bucket_path() / root
    rid = cfg.run_id or default_run_id(cfg)
    # Multi-GPU: ensure all ranks use the same run_id.
    # At this point dist may not be initialized yet (model loading does that),
    # so we use a file-based approach: rank 0 writes the run_id, others read it.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1 and not cfg.run_id:
        _rid_file = root / cfg.model_name / ".run_id_sync"
        if rank == 0:
            (root / cfg.model_name).mkdir(parents=True, exist_ok=True)
            _rid_file.write_text(rid)
        # Spin-wait for rank 0 to write (max 60s)
        import time as _time
        _t0 = _time.monotonic()
        while not _rid_file.exists():
            if _time.monotonic() - _t0 > 60:
                break
            _time.sleep(0.1)
        if _rid_file.exists():
            rid = _rid_file.read_text().strip()
            # Clean up after all ranks have read
            if rank == world_size - 1:
                try:
                    _rid_file.unlink()
                except Exception:
                    pass
    return root / cfg.model_name / rid


def should_skip_quantize(cfg: PipelineConfig, layer_id: int, weight: str) -> bool:
    """Check if a (layer_id, weight) pair should skip quantization."""
    for skip_layer_id, skip_weight in cfg.skip_quantize_layers:
        if int(skip_layer_id) == int(layer_id) and str(skip_weight).lower() == str(weight).lower():
            return True
    return False


@torch.no_grad()
def create_fullprec_artifact(
    model: torch.nn.Module,
    module_name: str,
    weight_name: str,
) -> LayerArtifact:
    """Create an artifact that stores the weight in full precision (no quantization)."""
    module = dict(model.named_modules())[module_name]
    W = module.weight.detach()

    payload: Dict[str, Any] = {
        "W_full": W.to(torch.float16).cpu(),  # Store in fp16 to save space
        "loss": 0.0,
        "entropy": 16.0,  # 16 bits per param (fp16)
        "rate_overhead": 0.0,
    }

    return LayerArtifact(
        method="fullprec",
        module_name=module_name,
        weight_name=weight_name,
        shape=tuple(int(x) for x in W.shape),
        payload=payload,
    )


@torch.no_grad()
def quantize_one_layer(
    *,
    model: torch.nn.Module,
    module_name: str,
    weight_name: str,
    H: torch.Tensor,
    cfg: PipelineConfig,
    gptq_cfg: Optional[GPTQConfig] = None,
    zsic_cfg: Optional[ZSICConfig] = None,
    # Qronos stats (optional)
    Sig_X: Optional[torch.Tensor] = None,
    Sig_hX: Optional[torch.Tensor] = None,
    Sig_X_hX: Optional[torch.Tensor] = None,
    # Residual compensation (optional, for wo/w2 layers)
    Sig_delta_R_Xhat: Optional[torch.Tensor] = None,
    # Unquantized covariance for dead dim detection (always from unquantized model).
    Sig_X_for_dead: torch.Tensor = None,
    # For multi-GPU ColumnParallel: total rows across all ranks (enables column sync)
    global_nrows: int = None,
    # Search-mode optimizations (coord-adapt)
    fp32_ldlq: bool = False,
    target_precomputed: Optional[torch.Tensor] = None,
) -> LayerArtifact:
    """Quantize exactly one module weight and return the artifact (not yet saved).

    For Qronos mode (when Qronos stats are provided):
    - Sig_X is the unquantized activations covariance E[X X^T]
    - Sig_hX is the quantized activations covariance E[X_hat X_hat^T]
    - Sig_X_hX is the cross-covariance E[X X_hat^T]
    - H is used as Sig_X fallback when Qronos stats are not available

    For residual compensation (for wo/w2 layers):
    - Sig_delta_R_Xhat is E[(R - R̂) X̂^T] where R is the residual stream
    """
    module = dict(model.named_modules())[module_name]

    # Pull current weight (still fp). Work in float32 where possible.
    W0 = module.weight.detach()
    work_dtype = torch.float32 if W0.dtype in (torch.float16, torch.bfloat16) else W0.dtype
    W = W0.to(work_dtype)

    # Convert H to work dtype. When Sig_X is provided, H is only needed as Sig_X_for_dead
    # (it's unused in ZSIC with Qronos stats), so we can skip it if Sig_X_for_dead != H.
    _dead_aliases_h = Sig_X_for_dead is H
    _need_h_work = Sig_X is None or _dead_aliases_h
    H_work = H.to(work_dtype) if _need_h_work else None

    # Qronos stats: convert to work_dtype (float32 for bf16/fp16 models).
    # compress_zsic casts to float64 internally, but the float64→float32→float64
    # roundtrip is intentional: the float32 precision acts as implicit regularization
    # that prevents over-fitting the Qronos covariance targeting, yielding better PPL.
    Sig_X_work = Sig_X.to(work_dtype) if Sig_X is not None else None
    del Sig_X
    Sig_hX_work = Sig_hX.to(work_dtype) if Sig_hX is not None else None
    del Sig_hX
    Sig_X_hX_work = Sig_X_hX.to(work_dtype) if Sig_X_hX is not None else None
    del Sig_X_hX
    Sig_delta_R_Xhat_work = Sig_delta_R_Xhat.to(work_dtype) if Sig_delta_R_Xhat is not None else None
    del Sig_delta_R_Xhat
    # Dead dim detection covariance (always unquantized).
    # Reuse existing tensor when Sig_X_for_dead aliases H or Sig_X (no extra alloc).
    if _dead_aliases_h and H_work is not None:
        Sig_X_for_dead_work = H_work
    elif Sig_X_for_dead is Sig_X_work:
        Sig_X_for_dead_work = Sig_X_work
    else:
        Sig_X_for_dead_work = Sig_X_for_dead.to(work_dtype)
    del Sig_X_for_dead

    Wq_in = W
    Hq_in = H_work

    method = cfg.method.lower()

    if method == "gptq":
        gcfg = gptq_cfg if gptq_cfg is not None else cfg.gptq
        if gcfg is None:
            raise ValueError("PipelineConfig.gptq must be provided for method='gptq'")
        What, loss, rate, frame = compress_gptq(Wq_in, Hq_in, cfg=gcfg, global_nrows=global_nrows)

        payload: Dict[str, Any] = {
            "Qint": frame["Qint"].to(torch.uint8).cpu(),
            "scales": frame["scales"].to(torch.float16).cpu() if frame["scales"] is not None else None,
            "zeros": frame["zeros"].to(torch.float16).cpu() if frame["zeros"] is not None else None,
            "groupsize": int(frame["groupsize"]),
            "blocksize": int(frame["blocksize"]),
            "percdamp": float(frame["percdamp"]),
            "actorder": bool(frame["actorder"]),
            "maxq": int(frame["maxq"]),
            "target_rate": float(gcfg.target_rate),
            "entropy": float(frame["entropy"]),
            "rate_overhead": float(frame["rate_overhead"]),
            "loss": float(loss),
            "relative_mse": float(frame["relative_mse"]),
        }

        art = LayerArtifact(
            method="gptq",
            module_name=module_name,
            weight_name=weight_name,
            shape=tuple(int(x) for x in W0.shape),
            payload=payload,
        )
        return art

    if method in ("zsic", "sic"):
        zcfg = zsic_cfg if zsic_cfg is not None else cfg.zsic
        if zcfg is None:
            raise ValueError("PipelineConfig.zsic must be provided for method='zsic'")

        # Determine dead row/col indices from zero_out_rows
        dead_row_indices = None
        forced_dead_col_indices = None
        if cfg.zero_out_rows:
            # Parse layer_id and weight from module_name ("layers.6.feed_forward.w1" -> 6, "w1")
            _parts = module_name.split(".")
            _layer_id = int(_parts[1])
            _weight = _parts[-1]
            zero_key = f"{_layer_id}.{_weight}"
            dims_to_zero = cfg.zero_out_rows.get(zero_key, [])
            if _weight.lower() == "w2":
                # For w2: collect dead rows from w1 and w3 at the same layer as forced dead columns
                w1_dims = cfg.zero_out_rows.get(f"{_layer_id}.w1", [])
                w3_dims = cfg.zero_out_rows.get(f"{_layer_id}.w3", [])
                combined_dims = set(w1_dims) | set(w3_dims)
                if combined_dims:
                    forced_dead_col_indices = sorted(combined_dims)
            elif _weight.lower() not in ("wo",):
                # For w1/w3/wq/wk/wv: zero rows = dead rows
                if dims_to_zero:
                    dead_row_indices = list(dims_to_zero)

        # Pass Qronos stats and residual compensation if available
        # Sig_X is required: use Hq_in (Hessian = E[XX^T]) as fallback when Sig_X not provided
        What, loss, rate, frame = compress_zsic(
            Wq_in, cfg=zcfg,
            Sig_X=Sig_X_work if Sig_X_work is not None else Hq_in,
            Sig_hX=Sig_hX_work,
            Sig_X_hX=Sig_X_hX_work,
            Sig_delta_R_Xhat=Sig_delta_R_Xhat_work,
            Sig_X_for_dead=Sig_X_for_dead_work,
            dead_row_indices=dead_row_indices,
            forced_dead_col_indices=forced_dead_col_indices,
            global_nrows=global_nrows,
            fp32_ldlq=fp32_ldlq,
            target_precomputed=target_precomputed,
        )

        payload = {
            "Z": frame["Z"].to(torch.int32).cpu(),
            "alpha": frame["alpha"].to(torch.float16).cpu(),
            "alpha_base": frame.get("alpha_base", None).to(torch.float16).cpu() if frame.get("alpha_base", None) is not None else None,
            "zero_point": frame.get("zero_point", None).to(torch.float16).cpu() if frame.get("zero_point", None) is not None else None,
            "apply_tgamma": bool(frame.get("apply_tgamma", False)),
            "t_vec": frame.get("t_vec", None).to(torch.float16).cpu() if frame.get("t_vec", None) is not None else None,
            "g_vec": frame.get("g_vec", None).to(torch.float16).cpu() if frame.get("g_vec", None) is not None else None,
            "sic_variant": str(frame.get("sic_variant", "compress_w2q")),
            "target_rate_bits": float(frame.get("target_rate_bits", zcfg.target_rate_bits)),
            "entropy": float(frame.get("entropy", 0.0)),
            "rate_overhead": float(frame.get("rate_overhead", 0.0)),
            "loss": float(loss),
            "c_param": float(frame.get("c_param", 0.0)),
            "percdamp": float(frame.get("percdamp", 0.0)),
            "hessian_damp_used": float(frame.get("hessian_damp_used", 0.0)),
            "cholesky_tries": int(frame.get("cholesky_tries", 0)),
            "qronos": bool(frame.get("qronos", False)),
            "residual_compensation": bool(frame.get("residual_compensation", False)),
            # Binary search fields (when binary_search=True in config)
            "binary_search_target_used": frame.get("binary_search_target_used", None),
            "binary_search_desired": frame.get("binary_search_desired", None),
            "binary_search_final_diff": frame.get("binary_search_final_diff", None),
            "binary_search_iterations": frame.get("binary_search_iterations", None),
            # Dead dimension handling (live-only tensors, expand at dequantization)
            "dead_indices": frame.get("dead_indices", []),
            "n_original": frame.get("n_original", None),
            "n_live": frame.get("n_live", None),
            "n_dead": frame.get("n_dead", 0),
            # Dead row handling (output dims excluded from quantization)
            "dead_row_indices": frame.get("dead_row_indices", []),
            "a_original": frame.get("a_original", None),
        }

        art = LayerArtifact(
            method="zsic",
            module_name=module_name,
            weight_name=weight_name,
            shape=tuple(int(x) for x in W0.shape),
            payload=payload,
        )
        return art

    raise ValueError(f"Unknown method: {cfg.method!r}")


###############################################################################
# Row→Column parallel resharding for RowParallel quantization equivalence
###############################################################################

def _enter_row_parallel_quant(
    model: torch.nn.Module,
    module_name: str,
    dist_rank: int,
    dist_world_size: int,
) -> Dict[str, Any]:
    """Reshard a RowParallel layer from column-split to row-split for quantization.

    RowParallel layers have weight (a, n_local) where n_local = n_full / ws.
    To match single-GPU quantization, we all-gather the weight to get (a, n_full),
    then each rank takes a row shard (a_local, n_full).  The ColumnParallel-style
    sync (global_nrows + all-reduce) then makes the quantization identical to
    single-GPU.

    Returns a context dict for _exit_row_parallel_quant.
    """
    module = dict(model.named_modules())[module_name]
    W_col = module.weight.data  # (a, n_local)
    a = W_col.shape[0]
    n_local = W_col.shape[1]
    n_full = n_local * dist_world_size

    # All-gather column shards → full weight on all ranks
    shards = [torch.zeros_like(W_col) for _ in range(dist_world_size)]
    _dist.all_gather(shards, W_col.contiguous())
    W_full = torch.cat(shards, dim=1)  # (a, n_full)

    # Row-shard: each rank takes its rows
    a_local = a // dist_world_size
    row_start = dist_rank * a_local
    # Last rank takes remaining rows (handles non-divisible case)
    row_end = a if dist_rank == dist_world_size - 1 else row_start + a_local
    W_row = W_full[row_start:row_end]  # (a_local, n_full)

    # Replace module weight with row shard
    orig_weight = module.weight.data.clone()
    module.weight.data = W_row.to(orig_weight.dtype)

    ctx = {
        "orig_weight": orig_weight,
        "a_full": a,
        "n_full": n_full,
        "n_local": n_local,
        "a_local": row_end - row_start,
        "row_start": row_start,
        "row_end": row_end,
        "col_start": dist_rank * n_local,
        "col_end": (dist_rank + 1) * n_local,
    }
    print(f"[row-parallel] {module_name}: resharded (a={a}, n_local={n_local}) → "
          f"row shard (a_local={row_end - row_start}, n_full={n_full}) on rank {dist_rank}", flush=True)
    return ctx


def _exit_row_parallel_quant_gptq(
    art: "LayerArtifact",
    ctx: Dict[str, Any],
    dist_rank: int,
    dist_world_size: int,
) -> "LayerArtifact":
    """Convert row-sharded GPTQ artifact to column-sharded for RowParallel storage.

    GPTQ artifacts contain:
      - Qint: (a_local, n_full) integer codes
      - scales: (num_groups, a_local) per-row, per-group scales
      - zeros: (num_groups, a_local) per-row, per-group zero points

    After resharding:
      - Qint: (a_full, n_local) — column-sharded
      - scales: (num_local_groups, a_full) — groups for this shard's columns
      - zeros: (num_local_groups, a_full)
    """
    payload = art.payload
    a_full = ctx["a_full"]
    n_local = ctx["n_local"]
    col_start = ctx["col_start"]
    col_end = ctx["col_end"]
    groupsize = int(payload["groupsize"])

    # Validate groupsize alignment with shard boundaries.
    # GPTQ groups are over columns; after column-sharding, each shard's column
    # range must start at a group boundary so dequantization stays correct.
    if groupsize > 0 and col_start % groupsize != 0:
        raise ValueError(
            f"GPTQ RowParallel resharding requires groupsize ({groupsize}) to divide "
            f"col_start ({col_start}). Ensure n_local ({n_local}) is a multiple of groupsize.")

    _gpu_device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")

    def _allgather_rows(t):
        """All-gather tensor along dim=0 (rows)."""
        was_cpu = not t.is_cuda
        if was_cpu:
            t = t.to(_gpu_device)
        max_rows = (a_full + dist_world_size - 1) // dist_world_size
        if t.shape[0] < max_rows:
            pad = torch.zeros(max_rows - t.shape[0], *t.shape[1:],
                              device=t.device, dtype=t.dtype)
            t_padded = torch.cat([t, pad], dim=0)
        else:
            t_padded = t
        shards = [torch.zeros_like(t_padded) for _ in range(dist_world_size)]
        _dist.all_gather(shards, t_padded.contiguous())
        full = torch.cat(shards, dim=0)[:a_full]
        if was_cpu:
            full = full.cpu()
        return full

    # Qint: (a_local, n_full) → (a_full, n_full) → column-shard (a_full, n_local)
    Qint_full = _allgather_rows(payload["Qint"])
    Qint_shard = Qint_full[:, col_start:col_end]

    # scales/zeros: (num_groups, a_local) — the a_local dim is the row dim.
    # All-gather along the row dimension (dim=1) by transposing.
    scales = payload["scales"]
    zeros = payload["zeros"]

    if scales is not None:
        # Transpose to (a_local, num_groups), all-gather rows, transpose back
        scales_full = _allgather_rows(scales.T).T  # (num_groups, a_full)
        zeros_full = _allgather_rows(zeros.T).T    # (num_groups, a_full)

        if groupsize == -1:
            # Single group: covers all columns. Each shard stores the same scale.
            scales_shard = scales_full
            zeros_shard = zeros_full
        else:
            # Select groups that belong to this shard's column range.
            # col_start is guaranteed to be a multiple of groupsize (checked above).
            group_start = col_start // groupsize
            group_end = (col_end + groupsize - 1) // groupsize
            scales_shard = scales_full[group_start:group_end]
            zeros_shard = zeros_full[group_start:group_end]
    else:
        scales_shard = None
        zeros_shard = None

    new_payload = dict(payload)
    new_payload["Qint"] = Qint_shard.cpu()
    new_payload["scales"] = scales_shard.to(torch.float16).cpu() if scales_shard is not None else None
    new_payload["zeros"] = zeros_shard.to(torch.float16).cpu() if zeros_shard is not None else None
    new_payload["groupsize"] = groupsize

    new_art = LayerArtifact(
        method=art.method,
        module_name=art.module_name,
        weight_name=art.weight_name,
        shape=(a_full, n_local),
        payload=new_payload,
    )
    return new_art


def _exit_row_parallel_quant(
    art: "LayerArtifact",
    ctx: Dict[str, Any],
    dist_rank: int,
    dist_world_size: int,
) -> "LayerArtifact":
    """Convert row-sharded artifact to column-sharded for RowParallel storage.

    After ColumnParallel-style quantization of a RowParallel layer, each rank has
    a row-sharded artifact.  This function all-gathers the row-sharded tensors and
    then each rank slices its column shard for storage (compatible with the existing
    per-shard artifact loading code).

    Supports both ZSIC and GPTQ methods.
    """
    payload = art.payload
    method = art.method.lower()

    if method == "gptq":
        return _exit_row_parallel_quant_gptq(art, ctx, dist_rank, dist_world_size)

    if method not in ("zsic", "sic"):
        raise NotImplementedError(
            f"Row→column artifact resharding not implemented for method={method!r}")

    a_full = ctx["a_full"]
    n_full = ctx["n_full"]
    n_local = ctx["n_local"]
    col_start = ctx["col_start"]
    device = payload["Z"].device

    # Verify no dead rows for RowParallel (wo/w2 don't have dead rows)
    dead_row_indices_local = payload.get("dead_row_indices", [])
    if dead_row_indices_local:
        raise NotImplementedError(
            "Dead row handling in RowParallel artifact resharding is not implemented. "
            "This should not occur for wo/w2 layers."
        )

    # Handle dead dimensions: Z and alpha/g_vec are in live-only space.
    # We need to map between global dead indices and local shard dead indices.
    dead_indices_global = payload.get("dead_indices", [])
    n_original = payload.get("n_original", None) or n_full

    if dead_indices_global:
        dead_set = set(dead_indices_global)
        # Build map: global live column → Z column index
        live_cols_global = [c for c in range(n_original) if c not in dead_set]
    else:
        dead_set = set()
        live_cols_global = list(range(n_full))

    # Determine GPU device for all-gather (payload tensors may be on CPU)
    _gpu_device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")

    # ---- All-gather row-sharded tensors → full ----
    def _allgather_rows(t):
        """All-gather tensor along dim=0 (rows). Handles CPU tensors."""
        was_cpu = not t.is_cuda
        if was_cpu:
            t = t.to(_gpu_device)
        # Pad to uniform size for all_gather (handle non-divisible a)
        max_rows = (a_full + dist_world_size - 1) // dist_world_size
        if t.shape[0] < max_rows:
            pad = torch.zeros(max_rows - t.shape[0], *t.shape[1:],
                              device=t.device, dtype=t.dtype)
            t_padded = torch.cat([t, pad], dim=0)
        else:
            t_padded = t
        shards = [torch.zeros_like(t_padded) for _ in range(dist_world_size)]
        _dist.all_gather(shards, t_padded.contiguous())
        full = torch.cat(shards, dim=0)[:a_full]  # trim padding
        if was_cpu:
            full = full.cpu()
        return full

    def _allgather_1d(t):
        """All-gather 1D tensor (per-row vectors like t_vec)."""
        was_cpu = not t.is_cuda
        if was_cpu:
            t = t.to(_gpu_device)
        max_rows = (a_full + dist_world_size - 1) // dist_world_size
        if t.shape[0] < max_rows:
            pad = torch.zeros(max_rows - t.shape[0], device=t.device, dtype=t.dtype)
            t_padded = torch.cat([t, pad], dim=0)
        else:
            t_padded = t
        shards = [torch.zeros_like(t_padded) for _ in range(dist_world_size)]
        _dist.all_gather(shards, t_padded.contiguous())
        full = torch.cat(shards, dim=0)[:a_full]
        if was_cpu:
            full = full.cpu()
        return full

    # Z: (a_local, n_live) → (a_full, n_live) → column-shard
    Z_full = _allgather_rows(payload["Z"])

    # Determine local shard's live/dead columns
    local_dead_indices = []  # Dead indices local to this shard (0-based within n_local)
    local_live_Z_cols = []   # Which Z column each local live column maps to
    for j_local in range(n_local):
        j_global = col_start + j_local
        if j_global in dead_set:
            local_dead_indices.append(j_local)
        else:
            # Find position of j_global in the global live column list
            z_col = live_cols_global.index(j_global)
            local_live_Z_cols.append(z_col)

    n_local_live = len(local_live_Z_cols)
    n_local_dead = len(local_dead_indices)

    # Slice Z to this shard's live columns
    if local_live_Z_cols:
        Z_shard = Z_full[:, local_live_Z_cols]  # (a_full, n_local_live)
    else:
        Z_shard = torch.zeros(a_full, 0, device=device, dtype=Z_full.dtype)

    # alpha and g_vec: per-column (n_live,) — same on all ranks (synced in quantization).
    # Slice to this shard's live columns.
    alpha = payload["alpha"]  # (n_live,)
    alpha_shard = alpha[local_live_Z_cols] if local_live_Z_cols else alpha[:0]

    alpha_base = payload.get("alpha_base", None)
    alpha_base_shard = None
    if alpha_base is not None:
        alpha_base_shard = alpha_base[local_live_Z_cols] if local_live_Z_cols else alpha_base[:0]

    g_vec = payload.get("g_vec", None)
    g_vec_shard = None
    if g_vec is not None:
        g_vec_shard = g_vec[local_live_Z_cols] if local_live_Z_cols else g_vec[:0]

    zero_point = payload.get("zero_point", None)
    zero_point_shard = None
    if zero_point is not None:
        zero_point_shard = zero_point[local_live_Z_cols] if local_live_Z_cols else zero_point[:0]

    # t_vec: per-row (a_local,) → all-gather → (a_full,)
    t_vec = payload.get("t_vec", None)
    t_vec_full = None
    if t_vec is not None:
        t_vec_full = _allgather_1d(t_vec)

    # Build new payload
    new_payload = dict(payload)
    new_payload["Z"] = Z_shard.to(torch.int32).cpu()
    new_payload["alpha"] = alpha_shard.to(torch.float16).cpu()
    new_payload["alpha_base"] = alpha_base_shard.to(torch.float16).cpu() if alpha_base_shard is not None else None
    new_payload["zero_point"] = zero_point_shard.to(torch.float16).cpu() if zero_point_shard is not None else None
    new_payload["t_vec"] = t_vec_full.to(torch.float16).cpu() if t_vec_full is not None else None
    new_payload["g_vec"] = g_vec_shard.to(torch.float16).cpu() if g_vec_shard is not None else None
    new_payload["dead_indices"] = local_dead_indices
    new_payload["n_original"] = n_local
    new_payload["n_live"] = n_local_live
    new_payload["n_dead"] = n_local_dead
    new_payload["dead_row_indices"] = []
    new_payload["a_original"] = a_full

    # Build new artifact with original RowParallel shape
    new_art = LayerArtifact(
        method=art.method,
        module_name=art.module_name,
        weight_name=art.weight_name,
        shape=(a_full, n_local),
        payload=new_payload,
    )
    return new_art


def get_dist_info() -> Tuple[int, int]:
    """Return (rank, world_size) from torch.distributed, or (0, 1) if not initialized.

    Also cross-checks against RANK / WORLD_SIZE env vars set by torchrun to
    catch cases where torch.distributed silently falls back to (0, 1).
    """
    if _dist is not None and _dist.is_available() and _dist.is_initialized():
        rank, world_size = int(_dist.get_rank()), int(_dist.get_world_size())
    else:
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    # Cross-check: torchrun always sets WORLD_SIZE.  If the env says multi-GPU
    # but we resolved to (0, 1), something went wrong.
    env_ws = int(os.environ.get("WORLD_SIZE", 1))
    if env_ws > 1 and world_size == 1:
        raise RuntimeError(
            f"get_dist_info() resolved to world_size=1 but WORLD_SIZE env var is {env_ws}. "
            f"torch.distributed may not be properly initialized."
        )
    return rank, world_size


def _artifact_relpath(module_name: str, *, method: str, rank: int = 0, world_size: int = 1) -> str:
    stem = safe_stem(module_name)
    if world_size > 1:
        return str(Path("layers") / f"{stem}.{method}.shard{rank}of{world_size}.pt")
    return str(Path("layers") / f"{stem}.{method}.pt")


def _save_artifact_and_manifest(
    art: LayerArtifact,
    module_name: str,
    run_dir: Path,
    manifest: RunManifest,
    manifest_path: Path,
    *,
    dist_rank: int,
    dist_world_size: int,
):
    """Save artifact (all ranks) and update manifest (rank 0 only).

    In multi-GPU mode, we barrier BEFORE updating the manifest so that all
    shard files exist on disk before the manifest claims the module is done.
    This prevents a resume bug where a crash between manifest write and
    barrier leaves the manifest with entries whose shard files don't exist.
    """
    # Every rank saves its own shard artifact
    relpath = _artifact_relpath(module_name, method=art.method, rank=dist_rank, world_size=dist_world_size)
    art.save(run_dir / relpath)

    # Barrier so all ranks finish saving before manifest is updated
    if dist_world_size > 1 and _dist is not None and _dist.is_initialized():
        _pipeline_collective("save_barrier")
        _dist.barrier()

    # Update in-memory manifest on all ranks (for skip checks), but only rank 0 writes to disk.
    relpath_r0 = _artifact_relpath(module_name, method=art.method, rank=0, world_size=dist_world_size)
    manifest.add(module_name, relpath_r0)
    if dist_rank == 0:
        manifest.save(manifest_path)


@torch.no_grad()
def _compute_wo_in_rel_mse(
    model: torch.nn.Module,
    model_unquant: torch.nn.Module,
    cache: "ActivationCache",
    cache_unquant: "ActivationCache",
    layer_id: int,
    W_quants: Dict[str, torch.Tensor],
    batch_size: int,
    max_samples: Optional[int] = None,
    *,
    precomputed_ref: Optional[Tuple] = None,
) -> float:
    """Compute relMSE of wo's input between unquantized and quantized wq/wk/wv.

    Reference: unquantized model forward with unquantized cache -> wo_in_ref
    Quantized: quantized model forward with quantized cache (+ new wq/wk/wv) -> wo_in_quant

    Args:
        model: The quantized model (will have W_quants applied temporarily)
        model_unquant: The unquantized model (for reference pass)
        cache: ActivationCache for quantized model at current layer
        cache_unquant: ActivationCache for unquantized model at current layer
        layer_id: Transformer block index
        W_quants: {"wq": tensor, "wk": tensor, "wv": tensor} quantized weights
        batch_size: Batch size for forward passes
        precomputed_ref: Optional (ref_acts, ref_sq_per_batch) from _precompute_wo_in_ref.
            When provided, skips unquantized model forward passes.

    Returns:
        relMSE = sum ||wo_in_quant - wo_in_ref||^2 / sum ||wo_in_ref||^2
    """
    layer_q = model.layers[layer_id]
    layer_u = model_unquant.layers[layer_id] if model_unquant is not None else None
    mods_q = dict(model.named_modules())
    total_mse = 0.0
    total_ref_sq = 0.0

    # Save original weights from quantized model (not yet quantized for wq/wk/wv)
    _W_save = {}
    for w in W_quants:
        _W_save[w] = mods_q[f"layers.{layer_id}.attention.{w}"].weight.data.clone()

    # Set quantized weights
    for w, W_q in W_quants.items():
        mods_q[f"layers.{layer_id}.attention.{w}"].weight.data.copy_(W_q)

    n = cache.nsamples
    if max_samples is not None:
        n = min(n, max_samples)
    _batch_idx = 0
    for i in range(0, n, batch_size):
        if precomputed_ref is not None:
            wo_ref = precomputed_ref[0][_batch_idx].cuda()
        else:
            h_u = cache_unquant.get_cached_activations_batch(i, batch_size)
            captured_ref = []
            hnd = layer_u.attention.wo.register_forward_pre_hook(
                lambda _m, inp, _c=captured_ref: _c.append(inp[0].detach()))
            _ = layer_u(h_u, start_pos=0, freqs_cis=cache_unquant._freqs_cis, mask=cache_unquant._mask)
            hnd.remove()
            wo_ref = captured_ref[0].float()

        h_q = cache.get_cached_activations_batch(i, batch_size)
        captured_q = []
        hnd2 = layer_q.attention.wo.register_forward_pre_hook(
            lambda _m, inp, _c=captured_q: _c.append(inp[0].detach()))
        _ = layer_q(h_q, start_pos=0, freqs_cis=cache._freqs_cis, mask=cache._mask)
        hnd2.remove()
        wo_quant = captured_q[0].float()

        total_mse += (wo_quant - wo_ref).pow(2).sum().item()
        if precomputed_ref is None:
            total_ref_sq += wo_ref.pow(2).sum().item()
        _batch_idx += 1

    if precomputed_ref is not None:
        total_ref_sq = sum(precomputed_ref[1][:_batch_idx])

    # Restore original weights in quantized model
    for w, W_s in _W_save.items():
        mods_q[f"layers.{layer_id}.attention.{w}"].weight.data.copy_(W_s)

    # All-reduce MSE across ranks for multi-GPU correctness
    if _dist is not None and _dist.is_available() and _dist.is_initialized() and _dist.get_world_size() > 1:
        _pipeline_collective("wo_mse_allreduce")
        t = torch.tensor([total_mse, total_ref_sq], device="cuda", dtype=torch.float64)
        _dist.all_reduce(t, op=_dist.ReduceOp.SUM)
        total_mse, total_ref_sq = t[0].item(), t[1].item()

    return total_mse / max(total_ref_sq, 1e-30)


@torch.no_grad()
def _precompute_wo_in_ref(model_unquant, cache_unquant, layer_id, batch_size, max_samples=None):
    """Precompute wo input reference activations from the unquantized model.

    Returns (ref_acts, ref_sq_per_batch) for use as precomputed_ref in _compute_wo_in_rel_mse.
    """
    layer_u = model_unquant.layers[layer_id]
    refs = []
    ref_sq = []
    n = cache_unquant.nsamples
    if max_samples is not None:
        n = min(n, max_samples)
    n_batches = (n + batch_size - 1) // batch_size
    print(f"[precompute] wo_in_ref layer {layer_id}: {n} samples, {n_batches} batches", flush=True)
    for i in range(0, n, batch_size):
        h_u = cache_unquant.get_cached_activations_batch(i, batch_size)
        captured = []
        hnd = layer_u.attention.wo.register_forward_pre_hook(
            lambda _m, inp, _c=captured: _c.append(inp[0].detach()))
        _ = layer_u(h_u, start_pos=0, freqs_cis=cache_unquant._freqs_cis, mask=cache_unquant._mask)
        hnd.remove()
        wo_ref = captured[0].float()
        refs.append(wo_ref.cpu())
        ref_sq.append(wo_ref.pow(2).sum().item())
    return refs, ref_sq



@torch.no_grad()
def _compute_w2_in_rel_mse(
    model, model_unquant, cache, cache_unquant,
    layer_id: int,
    W_w1: torch.Tensor,
    W_w3: torch.Tensor,
    batch_size: int,
    max_samples: Optional[int] = None,
    *,
    precomputed_ref: Optional[Tuple] = None,
) -> float:
    """Compute relMSE of w2's input between unquantized and quantized w1/w3.

    w2's input is the gated hidden state: SiLU(w1·x) ⊙ w3·x.
    We capture it via a forward pre-hook on w2.

    Args:
        model: The quantized model (will have W_w1/W_w3 applied temporarily)
        model_unquant: The unquantized model (for reference pass)
        cache: ActivationCache for quantized model at current layer
        cache_unquant: ActivationCache for unquantized model at current layer
        layer_id: Transformer block index
        W_w1: Candidate quantized w1 weight tensor
        W_w3: Candidate quantized w3 weight tensor
        batch_size: Batch size for forward passes
        max_samples: If set, limit evaluation to this many samples
        precomputed_ref: Optional (ref_acts, ref_sq_per_batch) from _precompute_w2_in_ref.
            When provided, skips unquantized model forward passes.

    Returns:
        relMSE = sum ||w2_in_quant - w2_in_ref||^2 / sum ||w2_in_ref||^2
    """
    layer_q = model.layers[layer_id]
    layer_u = model_unquant.layers[layer_id] if model_unquant is not None else None
    mods_q = dict(model.named_modules())
    total_mse = 0.0
    total_ref_sq = 0.0

    # Save original w1/w3 weights
    w1_mod_name = f"layers.{layer_id}.feed_forward.w1"
    w3_mod_name = f"layers.{layer_id}.feed_forward.w3"
    W_w1_save = mods_q[w1_mod_name].weight.data.clone()
    W_w3_save = mods_q[w3_mod_name].weight.data.clone()

    # Apply candidate w1/w3
    mods_q[w1_mod_name].weight.data.copy_(W_w1)
    mods_q[w3_mod_name].weight.data.copy_(W_w3)

    n = cache.nsamples
    if max_samples is not None:
        n = min(n, max_samples)
    _batch_idx = 0
    for i in range(0, n, batch_size):
        if precomputed_ref is not None:
            w2_ref = precomputed_ref[0][_batch_idx].cuda()
        else:
            # 1) Unquantized model forward -> reference w2 input
            h_u = cache_unquant.get_cached_activations_batch(i, batch_size)
            captured_ref = []
            hnd = layer_u.feed_forward.w2.register_forward_pre_hook(
                lambda _m, inp, _c=captured_ref: _c.append(inp[0].detach()))
            _ = layer_u(h_u, start_pos=0, freqs_cis=cache_unquant._freqs_cis, mask=cache_unquant._mask)
            hnd.remove()
            w2_ref = captured_ref[0].float()

        # 2) Quantized model forward -> quantized w2 input
        h_q = cache.get_cached_activations_batch(i, batch_size)
        captured_q = []
        hnd2 = layer_q.feed_forward.w2.register_forward_pre_hook(
            lambda _m, inp, _c=captured_q: _c.append(inp[0].detach()))
        _ = layer_q(h_q, start_pos=0, freqs_cis=cache._freqs_cis, mask=cache._mask)
        hnd2.remove()
        w2_quant = captured_q[0].float()

        total_mse += (w2_quant - w2_ref).pow(2).sum().item()
        if precomputed_ref is None:
            total_ref_sq += w2_ref.pow(2).sum().item()
        _batch_idx += 1

    if precomputed_ref is not None:
        total_ref_sq = sum(precomputed_ref[1][:_batch_idx])

    # Restore original w1/w3
    mods_q[w1_mod_name].weight.data.copy_(W_w1_save)
    mods_q[w3_mod_name].weight.data.copy_(W_w3_save)

    # All-reduce MSE across ranks for multi-GPU correctness
    if _dist is not None and _dist.is_available() and _dist.is_initialized() and _dist.get_world_size() > 1:
        _pipeline_collective("w2_mse_allreduce")
        t = torch.tensor([total_mse, total_ref_sq], device="cuda", dtype=torch.float64)
        _dist.all_reduce(t, op=_dist.ReduceOp.SUM)
        total_mse, total_ref_sq = t[0].item(), t[1].item()

    return total_mse / max(total_ref_sq, 1e-30)


@torch.no_grad()
def _precompute_w2_in_ref(model_unquant, cache_unquant, layer_id, batch_size, max_samples=None):
    """Precompute w2 input reference activations from the unquantized model.

    Returns (ref_acts, ref_sq_per_batch) for use as precomputed_ref in _compute_w2_in_rel_mse.
    """
    layer_u = model_unquant.layers[layer_id]
    refs = []
    ref_sq = []
    n = cache_unquant.nsamples
    if max_samples is not None:
        n = min(n, max_samples)
    n_batches = (n + batch_size - 1) // batch_size
    print(f"[precompute] w2_in_ref layer {layer_id}: {n} samples, {n_batches} batches", flush=True)
    for i in range(0, n, batch_size):
        h_u = cache_unquant.get_cached_activations_batch(i, batch_size)
        captured = []
        hnd = layer_u.feed_forward.w2.register_forward_pre_hook(
            lambda _m, inp, _c=captured: _c.append(inp[0].detach()))
        _ = layer_u(h_u, start_pos=0, freqs_cis=cache_unquant._freqs_cis, mask=cache_unquant._mask)
        hnd.remove()
        w2_ref = captured[0].float()
        refs.append(w2_ref.cpu())
        ref_sq.append(w2_ref.pow(2).sum().item())
    return refs, ref_sq


def run_pipeline(cfg: PipelineConfig, *, local_rank: int = 0) -> Path:
    """Run a full layerwise quantization job.

    Returns:
        Path to run directory.
    """
    # Distributed info: rank/world_size for tensor-parallel quantization.
    dist_rank, dist_world_size = get_dist_info()
    is_rank0 = dist_rank == 0

    run_dir = get_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifest.json"
    log_path = run_dir / "layer_logs.jsonl"

    # Load or create manifest.
    if cfg.resume and manifest_path.exists():
        manifest = RunManifest.load(manifest_path)
        # Ensure world_size matches current run
        if manifest.world_size != dist_world_size:
            if manifest.world_size == 1 and len(manifest.artifacts) == 0:
                # Empty manifest from legacy run, update world_size
                manifest.world_size = dist_world_size
            else:
                raise RuntimeError(
                    f"Manifest world_size={manifest.world_size} does not match "
                    f"current world_size={dist_world_size}. Cannot resume with "
                    f"different number of GPUs."
                )
    else:
        manifest = RunManifest(
            model_name=str(cfg.model_name),
            method=str(cfg.method),
            run_id=str(run_dir.name),
            config=asdict(cfg),
            artifacts={},
            world_size=dist_world_size,
        )
        manifest.save(manifest_path)

    if dist_world_size > 1 and _is_qwen3(cfg.model_name):
        raise RuntimeError(
            f"Multi-GPU quantization is not supported for Qwen3 models (world_size={dist_world_size}). "
            f"Qwen3 uses HuggingFace loading, not FairScale tensor parallelism."
        )

    if dist_world_size > 1:
        print(f"[pipeline] multi-GPU mode: rank={dist_rank}, world_size={dist_world_size}", flush=True)

    # --- Precomputed unquantized data ---
    # Auto-enable when features requiring the unquant model are active.
    # This avoids loading a second full model copy to GPU (2x memory).
    # Resolved BEFORE main model loading so that run_precompute (if needed)
    # can use the full GPU without competing with the main model.
    need_unquant_model = cfg.qronos or cfg.collect_qronos_stats or cfg.residual_compensation or cfg.attn_weighted_qkv
    _precomputed_dir = cfg.precomputed_dir
    if _precomputed_dir == "none":
        _precomputed_dir = None
        print("[precomputed] explicitly disabled (--precomputed_dir=none)", flush=True)
    elif _precomputed_dir is None and need_unquant_model:
        _precomputed_dir = "auto"
        print("[precomputed] auto-enabling precomputed mode (avoids 2x GPU memory from dual model)", flush=True)

    precomputed: Optional[PrecomputedUnquantData] = None
    if _precomputed_dir:
        if _precomputed_dir == "auto":
            precomputed = resolve_precomputed(
                cfg.model_name,
                calib_dataset=cfg.calib_dataset,
                calib_seed=cfg.calib_seed,
                seqlen=cfg.seqlen,
                calib_stride=cfg.calib_stride,
                nsamples=cfg.nsamples,
                hessian_batch_size=cfg.hessian_batch_size,
                local_rank=local_rank,
            )
        else:
            precomputed = PrecomputedUnquantData(_precomputed_dir)
        print(f"[precomputed] loaded from {precomputed.dir} "
              f"(model={precomputed.model_name}, nsamples={precomputed.nsamples})", flush=True)

    # Load model/tokenizer and apply any already-saved layers (resume).
    print(f"[pipeline] [{cfg.run_id}] loading model...", flush=True)
    model, tokenizer = load_model_and_tokenizer(cfg.model_name, local_rank=local_rank, max_seq_len=cfg.seqlen)
    print(f"[pipeline] [{cfg.run_id}] model loaded", flush=True)

    # Apply zero_out_rows to original weights BEFORE any Hessian computation.
    # This ensures that dead dimension detection for w2 sees the zeroed intermediate dims
    # from w1/w3, allowing automatic detection of corresponding dead columns.
    if cfg.zero_out_rows:
        print(f"[zero_out] applying {len(cfg.zero_out_rows)} zero_out_rows at startup", flush=True)
        modules = dict(model.named_modules())
        for zero_key, dims_to_zero in cfg.zero_out_rows.items():
            if not dims_to_zero:
                continue
            # Parse "layer_id.weight" -> module name
            parts = zero_key.split(".")
            if len(parts) != 2:
                print(f"[zero_out] warning: invalid key format '{zero_key}', expected 'layer_id.weight'", flush=True)
                continue
            layer_id_str, weight = parts
            try:
                layer_id = int(layer_id_str)
            except ValueError:
                print(f"[zero_out] warning: invalid layer_id '{layer_id_str}' in '{zero_key}'", flush=True)
                continue
            module_name = get_hess_name(layer_id, weight)
            if module_name not in modules:
                print(f"[zero_out] warning: module '{module_name}' not found", flush=True)
                continue
            module = modules[module_name]
            with torch.no_grad():
                if weight.lower() == "w2":
                    for col_idx in dims_to_zero:
                        module.weight.data[:, col_idx] = 0
                    print(f"[zero_out] startup: zeroed columns {list(dims_to_zero)} in {module_name}", flush=True)
                else:
                    for row_idx in dims_to_zero:
                        module.weight.data[row_idx, :] = 0
                    print(f"[zero_out] startup: zeroed rows {list(dims_to_zero)} in {module_name}", flush=True)

    # Need second unquantized model when:
    # - Qronos: computes X (original activations) for Σ_X, Σ_XX̂ statistics + applies Qronos targeting
    # - collect_qronos_stats: computes stats for diagnostics only (no Qronos targeting)
    # - residual_compensation: computes Σ_{ΔR,X̂} for wo/w2 layers
    # The main `model` becomes `model_quant` (progressively quantized)
    # Warn if adaptive eps flags are set but --qronos is not enabled.
    # Without qronos, the adapt branches silently fall through to normal quantization.
    if not cfg.qronos:
        _adapt_flags = []
        if cfg.qronos_adapt:
            _adapt_flags.append("--qronos_adapt")
        if cfg.w1w3_qronos_adapt:
            _adapt_flags.append("--w1w3_qronos_adapt")
        if cfg.attn_weighted_adapt_eps_joint:
            _adapt_flags.append("--attn_weighted_adapt_eps_joint")
        if _adapt_flags:
            print(f"[WARNING] {', '.join(_adapt_flags)} set but --qronos is not enabled. "
                  f"Adaptive eps blending requires --qronos; these flags will have no effect.",
                  flush=True)

    model_unquant = None
    if need_unquant_model:
        reasons = []
        if cfg.qronos:
            reasons.append("qronos")
        if cfg.collect_qronos_stats:
            reasons.append("collect_qronos_stats")
        if cfg.residual_compensation:
            reasons.append("residual_compensation")
        if cfg.attn_weighted_qkv:
            reasons.append("attn_weighted_qkv")

        if precomputed is not None:
            # With precomputed data: load unquant model to CPU only (not GPU).
            # Individual layers are moved to GPU on-demand for Qronos/residual replay.
            print(f"[{'+'.join(reasons)}] loading unquantized model to CPU (precomputed mode)", flush=True)
            model_unquant, _ = load_model_and_tokenizer(cfg.model_name, local_rank=local_rank, device="cpu", max_seq_len=cfg.seqlen)
            precomputed.set_model_cpu(model_unquant)
            torch.cuda.empty_cache()
            print("[precomputed] unquant model on CPU for layer replay", flush=True)
        else:
            print(f"[{'+'.join(reasons)}] loading second model copy (unquantized)", flush=True)
            model_unquant, _ = load_model_and_tokenizer(cfg.model_name, local_rank=local_rank, max_seq_len=cfg.seqlen)

        # model_unquant stays frozen (never gets quantized weights applied)
        # But we still apply zero_out_rows so residual compensation sees consistent activations
        if cfg.zero_out_rows:
            print("[zero_out] applying zero_out_rows to unquantized model", flush=True)
            modules_unquant = dict(model_unquant.named_modules())
            for zero_key, dims_to_zero in cfg.zero_out_rows.items():
                if not dims_to_zero:
                    continue
                parts = zero_key.split(".")
                if len(parts) != 2:
                    continue
                layer_id_str, weight = parts
                try:
                    layer_id = int(layer_id_str)
                except ValueError:
                    continue
                module_name = get_hess_name(layer_id, weight)
                if module_name not in modules_unquant:
                    continue
                module = modules_unquant[module_name]
                with torch.no_grad():
                    if weight.lower() == "w2":
                        for col_idx in dims_to_zero:
                            module.weight.data[:, col_idx] = 0
                    else:
                        for row_idx in dims_to_zero:
                            module.weight.data[row_idx, :] = 0

    # Resume: apply existing artifacts to model for correct Hessians.
    # For Qronos: only apply to the quantized model, not the unquantized one.
    # In multi-GPU mode, verify all shard files exist before applying.
    # A crash between rank 0 saving manifest and other ranks saving their shards
    # would leave the manifest claiming a module is done but shard files missing.
    _resume_modules = list(manifest.artifacts.keys())
    if dist_world_size > 1:
        _missing = []
        for module_name in _resume_modules:
            relpath = manifest.artifact_relpath_for_rank(module_name, dist_rank)
            if not (run_dir / relpath).exists():
                _missing.append(module_name)
        if _missing:
            print(f"[resume] WARNING: {len(_missing)} modules in manifest but shard files missing for rank {dist_rank}:", flush=True)
            for m in _missing:
                print(f"  {m}", flush=True)
            # Remove incomplete modules from manifest so they get re-quantized
            for m in _missing:
                manifest.remove(m)
                _resume_modules.remove(m)
            # Rank 0 saves the cleaned manifest
            if dist_rank == 0:
                manifest.save(manifest_path)
            if _dist is not None and _dist.is_initialized():
                _dist.barrier()
            print(f"[resume] removed {len(_missing)} incomplete modules, will re-quantize them", flush=True)

    print(f"[pipeline] [{cfg.run_id}] applying {len(_resume_modules)} existing artifacts...", flush=True)
    for module_name in _resume_modules:
        relpath = manifest.artifact_relpath_for_rank(module_name, dist_rank)
        art = LayerArtifact.load(run_dir / relpath)
        apply_layer_artifact(model, art)

    # Optional rate control (primarily for ZSIC): global budget + regression inversion.
    rate_ctrl: RateController | None = None
    print(f"[debug] cfg.rate_control={cfg.rate_control}", flush=True)
    print(f"[debug] cfg.zsic.binary_search={cfg.zsic.binary_search if cfg.zsic else None}", flush=True)
    if (
        cfg.rate_control is not None
        and bool(cfg.rate_control.enabled)
        and str(cfg.method).lower() in ("zsic", "sic")
    ):
        modules = dict(model.named_modules())
        layer_meta: Dict[str, Dict[str, Any]] = {}
        for layer_id, weight in cfg.layers:
            mod = get_hess_name(layer_id, weight)
            if mod not in modules:
                raise KeyError(f"Module not found in model: {mod}")
            w = modules[mod].weight
            if w.ndim != 2:
                raise ValueError(f"Expected 2D weight for {mod}, got shape {tuple(w.shape)}")
            a, n = int(w.shape[0]), int(w.shape[1])
            layer_meta[mod] = {"numel": int(a * n), "shape": [a, n], "weight": str(weight)}

        existing: Dict[str, Dict[str, Any]] = {}
        for mod in manifest.artifacts:
            # Only use artifacts in this run's target layer list.
            if mod not in layer_meta:
                continue
            # Load this rank's shard artifact (not always rank 0's)
            relpath = manifest.artifact_relpath_for_rank(mod, dist_rank)
            art = LayerArtifact.load(run_dir / relpath, map_location="cpu")
            entropy = float(art.payload.get("entropy", 0.0))
            overhead = float(art.payload.get("rate_overhead", 0.0))
            n_dead = int(art.payload.get("n_dead", 0))
            if n_dead > 0:
                # Account for dead dims: compute rate per original element
                n_original = int(art.payload.get("n_original", art.shape[1]))
                n_live = int(art.payload.get("n_live", n_original))
                a_art = art.shape[0]
                actual = float((entropy * n_live + 16 + 16 * n_live / a_art + 16 * n_dead / a_art) / n_original)
            else:
                actual = float(entropy + overhead)
            # For ZSIC artifacts, this is the input parameter we passed.
            target_x = float(art.payload.get("target_rate_bits", actual))
            existing[mod] = {
                "numel": int(layer_meta[mod]["numel"]),
                "actual_rate": float(actual),
                "target_x": float(target_x),
            }

        # Build set of skipped layer module names for budget adjustment
        skip_layer_modules = set()
        for skip_lid, skip_w in cfg.skip_quantize_layers:
            skip_mod = get_hess_name(int(skip_lid), str(skip_w))
            if skip_mod in layer_meta:
                skip_layer_modules.add(skip_mod)

        rate_ctrl = RateController(
            cfg=cfg.rate_control,
            layer_meta=layer_meta,
            existing=existing,
            skip_layers=skip_layer_modules,
        )
        # Save initial controller state for reproducibility.
        rate_ctrl.save_json(str(run_dir / "rate_control_state.json"))

    # Calibration data. None means stream the entire dataset.
    train_tokens = split_dataset(
        get_calibration_data(
            tokenizer,
            dataset=cfg.calib_dataset,
            nsamples=cfg.nsamples,
            seqlen=cfg.seqlen,
            seed=cfg.calib_seed,
        ),
        cfg.seqlen,
        stride=cfg.calib_stride,
    )
    actual_nsamples = train_tokens.shape[0]
    _stride_str = f", stride={cfg.calib_stride}" if cfg.calib_stride and cfg.calib_stride != cfg.seqlen else ""
    print(f"[calib] [{cfg.run_id}] using {actual_nsamples} calibration samples (seqlen={cfg.seqlen}{_stride_str})", flush=True)

    # Initialize activation cache for O(N) instead of O(N^2) complexity.
    # The cache stores hidden states at transformer block boundaries.
    # Barrier: disk I/O (unquant model load, artifact resume, calibration data)
    # causes variable delays across ranks.  ParallelEmbedding inside
    # ActivationCache does an all_gather, so all ranks must be here together.
    if _dist is not None and _dist.is_initialized() and _dist.get_world_size() > 1:
        _dist.barrier()
    device = next(model.parameters()).device
    cache = ActivationCache(
        model=model,
        dataset=train_tokens,
        seqlen=int(cfg.seqlen),
        nsamples=actual_nsamples,
        device=device,
        # dtype auto-detected from model (e.g., bfloat16)
        batch_size=cfg.hessian_batch_size,  # Resize KV caches if needed
    )
    print(f"[cache] [{cfg.run_id}] initialized with {cache.nsamples} samples, starting at block 0, batch_size={cfg.hessian_batch_size}", flush=True)

    # Validate precomputed data token hash against current calibration tokens.
    if precomputed is not None:
        from quant_layerwise.precompute import compute_token_hash
        _stored_hash = precomputed.meta.get("token_hash")
        if _stored_hash is not None:
            _cur_hash = compute_token_hash(train_tokens, cache.nsamples)
            if _cur_hash != _stored_hash:
                import shutil
                print(f"[precomputed] TOKEN HASH MISMATCH: stored={_stored_hash}, current={_cur_hash}", flush=True)
                print(f"[precomputed] deleting stale data at {precomputed.dir} and regenerating...", flush=True)
                shutil.rmtree(precomputed.dir)
                precomputed = resolve_precomputed(
                    cfg.model_name, calib_dataset=cfg.calib_dataset, calib_seed=cfg.calib_seed,
                    seqlen=cfg.seqlen, calib_stride=cfg.calib_stride, nsamples=cfg.nsamples,
                    hessian_batch_size=cfg.hessian_batch_size, local_rank=local_rank,
                )
            else:
                print(f"[precomputed] token hash OK ({_stored_hash})", flush=True)
        else:
            print(f"[precomputed] WARNING: no token_hash in meta.json — data may be stale. "
                  f"Delete {precomputed.dir} to force regeneration.", flush=True)

    # Second cache for unquantized model when:
    # - Qronos: computes X (original activations) for Σ_X, Σ_XX̂ statistics
    # - collect_qronos_stats: computes stats for diagnostics (no Qronos targeting)
    cache_unquant = None
    if need_unquant_model and model_unquant is not None and precomputed is None:
        # No precomputed data: create live cache for unquantized model on GPU.
        cache_unquant = ActivationCache(
            model=model_unquant,
            dataset=train_tokens,
            seqlen=int(cfg.seqlen),
            nsamples=actual_nsamples,
            device=device,
            batch_size=cfg.hessian_batch_size,
        )
        print(f"[unquant-cache] initialized with {cache_unquant.nsamples} samples")
    elif precomputed is not None and need_unquant_model:
        print("[precomputed] skipping unquant cache (using precomputed block outputs)", flush=True)

    # If resuming, advance cache through already-quantized blocks.
    # Find the first layer_id that needs to be quantized.
    layers_to_process = [(lid, w) for lid, w in cfg.layers if not manifest.has(get_hess_name(lid, w))]
    if layers_to_process:
        first_layer_id = layers_to_process[0][0]
        # Advance cache to the first block we need to process
        while cache.current_block_idx < first_layer_id:
            print(f"[cache] advancing through block {cache.current_block_idx} (resuming)")
            cache.advance_through_block(cache.current_block_idx, batch_size=cfg.hessian_batch_size)
            # Qronos: also advance unquantized cache
            if cache_unquant is not None:
                print(f"[unquant-cache] advancing through block {cache_unquant.current_block_idx} (resuming)")
                cache_unquant.advance_through_block(cache_unquant.current_block_idx, batch_size=cfg.hessian_batch_size)

    # Build a set of (layer_id, weight) pairs to know when we're at the last weight of a block.
    # We advance the cache after processing the last weight in each block.
    layer_weights_map: Dict[int, List[str]] = {}
    for layer_id, weight in cfg.layers:
        if layer_id not in layer_weights_map:
            layer_weights_map[layer_id] = []
        layer_weights_map[layer_id].append(weight)

    # Cache for per-layer attention importance weights (reused across wq/wk/wv within a layer)
    _attn_importance_cache: Dict[int, torch.Tensor] = {}

    # Cache for Qronos stats reuse within a sub-block.
    # wq/wk/wv share the same input (attention_norm output) — their Qronos stats are identical.
    # w1/w3 share the same input (ffn_norm output) — same.
    # Key: (layer_id, group) where group is "attn_qkv" or "ffn_w1w3".
    # This avoids redundant forward passes (~297 per weight × 2-4 weights saved per block).
    _qronos_stats_cache: Dict[tuple, "QronosStats"] = {}
    _qronos_stats_w_cache: Dict[tuple, "QronosStats"] = {}  # attention-weighted variant

    def _qronos_group(w: str):
        """Return the Qronos-reuse group for a weight type, or None if not groupable."""
        if w in ("wq", "wk", "wv"):
            return "attn_qkv"
        if w in ("w1", "w3"):
            return "ffn_w1w3"
        return None  # wo, w2 have unique inputs

    def _get_batch_size(w: str) -> int:
        """Per-weight batch size: w2 uses cfg.w2_batch_size if set, others use cfg.hessian_batch_size."""
        if w == "w2" and cfg.w2_batch_size is not None:
            return cfg.w2_batch_size
        return cfg.hessian_batch_size

    def _get_replay_batch_size(w: str) -> int:
        """Batch size for Qronos/residual single-layer replay (forward pass through full layer).

        Memory-aware: row-parallel weights (wo/w2) run the full unquant layer
        (all heads), needing more memory than column-parallel weights (sharded heads).
        - w2: uses w2_batch_size (smallest, due to huge 28672-dim covariances)
        - wo: uses w2_batch_size or replay_batch_size (full heads but small covariances)
        - wq/wk/wv/w1/w3: uses replay_batch_size or hessian_batch_size (sharded heads)
        """
        if w.lower() == "w2" and cfg.w2_batch_size is not None:
            return cfg.w2_batch_size
        if w.lower() == "wo" and cfg.w2_batch_size is not None:
            # wo has full heads like w2 but smaller covariances — use midpoint
            return min(cfg.w2_batch_size * 2, cfg.replay_batch_size or cfg.hessian_batch_size)
        if cfg.replay_batch_size is not None:
            return cfg.replay_batch_size
        return _get_batch_size(w)

    # Main loop.
    _model_modules = dict(model.named_modules())

    import time as _time
    _t_weight_start = None
    _prev_weight_name = None
    for layer_id, weight in cfg.layers:
        module_name = get_hess_name(layer_id, weight)
        weight_name = get_weight_name(layer_id, weight)

        if _t_weight_start is not None:
            _t_weight_elapsed = _time.monotonic() - _t_weight_start
            print(f"[timer] {_prev_weight_name} total: {_t_weight_elapsed:.1f}s", flush=True)
        _t_weight_start = _time.monotonic()
        _prev_weight_name = module_name

        if dist_world_size > 1:
            _check_nccl_counters_at_barrier(module_name)
            _pipeline_collective("barrier")
            _dist.barrier()

        # Skip layers that don't exist in the model (e.g. layer_end > n_layers)
        if module_name not in _model_modules:
            print(f"[skip] module not in model: {module_name}")
            continue

        if manifest.has(module_name):
            print(f"[skip] already quantized: {module_name}")
            # Check if we need to advance cache through this block
            # (mirrors the skip_quantize path — needed for non-default weight orderings
            # where the last weight in a block could be pre-quantized by an adapt branch)
            weights_in_block = layer_weights_map.get(layer_id, [])
            is_last_weight_in_block = (weight == weights_in_block[-1]) if weights_in_block else False
            if is_last_weight_in_block and cache.current_block_idx == layer_id:
                print(f"[cache] advancing through block {layer_id} (last weight already quantized)", flush=True)
                cache.advance_through_block(layer_id, batch_size=cfg.hessian_batch_size)
                if cache_unquant is not None:
                    cache_unquant.advance_through_block(layer_id, batch_size=cfg.hessian_batch_size)
            continue

        # Check if this layer should skip quantization and use full precision
        if should_skip_quantize(cfg, layer_id, weight):
            print(f"\n[layerwise] SKIPPING quantization for {module_name} (storing full precision)", flush=True)
            art = create_fullprec_artifact(model, module_name, weight_name)

            # Save artifact + manifest
            _save_artifact_and_manifest(art, module_name, run_dir, manifest, manifest_path,
                                        dist_rank=dist_rank, dist_world_size=dist_world_size)

            # Log the skip
            line = {
                "module": module_name,
                "weight": weight_name,
                "shape": list(art.shape),
                "method": "fullprec",
                "nseq": 0,
                "ntokens": 0,
                "actual_rate": 16.0,
                "target_rate_bits_used": None,
                "rate_control": None,
                "payload": {
                    "loss": 0.0,
                    "entropy": 16.0,
                    "rate_overhead": 0.0,
                },
                "ts": time.time(),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(line) + "\n")

            # Check if we need to advance cache through this block
            # (even though we skipped quantization, the cache still needs to advance)
            weights_in_block = layer_weights_map.get(layer_id, [])
            is_last_weight_in_block = (weight == weights_in_block[-1]) if weights_in_block else False
            if is_last_weight_in_block and cache.current_block_idx == layer_id:
                print(f"[cache] advancing through block {layer_id} (last weight skipped)", flush=True)
                cache.advance_through_block(layer_id, batch_size=cfg.hessian_batch_size)
                if cache_unquant is not None:
                    cache_unquant.advance_through_block(layer_id, batch_size=cfg.hessian_batch_size)
            continue

        # For multi-GPU: enable ColumnParallel-style sync for ALL weight types.
        # ColumnParallelLinear (wq/wk/wv/w1/w3): naturally row-sharded, use global_nrows sync.
        # RowParallelLinear (wo/w2): reshard from column-split to row-split before
        # quantization, so the same global_nrows sync gives single-GPU equivalence.
        _is_col_parallel = dist_world_size > 1 and weight.lower() not in ("wo", "w2")
        _is_row_parallel = dist_world_size > 1 and weight.lower() in ("wo", "w2")
        _global_nrows = None
        _rp_ctx = None  # RowParallel resharding context
        if _is_col_parallel:
            _global_nrows = _model_modules[module_name].weight.shape[0] * dist_world_size

        # Per-weight batch size: w2 uses smaller batch to avoid OOM (28672² accumulators).
        _bs = _get_batch_size(weight)
        _rbs = _get_replay_batch_size(weight)  # Smaller batch for layer replay (attention scores OOM on large models)
        print(f"\n[layerwise] [{cfg.run_id}] quantizing {module_name} ({cfg.method}, batch_size={_bs}" + (f", replay_bs={_rbs}" if _rbs != _bs else "") + ")", flush=True)

        # Evict Qronos stats caches when processing non-grouped weights (wo, w2).
        # By this point, all consumers of the cached group stats have been processed.
        # Frees ~4.5 GiB of float64 covariance matrices.
        if _qronos_group(weight) is None:
            for _gname in ("attn_qkv", "ffn_w1w3"):
                _evicted = _qronos_stats_cache.pop((layer_id, _gname), None)
                if _evicted is not None:
                    print(f"[qronos-cache] evicted {_gname} unweighted stats for layer {layer_id}", flush=True)
                _evicted_w = _qronos_stats_w_cache.pop((layer_id, _gname), None)
                if _evicted_w is not None:
                    print(f"[qronos-cache] evicted {_gname} weighted stats for layer {layer_id}", flush=True)
            del _evicted, _evicted_w
            torch.cuda.empty_cache()

        # Resize KV caches when per-weight batch size differs from the global
        # hessian_batch_size. This frees memory for w2 (80 layers of KV caches).
        # Restored before advance_through_block.
        _kv_resized = False
        if _bs != cfg.hessian_batch_size:
            from quant_layerwise.hessian_runtime import _resize_kv_caches
            _resize_kv_caches(model, _bs)
            torch.cuda.empty_cache()
            _kv_resized = True
            print(f"[kv-cache] resized to batch_size={_bs} (saving memory for {weight})", flush=True)

        # Hessian computation: always from the progressively-quantized model's
        # live cache. Using unquant H changes the quantization landscape
        # (different LDLQ → different coord-adapt search → PPL regression).
        H, nseq_used, ntokens_used = compute_module_hessian_cached(
            model,
            cache,
            layer_id,
            module_name,
            per_rank=False if _is_row_parallel else None,  # Gather full Hessian for RowParallel
            normalize=True,
            normalize_by="tokens",  # True E[X X^T] normalization
            dtype=torch.float64,
            verbose=True,
            batch_size=_bs,
        )

        # Compute Qronos stats (Σ_X, Σ_X̂, Σ_XX̂) when:
        # - qronos=True: stats are computed and used for Qronos targeting
        # - collect_qronos_stats=True: stats are computed for diagnostics only (no targeting)
        # Note: residual_compensation does NOT require full Qronos stats - it only needs Σ_{ΔR,X̂}
        qronos_stats = None
        should_compute_stats = (cfg.qronos or cfg.collect_qronos_stats) and model_unquant is not None and (cache_unquant is not None or precomputed is not None)
        if should_compute_stats:
            _qg = _qronos_group(weight)
            _cache_key = (layer_id, _qg) if _qg else None
            if _cache_key and _cache_key in _qronos_stats_cache:
                qronos_stats = _qronos_stats_cache[_cache_key]
                print(f"[qronos-cached] reusing unweighted stats from {_qg} for {module_name}", flush=True)
            elif precomputed is not None:
                # Use the SAME on-the-fly code path as non-precomputed mode,
                # just with a temporary cache built from precomputed block outputs.
                # This ensures bitwise-identical stats regardless of precomputed mode.
                from quant_layerwise.hessian_runtime import compute_qronos_stats_cached as _cqsc_pre
                from quant_layerwise.precompute import _resize_layer_kv as _rlkv
                _h_u_pre = precomputed.load_block_input(layer_id)
                if torch.cuda.is_available() and not _h_u_pre.is_pinned():
                    _h_u_pre = _h_u_pre.pin_memory()
                _tc = type(cache).__new__(type(cache))
                _tc._cached_h = _h_u_pre
                _tc.nsamples = min(_h_u_pre.shape[0], cache.nsamples)
                _tc.seqlen = _h_u_pre.shape[1]
                _tc.device = device
                _tc.dtype = cache.dtype
                _tc.current_block_idx = layer_id
                _tc._freqs_cis = cache._freqs_cis
                _tc._mask = cache._mask
                _lu = model_unquant.layers[layer_id]
                _was_cpu = next(_lu.parameters()).device.type == "cpu"
                if _was_cpu:
                    _lu.to(device)
                    _rlkv(_lu, _rbs)
                qronos_stats = _cqsc_pre(
                    model_unquant=model_unquant, model_quant=model,
                    cache_unquant=_tc, cache_quant=cache,
                    layer_id=layer_id, module_name=module_name,
                    normalize=True, normalize_by="tokens",
                    dtype=torch.float64, verbose=True, batch_size=_rbs,
                    gather_row_parallel=_is_row_parallel,
                )
                if _was_cpu:
                    _lu.to("cpu")
                    torch.cuda.empty_cache()
                del _h_u_pre, _tc
                if _cache_key:
                    _qronos_stats_cache[_cache_key] = qronos_stats
            else:
                from quant_layerwise.hessian_runtime import compute_qronos_stats_cached

                qronos_stats = compute_qronos_stats_cached(
                    model_unquant=model_unquant,
                    model_quant=model,
                    cache_unquant=cache_unquant,
                    cache_quant=cache,
                    layer_id=layer_id,
                    module_name=module_name,
                    normalize=True,
                    normalize_by="tokens",  # True E[X X^T] normalization
                    dtype=torch.float64,
                    verbose=True,
                    batch_size=_rbs,
                    gather_row_parallel=_is_row_parallel,
                )
                if _cache_key:
                    _qronos_stats_cache[_cache_key] = qronos_stats
            # Note: Qronos stats are saved AFTER residual stats computation (see below)
            # so that both can be saved together for wo/w2 layers

        # Unquantized Σ_X for dead dimension detection.
        # Dead feature erasure must use the fully-unquantized covariance (from model_unquant),
        # not H from the progressively-quantized model whose activations reflect errors in
        # previously quantized weights (critical for w2 whose input = SiLU(w1_q·x)⊙w3_q·x).
        # Prefer sigma_x from Qronos stats (same E[XX^T]) when available — this lets us
        # free H before quantize_one_layer, saving 6+ GiB for large layers like w2.
        if should_compute_stats and qronos_stats is not None:
            _sigma_x_for_dead = qronos_stats.sigma_x
        elif precomputed is not None and precomputed.has_hessian(module_name):
            # Load unquantized Σ_X from disk for dead dim detection (H is from quant model).
            _sigma_x_for_dead = precomputed.load_hessian(module_name, device=device)
        elif model_unquant is not None and cache_unquant is not None:
            # Compute unquantized covariance for dead dim detection even when qronos is off
            _sigma_x_for_dead, _, _ = compute_module_hessian_cached(
                model_unquant, cache_unquant, layer_id, module_name,
                per_rank=False if _is_row_parallel else None,
                normalize=True, normalize_by="tokens",
                dtype=torch.float64, verbose=False, batch_size=_bs,
            )
        else:
            # No unquantized model available; H is the best we have
            _sigma_x_for_dead = H

        # Multi-GPU ColumnParallel: broadcast Σ_X for dead-dim detection from
        # rank 0 so that ALL downstream uses (rate control dead_params,
        # compress_zsic) see bitwise-identical covariance.  For ColumnParallel
        # weights the input is replicated, but X.T @ X accumulation over
        # batches can diverge O(1e-10) across GPUs due to CUDA matmul
        # non-determinism.  RowParallel sigma_x is already synced via
        # broadcast_from_rank0() in Qronos stats computation.
        if _is_col_parallel and dist_world_size > 1 and _sigma_x_for_dead is not H:
            _pipeline_collective("sigma_x_dead_broadcast")
            _sxd_gpu = _sigma_x_for_dead.clone().cuda()
            _dist.broadcast(_sxd_gpu, src=0)
            _sigma_x_for_dead = _sxd_gpu.to(_sigma_x_for_dead.device, dtype=_sigma_x_for_dead.dtype)

        # Attention-weighted QKV calibration: mix weighted and unweighted Qronos stats
        is_attn_weighted = (
            cfg.attn_weighted_qkv
            and weight.lower() in [w.lower() for w in cfg.attn_weighted_weights]
            and model_unquant is not None
            and (cache_unquant is not None or precomputed is not None)
        )
        # For adaptive eps search, we keep both weighted/unweighted stats and defer mixing
        _attn_weighted_qronos_stats_uw = None  # unweighted (for adaptive eps evaluation)
        _attn_weighted_qronos_stats_w = None   # weighted (for adaptive eps mixing)

        if is_attn_weighted and should_compute_stats and qronos_stats is not None:
            # Compute attention importance (once per layer, cached across wq/wk/wv)
            if layer_id not in _attn_importance_cache:
                if precomputed is not None and precomputed.has_attention_importance(layer_id):
                    # Load from disk (precomputed with sum_mode=False).
                    _attn_importance_cache[layer_id] = precomputed.load_attention_importance(layer_id)
                elif precomputed is not None:
                    # Replay (not precomputed for this layer).
                    _attn_importance_cache[layer_id] = compute_attention_importance_precomputed(
                        precomputed, cache, layer_id,
                        batch_size=_rbs,
                        sum_mode=False,
                    )
                else:
                    from quant_layerwise.hessian_runtime import compute_attention_importance
                    _attn_importance_cache[layer_id] = compute_attention_importance(
                        model_unquant, cache_unquant, layer_id,
                        batch_size=_rbs,
                        sum_mode=False,
                    )
                p0 = _attn_importance_cache[layer_id][:, 0].mean().item()
                print(f"[attn-weight] layer {layer_id}: p_0={p0:.4f} (BOS attention importance)", flush=True)
            attn_w = _attn_importance_cache[layer_id]

            # Compute weighted Qronos stats (with group caching)
            _qg_w = _qronos_group(weight)
            _cache_key_w = (layer_id, _qg_w) if _qg_w else None
            if _cache_key_w and _cache_key_w in _qronos_stats_w_cache:
                qronos_stats_w = _qronos_stats_w_cache[_cache_key_w]
                print(f"[qronos-cached] reusing weighted stats from {_qg_w} for {module_name}", flush=True)
            elif precomputed is not None:
                # Use on-the-fly path with temp cache from precomputed block outputs
                from quant_layerwise.hessian_runtime import compute_qronos_stats_cached as _cqsc_w
                from quant_layerwise.precompute import _resize_layer_kv as _rlkv_w
                _h_u_w = precomputed.load_block_input(layer_id)
                if torch.cuda.is_available() and not _h_u_w.is_pinned():
                    _h_u_w = _h_u_w.pin_memory()
                _tc_w = type(cache).__new__(type(cache))
                _tc_w._cached_h = _h_u_w
                _tc_w.nsamples = min(_h_u_w.shape[0], cache.nsamples)
                _tc_w.seqlen = _h_u_w.shape[1]
                _tc_w.device = device
                _tc_w.dtype = cache.dtype
                _tc_w.current_block_idx = layer_id
                _tc_w._freqs_cis = cache._freqs_cis
                _tc_w._mask = cache._mask
                _lu_w = model_unquant.layers[layer_id]
                _was_cpu_w = next(_lu_w.parameters()).device.type == "cpu"
                if _was_cpu_w:
                    _lu_w.to(device)
                    _rlkv_w(_lu_w, _rbs)
                qronos_stats_w = _cqsc_w(
                    model_unquant=model_unquant, model_quant=model,
                    cache_unquant=_tc_w, cache_quant=cache,
                    layer_id=layer_id, module_name=module_name,
                    normalize=True, normalize_by="tokens",
                    dtype=torch.float64, verbose=False, batch_size=_rbs,
                    token_weights=attn_w,
                    gather_row_parallel=_is_row_parallel,
                )
                if _was_cpu_w:
                    _lu_w.to("cpu")
                    torch.cuda.empty_cache()
                del _h_u_w, _tc_w
                if _cache_key_w:
                    _qronos_stats_w_cache[_cache_key_w] = qronos_stats_w
            else:
                from quant_layerwise.hessian_runtime import compute_qronos_stats_cached as _compute_qronos_stats_cached
                qronos_stats_w = _compute_qronos_stats_cached(
                    model_unquant=model_unquant,
                    model_quant=model,
                    cache_unquant=cache_unquant,
                    cache_quant=cache,
                    layer_id=layer_id,
                    module_name=module_name,
                    normalize=True,
                    normalize_by="tokens",
                    dtype=torch.float64,
                    verbose=False,
                    batch_size=_rbs,
                    token_weights=attn_w,
                    gather_row_parallel=_is_row_parallel,
                )
                if _cache_key_w:
                    _qronos_stats_w_cache[_cache_key_w] = qronos_stats_w

            if cfg.attn_weighted_adapt_eps_joint:
                # Adaptive: keep both stats, defer mixing to quantization loop below
                _attn_weighted_qronos_stats_uw = qronos_stats
                _attn_weighted_qronos_stats_w = qronos_stats_w
                print(f"[attn-weight] joint eps search for {module_name}", flush=True)
            else:
                # Fixed eps: mix now
                eps = cfg.attn_weighted_qkv_eps
                from quant_layerwise.qronos_stats import QronosStats
                qronos_stats = QronosStats(
                    sigma_x=(1.0 - eps) * qronos_stats_w.sigma_x + eps * qronos_stats.sigma_x,
                    sigma_xhat=(1.0 - eps) * qronos_stats_w.sigma_xhat + eps * qronos_stats.sigma_xhat,
                    sigma_x_xhat=(1.0 - eps) * qronos_stats_w.sigma_x_xhat + eps * qronos_stats.sigma_x_xhat,
                    nseq=qronos_stats.nseq,
                    ntokens=qronos_stats.ntokens,
                )
                del qronos_stats_w
                print(f"[attn-weight] mixed Qronos stats for {module_name} (eps={eps})", flush=True)

        # Optional rate-control: choose a per-layer target_rate_bits to hit a global budget.
        zsic_cfg_layer: ZSICConfig | None = None
        rc_info: Dict[str, Any] | None = None
        target_x_used: float | None = None

        # If Qronos mode is enabled and we have stats, enable it in ZSIC config
        # But only if the layer is in the specified range [qronos_layer_min, qronos_layer_max)
        # and not in the skip list
        qronos_enabled = cfg.qronos and qronos_stats is not None
        if qronos_enabled:
            # Check layer range restrictions
            if cfg.qronos_layer_min is not None and layer_id < cfg.qronos_layer_min:
                qronos_enabled = False
                print(f"[qronos] layer {layer_id} < qronos_layer_min ({cfg.qronos_layer_min}), using standard quantization", flush=True)
            if cfg.qronos_layer_max is not None and layer_id >= cfg.qronos_layer_max:
                qronos_enabled = False
                print(f"[qronos] layer {layer_id} >= qronos_layer_max ({cfg.qronos_layer_max}), using standard quantization", flush=True)
        # Compute residual compensation stats for wo/w2 layers if enabled
        # Requires both unquant and quant models/caches (but not necessarily qronos mode)
        residual_stats = None
        residual_comp_enabled = (
            cfg.residual_compensation
            and weight.lower() in ("wo", "w2")
            and model_unquant is not None
            and (cache_unquant is not None or precomputed is not None)
            and layer_id >= cfg.rescomp_skip_prefix  # Skip first N layers
        )
        if cfg.residual_compensation and layer_id < cfg.rescomp_skip_prefix and weight.lower() in ("wo", "w2"):
            print(f"[residual] layer {layer_id} < rescomp_skip_prefix ({cfg.rescomp_skip_prefix}), skipping residual compensation", flush=True)
        if residual_comp_enabled:
            if precomputed is not None:
                # Use on-the-fly path with temp cache from precomputed block outputs
                from quant_layerwise.hessian_runtime import compute_residual_stats_cached as _crsc_pre
                from quant_layerwise.precompute import _resize_layer_kv as _rlkv2
                _h_u_res = precomputed.load_block_input(layer_id)
                if torch.cuda.is_available() and not _h_u_res.is_pinned():
                    _h_u_res = _h_u_res.pin_memory()
                _tc_r = type(cache).__new__(type(cache))
                _tc_r._cached_h = _h_u_res
                _tc_r.nsamples = min(_h_u_res.shape[0], cache.nsamples)
                _tc_r.seqlen = _h_u_res.shape[1]
                _tc_r.device = device
                _tc_r.dtype = cache.dtype
                _tc_r.current_block_idx = layer_id
                _tc_r._freqs_cis = cache._freqs_cis
                _tc_r._mask = cache._mask
                _lu_r = model_unquant.layers[layer_id]
                _was_cpu_r = next(_lu_r.parameters()).device.type == "cpu"
                if _was_cpu_r:
                    _lu_r.to(device)
                    _rlkv2(_lu_r, _rbs)
                residual_stats = _crsc_pre(
                    model_unquant=model_unquant,
                    model_quant=model,
                    cache_unquant=_tc_r,
                    cache_quant=cache,
                    layer_id=layer_id,
                    weight_type=weight.lower(),
                    normalize=True,
                    normalize_by="tokens",
                    dtype=torch.float64,
                    verbose=True,
                    batch_size=_rbs,
                    gather_row_parallel=_is_row_parallel,
                )
                if _was_cpu_r:
                    _lu_r.to("cpu")
                    torch.cuda.empty_cache()
                del _h_u_res, _tc_r
            else:
                from quant_layerwise.hessian_runtime import compute_residual_stats_cached

                residual_stats = compute_residual_stats_cached(
                    model_unquant=model_unquant,
                    model_quant=model,
                    cache_unquant=cache_unquant,
                    cache_quant=cache,
                    layer_id=layer_id,
                    weight_type=weight.lower(),
                    normalize=True,
                    normalize_by="tokens",
                    dtype=torch.float64,
                    verbose=True,
                    batch_size=_rbs,
                    gather_row_parallel=_is_row_parallel,
                )
            print(f"[residual] computed Σ_{{ΔR,X̂}} for {weight} layer (shape={residual_stats.sigma_delta_r_xhat.shape})", flush=True)

            # Note: with gather_row_parallel=True, the residual stats are computed
            # with full-dimensional X_hat, so no 1/world_size scaling is needed.
            # The ColumnParallel-style row-sharded quantization handles this correctly.

        # Save Qronos stats (and residual stats if available) to pkl file
        if qronos_stats is not None and cfg.save_qronos_stats:
            import pickle
            qronos_dir = run_dir / "qronos_stats"
            qronos_dir.mkdir(parents=True, exist_ok=True)
            qronos_path = qronos_dir / f"{safe_stem(module_name)}.pkl"
            # Save as simple dict: just load with pickle.load() and access keys
            qronos_dict = {
                "Sig_X": qronos_stats.sigma_x.cpu(),      # E[X X^T] - unquantized
                "Sig_hX": qronos_stats.sigma_xhat.cpu(),  # E[X̂ X̂^T] - quantized
                "Sig_X_hX": qronos_stats.sigma_x_xhat.cpu(),  # E[X X̂^T] - cross
                "module_name": module_name,
                "layer_id": layer_id,
            }
            # Include residual stats for wo/w2 layers when residual_compensation is enabled
            if residual_stats is not None:
                qronos_dict["Sig_delta_R_Xhat"] = residual_stats.sigma_delta_r_xhat.cpu()
                print("[qronos] including Σ_{ΔR,X̂} in stats (residual_compensation enabled)", flush=True)
            with open(qronos_path, "wb") as f:
                pickle.dump(qronos_dict, f)
            print(f"[qronos] saved stats to {qronos_path}", flush=True)

        # Determine if we need Qronos mode in ZSIC config
        # Only enable when qronos_enabled (full Qronos targeting)
        # Residual compensation can work without Qronos mode (simplified formula)
        use_qronos_mode = qronos_enabled

        # Pre-detect dead dims for rate controller budget adjustment.
        # Dead-dim savings are redistributed: budget is divided by
        # (current alive params + all remaining future params).
        _dead_params_for_rc = 0
        if rate_ctrl is not None and cfg.zsic is not None:
            # Use unquantized Σ_X for dead dim detection when available, else fall back to H
            _dead_mask = find_dead_dimensions(_sigma_x_for_dead, threshold_ratio=cfg.zsic.dead_dim_threshold)
            _n_dead_pre = int(_dead_mask.sum())
            if _n_dead_pre > 0:
                _a_pre = dict(model.named_modules())[module_name].weight.shape[0]
                _dead_params_for_rc = _a_pre * _n_dead_pre
                print(f"[rate] {module_name}: {_n_dead_pre} dead dims -> {_dead_params_for_rc} dead params excluded from budget", flush=True)

        if rate_ctrl is not None:
            if cfg.zsic is None:
                raise ValueError("rate_control enabled but cfg.zsic is None")
            target_x_used, rc_info = rate_ctrl.suggest_target_x(module_name, dead_params=_dead_params_for_rc)

            # Use the target rate from budget controller
            # Binary search (if enabled) will find the right internal target to achieve this rate
            zsic_cfg_layer = replace(
                cfg.zsic,
                target_rate_bits=float(target_x_used),
                qronos=use_qronos_mode,
                residual_compensation=residual_comp_enabled,
                rate_control_active=True,
            )
            print(
                "[rate] wtype={} target={:.4f} remaining_budget={:.2f}".format(
                    rc_info.get("weight_type", "?"),
                    float(target_x_used),
                    float(rc_info.get("remaining_budget_bits", 0)),
                ),
                flush=True,
            )
        elif use_qronos_mode and cfg.zsic is not None:
            # Enable qronos mode in config (for Qronos targeting)
            zsic_cfg_layer = replace(
                cfg.zsic,
                qronos=True,
                residual_compensation=residual_comp_enabled,
            )
        elif residual_comp_enabled and cfg.zsic is not None:
            # Residual compensation without Qronos (simplified formula)
            zsic_cfg_layer = replace(
                cfg.zsic,
                qronos=False,
                residual_compensation=True,
            )

        # RowParallel resharding: convert column-split weight to row-split
        # so that ColumnParallel-style sync gives single-GPU equivalent quantization.
        if _is_row_parallel:
            _rp_ctx = _enter_row_parallel_quant(model, module_name, dist_rank, dist_world_size)
            _global_nrows = _rp_ctx["a_full"]
            # Row-slice Sig_delta_R_Xhat to match the row-sharded weight (a_local, n_full)
            if residual_comp_enabled and residual_stats is not None:
                _rs, _re = _rp_ctx["row_start"], _rp_ctx["row_end"]
                residual_stats.sigma_delta_r_xhat = residual_stats.sigma_delta_r_xhat[_rs:_re]

        # Quantize layer and build artifact.
        # Pass Qronos stats only when qronos_enabled (actual Qronos targeting)
        # Residual compensation works independently - just needs Σ_{ΔR,X̂}
        if (cfg.qronos_adapt and cfg.attn_weighted_adapt_eps_joint
              and _attn_weighted_qronos_stats_uw is not None
              and weight == "wq" and qronos_enabled):
            # ============================================================
            # Combined coordinate descent: jointly optimize qronos_eps and
            # attnw_eps for wq/wk/wv via two sequential golden-section
            # searches minimizing wo_in relMSE.
            #   Stage 1 (qronos_eps): blend quantized↔standard within each
            #       weighting scheme.
            #   Stage 2 (attnw_eps): blend weighted↔unweighted on the
            #       denoised Hessian from Stage 1.
            # Corner cases:
            #   (q=0,a=0) pure weighted qronos
            #   (q=0,a=1) pure unweighted qronos
            #   (q=1,a=0) pure weighted standard
            #   (q=1,a=1) pure unweighted standard (vanilla LDLQ)
            # ============================================================
            from quant_layerwise.qronos_stats import QronosStats as _QS

            _attn_weights = ["wq", "wk", "wv"]
            _stats_uw = _attn_weighted_qronos_stats_uw
            _stats_w = _attn_weighted_qronos_stats_w
            batch_size = _rbs

            print(f"[coord-adapt] layer {layer_id}: gathering wq/wk/wv data", flush=True)

            # Gather W_orig, module refs for all 3
            _mods = dict(model.named_modules())
            _W_origs: Dict[str, torch.Tensor] = {}
            _module_names: Dict[str, str] = {}
            _weight_names: Dict[str, str] = {}
            _global_nrows_per_w: Dict[str, int | None] = {}
            for _w in _attn_weights:
                _mn = get_hess_name(layer_id, _w)
                _module_names[_w] = _mn
                _weight_names[_w] = get_weight_name(layer_id, _w)
                _W_origs[_w] = _mods[_mn].weight.data.clone()
                # Each weight type may have different row count (wq vs wk/wv with GQA)
                _global_nrows_per_w[_w] = _mods[_mn].weight.shape[0] * dist_world_size if dist_world_size > 1 else None

            _H_attn = H  # shared Hessian for wq/wk/wv

            # Search eval configs: secant re-targets each eval to match desired rate.
            # No Phase 1 calibration needed — each eval finds its own c_param via secant.
            _zcfg_ss: Dict[str, Any] = {}
            for _w in _attn_weights:
                if zsic_cfg_layer is not None:
                    _zcfg_ss[_w] = replace(zsic_cfg_layer,
                                           binary_search=True)
                else:
                    _zcfg_ss[_w] = zsic_cfg_layer

            # Precompute wo_in reference for relMSE evaluation
            if precomputed is not None:
                _wo_ref_cache = precompute_wo_in_ref(precomputed, cache, layer_id, batch_size)
            else:
                _wo_ref_cache = _precompute_wo_in_ref(model_unquant, cache_unquant, layer_id, batch_size)

            import math as _math
            _PHI = (_math.sqrt(5) - 1) / 2  # ~0.618
            _N_STEPS = cfg.coord_adapt_q_eps_steps

            _search_nsamples = max(1, int(cache.nsamples * cfg.adapt_search_sample_ratio)) if cfg.adapt_search_sample_ratio < 1.0 else None
            if _search_nsamples is not None:
                print(f"[coord-adapt] using {_search_nsamples}/{cache.nsamples} samples for search (ratio={cfg.adapt_search_sample_ratio})", flush=True)

            _all_search_targets = []  # collect (q_eps, a_eps, {w: target}) for target-range summary

            def _eval_coord_eps(q_eps, a_eps):
                """Quantize all 3 at (q_eps, a_eps), measure wo_in relMSE, restore."""
                # Stage 1: qronos blend within each weighting
                _hx_w  = (1.0 - q_eps) * _stats_w.sigma_xhat  + q_eps * _stats_w.sigma_x
                _xxh_w = (1.0 - q_eps) * _stats_w.sigma_x_xhat + q_eps * _stats_w.sigma_x
                _hx_uw  = (1.0 - q_eps) * _stats_uw.sigma_xhat  + q_eps * _stats_uw.sigma_x
                _xxh_uw = (1.0 - q_eps) * _stats_uw.sigma_x_xhat + q_eps * _stats_uw.sigma_x
                # Stage 2: attnw blend
                _m = _QS(
                    sigma_x=(1.0 - a_eps) * _stats_w.sigma_x + a_eps * _stats_uw.sigma_x,
                    sigma_xhat=(1.0 - a_eps) * _hx_w + a_eps * _hx_uw,
                    sigma_x_xhat=(1.0 - a_eps) * _xxh_w + a_eps * _xxh_uw,
                    nseq=_stats_uw.nseq, ntokens=_stats_uw.ntokens,
                )

                _search_targets = {}
                for _w2 in _attn_weights:
                    _a = quantize_one_layer(
                        model=model, module_name=_module_names[_w2], weight_name=_weight_names[_w2],
                        H=_H_attn, cfg=cfg, zsic_cfg=_zcfg_ss[_w2],
                        Sig_X=_m.sigma_x if qronos_enabled else None,
                        Sig_hX=_m.sigma_xhat if qronos_enabled else None,
                        Sig_X_hX=_m.sigma_x_xhat if qronos_enabled else None,
                        Sig_delta_R_Xhat=None,
                        Sig_X_for_dead=_sigma_x_for_dead,
                        global_nrows=_global_nrows_per_w[_w2],
                    )
                    _search_targets[_w2] = _a.payload.get("binary_search_target_used", None)
                    apply_layer_artifact(model, _a)
                _W_qs = {_w2: _mods[_module_names[_w2]].weight.data.clone() for _w2 in _attn_weights}
                _rel = _compute_wo_in_rel_mse(model, model_unquant, cache, cache_unquant, layer_id, _W_qs, batch_size, max_samples=_search_nsamples, precomputed_ref=_wo_ref_cache)
                for _w2 in _attn_weights:
                    _mods[_module_names[_w2]].weight.data.copy_(_W_origs[_w2])
                _tgt_str = ", ".join(f"{w}={_search_targets[w]:.4f}" for w in _attn_weights if _search_targets.get(w) is not None)
                print(f"[coord-adapt] eval(q={q_eps:.4f},a={a_eps:.4f}): relMSE={_rel:.6e} targets=[{_tgt_str}]", flush=True)
                _all_search_targets.append((q_eps, a_eps, dict(_search_targets)))
                return _rel

            def _print_target_ranges(label):
                if not _all_search_targets:
                    return
                for w in _attn_weights:
                    vals = [t[2][w] for t in _all_search_targets if w in t[2] and t[2][w] is not None]
                    if vals:
                        print(f"[coord-adapt] {label} target range for {w}: "
                              f"min={min(vals):.4f} max={max(vals):.4f} spread={max(vals)-min(vals):.4f}",
                              flush=True)

            # --- Phase 2a: golden section on q_eps with a_eps=0 ---
            print(f"[coord-adapt] layer {layer_id}: phase 2a — qronos_eps search (a_eps=0, steps={_N_STEPS})", flush=True)
            _f_00 = _eval_coord_eps(0.0, 0.0)
            print(f"[coord-adapt] layer {layer_id}: wo_in relMSE(q=0,a=0) = {_f_00:.6e}", flush=True)
            _all_evals = [(0.0, 0.0, _f_00)]  # track all (q_eps, a_eps, mse)

            _f_q1_a0 = _eval_coord_eps(1.0, 0.0)
            _all_evals.append((1.0, 0.0, _f_q1_a0))
            print(f"[coord-adapt] layer {layer_id}: relMSE(q=1,a=0) = {_f_q1_a0:.6e}", flush=True)

            _lo, _hi = 0.0, 1.0
            _m1 = _hi - _PHI * (_hi - _lo)
            _m2 = _lo + _PHI * (_hi - _lo)
            _f1 = _eval_coord_eps(_m1, 0.0)
            _f2 = _eval_coord_eps(_m2, 0.0)
            _all_evals.append((_m1, 0.0, _f1))
            _all_evals.append((_m2, 0.0, _f2))

            for _step in range(_N_STEPS - 1):
                if _f1 <= _f2:
                    _hi = _m2
                    _m2, _f2 = _m1, _f1
                    _m1 = _hi - _PHI * (_hi - _lo)
                    _f1 = _eval_coord_eps(_m1, 0.0)
                    _all_evals.append((_m1, 0.0, _f1))
                else:
                    _lo = _m1
                    _m1, _f1 = _m2, _f2
                    _m2 = _lo + _PHI * (_hi - _lo)
                    _f2 = _eval_coord_eps(_m2, 0.0)
                    _all_evals.append((_m2, 0.0, _f2))
                print(f"[coord-adapt] phase 2a step {_step+2}/{_N_STEPS}: q_eps in [{_lo:.4f}, {_hi:.4f}]", flush=True)

            _gs_q = (_lo + _hi) / 2
            _gs_q_mse = _eval_coord_eps(_gs_q, 0.0)
            _all_evals.append((_gs_q, 0.0, _gs_q_mse))
            _best_q, _ = min([(0.0, _f_00), (1.0, _f_q1_a0), (_gs_q, _gs_q_mse)], key=lambda x: x[1])
            print(f"[coord-adapt] layer {layer_id}: phase 2a best q_eps={_best_q:.4f}", flush=True)
            _print_target_ranges("phase 2a")

            # --- Phase 2b: golden section on a_eps with q_eps=best_q ---
            _N_STEPS_A = cfg.coord_adapt_a_eps_steps
            if _N_STEPS_A > 0:
                print(f"[coord-adapt] layer {layer_id}: phase 2b — attnw_eps search (q_eps={_best_q:.4f}, steps={_N_STEPS_A})", flush=True)

                _f_bq_a0 = _eval_coord_eps(_best_q, 0.0)
                _all_evals.append((_best_q, 0.0, _f_bq_a0))
                _f_bq_a1 = _eval_coord_eps(_best_q, 1.0)
                _all_evals.append((_best_q, 1.0, _f_bq_a1))
                print(f"[coord-adapt] layer {layer_id}: relMSE(q={_best_q:.4f},a=0) = {_f_bq_a0:.6e}", flush=True)
                print(f"[coord-adapt] layer {layer_id}: relMSE(q={_best_q:.4f},a=1) = {_f_bq_a1:.6e}", flush=True)

                _lo, _hi = 0.0, 1.0
                _m1 = _hi - _PHI * (_hi - _lo)
                _m2 = _lo + _PHI * (_hi - _lo)
                _f1 = _eval_coord_eps(_best_q, _m1)
                _f2 = _eval_coord_eps(_best_q, _m2)
                _all_evals.append((_best_q, _m1, _f1))
                _all_evals.append((_best_q, _m2, _f2))

                for _step in range(_N_STEPS_A - 1):
                    if _f1 <= _f2:
                        _hi = _m2
                        _m2, _f2 = _m1, _f1
                        _m1 = _hi - _PHI * (_hi - _lo)
                        _f1 = _eval_coord_eps(_best_q, _m1)
                        _all_evals.append((_best_q, _m1, _f1))
                    else:
                        _lo = _m1
                        _m1, _f1 = _m2, _f2
                        _m2 = _lo + _PHI * (_hi - _lo)
                        _f2 = _eval_coord_eps(_best_q, _m2)
                        _all_evals.append((_best_q, _m2, _f2))
                    print(f"[coord-adapt] phase 2b step {_step+2}/{_N_STEPS_A}: a_eps in [{_lo:.4f}, {_hi:.4f}]", flush=True)

                _gs_a = (_lo + _hi) / 2
                _gs_a_mse = _eval_coord_eps(_best_q, _gs_a)
                _all_evals.append((_best_q, _gs_a, _gs_a_mse))
            else:
                print(f"[coord-adapt] layer {layer_id}: phase 2b skipped (coord_adapt_a_eps_steps=0, using a_eps=0)", flush=True)

            # Pick global best from ALL evaluated points
            _best_q_eps, _best_a_eps, _best_mse = min(_all_evals, key=lambda x: x[2])
            _print_target_ranges("final")
            print(f"[coord-adapt] layer {layer_id}: best (q_eps={_best_q_eps:.4f}, a_eps={_best_a_eps:.4f}) "
                  f"wo_in relMSE={_best_mse:.6e}", flush=True)

            # --- Phase 3: final quantization with proper rate control ---
            _hx_w_best  = (1.0 - _best_q_eps) * _stats_w.sigma_xhat  + _best_q_eps * _stats_w.sigma_x
            _xxh_w_best = (1.0 - _best_q_eps) * _stats_w.sigma_x_xhat + _best_q_eps * _stats_w.sigma_x
            _hx_uw_best  = (1.0 - _best_q_eps) * _stats_uw.sigma_xhat  + _best_q_eps * _stats_uw.sigma_x
            _xxh_uw_best = (1.0 - _best_q_eps) * _stats_uw.sigma_x_xhat + _best_q_eps * _stats_uw.sigma_x
            _mixed_best = _QS(
                sigma_x=(1.0 - _best_a_eps) * _stats_w.sigma_x + _best_a_eps * _stats_uw.sigma_x,
                sigma_xhat=(1.0 - _best_a_eps) * _hx_w_best + _best_a_eps * _hx_uw_best,
                sigma_x_xhat=(1.0 - _best_a_eps) * _xxh_w_best + _best_a_eps * _xxh_uw_best,
                nseq=_stats_uw.nseq, ntokens=_stats_uw.ntokens,
            )

            for _w in _attn_weights:
                _mn = _module_names[_w]
                _wn = _weight_names[_w]

                # Rate controller
                _target_x = None
                _rc_info_w = None
                if rate_ctrl is not None and cfg.zsic is not None:
                    # Use unquantized Σ_X for dead dim detection when available
                    _dead_mask_w = find_dead_dimensions(_sigma_x_for_dead, threshold_ratio=cfg.zsic.dead_dim_threshold)
                    _n_dead_w = int(_dead_mask_w.sum())
                    _dead_p = 0
                    if _n_dead_w > 0:
                        _a_w = _mods[_mn].weight.shape[0]
                        _dead_p = _a_w * _n_dead_w
                    _target_x, _rc_info_w = rate_ctrl.suggest_target_x(_mn, dead_params=_dead_p)
                    _zcfg_final = replace(cfg.zsic,
                                          target_rate_bits=float(_target_x),
                                          qronos=use_qronos_mode,
                                          residual_compensation=False,
                                          rate_control_active=True)
                    print(f"[coord-adapt] {_mn}: rate target={_target_x:.4f}", flush=True)
                else:
                    _zcfg_final = replace(cfg.zsic,
                                          qronos=use_qronos_mode,
                                          residual_compensation=False) if cfg.zsic is not None else zsic_cfg_layer

                _art = quantize_one_layer(
                    model=model, module_name=_mn, weight_name=_wn,
                    H=_H_attn, cfg=cfg, zsic_cfg=_zcfg_final,
                    Sig_X=_mixed_best.sigma_x if qronos_enabled else None,
                    Sig_hX=_mixed_best.sigma_xhat if qronos_enabled else None,
                    Sig_X_hX=_mixed_best.sigma_x_xhat if qronos_enabled else None,
                    Sig_delta_R_Xhat=None,
                    Sig_X_for_dead=_sigma_x_for_dead,
                    global_nrows=_global_nrows_per_w[_w],
                )

                # Save artifact + manifest
                _save_artifact_and_manifest(_art, _mn, run_dir, manifest, manifest_path,
                                            dist_rank=dist_rank, dist_world_size=dist_world_size)

                # Apply to model
                apply_layer_artifact(model, _art)

                # inf/NaN check
                _w_applied = _mods[_mn].weight.data
                _n_inf = int(_w_applied.isinf().sum())
                _n_nan = int(_w_applied.isnan().sum())
                if _n_inf > 0 or _n_nan > 0:
                    print(f"[WARNING] {_mn}: dequantized weight has {_n_inf} inf, {_n_nan} NaN "
                          f"(max={_w_applied.abs().max():.2e}, shape={list(_w_applied.shape)})", flush=True)

                # Zero out rows if configured
                if cfg.zero_out_rows:
                    _zero_key = f"{layer_id}.{_w}"
                    _dims_to_zero = cfg.zero_out_rows.get(_zero_key, [])
                    if _dims_to_zero:
                        _method = cfg.method.lower()
                        _handled = (_method in ("zsic", "sic") and _w.lower() != "wo")
                        if _handled:
                            print(f"[zero_out] {_mn}: skipping post-quant zeroing (handled by dead row/col exclusion)", flush=True)
                        else:
                            _mod = _mods[_mn]
                            with torch.no_grad():
                                for _row_idx in _dims_to_zero:
                                    _mod.weight.data[_row_idx, :] = 0
                                print(f"[zero_out] zeroed rows {list(_dims_to_zero)} in {_mn}", flush=True)

                # Update rate controller
                _n_dead_art = int(_art.payload.get("n_dead", 0))
                if _n_dead_art > 0:
                    _n_original = int(_art.payload.get("n_original", _art.shape[1]))
                    _n_live = int(_art.payload.get("n_live", _n_original))
                    _entropy = float(_art.payload.get("entropy", 0.0))
                    _a = _art.shape[0]
                    _actual_rate = (_entropy * _n_live + 16 + 16 * _n_live / _a + 16 * _n_dead_art / _a) / _n_original
                else:
                    _actual_rate = float(_art.payload.get("entropy", 0.0)) + float(_art.payload.get("rate_overhead", 0.0))
                if rate_ctrl is not None:
                    _bs_target = _art.payload.get("binary_search_target_used", None)
                    if _bs_target is not None:
                        rate_ctrl.update(_mn, target_x=float(_bs_target), actual_rate=float(_actual_rate))
                    elif _target_x is not None:
                        rate_ctrl.update(_mn, target_x=float(_target_x), actual_rate=float(_actual_rate))
                    rate_ctrl.save_json(str(run_dir / "rate_control_state.json"))

                # Log line
                _line = {
                    "module": _mn,
                    "weight": _wn,
                    "shape": list(_art.shape),
                    "method": _art.method,
                    "nseq": int(nseq_used),
                    "ntokens": int(ntokens_used),
                    "actual_rate": float(_actual_rate),
                    "target_rate_bits_used": None if _target_x is None else float(_target_x),
                    "rate_control": _rc_info_w,
                    "payload": {
                        "loss": float(_art.payload.get("loss", 0.0)),
                        "entropy": float(_art.payload.get("entropy", 0.0)),
                        "rate_overhead": float(_art.payload.get("rate_overhead", 0.0)),
                        "n_dead": int(_art.payload.get("n_dead", 0)),
                        "qronos": bool(_art.payload.get("qronos", False)),
                        "binary_search_target_used": _art.payload.get("binary_search_target_used", None),
                        "binary_search_desired": _art.payload.get("binary_search_desired", None),
                        "binary_search_final_diff": _art.payload.get("binary_search_final_diff", None),
                        "qronos_adapt_eps": _best_q_eps,
                        "adapt_eps_joint": _best_a_eps,
                    },
                    "ts": time.time(),
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(_line) + "\n")

                print(f"[coord-adapt] {_mn}: quantized with (q_eps={_best_q_eps:.4f}, a_eps={_best_a_eps:.4f})", flush=True)

            del _attn_weighted_qronos_stats_uw, _attn_weighted_qronos_stats_w

            # Advance cache if needed
            weights_in_block = layer_weights_map.get(layer_id, [])
            is_last_weight_in_block = (weight == weights_in_block[-1]) if weights_in_block else False
            if is_last_weight_in_block and cache.current_block_idx == layer_id:
                print(f"[cache] advancing through block {layer_id} (all weights quantized)", flush=True)
                cache.advance_through_block(layer_id, batch_size=cfg.hessian_batch_size)
                if cache_unquant is not None:
                    print(f"[unquant-cache] advancing through block {layer_id}", flush=True)
                    cache_unquant.advance_through_block(layer_id, batch_size=cfg.hessian_batch_size)
                _attn_importance_cache.pop(layer_id, None)
                _qronos_stats_cache.pop((layer_id, "attn_qkv"), None)
                _qronos_stats_cache.pop((layer_id, "ffn_w1w3"), None)
                _qronos_stats_w_cache.pop((layer_id, "attn_qkv"), None)
                _qronos_stats_w_cache.pop((layer_id, "ffn_w1w3"), None)

            torch.cuda.empty_cache()
            continue
        elif _attn_weighted_qronos_stats_uw is not None and cfg.attn_weighted_adapt_eps_joint and weight == "wq" and qronos_enabled:
            # ============================================================
            # Joint adaptive eps: find single eps for wq/wk/wv that minimizes
            # wo_in relMSE via forward pass through the attention layer.
            # All three weights are quantized and saved during wq's iteration.
            # wk/wv will hit manifest.has() -> skip when they come up later.
            # Only triggers for wq (first attn weight). On resume, if wq was
            # already saved but wk/wv weren't, they fall through to normal path.
            # Requires qronos_enabled (eps mixing only affects Qronos stats).
            # ============================================================
            from quant_layerwise.qronos_stats import QronosStats as _QS

            _attn_weights = ["wq", "wk", "wv"]
            _stats_uw = _attn_weighted_qronos_stats_uw
            _stats_w = _attn_weighted_qronos_stats_w
            batch_size = _rbs

            print(f"[adapt-eps-joint] layer {layer_id}: gathering wq/wk/wv data", flush=True)

            # Gather W_orig, module refs for all 3
            _mods = dict(model.named_modules())
            _W_origs: Dict[str, torch.Tensor] = {}
            _module_names: Dict[str, str] = {}
            _weight_names: Dict[str, str] = {}
            _global_nrows_per_w: Dict[str, int | None] = {}
            for _w in _attn_weights:
                _mn = get_hess_name(layer_id, _w)
                _module_names[_w] = _mn
                _weight_names[_w] = get_weight_name(layer_id, _w)
                _W_origs[_w] = _mods[_mn].weight.data.clone()
                _global_nrows_per_w[_w] = _mods[_mn].weight.shape[0] * dist_world_size if dist_world_size > 1 else None

            # For each weight, we need its own Hessian (wq has H already; wk/wv share same input so same H)
            # wq/wk/wv all see the same input (attention_norm output), so H is the same for all three
            _H_attn = H  # shared Hessian for wq/wk/wv

            # For Qronos stats: wq has qronos_stats already computed.
            # wk/wv need their own too, but since input is identical, stats are the same for wq/wk/wv.
            # The weighted stats (_stats_w) and unweighted stats (_stats_uw) are the same for all QKV.

            # Search eval configs: secant re-targets each eval to match desired rate.
            # No Phase 1 calibration needed — each eval finds its own c_param via secant.
            _zcfg_ss: Dict[str, Any] = {}
            for _w in _attn_weights:
                if zsic_cfg_layer is not None:
                    _zcfg_ss[_w] = replace(zsic_cfg_layer,
                                           binary_search=True)
                else:
                    _zcfg_ss[_w] = zsic_cfg_layer

            # Precompute wo_in reference for relMSE evaluation
            if precomputed is not None:
                _wo_ref_cache = precompute_wo_in_ref(precomputed, cache, layer_id, batch_size)
            else:
                _wo_ref_cache = _precompute_wo_in_ref(model_unquant, cache_unquant, layer_id, batch_size)

            # --- Phase 2: golden section search on eps ∈ [0, 1] ---
            import math as _math
            _PHI = (_math.sqrt(5) - 1) / 2  # ≈ 0.618
            _N_STEPS = cfg.coord_adapt_q_eps_steps

            _search_nsamples = max(1, int(cache.nsamples * cfg.adapt_search_sample_ratio)) if cfg.adapt_search_sample_ratio < 1.0 else None
            if _search_nsamples is not None:
                print(f"[adapt-eps-joint] using {_search_nsamples}/{cache.nsamples} samples for search (ratio={cfg.adapt_search_sample_ratio})", flush=True)

            def _eval_joint_eps(eps_val):
                """Quantize all 3 at eps_val, measure wo_in relMSE, restore."""
                _search_targets: Dict[str, Any] = {}
                _m = _QS(
                    sigma_x=(1.0 - eps_val) * _stats_w.sigma_x + eps_val * _stats_uw.sigma_x,
                    sigma_xhat=(1.0 - eps_val) * _stats_w.sigma_xhat + eps_val * _stats_uw.sigma_xhat,
                    sigma_x_xhat=(1.0 - eps_val) * _stats_w.sigma_x_xhat + eps_val * _stats_uw.sigma_x_xhat,
                    nseq=_stats_uw.nseq, ntokens=_stats_uw.ntokens,
                )
                for _w2 in _attn_weights:
                    _a = quantize_one_layer(
                        model=model, module_name=_module_names[_w2], weight_name=_weight_names[_w2],
                        H=_H_attn, cfg=cfg, zsic_cfg=_zcfg_ss[_w2],
                        Sig_X=_m.sigma_x if qronos_enabled else None,
                        Sig_hX=_m.sigma_xhat if qronos_enabled else None,
                        Sig_X_hX=_m.sigma_x_xhat if qronos_enabled else None,
                        Sig_delta_R_Xhat=None,
                        Sig_X_for_dead=_sigma_x_for_dead,
                        global_nrows=_global_nrows_per_w[_w2],
                    )
                    _search_targets[_w2] = _a.payload.get("binary_search_target_used", None)
                    apply_layer_artifact(model, _a)
                _W_qs = {_w2: _mods[_module_names[_w2]].weight.data.clone() for _w2 in _attn_weights}
                _rel = _compute_wo_in_rel_mse(model, model_unquant, cache, cache_unquant, layer_id, _W_qs, batch_size, max_samples=_search_nsamples, precomputed_ref=_wo_ref_cache)
                for _w2 in _attn_weights:
                    _mods[_module_names[_w2]].weight.data.copy_(_W_origs[_w2])
                _tgt_str = ", ".join(f"{w}={_search_targets[w]:.4f}" for w in _attn_weights if _search_targets.get(w) is not None)
                print(f"[adapt-eps-joint] eval(eps={eps_val:.4f}): relMSE={_rel:.6e} targets=[{_tgt_str}]", flush=True)
                return _rel

            _f_eps0 = _eval_joint_eps(0.0)
            print(f"[adapt-eps-joint] layer {layer_id}: wo_in relMSE(eps=0.0) = {_f_eps0:.6e}", flush=True)
            _f_eps1 = _eval_joint_eps(1.0)
            print(f"[adapt-eps-joint] layer {layer_id}: wo_in relMSE(eps=1.0) = {_f_eps1:.6e}", flush=True)

            # Golden section search
            _lo, _hi = 0.0, 1.0
            _m1 = _hi - _PHI * (_hi - _lo)
            _m2 = _lo + _PHI * (_hi - _lo)
            _f1 = _eval_joint_eps(_m1)
            _f2 = _eval_joint_eps(_m2)

            for _step in range(_N_STEPS - 1):
                if _f1 <= _f2:
                    _hi = _m2
                    _m2, _f2 = _m1, _f1
                    _m1 = _hi - _PHI * (_hi - _lo)
                    _f1 = _eval_joint_eps(_m1)
                else:
                    _lo = _m1
                    _m1, _f1 = _m2, _f2
                    _m2 = _lo + _PHI * (_hi - _lo)
                    _f2 = _eval_joint_eps(_m2)
                print(f"[adapt-eps-joint] step {_step+2}/{_N_STEPS}: [{_lo:.4f}, {_hi:.4f}]", flush=True)

            _gs_eps = (_lo + _hi) / 2
            _gs_mse = _eval_joint_eps(_gs_eps)
            _best_eps, _best_mse = min([(0.0, _f_eps0), (1.0, _f_eps1), (_gs_eps, _gs_mse)], key=lambda x: x[1])
            print(f"[adapt-eps-joint] layer {layer_id}: best eps={_best_eps:.4f} (wo_in relMSE={_best_mse:.6e})", flush=True)

            # --- Phase 3: final quantization of ALL THREE with proper rate control ---
            _mixed_best = _QS(
                sigma_x=(1.0 - _best_eps) * _stats_w.sigma_x + _best_eps * _stats_uw.sigma_x,
                sigma_xhat=(1.0 - _best_eps) * _stats_w.sigma_xhat + _best_eps * _stats_uw.sigma_xhat,
                sigma_x_xhat=(1.0 - _best_eps) * _stats_w.sigma_x_xhat + _best_eps * _stats_uw.sigma_x_xhat,
                nseq=_stats_uw.nseq, ntokens=_stats_uw.ntokens,
            )

            for _w in _attn_weights:
                _mn = _module_names[_w]
                _wn = _weight_names[_w]

                # Rate controller: suggest_target_x is stateless, update consumes budget
                _target_x = None
                _rc_info_w = None
                if rate_ctrl is not None and cfg.zsic is not None:
                    # Use unquantized Σ_X for dead dim detection when available
                    _dead_mask_w = find_dead_dimensions(_sigma_x_for_dead, threshold_ratio=cfg.zsic.dead_dim_threshold)
                    _n_dead_w = int(_dead_mask_w.sum())
                    _dead_p = 0
                    if _n_dead_w > 0:
                        _a_w = _mods[_mn].weight.shape[0]
                        _dead_p = _a_w * _n_dead_w
                    _target_x, _rc_info_w = rate_ctrl.suggest_target_x(_mn, dead_params=_dead_p)
                    _zcfg_final = replace(cfg.zsic,
                                          target_rate_bits=float(_target_x),
                                          qronos=use_qronos_mode,
                                          residual_compensation=False,
                                          rate_control_active=True)
                    print(f"[adapt-eps-joint] {_mn}: rate target={_target_x:.4f}", flush=True)
                else:
                    _zcfg_final = replace(cfg.zsic,
                                          qronos=use_qronos_mode,
                                          residual_compensation=False) if cfg.zsic is not None else zsic_cfg_layer

                _art = quantize_one_layer(
                    model=model, module_name=_mn, weight_name=_wn,
                    H=_H_attn, cfg=cfg, zsic_cfg=_zcfg_final,
                    Sig_X=_mixed_best.sigma_x if qronos_enabled else None,
                    Sig_hX=_mixed_best.sigma_xhat if qronos_enabled else None,
                    Sig_X_hX=_mixed_best.sigma_x_xhat if qronos_enabled else None,
                    Sig_delta_R_Xhat=None,
                    Sig_X_for_dead=_sigma_x_for_dead,
                    global_nrows=_global_nrows_per_w[_w],
                )

                # Save artifact + manifest
                _save_artifact_and_manifest(_art, _mn, run_dir, manifest, manifest_path,
                                            dist_rank=dist_rank, dist_world_size=dist_world_size)

                # Apply to model
                apply_layer_artifact(model, _art)

                # inf/NaN check
                _w_applied = _mods[_mn].weight.data
                _n_inf = int(_w_applied.isinf().sum())
                _n_nan = int(_w_applied.isnan().sum())
                if _n_inf > 0 or _n_nan > 0:
                    print(f"[WARNING] {_mn}: dequantized weight has {_n_inf} inf, {_n_nan} NaN "
                          f"(max={_w_applied.abs().max():.2e}, shape={list(_w_applied.shape)})", flush=True)

                # Zero out rows if configured
                if cfg.zero_out_rows:
                    _zero_key = f"{layer_id}.{_w}"
                    _dims_to_zero = cfg.zero_out_rows.get(_zero_key, [])
                    if _dims_to_zero:
                        _method = cfg.method.lower()
                        _handled = (_method in ("zsic", "sic") and _w.lower() != "wo")
                        if _handled:
                            print(f"[zero_out] {_mn}: skipping post-quant zeroing (handled by dead row/col exclusion)", flush=True)
                        else:
                            _mod = _mods[_mn]
                            with torch.no_grad():
                                for _row_idx in _dims_to_zero:
                                    _mod.weight.data[_row_idx, :] = 0
                                print(f"[zero_out] zeroed rows {list(_dims_to_zero)} in {_mn}", flush=True)

                # Update rate controller
                _n_dead_art = int(_art.payload.get("n_dead", 0))
                if _n_dead_art > 0:
                    _n_original = int(_art.payload.get("n_original", _art.shape[1]))
                    _n_live = int(_art.payload.get("n_live", _n_original))
                    _entropy = float(_art.payload.get("entropy", 0.0))
                    _a = _art.shape[0]
                    _actual_rate = (_entropy * _n_live + 16 + 16 * _n_live / _a + 16 * _n_dead_art / _a) / _n_original
                else:
                    _actual_rate = float(_art.payload.get("entropy", 0.0)) + float(_art.payload.get("rate_overhead", 0.0))
                if rate_ctrl is not None:
                    _bs_target = _art.payload.get("binary_search_target_used", None)
                    if _bs_target is not None:
                        rate_ctrl.update(_mn, target_x=float(_bs_target), actual_rate=float(_actual_rate))
                    elif _target_x is not None:
                        rate_ctrl.update(_mn, target_x=float(_target_x), actual_rate=float(_actual_rate))
                    rate_ctrl.save_json(str(run_dir / "rate_control_state.json"))

                # Log line
                _line = {
                    "module": _mn,
                    "weight": _wn,
                    "shape": list(_art.shape),
                    "method": _art.method,
                    "nseq": int(nseq_used),
                    "ntokens": int(ntokens_used),
                    "actual_rate": float(_actual_rate),
                    "target_rate_bits_used": None if _target_x is None else float(_target_x),
                    "rate_control": _rc_info_w,
                    "payload": {
                        "loss": float(_art.payload.get("loss", 0.0)),
                        "entropy": float(_art.payload.get("entropy", 0.0)),
                        "rate_overhead": float(_art.payload.get("rate_overhead", 0.0)),
                        "n_dead": int(_art.payload.get("n_dead", 0)),
                        "qronos": bool(_art.payload.get("qronos", False)),
                        "binary_search_target_used": _art.payload.get("binary_search_target_used", None),
                        "binary_search_desired": _art.payload.get("binary_search_desired", None),
                        "binary_search_final_diff": _art.payload.get("binary_search_final_diff", None),
                        "adapt_eps_joint": _best_eps,
                    },
                    "ts": time.time(),
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(_line) + "\n")

                print(f"[adapt-eps-joint] {_mn}: quantized with eps={_best_eps:.4f}", flush=True)

            del _attn_weighted_qronos_stats_uw, _attn_weighted_qronos_stats_w

            # Skip normal post-processing (already handled above for all 3 weights).
            # Advance cache if this was the last weight in the block.
            weights_in_block = layer_weights_map.get(layer_id, [])
            is_last_weight_in_block = (weight == weights_in_block[-1]) if weights_in_block else False
            if is_last_weight_in_block and cache.current_block_idx == layer_id:
                print(f"[cache] advancing through block {layer_id} (all weights quantized)", flush=True)
                cache.advance_through_block(layer_id, batch_size=cfg.hessian_batch_size)
                if cache_unquant is not None:
                    print(f"[unquant-cache] advancing through block {layer_id}", flush=True)
                    cache_unquant.advance_through_block(layer_id, batch_size=cfg.hessian_batch_size)
                _attn_importance_cache.pop(layer_id, None)
                _qronos_stats_cache.pop((layer_id, "attn_qkv"), None)
                _qronos_stats_cache.pop((layer_id, "ffn_w1w3"), None)
                _qronos_stats_w_cache.pop((layer_id, "attn_qkv"), None)
                _qronos_stats_w_cache.pop((layer_id, "ffn_w1w3"), None)

            torch.cuda.empty_cache()
            continue
        elif cfg.qronos_adapt and weight == "wq" and qronos_enabled and qronos_stats is not None:
            # ============================================================
            # Qronos adapt: adaptively blend qronos stats with standard stats
            # for QKV. Finds eps in [0,1] that minimizes wo_in relMSE:
            #   Sig_hX_mix  = (1-eps)*Sigma_Xhat  + eps*Sigma_X
            #   Sig_X_hX_mix = (1-eps)*Sigma_X_Xhat + eps*Sigma_X
            # eps=0: pure qronos, eps=1: pure standard LDLQ
            # All three weights are quantized during wq's iteration.
            # wk/wv will hit manifest.has() -> skip when they come up later.
            # ============================================================
            from quant_layerwise.qronos_stats import QronosStats as _QS

            _attn_weights = ["wq", "wk", "wv"]
            _qs = qronos_stats  # unweighted qronos stats (already computed)
            batch_size = _rbs

            print(f"[qronos-adapt] layer {layer_id}: gathering wq/wk/wv data", flush=True)

            # Gather W_orig, module refs for all 3
            _mods = dict(model.named_modules())
            _W_origs: Dict[str, torch.Tensor] = {}
            _module_names: Dict[str, str] = {}
            _weight_names: Dict[str, str] = {}
            _global_nrows_per_w: Dict[str, int | None] = {}
            for _w in _attn_weights:
                _mn = get_hess_name(layer_id, _w)
                _module_names[_w] = _mn
                _weight_names[_w] = get_weight_name(layer_id, _w)
                _W_origs[_w] = _mods[_mn].weight.data.clone()
                _global_nrows_per_w[_w] = _mods[_mn].weight.shape[0] * dist_world_size if dist_world_size > 1 else None

            # wq/wk/wv all see the same input, so H is shared
            _H_attn = H

            # Search eval configs: secant re-targets each eval to match desired rate.
            # No Phase 1 calibration needed — each eval finds its own c_param via secant.
            _zcfg_ss: Dict[str, Any] = {}
            for _w in _attn_weights:
                if zsic_cfg_layer is not None:
                    _zcfg_ss[_w] = replace(zsic_cfg_layer,
                                           binary_search=True)
                else:
                    _zcfg_ss[_w] = zsic_cfg_layer

            # Precompute wo_in reference for relMSE evaluation
            if precomputed is not None:
                _wo_ref_cache = precompute_wo_in_ref(precomputed, cache, layer_id, batch_size)
            else:
                _wo_ref_cache = _precompute_wo_in_ref(model_unquant, cache_unquant, layer_id, batch_size)

            # --- Golden section search on eps in [0, 1] ---
            import math as _math
            _PHI = (_math.sqrt(5) - 1) / 2  # ~0.618
            _N_STEPS = cfg.coord_adapt_q_eps_steps

            _search_nsamples = max(1, int(cache.nsamples * cfg.adapt_search_sample_ratio)) if cfg.adapt_search_sample_ratio < 1.0 else None
            if _search_nsamples is not None:
                print(f"[qronos-adapt] using {_search_nsamples}/{cache.nsamples} samples for search (ratio={cfg.adapt_search_sample_ratio})", flush=True)

            def _eval_qronos_adapt_eps(eps_val):
                """Quantize all 3 at eps_val, measure wo_in relMSE, restore."""
                _sig_hx_mix = (1.0 - eps_val) * _qs.sigma_xhat + eps_val * _qs.sigma_x
                _sig_x_xhat_mix = (1.0 - eps_val) * _qs.sigma_x_xhat + eps_val * _qs.sigma_x
                _m = _QS(
                    sigma_x=_qs.sigma_x,
                    sigma_xhat=_sig_hx_mix,
                    sigma_x_xhat=_sig_x_xhat_mix,
                    nseq=_qs.nseq, ntokens=_qs.ntokens,
                )
                _search_targets = {}
                for _w2 in _attn_weights:
                    _a = quantize_one_layer(
                        model=model, module_name=_module_names[_w2], weight_name=_weight_names[_w2],
                        H=_H_attn, cfg=cfg, zsic_cfg=_zcfg_ss[_w2],
                        Sig_X=_m.sigma_x if qronos_enabled else None,
                        Sig_hX=_m.sigma_xhat if qronos_enabled else None,
                        Sig_X_hX=_m.sigma_x_xhat if qronos_enabled else None,
                        Sig_delta_R_Xhat=None,
                        Sig_X_for_dead=_sigma_x_for_dead,
                        global_nrows=_global_nrows_per_w[_w2],
                    )
                    _search_targets[_w2] = _a.payload.get("binary_search_target_used", None)
                    apply_layer_artifact(model, _a)
                _W_qs = {_w2: _mods[_module_names[_w2]].weight.data.clone() for _w2 in _attn_weights}
                _rel = _compute_wo_in_rel_mse(model, model_unquant, cache, cache_unquant, layer_id, _W_qs, batch_size, max_samples=_search_nsamples, precomputed_ref=_wo_ref_cache)
                for _w2 in _attn_weights:
                    _mods[_module_names[_w2]].weight.data.copy_(_W_origs[_w2])
                _tgt_str = ", ".join(f"{w}={_search_targets[w]:.4f}" for w in _attn_weights if _search_targets.get(w) is not None)
                print(f"[qronos-adapt] eval(eps={eps_val:.4f}): relMSE={_rel:.6e} targets=[{_tgt_str}]", flush=True)
                return _rel

            _f_eps0 = _eval_qronos_adapt_eps(0.0)
            print(f"[qronos-adapt] layer {layer_id}: wo_in relMSE(eps=0.0) = {_f_eps0:.6e}", flush=True)
            _f_eps1 = _eval_qronos_adapt_eps(1.0)
            print(f"[qronos-adapt] layer {layer_id}: wo_in relMSE(eps=1.0) = {_f_eps1:.6e}", flush=True)

            # Golden section search
            _lo, _hi = 0.0, 1.0
            _m1 = _hi - _PHI * (_hi - _lo)
            _m2 = _lo + _PHI * (_hi - _lo)
            _f1 = _eval_qronos_adapt_eps(_m1)
            _f2 = _eval_qronos_adapt_eps(_m2)

            for _step in range(_N_STEPS - 1):
                if _f1 <= _f2:
                    _hi = _m2
                    _m2, _f2 = _m1, _f1
                    _m1 = _hi - _PHI * (_hi - _lo)
                    _f1 = _eval_qronos_adapt_eps(_m1)
                else:
                    _lo = _m1
                    _m1, _f1 = _m2, _f2
                    _m2 = _lo + _PHI * (_hi - _lo)
                    _f2 = _eval_qronos_adapt_eps(_m2)
                print(f"[qronos-adapt] step {_step+2}/{_N_STEPS}: [{_lo:.4f}, {_hi:.4f}]", flush=True)

            _gs_eps = (_lo + _hi) / 2
            _gs_mse = _eval_qronos_adapt_eps(_gs_eps)
            _best_eps, _best_mse = min([(0.0, _f_eps0), (1.0, _f_eps1), (_gs_eps, _gs_mse)], key=lambda x: x[1])
            print(f"[qronos-adapt] layer {layer_id}: best eps={_best_eps:.4f} (wo_in relMSE={_best_mse:.6e})", flush=True)

            # --- Phase 3: final quantization of ALL THREE with proper rate control ---
            _sig_hx_best = (1.0 - _best_eps) * _qs.sigma_xhat + _best_eps * _qs.sigma_x
            _sig_x_xhat_best = (1.0 - _best_eps) * _qs.sigma_x_xhat + _best_eps * _qs.sigma_x
            _mixed_best = _QS(
                sigma_x=_qs.sigma_x,
                sigma_xhat=_sig_hx_best,
                sigma_x_xhat=_sig_x_xhat_best,
                nseq=_qs.nseq, ntokens=_qs.ntokens,
            )

            for _w in _attn_weights:
                _mn = _module_names[_w]
                _wn = _weight_names[_w]

                # Rate controller: suggest_target_x is stateless, update consumes budget
                _target_x = None
                _rc_info_w = None
                if rate_ctrl is not None and cfg.zsic is not None:
                    # Use unquantized Σ_X for dead dim detection when available
                    _dead_mask_w = find_dead_dimensions(_sigma_x_for_dead, threshold_ratio=cfg.zsic.dead_dim_threshold)
                    _n_dead_w = int(_dead_mask_w.sum())
                    _dead_p = 0
                    if _n_dead_w > 0:
                        _a_w = _mods[_mn].weight.shape[0]
                        _dead_p = _a_w * _n_dead_w
                    _target_x, _rc_info_w = rate_ctrl.suggest_target_x(_mn, dead_params=_dead_p)
                    _zcfg_final = replace(cfg.zsic,
                                          target_rate_bits=float(_target_x),
                                          qronos=use_qronos_mode,
                                          residual_compensation=False,
                                          rate_control_active=True)
                    print(f"[qronos-adapt] {_mn}: rate target={_target_x:.4f}", flush=True)
                else:
                    _zcfg_final = replace(cfg.zsic,
                                          qronos=use_qronos_mode,
                                          residual_compensation=False) if cfg.zsic is not None else zsic_cfg_layer

                _art = quantize_one_layer(
                    model=model, module_name=_mn, weight_name=_wn,
                    H=_H_attn, cfg=cfg, zsic_cfg=_zcfg_final,
                    Sig_X=_mixed_best.sigma_x if qronos_enabled else None,
                    Sig_hX=_mixed_best.sigma_xhat if qronos_enabled else None,
                    Sig_X_hX=_mixed_best.sigma_x_xhat if qronos_enabled else None,
                    Sig_delta_R_Xhat=None,
                    Sig_X_for_dead=_sigma_x_for_dead,
                    global_nrows=_global_nrows_per_w[_w],
                )

                # Save artifact + manifest
                _save_artifact_and_manifest(_art, _mn, run_dir, manifest, manifest_path,
                                            dist_rank=dist_rank, dist_world_size=dist_world_size)

                # Apply to model
                apply_layer_artifact(model, _art)

                # inf/NaN check
                _w_applied = _mods[_mn].weight.data
                _n_inf = int(_w_applied.isinf().sum())
                _n_nan = int(_w_applied.isnan().sum())
                if _n_inf > 0 or _n_nan > 0:
                    print(f"[WARNING] {_mn}: dequantized weight has {_n_inf} inf, {_n_nan} NaN "
                          f"(max={_w_applied.abs().max():.2e}, shape={list(_w_applied.shape)})", flush=True)

                # Zero out rows if configured
                if cfg.zero_out_rows:
                    _zero_key = f"{layer_id}.{_w}"
                    _dims_to_zero = cfg.zero_out_rows.get(_zero_key, [])
                    if _dims_to_zero:
                        _method = cfg.method.lower()
                        _handled = (_method in ("zsic", "sic") and _w.lower() != "wo")
                        if _handled:
                            print(f"[zero_out] {_mn}: skipping post-quant zeroing (handled by dead row/col exclusion)", flush=True)
                        else:
                            _mod = _mods[_mn]
                            with torch.no_grad():
                                for _row_idx in _dims_to_zero:
                                    _mod.weight.data[_row_idx, :] = 0
                                print(f"[zero_out] zeroed rows {list(_dims_to_zero)} in {_mn}", flush=True)

                # Update rate controller
                _n_dead_art = int(_art.payload.get("n_dead", 0))
                if _n_dead_art > 0:
                    _n_original = int(_art.payload.get("n_original", _art.shape[1]))
                    _n_live = int(_art.payload.get("n_live", _n_original))
                    _entropy = float(_art.payload.get("entropy", 0.0))
                    _a = _art.shape[0]
                    _actual_rate = (_entropy * _n_live + 16 + 16 * _n_live / _a + 16 * _n_dead_art / _a) / _n_original
                else:
                    _actual_rate = float(_art.payload.get("entropy", 0.0)) + float(_art.payload.get("rate_overhead", 0.0))
                if rate_ctrl is not None:
                    _bs_target = _art.payload.get("binary_search_target_used", None)
                    if _bs_target is not None:
                        rate_ctrl.update(_mn, target_x=float(_bs_target), actual_rate=float(_actual_rate))
                    elif _target_x is not None:
                        rate_ctrl.update(_mn, target_x=float(_target_x), actual_rate=float(_actual_rate))
                    rate_ctrl.save_json(str(run_dir / "rate_control_state.json"))

                # Log line
                _line = {
                    "module": _mn,
                    "weight": _wn,
                    "shape": list(_art.shape),
                    "method": _art.method,
                    "nseq": int(nseq_used),
                    "ntokens": int(ntokens_used),
                    "actual_rate": float(_actual_rate),
                    "target_rate_bits_used": None if _target_x is None else float(_target_x),
                    "rate_control": _rc_info_w,
                    "payload": {
                        "loss": float(_art.payload.get("loss", 0.0)),
                        "entropy": float(_art.payload.get("entropy", 0.0)),
                        "rate_overhead": float(_art.payload.get("rate_overhead", 0.0)),
                        "n_dead": int(_art.payload.get("n_dead", 0)),
                        "qronos": bool(_art.payload.get("qronos", False)),
                        "binary_search_target_used": _art.payload.get("binary_search_target_used", None),
                        "binary_search_desired": _art.payload.get("binary_search_desired", None),
                        "binary_search_final_diff": _art.payload.get("binary_search_final_diff", None),
                        "qronos_adapt_eps": _best_eps,
                    },
                    "ts": time.time(),
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(_line) + "\n")

                print(f"[qronos-adapt] {_mn}: quantized with eps={_best_eps:.4f}", flush=True)

            # Skip normal post-processing (already handled above for all 3 weights).
            # Advance cache if this was the last weight in the block.
            weights_in_block = layer_weights_map.get(layer_id, [])
            is_last_weight_in_block = (weight == weights_in_block[-1]) if weights_in_block else False
            if is_last_weight_in_block and cache.current_block_idx == layer_id:
                print(f"[cache] advancing through block {layer_id} (all weights quantized)", flush=True)
                cache.advance_through_block(layer_id, batch_size=cfg.hessian_batch_size)
                if cache_unquant is not None:
                    print(f"[unquant-cache] advancing through block {layer_id}", flush=True)
                    cache_unquant.advance_through_block(layer_id, batch_size=cfg.hessian_batch_size)
                _attn_importance_cache.pop(layer_id, None)
                _qronos_stats_cache.pop((layer_id, "attn_qkv"), None)
                _qronos_stats_cache.pop((layer_id, "ffn_w1w3"), None)
                _qronos_stats_w_cache.pop((layer_id, "attn_qkv"), None)
                _qronos_stats_w_cache.pop((layer_id, "ffn_w1w3"), None)

            torch.cuda.empty_cache()
            continue
        elif (cfg.w1w3_qronos_adapt and weight == "w1"
              and qronos_enabled and qronos_stats is not None):
            # ============================================================
            # w1w3 qronos adapt: adaptively blend qronos stats with standard
            # stats for w1 and w3 jointly. Finds eps in [0,1] that minimizes
            # w2 input relMSE (the gated hidden state SiLU(w1·x) ⊙ w3·x):
            #   Sig_hX_mix   = (1-eps)*Sigma_Xhat   + eps*Sigma_X
            #   Sig_X_hX_mix = (1-eps)*Sigma_X_Xhat + eps*Sigma_X
            # eps=0: pure qronos, eps=1: pure standard LDLQ
            # Both w1 and w3 are quantized during w1's iteration.
            # w3 will hit manifest.has() -> skip when it comes up later.
            # w1 and w3 share the same input (ffn_norm output) -> same H
            # and qronos_stats -> single shared eps.
            # ============================================================
            from quant_layerwise.qronos_stats import QronosStats as _QS

            _ffn_weights = ["w1", "w3"]
            _qs = qronos_stats  # unweighted qronos stats (already computed)
            batch_size = _rbs

            print(f"[w1w3-qronos-adapt] layer {layer_id}: gathering w1/w3 data", flush=True)

            # Gather W_orig, module refs for both
            _mods = dict(model.named_modules())
            _W_origs: Dict[str, torch.Tensor] = {}
            _module_names: Dict[str, str] = {}
            _weight_names: Dict[str, str] = {}
            _global_nrows_per_w: Dict[str, int | None] = {}
            for _w in _ffn_weights:
                _mn = get_hess_name(layer_id, _w)
                _module_names[_w] = _mn
                _weight_names[_w] = get_weight_name(layer_id, _w)
                _W_origs[_w] = _mods[_mn].weight.data.clone()
                _global_nrows_per_w[_w] = _mods[_mn].weight.shape[0] * dist_world_size if dist_world_size > 1 else None

            # w1 and w3 share the same input, so H is shared
            _H_ffn = H

            # Search eval configs: secant re-targets each eval to match desired rate.
            # No Phase 1 calibration needed — each eval finds its own c_param via secant.
            _zcfg_ss: Dict[str, Any] = {}
            for _w in _ffn_weights:
                if zsic_cfg_layer is not None:
                    _zcfg_ss[_w] = replace(zsic_cfg_layer,
                                           binary_search=True)
                else:
                    _zcfg_ss[_w] = zsic_cfg_layer

            # Precompute w2_in reference for relMSE evaluation
            if precomputed is not None:
                _w2_ref_cache = precompute_w2_in_ref(precomputed, cache, layer_id, batch_size)
            else:
                _w2_ref_cache = _precompute_w2_in_ref(model_unquant, cache_unquant, layer_id, batch_size)

            # --- Golden section search on eps in [0, 1] ---
            import math as _math
            _PHI = (_math.sqrt(5) - 1) / 2  # ~0.618
            _N_STEPS = cfg.coord_adapt_q_eps_steps

            _search_nsamples = max(1, int(cache.nsamples * cfg.adapt_search_sample_ratio)) if cfg.adapt_search_sample_ratio < 1.0 else None
            if _search_nsamples is not None:
                print(f"[w1w3-qronos-adapt] using {_search_nsamples}/{cache.nsamples} samples for search (ratio={cfg.adapt_search_sample_ratio})", flush=True)

            def _eval_w1w3_eps(eps_val):
                """Quantize both w1+w3 at eps_val, measure w2_in relMSE, restore."""
                _sig_hx_mix = (1.0 - eps_val) * _qs.sigma_xhat + eps_val * _qs.sigma_x
                _sig_x_xhat_mix = (1.0 - eps_val) * _qs.sigma_x_xhat + eps_val * _qs.sigma_x
                _m = _QS(
                    sigma_x=_qs.sigma_x,
                    sigma_xhat=_sig_hx_mix,
                    sigma_x_xhat=_sig_x_xhat_mix,
                    nseq=_qs.nseq, ntokens=_qs.ntokens,
                )
                _search_targets_ffn = {}
                for _w2 in _ffn_weights:
                    _a = quantize_one_layer(
                        model=model, module_name=_module_names[_w2], weight_name=_weight_names[_w2],
                        H=_H_ffn, cfg=cfg, zsic_cfg=_zcfg_ss[_w2],
                        Sig_X=_m.sigma_x if qronos_enabled else None,
                        Sig_hX=_m.sigma_xhat if qronos_enabled else None,
                        Sig_X_hX=_m.sigma_x_xhat if qronos_enabled else None,
                        Sig_delta_R_Xhat=None,
                        Sig_X_for_dead=_sigma_x_for_dead,
                        global_nrows=_global_nrows_per_w[_w2],
                    )
                    _search_targets_ffn[_w2] = _a.payload.get("binary_search_target_used", None)
                    apply_layer_artifact(model, _a)
                _W_w1_q = _mods[_module_names["w1"]].weight.data.clone()
                _W_w3_q = _mods[_module_names["w3"]].weight.data.clone()
                _rel = _compute_w2_in_rel_mse(model, model_unquant, cache, cache_unquant, layer_id, _W_w1_q, _W_w3_q, batch_size, max_samples=_search_nsamples, precomputed_ref=_w2_ref_cache)
                for _w2 in _ffn_weights:
                    _mods[_module_names[_w2]].weight.data.copy_(_W_origs[_w2])
                _tgt_str = ", ".join(f"{w}={_search_targets_ffn[w]:.4f}" for w in _ffn_weights if _search_targets_ffn.get(w) is not None)
                print(f"[w1w3-adapt] eval(eps={eps_val:.4f}): relMSE={_rel:.6e} targets=[{_tgt_str}]", flush=True)
                return _rel

            _f_eps0 = _eval_w1w3_eps(0.0)
            print(f"[w1w3-qronos-adapt] layer {layer_id}: w2_in relMSE(eps=0.0) = {_f_eps0:.6e}", flush=True)
            _f_eps1 = _eval_w1w3_eps(1.0)
            print(f"[w1w3-qronos-adapt] layer {layer_id}: w2_in relMSE(eps=1.0) = {_f_eps1:.6e}", flush=True)

            # Golden section search
            _lo, _hi = 0.0, 1.0
            _m1 = _hi - _PHI * (_hi - _lo)
            _m2 = _lo + _PHI * (_hi - _lo)
            _f1 = _eval_w1w3_eps(_m1)
            _f2 = _eval_w1w3_eps(_m2)

            for _step in range(_N_STEPS - 1):
                if _f1 <= _f2:
                    _hi = _m2
                    _m2, _f2 = _m1, _f1
                    _m1 = _hi - _PHI * (_hi - _lo)
                    _f1 = _eval_w1w3_eps(_m1)
                else:
                    _lo = _m1
                    _m1, _f1 = _m2, _f2
                    _m2 = _lo + _PHI * (_hi - _lo)
                    _f2 = _eval_w1w3_eps(_m2)
                print(f"[w1w3-qronos-adapt] step {_step+2}/{_N_STEPS}: [{_lo:.4f}, {_hi:.4f}]", flush=True)

            _gs_eps = (_lo + _hi) / 2
            _gs_mse = _eval_w1w3_eps(_gs_eps)
            _best_eps, _best_mse = min([(0.0, _f_eps0), (1.0, _f_eps1), (_gs_eps, _gs_mse)], key=lambda x: x[1])
            print(f"[w1w3-qronos-adapt] layer {layer_id}: best eps={_best_eps:.4f} (w2_in relMSE={_best_mse:.6e})", flush=True)

            # --- Phase 3: final quantization of BOTH with proper rate control ---
            _sig_hx_best = (1.0 - _best_eps) * _qs.sigma_xhat + _best_eps * _qs.sigma_x
            _sig_x_xhat_best = (1.0 - _best_eps) * _qs.sigma_x_xhat + _best_eps * _qs.sigma_x
            _mixed_best = _QS(
                sigma_x=_qs.sigma_x,
                sigma_xhat=_sig_hx_best,
                sigma_x_xhat=_sig_x_xhat_best,
                nseq=_qs.nseq, ntokens=_qs.ntokens,
            )

            for _w in _ffn_weights:
                _mn = _module_names[_w]
                _wn = _weight_names[_w]

                # Rate controller: suggest_target_x is stateless, update consumes budget
                _target_x = None
                _rc_info_w = None
                if rate_ctrl is not None and cfg.zsic is not None:
                    # Use unquantized Σ_X for dead dim detection when available
                    _dead_mask_w = find_dead_dimensions(_sigma_x_for_dead, threshold_ratio=cfg.zsic.dead_dim_threshold)
                    _n_dead_w = int(_dead_mask_w.sum())
                    _dead_p = 0
                    if _n_dead_w > 0:
                        _a_w = _mods[_mn].weight.shape[0]
                        _dead_p = _a_w * _n_dead_w
                    _target_x, _rc_info_w = rate_ctrl.suggest_target_x(_mn, dead_params=_dead_p)
                    _zcfg_final = replace(cfg.zsic,
                                          target_rate_bits=float(_target_x),
                                          qronos=use_qronos_mode,
                                          residual_compensation=False,
                                          rate_control_active=True)
                    print(f"[w1w3-qronos-adapt] {_mn}: rate target={_target_x:.4f}", flush=True)
                else:
                    _zcfg_final = replace(cfg.zsic,
                                          qronos=use_qronos_mode,
                                          residual_compensation=False) if cfg.zsic is not None else zsic_cfg_layer

                _art = quantize_one_layer(
                    model=model, module_name=_mn, weight_name=_wn,
                    H=_H_ffn, cfg=cfg, zsic_cfg=_zcfg_final,
                    Sig_X=_mixed_best.sigma_x if qronos_enabled else None,
                    Sig_hX=_mixed_best.sigma_xhat if qronos_enabled else None,
                    Sig_X_hX=_mixed_best.sigma_x_xhat if qronos_enabled else None,
                    Sig_delta_R_Xhat=None,
                    Sig_X_for_dead=_sigma_x_for_dead,
                    global_nrows=_global_nrows_per_w[_w],
                )

                # Save artifact + manifest
                _save_artifact_and_manifest(_art, _mn, run_dir, manifest, manifest_path,
                                            dist_rank=dist_rank, dist_world_size=dist_world_size)

                # Apply to model
                apply_layer_artifact(model, _art)

                # inf/NaN check
                _w_applied = _mods[_mn].weight.data
                _n_inf = int(_w_applied.isinf().sum())
                _n_nan = int(_w_applied.isnan().sum())
                if _n_inf > 0 or _n_nan > 0:
                    print(f"[WARNING] {_mn}: dequantized weight has {_n_inf} inf, {_n_nan} NaN "
                          f"(max={_w_applied.abs().max():.2e}, shape={list(_w_applied.shape)})", flush=True)

                # Zero out rows if configured
                if cfg.zero_out_rows:
                    _zero_key = f"{layer_id}.{_w}"
                    _dims_to_zero = cfg.zero_out_rows.get(_zero_key, [])
                    if _dims_to_zero:
                        _method = cfg.method.lower()
                        _handled = (_method in ("zsic", "sic") and _w.lower() != "wo")
                        if _handled:
                            print(f"[zero_out] {_mn}: skipping post-quant zeroing (handled by dead row/col exclusion)", flush=True)
                        else:
                            _mod = _mods[_mn]
                            with torch.no_grad():
                                for _row_idx in _dims_to_zero:
                                    _mod.weight.data[_row_idx, :] = 0
                                print(f"[zero_out] zeroed rows {list(_dims_to_zero)} in {_mn}", flush=True)

                # Update rate controller
                _n_dead_art = int(_art.payload.get("n_dead", 0))
                if _n_dead_art > 0:
                    _n_original = int(_art.payload.get("n_original", _art.shape[1]))
                    _n_live = int(_art.payload.get("n_live", _n_original))
                    _entropy = float(_art.payload.get("entropy", 0.0))
                    _a = _art.shape[0]
                    _actual_rate = (_entropy * _n_live + 16 + 16 * _n_live / _a + 16 * _n_dead_art / _a) / _n_original
                else:
                    _actual_rate = float(_art.payload.get("entropy", 0.0)) + float(_art.payload.get("rate_overhead", 0.0))
                if rate_ctrl is not None:
                    _bs_target = _art.payload.get("binary_search_target_used", None)
                    if _bs_target is not None:
                        rate_ctrl.update(_mn, target_x=float(_bs_target), actual_rate=float(_actual_rate))
                    elif _target_x is not None:
                        rate_ctrl.update(_mn, target_x=float(_target_x), actual_rate=float(_actual_rate))
                    rate_ctrl.save_json(str(run_dir / "rate_control_state.json"))

                # Log line
                _line = {
                    "module": _mn,
                    "weight": _wn,
                    "shape": list(_art.shape),
                    "method": _art.method,
                    "nseq": int(nseq_used),
                    "ntokens": int(ntokens_used),
                    "actual_rate": float(_actual_rate),
                    "target_rate_bits_used": None if _target_x is None else float(_target_x),
                    "rate_control": _rc_info_w,
                    "payload": {
                        "loss": float(_art.payload.get("loss", 0.0)),
                        "entropy": float(_art.payload.get("entropy", 0.0)),
                        "rate_overhead": float(_art.payload.get("rate_overhead", 0.0)),
                        "n_dead": int(_art.payload.get("n_dead", 0)),
                        "qronos": bool(_art.payload.get("qronos", False)),
                        "binary_search_target_used": _art.payload.get("binary_search_target_used", None),
                        "binary_search_desired": _art.payload.get("binary_search_desired", None),
                        "binary_search_final_diff": _art.payload.get("binary_search_final_diff", None),
                        "w1w3_qronos_adapt_eps": _best_eps,
                    },
                    "ts": time.time(),
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(_line) + "\n")

                print(f"[w1w3-qronos-adapt] {_mn}: quantized with eps={_best_eps:.4f}", flush=True)

            # Skip normal post-processing (already handled above for both weights).
            # Advance cache if this was the last weight in the block.
            weights_in_block = layer_weights_map.get(layer_id, [])
            is_last_weight_in_block = (weight == weights_in_block[-1]) if weights_in_block else False
            if is_last_weight_in_block and cache.current_block_idx == layer_id:
                print(f"[cache] advancing through block {layer_id} (all weights quantized)", flush=True)
                cache.advance_through_block(layer_id, batch_size=cfg.hessian_batch_size)
                if cache_unquant is not None:
                    print(f"[unquant-cache] advancing through block {layer_id}", flush=True)
                    cache_unquant.advance_through_block(layer_id, batch_size=cfg.hessian_batch_size)
                _attn_importance_cache.pop(layer_id, None)
                _qronos_stats_cache.pop((layer_id, "attn_qkv"), None)
                _qronos_stats_cache.pop((layer_id, "ffn_w1w3"), None)
                _qronos_stats_w_cache.pop((layer_id, "attn_qkv"), None)
                _qronos_stats_w_cache.pop((layer_id, "ffn_w1w3"), None)

            torch.cuda.empty_cache()
            continue
        else:
            # Extract tensor references for the call, then free float64 originals
            # from dataclasses for non-cached weights (wo, w2) to reduce peak GPU memory.
            _sig_x_arg = qronos_stats.sigma_x if qronos_enabled and qronos_stats is not None else None
            _sig_hx_arg = qronos_stats.sigma_xhat if qronos_enabled and qronos_stats is not None else None
            _sig_x_hx_arg = qronos_stats.sigma_x_xhat if qronos_enabled and qronos_stats is not None else None
            _sig_dr_arg = residual_stats.sigma_delta_r_xhat if residual_comp_enabled and residual_stats is not None else None
            if qronos_stats is not None and _qronos_group(weight) is None:
                # Non-cached weight (wo/w2): free float64 originals from dataclass
                qronos_stats.sigma_x = qronos_stats.sigma_xhat = qronos_stats.sigma_x_xhat = None
            if residual_stats is not None:
                residual_stats.sigma_delta_r_xhat = None

            # When Sig_X is available (Qronos), H is only needed as fallback.
            # Free it to save 6+ GiB (critical for w2 with 28672² Hessian).
            _h_arg = H if _sig_x_arg is None else None
            if _sig_x_arg is not None and _sigma_x_for_dead is not H:
                del H  # Free float64 Hessian (sigma_x used for dead dims instead)

            # For w2 (RowParallel, input_dim=28672 or 14336 native): move covariance
            # matrices to CPU to free GPU memory. Applies to resharded row-parallel w2.
            _needs_cpu_offload = _is_row_parallel
            if _needs_cpu_offload and _sig_x_arg is not None:
                _sxd_aliases_sx = (_sigma_x_for_dead is _sig_x_arg)
                _sig_x_arg = _sig_x_arg.cpu()
                if _sxd_aliases_sx:
                    _sigma_x_for_dead = _sig_x_arg
                elif _sigma_x_for_dead is not None and _sigma_x_for_dead.is_cuda:
                    _sigma_x_for_dead = _sigma_x_for_dead.cpu()
                if _sig_hx_arg is not None:
                    _sig_hx_arg = _sig_hx_arg.cpu()
                if _sig_x_hx_arg is not None:
                    _sig_x_hx_arg = _sig_x_hx_arg.cpu()
                torch.cuda.empty_cache()
                print(f"[memory] moved covariance matrices to CPU for {weight} (freeing GPU for dead-dim slicing)", flush=True)

            art = quantize_one_layer(
                model=model,
                module_name=module_name,
                weight_name=weight_name,
                H=_h_arg,
                cfg=cfg,
                zsic_cfg=zsic_cfg_layer,
                Sig_X=_sig_x_arg,
                Sig_hX=_sig_hx_arg,
                Sig_X_hX=_sig_x_hx_arg,
                Sig_delta_R_Xhat=_sig_dr_arg,
                Sig_X_for_dead=_sigma_x_for_dead,
                global_nrows=_global_nrows,
            )

        # RowParallel post-processing: convert row-sharded artifact to column-sharded,
        # restore original column-sharded weight, then apply column-shard artifact.
        if _rp_ctx is not None:
            # Restore module weight to original column shard before artifact conversion
            _mod = dict(model.named_modules())[module_name]
            _mod.weight.data = _rp_ctx["orig_weight"]

            art = _exit_row_parallel_quant(art, _rp_ctx, dist_rank, dist_world_size)
            _codes_key = "Z" if "Z" in art.payload else "Qint"
            print(f"[row-parallel] {module_name}: converted artifact to column-shard "
                  f"({_codes_key}={list(art.payload[_codes_key].shape)}, shape={art.shape})", flush=True)
            _rp_ctx = None

        # Save artifact (all ranks) + manifest (rank 0 only).
        _save_artifact_and_manifest(art, module_name, run_dir, manifest, manifest_path,
                                    dist_rank=dist_rank, dist_world_size=dist_world_size)

        # Apply to model so subsequent Hessians see it.
        apply_layer_artifact(model, art)

        # Check for non-finite values in dequantized weight (inf/NaN detection).
        _w_applied = dict(model.named_modules())[module_name].weight.data
        _n_inf = int(_w_applied.isinf().sum())
        _n_nan = int(_w_applied.isnan().sum())
        if _n_inf > 0 or _n_nan > 0:
            print(f"[WARNING] {module_name}: dequantized weight has {_n_inf} inf, {_n_nan} NaN "
                  f"(max={_w_applied.abs().max():.2e}, shape={list(_w_applied.shape)})", flush=True)

        # Zero out specific rows/columns if configured (for outlier removal).
        # For ZSIC: w1/w3/wq/wk/wv dead rows and w2 forced dead cols are already
        # handled by dead row/col exclusion in compress_zsic — skip post-quant zeroing.
        # For other methods or wo: apply post-quant zeroing as before.
        if cfg.zero_out_rows:
            zero_key = f"{layer_id}.{weight}"
            dims_to_zero = cfg.zero_out_rows.get(zero_key, [])
            if dims_to_zero:
                # Check if handled by ZSIC dead row/col exclusion
                _method = cfg.method.lower()
                _handled = (_method in ("zsic", "sic") and weight.lower() != "wo")
                # For w2, check if w1/w3 dead rows propagated as forced dead cols
                if _method in ("zsic", "sic") and weight.lower() == "w2":
                    _w1_dims = cfg.zero_out_rows.get(f"{layer_id}.w1", [])
                    _w3_dims = cfg.zero_out_rows.get(f"{layer_id}.w3", [])
                    _handled = bool(set(_w1_dims) | set(_w3_dims))

                if _handled:
                    print(f"[zero_out] {module_name}: skipping post-quant zeroing (handled by dead row/col exclusion)", flush=True)
                else:
                    module = dict(model.named_modules())[module_name]
                    with torch.no_grad():
                        if weight.lower() == "w2":
                            for col_idx in dims_to_zero:
                                module.weight.data[:, col_idx] = 0
                            print(f"[zero_out] zeroed columns {list(dims_to_zero)} in {module_name}", flush=True)
                        else:
                            for row_idx in dims_to_zero:
                                module.weight.data[row_idx, :] = 0
                            print(f"[zero_out] zeroed rows {list(dims_to_zero)} in {module_name}", flush=True)

        # Update rate controller with achieved rate.
        # Must account for dead dimensions: rate is per-original-element, not per-live-element
        n_dead = int(art.payload.get("n_dead", 0))
        if n_dead > 0:
            # Dead dims: recompute rate per original element
            n_original = int(art.payload.get("n_original", art.shape[1]))
            n_live = int(art.payload.get("n_live", n_original))
            entropy = float(art.payload.get("entropy", 0.0))
            a = art.shape[0]
            # total_bits = entropy * a * n_live + 16*a + 16*n_live + 16*n_dead
            # actual_rate = total_bits / (a * n_original)
            actual_rate = (entropy * n_live + 16 + 16 * n_live / a + 16 * n_dead / a) / n_original
        else:
            actual_rate = float(art.payload.get("entropy", 0.0)) + float(art.payload.get("rate_overhead", 0.0))
        if rate_ctrl is not None:
            # When binary search was used, get the actual target it found
            bs_target = art.payload.get("binary_search_target_used", None)
            if bs_target is not None:
                rate_ctrl.update(module_name, target_x=float(bs_target), actual_rate=float(actual_rate))
            elif target_x_used is not None:
                rate_ctrl.update(module_name, target_x=float(target_x_used), actual_rate=float(actual_rate))
            rate_ctrl.save_json(str(run_dir / "rate_control_state.json"))

        # Append per-layer log line.
        line = {
            "module": module_name,
            "weight": weight_name,
            "shape": list(art.shape),
            "method": art.method,
            "nseq": int(nseq_used),
            "ntokens": int(ntokens_used),
            "actual_rate": float(actual_rate),
            "target_rate_bits_used": None if target_x_used is None else float(target_x_used),
            "rate_control": rc_info,
            "payload": {
                "loss": float(art.payload.get("loss", 0.0)),
                "entropy": float(art.payload.get("entropy", 0.0)),
                "rate_overhead": float(art.payload.get("rate_overhead", 0.0)),
                "n_dead": int(art.payload.get("n_dead", 0)),
                "qronos": bool(art.payload.get("qronos", False)),
                "binary_search_target_used": art.payload.get("binary_search_target_used", None),
                "binary_search_desired": art.payload.get("binary_search_desired", None),
                "binary_search_final_diff": art.payload.get("binary_search_final_diff", None),
            },
            "ts": time.time(),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(line) + "\n")

        # Restore KV caches to full batch size before advance_through_block.
        if _kv_resized:
            from quant_layerwise.hessian_runtime import _resize_kv_caches
            _resize_kv_caches(model, cfg.hessian_batch_size)
            _kv_resized = False
            print(f"[kv-cache] restored to batch_size={cfg.hessian_batch_size}", flush=True)

        # Advance cache through this block if we've processed all weights in it.
        # This propagates activations through the now-quantized block.
        weights_in_block = layer_weights_map.get(layer_id, [])
        is_last_weight_in_block = (weight == weights_in_block[-1]) if weights_in_block else False
        if is_last_weight_in_block and cache.current_block_idx == layer_id:
            print(f"[cache] advancing through block {layer_id} (all weights quantized)", flush=True)
            cache.advance_through_block(layer_id, batch_size=cfg.hessian_batch_size)
            # Qronos: also advance unquantized cache
            if cache_unquant is not None:
                print(f"[unquant-cache] advancing through block {layer_id}", flush=True)
                cache_unquant.advance_through_block(layer_id, batch_size=cfg.hessian_batch_size)
            # Evict attention importance cache for this layer (no longer needed)
            _attn_importance_cache.pop(layer_id, None)
            _qronos_stats_cache.pop((layer_id, "attn_qkv"), None)
            _qronos_stats_cache.pop((layer_id, "ffn_w1w3"), None)
            _qronos_stats_w_cache.pop((layer_id, "attn_qkv"), None)
            _qronos_stats_w_cache.pop((layer_id, "ffn_w1w3"), None)

        torch.cuda.empty_cache()

    # Report resulting rate after the model (requested layers) is done.
    # Only rank 0 computes summary (reads rank-0 artifacts from manifest).
    if is_rank0:
        rate_summary = compute_run_rate_summary(run_dir, manifest)
        (run_dir / "rate_summary.json").write_text(json.dumps(rate_summary, indent=2))
        print(
            "\n[done] run_dir={}  avg_rate_bits_per_param={:.4f}  total_params={}".format(
                run_dir,
                float(rate_summary.get("avg_rate_bits_per_param", float("nan"))),
                int(rate_summary.get("total_params", 0)),
            )
        )

        # Generate activation MSE plot if requested and stats were collected
        if cfg.plot_activation_mse and (cfg.qronos or cfg.collect_qronos_stats):
            qronos_dir = run_dir / "qronos_stats"
            if qronos_dir.exists() and any(qronos_dir.glob("*.pkl")):
                try:
                    from scripts.plot_activation_mse import plot_activation_mse
                    plot_path = run_dir / "activation_mse.png"
                    print("\n[plot] generating activation MSE plot...", flush=True)
                    plot_activation_mse(
                        run_dir=run_dir,
                        output_path=plot_path,
                        title=f"Activation Drift: {cfg.model_name} ({cfg.method}, r={rate_summary.get('avg_rate_bits_per_param', 0):.2f})",
                        show_correlation=False,  # Single panel scatter plot
                    )
                    print(f"[plot] saved to {plot_path}", flush=True)
                except Exception as e:
                    print(f"[plot] warning: failed to generate plot: {e}", flush=True)
            else:
                print("[plot] no qronos_stats found, skipping activation MSE plot", flush=True)

    # Final barrier so all ranks finish before returning.
    if dist_world_size > 1 and _dist is not None and _dist.is_initialized():
        _dist.barrier()

    return run_dir


def compute_run_rate_summary(run_dir: Path, manifest: RunManifest) -> Dict[str, Any]:
    total_bits = 0.0
    total_params = 0
    per_layer = []
    ws = manifest.world_size  # account for sharded artifacts

    for module_name in sorted(manifest.artifacts.keys()):
        relpath = manifest.artifacts[module_name]
        art = LayerArtifact.load(run_dir / relpath, map_location="cpu")

        # Shape is per-shard. Scale by world_size for total params.
        a, n = (int(art.shape[0]), int(art.shape[1]))
        numel = a * n

        entropy = float(art.payload.get("entropy", 0.0))
        overhead = float(art.payload.get("rate_overhead", 0.0))
        n_dead = int(art.payload.get("n_dead", 0))

        # Account for dead dimensions when computing rate
        if n_dead > 0:
            n_original = int(art.payload.get("n_original", n))
            n_live = int(art.payload.get("n_live", n_original))
            # total_bits = entropy * a * n_live + 16*a + 16*n_live + 16*n_dead
            # rate = total_bits / (a * n_original)
            rate = (entropy * n_live + 16 + 16 * n_live / a + 16 * n_dead / a) / n_original
        else:
            rate = float(entropy + overhead)

        # Rate is bits-per-param (same across shards), but numel must be scaled
        full_numel = numel * ws
        total_bits += rate * float(full_numel)
        total_params += int(full_numel)

        per_layer.append(
            {
                "module": module_name,
                "method": art.method,
                "shape": [a, n],
                "entropy": entropy,
                "rate_overhead": overhead,
                "rate": rate,
                "n_dead": n_dead,
                "numel": int(full_numel),
                "world_size": ws,
            }
        )

    avg_rate = total_bits / float(total_params) if total_params > 0 else None

    return {
        "model_name": manifest.model_name,
        "method": manifest.method,
        "run_id": manifest.run_id,
        "n_layers_quantized": int(len(manifest.artifacts)),
        "total_params": int(total_params),
        "total_bits": float(total_bits),
        "avg_rate_bits_per_param": None if avg_rate is None else float(avg_rate),
        "per_layer": per_layer,
    }


def build_layers(
    *,
    layer_ids: Iterable[int],
    weights: Sequence[str] = ("wq", "wk", "wv", "wo", "w1", "w3", "w2"),  # w2 last so Hessian sees quantized w1,w3
) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for lid in layer_ids:
        for w in weights:
            out.append((int(lid), str(w)))
    return out
