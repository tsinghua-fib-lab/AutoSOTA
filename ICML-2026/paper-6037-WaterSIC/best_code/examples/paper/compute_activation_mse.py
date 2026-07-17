"""Compute per-layer activation MSE between unquantized and quantized models.

Loads quantized model artifacts and computes ||X - X̂|| / ||X|| for each layer
by running inference. No pre-computed qronos_stats needed.

Usage:
    # Compare WaterSICR (with residual compensation) vs WaterSIC (without)
    python scripts/compute_activation_mse.py \
        --run_a /path/to/rescomp_run \
        --run_b /path/to/no_rescomp_run \
        --label_a "WaterSICR" \
        --label_b "WaterSIC" \
        --output watersicr_vs_watersic.png \
        --init_dist
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_layerwise.data import get_wikitext2, split_dataset, take_nseq
from quant_layerwise.names import get_hess_name
from quant_layerwise.partial_model import load_and_apply_manifest
from quant_layerwise.pipeline import ensure_single_process_distributed, load_model_and_tokenizer
from quant_layerwise.storage import RunManifest


def apply_zero_out_rows(model: torch.nn.Module, zero_out_rows: str):
    """Parse and apply zero_out_rows spec to a model.

    Format: "6.w1:5723,8518;6.w3:5723,8518;16.w1:2271,1875"
    For w1/w3/wq/wk/wv: zeros rows. For w2/wo: zeros columns.
    """
    modules = dict(model.named_modules())
    for item in zero_out_rows.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid zero_out_rows format: '{item}'")
        key = parts[0].strip()
        rows = [int(r.strip()) for r in parts[1].split(",") if r.strip()]
        layer_id_str, weight = key.split(".")
        module_name = get_hess_name(int(layer_id_str), weight)
        if module_name not in modules:
            print(f"[zero_out] warning: module '{module_name}' not found", flush=True)
            continue
        module = modules[module_name]
        with torch.no_grad():
            if weight.lower() in ("w2", "wo"):
                for col_idx in rows:
                    module.weight.data[:, col_idx] = 0
                print(f"[zero_out] zeroed columns {rows} in {module_name}", flush=True)
            else:
                for row_idx in rows:
                    module.weight.data[row_idx, :] = 0
                print(f"[zero_out] zeroed rows {rows} in {module_name}", flush=True)


def get_layer_modules(model) -> Dict[str, torch.nn.Module]:
    """Get all weight modules we want to track activations for."""
    modules = {}
    for layer_idx, layer in enumerate(model.layers):
        modules[f"layers.{layer_idx}.attention_norm"] = layer.attention_norm
        modules[f"layers.{layer_idx}.ffn_norm"] = layer.ffn_norm
        modules[f"layers.{layer_idx}.attention.wq"] = layer.attention.wq
        modules[f"layers.{layer_idx}.attention.wk"] = layer.attention.wk
        modules[f"layers.{layer_idx}.attention.wv"] = layer.attention.wv
        modules[f"layers.{layer_idx}.attention.wo"] = layer.attention.wo
        modules[f"layers.{layer_idx}.feed_forward.w1"] = layer.feed_forward.w1
        modules[f"layers.{layer_idx}.feed_forward.w2"] = layer.feed_forward.w2
        modules[f"layers.{layer_idx}.feed_forward.w3"] = layer.feed_forward.w3
    return modules


class ActivationCapture:
    """Hook to capture input activations for a module.

    When gpu=True, activations stay on GPU (fast, no sync). When gpu=False,
    activations are moved to CPU (blocking sync per hook, but uses less GPU memory).
    """

    def __init__(self, gpu: bool = False):
        self.activations: List[torch.Tensor] = []
        self.handle = None
        self.gpu = gpu

    def hook(self, module, input, output):
        x = input[0].detach()
        self.activations.append(x.clone() if self.gpu else x.cpu())

    def register(self, module: torch.nn.Module):
        self.handle = module.register_forward_hook(self.hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def get_concatenated(self) -> torch.Tensor:
        if not self.activations:
            return None
        return torch.cat(self.activations, dim=0)

    def clear(self):
        self.activations = []


class CompareCapture:
    """Hook that computes MSE/cosine metrics on-the-fly against a stored reference.

    Avoids storing quant activations entirely — metrics are computed inside the
    hook during the quant model's forward pass, and the reference is freed
    immediately after.
    """

    def __init__(self, ref_activation: torch.Tensor):
        self.ref = ref_activation  # GPU or CPU tensor
        self.handle = None
        self.mse_num = 0.0
        self.ref_norm_sq = 0.0
        self.dot = 0.0
        self.q_norm_sq = 0.0

    def hook(self, module, input, output):
        x_q = input[0].detach()
        x_ref = self.ref
        # Ensure same device
        if x_ref.device != x_q.device:
            x_ref = x_ref.to(x_q.device)
        x_q = x_q.flatten().double()
        x_ref = x_ref.flatten().double()
        diff = x_ref - x_q
        self.mse_num += torch.sum(diff * diff).item()
        self.ref_norm_sq += torch.sum(x_ref * x_ref).item()
        self.dot += torch.sum(x_ref * x_q).item()
        self.q_norm_sq += torch.sum(x_q * x_q).item()
        # Free ref immediately — this module is done
        self.ref = None
        del x_ref, x_q, diff

    def register(self, module: torch.nn.Module):
        self.handle = module.register_forward_hook(self.hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def _resize_kv_cache(model, batch_size: int):
    """Resize KV caches to fit batch_size if needed."""
    if hasattr(model, "resize_kv_caches"):
        model.resize_kv_caches(batch_size)
    elif hasattr(model, "layers") and hasattr(model.layers[0], "attention"):
        for layer in model.layers:
            attn = layer.attention
            if hasattr(attn, "cache_k") and attn.cache_k.shape[0] < batch_size:
                old_shape = attn.cache_k.shape
                new_shape = (batch_size, old_shape[1], old_shape[2], old_shape[3])
                attn.cache_k = torch.zeros(new_shape, device=attn.cache_k.device, dtype=attn.cache_k.dtype)
                attn.cache_v = torch.zeros(new_shape, device=attn.cache_v.device, dtype=attn.cache_v.dtype)


def parse_layer_name(name: str) -> Tuple[int, str, str]:
    parts = name.split(".")
    if len(parts) >= 4 and parts[0] == "layers":
        layer_id = int(parts[1])
        block_type = parts[2]
        weight_type = parts[3]
        return layer_id, block_type, weight_type
    elif len(parts) == 3 and parts[0] == "layers":
        # attention_norm, ffn_norm
        return int(parts[1]), "residual", parts[2]
    return -1, "", name


def get_sort_key(name: str) -> Tuple[int, int, int]:
    layer_id, block_type, weight_type = parse_layer_name(name)
    if block_type == "residual":
        if weight_type == "attention_norm":
            return (layer_id, -1, 0)  # before attention block
        elif weight_type == "ffn_norm":
            return (layer_id, 0, 99)  # between attention and ffn
    block_order = 0 if block_type == "attention" else 1
    weight_order_map = {"wq": 0, "wk": 1, "wv": 2, "wo": 3, "w1": 0, "w3": 1, "w2": 2}  # w2 last
    weight_order = weight_order_map.get(weight_type, 99)
    return (layer_id, block_order, weight_order)


def get_merged_weight_type(weight_type: str) -> str:
    if weight_type in ("wq", "wk", "wv"):
        return "qkv"
    elif weight_type in ("w1", "w3"):
        return "w1w3"
    elif weight_type in ("attention_norm", "ffn_norm"):
        return weight_type  # no merging
    return weight_type


def get_weight_type_color(weight_type: str) -> str:
    color_map = {
        "qkv": "tab:blue", "wo": "tab:cyan",
        "w1w3": "tab:orange", "w2": "tab:green",
        "attention_norm": "tab:red", "ffn_norm": "tab:purple",
    }
    return color_map.get(weight_type, "tab:gray")


def get_cache_path(run_dir: Path, nsamples: int, seqlen: int) -> Path:
    """Get cache file path for a run's metrics."""
    return run_dir / f"activation_metrics_cache_n{nsamples}_s{seqlen}.pt"


def load_cached_metrics(run_dir: Path, nsamples: int, seqlen: int) -> Optional[Tuple[Dict, Dict]]:
    """Load cached metrics if available."""
    cache_path = get_cache_path(run_dir, nsamples, seqlen)
    if cache_path.exists():
        try:
            data = torch.load(cache_path, map_location="cpu", weights_only=False)
            print(f"  Loaded cached metrics from {cache_path.name}")
            return data["mse"], data["cos_sim"]
        except Exception as e:
            print(f"  Cache load failed: {e}")
    return None


def save_cached_metrics(run_dir: Path, nsamples: int, seqlen: int, mse: Dict, cos_sim: Dict):
    """Save metrics to cache."""
    cache_path = get_cache_path(run_dir, nsamples, seqlen)
    torch.save({"mse": mse, "cos_sim": cos_sim}, cache_path)
    print(f"  Saved metrics cache to {cache_path.name}")


def _finalize_metrics(
    running_stats: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Convert running stats to final relative-MSE and cosine-similarity dicts."""
    mse_results = {}
    cos_sim_results = {}
    for name, stats in running_stats.items():
        if stats["ref_norm_sq"] > 0:
            mse_results[name] = stats["mse_num"] / stats["ref_norm_sq"]
        else:
            mse_results[name] = 0.0

        norm_ref = np.sqrt(stats["ref_norm_sq"])
        norm_q = np.sqrt(stats["q_norm_sq"])
        if norm_ref > 0 and norm_q > 0:
            cos_sim_results[name] = stats["dot"] / (norm_ref * norm_q)
        else:
            cos_sim_results[name] = 1.0

    return mse_results, cos_sim_results


# ---------------------------------------------------------------------------
# Sequential: one model on GPU at a time (for large models like 70B)
# ---------------------------------------------------------------------------


def compute_metrics_sequential(
    model_name: str,
    run_dir: Path,
    local_rank: int = 0,
    batch_size: int = 4,
    nsamples: int = 64,
    seqlen: int = 2048,
    use_cache: bool = True,
    zero_out_rows: str = "",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute per-layer metrics loading one model at a time.

    Phase 1: load ref model, forward all batches, capture activations to CPU.
    Phase 2: free ref, load quant model, forward all batches with CompareCapture.

    Peak GPU = one model.  Peak CPU = all ref activations (nsamples * ~25GB for 70B).
    Use small nsamples (4-16) to keep CPU memory reasonable.
    """
    if use_cache:
        cached = load_cached_metrics(run_dir, nsamples, seqlen)
        if cached is not None:
            return cached

    is_rank0 = local_rank == 0

    # --- Phase 1: ref model → capture activations to CPU ---
    if is_rank0:
        print("  Phase 1: loading reference model...")
    model_ref, tokenizer = load_model_and_tokenizer(model_name, local_rank=local_rank)

    if is_rank0:
        print(f"  Preparing eval data (nsamples={nsamples}, seqlen={seqlen})...")
    eval_tokens = split_dataset(get_wikitext2(tokenizer, split="test"), seqlen)
    eval_tokens = take_nseq(eval_tokens, nsamples)
    n_total = eval_tokens.shape[0]
    n_batches = (n_total + batch_size - 1) // batch_size
    if is_rank0:
        print(f"  Using {n_total} samples, {n_batches} batches of {batch_size}")
    if zero_out_rows:
        apply_zero_out_rows(model_ref, zero_out_rows)
    model_ref.eval()
    modules_ref = get_layer_modules(model_ref)
    module_names = list(modules_ref.keys())
    device = next(model_ref.parameters()).device
    _resize_kv_cache(model_ref, batch_size)

    # Store ref activations on CPU: list of {module_name: cpu_tensor}
    ref_acts_per_batch = []
    if is_rank0:
        print(f"  Phase 1: capturing ref activations ({n_batches} batches, {len(module_names)} modules)...")

    with torch.no_grad():
        for bi_start in range(0, n_total, batch_size):
            batch = eval_tokens[bi_start : bi_start + batch_size].to(device)
            caps = {name: ActivationCapture(gpu=False) for name in module_names}
            for name in module_names:
                caps[name].register(modules_ref[name])
            t0 = time.time()
            _ = model_ref(batch, start_pos=0)
            t_fwd = time.time() - t0
            batch_acts = {}
            for name in module_names:
                caps[name].remove()
                batch_acts[name] = caps[name].get_concatenated()
                caps[name].clear()
            ref_acts_per_batch.append(batch_acts)
            del caps
            bi = bi_start // batch_size + 1
            if is_rank0:
                print(f"    Ref batch {bi}/{n_batches}: {t_fwd:.1f}s")

    del modules_ref, model_ref
    import gc; gc.collect()
    torch.cuda.empty_cache()
    if is_rank0:
        print("  Phase 1 done. Freed ref model from GPU.")

    # --- Phase 2: quant model → CompareCapture against CPU ref acts ---
    if is_rank0:
        print(f"  Phase 2: loading quantized model from {run_dir.name}...")
    model_q, _ = load_model_and_tokenizer(model_name, local_rank=local_rank)
    device = next(model_q.parameters()).device
    from fairscale.nn.model_parallel.initialize import get_model_parallel_rank
    mp_rank = get_model_parallel_rank()
    load_and_apply_manifest(model_q, run_dir, rank=mp_rank, map_location=f"cuda:{local_rank}")
    if zero_out_rows:
        apply_zero_out_rows(model_q, zero_out_rows)
    model_q.eval()
    modules_q = get_layer_modules(model_q)
    _resize_kv_cache(model_q, batch_size)

    running_stats = {
        name: {"mse_num": 0.0, "ref_norm_sq": 0.0, "dot": 0.0, "q_norm_sq": 0.0}
        for name in module_names
    }

    if is_rank0:
        print(f"  Phase 2: computing metrics ({n_batches} batches)...")

    with torch.no_grad():
        for bi_idx, bi_start in enumerate(range(0, n_total, batch_size)):
            batch = eval_tokens[bi_start : bi_start + batch_size].to(device)
            batch_ref_acts = ref_acts_per_batch[bi_idx]

            cmp_caps = {}
            for name in module_names:
                cmp_caps[name] = CompareCapture(batch_ref_acts[name])
                cmp_caps[name].register(modules_q[name])

            t0 = time.time()
            _ = model_q(batch, start_pos=0)
            t_fwd = time.time() - t0

            for name in module_names:
                cmp_caps[name].remove()
                running_stats[name]["mse_num"] += cmp_caps[name].mse_num
                running_stats[name]["ref_norm_sq"] += cmp_caps[name].ref_norm_sq
                running_stats[name]["dot"] += cmp_caps[name].dot
                running_stats[name]["q_norm_sq"] += cmp_caps[name].q_norm_sq
            del cmp_caps, batch_ref_acts
            ref_acts_per_batch[bi_idx] = None  # free as we go

            bi = bi_idx + 1
            if is_rank0:
                print(f"    Quant batch {bi}/{n_batches}: {t_fwd:.1f}s")

    mse_results, cos_sim_results = _finalize_metrics(running_stats)

    del model_q
    torch.cuda.empty_cache()

    if use_cache:
        save_cached_metrics(run_dir, nsamples, seqlen, mse_results, cos_sim_results)

    return mse_results, cos_sim_results


# ---------------------------------------------------------------------------
# Core: ref + quant on GPU together, batch-by-batch, metrics on CPU
# ---------------------------------------------------------------------------


def compute_metrics_for_run(
    model_ref,
    run_dir: Path,
    eval_tokens: torch.Tensor,
    local_rank: int = 0,
    batch_size: int = 32,
    nsamples: int = 64,
    seqlen: int = 2048,
    use_cache: bool = True,
    zero_out_rows: str = "",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute per-layer metrics for a single quantized run vs reference model.

    Memory-efficient: captures ref activations to CPU (~85 GB for 8B model),
    then computes metrics on-the-fly during the quant forward pass using
    CompareCapture hooks.  Quant activations are never stored — metrics are
    computed inside the hook and ref tensors are freed progressively as each
    layer executes.  Peak CPU memory ≈ ref activations only.
    """
    if use_cache:
        cached = load_cached_metrics(run_dir, nsamples, seqlen)
        if cached is not None:
            return cached

    manifest = RunManifest.load(run_dir / "manifest.json")

    print(f"  Loading quantized model from {run_dir.name}...")
    model_q, _ = load_model_and_tokenizer(manifest.model_name, local_rank=local_rank)
    load_and_apply_manifest(model_q, run_dir)

    if zero_out_rows:
        apply_zero_out_rows(model_q, zero_out_rows)

    modules_ref = get_layer_modules(model_ref)
    modules_q = get_layer_modules(model_q)
    module_names = list(modules_ref.keys())

    model_q.eval()
    model_ref.eval()
    device = next(model_q.parameters()).device

    _resize_kv_cache(model_ref, batch_size)
    _resize_kv_cache(model_q, batch_size)

    running_stats = {
        name: {"mse_num": 0.0, "ref_norm_sq": 0.0, "dot": 0.0, "q_norm_sq": 0.0}
        for name in module_names
    }

    print(f"  Computing metrics batch-by-batch ({len(module_names)} modules)...")
    n_total = eval_tokens.shape[0]
    n_batches = (n_total + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(0, n_total, batch_size):
            batch = eval_tokens[batch_idx : batch_idx + batch_size].to(device)

            # Step 1: Ref forward — capture activations on GPU (no CPU sync!)
            ref_caps = {name: ActivationCapture(gpu=True) for name in module_names}
            for name in module_names:
                ref_caps[name].register(modules_ref[name])
            t_fwd = time.time()
            _ = model_ref(batch, start_pos=0)
            t_ref = time.time() - t_fwd
            for name in module_names:
                ref_caps[name].remove()

            # Step 2: Quant forward — CompareCapture hooks compute metrics
            # on GPU against stored ref, freeing ref as each layer runs.
            # No quant activations are stored.
            cmp_caps = {}
            for name in module_names:
                ref_act = ref_caps[name].get_concatenated()
                ref_caps[name].clear()
                cmp_caps[name] = CompareCapture(ref_act)
                cmp_caps[name].register(modules_q[name])
            del ref_caps

            t_fwd = time.time()
            _ = model_q(batch, start_pos=0)
            t_quant = time.time() - t_fwd

            for name in module_names:
                cmp_caps[name].remove()
                running_stats[name]["mse_num"] += cmp_caps[name].mse_num
                running_stats[name]["ref_norm_sq"] += cmp_caps[name].ref_norm_sq
                running_stats[name]["dot"] += cmp_caps[name].dot
                running_stats[name]["q_norm_sq"] += cmp_caps[name].q_norm_sq
            del cmp_caps

            bi = batch_idx // batch_size + 1
            print(f"    Batch {bi}/{n_batches}: ref_fwd={t_ref:.1f}s quant_fwd={t_quant:.1f}s")

    mse_results, cos_sim_results = _finalize_metrics(running_stats)

    del model_q
    torch.cuda.empty_cache()

    if use_cache:
        save_cached_metrics(run_dir, nsamples, seqlen, mse_results, cos_sim_results)

    return mse_results, cos_sim_results


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------


def plot_comparison(
    run_a: Path,
    run_b: Path,
    output_path: Path = None,
    label_a: str = None,
    label_b: str = None,
    title: str = None,
    figsize: Tuple[float, float] = (22, 7),
    merge_inputs: bool = True,
    nsamples: int = 64,
    batch_size: int = 32,
    seqlen: int = 2048,
    local_rank: int = 0,
    init_dist: bool = False,
    master_port_base: int = 29500,
    use_cache: bool = True,
    zero_out_rows: str = "",
):
    """Compare per-layer activation MSE between two quantized runs.

    For each run: loads ref + quant model on GPU together, runs one batch at
    a time (ref forward → quant forward → compare on CPU → free activations).
    Only a few GB of CPU memory used at any time.
    """

    if init_dist:
        ensure_single_process_distributed(local_rank=local_rank, master_port=master_port_base)

    manifest_a = RunManifest.load(run_a / "manifest.json")
    manifest_b = RunManifest.load(run_b / "manifest.json")

    if manifest_a.model_name != manifest_b.model_name:
        raise ValueError(f"Different models: {manifest_a.model_name} vs {manifest_b.model_name}")

    model_name = manifest_a.model_name
    if label_a is None:
        label_a = run_a.name
    if label_b is None:
        label_b = run_b.name

    print("Computing activation MSE comparison:")
    print(f"  Run A ({label_a}): {run_a}")
    print(f"  Run B ({label_b}): {run_b}")
    print(f"  Model: {model_name}")

    # Check caches
    cached_a = load_cached_metrics(run_a, nsamples, seqlen) if use_cache else None
    cached_b = load_cached_metrics(run_b, nsamples, seqlen) if use_cache else None

    if cached_a is not None and cached_b is not None:
        print("Both runs have cached metrics, skipping model loading.")
        mse_a, cos_sim_a = cached_a
        mse_b, cos_sim_b = cached_b
    else:
        # Load reference model (stays on GPU for all runs)
        print("\nLoading reference (unquantized) model...")
        model_ref, tokenizer = load_model_and_tokenizer(model_name, local_rank=local_rank)

        if zero_out_rows:
            apply_zero_out_rows(model_ref, zero_out_rows)

        print(f"Preparing evaluation data (nsamples={nsamples}, seqlen={seqlen})...")
        eval_tokens = split_dataset(get_wikitext2(tokenizer, split="test"), seqlen)
        eval_tokens = take_nseq(eval_tokens, nsamples)
        print(f"Using {eval_tokens.shape[0]} samples")

        # Compute metrics for each run (ref + quant on GPU, batch by batch)
        if cached_a is not None:
            mse_a, cos_sim_a = cached_a
        else:
            print(f"\nComputing metrics for {label_a}...")
            t0 = time.time()
            mse_a, cos_sim_a = compute_metrics_for_run(
                model_ref, run_a, eval_tokens, local_rank, batch_size, nsamples, seqlen, use_cache,
                zero_out_rows=zero_out_rows,
            )
            print(f"  {label_a} done in {time.time() - t0:.1f}s")

        if cached_b is not None:
            mse_b, cos_sim_b = cached_b
        else:
            print(f"\nComputing metrics for {label_b}...")
            t0 = time.time()
            mse_b, cos_sim_b = compute_metrics_for_run(
                model_ref, run_b, eval_tokens, local_rank, batch_size, nsamples, seqlen, use_cache,
                zero_out_rows=zero_out_rows,
            )
            print(f"  {label_b} done in {time.time() - t0:.1f}s")

        del model_ref
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Build per-layer result records
    # ------------------------------------------------------------------
    results_a = []
    results_b = []
    for name in sorted(mse_a.keys(), key=get_sort_key):
        if name in mse_b and name in cos_sim_a and name in cos_sim_b:
            layer_id, block_type, weight_type = parse_layer_name(name)
            results_a.append({
                "name": name,
                "rel_mse": mse_a[name],
                "cos_sim": cos_sim_a[name],
                "weight_type": weight_type,
                "merged_type": get_merged_weight_type(weight_type),
                "layer_id": layer_id,
            })
            results_b.append({
                "name": name,
                "rel_mse": mse_b[name],
                "cos_sim": cos_sim_b[name],
                "weight_type": weight_type,
                "merged_type": get_merged_weight_type(weight_type),
                "layer_id": layer_id,
            })

    # Merge weight types if requested
    if merge_inputs:
        def merge_results(results):
            grouped = defaultdict(list)
            for r in results:
                key = (r["layer_id"], r["merged_type"])
                grouped[key].append(r)

            merged = []
            for (layer_id, merged_type), items in grouped.items():
                avg_mse = np.mean([r["rel_mse"] for r in items])
                avg_cos_sim = np.mean([r["cos_sim"] for r in items])
                merged.append({
                    "short_name": f"L{layer_id}_{merged_type}",
                    "rel_mse": avg_mse,
                    "cos_sim": avg_cos_sim,
                    "weight_type": merged_type,
                    "layer_id": layer_id,
                    "sort_key": (layer_id, -1 if merged_type == "attention_norm" else 0 if merged_type == "qkv" else 1 if merged_type == "wo" else 2 if merged_type == "ffn_norm" else 3 if merged_type == "w1w3" else 4),
                })
            merged.sort(key=lambda x: x["sort_key"])
            return merged

        results_a = merge_results(results_a)
        results_b = merge_results(results_b)
    else:
        for r in results_a:
            r["short_name"] = f"L{r['layer_id']}_{r['weight_type']}"
        for r in results_b:
            r["short_name"] = f"L{r['layer_id']}_{r['weight_type']}"

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    names = [r["short_name"] for r in results_a]
    rel_mses_a = [r["rel_mse"] * 100 for r in results_a]
    rel_mses_b = [r["rel_mse"] * 100 for r in results_b]
    cos_dist_a = [(1 - r["cos_sim"]) * 100 for r in results_a]
    cos_dist_b = [(1 - r["cos_sim"]) * 100 for r in results_b]
    weight_types = [r["weight_type"] for r in results_a]

    import re
    rate_match = re.search(r"\.r(\d+\.\d+)", run_a.name)
    rate_str = f"Rate={rate_match.group(1)}" if rate_match else ""

    from matplotlib.lines import Line2D
    x = np.arange(len(names))
    weight_type_order = ["attention_norm", "qkv", "wo", "ffn_norm", "w1w3", "w2"]

    def make_comparison_plot(values_a, values_b, ylabel, plot_title, save_path):
        fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="white")
        ax.set_facecolor("white")

        for wt in weight_type_order:
            indices = [i for i, w in enumerate(weight_types) if w == wt]
            if indices:
                color = get_weight_type_color(wt)
                ax.scatter([x[i] - 0.15 for i in indices], [values_a[i] for i in indices],
                           c=color, s=60, alpha=0.9, edgecolors="none", marker="o", zorder=3)
                ax.scatter([x[i] + 0.15 for i in indices], [values_b[i] for i in indices],
                           c="none", s=60, alpha=0.9, edgecolors=color, linewidths=2, marker="o", zorder=3)

        for i in range(len(names)):
            ax.plot([x[i] - 0.15, x[i] + 0.15], [values_a[i], values_b[i]],
                    color="gray", alpha=0.3, linewidth=0.8, zorder=1)

        ax.grid(True, which="major", axis="both", alpha=0.3, linewidth=1, color="lightgray")
        ax.set_axisbelow(True)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_xlabel("Layer", fontsize=13)
        ax.set_title(plot_title, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=10)

        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=8, label=f"{label_a} (filled)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="gray",
                   markeredgewidth=2, markersize=8, label=f"{label_b} (hollow)"),
            Line2D([0], [0], marker="", color="none", label=""),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red", markersize=8, label="attn_in (residual)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markersize=8, label="qkv"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:cyan", markersize=8, label="wo"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:purple", markersize=8, label="ffn_in (residual)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange", markersize=8, label="w1w3"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:green", markersize=8, label="w2"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=10, ncol=2)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"Saved: {save_path}")
        else:
            plt.show()
        plt.close(fig)

    # Plot 1: Relative MSE
    mse_title = title or f"Relative MSE: {label_a} vs {label_b}\n{model_name} {rate_str}"
    mse_path = output_path if output_path else None
    make_comparison_plot(rel_mses_a, rel_mses_b, "Relative Activation MSE (%)", mse_title, mse_path)

    # Plot 2: Cosine Distance
    cos_title = f"Cosine Distance: {label_a} vs {label_b}\n{model_name} {rate_str}"
    cos_path = output_path.with_stem(output_path.stem + "_cosine") if output_path else None
    make_comparison_plot(cos_dist_a, cos_dist_b, "Cosine Distance (1 - cos_sim) %", cos_title, cos_path)

    # Summary
    print("\nSummary (Relative MSE %):")
    print(f"  {label_a}: mean={np.mean(rel_mses_a):.2f}%, max={max(rel_mses_a):.2f}%")
    print(f"  {label_b}: mean={np.mean(rel_mses_b):.2f}%, max={max(rel_mses_b):.2f}%")
    diffs_mse = [b - a for a, b in zip(rel_mses_a, rel_mses_b)]
    print(f"  Diff ({label_b} - {label_a}): mean={np.mean(diffs_mse):.2f}%")

    print("\nSummary (Cosine Distance %):")
    print(f"  {label_a}: mean={np.mean(cos_dist_a):.4f}%, max={max(cos_dist_a):.4f}%")
    print(f"  {label_b}: mean={np.mean(cos_dist_b):.4f}%, max={max(cos_dist_b):.4f}%")
    diffs_cos = [b - a for a, b in zip(cos_dist_a, cos_dist_b)]
    print(f"  Diff ({label_b} - {label_a}): mean={np.mean(diffs_cos):.4f}%")

    # Save JSON
    json_path = output_path.with_suffix(".json") if output_path else run_a / "activation_metrics_compare.json"
    data = {
        "run_a": str(run_a), "run_b": str(run_b),
        "label_a": label_a, "label_b": label_b,
        "nsamples": nsamples, "seqlen": seqlen,
        "layers": [{
            "name": names[i],
            "rel_mse_a_pct": rel_mses_a[i], "rel_mse_b_pct": rel_mses_b[i],
            "cos_dist_a_pct": cos_dist_a[i], "cos_dist_b_pct": cos_dist_b[i],
        } for i in range(len(names))],
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON: {json_path}")


def _merge_results(results, merge_inputs: bool = True):
    """Merge weight types (qkv, w1w3) and sort."""
    if merge_inputs:
        grouped = defaultdict(list)
        for r in results:
            key = (r["layer_id"], r["merged_type"])
            grouped[key].append(r)

        merged = []
        for (layer_id, merged_type), items in grouped.items():
            avg_mse = np.mean([r["rel_mse"] for r in items])
            avg_cos_sim = np.mean([r["cos_sim"] for r in items])
            merged.append({
                "short_name": f"L{layer_id}_{merged_type}",
                "rel_mse": avg_mse,
                "cos_sim": avg_cos_sim,
                "weight_type": merged_type,
                "layer_id": layer_id,
                "sort_key": (layer_id, -1 if merged_type == "attention_norm" else 0 if merged_type == "qkv" else 1 if merged_type == "wo" else 2 if merged_type == "ffn_norm" else 3 if merged_type == "w1w3" else 4),
            })
        merged.sort(key=lambda x: x["sort_key"])
        return merged
    else:
        for r in results:
            r["short_name"] = f"L{r['layer_id']}_{r['weight_type']}"
        return results


def _scatter_on_ax(ax, plot_results, weight_type_order):
    """Draw scatter points on a single axes."""
    names = [r["short_name"] for r in plot_results]
    rel_mses = [r["rel_mse"] * 100 for r in plot_results]
    weight_types = [r["weight_type"] for r in plot_results]
    x = np.arange(len(names))

    for wt in weight_type_order:
        indices = [i for i, w in enumerate(weight_types) if w == wt]
        if indices:
            color = get_weight_type_color(wt)
            ax.scatter(
                [x[i] for i in indices], [rel_mses[i] for i in indices],
                c=color, s=55, alpha=0.8, label=wt, edgecolors='none', zorder=3,
            )

    ax.grid(True, which="major", axis="both", alpha=0.3, linewidth=1, color='lightgray')
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.tick_params(axis='y', labelsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('black')


def plot_single_run(
    run_dir: Path,
    output_path: Path = None,
    title: str = None,
    merge_inputs: bool = True,
    nsamples: int = 64,
    batch_size: int = 32,
    seqlen: int = 2048,
    use_cache: bool = True,
    zero_out_rows: str = "",
    split: int = 0,
):
    """Compute and plot per-layer activation MSE for a single run.

    If split > 1, produces a multi-panel figure with layers divided evenly.
    Works with torchrun for multi-GPU models (no --init_dist needed).
    """
    import re

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_rank0 = local_rank == 0

    manifest = RunManifest.load(run_dir / "manifest.json")
    model_name = manifest.model_name

    # Try cache first
    cached = load_cached_metrics(run_dir, nsamples, seqlen) if use_cache else None

    if cached is not None:
        mse_dict, cos_sim_dict = cached
    else:
        if is_rank0:
            print(f"Computing metrics for {run_dir.name} (sequential, one model at a time)...")
        t0 = time.time()
        mse_dict, cos_sim_dict = compute_metrics_sequential(
            model_name, run_dir, local_rank=local_rank,
            batch_size=batch_size, nsamples=nsamples, seqlen=seqlen,
            use_cache=use_cache, zero_out_rows=zero_out_rows,
        )
        if is_rank0:
            print(f"Done in {time.time() - t0:.1f}s")

    # Only rank 0 does plotting
    if not is_rank0:
        return

    # Build result records
    results = []
    for name in sorted(mse_dict.keys(), key=get_sort_key):
        layer_id, block_type, weight_type = parse_layer_name(name)
        results.append({
            "name": name,
            "rel_mse": mse_dict[name],
            "cos_sim": cos_sim_dict.get(name, 0.0),
            "weight_type": weight_type,
            "merged_type": get_merged_weight_type(weight_type),
            "layer_id": layer_id,
        })

    plot_results = _merge_results(results, merge_inputs)
    weight_type_order = ["attention_norm", "qkv", "wo", "ffn_norm", "w1w3", "w2"]

    rate_match = re.search(r"\.r(\d+\.\d+)", run_dir.name)
    rate_str = f"rate={rate_match.group(1)}" if rate_match else ""
    default_title = f"Activation Drift  —  {model_name}  {rate_str}"

    if split > 1:
        # Multi-panel split
        layer_ids = sorted(set(r["layer_id"] for r in plot_results))
        n_layers = len(layer_ids)
        per_panel = (n_layers + split - 1) // split

        panels = []
        for p in range(split):
            lo = p * per_panel
            hi = min((p + 1) * per_panel, n_layers)
            panel_layer_ids = set(layer_ids[lo:hi])
            panel_data = [r for r in plot_results if r["layer_id"] in panel_layer_ids]
            if panel_data:
                panels.append(panel_data)

        n_panels = len(panels)
        fig, axes = plt.subplots(n_panels, 1, figsize=(22, 5 * n_panels),
                                 facecolor='white', squeeze=False)

        all_rel = [r["rel_mse"] * 100 for r in plot_results]
        y_max = max(all_rel) * 1.08

        for i, panel_data in enumerate(panels):
            ax = axes[i, 0]
            ax.set_facecolor('white')
            _scatter_on_ax(ax, panel_data, weight_type_order)
            ax.set_ylim(0, y_max)
            ax.set_ylabel("Relative MSE (%)", fontsize=12)
            lo_l = min(r["layer_id"] for r in panel_data)
            hi_l = max(r["layer_id"] for r in panel_data)
            ax.set_title(f"Layers {lo_l}–{hi_l}", fontsize=13)
            if i == 0:
                ax.legend(loc="upper right", frameon=True, fancybox=True,
                          framealpha=0.95, edgecolor='lightgray', fontsize=10)

        axes[-1, 0].set_xlabel("Layer", fontsize=12)
        fig.suptitle(title or default_title, fontsize=15, y=1.0)
    else:
        # Single panel
        fig, ax = plt.subplots(1, 1, figsize=(22, 7), facecolor='white')
        ax.set_facecolor('white')
        _scatter_on_ax(ax, plot_results, weight_type_order)
        ax.set_ylabel("Relative Activation MSE (%)", fontsize=13)
        ax.set_xlabel("Layer", fontsize=13)
        ax.set_title(title or default_title, fontsize=15)
        ax.legend(loc="upper right", frameon=True, fancybox=True,
                  framealpha=0.95, edgecolor='lightgray', fontsize=10)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    plt.close(fig)

    # Summary
    rel_mses = [r["rel_mse"] * 100 for r in plot_results]
    print("\nSummary (Relative MSE %):")
    print(f"  {len(plot_results)} points, mean={np.mean(rel_mses):.2f}%, max={max(rel_mses):.2f}%")
    top = sorted(plot_results, key=lambda r: -r["rel_mse"])[:10]
    print("  Top 10:")
    for r in top:
        print(f"    {r['short_name']}: {r['rel_mse']*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Compute and plot per-layer activation MSE")
    # Single-run mode
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Single run directory (plot one run)")
    # Comparison mode
    parser.add_argument("--run_a", type=str, default=None)
    parser.add_argument("--run_b", type=str, default=None)
    parser.add_argument("--label_a", type=str, default=None)
    parser.add_argument("--label_b", type=str, default=None)
    # Common
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--no_merge", action="store_true")
    parser.add_argument("--figsize", type=str, default="22,7")
    parser.add_argument("--nsamples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for activation capture (larger = faster but more VRAM)")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--init_dist", action="store_true")
    parser.add_argument("--master_port_base", type=int, default=29500)
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable caching of metrics (recompute even if cache exists)")
    parser.add_argument("--zero_out_rows", type=str, default="",
                        help="Zero out rows in reference model. Format: '6.w1:5723,8518;16.w1:2271,1875'")
    parser.add_argument("--split", type=int, default=0, metavar="N",
                        help="Split into N vertical panels (e.g. --split 4 for 80 layers)")

    args = parser.parse_args()

    if args.run_dir:
        # Single-run mode
        run_dir = Path(args.run_dir)
        output_path = Path(args.output) if args.output else run_dir / "activation_mse.png"
        plot_single_run(
            run_dir=run_dir, output_path=output_path, title=args.title,
            merge_inputs=not args.no_merge,
            nsamples=args.nsamples, batch_size=args.batch_size, seqlen=args.seqlen,
            use_cache=not args.no_cache,
            zero_out_rows=args.zero_out_rows, split=args.split,
        )
    elif args.run_a and args.run_b:
        # Comparison mode
        run_a = Path(args.run_a)
        run_b = Path(args.run_b)
        output_path = Path(args.output) if args.output else run_a / "activation_mse_compare.png"
        figsize = tuple(map(float, args.figsize.split(",")))
        plot_comparison(
            run_a=run_a, run_b=run_b, output_path=output_path,
            label_a=args.label_a, label_b=args.label_b, title=args.title,
            figsize=figsize, merge_inputs=not args.no_merge,
            nsamples=args.nsamples, batch_size=args.batch_size, seqlen=args.seqlen,
            init_dist=args.init_dist, master_port_base=args.master_port_base,
            use_cache=not args.no_cache, zero_out_rows=args.zero_out_rows,
        )
    else:
        parser.error("Provide --run_dir for single-run mode, or --run_a and --run_b for comparison")


if __name__ == "__main__":
    main()
