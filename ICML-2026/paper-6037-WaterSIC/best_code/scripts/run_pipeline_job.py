"""Run one layerwise quantization job.

This script is designed to be used in two ways:
  1) CLI: `python -m scripts.run_pipeline_job ...`
  2) As a callable from your GPU scheduler (mimicking your baseline runners):
        from scripts.run_pipeline_job import run_pipeline_job

If you're running without torchrun (WORLD_SIZE=1), you can pass `init_dist=True`
so that `parallel.start.start` can initialize NCCL using a per-GPU master port.
This only works when your ckpt_dir has a *single* checkpoint shard.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

# ── NUMA pinning ────────────────────────────────────────────────────────
# Pin each torchrun worker to the NUMA node of its GPU to avoid cross-NUMA
# memory traffic (PCIe + SMP bridge).  Topology assumed:
#   GPU 0,1 → NUMA 0    GPU 2,3 → NUMA 1
# For machines with different topology, override with QUANT_NUMA_MAP env var
# (comma-separated list: "0,0,1,1" means rank0→NUMA0, rank1→NUMA0, etc.)
# Set QUANT_NO_NUMA=1 to disable.
def _pin_numa() -> None:
    if os.environ.get("QUANT_NO_NUMA", "0") == "1":
        return
    local_rank_str = os.environ.get("LOCAL_RANK")
    if local_rank_str is None:
        return  # not launched via torchrun
    local_rank = int(local_rank_str)
    try:
        import subprocess
        # Read available NUMA nodes
        result = subprocess.run(
            ["lscpu", "--parse=CPU,NODE"], capture_output=True, text=True
        )
        if result.returncode != 0:
            return
        # Parse CPU→NUMA mapping
        node_cpus: dict[int, list[int]] = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split(",")
            cpu_id, node_id = int(parts[0]), int(parts[1])
            node_cpus.setdefault(node_id, []).append(cpu_id)

        # Determine which NUMA node this rank should use
        numa_map_str = os.environ.get("QUANT_NUMA_MAP")
        if numa_map_str:
            numa_map = [int(x) for x in numa_map_str.split(",")]
            target_node = numa_map[local_rank]
        else:
            # Default: ranks 0,1 → node 0; ranks 2,3 → node 1; etc. (2 GPUs per NUMA node)
            target_node = local_rank // 2

        if target_node not in node_cpus:
            return

        cpus = node_cpus[target_node]
        os.sched_setaffinity(0, cpus)
        if local_rank == 0:
            print(f"[NUMA] Pinned rank {local_rank} to NUMA node {target_node} "
                  f"({len(cpus)} CPUs)")
    except Exception as e:
        if local_rank == 0:
            print(f"[NUMA] Pinning failed (non-fatal): {e}")

_pin_numa()

# ── NCCL tuning for multi-NUMA topologies ───────────────────────────────
# Ring algorithm avoids the tree-reduction bottleneck on SYS (PCIe+SMP) links.
# LL (Low Latency) protocol reduces per-message overhead for small collectives
# (e.g. the remaining all-reduces/all-gathers after gathered-LDLQ).
# Set QUANT_NO_NCCL_TUNE=1 to disable.
if os.environ.get("QUANT_NO_NCCL_TUNE", "0") != "1":
    _nccl_defaults = {
        "NCCL_ALGO": "Ring",
        "NCCL_PROTO": "LL",
    }
    for k, v in _nccl_defaults.items():
        if k not in os.environ:
            os.environ[k] = v

from quant_layerwise.pipeline import (
    PipelineConfig,
    build_layers,
    ensure_single_process_distributed,
    run_pipeline,
)
from quant_layerwise.rate_control import RateControlConfig
from quant_layerwise.methods.gptq import GPTQConfig
from quant_layerwise.methods.zsic import ZSICConfig


def run_pipeline_job(
    model_name: str,
    method: str,
    target_rate: float,
    *,
    # layer selection
    layer_begin: int = 0,
    layer_end: int = 32,
    weights: str = "wq,wk,wv,wo,w1,w3,w2",  # w2 after w1,w3 so Hessian sees quantized activations
    # calib
    calib_dataset: str = "redpajama",  # "redpajama" or "wikitext2"
    calib_seed: int = 42,  # Seed for RedPajama shuffling
    seqlen: int = 2048,
    calib_stride: int | None = None,  # Stride for overlapping calib windows (None = seqlen)
    nsamples: int | None = None,  # Max calibration samples (None = use all)
    hessian_batch_size: int = 1,  # Batch size for Hessian computation (higher = faster)
    w2_batch_size: int | None = None,  # Override batch size for w2 (default: use hessian_batch_size)
    replay_batch_size: int | None = None,  # Override batch size for Qronos/residual layer replay (default: hessian_batch_size)
    # GPTQ
    groupsize: int = -1,
    blocksize: int = 128,
    percdamp: float = 0.0,
    actorder: bool = False,
    gptq_maxq: int | None = None,  # Override maxq (if None, compute from target_rate)
    # ZSIC
    zsic_percdamp: float | None = None,
    zsic_binary_search: bool = False,
    zsic_binary_search_row_fraction: float = 0.1,
    zsic_dead_dim_threshold: float = 0.001,  # Remove dims with var < threshold * mean(diag)
    # Rate control (mainly for ZSIC)
    rate_control: bool = False,
    global_rate_bits: float | None = None,
    rate_xmin: float = 0.05,
    rate_xmax: float = 16.0,
    rate_weight_budgets: str = "",  # Format: "wk:1.5,wq:1.25" to give wk 50% more bits, wq 25% more
    # Skip quantization for specific layers (store full precision)
    skip_quantize: str = "",  # Format: "0.wq,0.wk,1.wq,1.wk" to skip layers 0-1 wq/wk
    # Zero out specific rows after quantization (for outlier removal)
    zero_out_rows: str = "",  # Format: "16.w1:2271,1875;16.w3:2271,1875"
    # Qronos: compute and save Σ_X̂ and Σ_XX̂ statistics
    qronos: bool = False,
    # Qronos layer range: only apply Qronos targeting to layers in [min, max)
    qronos_layer_min: int | None = None,
    qronos_layer_max: int | None = None,
    # Collect Qronos stats for diagnostics (no targeting) + plot activation MSE
    collect_qronos_stats: bool = False,
    save_qronos_stats: bool = False,
    plot_activation_mse: bool = False,
    # Residual stream compensation for wo/w2 layers (requires qronos=True)
    residual_compensation: bool = False,
    # Skip residual compensation on the first N layers (0 = apply to all)
    rescomp_skip_prefix: int = 0,
    # Attention-weighted QKV calibration (BOS fix)
    attn_weighted_qkv: bool = False,
    attn_weighted_qkv_eps: float = 0.0,
    attn_weighted_weights: str = "wq,wk,wv",
    attn_weighted_adapt_eps_joint: bool = False,
    # Adaptive qronos/standard blending for QKV
    qronos_adapt: bool = False,
    # Adaptive qronos/standard blending for w1/w3 jointly (w2 input relMSE)
    w1w3_qronos_adapt: bool = False,
    # Subsample calibration during golden-section search
    adapt_search_sample_ratio: float = 1.0,
    # Number of golden-section steps for q_eps in all adapt searches
    coord_adapt_q_eps_steps: int = 10,
    # Number of golden-section steps for a_eps in coord-adapt (0 = skip a_eps search)
    coord_adapt_a_eps_steps: int = 10,
    # Precomputed dir: "none" to disable auto-enable, "auto" to auto-resolve, or a path
    precomputed_dir: str | None = None,
    # output
    run_root: str = "quant_runs",
    run_id: str = "",
    resume: bool = True,
    # distributed init (single process)
    init_dist: bool = False,
    master_port_base: int = 29500,
    local_rank: int | None = None,
):
    print(f"[debug run_pipeline_job] zsic_binary_search={zsic_binary_search}, rate_control={rate_control}")
    if local_rank is None:
        # Prefer LOCAL_RANK env var (set by torchrun or sweep workers).
        # torch.cuda.current_device() returns 0 before any set_device call,
        # which is wrong for multi-GPU torchrun (all ranks would get 0).
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if init_dist:
        # Use a different port per GPU to avoid collisions when multiple single-process
        # jobs run on the same machine.
        ensure_single_process_distributed(local_rank=local_rank, master_port=master_port_base + int(local_rank))

    wlist = [w.strip() for w in weights.split(",") if w.strip()]
    layers: List[Tuple[int, str]] = build_layers(layer_ids=range(int(layer_begin), int(layer_end)), weights=wlist)

    method_l = method.lower()
    gptq_cfg = None
    zsic_cfg = None
    rate_cfg = None

    if method_l == "gptq":
        gptq_cfg = GPTQConfig(
            target_rate=float(target_rate),
            groupsize=int(groupsize),
            blocksize=int(blocksize),
            percdamp=float(percdamp),
            actorder=bool(actorder),
            maxq=int(gptq_maxq) if gptq_maxq is not None else None,
        )
    elif method_l in ("zsic", "sic"):
        # For ZSIC, use percdamp=0.0001 by default for numerical stability
        # (especially important for Qronos mode with Sigma_hatX)
        if zsic_percdamp is None:
            if percdamp > 0.0:
                zsic_percdamp = float(percdamp)
            else:
                zsic_percdamp = 0.0001  # Default for ZSIC
        zsic_cfg = ZSICConfig(
            target_rate_bits=float(target_rate),
            percdamp=float(zsic_percdamp),
            binary_search=bool(zsic_binary_search),
            binary_search_row_fraction=float(zsic_binary_search_row_fraction),
            qronos=bool(qronos),
            residual_compensation=bool(residual_compensation),
            dead_dim_threshold=float(zsic_dead_dim_threshold),
        )

        # Rate control: if enabled, we try to hit a *global* average rate.
        if bool(rate_control) or global_rate_bits is not None:
            g = float(target_rate) if global_rate_bits is None else float(global_rate_bits)

            # Parse weight budget multipliers: "wk:1.5,wq:1.25" -> {"wk": 1.5, "wq": 1.25}
            weight_mults = None
            if rate_weight_budgets:
                weight_mults = {}
                for item in rate_weight_budgets.split(","):
                    item = item.strip()
                    if not item:
                        continue
                    parts = item.split(":")
                    if len(parts) != 2:
                        raise ValueError(f"Invalid rate_weight_budgets format: '{item}'. Expected 'weight:multiplier' (e.g., 'wk:1.5')")
                    wtype = parts[0].strip()
                    mult = float(parts[1].strip())
                    weight_mults[wtype] = mult

            rate_cfg = RateControlConfig(
                enabled=True,
                global_target_rate_bits=float(g),
                xmin=float(rate_xmin),
                xmax=float(rate_xmax),
                weight_budget_multipliers=weight_mults,
            )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Parse skip_quantize: "0.wq,0.wk,1.wq,1.wk" -> [(0, "wq"), (0, "wk"), ...]
    skip_layers: List[Tuple[int, str]] = []
    if skip_quantize:
        for item in skip_quantize.split(","):
            item = item.strip()
            if not item:
                continue
            parts = item.split(".")
            if len(parts) != 2:
                raise ValueError(f"Invalid skip_quantize format: '{item}'. Expected 'layer_id.weight' (e.g., '0.wq')")
            layer_id = int(parts[0])
            weight = parts[1].strip()
            skip_layers.append((layer_id, weight))

    # Parse zero_out_rows: "16.w1:2271,1875;16.w3:2271,1875" -> {"16.w1": [2271, 1875], "16.w3": [2271, 1875]}
    zero_out_dict: Dict[str, List[int]] | None = None
    if zero_out_rows:
        zero_out_dict = {}
        for item in zero_out_rows.split(";"):
            item = item.strip()
            if not item:
                continue
            parts = item.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid zero_out_rows format: '{item}'. Expected 'layer.weight:row1,row2,...' (e.g., '16.w1:2271,1875')")
            key = parts[0].strip()  # e.g., "16.w1"
            rows = [int(r.strip()) for r in parts[1].split(",") if r.strip()]
            zero_out_dict[key] = rows
        print(f"[zero_out_rows] parsed: {zero_out_dict}", flush=True)

    cfg = PipelineConfig(
        model_name=str(model_name),
        method=str(method_l),
        layers=layers,
        calib_dataset=str(calib_dataset),
        calib_seed=int(calib_seed),
        seqlen=int(seqlen),
        calib_stride=int(calib_stride) if calib_stride is not None else None,
        nsamples=int(nsamples) if nsamples is not None else None,
        hessian_batch_size=int(hessian_batch_size),
        w2_batch_size=int(w2_batch_size) if w2_batch_size is not None else None,
        replay_batch_size=int(replay_batch_size) if replay_batch_size is not None else None,
        run_root=str(run_root),
        run_id=str(run_id),
        resume=bool(resume),
        gptq=gptq_cfg,
        zsic=zsic_cfg,
        rate_control=rate_cfg,
        skip_quantize_layers=skip_layers,
        qronos=bool(qronos),
        qronos_layer_min=qronos_layer_min,
        qronos_layer_max=qronos_layer_max,
        collect_qronos_stats=bool(collect_qronos_stats),
        save_qronos_stats=bool(save_qronos_stats),
        plot_activation_mse=bool(plot_activation_mse),
        residual_compensation=bool(residual_compensation),
        rescomp_skip_prefix=int(rescomp_skip_prefix),
        attn_weighted_qkv=bool(attn_weighted_qkv),
        attn_weighted_qkv_eps=float(attn_weighted_qkv_eps),
        attn_weighted_weights=tuple(w.strip() for w in attn_weighted_weights.split(",") if w.strip()),
        attn_weighted_adapt_eps_joint=bool(attn_weighted_adapt_eps_joint),
        qronos_adapt=bool(qronos_adapt),
        w1w3_qronos_adapt=bool(w1w3_qronos_adapt),
        adapt_search_sample_ratio=float(adapt_search_sample_ratio),
        coord_adapt_q_eps_steps=int(coord_adapt_q_eps_steps),
        coord_adapt_a_eps_steps=int(coord_adapt_a_eps_steps),
        zero_out_rows=zero_out_dict,
        precomputed_dir=precomputed_dir if precomputed_dir else None,
    )

    return run_pipeline(cfg, local_rank=int(local_rank))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--method", required=True, choices=["gptq", "zsic"])
    p.add_argument("--target_rate", required=True, type=float)

    p.add_argument("--layer_begin", type=int, default=0)
    p.add_argument("--layer_end", type=int, default=32)
    p.add_argument("--weights", type=str, default="wq,wk,wv,wo,w1,w3,w2")

    p.add_argument("--calib_dataset", type=str, default="redpajama",
                   help="Calibration dataset: redpajama, c4, wikitext2, or mix like 'redpajama:0.5,c4:0.4,wikitext2:0.1'")
    p.add_argument("--calib_seed", type=int, default=42,
                   help="Seed for RedPajama shuffling (default: 42)")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--calib_stride", type=int, default=None,
                   help="Stride for overlapping calib windows (default: seqlen = no overlap). "
                        "E.g. --calib_stride 2048 with --seqlen 4096 gives 50%% overlap and ~2x more sequences.")
    p.add_argument("--nsamples", type=int, default=None,
                   help="Max calibration samples (default: use all available)")
    p.add_argument("--hessian_batch_size", type=int, default=1, help="Batch size for Hessian computation (higher = faster, more memory)")
    p.add_argument("--w2_batch_size", type=int, default=None, help="Override batch size for w2 (default: use hessian_batch_size)")
    p.add_argument("--replay_batch_size", type=int, default=None, help="Override batch size for Qronos/residual layer replay (default: hessian_batch_size). Lower for large models where attention scores OOM.")

    # GPTQ
    p.add_argument("--groupsize", type=int, default=-1)
    p.add_argument("--blocksize", type=int, default=128)
    p.add_argument("--percdamp", type=float, default=0.0)
    p.add_argument("--actorder", action="store_true")
    p.add_argument("--gptq_maxq", type=int, default=None,
                   help="Override maxq for GPTQ (default: compute from target_rate as 2^(rate+1)-1)")

    # ZSIC
    # Optional: override percdamp used inside ZSIC (defaults to --percdamp)
    p.add_argument("--zsic_percdamp", type=float, default=None)

    # Optional: binary search to hit target rate more precisely
    p.add_argument("--zsic_binary_search", action="store_true",
                   help="Binary search for target_rate that achieves desired actual rate")
    p.add_argument("--zsic_binary_search_row_fraction", type=float, default=0.1,
                   help="Fraction of rows to use during binary search (e.g., 0.1 for 10%%). Default 0.1 for speed.")
    p.add_argument("--zsic_dead_dim_threshold", type=float, default=0.001,
                   help="Remove dims with variance < threshold * mean(diag). Prevents Cholesky failures from dead dims.")

    # Optional: rate control (mainly for ZSIC)
    p.add_argument("--rate_control", action="store_true", help="enable global rate budget tracking")
    p.add_argument("--global_rate_bits", type=float, default=None, help="global avg target bits/param (default: --target_rate)")
    p.add_argument("--rate_xmin", type=float, default=0.05, help="minimum allowed target rate")
    p.add_argument("--rate_xmax", type=float, default=16.0, help="maximum allowed target rate")
    p.add_argument("--rate_weight_budgets", type=str, default="",
                   help="Weight-type budget multipliers. Format: 'wk:1.5,wq:1.25' gives wk 50%% more bits, wq 25%% more")

    p.add_argument("--skip_quantize", type=str, default="",
                   help="Skip quantization for specific layers (store full precision). Format: '0.wq,0.wk,1.wq,1.wk'")
    p.add_argument("--zero_out_rows", type=str, default="",
                   help="Zero out specific rows after quantization (for outlier removal). Format: '16.w1:2271,1875;16.w3:2271,1875'")

    p.add_argument("--qronos", action="store_true",
                   help="Compute and save Qronos statistics (Σ_X̂ and Σ_XX̂) for each layer")
    p.add_argument("--qronos_layer_min", type=int, default=None,
                   help="Only apply Qronos targeting to layers >= this (default: all layers)")
    p.add_argument("--qronos_layer_max", type=int, default=None,
                   help="Only apply Qronos targeting to layers < this (default: all layers)")
    p.add_argument("--collect_qronos_stats", action="store_true",
                   help="Collect Qronos stats for diagnostics (no Qronos targeting)")
    p.add_argument("--save_qronos_stats", action="store_true",
                   help="Save Qronos stats (Σ_X, Σ_X̂, Σ_{X,X̂}) to disk as .pkl files")
    p.add_argument("--plot_activation_mse", action="store_true",
                   help="Plot activation MSE at end of run (requires --qronos or --collect_qronos_stats)")

    p.add_argument("--residual_compensation", action="store_true",
                   help="Enable residual stream compensation for wo/w2 layers (automatically enables Qronos mode for wo/w2)")
    p.add_argument("--rescomp_skip_prefix", type=int, default=0,
                   help="Skip residual compensation on the first N layers (0 = apply to all)")

    p.add_argument("--attn_weighted_qkv", action="store_true",
                   help="Weight QKV calibration stats by per-token attention importance (BOS fix)")
    p.add_argument("--attn_weighted_qkv_eps", type=float, default=0.01,
                   help="Mixing: Sig = (1-eps)*weighted + eps*unweighted (default: 0.01)")
    p.add_argument("--attn_weighted_weights", type=str, default="wq,wk,wv",
                   help="Which QKV weights to apply attention weighting to (default: 'wq,wk,wv')")
    p.add_argument("--attn_weighted_adapt_eps_joint", action="store_true",
                   help="Joint eps search for wq/wk/wv (minimize wo_in relMSE via forward pass)")

    p.add_argument("--qronos_adapt", action="store_true",
                   help="Adaptive qronos/standard blending for QKV (minimize wo_in relMSE)")
    p.add_argument("--w1w3_qronos_adapt", action="store_true",
                   help="Adaptive qronos/standard blending for w1/w3 jointly (minimize w2 input relMSE)")

    p.add_argument("--adapt_search_sample_ratio", type=float, default=1.0,
                   help="Fraction of calibration samples for golden-section search (default: 1.0 = all)")
    p.add_argument("--coord_adapt_q_eps_steps", type=int, default=10,
                   help="Golden-section steps for q_eps in all adapt searches (default: 10)")
    p.add_argument("--coord_adapt_a_eps_steps", type=int, default=10,
                   help="Golden-section steps for a_eps in coord-adapt (0 = skip a_eps search, default: 10)")

    p.add_argument("--precomputed_dir", type=str, default=None,
                   help="Precomputed data dir. 'none' to disable auto-enable, 'auto' to auto-resolve, or a path.")

    p.add_argument("--run_root", type=str, default="quant_runs")
    p.add_argument("--run_id", type=str, default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.set_defaults(resume=True)

    p.add_argument("--init_dist", action="store_true", help="init NCCL env for single-process jobs")
    p.add_argument("--master_port_base", type=int, default=29500)

    args = p.parse_args()

    run_pipeline_job(
        args.model,
        args.method,
        args.target_rate,
        layer_begin=args.layer_begin,
        layer_end=args.layer_end,
        weights=args.weights,
        calib_dataset=args.calib_dataset,
        calib_seed=args.calib_seed,
        seqlen=args.seqlen,
        calib_stride=args.calib_stride,
        nsamples=args.nsamples,
        hessian_batch_size=args.hessian_batch_size,
        w2_batch_size=args.w2_batch_size,
        replay_batch_size=args.replay_batch_size,
        groupsize=args.groupsize,
        blocksize=args.blocksize,
        percdamp=args.percdamp,
        actorder=args.actorder,
        gptq_maxq=args.gptq_maxq,
        zsic_percdamp=args.zsic_percdamp,
        zsic_binary_search=args.zsic_binary_search,
        zsic_binary_search_row_fraction=args.zsic_binary_search_row_fraction,
        zsic_dead_dim_threshold=args.zsic_dead_dim_threshold,
        rate_control=args.rate_control,
        global_rate_bits=args.global_rate_bits,
        rate_xmin=args.rate_xmin,
        rate_xmax=args.rate_xmax,
        rate_weight_budgets=args.rate_weight_budgets,
        skip_quantize=args.skip_quantize,
        zero_out_rows=args.zero_out_rows,
        qronos=args.qronos,
        qronos_layer_min=args.qronos_layer_min,
        qronos_layer_max=args.qronos_layer_max,
        collect_qronos_stats=args.collect_qronos_stats,
        save_qronos_stats=args.save_qronos_stats,
        plot_activation_mse=args.plot_activation_mse,
        residual_compensation=args.residual_compensation,
        rescomp_skip_prefix=args.rescomp_skip_prefix,
        attn_weighted_qkv=args.attn_weighted_qkv,
        attn_weighted_qkv_eps=args.attn_weighted_qkv_eps,
        attn_weighted_weights=args.attn_weighted_weights,
        attn_weighted_adapt_eps_joint=args.attn_weighted_adapt_eps_joint,
        qronos_adapt=args.qronos_adapt,
        w1w3_qronos_adapt=args.w1w3_qronos_adapt,
        adapt_search_sample_ratio=args.adapt_search_sample_ratio,
        coord_adapt_q_eps_steps=args.coord_adapt_q_eps_steps,
        coord_adapt_a_eps_steps=args.coord_adapt_a_eps_steps,
        precomputed_dir=args.precomputed_dir,
        run_root=args.run_root,
        run_id=args.run_id,
        resume=args.resume,
        init_dist=args.init_dist,
        master_port_base=args.master_port_base,
    )


if __name__ == "__main__":
    main()
