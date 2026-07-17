"""Evaluate a quantized run directory.

Computes:
  * PPL on WikiText-2 test split
  * KL(P_unquantized || P_quantized)

By default this loads *two* models (ref + quantized). If you can't fit both,
run with `--ppl_only`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from quant_layerwise.data import get_calibration_data, get_wikitext2, split_dataset, take_nseq
from quant_layerwise.eval import eval_kl, eval_ppl
from quant_layerwise.names import get_hess_name
from quant_layerwise.partial_model import load_and_apply_manifest
from quant_layerwise.pipeline import ensure_single_process_distributed, load_model_and_tokenizer, get_dist_info
from quant_layerwise.storage import RunManifest


def _apply_zero_out_rows(model: torch.nn.Module, zero_out_rows: str):
    """Parse and apply zero_out_rows spec to a model.

    Format: "6.w1:5723,8518;16.w1:2271,1875"
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


def run_eval_job(
    run_dir: str | Path,
    *,
    seqlen: int = 2048,
    eval_nsamples: int | None = None,  # None means use all available samples
    max_batches: int | None = None,  # None means use all batches
    ppl_only: bool = False,
    sequential: bool = False,  # Load models one at a time to save memory
    split: str = "test",  # "test" or "train" - use "train" to eval on calibration data
    eval_dataset: str = "wikitext2",  # "wikitext2", "redpajama_sample", "redpajama", "c4"
    eval_nsamples_calib: int | None = 1024,  # nsamples for non-wikitext2 eval datasets
    force_bos: bool = False,  # Replace first token of each sequence with BOS
    zero_out_rows: str = "",  # e.g. "6.w1:5723,8518;16.w1:2271,1875"
    init_dist: bool = False,
    master_port_base: int = 29600,
    local_rank: int | None = None,
):
    run_dir = Path(run_dir)
    if local_rank is None:
        # Prefer LOCAL_RANK env var (set by torchrun or sweep workers).
        # torch.cuda.current_device() returns 0 before any set_device call,
        # which is wrong for multi-GPU torchrun (all ranks would get 0).
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if init_dist:
        ensure_single_process_distributed(local_rank=local_rank, master_port=master_port_base + int(local_rank))

    # Read manifest first to know which base model to load.
    manifest = RunManifest.load(run_dir / "manifest.json")

    # Distributed info for loading correct shard artifacts.
    dist_rank, dist_world_size = get_dist_info()

    # Load quantized model (separate model so we can keep an unquantized reference if needed)
    # Pass seqlen as max_seq_len to ensure KV cache and RoPE are sized correctly
    model_q, tokenizer = load_model_and_tokenizer(
        manifest.model_name, local_rank=local_rank, max_seq_len=seqlen
    )

    # Apply all saved layers (each rank loads its own shard)
    _manifest2 = load_and_apply_manifest(model_q, run_dir, rank=dist_rank)
    assert _manifest2.model_name == manifest.model_name

    # Apply zero_out_rows to ensure exact zeros after dequantization
    # (pipeline applies this post-quantization, but load_and_apply_manifest doesn't)
    if zero_out_rows:
        _apply_zero_out_rows(model_q, zero_out_rows)

    # Dataset
    if eval_dataset == "wikitext2":
        eval_tokens = split_dataset(get_wikitext2(tokenizer, split=split), seqlen)
    elif eval_dataset == "c4":
        # Standard C4 eval: random windows from validation split (GPTQ/QuIP protocol)
        from quant_layerwise.data import get_c4_val
        n = eval_nsamples if eval_nsamples is not None else 256
        eval_tokens = split_dataset(get_c4_val(tokenizer, nsamples=n, seqlen=seqlen, seed=0), seqlen)
    else:
        eval_tokens = split_dataset(
            get_calibration_data(tokenizer, dataset=eval_dataset, nsamples=eval_nsamples_calib, seqlen=seqlen),
            seqlen,
        )
    eval_tokens = take_nseq(eval_tokens, eval_nsamples)  # None means all samples
    actual_nsamples = eval_tokens.shape[0]

    if force_bos:
        bos_id = tokenizer.bos_id
        if bos_id is None:
            print("[eval] WARNING: force_bos requested but tokenizer has no bos_id, skipping")
        else:
            n_replaced = int((eval_tokens[:, 0] != bos_id).sum())
            eval_tokens[:, 0] = bos_id
            print(f"[eval] force_bos: replaced first token with BOS (id={bos_id}) in {n_replaced}/{actual_nsamples} sequences")

    print(f"[eval] using {actual_nsamples} eval samples from {split} split (seqlen={seqlen})")

    ppl_q, nll_q = eval_ppl(model_q, eval_tokens, max_batches=max_batches)

    out = {
        "run_dir": str(run_dir),
        "model_name": manifest.model_name,
        "method": manifest.method,
        "eval": {
            "split": split,
            "seqlen": int(seqlen),
            "eval_nsamples": int(actual_nsamples),
            "max_batches": None if max_batches is None else int(max_batches),
            "ppl_quant": float(ppl_q),
            "nll_quant": float(nll_q),
        },
    }

    if not ppl_only:
        if sequential:
            # Sequential mode: delete quant model first to free memory
            print("[eval] sequential mode: unloading quantized model to free memory...")
            del model_q
            torch.cuda.empty_cache()

            model_ref, _tok2 = load_model_and_tokenizer(
                manifest.model_name, local_rank=local_rank, max_seq_len=seqlen
            )
            if zero_out_rows:
                _apply_zero_out_rows(model_ref, zero_out_rows)
            ppl_ref, nll_ref = eval_ppl(model_ref, eval_tokens, max_batches=max_batches)

            # Reload quantized model for KL computation
            print("[eval] sequential mode: reloading quantized model for KL...")
            model_q, _ = load_model_and_tokenizer(
                manifest.model_name, local_rank=local_rank, max_seq_len=seqlen
            )
            _manifest2 = load_and_apply_manifest(model_q, run_dir, rank=dist_rank)
            if zero_out_rows:
                _apply_zero_out_rows(model_q, zero_out_rows)
            kl = eval_kl(model_ref, model_q, eval_tokens, max_batches=max_batches)
        else:
            # Standard mode: load both models simultaneously
            model_ref, _tok2 = load_model_and_tokenizer(
                manifest.model_name, local_rank=local_rank, max_seq_len=seqlen
            )
            if zero_out_rows:
                _apply_zero_out_rows(model_ref, zero_out_rows)
            ppl_ref, nll_ref = eval_ppl(model_ref, eval_tokens, max_batches=max_batches)
            kl = eval_kl(model_ref, model_q, eval_tokens, max_batches=max_batches)

        out["eval"].update(
            {
                "ppl_ref": float(ppl_ref),
                "nll_ref": float(nll_ref),
                "kl_ref_to_quant": float(kl),
            }
        )

    # Only rank 0 writes eval results to disk.
    if dist_rank == 0:
        ds_tag = eval_dataset if eval_dataset != "wikitext2" else ""
        if force_bos:
            out_filename = f"eval_{split}_bos.json" if split != "test" else "eval_bos.json"
        elif ds_tag:
            out_filename = f"eval_{ds_tag}.json"
        else:
            out_filename = f"eval_{split}.json" if split != "test" else "eval.json"
        out_path = run_dir / out_filename
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

        print(f"[eval] wrote {out_path}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--eval_nsamples", type=int, default=None, help="Number of eval samples (default: all available)")
    p.add_argument("--max_batches", type=int, default=None, help="Max batches for eval (default: all)")
    p.add_argument("--ppl_only", action="store_true")
    p.add_argument("--sequential", action="store_true",
                   help="Load models sequentially to save GPU memory (slower but uses less VRAM)")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"],
                   help="Dataset split to evaluate on (default: test). Use 'train' to eval on calibration data.")
    p.add_argument("--eval_dataset", type=str, default="wikitext2",
                   choices=["wikitext2", "redpajama_sample", "redpajama", "c4"],
                   help="Eval dataset (default: wikitext2)")
    p.add_argument("--eval_nsamples_calib", type=int, default=1024,
                   help="Number of samples for non-wikitext2 eval datasets (default: 1024)")
    p.add_argument("--force_bos", action="store_true",
                   help="Replace first token of each sequence with BOS token")
    p.add_argument("--zero_out_rows", type=str, default="",
                   help="Zero out rows in both models. Format: '6.w1:5723,8518;16.w1:2271,1875'")

    p.add_argument("--init_dist", action="store_true")
    p.add_argument("--master_port_base", type=int, default=29600)

    args = p.parse_args()
    run_eval_job(
        args.run_dir,
        seqlen=args.seqlen,
        eval_nsamples=args.eval_nsamples,
        max_batches=args.max_batches,
        ppl_only=args.ppl_only,
        sequential=args.sequential,
        split=args.split,
        eval_dataset=args.eval_dataset,
        eval_nsamples_calib=args.eval_nsamples_calib,
        force_bos=args.force_bos,
        zero_out_rows=args.zero_out_rows,
        init_dist=args.init_dist,
        master_port_base=args.master_port_base,
    )


if __name__ == "__main__":
    main()
