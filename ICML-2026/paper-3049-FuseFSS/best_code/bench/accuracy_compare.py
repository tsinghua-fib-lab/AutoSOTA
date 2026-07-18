#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    def tqdm(x, **_kwargs):  # type: ignore
        return x

try:
    from datasets import load_dataset
except Exception as exc:
    print("[error] datasets is required: pip install datasets", file=sys.stderr)
    raise

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from transformers.utils import logging as hf_logging
except Exception as exc:
    print("[error] transformers is required: pip install transformers", file=sys.stderr)
    raise

from fixed_point import FixedPointConfig, apply_fixed_point_emulation


GLUE_TEXT_KEYS = {
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Accuracy comparison for Table 4 style eval")
    parser.add_argument("--config", required=True, help="Path to accuracy_table4.json")
    parser.add_argument("--out-json", required=True, help="Output JSON path")
    parser.add_argument("--out-md", required=True, help="Output markdown path")
    parser.add_argument("--device", default=None, help="Legacy alias for --pytorch-device")
    parser.add_argument("--pytorch-device", default=None, help="Device for float32 baseline (cpu or cuda)")
    parser.add_argument("--fusefss-device", "--suf-device", dest="fusefss_device",
                        default=None, help="Device for FuseFSS emulation (cpu or cuda)")
    parser.add_argument("--mpc-device", default=None, help="Device for FuseFSS MPC emulation (cpu or cuda)")
    parser.add_argument("--run-mpc", action="store_true", help="Run FuseFSS MPC emulation")
    parser.add_argument("--skip-pytorch", action="store_true", help="Skip PyTorch baseline eval")
    parser.add_argument("--skip-fusefss", "--skip-suf", dest="skip_fusefss",
                        action="store_true", help="Skip FuseFSS emulation eval")
    parser.add_argument("--skip-mpc", action="store_true", help="Skip MPC emulation eval")
    parser.add_argument("--mpc-rounding", default=None, help="Rounding for MPC emulation (round|trunc|floor)")
    parser.add_argument("--only", default=None, help="Comma-separated list of row ids to run")
    parser.add_argument("--append", action="store_true", help="Append to existing outputs if present")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for GLUE")
    parser.add_argument("--lm-batch-size", type=int, default=1, help="Batch size for LAMBADA")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit eval examples")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--cache-dir", default=None, help="HF/datasets cache dir")
    parser.add_argument("--strict-sigma-match", action="store_true", help="Fail if PyTorch acc diverges from Sigma ref")
    parser.add_argument("--sigma-tol-pp", type=float, default=1.0, help="Tolerance (pp) for strict match")
    parser.add_argument("--debug-sync", action="store_true", help="Synchronize CUDA each batch to surface errors")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)




def maybe_select(dataset, max_examples: Optional[int]):
    if max_examples is None:
        return dataset
    return dataset.select(range(min(max_examples, len(dataset))))


def build_glue_dataloader(task: str, tokenizer, seq_len: int, cache_dir: Optional[str],
                          batch_size: int, max_examples: Optional[int]) -> Tuple[Iterable[Dict[str, torch.Tensor]], int, int]:
    if task not in GLUE_TEXT_KEYS:
        raise ValueError(f"Unsupported GLUE task: {task}")

    train_ds = load_dataset("glue", task, split="train", cache_dir=cache_dir)
    val_ds = load_dataset("glue", task, split="validation", cache_dir=cache_dir)

    train_size = len(train_ds)
    val_ds = maybe_select(val_ds, max_examples)
    val_size = len(val_ds)

    key_a, key_b = GLUE_TEXT_KEYS[task]

    def tokenize_fn(batch):
        if key_b is None:
            return tokenizer(
                batch[key_a],
                truncation=True,
                padding="max_length",
                max_length=seq_len,
            )
        return tokenizer(
            batch[key_a],
            batch[key_b],
            truncation=True,
            padding="max_length",
            max_length=seq_len,
        )

    val_ds = val_ds.map(tokenize_fn, batched=True)

    columns = ["input_ids", "attention_mask", "label"]
    if "token_type_ids" in val_ds.column_names:
        columns.append("token_type_ids")
    val_ds.set_format(type="torch", columns=columns)

    dataloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return dataloader, train_size, val_size


def eval_seqcls(
    model: torch.nn.Module,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    device: str,
    debug_sync: bool = False,
) -> float:
    correct = 0
    total = 0
    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            labels = batch.pop("label")
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = labels.to(device)
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            if debug_sync and device == "cuda":
                torch.cuda.synchronize()
    if total == 0:
        return 0.0
    return correct / total


def load_lambada_dataset(
    cache_dir: Optional[str],
    max_examples: Optional[int],
) -> Tuple[Iterable[Dict[str, Any]], int, str, str]:
    dataset_name = "lambada_openai"
    split_name = "validation"
    try:
        val_ds = load_dataset(dataset_name, split=split_name, cache_dir=cache_dir)
        if len(val_ds) != 5153:
            try:
                test_ds = load_dataset(dataset_name, split="test", cache_dir=cache_dir)
                if len(test_ds) == 5153:
                    val_ds = test_ds
                    split_name = "test"
            except Exception:
                pass
    except Exception:
        dataset_name = "lambada"
        split_name = "validation"
        val_ds = load_dataset(dataset_name, split=split_name, cache_dir=cache_dir)
        if len(val_ds) != 5153:
            try:
                test_ds = load_dataset(dataset_name, split="test", cache_dir=cache_dir)
                if len(test_ds) == 5153:
                    val_ds = test_ds
                    split_name = "test"
            except Exception:
                pass
    val_ds = maybe_select(val_ds, max_examples)
    return val_ds, len(val_ds), dataset_name, split_name


def eval_lambada(
    model: torch.nn.Module,
    tokenizer,
    dataset: Iterable[Dict[str, Any]],
    device: str,
    seq_len: int,
    batch_size: int,
    debug_sync: bool = False,
) -> Tuple[float, int]:
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    correct = 0
    total = 0

    batch_prompts: List[List[int]] = []
    batch_labels: List[int] = []

    def run_batch(prompts: List[List[int]], labels: List[int]) -> None:
        nonlocal correct, total
        if not prompts:
            return
        max_len = max(len(p) for p in prompts)
        input_ids = []
        attention_mask = []
        for p in prompts:
            pad_len = max_len - len(p)
            input_ids.append(p + [tokenizer.pad_token_id] * pad_len)
            attention_mask.append([1] * len(p) + [0] * pad_len)
        input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
        mask_t = torch.tensor(attention_mask, dtype=torch.long, device=device)
        with torch.inference_mode():
            outputs = model(input_ids=input_ids_t, attention_mask=mask_t)
            logits = outputs.logits
        last_idx = mask_t.sum(dim=1) - 1
        batch_indices = torch.arange(logits.size(0), device=device)
        last_logits = logits[batch_indices, last_idx]
        preds = torch.argmax(last_logits, dim=-1)
        labels_t = torch.tensor(labels, dtype=torch.long, device=device)
        correct += (preds == labels_t).sum().item()
        total += labels_t.numel()
        if debug_sync and device == "cuda":
            torch.cuda.synchronize()

    text_key = "text"
    if dataset and isinstance(dataset[0], dict) and "text" not in dataset[0]:
        if "sentence" in dataset[0]:
            text_key = "sentence"

    for sample in tqdm(dataset, desc="LAMBADA"):
        text = sample.get(text_key)
        if text is None:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < 2:
            continue
        label_id = ids[-1]
        prompt = ids[:-1]
        if len(prompt) > seq_len:
            prompt = prompt[-seq_len:]
        batch_prompts.append(prompt)
        batch_labels.append(label_id)
        if len(batch_prompts) >= batch_size:
            run_batch(batch_prompts, batch_labels)
            batch_prompts = []
            batch_labels = []

    run_batch(batch_prompts, batch_labels)

    if total == 0:
        return 0.0, total
    return correct / total, total


def load_model_for_row(family: str, checkpoint: str, cache_dir: Optional[str]) -> torch.nn.Module:
    if family == "bert_seqcls":
        return AutoModelForSequenceClassification.from_pretrained(checkpoint, cache_dir=cache_dir)
    if family == "gpt_lm":
        return AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=cache_dir)
    raise ValueError(f"Unknown family: {family}")


def load_tokenizer(checkpoint: str, cache_dir: Optional[str]):
    return AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir, use_fast=True)


def format_pct(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def escape_md(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value)
    text = text.replace("|", "\\|")
    text = text.replace("\n", " ")
    return text


def main() -> int:
    args = parse_args()
    hf_logging.set_verbosity_error()

    cfg = load_config(args.config)
    global_cfg = cfg.get("global", {})

    pytorch_device = args.pytorch_device or args.device or global_cfg.get("default_device", "cpu")
    fusefss_device = args.fusefss_device or pytorch_device
    mpc_device = args.mpc_device or fusefss_device

    if pytorch_device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested for baseline but not available, falling back to CPU")
        pytorch_device = "cpu"
    if fusefss_device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested for FuseFSS but not available, falling back to CPU")
        fusefss_device = "cpu"
    if mpc_device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested for MPC but not available, falling back to CPU")
        mpc_device = "cpu"

    seed = args.seed if args.seed is not None else global_cfg.get("seed", 0)
    set_seed(seed)

    cache_dir = args.cache_dir if args.cache_dir is not None else global_cfg.get("cache_dir")
    seq_len = global_cfg.get("seq_len", 128)
    frac_bits = global_cfg.get("frac_bits", 12)

    results: List[Dict[str, Any]] = []
    only_ids = None
    if args.only:
        only_ids = {rid.strip() for rid in args.only.split(",") if rid.strip()}

    existing_by_id: Dict[str, Dict[str, Any]] = {}
    out_json = Path(args.out_json)
    if args.append and out_json.exists():
        with out_json.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        if isinstance(existing, list):
            existing_by_id = {item.get("id"): item for item in existing}

    run_mpc_global = (args.run_mpc or global_cfg.get("run_mpc", False)) and not args.skip_mpc

    for row in cfg.get("rows", []):
        row_id = row.get("id", "")
        if only_ids is not None and row_id not in only_ids:
            continue
        existing_row = existing_by_id.get(row_id)
        if row.get("skip", False):
            results.append({
                "id": row_id,
                "status": "skipped",
                "notes": row.get("notes", ""),
                "sigma_ref": row.get("sigma_ref", {}),
                "family": row.get("family"),
                "task": row.get("task"),
                "n_bits": row.get("n_bits"),
                "frac_bits": row.get("frac_bits", frac_bits),
            })
            continue

        family = row.get("family")
        task = row.get("task")
        checkpoint = row.get("hf_checkpoint")
        if not family or not task or not checkpoint:
            raise ValueError(f"Row {row_id} missing required fields")

        row_seq_len = row.get("seq_len", seq_len)
        row_frac_bits = row.get("frac_bits", frac_bits)
        n_bits = row.get("n_bits")
        sigma_ref = row.get("sigma_ref", {})
        row_pytorch_device = row.get("pytorch_device", pytorch_device)
        row_fusefss_device = row.get("fusefss_device", fusefss_device)
        row_mpc_device = row.get("mpc_device", mpc_device)
        fusefss_rounding = row.get(
            "fusefss_rounding",
            row.get("suf_rounding", global_cfg.get("fusefss_rounding", global_cfg.get("suf_rounding", "round"))),
        )
        mpc_rounding = row.get("mpc_rounding", args.mpc_rounding or global_cfg.get("mpc_rounding", "trunc"))

        if row_pytorch_device == "cuda" and not torch.cuda.is_available():
            print(f"[warn] {row_id}: CUDA requested for baseline but not available, falling back to CPU")
            row_pytorch_device = "cpu"
        if row_fusefss_device == "cuda" and not torch.cuda.is_available():
            print(f"[warn] {row_id}: CUDA requested for FuseFSS but not available, falling back to CPU")
            row_fusefss_device = "cpu"
        if row_mpc_device == "cuda" and not torch.cuda.is_available():
            print(f"[warn] {row_id}: CUDA requested for MPC but not available, falling back to CPU")
            row_mpc_device = "cpu"

        print(f"[info] running {row_id} ({task}) on {checkpoint}")
        print(f"[info] devices: baseline={row_pytorch_device}, fusefss={row_fusefss_device}, mpc={row_mpc_device}")
        t0 = time.time()

        tokenizer = load_tokenizer(row.get("tokenizer", checkpoint), cache_dir)

        extra_note = None
        run_pytorch = not args.skip_pytorch
        run_fusefss = not args.skip_fusefss and n_bits is not None
        run_mpc = run_mpc_global and n_bits is not None

        if not (run_pytorch or run_fusefss or run_mpc) and existing_row:
            results.append(existing_row)
            continue

        if family == "bert_seqcls":
            glue_task = task.split("/")[-1]
            dataloader, train_size, val_size = build_glue_dataloader(
                glue_task,
                tokenizer,
                row_seq_len,
                cache_dir,
                row.get("batch_size", args.batch_size),
                args.max_examples,
            )

            pytorch_acc = None
            if run_pytorch:
                model = load_model_for_row(family, checkpoint, cache_dir).to(row_pytorch_device)
                pytorch_acc = eval_seqcls(model, dataloader, row_pytorch_device, debug_sync=args.debug_sync)
                del model
                if row_pytorch_device == "cuda":
                    torch.cuda.empty_cache()
            elif existing_row and existing_row.get("pytorch"):
                pytorch_acc = existing_row["pytorch"].get("acc")

            fusefss_acc = None
            if run_fusefss:
                model = load_model_for_row(family, checkpoint, cache_dir).to(row_fusefss_device)
                cfg_fp = FixedPointConfig(n_bits=n_bits, frac_bits=row_frac_bits, rounding=fusefss_rounding)
                apply_fixed_point_emulation(model, cfg_fp, quantize_weights=True)
                fusefss_acc = eval_seqcls(model, dataloader, row_fusefss_device, debug_sync=args.debug_sync)
                del model
                if row_fusefss_device == "cuda":
                    torch.cuda.empty_cache()
            elif existing_row and existing_row.get("fusefss"):
                fusefss_acc = existing_row["fusefss"].get("acc")

            mpc_acc = None
            if run_mpc:
                model = load_model_for_row(family, checkpoint, cache_dir).to(row_mpc_device)
                cfg_fp = FixedPointConfig(n_bits=n_bits, frac_bits=row_frac_bits, rounding=mpc_rounding)
                apply_fixed_point_emulation(model, cfg_fp, quantize_weights=True)
                mpc_acc = eval_seqcls(model, dataloader, row_mpc_device, debug_sync=args.debug_sync)
                del model
                if row_mpc_device == "cuda":
                    torch.cuda.empty_cache()
            elif existing_row and existing_row.get("mpc"):
                mpc_acc = existing_row["mpc"].get("acc")

        elif family == "gpt_lm":
            dataset, val_size, dataset_name, split_name = load_lambada_dataset(cache_dir, args.max_examples)
            dataset_note = None
            if dataset_name != "lambada_openai" or split_name != "validation":
                dataset_note = f"dataset={dataset_name}:{split_name}"
            extra_note = dataset_note
            train_size = sigma_ref.get("train_size")

            pytorch_acc = None
            if run_pytorch:
                model = load_model_for_row(family, checkpoint, cache_dir).to(row_pytorch_device)
                pytorch_acc, used = eval_lambada(
                    model,
                    tokenizer,
                    dataset,
                    row_pytorch_device,
                    row_seq_len,
                    row.get("batch_size", args.lm_batch_size),
                    debug_sync=args.debug_sync,
                )
                if used and used != val_size:
                    val_size = used
                del model
                if row_pytorch_device == "cuda":
                    torch.cuda.empty_cache()
            elif existing_row and existing_row.get("pytorch"):
                pytorch_acc = existing_row["pytorch"].get("acc")

            fusefss_acc = None
            if run_fusefss:
                model = load_model_for_row(family, checkpoint, cache_dir).to(row_fusefss_device)
                cfg_fp = FixedPointConfig(n_bits=n_bits, frac_bits=row_frac_bits, rounding=fusefss_rounding)
                apply_fixed_point_emulation(model, cfg_fp, quantize_weights=True)
                fusefss_acc, used = eval_lambada(
                    model,
                    tokenizer,
                    dataset,
                    row_fusefss_device,
                    row_seq_len,
                    row.get("batch_size", args.lm_batch_size),
                    debug_sync=args.debug_sync,
                )
                if used and used != val_size:
                    val_size = used
                del model
                if row_fusefss_device == "cuda":
                    torch.cuda.empty_cache()
            elif existing_row and existing_row.get("fusefss"):
                fusefss_acc = existing_row["fusefss"].get("acc")

            mpc_acc = None
            if run_mpc:
                model = load_model_for_row(family, checkpoint, cache_dir).to(row_mpc_device)
                cfg_fp = FixedPointConfig(n_bits=n_bits, frac_bits=row_frac_bits, rounding=mpc_rounding)
                apply_fixed_point_emulation(model, cfg_fp, quantize_weights=True)
                mpc_acc, used = eval_lambada(
                    model,
                    tokenizer,
                    dataset,
                    row_mpc_device,
                    row_seq_len,
                    row.get("batch_size", args.lm_batch_size),
                    debug_sync=args.debug_sync,
                )
                if used and used != val_size:
                    val_size = used
                del model
                if row_mpc_device == "cuda":
                    torch.cuda.empty_cache()
            elif existing_row and existing_row.get("mpc"):
                mpc_acc = existing_row["mpc"].get("acc")
            task = f"{task} ({dataset_name}:{split_name})"
        else:
            raise ValueError(f"Unknown family: {family}")

        pytorch_pct = pytorch_acc * 100.0 if pytorch_acc is not None else None
        fusefss_pct = fusefss_acc * 100.0 if fusefss_acc is not None else None
        mpc_pct = mpc_acc * 100.0 if mpc_acc is not None else None
        sigma_pytorch = sigma_ref.get("pytorch_acc")
        sigma_acc = sigma_ref.get("sigma_acc")

        if args.strict_sigma_match and sigma_pytorch is not None:
            if pytorch_pct is None:
                raise RuntimeError(f"{row_id}: strict match requested but pytorch acc missing")
            if abs(pytorch_pct - sigma_pytorch) > args.sigma_tol_pp:
                raise RuntimeError(
                    f"{row_id}: PyTorch acc {pytorch_pct:.2f} diverges from sigma ref {sigma_pytorch:.2f}"
                )

        notes = row.get("notes", "")
        if extra_note:
            notes = f"{notes} | {extra_note}" if notes else extra_note
        sigma_val = sigma_ref.get("val_size")
        if sigma_val and val_size and val_size != sigma_val:
            suffix = f"val_size={val_size} (sigma {sigma_val})"
            notes = f"{notes} | {suffix}" if notes else suffix

        result = {
            "id": row_id,
            "family": family,
            "task": task,
            "seq_len": row_seq_len,
            "n_bits": n_bits,
            "frac_bits": row_frac_bits,
            "train_size": train_size,
            "val_size": val_size,
            "pytorch": {"acc": pytorch_acc, "acc_pct": pytorch_pct},
            "fusefss": {"acc": fusefss_acc, "acc_pct": fusefss_pct},
            "mpc": {"acc": mpc_acc, "acc_pct": mpc_pct},
            "sigma_ref": sigma_ref,
            "delta": {
                "fusefss_minus_pytorch_pp": None if (fusefss_pct is None or pytorch_pct is None) else fusefss_pct - pytorch_pct,
                "mpc_minus_pytorch_pp": None if (mpc_pct is None or pytorch_pct is None) else mpc_pct - pytorch_pct,
                "mpc_minus_fusefss_pp": None if (mpc_pct is None or fusefss_pct is None) else mpc_pct - fusefss_pct,
                "fusefss_minus_sigma_pp": None if (fusefss_pct is None or sigma_acc is None) else fusefss_pct - sigma_acc,
                "mpc_minus_sigma_pp": None if (mpc_pct is None or sigma_acc is None) else mpc_pct - sigma_acc,
                "pytorch_minus_sigma_pp": None if (pytorch_pct is None or sigma_acc is None) else pytorch_pct - sigma_acc,
            },
            "notes": notes,
            "runtime_s": time.time() - t0,
        }
        results.append(result)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    merged_results = results
    if existing_by_id:
        for item in results:
            existing_by_id[item.get("id")] = item
        merged_results = list(existing_by_id.values())
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(merged_results, f, indent=2)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w", encoding="utf-8") as f:
        f.write("| Model | Dataset | Train Size | Val Size | PyTorch Acc | Sigma Acc | FuseFSS Acc | FuseFSS MPC Acc | Bitwidth | frac_bits | Notes |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for r in merged_results:
            if r.get("status") == "skipped":
                sigma_ref = r.get("sigma_ref", {})
                f.write(
                    f"| {escape_md(r.get('id',''))} | SKIPPED | {sigma_ref.get('train_size','-')} | {sigma_ref.get('val_size','-')} | - | {sigma_ref.get('sigma_acc','-')} | - | - | {r.get('n_bits','-')} | {r.get('frac_bits','-')} | {escape_md(r.get('notes',''))} |\n"
                )
                continue
            sigma_ref = r.get("sigma_ref", {})
            model_name = escape_md(r.get("id", ""))
            dataset = escape_md(r.get("task", ""))
            f.write(
                "| {model} | {dataset} | {train} | {val} | {pt} | {sigma} | {fusefss} | {mpc} | {bits} | {frac} | {notes} |\n".format(
                    model=model_name,
                    dataset=dataset,
                    train=r.get("train_size", "-"),
                    val=r.get("val_size", "-"),
                    pt=format_pct(r.get("pytorch", {}).get("acc_pct")),
                    sigma=format_pct(sigma_ref.get("sigma_acc")),
                    fusefss=format_pct(r.get("fusefss", {}).get("acc_pct")),
                    mpc=format_pct(r.get("mpc", {}).get("acc_pct")),
                    bits=r.get("n_bits", "-"),
                    frac=r.get("frac_bits", "-"),
                    notes=escape_md(r.get("notes", "")),
                )
            )

    print(f"[done] wrote {out_json} and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
