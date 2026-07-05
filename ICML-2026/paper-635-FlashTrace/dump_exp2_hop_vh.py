#!/usr/bin/env python3
"""One-off: add per-hop IFR vectors (vh) into an existing exp2 trace .npz.

This is useful when the original exp2 run saved sample-level traces but did not
include per-hop vectors for some multi-hop IFR variants (e.g. ifr_multi_hop_both).

Defaults are written to match the reference commands in `exp/exp2/README.md`.

Example (matches the path in the question):

python dump_exp2_hop_vh.py \
  --trace_npz exp/exp2/output/traces/exp/exp2/data/morehopqa.jsonl/qwen-8B/ifr_multi_hop_both_n1_mfaithfulness_gen_95ex/ex_000026.npz \
  --dataset exp/exp2/data/morehopqa.jsonl \
  --attr_func ifr_multi_hop_both \
  --model qwen-8B \
  --model_path /opt/share/models/Qwen/Qwen3-8B/ \
  --cuda 2,3,4,5,6,7 \
  --n_hops 1 \
  --chunk_tokens 128 \
  --sink_chunk_tokens 32 \
  --inplace
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def _early_set_cuda_visible_devices() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cuda", type=str, default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    if args.cuda and "," in str(args.cuda):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)


_early_set_cuda_visible_devices()

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ft_ifr_improve
import llm_attr
from exp.exp2 import dataset_utils as ds_utils


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _resolve_device(cuda: Optional[str], cuda_num: int) -> str:
    """Mirror exp/exp2/run_exp.py device selection policy."""
    if cuda is not None and "," in cuda:
        # _early_set_cuda_visible_devices already applied.
        return "auto"
    if cuda is not None and str(cuda).strip():
        return f"cuda:{cuda}" if torch.cuda.is_available() else "cpu"
    return f"cuda:{int(cuda_num)}" if torch.cuda.is_available() else "cpu"


def _load_model(model_name: str, device: str):
    """Mirror exp/exp2/run_exp.py model loading knobs."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device == "auto" else {"": int(device.split(":")[1])} if device.startswith("cuda:") else None,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


@dataclass(frozen=True)
class ManifestRecord:
    example_idx: int
    prompt_sha1: str
    target_sha1: Optional[str]


def _load_manifest_record(manifest_path: Path, *, example_idx: int) -> Optional[ManifestRecord]:
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if int(obj.get("example_idx", -1)) != int(example_idx):
                continue
            return ManifestRecord(
                example_idx=int(example_idx),
                prompt_sha1=str(obj.get("prompt_sha1") or ""),
                target_sha1=str(obj["target_sha1"]) if obj.get("target_sha1") is not None else None,
            )
    return None


def _parse_example_idx_from_npz_name(path: Path) -> Optional[int]:
    m = re.match(r"^ex_(\d+)$", path.stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _pick_example(
    examples: list[ds_utils.CachedExample],
    *,
    example_idx: int,
    record: Optional[ManifestRecord],
) -> ds_utils.CachedExample:
    if record is not None and record.prompt_sha1:
        matches: list[ds_utils.CachedExample] = []
        for ex in examples:
            if _sha1_text(ex.prompt) != record.prompt_sha1:
                continue
            if record.target_sha1 is None:
                if ex.target is None:
                    matches.append(ex)
            else:
                if ex.target is not None and _sha1_text(ex.target) == record.target_sha1:
                    matches.append(ex)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise SystemExit(
                f"Manifest sha1 matched multiple dataset entries ({len(matches)}). "
                "Please pass --example_idx to select by index or use a smaller dataset cache."
            )
        raise SystemExit(
            "Failed to locate the trace example in the provided dataset by sha1. "
            "Ensure --dataset points to the same cached JSONL used to produce the trace."
        )

    if not (0 <= int(example_idx) < len(examples)):
        raise SystemExit(f"example_idx out of range: {example_idx} not in [0, {len(examples)}).")
    return examples[int(example_idx)]


def _extract_vh(attr: Any) -> np.ndarray:
    ifr = (getattr(attr, "metadata", None) or {}).get("ifr") or {}
    per_hop = ifr.get("per_hop_projected") or []
    if not per_hop:
        raise RuntimeError("Attribution result missing metadata['ifr']['per_hop_projected']; cannot build vh.")
    stacked = torch.stack([torch.as_tensor(v, dtype=torch.float32).reshape(-1) for v in per_hop], dim=0)
    return stacked.detach().cpu().numpy().astype(np.float32, copy=False)


def _run_ifr_attr(
    attr_func: str,
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    target: str,
    sink_span: Optional[tuple[int, int]],
    thinking_span: Optional[tuple[int, int]],
    n_hops: int,
    chunk_tokens: int,
    sink_chunk_tokens: int,
) -> Any:
    if attr_func == "ifr_multi_hop":
        attributor = llm_attr.LLMIFRAttribution(
            model,
            tokenizer,
            chunk_tokens=chunk_tokens,
            sink_chunk_tokens=sink_chunk_tokens,
        )
        return attributor.calculate_ifr_multi_hop(
            prompt,
            target=target,
            sink_span=sink_span,
            thinking_span=thinking_span,
            n_hops=int(n_hops),
        )
    if attr_func == "ifr_in_all_gen":
        attributor = ft_ifr_improve.LLMIFRAttributionInAllGen(
            model,
            tokenizer,
            chunk_tokens=chunk_tokens,
            sink_chunk_tokens=sink_chunk_tokens,
        )
        return attributor.calculate_ifr_in_all_gen(
            prompt,
            target=target,
            sink_span=sink_span,
            thinking_span=thinking_span,
            n_hops=int(n_hops),
        )
    if attr_func == "ifr_multi_hop_stop_words":
        attributor = ft_ifr_improve.LLMIFRAttributionImproved(
            model,
            tokenizer,
            chunk_tokens=chunk_tokens,
            sink_chunk_tokens=sink_chunk_tokens,
        )
        return attributor.calculate_ifr_multi_hop_stop_words(
            prompt,
            target=target,
            sink_span=sink_span,
            thinking_span=thinking_span,
            n_hops=int(n_hops),
        )
    if attr_func == "ifr_multi_hop_both":
        attributor = ft_ifr_improve.LLMIFRAttributionBoth(
            model,
            tokenizer,
            chunk_tokens=chunk_tokens,
            sink_chunk_tokens=sink_chunk_tokens,
        )
        return attributor.calculate_ifr_multi_hop_both(
            prompt,
            target=target,
            sink_span=sink_span,
            thinking_span=thinking_span,
            n_hops=int(n_hops),
        )
    if attr_func == "ifr_multi_hop_split_hop":
        attributor = ft_ifr_improve.LLMIFRAttributionSplitHop(
            model,
            tokenizer,
            chunk_tokens=chunk_tokens,
            sink_chunk_tokens=sink_chunk_tokens,
        )
        return attributor.calculate_ifr_multi_hop_split_hop(
            prompt,
            target=target,
            sink_span=sink_span,
            thinking_span=thinking_span,
            n_hops=int(n_hops),
        )
    raise SystemExit(
        f"Unsupported --attr_func '{attr_func}'. "
        "Supported (vh-capable IFR variants): "
        "ifr_multi_hop, ifr_in_all_gen, ifr_multi_hop_stop_words, ifr_multi_hop_both, ifr_multi_hop_split_hop."
    )


def _save_npz(
    out_path: Path,
    *,
    payload: dict[str, np.ndarray],
    inplace_src: Optional[Path] = None,
    backup: bool = True,
    overwrite_backup: bool = False,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if inplace_src is not None:
        if backup and inplace_src.exists():
            backup_path = inplace_src.with_name(inplace_src.name + ".bak")
            if overwrite_backup and backup_path.exists():
                backup_path.unlink()
            if not backup_path.exists():
                backup_path.write_bytes(inplace_src.read_bytes())

        # NOTE: numpy.savez* appends ".npz" if the filename does not already end with ".npz".
        # So we must ensure our temporary path ends with ".npz", otherwise we'd write
        # "<name>.tmp.npz" but later try to os.replace("<name>.tmp", ...).
        tmp_path = out_path.with_name(out_path.stem + ".tmp.npz")
        if tmp_path.exists():
            tmp_path.unlink()
        np.savez_compressed(tmp_path, **payload)
        os.replace(tmp_path, out_path)
        return

    if out_path.exists():
        raise SystemExit(f"Refusing to overwrite existing file: {out_path} (use --inplace).")
    np.savez_compressed(out_path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser("One-off exp2 trace patcher: add per-hop vh vectors.")
    parser.add_argument(
        "--trace_npz",
        type=str,
        default=(
            "exp/exp2/output/traces/exp/exp2/data/morehopqa.jsonl/qwen-8B/"
            "ifr_multi_hop_both_n1_mfaithfulness_gen_95ex/ex_000026.npz"
        ),
        help="Path to the existing exp2 trace npz (ex_*.npz).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="exp/exp2/data/morehopqa.jsonl",
        help="Path to the exp2 cached dataset JSONL used to produce the trace.",
    )
    parser.add_argument(
        "--attr_func",
        type=str,
        default="ifr_multi_hop_both",
        help="Attribution method to rerun (vh-capable IFR variants only).",
    )
    parser.add_argument("--example_idx", type=int, default=None, help="Override example_idx (0-based).")
    parser.add_argument("--sample", type=int, default=None, help="If the original run used --sample, set it here.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for --sample shuffling (must match original).")

    parser.add_argument("--model", type=str, default="qwen-8B", help="HF repo id (used when --model_path not set).")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/opt/share/models/Qwen/Qwen3-8B/",
        help="Local model path; overrides --model for loading (matches exp2 README examples).",
    )
    parser.add_argument(
        "--cuda",
        type=str,
        default="2,3,4,5,6,7",
        help="CUDA selection (same semantics as exp2): '0' or '0,1,2'.",
    )
    parser.add_argument("--cuda_num", type=int, default=0, help="Single-device index when --cuda not set.")

    parser.add_argument("--chunk_tokens", type=int, default=128)
    parser.add_argument("--sink_chunk_tokens", type=int, default=32)
    parser.add_argument("--n_hops", type=int, default=1)

    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the trace npz in place (recommended so manifest.jsonl stays valid).",
    )
    parser.add_argument("--no_backup", action="store_true", help="Disable .bak creation when using --inplace.")
    parser.add_argument(
        "--overwrite_backup",
        action="store_true",
        help="Allow replacing an existing .bak when using --inplace.",
    )
    args = parser.parse_args()

    trace_npz = Path(args.trace_npz)
    if not trace_npz.exists():
        raise SystemExit(f"Missing trace npz: {trace_npz}")

    example_idx = args.example_idx
    if example_idx is None:
        example_idx = _parse_example_idx_from_npz_name(trace_npz)
    if example_idx is None:
        raise SystemExit("Failed to infer --example_idx from trace filename; please pass --example_idx explicitly.")

    manifest_path = trace_npz.with_name("manifest.jsonl")
    record = _load_manifest_record(manifest_path, example_idx=int(example_idx))

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Missing cached dataset JSONL: {dataset_path}")
    examples = ds_utils.load_cached(dataset_path, sample=args.sample, seed=args.seed)
    ex = _pick_example(examples, example_idx=int(example_idx), record=record)

    if ex.target is None:
        raise SystemExit("Cached dataset example has target=None; this script requires cached targets (CoT+answer).")
    prompt = ex.prompt
    target = ex.target

    sink_span = tuple(ex.sink_span) if ex.sink_span else None
    thinking_span = tuple(ex.thinking_span) if ex.thinking_span else None

    model_name = str(args.model_path or args.model).strip()
    if not model_name:
        raise SystemExit("Please set --model or --model_path.")
    device = _resolve_device(args.cuda, args.cuda_num)
    model, tokenizer = _load_model(model_name, device)

    attr = _run_ifr_attr(
        str(args.attr_func),
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        target=target,
        sink_span=sink_span,
        thinking_span=thinking_span,
        n_hops=int(args.n_hops),
        chunk_tokens=int(args.chunk_tokens),
        sink_chunk_tokens=int(args.sink_chunk_tokens),
    )
    vh = _extract_vh(attr)

    with np.load(trace_npz, allow_pickle=False) as old:
        payload = {k: old[k] for k in old.files}
    payload["vh"] = vh

    if args.inplace:
        out_path = trace_npz
    else:
        out_path = trace_npz.with_name(trace_npz.stem + "_with_vh.npz")

    _save_npz(
        out_path,
        payload=payload,
        inplace_src=trace_npz if args.inplace else None,
        backup=not bool(args.no_backup),
        overwrite_backup=bool(args.overwrite_backup),
    )

    print(f"Saved vh -> {out_path}")
    print(f"vh shape: {vh.shape} (n_hops+1, prompt_len+gen_len)")


if __name__ == "__main__":
    main()
