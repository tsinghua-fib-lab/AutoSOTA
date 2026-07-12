"""Export per-token explorer data for the project page (static, pre-computed).

Produces the small JSON the project page's per-token explorer reads at runtime
(no model is run in the browser; see docs/adr/0001). For each model in the
``tab:progressive_modifications`` table (4 families x {baseline, low-dim}) and a
handful of samples, it emits, per token of a crammed sample:

  * the display text of the token (exact slice of the original text),
  * its base-model surprisal  s(L) = -log2 p(x_L | x_<L)  in bits (one frozen
    forward pass per (model, sample) -- forward only, no training),
  * its steps-to-converge under progressive cramming (read from the run's
    ``progressive_prefixes`` dataset; per-stage vs cumulative auto-detected),
  * whether that stage converged, and the compression horizon (longest exactly
    reconstructed prefix).

This must run on a machine that has the progressive run dirs (the private
research artifacts); the committed JSON under ``page/data/`` is the shipped
artifact. Re-run example (from the research repo root)::

    HF_HOME=/path/to/hf_cache python scripts/export_token_explorer_data.py \
        --exp_root artifacts/experiments_progressive \
        --out_dir /path/to/progressive_cramming/page/data \
        --samples 0 1 2

Surprisal alignment mirrors ``scripts/analyze_surprisal_vs_steps.py``: entry j of
the surprisal array predicts the token at 0-index j+1, so the token shown at
0-index ``i`` (prefix length ``L = i+1``) has surprisal ``surp[i-1]`` and
steps-to-converge from the stage whose ``stage_seq_len == L``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

_TRAIN_SPLIT_DATASETS = {"LarryLovestein/pg19_1k", "LarryLovestein/fanfics_1k"}


def model_registry(exp_root: str):
    """The 8 runs behind ``tab:progressive_modifications`` (baseline + low-dim per family)."""
    e = exp_root.rstrip("/")
    return [
        {
            "key": "llama31_8b",
            "name": "Llama-3.1-8B",
            "ckpt": "unsloth/Meta-Llama-3.1-8B",
            "lr": "0.1",
            "lowdim_label": "dim=256",
            "variants": {
                "base": f"{e}/sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lr_0.1",
                "lowdim": f"{e}/sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lowdim_256_lowproj_lr_0.1",
            },
        },
        {
            "key": "pythia_1.4b",
            "name": "Pythia-1.4B",
            "ckpt": "EleutherAI/pythia-1.4b",
            "lr": "0.5",
            "lowdim_label": "dim=256",
            "variants": {
                "base": f"{e}/sl_4096_pythia-1.4b_ds_pg19_1k_limit_50_lr_0.5",
                "lowdim": f"{e}/sl_4096_pythia-1.4b_ds_pg19_1k_limit_50_lowdim_256_lowproj_lr_0.5",
            },
        },
        {
            "key": "smollm2_1.7b",
            "name": "SmolLM2-1.7B",
            "ckpt": "HuggingFaceTB/SmolLM2-1.7B",
            "lr": "0.1",
            "lowdim_label": "dim=256",
            "variants": {
                "base": f"{e}/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lr_0.1",
                "lowdim": f"{e}/sl_4096_SmolLM2-1.7B_ds_pg19_1k_limit_50_lowdim_256_lowproj_lr_0.1",
            },
        },
        {
            "key": "gemma3_4b",
            "name": "Gemma-3-4B",
            "ckpt": "unsloth/gemma-3-4b-pt",
            "lr": "0.1",
            "lowdim_label": "dim=32",
            "variants": {
                "base": f"{e}/sl_4096_gemma-3-4b-pt_ds_pg19_1k_limit_50_lr_0.1",
                "lowdim": f"{e}/sl_4096_gemma-3-4b-pt_ds_pg19_1k_limit_50_lowdim_32_lowproj_lr_0.1",
            },
        },
    ]


def parse_cmd_txt(run_dir: str) -> dict:
    """Recover the run's key knobs from the persisted CLI (cmd.txt)."""
    path = os.path.join(run_dir, "cmd.txt")
    cfg: dict = {}
    if not os.path.exists(path):
        return cfg
    with open(path, encoding="utf-8") as f:
        toks = shlex.split(f.read().strip())
    flag_keys = {
        "--model_checkpoint": "model_checkpoint",
        "--dataset_name": "dataset_name",
        "--max_sequence_length": "max_sequence_length",
        "--limit_dataset_items": "limit",
        "--progressive_step": "progressive_step",
        "--no_bos_token": "no_bos_token",
    }
    i = 0
    while i < len(toks):
        if toks[i] in flag_keys:
            cfg[flag_keys[toks[i]]] = toks[i + 1] if i + 1 < len(toks) else True
            i += 2
        else:
            i += 1
    return cfg


@torch.no_grad()
def per_token_surprisal_bits(model, input_ids: torch.Tensor) -> np.ndarray:
    """s_i = -log2 p(x_i | x_<i) in bits; entry j predicts token at 0-index j+1."""
    logits = model(input_ids=input_ids).logits
    logp = F.log_softmax(logits[0, :-1].float(), dim=-1)
    targets = input_ids[0, 1:]
    nll = -logp[torch.arange(targets.shape[0], device=targets.device), targets]
    return (nll / math.log(2)).cpu().numpy()


def stage_map(run_dir: str):
    """Return {sample_id: sorted [(seq_len, steps_taken, convergence)]} and cumulative flag."""
    ds = load_from_disk(os.path.join(run_dir, "progressive_prefixes"))
    keep = [c for c in ("sample_id", "stage_seq_len", "steps_taken", "final_convergence") if c in ds.column_names]
    ds = ds.select_columns(keep)
    rows = defaultdict(list)
    sid = ds["sample_id"]
    seq = ds["stage_seq_len"]
    steps = ds["steps_taken"]
    conv = ds["final_convergence"]
    for s, L, st, c in zip(sid, seq, steps, conv):
        rows[int(s)].append((int(L), int(st), float(c)))
    for rr in rows.values():
        rr.sort()
    # detect cumulative vs per-stage: a cumulative steps sequence never decreases across
    # consecutive converged unit-step stages.
    dec = comp = 0
    for rr in rows.values():
        for k in range(1, len(rr)):
            (L, st, c), (Lp, stp, cp) = rr[k], rr[k - 1]
            if L - Lp == 1 and c == 1.0 and cp == 1.0:
                comp += 1
                dec += int(st < stp)
    cumulative = comp > 0 and (dec / comp) < 0.02
    return rows, cumulative


def per_token_steps(rr, cumulative: bool, n_tokens: int):
    """Arrays steps[L-1], converged[L-1] for prefix length L=1..n_tokens (NaN/0 where missing)."""
    by_len = {L: (st, c) for (L, st, c) in rr}
    steps = [None] * n_tokens
    converged = [0] * n_tokens
    prev_steps = None
    for L in range(1, n_tokens + 1):
        if L not in by_len:
            prev_steps = None
            continue
        st, c = by_len[L]
        converged[L - 1] = int(c >= 0.999)
        if cumulative and prev_steps is not None:
            cost = st - prev_steps
        else:
            cost = st
        steps[L - 1] = int(cost) if cost is not None else None
        prev_steps = st
    return steps, converged


def consecutive_horizon(converged) -> int:
    """Longest exactly-reconstructed prefix: first index that failed, else full length."""
    h = 0
    for c in converged:
        if c:
            h += 1
        else:
            break
    return h


def display_tokens(tok, text: str, input_ids, no_bos: bool):
    """Exact display slice per token via fast-tokenizer offsets, with a BOS marker."""
    enc = tok(
        text,
        truncation=True,
        max_length=len(input_ids),
        add_special_tokens=not no_bos,
        return_offsets_mapping=True,
    )
    offs = enc["offset_mapping"]
    out = []
    for (a, b) in offs:
        if b <= a:  # special token (e.g. BOS) has empty span
            out.append("‹bos›")
        else:
            out.append(text[a:b])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp_root", default="artifacts/experiments_progressive")
    ap.add_argument("--out_dir", required=True, help="Where to write <model_key>.json + manifest.json")
    ap.add_argument("--samples", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--display_margin", type=int, default=30, help="Tokens shown past the horizon (the stall region).")
    ap.add_argument("--text_column", default="text")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    manifest = {"models": []}

    for spec in model_registry(args.exp_root):
        base_dir = spec["variants"]["base"]
        if not os.path.exists(os.path.join(base_dir, "progressive_prefixes")):
            print(f"[skip] {spec['name']}: missing {base_dir}")
            continue
        cfg = parse_cmd_txt(base_dir)
        ckpt = cfg.get("model_checkpoint", spec["ckpt"])
        dataset_name = cfg.get("dataset_name", "LarryLovestein/pg19_1k")
        max_length = int(cfg.get("max_sequence_length", 4096))
        no_bos = bool(cfg.get("no_bos_token", False))
        split = "train" if dataset_name in _TRAIN_SPLIT_DATASETS else "test"

        print(f"\n=== {spec['name']} ({ckpt}) ===")
        raw = load_dataset(dataset_name, split=split)
        tok = AutoTokenizer.from_pretrained(ckpt)
        tok.pad_token = tok.eos_token
        if no_bos and hasattr(tok, "add_bos_token"):
            tok.add_bos_token = False
        model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16).to(device).eval()

        # stage maps per variant
        variant_maps = {}
        for vk, vdir in spec["variants"].items():
            if os.path.exists(os.path.join(vdir, "progressive_prefixes")):
                variant_maps[vk] = stage_map(vdir)
            else:
                print(f"  [warn] missing variant {vk}: {vdir}")

        model_out = {k: spec[k] for k in ("key", "name", "ckpt", "lr", "lowdim_label")}
        model_out["ckpt"] = ckpt
        model_out["samples"] = {}
        man_samples = []

        for sid in args.samples:
            if sid >= len(raw):
                continue
            text = raw[sid][args.text_column]
            enc = tok(text, truncation=True, max_length=max_length, add_special_tokens=not no_bos)
            ids = torch.tensor(enc["input_ids"]).unsqueeze(0).to(device)
            surp = per_token_surprisal_bits(model, ids)  # length n-1
            disp = display_tokens(tok, text, enc["input_ids"], no_bos)
            n_full = len(enc["input_ids"])

            # horizon across variants -> display length
            horizons = {}
            steps_conv = {}
            for vk, (rows, cumulative) in variant_maps.items():
                rr = rows.get(sid, [])
                steps, converged = per_token_steps(rr, cumulative, n_full)
                horizons[vk] = consecutive_horizon(converged)
                steps_conv[vk] = (steps, converged)
            if not horizons:
                continue
            n_disp = min(n_full, max(horizons.values()) + args.display_margin)

            # surprisal[i] = surp[i-1] for i>=1 (token at 0-index i predicted by entry i-1); i=0 -> None
            surprisal = [None] + [round(float(surp[i - 1]), 3) for i in range(1, n_disp)]
            sample_entry = {
                "tokens": disp[:n_disp],
                "surprisal": surprisal[:n_disp],
                "variants": {},
            }
            for vk, (steps, converged) in steps_conv.items():
                sample_entry["variants"][vk] = {
                    "horizon": horizons[vk],
                    "steps": steps[:n_disp],
                    "converged": converged[:n_disp],
                }
            model_out["samples"][str(sid)] = sample_entry

            preview = "".join(disp[1 : min(n_disp, 14)]).replace("\n", " ").strip()
            man_samples.append(
                {
                    "id": sid,
                    "n_tokens": n_disp,
                    "horizon_base": horizons.get("base"),
                    "horizon_lowdim": horizons.get("lowdim"),
                    "preview": (preview[:80] + "…") if len(preview) > 80 else preview,
                }
            )
            print(
                f"  sample {sid}: n_disp={n_disp} horizons={horizons} "
                f"surprisal[mean]={np.nanmean([s for s in surprisal[1:] if s is not None]):.2f}"
            )

        out_path = os.path.join(args.out_dir, f"{spec['key']}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(model_out, f, ensure_ascii=False, separators=(",", ":"))
        sz = os.path.getsize(out_path) / 1024
        print(f"  wrote {out_path} ({sz:.0f} KB)")
        manifest["models"].append(
            {
                "key": spec["key"],
                "name": spec["name"],
                "ckpt": ckpt,
                "lr": spec["lr"],
                "lowdim_label": spec["lowdim_label"],
                "samples": man_samples,
            }
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {os.path.join(args.out_dir, 'manifest.json')} ({len(manifest['models'])} models)")


if __name__ == "__main__":
    main()
