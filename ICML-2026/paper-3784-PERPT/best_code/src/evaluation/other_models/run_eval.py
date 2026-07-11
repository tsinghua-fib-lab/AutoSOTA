#!/usr/bin/env python3
"""
one-pass embedding cache + 4 tasks

inputs (under regions_root):
  regions/CATEGORY/chr*.bed
  regions/CATEGORY_upstream/chr*.bed
  regions/CATEGORY_random/chr*.bed
  regions/CATEGORY_random-noannot/chr*.bed

assumptions:
- all BEDs have 7 columns: chrom, start, end, name, score, strand, pair_id
- all functional ROIs are already exclusive / non-overlapping (built upstream)
- pair_id is shared across ROI + upstream + random + random-noannot (per category)

goals:
- minimize forward passes: embed each unique (region window) exactly once per split
- avoid memory blowup: never keep token-level reps; pool immediately; save cache per split

nt note:
- nucleotide transformer uses nonoverlapping 6-mer tokenization
- mapping bp spans -> token spans must account for that + a leading special token
"""

import gc
import argparse
import os
import json
import logging
import pathlib
import shutil
from pathlib import Path
from types import MethodType

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import pyBigWig
from pyfaidx import Fasta
from tqdm import tqdm

#from evo2 import Evo2
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoModel,
    AutoTokenizer,
    AutoConfig,
)

import umap
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
)

import sys
sys.path.append("/home/mica/scratch/gamba/")
from src.evaluation.utils.helpers import extract_context  # uses bigwig + genome


# -----------------------------------------------------------------------------
# logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -----------------------------------------------------------------------------
# model init + loaders
# -----------------------------------------------------------------------------
def vishniakov_init(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def patched_forward(
    self,
    input_ids=None,
    inputs_embeds=None,
    labels=None,
    loss_weights=None,
    output_hidden_states=None,
    return_dict=None,
):
    from transformers.modeling_outputs import MaskedLMOutput
    from torch.nn.functional import cross_entropy

    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.caduceus(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states).float()

    loss = None
    if labels is not None:
        loss = cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return MaskedLMOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
    )


# evo2 config
EVO2_LAYER_NAME = "blocks.26"


def load_model(model_type="nt-ms", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"[load_model] type={model_type} device={device}")

    if model_type == "nt-ms":
        name = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
        model = AutoModelForMaskedLM.from_pretrained(name, trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        return model.to(device).eval(), tok

    if model_type == "nt-ms-random-init":
        name = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
        torch.manual_seed(42)
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_config(cfg, trust_remote_code=True)
        model.apply(vishniakov_init)
        return model.to(device).eval(), tok

    if model_type == "nt-human":
        name = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
        model = AutoModelForMaskedLM.from_pretrained(name, trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        return model.to(device).eval(), tok

    if model_type == "nt-human-random-init":
        name = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
        torch.manual_seed(42)
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_config(cfg, trust_remote_code=True)
        model.apply(vishniakov_init)
        return model.to(device).eval(), tok

    if model_type == "hyenaDNA":
        ckpt = "LongSafari/hyenadna-medium-160k-seqlen-hf"
        tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            ckpt,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        return model.to(device).eval(), tok

    if model_type == "hyenaDNA-random-init":
        ckpt = "LongSafari/hyenadna-medium-160k-seqlen-hf"
        torch.manual_seed(42)
        tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_config(
            cfg, trust_remote_code=True
        )
        model.apply(vishniakov_init)
        return model.to(device).eval(), tok

    if model_type == "phyloGPN":
        ckpt = "songlab/PhyloGPN"
        tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        model = AutoModel.from_pretrained(ckpt, trust_remote_code=True)
        return model.to(device).eval(), tok

    if model_type == "phyloGPN-random-init":
        ckpt = "songlab/PhyloGPN"
        torch.manual_seed(42)
        tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
        model = AutoModel.from_config(cfg, trust_remote_code=True)
        model.apply(vishniakov_init)
        return model.to(device).eval(), tok

    if model_type == "caduceus-theirs":
        model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True
        ).to(device)
        model.forward = MethodType(patched_forward, model)
        return model.eval(), tokenizer

    if model_type == "caduceus-theirs-random-init":
        model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
        torch.manual_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_config(cfg, trust_remote_code=True)
        model.apply(vishniakov_init)
        model = model.to(device).eval()
        model.forward = MethodType(patched_forward, model)
        return model, tokenizer

    if model_type == "evo2":
        model = Evo2("evo2_7b")
        return model, None

    if model_type == "evo2-random-init":
        model = Evo2("evo2_7b")
        return model, None

    raise ValueError(f"unsupported model_type: {model_type}")


# -----------------------------------------------------------------------------
# io helpers
# -----------------------------------------------------------------------------
def read_pair_bed(path: Path, category: str, role: str):
    """
    expects:
      chrom  start  end  name  score  strand  pair_id
    """
    regions = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 7:
                continue
            try:
                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                name = parts[3]
                score = float(parts[4])
                strand = parts[5]
                pair_id = parts[6]
            except ValueError:
                continue
            if end <= start:
                continue
            regions.append(
                dict(
                    chrom=chrom,
                    start=start,
                    end=end,
                    name=name,
                    score=score,
                    strand=strand,
                    pair_id=pair_id,
                    category=category,
                    role=role,
                )
            )
    return regions


def iter_category_beds(dirpath: Path):
    if not dirpath.exists():
        return []
    return sorted(dirpath.glob("chr*.bed"))


# -----------------------------------------------------------------------------
# pooling helpers (no token-level retention)
# -----------------------------------------------------------------------------
def _valid_T_from_attention_mask(attn_mask_1d):
    if attn_mask_1d is None:
        return None
    return int(attn_mask_1d.long().sum().item())


def _nt_tok0_from_tokenizer(tokenizer):
    # based on your empirical check: 60bp -> 11 tokens => 10 kmers + 1 special
    return 1


def _bp_span_to_token_span_nt_nonoverlap(fs_bp, fe_bp, valid_T, tok0=1, k=6):
    """
    nonoverlapping k-mer tokenization with a leading special token:
      tokens = [SPECIAL] + kmer0 + kmer1 + ...
    """
    if fe_bp <= fs_bp:
        return None

    tfs = fs_bp // k
    tfe = (fe_bp + k - 1) // k  # ceil

    tfs += tok0
    tfe += tok0

    if valid_T is None:
        valid_T = tfe

    tfs = max(tok0, min(tfs, valid_T - 1))
    tfe = max(tfs + 1, min(tfe, valid_T))
    return tfs, tfe


def _bp_span_to_token_span_linear(T, seq_len_bp, fs_bp, fe_bp):
    # fallback mapping for models without a fixed k-mer stride
    scale = T / max(1, seq_len_bp)
    tfs = max(0, min(int(np.floor(fs_bp * scale)), T - 1))
    tfe = max(tfs + 1, min(int(np.ceil(fe_bp * scale)), T))
    return tfs, tfe


def pool_from_rep(
    rep_TH,
    window_len_bp,
    fs_bp,
    fe_bp,
    roi100_seed=None,
    mapping="linear",      # "linear" | "nt6"
    valid_T=None,          # number of non-pad tokens
    tok0=1,
    k=6,
):
    """
    rep_TH: torch.Tensor [T,H] (cpu float32 preferred)
    returns:
      roi_mean [H], full_mean [H], roi100_mean [H]
    """
    rep_TH = rep_TH.to(torch.float32)
    T_all, H = rep_TH.shape

    T = valid_T if valid_T is not None else T_all
    rep = rep_TH[:T]  # ignore padding tokens for pooling

    # full mean over valid tokens
    full_vec = rep.mean(dim=0)

    # roi mean
    if mapping == "nt6":
        span = _bp_span_to_token_span_nt_nonoverlap(fs_bp, fe_bp, valid_T=T, tok0=tok0, k=k)
        if span is None:
            return None
        tfs, tfe = span
    else:
        tfs, tfe = _bp_span_to_token_span_linear(T, window_len_bp, fs_bp, fe_bp)

    roi_vec = rep[tfs:tfe].mean(dim=0)

    # roi 100bp mean (uniform start within [fs, fe-100])
    roi100_vec = roi_vec
    if roi100_seed is not None:
        roi_len = max(0, fe_bp - fs_bp)
        if roi_len >= 100:
            rng = np.random.RandomState(int(roi100_seed) & 0xFFFFFFFF)
            sub_start = int(rng.randint(fs_bp, fe_bp - 100 + 1))
            sub_end = sub_start + 100

            if mapping == "nt6":
                span2 = _bp_span_to_token_span_nt_nonoverlap(sub_start, sub_end, valid_T=T, tok0=tok0, k=k)
                if span2 is not None:
                    tss, tse = span2
                    roi100_vec = rep[tss:tse].mean(dim=0)
            else:
                tss, tse = _bp_span_to_token_span_linear(T, window_len_bp, sub_start, sub_end)
                roi100_vec = rep[tss:tse].mean(dim=0)

    return (
        roi_vec.detach().cpu().numpy().astype(np.float32),
        full_vec.detach().cpu().numpy().astype(np.float32),
        roi100_vec.detach().cpu().numpy().astype(np.float32),
    )


# -----------------------------------------------------------------------------
# caching: build item list and embed once per split
# -----------------------------------------------------------------------------
def build_items_for_group(regions_root: Path, categories, group_chroms):
    group_chroms = set(group_chroms)

    role_dirs = {
        "roi": lambda c: regions_root / c,
        "upstream": lambda c: regions_root / f"{c}_upstream",
        "random": lambda c: regions_root / f"{c}_random",
        "random-noannot": lambda c: regions_root / f"{c}_random-noannot",
    }

    # load all region maps first
    per = {c: {r: {} for r in role_dirs.keys()} for c in categories}
    for c in categories:
        for role, dfn in role_dirs.items():
            d = dfn(c)
            regs = []
            for bf in iter_category_beds(d):
                regs.extend(read_pair_bed(bf, c, role))
            regs = [r for r in regs if r["chrom"] in group_chroms]
            per[c][role] = {str(r["pair_id"]): r for r in regs}

    # enforce 4-way intersection
    pairs_common = {}
    for c in categories:
        roi_p = set(per[c]["roi"].keys())
        up_p  = set(per[c]["upstream"].keys())
        r_p   = set(per[c]["random"].keys())
        rn_p  = set(per[c]["random-noannot"].keys())
        pairs_common[c] = sorted(roi_p & up_p & r_p & rn_p)

    needed = set()
    for c in categories:
        for pid in pairs_common[c]:
            for role in ("roi", "upstream", "random", "random-noannot"):
                needed.add((c, pid, role))

    items = []
    for (c, pid, role) in sorted(needed):
        r = per[c][role].get(pid)
        if r is None:
            continue
        items.append(dict(key=f"{c}|{pid}|{role}", **r))

    # keep return shape unchanged for rest of pipeline
    pairs_upstream = pairs_common
    pairs_random = pairs_common
    pairs_random_noannot = pairs_common
    
    return items, pairs_upstream, pairs_random, pairs_random_noannot


def model_name_for_context(model_type: str):
    if "random-init" in model_type:
        model_type = model_type.replace("-random-init", "")
    if "caduceus" in model_type:
        return "caduceus-theirs"
    return model_type


def embed_items_to_cache(
    model,
    tokenizer,
    model_type: str,
    genome: Fasta,
    bigwig_file: str,
    items,
    batch_size: int,
    device: torch.device,
    cache_npz_path: Path,
    cache_meta_path: Path,
    roi100_seed_base: int = 12345,
):
    out_n = len(items)
    if out_n == 0:
        raise ValueError("no items to embed")

    cache_npz_path.parent.mkdir(parents=True, exist_ok=True)
    cache_meta_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Progress tracking
    progress_file = cache_npz_path.with_suffix('.progress.json')
    
    # Check for existing progress
    start_idx = 0
    H = None
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            start_idx = progress.get('last_completed_idx', 0) + 1
            H = progress.get('embedding_dim')
            logging.info(f"[resume] continuing from index {start_idx}, H={H}")
    
    # Memory-mapped arrays (auto-saves to disk)
    memmap_dir = cache_npz_path.parent / f"{cache_npz_path.stem}_memmap"
    memmap_dir.mkdir(exist_ok=True)
    
    def get_or_create_memmap(name, shape, dtype=np.float32):
        path = memmap_dir / f"{name}.npy"
        if path.exists() and start_idx > 0:
            # Resume: load existing memmap
            return np.memmap(path, dtype=dtype, mode='r+', shape=shape)
        else:
            # New: create fresh memmap
            return np.memmap(path, dtype=dtype, mode='w+', shape=shape)
    
    # Metadata tracking
    meta_rows = []
    meta_cache = cache_meta_path.with_suffix('.meta_cache.jsonl')
    
    if meta_cache.exists() and start_idx > 0:
        # Load existing metadata
        with open(meta_cache, 'r') as f:
            for line in f:
                meta_rows.append(json.loads(line))
        logging.info(f"[resume] loaded {len(meta_rows)} existing metadata rows")
    
    ctx_model_name = model_name_for_context(model_type)
    tok0_nt = _nt_tok0_from_tokenizer(tokenizer) if model_type.startswith("nt-") else 1
    
    # Initialize on first batch or resume
    roi_mean = None
    full_mean = None
    roi100_mean = None
    valid = None
    
    pbar = tqdm(total=out_n, initial=start_idx, desc="embedding items (pooled)")
    idx = start_idx

    while idx < out_n:
        batch_items = items[idx : idx + batch_size]
        
        contexts = []
        ctx_indices = []

        for j, it in enumerate(batch_items):
            r = dict(it)
            ctx = extract_context(bigwig_file, r, genome, model_type=ctx_model_name)
            if not ctx or "sequence" not in ctx or not ctx["sequence"]:
                if idx + j >= len(meta_rows):
                    meta_rows.append({**it, "window_len": None, "fs": None, "fe": None, "valid": False})
                pbar.update(1)
                continue

            fs = int(ctx.get("feature_start_in_window", 0))
            fe = int(ctx.get("feature_end_in_window", len(ctx["sequence"])))
            ctx["feature_start_in_window"] = fs
            ctx["feature_end_in_window"] = fe
            ctx["category"] = it["category"]
            ctx["role"] = it["role"]
            ctx["pair_id"] = it["pair_id"]
            ctx["key"] = it["key"]

            contexts.append(ctx)
            ctx_indices.append(idx + j)

        if not contexts:
            idx += batch_size
            continue

        # evo2 path (per sequence)
        if model_type.startswith("evo2"):
            for ctx, gi in zip(contexts, ctx_indices):
                seq = ctx["sequence"]
                fs = int(ctx["feature_start_in_window"])
                fe = int(ctx["feature_end_in_window"])
                window_len = int(len(seq))

                token_ids = torch.tensor(
                    model.tokenizer.tokenize(seq),
                    dtype=torch.int,
                    device=device,
                ).unsqueeze(0)

                with torch.no_grad():
                    _, emb_dict = model(
                        token_ids,
                        return_embeddings=True,
                        layer_names=[EVO2_LAYER_NAME],
                    )
                    rep = emb_dict[EVO2_LAYER_NAME][0].to(torch.float32).cpu()  # [T,H]

                H_batch = rep.shape[1]
                
                # Initialize memmap on first batch
                if roi_mean is None:
                    H = H_batch
                    roi_mean = get_or_create_memmap("roi_mean", (out_n, H))
                    full_mean = get_or_create_memmap("full_mean", (out_n, H))
                    roi100_mean = get_or_create_memmap("roi100_mean", (out_n, H))
                    valid = get_or_create_memmap("valid", (out_n,), dtype=bool)
                    logging.info(f"[memmap] initialized with H={H}")

                seed = None
                if ctx["role"] == "roi":
                    seed = roi100_seed_base + (hash(ctx["key"]) & 0x7FFFFFFF)

                pooled = pool_from_rep(
                    rep,
                    window_len,
                    fs,
                    fe,
                    roi100_seed=seed,
                    mapping="linear",
                    valid_T=None,
                )
                if pooled is None:
                    if gi >= len(meta_rows):
                        meta_rows.append({**items[gi], "window_len": window_len, "fs": fs, "fe": fe, "valid": False})
                    continue
                rvec, fvec, r100 = pooled

                roi_mean[gi] = rvec
                full_mean[gi] = fvec
                roi100_mean[gi] = r100
                valid[gi] = True

                if gi >= len(meta_rows):
                    meta_rows.append({**items[gi], "window_len": window_len, "fs": fs, "fe": fe, "valid": True})

                del rep, token_ids
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            # Flush memmap after evo2 batch
            roi_mean.flush()
            full_mean.flush()
            roi100_mean.flush()
            valid.flush()
            
            # Save metadata incrementally
            with open(meta_cache, 'w') as f:
                for row in meta_rows:
                    f.write(json.dumps(row) + '\n')
            
            # Save progress
            with open(progress_file, 'w') as f:
                json.dump({
                    'last_completed_idx': idx + len(batch_items) - 1,
                    'embedding_dim': H,
                    'total_items': out_n,
                }, f)

            pbar.update(len(contexts))
            idx += batch_size
            continue

        # HF batched path
        batch_seqs = [c["sequence"] for c in contexts]
        batch_fs = [int(c["feature_start_in_window"]) for c in contexts]
        batch_fe = [int(c["feature_end_in_window"]) for c in contexts]
        batch_wl = [int(len(c["sequence"])) for c in contexts]

        # phyloGPN length constraint
        if model_type.startswith("phyloGPN"):
            keep = [k for k, s in enumerate(batch_seqs) if len(s) == 481]
            if len(keep) != len(batch_seqs):
                drop = len(batch_seqs) - len(keep)
                logging.warning(f"[phyloGPN] dropping {drop} invalid-length sequences")
            contexts = [contexts[k] for k in keep]
            ctx_indices = [ctx_indices[k] for k in keep]
            batch_seqs = [batch_seqs[k] for k in keep]
            batch_fs = [batch_fs[k] for k in keep]
            batch_fe = [batch_fe[k] for k in keep]
            batch_wl = [batch_wl[k] for k in keep]
            if not batch_seqs:
                idx += batch_size
                continue

        if model_type.startswith("nt-"):
            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1000,  # tokens (6kb bp)
            )
        elif model_type.startswith("hyenaDNA"):
            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
        else:
            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if model_type.startswith("phyloGPN") and hasattr(model, "get_embeddings"):
                last_hidden = model.get_embeddings(inputs["input_ids"]).to(torch.float32)
                attn = inputs.get("attention_mask", None)
            else:
                out = model(**inputs, output_hidden_states=True)
                last_hidden = out.hidden_states[-1].to(torch.float32)
                attn = inputs.get("attention_mask", None)

        B, T, H_batch = last_hidden.shape
        
        # Initialize memmap on first batch
        if roi_mean is None:
            H = H_batch
            roi_mean = get_or_create_memmap("roi_mean", (out_n, H))
            full_mean = get_or_create_memmap("full_mean", (out_n, H))
            roi100_mean = get_or_create_memmap("roi100_mean", (out_n, H))
            valid = get_or_create_memmap("valid", (out_n,), dtype=bool)
            logging.info(f"[memmap] initialized with H={H}")

        for b, (rep, gi, fs, fe, wl, ctx) in enumerate(
            zip(last_hidden, ctx_indices, batch_fs, batch_fe, batch_wl, contexts)
        ):
            rep = rep.detach().cpu()  # [T,H]

            valid_T = _valid_T_from_attention_mask(attn[b]) if attn is not None else None

            seed = None
            if ctx["role"] == "roi":
                seed = roi100_seed_base + (hash(ctx["key"]) & 0x7FFFFFFF)

            if model_type.startswith("nt-"):
                pooled = pool_from_rep(
                    rep, wl, fs, fe,
                    roi100_seed=seed,
                    mapping="nt6",
                    valid_T=valid_T,
                    tok0=tok0_nt,
                    k=6,
                )
            else:
                pooled = pool_from_rep(
                    rep, wl, fs, fe,
                    roi100_seed=seed,
                    mapping="linear",
                    valid_T=valid_T,
                )

            if pooled is None:
                if gi >= len(meta_rows):
                    meta_rows.append({**items[gi], "window_len": wl, "fs": fs, "fe": fe, "valid": False})
                continue

            rvec, fvec, r100 = pooled
            
            # Write directly to memmap (auto-saves to disk)
            roi_mean[gi] = rvec
            full_mean[gi] = fvec
            roi100_mean[gi] = r100
            valid[gi] = True

            if gi >= len(meta_rows):
                meta_rows.append({**items[gi], "window_len": wl, "fs": fs, "fe": fe, "valid": True})

        # Flush memmap to disk after each batch
        roi_mean.flush()
        full_mean.flush()
        roi100_mean.flush()
        valid.flush()
        
        # Save metadata incrementally
        with open(meta_cache, 'w') as f:
            for row in meta_rows:
                f.write(json.dumps(row) + '\n')
        
        # Save progress
        with open(progress_file, 'w') as f:
            json.dump({
                'last_completed_idx': idx + len(batch_items) - 1,
                'embedding_dim': H,
                'total_items': out_n,
            }, f)

        del last_hidden, inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

        pbar.update(len(contexts))
        idx += batch_size

    pbar.close()

    # Final: convert memmap to compressed npz
    logging.info("[finalize] converting memmap to compressed npz...")
    valid_mask = np.asarray(valid).astype(bool)
    
    np.savez_compressed(
        cache_npz_path,
        roi_mean=np.asarray(roi_mean),
        full_mean=np.asarray(full_mean),
        roi100_mean=np.asarray(roi100_mean),
        valid=valid_mask,
        keys=np.asarray([it["key"] for it in items]),
    )

    # Save metadata
    meta_df = pd.DataFrame(items)
    meta_df = meta_df.merge(
        pd.DataFrame(meta_rows),
        on=["key", "chrom", "start", "end", "name", "score", "strand", "pair_id", "category", "role"],
        how="left",
        suffixes=("", "_y"),
    )
    if "valid" not in meta_df.columns:
        meta_df["valid"] = False
    meta_df["valid"] = meta_df["valid"].fillna(False).astype(bool)
    meta_df["valid"] = valid_mask
    meta_df.to_parquet(cache_meta_path, index=False)

    # Cleanup temp files
    logging.info("[cleanup] removing temporary memmap files...")

    # Explicitly delete memmap references to close file handles
    del roi_mean, full_mean, roi100_mean, valid

    # Force garbage collection to ensure file handles are released
    import gc
    gc.collect()

    # Now try to remove the directory with error handling
    try:
        shutil.rmtree(memmap_dir)
    except OSError as e:
        logging.warning(f"[cleanup] failed to remove {memmap_dir}: {e}")
        logging.warning("[cleanup] attempting manual file removal...")
        for item in memmap_dir.iterdir():
            try:
                if item.is_file():
                    item.unlink()
            except Exception as e2:
                logging.warning(f"[cleanup] failed to remove {item}: {e2}")
        try:
            memmap_dir.rmdir()
        except Exception as e3:
            logging.warning(f"[cleanup] failed to remove directory: {e3}")

    # Check if files exist before unlinking
    if progress_file.exists():
        progress_file.unlink()
    if meta_cache.exists():
        meta_cache.unlink()

    logging.info(f"[cache] wrote {cache_npz_path}")
    logging.info(f"[cache] wrote {cache_meta_path}")

    return cache_npz_path, cache_meta_path


def load_cache(cache_npz_path: Path, cache_meta_path: Path):
    z = np.load(cache_npz_path, allow_pickle=True)
    meta = pd.read_parquet(cache_meta_path)

    valid = np.asarray(z["valid"]).astype(bool)
    meta = meta.loc[valid].reset_index(drop=True)

    roi_mean = np.asarray(z["roi_mean"])[valid]
    full_mean = np.asarray(z["full_mean"])[valid]
    roi100_mean = np.asarray(z["roi100_mean"])[valid]

    return meta, roi_mean, full_mean, roi100_mean


# -----------------------------------------------------------------------------
# metrics + plotting
# -----------------------------------------------------------------------------
def plot_umap(embeddings, labels, output_path, title):
    if len(embeddings) == 0:
        logging.warning(f"[UMAP] no embeddings for {title}")
        return
    um = umap.UMAP()
    emb2d = um.fit_transform(np.asarray(embeddings))
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=emb2d[:, 0], y=emb2d[:, 1], hue=labels, s=20, alpha=0.8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def loo_1nn_predictions(embeddings, labels):
    labels = np.asarray(labels)
    X = np.asarray(embeddings)
    nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(X)
    _, idx = nn.kneighbors(X)
    y_true = labels
    y_pred = labels[idx[:, 1]]
    return y_true, y_pred


def eval_metrics(y_true, y_pred, label_order=None):
    if label_order is None:
        label_order = np.unique(y_true)

    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    row_sums = cm.sum(axis=1, keepdims=True)
    per_class_recall = np.diag(cm) / np.where(row_sums == 0, 1, row_sums).squeeze()

    valid = ~np.isnan(per_class_recall)
    ba = float(np.mean(per_class_recall[valid]))
    sem = float(np.std(per_class_recall[valid], ddof=1) / np.sqrt(np.sum(valid)))
    ci95 = float(1.96 * sem)

    metrics = {
        "micro_accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": ba,
        "balanced_accuracy_sem": sem,
        "balanced_accuracy_ci95": ci95,
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                y_true, y_pred, labels=label_order, average="weighted", zero_division=0
            )
        ),
        "cohens_kappa": float(cohen_kappa_score(y_true, y_pred, labels=label_order)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "per_class_recall": dict(zip(label_order, per_class_recall.astype(float))),
        "support": dict(zip(label_order, cm.sum(axis=1).astype(int))),
    }
    return cm, metrics, label_order


def plot_knn_heatmap(embeddings, labels, output_path, title):
    if len(embeddings) == 0:
        logging.warning(f"[KNN] no embeddings for {title}")
        return None, None, None

    y_true, y_pred = loo_1nn_predictions(embeddings, labels)
    present = sorted(set(labels))
    cm, metrics, label_order = eval_metrics(y_true, y_pred, label_order=present)

    with np.errstate(invalid="ignore", divide="ignore"):
        acc_matrix = cm.astype(float) / np.where(
            cm.sum(axis=1, keepdims=True) == 0, 1, cm.sum(axis=1, keepdims=True)
        )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        acc_matrix,
        xticklabels=label_order,
        yticklabels=label_order,
        vmin=0,
        vmax=1,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Per-class recall"},
    )
    plt.title(
        f"{title}\n"
        f"micro={metrics['micro_accuracy']:.2%} | "
        f"balanced={metrics['balanced_accuracy']:.2%} | "
        f"macro-F1={metrics['macro_f1']:.2%}"
    )
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    logging.info(
        f"[KNN] {title} | micro={metrics['micro_accuracy']:.3f}, "
        f"balanced={metrics['balanced_accuracy']:.3f}, "
        f"macroF1={metrics['macro_f1']:.3f}, "
        f"weightedF1={metrics['weighted_f1']:.3f}, "
        f"kappa={metrics['cohens_kappa']:.3f}, "
        f"mcc={metrics['mcc']:.3f}"
    )
    return metrics, label_order, acc_matrix


def _save_per_class_json(json_path, label_order, acc_matrix):
    data = {
        "label_order": list(map(str, label_order)),
        "per_class_recall": {str(lbl): float(acc_matrix[i, i]) for i, lbl in enumerate(label_order)},
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def _append_summary(csv_path, row_dict):
    csv_path = pathlib.Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row_dict])
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False)


def save_reps(output_dir, model_id, tag, X, labels, metas):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels)

    np.savez_compressed(
        out / f"reps_{model_id}_{tag}.npz",
        embeddings=X,
        labels=labels,
    )
    mdf = pd.DataFrame(metas)
    if "label" in mdf.columns:
        mdf["label"] = labels
    else:
        mdf.insert(0, "label", labels)
    mdf.to_parquet(out / f"reps_{model_id}_{tag}_meta.parquet", index=False)


# -----------------------------------------------------------------------------
# task evaluators (no more forward passes)
# -----------------------------------------------------------------------------
def eval_binary_task_from_cache(
    meta,
    roi_mean,
    full_mean,
    roi100_mean,
    output_dir: Path,
    model_type: str,
    group_name: str,
    categories,
    task_name: str,
    use_scope: str,
):
    assert use_scope in ("roi", "full")
    scope_arr = roi_mean if use_scope == "roi" else full_mean

    summary_csv = output_dir / f"binary_{task_name}_knn_summary.csv"

    for cat in categories:
        m_roi = meta[(meta["category"] == cat) & (meta["role"] == "roi")]
        m_ctl = meta[(meta["category"] == cat) & (meta["role"] == task_name)]

        if len(m_roi) == 0 or len(m_ctl) == 0:
            logging.warning(f"[{group_name}] {cat} {task_name}: missing roi/control, skipping")
            continue

        roi_idx = m_roi.reset_index().rename(columns={"index": "i_roi"})[["pair_id", "i_roi"]]
        ctl_idx = m_ctl.reset_index().rename(columns={"index": "i_ctl"})[["pair_id", "i_ctl"]]
        jn = roi_idx.merge(ctl_idx, on="pair_id", how="inner")
        if len(jn) == 0:
            logging.warning(f"[{group_name}] {cat} {task_name}: no matching pair_ids, skipping")
            continue

        X_roi = scope_arr[jn["i_roi"].to_numpy()]
        X_ctl = scope_arr[jn["i_ctl"].to_numpy()]

        X = np.concatenate([X_roi, X_ctl], axis=0)
        y = np.asarray(["feature"] * len(X_roi) + [task_name] * len(X_ctl))

        metas = []
        for _, row in meta.loc[jn["i_roi"].to_numpy()].iterrows():
            metas.append(dict(**row.to_dict(), pair_label="feature", task=task_name, scope=use_scope))
        for _, row in meta.loc[jn["i_ctl"].to_numpy()].iterrows():
            metas.append(dict(**row.to_dict(), pair_label=task_name, task=task_name, scope=use_scope))

        title = f"{cat} – feature vs {task_name} ({group_name}, {use_scope})"
        plot_umap(
            X,
            y,
            output_dir / f"umap_{model_type}_{task_name}_{cat}_{group_name}_{use_scope}.png",
            title=title,
        )
        metrics, order, mat = plot_knn_heatmap(
            X,
            y,
            output_dir / f"knn_{model_type}_{task_name}_{cat}_{group_name}_{use_scope}.png",
            title=title,
        )
        if metrics is None:
            continue

        _save_per_class_json(
            output_dir / f"per_class_{model_type}_{task_name}_{cat}_{group_name}_{use_scope}.json",
            order,
            mat,
        )

        tag = f"{group_name}_{task_name}_{cat}_{use_scope}"
        save_reps(output_dir, model_type, tag, X, y, metas)

        _append_summary(
            summary_csv,
            dict(
                Model=model_type,
                Group=group_name,
                Task=task_name,
                Category=cat,
                Scope=use_scope,
                NPairs=int(len(jn)),
                BalancedAccuracyPct=100.0 * metrics["balanced_accuracy"],
                BalancedAccuracySEM_Pct=100.0 * metrics["balanced_accuracy_sem"],
                MicroAccuracyPct=100.0 * metrics["micro_accuracy"],
                MacroF1Pct=100.0 * metrics["macro_f1"],
            ),
        )
        
def eval_multiclass_from_cache(
    meta,
    roi_mean,
    roi100_mean,
    output_dir: Path,
    model_type: str,
    group_name: str,
    categories,
):
    m = meta[(meta["role"] == "roi") & (meta["category"].isin(categories))].copy()
    if len(m) == 0:
        logging.warning(f"[{group_name}] multiclass: no roi rows, skipping")
        return

    # REMOVED: All the balancing logic
    # Just use all ROI regions directly
    idx = m.index.to_numpy()
    y = m["category"].to_numpy()

    for scope_name, Xsrc in [("roi", roi_mean), ("roi100bp", roi100_mean)]:
        X = Xsrc[idx]

        title = f"multiclass ROI ({group_name}, {scope_name})"
        plot_umap(
            X,
            y,
            output_dir / f"umap_{model_type}_multiclass_{group_name}_{scope_name}.png",
            title=title,
        )
        metrics, order, mat = plot_knn_heatmap(
            X,
            y,
            output_dir / f"knn_{model_type}_multiclass_{group_name}_{scope_name}.png",
            title=title,
        )
        if metrics is None:
            continue

        _save_per_class_json(
            output_dir / f"per_class_{model_type}_multiclass_{group_name}_{scope_name}.json",
            order,
            mat,
        )

        metas = m.to_dict("records")
        tag = f"{group_name}_multiclass_{scope_name}"
        save_reps(output_dir, model_type, tag, X, y, metas)

        _append_summary(
            output_dir / "multiclass_knn_summary.csv",
            dict(
                Model=model_type,
                Group=group_name,
                Task="multiclass",
                Scope=scope_name,
                NTotal=int(len(m)),  # Changed from NPerClass
                NCats=int(len(np.unique(y))),  # Changed to count actual categories
                BalancedAccuracyPct=100.0 * metrics["balanced_accuracy"],
                BalancedAccuracySEM_Pct=100.0 * metrics["balanced_accuracy_sem"],
                MicroAccuracyPct=100.0 * metrics["micro_accuracy"],
                MacroF1Pct=100.0 * metrics["macro_f1"],
            ),
        )


# -----------------------------------------------------------------------------
# main analysis: one cache pass per group, then run tasks from cache
# -----------------------------------------------------------------------------
def analyze_all_tasks(
    genome_fasta: str,
    bigwig_file: str,
    regions_root: str,
    output_dir: str,
    categories,
    chromosomes,
    training_chromosomes,
    test_chromosomes,
    batch_size: int,
    model_type: str,
    num_regions_per_category: int = None,
    resume_cache: bool = False,
):
    regions_root = Path(regions_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device: {device}")

    model, tokenizer = load_model(model_type=model_type, device=device)
    genome = Fasta(genome_fasta)

    if training_chromosomes and test_chromosomes:
        chrom_groups = {
            "training": training_chromosomes,
            "test": test_chromosomes,
        }
    else:
        chrom_groups = {"all": chromosomes}

    for group_name, group_chroms in chrom_groups.items():
        logging.info(f"=== group={group_name} ===")

        grp_out = output_dir / group_name
        grp_out.mkdir(parents=True, exist_ok=True)

        cache_npz = grp_out / f"cache_{model_type}_{group_name}.npz"
        cache_meta = grp_out / f"cache_{model_type}_{group_name}_meta.parquet"

        if resume_cache and cache_npz.exists() and cache_meta.exists():
            logging.info(f"[resume] using existing cache: {cache_npz.name}")
        else:
            items, pairs_up, pairs_r, pairs_rn = build_items_for_group(
                regions_root=regions_root,
                categories=categories,
                group_chroms=group_chroms,
            )

            if num_regions_per_category is not None:
                cap = int(num_regions_per_category)
                keep_keys = set()
                rng = np.random.RandomState(42)

                for cat in categories:
                    roi_keys = [it["key"] for it in items if it["category"] == cat and it["role"] == "roi"]
                    if len(roi_keys) > cap:
                        roi_keys = list(rng.choice(roi_keys, size=cap, replace=False))
                    keep_keys.update(roi_keys)

                def cap_task(role, pairs_by_cat):
                    for cat in categories:
                        pids = list(pairs_by_cat.get(cat, []))
                        if len(pids) > cap:
                            pids = list(rng.choice(pids, size=cap, replace=False))
                        for pid in pids:
                            keep_keys.add(f"{cat}|{pid}|roi")
                            keep_keys.add(f"{cat}|{pid}|{role}")

                cap_task("upstream", pairs_up)
                cap_task("random", pairs_r)
                cap_task("random-noannot", pairs_rn)

                items = [it for it in items if it["key"] in keep_keys]
                logging.info(f"[cap] kept {len(items)} items after cap={cap}")

            embed_items_to_cache(
                model=model,
                tokenizer=tokenizer,
                model_type=model_type,
                genome=genome,
                bigwig_file=bigwig_file,
                items=items,
                batch_size=batch_size,
                device=device,
                cache_npz_path=cache_npz,
                cache_meta_path=cache_meta,
            )

        meta, roi_mean, full_mean, roi100_mean = load_cache(cache_npz, cache_meta)
        logging.info(f"[cache] loaded valid rows: {len(meta)}")

        for task in ["upstream", "random", "random-noannot"]:
            for scope in ["roi", "full"]:
                eval_binary_task_from_cache(
                    meta=meta,
                    roi_mean=roi_mean,
                    full_mean=full_mean,
                    roi100_mean=roi100_mean,
                    output_dir=grp_out,
                    model_type=model_type,
                    group_name=group_name,
                    categories=categories,
                    task_name=task,
                    use_scope=scope,
                )

        eval_multiclass_from_cache(
            meta=meta,
            roi_mean=roi_mean,
            roi100_mean=roi100_mean,
            output_dir=grp_out,
            model_type=model_type,
            group_name=group_name,
            categories=categories,
        )

        del meta, roi_mean, full_mean, roi100_mean
        if device.type == "cuda":
            torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# cli
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="one-pass cache embedding + upstream/random/random-noannot/multiclass tasks"
    )
    p.add_argument(
        "--bigwig_file",
        type=str,
        default="/home/mica/scratch/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig",
    )
    p.add_argument(
        "--genome_fasta",
        type=str,
        default="/home/mica/scratch/gamba/data_processing/data/240-mammalian/hg38.ml.fa",
    )
    p.add_argument(
        "--regions_root",
        type=str,
        default="/home/mica/scratch/gamba/data_processing/data/regions_common",
        help="root containing CATEGORY/, CATEGORY_upstream/, CATEGORY_random/, CATEGORY_random-noannot/",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/home/mica/scratch/gamba/other-models/final_representations/all_tasks",
    )
    p.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=[
            "vista_enhancer",
            "UCNE",
            "repeats",
            "exons",
            "introns",
            "noncoding_regions",
            "coding_regions",
            "upstream_TSS",
            "UTR5",
            "UTR3",
            "promoters",
        ],
    )
    p.add_argument("--num_regions_per_category", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument(
        "--model_type",
        type=str,
        choices=[
            "hyenaDNA",
            "hyenaDNA-random-init",
            "nt-ms",
            "nt-ms-random-init",
            "nt-human",
            "nt-human-random-init",
            "phyloGPN",
            "phyloGPN-random-init",
            "caduceus-theirs",
            "caduceus-theirs-random-init",
            "evo2",
            "evo2-random-init",
        ],
        required=True,
    )
    p.add_argument(
        "--chromosomes",
        type=str,
        nargs="+",
        default=[
            "chr1","chr2","chr3","chr4","chr5","chr6","chr7","chr8","chr9","chr10",
            "chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19","chr20",
            "chr21","chr22","chrX",
        ],
    )
    p.add_argument("--training_chromosomes", type=str, nargs="+", default=None)
    p.add_argument("--test_chromosomes", type=str, nargs="+", default=None)
    p.add_argument(
        "--resume_cache",
        action="store_true",
        help="skip embedding cache if cache files already exist for each group",
    )

    args = p.parse_args()

    outdir = Path(args.output_dir) / args.model_type
    outdir.mkdir(parents=True, exist_ok=True)

    analyze_all_tasks(
        genome_fasta=args.genome_fasta,
        bigwig_file=args.bigwig_file,
        regions_root=args.regions_root,
        output_dir=str(outdir),
        categories=args.categories,
        chromosomes=args.chromosomes,
        training_chromosomes=args.training_chromosomes,
        test_chromosomes=args.test_chromosomes,
        batch_size=args.batch_size,
        model_type=args.model_type,
        num_regions_per_category=args.num_regions_per_category,
        resume_cache=args.resume_cache,
    )


if __name__ == "__main__":
    main()