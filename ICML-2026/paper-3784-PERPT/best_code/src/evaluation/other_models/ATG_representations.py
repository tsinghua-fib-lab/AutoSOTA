#!/usr/bin/env python3
"""
ATG 5-way representation eval for "other models" (nt/hyena/phyloGPN/evo2), with stratified sampling.

what it does:
- loads ONE TSV: the simplified 5-way format
- samples N_EXAMPLES total, evenly across chromosomes present in the TSV
  (default: 10000 examples total -> 50000 contexts since 5 labels/example)
- builds 5 contexts per example via extract_context (codon windows at each label position)
- embeds each context, pools ROI tokens to one vector
- saves:
    reps_{model_type}_ATG5way_all_labels_roi.npz
    reps_{model_type}_ATG5way_all_labels_roi_meta.parquet
- runs:
  - 5-way 1-NN confusion heatmap on labels 1..5
  - binary 1-NN tasks: 1 vs each of 2..5

notes:
- phyloGPN (drops contexts with seq length != 481).
"""

import argparse
import os
import json
import logging
from pathlib import Path
from collections import defaultdict
from types import MethodType

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

#from evo2 import Evo2  

import pyBigWig  # noqa: F401
from pyfaidx import Fasta

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoModel,
    AutoTokenizer,
    AutoConfig,
)

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
)

import sys
sys.path.append("/home/mica/scratch/gamba/")
from src.evaluation.utils.helpers import extract_context


# ---------------- logging ----------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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

EVO2_LAYER_NAME = "blocks.26"

# ---------------- 5-way schema ----------------

LABEL_COLS_5WAY = {
    1: "label1_start_pos",
    2: "label2_noncoding_near_pos",
    3: "label3_noncoding_far_pos",
    4: "label4_same_inframe_met_pos",
    5: "label5_same_outframe_atg_pos",
}
DELTA_COLS_5WAY = {
    1: None,
    2: "label2_delta_bp",
    3: "label3_delta_bp",
    4: "label4_delta_bp",
    5: "label5_delta_bp",
}


# ---------------- init helpers + model loader ----------------

def vishniakov_init(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


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
        model = AutoModelForSequenceClassification.from_config(cfg, trust_remote_code=True)
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

    if model_type == "evo2":
        from evo2 import Evo2
        model = Evo2("evo2_7b")
        return model, None

    if model_type == "evo2-random-init":
        from evo2 import Evo2
        model = Evo2("evo2_7b")
        logging.warning("[evo2-random-init] random initialization not implemented for Evo2; using pretrained weights")
        return model, None

    raise ValueError(f"unsupported model_type: {model_type}")


# ---------------- saving reps ----------------

def save_reps(base_dir, model_type, name, X, labels, metas, extra=None):
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels)

    prefix = f"reps_{model_type}_{name}"
    np.savez_compressed(
        base_dir / f"{prefix}.npz",
        embeddings=X,
        labels=labels,
    )

    mdf = pd.DataFrame(metas)
    if "label" in mdf.columns:
        mdf["label"] = labels
    else:
        mdf.insert(0, "label", labels)

    if extra:
        for k, v in extra.items():
            mdf[k] = v

    mdf.to_parquet(base_dir / f"{prefix}_meta.parquet", index=False)


# ---------------- KNN + metrics helpers ----------------

def loo_1nn_predictions(embeddings, labels):
    labels = np.asarray(labels)
    X = np.asarray(embeddings)
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(X)
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
    sem = float(np.std(per_class_recall[valid], ddof=1) / np.sqrt(np.sum(valid))) if np.sum(valid) > 1 else 0.0
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
            f1_score(y_true, y_pred, labels=label_order, average="weighted", zero_division=0)
        ),
        "cohens_kappa": float(cohen_kappa_score(y_true, y_pred, labels=label_order)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "per_class_recall": dict(zip(label_order, per_class_recall.astype(float))),
        "support": dict(zip(label_order, cm.sum(axis=1).astype(int))),
    }
    return cm, metrics, label_order


def plot_knn_heatmap(embeddings, labels, output_path, title="1-NN"):
    if len(embeddings) == 0:
        logging.warning("[plot_knn_heatmap] no embeddings to plot")
        return None, None, None

    labels = np.asarray(labels)
    present = sorted(set(labels))
    y_true, y_pred = loo_1nn_predictions(embeddings, labels)

    cm, metrics, label_order = eval_metrics(y_true, y_pred, label_order=present)

    with np.errstate(invalid="ignore", divide="ignore"):
        acc_matrix = cm.astype(float) / np.where(
            cm.sum(axis=1, keepdims=True) == 0,
            1,
            cm.sum(axis=1, keepdims=True),
        )

    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(
        acc_matrix,
        xticklabels=label_order,
        yticklabels=label_order,
        vmin=0,
        vmax=1.0,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "per-class recall"},
    )
    plt.title(
        f"{title}\n"
        f"micro={metrics['micro_accuracy']:.2%} | balanced={metrics['balanced_accuracy']:.2%} | macro-F1={metrics['macro_f1']:.2%}"
    )
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    logging.info(
        f"[KNN] {title} | "
        f"micro={metrics['micro_accuracy']:.3f}, bal={metrics['balanced_accuracy']:.3f}, macroF1={metrics['macro_f1']:.3f}"
    )

    return metrics, label_order, acc_matrix


def plot_binary_knn(embeddings, labels, output_path, title):
    if len(embeddings) == 0:
        logging.warning(f"[KNN] no embeddings for {title}")
        return None, None, None

    y_true, y_pred = loo_1nn_predictions(embeddings, labels)
    present = sorted(set(labels))
    cm, metrics, label_order = eval_metrics(y_true, y_pred, label_order=present)

    with np.errstate(invalid="ignore", divide="ignore"):
        acc_matrix = cm.astype(float) / np.where(
            cm.sum(axis=1, keepdims=True) == 0,
            1,
            cm.sum(axis=1, keepdims=True),
        )

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        acc_matrix,
        xticklabels=label_order,
        yticklabels=label_order,
        vmin=0,
        vmax=1,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "per-class recall"},
    )
    plt.title(title)
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


# ---------------- sampling + context loading (single TSV) ----------------

def sample_examples_even_by_chrom(df: pd.DataFrame, n_total: int, seed: int) -> pd.DataFrame:
    """
    sample ~evenly across chromosomes (by rows/examples), without replacement.
    if a chromosome has fewer than its target quota, we take all and redistribute leftover.
    """
    rng = np.random.default_rng(seed)
    chroms = sorted(df["chrom"].unique().tolist())
    if n_total >= len(df):
        logging.info(f"[sample] n_total >= len(df) ({n_total} >= {len(df)}), using all rows")
        return df.copy()

    base = n_total // len(chroms)
    rem = n_total % len(chroms)

    # initial quotas
    quota = {c: base for c in chroms}
    for c in chroms[:rem]:
        quota[c] += 1

    taken = []
    leftover = 0

    for c in chroms:
        sub = df[df["chrom"] == c]
        k = quota[c]
        if len(sub) <= k:
            taken.append(sub)
            leftover += (k - len(sub))
        else:
            idx = rng.choice(len(sub), size=k, replace=False)
            taken.append(sub.iloc[idx])

    out = pd.concat(taken, ignore_index=True)

    # redistribute leftover to chroms that still have unused rows
    if leftover > 0:
        remaining = df.merge(out[["chrom", "transcript_id", "label1_start_pos"]],
                             on=["chrom", "transcript_id", "label1_start_pos"],
                             how="left", indicator=True)
        remaining = remaining[remaining["_merge"] == "left_only"].drop(columns=["_merge"])
        if len(remaining) == 0:
            logging.info("[sample] no remaining rows to fill leftover; returning current sample")
            return out

        k2 = min(leftover, len(remaining))
        idx2 = rng.choice(len(remaining), size=k2, replace=False)
        out = pd.concat([out, remaining.iloc[idx2]], ignore_index=True)

    # final trim in case of weirdness
    if len(out) > n_total:
        out = out.sample(n=n_total, random_state=seed)

    return out


def load_atg_5way_contexts_from_single_tsv_sampled(
    atg_tsv_path: str,
    bigwig_file: str,
    genome: Fasta,
    model_type: str,
    n_examples_total: int,
    seed: int = 0,
    sampled_examples_tsv: str | None = None,
    output_dir_for_sampling: Path | None = None,
):
    _ = pyBigWig.open(bigwig_file).close()

    if sampled_examples_tsv is not None:
        df_s = pd.read_csv(sampled_examples_tsv, sep="\t")
        logging.info(f"[sample] loaded sampled examples from: {sampled_examples_tsv} (n={len(df_s)})")
    else:
        df = pd.read_csv(atg_tsv_path, sep="\t")
        required = ["chrom", "transcript_id", "gene_id", "strand"] + list(LABEL_COLS_5WAY.values())
        for c in required:
            if c not in df.columns:
                raise ValueError(f"missing required column in TSV: {c}")

        df_s = sample_examples_even_by_chrom(df, n_total=n_examples_total, seed=seed)
        logging.info(f"[sample] selected {len(df_s)} examples -> expecting {len(df_s) * 5} contexts")

        if output_dir_for_sampling is not None:
            output_dir_for_sampling = Path(output_dir_for_sampling)
            output_dir_for_sampling.mkdir(parents=True, exist_ok=True)
            save_sampled_examples_tsv(output_dir_for_sampling, df_s, seed=seed, n_examples=n_examples_total)

    contexts = []
    kept_examples = 0
    dropped_examples = 0

    for _, row in df_s.iterrows():
        try:
            pos_dict = {lid: int(row[col]) for lid, col in LABEL_COLS_5WAY.items()}
        except Exception:
            dropped_examples += 1
            continue

        anchor = pos_dict[1]
        example_id = f"{row['chrom']}|{row['transcript_id']}|{row['strand']}|{anchor}"

        example_contexts = []
        ok = True

        for lid, pos in pos_dict.items():
            region = {
                "chrom": row["chrom"],
                "start": pos,
                "end": pos + 3,
                "feature_id": f"{row['transcript_id']}_L{lid}",
                "strand": row["strand"],
            }
            ctx_model_name = model_name_for_context(model_type)
            ctx = extract_context(bigwig_file, region, genome, model_type=ctx_model_name)
            if not ctx or "sequence" not in ctx:
                ok = False
                break

            ctx["example_id"] = example_id
            ctx["label_id"] = lid
            ctx["delta_bp"] = 0 if DELTA_COLS_5WAY[lid] is None else int(row[DELTA_COLS_5WAY[lid]])

            ctx["transcript_id"] = row["transcript_id"]
            ctx["gene_id"] = row["gene_id"]
            ctx["strand"] = row["strand"]

            example_contexts.append(ctx)

        if not ok:
            dropped_examples += 1
            continue

        contexts.extend(example_contexts)
        kept_examples += 1

    logging.info(f"[contexts] kept_examples={kept_examples}, dropped_examples={dropped_examples}")
    logging.info(f"[contexts] total contexts={len(contexts)} (should be kept_examples*5)")
    return contexts


# ---------------- embedding helper (supports evo2) ----------------


def model_name_for_context(model_type: str):
    if "random-init" in model_type:
        model_type = model_type.replace("-random-init", "")
    if "caduceus" in model_type:
        return "caduceus-theirs"
    return model_type


def embed_atg_regions_batched(model, tokenizer, regions, batch_size, device, model_type):
    """
    regions: list of dicts from extract_context, each with:
      - sequence
      - feature_start_in_window, feature_end_in_window
      - example_id
      - label_id (1..5)
      - delta_bp
      - transcript_id, gene_id, strand
    returns:
      roi_embeds [N,H], full_embeds [N,H], labels [N] (label_id), metas [N] dict
    """
    roi_embeds, full_embeds, labels, metas = [], [], [], []

    for batch_start in range(0, len(regions), batch_size):
        batch_regions = regions[batch_start : batch_start + batch_size]

        batch_sequences = []
        batch_info = []
        for r in batch_regions:
            seq = r.get("sequence")
            if not seq:
                continue

            fs = r.get("feature_start_in_window", 0)
            fe = r.get("feature_end_in_window", len(seq))

            info = {
                "chrom": r.get("chrom"),
                "start": int(r.get("start", -1)),
                "end": int(r.get("end", -1)),
                "feature_id": r.get("feature_id", "unknown"),
                "window_len": len(seq),
                "feature_start_in_window": int(fs),
                "feature_end_in_window": int(fe),
                "example_id": r.get("example_id"),
                "label_id": int(r.get("label_id")),
                "delta_bp": int(r.get("delta_bp", 0)),
                "transcript_id": r.get("transcript_id", ""),
                "gene_id": r.get("gene_id", ""),
                "strand": r.get("strand", ""),
            }

            batch_sequences.append(seq)
            batch_info.append(info)

        if not batch_sequences:
            continue

        # evo2 path (per sequence; keep existing behavior)
        if model_type.startswith("evo2"):
            for seq, info in zip(batch_sequences, batch_info):
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

                T, _H = rep.shape
                seq_len = int(info["window_len"])
                fs = info["feature_start_in_window"]
                fe = info["feature_end_in_window"]

                full_vec = rep.mean(dim=0).numpy()

                # map character span to token span (approx)
                scale = T / max(1, seq_len)
                tfs = max(0, min(int(np.floor(fs * scale)), T - 1))
                tfe = max(tfs + 1, min(int(np.ceil(fe * scale)), T))
                roi_vec = rep[tfs:tfe].mean(dim=0).numpy()

                roi_embeds.append(roi_vec.astype(np.float32))
                full_embeds.append(full_vec.astype(np.float32))
                labels.append(int(info["label_id"]))
                metas.append({k: info[k] for k in info})

                del rep, token_ids
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            continue

        # non-evo2 models: compute last hidden [B,T,H]
        if model_type.startswith("phyloGPN"):
            keep = [i for i, s in enumerate(batch_sequences) if len(s) == 481]
            if len(keep) != len(batch_sequences):
                logging.warning(
                    f"[phyloGPN] dropping {len(batch_sequences) - len(keep)} invalid-length sequences"
                )
            batch_sequences = [batch_sequences[i] for i in keep]
            batch_info = [batch_info[i] for i in keep]
            if not batch_sequences:
                continue

            inputs = tokenizer(batch_sequences, return_tensors="pt", padding=False, truncation=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                last_hidden = model.get_embeddings(inputs["input_ids"]).to(torch.float32)

        elif model_type.startswith("nt-"):
            inputs = tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1000,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
                last_hidden = out["hidden_states"][-1].to(torch.float32)

        elif model_type.startswith("hyenaDNA"):
            inputs = tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
                last_hidden = out.hidden_states[-1].to(torch.float32)

        else:
            inputs = tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                if hasattr(model, "get_embeddings"):
                    last_hidden = model.get_embeddings(inputs["input_ids"]).to(torch.float32)
                else:
                    out = model(**inputs, output_hidden_states=True)
                    last_hidden = out.hidden_states[-1].to(torch.float32)

        assert last_hidden.shape[0] == len(batch_info)

        for rep, info in zip(last_hidden, batch_info):
            rep = rep.cpu()  # [T,H]
            T, _H = rep.shape
            seq_len = int(info["window_len"])
            fs = info["feature_start_in_window"]
            fe = info["feature_end_in_window"]

            full_vec = rep.mean(dim=0).numpy()

            scale = T / max(1, seq_len)
            tfs = max(0, min(int(np.floor(fs * scale)), T - 1))
            tfe = max(tfs + 1, min(int(np.ceil(fe * scale)), T))
            roi_vec = rep[tfs:tfe].mean(dim=0).numpy()

            roi_embeds.append(roi_vec.astype(np.float32))
            full_embeds.append(full_vec.astype(np.float32))
            labels.append(int(info["label_id"]))
            metas.append({k: info[k] for k in info})

        del last_hidden, inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return (
        np.stack(roi_embeds),
        np.stack(full_embeds),
        np.asarray(labels, dtype=int),
        metas,
    )


# ---------------- main analysis ----------------

def analyze_atg_5way_other_models(
    atg_tsv_path,
    genome_fasta,
    bigwig_file,
    output_dir,
    model_type,
    batch_size,
    n_examples_total,
    seed,
    sampled_examples_tsv=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # cache check (roi is the main thing you plot)
    cached = maybe_load_cached_reps(output_dir, model_type, "ATG5way_all_labels_roi")
    if cached is not None:
        roi_embeds, label_ids, metas = cached
        logging.info(f"[cache] loaded roi_embeds shape={roi_embeds.shape}")

        plot_knn_heatmap(
            roi_embeds,
            label_ids,
            output_path=output_dir / f"knn_heatmap_{model_type}_ATG5way_all_labels.png",
            title=f"ATG 5-way 1-NN ({model_type})",
        )

        for target_label in range(2, 6):
            idx = np.where((label_ids == 1) | (label_ids == target_label))[0]
            if len(idx) == 0:
                continue
            plot_knn_heatmap(
                roi_embeds[idx],
                label_ids[idx],
                output_path=output_dir / f"knn_heatmap_{model_type}_ATG1_vs_{target_label}.png",
                title=f"ATG 1-vs-{target_label} 1-NN ({model_type})",
            )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"using device: {device}")

    model, tokenizer = load_model(model_type=model_type, device=device)
    genome = Fasta(genome_fasta)

    contexts = load_atg_5way_contexts_from_single_tsv_sampled(
        atg_tsv_path=atg_tsv_path,
        bigwig_file=bigwig_file,
        genome=genome,
        model_type=model_type,
        n_examples_total=n_examples_total,
        seed=seed,
        sampled_examples_tsv=sampled_examples_tsv,
        output_dir_for_sampling=output_dir,  # writes sampled_examples_atg5.tsv if sampling
    )
    if not contexts:
        logging.error("no contexts loaded, aborting")
        return

    roi_embeds, full_embeds, label_ids, metas = embed_atg_regions_batched(
        model,
        tokenizer,
        contexts,
        batch_size=batch_size,
        device=device,
        model_type=model_type,
    )

    # enforce strict 5-per-example
    index_by_example = defaultdict(dict)
    for i, m in enumerate(metas):
        index_by_example[m["example_id"]][int(m["label_id"])] = i

    valid_examples = [ex for ex, d in index_by_example.items() if all(k in d for k in range(1, 6))]
    logging.info(f"valid examples with all 5 labels after embedding: {len(valid_examples)}")
    if len(valid_examples) == 0:
        logging.error("no valid examples with all 5 labels after embedding; aborting")
        return

    keep = []
    for ex in valid_examples:
        for lid in range(1, 6):
            keep.append(index_by_example[ex][lid])
    keep = np.asarray(keep, dtype=int)

    roi_embeds = roi_embeds[keep]
    full_embeds = full_embeds[keep]
    label_ids = label_ids[keep]
    metas = [metas[i] for i in keep.tolist()]

    extra_roi = {"model_type": model_type, "scope": "roi_all", "n_examples": len(valid_examples), "seed": seed}
    extra_full = {"model_type": model_type, "scope": "full_all", "n_examples": len(valid_examples), "seed": seed}
    save_reps(output_dir, model_type, "ATG5way_all_labels_roi", roi_embeds, label_ids, metas, extra=extra_roi)
    save_reps(output_dir, model_type, "ATG5way_all_labels_full", full_embeds, label_ids, metas, extra=extra_full)

    # plots (5-way + 1-vs-k)
    metrics5, _, _ = plot_knn_heatmap(
        roi_embeds,
        label_ids,
        output_path=output_dir / f"knn_heatmap_{model_type}_ATG5way_all_labels.png",
        title=f"ATG 5-way 1-NN ({model_type})",
    )

    task_metrics = {}
    if metrics5:
        task_metrics["task5way_balanced_accuracy"] = float(metrics5["balanced_accuracy"])
        task_metrics["task5way_micro_accuracy"] = float(metrics5["micro_accuracy"])

    for target_label in range(2, 6):
        idx = np.where((label_ids == 1) | (label_ids == target_label))[0]
        if len(idx) == 0:
            continue
        mk, _, _ = plot_knn_heatmap(
            roi_embeds[idx],
            label_ids[idx],
            output_path=output_dir / f"knn_heatmap_{model_type}_ATG1_vs_{target_label}.png",
            title=f"ATG 1-vs-{target_label} 1-NN ({model_type})",
        )
        if mk:
            task_metrics[f"1_vs_{target_label}_balanced_accuracy"] = float(mk["balanced_accuracy"])

    with open(output_dir / f"balanced_accuracy_{model_type}_ATG5way.json", "w") as f:
        json.dump(task_metrics, f, indent=2)

    logging.info(f"task balanced accuracies: {task_metrics}")


def maybe_load_cached_reps(output_dir: Path, model_tag: str, name: str):
    prefix = f"reps_{model_tag}_{name}"
    npz = output_dir / f"{prefix}.npz"
    meta = output_dir / f"{prefix}_meta.parquet"
    if npz.exists() and meta.exists():
        logging.info(f"[cache] found existing reps, skipping embedding: {npz.name}")
        d = np.load(npz, allow_pickle=True)
        X = d["embeddings"].astype(np.float32)
        y = d["labels"].astype(int)
        mdf = pd.read_parquet(meta)
        metas = mdf.to_dict(orient="records")
        return X, y, metas
    return None


def save_sampled_examples_tsv(output_dir: Path, df_sampled: pd.DataFrame, seed: int, n_examples: int):
    out = output_dir / "sampled_examples_atg5.tsv"
    meta = {
        "n_examples_requested": int(n_examples),
        "n_examples_saved": int(len(df_sampled)),
        "seed": int(seed),
    }
    df_sampled.to_csv(out, sep="\t", index=False)
    with open(output_dir / "sampled_examples_atg5.meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logging.info(f"wrote sampled examples TSV: {out}")


# ---------------- cli ----------------

def main():
    p = argparse.ArgumentParser(description="ATG 5-way eval for other models with even-by-chrom sampling")
    p.add_argument(
        "--atg_tsv_path",
        type=str,
        default="/home/mica/gamba/data_processing/data/ATGs_simplified/all_chr_atg_5way.tsv",
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
        "--output_dir",
        type=str,
        default="/home/mica/scratch/gamba/other-models/ATG_reps_5way",
    )
    p.add_argument("--sampled_examples_tsv", type=str, default="/home/mica/scratch/gamba/data_processing/data/240-mammalian/ATG_reps_5way/sampled_examples_atg5.tsv")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_examples_total", type=int, default=5000, help="total examples (rows) to sample; contexts=5x")
    p.add_argument("--seed", type=int, default=42)

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
            "evo2",
            "caduceus-theirs",
            "caduceus-theirs-random-init"
        ],
        required=True,
    )

    args = p.parse_args()
    outdir = os.path.join(args.output_dir, args.model_type)
    os.makedirs(outdir, exist_ok=True)

    analyze_atg_5way_other_models(
        atg_tsv_path=args.atg_tsv_path,
        genome_fasta=args.genome_fasta,
        bigwig_file=args.bigwig_file,
        output_dir=outdir,
        model_type=args.model_type,
        batch_size=args.batch_size,
        n_examples_total=args.n_examples_total,
        sampled_examples_tsv=args.sampled_examples_tsv,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()