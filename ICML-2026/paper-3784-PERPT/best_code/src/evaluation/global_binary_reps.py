#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pyBigWig
from pyfaidx import Fasta
import json
from tqdm import tqdm
import pandas as pd
import logging
import sys
import random
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

sys.path.append("../gamba")
sys.path.append("/home/mica/gamba/")
import pathlib
from torch.nn import MSELoss, CrossEntropyLoss
from sequence_models.constants import MSA_PAD, START, STOP
from evodiff.utils import Tokenizer
from gamba.constants import TaskType, DNA_ALPHABET_PLUS
from gamba.collators import gLMCollator, gLMMLMCollator
from gamba.model import create_model, JambagambaModel, JambaGambaNoConsModel, JambaGambaNOALMModel
from my_caduceus.configuration_caduceus import CaduceusConfig
from my_caduceus.modeling_caduceus import (
    CaduceusConservationForMaskedLM,
    CaduceusForMaskedLM,
    CaduceusConservation
)
from src.evaluation.utils.helpers import load_bed_file, extract_context
from src.evaluation.utils.specific_helpers import load_model, predict_scores_batched

from Bio.Seq import Seq
from collections import Counter

import umap
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
)

CATEGORY_ORDER = [
    "vista_enhancer", "UCNE", "repeats", "exons", "introns",
    "noncoding_regions", "coding_regions", "upstream_TSS",
    "UTR5", "UTR3", "promoters",
]

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------- helpers for region → embeddings ----------------

def extract_region_embeddings(representations, region_info, mode="roi"):
    """
    Extract pooled embeddings for region-of-interest (ROI) or full sequence.

    representations: list of np.arrays [seq_len, hidden_dim]
    region_info: list of dicts with keys including:
        'feature_start_in_window', 'feature_end_in_window',
        'category', 'class_label', 'chrom', 'start', 'end'
    mode: "roi" or "full"

    returns:
        embeddings [N, D], labels (class_label), metas (dicts)
    """
    embeddings = []
    labels = []
    metas = []

    for rep, info in zip(representations, region_info):
        if isinstance(rep, float) or np.isnan(rep).any():
            continue

        if mode == "roi":
            fs = int(info["feature_start_in_window"])
            fe = int(info["feature_end_in_window"])
            rep_slice = rep[fs:fe]
        elif mode == "full":
            fs = 0
            fe = rep.shape[0]
            rep_slice = rep
        else:
            raise ValueError(f"unsupported mode: {mode}")

        if rep_slice.shape[0] == 0:
            continue

        pooled = rep_slice.mean(axis=0)
        cls = info.get("class_label", info.get("category", "unknown"))
        embeddings.append(pooled.astype(np.float32))
        labels.append(cls)
        metas.append({
            "chrom": info.get("chrom"),
            "start": int(info.get("start", -1)),
            "end":   int(info.get("end", -1)),
            "feature_start_in_window": fs,
            "feature_end_in_window": fe,
            "category": info.get("category", "unknown"),
            "class_label": cls,
        })

    logging.info(f"[extract_region_embeddings] mode={mode}, returning {len(embeddings)} embeds")
    if len(embeddings) == 0:
        return np.empty((0, 1), dtype=np.float32), [], []

    return np.stack(embeddings), labels, metas


def _roi_span(info):
    fs = int(info["feature_start_in_window"])
    fe = int(info["feature_end_in_window"])
    if fe <= fs:
        return None
    return fs, fe


def _build_kmer_index(k=6, alphabet="ACGT"):
    from itertools import product
    kmers = ["".join(p) for p in product(alphabet, repeat=k)]
    return {kmer: i for i, kmer in enumerate(kmers)}


def _seq_to_kmer_vec(seq, k, kmer_index):
    n = len(kmer_index)
    vec = np.zeros(n, dtype=np.float32)
    L = len(seq)
    if L < k:
        return vec
    for i in range(L - k + 1):
        kmer = seq[i:i + k].upper()
        if kmer in kmer_index:
            vec[kmer_index[kmer]] += 1.0
    s = vec.sum()
    if s > 0:
        vec /= s
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def compute_kmer_embeddings(valid_regions, mode="roi", k=6):
    from itertools import product
    kmers = ["".join(p) for p in product("ACGT", repeat=k)]
    kmer_index = {kmer: i for i, kmer in enumerate(kmers)}

    embeddings, labels, metas = [], [], []
    for r in valid_regions:
        seq = r["sequence"]
        fs = fe = None
        if mode == "roi":
            span = _roi_span(r)
            if span is None:
                continue
            fs, fe = span
            seq = seq[fs:fe]
        if not seq or len(seq) < k:
            continue
        vec = _seq_to_kmer_vec(seq, k, kmer_index)
        cls = r.get("class_label", r.get("category", "unknown"))
        embeddings.append(vec.astype(np.float32))
        labels.append(cls)
        metas.append({
            "chrom": r.get("chrom"),
            "start": int(r.get("start", -1)),
            "end":   int(r.get("end", -1)),
            "feature_start_in_window": int(fs) if fs is not None else 0,
            "feature_end_in_window":   int(fe) if fe is not None else len(seq),
            "category": r.get("category", "unknown"),
            "class_label": cls,
        })

    if len(embeddings) == 0:
        return np.empty((0, 4 ** k), dtype=np.float32), [], []

    return np.vstack(embeddings), labels, metas


def _summarize_scores(scores):
    s = np.asarray(scores, dtype=np.float32)
    if s.size == 0 or np.isnan(s).all():
        return None
    m = np.nanmean(s)
    st = np.nanstd(s)
    pos = s[s > 0]
    neg = s[s < 0]
    denom = float(np.sum(~np.isnan(s))) if np.sum(~np.isnan(s)) else 0.0
    fpos = float(np.sum(s > 0)) / denom if denom else 0.0
    fneg = float(np.sum(s < 0)) / denom if denom else 0.0
    mpos = float(np.nanmean(pos)) if pos.size else 0.0
    mneg = float(np.nanmean(neg)) if neg.size else 0.0
    return np.array([m, st, fpos, fneg, mpos, mneg], dtype=np.float32)


def compute_phylop_embeddings(valid_regions, mode="roi"):
    embeddings, labels, metas = [], [], []
    for r in valid_regions:
        scores = r.get("scores", None)
        if scores is None:
            continue
        fs = fe = None
        if mode == "roi":
            span = _roi_span(r)
            if span is None:
                continue
            fs, fe = span
            ss = scores[fs:fe]
        else:
            ss = scores
        feat = _summarize_scores(ss)
        if feat is None:
            continue
        cls = r.get("class_label", r.get("category", "unknown"))
        embeddings.append(feat.astype(np.float32))
        labels.append(cls)
        metas.append({
            "chrom": r.get("chrom"),
            "start": int(r.get("start", -1)),
            "end":   int(r.get("end", -1)),
            "feature_start_in_window": int(fs) if fs is not None else 0,
            "feature_end_in_window":   int(fe) if fe is not None else len(ss),
            "category": r.get("category", "unknown"),
            "class_label": cls,
        })

    if len(embeddings) == 0:
        return np.empty((0, 6), dtype=np.float32), [], []

    return np.vstack(embeddings), labels, metas


# ---------------- plotting + metrics ----------------

def plot_umap(embeddings, labels, output_path, title="UMAP"):
    if len(embeddings) != len(labels):
        logging.error(f"[plot_umap] {len(embeddings)} embeds vs {len(labels)} labels")
        return
    if len(embeddings) == 0:
        logging.warning("[plot_umap] no embeddings to plot")
        return

    umap_model = umap.UMAP()
    embedding_2d = umap_model.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=embedding_2d[:, 0],
        y=embedding_2d[:, 1],
        hue=labels,
        palette="tab10",
        s=40,
        alpha=0.8,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def loo_1nn_predictions(embeddings, labels):
    labels = np.asarray(labels)
    X = np.asarray(embeddings)
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(X)
    _, indices = nn.kneighbors(X)
    y_true = labels
    y_pred = labels[indices[:, 1]]
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
        "macro_f1": float(f1_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=label_order, average="weighted", zero_division=0)),
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

    present = sorted(set(labels))
    y_true, y_pred = loo_1nn_predictions(embeddings, labels)
    cm, metrics, label_order = eval_metrics(y_true, y_pred, label_order=present)

    with np.errstate(invalid="ignore", divide="ignore"):
        acc_matrix = cm.astype(float) / np.where(
            cm.sum(axis=1, keepdims=True) == 0,
            1,
            cm.sum(axis=1, keepdims=True),
        )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        acc_matrix,
        xticklabels=label_order,
        yticklabels=label_order,
        vmin=0,
        vmax=1.0,
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
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    logging.info(
        f"[KNN] {title} | micro={metrics['micro_accuracy']:.3f}, "
        f"balanced={metrics['balanced_accuracy']:.3f}, "
        f"macroF1={metrics['macro_f1']:.3f}, "
        f"weightedF1={metrics['weighted_f1']:.3f}, "
        f"kappa={metrics['cohens_kappa']:.3f}, mcc={metrics['mcc']:.3f}"
    )

    return metrics, label_order, acc_matrix


def _save_summary(csv_path, row_dict):
    csv_path = pathlib.Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row_dict])
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False)


def save_reps(output_dir, model_id, group_name, category, scope, X, labels, metas, extra=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels)

    tag = f"{group_name}_{category}_{scope}"

    # npz with embeddings + labels
    np.savez_compressed(
        out / f"reps_{model_id}_{tag}.npz",
        embeddings=X,
        labels=labels,
    )

    # metadata dataframe
    mdf = pd.DataFrame(metas)

    # ensure these columns exist with the values we want, but
    # don't crash if they’re already present.
    if "label" in mdf.columns:
        mdf["label"] = labels
    else:
        mdf.insert(0, "label", labels)

    if "scope" in mdf.columns:
        mdf["scope"] = scope
    else:
        mdf.insert(0, "scope", scope)

    if "category" in mdf.columns:
        mdf["category"] = category
    else:
        mdf.insert(0, "category", category)

    if "group" in mdf.columns:
        mdf["group"] = group_name
    else:
        mdf.insert(0, "group", group_name)

    # add model-level metadata
    if extra:
        for k, v in extra.items():
            mdf[k] = v

    mdf.to_parquet(out / f"reps_{model_id}_{tag}_meta.parquet", index=False)

def _save_per_class_json(json_path, label_order, acc_matrix):
    data = {
        "label_order": list(map(str, label_order)),
        "per_class_recall": {str(lbl): float(acc_matrix[i, i]) for i, lbl in enumerate(label_order)},
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------- upstream-pair logic ----------------
REGION_ROOT = "/home/mica/gamba/data_processing/data/regions"


def _pair_key(r: dict) -> str:
    """
    build a key for pairing roi and upstream regions.

    we use the last BED column (pair id) plus chrom, so that:
      chr22 ... 54049   (roi)
      chr22 ... 54049   (upstream)
    get matched.

    adjust the key name ('pair_id' vs 'id') to whatever load_bed_file uses.
    """
    chrom = r.get("chrom", "")
    # try common possibilities; change to the real one if needed
    pid = r.get("pair_id", r.get("id"))
    if pid is not None:
        return f"{chrom}:{pid}"
    # hard fallback if the id key name is different
    return f"{chrom}:{r.get('start')}-{r.get('end')}"



def load_paired_regions(category, group_chroms, genome, bw, num_regions=None):
    """
    load roi and upstream regions for a category and a set of chromosomes, and
    pair them by the pair id in the last BED column.

    expects:
      roi bed:  REGION_ROOT/{category}/{chrom}.bed
      up bed:   REGION_ROOT/{category}_upstream/{chrom}.bed

    returns:
      list of (roi_region_dict, upstream_region_dict)
    """
    pairs = []

    for chrom in group_chroms:
        roi_bed = os.path.join(REGION_ROOT, category, f"{chrom}.bed")
        up_bed = os.path.join(REGION_ROOT, f"{category}_upstream", f"{chrom}.bed")

        if not os.path.exists(roi_bed):
            logging.warning(f"[load_paired_regions] missing roi bed: {roi_bed}")
            continue
        if not os.path.exists(up_bed):
            logging.warning(f"[load_paired_regions] missing upstream bed: {up_bed}")
            continue

        roi_regions = load_bed_file(roi_bed, category, genome, bw)
        up_regions = load_bed_file(up_bed, category, genome, bw)

        # build key → upstream region map
        up_by_key = {}
        for r in up_regions:
            if r.get("chrom") != chrom:
                continue
            key = _pair_key(r)
            up_by_key[key] = r

        # pair roi with upstream by pair id
        for r in roi_regions:
            if r.get("chrom") != chrom:
                continue
            key = _pair_key(r)
            up = up_by_key.get(key)
            if up is None:
                continue
            pairs.append((r, up))
            if num_regions is not None and len(pairs) >= num_regions:
                return pairs

    return pairs


def analyze_upstream_pairs(
    genome_fasta,
    bigwig_file,
    checkpoint_dir,
    config_fpath,
    output_dir,
    baseline="none",
    num_regions=100,
    chromosomes=None,
    last_step=44000,
    batch_size=8,
    training_chromosomes=None,
    test_chromosomes=None,
    training_task="dual",
    model_type="gamba",
):
    """
    for each category and chromosome group, compare:
      - roi region
      - region shifted 2kb upstream (same length)

    supports baselines (kmer6, phylop) and trained models (gamba/caduceus).
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"using device: {device}")

    model = tokenizer = None
    if baseline == "none":
        model, tokenizer = load_model(
            checkpoint_dir,
            config_fpath,
            last_step=last_step,
            device=device,
            training_task=training_task,
            model_type=model_type,
        )

    genome = Fasta(genome_fasta)
    bw = pyBigWig.open(bigwig_file)

    categories = [
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
    ]

    if training_chromosomes and test_chromosomes:
        chromosome_groups = {
            "training": training_chromosomes,
            "test": test_chromosomes,
        }
    else:
        chromosome_groups = {"all": chromosomes}

    summary_csv = os.path.join(output_dir, "representation_knn_summary.csv")

    for group_name, group_chroms in chromosome_groups.items():
        logging.info(f"analyzing group={group_name}, chroms={group_chroms}")

        for category in categories:
            logging.info(f"  category={category}")

            # load paired roi / upstream regions from bed files on scratch
            paired_regions = load_paired_regions(
                category=category,
                group_chroms=group_chroms,
                genome=genome,
                bw=bw,
                num_regions=num_regions,
            )

            if not paired_regions:
                logging.warning(f"    no paired regions for {category} in group {group_name}")
                continue

            logging.info(f"    using {len(paired_regions)} roi/upstream pairs")

            # build paired contexts: roi + upstream (from precomputed beds)
            valid_regions = []
            n_pairs = 0

            for r_roi, r_up in paired_regions:
                # choose context mode
                ctx_model_type = "baseline" if baseline in ("kmer6", "phylop") else model_type

                pos_ctx = extract_context(bigwig_file, r_roi, genome, ctx_model_type)
                neg_ctx = extract_context(bigwig_file, r_up, genome, ctx_model_type)

                if not pos_ctx or "sequence" not in pos_ctx:
                    continue
                if not neg_ctx or "sequence" not in neg_ctx:
                    continue

                # attach labels and category
                for ctx, cls, reg in (
                    (pos_ctx, "roi", r_roi),
                    (neg_ctx, "upstream", r_up),
                ):
                    ctx["chrom"] = reg["chrom"]
                    ctx["category"] = category
                    ctx["class_label"] = cls
                    ctx["start"] = int(reg["start"])
                    ctx["end"] = int(reg["end"])

                valid_regions.append(pos_ctx)
                valid_regions.append(neg_ctx)
                n_pairs += 1

            logging.info(f"    built {n_pairs} roi/upstream pairs, {len(valid_regions)} total contexts")

            if len(valid_regions) < 4:
                logging.warning(f"    not enough valid regions for {category} in group {group_name}")
                continue

            # embeddings for this category + group
            if baseline == "kmer6":
                full_embeds, full_labels, full_metas = compute_kmer_embeddings(valid_regions, mode="full", k=6)
                roi_embeds, roi_labels, roi_metas = compute_kmer_embeddings(valid_regions, mode="roi", k=6)
            elif baseline == "phylop":
                full_embeds, full_labels, full_metas = compute_phylop_embeddings(valid_regions, mode="full")
                roi_embeds, roi_labels, roi_metas = compute_phylop_embeddings(valid_regions, mode="roi")
            else:
                # trained model
                sequence_representations, region_info = predict_scores_batched(
                    model,
                    tokenizer,
                    valid_regions,
                    batch_size=batch_size,
                    device=device,
                    model_type=model_type,
                    training_task=training_task,
                )

                # propagate category + class_label from input regions
                for ctx, info in zip(valid_regions, region_info):
                    info["chrom"] = ctx.get("chrom")
                    info["start"] = ctx.get("start")
                    info["end"] = ctx.get("end")
                    info["category"] = category
                    info["class_label"] = ctx.get("class_label", "unknown")

                full_embeds, full_labels, full_metas = extract_region_embeddings(
                    sequence_representations, region_info, mode="full"
                )
                roi_embeds, roi_labels, roi_metas = extract_region_embeddings(
                    sequence_representations, region_info, mode="roi"
                )

            # sanity checks
            assert len(full_embeds) == len(full_labels), "full labels/embeds mismatch"
            assert len(roi_embeds) == len(roi_labels), "roi labels/embeds mismatch"

            # plotting + metrics for this group/category
            cat_outdir = output_dir / group_name / category
            cat_outdir.mkdir(parents=True, exist_ok=True)

            model_id = model_type if baseline == "none" else baseline
            tag = f"{model_id}_{group_name}_{category}"

            # umap
            plot_umap(
                roi_embeds,
                roi_labels,
                cat_outdir / f"umap_roi_{tag}.png",
                title=f"{category} roi vs upstream ({group_name})",
            )
            roi_metrics, roi_lbl_order, roi_mat = plot_knn_heatmap(
                roi_embeds,
                roi_labels,
                cat_outdir / f"knn_roi_{tag}.png",
                title=f"{category} roi vs upstream ({group_name})",
            )

            plot_umap(
                full_embeds,
                full_labels,
                cat_outdir / f"umap_full_{tag}.png",
                title=f"{category} full-window ({group_name})",
            )
            full_metrics, full_lbl_order, full_mat = plot_knn_heatmap(
                full_embeds,
                full_labels,
                cat_outdir / f"knn_full_{tag}.png",
                title=f"{category} full-window ({group_name})",
            )

            # save per-class recall jsons
            if roi_metrics is not None:
                _save_per_class_json(
                    cat_outdir / f"per_class_roi_{tag}.json",
                    roi_lbl_order,
                    roi_mat,
                )
            if full_metrics is not None:
                _save_per_class_json(
                    cat_outdir / f"per_class_full_{tag}.json",
                    full_lbl_order,
                    full_mat,
                )

            # save embeddings + meta
            extra = {
                "model_id": model_id,
                "training_task": training_task,
                "baseline": baseline,
                "last_step": last_step,
            }
            save_reps(
                cat_outdir,
                model_id,
                group_name,
                category,
                "roi",
                roi_embeds,
                roi_labels,
                roi_metas,
                extra,
            )
            save_reps(
                cat_outdir,
                model_id,
                group_name,
                category,
                "full",
                full_embeds,
                full_labels,
                full_metas,
                extra,
            )

            # summary csv rows
            if roi_metrics is not None:
                _save_summary(
                    summary_csv,
                    {
                        "Group": group_name,
                        "Category": category,
                        "Scope": "roi",
                        "Model": model_id,
                        "Baseline": baseline,
                        "BalancedAccuracyPct": 100.0 * roi_metrics["balanced_accuracy"],
                        "BalancedAccuracySEM_Pct": 100.0 * roi_metrics["balanced_accuracy_sem"],
                        "MicroAccuracyPct": 100.0 * roi_metrics["micro_accuracy"],
                        "MacroF1Pct": 100.0 * roi_metrics["macro_f1"],
                    },
                )
            if full_metrics is not None:
                _save_summary(
                    summary_csv,
                    {
                        "Group": group_name,
                        "Category": category,
                        "Scope": "full",
                        "Model": model_id,
                        "Baseline": baseline,
                        "BalancedAccuracyPct": 100.0 * full_metrics["balanced_accuracy"],
                        "BalancedAccuracySEM_Pct": 100.0 * full_metrics["balanced_accuracy_sem"],
                        "MicroAccuracyPct": 100.0 * full_metrics["micro_accuracy"],
                        "MacroF1Pct": 100.0 * full_metrics["macro_f1"],
                    },
                )

    bw.close()


# ---------------- cli ----------------

def main():
    parser = argparse.ArgumentParser(
        description="analyze roi vs 2kb-upstream separability for different categories"
    )
    parser.add_argument(
        "--bigwig_file",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig",
        help="path to phyloP bigwig file",
    )
    parser.add_argument(
        "--genome_fasta",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa",
        help="path to genome fasta",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/final_representations/upstream_pairs",
        help="directory to save analysis results",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/home/mica/gamba/",
        help="directory containing model checkpoints",
    )
    parser.add_argument(
        "--config_fpath",
        type=str,
        default="/home/mica/gamba/configs/jamba-small-240mammalian.json",
        help="path to model config json",
    )
    parser.add_argument(
        "--num_regions",
        type=int,
        default=1000,
        help="max number of base regions per category",
    )
    parser.add_argument(
        "--chromosomes",
        type=str,
        nargs="+",
        default=[
            "chr1", "chr2", "chr3", "chr4", "chr5", "chr6",
            "chr7", "chr8", "chr9", "chr10", "chr11", "chr12",
            "chr13", "chr14", "chr15", "chr16", "chr17", "chr18",
            "chr19", "chr20", "chr21", "chrX",
        ],
        help="chromosomes to analyze",
    )
    parser.add_argument(
        "--training_chromosomes",
        type=str,
        nargs="+",
        default=[
            "chr1", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9",
            "chr10", "chr11", "chr12", "chr13", "chr14", "chr15",
            "chr17", "chr18", "chr19", "chr20", "chr21", "chrX",
        ],
        help="training chromosomes",
    )
    parser.add_argument(
        "--test_chromosomes",
        type=str,
        nargs="+",
        default=["chr2", "chr22", "chr16", "chr3"],
        help="held-out test chromosomes",
    )
    parser.add_argument(
        "--last_step",
        type=int,
        default=44000,
        help="checkpoint step to use (0 = random init)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size for model predictions",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["gamba", "caduceus"],
        default=None,
        help="required when baseline == 'none'",
    )
    parser.add_argument(
        "--training_task",
        type=str,
        choices=["dual", "cons_only", "seq_only"],
        default=None,
        help="required when baseline == 'none'",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=["none", "kmer6", "phylop"],
        default="none",
        help="baseline embedding instead of trained model",
    )

    args = parser.parse_args()

    logging.info(f"starting upstream-pair analysis")
    logging.info(f"chromosomes: {args.chromosomes}")
    logging.info(f"training chromosomes: {args.training_chromosomes}")
    logging.info(f"test chromosomes: {args.test_chromosomes}")

    if args.baseline == "none":
        if args.model_type is None or args.training_task is None:
            raise SystemExit("when --baseline=none, provide --model_type and --training_task")
        logging.info(f"using MODEL: type={args.model_type}, task={args.training_task}")
    else:
        logging.info(f"using BASELINE: {args.baseline}")
        # for baselines, treat all chroms as training, no test split
        args.training_chromosomes = args.chromosomes
        args.test_chromosomes = None
        args.model_type = args.baseline
        args.training_task = "baseline"

    # prepare output dir + checkpoint dir naming
    if args.baseline != "none":
        output_dir = os.path.join(args.output_dir, "baseline", args.baseline)
        checkpoint_dir = None
    else:
        if args.model_type == "gamba":
            checkpoint_dir = os.path.join(args.checkpoint_dir, "clean_dcps/CCP/")
        else:
            checkpoint_dir = args.checkpoint_dir

        if args.last_step == 0:
            last_tag = "random_init"
        else:
            last_tag = args.last_step
        output_dir = os.path.join(
            args.output_dir,
            f"{args.model_type}_{args.training_task}_ALLPOSstep_{last_tag}",
        )

    try:
        analyze_upstream_pairs(
            genome_fasta=args.genome_fasta,
            bigwig_file=args.bigwig_file,
            checkpoint_dir=checkpoint_dir,
            config_fpath=args.config_fpath,
            output_dir=output_dir,
            baseline=args.baseline,
            num_regions=args.num_regions,
            chromosomes=args.chromosomes,
            last_step=args.last_step,
            batch_size=args.batch_size,
            training_chromosomes=args.training_chromosomes,
            test_chromosomes=args.test_chromosomes,
            training_task=args.training_task,
            model_type=args.model_type if args.baseline == "none" else "baseline",
        )
        logging.info("analysis completed successfully")
    except Exception as e:
        logging.error(f"error in analysis: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
