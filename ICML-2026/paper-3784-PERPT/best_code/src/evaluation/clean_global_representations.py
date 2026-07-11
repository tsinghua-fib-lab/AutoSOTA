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
#import Counter
from collections import Counter

import umap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors

CATEGORY_ORDER = [
    "vista_enhancer", "UCNE", "repeats", "exons", "introns",
    "noncoding_regions", "coding_regions", "upstream_TSS",
    "UTR5", "UTR3", "promoters", #"phyloP_negative", "phyloP_neutral", "phyloP_positive",
]


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def extract_region_embeddings(representations, region_info, mode="roi"):
    """
    Extract embeddings for the region of interest (ROI) or full sequence.

    Args:
        representations: list of np.arrays of shape (seq_len, hidden_dim)
        region_info: list of dicts with keys 'feature_start_in_window', 'feature_end_in_window', 'category'
        mode: "roi" for region-of-interest, "full" for full-sequence average

    Returns:
        List of pooled embeddings and corresponding category labels
    """
    embeddings = []
    labels = []
    metas = []

    for rep, info in zip(representations, region_info):
        if isinstance(rep, float) or np.isnan(rep).any():
            continue

        if mode == "roi":
            fs = info["feature_start_in_window"]
            fe = info["feature_end_in_window"]
            rep_slice = rep[fs:fe]
        elif mode == "full":
            fs = 0
            fe = rep.shape[0]
            rep_slice = rep
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if rep_slice.shape[0] == 0:
            continue

        pooled = rep_slice.mean(axis=0)
        embeddings.append(pooled.astype(np.float32))
        labels.append(info.get("category", "unknown"))
        metas.append({
            "chrom": info.get("chrom"),
            "start": int(info.get("start", -1)),
            "end":   int(info.get("end", -1)),
            "feature_start_in_window": fs,
            "feature_end_in_window": fe,
            "category": info.get("category", "unknown"),
        })
    print(f"[extract_region_embeddings] mode={mode}, returning {len(embeddings)} embeds and {len(labels)} labels")

    return np.stack(embeddings), labels, metas

def _roi_span(info):
    fs = int(info["feature_start_in_window"])
    fe = int(info["feature_end_in_window"])
    if fe <= fs:
        return None
    return fs, fe

def _build_kmer_index(k=6, alphabet="ACGT"):
    # lexicographic index: 'AAAAAA'..'TTTTTT'
    from itertools import product
    kmers = [''.join(p) for p in product(alphabet, repeat=k)]
    return {kmer: i for i, kmer in enumerate(kmers)}

def _seq_to_kmer_vec(seq, k, kmer_index):
    # ignore k-mers with non-ACGT chars
    n = len(kmer_index)
    vec = np.zeros(n, dtype=np.float32)
    L = len(seq)
    if L < k:
        return vec
    for i in range(L - k + 1):
        kmer = seq[i:i+k].upper()
        if kmer in kmer_index:  # skip if contains N/others
            vec[kmer_index[kmer]] += 1.0
    s = vec.sum()
    if s > 0:
        vec /= s  # normalize to frequencies
    # L2 normalize for cosine-ish geometry in euclidean space
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec

def compute_kmer_embeddings(valid_regions, mode="roi", k=6):
    from itertools import product
    kmers = [''.join(p) for p in product("ACGT", repeat=k)]
    kmer_index = {kmer: i for i, kmer in enumerate(kmers)}
    embeddings, labels, metas = [], [], []
    for r in valid_regions:
        seq = r["sequence"]; fs = fe = None
        if mode == "roi":
            span = _roi_span(r)
            if span is None: continue
            fs, fe = span
            seq = seq[fs:fe]
        if not seq or len(seq) < k: continue
        vec = _seq_to_kmer_vec(seq, k, kmer_index)
        embeddings.append(vec.astype(np.float32))
        labels.append(r.get("category", "unknown"))
        metas.append({
            "chrom": r.get("chrom"),
            "start": int(r.get("start", -1)),
            "end":   int(r.get("end", -1)),
            "feature_start_in_window": int(fs) if fs is not None else 0,
            "feature_end_in_window":   int(fe) if fe is not None else len(seq),
            "category": r.get("category", "unknown"),
        })
    if len(embeddings) == 0:
        return np.empty((0, 4**k), dtype=np.float32), [], []
    return np.vstack(embeddings), labels, metas


def compute_phylop_embeddings(valid_regions, mode="roi"):
    embeddings, labels, metas = [], [], []
    for r in valid_regions:
        scores = r.get("scores", None)
        if scores is None: continue
        fs = fe = None
        if mode == "roi":
            span = _roi_span(r)
            if span is None: continue
            fs, fe = span
            ss = scores[fs:fe]
        else:
            ss = scores
        feat = _summarize_scores(ss)
        if feat is None: continue
        embeddings.append(feat.astype(np.float32))
        labels.append(r.get("category", "unknown"))
        metas.append({
            "chrom": r.get("chrom"),
            "start": int(r.get("start", -1)),
            "end":   int(r.get("end", -1)),
            "feature_start_in_window": int(fs) if fs is not None else 0,
            "feature_end_in_window":   int(fe) if fe is not None else len(ss),
            "category": r.get("category", "unknown"),
        })
    if len(embeddings) == 0:
        return np.empty((0, 6), dtype=np.float32), [], []
    return np.vstack(embeddings), labels, metas

def _summarize_scores(scores):
    s = np.asarray(scores, dtype=np.float32)
    if s.size == 0 or np.isnan(s).all():
        return None
    m = np.nanmean(s)
    st = np.nanstd(s)
    pos = s[s > 0]
    neg = s[s < 0]
    fpos = float(np.sum(s > 0)) / float(np.sum(~np.isnan(s))) if np.sum(~np.isnan(s)) else 0.0
    fneg = float(np.sum(s < 0)) / float(np.sum(~np.isnan(s))) if np.sum(~np.isnan(s)) else 0.0
    mpos = float(np.nanmean(pos)) if pos.size else 0.0
    mneg = float(np.nanmean(neg)) if neg.size else 0.0
    return np.array([m, st, fpos, fneg, mpos, mneg], dtype=np.float32)



def plot_umap(embeddings, labels, output_path, title="UMAP of Representations"):
    if len(embeddings) != len(labels):
        print(f"[ERROR] plot_umap: {len(embeddings)} embeddings vs {len(labels)} labels")
        return

    if len(embeddings) == 0:
        print("[WARNING] No embeddings to plot")
        return
    umap_model = umap.UMAP()
    embedding_2d = umap_model.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embedding_2d[:, 0], y=embedding_2d[:, 1],
        hue=pd.Categorical(labels, categories=CATEGORY_ORDER, ordered=True),
        palette="tab10", s=40, alpha=0.8
    )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def leave_one_out_1nn_accuracy(embeddings, labels):
    labels = np.array(labels)
    embeddings = np.array(embeddings)

    nn = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    # Use the second closest point (excluding the point itself)
    nearest_neighbor_indices = indices[:, 1]
    preds = labels[nearest_neighbor_indices]

    accuracy = np.mean(preds == labels)
    conf_mat = confusion_matrix(labels, preds, labels=np.unique(labels))
    return accuracy, conf_mat

from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef

def loo_1nn_predictions(embeddings, labels):
    labels = np.asarray(labels)
    X = np.asarray(embeddings)
    nn = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(X)
    _, indices = nn.kneighbors(X)
    y_true = labels
    y_pred = labels[indices[:, 1]]  # exclude self
    return y_true, y_pred

def eval_metrics(y_true, y_pred, label_order=None):
    if label_order is None:
        label_order = np.unique(y_true)

    cm = confusion_matrix(y_true, y_pred, labels=label_order)

    # per-class recall (row-normalized cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    per_class_recall = np.diag(cm) / np.where(row_sums == 0, 1, row_sums).squeeze()

    # balanced accuracy and SEM across classes
    valid = ~np.isnan(per_class_recall)
    ba   = float(np.mean(per_class_recall[valid]))
    sem  = float(np.std(per_class_recall[valid], ddof=1) / np.sqrt(np.sum(valid)))
    ci95 = float(1.96 * sem)

    metrics = {
        "micro_accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": ba,                  # 0–1
        "balanced_accuracy_sem": sem,             # 0–1
        "balanced_accuracy_ci95": ci95,           # 0–1
        "macro_f1": float(f1_score(y_true, y_pred, labels=label_order, average='macro', zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=label_order, average='weighted', zero_division=0)),
        "cohens_kappa": float(cohen_kappa_score(y_true, y_pred, labels=label_order)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "per_class_recall": dict(zip(label_order, per_class_recall.astype(float))),
        "support": dict(zip(label_order, cm.sum(axis=1).astype(int))),
    }
    return cm, metrics, label_order

def plot_knn_heatmap(embeddings, labels, output_path, title="1-NN Classification"):
    present = [c for c in CATEGORY_ORDER if c in set(labels)]
    if not present:
        present = sorted(set(labels))

    y_true, y_pred = loo_1nn_predictions(embeddings, labels)
    cm, metrics, label_order = eval_metrics(y_true, y_pred, label_order=present)

    # Row-normalized matrix for visualization (per-class recall)
    with np.errstate(invalid='ignore', divide='ignore'):
        acc_matrix = cm.astype(float) / np.where(cm.sum(axis=1, keepdims=True)==0, 1, cm.sum(axis=1, keepdims=True))

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        acc_matrix,
        xticklabels=label_order,
        yticklabels=label_order,
        vmin=0, vmax=0.85,
        cmap="Blues", annot=True, fmt=".2f",
        cbar_kws={"label": "Per-class recall"}
    )
    plt.title(f"{title}\n"
              f"micro={metrics['micro_accuracy']:.2%} | "
              f"balanced={metrics['balanced_accuracy']:.2%} | "
              f"macro‑F1={metrics['macro_f1']:.2%}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    # (Optional) also print a terse summary to logs
    logging.info(f"[KNN] micro={metrics['micro_accuracy']:.3f}, "
                 f"balanced={metrics['balanced_accuracy']:.3f}, "
                 f"macroF1={metrics['macro_f1']:.3f}, "
                 f"weightedF1={metrics['weighted_f1']:.3f}, "
                 f"kappa={metrics['cohens_kappa']:.3f}, mcc={metrics['mcc']:.3f}")
    return metrics, label_order, acc_matrix  

def _save_summary(csv_path, row_dict):
    csv_path = pathlib.Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row_dict])
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False)


def save_reps(output_dir, model_id, group_name, scope, X, labels, metas, extra=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels)

    # 3a) embeddings + labels as NPZ (compressed)
    np.savez_compressed(out / f"reps_{model_id}_{group_name}_{scope}.npz",
                        embeddings=X, labels=labels)

    # 3b) metadata as Parquet for convenience
    mdf = pd.DataFrame(metas)
    mdf.insert(0, "label", labels)
    mdf.insert(0, "scope", scope)
    mdf.insert(0, "group", group_name)
    if extra:  # model metadata
        for k, v in extra.items(): mdf[k] = v
    mdf.to_parquet(out / f"reps_{model_id}_{group_name}_{scope}_meta.parquet", index=False)


def _save_per_class_json(json_path, label_order, acc_matrix):
    data = {
        "label_order": list(map(str, label_order)),
        "per_class_recall": {str(lbl): float(acc_matrix[i,i]) for i, lbl in enumerate(label_order)}
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

def analyze_agreement(
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
    training_task='dual',
    model_type='gamba'
):
    """
    Analyze agreement between predicted and true phyloP scores across different
    genomic regions and chromosomes using pre-saved BED files.
    """
    import logging
    from pathlib import Path
    import torch
    import glob
    from pyfaidx import Fasta

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = tokenizer = None
    if baseline == "none":
        model, tokenizer = load_model(checkpoint_dir, config_fpath, last_step=last_step, device=device, training_task=training_task, model_type=model_type)
    genome = Fasta(genome_fasta)

    bw = pyBigWig.open(bigwig_file)

    categories = [
        #"phyloP_negative", "phyloP_neutral", "phyloP_positive", 
        "vista_enhancer", "UCNE", "repeats", "exons", "introns", "noncoding_regions", "coding_regions", "upstream_TSS", "UTR5", "UTR3", "promoters"]

    chromosome_groups = {}
    if training_chromosomes and test_chromosomes:
        chromosome_groups = {
            "training": training_chromosomes,
            "test": test_chromosomes
        }
    else:
        chromosome_groups = {"all": chromosomes}

    all_group_embeddings = {
        group_name: {"roi": [], "full": [], "roi_labels": [], "full_labels": [],
                    "roi_meta": [], "full_meta": []}
        for group_name in chromosome_groups
    }


    for group_name, group_chroms in chromosome_groups.items():
        logging.info(f"Analyzing {group_name} chromosomes: {group_chroms}")
        for category in categories:
            bed_files = glob.glob(f"/home/mica/gamba/data_processing/data/regions/{category}/*.bed")
            group_regions = []
            for bed_file in bed_files:
                loaded = load_bed_file(bed_file, category, genome, bw)
                group_regions.extend([r for r in loaded if r["chrom"] in group_chroms])

            if not group_regions:
                logging.warning(f"No regions found for {category} in {group_name} chromosomes")
                continue

            group_regions = group_regions[:num_regions]

            valid_regions = []
            for i, region in enumerate(group_regions):
                if model_type in ["kmer6", "phylop"]:
                    context = extract_context(bigwig_file, region, genome, model_type="baseline")
                else:
                    context = extract_context(bigwig_file, region, genome, model_type)
                if not context or "sequence" not in context:
                    print(f"[WARN] Region {i} has invalid or truncated sequence")
                    continue
                # keep per-base phyloP for baseline
                if "scores" in region and isinstance(region["scores"], (list, np.ndarray)):
                    context["scores"] = np.asarray(region["scores"], dtype=np.float32)
                valid_regions.append(context)

            baseline = None
            if baseline == "kmer6":
                # ensure labels exist
                for r in valid_regions:
                    r["category"] = category

                full_embeds, full_labels, full_metas = compute_kmer_embeddings(valid_regions, mode="full", k=6)
                roi_embeds,  roi_labels, roi_metas  = compute_kmer_embeddings(valid_regions,  mode="roi",  k=6)

            elif baseline == "phylop":
                for r in valid_regions:
                    r["category"] = category

                full_embeds, full_labels, full_metas = compute_phylop_embeddings(valid_regions, mode="full")
                roi_embeds,  roi_labels, roi_metas  = compute_phylop_embeddings(valid_regions,  mode="roi")

            else:
                # --- trained model path ---
                sequence_representations, region_info = predict_scores_batched(
                    model, tokenizer, valid_regions,
                    batch_size=batch_size,
                    device=device,
                    model_type=model_type,
                    training_task=training_task
                )

                # region_info exists ONLY here
                for r in region_info:
                    r["category"] = category

                full_embeds, full_labels, full_metas = extract_region_embeddings(
                    sequence_representations, region_info, mode="full"
                )
                roi_embeds,  roi_labels, roi_metas  = extract_region_embeddings(
                    sequence_representations, region_info, mode="roi"
                )

            # common accumulation (works for both paths)
            all_group_embeddings[group_name]["roi"].extend(roi_embeds)
            all_group_embeddings[group_name]["full"].extend(full_embeds)
            all_group_embeddings[group_name]["roi_meta"].extend(roi_metas)
            all_group_embeddings[group_name]["full_meta"].extend(full_metas)
            assert len(full_labels) == len(full_embeds), "Full labels and embeddings mismatch!"
            assert len(roi_labels)  == len(roi_embeds),  "ROI labels and embeddings mismatch!"
            all_group_embeddings[group_name]["full_labels"].extend(full_labels)
            all_group_embeddings[group_name]["roi_labels"].extend(roi_labels)


    for group_name, group_data in all_group_embeddings.items():
        plot_umap(group_data["roi"], group_data["roi_labels"], output_dir / f"global_umap_roi_{group_name}.png", title=f"Global UMAP - ROI ({group_name})")
        roi_metrics, roi_labels, roi_mat  = plot_knn_heatmap(group_data["roi"], group_data["roi_labels"], output_dir / f"global_knn_roi_{group_name}.png", title=f"1-NN Accuracy - ROI ({group_name})")
        plot_umap(group_data["full"], group_data["full_labels"], output_dir / f"global_umap_full_{group_name}.png", title=f"Global UMAP - Full ({group_name})")
        full_metrics, full_labels, full_mat  = plot_knn_heatmap(group_data["full"], group_data["full_labels"], output_dir / f"global_knn_full_{group_name}.png", title=f"1-NN Accuracy - Full ({group_name})")
    # save JSON per-class recalls (optional but handy)
    _save_per_class_json(output_dir / f"per_class_roi_{group_name}.json",  roi_labels,  roi_mat)
    _save_per_class_json(output_dir / f"per_class_full_{group_name}.json", full_labels, full_mat)

    model_id = (model_type if baseline==None else baseline)  # e.g., gamba, caduceus, kmer6, phylop
    extra = {
        "model_id": model_id,
        "training_task": training_task,
        "baseline": baseline,
        "last_step": last_step,
    }

    ag = all_group_embeddings[group_name]
    save_reps(output_dir, model_id, group_name, "roi",  ag["roi"],  ag["roi_labels"],  ag["roi_meta"],  extra)
    save_reps(output_dir, model_id, group_name, "full", ag["full"], ag["full_labels"], ag["full_meta"], extra)

    # save one CSV row per scope
    summary_csv = os.path.join(output_dir, "representation_knn_summary.csv")
    for scope, m in [("roi", roi_metrics), ("full", full_metrics)]:
        _save_summary(summary_csv, {
            "Group": group_name,                # all / train / test
            "Scope": scope,                     # roi / full
            "Model": model_type,                # class or baseline
            "BalancedAccuracyPct": 100.0 * m["balanced_accuracy"],
            "BalancedAccuracySEM_Pct": 100.0 * m["balanced_accuracy_sem"],
            "MicroAccuracyPct": 100.0 * m["micro_accuracy"],
            "MacroF1Pct": 100.0 * m["macro_f1"],
        })

    bw.close()
    

def main():
    parser = argparse.ArgumentParser(
        description="Analyze agreement between predicted and true phyloP scores"
    )
    parser.add_argument(
        "--bigwig_file",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig",
        help="Path to the bigwig file with phyloP scores",
    )
    parser.add_argument(
        "--genome_fasta",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa",
        help="Path to the genome fasta file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/global_representations",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default='/home/mica/gamba/',
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--config_fpath",
        type=str,
        default='/home/mica/gamba/configs/jamba-small-240mammalian.json',
        help="Path to model config JSON",
    )
    parser.add_argument(
        "--num_regions",
        type=int,
        default=1000,
        help="Number of regions to sample per category",
    )
    parser.add_argument(
        "--chromosomes",
        type=str,
        nargs="+",
        default=["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chrX"],
        help="List of chromosomes to analyze",
    )
    parser.add_argument(
        "--training_chromosomes",
        type=str,
        nargs="+",
        #default= None,
        default=["chr1", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10","chr11", "chr12", "chr13", "chr14", "chr15", "chr17", "chr18", "chr19", "chr20", "chr21", "chrX"],
        help="List of chromosomes used in training",
    )
    parser.add_argument(
        "--test_chromosomes",
        type=str,
        nargs="+",
        #default= None,
        default=["chr2", "chr22", "chr16", "chr3"],
        help="List of chromosomes held out for testing",
    )
    parser.add_argument(
        "--last_step",
        type=int,
        default=44000, #0, #44000
        help="Checkpoint step to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for model predictions",
    )
    parser.add_argument(
        "--model_type", type=str, choices=["gamba", "caduceus"],
        default=None,
        help="Only required when baseline == 'none'"
    )

    parser.add_argument(
        "--training_task", type=str, choices=["dual", "cons_only", "seq_only"],
        default=None,
        help="Only required when baseline == 'none'"
    )

    parser.add_argument(
        "--baseline",
        type=str,
        choices=["none", "kmer6", "phylop"],
        default="none",
        help="Baseline embedding instead of a trained model"
    )


    args = parser.parse_args()
    
    # Configure logging to include timestamps
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

        
    logging.info(f"Starting analysis script with chromosomes: {args.chromosomes}")
    logging.info(f"Training chromosomes: {args.training_chromosomes}")
    logging.info(f"Test chromosomes: {args.test_chromosomes}")

    # Validate conditional requirements
    if args.baseline == "none":
        if args.model_type is None or args.training_task is None:
            raise SystemExit("When --baseline=none, you must provide --model_type and --training_task.")
        logging.info(f"Using MODEL: type={args.model_type}, task={args.training_task}")
    else:
        # clear model-related args; they won't be used
        logging.info(f"Using BASELINE: {args.baseline}")
        #set all chromosomes as train, no test
        args.training_chromosomes = args.chromosomes
        args.test_chromosomes = None
        args.model_type = None
        args.training_task = None

    if args.baseline != "none":
        # e.g., /.../global_representations/baseline/kmer6/
        output_dir = os.path.join(args.output_dir, "baseline", args.baseline)
        args.model_type = args.baseline
    else:
        # keep your existing convention for trained models
        if args.model_type == 'gamba':
            checkpoint_dir = args.checkpoint_dir + f"/clean_dcps/CCP/"
            #checkpoint_dir = args.checkpoint_dir + f"/clean_dcps/focal_loss/"
            # if args.training_task == "seq_only":
            #     checkpoint_dir = args.checkpoint_dir + f"/clean_dcps/CCP/"
            #     args.last_step = 56000 #0 #56000
        else:
            checkpoint_dir = args.checkpoint_dir #+ f"/clean_caduceus_dcps/"
            #args.last_step = 56000

        if args.last_step == 0:
            last_step = "random_init"
        else:
            last_step = args.last_step
        output_dir = args.output_dir + f"/{args.model_type}_{args.training_task}_ALLPOSstep_{last_step}/"

    try:
        analyze_kwargs = dict(
            genome_fasta=args.genome_fasta,
            bigwig_file=args.bigwig_file,
            checkpoint_dir=None,
            config_fpath=args.config_fpath,
            output_dir=output_dir,
            num_regions=args.num_regions,
            chromosomes=args.chromosomes,
            training_chromosomes=args.training_chromosomes,
            test_chromosomes=args.test_chromosomes,
            last_step=args.last_step,
            batch_size=args.batch_size,
            training_task=args.training_task,
            model_type=args.model_type,
            baseline=args.baseline
        )
        if args.baseline == "none":
            analyze_kwargs["checkpoint_dir"] = checkpoint_dir

        analyze_agreement(**analyze_kwargs)
        logging.info("Analysis completed successfully")
    except Exception as e:
        logging.error(f"Error in analysis: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()