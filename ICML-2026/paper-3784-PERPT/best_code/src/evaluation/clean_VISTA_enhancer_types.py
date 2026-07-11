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
import re
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
sys.path.append("../gamba")
sys.path.append("/home/mica/gamba/")
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
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef


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

    for rep, info in zip(representations, region_info):
        if isinstance(rep, float) or np.isnan(rep).any():
            continue

        if mode == "roi":
            fs = info["feature_start_in_window"]
            fe = info["feature_end_in_window"]
            rep_slice = rep[fs:fe]
        elif mode == "full":
            rep_slice = rep
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if rep_slice.shape[0] == 0:
            continue

        pooled = rep_slice.mean(axis=0)  # shape: (hidden_dim,)
        embeddings.append(pooled)
        labels.append(info.get("category", "unknown"))
    print(f"[extract_region_embeddings] mode={mode}, returning {len(embeddings)} embeds and {len(labels)} labels")

    return np.stack(embeddings), labels

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
    kmer_index = _build_kmer_index(k=k)
    embeddings, labels = [], []
    for r in valid_regions:
        seq = r["sequence"]
        if mode == "roi":
            span = _roi_span(r)
            if span is None: 
                continue
            fs, fe = span
            seq = seq[fs:fe]
        # skip empty/too short
        if not seq or len(seq) < k:
            continue
        vec = _seq_to_kmer_vec(seq, k, kmer_index)
        embeddings.append(vec)
        labels.append(r.get("category", "unknown"))
    if len(embeddings) == 0:
        return np.empty((0, 4**k), dtype=np.float32), []
    return np.vstack(embeddings), labels

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

def compute_phylop_embeddings(valid_regions, mode="roi"):
    embeddings, labels = [], []
    for r in valid_regions:
        scores = r.get("scores", None)
        if scores is None:
            continue
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
        embeddings.append(feat)
        labels.append(r.get("category", "unknown"))
    if len(embeddings) == 0:
        return np.empty((0, 6), dtype=np.float32), []
    return np.vstack(embeddings), labels

def plot_umap(embeddings, labels, output_path, title, label_order, palette):
    if len(embeddings) == 0: return
    umap_model = umap.UMAP()
    embedding_2d = umap_model.fit_transform(embeddings)
    df = pd.DataFrame({"x": embedding_2d[:,0], "y": embedding_2d[:,1], "label": labels})
    sns.scatterplot(data=df, x="x", y="y", hue="label", palette=palette, s=40, alpha=0.8,
                    hue_order=label_order, legend="full")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def loo_1nn_predictions(embeddings, labels):
    labels = np.asarray(labels)
    X = np.asarray(embeddings)
    nn = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(X)
    _, indices = nn.kneighbors(X)
    y_true = labels
    y_pred = labels[indices[:, 1]]  # exclude self
    return y_true, y_pred

def eval_metrics(y_true, y_pred, label_order=None):
    # Choose a consistent order for confusion matrix + reporting
    if label_order is None:
        label_order = np.unique(y_true)

    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    micro_acc = (y_true == y_pred).mean()
    bal_acc   = balanced_accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, labels=label_order, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=label_order, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred, labels=label_order)
    mcc   = matthews_corrcoef(y_true, y_pred)

    # Per-class recall (row-normalized cm)
    with np.errstate(invalid='ignore', divide='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        per_class_recall = np.diag(cm) / np.where(row_sums==0, 1, row_sums).squeeze()

    metrics = {
        "micro_accuracy": float(micro_acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "cohens_kappa": float(kappa),
        "mcc": float(mcc),
        "per_class_recall": dict(zip(label_order, per_class_recall.astype(float))),
        "support": dict(zip(label_order, cm.sum(axis=1).astype(int))),
    }
    return cm, metrics, label_order

def plot_knn_heatmap(embeddings, labels, output_path, title, label_order):
    y_true, y_pred = loo_1nn_predictions(embeddings, labels)
    cm, metrics, order = eval_metrics(y_true, y_pred, label_order=label_order)
    acc_matrix = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(12, 10))
    sns.heatmap(acc_matrix, xticklabels=order, yticklabels=order,
                vmin=0, vmax=0.85, cmap="Blues", annot=True, fmt=".2f",
                cbar_kws={"label": "Per-class recall"})
    plt.title(f"{title}\n"
              f"micro={metrics['micro_accuracy']:.2%} | "
              f"balanced={metrics['balanced_accuracy']:.2%} | "
              f"macro-F1={metrics['macro_f1']:.2%}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()




def load_enhancer_data(tsv_path: str) -> pd.DataFrame:
    """Load enhancer data from TSV file."""
    print(f"Reading data from {tsv_path}")
    
    # Read the TSV file
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        print(f"Loaded {len(df)} total records")
    except Exception as e:
        print(f"Error reading TSV file: {e}")
        return pd.DataFrame()
    
    # Print column names to debug
    print("Columns in file:", df.columns.tolist())
    
    # Handle tissue sets with more robust error checking
    df['Tissues'] = df['Tissues'].fillna('').astype(str)
    df['tissue_set'] = df['Tissues'].apply(
        lambda x: set(x.split(', ')) if x and x.strip() else set()
    )
    
    # Filter for positive expression with error checking
    if 'Expression' not in df.columns:
        print("Error: 'Expression' column not found in data")
        return pd.DataFrame()
        
    positive_df = df[df['Expression'] == 'positive'].copy()
    print(f"Found {len(positive_df)} records with positive expression")
    
    # Parse coordinates
    def parse_coordinates(coord_str):
        try:
            if pd.isna(coord_str):
                return None, None, None
            # Try both formats: chr:start-end and just the coordinates
            match = re.match(r'(?:chr)?(\d+):(\d+)-(\d+)', str(coord_str))
            if match:
                return f"chr{match.group(1)}", int(match.group(2)), int(match.group(3))
            return None, None, None
        except Exception as e:
            print(f"Error parsing coordinates {coord_str}: {e}")
            return None, None, None
    
    # Apply coordinate parsing
    coords = positive_df['Element Coordinates'].apply(parse_coordinates)
    positive_df['chrom'] = coords.apply(lambda x: x[0])
    positive_df['start'] = coords.apply(lambda x: x[1])
    positive_df['end'] = coords.apply(lambda x: x[2])
    
    # Filter out rows with invalid coordinates
    valid_coords = (
        positive_df['chrom'].notna() & 
        positive_df['start'].notna() & 
        positive_df['end'].notna()
    )
    valid_df = positive_df[valid_coords].copy()
    
    # Convert coordinates to integers
    valid_df['start'] = valid_df['start'].astype(int)
    valid_df['end'] = valid_df['end'].astype(int)
    
    print(f"Final dataset: {len(valid_df)} valid enhancers with positive expression")
    print("Sample of processed data:")
    print(valid_df[['Element ID', 'chrom', 'start', 'end', 'Tissues']].head())
    
    return valid_df


def analyze_enhancers(
    genome_fasta,
    bigwig_file,
    checkpoint_dir,
    config_fpath,
    output_dir,
    enhancer_file=None,          # NEW: TSV path
    baseline="none",
    num_regions=250,
    chromosomes=None,
    last_step=44000,
    batch_size=8,
    training_chromosomes=None,
    test_chromosomes=None,
    training_task='dual',
    model_type='gamba'
):
    """
    TSV-only, multi-label pipeline for VISTA enhancer tissues.

    - One embedding per enhancer.
    - Labels are sets of tissues (multi-label).
    - UMAP uses a single dominant tissue for coloring (for visualization only).
    - k-NN is multi-label (LOO), metrics saved as JSON.
    """
    import logging, json
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    import pyBigWig
    from pyfaidx import Fasta
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.metrics import f1_score, hamming_loss, accuracy_score
    from sklearn.neighbors import NearestNeighbors

    # ------------------------- helpers (scoped) -------------------------
    def build_multilabel_targets(label_sets, tissue_order=None):
        all_labels = sorted(set().union(*label_sets)) if tissue_order is None else list(tissue_order)
        mlb = MultiLabelBinarizer(classes=all_labels)
        Y = mlb.fit_transform(label_sets)  # (N, C)
        return Y.astype(np.int32), mlb.classes_.tolist()

    def knn_multilabel_predict(X, Y, k=5, thresh=0.5, metric='euclidean'):
        nn = NearestNeighbors(n_neighbors=k+1, metric=metric).fit(X)
        _, idx = nn.kneighbors(X)
        neigh = idx[:, 1:k+1]              # drop self
        votes = Y[neigh].sum(axis=1)       # (N, C)
        freq = votes / k
        Y_pred = (freq >= thresh).astype(np.int32)
        return Y_pred, freq

    def multilabel_metrics(Y_true, Y_pred, label_names):
        out = {}
        out["subset_accuracy"] = float(accuracy_score(Y_true, Y_pred))
        out["hamming_loss"]    = float(hamming_loss(Y_true, Y_pred))
        out["micro_f1"]        = float(f1_score(Y_true, Y_pred, average='micro', zero_division=0))
        out["macro_f1"]        = float(f1_score(Y_true, Y_pred, average='macro', zero_division=0))
        per_label_f1 = f1_score(Y_true, Y_pred, average=None, zero_division=0)
        out["per_label_f1"]    = dict(zip(label_names, per_label_f1.astype(float)))
        return out

    def dominant_label_for_plot(label_sets, label_order):
        order_index = {lab:i for i,lab in enumerate(label_order)}
        dom = []
        for s in label_sets:
            if not s:
                dom.append(None)
            else:
                dom.append(sorted(list(s), key=lambda x: order_index.get(x, 1e9))[0])
        return dom

    def plot_umap_multilabel(E, label_sets, out_png, title, label_order, palette):
        if len(E) == 0: return
        emb2d = umap.UMAP().fit_transform(np.asarray(E))
        one_label = dominant_label_for_plot(label_sets, label_order)
        df = pd.DataFrame({"x": emb2d[:,0], "y": emb2d[:,1], "label": one_label})
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=df, x="x", y="y", hue="label",
            hue_order=label_order, palette=palette, s=40, alpha=0.9, legend=True
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=300); plt.close()

    def eval_and_log_knn_multilabel(E, label_sets, out_json, title, label_order, k=5, thresh=0.5):
        if len(E) == 0: return
        X = np.asarray(E)
        Y_true, classes = build_multilabel_targets(label_sets, tissue_order=label_order)
        Y_pred, _ = knn_multilabel_predict(X, Y_true, k=k, thresh=thresh)
        m = multilabel_metrics(Y_true, Y_pred, classes)
        m["title"] = title
        m["k"] = k
        m["threshold"] = thresh
        with open(out_json, "w") as f:
            json.dump(m, f, indent=2)
        logging.info(f"[{title}] subset_acc={m['subset_accuracy']:.3f}, "
                     f"hamming={m['hamming_loss']:.3f}, microF1={m['micro_f1']:.3f}, macroF1={m['macro_f1']:.3f}")

    # ------------------------- setup -------------------------
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"[analyze_enhancers] device={device} baseline={baseline} model_type={model_type} task={training_task}")

    # Load model if needed
    model = tokenizer = None
    if baseline == "none":
        model, tokenizer = load_model(
            checkpoint_dir, config_fpath,
            last_step=last_step, device=device,
            training_task=training_task, model_type=model_type
        )

    genome = Fasta(genome_fasta)
    bw = pyBigWig.open(bigwig_file)

    # Decide grouping of chromosomes
    if training_chromosomes and test_chromosomes:
        chromosome_groups = {"training": set(training_chromosomes),
                             "test": set(test_chromosomes)}
    else:
        chromosome_groups = {"all": set(chromosomes or [])}

    # Load TSV of VISTA enhancers
    if enhancer_file is None:
        # fall back to global args if present (keeps backward-compat with your earlier script)
        try:
            enhancer_path = args.enhancer_file
        except NameError:
            raise RuntimeError("enhancer_file must be provided to analyze_enhancers")
    else:
        enhancer_path = enhancer_file

    logging.info("Loading enhancer TSV…")
    enhancer_df = load_enhancer_data(enhancer_path)
    if enhancer_df.empty:
        logging.warning("No valid enhancers loaded from TSV. Exiting.")
        bw.close(); return

    # Require at least one tissue
    enhancer_df = enhancer_df[enhancer_df["tissue_set"].apply(lambda s: len(s) > 0)].copy()
    logging.info(f"Enhancers with ≥1 tissue: {len(enhancer_df)}")

    # Storage per group
    group_store = {
        gname: {"roi": [], "full": [], "roi_label_sets": [], "full_label_sets": []}
        for gname in chromosome_groups
    }

    # helper to cap count per tissue
    def cap_per_tissue(exploded_df, per_label_cap):
        out_parts = []
        for lab, sub in exploded_df.groupby("tissue"):
            out_parts.append(sub.iloc[:per_label_cap])
        return pd.concat(out_parts, ignore_index=True) if out_parts else exploded_df

    # ------------------------- main loop -------------------------
    for group_name, chrom_set in chromosome_groups.items():
        logging.info(f"[Group: {group_name}] Chromosomes: {sorted(list(chrom_set))}")
        group_df = enhancer_df[enhancer_df["chrom"].isin(chrom_set)].copy()
        if group_df.empty:
            logging.warning(f"[{group_name}] No enhancers on these chromosomes."); continue

        # Explode for selection & capping (but we will embed each enhancer once)
        exploded = (
            group_df
            .assign(tissue=lambda df: df["tissue_set"])
            .explode("tissue")
            .dropna(subset=["tissue"])
        )
        exploded_capped = cap_per_tissue(exploded, num_regions)

        # Keys of enhancers to include
        exploded_capped["key"] = (
            exploded_capped["chrom"].astype(str) + ":" +
            exploded_capped["start"].astype(int).astype(str) + "-" +
            exploded_capped["end"].astype(int).astype(str)
        )
        need_keys = exploded_capped["key"].unique().tolist()

        # Build maps for region + tissue_set
        group_df["key"] = (
            group_df["chrom"].astype(str) + ":" +
            group_df["start"].astype(int).astype(str) + "-" +
            group_df["end"].astype(int).astype(str)
        )
        key_to_region = {
            row["key"]: {"chrom": str(row["chrom"]), "start": int(row["start"]), "end": int(row["end"]), "strand": "+"}
            for _, row in group_df.iterrows() if row["key"] in need_keys
        }
        key_to_tissues = {
            row["key"]: set(map(str, row["tissue_set"])) for _, row in group_df.iterrows() if row["key"] in need_keys
        }

        # Extract context once per enhancer
        valid_keys, valid_regions = [], []
        for k in need_keys:
            region = key_to_region.get(k)
            if region is None: continue
            ctx = extract_context(bigwig_file, region, genome, model_type)
            if not ctx or "sequence" not in ctx:
                logging.warning(f"[{group_name}] context failed for {k}")
                continue
            if "scores" in ctx and isinstance(ctx["scores"], (list, np.ndarray)):
                ctx["scores"] = np.asarray(ctx["scores"], dtype=np.float32)
            valid_keys.append(k); valid_regions.append(ctx)

        if len(valid_regions) == 0:
            logging.warning(f"[{group_name}] No valid regions after context extraction."); continue

        # Compute embeddings (baseline or model)
        if baseline == "kmer6":
            full_embeds, _ = compute_kmer_embeddings(valid_regions, mode="full", k=6)
            roi_embeds,  _ = compute_kmer_embeddings(valid_regions, mode="roi",  k=6)
        elif baseline == "phylop":
            full_embeds, _ = compute_phylop_embeddings(valid_regions, mode="full")
            roi_embeds,  _ = compute_phylop_embeddings(valid_regions, mode="roi")
        else:
            seq_reps, region_info = predict_scores_batched(
                model, tokenizer, valid_regions,
                batch_size=batch_size, device=device,
                model_type=model_type, training_task=training_task
            )
            full_embeds, _ = extract_region_embeddings(seq_reps, region_info, mode="full")
            roi_embeds,  _ = extract_region_embeddings(seq_reps, region_info, mode="roi")

        # Align label sets to valid_keys order
        label_sets = [key_to_tissues[k] for k in valid_keys]

        # Accumulate
        group_store[group_name]["roi"] = np.asarray(roi_embeds)
        group_store[group_name]["full"] = np.asarray(full_embeds)
        group_store[group_name]["roi_label_sets"]  = label_sets
        group_store[group_name]["full_label_sets"] = label_sets

        # Log counts for visibility
        flat = pd.Series([t for s in label_sets for t in s])
        logging.info(f"[{group_name}] samples per tissue (unique enhancers): {flat.value_counts().to_dict()}")

    # ------------------------- global order + palette -------------------------
    # Collect all tissues present across groups
    all_sets = []
    for g in group_store.values():
        all_sets.extend(g.get("roi_label_sets", []))
    ALL_TISSUES = sorted(set().union(*all_sets)) if all_sets else []

    # Fixed palette across all plots
    tissue_palette = dict(zip(ALL_TISSUES, sns.color_palette("tab20", len(ALL_TISSUES))))

    # ------------------------- plotting & metrics -------------------------
    for group_name, data in group_store.items():
        E_roi   = np.asarray(data["roi"])
        E_full  = np.asarray(data["full"])
        L_roi   = data["roi_label_sets"]
        L_full  = data["full_label_sets"]

        # UMAPs (dominant label for color)
        plot_umap_multilabel(
            E_roi, L_roi, Path(outdir, f"vista_umap_roi_{group_name}.png"),
            f"VISTA tissues (ROI) — {group_name}", ALL_TISSUES, tissue_palette
        )
        plot_umap_multilabel(
            E_full, L_full, Path(outdir, f"vista_umap_full_{group_name}.png"),
            f"VISTA tissues (Full) — {group_name}", ALL_TISSUES, tissue_palette
        )

        # Multi-label 1-NN metrics (saved as JSON reports)
        eval_and_log_knn_multilabel(
            E_roi, L_roi, Path(outdir, f"vista_knn_roi_{group_name}.json"),
            f"1‑NN multilabel — ROI ({group_name})", ALL_TISSUES, k=5, thresh=0.5
        )
        eval_and_log_knn_multilabel(
            E_full, L_full, Path(outdir, f"vista_knn_full_{group_name}.json"),
            f"1‑NN multilabel — Full ({group_name})", ALL_TISSUES, k=5, thresh=0.5
        )

    bw.close()
    logging.info("TSV-only multi‑label tissue analysis complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Check separation of VISTA enhancer types"
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
        default="/home/mica/gamba/data_processing/data/240-mammalian/VISTA_enhancer_types/",
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
    parser.add_argument('--enhancer_file', type=str, default ='/home/mica/gamba/data_processing/data/VISTA_enhancers/experiments.tsv', help='BED file for VISTA enhancers')
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
        default=["chr1", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10","chr11", "chr12", "chr13", "chr14", "chr15", "chr17", "chr18", "chr19", "chr20", "chr21", "chrX"],
        help="List of chromosomes used in training",
    )
    parser.add_argument(
        "--test_chromosomes",
        type=str,
        nargs="+",
        default=["chr2", "chr22", "chr16", "chr3"],
        help="List of chromosomes held out for testing",
    )
    parser.add_argument(
        "--last_step",
        type=int,
        default=44000,
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
    parser.add_argument(
        "--num_regions",
        type=int,
        default=1000,
        help="Maximum number of regions to analyze per tissue type (for VISTA enhancers)",
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
    else:
        # keep your existing convention for trained models
        if args.model_type == 'gamba':
            checkpoint_dir = args.checkpoint_dir + f"/clean_dcps/CCP/"
            if args.training_task == "seq_only":
                checkpoint_dir = args.checkpoint_dir + f"/clean_dcps/"
                args.last_step = 56000
        else:
            checkpoint_dir = args.checkpoint_dir + f"/clean_caduceus_dcps/"
            args.last_step = 56000

        output_dir = args.output_dir + f"/{args.model_type}_{args.training_task}_step_{args.last_step}/"

    try:
        analyze_kwargs = dict(
            genome_fasta=args.genome_fasta,
            bigwig_file=args.bigwig_file,
            checkpoint_dir=None,
            config_fpath=args.config_fpath,
            output_dir=output_dir,
            enhancer_file = args.enhancer_file,
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

        analyze_enhancers(**analyze_kwargs)
        logging.info("Analysis completed successfully")
    except Exception as e:
        logging.error(f"Error in analysis: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()