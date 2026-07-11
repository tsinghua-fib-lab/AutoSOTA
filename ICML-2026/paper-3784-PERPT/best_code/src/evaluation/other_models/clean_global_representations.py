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
from torch.nn import MSELoss, CrossEntropyLoss
from tqdm import tqdm
from transformers.modeling_outputs import MaskedLMOutput
from torch.nn import functional as F
from types import MethodType
# Update imports
from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoModel, AutoTokenizer
from transformers import TrainingArguments, Trainer
import torch
sys.path.append("/home/mica/gamba/src/")
from evaluation.utils.helpers import load_bed_file, extract_context, extract_phyloP_scores

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


# Patch the model's forward() to use ignore_index = -100
def patched_forward(self, input_ids=None, inputs_embeds=None, labels=None, loss_weights=None, output_hidden_states=None, return_dict=None):
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
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
        if loss_weights is not None:
            loss = weighted_cross_entropy(logits, labels, loss_weights, ignore_index=-100)
        else:
            loss = cross_entropy(logits, labels, ignore_index=-100)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return MaskedLMOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
    )

def load_model(
    model_type="gamba",
    device=None
):
    """Load model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Loading model of type {model_type} on device {device}")
    if model_type == "nt-ms":
        model_name = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = model.to(device)
        return model, tokenizer

    if model_type=="caduceus_small":
        model_name = "kuleshov-group/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        model.forward = MethodType(patched_forward, model)
        model.eval()
        return model, tokenizer

    if model_type=="caduceus":
        model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        model.forward = MethodType(patched_forward, model)
        model.eval()
        return model, tokenizer

    elif model_type == "nt-human":
        model_name = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = model.to(device)
        return model, tokenizer

    elif model_type == "hyenaDNA":
        checkpoint = 'LongSafari/hyenadna-medium-160k-seqlen-hf'
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        model = model.to(device)
        return model, tokenizer
    else:
        #model type is gpn
        model_name = "phyloGPN"
        checkpoint = "songlab/PhyloGPN"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
        model = model.to(device)
        return model, tokenizer
    

def predict_scores_batched(model, tokenizer, regions, batch_size, device, model_type):
    """
    Extract representations and metadata from regions in batches.
    - Filters invalid sequences per batch, not globally.
    - Extends global region_info only after each batch is processed.
    """

    all_embeddings = []
    all_region_info = []

    for batch_start in range(0, len(regions), batch_size):
        batch_regions = regions[batch_start:batch_start + batch_size]

        # ---- Build batch-local info & sequences ----
        batch_region_info = []
        batch_sequences = []

        for r in batch_regions:
            seq = r.get("sequence", None)
            if not seq:
                continue

            fs = r.get("feature_start_in_window", 0)
            fe = r.get("feature_end_in_window", len(seq))

            batch_region_info.append({
                "chrom": r["chrom"],
                "start": r["start"],
                "end": r["end"],
                "feature_id": r.get("feature_id", "unknown"),
                "mean_score": r.get("mean_score", 0.0),
                "feature_start_in_window": fs,
                "feature_end_in_window": fe
            })
            batch_sequences.append(seq)

        # ---- Model-specific handling ----
        if model_type == "phyloGPN":
            keep_idx = [i for i, seq in enumerate(batch_sequences) if len(seq) == 481]
            if len(keep_idx) != len(batch_sequences):
                print(f"[WARNING] Dropping {len(batch_sequences) - len(keep_idx)} invalid-length sequences for phyloGPN")
            batch_sequences = [batch_sequences[i] for i in keep_idx]
            batch_region_info = [batch_region_info[i] for i in keep_idx]

        elif model_type.startswith("nt-"):
            keep_idx = [i for i, seq in enumerate(batch_sequences) if len(seq) > 0]
            if len(keep_idx) != len(batch_sequences):
                print(f"[WARNING] Dropping {len(batch_sequences) - len(keep_idx)} empty sequences for NT model")
            batch_sequences = [batch_sequences[i] for i in keep_idx]
            batch_region_info = [batch_region_info[i] for i in keep_idx]

        elif model_type == "caduceus" or model_type == "caduceus_small":
            keep_idx = [i for i, seq in enumerate(batch_sequences) if len(seq) > 0]
            if len(keep_idx) != len(batch_sequences):
                print(f"[WARNING] Dropping {len(batch_sequences) - len(keep_idx)} empty sequences for caduceus")
            batch_sequences = [batch_sequences[i] for i in keep_idx]
            batch_region_info = [batch_region_info[i] for i in keep_idx]

        elif model_type == "hyenaDNA":
            keep_idx = [i for i, seq in enumerate(batch_sequences) if len(seq) > 0]
            if len(keep_idx) != len(batch_sequences):
                print(f"[WARNING] Dropping {len(batch_sequences) - len(keep_idx)} empty sequences for hyenaDNA")
            batch_sequences = [batch_sequences[i] for i in keep_idx]
            batch_region_info = [batch_region_info[i] for i in keep_idx]

            if not batch_sequences:
                continue  # skip if nothing remains

            # Tokenize & run model
            inputs = tokenizer(batch_sequences, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1].to(torch.float32)  # cast to avoid bfloat16 numpy issue

                #print(f"[DEBUG] input_ids shape: {inputs['input_ids'].shape}")
                #print(f"[DEBUG] hidden state shape: {hidden_states.shape}")

                for idx in range(len(batch_sequences)):
                    region_repr = hidden_states[idx]  # (seq_len, hidden_dim)
                    all_embeddings.append(region_repr.cpu().numpy())
                    all_region_info.append(batch_region_info[idx])

            continue  # skip generic handling for HyenaDNA

        # ---- Skip empty batches after filtering ----
        if not batch_sequences:
            continue

        # ---- Generic tokenization & embedding extraction ----
        inputs = tokenizer(batch_sequences, return_tensors="pt", padding=True, truncation=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if hasattr(model, "get_embeddings"):
                reps = model.get_embeddings(inputs["input_ids"])  # (B, L, H) or (B, H)
            else:
                outputs = model(**inputs, output_hidden_states=True)
                reps = outputs.hidden_states[-1]  # last layer (B, L, H)

        # ---- Pool token-level output to region-level ----
        # if reps.ndim == 3:
        #     reps = reps.mean(dim=1)
        reps = reps.float()

        # ---- Append outputs ----
        all_embeddings.extend(reps.cpu().numpy())
        all_region_info.extend(batch_region_info)

    # ---- Final sanity check ----
    assert len(all_embeddings) == len(all_region_info), \
        f"Mismatch: {len(all_embeddings)} embeddings vs {len(all_region_info)} region_info"

    return all_embeddings, all_region_info


def extract_region_embeddings(representations, region_info, mode="roi", model_type=None):
    """
    Extract embeddings for the region of interest (ROI) or full sequence.

    Args:
        representations: list of np.arrays of shape (seq_len, hidden_dim)
        region_info: list of dicts with keys 'feature_start_in_window', 'feature_end_in_window', 'category'
        mode: "roi" for region-of-interest, "full" for full-sequence average
        model_type: used to apply token index correction for k-mer models

    Returns:
        Tuple of:
        - np.array of pooled embeddings, shape (n_samples, hidden_dim)
        - list of category labels
    """
    embeddings = []
    labels = []
    print(f"[DEBUG] Received {len(representations)} representations and {len(region_info)} regions")
    
    for rep, info in zip(representations, region_info):
        if isinstance(rep, float) or np.isnan(rep).any():
            print(f"[SKIP] index {i}: rep is invalid")
            continue
        if rep.shape[0] == 0:
            print(f"[SKIP] index {i}: rep.shape[0] == 0")
            continue
        print(f"[DEBUG] Processing representation with shape {rep.shape} and info {info}")
        
        if mode == "roi":
            fs = info["feature_start_in_window"]
            fe = info["feature_end_in_window"]
            
            # Adjust for 6-mer tokenization used by NT
            if model_type in ["nt", "nt-ms"]:
                fs = fs // 6
                fe = fe // 6

            # Skip if slice out of bounds
            if fe > rep.shape[0]:
                print(f"[WARNING] ROI slice out of bounds: {fs}-{fe} vs rep.shape[0]={rep.shape[0]}")
                continue

            rep_slice = rep[fs:fe]
        
        elif mode == "full":
            rep_slice = rep
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        print(f"[DEBUG] Extracted slice with shape {rep_slice.shape} for mode {mode}")
        if rep_slice.shape[0] == 0:
            continue
        pooled = rep_slice.mean(axis=0)

        embeddings.append(pooled)
        print(f"[DEBUG] Appending pooled rep of shape {pooled.shape}, label={info.get('category', 'unknown')}")
        labels.append(info.get("category", "unknown"))

    print(f"[DEBUG] Total embeddings collected: {len(embeddings)}, len of labels: {len(labels)}")

    return np.stack(embeddings), labels

import umap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_umap(embeddings, labels, output_path, title="UMAP of Representations"):
    print(f"[UMAP] embeddings={np.array(embeddings).shape}, labels={len(labels)}")
    assert len(embeddings) == len(labels), f"UMAP mismatch before transform: {len(embeddings)} vs {len(labels)}"
    umap_model = umap.UMAP()
    embedding_2d = umap_model.fit_transform(embeddings)
    print(f"[CHECK] embedding_2d shape: {embedding_2d.shape}")
    print(f"[CHECK] labels length: {len(labels)}")
    assert embedding_2d.shape[0] == len(labels), f"Mismatch: {embedding_2d.shape[0]} != {len(labels)}"
    if embedding_2d.shape[0] != len(labels):
        print(f"[ERROR] UMAP plotting mismatch: {embedding_2d.shape[0]} embeddings vs {len(labels)} labels")
        min_len = min(embedding_2d.shape[0], len(labels))
        embedding_2d = embedding_2d[:min_len]
        labels = labels[:min_len]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=labels, palette="tab10", s=40, alpha=0.8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved figure to {output_path}.")


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors

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


def analyze_agreement(
    genome_fasta,
    bigwig_file,
    output_dir,
    num_regions=100,
    chromosomes=None,
    last_step=44000,
    batch_size=8,
    training_chromosomes=None,
    test_chromosomes=None,
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

    model, tokenizer = load_model( model_type=model_type, device =device)
    genome = Fasta(genome_fasta)

    bw = pyBigWig.open(bigwig_file)

    categories = [
        #"phyloP_negative", "phyloP_neutral", "phyloP_positive", 
        "UCNE", "repeats", "exons", "introns", "noncoding_regions", "coding_regions", "upstream_TSS", "UTR5", "UTR3", "vista_enhancer", "promoters"]
    #categories = ["vista_enhancer"]

    chromosome_groups = {}
    if training_chromosomes and test_chromosomes:
        chromosome_groups = {
            "training": training_chromosomes,
            "test": test_chromosomes
        }
    else:
        chromosome_groups = {"all": chromosomes}

    all_group_embeddings = {
        group_name: {"roi": [], "full": [], "roi_labels": [], "full_labels": []}
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
                if model_type == "caduceus_small":
                    model_name = "caduceus"
                else:
                    model_name = model_type
                context = extract_context(bigwig_file, region, genome, model_type=model_name)
                if not context or "sequence" not in context:
                    print(f"[WARN] Region {i} has invalid or truncated sequence")
                    continue
                valid_regions.append(context)

            if not valid_regions:
                logging.warning(f"[SKIP] All regions in group {group_name} were invalid.")
                continue
            sequence_representations, region_info = predict_scores_batched(
                model, tokenizer, valid_regions, batch_size=batch_size, device=device, model_type=model_type
            )

            for r in region_info:
                r["category"] = category
            print(f"[DEBUG] Received {len(sequence_representations)} representations")
            print(f"[DEBUG] Received {len(region_info)} region metadata entries")


            # Extract embeddings
            if model_type == "phyloGPN":
                full_embeds = sequence_representations  # Already in correct shape
                full_labels = [info["category"] for info in region_info]
                roi_embeds = sequence_representations  # Same for ROI in this case
                roi_labels = full_labels  # Same labels for ROI
            else:
                full_embeds, full_labels = extract_region_embeddings(sequence_representations, region_info, mode="full", model_type =model_type)
                roi_embeds, roi_labels = extract_region_embeddings(sequence_representations, region_info, mode="roi", model_type =model_type)

            all_group_embeddings[group_name]["roi"].extend(roi_embeds)
            all_group_embeddings[group_name]["full"].extend(full_embeds)
            all_group_embeddings[group_name]["full_labels"].extend(full_labels)
            all_group_embeddings[group_name]["roi_labels"].extend(roi_labels)

    for group_name, group_data in all_group_embeddings.items():
        plot_umap(group_data["roi"], group_data["roi_labels"], output_dir / f"global_umap_roi_{group_name}.png", title=f"Global UMAP - ROI ({group_name})")
        plot_knn_heatmap(group_data["roi"], group_data["roi_labels"], output_dir / f"global_knn_roi_{group_name}.png", title=f"1-NN Accuracy - ROI ({group_name})")
        plot_umap(group_data["full"], group_data["full_labels"], output_dir / f"global_umap_full_{group_name}.png", title=f"Global UMAP - Full ({group_name})")
        plot_knn_heatmap(group_data["full"], group_data["full_labels"], output_dir / f"global_knn_full_{group_name}.png", title=f"1-NN Accuracy - Full ({group_name})")

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
        default="/home/mica/caduceus/global_representations",
        help="Directory to save analysis results",
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
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for model predictions",
    )
    parser.add_argument(
        "--model_type", type=str, choices=["hyenaDNA", "phyloGPN", "nt-ms", "nt-human", "caduceus", "caduceus_small"], required=True,
        help="Which model type to use (gamba or caduceus)"
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
    
    #change outputdir to + dcp checkpoint 
    output_dir = args.output_dir + f"/{args.model_type}/"
    try:
        analyze_agreement(
            args.genome_fasta,
            args.bigwig_file,
            output_dir,
            num_regions=args.num_regions,
            chromosomes=args.chromosomes,
            training_chromosomes=args.training_chromosomes,
            test_chromosomes=None, #No test since I dont know whats held out
            batch_size=args.batch_size,
            model_type=args.model_type
        )
        logging.info("Analysis completed successfully")
    except Exception as e:
        logging.error(f"Error in analysis: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()