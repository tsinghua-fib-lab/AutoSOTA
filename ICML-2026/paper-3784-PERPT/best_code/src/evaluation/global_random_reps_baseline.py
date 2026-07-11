#!/usr/bin/env python3
import argparse
import os
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pyBigWig
from pyfaidx import Fasta
from tqdm import tqdm

import umap
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
)

import sys
sys.path.append("../gamba")
sys.path.append("/home/mica/gamba/")

from src.evaluation.utils.helpers import extract_context  # <- only helper we need

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

CATEGORY_ORDER = [
    "vista_enhancer", "UCNE", "repeats", "exons", "introns",
    "noncoding_regions", "coding_regions", "upstream_TSS",
    "UTR5", "UTR3", "promoters",
]

# -------------------------------------------------------------------
# k-mer + phyloP embedding helpers (mostly copied from your script)
# -------------------------------------------------------------------

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
        elif mode == "full":
            fs = 0
            fe = len(seq)
        else:
            raise ValueError(f"unsupported mode: {mode}")

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
        elif mode == "full":
            fs = 0
            fe = len(scores)
            ss = scores
        else:
            raise ValueError(f"unsupported mode: {mode}")

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


# -------------------------------------------------------------------
# basic 1-NN helpers (optional, for debugging)
# -------------------------------------------------------------------

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


def save_reps(output_dir, model_id, group_name, category, scope, X, labels, metas, extra=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels)

    tag = f"{group_name}_{category}_{scope}"

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

    if extra:
        for k, v in extra.items():
            mdf[k] = v

    mdf.to_parquet(out / f"reps_{model_id}_{tag}_meta.parquet", index=False)


# -------------------------------------------------------------------
# loading random regions + feature embeddings
# -------------------------------------------------------------------

def load_random_regions_for_category(
    category: str,
    random_root: Path,
    genome: Fasta,
    bw: pyBigWig.pyBigWig,
    ctx_model_type: str = "baseline",
):
    """
    load random_<category>.bed and build contexts with extract_context.
    we treat the whole window as ROI (feature_start=0, feature_end=len(seq)).
    """
    bed_path = random_root / category / f"random_{category}.bed"
    if not bed_path.exists():
        logging.warning(f"[random] missing BED for {category}: {bed_path}")
        return []

    regions = []
    with bed_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            r = {"chrom": chrom, "start": start, "end": end}
            ctx = extract_context(
                bw,
                r,
                genome,
                model_type=ctx_model_type,
            )
            if not ctx or "sequence" not in ctx:
                continue

            seq_len = len(ctx["sequence"])
            ctx["feature_start_in_window"] = 0
            ctx["feature_end_in_window"] = seq_len
            ctx["category"] = category
            ctx["class_label"] = "random"
            regions.append(ctx)

    logging.info(f"[random] {category}: built {len(regions)} random contexts")
    return regions


def load_feature_embeddings_for_baseline(
    baseline: str,
    category: str,
    scope: str,
    feature_root: Path,
    group_name: str = "all",
):
    """
    load existing baseline embeddings for real features (label == 'roi')
    from upstream-pair pipeline:
      feature_root/baseline/<group_name>/<category>/
        reps_<baseline>_<group_name>_<category>_<scope>.npz
        reps_<baseline>_<group_name>_<category>_<scope>_meta.parquet
    """
    cat_dir = feature_root / baseline / group_name / category
    npz_path = cat_dir / f"reps_{baseline}_{group_name}_{category}_{scope}.npz"
    meta_path = cat_dir / f"reps_{baseline}_{group_name}_{category}_{scope}_meta.parquet"

    if not npz_path.exists() or not meta_path.exists():
        logging.warning(f"[feature] missing npz/meta for {baseline} {category} {scope}")
        return None, None, None

    data = np.load(npz_path, allow_pickle=True)
    X = np.asarray(data["embeddings"])
    meta = pd.read_parquet(meta_path)

    if "label" not in meta.columns:
        logging.warning(f"[feature] meta has no 'label'; using all rows")
        mask = np.ones(len(meta), dtype=bool)
    else:
        # keep only true features (roi)
        mask = (meta["label"].astype(str) == "roi").values

    X_feat = X[mask]
    meta_feat = meta.loc[mask].copy()

    # we relabel these as "feature" for the new random-vs-feature task
    y_feat = np.array(["feature"] * X_feat.shape[0], dtype=object)

    # force class_label in meta for compatibility
    meta_feat["class_label"] = "feature"

    logging.info(
        f"[feature] {baseline} {category} {scope}: "
        f"{X_feat.shape[0]} feature embeddings"
    )
    return X_feat, y_feat, meta_feat.to_dict(orient="records")


# -------------------------------------------------------------------
# main routine: build random-vs-feature baseline reps
# -------------------------------------------------------------------

def build_random_vs_category_baseline(
    baseline: str,
    random_root: Path,
    feature_root: Path,
    output_root: Path,
    genome_fasta: str,
    bigwig_file: str,
    scopes=("roi", "full"),
    group_name: str = "all",
    max_random: int = None,
):
    assert baseline in ("kmer6", "phylop")

    genome = Fasta(genome_fasta)
    bw = pyBigWig.open(bigwig_file)

    out_root_model = output_root / baseline / group_name

    for category in CATEGORY_ORDER:
        # 1) random regions → contexts
        random_regions = load_random_regions_for_category(
            category,
            random_root,
            genome,
            bigwig_file,
            ctx_model_type="baseline",
        )
        if not random_regions:
            continue

        if max_random is not None:
            random_regions = random_regions[:max_random]

        # 2) random embeddings per scope
        if baseline == "kmer6":
            X_rand_roi, y_rand_roi, meta_rand_roi = compute_kmer_embeddings(
                random_regions, mode="roi", k=6
            )
            X_rand_full, y_rand_full, meta_rand_full = compute_kmer_embeddings(
                random_regions, mode="full", k=6
            )
        else:  # phylop
            X_rand_roi, y_rand_roi, meta_rand_roi = compute_phylop_embeddings(
                random_regions, mode="roi"
            )
            X_rand_full, y_rand_full, meta_rand_full = compute_phylop_embeddings(
                random_regions, mode="full"
            )

        # 3) feature embeddings from existing upstream baseline
        X_feat_roi, y_feat_roi, meta_feat_roi = load_feature_embeddings_for_baseline(
            baseline,
            category,
            scope="roi",
            feature_root=feature_root,
            group_name=group_name,
        )
        X_feat_full, y_feat_full, meta_feat_full = load_feature_embeddings_for_baseline(
            baseline,
            category,
            scope="full",
            feature_root=feature_root,
            group_name=group_name,
        )

        # if any of these is missing, skip that scope
        scope_data = {
            "roi": (X_feat_roi, y_feat_roi, meta_feat_roi, X_rand_roi, y_rand_roi, meta_rand_roi),
            "full": (X_feat_full, y_feat_full, meta_feat_full, X_rand_full, y_rand_full, meta_rand_full),
        }

        for scope in scopes:
            Xf, yf, mf, Xr, yr, mr = scope_data[scope]
            if Xf is None or Xr is None:
                logging.warning(f"[{baseline}] {category} {scope}: missing feature/random, skipping")
                continue
            if Xf.shape[0] == 0 or Xr.shape[0] == 0:
                logging.warning(f"[{baseline}] {category} {scope}: empty feature/random, skipping")
                continue

            # balance classes
            n = min(Xf.shape[0], Xr.shape[0])
            if n < 5:
                logging.warning(f"[{baseline}] {category} {scope}: too few samples ({n}), skipping")
                continue

            rng = np.random.default_rng(1337)
            idx_f = rng.choice(Xf.shape[0], size=n, replace=False)
            idx_r = rng.choice(Xr.shape[0], size=n, replace=False)

            # make sure labels are numpy arrays for fancy indexing
            yf = np.asarray(yf, dtype=object)
            yr = np.asarray(yr, dtype=object)

            X = np.concatenate([Xf[idx_f], Xr[idx_r]], axis=0)
            y = np.concatenate([yf[idx_f], yr[idx_r]], axis=0)

            # metas are lists of dicts; index them via list comprehension
            metas = [mf[i] for i in idx_f] + [mr[i] for i in idx_r]

            # save in the format expected by your aggregator:
            # output_root/baseline/all/category/reps_<baseline>_all_<category>_<scope>.npz
            cat_outdir = out_root_model / category
            extra = {
                "baseline": baseline,
                "source_feature_root": str(feature_root),
                "source_random_root": str(random_root),
            }
            save_reps(
                output_dir=cat_outdir,
                model_id=baseline,
                group_name=group_name,
                category=category,
                scope=scope,
                X=X,
                labels=y,
                metas=metas,
                extra=extra,
            )

            logging.info(
                f"[{baseline}] {category} {scope}: saved {X.shape[0]} reps "
                f"to {cat_outdir}"
            )


    bw.close()


# -------------------------------------------------------------------
# cli
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="build kmer6 / phylop random-vs-feature baseline reps"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=["kmer6", "phylop", "both"],
        default="both",
        help="which baseline(s) to run",
    )
    parser.add_argument(
        "--random_regions_root",
        type=str,
        default="/home/mica/gamba/data_processing/data/random_regions_matched",
    )
    parser.add_argument(
        "--feature_reps_root",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/global_representations_upstream_pairs/baseline",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/global_random_baseline_representations/baseline",
    )
    parser.add_argument(
        "--bigwig_file",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/241-mammalian-2020v2.bigWig",
    )
    parser.add_argument(
        "--genome_fasta",
        type=str,
        default="/home/mica/gamba/data_processing/data/240-mammalian/hg38.ml.fa",
    )
    parser.add_argument(
        "--max_random",
        type=int,
        default=None,
        help="optional cap on number of random regions per category",
    )
    args = parser.parse_args()

    random_root = Path(args.random_regions_root)
    feature_root = Path(args.feature_reps_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    baselines = []
    if args.baseline == "both":
        baselines = ["kmer6", "phylop"]
    else:
        baselines = [args.baseline]

    for b in baselines:
        logging.info(f"=== building random-vs-feature baseline for {b} ===")
        build_random_vs_category_baseline(
            baseline=b,
            random_root=random_root,
            feature_root=feature_root,
            output_root=output_root,
            genome_fasta=args.genome_fasta,
            bigwig_file=args.bigwig_file,
            scopes=("roi", "full"),
            group_name="all",
            max_random=args.max_random,
        )


if __name__ == "__main__":
    main()
