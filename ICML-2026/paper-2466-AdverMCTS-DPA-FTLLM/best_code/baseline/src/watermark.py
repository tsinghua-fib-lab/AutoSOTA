#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced Baseline watermark dataset generator (Homoglyphs via src.data.perturb_modified)

Advanced features:
- All users: generate homoglyph baseline
- Only user_0001 ~ user_0005:
    - generate null
    - generate propagation_inputs.csv (official format)
- user_0006+:
    - do not generate null
    - do not generate propagation
    - but homoglyph still exists in the training set

Statistical meaning:
- The null distribution is only used as a representative statistic
- Greatly saves GPU time
- Fully consistent with the official unicode_properties experimental logic

Anonymization notes:
- Removed any hard-coded, user-specific absolute paths (e.g., /home/<user>/...)
- External dependency paths are now provided via CLI or environment variables
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import tempfile
import shutil

# =========================
# project root (relative, safe)
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw")

# Make local imports work (repo-relative)
sys.path.append(PROJECT_ROOT)

# =========================
# CLI
# =========================
parser = argparse.ArgumentParser()

parser.add_argument("--source", choices=["blog1k", "Gutenberg", "poems", "cnn_news"], default="blog1k")
parser.add_argument("--train-size-total", type=int, default=1000)
parser.add_argument("--num-users", type=int, default=50)
parser.add_argument("--wm-per-user", type=int, default=40)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--version", type=str, default="xp_homoglyph_advanced")

# dataset inputs (default: repo-relative)
parser.add_argument("--blog1k-csv", type=str, default=os.path.join(DATASET_DIR, "blog1000.csv"))
parser.add_argument("--cnn-news-csv", type=str, default=os.path.join(DATASET_DIR, "cnn_dm_train_article_8k_10k_chars.csv"))
parser.add_argument("--poems-csv", type=str, default=os.path.join(DATASET_DIR, "PoetryFoundationData.csv"))

parser.add_argument("--output-dir-root", type=str, default="./")
parser.add_argument("--max-wm-per-user", type=int, default=200)
parser.add_argument("--max-tokens", type=int, default=3800)

# homoglyph controls
parser.add_argument("--base-seed", type=int, default=123)
parser.add_argument("--group-folder", choices=["sampled_perturbation", "constant_perturbation"], default="sampled_perturbation")
parser.add_argument("--null-n-seq", type=int, default=100)

# OPTIONAL: external dependency root (instead of hard-coded /home/<user>/...)
# Example usage:
#   --external-lib-root /path/to/datawatermarks
# or set env var:
#   export EXTERNAL_LIB_ROOT=/path/to/datawatermarks
parser.add_argument("--external-lib-root", type=str, default=os.environ.get("EXTERNAL_LIB_ROOT", ""))

args = parser.parse_args()

# If provided, add external dependency path safely
if args.external_lib_root:
    sys.path.append(os.path.abspath(args.external_lib_root))

# =========================
# paths
# =========================
output_dir = os.path.join(args.output_dir_root, args.version)
train_out = os.path.join(output_dir, "train.jsonl")
users_dir = os.path.join(output_dir, "users")
prop_dir = os.path.join(output_dir, "propagation_inputs")
os.makedirs(users_dir, exist_ok=True)
os.makedirs(prop_dir, exist_ok=True)

# =========================
# utils
# =========================
def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# =========================
# homoglyph via official perturb_dataset
# =========================
def homoglyph_via_perturb_dataset(texts, seed, group_folder, null_n_seq, uid):
    from src.data.perturb_modified import perturb_dataset

    k = len(texts)
    tmp_dir = tempfile.mkdtemp(prefix="tmp_homo_")
    out_dir = os.path.join(tmp_dir, "out")
    os.makedirs(out_dir, exist_ok=True)

    tmp_in = os.path.join(tmp_dir, "in.jsonl")
    with open(tmp_in, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    perturb_dataset(
        exp_name="unicode_properties",
        group_folder=group_folder,
        raw_dataset=tmp_in,
        out_dir=out_dir,
        seed=seed,
        repetition=k,
        null_n_seq=null_n_seq,
        num_proc=1,
        num_watermarks=1,
        start_range=0,
    )

    out_jsonl = os.path.join(out_dir, f"{k}_dataset.jsonl")
    wm_texts = [json.loads(l)["text"] for l in open(out_jsonl, encoding="utf-8")]

    # Only save propagation for the first 5 users
    pattern_id = int(uid.split("_")[1])
    if pattern_id <= 5 and null_n_seq > 0:
        prop_csv = os.path.join(out_dir, f"{k}_propagation_inputs.csv")
        final_prop = os.path.join(prop_dir, f"{uid}_k{k}_null{null_n_seq}_propagation_inputs.csv")
        shutil.copy(prop_csv, final_prop)
        print(f"[propagation] saved for {uid}")
    else:
        print(f"[propagation] skipped for {uid}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return wm_texts

# =========================
# load data
# =========================
if args.source == "blog1k":
    df = pd.read_csv(args.blog1k_csv)
    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    df = df[df["word_count"] >= 200].copy()
    df = df[["text"]].dropna().copy()

elif args.source == "poems":
    df = pd.read_csv(args.poems_csv)

    df["text"] = (
        df["Title"].fillna("").astype(str)
        + "\n\n"
        + df["Poem"].fillna("").astype(str)
        + "\n\n--- Poet: "
        + df["Poet"].fillna("").astype(str)
        + "\n--- Tags: "
        + df["Tags"].fillna("").astype(str)
    )

    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    df = df[df["word_count"] >= 200].copy()
    df = df[["text"]].dropna().copy()

elif args.source == "cnn_news":
    df = pd.read_csv(args.cnn_news_csv)
    df["text"] = df["article"].astype(str)
    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    df = df[df["word_count"] >= 200].copy()
    df = df[["text"]].dropna().copy()

else:
    raise ValueError(f"Unknown source: {args.source}")

df = df[["text"]].dropna().copy()

rng = np.random.default_rng(args.seed)
eligible_idx = rng.choice(df.index, size=args.train_size_total, replace=False)

# =========================
# assign users
# =========================
df["is_watermarked"] = False
df["user_id"] = ""
df["watermarked"] = df["text"]

all_wm_indices = []

for u in range(args.num_users):
    uid = f"user_{u+1:04d}"
    base = u * args.max_wm_per_user
    indices = eligible_idx[base: base + args.wm_per_user]

    df.loc[indices, "is_watermarked"] = True
    df.loc[indices, "user_id"] = uid

    texts = df.loc[indices, "text"].tolist()
    user_seed = args.base_seed + u

    # Advanced logic: only the first 5 users generate null
    null_n_seq = args.null_n_seq if u < 5 else 0

    print(f"[homoglyph] {uid} | docs={len(texts)} | seed={user_seed} | null_n_seq={null_n_seq}")

    wm_texts = homoglyph_via_perturb_dataset(
        texts=texts,
        seed=user_seed,
        group_folder=args.group_folder,
        null_n_seq=null_n_seq,
        uid=uid
    )

    df.loc[indices, "watermarked"] = wm_texts
    all_wm_indices.extend(indices)

    # user manifest (no machine-specific paths)
    write_jsonl(
        os.path.join(users_dir, f"{uid}_watermarks.jsonl"),
        [{
            "user_id": uid,
            "row_indices": indices.tolist(),
            "homoglyph_seed": int(user_seed),
            "null_n_seq": int(null_n_seq)
        }]
    )

# =========================
# build train
# =========================
all_wm_indices = np.array(all_wm_indices)
non_wm_needed = args.train_size_total - len(all_wm_indices)
non_wm_pool = np.setdiff1d(eligible_idx, all_wm_indices)
non_wm_indices = rng.choice(non_wm_pool, size=non_wm_needed, replace=False)

train_indices = np.concatenate([all_wm_indices, non_wm_indices])
train_df = df.loc[train_indices].copy()

os.makedirs(output_dir, exist_ok=True)
write_jsonl(train_out, train_df.to_dict(orient="records"))

print("\n================ SUMMARY ================")
print(f"Users total: {args.num_users}")
print("Users with null: 5")
print(f"Null per user: {args.null_n_seq}")
print(f"Watermarked docs: {len(all_wm_indices)}")
print(f"Train size: {len(train_df)}")
print(f"Propagation saved in: {prop_dir}")
print(f"Train dataset saved to: {train_out}")
print("========================================")
