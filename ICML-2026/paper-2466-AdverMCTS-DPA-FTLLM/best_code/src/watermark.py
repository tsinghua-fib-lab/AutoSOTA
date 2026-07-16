#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import pandas as pd
import numpy as np
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset/raw")

# =========================
# CLI arguments
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--source", choices=["blog1k", "Gutenberg", "poems", "cnn_news"], default="blog1k")
parser.add_argument("--train-size-total", type=int, default=1000)
parser.add_argument("--num-users", type=int, default=1)
parser.add_argument("--wm-per-user", type=int, default=50)
parser.add_argument("--test-wm-ratio", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--version", type=str, default="xp2-2K")
parser.add_argument("--blog1k-csv", type=str, default=os.path.join(DATASET_DIR, "blog1000.csv"))
parser.add_argument("--gutenberg-csv", type=str, default=os.path.join(DATASET_DIR, "Gutenberg.csv"))
parser.add_argument("--poems-csv", type=str, default=os.path.join(DATASET_DIR, "PoetryFoundationData.csv"))
parser.add_argument(
    "--cnn-news-csv",
    type=str,
    default=os.path.join(DATASET_DIR, "cnn_dm_3.0.0_train_article_8k_10k_chars.csv"),
)
parser.add_argument("--output-dir-root", type=str, default="./")
parser.add_argument(
    "--max-wm-per-user",
    type=int,
    default=200,
    help="Max number of watermarks per user used for slicing (to guarantee inclusion property).",
)
parser.add_argument(
    "--max-tokens",
    type=int,
    default=3800,
    help="Filter out overly long texts (0 means no limit).",
)
parser.add_argument(
    "--wm-offset",
    type=int,
    default=0,
    help="Global integer offset added to each per-user watermark RNG seed. "
         "Does NOT change dataset sampling; only changes watermark strings. Example: 1 means +1.",
)
args = parser.parse_args()
WM_OFFSET = int(args.wm_offset)
# =========================
# Path configuration
# =========================
output_dir = f"{args.output_dir_root}/{args.version}"
train_out = f"{output_dir}/train.jsonl"
test_out = f"{output_dir}/test.jsonl"
users_dir = f"{output_dir}/users"
users_manifest_out = f"{output_dir}/users_manifest.json"

prompts_out = f"{output_dir}/cnn_news_prompts.jsonl"

NUM_USERS = args.num_users
WM_PER_USER = args.wm_per_user
TRAIN_SIZE_TOTAL = args.train_size_total

NUM_NON_WM = TRAIN_SIZE_TOTAL - NUM_USERS * WM_PER_USER
if NUM_NON_WM < 0:
    raise ValueError(
        f"TRAIN_SIZE_TOTAL={TRAIN_SIZE_TOTAL} is smaller than the total number of watermarked samples "
        f"{NUM_USERS * WM_PER_USER}"
    )

TEST_WM_RATIO = args.test_wm_ratio
RANDOM_SEED = args.seed
sample_rng = np.random.default_rng(RANDOM_SEED)  # used only for sampling the global pool


def load_alphabet_from_txt(path: str) -> list[str]:
    """
    Parse lines like:
      U+E0020  ''  width=1  TAG SPACE  Cf
    Extract the leading U+XXXX / U+YYYYY codepoints and convert them to characters.
    """
    alphabet = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Match a leading U+XXXX or U+YYYYY
            m = re.match(r"^U\+([0-9A-Fa-f]+)", line)
            if not m:
                continue

            hex_code = m.group(1)  # e.g., "E0020"
            code_point = int(hex_code, 16)

            try:
                ch = chr(code_point)  # convert to actual character
            except ValueError:
                # Skip invalid code points
                continue

            alphabet.append(ch)

    # Deduplicate while preserving order
    seen = set()
    dedup = []
    for ch in alphabet:
        if ch not in seen:
            seen.add(ch)
            dedup.append(ch)

    return dedup


# Assume the txt file is in the same folder as this script, e.g. "alphabet.txt"
ALPHABET_FILE = os.path.join(os.path.dirname(__file__), "alphabet.txt")
ZWS_ALPHABET = load_alphabet_from_txt(ALPHABET_FILE)

print(f"✔ Loaded {len(ZWS_ALPHABET)} ZW characters from {ALPHABET_FILE}")
print("  First few code points:", [f"U+{ord(c):04X}" for c in ZWS_ALPHABET[:8]])


# =========================
# Zero-width alphabet & insertion params
# =========================
# ZWS_ALPHABET = ["\u2060", "\u2061", "\u2062", "\u2063"]
# ZWS_ALPHABET = ["\u200b", "\u200c", "\u200d", "\u2060"]
# U+2060, U+2061, U+2062, U+2063, U+2064

CLUSTER_SIZE = 4
SPACING = 4
HALF_SYMBOLS = 16
PAIR_SYMBOLS = 32


def gen_random_zws(n: int, rng: np.random.Generator) -> str:
    idx = rng.integers(0, len(ZWS_ALPHABET), size=n)
    return "".join(ZWS_ALPHABET[i] for i in idx)


def gen_random_pair(rng: np.random.Generator) -> tuple[str, str]:
    a = gen_random_zws(HALF_SYMBOLS, rng)
    while True:
        b = gen_random_zws(HALF_SYMBOLS, rng)
        if b != a:
            return a, b


def _group_into_clusters(zws: str, cluster_size: int = CLUSTER_SIZE) -> list[str]:
    m = len(zws) - (len(zws) % cluster_size)
    return ["".join(zws[i:i + cluster_size]) for i in range(0, m, cluster_size)]


def embed_watermark_in_text(
    text: str,
    watermark_zws: str,
    cluster_size: int = CLUSTER_SIZE,
    spacing: int = SPACING
) -> str:
    clusters = _group_into_clusters(watermark_zws, cluster_size)
    if not clusters:
        return text

    words = text.split(" ")
    out = []
    cluster_idx = 0
    total_inserted = 0

    for i, w in enumerate(words):
        out.append(w)
        if i < len(words) - 1:
            out.append(" ")
            if i % spacing == 0:
                c = clusters[cluster_idx % len(clusters)]
                out.append(c)
                cluster_idx += 1
                total_inserted += 1

    remainder = total_inserted % len(clusters)
    if remainder != 0:
        need = len(clusters) - remainder
        for k in range(need):
            out.append(clusters[(cluster_idx + k) % len(clusters)])

    return "".join(out)


def split_text_in_half_words(text: str) -> tuple[str, str]:
    words = text.split()
    mid = len(words) // 2
    return " ".join(words[:mid]), " ".join(words[mid:])


def apply_dual_watermark(text: str, pair: tuple[str, str]) -> str:
    a, b = split_text_in_half_words(text)
    return embed_watermark_in_text(a, pair[0]) + " " + embed_watermark_in_text(b, pair[1])


def _split_into_word_chunks_preserve_ws(text: str, chunk_words: int = 400) -> list[str]:
    """
    Chunk by word count while trying to preserve the original whitespace (spaces/newlines/multiple spaces).
    Approach: match `non-whitespace token + trailing whitespace` as a "word unit".
    Group units by chunk_words and join them back into chunks.
    """
    text = str(text)
    units = list(re.finditer(r"\S+\s*", text))
    if not units:
        return [text]

    chunks = []
    cur = []
    cnt = 0
    for m in units:
        cur.append(m.group(0))
        cnt += 1
        if cnt >= chunk_words:
            chunks.append("".join(cur))
            cur = []
            cnt = 0
    if cur:
        chunks.append("".join(cur))
    return chunks


def _group_into_word_halves_preserve_ws(text: str) -> tuple[str, str]:
    """
    Split text into two halves by word count without breaking original whitespace structure
    (the cut point is at a word-unit boundary).
    """
    text = str(text)
    units = list(re.finditer(r"\S+\s*", text))
    if not units:
        return text, ""

    mid = len(units) // 2
    if mid == 0:
        cut = 0
    else:
        cut = units[mid - 1].end()

    return text[:cut], text[cut:]


def embed_watermark_in_text_preserve_ws(
    text: str,
    watermark_zws: str,
    cluster_size: int = CLUSTER_SIZE,
    spacing: int = SPACING,
) -> str:
    """
    Similar to embed_watermark_in_text, but tries to preserve original whitespace (including newlines/multiple spaces).
    Procedure:
      - Preserve leading whitespace
      - Iterate over `word + trailing whitespace` units
      - Insert clusters at the specified spacing without rewriting existing whitespace
    """
    text = str(text)
    clusters = _group_into_clusters(watermark_zws, cluster_size)
    if not clusters:
        return text

    m0 = re.match(r"^\s*", text)
    leading_ws = m0.group(0) if m0 else ""
    rest = text[len(leading_ws):]

    units = list(re.finditer(r"\S+\s*", rest))
    if not units:
        return text  # all whitespace

    out = [leading_ws]
    cluster_idx = 0
    total_inserted = 0

    for i, m in enumerate(units):
        out.append(m.group(0))
        if i < len(units) - 1 and (i % spacing == 0):
            c = clusters[cluster_idx % len(clusters)]
            out.append(c)
            cluster_idx += 1
            total_inserted += 1

    # Pad cluster cycle so that total_inserted is a multiple of len(clusters) (keep original semantics)
    remainder = total_inserted % len(clusters)
    if remainder != 0:
        need = len(clusters) - remainder
        for k in range(need):
            out.append(clusters[(cluster_idx + k) % len(clusters)])

    return "".join(out)


def apply_dual_watermark_preserve_ws(text: str, pair: tuple[str, str]) -> str:
    """
    Dual-watermark version (first half uses wm_first, second half uses wm_second) while preserving whitespace.
    """
    a, b = _group_into_word_halves_preserve_ws(text)
    return embed_watermark_in_text_preserve_ws(a, pair[0]) + embed_watermark_in_text_preserve_ws(b, pair[1])


def apply_dual_watermark_chunked_400_words(text: str, pair: tuple[str, str], chunk_words: int = 400) -> str:
    """
    cnn_news requirement:
      1) split into 400-word chunks
      2) watermark each chunk
      3) concatenate back in the original order (preserving whitespace as much as possible)
    """
    chunks = _split_into_word_chunks_preserve_ws(text, chunk_words=chunk_words)
    return "".join(apply_dual_watermark_preserve_ws(ch, pair) for ch in chunks)


def write_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# Fixed-length random slicing (~1000 characters) for Gutenberg texts
TARGET_CHARS = 1000


def random_char_slice(text: str, target_len: int = TARGET_CHARS) -> str:
    text = str(text)
    if len(text) <= target_len:
        return text
    import random
    start = random.randint(0, len(text) - target_len)
    return text[start: start + target_len]


# =========================
# New: inclusion-preserving watermark allocation strategy
# =========================
def get_deterministic_user_watermark_indices(
    total_users: int,
    target_user_id: int,
    wm_per_user: int,
    max_wm_per_user: int,
    all_eligible_idx: np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Assign watermark indices to a user with inclusion guarantees.

    Property 1: When total_users=3, user_0's watermarks include those from total_users=2 for user_0.
    Property 2: When wm_per_user=10, it includes the indices from wm_per_user=5.
    Property 3: Across all scenarios, the watermark text assignment for the same user ID is identical.

    Args:
        total_users: total number of users in the current scenario
        target_user_id: target user id (0-indexed)
        wm_per_user: number of watermarked texts for this user in the current scenario
        max_wm_per_user: maximum watermarks per user (reserved capacity)
        all_eligible_idx: all eligible text indices
        rng: RNG (kept for API consistency; not required by the slicing logic)

    Returns:
        Numpy array of selected text indices for this user.
    """
    # Base offset per user (fixed order; independent of total_users)
    base_offset = target_user_id * max_wm_per_user

    # Ensure sufficient texts
    required_total = total_users * max_wm_per_user
    if len(all_eligible_idx) < required_total:
        raise ValueError(f"Need at least {required_total} samples, but only have {len(all_eligible_idx)}.")

    # Take this user's dedicated block
    user_block_indices = all_eligible_idx[base_offset: base_offset + max_wm_per_user]

    # Return the first wm_per_user indices (inclusion property)
    return user_block_indices[:wm_per_user]


def generate_user_watermarks_deterministic(
    num_users: int,
    wm_per_user: int,
    max_wm_per_user: int,
    all_train_idx: np.ndarray,
    rng: np.random.Generator
) -> dict[str, dict]:
    """
    Generate inclusion-preserving user watermark assignments, and strictly enforce that:
    - For any users i, j (same or different):
      wm_first_i, wm_second_i, wm_first_j, wm_second_j are all pairwise distinct.
      In other words, all halves are globally unique across users and do not collide.

    Original properties are preserved:
    - Each user's watermark halves are deterministic (seeded by RANDOM_SEED + uid_int)
    - Text index assignment uses the inclusion strategy (get_deterministic_user_watermark_indices)
    """
    user_assignments: dict[str, dict] = {}

    # Global set for all used watermark halves (no distinction between first/second)
    used_halves: set[str] = set()

    for uid_int in range(num_users):
        uid_str = f"user_{uid_int + 1:04d}"

        # Deterministic per-user RNG (seed is offsettable without changing --seed)
        # Base offset +4 keeps original behavior; WM_OFFSET shifts all users consistently.
        user_seed = RANDOM_SEED + uid_int + 4 + WM_OFFSET
        user_rng = np.random.default_rng(user_seed)

        # Generate first half: must be globally unique
        while True:
            wm_first = gen_random_zws(HALF_SYMBOLS, user_rng)
            if wm_first not in used_halves:
                break

        # Generate second half: must differ from first and be globally unique
        while True:
            wm_second = gen_random_zws(HALF_SYMBOLS, user_rng)
            if wm_second != wm_first and wm_second not in used_halves:
                break

        used_halves.add(wm_first)
        used_halves.add(wm_second)

        # Inclusion-preserving index assignment
        user_indices = get_deterministic_user_watermark_indices(
            total_users=num_users,
            target_user_id=uid_int,
            wm_per_user=wm_per_user,
            max_wm_per_user=max_wm_per_user,
            all_eligible_idx=all_train_idx,
            rng=rng
        )

        user_assignments[uid_str] = {
            "indices": user_indices,
            "wm_first": wm_first,
            "wm_second": wm_second,
            "row_indices": [int(i) for i in user_indices],
        }

    return user_assignments


# =========================
# Load data & capacity filtering
# =========================
if args.source == "blog1k":
    print(">>> Using blog1k data:", args.blog1k_csv)
    df = pd.read_csv(args.blog1k_csv)
    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    df = df[df["word_count"] >= 200].copy()
    print(f"[OK] {len(df)} samples satisfy length >= 200 words")
    df = df[["text"]].dropna().copy()
    df["num_spaces"] = df["text"].str.count(" ")

elif args.source == "Gutenberg":
    print(">>> Using Gutenberg CSV data:", args.gutenberg_csv)
    df = pd.read_csv(args.gutenberg_csv)
    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    df = df[df["word_count"] >= 200].copy()
    print(f"[OK] {len(df)} samples satisfy length >= 200 words")
    df = df[["text"]].dropna().copy()
    df["text"] = df["text"].apply(lambda x: random_char_slice(x, TARGET_CHARS))
    df["num_spaces"] = df["text"].str.count(" ")

elif args.source == "poems":
    print(">>> Using Kaggle poems dataset:", args.poems_csv)
    df = pd.read_csv(args.poems_csv)

    df["text"] = (
        df["Title"].fillna("")
        + "\n\n"
        + df["Poem"].fillna("")
        + "\n\n--- Poet: "
        + df["Poet"].fillna("")
        + "\n--- Tags: "
        + df["Tags"].fillna("")
    )

    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    df = df[df["word_count"] >= 200].copy()
    print(f"[OK] {len(df)} samples satisfy length >= 200 words")

    df = df[["text"]].dropna().copy()
    df["num_spaces"] = df["text"].str.count(" ")

elif args.source == "cnn_news":
    print(">>> Using cnn_news data:", args.cnn_news_csv)
    df = pd.read_csv(args.cnn_news_csv)

    if "article" not in df.columns:
        raise ValueError(f"cnn_news CSV is missing the 'article' column. Actual columns: {list(df.columns)}")

    df["text"] = df["article"].astype(str)

    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    df = df[df["word_count"] >= 200].copy()
    print(f"[OK] {len(df)} samples satisfy length >= 200 words")

    df = df[["text"]].dropna().copy()
    df["num_spaces"] = df["text"].str.count(" ")


# =========================
# Optional: filter overly long texts by token count
# =========================
if args.max_tokens > 0:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    df["num_tokens"] = df["text"].apply(lambda x: len(tokenizer.encode(x, add_special_tokens=False)))
    before = len(df)
    df = df[df["num_tokens"] <= args.max_tokens].copy()
    after = len(df)
    print(f"[OK] Token-length filter: max_tokens={args.max_tokens}, kept {after}/{before} (dropped {before - after})")
else:
    print("[WARN] Token-length filtering is disabled (--max-tokens=0)")


# =========================
# cnn_news: prompt construction
# =========================
def get_first_half_prompt(text: str) -> str:
    words = text.split()
    split_idx = len(words) // 2 + 4
    return " ".join(words[:split_idx])


def watermark_and_build_chunk_prompts_cnn(
    text: str,
    pair: tuple[str, str],
    chunk_words: int = 400,
) -> tuple[str, list[str]]:
    """
    Returns:
      - The full watermarked article (re-concatenated)
      - prompts: one prompt per 400-word chunk (using the same get_first_half rule as the probe)
    """
    chunks = _split_into_word_chunks_preserve_ws(text, chunk_words=chunk_words)

    watermarked_chunks = []
    prompts = []

    for ch in chunks:
        ch_wm = apply_dual_watermark_preserve_ws(ch, pair)  # watermark each chunk independently
        watermarked_chunks.append(ch_wm)

        # Build prompt following the probe format (note: it flattens whitespace, matching probe behavior)
        prompts.append(get_first_half_prompt(ch_wm))

    return "".join(watermarked_chunks), prompts


# =========================
# Select TRAIN / TEST
# =========================
TOTAL_CLUSTERS = (PAIR_SYMBOLS // CLUSTER_SIZE)
min_spaces = TOTAL_CLUSTERS * SPACING

required_train = TRAIN_SIZE_TOTAL
eligible_idx = df.index[df["num_spaces"] >= min_spaces]

if len(df) < required_train:
    raise ValueError(f"Dataset rows={len(df)} < required TRAIN_SIZE_TOTAL={required_train}")

# Global pool (fixed order for reproducibility)
all_train_idx = sample_rng.choice(eligible_idx, size=TRAIN_SIZE_TOTAL, replace=False)

# =========================
# Use inclusion-preserving watermark allocation strategy
# =========================
print("\n=== Using inclusion-preserving watermark allocation strategy ===")
print(f"Total users: {NUM_USERS}, watermarks per user: {WM_PER_USER}")
print(f"Max watermarks per user (reserved): {args.max_wm_per_user}")

# Capacity check
required_capacity = NUM_USERS * args.max_wm_per_user
if len(eligible_idx) < required_capacity:
    available_users = len(eligible_idx) // args.max_wm_per_user
    raise ValueError(
        f"Insufficient capacity: need {required_capacity} samples, but only have {len(eligible_idx)}.\n"
        f"At most {available_users} users are supported (max {args.max_wm_per_user} watermarks per user)."
    )

# Ensure eligible_idx has enough samples for the training set
if len(eligible_idx) < TRAIN_SIZE_TOTAL:
    raise ValueError(
        f"Insufficient eligible samples for training: need {TRAIN_SIZE_TOTAL}, "
        f"but only have {len(eligible_idx)} eligible samples."
    )

# Generate user assignments (use the first required_capacity eligible indices)
user_assignments = generate_user_watermarks_deterministic(
    num_users=NUM_USERS,
    wm_per_user=WM_PER_USER,
    max_wm_per_user=args.max_wm_per_user,
    all_train_idx=eligible_idx[:required_capacity],
    rng=sample_rng
)

# =========================
# Initialize
# =========================
df["is_watermarked"] = False
df["user_id"] = ""
df["wm_first"] = ""
df["wm_second"] = ""
df["watermarked"] = df["text"]

cnn_prompt_records = []

# =========================
# Apply watermarks
# =========================
all_wm_indices = []
for uid, assignment in user_assignments.items():
    indices = assignment["indices"]
    wm_first = assignment["wm_first"]
    wm_second = assignment["wm_second"]

    # Mark and apply watermarks
    df.loc[indices, "user_id"] = uid
    df.loc[indices, "wm_first"] = wm_first
    df.loc[indices, "wm_second"] = wm_second
    df.loc[indices, "is_watermarked"] = True

    texts = df.loc[indices, "text"].tolist()

    if args.source == "cnn_news":
        wm_texts = []
        for local_k, t in enumerate(texts):
            wm_text, prompts = watermark_and_build_chunk_prompts_cnn(
                t, (wm_first, wm_second), chunk_words=400
            )
            wm_texts.append(wm_text)

            # Record: one prompt per chunk
            row_id = int(df.loc[indices[local_k]].name)
            for chunk_id, p in enumerate(prompts):
                cnn_prompt_records.append({
                    "row_id": row_id,
                    "user_id": uid,
                    "chunk_id": chunk_id,
                    "prompt": p,
                    "wm_first": wm_first,
                    "wm_second": wm_second,
                })
    else:
        wm_texts = [apply_dual_watermark(t, (wm_first, wm_second)) for t in texts]

    df.loc[indices, "watermarked"] = wm_texts

    all_wm_indices.extend(indices)

    # Save per-user file
    user_file = os.path.join(users_dir, f"{uid}_watermarks.jsonl")
    write_jsonl(user_file, [{
        "user_id": uid,
        "wm_first": wm_first,
        "wm_second": wm_second,
        "row_indices": assignment["row_indices"],
    }])

# =========================
# Build TRAIN
# =========================
all_wm_indices = []
for assignment in user_assignments.values():
    all_wm_indices.extend(assignment["indices"])
all_wm_indices = np.array(all_wm_indices)

# Select non-watermarked samples from eligible_idx, ensuring no overlap with watermarked samples
eligible_for_non_wm = np.setdiff1d(eligible_idx, all_wm_indices)

if len(eligible_for_non_wm) < NUM_NON_WM:
    raise ValueError(
        f"Not enough non-watermarked samples: need {NUM_NON_WM}, but only have {len(eligible_for_non_wm)}.\n"
        f"Consider reducing train size or number of watermarks."
    )

# Randomly select non-watermarked samples
non_wm_indices = sample_rng.choice(eligible_for_non_wm, size=NUM_NON_WM, replace=False)

# Build final train indices
train_indices = np.concatenate([all_wm_indices, non_wm_indices])

# Re-sort for consistency (optional)
train_indices = np.sort(train_indices)

train_df = df.loc[train_indices].copy()
train_df["is_member"] = True

print("=== Allocation summary ===")
print(f"Watermarked samples: {len(all_wm_indices)}")
print(f"Non-watermarked samples: {len(non_wm_indices)}")
print(f"Total training set: {len(train_indices)} (target: {TRAIN_SIZE_TOTAL})")

if len(train_indices) != TRAIN_SIZE_TOTAL:
    print(f"[WARN] Training set size mismatch! Expected: {TRAIN_SIZE_TOTAL}, actual: {len(train_indices)}")

# =========================
# Save
# =========================
for d in (train_df,):
    if "num_spaces" in d.columns:
        d.drop(columns=["num_spaces"], inplace=True)
    if "num_tokens" in d.columns:
        d.drop(columns=["num_tokens"], inplace=True)

os.makedirs(output_dir, exist_ok=True)
with open(train_out, "w", encoding="utf-8") as f:
    for rec in train_df.to_dict(orient="records"):
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if args.source == "cnn_news":
    write_jsonl(prompts_out, cnn_prompt_records)
    print(f"[OK] cnn_news prompts saved to: {prompts_out} (num_prompts={len(cnn_prompt_records)})")

with open(users_manifest_out, "w", encoding="utf-8") as f:
    json.dump({
        "num_users": NUM_USERS,
        "wm_per_user": WM_PER_USER,
        "train_size_total": TRAIN_SIZE_TOTAL,
        "num_nonwm": NUM_NON_WM,
        "train_rows": int(len(train_df)),
        "min_spaces": int(min_spaces),
        "source": args.source,
        "max_wm_per_user": args.max_wm_per_user,
        "random_seed": RANDOM_SEED,
        "users": [
            {
                "user_id": uid,
                "file": os.path.join(users_dir, f"{uid}_watermarks.jsonl"),
                "count": len(assignment["indices"])
            }
            for uid, assignment in user_assignments.items()
        ]
    }, f, ensure_ascii=False, indent=2)

print("\n=== Inclusion-preserving allocation report ===")
for uid, assignment in user_assignments.items():
    print(f"{uid}: {len(assignment['indices'])} watermarked texts")
print(
    f"Training set total: {len(train_df)} | "
    f"watermarked: {int(train_df['is_watermarked'].sum())} | "
    f"non-watermarked: {int((~train_df['is_watermarked']).sum())}"
)

print(f"\n[OK] Training set saved to: {train_out}")
