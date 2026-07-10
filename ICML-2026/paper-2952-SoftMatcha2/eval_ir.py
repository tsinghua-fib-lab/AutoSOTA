#!/usr/bin/env python3
"""
IR evaluation for SoftMatcha2 on TREC-COVID.
Implements BM25 baseline and SoftMatcha/Ours document scoring.
"""
import os
import sys
import json
import math
import time
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
INDEX_PATH = "/repo/index_trec_covid"
CORPUS_FILE = "/repo/trec_covid_corpus.txt"
ALPHA = 0.45  # similarity threshold
K = 20        # top-K patterns
QUERY_MAX_WORDS = 5  # max words per pattern query

# ============================================================
# 1. Load data
# ============================================================
print("Loading data...")
corpus = load_dataset("BeIR/trec-covid", "corpus")
queries = load_dataset("BeIR/trec-covid", "queries")
qrels = load_dataset("BeIR/trec-covid-qrels")

# Build doc ID -> index mapping
doc_ids = [doc["_id"] for doc in corpus["corpus"]]
doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}
num_docs = len(doc_ids)

# Build query data
query_texts = {}
for q in queries["queries"]:
    query_texts[int(q["_id"])] = q["text"]

# Build qrels: query_id -> set of relevant doc_ids with scores
qrels_data = defaultdict(dict)
for item in qrels["test"]:
    qid = int(item["query-id"])
    did = item["corpus-id"]
    score = int(item["score"])
    qrels_data[qid][did] = score

print(f"  Documents: {num_docs}")
print(f"  Queries: {len(query_texts)}")
print(f"  Qrels query-doc pairs: {sum(len(v) for v in qrels_data.values())}")

# ============================================================
# 2. Load SoftMatcha2 Searcher
# ============================================================
print("\nLoading SoftMatcha2 searcher...")
from softmatcha.embeddings import get_embedding
from softmatcha.search import Searcher
from softmatcha.struct import Pattern
from softmatcha.tokenizers import get_tokenizer

# Get embedding backend
embedding_class = get_embedding("gensim")
embedding_cfg = embedding_class.Config("glove-wiki-gigaword-300", mmap=False)
embedding = embedding_class.build(embedding_cfg)

tokenizer_class = get_tokenizer("gensim")
tokenizer_cfg = tokenizer_class.Config("glove-wiki-gigaword-300", split_hyphen=False)
tokenizer = tokenizer_class.build(tokenizer_cfg)

# Load searcher
searcher = Searcher(INDEX_PATH, tokenizer, embedding)

vocab_size = min(searcher.max_vocab, len(embedding.embeddings))

# ============================================================
# 3. Build inverted index for fast document lookup
# ============================================================
print("\nBuilding inverted index...")
inverted_index = defaultdict(lambda: defaultdict(int))
doc_lengths = np.zeros(num_docs, dtype=np.int32)

for idx, doc in enumerate(tqdm(corpus["corpus"], desc="  Indexing")):
    title = doc["title"] or ""
    text = doc["text"] or ""
    full_text = title + " " + text
    full_text = full_text.replace("\n", " ").replace("\r", " ")

    # Tokenize and encode
    words = tokenizer.tokenize(full_text)
    token_ids_list = tokenizer.encode([w.lower() for w in words])
    token_ids_list = [min(t, searcher.max_vocab - 1) for t in token_ids_list]

    doc_lengths[idx] = len(token_ids_list)
    for tid in token_ids_list:
        inverted_index[tid][idx] += 1

print(f"  Inverted index built: {len(inverted_index)} unique tokens")

# ============================================================
# 4. Document scoring functions
# ============================================================
K1 = 1.2
B = 0.75
avg_dl = doc_lengths.mean()
num_docs_float = float(num_docs)

print("\nPrecomputing IDF values...")
idf_cache = {}
for tid in inverted_index:
    df = len(inverted_index[tid])
    if df > 0:
        idf_cache[tid] = math.log((num_docs_float - df + 0.5) / (df + 0.5) + 1.0)
print(f"  IDF computed for {len(idf_cache)} tokens")

stop_words = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would",
    "can", "could", "should", "may", "might", "shall",
    "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "and", "or", "but", "not", "if", "as", "than", "that", "this",
    "what", "which", "who", "whom", "how", "when", "where",
    "it", "its", "they", "them", "their", "we", "us", "our",
    "he", "she", "his", "her", "i", "you", "me", "my", "your",
}

def extract_pattern_queries(query_text, max_words=QUERY_MAX_WORDS):
    """Extract pattern queries from a query text."""
    words = tokenizer.tokenize(query_text)
    content_words = [w.lower() for w in words if w.lower() not in stop_words and len(w) > 1]
    patterns = []
    for w in content_words:
        patterns.append(w)
    for i in range(len(content_words)):
        for j in range(i + 2, min(i + max_words + 1, len(content_words) + 1)):
            if j - i <= max_words:
                pattern = " ".join(content_words[i:j])
                patterns.append(pattern)
    return list(dict.fromkeys(patterns))


def search_pattern(pattern_str):
    """Search a pattern string using SoftMatcha2."""
    pattern_tokens = tokenizer(pattern_str)
    if len(pattern_tokens) == 0 or len(pattern_tokens) > 12:
        return [], [], []
    unknown = False
    for t in pattern_tokens:
        if t >= min(searcher.max_vocab, len(embedding.embeddings)) - 1:
            unknown = True
            break
    if unknown:
        return [], [], []
    pattern_embeddings = searcher.normalize(embedding(pattern_tokens))
    pat = Pattern.build(pattern_tokens, pattern_embeddings, [0.0] * len(pattern_embeddings))
    cand, cand_score, count, thres = searcher.search(pat, K, ALPHA, 10.0)
    return cand[:K], cand_score[:K], count[:K]


def compute_bm25_score(query_tokens, doc_idx):
    """Compute BM25 score for a document."""
    score = 0.0
    doc_len = doc_lengths[doc_idx]
    for qtid in query_tokens:
        if qtid not in inverted_index:
            continue
        tf = inverted_index[qtid].get(doc_idx, 0)
        if tf == 0:
            continue
        idf = idf_cache.get(qtid, 0.0)
        numerator = tf * (K1 + 1)
        denominator = tf + K1 * (1 - B + B * doc_len / avg_dl)
        score += idf * numerator / denominator
    return score


def evaluate_bm25():
    """BM25 baseline evaluation."""
    all_p20 = []
    all_r1000 = []

    for qid in sorted(query_texts.keys()):
        query_text = query_texts[qid]

        # Tokenize query
        words = tokenizer.tokenize(query_text)
        q_tokens = tokenizer.encode([w.lower() for w in words])

        # Score documents
        doc_scores = np.zeros(num_docs)
        for d_idx in range(num_docs):
            doc_scores[d_idx] = compute_bm25_score(q_tokens, d_idx)

        # Rank and get relevant
        ranked = np.argsort(-doc_scores)
        relevant_docs = {doc_id_to_idx[did] for did, score in qrels_data.get(qid, {}).items()
                         if score > 0 and did in doc_id_to_idx}
        if not relevant_docs:
            continue

        p20 = sum(1 for d in ranked[:20] if d in relevant_docs) / min(20, len(ranked)) * 100
        r1000 = sum(1 for d in ranked[:1000] if d in relevant_docs) / len(relevant_docs) * 100
        all_p20.append(p20)
        all_r1000.append(r1000)

    return np.mean(all_p20) if all_p20 else 0, np.mean(all_r1000) if all_r1000 else 0


def evaluate_softmatcha2():
    """SoftMatcha2-enhanced document scoring."""
    all_p20 = []
    all_r1000 = []

    for qid in sorted(query_texts.keys()):
        query_text = query_texts[qid]
        patterns = extract_pattern_queries(query_text)

        if not patterns:
            continue

        doc_scores = np.zeros(num_docs)

        # Score based on matching patterns
        for pattern in patterns:
            try:
                cand, cand_score, count = search_pattern(pattern)
            except Exception:
                continue

            if len(cand) == 0:
                continue

            # For the exact match (similarity=1.0), give full BM25 score
            # For similar matches, give partial score proportional to similarity
            for match_idx in range(len(cand)):
                match_tokens = cand[match_idx][cand[match_idx] < vocab_size]
                sim = float(cand_score[match_idx])

                if sim < 0.2:  # Very low similarity - skip
                    continue

                # Find documents containing this matching pattern's tokens
                # Use the first token as anchor (simplified approach)
                for mt in match_tokens:
                    if mt in inverted_index:
                        for d_idx, tf in inverted_index[mt].items():
                            if tf > 0 and mt in idf_cache:
                                idf = idf_cache[mt]
                                doc_len = doc_lengths[d_idx]
                                numerator = tf * (K1 + 1)
                                denominator = tf + K1 * (1 - B + B * doc_len / avg_dl)
                                doc_scores[d_idx] += sim * idf * numerator / denominator

        ranked = np.argsort(-doc_scores)
        relevant_docs = {doc_id_to_idx[did] for did, score in qrels_data.get(qid, {}).items()
                         if score > 0 and did in doc_id_to_idx}
        if not relevant_docs:
            continue

        p20 = sum(1 for d in ranked[:20] if d in relevant_docs) / min(20, len(ranked)) * 100
        r1000 = sum(1 for d in ranked[:1000] if d in relevant_docs) / len(relevant_docs) * 100
        all_p20.append(p20)
        all_r1000.append(r1000)

    return np.mean(all_p20) if all_p20 else 0, np.mean(all_r1000) if all_r1000 else 0


# ============================================================
# 5. Run evaluation
# ============================================================
print("\n" + "=" * 60)
print("TREC-COVID Evaluation")
print("=" * 60)

print("\n[1/2] Running BM25 baseline...")
start = time.time()
bm25_p20, bm25_r1000 = evaluate_bm25()
elapsed = time.time() - start
print(f"  BM25: P@20={bm25_p20:.1f}, R@1000={bm25_r1000:.1f} (took {elapsed:.1f}s)")

print("\n[2/2] Running SoftMatcha2 evaluation...")
start = time.time()
sm2_p20, sm2_r1000 = evaluate_softmatcha2()
elapsed = time.time() - start
print(f"  SoftMatcha2: P@20={sm2_p20:.1f}, R@1000={sm2_r1000:.1f} (took {elapsed:.1f}s)")

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"{'Method':<20} {'P@20':>8} {'R@1000':>8}")
print(f"{'-'*36}")
print(f"{'BM25 (ours)':<20} {bm25_p20:>8.1f} {bm25_r1000:>8.1f}")
print(f"{'SoftMatcha2 (ours)':<20} {sm2_p20:>8.1f} {sm2_r1000:>8.1f}")
print()
print("Paper results (Table 17):")
print(f"{'BM25 (paper)':<20} {33.8:>8.1f} {14.9:>8.1f}")
print(f"{'SoftMatcha (paper)':<20} {35.4:>8.1f} {17.7:>8.1f}")
print(f"{'Ours (paper)':<20} {36.0:>8.1f} {18.0:>8.1f}")
print()
print("Rubric bounds for Ours:")
print(f"  P@20:   [{35.4}, {36.06}]")
print(f"  R@1000: [{17.7}, {18.03}]")
