#!/usr/bin/env python3
"""Convert DBpedia1536 parquet data to PAG fbin format and compute ground truth."""
import os
import sys
import struct
import time
import numpy as np
import torch

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

OUTPUT_DIR = "/datasets/dbpedia1536"
BASE_COUNT = 999000
QUERY_COUNT = 1000
DIM = 1536
GT_K = 1000

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Stream dataset and extract first BASE_COUNT + QUERY_COUNT vectors
    from datasets import load_dataset
    
    total_needed = BASE_COUNT + QUERY_COUNT
    print(f"Streaming {total_needed} vectors from HF dataset...")
    
    ds = load_dataset(
        "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M",
        split="train",
        streaming=True,
        token=os.environ.get("HF_TOKEN"),
    )
    
    all_embeddings = np.zeros((total_needed, DIM), dtype=np.float32)
    start_time = time.time()
    
    for i, example in enumerate(ds):
        if i >= total_needed:
            break
        emb = np.array(example["text-embedding-3-large-1536-embedding"], dtype=np.float32)
        all_embeddings[i] = emb
        
        if (i + 1) % 100000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (total_needed - i - 1) / rate
            print(f"  Loaded {i+1}/{total_needed} vectors, {rate:.0f} vec/s, ETA: {eta:.0f}s")
    
    elapsed = time.time() - start_time
    print(f"Loaded {total_needed} vectors in {elapsed:.1f}s ({total_needed/elapsed:.0f} vec/s)")
    
    # 2. Split into base and query
    base = all_embeddings[:BASE_COUNT]
    query = all_embeddings[BASE_COUNT:]
    del all_embeddings
    print(f"Base: {base.shape}, Query: {query.shape}")
    
    # 3. Write fbin files
    print("\nWriting fbin files...")
    with open(os.path.join(OUTPUT_DIR, "base.fbin"), "wb") as f:
        f.write(struct.pack("<II", BASE_COUNT, DIM))
        f.write(base.tobytes())
    print(f"  base.fbin: {os.path.getsize(os.path.join(OUTPUT_DIR, 'base.fbin'))} bytes")
    
    with open(os.path.join(OUTPUT_DIR, "query.fbin"), "wb") as f:
        f.write(struct.pack("<II", QUERY_COUNT, DIM))
        f.write(query.tobytes())
    print(f"  query.fbin: {os.path.getsize(os.path.join(OUTPUT_DIR, 'query.fbin'))} bytes")
    
    # 4. Compute ground truth using GPU brute force
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nComputing ground truth (K={GT_K}) on {device}...")
    
    base_t = torch.from_numpy(base).to(device)
    query_t = torch.from_numpy(query).to(device)
    
    # Process queries in batches to manage GPU memory
    # 1K queries × 1M base × 1536 dim is ~6GB for distance matrix
    n_query = query_t.shape[0]
    n_base = base_t.shape[0]
    
    # Pre-compute base norms
    base_norm_sq = (base_t ** 2).sum(dim=1)  # [N]
    
    gt_indices = np.zeros((n_query, GT_K), dtype=np.uint32)
    gt_start = time.time()
    
    for i in range(n_query):
        q = query_t[i:i+1]  # [1, D]
        q_norm_sq = (q ** 2).sum(dim=1, keepdim=True)  # [1, 1]
        dots = torch.mm(q, base_t.t())  # [1, N]
        dists = q_norm_sq + base_norm_sq.unsqueeze(0) - 2 * dots
        dists = torch.clamp(dists, min=0.0)
        
        # Get top-K smallest distances (L2)
        _, topk_idx = torch.topk(dists, k=GT_K, dim=1, largest=False)
        gt_indices[i] = topk_idx.cpu().numpy().astype(np.uint32)
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - gt_start
            rate = (i + 1) / elapsed
            eta = (n_query - i - 1) / rate
            print(f"  Ground truth: {i+1}/{n_query} queries, {rate:.1f} q/s, ETA: {eta:.0f}s")
    
    gt_elapsed = time.time() - gt_start
    print(f"Ground truth computed in {gt_elapsed:.1f}s ({n_query/gt_elapsed:.1f} q/s)")
    
    # 5. Write ground truth ibin
    with open(os.path.join(OUTPUT_DIR, "gt1000.ibin"), "wb") as f:
        f.write(struct.pack("<II", n_query, GT_K))
        f.write(gt_indices.tobytes())
    print(f"  gt1000.ibin: {os.path.getsize(os.path.join(OUTPUT_DIR, 'gt1000.ibin'))} bytes")
    
    print("\nDone! Files in", OUTPUT_DIR)
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        fp = os.path.join(OUTPUT_DIR, fn)
        print(f"  {fn}: {os.path.getsize(fp)} bytes")

if __name__ == "__main__":
    main()
