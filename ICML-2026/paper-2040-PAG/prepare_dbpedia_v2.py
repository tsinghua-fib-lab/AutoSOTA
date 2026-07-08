#!/usr/bin/env python3
"""Convert DBpedia1536 parquet to PAG fbin and compute ground truth (GPU-accelerated)."""
import os, struct, time
import numpy as np
import torch

PARQUET_DIR = "/datasets/dbpedia1536_parquet/data"
OUTPUT_DIR = "/datasets/dbpedia1536"
BASE_COUNT = 999000
QUERY_COUNT = 1000
DIM = 1536
GT_K = 1000
NUM_PARQUET_FILES = 26

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Read all parquet files and extract embeddings
    import pandas as pd
    all_vecs = np.zeros((BASE_COUNT + QUERY_COUNT, DIM), dtype=np.float32)
    offset = 0
    start = time.time()
    
    print(f"Reading {NUM_PARQUET_FILES} parquet files...")
    for i in range(NUM_PARQUET_FILES):
        fname = f"train-{i:05d}-of-00026.parquet"
        fpath = os.path.join(PARQUET_DIR, fname)
        if not os.path.exists(fpath):
            print(f"ERROR: {fpath} not found!")
            continue
        
        df = pd.read_parquet(fpath)
        emb_col = "text-embedding-3-large-1536-embedding"
        embs = np.array(df[emb_col].tolist(), dtype=np.float32)
        n = embs.shape[0]
        
        end = min(offset + n, BASE_COUNT + QUERY_COUNT)
        to_copy = end - offset
        if to_copy > 0:
            all_vecs[offset:end] = embs[:to_copy]
            offset = end
        
        if (i+1) % 5 == 0:
            print(f"  [{i+1}/{NUM_PARQUET_FILES}] Processed, {offset} vectors loaded")
        
        if offset >= BASE_COUNT + QUERY_COUNT:
            print(f"  Got all {offset} vectors, stopping early at file {i+1}")
            break
    
    elapsed = time.time() - start
    print(f"Total: {offset} vectors in {elapsed:.1f}s")
    
    # 2. Split
    base = all_vecs[:BASE_COUNT]
    query = all_vecs[BASE_COUNT:BASE_COUNT + QUERY_COUNT]
    del all_vecs
    print(f"Base: {base.shape}, Query: {query.shape}")
    
    # 3. Write fbin files
    with open(os.path.join(OUTPUT_DIR, "base.fbin"), "wb") as f:
        f.write(struct.pack("<II", BASE_COUNT, DIM))
        f.write(base.tobytes())
    print(f"  base.fbin: {os.path.getsize(os.path.join(OUTPUT_DIR, 'base.fbin')) / (1024*1024):.0f} MB")
    
    with open(os.path.join(OUTPUT_DIR, "query.fbin"), "wb") as f:
        f.write(struct.pack("<II", QUERY_COUNT, DIM))
        f.write(query.tobytes())
    print(f"  query.fbin: {os.path.getsize(os.path.join(OUTPUT_DIR, 'query.fbin'))} bytes")
    
    # 4. GPU ground truth computation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nComputing ground truth on {device}...")
    
    base_t = torch.from_numpy(base).to(device)  # [N, D]
    base_norm_sq = (base_t ** 2).sum(dim=1)  # [N]
    
    n_query = query.shape[0]
    gt_indices = np.zeros((n_query, GT_K), dtype=np.uint32)
    gt_start = time.time()
    
    for i in range(n_query):
        q = torch.from_numpy(query[i:i+1]).to(device)  # [1, D]
        q_norm_sq = (q ** 2).sum(dim=1, keepdim=True)
        dots = torch.mm(q, base_t.t())
        dists = q_norm_sq + base_norm_sq.unsqueeze(0) - 2 * dots
        dists = torch.clamp(dists, min=0.0)
        _, topk_idx = torch.topk(dists, k=GT_K, dim=1, largest=False)
        gt_indices[i] = topk_idx.cpu().numpy().astype(np.uint32)
        
        if (i+1) % 200 == 0:
            e = time.time() - gt_start
            print(f"  {i+1}/{n_query} queries, {n_query*e/(i+1):.0f}s total ETA")
    
    gt_elapsed = time.time() - gt_start
    print(f"Ground truth: {gt_elapsed:.1f}s ({n_query/gt_elapsed:.1f} q/s)")
    
    # 5. Write gt1000.ibin
    with open(os.path.join(OUTPUT_DIR, "gt1000.ibin"), "wb") as f:
        f.write(struct.pack("<II", n_query, GT_K))
        f.write(gt_indices.tobytes())
    
    print(f"\nDone! {OUTPUT_DIR}:")
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        fp = os.path.join(OUTPUT_DIR, fn)
        print(f"  {fn}: {os.path.getsize(fp):,} bytes")
    
    return 0

if __name__ == "__main__":
    exit(main())
