#!/usr/bin/env python3
"""Convert DBpedia1536 parquet to PAG fbin and compute ground truth (GPU-accelerated, batched)."""
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
QUERY_BATCH_SIZE = 64  # Process multiple queries at once on GPU

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Read parquet files
    import pandas as pd
    all_vecs = np.zeros((BASE_COUNT + QUERY_COUNT, DIM), dtype=np.float32)
    offset = 0
    t0 = time.time()
    
    print(f"Reading {NUM_PARQUET_FILES} parquet files...")
    for i in range(NUM_PARQUET_FILES):
        fname = f"train-{i:05d}-of-00026.parquet"
        fpath = os.path.join(PARQUET_DIR, fname)
        assert os.path.exists(fpath), f"Missing: {fpath}"
        
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
            print(f"  [{i+1}/{NUM_PARQUET_FILES}] {offset:,} vectors loaded ({time.time()-t0:.0f}s)")
        
        if offset >= BASE_COUNT + QUERY_COUNT:
            break
    
    print(f"Loaded {offset:,} vectors in {time.time()-t0:.0f}s")
    
    # Step 2: Split and write fbin
    base = all_vecs[:BASE_COUNT]
    query = all_vecs[BASE_COUNT:BASE_COUNT + QUERY_COUNT]
    del all_vecs
    
    with open(os.path.join(OUTPUT_DIR, "base.fbin"), "wb") as f:
        f.write(struct.pack("<II", BASE_COUNT, DIM))
        f.write(base.tobytes())
    base_size_mb = os.path.getsize(os.path.join(OUTPUT_DIR, "base.fbin")) / (1024*1024)
    print(f"base.fbin: {base_size_mb:.0f} MB")
    
    with open(os.path.join(OUTPUT_DIR, "query.fbin"), "wb") as f:
        f.write(struct.pack("<II", QUERY_COUNT, DIM))
        f.write(query.tobytes())
    print(f"query.fbin: {os.path.getsize(os.path.join(OUTPUT_DIR, 'query.fbin'))} bytes")
    
    # Step 3: GPU ground truth (batched)
    device = torch.device("cuda:0")
    print(f"\nGPU Ground Truth on {device} (batch_size={QUERY_BATCH_SIZE})...")
    print(f"  Base: {base.shape}, dtype={base.dtype}")
    
    base_t = torch.from_numpy(base).to(device)
    base_norm_sq = (base_t ** 2).sum(dim=1)
    
    n_q = query.shape[0]
    gt_indices = np.zeros((n_q, GT_K), dtype=np.uint32)
    gt_t0 = time.time()
    
    for start_i in range(0, n_q, QUERY_BATCH_SIZE):
        end_i = min(start_i + QUERY_BATCH_SIZE, n_q)
        q_batch = torch.from_numpy(query[start_i:end_i]).to(device)  # [B, D]
        
        q_norm = (q_batch ** 2).sum(dim=1, keepdim=True)  # [B, 1]
        dots = torch.mm(q_batch, base_t.t())  # [B, N]
        dists = q_norm + base_norm_sq.unsqueeze(0) - 2 * dots
        dists = torch.clamp(dists, min=0.0)
        
        _, topk_idx = torch.topk(dists, k=GT_K, dim=1, largest=False)
        gt_indices[start_i:end_i] = topk_idx.cpu().numpy().astype(np.uint32)
        
        elapsed = time.time() - gt_t0
        progress = end_i / n_q
        eta = elapsed / progress * (1 - progress) if progress > 0 else 0
        print(f"  [{end_i}/{n_q}] {progress*100:.0f}%, {elapsed:.0f}s elapsed, ETA {eta:.0f}s")
    
    gt_elapsed = time.time() - gt_t0
    print(f"Ground truth: {gt_elapsed:.1f}s ({n_q/gt_elapsed:.1f} q/s)")
    
    # Step 4: Write gt1000.ibin
    with open(os.path.join(OUTPUT_DIR, "gt1000.ibin"), "wb") as f:
        f.write(struct.pack("<II", n_q, GT_K))
        f.write(gt_indices.tobytes())
    
    print(f"\nDone! {OUTPUT_DIR}:")
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        fp = os.path.join(OUTPUT_DIR, fn)
        print(f"  {fn}: {os.path.getsize(fp):,} bytes")

if __name__ == "__main__":
    main()
