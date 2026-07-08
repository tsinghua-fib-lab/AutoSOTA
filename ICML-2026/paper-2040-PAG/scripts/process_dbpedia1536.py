import os, struct, time, gc
import numpy as np
import pyarrow.parquet as pq
from glob import glob

N_BASE = 999000
N_QUERY = 1000
TOP_K = 100
DIM = 1536
OUT_DIR = "/repo/data/dbpedia1536"
SRC_DIR = "/autosota_cache/dbpedia1536_raw"

os.makedirs(OUT_DIR, exist_ok=True)

# Step 1: Stream embeddings directly to fbin files
print("Step 1: Extracting embeddings from cached parquet files...")
parquet_files = sorted(glob(f"{SRC_DIR}/data/train-*.parquet"))
assert len(parquet_files) == 26, f"Expected 26 files, got {len(parquet_files)}"

total_written = 0
t0 = time.time()

base_file = open(os.path.join(OUT_DIR, "base.fbin"), "wb")
base_file.write(struct.pack("<I", N_BASE))
base_file.write(struct.pack("<I", DIM))

query_file = open(os.path.join(OUT_DIR, "query.fbin"), "wb")
query_file.write(struct.pack("<I", N_QUERY))
query_file.write(struct.pack("<I", DIM))

query_vectors = np.zeros((N_QUERY, DIM), dtype=np.float32)
query_count = 0

for fpath in parquet_files:
    table = pq.read_table(fpath, columns=["text-embedding-3-large-1536-embedding"])
    col = table.column("text-embedding-3-large-1536-embedding")
    arr = np.array(col.to_pylist(), dtype=np.float32)
    
    for j in range(arr.shape[0]):
        if total_written < N_BASE:
            base_file.write(arr[j].tobytes())
            total_written += 1
        elif total_written < N_BASE + N_QUERY:
            query_vectors[query_count] = arr[j]
            query_count += 1
            total_written += 1
        else:
            break
    
    del arr, table, col
    gc.collect()
    elapsed = time.time() - t0
    print(f"  {os.path.basename(fpath)}: total_written={total_written}, elapsed={elapsed:.0f}s")
    if total_written >= N_BASE + N_QUERY:
        break

query_file.write(query_vectors.tobytes())
base_file.close()
query_file.close()
print(f"Base and query files written in {time.time()-t0:.1f}s")

# Step 2: Compute ground truth (brute force)
print("Step 2: Computing brute-force ground truth...")
t0 = time.time()

gt_ids = np.zeros((N_QUERY, TOP_K), dtype=np.int32)

# Read whole base into memory (6.13 GB)
print("  Loading base vectors into memory...")
with open(os.path.join(OUT_DIR, "base.fbin"), "rb") as bf:
    rows = struct.unpack("<I", bf.read(4))[0]
    dim = struct.unpack("<I", bf.read(4))[0]
    assert rows == N_BASE and dim == DIM
    base = np.frombuffer(bf.read(), dtype=np.float32).reshape(rows, dim)
print(f"  Base loaded: {base.shape}, {base.nbytes/1024/1024/1024:.2f} GB")

# Compute ground truth
BATCH_Q = 50
BATCH_B = 50000
for q_start in range(0, N_QUERY, BATCH_Q):
    q_end = min(q_start + BATCH_Q, N_QUERY)
    q_batch = query_vectors[q_start:q_end]
    q_norms = np.sum(q_batch.astype(np.float32) ** 2, axis=1)
    
    # Use float32 to save memory
    best_idx = np.zeros((q_end - q_start, TOP_K), dtype=np.int32)
    best_dist = np.full((q_end - q_start, TOP_K), np.inf, dtype=np.float32)
    
    for b_start in range(0, N_BASE, BATCH_B):
        b_end = min(b_start + BATCH_B, N_BASE)
        chunk = base[b_start:b_end]
        chunk_norms = np.sum(chunk ** 2, axis=1).astype(np.float32)
        dots = (chunk @ q_batch.T).astype(np.float32)
        
        for i in range(q_end - q_start):
            dists = q_norms[i] + chunk_norms - 2.0 * dots[:, i]
            dists = np.maximum(dists, 0.0)
            
            all_dists = np.concatenate([best_dist[i], dists])
            all_idx = np.concatenate([best_idx[i], np.arange(b_start, b_end, dtype=np.int32)])
            topk = np.argpartition(all_dists, TOP_K)[:TOP_K]
            sort_k = topk[np.argsort(all_dists[topk])]
            best_dist[i] = all_dists[sort_k]
            best_idx[i] = all_idx[sort_k]
    
    gt_ids[q_start:q_end] = best_idx
    
    elapsed = time.time() - t0
    eta = elapsed / q_end * (N_QUERY - q_end)
    print(f"  {q_end}/{N_QUERY} queries, elapsed={elapsed:.0f}s, ETA={eta:.0f}s")

print(f"Ground truth computed in {time.time()-t0:.1f}s")

# Step 3: Write gt100.ibin
with open(os.path.join(OUT_DIR, "gt100.ibin"), "wb") as f:
    f.write(struct.pack("<I", N_QUERY))
    f.write(struct.pack("<I", TOP_K))
    f.write(gt_ids.tobytes())

print("Done!")
for fname in ["base.fbin", "query.fbin", "gt100.ibin"]:
    path = os.path.join(OUT_DIR, fname)
    size = os.path.getsize(path) / (1024**3)
    print(f"  {fname}: {size:.2f} GB")
