import os, sys, struct, time, gc
import numpy as np

# Ensure no_proxy includes HF domains
os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",hf-mirror.com,huggingface.co"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

DATASET = "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M"
N_BASE = 999000
N_QUERY = 1000
TOP_K = 100
DIM = 1536
OUT_DIR = "/repo/data/dbpedia1536"
TMP_DIR = "/autosota_cache/tmp/dbpedia1536"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

PARQUET_FILES = [f"data/train-{i:05d}-of-00026.parquet" for i in range(26)]

# Step 1: Download parquet files (cached by HF hub) and stream embeddings to disk
print("Step 1: Downloading and extracting embeddings...")
total_written = 0
base_file = open(os.path.join(OUT_DIR, "base.fbin"), "wb")
query_file = open(os.path.join(OUT_DIR, "query.fbin"), "wb")

# Write headers
base_file.write(struct.pack("<I", N_BASE))
base_file.write(struct.pack("<I", DIM))
query_file.write(struct.pack("<I", N_QUERY))
query_file.write(struct.pack("<I", DIM))

# Accumulate query vectors
query_vectors = np.zeros((N_QUERY, DIM), dtype=np.float32)
query_count = 0

for i, fname in enumerate(PARQUET_FILES):
    local = hf_hub_download(DATASET, fname, repo_type="dataset", cache_dir=TMP_DIR)
    table = pq.read_table(local, columns=["text-embedding-3-large-1536-embedding"])
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
    print(f"  File {i+1}/26 done, total written: {total_written}")
    if total_written >= N_BASE + N_QUERY:
        break

base_file.close()
query_file.write(query_vectors.tobytes())
query_file.close()
print(f"Base and query files written. Base={N_BASE}, Query={query_count}")

# Step 2: Compute ground truth
print("Step 2: Computing ground truth (brute force KNN)...")
t0 = time.time()

gt_ids = np.zeros((N_QUERY, TOP_K), dtype=np.int32)

# Read base from file in chunks
with open(os.path.join(OUT_DIR, "base.fbin"), "rb") as bf:
    base_rows = struct.unpack("<I", bf.read(4))[0]
    base_dim = struct.unpack("<I", bf.read(4))[0]
    assert base_rows == N_BASE and base_dim == DIM
    base_data_start = bf.tell()

def load_base_chunk(f, start_row, count):
    f.seek(base_data_start + start_row * DIM * 4)
    raw = f.read(count * DIM * 4)
    return np.frombuffer(raw, dtype=np.float32).reshape(count, DIM)

CHUNK = 50000
with open(os.path.join(OUT_DIR, "base.fbin"), "rb") as bf:
    for qi in range(N_QUERY):
        q = query_vectors[qi].astype(np.float32)
        q_norm_sq = float(np.dot(q, q))
        
        best_idx = np.zeros(TOP_K, dtype=np.int32)
        best_dist = np.full(TOP_K, np.inf, dtype=np.float32)
        
        for chunk_start in range(0, N_BASE, CHUNK):
            chunk_end = min(chunk_start + CHUNK, N_BASE)
            chunk = load_base_chunk(bf, chunk_start, chunk_end - chunk_start)
            chunk_norms_sq = np.sum(chunk.astype(np.float32) ** 2, axis=1).astype(np.float32)
            dots = (chunk.astype(np.float32) @ q).astype(np.float32)
            dists = np.float32(q_norm_sq) + chunk_norms_sq - np.float32(2.0) * dots
            dists = np.maximum(dists, 0.0)
            
            # Merge top-K
            all_dists = np.concatenate([best_dist, dists])
            all_idx = np.concatenate([best_idx, np.arange(chunk_start, chunk_end, dtype=np.int32)])
            topk = np.argpartition(all_dists, TOP_K)[:TOP_K]
            sort_k = topk[np.argsort(all_dists[topk])]
            best_dist = all_dists[sort_k].copy()
            best_idx = all_idx[sort_k].copy()
        
        gt_ids[qi] = best_idx
        
        if (qi + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (qi + 1) * (N_QUERY - qi - 1)
            print(f"  Query {qi+1}/{N_QUERY}, elapsed={elapsed:.0f}s, ETA={eta:.0f}s, dist[0]={best_dist[0]:.6f}")

print(f"Ground truth computed in {time.time()-t0:.1f}s")

# Step 3: Write gt100.ibin
with open(os.path.join(OUT_DIR, "gt100.ibin"), "wb") as f:
    f.write(struct.pack("<I", N_QUERY))
    f.write(struct.pack("<I", TOP_K))
    f.write(gt_ids.tobytes())

print("Written gt100.ibin")
print("Done!")

# Validate
print("Validating files...")
for fname in ["base.fbin", "query.fbin", "gt100.ibin"]:
    path = os.path.join(OUT_DIR, fname)
    size = os.path.getsize(path)
    print(f"  {fname}: {size/1024/1024:.1f} MB")
