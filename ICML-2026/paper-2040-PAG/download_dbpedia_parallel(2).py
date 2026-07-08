#!/usr/bin/env python3
"""Download DBpedia parquet files in parallel using wget."""
import os, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed

OUTPUT_DIR = "/datasets/dbpedia1536_parquet/data"
NUM_FILES = 26
BASE_URL = "https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M/resolve/main/data"
MAX_WORKERS = 4

def download_file(i):
    fname = f"train-{i:05d}-of-00026.parquet"
    outpath = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(outpath):
        size_mb = os.path.getsize(outpath) / (1024*1024)
        return (i, True, f"already exists ({size_mb:.0f} MB)")
    
    url = f"{BASE_URL}/{fname}"
    env = os.environ.copy()
    env["http_proxy"] = "http://172.17.0.1:17890"
    env["https_proxy"] = "http://172.17.0.1:17890"
    env.pop("ALL_PROXY", None)
    env.pop("all_proxy", None)
    
    t0 = time.time()
    try:
        result = subprocess.run(
            ["wget", "-q", "--timeout=120", "--tries=3",
             "-O", outpath, url],
            capture_output=True, text=True, timeout=900,
            env=env
        )
        elapsed = time.time() - t0
        if result.returncode == 0 and os.path.exists(outpath):
            size_mb = os.path.getsize(outpath) / (1024*1024)
            return (i, True, f"OK {size_mb:.0f}MB in {elapsed:.0f}s")
        else:
            return (i, False, f"FAILED: {result.stderr[:200]}")
    except Exception as e:
        return (i, False, f"ERROR: {e}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check existing files
    existing = sum(1 for i in range(NUM_FILES) 
                   if os.path.exists(os.path.join(OUTPUT_DIR, f"train-{i:05d}-of-00026.parquet")))
    print(f"Existing files: {existing}/{NUM_FILES}")
    
    if existing == NUM_FILES:
        print("All files already downloaded!")
        return 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_file, i): i for i in range(NUM_FILES)}
        success = 0
        failed = []
        
        for future in as_completed(futures):
            i, ok, msg = future.result()
            print(f"  [{i+1:2d}/{NUM_FILES}] {msg}")
            if ok:
                success += 1
            else:
                failed.append(i)
    
    print(f"\nDownloaded: {success}/{NUM_FILES}")
    if failed:
        print(f"Failed: {failed}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
