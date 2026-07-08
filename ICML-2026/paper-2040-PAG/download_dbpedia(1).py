#!/usr/bin/env python3
"""Download DBpedia parquet files using wget."""
import os
import subprocess
import sys

OUTPUT_DIR = "/datasets/dbpedia1536_parquet/data"
NUM_FILES = 26
BASE_URL = "https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M/resolve/main/data"

proxies = {
    "http": "http://172.17.0.1:17890",
    "https": "http://172.17.0.1:17890",
}

def download_file(i):
    fname = f"train-{i:05d}-of-00026.parquet"
    outpath = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(outpath):
        size = os.path.getsize(outpath)
        print(f"  [{i+1}/{NUM_FILES}] {fname} already exists ({size} bytes)")
        return True
    
    url = f"{BASE_URL}/{fname}"
    cmd = [
        "wget", "-q", "--timeout=60", "--tries=3",
        "-e", f"http_proxy={proxies['http']}",
        "-e", f"https_proxy={proxies['https']}",
        "-O", outpath, url
    ]
    print(f"  [{i+1}/{NUM_FILES}] Downloading {fname}...", end=" ", flush=True)
    
    try:
        # Need to set env for wget
        env = os.environ.copy()
        env["http_proxy"] = proxies["http"]
        env["https_proxy"] = proxies["https"]
        env.pop("ALL_PROXY", None)
        env.pop("all_proxy", None)
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        if result.returncode == 0 and os.path.exists(outpath):
            size_mb = os.path.getsize(outpath) / (1024*1024)
            print(f"OK ({size_mb:.1f} MB)")
            return True
        else:
            print(f"FAILED: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    success = 0
    failed = []
    
    for i in range(NUM_FILES):
        if download_file(i):
            success += 1
        else:
            failed.append(i)
    
    print(f"\nDownloaded {success}/{NUM_FILES} files")
    if failed:
        print(f"Failed: {failed}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
