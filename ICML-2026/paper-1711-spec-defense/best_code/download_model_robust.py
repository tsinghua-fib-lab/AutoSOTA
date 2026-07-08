#!/usr/bin/env python3
"""Robust model downloader with resume and retry support."""

import os
import time
import requests

URL = "https://hf-mirror.com/openai/clip-vit-base-patch16/resolve/main/pytorch_model.bin"
LOCAL_PATH = "/models/clip-vit-base-patch16/pytorch_model.bin"
TEMP_PATH = LOCAL_PATH + ".part"

# Get expected file size
try:
    resp = requests.head(URL, timeout=30, allow_redirects=True)
    expected_size = int(resp.headers.get("Content-Length", 0))
    print(f"Expected size: {expected_size} bytes ({expected_size/1024/1024:.1f} MB)")
except Exception as e:
    print(f"HEAD request failed: {e}")
    expected_size = 0

# Check existing progress
resume_pos = 0
if os.path.exists(TEMP_PATH):
    resume_pos = os.path.getsize(TEMP_PATH)
    print(f"Resuming from byte {resume_pos} ({resume_pos/1024/1024:.1f} MB)")
elif os.path.exists(LOCAL_PATH):
    resume_pos = os.path.getsize(LOCAL_PATH)
    print(f"Resuming from byte {resume_pos} ({resume_pos/1024/1024:.1f} MB)")
    # Rename to temp
    if resume_pos < expected_size:
        os.rename(LOCAL_PATH, TEMP_PATH)
    else:
        print("File already complete!")
        exit(0)

# Download with retry
max_retries = 50
retry_count = 0
while resume_pos < expected_size:
    try:
        headers = {}
        if resume_pos > 0:
            headers["Range"] = f"bytes={resume_pos}-"

        print(f"Downloading from byte {resume_pos}...", end=" ", flush=True)
        resp = requests.get(URL, headers=headers, timeout=120, stream=True)

        if resp.status_code not in (200, 206):
            print(f"Bad status: {resp.status_code}")
            retry_count += 1
            if retry_count > max_retries:
                break
            time.sleep(5)
            continue

        with open(TEMP_PATH, "ab") as f:
            for chunk in resp.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    resume_pos += len(chunk)
                    if resume_pos % (10*1024*1024) < 1024*1024:
                        pct = resume_pos * 100 / expected_size if expected_size else 0
                        print(f"{resume_pos/1024/1024:.0f}MB({pct:.0f}%)...", end=" ", flush=True)

        print(f"Done! Total: {resume_pos/1024/1024:.1f} MB")

        # Verify completeness
        if expected_size > 0 and resume_pos >= expected_size:
            break

    except Exception as e:
        retry_count += 1
        print(f"Error ({retry_count}/{max_retries}): {e}")
        if retry_count > max_retries:
            print("Max retries reached!")
            break
        time.sleep(5)

# Finalize
if expected_size > 0 and resume_pos >= expected_size:
    os.rename(TEMP_PATH, LOCAL_PATH)
    print(f"SUCCESS: File downloaded ({os.path.getsize(LOCAL_PATH)} bytes)")
else:
    print(f"Download incomplete: {resume_pos} / {expected_size} bytes")
    if os.path.exists(TEMP_PATH):
        print("Partial download saved, can resume later")
