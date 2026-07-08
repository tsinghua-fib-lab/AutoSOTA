#!/usr/bin/env python3
"""Download CLIP ViT-B/16 essential files from hf-mirror.com."""

import os
import sys
import requests

MIRROR = 'https://hf-mirror.com'
MODEL = 'openai/clip-vit-base-patch16'
LOCAL = '/models/clip-vit-base-patch16'

os.makedirs(LOCAL, exist_ok=True)

# Essential files for PyTorch (skip flax_model.msgpack)
ESSENTIAL_FILES = [
    'config.json',
    'preprocessor_config.json',
    'pytorch_model.bin',
    'merges.txt',
    'vocab.json',
    'tokenizer.json',
    'special_tokens_map.json',
    'tokenizer_config.json',
]

def download_file(url, local_path, use_stream=False):
    if os.path.exists(local_path):
        size = os.path.getsize(local_path)
        print(f'  SKIP: {os.path.basename(local_path)} ({size} bytes)')
        return True

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f'  DOWNLOAD: {os.path.basename(local_path)} ...', end=' ', flush=True)

    try:
        if use_stream:
            resp = requests.get(url, stream=True, timeout=600,
                              headers={'User-Agent': 'Mozilla/5.0'})
            if resp.status_code == 200:
                total = 0
                with open(local_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=1024*1024):
                        f.write(chunk)
                        total += len(chunk)
                        if total % (50*1024*1024) == 0:
                            print(f'{total//(1024*1024)}MB...', end=' ', flush=True)
                print(f'OK ({total} bytes)')
                return True
        else:
            resp = requests.get(url, timeout=120,
                              headers={'User-Agent': 'Mozilla/5.0'})
            if resp.status_code == 200:
                with open(local_path, 'wb') as f:
                    f.write(resp.content)
                print(f'OK ({len(resp.content)} bytes)')
                return True

        print(f'FAILED (status={resp.status_code})')
        return False
    except Exception as e:
        print(f'ERROR: {e}')
        return False

print(f'Downloading CLIP ViT-B/16 to {LOCAL}')
print(f'Mirror: {MIRROR}')

for fname in ESSENTIAL_FILES:
    url = f'{MIRROR}/{MODEL}/resolve/main/{fname}'
    local_path = os.path.join(LOCAL, fname)
    use_stream = fname == 'pytorch_model.bin'
    success = download_file(url, local_path, use_stream=use_stream)
    if not success:
        print(f'ERROR: Failed to download {fname}')
        sys.exit(1)

print()
print('All essential files downloaded successfully!')
print('Files:')
for f in sorted(os.listdir(LOCAL)):
    fp = os.path.join(LOCAL, f)
    if os.path.isfile(fp):
        size_mb = os.path.getsize(fp) / (1024*1024)
        print(f'  {f}: {size_mb:.1f} MB')
