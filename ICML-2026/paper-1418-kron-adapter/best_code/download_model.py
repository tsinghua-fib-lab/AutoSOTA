import requests
import os
import sys

HF_CACHE = '/autosota_cache/hf'
REPO_ID = 'google-t5/t5-base'
MIRROR_BASE = 'https://hf-mirror.com'

# List of files needed (skip large format-specific files)
FILES_NEEDED = [
    'config.json',
    'model.safetensors', 
    'tokenizer.json',
    'spiece.model',
    'generation_config.json',
    'tokenizer_config.json',
    'special_tokens_map.json',
]

# First get all files listing
api_url = f'{MIRROR_BASE}/api/models/{REPO_ID}/tree/main?recursive=True&expand=False'
print(f'Fetching file list from {api_url}')
r = requests.get(api_url, allow_redirects=True, timeout=30)
all_files = r.json()
print(f'Found {len(all_files)} files')

# Get files needed (include any missing from our list)
file_map = {f['path']: f for f in all_files}
for path in list(FILES_NEEDED):
    if path not in file_map:
        print(f'WARNING: {path} not found in repo, removing from list')
        FILES_NEEDED.remove(path)

print(f'Files to download: {FILES_NEEDED}')

# Setup cache structure
hub_dir = os.path.join(HF_CACHE, 'hub', 'models--google-t5--t5-base')
blob_dir = os.path.join(hub_dir, 'blobs')
snap_dir = os.path.join(hub_dir, 'snapshots', 'main')
os.makedirs(blob_dir, exist_ok=True)
os.makedirs(snap_dir, exist_ok=True)

for path in FILES_NEEDED:
    f_info = file_map[path]
    oid = f_info['oid']
    size = f_info['size']
    
    # Download URL 
    dl_url = f'{MIRROR_BASE}/{REPO_ID}/resolve/main/{path}'
    print(f'\nDownloading {path} ({size/1024/1024:.1f} MB) from {dl_url}')
    
    blob_path = os.path.join(blob_dir, oid)
    
    if os.path.exists(blob_path) and os.path.getsize(blob_path) == size:
        print(f'  Already cached, skipping')
    else:
        r = requests.get(dl_url, allow_redirects=True, timeout=600, stream=True)
        r.raise_for_status()
        
        with open(blob_path + '.tmp', 'wb') as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=1024*1024):
                f.write(chunk)
                downloaded += len(chunk)
                if size:
                    print(f'\r  {downloaded/1024/1024:.1f}/{size/1024/1024:.1f} MB ({100*downloaded//size}%)', end='', flush=True)
        
        os.rename(blob_path + '.tmp', blob_path)
        print(f'\n  Done: {path}')
    
    # Create symlink in snapshots
    snap_path = os.path.join(snap_dir, path)
    snap_dirname = os.path.dirname(snap_path)
    os.makedirs(snap_dirname, exist_ok=True)
    if os.path.exists(snap_path):
        os.remove(snap_path)
    os.symlink(os.path.relpath(blob_path, snap_dirname), snap_path)
    print(f'  Cached to: {snap_path}')

print('\n=== All files downloaded and cached ===')
