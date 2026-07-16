import os, requests, time
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

outdir = '/autosota_cache/tmp/skew40_raw'
os.makedirs(outdir, exist_ok=True)

for fname in ['test_zeroshot.parquet', 'train.parquet']:
    outpath = os.path.join(outdir, fname)
    if os.path.exists(outpath):
        sz = os.path.getsize(outpath) / (1024**3)
        print(f'{fname} already exists: {sz:.2f} GB')
        continue
    
    url = f'https://hf-mirror.com/datasets/GilpinLab/skew40/resolve/main/{fname}'
    print(f'Downloading {fname}...')
    r = requests.get(url, stream=True, timeout=300)
    total = int(r.headers.get('content-length', 0))
    downloaded = 0
    start = time.time()
    with open(outpath, 'wb') as f:
        for chunk in r.iter_content(chunk_size=16*1024*1024):
            f.write(chunk)
            downloaded += len(chunk)
            elapsed = time.time() - start
            pct = downloaded/total*100 if total else 0
            print(f'\r{fname}: {downloaded/(1024**3):.1f}/{total/(1024**3):.1f} GB ({pct:.0f}%)', end='')
    print(f'\nDone: {fname} ({os.path.getsize(outpath)/(1024**3):.2f} GB)')
