"""Test download speed for full 1482-day range."""
import time, requests, numpy as np
from datetime import datetime

t0 = time.time()
url = ('https://wikimedia.org/api/rest_v1/metrics/pageviews/'
       'per-article/en.wikipedia.org/desktop/user/Classical_conditioning/'
       'daily/20201009/20241029')
r = requests.get(url, headers={'User-Agent': 'MILCCI/1.0'}, timeout=120)
dt = time.time() - t0
print(f'Status: {r.status_code}, time: {dt:.1f}s')
if r.status_code == 200:
    items = r.json().get('items', [])
    total = sum(it['views'] for it in items)
    print(f'Items: {len(items)}, total views: {total}')
else:
    print(f'Error: {r.text[:200]}')
