import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import requests
from requests.adapters import HTTPAdapter, BaseAdapter

# Check what the huggingface_hub session does
from huggingface_hub import get_session
session = get_session()

# Try a HEAD request with the session
url = 'https://hf-mirror.com/google-t5/t5-base/resolve/main/config.json'
print(f'Testing HEAD to: {url}')

# 1. Default session head (should have the adapter)
r1 = session.head(url, timeout=30)
print(f'Session HEAD status: {r1.status_code}, url: {r1.url}, hist: {len(r1.history)}')

# 2. Session head with explicit allow_redirects
r2 = session.head(url, allow_redirects=True, timeout=30)
print(f'Session HEAD (allow_redir) status: {r2.status_code}, url: {r2.url}, hist: {len(r2.history)}')

# 3. Plain requests head with allow_redirects
r3 = requests.head(url, allow_redirects=True, timeout=30)
print(f'Plain HEAD status: {r3.status_code}, url: {r3.url}, hist: {len(r3.history)}')

# 4. Plain session without custom adapters
s = requests.Session()
r4 = s.head(url, allow_redirects=True, timeout=30)
print(f'Plain session HEAD status: {r4.status_code}, url: {r4.url}, hist: {len(r4.history)}')
