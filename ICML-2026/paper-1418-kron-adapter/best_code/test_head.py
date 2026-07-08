import requests
import os

# Test what happens with HEAD request and 308 redirect
url = 'https://hf-mirror.com/google-t5/t5-base/resolve/main/config.json'

# 1. HEAD without following redirects
r1 = requests.head(url, allow_redirects=False, timeout=10)
print(f'HEAD no redirects: {r1.status_code}')
print(f'  headers: {dict(r1.headers)}')

# 2. HEAD with following redirects
r2 = requests.head(url, allow_redirects=True, timeout=10)
print(f'HEAD with redirects: {r2.status_code}')
print(f'  final url: {r2.url}')
print(f'  headers: {dict(r2.headers)}')

# 3. What does requests do by default?
r3 = requests.head(url, timeout=10)
print(f'HEAD default: {r3.status_code}')
print(f'  final url: {r3.url}')

# 4. The huggingface_hub library uses a session
s = requests.Session()
r4 = s.head(url, timeout=10)
print(f'Session HEAD default: {r4.status_code}')
print(f'  final url: {r4.url}')
print(f'  history: {[h.status_code for h in r4.history]}')
