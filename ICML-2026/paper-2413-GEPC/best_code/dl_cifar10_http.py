import requests, tarfile, os, shutil

# Try HTTP-only mirrors
http_urls = [
    'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
    'http://data.csail.mit.edu/places/places365/cifar-10-python.tar.gz',
]

for url in http_urls:
    try:
        print(f'Trying HTTP: {url}')
        resp = requests.get(url, stream=True, timeout=30, allow_redirects=True)
        print(f'  Status: {resp.status_code}, final URL: {resp.url[:100]}')
    except Exception as e:
        print(f'  Error: {type(e).__name__}: {e}')
