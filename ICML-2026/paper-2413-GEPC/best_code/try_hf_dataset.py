import os, requests

token = os.environ.get("HF_TOKEN", "")
url = "https://hf-mirror.com/datasets/torch/cifar10/resolve/main/cifar-10-python.tar.gz"
headers = {}
if token:
    headers["Authorization"] = f"Bearer {token}"
try:
    resp = requests.head(url, headers=headers, timeout=30)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        print(f"Content-Length: {resp.headers.get('content-length', 'N/A')}")
    elif resp.status_code == 302:
        print(f"Redirect: {resp.headers.get('location', 'N/A')}")
except Exception as e:
    print(f"Error: {e}")
