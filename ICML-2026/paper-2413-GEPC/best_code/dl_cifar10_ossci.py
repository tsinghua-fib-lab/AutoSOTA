import requests

# torchvision 0.17+ uses this mirror
urls = [
    'https://ossci-datasets.s3.amazonaws.com/cifar-10-python.tar.gz',
    'https://ossci-datasets.s3.us-east-1.amazonaws.com/cifar-10-python.tar.gz',
]

for url in urls:
    try:
        print(f'Trying: {url}')
        resp = requests.head(url, timeout=20, allow_redirects=True)
        print(f'  Status: {resp.status_code}')
        if resp.status_code == 200:
            print(f'  Content-Length: {resp.headers.get("content-length", "N/A")}')
    except Exception as e:
        print(f'  Error: {type(e).__name__}: {str(e)[:100]}')
