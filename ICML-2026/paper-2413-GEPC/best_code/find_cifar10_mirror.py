import requests

urls = [
    'https://download.pytorch.org/data/cifar-10-python.tar.gz',
    'https://github.com/kuangliu/pytorch-cifar/raw/master/cifar-10-python.tar.gz',
    'https://raw.githubusercontent.com/kuangliu/pytorch-cifar/master/cifar-10-python.tar.gz',
    'https://figshare.com/ndownloader/files/36501643',  # figshare
    'https://zenodo.org/record/2535967/files/cifar-10-python.tar.gz',
]

for url in urls:
    try:
        print(f'Testing HEAD: {url}')
        resp = requests.head(url, timeout=20, allow_redirects=True)
        print(f'  Status: {resp.status_code}, final: {resp.url[:80]}')
        if resp.status_code == 200:
            print(f'  Content-Length: {resp.headers.get("content-length", "N/A")}')
    except Exception as e:
        print(f'  Error: {type(e).__name__}: {str(e)[:100]}')
