import os, requests, tarfile, shutil, sys

datasets_dir = '/datasets'

# Download CIFAR-10
cifar10_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
cifar10_tmp = '/autosota_cache/tmp/cifar-10-python.tar.gz'
cifar10_dir = os.path.join(datasets_dir, 'cifar-10-batches-py')

if not os.path.exists(cifar10_dir) or not os.path.exists(os.path.join(cifar10_dir, 'data_batch_1')):
    os.makedirs('/autosota_cache/tmp', exist_ok=True)
    print(f'Downloading CIFAR-10 from {cifar10_url} ...', flush=True)
    resp = requests.get(cifar10_url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0))
    print(f'Content-Length: {total} bytes ({total/1024/1024:.1f} MB)', flush=True)
    dl = 0
    with open(cifar10_tmp, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8*1024*1024):
            f.write(chunk)
            dl += len(chunk)
            print(f'\r  {dl/1024/1024:.1f}/{total/1024/1024:.1f} MB', end='', flush=True)
    print()
    print(f'Extracting...', flush=True)
    with tarfile.open(cifar10_tmp, 'r:gz') as tar:
        tar.extractall(path=datasets_dir)
    os.remove(cifar10_tmp)
    print(f'CIFAR-10 extracted to {cifar10_dir}', flush=True)
else:
    print(f'CIFAR-10 already exists at {cifar10_dir}', flush=True)

# SVHN
svhn_dir = os.path.join(datasets_dir, 'svhn')
svhn_train = os.path.join(svhn_dir, 'train_32x32.mat')
svhn_test = os.path.join(svhn_dir, 'test_32x32.mat')

def download_svhn_file(url, dest, name):
    tmp = '/autosota_cache/tmp/' + os.path.basename(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    os.makedirs('/autosota_cache/tmp', exist_ok=True)
    print(f'Downloading {name} from {url} ...', flush=True)
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0))
    dl = 0
    with open(tmp, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8*1024*1024):
            f.write(chunk)
            dl += len(chunk)
            if total > 0:
                print(f'\r  {dl/1024/1024:.1f}/{total/1024/1024:.1f} MB', end='', flush=True)
    print()
    shutil.move(tmp, dest)
    print(f'{name} saved ({os.path.getsize(dest)} bytes)', flush=True)

if not os.path.exists(svhn_test):
    download_svhn_file('http://ufldl.stanford.edu/housenumbers/test_32x32.mat', svhn_test, 'SVHN test')
else:
    print(f'SVHN test already exists at {svhn_test}', flush=True)

if not os.path.exists(svhn_train):
    download_svhn_file('http://ufldl.stanford.edu/housenumbers/train_32x32.mat', svhn_train, 'SVHN train')
else:
    print(f'SVHN train already exists at {svhn_train}', flush=True)

print('\nAll datasets ready.', flush=True)
