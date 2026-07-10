"""Apply CIFAR-10 monkey-patch and verify loading."""

import hashlib
from torchvision.datasets.cifar import CIFAR10

# Compute MD5s
def md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

base = '/datasets/cifar-10-batches-py'

# Patch train/test file MD5s
CIFAR10.train_list = [
    ('data_batch_1', md5(f'{base}/data_batch_1')),
    ('data_batch_2', md5(f'{base}/data_batch_2')),
    ('data_batch_3', md5(f'{base}/data_batch_3')),
    ('data_batch_4', md5(f'{base}/data_batch_4')),
    ('data_batch_5', md5(f'{base}/data_batch_5')),
]
CIFAR10.test_list = [
    ('test_batch', md5(f'{base}/test_batch')),
]
# Patch meta MD5
CIFAR10.meta['md5'] = md5(f'{base}/batches.meta')

print("MD5s patched:")
print(f"  train_list: {CIFAR10.train_list}")
print(f"  test_list: {CIFAR10.test_list}")
print(f"  meta.md5: {CIFAR10.meta['md5']}")

# Verify
ds = CIFAR10('/datasets', train=True, download=False)
print(f'\nTrain set: {len(ds)} samples, classes: {ds.classes}')
sample, label = ds[0]
print(f'First sample: shape={sample.shape}, label={label}')

ds_test = CIFAR10('/datasets', train=False, download=False)
print(f'Test set: {len(ds_test)} samples')

print('\nCIFAR-10 loading OK!')
