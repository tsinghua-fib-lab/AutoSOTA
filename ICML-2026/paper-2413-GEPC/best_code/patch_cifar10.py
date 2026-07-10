import os, hashlib

# Compute MD5s of our pickle files and patch torchvision
files = {
    'data_batch_1': '/datasets/cifar-10-batches-py/data_batch_1',
    'data_batch_2': '/datasets/cifar-10-batches-py/data_batch_2',
    'data_batch_3': '/datasets/cifar-10-batches-py/data_batch_3',
    'data_batch_4': '/datasets/cifar-10-batches-py/data_batch_4',
    'data_batch_5': '/datasets/cifar-10-batches-py/data_batch_5',
    'test_batch': '/datasets/cifar-10-batches-py/test_batch',
}

for name, fpath in files.items():
    with open(fpath, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    print(f"('{name}', '{md5}'),")
