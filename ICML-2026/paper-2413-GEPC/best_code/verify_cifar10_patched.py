from torchvision.datasets.cifar import CIFAR10

# Monkey-patch MD5 checksums to match our converted data
CIFAR10.train_list = [
    ('data_batch_1', 'd8e1be272e8d2aa6baa629b03bd04850'),
    ('data_batch_2', '546b6df25677e449b26997f8bd12bf97'),
    ('data_batch_3', 'ffd1f7961980d5ab3f92f3a80b96321d'),
    ('data_batch_4', '1255dfba9d2199366fb19ae52e41b745'),
    ('data_batch_5', 'b3acb8a16cb14777c6a54ab2715a9ded'),
]
CIFAR10.test_list = [
    ('test_batch', 'bb57009b7e00499af06325b0c595117d'),
]

# Verify
ds = CIFAR10('/datasets', train=True, download=False)
print(f'Train set: {len(ds)} samples')
print(f'First sample shape: {ds[0][0].shape}, label: {ds[0][1]}')

ds_test = CIFAR10('/datasets', train=False, download=False)
print(f'Test set: {len(ds_test)} samples')
print('All good!')
