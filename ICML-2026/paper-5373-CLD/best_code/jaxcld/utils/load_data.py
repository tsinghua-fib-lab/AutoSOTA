"""
Datasets to load (MNIST, FMNIST, CIFAR-10, FOOD, IMAGENET, Synthetic)
JAXv3, all datatypes hard coded to load only in jax and all labels +-1
DO NOT JIT
"""
import os
import numpy as np
from numpy.random import randn
from math import ceil
import pickle, gzip
import pandas as pd
from os.path import dirname, join, abspath
#from scipy.stats import ortho_group 
import os
from PIL import Image
import math
from jax.scipy.linalg import qr
import sys

import jax
import jax.numpy as jnp
#import numpy as np
from jax import grad, jit, vmap,  pmap
from jax import random
from jax import lax
from jax import make_jaxpr 
from jax import config
from jax import device_put

CPUS = jax.devices("cpu")
GPUS = jax.devices("gpu")

key = jax.random.PRNGKey(2024)

#fmnist 
def load_fmnist(dataset_rel_path=join('datasets', 'Fashion MNIST'), 
                n=1000,
                binary_classes=False, 
                downsample=False, 
                stride=3, 
                normalize=True):
    
    project_root = dirname(abspath('content'))
    path = join(project_root, dataset_rel_path)
    training_labels_path = join(path, 'train-labels-idx1-ubyte')
    training_images_path = join(path, 'train-images-idx3-ubyte')
    test_labels_path = join(path, 't10k-labels-idx1-ubyte')
    test_images_path = join(path, 't10k-images-idx3-ubyte')

    with open(training_labels_path, 'rb') as training_lbpath:
        training_y = np.frombuffer(training_lbpath.read(),
                                   dtype=np.uint8, offset=8)
    with open(training_images_path, 'rb') as training_imgpath:
        X_training_raw = np.frombuffer(training_imgpath.read(),
                                       dtype=np.uint8, offset=16).reshape(len(training_y), 784)
    with open(test_labels_path, 'rb') as test_lbpath:
        test_y = np.frombuffer(test_lbpath.read(),
                               dtype=np.uint8, offset=8)
    with open(test_images_path, 'rb') as test_imgpath:
        X_test_raw = np.frombuffer(test_imgpath.read(),
                                   dtype=np.uint8, offset=16).reshape(len(test_y), 784)
    dim = ceil(28 / stride)
    
    if binary_classes:
      training_mask = (training_y == 2) | (training_y == 8)
      X_training_raw = X_training_raw[training_mask, :]
      training_y = np.sign(training_y[training_mask] - 5.)
    
    if downsample:
        training_X = np.zeros([training_y.size, dim ** 2])
        for i in range(training_y.size):
                  x = X_training_raw[i, :].reshape([28, 28])
                  x = x[::stride, ::stride]
                  training_X[i, :] = x.reshape(dim ** 2)
    else:
        training_X = X_training_raw

    if binary_classes:
      test_mask = (test_y == 2) | (test_y == 8)
      X_test_raw = X_test_raw[test_mask, :]
      test_y = np.sign(test_y[test_mask] - 5.)
    
    if downsample:
        test_X = np.zeros([test_y.size, dim ** 2])
        for i in range(test_y.size):
            x = X_test_raw[i, :].reshape([28, 28])
            x = x[::stride, ::stride]
            test_X[i, :] = x.reshape(dim ** 2)
    else:
        test_X = X_test_raw

    training_X = training_X[:n, :]
    training_y = training_y[:n]

    if normalize:
        X_mean, X_std = training_X.mean(), training_X.std()
        training_X = (training_X - X_mean) / (X_std + 1e-10)
        test_X = (test_X - X_mean) / (X_std + 1e-10)
    return training_X, training_y.astype(int), test_X, test_y.astype(int)

# CIFAR-10
# labels in binary are -1 and 1
def load_cifar(classes,
               dataset_rel_path = join('datasets', 'cifar-10-batches-py'), 
               n=50000, 
               downsample=False, 
               binary_classes=True,
               stride=3, 
               normalize=True):
    
    project_root = dirname(abspath('content'))
    path = join(project_root, dataset_rel_path)
    print('path to datasets is = ', path)
    #exit()
    dim = ceil(32 / stride)

    dats_training = []
    for i in range(1, 6):
        training_file_name = 'data_batch_' + str(i)
        with open(join(path, training_file_name), 'rb') as ftrain:
            dats_training += [pickle.load(ftrain, encoding='latin1')]
    X_training_raw = jnp.concatenate([jnp.array(dat_training['data'])
                                     for dat_training in dats_training], axis=0)  # (50000, 3072)
    # X_training_raw = X_training_raw.reshape(10000, 3, 32, 32)
    training_y = jnp.concatenate([jnp.array(dat_training['labels'])
                                 for dat_training in dats_training], axis=0)  # (50000,)
    
    if binary_classes:
        training_mask = (training_y == classes[0]) | (training_y == classes[1])
        X_training_raw = X_training_raw[training_mask, :]
        training_y = training_y[training_mask]
        Idx = np.where(training_y==classes[0])[0]
        Jdx = np.where(training_y==classes[1])[0]
        training_y = training_y.at[Idx].set(0)
        training_y = training_y.at[Jdx].set(1)

    if downsample:
        training_X = jnp.zeros([training_y.size, 3 * (dim ** 2)])
        for i in range(training_y.size):
            x = X_training_raw[i, :].reshape([32, 32, 3])
            x = x[::stride, ::stride, :]
            training_X[i, :] = x.reshape(3 * (dim ** 2))
    else:
        training_X = X_training_raw

    test_file_name = 'test_batch'
    with open(join(path, test_file_name), 'rb') as ftest:
        dat_test = pickle.load(ftest, encoding='latin1')
    images = dat_test['data']
    labels = dat_test['labels']
    X_test_raw = jnp.array(images)  # (10000, 3072)
    test_y = jnp.array(labels)  # (10000,)

    if binary_classes:
        test_mask = (test_y == classes[0]) | (test_y == classes[1])
        X_test_raw = X_test_raw[test_mask, :]
        #test_y = np.sign(test_y[test_mask] - 0.5)
        test_y = test_y[test_mask]
        Idx = np.where(test_y==classes[0])[0]
        Jdx = np.where(test_y==classes[1])[0]
        test_y = test_y.at[Idx].set(0)
        test_y = test_y.at[Jdx].set(1)
        
    if downsample:
        test_X = jnp.zeros([test_y.size, 3 * (dim ** 2)])
        for i in range(test_y.size):
            x = X_test_raw[i, :].reshape([32, 32, 3])
            x = x[::stride, ::stride, :]
            test_X[i, :] = x.reshape(3 * (dim ** 2))
    else:
        test_X = X_test_raw

    training_X = training_X[:n, :]
    training_y = training_y[:n]

    if normalize:
        #X_mean, X_std = training_X.mean(), training_X.std()
        #training_X = (training_X - X_mean) / (X_std + 1e-10)
        training_X = training_X/(255)
        test_X = test_X/(255)
        #test_X = (test_X - X_mean) / (X_std + 1e-10)
    
    # # convert to jax arrays
    # training_X = jnp.asarray(training_X)
    # training_y = jnp.asarray(training_y)
    # test_X = jnp.asarray(test_X)
    # test_y = jnp.asarray(test_y)
    #print("types of input for cifar is currently :")
    print(type(training_X), type(training_y), type(test_X), type(test_y))
    #exit()
    # print(test_y)
    # exit()
    return training_X, training_y, test_X, test_y



# # preprocess the already loaded CIFAR10
# # only used if data is loaded another way except for our custom load_cifar()
# def preprocess_cifar10(X_train=[], 
#                      X_test=[], 
#                      Y_train=[], 
#                      Y_test=[], 
#                      n_train=10000,
#                      n_test=1000, 
#                      downsample=False, 
#                      binary_classes=True,
#                      stride=3, 
#                      normalize=True):
    
#     assert not (len(X_train) == 0 or len(X_test) == 0 or len(Y_train) == 0 or len(Y_test) == 0), "Must provide valid train/test CIFAR data. Retrieve using (X_train,Y_train),(X_test,Y_test) = keras.datasets.cifar10.load_data()"

#     # limit classes only to class 0 and 1
#     if binary_classes:
#         training_mask = (Y_train == 0) | (Y_train == 1)
#         training_mask = training_mask.flatten()
#         X_train = X_train[training_mask, :, :, :]
#         Y_train = np.sign(Y_train[training_mask] - 0.5)

#         test_mask = (Y_test == 0) | (Y_test == 1)
#         test_mask = test_mask.flatten()
#         X_test = X_test[test_mask, :]
#         Y_test = np.sign(Y_test[test_mask] - 0.5)

#     # downsample images to 1/stride of their pixels
#     if downsample:
#         dim = ceil(32 / stride)
#         X_train_proc = np.zeros([Y_train.size, 3 * (dim ** 2)])
#         for i in range(Y_train.size):
#             x = X_train[i, :][::stride, ::stride, :]
#             X_train_proc[i, :] = x.reshape(3 * (dim ** 2))

#         X_test_proc = np.zeros([Y_test.size, 3 * (dim ** 2)])
#         for i in range(Y_test.size):
#             x = X_test[i, :][::stride, ::stride, :]
#             X_test_proc[i, :] = x.reshape(3 * (dim ** 2))
#     else:
#         X_train_proc = X_train.reshape(3 * 32**2)
#         X_test_proc = X_test.reshape(3 * 32**2)

#     X_train_proc = X_train_proc[:n_train, :]
#     Y_train = Y_train[:n_train]

#     X_test_proc = X_test_proc[:n_test, :]
#     Y_test = Y_test[:n_test]

#     if normalize:
#         X_mean, X_std = X_train_proc.mean(), X_train_proc.std()
#         X_train_proc = (X_train_proc - X_mean) / (X_std + 1e-10)
#         X_test_proc = (X_test_proc - X_mean) / (X_std + 1e-10)

#     return X_train_proc, Y_train.squeeze(), X_test_proc, Y_test.squeeze()



def load_images_and_labels(folder_path):
    image_vectors = []
    labels = []

    valid_extensions = ('.png', '.jpg', '.gif', '.jpeg')
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(valid_extensions):
            if filename[0] in ('1'):
                labels.append(int(filename[0]))
                with Image.open(join(folder_path, filename)) as img:
                    image_vectors.append(jnp.array(img).reshape(-1))
            elif filename[0] in ('0'):
                labels.append(int(-1))
                with Image.open(join(folder_path, filename)) as img:
                    image_vectors.append(jnp.array(img).reshape(-1))
            else:
                print(f'Error: filename {filename} does not start with 0 or 1')
    
    return jnp.stack(image_vectors), jnp.array(labels)


# # FOOD 256x256x3 or Food-11 512x512x3
# # labels are -1 or +1
def load_food(dataset_rel_path=join('datasets', 'Food_cleaned'), 
                    n=3000, 
                    downsample=False, 
                    binary_classes=True,
                    stride=3, 
                    normalize=True):
    project_root = dirname(abspath(''))
    folder_path_train = join(project_root, dataset_rel_path, 'training')
    folder_path_val = join(project_root, dataset_rel_path, 'validation')

    X_train, Y_train = load_images_and_labels(folder_path_train)
    X_val, Y_val = load_images_and_labels(folder_path_val)
    
    # standardize the data
    X_mean, X_std = X_train.mean(), X_train.std()
    X_train = (X_train - X_mean) / (X_std + 1e-10)

    X_mean, X_std = X_val.mean(), X_val.std()
    X_val = (X_val - X_mean) / (X_std + 1e-10)

    permuted_indices = jax.random.permutation(key, len(Y_train))
    permuted_indices2 = jax.random.permutation(key, len(Y_val))

    X_train = X_train[permuted_indices, :]
    Y_train = Y_train[permuted_indices]
    X_val = X_val[permuted_indices2, :]
    Y_val = Y_val[permuted_indices2]

    return X_train, Y_train, X_val, Y_val



# loads chicken or shark classes from Imagenet (https://www.kaggle.com/datasets/ambityga/imagenet100)
# labels are -1=chicken or 1=shark
# downsampling is down outside of this function, to 171x171x3
def load_imagenet(dataset_rel_path=join('datasets', 'imagenet100'), 
                    n=2600, 
                    downsample=False, 
                    binary_classes=True,    
                    stride=3, 
                    normalize=True): # todo: add normalize
    project_root = dirname(abspath(''))
    folder_path_train = join(project_root, dataset_rel_path, 'training')
    #print(folder_path_train)
    folder_path_val = join(project_root, dataset_rel_path, 'validation')
    #print(folder_path_val)
    X_train, Y_train = load_images_and_labels(folder_path_train)
    X_val, Y_val = load_images_and_labels(folder_path_val)

    # standardize the data
    X_mean, X_std = X_train.mean(), X_train.std()
    X_train = (X_train - X_mean) / (X_std + 1e-10)

    X_mean, X_std = X_val.mean(), X_val.std()
    X_val = (X_val - X_mean) / (X_std + 1e-10)

    permuted_indices = jax.random.permutation(key, len(Y_train))
    permuted_indices2 = jax.random.permutation(key, len(Y_val))

    X_train = X_train[permuted_indices, :]
    Y_train = Y_train[permuted_indices]
    X_val = X_val[permuted_indices2, :]
    Y_val = Y_val[permuted_indices2]
    

    # print(Y_train)
    # exit()
    return X_train, Y_train, X_val, Y_val

# loads chicken or shark classes from Imagenet
# labels are -1=chicken or 1=shark
# size 512x512x3
def load_imagenet512(dataset_rel_path=join('datasets', 'imagenet_cleaned'), 
                    n=2600, 
                    downsample=False, 
                    binary_classes=True,    
                    stride=3, 
                    normalize=True): # todo: add normalize
    project_root = dirname(abspath(''))
    folder_path_train = join(project_root, dataset_rel_path, 'training')
    #print(folder_path_train)
    folder_path_val = join(project_root, dataset_rel_path, 'validation')
    #print(folder_path_val)
    X_train, Y_train = load_images_and_labels(folder_path_train)
    X_val, Y_val = load_images_and_labels(folder_path_val)

    # standardize the data
    X_mean, X_std = X_train.mean(), X_train.std()
    X_train = (X_train - X_mean) / (X_std + 1e-10)

    X_mean, X_std = X_val.mean(), X_val.std()
    X_val = (X_val - X_mean) / (X_std + 1e-10)

    permuted_indices = jax.random.permutation(key, len(Y_train))
    permuted_indices2 = jax.random.permutation(key, len(Y_val))

    X_train = X_train[permuted_indices, :]
    Y_train = Y_train[permuted_indices]
    X_val = X_val[permuted_indices2, :]
    Y_val = Y_val[permuted_indices2]
    

    # print(Y_train)
    # exit()
    return X_train, Y_train, X_val, Y_val


# # synthetic data generation function adapted from https://github.com/pilancilab/scnn
# def load_synthetic(n: int, d: int, hidden_units: int = 50, kappa: float = 1.0):
#     """Create a binary classification dataset with a random Gaussian design matrix."""
#     key = random.PRNGKey(0)  # Replace 0 with your chosen seed

#     # Generate random weights
#     key, subkey = random.split(key)
#     w1 = random.normal(subkey, (hidden_units, d))
#     key, subkey = random.split(key)
#     w2 = random.normal(subkey, (1, hidden_units))

#     # Generate covariance matrix
#     Sigma = sample_covariance_matrix(subkey, d, kappa)  # Ensure this function is JAX compatible

#     # Initialize lists to collect data
#     X, y = [], []
#     n_pos, n_neg = 0, 0
#     n_total = n

#     # Simple rejection sampling
#     while n_pos + n_neg < n_total:
#         key, subkey = random.split(key)
#         xi = random.multivariate_normal(subkey, jnp.zeros(d), Sigma)
#         # Compute forward pass
#         yi = jnp.maximum(xi @ w1.T, 0) @ w2.T
#         if yi <= 0 and n_neg < math.ceil(n_total / 2):
#             y.append(-1)
#             X.append(xi)
#             n_neg += 1
#         elif yi > 0 and n_pos < math.ceil(n_total / 2):
#             y.append(1)
#             X.append(xi)
#             n_pos += 1

#     # Convert lists to JAX arrays
#     X_jax = jnp.array(X)
#     y_jax = jnp.array(y).reshape(-1, 1)

#     # Shuffle dataset
#     key, subkey = random.split(key)
#     indices = random.permutation(subkey, n_total)
#     X_jax, y_jax = X_jax[indices], y_jax[indices]

#     # Split into training and test sets
#     train_set = (X_jax[:n], y_jax[:n])
#     test_set = (X_jax[n:], y_jax[n:])

#     return train_set[0], train_set[1], test_set[0], test_set[1]


# def sample_covariance_matrix(key: random.PRNGKey, d: int, kappa: float) -> jnp.ndarray:
#     """Sample a covariance matrix with a specific condition number using JAX.

#     Args:
#         key: a JAX random number generator key.
#         d: the dimensionality of the covariance matrix.
#         kappa: condition number of the covariance matrix.

#     Returns:
#         A (d, d) matrix :math:`\\Sigma` with condition number `kappa`.
#     """
#     # Sample a random matrix and perform QR decomposition to get an orthonormal matrix Q
#     key, subkey = random.split(key)
#     A = random.normal(subkey, (d, d))
#     Q, _ = qr(A)

#     # Sample eigenvalues so that lambda_1 / lambda_d = kappa.
#     key, subkey = random.split(key)
#     eigs = random.uniform(subkey, (d - 2,), minval=1, maxval=kappa)
#     eigs = jnp.concatenate([jnp.array([kappa, 1]), eigs])

#     # Compute covariance
#     Sigma = Q.T @ jnp.diag(eigs) @ Q

#     return Sigma
