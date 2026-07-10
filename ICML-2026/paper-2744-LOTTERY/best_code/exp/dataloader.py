"""
Data loading utilities for LoTT experiments.

Datasets:
  - BLOB: 2D mixture of Gaussians (synthetic)
  - CIFAR-10: Adversarial detection with ResNet-18 / WRN-28 embeddings
  - Higgs: High-dimensional tabular physics data
"""

import torchvision.transforms as transforms
from torchvision import datasets
from sklearn.utils import check_random_state
import pickle
import torch
import numpy as np
import os
from functools import lru_cache


def check_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_data(dataset, N_ref, N_query, rs, check, need_labels=True, data_root=None, model_arch="Res18"):
    """
    Load reference (X) and query (Y) samples.

    Args:
        dataset: 'blob', 'cifar10', or 'higgs'
        N_ref: Number of reference samples
        N_query: Number of query samples
        rs: Random seed
        check: 1 = alternative (P != Q), 0 = null (P = Q)
        need_labels: Whether to return reference labels
        data_root: Path to data directory (default: ../data/)

    Returns:
        X, Y: Tensors of reference and query samples
        X_labels: Reference labels (or None)
    """
    if data_root is None:
        data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

    if 'blob' in dataset:
        X, Y, X_labels = sample_blob(N_ref, N_query, rs, check)
    elif 'higgs' in dataset:
        X, Y, X_labels = sample_higgs(N_ref, N_query, rs, check, data_root)
    elif 'cifar10' in dataset:
        X, Y, X_labels = load_cifar10(N_ref, N_query, rs, check, data_root, model_arch)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    if X.shape[1:] != Y.shape[1:]:
        raise ValueError(f"X and Y must have the same feature size, but got X: {X.shape[1:]}, Y: {Y.shape[1:]}")

    if need_labels:
        return X, Y, X_labels
    else:
        return X, Y


# ---------------------------------------------------------------------------
# BLOB (synthetic)
# ---------------------------------------------------------------------------

def create_grid(n_rows, n_cols):
    return np.array([[i, j] for i in range(n_rows) for j in range(n_cols)])


def create_cov_matrix(n_locs=9, variance=0.03, min_corr=0.02):
    n_side = n_locs // 2
    correlations = min_corr + np.arange(n_side) * 0.002
    correlations = np.concatenate([correlations[::-1] * -1, [0], correlations])
    return np.array([
        [[variance, corr], [corr, variance]]
        for corr in correlations
    ]).round(4)


def sample_blob(N_ref, N_query, rs, check, rows=3, cols=3, var=0.03, min_corr=0.02):
    mu = np.zeros(2)
    sigma = np.eye(2) * (var - 0.01)
    sigmas = create_cov_matrix(n_locs=rows * cols, variance=var, min_corr=min_corr)
    random_state = check_random_state(rs)

    X = random_state.multivariate_normal(mu, sigma, size=N_ref)
    X_row = random_state.randint(rows, size=N_ref)
    X_col = random_state.randint(cols, size=N_ref)
    X[:, 0] += X_row
    X[:, 1] += X_col
    X_labels = X_row * cols + X_col

    if check:
        Y = random_state.multivariate_normal(mu, np.eye(2), size=N_query)
        Y_row = random_state.randint(rows, size=N_query)
        Y_col = random_state.randint(cols, size=N_query)
        locs = create_grid(rows, cols)
        for i, loc in enumerate(locs):
            tgt_row, tgt_col = loc
            L = np.linalg.cholesky(sigmas[i])
            mask = (Y_row == tgt_row) & (Y_col == tgt_col)
            Y[mask] = Y[mask] @ L + loc
    else:
        Y = random_state.multivariate_normal(mu, sigma, size=N_query)
        Y[:, 0] += random_state.randint(rows, size=N_query)
        Y[:, 1] += random_state.randint(cols, size=N_query)

    return X, Y, X_labels


# ---------------------------------------------------------------------------
# CIFAR-10 (adversarial detection)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_cifar10_test(path):
    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=0)
    imgs, labels = next(iter(test_loader))
    return imgs.numpy(), labels.numpy()


@lru_cache(maxsize=1)
def _load_cifar10_train(path):
    transform_test = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.CIFAR10(root=path, train=True, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=0)
    imgs, labels = next(iter(train_loader))
    return imgs.numpy(), labels.numpy()


_CACHED_ADV = {}

def _load_cifar10_adv(path, model_arch):
    if model_arch in _CACHED_ADV:
        return _CACHED_ADV[model_arch]

    if model_arch == "Res18":
        adv_path = os.path.join(path, "Adv_cifar10_pgd_5_eps4_linf.npz")
    elif model_arch == "WRN28":
        adv_path = os.path.join(path, "Adv_cifar10_pgd_5_eps4_linf_transfer_wrn28.npz")
    else:
        raise ValueError(f"No adversarial data for model: {model_arch}")

    data = np.load(adv_path)
    adv = data['X_adv']
    original_labels = data['predicted_original_labels']
    predicted_labels = data['predicted_adv_labels']
    mask = (predicted_labels != original_labels)
    result = adv[mask]
    _CACHED_ADV[model_arch] = result
    return result


def load_cifar10(N_ref, N_query, rs, check, data_root, model_arch="Res18"):
    random_state = check_random_state(rs)
    path = os.path.join(data_root, "cifar10")

    X_all, X_labels_all = _load_cifar10_test(path)
    X_indices = random_state.choice(len(X_all), size=min(N_ref, len(X_all)), replace=False)
    X = X_all[X_indices]
    X_labels = X_labels_all[X_indices]

    if check:
        Y_all = _load_cifar10_adv(path, model_arch)
        Y_indices = random_state.choice(len(Y_all), size=min(N_query, len(Y_all)), replace=False)
        Y = Y_all[Y_indices]
    else:
        Y_all, _ = _load_cifar10_train(path)
        Y_indices = random_state.choice(len(Y_all), size=min(N_query, len(Y_all)), replace=False)
        Y = Y_all[Y_indices]

    return X, Y, X_labels


# ---------------------------------------------------------------------------
# Higgs
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_higgs_data(path):
    return pickle.load(open(path, "rb"))


def sample_higgs(N_ref, N_query, rs, check, data_root):
    torch.manual_seed(rs)
    higgs_path = os.path.join(data_root, "HIGGS_TST.pckl")
    data = _load_higgs_data(higgs_path)

    if check:
        X, Y = data[0], data[1]
    else:
        tmp = data[0]
        n = len(tmp)
        indices = np.random.choice(n, N_ref + N_query, replace=False)
        X = tmp[indices[:N_ref]]
        Y = tmp[indices[N_ref:]]

    return X[:N_ref], Y[:N_query], None
