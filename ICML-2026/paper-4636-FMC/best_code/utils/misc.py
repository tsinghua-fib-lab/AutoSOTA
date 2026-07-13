import json
from math import nan
import re
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mlxp import ConfigDict
from mlxp.data_structures.dataframe import GroupedDataFrame
import numpy as np
import torch
from torch.nn.modules import Module
from torch.utils.data import DataLoader, TensorDataset, random_split


import torch.nn as nn


class SpikeAndSlabCauchyNoise(nn.Module):
    """
    Implements the spike-and-slab error model from Ward et al (2022):
       with probability (1−rho) you add Gaussian noise (spike),
       with probability rho you add Cauchy noise (slab).
    The noise is added to each dimension of the summary vector independently.
    """

    def __init__(self, rho: float, sigma: float = 0.01, tau: float = 0.25):
        """
        Args:
          rho   : probability of slab (Cauchy) component
          sigma : std dev of Gaussian (spike) component
          tau   : scale parameter of Cauchy (slab) component
        """
        super().__init__()
        self.rho = rho
        self.sigma = sigma
        self.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds noise to x according to the mixture model.
        Args:
          x : tensor of shape (batch_size, D) (or more dims where last dim = D)
        Returns:
          noisy_x : same shape as x
        """
        device = x.device
        shape = x.shape
        D = shape[-1]

        # Sample which dims use slab vs spike
        # mask = 1 means slab (Cauchy), 0 means spike (Gaussian)
        # Bernoulli per‐dimension
        mask = torch.bernoulli(torch.full((shape[:-1] + (D,)), self.rho, device=device))

        # Gaussian noise (spike)
        gaussian_noise = torch.randn_like(x) * self.sigma

        # Cauchy noise (slab): can be generated via torch.tan(pi*(u−0.5))*scale
        u = torch.rand_like(x)
        cauchy_noise = self.tau * torch.tan(torch.pi * (u - 0.5))

        # Compose
        noise = mask * cauchy_noise + (1.0 - mask) * gaussian_noise
        return x + noise


def random_psd(dim: int, scale: float = 1.0, seed: Optional[int] = None) -> torch.Tensor:
    """
    Generates a random symmetric positive semi-definite (PSD) matrix by symmetrizing a random matrix,
    removing the diagonal, and adding a scaled identity for stability.

    Args:
        dim: Dimension of the square matrix (D x D).
        scale: Standard deviation of the random values.
        seed: Optional random seed for reproducibility.

    Returns:
        A (dim, dim) PSD matrix.
    """
    if seed is not None:
        torch.manual_seed(seed)

    U = torch.randn(dim, dim) * scale
    U = torch.tril(U)  # Lower triangular matrix
    A = U + U.T
    A.fill_diagonal_(0.0)
    A += torch.eye(dim)
    return A


def rbf_kernel_matrix(X, lengthscale=1.0):
    """Compute kernel matrix and pairwise differences for X (N x D)."""
    N, D = X.shape
    X_sq = (X**2).sum(dim=1, keepdim=True)
    sq_dists = X_sq + X_sq.T - 2 * X @ X.T
    K = torch.exp(-0.5 * sq_dists / lengthscale**2)
    return K, sq_dists


def rbf_kernel_hessian(x1, x2, lengthscale=1.0):
    """
    Computes the mixed Hessian ∇_{x1} ∇_{x2}^T of the RBF kernel
    between each pair of points in x1 and x2.

    Args:
        x1: (N, D)
        x2: (M, D)
        lengthscale: float

    Returns:
        Hessian: (N, M, D, D) tensor
    """
    N, D = x1.shape
    M = x2.shape[0]
    x1_ = x1.unsqueeze(1)  # (N, 1, D)
    x2_ = x2.unsqueeze(0)  # (1, M, D)
    diff = x1_ - x2_  # (N, M, D)

    sq_dist = (diff**2).sum(dim=-1)  # (N, M)
    K = torch.exp(-0.5 * sq_dist / lengthscale**2)  # (N, M)

    # Outer product: (N, M, D, D)
    outer = diff.unsqueeze(-1) @ diff.unsqueeze(-2)  # (N, M, D, D)

    # Hessian
    H = (outer / lengthscale**4 - torch.eye(D, device=x1.device) / lengthscale**2) * K.unsqueeze(
        -1
    ).unsqueeze(-1)
    return H  # (N, M, D, D)


def tensor_to_df(tensor, label, dim_names: Optional[List[str]]) -> pd.DataFrame:
    if dim_names is not None:
        assert len(dim_names) == tensor.shape[-1]
        dict_val = {"type": label}
        for i in range(len(dim_names)):
            dict_val[dim_names[i]] = tensor[:, i].numpy()
    else:
        dict_val = {"type": label}
        for i in range(tensor.shape[1]):
            dict_val[f"dim_{i}"] = tensor[:, i].numpy()
    df = pd.DataFrame(dict_val)
    return df


def get_params(reader: GroupedDataFrame, field: str) -> Dict:
    """
    Return reader fields as dictonary.

    Args:
        reader: mlxp reader object
        field: String root e.g. 'config.task.simulator.params'

    Returns:
        params: Dictionary key,val with all the field.* value e.g. {'seed': 0, 'prior_scale': 5}
    """
    params = {}
    for v in reader.keys():
        if re.search(f"^{field}.", v):
            if str(reader[v]) == "nan":
                continue
            key = re.sub(f"{field}.", "", v)
            print(key, reader[v])
            # if key is of form 'field.subfield', split it
            if "model_config" in key:
                key_parts = key.split(".")
                if len(key_parts) == 2:
                    key = key_parts[0]
                    part = key_parts[1]
                    if key not in params.keys():
                        params[key] = {}
                    params[key][part] = reader[v]
                elif len(key_parts) == 3:
                    key = key_parts[0]
                    part1 = key_parts[1]
                    part2 = key_parts[2]
                    if key not in params.keys():
                        params[key] = {}
                    if part1 not in params[key].keys():
                        params[key][part1] = {}
                    params[key][part1][part2] = reader[v]
                else:
                    raise ValueError()
            else:
                value = reader[v]
                params[key] = value
    return params


def get_last_logid(logdir: Path):
    logid = 0
    for p in logdir.iterdir():
        if p.is_dir() and int(p.name) > logid:
            logid = int(p.name)
    return logid


def combine_loss_by_task(*loss_by_task_dicts):
    combined_loss_by_task = {}

    for loss_by_task in loss_by_task_dicts:
        for task_key, task_data in loss_by_task.items():
            if task_key not in combined_loss_by_task:
                combined_loss_by_task[task_key] = task_data
            else:
                for metric_type, metrics in task_data.items():
                    if metric_type not in combined_loss_by_task[task_key]:
                        combined_loss_by_task[task_key][metric_type] = metrics
                    else:
                        for metric_name, value in metrics.items():
                            if metric_name not in combined_loss_by_task[task_key][metric_type]:
                                combined_loss_by_task[task_key][metric_type][metric_name] = value

    return combined_loss_by_task


def save_or_merge_metrics(metrics: Dict[str, Any], path: Path) -> None:
    """
    Save or merge metrics to a given JSON file path.

    If the file exists, merge and overwrite existing metrics with the new ones.
    If it doesn't exist, save the new metrics directly.

    Args:
        metrics (dict): The new metrics to save.
        path (Path): Path to the metrics.json file.
    """
    if path.exists():
        with path.open("r") as f:
            existing = json.load(f)
    else:
        existing = {}

    def recursive_merge(source: Dict, target: Dict):
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                recursive_merge(value, target[key])
            else:
                target[key] = value

    recursive_merge(metrics, existing)

    with path.open("w") as f:
        json.dump(existing, f, indent=2)


def convert_tensors_to_float(nested_dict) -> Dict:
    """
    Recursively converts all PyTorch tensors in a nested dictionary to float values.

    Args:
        nested_dict (dict): A nested dictionary potentially containing PyTorch tensors.

    Returns:
        dict: A new dictionary with all tensors converted to float.
    """

    def to_float(value):
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                return value.item()
            elif value.dim() == 1:
                return value[0].float()
        elif isinstance(value, dict):
            return {k: to_float(v) for k, v in value.items()}  # Recurse if it's a dictionary
        elif isinstance(value, list):
            return to_float(value[0])  # Recurse if it's a list
        else:
            return float(value)  # Return value as is for other types

    return to_float(nested_dict)


def print_mem_cuda(prefix: str, device: torch.device) -> None:
    if device.type != "cuda":
        return None
    free, total = torch.cuda.mem_get_info(device)
    mem_used_MB = (total - free) / 1024**2
    print(prefix, mem_used_MB, "MB")


def t_stat_scorer(clf, X, y):
    # Get the probability estimates for the positive class
    proba = clf.predict_proba(X)[:, 1]  # Assuming binary classification
    # Compute the score
    score = np.mean((proba - 0.5) ** 2)
    return -1.0 * score


def create_train_val_loaders(x, batch_size, training_size, shuffle=True):
    """
    Create training and validation DataLoaders from data.

    Args:
        x (numpy.ndarray or torch.Tensor): Input data of shape (nsamples, dim).
        batch_size (int): Batch size for DataLoaders.
        training_size (float): Proportion of data to use for training (0 < training_size <= 1).
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        DataLoader: Training DataLoader.
        DataLoader: Validation DataLoader.
    """
    # Convert to tensor if needed
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    # Create a TensorDataset
    dataset = TensorDataset(x)

    # Compute split sizes
    total_size = len(dataset)
    train_size = int(training_size * total_size)
    val_size = total_size - train_size

    # Split the dataset into training and validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader


def loader_from_tensor(
    x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool = False
) -> Tuple[DataLoader, DataLoader]:
    xdset = TensorDataset(x)
    ydset = TensorDataset(y)
    xloader = DataLoader(xdset, batch_size=batch_size, shuffle=shuffle)
    yloader = DataLoader(ydset, batch_size=batch_size, shuffle=shuffle)
    return xloader, yloader


def train_val_split(
    theta: torch.Tensor, x: torch.Tensor, train_size: float, batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    theta_x_dset = TensorDataset(theta, x)
    train, val = random_split(theta_x_dset, [train_size, 1 - train_size])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_val_split_n(
    *tensors: torch.Tensor, train_size: float = 0.8, batch_size: int = 256
) -> Tuple[DataLoader, DataLoader]:
    """
    Splits an arbitrary number of tensors into train and validation DataLoaders.

    Args:
        *tensors: any number of torch.Tensors, each of the same first dimension (N).
        train_size (float): fraction of the dataset to use for training (e.g., 0.8).
        batch_size (int): batch size for DataLoaders.

    Returns:
        (train_loader, val_loader): Tuple of DataLoaders
    """
    if len(tensors) == 0:
        raise ValueError("At least one tensor must be provided.")

    # Check that all tensors have the same first dimension
    n_samples = tensors[0].size(0)
    for tensor in tensors:
        if tensor.size(0) != n_samples:
            raise ValueError(
                "All input tensors must have the same number of samples in the first dimension."
            )

    # Create dataset
    dataset = TensorDataset(*tensors)

    # Compute sizes
    n_train = int(train_size * n_samples)
    n_val = n_samples - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_fm_data(
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    embedding_net: Optional[Module],
    space: str,
    batch_size: int,
    rescale: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if embedding_net is not None and space == "latent":
        xlist, ylist = [], []
        xloader, yloader = loader_from_tensor(x, y, batch_size)
        for (x_batch,), (y_batch,) in zip(xloader, yloader):
            x = embedding_net(x_batch.to(device))
            y = embedding_net(y_batch.to(device))
            xlist.append(x.detach().cpu())
            ylist.append(y.detach().cpu())
        xemb = torch.cat(xlist, dim=0)
        yemb = torch.cat(ylist, dim=0)
    elif space == "data":
        xemb = x
        yemb = y
    else:
        raise ValueError("Invalid space for flow matching")
    if rescale and space == "data":
        xemb = (xemb - xemb.mean(0)) / xemb.std(0)
        yemb = (yemb - yemb.mean(0)) / yemb.std(0)
    return xemb, yemb


def whiten_tensor_pytorch(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Whitens a 2D PyTorch tensor (n_samples, n_features).

    Args:
        tensor (torch.Tensor): Input tensor of shape (n_samples, n_features).

    Returns:
        torch.Tensor: Whitened tensor of the same shape as the input.
    """
    # Ensure the input tensor is 2D
    if tensor.ndim != 2:
        raise ValueError("Input tensor must be 2D (n_samples, n_features).")

    # Step 1: Center the data (zero mean)
    mean = torch.mean(tensor, dim=0)
    centered_tensor = tensor - mean

    # Step 2: Compute covariance matrix
    cov_matrix = torch.mm(centered_tensor.T, centered_tensor) / (tensor.size(0) - 1)

    # Step 3: Eigenvalue decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

    # Step 4: Whiten the data
    whitening_matrix = (
        eigenvectors @ torch.diag(1.0 / torch.sqrt(eigenvalues + 1e-6)) @ eigenvectors.T
    )
    whitened_tensor = torch.mm(centered_tensor, whitening_matrix)

    return whitened_tensor, mean, whitening_matrix


def rescale(tensor: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Rescale a tensor using the given mean and standard deviation.
    Args:
        tensor (torch.Tensor): Input tensor.
        mean (torch.Tensor): Mean to rescale the tensor.
        scale (torch.Tensor): Standard deviation to rescale the tensor.
    Returns:
        torch.Tensor: Rescaled tensor.
    """
    if mean is None:
        return tensor
    elif scale.ndim == 1:
        return (tensor * scale.to(tensor.device)) + mean.to(tensor.device)
    elif scale.ndim == 2:
        return tensor @ torch.linalg.inv(scale).to(tensor.device) + mean.to(tensor.device)
    else:
        raise ValueError(f"Invalid scale tensor dimensions.{scale.ndim}")


def scale(tensor: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Reverse the rescaling of a tensor using the given mean and standard deviation.
    Args:
        tensor (torch.Tensor): Input tensor.
        mean (torch.Tensor): Mean used during rescaling.
        scale (torch.Tensor): Standard deviation or scale matrix used during rescaling.
    Returns:
        torch.Tensor: Scaled tensor (original space before rescaling).
    """
    centered = tensor - mean.to(tensor.device)
    if scale.ndim == 1:
        return centered / scale.to(tensor.device)
    elif scale.ndim == 2:
        return centered @ scale.to(tensor.device)
    else:
        raise ValueError(f"Invalid scale tensor dimensions: {scale.ndim}")


def find_closest_sample(x: torch.Tensor, samples: torch.Tensor, n_closest: int) -> torch.Tensor:
    """
    Find the n_closest samples in the samples tensor to the input x tensor.
    Args:
        x (torch.Tensor): Input tensor.
        samples (torch.Tensor): Samples tensor.
        n_closest (int): Number of closest samples to find.
    Returns:
        torch.Tensor: Closest samples tensor.
    """
    distances = torch.cdist(x, samples)
    closest_indices = torch.argsort(distances, dim=1)[:, :n_closest]
    closest_samples = samples[closest_indices]
    return closest_samples


def merge_loss_json(folder1, folder2, output_folder):
    """
    Merge JSON files from two folders based on metric types.
    Extracts 'c2st' metrics from folder1 and 'lpp' metrics from folder2,
    then combines them into new JSON files in the output folder.

    Args:
        folder1 (str or Path): Path to the first folder containing JSON files with 'c2st' metrics.
        folder2 (str or Path): Path to the second folder containing JSON files with 'lpp' metrics.
        output_folder (str or Path): Path to the folder where merged JSON files will be saved.
    """
    folder1, folder2, output_folder = Path(folder1), Path(folder2), Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Collect all JSON files in folder1 and folder2
    json_files1 = {f.name: f for f in folder1.glob("loss_by_task_*.json")}
    json_files2 = {f.name: f for f in folder2.glob("loss_by_task_*.json")}

    # Get the union of file names in both folders
    all_files = set(json_files1.keys()).union(json_files2.keys())

    for file_name in all_files:
        combined_data = {}

        # Load data from folder1 (C2ST metrics)
        if file_name in json_files1:
            with open(json_files1[file_name], "r") as f:
                data1 = json.load(f)

            # Filter for C2ST-related fields
            for task, task_data in data1.items():
                combined_data.setdefault(task, {})
                for cal_samples, metrics in task_data.items():
                    combined_data[task].setdefault(cal_samples, {})
                    for loss_name, loss_data in metrics.items():
                        if "c2st" in loss_name.lower():  # Keep only C2ST-related metrics
                            combined_data[task][cal_samples][loss_name] = loss_data

        # Load data from folder2 (LPP metrics)
        if file_name in json_files2:
            with open(json_files2[file_name], "r") as f:
                data2 = json.load(f)

            # Filter for LPP-related fields
            for task, task_data in data2.items():
                combined_data.setdefault(task, {})
                for cal_samples, metrics in task_data.items():
                    combined_data[task].setdefault(cal_samples, {})
                    for loss_name, loss_data in metrics.items():
                        if "lpp" in loss_name.lower():  # Keep only LPP-related metrics
                            combined_data[task][cal_samples][loss_name] = loss_data

        # Save the merged JSON file
        output_path = output_folder / file_name
        with open(output_path, "w") as f:
            json.dump(combined_data, f, indent=4)

        print(f"Merged file saved: {output_path}")
