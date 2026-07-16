"""Spectrum analysis and covariance tracking utilities for ablation experiments."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from .types import Tensor, CovarianceDict
from .recorder import KFACRecorder
from .covariance import compute_covariances, CovariancePair
from .optimizer import generate_pink_noise


def compute_kfac_eigenvalues(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_batches: int = 10,
    num_classes: int = 10,
    noise_type: Optional[str] = None,
    input_shape: Optional[Tuple[int, ...]] = None,
) -> Dict[str, np.ndarray]:
    """Compute eigenvalues of KFAC approximation (A and G) per layer.

    Args:
        noise_type: None for real data, 'white' or 'pink' for synthetic.
        input_shape: Required if noise_type is not None.

    Returns dict mapping layer_name -> {
        'eig_A': sorted eigenvalues of A (descending),
        'eig_G': sorted eigenvalues of G (descending),
    }
    """
    target = model._module if hasattr(model, "_module") else model
    recorder = KFACRecorder(model, only_trainable=False)
    recorder.enable()

    all_A: Dict[str, List[Tensor]] = {}
    all_G: Dict[str, List[Tensor]] = {}

    model.eval()
    data_iter = iter(data_loader)
    for _ in range(num_batches):
        model.zero_grad()

        if noise_type is None:
            try:
                data, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                data, labels = next(data_iter)
            data, labels = data.to(device), labels.to(device)
        else:
            assert input_shape is not None
            batch_size = data_loader.batch_size or 256
            if noise_type == "pink":
                data = generate_pink_noise(batch_size, input_shape, device)
            else:
                from .methods import generate_white_noise
                data = generate_white_noise(batch_size, input_shape, device)
            labels = torch.randint(0, num_classes, (batch_size,), device=device)

        output = target(data)
        loss = F.cross_entropy(output, labels)
        loss.backward()

        cov = compute_covariances(model, recorder.activations, recorder.backprops)
        for name in cov.A:
            all_A.setdefault(name, []).append(cov.A[name])
        for name in cov.G:
            all_G.setdefault(name, []).append(cov.G[name])
        recorder.clear()

    recorder.disable()
    recorder.remove()
    model.zero_grad()

    result: Dict[str, Dict[str, np.ndarray]] = {}
    for name in all_A:
        avg_A = torch.stack(all_A[name]).mean(dim=0)
        avg_G = torch.stack(all_G[name]).mean(dim=0)

        eig_A = torch.linalg.eigvalsh(avg_A).flip(0).cpu().numpy()
        eig_G = torch.linalg.eigvalsh(avg_G).flip(0).cpu().numpy()

        result[name] = {"eig_A": eig_A, "eig_G": eig_G}

    return result


def compute_condition_number(evals: np.ndarray, eps: float = 1e-10) -> float:
    """Compute condition number from eigenvalue array."""
    pos = evals[evals > eps]
    if len(pos) < 2:
        return float("inf")
    return float(pos[0] / pos[-1])


def compute_covariance_similarity(
    cov_source: CovariancePair,
    cov_target: CovariancePair,
) -> Dict[str, Dict[str, float]]:
    """Compare covariance matrices using cosine similarity and relative Frobenius norm.

    Returns per-layer dict with keys:
        'cos_sim_A', 'cos_sim_G': cosine similarity (1.0 = identical)
        'frob_rel_A', 'frob_rel_G': relative Frobenius norm distance
    """
    result: Dict[str, Dict[str, float]] = {}

    for name in cov_source.A:
        if name not in cov_target.A:
            continue

        src_A = cov_source.A[name].flatten().float()
        tgt_A = cov_target.A[name].flatten().float()
        cos_A = F.cosine_similarity(src_A.unsqueeze(0), tgt_A.unsqueeze(0)).item()
        frob_A = (src_A - tgt_A).norm().item() / (tgt_A.norm().item() + 1e-10)

        src_G = cov_source.G[name].flatten().float()
        tgt_G = cov_target.G[name].flatten().float()
        cos_G = F.cosine_similarity(src_G.unsqueeze(0), tgt_G.unsqueeze(0)).item()
        frob_G = (src_G - tgt_G).norm().item() / (tgt_G.norm().item() + 1e-10)

        result[name] = {
            "cos_sim_A": cos_A,
            "cos_sim_G": cos_G,
            "frob_rel_A": frob_A,
            "frob_rel_G": frob_G,
        }

    return result


def track_covariances_epoch(
    model: nn.Module,
    private_loader: DataLoader,
    public_loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
    num_batches: int = 10,
    input_shape: Optional[Tuple[int, ...]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute covariance similarity snapshot for one epoch.

    Compares oracle (private) vs public and synthetic pink noise sources.
    Returns dict with keys 'public' and 'pink_noise', each containing
    per-layer similarity metrics.
    """
    target = model._module if hasattr(model, "_module") else model

    def _compute_covs(loader, use_noise=False):
        recorder = KFACRecorder(model, only_trainable=False)
        recorder.enable()
        cov_list = []
        data_iter = iter(loader)

        for _ in range(num_batches):
            model.zero_grad()
            if use_noise:
                assert input_shape is not None
                bs = loader.batch_size or 256
                data = generate_pink_noise(bs, input_shape, device)
                labels = torch.randint(0, num_classes, (bs,), device=device)
            else:
                try:
                    data, labels = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    data, labels = next(data_iter)
                data, labels = data.to(device), labels.to(device)

            output = target(data)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            cov = compute_covariances(model, recorder.activations, recorder.backprops)
            cov_list.append(cov)
            recorder.clear()

        recorder.disable()
        recorder.remove()
        model.zero_grad()

        # Average
        avg_A: CovarianceDict = {}
        avg_G: CovarianceDict = {}
        for name in cov_list[0].A:
            avg_A[name] = torch.stack([c.A[name] for c in cov_list]).mean(0)
            avg_G[name] = torch.stack([c.G[name] for c in cov_list]).mean(0)
        return CovariancePair(A=avg_A, G=avg_G)

    model.eval()
    oracle_cov = _compute_covs(private_loader)
    public_cov = _compute_covs(public_loader)
    pink_cov = _compute_covs(private_loader, use_noise=True)

    return {
        "public": compute_covariance_similarity(public_cov, oracle_cov),
        "pink_noise": compute_covariance_similarity(pink_cov, oracle_cov),
    }
