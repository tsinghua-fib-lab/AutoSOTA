"""
CaPS backend interface wrapper
Wraps the CaPS algorithm into a unified interface for easy OCMB invocation
"""
import sys
import os
import numpy as np
import torch
from pathlib import Path

# Import utils module (in the same directory)
from .utils import *


def Stein_hess(X, eta_G, eta_H, s=None, device="cuda:0", auto_tune_s=False, lambda_ksd=0.0):
    """
    Estimate diagonal of Hessian of log p_X using first-order and second-order Stein identity

    Parameters
    ----------
    X : torch.Tensor, shape (n_samples, n_nodes)
        Observed data
    eta_G : float
        First-order regularization parameter
    eta_H : float
        Second-order regularization parameter
    s : float, optional
        Kernel bandwidth, automatically computed if None
    device : str
        Computing device

    Returns
    -------
    H : torch.Tensor, shape (n_samples, n_nodes)
        Hessian diagonal estimate
    """
    n, d = X.shape
    X = X.to(device)
    X_diff = X.unsqueeze(1) - X

    if s is None:
        D = torch.norm(X_diff, dim=2, p=2)
        s = D.flatten().median()

        # Auto-tune kernel bandwidth (CODE-04)
        if auto_tune_s:
            multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
            best_s = s
            best_loss = float("inf")
            for mult in multipliers:
                sc = s * mult
                Kc = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * sc**2)) / sc
                nKc = -torch.einsum("kij,ik->kj", X_diff, Kc) / sc**2
                Ic = Kc + eta_G * torch.eye(n).to(device)
                Gc = torch.matmul(torch.inverse(Ic), nKc)
                loss_c = torch.norm(Gc, p="fro").item()
                if loss_c < best_loss:
                    best_loss = loss_c
                    best_s = sc
            s = best_s

    K = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * s**2)) / s

    nablaK = -torch.einsum('kij,ik->kj', X_diff, K) / s**2
    G = torch.matmul(torch.inverse(K + eta_G * torch.eye(n).to(device)), nablaK)

    nabla2K = torch.einsum('kij,ik->kj', -1/s**2 + X_diff**2/s**4, K)
    H = (-G**2 + torch.matmul(torch.inverse(K + eta_H * torch.eye(n).to(device)), nabla2K))
    
    # KSD regularization (ALGO-03): sharpen Hessian via Kernel Stein Discrepancy
    if lambda_ksd > 0:
        K_inv = torch.inverse(K + eta_G * torch.eye(n).to(device))
        ksd_val = (K @ G).mean().item() + torch.trace(K_inv).item() / n - 1.0
        H = H - lambda_ksd * ksd_val

    return H.to('cpu')


def get_parents_score_from_hessian(curr_H, full_H, i):
    """
    Calculate parent node scores from Hessian estimates

    Parameters
    ----------
    curr_H : torch.Tensor
        Hessian after removing the i-th variable
    full_H : torch.Tensor
        Full Hessian
    i : int
        Index of removed variable

    Returns
    -------
    parents_score : torch.Tensor
        Parent node score vector
    """
    full_H = torch.hstack([full_H[0:i], full_H[i+1:]])
    parents_score = np.abs(curr_H - full_H)
    parents_score = torch.cat([parents_score[:i], torch.tensor([0.0]), parents_score[i:]])
    return parents_score


def run_caps(X, eta_G=0.001, eta_H=0.001, dispersion="mean", device="cuda:0", auto_tune_s=False, lambda_ksd=0.0):
    """
    Unified interface for calling CaPS algorithm

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_nodes)
        Observed data (already standardized)
    eta_G : float, default=0.001
        First-order Stein regularization parameter
    eta_H : float, default=0.001
        Second-order Stein regularization parameter
    dispersion : str, default="mean"
        Dispersion criterion ("mean" uses mean)
    device : str, default='cuda:0'
        Computing device

    Returns
    -------
    order : np.ndarray, shape (n_nodes,)
        Topological order, order[i] represents node index at position i
        (nodes earlier in the order are more upstream in causality)
    parent_scores : np.ndarray, shape (n_nodes, n_nodes)
        Parent node score matrix, parent_scores[j, i] represents score of X_j -> X_i
        (higher score means j is more likely a parent of i)
    """
    if not torch.cuda.is_available() and device.startswith('cuda'):
        device = 'cpu'
        print(f"Warning: CUDA unavailable, switching to CPU")

    device = torch.device(device)
    n, d = X.shape

    # Convert to PyTorch tensor
    X_tensor = torch.Tensor(X).to(device)
    full_X = X_tensor.clone()

    # Calculate topological order
    order = []
    layer = [0 for _ in range(d)]
    active_nodes = list(range(d))

    print(f"Starting CaPS topological ordering computation...")
    for i in range(d - 1):
        # Calculate Hessian of current active variables
        H = Stein_hess(X_tensor, eta_G, eta_H, device=device, auto_tune_s=auto_tune_s, lambda_ksd=lambda_ksd)

        if dispersion == "mean":  # Lemma 1 of CaPS
            l = int(H.mean(axis=0).argmax())
        else:
            raise ValueError(f"Unknown dispersion criterion: {dispersion}")

        # Add node with maximum Hessian to ordering (this is the most downstream causal node)
        order.append(active_nodes[l])
        active_nodes.pop(l)

        # Remove that node
        X_tensor = torch.hstack([X_tensor[:, 0:l], X_tensor[:, l+1:]])

    order.append(active_nodes[0])
    order.reverse()  # Reverse so upstream nodes come first

    print(f"Topological ordering completed: {order}")

    # Calculate parent score matrix
    print(f"Calculating parent score matrix...")
    active_nodes = list(range(d))
    full_H = Stein_hess(full_X, eta_G, eta_H, device=device, auto_tune_s=auto_tune_s, lambda_ksd=lambda_ksd).mean(axis=0)
    parent_scores = np.zeros((d, d))

    for i in range(d):
        curr_X = torch.hstack([full_X[:, 0:i], full_X[:, i+1:]])
        curr_H = Stein_hess(curr_X, eta_G, eta_H, device=device, auto_tune_s=auto_tune_s, lambda_ksd=lambda_ksd).mean(axis=0)
        parent_scores[i] = get_parents_score_from_hessian(curr_H, full_H, i).numpy()

    print(f"Parent score matrix calculation completed")
    print(f"Score matrix statistics: min={parent_scores.min():.4f}, max={parent_scores.max():.4f}, mean={parent_scores.mean():.4f}")

    return np.array(order), parent_scores


if __name__ == '__main__':
    """Test CaPS interface"""
    print("=" * 60)
    print("Testing CaPS Interface")
    print("=" * 60)

    # Generate simple test data
    np.random.seed(42)
    n_samples = 500

    # Create a simple causal DAG: X0 -> X1 -> X2
    X0 = np.random.randn(n_samples)
    X1 = 0.8 * X0 + 0.3 * np.random.randn(n_samples)
    X2 = 0.7 * X1 + 0.3 * np.random.randn(n_samples)

    X = np.column_stack([X0, X1, X2])

    # Standardization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"Test data shape: {X.shape}")
    print(f"True causal order should be: [0, 1, 2]")

    # Call CaPS
    order, parent_scores = run_caps(X, eta_G=0.001, eta_H=0.001, device='cuda:0')

    print(f"\nCaPS predicted topological order: {order}")
    print(f"\nParent score matrix:")
    print(parent_scores)
    print(f"\nExpected: parent_scores[0,1] and parent_scores[1,2] should be large")
    print(f"Actual: parent_scores[0,1]={parent_scores[0,1]:.4f}, parent_scores[1,2]={parent_scores[1,2]:.4f}")
