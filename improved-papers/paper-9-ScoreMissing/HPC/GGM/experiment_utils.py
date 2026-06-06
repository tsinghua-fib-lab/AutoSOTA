import numpy as np
import torch


def make_prec(num_blocks, block_size, prob_connected, min_eval, val_interval=(0.5, 1)):
    if val_interval[0] > val_interval[1]:
        raise ValueError("Interval must be lower bound then upper bound, both in [0,1].")
    dim = block_size * num_blocks

    prec = torch.zeros(dim, dim)
    for i in range(num_blocks):
        # Get non zero entries
        nonzeros = (torch.rand((block_size, block_size)) < prob_connected).float()
        # Uniformly sample non-zero entries on [val_interval[0],val_interval[1]]
        values = nonzeros * (
            (val_interval[1]-val_interval[0]) * torch.rand_like(nonzeros) + val_interval[0])
        # Take upper triagle as both for symmetry.
        values = torch.tril(values) + torch.tril(values).T
        # Add to precision matrix
        prec[i * block_size: (i + 1) * block_size, :][
            :, i * block_size: (i + 1) * block_size
        ] = values

    # Change diagonals to 0 and then calculate value to give min eval 0.1
    prec = prec * (1 - torch.eye(dim))
    diagonal_val = -1 * torch.linalg.eigvalsh(prec).min() + min_eval
    prec = prec + diagonal_val * torch.eye(dim)
    return prec


def make_star_prec(dim, prob_connected, val_interval=(0.7, 1)):
    if val_interval[0] > val_interval[1]:
        raise ValueError("Interval must be lower bound then upper bound, both in [0, 1].")
    prec = torch.eye(dim)
    max_offdiag = np.sqrt(1/(dim-1))
    nonzeros = torch.bernoulli(torch.ones(dim-1)*prob_connected)
    # Uniformly sample on [val_interval[0]*max_offdiag, val_interval[1]*max_offdiag]
    values = nonzeros*max_offdiag*((torch.rand(dim-1)*(val_interval[1]-val_interval[0]))+val_interval[0])
    prec[1:, 0] = values.detach().clone()
    prec[0, 1:] = values.detach().clone()
    return prec


def make_multistar_prec(dim, prob_connected, min_eval, val_interval=(0.7, 1), num_stars=1):
    if val_interval[0] > val_interval[1]:
        raise ValueError("Interval must be lower bound then upper bound, both in [0, 1].")
    prec = torch.eye(dim)
    max_offdiag = np.sqrt(1/(dim-1))
    centres = torch.randperm(dim)[:num_stars]

    nonzeros = torch.bernoulli(torch.ones(centres.shape[0], dim)*prob_connected)
    # Uniformly sample on [val_interval[0], val_interval[1]]
    values = nonzeros*max_offdiag*(
        (torch.rand(centres.shape[0], dim)*(val_interval[1]-val_interval[0]))+val_interval[0])
    prec[centres, :] = torch.maximum(prec[centres, :], values.detach().clone())
    prec[:, centres] = torch.maximum(prec[:, centres], values.detach().clone().t())
    prec[centres, centres] = 1

    return prec


def make_multistar_fixed_prec(dim, prob_connected, min_eval, val=0.7, num_stars=1):
    prec = torch.eye(dim)
    centres = torch.arange(num_stars).int()

    nonzeros = torch.bernoulli(torch.ones(centres.shape[0], dim)*prob_connected)
    # Uniformly sample on [val_interval[0], val_interval[1]]
    values = nonzeros*val
    prec[centres, :] = torch.maximum(prec[centres, :], values.detach().clone())
    prec[:, centres] = torch.maximum(prec[:, centres], values.detach().clone().t())
    prec[centres, centres] = 1

    # Change diagonals to 0 and then calculate value to give min eval 0.1
    prec = prec * (1 - torch.eye(dim))
    diagonal_val = -1 * torch.linalg.eigvalsh(prec).min() + min_eval
    prec = prec + diagonal_val * torch.eye(dim)
    return prec


def tpr(true_adj: torch.Tensor, est_adj: torch.Tensor):
    # Filter out diagonal
    true_adj = true_adj[..., ~torch.eye(true_adj.shape[-1]).bool()]
    est_adj = est_adj[..., ~torch.eye(est_adj.shape[-1]).bool()]
    # Calculate TPR
    return torch.sum((est_adj == 1) & (true_adj == 1), dim=-1) / torch.sum(
        true_adj == 1, dim=-1
    )


def fpr(true_adj: torch.Tensor, est_adj: torch.Tensor):
    # Filter out diagonal
    true_adj = true_adj[..., ~torch.eye(true_adj.shape[-1]).bool()]
    est_adj = est_adj[..., ~torch.eye(est_adj.shape[-1]).bool()]
    # Calculate FPR
    return torch.sum((est_adj == 1) & (true_adj == 0), dim=-1) / torch.sum(
        true_adj == 0, dim=-1
    )


def roc(true_adj: torch.Tensor, est_prec: torch.Tensor):
    # Remove diagonal elements
    true_adj = true_adj[..., ~torch.eye(true_adj.shape[-1]).bool()]
    est_prec = est_prec[..., ~torch.eye(est_prec.shape[-1]).bool()]

    values, indices = torch.sort(est_prec, descending=True)
    # Sort in increasing order
    true_adj = true_adj[indices]
    tpr = torch.cumsum(true_adj == 1, -1)/torch.sum(true_adj, -1, keepdim=True)
    fpr = torch.cumsum(true_adj == 0, -1)/torch.sum(1-true_adj, -1, keepdim=True)
    return tpr, fpr


def AUC(tprs, fprs):
    return torch.trapz(tprs, fprs, dim=-1)


def make_planar_funcs(norm_vec: torch.Tensor, intercept: float):
    """Creates planar g functions

    Args:
        norm_vec (torch.Tensor): Vector perp to hyperplane
        intercept (float): intercept.

    Returns:
        (planar_g, planar_g_mask, planar_boundary)
    """
    def planar_g(x: torch.Tensor):
        return torch.clamp(x @ norm_vec + intercept, 0.0, 1.0)

    def planar_boundary(x: torch.Tensor):
        return planar_g(x) > 0

    def planar_g_mask(x: torch.tensor, mask: torch.tensor):
        "Returns g is every coordinate is 1 otherwise returns 0"
        all_obs = torch.min(mask, dim=-1)[0]
        return all_obs * planar_g(x) + (1 - all_obs) * torch.ones_like(x[..., 0])

    return (planar_g, planar_g_mask, planar_boundary)
