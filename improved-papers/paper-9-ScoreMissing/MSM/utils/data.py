import torch
from torch import distributions as dists
from typing import Callable, Tuple


def torch_nanstd(X: torch.Tensor, dim):
    squared_diff = (X - torch.nanmean(X, dim=dim, keepdim=True))**2
    return torch.sqrt(torch.nanmean(squared_diff, dim=dim))


def torch_normcdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1 + torch.erf(x / (2**0.5)))


def my_all(X: torch.Tensor, dim=None):
    if dim is None:
        dim = tuple(range(len(X.shape)))
    if isinstance(dim, int):
        dim = [dim]
    return torch.sum(X, dim=dim) == torch.prod(torch.tensor(X.shape)[list(dim)])


def my_any(X: torch.Tensor, dim):
    return torch.sum(X, dim=dim) > 0


def my_allclose(X: torch.Tensor, Y: torch.Tensor, rtol=1e-05, atol=1e-08, dim=None):
    if dim is None:
        dim = tuple(range(len(X.shape)))
    logic_mat = torch.abs(X-Y) < atol + rtol * torch.abs(Y)
    return my_all(logic_mat, dim=dim)


def dist_to_symettry(A: torch.Tensor):
    dims = len(A.shape)
    return 0.5*torch.norm(A-A.permute((*tuple(range(dims-2)), dims-1, dims-2)), dim=(dims-1, dims-2))


def torch_nancov(X: torch.Tensor):
    d = X.shape[0]
    nan_bool = torch.isnan(X)
    cov_mat = torch.empty((d, d))
    inds = torch.tril_indices(d, d, -1)
    for ind in inds.T:
        X_sub = X[ind, :]
        nan_sub = nan_bool[ind, :]
        X_sub = X_sub[:, ~my_any(nan_sub, 0)]
        temp_cov = torch.cov(X_sub)
        for i, sub_ind in enumerate(ind):
            cov_mat[sub_ind, ind] = temp_cov[i, :]
    return cov_mat


def gen_inverse_ind(tensor: torch.Tensor, threshold: float, check_threshold=False):
    """Returns the first instance of a value greater than a threshold for a 1-dim tensor

    Args:
        tensor (torch.Tensor): The tensor to search for the threshold
        threshold (float): The threshold which the tensor needs to be larger than
        check_threshold (bool, optional): Whether to allow for threshold not to be met. Defaults to False.

    Returns:
        _type_: _description_
    """
    if not check_threshold:
        return torch.min(torch.nonzero(tensor > threshold)).item()
    else:
        return torch.min(torch.cat(
            (torch.nonzero(tensor > threshold), torch.tensor([tensor.shape]) - 1))
            ).item()


def get_ci(vec: torch.Tensor, dim: int, verbose=False, na_rm=False) -> torch.Tensor:
    """Get mean and CI for mean from vector

    Args:
        vec (Tensor): The vector of values you want the C.I. for the mean from
        dim (int): The dimension to calculate the mean and CI over
        verbose (bool, optional): Whether or not to print the CI. Defaults to True.
    """
    n = vec.shape[dim]
    if na_rm:
        n_samples = torch.sum(~torch.isnan(vec), dim=dim)
        mean = torch.nanmean(vec, dim=dim)
        se = torch_nanstd(vec, dim=dim)/(n_samples**0.5)
    else:
        mean = torch.mean(vec, dim=dim)
        se = torch.std(vec, dim=dim)/(n**0.5)
    ci_up = mean+1.96*se
    ci_low = mean-1.96*se
    # if verbose:
    #     print(f"Our Estimated Expected Power is: {mean:4.3f}")
    #     print(f"With ci({ci_low:4.3f}, {ci_up:4.3f})")
    return torch.stack([mean, ci_low, ci_up], dim=0)


def recursive_tensorize(list_of_lists) -> torch.Tensor:
    """Recursively convert a list of lists to a tensor via iterative implementation of stack on the 0th dimension

    Args:
        list_of_lists (list[list[torch.Tensor]]): A nested list of lists of tensors to convert

    Returns:
        torch.Tensor: The converted tensor
    """
    if isinstance(list_of_lists, list):
        if isinstance(list_of_lists[0], (list, tuple, torch.Tensor)):
            return torch.stack([recursive_tensorize(list) for list in list_of_lists])
        else:
            return torch.tensor(list_of_lists)
    else:
        return list_of_lists


def tpr(true_adj: torch.Tensor, est_adj: torch.Tensor):
    # Filter out diagonal
    true_adj = true_adj[..., ~torch.eye(true_adj.shape[-1]).bool()]
    est_adj = est_adj[..., ~torch.eye(est_adj.shape[-1]).bool()]
    # Calculate TPR
    return torch.sum((est_adj == 1) & (true_adj == 1), dim=-1)/torch.sum(true_adj == 1, dim=-1)


def fpr(true_adj: torch.Tensor, est_adj: torch.Tensor):
    # Filter out diagonal
    true_adj = true_adj[..., ~torch.eye(true_adj.shape[-1]).bool()]
    est_adj = est_adj[..., ~torch.eye(est_adj.shape[-1]).bool()]
    # Calculate FPR
    return torch.sum((est_adj == 1) & (true_adj == 0), dim=-1)/torch.sum(true_adj == 0, dim=-1)


# Create fast roc function
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


def create_sampling_func(
    sampling_dist: dists.Distribution, weight_function=None, missing_prob=None
) -> Callable[[int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

    # Create a sampling function that samples from a distribution
    def our_sampling_func(n: int, copies: int, verbose=False):
        X = sampling_dist.sample((n * copies,))
        if weight_function is not None:
            X = X[weight_function(X) > 0, :]
            if verbose:
                print(
                    f"Proprtion of samples in truncation: {100*X.shape[0]/(n*copies): 4.3f}%"
                )
            if X.shape[0] < n:
                print("Warning: not enough samples")
            else:
                X = X[:n, :]
        if missing_prob is None:
            return X
        else:
            mask = torch.bernoulli(torch.ones_like(X) - missing_prob)
            X_corrupted = X * mask
            return X_corrupted, mask, X

    return our_sampling_func


############################################
# ###### Missing Truncation Functions ######
############################################
def default_g(X: torch.Tensor, mask: torch.Tensor, r: float,
              centre: torch.Tensor, **kwargs):
    return r - torch.linalg.norm(mask*(X - centre), dim=-1)


def default_square_g(X: torch.Tensor, mask: torch.Tensor,
                     r: float, centre: torch.Tensor, **kwargs):
    return r - torch.linalg.norm(mask*(X - centre), dim=-1, ord=torch.inf)


def default_rectangular_g(X: torch.Tensor, mask: torch.Tensor,
                          r: torch.Tensor, centre: torch.Tensor, **kwargs):
    return torch.min(r - mask*(torch.abs(X - centre)), dim=-1)[0]
