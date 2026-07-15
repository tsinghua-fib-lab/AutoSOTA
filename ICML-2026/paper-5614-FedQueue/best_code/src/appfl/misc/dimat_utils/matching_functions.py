"""
Matching algorithms for feature space alignment between models.
Ported from DIMAT/utils/matching_functions.py.
"""

import torch
import scipy
import networkx as nx


def remove_col(x, idx):
    return torch.cat([x[:, :idx], x[:, idx + 1 :]], dim=-1)


def compute_correlation(covariance, corrsave_path, node, eps=1e-7):
    std = torch.diagonal(covariance).sqrt()
    covariance = covariance / (torch.clamp(torch.outer(std, std), min=eps))
    return covariance


def add_bias_to_mats(mats):
    """Maybe add bias to input."""
    pad_value = 0
    pad_func = torch.nn.ConstantPad1d((0, 1, 0, 1), pad_value)
    biased_mats = []
    for mat in mats:
        padded_mat = pad_func(mat)
        padded_mat[-1, -1] = 1
        biased_mats.append(padded_mat)
    return biased_mats


def match_tensors_zipit(
    metric,
    corrsave_path,
    node,
    r=0.5,
    a=0.3,
    b=0.125,
    print_merges=False,
    get_merge_value=False,
    check_doubly_stochastic=False,
    add_bias=False,
    **kwargs,
):
    """
    ZipIt! matching algorithm. Given metric dict, computes matching.
    Args:
    - metric: dictionary containing metrics with covariance or cossim matrix.
    - r: Amount to reduce total input feature dimension.
    - a: alpha hyperparameter.
    - b: beta hyperparameter.
    Returns:
    - (un)merge matrices
    """
    if "covariance" in metric:
        sims = compute_correlation(metric["covariance"], corrsave_path, node)
    elif "cossim" in metric:
        sims = metric["cossim"]
    out_dim = sims.shape[0]
    remainder = int(out_dim * (1 - r) + 1e-4)
    permutation_matrix = torch.eye(out_dim, out_dim)

    torch.diagonal(sims)[:] = -torch.inf

    num_models = int(1 / (1 - r) + 0.5)
    Om = out_dim // num_models

    original_model = torch.zeros(out_dim, device=sims.device).long()
    for i in range(num_models):
        original_model[i * Om : (i + 1) * Om] = i

    to_remove = permutation_matrix.shape[1] - remainder
    budget = torch.zeros(num_models, device=sims.device).long() + int(
        (to_remove // num_models) * b + 1e-4
    )

    merge_value = []

    while permutation_matrix.shape[1] > remainder:
        best_idx = sims.reshape(-1).argmax()
        row_idx = best_idx % sims.shape[1]
        col_idx = best_idx // sims.shape[1]

        merge_value.append(permutation_matrix[row_idx, col_idx])

        if col_idx < row_idx:
            row_idx, col_idx = col_idx, row_idx

        row_origin = original_model[row_idx]
        col_origin = original_model[col_idx]

        permutation_matrix[:, row_idx] += permutation_matrix[:, col_idx]
        permutation_matrix = remove_col(permutation_matrix, col_idx)

        sims[:, row_idx] = torch.minimum(sims[:, row_idx], sims[:, col_idx])

        if "magnitudes" in metric:
            metric["magnitudes"][row_idx] = torch.minimum(
                metric["magnitudes"][row_idx], metric["magnitudes"][col_idx]
            )
            metric["magnitudes"] = remove_col(metric["magnitudes"][None], col_idx)[0]

        if a <= 0:
            sims[row_origin * Om : (row_origin + 1) * Om, row_idx] = -torch.inf
            sims[col_origin * Om : (col_origin + 1) * Om, row_idx] = -torch.inf
        else:
            sims[:, row_idx] *= a
        sims = remove_col(sims, col_idx)

        sims[row_idx, :] = torch.minimum(sims[row_idx, :], sims[col_idx, :])
        if a <= 0:
            sims[row_idx, row_origin * Om : (row_origin + 1) * Om] = -torch.inf
            sims[row_idx, col_origin * Om : (col_origin + 1) * Om] = -torch.inf
        else:
            sims[row_idx, :] *= a
        sims = remove_col(sims.T, col_idx).T

        row_origin, col_origin = original_model[row_idx], original_model[col_idx]
        original_model = remove_col(original_model[None, :], col_idx)[0]

        if row_origin == col_origin:
            origin = original_model[row_idx].item()
            budget[origin] -= 1

            if budget[origin] <= 0:
                selector = original_model == origin
                sims[selector[:, None] & selector[None, :]] = -torch.inf

    if add_bias:
        unmerge_mats = permutation_matrix.chunk(num_models, dim=0)
        unmerge_mats = add_bias_to_mats(unmerge_mats)
        unmerge = torch.cat(unmerge_mats, dim=0)
    else:
        unmerge = permutation_matrix

    merge = permutation_matrix / (permutation_matrix.sum(dim=0, keepdim=True) + 1e-5)

    merge = merge.to(sims.device)
    unmerge = unmerge.to(sims.device)

    if get_merge_value:
        merge_value = sum(merge_value) / len(merge_value)
        return merge.T, unmerge, merge_value
    return merge.T, unmerge


def match_tensors_optimal(metric, corrsave_path, node, r=0.5, add_bias=False, **kwargs):
    """
    Apply optimal algorithm to compute matching using max-weight matching.
    """
    correlation = metric["covariance"]
    corr_mtx_a = compute_correlation(correlation, corrsave_path, node).cpu().numpy()
    min_num = corr_mtx_a[corr_mtx_a != -torch.inf].min() - 1
    G = nx.Graph()
    out_dim = corr_mtx_a.shape[0] // 2
    for i in range(2 * out_dim):
        G.add_node(i)
    for i in range(2 * out_dim):
        for j in range(0, i):
            G.add_edge(i, j, weight=(corr_mtx_a[i, j] - min_num))
    matches = nx.max_weight_matching(G)
    matches_matrix = torch.zeros(2 * out_dim, out_dim, device=correlation.device)
    for i, (a, b) in enumerate(matches):
        matches_matrix[a, i] = 1
        matches_matrix[b, i] = 1
    merge = matches_matrix / (matches_matrix.sum(dim=0, keepdim=True) + 1e-5)
    unmerge = matches_matrix
    return merge.T, unmerge


def match_tensors_permute(
    metric,
    corrsave_path,
    node,
    r=0.5,
    get_merge_value=False,
    check_doubly_stochastic=False,
    add_bias=False,
    **kwargs,
):
    """
    Matches arbitrary models by permuting all to the space of the first.
    Mimics Rebasin methods using the Hungarian algorithm.
    """
    correlation = compute_correlation(metric["covariance"], corrsave_path, node)
    out_dim = correlation.shape[0]

    num_models = int(1 / (1 - r) + 0.5)
    Om = out_dim // num_models
    device = correlation.device

    mats = [torch.eye(Om, device=device)]
    for i in range(1, num_models):
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(
            correlation[:Om, Om * i : Om * (i + 1)].cpu().numpy(), maximize=True
        )
        mats.append(
            torch.eye(Om, device=device)[torch.tensor(col_ind).long().to(device)].T
        )

    if add_bias:
        unmerge_mats = add_bias_to_mats(mats)
    else:
        unmerge_mats = mats

    unmerge = torch.cat(unmerge_mats, dim=0)
    merge = torch.cat(mats, dim=0)
    merge = merge / (merge.sum(dim=0, keepdim=True) + 1e-5)

    if get_merge_value:
        merge_value = (
            correlation[:Om, Om * i : Om * (i + 1)]
            .cpu()
            .numpy()[row_ind, col_ind]
            .mean()
        )
        return merge.T, unmerge, merge_value
    return merge.T, unmerge


def match_tensors_identity(metric, r=0.5, add_bias=False, **kwargs):
    """
    Match feature spaces from different models by simple weight averaging.
    """
    correlation = metric["covariance"]
    out_dim = correlation.shape[0]

    N = int(1 / (1 - r) + 0.5)
    Om = out_dim // N
    device = correlation.device

    mats = [torch.eye(Om, device=device) for _ in range(N)]

    if add_bias:
        unmerge_mats = add_bias_to_mats(mats)
    else:
        unmerge_mats = mats

    unmerge = torch.cat(unmerge_mats, dim=0)
    merge = torch.cat(mats, dim=0)
    merge = merge / (merge.sum(dim=0, keepdim=True) + 1e-5)

    return merge.T, unmerge
