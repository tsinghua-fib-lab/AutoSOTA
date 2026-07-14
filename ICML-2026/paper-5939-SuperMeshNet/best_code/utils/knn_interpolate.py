# k-NN interpolation using pure PyTorch (no pytorch3d or pyg-lib needed)
import torch


def knn_interpolate(x, pos1, pos2, k=3):
    """k-NN interpolation from source positions pos1 to target positions pos2.

    For each point in pos2, finds its k nearest neighbors in pos1,
    then computes distance-weighted interpolation of features x.
    """
    # Compute pairwise squared distances: [N2, N1]
    # pos1: [N1, d], pos2: [N2, d]
    dists_sq = torch.cdist(pos2, pos1, p=2)  # [N2, N1]

    # Find k nearest neighbors
    topk_dists, topk_idx = torch.topk(dists_sq, k=k, dim=-1, largest=False)  # [N2, k]

    # Compute weights: 1 / (distance + epsilon), matching original pytorch3d version
    eps = 1e-8
    weights = 1.0 / (topk_dists + eps)  # [N2, k]
    weights = weights / weights.sum(dim=-1, keepdim=True)  # normalize

    # Gather features of k nearest neighbors
    # x: [N1, F], topk_idx: [N2, k] -> feats_knn: [N2, k, F]
    feats_knn = x[topk_idx]

    # Weighted sum: [N2, k, F] * [N2, k, 1] -> sum -> [N2, F]
    interpolated = (weights.unsqueeze(-1) * feats_knn).sum(dim=1)

    return interpolated
