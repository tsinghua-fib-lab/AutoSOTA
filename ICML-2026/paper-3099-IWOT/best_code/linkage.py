import torch
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

import utils


def compute_pseudolabels(Z, num_targets, num_classes, soft=True):
    coph_dists = cophenet(Z)
    # Convert to square matrix
    ultra_dist_mat = squareform(coph_dists)
    ultra_dists_to_source = torch.tensor(ultra_dist_mat[num_targets:, :-num_classes])

    # print(dists)
    if soft is True:
        return torch.nn.functional.softmin(ultra_dists_to_source, dim=0).T
    else:
        y_pred = torch.argmin(ultra_dists_to_source, dim=0)
        return utils.one_hot(y_pred.T, num_classes)


def compute_soft_cluster(source_feat, target_feat, tau_min=0.1, tau_max=0.1):
    def soft_max(a, b, tau):
        # a: [1] or [N], b: [N]
        # returns smooth max elementwise
        m = torch.maximum(a, b)
        return m + tau * torch.log(torch.exp((a - m) / tau) + torch.exp((b - m) / tau))

    def soft_min2(a, b, tau):
        # smooth min elementwise between two tensors a,b of same shape
        m = torch.minimum(a, b)
        return m - tau * torch.log(
            torch.exp(-(a - m) / tau) + torch.exp(-(b - m) / tau)
        )

    all_feat = torch.vstack((source_feat, target_feat))
    U = torch.cdist(all_feat, all_feat, p=2)  # [N,N], requires grad

    n = U.shape[0]
    for k in range(n):
        U_prev = U  # keep reference
        U = U_prev.clone()  # new tensor (no inplace on U_prev)

        # vectorize over i for each k to avoid inner loop where possible
        # candidate[i,j] = soft_max(U_prev[i,k], U_prev[k,j])
        cand = soft_max(U_prev[:, k].unsqueeze(1), U_prev[k, :].unsqueeze(0), tau_max)
        U = soft_min2(U, cand, tau_min)  # not inplace; creates new ops

        # optional: enforce diagonal = 0 without inplace on autograd-needed tensor
        # (diagonal isn't used in your slicing anyway)
        # U = U - torch.diag_embed(torch.diagonal(U))

    num_source = source_feat.shape[0]
    return U[:num_source, num_source:]


@torch.no_grad()
def compute_cluster_full(source_feat, target_feat, num_classes, method):
    all_feat = torch.vstack((source_feat, target_feat))
    dist_mat = torch.cdist(all_feat, all_feat, p=2)

    # Perform linkage hierarchical clustering
    idx = torch.triu_indices(dist_mat.size(0), dist_mat.size(1), offset=1)
    y = dist_mat[idx[0], idx[1]]
    Z = linkage(y, method, metric="euclidean", optimal_ordering=True)
    return Z


@torch.no_grad()
def compute_cluster(source_feat, target_feat, method):
    num_targets = target_feat.shape[0]
    num_classes = len(source_feat)
    dist_targets = torch.cdist(target_feat, target_feat, p=2)
    dist_mat = torch.zeros(num_targets + num_classes, num_targets + num_classes)
    dist_mat[:num_targets, :num_targets] = dist_targets

    for i in range(num_classes):
        for j in range(num_classes):
            try:
                dists = torch.cdist(source_feat[i], source_feat[j], p=2)
                min_dist = torch.min(dists)
            except Exception:
                min_dist = 1e3
            dist_mat[num_targets + i, num_targets + j] = min_dist
            # print(dist_mat[num_targets+i, num_targets+j])

    for i in range(num_classes):
        try:
            dist_to_source = torch.cdist(target_feat, source_feat[i], p=2)
            min_dist_to_source, _ = torch.min(dist_to_source, dim=1)
        except Exception:
            min_dist_to_source = 1e3 * torch.ones(num_targets)
        # print(min_dist_to_source)
        # Expand target distances with source cond as a new node
        dist_mat[num_targets + i, :num_targets] = min_dist_to_source
        dist_mat[:num_targets, num_targets + i] = min_dist_to_source
    # Perform linkage hierarchical clustering
    idx = torch.triu_indices(dist_mat.size(0), dist_mat.size(1), offset=1)
    y = dist_mat[idx[0], idx[1]]
    Z = linkage(y, method, metric="euclidean", optimal_ordering=True)
    return Z


def plot_cluster(Z, num_targets, num_classes):
    plt.figure()

    # Plot the dendrogram
    def llf(idx):
        if idx < num_targets:
            return str(idx)
        elif idx >= num_targets:
            return "S" + str(idx - num_targets)

    dendrogram(Z, labels=None, leaf_label_func=llf)
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    plt.show()
