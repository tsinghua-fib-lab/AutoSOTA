"""Monkey-patch knn_graph with sklearn fallback to avoid pyg-lib dependency."""
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def _knn_graph_fallback(x, k, loop=False, flow='source_to_target',
                        cosine=False, num_workers=1):
    """Fallback KNN graph using sklearn to avoid pyg-lib dependency."""
    x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
    k = min(int(k), len(x_np) - 1)  # Ensure k < n_nodes
    if k < 1:
        k = 1

    if cosine:
        from sklearn.preprocessing import normalize
        x_np = normalize(x_np, norm='l2')

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto',
                            metric='euclidean' if not cosine else 'cosine').fit(x_np)
    distances, indices = nbrs.kneighbors(x_np)

    # Build edge index
    src_list = []
    dst_list = []
    n = len(x_np)
    for i in range(n):
        if not loop:
            neighbors = indices[i, 1:]  # Skip self (first neighbor)
        else:
            neighbors = indices[i, :]

        if flow == 'source_to_target':
            src_list.extend([i] * len(neighbors))
            dst_list.extend(neighbors.tolist())
        else:
            src_list.extend(neighbors.tolist())
            dst_list.extend([i] * len(neighbors))

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    return edge_index


# Apply monkey-patch
import torch_geometric.nn.pool as pool_mod
pool_mod.knn_graph = _knn_graph_fallback
print("Patched knn_graph with sklearn fallback")
