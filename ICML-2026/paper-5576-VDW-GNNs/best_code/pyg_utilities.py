"""
Utility functions for pytorch-geometric
projects.
"""
import math

import torch
import torch.nn as nn
from torch import linalg
from torch_geometric.data import (
    Data,
    Batch,
    Dataset,
    InMemoryDataset
)
from torch_sparse import SparseTensor, matmul
from typing import (
    Tuple,
    List,
    Dict,
    Any,
    Optional,
    Iterable,
    Callable,
    Literal
)

from models.nn_utilities import ProjectionMLP


def infer_device_from_batch(
    batch: Batch,
    *,
    feature_keys: Optional[List[str]] = None,
    operator_keys: Optional[List[str]] = None,
    edge_index_key: str = "edge_index",
) -> torch.device:
    """
    Infer device from tensors in a PyG Batch/Data object.
    """
    keys: List[str] = []
    if feature_keys:
        keys.extend([key for key in feature_keys if key])
    if operator_keys:
        keys.extend([key for key in operator_keys if key])
    if edge_index_key:
        keys.append(edge_index_key)

    for key in keys:
        if hasattr(batch, key):
            value = getattr(batch, key)
            if isinstance(value, torch.Tensor):
                return value.device

    if isinstance(batch, torch.Tensor):
        return batch.device

    return torch.device("cpu")


def check_if_undirected(edge_index: torch.Tensor) -> bool:
    """
    Checks if an edge_index tensor is undirected (has two
    directions for each edge).

    Args:
        edge_index: edge_index tensor of shape (2, E).
    Returns:
        True if the edge_index is undirected, False otherwise.
    """
    edges = set(map(tuple, edge_index.t().tolist()))
    reverse_edges = set((j, i) for i, j in edges)
    return edges == reverse_edges


class GraphListDataset(Dataset):
    """
    This class is a simple PyG Dataset subclass useful 
    for converting a list of Data objects into a Dataset,
    and applying transformations on every Data (graph) 
    object access, if needed.

    __init__ args:
        data_list: list of pytorch-geometric Data objects.
        transform: a callable function to apply to each
            Data object on every access.
    """
    def __init__(
        self,
        data_list: List[Data],
        transform: Optional[Callable] = None):
        super().__init__(transform=transform)
        self.data_list = data_list  # Store the list of Data objects

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        # note self.transform will be applied right after
        # self.get, within self.__get_item parent call
        return self.data_list[idx]


def exclude_features_sparsify_Data_list(
    data_list: List[Data],
    target_tensor_dtype,
    exclude_feat_idx: Optional[List[int]] = None,
    sparsify_feats: bool = False,
    use_k_drop: bool = False
) -> List[Data]:
    """
    Given a list of (graphs stored in) PyG Data objects,
    this function (1) converts the target tensors to the
    desired dtype; (2) removes any desired features to 
    exclude; and (3) converts the node features (Data.x)
    to sparse tensors, if desired.

    Args:
        data_list: list of PyG Data objects.
        target_tensor_dtype: torch.dtype for target
            tensors (e.g. torch.float).
        exclude_feat_idx: optional list of indices for
            features to exclude from each Data object
            node feature tensor (Data.x).
        sparsify_feats: whether to convert node feature
            matrix tensors to torch sparse tensors.
        use_k_drop: whether to use k_drop methods (need
            to use k_drop.KDropData objects instead of
            Data objects).
    Returns:
        Processed list of PyG Data objects.
    """
    if exclude_feat_idx is not None:
        orig_num_feats = data_list[0].x.shape[1]
        incl_mask = torch.ones(orig_num_feats, dtype=torch.bool)
        incl_mask[exclude_feat_idx] = False
        xs = [
            g.x[:, incl_mask].to(target_tensor_dtype) \
            for g in data_list
        ]
    else: 
        xs = [
            g.x.to(target_tensor_dtype) \
            for g in data_list
        ]

    data_list = [
        Data(
            x=x.to_sparse() if sparsify_feats else x, 
            edge_index=g.edge_index, 
            y=g.y.to(target_tensor_dtype)
        ) \
        for x, g in zip(xs, data_list)
    ]
    return data_list


def get_attribute_global_col_mins_maxs_for_Data_list(
    data_list: List[Data], 
    attr_name: str = 'x'
) -> Tuple[torch.Tensor]:
    """
    Finds the global, by-column mins and maxes of a 
    matrix tensor feature/attribute shared by all 
    graphs in a list of pytorch-geometric Data objects.

    Args:
        data_list: list of pytorch-geometric Data
            objects with a shared tensor feature 
            attribute.
        attr_name: string key for the attribute;
            defaults to 'x', the node features
            tensor.
    Returns:
        2-tuple of mins and maxs tensors (each of which
        has length equal to the number of columns in the
        attribute tensor).
    """
    first_tensor = getattr(data_list[0], attr_name)
    min_vals = first_tensor.min(dim=0).values
    max_vals = first_tensor.max(dim=0).values

    for data in data_list[1:]:
        tensor = getattr(data, attr_name)
        min_vals = torch.minimum(min_vals, tensor.min(dim=0).values)
        max_vals = torch.maximum(max_vals, tensor.max(dim=0).values)

    return min_vals, max_vals


def min_max_scale_data_list(
    data_list: List[Data],
    min_vals: torch.Tensor,
    max_vals: torch.Tensor,
    attr_name: str = 'x'
) -> None: # sets new attribute values in place
    """
    Uses mins and maxs (such as those found by
    'get_attribute_global_col_mins_maxs_for_Data_list')
    to min-max scale a shared attribute of each Data
    object (in place) in a list of Data objects.

    Args:
        data_list: list of pytorch-geometric Data
            objects with a shared tensor feature
            attribute.
        min_vals: tensor of minimum values for each
            column in the common attribute tensor
            across all Data objects in the list.
        max_vals:tensor of maximum values for each
            column in the common attribute tensor
            across all Data objects in the list.
        attr_name: string key for the attribute;
            defaults to 'x', the node features
            tensor.
    Returns:
        None; re-assigns min-max scaled attribute
        tensors in-place on the Data objects in the
        data_list.
    """
    # Avoid division by zero
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0  # Prevent divide-by-zero

    for data in data_list:
        attr = getattr(data, attr_name)
        scaled_tensor = (attr - min_vals) / range_vals
        setattr(data, attr_name, scaled_tensor)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


# def torch_sparse_identity(size: int) -> torch.Tensor:
#     """
#     Constructs an identity matrix tensor in sparse
#     COO form.

#     Args:
#         size: number of rows/columns in the (square)
#             identity matrix to be created.
#     Returns:
#         Sparse identity matrix tensor in COO format.
#     """
#     indices = torch.arange(size).unsqueeze(0).repeat(2, 1)
#     values = torch.ones(size)
#     return torch.sparse_coo_tensor(indices, values, (size, size))


# DEPRECATED: use 'get_P' from 'process_pyg_data.py' instead
# (THIS HAS COLUMN-NORMALIZATION, WHICH IS WRONG FOR Px multiplication)
# def get_Batch_P_sparse(
#     edge_index: torch.Tensor, 
#     edge_weight: Optional[torch.Tensor] = None,
#     n_nodes: int = None,
#     lazy: bool = True,
#     device: Optional[str] = None
# ) -> torch.Tensor:
#     r"""
#     Computes P, the lazy random walk diffusion 
#     operator on a graph defined as 
#     $$P = 0.5 (I - AD^{-1})$$,
#     where the graph is the disconnected batch
#     graph of a torch_geometric Batch object.

#     WARNING: leaving 'n_nodes' as None has the
#     'to_torch_coo_tensor' method infer the size
#     of A_sparse, etc. It may get it wrong in edge
#     cases; it is best to provide this value.

#     Args:
#         edge_index: edge_index (e.g., from a
#             pytorch_geometric Data or Batch object).
#         edge_weight: edge_weight (e.g., from a
#             pytorch_geometric Data or Batch object).
#         n_nodes: total number of nodes in batch 
#             'x' tensor (e.g., from a pytorch_geometric 
#             Batch object). 
#         lazy: whether to return lazy random walk operator
#             matrix.
#         device: string device key (e.g., 'cpu', 'cuda', 
#             'mps') for placing the output tensor; if
#             None, will check for cuda, else assign to cpu.
#     Returns:
#         Sparse P matrix tensor, of shape 
#         (N, N), where N = data.x.shape[0], 
#         the  total number of nodes across all
#         batched graphs. Note P_sparse is 'doubly
#         sparse': sparse off of block diagonals,
#         and each block is itself a sparse operator
#         P_i for each graph x_i.
#     """
#     from torch_geometric.utils import to_torch_coo_tensor
#     if device is None:
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     # should we use 'torch.sparse_coo_tensor' here instead?
#     A_sparse = to_torch_coo_tensor(
#         edge_index, 
#         edge_weight,
#         (n_nodes, n_nodes)
#     ).to(device)
#     D = A_sparse.sum(dim=1).to_dense()
#     # as of Oct 2024, 'torch.sparse.spdiags' doesn't work on cuda 12.4,
#     # -> use function with cpu tensors, then move resulting sparse
#     # tensor to device
#     D = D.squeeze().to('cpu')
#     # for nodes with degree 0, prevent division by 0 error
#     D_inv = torch.where(D > 0, (1. / D), 0.)
#     D_inv = torch.sparse.spdiags(
#         diagonals=D_inv, 
#         offsets=torch.zeros(1).long().to('cpu'),
#         shape=(len(D), len(D))
#     ).to(device)
#     I = torch_sparse_identity(len(D)).to(device)
#     P_sparse = (I + torch.sparse.mm(A_sparse, D_inv)) # .to(device)
#     if lazy:
#         P_sparse = 0.5 * P_sparse
#     # P_sparse = P_sparse.coalesce()
#     return P_sparse






def channel_pool(
    channel_pool_key: str,
    x: torch.Tensor,
    num_graphs: int,
    batch_index: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Pools a feature tensor's channels (within
    channels, across nodes), using 'sum', 'max',
    'mean', or 'max+mean' methods.

    Args:
        channel_pool_key: string key for the type
            of channel pooling to apply: 'sum', 'max',
            'mean', or 'max+mean'.
        x: tensor of batched hidden node features, of 
            shape (total num. nodes, num. channels).
        num_graphs: number of graphs in the batch.
        batch_index: the Batch.batch attribute (the
            tensor with each graph's index at identifying
            which rows of the batched feature matrix belong
            to the ith graph).
    Returns:
        Tensor of the hidden features pooled within channels.
    """
    if channel_pool_key not in (
        'sum', 
        'max', 
        'mean', 
        'max+mean', 
        'mean+max'
    ):
        raise NotImplementedError(
            f"'{channel_pool_key}' channel pooling not yet implemented."
        )
        
    x_i_chan_pools_max = [None] * num_graphs
    x_i_chan_pools_mean = [None] * num_graphs
    for i in range(num_graphs):
        if num_graphs > 1:
            # subset out ith graph
            x_i_mask = (batch_index == i)
            x_i = x[x_i_mask]
        else:
            x_i = x
        if 'sum' in channel_pool_key:
            x_i_chan_pools_mean[i] = torch.sum(x_i, dim=0)
        if 'max' in channel_pool_key:
            x_i_chan_pools_max[i] = torch.max(x_i, dim=0).values
        if 'mean' in channel_pool_key:
            x_i_chan_pools_mean[i] = torch.mean(x_i, dim=0)

    if ('max' in channel_pool_key) \
    and ('mean' not in channel_pool_key):
        x = torch.stack(x_i_chan_pools_max)
        # x shape: (n_graphs, [final_]n_channels_3)
        
    if ('mean' in channel_pool_key) \
    and ('max' not in channel_pool_key):
        x = torch.stack(x_i_chan_pools_mean)
        # x shape: (n_graphs, [final_]n_channels_3)
        
    if ('mean' in channel_pool_key) \
    and ('max' in channel_pool_key):
        maxs = torch.stack(x_i_chan_pools_max)
        means = torch.stack(x_i_chan_pools_mean)
        x = torch.stack((maxs, means), dim=1)
        # x shape: (n_graphs, 2, [final_]n_channels_3)
    return x


def moments_channel_pool(
    x: torch.Tensor,
    batch_index: Optional[torch.Tensor],
    num_graphs: int,
    channel_pool_moments: Tuple[int] = (1, 2, 3, 4),
    rescale_moments: bool = False
) -> torch.Tensor:
    """
    Pools a feature tensor's channels into moments (across
    nodes).

    Args:
        x: feature tensor of shape (N, c), where N is the 
            total number of nodes in the batched graphs,
            and c is the number of (hidden) features/channels.
        batch_index: 1-d batch index tensor of length N
            that identifies individual graph's node indices 
            in a py-g Batch object, which collates graphs into 
            one large disconnected graph. E.g.,
            [0, ..., 0, 1, ..., 1, ..., n-1, ..., n-1]
        num_graphs: number of individual graphs in the Batch.
        channel_pool_moments: tuple of moments to return, e.g.
            (1, 2, 3, 4) returns the 1st through 4th moments.
        rescale_moments: if True, rescales moments
            (across graphs, within moment-channels)
            onto interval [-1, 1], so that all new moment
            pooled features values are on the same
            scale. Note that this centers all moment 
            features at 0, losing any cross-channel differences 
            in their distribution centers. Hence, use with
            caution: the gain in numerical optimization
            might come at the cost of hindered learning.

    Returns:
        Tensor of containing moments of each channel
        in the original input x, shape (b, Q, c), where
        b is the number of graphs in the batch, Q is 
        the number of moments, and c is the number of
        channels.
    """
    # pool individual graph's channels as moments
    # (across nodes)
    q_norms = [None] * num_graphs
    for i in range(num_graphs):

        if num_graphs > 1:
            # subset out ith graph
            x_i_mask = (batch_index == i)
            x_i = x[x_i_mask]
        else:
            x_i = x
    
        # compute q-norms of its columns
        # technically, moments as defined in MFCN are ||Wjx||_q^q,
        # so we should raise norms to their qs here, but it shouldn't matter
        # for learning
        x_i_q_norms = torch.stack([
            linalg.vector_norm(
                x=x_i, 
                ord=q, 
                dim=0 # 0 for norms of col vecs
            ) ** q \
            for q in channel_pool_moments
        ]) # x_i_q_norms shape: (Q, n_channels)
        q_norms[i] = x_i_q_norms

    # stack all graph's moments
    x = torch.stack(q_norms)
    # x shape = (n_graphs, Q, n_channels)

    if rescale_moments:
        # compute min and max along the first dimension (n_graphs), keeping dimensions
        min_vals = x.min(dim=0, keepdim=True).values  # shape: (1, Q, n_channels)
        max_vals = x.max(dim=0, keepdim=True).values  # shape: (1, Q, n_channels)
    
        # avoid division by zero in case max == min
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # to prevent division by zero
    
        # apply min-max scaling -> interval [0, 1]
        x = (x - min_vals) / range_vals

        # -> interval [-1, 1]
        x = 2. * x - 1.
    
    return x


def node_pool(
    x: torch.Tensor, 
    node_pooling_key: str, 
    node_pool_wts: Optional[torch.Tensor],
    node_pool_bias: Optional[torch.Tensor],
    pool_dim: int = 1
) -> torch.Tensor:
    """
    Applies various simple node pooling operations
    to a tensor of node features.

    Args:
        x: tensor of hidden node features.
        node_pooling_key: string key for node
            pooling type: 'mean', 'max', 'sum', 
            or 'linear' (for a simple linear layer).
        node_pool_wts: a torch.Parameter tensor of
            learnable node pooling linear layer weights.
        node_pool_bias: a torch.Parameter tensor of
            learnable node pooling linear layer bias.
        pool_dim: the tensor 'dim' (dimension index)
            at which to perform mean, sum, or max 
            node pooling.
    Returns:
        Tensor of node pooling values.
    """
    if 'mean' in node_pooling_key:
        x = torch.mean(x, dim=pool_dim)
    elif 'sum' in node_pooling_key:
        x = torch.sum(x, dim=pool_dim)
    elif 'max' in node_pooling_key:
        x = torch.max(x, dim=pool_dim).values
    elif 'linear' in node_pooling_key:
        # channels are linearly-combined within nodes,
        # x' = wx + b, using the same learned w and b
        # parameters for all nodes
        # (N, d) @ (d, 1) -> (N, 1)
        x = torch.matmul(x, node_pool_wts) + node_pool_bias
    else:
        raise NotImplementedError(
            f"'{node_pooling_key}' node pooling method not implemented."
            f"Did you mean 'mean', 'sum', 'max', or 'linear'?"
        )
    return x

