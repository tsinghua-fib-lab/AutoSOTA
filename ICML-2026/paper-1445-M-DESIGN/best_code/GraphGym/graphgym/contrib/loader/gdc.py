from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy.linalg import expm
from torch import Tensor
from torch_scatter import scatter_add

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    add_self_loops,
    coalesce,
    is_undirected,
    to_dense_adj,
)

from torch_geometric.transforms import GDC


class MyGDC(GDC):
    def __call__(self, data: Data) -> Data:
        N = data.num_nodes
        edge_index = data.edge_index
        if data.edge_attr is None:
            edge_weight = torch.ones(edge_index.size(1),
                                     device=edge_index.device)
        else:
            edge_weight = data.edge_attr
            assert self.exact
            assert edge_weight.dim() == 1

        if self.self_loop_weight:
            edge_index, edge_weight = add_self_loops(
                edge_index, edge_weight, fill_value=self.self_loop_weight,
                num_nodes=N)

        edge_index, edge_weight = coalesce(edge_index, edge_weight, N)

        if self.exact:
            edge_index, edge_weight = self.transition_matrix(
                edge_index, edge_weight, N, self.normalization_in)
            diff_mat = self.diffusion_matrix_exact(edge_index, edge_weight, N,
                                                   **self.diffusion_kwargs)
            edge_index, edge_weight = self.sparsify_dense(
                diff_mat, **self.sparsification_kwargs)
        else:
            edge_index, edge_weight = self.diffusion_matrix_approx(
                edge_index, edge_weight, N, self.normalization_in,
                **self.diffusion_kwargs)
            edge_index, edge_weight = self.sparsify_sparse(
                edge_index, edge_weight, N, **self.sparsification_kwargs)

        edge_index, edge_weight = coalesce(edge_index, edge_weight, N)
        edge_index, edge_weight = self.transition_matrix(
            edge_index, edge_weight, N, self.normalization_out)

        # data.edge_index = edge_index
        # data.edge_attr = edge_weight
        # return data

        attr = f"edge_index_gdc_{self.diffusion_kwargs['method']}"
        data[attr] = edge_index
        return data
