import numpy as np
import torch
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from torch_geometric.data import Data
from typing import Any, Optional
import torch.nn.functional as F
from scipy.sparse.linalg import eigs, eigsh


def add_node_attr(data: Data, value: Any, attr_name: Optional[str] = None) -> Data:
    if attr_name is None:
        if data.x is not None:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data

class MyAddLaplacianEigenvectorPE(AddLaplacianEigenvectorPE):
    def __init__(
        self,
        k: int,
        attr_name: Optional[str] = 'laplacian_eigenvector_pe',
        is_undirected: bool = False,
        **kwargs,
    ):
        self.k = k
        self.attr_name = attr_name
        self.is_undirected = is_undirected
        self.kwargs = kwargs

    def __call__(self, data: Data) -> Data:
        eig_fn = eigs if not self.is_undirected else eigsh
        num_nodes = data.num_nodes

        edge_index, edge_weight = get_laplacian(data.edge_index,
                                                normalization='sym',
                                                num_nodes=num_nodes)
        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
        eig_vals, eig_vecs = eig_fn(L,
                                    k=self.k + 1,
                                    which='SR' if not self.is_undirected else 'SA',
                                    return_eigenvectors=True,
                                    **self.kwargs)
        eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
        pe = torch.from_numpy(eig_vecs[:, 1:self.k + 1])
        data = add_node_attr(data, pe, attr_name=self.attr_name)

        # assert num_nodes >= self.k + 1
        # zero padding in case of the assertion error
        # if num_nodes <= self.k:
        if not (num_nodes >= self.k + 1):
            data['laplacian_eigenvector_pe'] = F.pad(data['laplacian_eigenvector_pe'],
                                                     (0, self.k + 1 - num_nodes),
                                                     value=float('0'))
        sign = -1 + 2 * torch.randint(low=0, high=2, size=(self.k, ))
        data['laplacian_eigenvector_pe'] *= sign
        return data

