"""Utils module."""

import multiprocessing
from functools import partial
from typing import NamedTuple
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_sparse import SparseTensor
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


def print_settings(model):
    """Print parameters to screen"""

    print("\n---- Settings: \n")

    for x in model.params:
        print(x, ":", model.params[x])

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_features = model.enc.in_channels

    print("\n---- Number of features to pass to the MLP: ", n_features)
    print("---- Total number of parameters: ", n_parameters)
    print(f"\nUsing device {device}")


def parallel_proc(fun, iterable, inputs, processes=-1, desc=""):
    """Distribute an iterable function between processes"""

    if processes == -1:
        processes = multiprocessing.cpu_count()

    if processes > 1 and len(iterable) > 1:
        with multiprocessing.Pool(processes=processes) as pool:
            fun = partial(fun, inputs)
            result = list(tqdm(pool.imap(fun, iterable), total=len(iterable), desc=desc))
    else:
        result = [fun(inputs, i) for i in tqdm(iterable, desc=desc)]

    return result


def move_to_gpu(model, data, adjs=None):
    """Move stuff to gpu"""

    assert hasattr(data, "kernels"), "It seems that data is not preprocessed. Run preprocess(data)!"

    model = model.to(device)
    data.x = data.x.to(device)
    data.pos = data.pos.to(device)
    data.mask = data.mask.to(device)

    if hasattr(data, "L"):
        if len(data.L) == 2:
            data.L = [_l.to(device) for _l in data.L]
        else:
            data.L = data.L.to(device)
    else:
        data.L = None

    if hasattr(data, "Lc"):
        if len(data.Lc) == 2:
            data.Lc = [_l.to(device) for _l in data.Lc]
        else:
            data.Lc = data.Lc.to(device)
    else:
        data.Lc = None

    data.kernels = [K.to(device) for K in data.kernels]
    data.gauges = data.gauges.to(device)

    if adjs is None:
        return model, data, None

    adjs = [adj.to(device) for adj in adjs]
    return model, data, adjs


def detach_from_gpu(model, data, adjs=None):
    """detach stuff from gpu"""

    assert hasattr(data, "kernels"), "It seems that data is not preprocessed. Run preprocess(data)!"

    model = model.to(device)
    data.x = data.x.detach().cpu()
    data.pos = data.pos.detach().cpu()
    data.mask = data.mask.detach().cpu()

    if hasattr(data, "L"):
        data.L = [_l.detach().cpu() for _l in data.L]
    else:
        data.L = None

    if hasattr(data, "Lc"):
        data.Lc = [_l.detach().cpu() for _l in data.Lc]
    else:
        data.Lc = None

    data.kernels = [K.detach().cpu() for K in data.kernels]
    data.gauges = data.gauges.detach().cpu()

    if adjs is None:
        return model, data, None

    for i, adj in enumerate(adjs):
        adjs[i] = [adj[0].detach().cpu(), adj[1].detach().cpu(), adj[2]]
    return model, data, adjs


def to_SparseTensor(edge_index, size=None, value=None):
    """
    Adjacency matrix as torch_sparse tensor

    Args:
        edge_index (2xE matrix): edge indices
        size: pair (rows,cols) giving the size of the matrix.
            The default is the largest node of the edge_index.
        value: list of weights. The default is unit values.

    Returns:
        adjacency matrix in SparseTensor format
    """
    if value is None:
        value = torch.ones(edge_index.shape[1])
    if size is None:
        size = (int(edge_index[0].max()) + 1, int(edge_index[1].max()) + 1)

    adj = SparseTensor(
        row=edge_index[0], col=edge_index[1], value=value, sparse_sizes=(size[0], size[1])
    )

    return adj


def np2torch(x, dtype=None):
    """Convert numpy to torch"""
    if dtype is None:
        return torch.from_numpy(x).float()
    if dtype == "double":
        return torch.tensor(x, dtype=torch.int64)
    raise NotImplementedError


def to_list(x):
    """Convert to list"""
    if not isinstance(x, list):
        x = [x]

    return x


def to_pandas(x, augment_time=True):
    """Convert numpy to pandas"""
    columns = [str(i) for i in range(x.shape[1])]

    if augment_time:
        xaug = np.hstack([np.arange(len(x))[:, None], x])
        df = pd.DataFrame(xaug, columns=["Time"] + columns, index=np.arange(len(x)))
    else:
        df = pd.DataFrame(xaug, columns=columns, index=np.arange(len(x)))

    return df


class EdgeIndex(NamedTuple):
    """Edge Index."""

    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        """to"""
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)


def expand_index(ind, dim):
    """Interleave dim incremented copies of ind"""

    n = len(ind)
    ind = [ind * dim + i for i in range(dim)]
    ind = torch.hstack(ind).view(dim, n).t().flatten()

    return ind


def to_block_diag(sp_tensors):
    """To block diagonal."""
    sizes = [torch.tensor(t.size()).unsqueeze(1) for t in sp_tensors]
    ind = [t.indices() for t in sp_tensors]
    val = [t.values() for t in sp_tensors]

    for i in range(1, len(sp_tensors)):
        for j in range(i):
            ind[i] += sizes[j]

    ind = torch.hstack(ind)
    val = torch.hstack(val)

    return torch.sparse_coo_tensor(ind, val)


def expand_edge_index(edge_index, dim=1):
    """When using rotations, we replace nodes by vector spaces so
    need to expand adjacency matrix from nxn -> n*dimxn*dim matrices"""

    if dim == 1:
        return edge_index

    dev = edge_index.device
    if dev != "cpu":
        edge_index = edge_index.to("cpu")

    n = edge_index.shape[1]
    ind = [torch.tensor([i, j]) for i in range(dim) for j in range(dim)]
    edge_index = [edge_index * dim + i.unsqueeze(1) for i in ind]
    edge_index = torch.stack(edge_index, dim=2).view(2, n * len(ind))

    if dev != "cpu":
        edge_index.to(dev)

    return edge_index


def tile_tensor(tensor, dim):
    """Enlarge nxn tensor to d*dim x n*dim block matrix. Effectively
    computing a sparse version of torch.kron(K, torch.ones((dim,dim)))"""

    tensor = tensor.coalesce()
    edge_index = tensor.indices()
    edge_index = expand_edge_index(edge_index, dim=dim)
    return torch.sparse_coo_tensor(edge_index, tensor.values().repeat_interleave(dim * dim))


def restrict_dimension(sp_tensor, d, m):
    """Limit the dimension of the tensor"""
    n = sp_tensor.size(0)
    idx = torch.ones(n)
    for _ in range(m, d):
        idx[m::d] = 0
    idx = torch.where(idx)[0]
    sp_tensor = torch.index_select(sp_tensor, 0, idx).coalesce()
    return torch.index_select(sp_tensor, 1, idx).coalesce()


def restrict_to_batch(sp_tensor, idx):
    """Restrict tensor to current batch"""

    idx = [i.to(sp_tensor.device) for i in idx]

    if len(idx) == 1:
        return torch.index_select(sp_tensor, 0, idx[0]).coalesce()
    if len(idx) == 2:
        sp_tensor = torch.index_select(sp_tensor, 0, idx[0])
        return torch.index_select(sp_tensor, 1, idx[1]).coalesce()

    raise NotImplementedError


def standardize(X):
    """Standarsise data row-wise"""

    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)

    return (X - mean) / std
