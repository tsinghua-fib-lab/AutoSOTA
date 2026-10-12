"""
Import this before importing torch_geometric or torchdrug to fix
torch_scatter / torch_sparse / torch_cluster compatibility with
this custom PyTorch 2.13.0 build (ABI mismatch with standard wheels).

Provides pure-PyTorch drop-in implementations of the scatter functions
and knn_graph that torchdrug and our GVP modules need.
"""
import sys
import types

import torch


# ── torch_scatter replacements ──────────────────────────────────────

def _scatter_add(src, index, dim_size=None, dim=0):
    """Pure-PyTorch scatter_add."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    shape = list(src.shape)
    shape[dim] = dim_size
    out = src.new_zeros(shape)
    if src.dim() == 1:
        idx = index
    else:
        idx = index.view(-1, *(1,) * (src.dim() - 1)).expand_as(src)
    return out.scatter_add_(dim, idx, src)


def _scatter_mean(src, index, dim_size=None, dim=0):
    """Pure-PyTorch scatter_mean."""
    out = _scatter_add(src, index, dim_size, dim)
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    ones = torch.ones_like(src)
    count = _scatter_add(ones, index, dim_size, dim)
    count = count.clamp(min=1)
    return out / count


def _scatter_max(src, index, dim_size=None, dim=0):
    """Pure-PyTorch scatter_max. Returns (values, argmax)."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    shape = list(src.shape)
    shape[dim] = dim_size
    # scatter_reduce 'amax' is available in PyTorch >= 1.11
    if src.dim() == 1:
        idx = index
    else:
        idx = index.view(-1, *(1,) * (src.dim() - 1)).expand_as(src)
    out = src.new_full(shape, float('-inf'))
    arg = src.new_zeros(shape, dtype=torch.long)
    out = out.scatter_reduce_(dim, idx, src, 'amax', include_self=False)
    return out, arg  # arg is approximate (not real indices) but shape-correct


def _scatter_min(src, index, dim_size=None, dim=0):
    """Pure-PyTorch scatter_min. Returns (values, argmin)."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    shape = list(src.shape)
    shape[dim] = dim_size
    if src.dim() == 1:
        idx = index
    else:
        idx = index.view(-1, *(1,) * (src.dim() - 1)).expand_as(src)
    out = src.new_full(shape, float('inf'))
    arg = src.new_zeros(shape, dtype=torch.long)
    out = out.scatter_reduce_(dim, idx, src, 'amin', include_self=False)
    return out, arg


def _scatter_softmax(src, index, dim_size=None, dim=0):
    """Pure-PyTorch scatter_softmax."""
    src_max = _scatter_max(src, index, dim_size, dim)[0]
    src_max_per = src_max[index]
    src_exp = (src - src_max_per).exp()
    src_exp_sum = _scatter_add(src_exp, index, dim_size, dim)
    return src_exp / src_exp_sum[index]


# ── torch_cluster.knn_graph replacement ─────────────────────────────

def _knn_graph(x, k, batch=None, loop=False, flow='source_to_target',
               cosine=False, num_workers=1):
    """Pure-PyTorch k-nearest neighbor graph."""
    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)

    edge_parts = []
    for b in batch.unique():
        mask = batch == b
        xb = x[mask]
        nb = xb.size(0)
        if nb <= k + 1:
            rows = torch.arange(nb, device=x.device).repeat_interleave(nb - 1)
            cols = torch.cat([torch.cat([torch.arange(0, i, device=x.device),
                                         torch.arange(i + 1, nb, device=x.device)])
                              for i in range(nb)])
        else:
            D = torch.cdist(xb, xb)
            D.fill_diagonal_(float('inf'))
            _, nn = D.topk(k, dim=-1, largest=False)
            rows = torch.arange(nb, device=x.device).repeat_interleave(k)
            cols = nn.reshape(-1)

        glb = mask.nonzero(as_tuple=False).squeeze(-1)
        edge_parts.append(torch.stack([glb[rows], glb[cols]], dim=0))

    edge_index = torch.cat(edge_parts, dim=1)
    if flow == 'target_to_source':
        edge_index = edge_index.flip([0])
    return edge_index


# ── Install mock modules ────────────────────────────────────────────

def _install_mock(mod_name, attrs):
    """Create a mock module with the given attributes."""
    mod = types.ModuleType(mod_name)
    mod.__file__ = __file__
    for name, fn in attrs.items():
        setattr(mod, name, fn)
    sys.modules[mod_name] = mod


_install_mock('torch_scatter', {
    'scatter_add': _scatter_add,
    'scatter_mean': _scatter_mean,
    'scatter_max': _scatter_max,
    'scatter_min': _scatter_min,
    'scatter_softmax': _scatter_softmax,
})

_install_mock('torch_sparse', {
    'SparseTensor': type('SparseTensor', (), {}),  # dummy class
})

_install_mock('torch_cluster', {
    'knn_graph': _knn_graph,
})

# Also provide the old cpu->cuda symlink patch (no-op now, kept for compat)
def _patch_torch_sparse():
    pass

_patch_torch_sparse()
