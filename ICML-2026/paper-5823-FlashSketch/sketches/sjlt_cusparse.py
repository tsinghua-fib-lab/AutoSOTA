from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from dataclasses import dataclass
import math

import torch

import globals as g
from torch_utils import resolve_dtype


@dataclass(frozen=True)
class SjltCusparseConfig:
    """Config for an SJLT sketch using torch sparse (cuSPARSE backend)."""

    method: str = g.METHOD_SJLT_CUSPARSE
    k: int = 512
    s: int = 4
    seed: int = 0
    dtype: str = g.DTYPE_FP32
    use_csr: bool = True


def _make_hash_tensors(cfg, device, d, generator):
    """Return (row_indices, col_indices, signs) for the SJLT hash."""
    col_indices = torch.arange(d, device=device, dtype=torch.int32).repeat_interleave(cfg.s)
    row_indices = torch.randint(
        0,
        cfg.k,
        (d * cfg.s,),
        generator=generator,
        device=device,
        dtype=torch.int32,
    )
    signs = torch.randint(
        0,
        2,
        (d * cfg.s,),
        generator=generator,
        device=device,
        dtype=torch.int32,
    )
    signs = signs * 2 - 1
    return row_indices, col_indices, signs


def _maybe_to_csr(sparse_tensor, use_csr):
    """Return a CSR tensor when requested."""
    if not use_csr:
        return sparse_tensor
    if not sparse_tensor.is_cuda:
        raise ValueError("CSR sketch matrix requires CUDA; set use_csr=False for CPU.")
    csr = sparse_tensor.to_sparse_csr()
    if csr.crow_indices().dtype != torch.int32:
        crow = csr.crow_indices().to(torch.int32)
        col = csr.col_indices().to(torch.int32)
        csr = torch.sparse_csr_tensor(crow, col, csr.values(), size=csr.shape, device=csr.device)
    return csr


def _build_sjlt_sparse_matrix(cfg, device, d, dtype, generator):
    """Construct a sparse SJLT matrix S of shape (k, d) with s nonzeros per column."""
    row_indices, col_indices, signs = _make_hash_tensors(cfg, device, d, generator)
    values = signs.to(dtype=dtype) / math.sqrt(cfg.s)

    indices = torch.stack([row_indices, col_indices], dim=0)
    S = torch.sparse_coo_tensor(indices, values, size=(cfg.k, d), device=device).coalesce()
    S = _maybe_to_csr(S, cfg.use_csr)
    return S


def sketch(A, cfg):
    """Apply an SJLT sketch using cuSPARSE sparse-dense matmul."""
    if A.ndim != 2:
        raise ValueError("A must be 2D with shape (d, n)")

    target_dtype = resolve_dtype(cfg.dtype)
    device = A.device

    if target_dtype != torch.float32:
        raise ValueError("Only fp32 is supported for sjlt_cusparse.")
    if A.dtype != torch.float32:
        raise ValueError("Input A must be fp32 for sjlt_cusparse.")
    A_work = A
    if not A_work.is_contiguous():
        A_work = A_work.contiguous()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(cfg.seed))
    S = _build_sjlt_sparse_matrix(cfg, device, A_work.shape[0], target_dtype, generator)

    SA = torch.sparse.mm(S, A_work)
    return SA
