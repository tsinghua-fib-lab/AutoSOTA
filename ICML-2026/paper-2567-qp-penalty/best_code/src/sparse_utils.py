"""Sparse linear solver detection and utility functions."""

import torch
import scipy.sparse as spa
import scipy.sparse.linalg as spla

_SPARSE_SOLVER = "scipy"
try:
    import pypardiso
    _SPARSE_SOLVER = "pardiso"
except ImportError:
    pass
try:
    import qdldl
    if _SPARSE_SOLVER == "scipy":
        _SPARSE_SOLVER = "qdldl"
except ImportError:
    pass
try:
    from sksparse.cholmod import cholesky as sparse_cholesky
    if _SPARSE_SOLVER == "scipy":
        _SPARSE_SOLVER = "cholmod"
except ImportError:
    pass


def sparse_solve_spd(H, b, solver=None):
    if solver is None:
        solver = _SPARSE_SOLVER
    H_csc = H.tocsc() if not spa.isspmatrix_csc(H) else H
    
    # Density check: if nnz exceeds 10% of dimension, consider it dense
    dim = H.shape[0]
    density = H.nnz / (dim * dim)
    
    # For very large and dense matrices, use iterative solver directly
    if dim > 5000 and density > 0.1:
        try:
            x, info = spla.minres(H, b, atol=1e-6, maxiter=500)
        except TypeError:
            x, info = spla.minres(H, b, tol=1e-6, maxiter=500)
        return x

    try:
        if solver == "pardiso":
            return pypardiso.spsolve(H_csc, b)
        elif solver == "qdldl":
            factor = qdldl.Solver(H_csc)
            return factor.solve(b)
        elif solver == "cholmod":
            factor = sparse_cholesky(H_csc)
            return factor(b)
        else:
            lu = spla.splu(H_csc)
            return lu.solve(b)
    except Exception as e:
        # Fallback to iterative solver
        try:
            x, info = spla.cg(H, b, atol=1e-9, maxiter=2000)
            if info != 0:
                # CG did not converge, try minres
                try:
                    x, info = spla.minres(H, b, atol=1e-9, maxiter=2000)
                except TypeError:
                    x, info = spla.minres(H, b, tol=1e-9, maxiter=2000)
                if info != 0:
                    raise RuntimeError(f"Iterative solver failed to converge: info={info}")
            return x
        except Exception as e2:
            raise RuntimeError(f"Sparse solve failed: {e}, fallback also failed: {e2}")


def is_sparse_tensor(t):
    if t is None: return False
    return t.layout in [torch.sparse_coo, torch.sparse_csc, torch.sparse_csr, torch.sparse_bsr, torch.sparse_bsc]


def torch_sparse_to_scipy(t: torch.Tensor) -> spa.spmatrix:
    t = t.detach().cpu()
    if t.layout == torch.sparse_csc:
        return spa.csc_matrix((t.values().numpy(), t.row_indices().numpy(), t.ccol_indices().numpy()), shape=t.size())
    elif t.layout == torch.sparse_csr:
        return spa.csr_matrix((t.values().numpy(), t.col_indices().numpy(), t.crow_indices().numpy()), shape=t.size())
    else:
        t = t.coalesce()
        return spa.coo_matrix((t.values().numpy(), (t.indices()[0].numpy(), t.indices()[1].numpy())), shape=t.size())
