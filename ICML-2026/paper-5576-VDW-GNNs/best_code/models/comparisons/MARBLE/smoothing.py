"""Smoothing module."""

import torch


def scalar_diffusion(x, t, method="matrix_exp", par=None):
    """Scalar diffusion."""
    if len(x.shape) == 1:
        x = x.unsqueeze(1)

    if method == "matrix_exp":
        if par.is_sparse:
            par = par.to_dense()
        return torch.matrix_exp(-t * par.to_dense()).mm(x)

    if method == "spectral":
        assert (
            isinstance(par, (list, tuple)) and len(par) == 2
        ), "For spectral method, par must be a tuple of \
            eigenvalues, eigenvectors!"
        evals, evecs = par

        # Transform to spectral
        x_spec = torch.mm(evecs.T, x)

        # Diffuse
        diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * t.unsqueeze(0))
        x_diffuse_spec = diffusion_coefs * x_spec

        # Transform back to per-vertex
        return evecs.mm(x_diffuse_spec)

    raise NotImplementedError


def vector_diffusion(x, t, Lc, L=None, method="spectral", normalise=True):
    """Vector diffusion."""
    n, d = x.shape[0], x.shape[1]

    if method == "spectral":
        assert len(Lc) == 2, "Lc must be a tuple of eigenvalues, eigenvectors!"
        nd = Lc[1].shape[0]
    else:
        nd = Lc.shape[0]

    assert (
        n * d % nd
    ) == 0, "Data dimension must be an integer multiple of the dimensions \
         of the connection Laplacian!"

    # vector diffusion with connection Laplacian
    out = x.view(nd, -1)
    out = scalar_diffusion(out, t, method, Lc)
    out = out.view(x.shape)

    if normalise:
        assert L is not None, "Need Laplacian for normalised diffusion!"
        x_abs = x.norm(dim=-1, p=2, keepdim=True)
        out_abs = scalar_diffusion(x_abs, t, method, L)
        ind = scalar_diffusion(torch.ones(x.shape[0], 1).to(x.device), t, method, L)
        out = out * out_abs / (ind * out.norm(dim=-1, p=2, keepdim=True))

    return out
