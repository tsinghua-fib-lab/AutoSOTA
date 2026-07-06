from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import torch


def relative_residual(A, x, b):
    """Compute relative residual ||Ax - b|| / ||b||."""
    resid = A @ x - b
    denom = torch.linalg.norm(b)
    if denom == 0:
        return float(torch.linalg.norm(resid).item())
    return float((torch.linalg.norm(resid) / denom).item())


def gram_errors(A, SA):
    """Compute Frobenius and spectral norm errors for Gram approximation."""
    G = A.T @ A
    G_hat = SA.T @ SA
    diff = G_hat - G

    fro_err = torch.linalg.norm(diff)
    fro_ref = torch.linalg.norm(G)
    rel_fro = fro_err / fro_ref if fro_ref > 0 else fro_err

    spectral_err = torch.linalg.norm(diff, ord=2)
    spectral_ref = torch.linalg.norm(G, ord=2)
    rel_spec = spectral_err / spectral_ref if spectral_ref > 0 else spectral_err

    return {
        "gram_fro_error": float(fro_err.item()),
        "gram_fro_rel": float(rel_fro.item()),
        "gram_spec_error": float(spectral_err.item()),
        "gram_spec_rel": float(rel_spec.item()),
    }


def ose_errors(SQ):
    """Compute OSE spectral and Frobenius errors for an embedded subspace."""
    if SQ.ndim != 2:
        raise ValueError("SQ must be 2D with shape (k, r)")
    r = SQ.shape[1]
    if r <= 0:
        raise ValueError("SQ must have a non-zero subspace dimension")

    gram = SQ.T @ SQ
    eye = torch.eye(r, device=SQ.device, dtype=SQ.dtype)
    diff = gram - eye

    fro_err = torch.linalg.norm(diff)
    spec_err = torch.linalg.norm(diff, ord=2)
    svals = torch.linalg.svdvals(SQ)
    max_dev = torch.max(torch.abs(svals.square() - 1.0))

    return {
        "ose_fro_err": float(fro_err.item()),
        "ose_spec_err": float(spec_err.item()),
        "ose_max_sv_dev": float(max_dev.item()),
    }
