import numpy as np
import torch
from typing import Dict, Union, List, Optional

_EPS = 1e-9  # matches the baseline's 1e-9 stabilizers for precision/recall


# ----------------------------
# Torch helpers (GPU-friendly)
# ----------------------------

def adj_from_W_torch(W: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Torch version of adj_from_W:
      A[i,j] = 1 iff |W[i,j]| > eps, diagonal forced to 0.
    Supports W shaped (..., D, D).
    Returns int64 adjacency.
    """
    A = (W.abs() > eps).to(torch.int64)
    D = A.shape[-1]
    eye = torch.eye(D, device=A.device, dtype=torch.bool)
    # broadcast eye across batch dims
    A = A.masked_fill(eye, 0)
    return A


def skel_from_adj_torch(A: torch.Tensor) -> torch.Tensor:
    """
    Torch version of skel_from_adj:
      Skeleton is undirected presence: S = 1 if A[i,j]==1 or A[j,i]==1, diagonal 0.
    Supports A shaped (..., D, D).
    Returns int64 skeleton.
    """
    S = ((A + A.transpose(-1, -2)) > 0).to(torch.int64)
    D = S.shape[-1]
    eye = torch.eye(D, device=S.device, dtype=torch.bool)
    S = S.masked_fill(eye, 0)
    return S


# ----------------------------
# Metrics for a single sample
# ----------------------------

def shd_directed_torch(A_pred: torch.Tensor, A_true: torch.Tensor) -> torch.Tensor:
    """
    Structural Hamming Distance where an orientation flip counts as 1.
    Vectorized Torch version of shd_directed.

    A_pred, A_true: (..., D, D) int/bool
    Returns: (...) tensor of SHD values (int64)
    """
    # convert to bool
    Ap = A_pred.to(torch.bool)
    At = A_true.to(torch.bool)

    D = At.shape[-1]
    # upper-triangular mask excluding diagonal
    triu = torch.triu(torch.ones((D, D), device=At.device, dtype=torch.bool), diagonal=1)

    # has edge in skeleton?
    has_true = (At | At.transpose(-1, -2)) & triu
    has_pred = (Ap | Ap.transpose(-1, -2)) & triu

    # skeleton mismatch contributes 1
    skel_mismatch = (has_true ^ has_pred)

    # if both have an edge, orientation differs if the ordered pair bits differ
    # (tp != pp) in the baseline means either direction bit differs.
    both_have = has_true & has_pred
    # Compare directed bits for (i,j) and (j,i) on upper triangle
    ori_diff = both_have & (
        (At & triu) != (Ap & triu)  # compares i->j on upper triangle
    )
    # Need also compare the reverse direction bits (j->i). We can do that by comparing
    # the transposed matrices on upper triangle.
    ori_diff = ori_diff | (both_have & (((At.transpose(-1, -2) & triu) != (Ap.transpose(-1, -2) & triu))))

    shd = (skel_mismatch | ori_diff).to(torch.int64).sum(dim=(-2, -1))
    return shd


def metrics_single_sample_torch(A_pred: torch.Tensor, A_true: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Structural metrics for ONE predicted directed adjacency vs true (torch version).

    Returns a dict of scalar tensors:
      - skeleton precision/recall/F1
      - orientation precision/recall/F1 (exact arrow match)
      - SHD (flip=1)
    """
    Ap = A_pred.to(torch.int64)
    At = A_true.to(torch.int64)

    S_pred = skel_from_adj_torch(Ap)
    S_true = skel_from_adj_torch(At)

    D = At.shape[0]
    triu = torch.triu(torch.ones((D, D), device=At.device, dtype=torch.bool), diagonal=1)

    # Skeleton counts: count undirected edges once via upper triangle
    tp_skel = ((S_pred == 1) & (S_true == 1) & triu).sum().to(torch.float64)
    fp_skel = ((S_pred == 1) & (S_true == 0) & triu).sum().to(torch.float64)
    fn_skel = ((S_pred == 0) & (S_true == 1) & triu).sum().to(torch.float64)

    prec_skel = tp_skel / (tp_skel + fp_skel + 1e-9)
    rec_skel = tp_skel / (tp_skel + fn_skel + 1e-9)
    f1_skel = 2 * prec_skel * rec_skel / (prec_skel + rec_skel + 1e-9)

    # Orientation counts (exact arrow match): full directed matrix (excluding diag doesn’t matter)
    tp_or = ((Ap == 1) & (At == 1)).sum().to(torch.float64)
    fp_or = ((Ap == 1) & (At == 0)).sum().to(torch.float64)
    fn_or = ((Ap == 0) & (At == 1)).sum().to(torch.float64)

    prec_or = tp_or / (tp_or + fp_or + 1e-9)
    rec_or = tp_or / (tp_or + fn_or + 1e-9)
    f1_or = 2 * prec_or * rec_or / (prec_or + rec_or + 1e-9)

    shd = shd_directed_torch(Ap, At).to(torch.float64)

    return dict(
        skel_precision=prec_skel, skel_recall=rec_skel, skel_f1=f1_skel,
        orient_precision=prec_or, orient_recall=rec_or, orient_f1=f1_or,
        shd=shd,
    )


# -----------------------------------------
# Weighted (Bayesian model averaged) metrics
# -----------------------------------------

def metrics_from_weighted_samples(
    particles: Union[List[np.ndarray], torch.Tensor],
    weights: Union[np.ndarray, torch.Tensor],
    A_true: np.ndarray,
    *,
    device: Optional[torch.device] = None,
    eps_w: float = 1e-12,
) -> Dict[str, float]:
    """
    Weighted structural metrics:
      E_q[metric(G)] ~= sum_s w_s metric(G_s)

    Supports:
      - particles as list of NumPy (CPU baseline mode)
      - particles as torch.Tensor (S,D,D) (GPU/vectorized mode)

    Returns:
      dict of Python floats (same keys as baseline).
    """
    # --- Normalize weights ---
    if isinstance(weights, torch.Tensor):
        w = weights.to(torch.float64)
        w = w / (w.sum() + eps_w)
    else:
        w = np.asarray(weights, dtype=float)
        w = w / (w.sum() + eps_w)

    # --- If particles are not torch tensors, fall back to CPU loop (faithful) ---
    if not isinstance(particles, torch.Tensor):
        # Import baseline helpers (same as your original file did)
        from dag.dag_ops import adj_from_W, skel_from_adj  # noqa: F401

        def shd_directed_np(A_pred: np.ndarray, A_true_np: np.ndarray) -> int:
            D = A_true_np.shape[0]
            shd = 0
            for i in range(D):
                for j in range(i + 1, D):
                    tp = (A_true_np[i, j], A_true_np[j, i])
                    pp = (A_pred[i, j], A_pred[j, i])
                    has_true = (tp[0] + tp[1]) > 0
                    has_pred = (pp[0] + pp[1]) > 0
                    if not has_true and not has_pred:
                        continue
                    if has_true != has_pred:
                        shd += 1
                    else:
                        if tp != pp:
                            shd += 1
            return shd

        def metrics_single_sample_np(A_pred: np.ndarray, A_true_np: np.ndarray) -> dict:
            S_pred = skel_from_adj(A_pred)
            S_true = skel_from_adj(A_true_np)

            tp_skel = int(np.sum((S_pred == 1) & (S_true == 1)) // 2)
            fp_skel = int(np.sum((S_pred == 1) & (S_true == 0)) // 2)
            fn_skel = int(np.sum((S_pred == 0) & (S_true == 1)) // 2)

            prec_skel = tp_skel / (tp_skel + fp_skel + 1e-9)
            rec_skel = tp_skel / (tp_skel + fn_skel + 1e-9)
            f1_skel = 2 * prec_skel * rec_skel / (prec_skel + rec_skel + 1e-9)

            tp_or = int(np.sum((A_pred == 1) & (A_true_np == 1)))
            fp_or = int(np.sum((A_pred == 1) & (A_true_np == 0)))
            fn_or = int(np.sum((A_pred == 0) & (A_true_np == 1)))

            prec_or = tp_or / (tp_or + fp_or + 1e-9)
            rec_or = tp_or / (tp_or + fn_or + 1e-9)
            f1_or = 2 * prec_or * rec_or / (prec_or + rec_or + 1e-9)

            shd_val = shd_directed_np(A_pred, A_true_np)

            return dict(
                skel_precision=prec_skel, skel_recall=rec_skel, skel_f1=f1_skel,
                orient_precision=prec_or, orient_recall=rec_or, orient_f1=f1_or,
                shd=float(shd_val),
            )

        agg = dict(
            skel_precision=0.0, skel_recall=0.0, skel_f1=0.0,
            orient_precision=0.0, orient_recall=0.0, orient_f1=0.0,
            shd=0.0,
        )
        for W_s, ww in zip(particles, w):
            from dag.dag_ops import adj_from_W
            A_s = adj_from_W(W_s)
            m = metrics_single_sample_np(A_s, A_true)
            for k in agg:
                agg[k] += float(ww) * float(m[k])
        return agg

    # --- GPU/vectorized path ---
    device = device if device is not None else particles.device
    W = particles.to(device=device)
    S = W.shape[0]

    if isinstance(w, np.ndarray):
        w_t = torch.as_tensor(w, device=device, dtype=torch.float64)
    else:
        w_t = w.to(device=device, dtype=torch.float64)

    A_true_t = torch.as_tensor(A_true, device=device, dtype=torch.int64)

    # Convert particles to adjacency
    A_pred = adj_from_W_torch(W)  # (S,D,D) int64

    # Skeleton metrics
    S_pred = skel_from_adj_torch(A_pred)          # (S,D,D)
    S_true = skel_from_adj_torch(A_true_t)        # (D,D)

    D = A_true_t.shape[0]
    triu = torch.triu(torch.ones((D, D), device=device, dtype=torch.bool), diagonal=1)

    tp_skel = ((S_pred == 1) & (S_true.unsqueeze(0) == 1) & triu).sum(dim=(-2, -1)).to(torch.float64)
    fp_skel = ((S_pred == 1) & (S_true.unsqueeze(0) == 0) & triu).sum(dim=(-2, -1)).to(torch.float64)
    fn_skel = ((S_pred == 0) & (S_true.unsqueeze(0) == 1) & triu).sum(dim=(-2, -1)).to(torch.float64)

    prec_skel = tp_skel / (tp_skel + fp_skel + 1e-9)
    rec_skel = tp_skel / (tp_skel + fn_skel + 1e-9)
    f1_skel = 2 * prec_skel * rec_skel / (prec_skel + rec_skel + 1e-9)

    # Orientation metrics
    tp_or = ((A_pred == 1) & (A_true_t.unsqueeze(0) == 1)).sum(dim=(-2, -1)).to(torch.float64)
    fp_or = ((A_pred == 1) & (A_true_t.unsqueeze(0) == 0)).sum(dim=(-2, -1)).to(torch.float64)
    fn_or = ((A_pred == 0) & (A_true_t.unsqueeze(0) == 1)).sum(dim=(-2, -1)).to(torch.float64)

    prec_or = tp_or / (tp_or + fp_or + 1e-9)
    rec_or = tp_or / (tp_or + fn_or + 1e-9)
    f1_or = 2 * prec_or * rec_or / (prec_or + rec_or + 1e-9)

    # SHD (flip=1)
    shd = shd_directed_torch(A_pred, A_true_t.unsqueeze(0)).to(torch.float64)

    # Weighted aggregation
    agg = dict(
        skel_precision=float((w_t * prec_skel).sum().item()),
        skel_recall=float((w_t * rec_skel).sum().item()),
        skel_f1=float((w_t * f1_skel).sum().item()),
        orient_precision=float((w_t * prec_or).sum().item()),
        orient_recall=float((w_t * rec_or).sum().item()),
        orient_f1=float((w_t * f1_or).sum().item()),
        shd=float((w_t * shd).sum().item()),
    )
    return agg

