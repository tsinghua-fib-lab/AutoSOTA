import torch
import torch.backends.cudnn as cudnn
from typing import Optional, Tuple, List

cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@torch.no_grad()
def _sinkhorn_project_inplace(
    X: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    iters: int = 4,
    eps: float = 1e-10,
) -> None:
    n = X.size(0)
    device = X.device
    dtype = X.dtype
    K = X.clamp_min(eps)
    u = torch.ones(n, device=device, dtype=dtype)
    v = torch.ones(n, device=device, dtype=dtype)
    for _ in range(iters):
        Kv = K @ v
        u = a / (Kv + eps)
        KTu = K.t() @ u
        v = b / (KTu + eps)
    X.copy_(K * u.view(-1, 1) * v.view(1, -1))


@torch.no_grad()
def _block_coordinate_gw_r3(
    X: torch.Tensor,
    P: torch.Tensor,
    active_indices: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_n: torch.Tensor,
    B_n: torch.Tensor,
    M: torch.Tensor,
    BT: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    iters: int = 4,
    epsilon: float = 1e-1,
    sinkhorn_iters: int = 2,
    eps: float = 1e-10,
) -> bool:
    if active_indices.numel() == 0:
        return False
    try:
        n = X.size(0)
        n_active = active_indices.size(0)
        density = n_active / (n * n)
        if density > 0.30:
            return False
        rows = active_indices[:, 0]
        cols = active_indices[:, 1]
        row_mask = torch.zeros(n, dtype=torch.bool, device=X.device)
        col_mask = torch.zeros(n, dtype=torch.bool, device=X.device)
        row_mask.scatter_(0, rows, True)
        col_mask.scatter_(0, cols, True)
        n_rows = int(row_mask.sum().item())
        n_cols = int(col_mask.sum().item())
        if n_rows > n * 0.6 or n_cols > n * 0.6:
            return False
        I = torch.where(row_mask)[0]
        J = torch.where(col_mask)[0]
        P_row_sums = P.sum(dim=1)
        P_col_sums = P.sum(dim=0)
        P_sub_old = P[I][:, J]
        P_sub_row_old = P_sub_old.sum(dim=1)
        P_sub_col_old = P_sub_old.sum(dim=0)
        a_sub = (a[I] - (P_row_sums[I] - P_sub_row_old)).clamp_min(eps)
        b_sub = (b[J] - (P_col_sums[J] - P_sub_col_old)).clamp_min(eps)
        A_sub = A_n[I][:, I].contiguous()
        B_sub = B_n[J][:, J].contiguous()
        X_sub = X[I][:, J]
        u_sub = u[I]
        v_sub = v[J]
        X_sub_old = X_sub.clone()
        inner_max = max(2, sinkhorn_iters)
        tol = 1e-6
        for _ in range(inner_max):
            M_sub = A_sub @ (X_sub @ B_sub.t())
            F_sub = u_sub.unsqueeze(1) + v_sub.unsqueeze(0) - 2.0 * M_sub
            X_sub = X_sub * torch.exp(-F_sub / epsilon)
            K = X_sub.clamp_min(eps)
            u_sink = torch.ones(n_rows, device=X.device, dtype=X.dtype)
            v_sink = torch.ones(n_cols, device=X.device, dtype=X.dtype)
            for _ in range(iters):
                Kv = K @ v_sink
                u_sink = a_sub / (Kv + eps)
                KTu = K.t() @ u_sink
                v_sink = b_sub / (KTu + eps)
            X_sub = K * u_sink.unsqueeze(1) * v_sink.unsqueeze(0)
            if (X_sub - X_sub_old).abs().max() < tol:
                break
            X_sub_old.copy_(X_sub)
        M_sub = A_sub @ (X_sub @ B_sub.t())
        I_expanded = I.unsqueeze(1).expand(n_rows, n_cols)
        J_expanded = J.unsqueeze(0).expand(n_rows, n_cols)
        X[I_expanded, J_expanded] = X_sub
        M[I_expanded, J_expanded] = M_sub
        return True
    except Exception:
        return False


@torch.no_grad()
def _efficient_adaptive_sinkhorn_r3(
    X: torch.Tensor,
    P: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_n: torch.Tensor,
    B_n: torch.Tensor,
    M: torch.Tensor,
    active_mask: Optional[torch.Tensor] = None,
    iters: int = 4,
    epsilon: float = 1e-1,
    sinkhorn_iters: int = 2,
    eps: float = 1e-10,
    use_local: bool = True,
    u: torch.Tensor = None,
    v: torch.Tensor = None,
    BT: torch.Tensor = None,
) -> float:
    if not use_local or active_mask is None:
        _sinkhorn_project_inplace(X, a, b, iters=iters, eps=eps)
        return 1.0
    active_count = int(active_mask.sum().item())
    if active_count == 0:
        return 0.0
    density = active_count / X.numel()
    active_indices = torch.nonzero(active_mask, as_tuple=False)
    if _block_coordinate_gw_r3(
        X, P, active_indices, a, b, A_n, B_n, M, BT, u, v,
        iters=iters, epsilon=epsilon, sinkhorn_iters=sinkhorn_iters, eps=eps
    ):
        return density
    _sinkhorn_project_inplace(X, a, b, iters=iters, eps=eps)
    return density


@torch.no_grad()
def _sym_normalize(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    d = A.sum(dim=1).clamp_min(eps)
    inv_sqrt = d.rsqrt()
    return inv_sqrt.unsqueeze(1) * A * inv_sqrt.unsqueeze(0)


@torch.no_grad()
def _degree_init(
    A: torch.Tensor, B: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
    temp: float = 0.1, iters: int = 6, eps: float = 1e-12
) -> torch.Tensor:
    dA = A.sum(dim=1, keepdim=True)
    dB = B.sum(dim=1, keepdim=True).t()
    C = (dA - dB).abs()
    scale = C.mean() * temp + eps
    X0 = torch.exp(-C / scale)
    _sinkhorn_project_inplace(X0, a, b, iters=iters, eps=eps)
    return X0


@torch.no_grad()
def _sharpen_and_project(
    X: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
    tau: float = 1.5, sinkhorn_iters: int = 20
) -> torch.Tensor:
    X = (X + 1e-32).pow(tau)
    _sinkhorn_project_inplace(X, a, b, iters=sinkhorn_iters)
    return X


@torch.no_grad()
def _compute_u_v(
    A: torch.Tensor, B: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (A * A) @ a, (B * B) @ b


@torch.no_grad()
def _compute_obj(
    ua: torch.Tensor, vb: torch.Tensor, M: torch.Tensor, P: torch.Tensor
) -> torch.Tensor:
    return ua + vb - 2.0 * (M * P).sum()


@torch.no_grad()
def DynamicVI_GW(
    A: torch.Tensor,
    B: torch.Tensor,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    X_init: Optional[torch.Tensor] = None,
    C_dummy: float = 1e3,
    rho0: float = 1e-1,
    min_rho: float = 1e-1,
    eps: float = 1e-6,
    max_iter: int = 500,
    Tw: int = 50,
    rho_decay: float = 1.0,
    warmup_iter: int = 10,
    r_ratio_cut: float = 0.00,
    T_full: int = 50,
    delta_thresh: float = 1e-7,
    sinkhorn_iters: int = 4,
    check_every: int = 10,
    enable_safe_local_sinkhorn: bool = True,
    fast_sampling: bool = True,
    reduce_warmup: bool = True,
    aggressive_early_stop: bool = True,
    optimize_chunk_size: bool = True,
    print_every: int = 50
) -> Tuple[torch.Tensor, List[float]]:
    common_dtype = torch.promote_types(A.dtype, B.dtype)
    device = A.device
    if B.device != device:
        B = B.to(device)
    if A.dtype != common_dtype:
        A = A.to(dtype=common_dtype)
    if B.dtype != common_dtype:
        B = B.to(dtype=common_dtype)
    n = A.size(0)
    A_n = _sym_normalize(A)
    B_n = _sym_normalize(B)
    a = (torch.ones(n, device=device, dtype=common_dtype) / n) if a is None else a.to(device=device, dtype=common_dtype)
    b = (torch.ones(n, device=device, dtype=common_dtype) / n) if b is None else b.to(device=device, dtype=common_dtype)
    if X_init is None:
        X = _degree_init(A, B, a, b, temp=0.1, iters=6)
    else:
        X = X_init.to(device=device, dtype=common_dtype).clone()
        _sinkhorn_project_inplace(X, a, b, iters=sinkhorn_iters)
    u, v = _compute_u_v(A_n, B_n, a, b)
    ua = (u * a).sum()
    vb = (v * b).sum()
    u_col = u.view(-1, 1)
    v_row = v.view(1, -1)
    BT = B_n.t().contiguous()
    M = A_n @ (X @ BT)
    F = u_col + v_row - 2.0 * M
    obj_hist: List[float] = []
    rho = rho0
    actual_warmup = 25 if reduce_warmup else warmup_iter
    actual_T_full = 5000 if optimize_chunk_size else T_full
    adaptive_T_full = optimize_chunk_size  # use density-driven adaptive T_full when chunk optimization is on
    actual_check_every = 15 if aggressive_early_stop else check_every
    if n <= 2000:
        temp_delta = torch.empty_like(X)
        use_temp_delta = True
    else:
        use_temp_delta = False
    internal_delta_thresh = max(delta_thresh * 2.5, 1e-6)
    density_log: List[float] = []
    best_obj = float("inf")
    best_X = None
    best_iter = -1
    for k in range(max_iter):
        rho = max(rho * rho_decay, min_rho)
        F_clamped = torch.clamp(-F / rho, -50, 50)
        X_new = X * torch.exp(F_clamped)
        current_sinkhorn_iters = 2 if k < actual_warmup else sinkhorn_iters
        current_density = 0.0
        if enable_safe_local_sinkhorn and k >= actual_warmup:
            if use_temp_delta:
                torch.sub(X_new, X, out=temp_delta)
                active_mask = torch.abs(temp_delta) > internal_delta_thresh
            else:
                Delta = X_new - X
                active_mask = torch.abs(Delta) > internal_delta_thresh
            current_density = _efficient_adaptive_sinkhorn_r3(
                X_new, X, a, b, A_n, B_n, M, active_mask,
                iters=current_sinkhorn_iters, epsilon=rho,
                sinkhorn_iters=4, eps=1e-10, use_local=True, u=u, v=v, BT=BT
            )
        else:
            _sinkhorn_project_inplace(X_new, a, b, iters=current_sinkhorn_iters)
            current_density = 1.0
        density_log.append(float(current_density))
        if aggressive_early_stop and k >= actual_warmup + 100 and len(density_log) >= 30:
            recent_densities = density_log[-30:]
            low_density_count = sum(1 for d in recent_densities if d < 0.02)
            if low_density_count >= 29:
                if best_X is not None and best_obj < _compute_obj(ua, vb, M, X_new).item():
                    X_new = best_X
                X = _sharpen_and_project(X_new, a, b, tau=1.5, sinkhorn_iters=max(18, sinkhorn_iters))
                U = X @ BT
                M = A_n @ U
                obj = _compute_obj(ua, vb, M, X).item()
                obj_hist.append(float(obj))
                break
        if adaptive_T_full and k >= actual_warmup and len(density_log) > 0:
            current_density_val = density_log[-1] if density_log else 1.0
            if current_density_val < 0.02:
                adaptive_T = 5
            elif current_density_val < 0.05:
                adaptive_T = 10
            elif current_density_val < 0.15:
                adaptive_T = 50
            else:
                adaptive_T = 200
            use_full = (k < actual_warmup) or ((k > 0) and (k % max(adaptive_T, 1) == 0))
        else:
            use_full = (k < actual_warmup) or ((k > 0) and (k % max(actual_T_full, 1) == 0))
        if use_full:
            U = X_new @ BT
            M = A_n @ U
        X = X_new
        F = u_col + v_row - 2.0 * M
        should_check = (k % actual_check_every == 0) or (k == max_iter - 1)
        if aggressive_early_stop and k > actual_warmup + 50 and k % 20 == 0:
            should_check = True
        if should_check:
            obj = _compute_obj(ua, vb, M, X).item()
            obj_hist.append(float(obj))
            if obj < best_obj:
                best_obj = obj
                best_X = X.clone()
                best_iter = k
            if aggressive_early_stop and len(obj_hist) >= 5:
                recent = obj_hist[-5:]
                diffs = [abs(recent[i + 1] - recent[i]) for i in range(4)]
                if all(abs(recent[i]) > 0 for i in range(5)):
                    rel_changes = [diffs[i] / abs(recent[i]) for i in range(4)]
                    if all(rel < eps * 0.5 for rel in rel_changes):
                        if best_X is not None and best_obj < obj:
                            X = best_X
                        else:
                            X = X_new
                        X = _sharpen_and_project(X, a, b, tau=1.5, sinkhorn_iters=max(18, sinkhorn_iters))
                        U = X @ BT
                        M = A_n @ U
                        break
            elif aggressive_early_stop and len(obj_hist) >= 2:
                prev, curr = obj_hist[-2], obj_hist[-1]
                if abs(prev) > 0:
                    rel = abs((curr - prev) / prev)
                    if rel < eps * 2.0:
                        if best_X is not None and best_obj < curr:
                            X = best_X
                        else:
                            X = X_new
                        X = _sharpen_and_project(X, a, b, tau=1.5, sinkhorn_iters=max(18, sinkhorn_iters))
                        U = X @ BT
                        M = A_n @ U
                        break
    use_best = best_X is not None and obj_hist and best_obj < obj_hist[-1]
    if use_best:
        X = best_X
    X_dbl = X.double()
    a_dbl = a.double()
    b_dbl = b.double()
    _sinkhorn_project_inplace(X_dbl, a_dbl, b_dbl, iters=300, eps=1e-16)
    X = X_dbl.float()
    return X, obj_hist


@torch.no_grad()
def DynamicVI_GW_Simple(
    A: torch.Tensor,
    B: torch.Tensor,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    X_init: Optional[torch.Tensor] = None,
    C_dummy: float = 1e2,
    rho0: float = 1e-1,
    min_rho: float = 1e-1,
    eps: float = 1e-6,
    max_iter: int = 500,
    rho_decay: float = 1.0,
    r_ratio_cut: float = 0.10,
    sinkhorn_iters: int = 8,
    check_every: int = 5,
) -> Tuple[torch.Tensor, List[float]]:
    return DynamicVI_GW(
        A, B, a, b, X_init,
        C_dummy, rho0, min_rho, eps, max_iter,
        Tw=max_iter + 1, rho_decay=rho_decay,
        warmup_iter=7,
        r_ratio_cut=r_ratio_cut,
        T_full=38,
        delta_thresh=1e-7,
        sinkhorn_iters=sinkhorn_iters,
        check_every=7,
        enable_safe_local_sinkhorn=True,
        fast_sampling=True,
        reduce_warmup=True,
        aggressive_early_stop=True,
        optimize_chunk_size=True,
    )


def create_dynamic_vi_gw(A: torch.Tensor, B: torch.Tensor, **kwargs):
    return DynamicVI_GW(A, B, **kwargs)


@torch.no_grad()
def round_hungarian_from_affinity(A: torch.Tensor, B: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        X_cpu = X.detach().to("cpu").numpy()
        r, c = linear_sum_assignment(-X_cpu)
        n = X_cpu.shape[0]
        P = np.zeros((n, n), dtype=np.float32)
        P[r, c] = 1.0
        return torch.from_numpy(P).to(X.device)
    except Exception:
        n = X.size(0)
        used_r = torch.zeros(n, dtype=torch.bool, device=X.device)
        used_c = torch.zeros(n, dtype=torch.bool, device=X.device)
        P = torch.zeros((n, n), dtype=torch.float32, device=X.device)
        _, idx = torch.topk(X.flatten(), k=n * n, largest=True)
        for lin in idx.tolist():
            i = lin // n
            j = lin % n
            if (not used_r[i]) and (not used_c[j]):
                P[i, j] = 1.0
                used_r[i] = True
                used_c[j] = True
                if used_r.all():
                    break
        return P
