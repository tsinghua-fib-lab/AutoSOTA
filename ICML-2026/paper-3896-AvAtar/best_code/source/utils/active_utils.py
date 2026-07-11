import time
import torch
import torch.nn.functional as F


def get_adjoint_grad_scores(T, C, eps, mode, tol=1e-9, max_iter=5000, **kwargs):
    n, m = T.shape
    a = T.sum(1)
    b = T.sum(0)

    # --- Step 0: Compute ∇_T f(T) ---
    def get_grad_T(T, C, mode, **kwargs):
        if mode == 'entropy':
            return torch.log(T + 1e-20) + 1.0
        elif mode == 'sq_l2':
            return 2 * T
        elif mode == 'kl':
            return torch.log(T/C + 1e-20) + 1.0
        elif mode == 'consistency':
            lap_mat1 = kwargs['lap_mat1']
            lap_mat2 = kwargs['lap_mat2']
            return 2 * (lap_mat1 @ T + T @ lap_mat2)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    if mode == 'consistency':
        assert 'lap_mat1' in kwargs and 'lap_mat2' in kwargs, "Laplacians required for consistency mode"
        grad_T = get_grad_T(T, C, mode, 
                            lap_mat1=kwargs['lap_mat1'], 
                            lap_mat2=kwargs['lap_mat2'])
    else:
        grad_T = get_grad_T(T, C, mode)

    # --- Step 1: Compute s, t on the RHS ---
    s = (T * grad_T).sum(dim=1)       # shape (n,)
    t = (T * grad_T).sum(dim=0)       # shape (m,)

    # --- Step 2: Define linear operator A(y_alpha, y_beta) ---
    def A_mv(y):
        y_alpha = y[:n]
        y_beta = y[n:]
        u = a * y_alpha + T @ y_beta
        v = T.T @ y_alpha + b * y_beta
        return torch.cat([u, v])

    # --- Step 3: Solve A y = [s; t] via conjugate gradient ---
    rhs = torch.cat([s, t])
    y = torch.zeros_like(rhs)

    r = rhs - A_mv(y)
    p = r.clone()
    rs_old = torch.dot(r, r)

    start_time = time.time()
    for i in range(max_iter):
        Ap = A_mv(p)
        alpha = rs_old / (torch.dot(p, Ap) + 1e-12)
        y += alpha * p
        r -= alpha * Ap
        rs_new = torch.dot(r, r)
        if torch.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / (rs_old + 1e-12)) * p
        rs_old = rs_new
    end_time = time.time()
    
    print(f"Conjugate gradient took {end_time - start_time:.4f} seconds")

    y_alpha = y[:n]
    y_beta = y[n:]

    # --- Step 4: Assemble ∇_C f = (1/ε) T ⊙ (y_alpha 1ᵀ + 1 y_betaᵀ - ∇_T f)
    T_norm = T / T.sum(dim=1, keepdim=True)
    pair_inf_scores = (1.0 / eps) * T_norm * (y_alpha.unsqueeze(1) + y_beta.unsqueeze(0) - grad_T) * C
    node_inf_scores = torch.sum(pair_inf_scores, dim=1)

    return node_inf_scores


def get_adjoint_grad_scores_sparse(T, C, eps, mode, tol=1e-9, max_iter=5000, sparse_thresh=1e-20, **kwargs):
    n, m = T.shape
    a = T.sum(1)
    b = T.sum(0)

    # --- Step 0: Convert T to sparse CSR format ---
    T_mask = T.abs() > sparse_thresh
    idx = T_mask.nonzero(as_tuple=False)
    vals = T[idx[:,0], idx[:,1]]

    T_sp = torch.sparse_coo_tensor(
        idx.t(), vals, size=T.shape,
        device=T.device, dtype=T.dtype
    ).coalesce().to_sparse_csr()

    a = T.sum(dim=1)     # shape (n,)
    b = T.sum(dim=0)     # shape (m,)

    # --- Step 1: Compute ∇_T f(T) in dense form ---
    def get_grad_T(T, C, mode, **kwargs):
        if mode == 'entropy':
            return torch.log(T + 1e-20) + 1.0
        elif mode == 'sq_l2':
            return 2 * T
        elif mode == 'kl':
            return torch.log(T/C + 1e-20) + 1.0
        elif mode == 'consistency':
            lap_mat1 = kwargs['lap_mat1']
            lap_mat2 = kwargs['lap_mat2']
            return 2 * (lap_mat1 @ T + T @ lap_mat2)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    if mode == 'consistency':
        assert 'lap_mat1' in kwargs and 'lap_mat2' in kwargs, "Laplacians required for consistency mode"
        grad_T = get_grad_T(T, C, mode, 
                            lap_mat1=kwargs['lap_mat1'], 
                            lap_mat2=kwargs['lap_mat2'])
    else:
        grad_T = get_grad_T(T, C, mode)

    # --- Step 2: Compute s, t on the RHS in dense form ---
    s = (T * grad_T).sum(dim=1)     # shape (n,)
    t = (T * grad_T).sum(dim=0)     # shape (m,)

    # --- Step 3: Define sparse linear operator A(y_alpha, y_beta) ---
    T_sp_T = T_sp.transpose(0, 1).to_sparse_csr()

    def A_mv_sp(y):
        """Matrix-vector product using sparse T for the CG step."""
        y_alpha = y[:n]   # (n,)
        y_beta  = y[n:]   # (m,)

        # Sparse multiplications
        u = a * y_alpha + T_sp @ y_beta                     # (n,)
        v = T_sp_T @ y_alpha + b * y_beta    # (m,)
        return torch.cat([u, v])

    # --- Step 4: Solve A y = [s; t] via conjugate gradient ---
    rhs = torch.cat([s, t])
    y = torch.zeros_like(rhs)

    r = rhs - A_mv_sp(y)
    p = r.clone()
    rs_old = torch.dot(r, r)

    start_time = time.time()
    for i in range(max_iter):
        Ap = A_mv_sp(p)
        alpha = rs_old / (torch.dot(p, Ap) + 1e-12)
        y += alpha * p
        r -= alpha * Ap
        rs_new = torch.dot(r, r)
        if torch.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / (rs_old + 1e-12)) * p
        rs_old = rs_new
    end_time = time.time()
    print(f"Sparse CG took {end_time - start_time:.4f} seconds")

    y_alpha = y[:n]
    y_beta = y[n:]

    # --- Step 5: Assemble ∇_C f = (1/ε) T ⊙ (y_alpha 1ᵀ + 1 y_betaᵀ - ∇_T f) ---
    T_norm = T / T.sum(dim=1, keepdim=True)
    pair_inf_scores = (1.0/eps) * T_norm * (y_alpha[:,None] + y_beta[None,:] - grad_T) * C
    node_inf_scores = (pair_inf_scores * T_norm).sum(dim=1)

    return node_inf_scores


# Query selection
def query_anchor_offline(utility_scores, visited, gnd_map, query_size):
    # Select the top-k anchors based on utility scores
    sorted_utility, sorted_indices = torch.sort(utility_scores, descending=True)
    queried_anchors, queried_utility = [], []
    num_selected = 0
    for utility, idx in zip(sorted_utility, sorted_indices):
        if num_selected >= query_size:
            break
        if idx.item() in visited:
            continue

        visited.add(idx.item())
        if idx.item() in gnd_map:
            queried_anchors.append([idx.item(), gnd_map[idx.item()]])
            queried_utility.append(utility.item())
            num_selected += 1

    queried_anchors = torch.tensor(queried_anchors, device=utility_scores.device, dtype=torch.long)
    queried_utility = torch.tensor(queried_utility, device=utility_scores.device, dtype=utility_scores.dtype)

    return queried_anchors, queried_utility


def query_anchors(T, Q, eps, modes, visited, gnd_map, query_size, weights=None, **kwargs):
    if weights is None:
        weights = [1.0 / len(modes)] * len(modes)
    assert len(modes) == len(weights), "Length of modes and weights must be the same."

    utility_scores = torch.zeros(T.shape[0], device=T.device, dtype=T.dtype)
    for mode, weight in zip(modes, weights):
        if mode.endswith('_adjoint_grad'):
            util_func = mode.split('_adjoint_grad')[0]
            raw_score = get_adjoint_grad_scores_sparse(T, Q, eps, mode=util_func, kwargs=kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        utility_scores += weight * F.normalize(raw_score, p=2, dim=0)

    # ALGO-01: Entropy-weighted query utility scores — prioritize uncertain nodes
    H_i = -(T * torch.log(T + 1e-10)).sum(dim=1)  # per-node coupling entropy
    entropy_weight = H_i / (H_i.max() + 1e-10)
    utility_scores = utility_scores * entropy_weight

    return query_anchor_offline(utility_scores, visited, gnd_map, query_size)
