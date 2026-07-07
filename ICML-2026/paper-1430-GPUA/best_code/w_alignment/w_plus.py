import torch

def procrustes_align(
    X: torch.Tensor,
    Y: torch.Tensor,
    beta: float = 0.0,                # 保持你之前的 beta（是否回退到单位阵）
    lambda_reg: float = 0.5,
    gamma_reg: float = 0.5,
    use_manifold: bool = True,
    k_neighbors: int = 3,
    sigma: float = 0.5,
    eps: float = 1e-6,
    batch_size: int = 32768,
    mu_svd_prior: float = 0.3         # 新增：SVD 先验强度（mu=0 表示不使用 SVD 先验）
) -> torch.Tensor:
    """
    结合 SVD 初始化与闭式精炼的对齐：
      - 保留三项：最小二乘、Laplacian 正则 (可选)、Frobenius 正则
      - 使用 SVD(M=X^T Y) 得到 W0 = U @ Vh 作为先验（近似正交）
      - 通过带先验的闭式解精炼 W：
            (XT_X + lambda XT_L_X + gamma I + mu I) W = XT_Y + mu * W0
      - 支持 d_X != d_Y
    参数说明见代码注释；mu_svd_prior 控制 SVD 先验权重（越大越靠近 SVD 初始解）
    """
    device = X.device
    n_samples, d_X = X.shape
    d_Y = Y.shape[1]

    # --- 基本项 ---
    XT_X = X.T @ X                   # (d_X, d_X)
    XT_Y = X.T @ Y                   # (d_X, d_Y)

    # --- Laplacian 项（如果启用） ---
    if use_manifold and lambda_reg > 0:
        XT_L_X = compute_XTLX_batched(X, k=k_neighbors, sigma=sigma)
    else:
        XT_L_X = torch.zeros((d_X, d_X), device=device)

    # --- SVD 先验 W0 (尽量正交的初始化) ---
    # 利用 M = X^T Y 的奇异值分解： M = U S Vh -> W0 = U @ Vh
    # 注意 torch.linalg.svd 返回 (U, S, Vh)
    M = XT_Y  # X^T Y
    try:
        U, _, Vh = torch.linalg.svd(M, full_matrices=False)
        W0 = U @ Vh                  # (d_X, d_Y)
    except RuntimeError:
        # 若 SVD 出错（极少），退回零先验
        W0 = torch.zeros((d_X, d_Y), device=device)

    # --- 构造线性系统 A W = B ---
    A = XT_X + lambda_reg * XT_L_X + (gamma_reg + eps) * torch.eye(d_X, device=device)  # (d_X, d_X)

    if mu_svd_prior and mu_svd_prior > 0:
        A_left = A + mu_svd_prior * torch.eye(d_X, device=device)   # (d_X, d_X)
        B_right = XT_Y + mu_svd_prior * W0                          # (d_X, d_Y)
    else:
        A_left = A
        B_right = XT_Y

    # --- 求解闭式解 ---
    try:
        W = torch.linalg.solve(A_left, B_right)  # (d_X, d_Y)
    except torch.linalg.LinAlgError:
        W = torch.linalg.pinv(A_left) @ B_right

    # --- 可选 beta 回退（保持行为与原来一致） ---
    if beta and beta > 0:
        min_dim = min(d_X, d_Y)
        identity = torch.eye(min_dim, device=W.device)
        W[:min_dim, :min_dim] = W[:min_dim, :min_dim] - (W[:min_dim, :min_dim] - identity) * beta

    return W


def compute_XTLX_batched(
    X: torch.Tensor,
    k: int = 10,
    sigma: float = 1.0,
    batch_size: int = 2048
) -> torch.Tensor:
    """向量化版本的邻接矩阵计算"""
    device = X.device
    n, d = X.shape
    
    XT_A_X = torch.zeros((d, d), device=device)
    degree = torch.zeros(n, device=device)
    
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_batch = X[start:end]  # (b, d)
        
        # 计算相似度矩阵
        dist_sq = torch.cdist(X_batch, X, p=2).pow(2)
        sims = torch.exp(-dist_sq / (2 * sigma ** 2))
        
        # 获取kNN（排除自身）
        topk_vals, topk_idx = torch.topk(sims, k=k + 1, dim=1)
        vals = topk_vals[:, 1:]  # (b, k)
        idx = topk_idx[:, 1:]    # (b, k)
        
        # 向量化计算度数
        degree[start:end] = vals.sum(dim=1)
        
        # 向量化计算 A 部分贡献
        batch_size_curr = X_batch.size(0)
        row_indices = torch.arange(start, start + batch_size_curr, device=device).repeat_interleave(k)
        col_indices = idx.flatten()
        weights = vals.flatten()
        
        # 使用稀疏矩阵快速计算
        from torch.sparse import softmax
        # 构造稀疏邻接矩阵
        indices = torch.stack([row_indices, col_indices])
        sparse_A = torch.sparse_coo_tensor(indices, weights, (n, n))
        sparse_A = (sparse_A + sparse_A.t()) / 2  # 对称化
        
        # 直接计算 X^T A X
        AX = torch.sparse.mm(sparse_A, X)
        XT_A_X_batch = X_batch.T @ AX[start:end]
        XT_A_X = XT_A_X + XT_A_X_batch
    
    XT_D_X = X.T @ (degree.unsqueeze(1) * X)
    XT_L_X = XT_D_X - XT_A_X
    return XT_L_X