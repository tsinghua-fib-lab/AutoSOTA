import torch

def procrustes_align(
    X: torch.Tensor,
    Y: torch.Tensor,
    beta: float = 0.0,
    lambda_reg: float = 0.1,
    gamma_reg: float = 0.3,
    use_manifold: bool = True,
    k_neighbors: int = 3,
    sigma: float = 0.3,
    eps: float = 1e-6,
    batch_size: int = 2048,
    mu_svd_prior: float = 0.3,
    n_iter: int = 10,                # 迭代SVD精炼次数
    orthogonal: bool = True          # 是否保持正交约束
) -> torch.Tensor:
    """
    模仿 SVD 方式求解带正则的 Procrustes 对齐问题：
      min_W ||XW - Y||_F^2 + λ tr(Wᵀ Xᵀ L X W) + γ||W||_F^2
      - 使用 SVD(M=XᵀY) 初始化，并通过迭代SVD近似求解
      - 支持 Laplacian 正则与 Frobenius 正则
      - 可选保持 W 正交

    输入:
        X, Y: 对齐特征矩阵
        mu_svd_prior: 初始SVD解权重
        n_iter: SVD精炼迭代次数
    输出:
        优化后的对齐矩阵 W
    """
    device = X.device
    n_samples, d_X = X.shape
    d_Y = Y.shape[1]

    # --- 基础项 ---
    XT_X = X.T @ X
    XT_Y = X.T @ Y

    # --- Laplacian 项 ---
    if use_manifold and lambda_reg > 0:
        XT_L_X = compute_XTLX_batched(X, k=k_neighbors, sigma=sigma, batch_size=batch_size)
    else:
        XT_L_X = torch.zeros((d_X, d_X), device=device)

    # --- 初始化 (SVD)
    M = XT_Y
    try:
        U, _, Vh = torch.linalg.svd(M, full_matrices=False)
        W = U @ Vh
    except RuntimeError:
        W = torch.eye(d_X, d_Y, device=device)

    # --- 迭代SVD精炼 ---
    for _ in range(n_iter):
        # 梯度步: 近似最小化目标
        grad = XT_X @ W - XT_Y + lambda_reg * XT_L_X @ W + gamma_reg * W
        W = W - 0.1 * grad  # 简单梯度步，可替换为自适应学习率

        # SVD重投影保持正交约束
        if orthogonal:
            try:
                U, _, Vh = torch.linalg.svd(W, full_matrices=False)
                W = U @ Vh
            except RuntimeError:
                pass

    # --- 可选 β 回退 ---
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
    """分块计算 Xᵀ L X (与原实现一致)。"""
    device = X.device
    n, d = X.shape
    XT_A_X = torch.zeros((d, d), device=device)
    degree = torch.zeros(n, device=device)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_batch = X[start:end]
        dist_sq = torch.cdist(X_batch, X, p=2).pow(2)
        sims = torch.exp(-dist_sq / (2 * sigma ** 2))
        topk_vals, topk_idx = torch.topk(sims, k=k + 1, dim=1)
        vals = topk_vals[:, 1:]
        idx = topk_idx[:, 1:]
        degree[start:end] = vals.sum(dim=1)

        for i in range(X_batch.size(0)):
            x_i = X_batch[i]
            x_neighbors = X[idx[i]]
            weights = vals[i].unsqueeze(1)
            weighted_sum = (weights * x_neighbors).sum(dim=0)
            XT_A_X += torch.outer(weighted_sum, x_i)

    XT_A_X = (XT_A_X + XT_A_X.T) / 2
    XT_D_X = X.T @ (degree.unsqueeze(1) * X)
    XT_L_X = XT_D_X - XT_A_X
    return XT_L_X