import torch

def procrustes_align(X, Y, beta = 0,lambda_reg=0.1, gamma_reg=0.01, k_neighbors=5):
    """
    使用方案B（Laplacian 正则项）求解最优映射矩阵 W
    目标: min ||XW - Y||^2 + λ tr(W^T X^T L X W) + γ ||W||^2
    
    参数:
        X: [n, 768] 输入特征
        Y: [n, 512] 输出特征
        lambda_reg: 拉普拉斯正则项权重 λ
        gamma_reg: L2 正则项权重 γ
        k_neighbors: 构造 S_Y 时使用的 kNN 邻居数

    返回:
        W: [768, 512] 最优线性映射矩阵
    """

    n = Y.shape[0]

    # 1. 计算样本相似性矩阵 S_Y (使用余弦相似度)
    Y_norm = Y / (Y.norm(dim=1, keepdim=True) + 1e-8)
    S_Y = torch.mm(Y_norm, Y_norm.t())  # [n, n]

    # 仅保留每行前 k 个邻居（稀疏化）
    S_Y_topk = torch.zeros_like(S_Y)
    topk_vals, topk_idx = torch.topk(S_Y, k=k_neighbors + 1, dim=1)
    for i in range(n):
        S_Y_topk[i, topk_idx[i]] = topk_vals[i]
    S_Y = (S_Y_topk + S_Y_topk.t()) / 2  # 保证对称性

    # 2. 构造拉普拉斯矩阵 L = D - S
    D = torch.diag(S_Y.sum(dim=1))
    L = D - S_Y

    # 3. 闭式解 (X^T X + λ X^T L X + γ I)^(-1) X^T Y
    d = X.shape[1]
    I = torch.eye(d, device=X.device)
    A = X.T @ X + lambda_reg * (X.T @ L @ X) + gamma_reg * I
    B = X.T @ Y

    W = torch.linalg.solve(A, B)  # 更稳定的线性方程解法

    return W
