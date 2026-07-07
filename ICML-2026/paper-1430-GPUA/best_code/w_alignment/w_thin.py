import torch

def procrustes_align(features_src: torch.Tensor,
                          features_tgt: torch.Tensor,
                          beta: float = 0) -> torch.Tensor:
    """
    使用 SVD 对齐 X -> Y（支持不同维度）
    
    features_src: (n_samples, d_X)
    features_tgt: (n_samples, d_Y)
    beta: 控制缩放到单位矩阵的程度，beta=1 表示完全回到单位矩阵
    返回:
        W: (d_X, d_Y)
    """
    # 计算最优正交矩阵的 SVD
    # 注意 X^T Y 维度为 (d_X, d_Y)
    u, _, v = torch.linalg.svd(features_src.T @ features_tgt, full_matrices=False)
    W = u @ v  # (d_X, d_Y)

    # 可选地让 W 更接近单位矩阵
    # 对非方阵，这里只对左上方方阵做加权
    min_dim = min(W.shape)
    identity = torch.eye(min_dim, device=W.device)
    W[:min_dim, :min_dim] = W[:min_dim, :min_dim] - (W[:min_dim, :min_dim] - identity) * beta

    return W
