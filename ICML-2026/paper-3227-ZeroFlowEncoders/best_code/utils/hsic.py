import torch

def hsic(x, y):
    """
    Computes the Hilbert-Schmidt Independence Criterion (HSIC) between x and y.
    
    The HSIC determines if two random variables are independent.
    HSIC = 0 implies independence (if the kernel is characteristic, like Gaussian).
    Higher values imply dependence.
    
    Args:
        x (torch.Tensor): Input tensor of shape (Batch, Dim_X)
        y (torch.Tensor): Input tensor of shape (Batch, Dim_Y)
        
    Returns:
        torch.Tensor: Scalar HSIC value (non-negative).
    """
    
    # 1. Helper to compute pairwise squared Euclidean distances
    def pairwise_distances(data):
        # (x - y)^2 = x^2 + y^2 - 2xy
        # shape: (Batch, Batch)
        n = data.size(0)
        dot = torch.mm(data, data.t())             # x @ x.T
        sq_norm = torch.diag(dot)                  # x^2
        
        # Expand sq_norm to (N, N) via broadcasting
        dist = sq_norm.unsqueeze(0) + sq_norm.unsqueeze(1) - 2 * dot
        
        # Clamp negative values due to numerical float precision
        dist = torch.clamp(dist, min=0.0)
        return dist

    # 2. Helper to compute Gaussian Kernel with Median Heuristic
    def gaussian_kernel_matrix(dist):
        # Flatten distance matrix to calculate median
        # We ideally want the median of the off-diagonal elements
        n = dist.size(0)
        if n > 1:
            # We filter out the diagonal zeros for a robust median
            # (masking the diagonal)
            mask = torch.ones_like(dist) - torch.eye(n, device=dist.device)
            median_dist = torch.median(dist[mask.bool()])
        else:
            median_dist = torch.median(dist)

        # Prevent division by zero
        median_dist = median_dist + 1e-5
        
        # Bandwidth: sigma^2 = median / 2 (standard heuristic)
        # Kernel: exp( - ||x-y||^2 / (2 * sigma^2) )
        #         exp( - ||x-y||^2 / median )
        sigma_sq = median_dist
        
        K = torch.exp(-dist / sigma_sq)
        return K

    # --- Main HSIC Computation ---
    
    n = x.size(0)
    
    # 3. Compute Kernel Matrices
    dist_x = pairwise_distances(x)
    dist_y = pairwise_distances(y)
    
    K = gaussian_kernel_matrix(dist_x)
    L = gaussian_kernel_matrix(dist_y)
    
    # 4. Compute Centering Matrix H
    # H = I - 1/n * J (where J is matrix of ones)
    H = torch.eye(n, device=x.device) - (1.0 / n) * torch.ones((n, n), device=x.device)
    
    # 5. Compute HSIC
    # Formula: Tr(K * H * L * H) / (n-1)^2
    # Note: K, L, H are all symmetric.
    
    # Centered Kernels: Kc = HKH, Lc = HLH
    Kc = torch.mm(torch.mm(H, K), H)
    Lc = torch.mm(torch.mm(H, L), H)
    
    # HSIC = Trace(Kc * Lc) / (n-1)^2
    hsic_value = torch.trace(torch.mm(Kc, Lc)) / ((n - 1) ** 2)
    
    return hsic_value

# --- Example Usage ---
if __name__ == "__main__":
    batch_size = 10000
    dim = 3*32*32
    
    # Case 1: Dependent Variables (y = x)
    x = torch.randn(batch_size, dim, device='cuda')
    y_dep = x.clone() # Identical dependence
    
    loss_dep = hsic(x, y_dep)
    print(f"HSIC (Dependent): {loss_dep.item():.6f}") # Should be High
    
    # Case 2: Independent Variables (Random y)
    y_indep = torch.randn(batch_size, dim, device='cuda')
    
    loss_indep = hsic(x, y_indep)
    print(f"HSIC (Independent): {loss_indep.item():.6f}") # Should be Low (close to 0)