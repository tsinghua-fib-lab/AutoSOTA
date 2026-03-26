import torch

def rand_projections(dim, num_projections=50, device='cuda'): # For SW
    projections = torch.randn((num_projections, dim), device=device)
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections
def one_dimensional_Wasserstein_prod(X_prod, Y_prod, p=2): # For SW
    wasserstein_distance = torch.abs(X_prod - Y_prod)
    wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p), dim=0)
    return wasserstein_distance
def sliced_wasserstein(X, Y, num_projections=50, p=2, device="cuda"): # For SW
    if X.dim() < 2 or Y.dim() < 2: return torch.tensor(0.0, device=device)
    if X.shape[1] != Y.shape[1]: raise ValueError(f"Feature dimensions must match: X={X.shape}, Y={Y.shape}")
    if X.size(0) == 0 or Y.size(0) == 0 : return torch.tensor(0.0, device=device)
    dim = X.size(1); theta = rand_projections(dim, num_projections, device=device)
    X_prod = torch.matmul(X, theta.transpose(0, 1)); Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    X_prod_sorted, _ = torch.sort(X_prod, dim=0); Y_prod_sorted, _ = torch.sort(Y_prod, dim=0)
    sw_per_projection = one_dimensional_Wasserstein_prod(X_prod_sorted, Y_prod_sorted, p=p)
    sw = torch.pow(sw_per_projection.mean(), 1. / p); return sw

def __main__():
    x = torch.randn(30, 2, device='cuda')
    y = torch.randn(40, 2, device='cuda')
    print(sliced_wasserstein(x, y, 10))

if __name__ == "__main__":
    __main__()