import torch
import torch.nn.functional as F
import numpy as np

def sora(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float= 8/255, max_alpha: float=16 / 255, method: str="Second Order Theory Sign", linearity_coef: float=None, **kwargs):
    # Normalize perturbations
    alpha_scalar = sora_max_alpha_function(max_alpha, linearity_coef=linearity_coef)
    if method == "Second Order Theory Sign":
        eps = (epsilon / std).view(1, -1, 1, 1)
        alpha = (alpha_scalar / std).view(1, -1, 1, 1)
    elif method == "Second Order Theory":
        eps = torch.sqrt(torch.sum((epsilon / std) ** 2)).item()
        alpha = torch.sqrt(torch.sum((alpha_scalar / std) ** 2)).item()
    
    # Initialize random step
    if method == "Second Order Theory Sign":
        eta = torch.empty_like(x).uniform_(-1, 1)
        eta *= eps
    elif method == "Second Order Theory":
        eta = torch.empty_like(x).normal_()
        d_flat = eta.view(x.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(x.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        eta *= r / n * eps
    eta = torch.clamp(eta, lower_limit - x, upper_limit - x)
    eta.requires_grad = True

    output = model(x + eta)
    loss = F.cross_entropy(output, y)
    grad = torch.autograd.grad(loss, eta)[0]
    grad = grad.detach()
    
    # Compute perturbation based on sign of gradient
    interpolation_coeff = torch.rand_like(grad).float()
    if method == "Second Order Theory Sign":
        delta = eta + alpha * interpolation_coeff * grad.sign()
    elif method == "Second Order Theory":
        grad_normalized = grad / (grad.view(grad.size(0), -1).norm(p=2, dim=1).view(-1, 1, 1, 1) + 1e-10)
        delta = eta + alpha * interpolation_coeff * grad_normalized
    delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
    delta = delta.detach()
    
    return delta, grad, alpha_scalar

# Function For Mapping Alignment To Max Alpha For SORA Method 
def sora_max_alpha_function(max_alpha, linearity_coef=None):
    linearity_coef = 0 if linearity_coef is None else linearity_coef
    if linearity_coef == 1:
        coef = 1
    else:
        coef = min(1, 0.02 / (1 - linearity_coef))
    alpha = coef * max_alpha
    return alpha
