import torch
import torch.nn.functional as F


def fgsm_rs(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float = 8/255, alpha: float = 10/255, k: float = 1.0):
    """
    Fast Gradient Sign Method with Random Start (FGSM-RS).

    FGSM-RS is a one-step adversarial attack used for adversarial training, where the
    perturbation is initialized with a random step within the ε-ball before applying
    the standard FGSM update based on the sign of the loss gradient. This random
    initialization improves robustness training over deterministic FGSM by avoiding
    gradient masking.

    Reference:
        Wong, E., Rice, L., & Kolter, J. Z. (2020).
        "Fast is better than free: Revisiting adversarial training."
        In International Conference on Learning Representations (ICLR).
        arXiv:2001.03994 (https://arxiv.org/abs/2001.03994)

    Args:
        model (torch.nn.Module): The target network.
        x (torch.Tensor): Clean input batch (B, C, H, W).
        y (torch.Tensor): Ground-truth labels.
        upper_limit (torch.Tensor): Per-channel normalized maximum limit for inputs.
        lower_limit (torch.Tensor): Per-channel normalized minimum limit for inputs.
        mu (torch.Tensor): Per-channel normalization mean.
        std (torch.Tensor): Per-channel normalization std.
        epsilon (float): Maximum perturbation magnitude (default: 8/255).
        alpha (float): Step size for FGSM update (default: 10/255).
        k (float): Randomization scale factor for initial perturbation (default: 1.0).

    Returns:
        tuple:
            - torch.Tensor: Final perturbation tensor (`delta`).
            - torch.Tensor: Gradient tensor from first backward pass.
    """
    # Normalize perturbations
    eps = (epsilon / std).view(1, -1, 1, 1)
    alpha = (alpha / std).view(1, -1, 1, 1)

    # Initialize random step
    eta = torch.empty_like(x).uniform_(-k, k) * eps
    eta = torch.clamp(eta, lower_limit - x, upper_limit - x)
    eta.requires_grad = True
    
    output = model(x + eta)
    loss = F.cross_entropy(output, y)
    grad = torch.autograd.grad(loss, eta)[0].detach()
    
    # Compute perturbation based on sign of gradient
    delta = eta + alpha * torch.sign(grad)
    delta = torch.clamp(delta, -eps, +eps)
    delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
    delta = delta.detach()
    
    return delta, grad
