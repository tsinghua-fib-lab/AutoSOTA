import torch
import torch.nn.functional as F


def nfgsm(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float = 8/255, alpha: float = 8/255, k: float = 2.0):
    """
    Noisy Fast Gradient Sign Method (NFGSM).

    NFGSM is a single-step adversarial attack for training robust models, where the
    perturbation is initialized with an amplified random noise (scaled by `k`) within
    the allowed ε-ball before applying the FGSM update. This increased diversity of
    starting points improves adversarial training efficiency and robustness, while
    mitigating catastrophic overfitting.

    Reference:
        de Jorge, P., Bibi, A., Volpi, R., Sanyal, A., Torr, P. H. S., Rogez, G., &
        Dokania, P. K. (2022).
        "Make Some Noise: Reliable and Efficient Single-Step Adversarial Training."
        arXiv:2202.01181 (https://arxiv.org/abs/2202.01181)

    Args:
        model (torch.nn.Module): Target neural network.
        x (torch.Tensor): Clean input batch (B, C, H, W).
        y (torch.Tensor): Ground-truth labels.
        upper_limit (torch.Tensor): Per-channel normalized upper bound for inputs.
        lower_limit (torch.Tensor): Per-channel normalized lower bound for inputs.
        mu (torch.Tensor): Per-channel normalization mean.
        std (torch.Tensor): Per-channel normalization std.
        epsilon (float): Maximum perturbation magnitude (default: 8/255).
        alpha (float): FGSM gradient step size (default: 8/255).
        k (float): Random initialization scaling factor; higher values generate
            larger initial noise before FGSM update (default: 2.0).

    Returns:
        tuple:
            - torch.Tensor: Final adversarial perturbation (`delta`).
            - torch.Tensor: Gradient tensor from the initial backward pass.
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
    grad = torch.autograd.grad(loss, eta)[0]
    grad = grad.detach()
    
    # Compute perturbation based on sign of gradient
    delta = eta + alpha * torch.sign(grad)
    delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
    delta = delta.detach()
    
    return delta, grad
