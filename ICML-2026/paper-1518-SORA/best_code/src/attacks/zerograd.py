import torch
import torch.nn.functional as F


def zero_grad(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float = 8/255, alpha: float = 10/255, q_val: float = 0.35, k: float = 1.0, clip: bool = True):
    """
    ZeroGrad Adversarial Training Step.

    This variant addresses catastrophic overfitting in single-step adversarial
    training by zeroing out weak gradient components before applying the FGSM update.
    The threshold for zeroing is based on the per-sample quantile (`q_val`) of the
    absolute gradient magnitudes. By suppressing low-significance gradient directions,
    the method avoids spurious perturbations and improves robustness.

    Random initialization within the ε-ball (scaled by `k`) provides additional
    stochasticity to improve attack diversity.

    Reference:
        Golgooni, Z., Saberi, M., Eskandar, M., & Rohban, M. H. (2021).
        "ZeroGrad: Mitigating and Explaining Catastrophic Overfitting in FGSM
        Adversarial Training."
        arXiv:2103.15476 (https://arxiv.org/abs/2103.15476)

    Args:
        model (torch.nn.Module): Target neural network.
        x (torch.Tensor): Clean input batch (B, C, H, W).
        y (torch.Tensor): Ground-truth labels.
        upper_limit (torch.Tensor): Per-channel normalized upper bound for inputs.
        lower_limit (torch.Tensor): Per-channel normalized lower bound for inputs.
        mu (torch.Tensor): Per-channel normalization mean.
        std (torch.Tensor): Per-channel normalization std.
        epsilon (float): Maximum perturbation magnitude (default: 8/255).
        alpha (float): FGSM step size for perturbation update (default: 10/255).
        q_val (float): Quantile threshold for zeroing low-magnitude gradients
            (0 ≤ q_val ≤ 1) (default: 0.35).
        k (float): Random initialization scaling factor (default: 1.0).
        clip (bool): Whether to clamp perturbations to the ε-ball after update
            (default: True).

    Returns:
        tuple:
            - torch.Tensor: Final adversarial perturbation (`delta`).
            - torch.Tensor: Gradient tensor after zeroing low-magnitude components.
    """
    # Normalize perturbations
    eps = (epsilon / std).view(1, -1, 1, 1)
    alpha = (alpha / std).view(1, -1, 1, 1)
    
    # Initialize random step
    delta = torch.empty_like(x).uniform_(-k, k) * eps
    delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
    delta.requires_grad = True

    output = model(x + delta)
    F.cross_entropy(output, y).backward()
    grad = delta.grad.detach()
    q_grad = torch.quantile(torch.abs(grad).view(grad.size(0), -1), q_val, dim=1)
    grad[torch.abs(grad) < q_grad.view(grad.size(0), 1, 1, 1)] = 0

    delta = delta + alpha * torch.sign(grad)
    if clip:
        delta = torch.clamp(delta, min=-eps, max=eps)
    delta = torch.clamp(delta, lower_limit - x, upper_limit - x).detach()

    return delta, grad
