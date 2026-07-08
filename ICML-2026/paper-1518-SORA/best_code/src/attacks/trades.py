import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementation of Theoretically Principled Trade-off between Robustness and Accuracy(TRADES)
def trades(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float = 8/255, perturb_steps: int = 10, alpha: float= 2/255):
    """
    Implementation of TRADES (Theoretically Principled Trade-off between Robustness and Accuracy) adversarial 
    training method [Zhang et al., 2019].

    This function generates adversarial examples via KL-divergence-based perturbations and computes the 
    robust loss for model training. The method aims to balance the trade-off between standard accuracy 
    and adversarial robustness.

    Reference:
        H. Zhang, Y. Yu, J. Jiao, E. P. Xing, L. E. Ghaoui, M. I. Jordan.
        "Theoretically Principled Trade-off between Robustness and Accuracy" 
        In Proceedings of ICML 2019.
        arXiv:1901.08573 (https://arxiv.org/abs/1901.08573)

    Args:
        model (torch.nn.Module): Neural network to be trained.
        x (torch.Tensor): Clean input batch (B, C, H, W).
        y (torch.Tensor): Ground-truth labels (unused here, kept for compatibility).
        upper_limit (torch.Tensor): Upper bound of normalized inputs.
        lower_limit (torch.Tensor): Lower bound of normalized inputs.
        mu (torch.Tensor): Mean used for input normalization.
        std (torch.Tensor): Standard deviation used for input normalization.
        epsilon (float): Maximum perturbation magnitude (default: 8/255).
        perturb_steps (int): Number of PGD steps for adversarial example generation (default: 10).
        alpha (float): Step size for PGD updates (default: 2/255).

    Returns:
        tuple:
            - torch.Tensor: Zero tensor (placeholder for compatibility with other training steps).
            - torch.Tensor: Robust loss computed from adversarial examples.
            - torch.Tensor: Gradient of KL divergence wrt. adversarial inputs at first step.
    """
    # Normalize perturbations
    eps = (epsilon / std).view(1, -1, 1, 1)
    alpha = (alpha / std).view(1, -1, 1, 1)

    batch_size = x.shape[0]
    kl_criterion = nn.KLDivLoss(reduction = "sum")
    model_training = model.training
    model.eval()
    delta = 0.001 * torch.randn(x.shape, device=x.device).detach()
    delta = delta * (1 / std).view(1, -1, 1, 1)

    x_trades = x + delta
    for step in range(perturb_steps):
        x_trades.requires_grad_(True)
        with torch.enable_grad():
            loss_kl = kl_criterion(F.log_softmax(model(x_trades), dim=1), F.softmax(model(x), dim=1))
        grad = torch.autograd.grad(loss_kl, x_trades)[0]
        if step == 0:
            grad_zero = grad.detach()
        x_trades = x_trades.detach() + alpha * torch.sign(grad.detach())
        x_trades = torch.clamp(x_trades, min=x - eps, max=x + eps)
        x_trades = torch.clamp(x_trades, min=lower_limit, max=upper_limit)
    if model_training:
        model.train()
    loss_robust = (1.0 / batch_size) * kl_criterion(F.log_softmax(model(x_trades), dim=1), F.softmax(model(x), dim=1))
    
    return torch.zeros_like(x), loss_robust, grad_zero
