import torch
import torch.nn.functional as F


def nuclear(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float = 8/255, alpha: float = 8/255, steps: int = 1, nuc_reg: float = 4, k: float = 0.5):
    """
    Nuclear-Norm Adversarial Training (NuAT).

    NuAT is a single-step adversarial training method that uses a regularizer with nuclear norm.

    Reference:
        Sriramanan, G., Addepalli, S., Baburaj, A., & Babu, R. V. (2021).
        "Towards Efficient and Effective Adversarial Training."
        NeurIPS 2021 Poster (https://openreview.net/forum?id=kuK2VARZGnI)

    Args:
        model (torch.nn.Module): The target network.
        x (torch.Tensor): Clean input batch (B, C, H, W).
        y (torch.Tensor): Ground-truth labels.
        upper_limit (torch.Tensor): Per-channel normalized maximum limit for inputs.
        lower_limit (torch.Tensor): Per-channel normalized minimum limit for inputs.
        mu (torch.Tensor): Per-channel normalization mean.
        std (torch.Tensor): Per-channel normalization std.
        epsilon (float): Maximum perturbation magnitude (default: 8/255).
        steps (int): The number of steps (default: 1).
        nuc_reg (float): Regularizer for the loss (default: 4.0).
        k (float): Bernoulli initialization noise magnitude (default: 4/255).

    Returns:
        tuple:
            - torch.Tensor: Zero tensor (placeholder for compatibility with other training steps).
            - torch.Tensor: Regularization loss computed from adversarial examples.
            - torch.Tensor: Gradient tensor from the final backward pass.
    """

    out = model(x)
    model_training = model.training
    
    eps = (epsilon / std).view(1, -1, 1, 1)
    alpha = (alpha / std).view(1, -1, 1, 1) / steps

    x_adv = x.clone().detach() + k * eps * torch.sign(torch.tensor([0.5]).to(x.device) - torch.rand_like(x))
    x_adv = torch.clamp(x_adv, lower_limit, upper_limit)

    for step in range(steps):
        x_adv.requires_grad = True
        output = model(x_adv)
        loss = F.cross_entropy(output, y) + nuc_reg * torch.norm(out - output, 'nuc') / y.shape[0] # Batch size
        grad = torch.autograd.grad(loss, x_adv)[0].detach()
      
        delta = alpha * torch.sign(grad)
        x_adv = torch.clamp(x_adv + delta, lower_limit, upper_limit)

    delta = torch.clamp(delta, -eps, +eps)
    delta = torch.clamp(x_adv - x, lower_limit - x, upper_limit - x)
    delta = delta.detach()

    if model_training:
        model.train()
    
    out = model(x)
    adv_out = model(x + delta)
    reg_loss =  torch.norm(out - adv_out, 'nuc') / y.shape[0] # Batch size
    
    return torch.zeros_like(x), reg_loss, grad
