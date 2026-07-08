import torch
import torch.nn.functional as F


def multi_grad(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float = 8/255, alpha: float = 16/255, samples: int = 3, zeroing_th: float = -1, k: float = 1.0, parallel: bool = True):
    """
    Multi-Gradient Method (ZeroGrad) for FGSM Adversarial Training.

    This method, inspired by the ZeroGrad framework, computes multiple gradients from
    randomly perturbed inputs and combines them to reduce catastrophic overfitting
    observed in single-step adversarial training. Small-magnitude aggregate gradients
    are zeroed out according to a threshold, preventing harmful updates from unstable
    directions.

    The `parallel` option computes gradients for multiple noise initializations in a
    single forward/backward pass for efficiency. The `zeroing_th` argument controls
    how aggressively small aggregate gradients are set to zero.

    Reference:
        Golgooni, Z., Saberi, M., Eskandar, M., & Rohban, M. H. (2021).
        "ZeroGrad: Mitigating and Explaining Catastrophic Overfitting in FGSM
        Adversarial Training."
        arXiv:2103.15476 (https://arxiv.org/abs/2103.15476)

    Args:
        model (torch.nn.Module): The target model.
        x (torch.Tensor): Clean input batch (B, C, H, W).
        y (torch.Tensor): Ground-truth labels.
        upper_limit (torch.Tensor): Per-channel normalized upper bound for inputs.
        lower_limit (torch.Tensor): Per-channel normalized lower bound for inputs.
        mu (torch.Tensor): Per-channel normalization mean.
        std (torch.Tensor): Per-channel normalization std.
        epsilon (float): Maximum perturbation magnitude (default: 8/255).
        alpha (float): Step size for FGSM update (default: 16/255).
        samples (int): Number of random perturbations to sample (default: 3).
        zeroing_th (float): Zeroing threshold for small-magnitude gradients. If set
            to -1, defaults to `samples` (default: -1).
        k (float): Randomization scale factor for the initial noise (default: 1.0).
        parallel (bool): If True, compute gradients for all samples in a single
            batch; otherwise loop over them sequentially.

    Returns:
        tuple:
            - torch.Tensor: Perturbation tensor (`d`) for adversarial training.
            - torch.Tensor: Average gradient across sampled perturbations.
    """
    if zeroing_th==-1:
        zeroing_th = samples
    # Normalize perturbations
    eps = (epsilon / std).view(1, -1, 1, 1)
    alpha = (alpha / std).view(1, -1, 1, 1)
    
    if parallel:
        x_cat = torch.cat([x for i in range(samples)], dim=0)
        # Initialize random step
        delta_cat = torch.empty_like(x_cat).uniform_(-k, k) * eps
        delta_cat = torch.clamp(delta_cat, lower_limit - x_cat, upper_limit - x_cat)
        delta_cat.requires_grad = True

        y_cat = torch.cat([y for i in range(samples)], dim=0)
        output = model(x_cat + delta_cat)
        F.cross_entropy(output, y_cat).backward()
        grad_cat = delta_cat.grad.detach()
        grads = [grad_cat[i*x.size(0):(i+1)*x.size(0)] for i in range(samples)]
    else:
        grads = []
        for _ in range(samples):
            # Initialize random step
            delta = torch.empty_like(x).uniform_(-k, k) * eps
            delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
            delta.requires_grad = True

            output = model(x + delta)
            F.cross_entropy(output, y).backward()

            grads += [torch.clone(delta.grad.detach())]
    g = sum([torch.sign(grads[i]) for i in range(samples)])
    grad = torch.where(torch.abs(g) < 
            (zeroing_th - (samples - zeroing_th)),
            torch.zeros_like(g), g)
    
    d = torch.clamp(alpha * torch.sign(grad), min=-eps, max=eps)
    d = torch.clamp(d, lower_limit - x, upper_limit - x)
    
    avg_grad = sum(grads).detach() / samples
    return d.detach(), avg_grad
