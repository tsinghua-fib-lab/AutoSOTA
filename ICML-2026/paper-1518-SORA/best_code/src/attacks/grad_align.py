import torch
import torch.nn.functional as F

#Implementation of GradAlign Regularizer
def grad_align(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float = 8/255, alpha: float = 10/255, k: float = 1.0):
    """
    Gradient Alignment Regularizer (GradAlign).

    GradAlign regularization encourages alignment between the input-gradient
    directions of the loss for clean and perturbed images. By penalizing gradient
    misalignment, it mitigates catastrophic overfitting in fast adversarial training
    settings and improves robustness. This implementation uses a per-sample cosine
    similarity measure between the gradients from clean and noise-perturbed inputs.

    Note:
        The original GradAlign repository had a mismatch between the `README` and
        `train.py` code. Here, `create_graph=False` is used for the first gradient
        computation to reduce memory and time costs, as it is not reused for higher-
        order derivatives.

    Reference:
        Andriushchenko, M., & Flammarion, N. (2020).
        "Understanding and Improving Fast Adversarial Training."
        arXiv:2007.02617 (https://arxiv.org/abs/2007.02617)

    Args:
        model (torch.nn.Module): Target model.
        x (torch.Tensor): Clean input batch (B, C, H, W).
        y (torch.Tensor): Ground-truth labels.
        upper_limit (torch.Tensor): Per-channel normalized upper bound for inputs.
        lower_limit (torch.Tensor): Per-channel normalized lower bound for inputs.
        mu (torch.Tensor): Per-channel normalization mean.
        std (torch.Tensor): Per-channel normalization std.
        epsilon (float): Maximum perturbation magnitude (default: 8/255).
        alpha (float): Step size for perturbation update (default: 10/255).
        k (float): Scaling factor for random initialization noise (default: 1.0).

    Returns:
        tuple:
            - torch.Tensor: Perturbation tensor (`delta`) from FGSM-RS update.
            - torch.Tensor: GradAlign regularization term (1 - mean cosine similarity).
            - torch.Tensor: Gradient from second backward pass (`grad2_copy`).
    """
    # Normalize perturbations
    eps = (epsilon / std).view(1, -1, 1, 1)
    alpha = (alpha / std).view(1, -1, 1, 1)
    
    x.requires_grad = True
    preds1 = model(x)
    cost1 = F.cross_entropy(preds1, y)
    # There was a mismatch between the code snippet in README and train.py of the GradAlign repository 
    # regarding the original GradAlign implementation. We set create_graph=False to improve memory and time efficiency.
    grad1 = torch.autograd.grad(cost1, x, create_graph=False)[0]
    grad1 = grad1.detach()
    eta = torch.empty_like(x).uniform_(-k, k) * eps
    eta = torch.clamp(eta, lower_limit - x, upper_limit - x)
    
    x_aug = x + eta
    preds2 = model(x_aug)
    cost2 = F.cross_entropy(preds2, y)
    grad2 = torch.autograd.grad(cost2, x, create_graph=True)[0]
    grad2_copy = grad2.clone().detach()

    alignment = F.cosine_similarity(grad1.reshape(grad1.shape[0], -1), grad2.reshape(grad2.shape[0], -1), dim=1)
    
    # Generate FGSM-RS Sample
    delta = eta + alpha * grad2.sign()
    delta = torch.clamp(delta, min=-eps, max=eps)
    delta = torch.clamp(delta, min=lower_limit - x, max=upper_limit - x)
    delta = delta.detach()
    
    return delta, 1 - alignment.mean(), grad2_copy
