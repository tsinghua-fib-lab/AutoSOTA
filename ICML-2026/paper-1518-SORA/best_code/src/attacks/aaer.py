import torch
import torch.nn.functional as F


def l2_square(x,y):
    """
    Computes the mean squared L2 distance between two tensors along the channel dimension.

    This function is typically used to measure the change in logits (or intermediate feature 
    representations) between clean and adversarial examples in robustness-related methods.

    Args:
        x (torch.Tensor): First tensor of shape (B, C, ...).
        y (torch.Tensor): Second tensor of the same shape as `x`.

    Returns:
        torch.Tensor: Mean squared L2 distance over the batch.
    """
    diff = x - y
    diff = diff * diff
    diff = diff.sum(1).mean(0)
    return diff

def fgsm(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float = 8/255, alpha: float = 8/255, k: float = 2.0, clip: bool = False):
    """
    Fast Gradient Sign Method (FGSM) with randomized initialization and optional 
    step-size clipping, adapted for normalized inputs.

    This function perturbs inputs in the direction of the sign of the gradient of the 
    loss function with respect to the input, optionally with a random initial step.

    Reference:
        Runqi Lin, Chaojian Yu, and Tongliang Liu. (2024).
        "Eliminating Catastrophic Overfitting Via Abnormal Ad-versarial Examples Regularization."
        arXiv:2404.08154 (https://arxiv.org/abs/2404.08154)

    Args:
        model (torch.nn.Module): Target model used for gradient computation.
        x (torch.Tensor): Clean input tensor (B, C, H, W).
        y (torch.Tensor): Ground truth labels.
        upper_limit (torch.Tensor): Maximum bound after normalization.
        lower_limit (torch.Tensor): Minimum bound after normalization.
        mu (torch.Tensor): Mean used for normalization.
        std (torch.Tensor): Standard deviation used for normalization.
        epsilon (float): Maximum perturbation magnitude per pixel (default: 8/255).
        alpha (float): Step size for FGSM update (default: 8/255).
        k (float): Scaling factor for randomized initialization.
        clip (bool): Whether to clip perturbations within epsilon bounds.

    Returns:
        tuple:
            - torch.Tensor: Final perturbation tensor.
            - torch.Tensor: Gradient from the first backward pass.
            - torch.Tensor: Clean logits before perturbation.
            - torch.Tensor: Per-sample loss values before attack.
    """
    # Normalize perturbations
    eps = (epsilon / std).view(1, -1, 1, 1)
    alpha = (alpha / std).view(1, -1, 1, 1)
    
    # Initialize random step
    eta = torch.empty_like(x).uniform_(-k, k) * eps
    eta = torch.clamp(eta, lower_limit - x, upper_limit - x)
    eta.requires_grad = True
    
    output = model(x + eta)
    clean_logit = output.detach()
    loss = F.cross_entropy(output, y, reduction='none')
    loss_before = loss.detach()
    loss = loss.mean()
    loss.backward()
    grad = eta.grad.detach()

    delta = eta + alpha * torch.sign(grad)
    if clip:
        delta = torch.clamp(delta, -eps, +eps)
    delta = torch.clamp(delta, lower_limit - x, upper_limit - x)
    delta = delta.detach()
    delta.requires_grad = True
    
    return delta, grad, clean_logit, loss_before


def aaer(loss_before, clean_logit, adv_logit, labels, lambda1: float = 1.0, lambda2: float = 4.0, lambda3: float = 1.5):
    """
    Abnormal Adversarial Example Regularization (AAER) loss.

    AAER compares variations between clean and adversarial logits to identify 
    abnormal adversarial examples (where adversarial loss is lower than clean loss) 
    and applies additional regularization to mitigate them.

    Reference:
        Runqi Lin, Chaojian Yu, and Tongliang Liu. (2024).
        "Eliminating Catastrophic Overfitting Via Abnormal Ad-versarial Examples Regularization."
        arXiv:2404.08154 (https://arxiv.org/abs/2404.08154)

    Args:
        loss_before (torch.Tensor): Per-sample loss values on clean inputs.
        clean_logit (torch.Tensor): Logits on clean inputs.
        adv_logit (torch.Tensor): Logits on adversarial inputs.
        labels (torch.Tensor): Ground-truth labels.
        lambda1 (float): Scaling factor for the abnormal-count proportion term.
        lambda2 (float): Scaling factor for abnormal cross-entropy term.
        lambda3 (float): Scaling factor for constrained variation term.

    Returns:
        torch.Tensor: The AAER-augmented loss value.
    """
    loss = F.cross_entropy(adv_logit, labels, reduction='none')
    loss_after = loss.detach()
    loss = loss.mean()
    abnormal_example = loss_before > loss_after
    normal_example = loss_before <= loss_after
    abnormal_count = torch.count_nonzero(abnormal_example)
    normal_count = torch.count_nonzero(normal_example)
    total_count = abnormal_count + normal_count

    # AAE-CE and AAE-L2
    if abnormal_count != 0:
        abnormal_variation = l2_square(clean_logit[abnormal_example], adv_logit[abnormal_example])
        abnormal_ce = abnormal_example * (loss_before - loss_after)
        abnormal_ce = abnormal_ce.sum() / abnormal_count
    # NAE-L2
    if normal_count != 0:
        normal_variation = l2_square(clean_logit[normal_example], adv_logit[normal_example])
    # AAER
    if abnormal_count != 0 and normal_count != 0:
        constrained_variation = max(abnormal_variation - normal_variation.item(), 0)
        loss = loss + (lambda1 * abnormal_count / total_count) * (lambda2 * abnormal_ce + lambda3 * constrained_variation) # '* min((epoch/20), 1)' warm-up for long training schedule
    
    return loss

