import torch
import torch.nn.functional as F
import copy

#Implement ELLE Regularizer
def elle(model, x, y, upper_limit, lower_limit, mu, std, epsilon: float = 8/255, alpha: float = 8/255, k: float = 1.0):
    """
    Implementation of ELLE (Efficient Local Linearity Enforcement) regularizer.

    ELLE improves adversarial robustness by enforcing the *linearity* of the loss
    across interpolations between adversarial examples in input space. It is applied
    in conjunction with adversarial training and helps the model maintain consistent
    gradients for better generalization against perturbations.

    Reference:
        Elias Abad Rocamora, Fanghui Liu, Grigorios G. Chrysos, Pablo M. Olmos, and Volkan Cevher. (2024).
        "Efficient Local Linearity Regularization to Overcome Catastrophic Overfitting."
        arXiv:2401.11618 (https://arxiv.org/abs/2401.11618)
        Github: https://github.com/LIONS-EPFL/ELLE

    Args:
        model (torch.nn.Module): Model to be regularized.
        x (torch.Tensor): Clean input batch (B, C, H, W).
        y (torch.Tensor): Ground-truth labels.
        upper_limit (torch.Tensor): Per-channel normalized maximum.
        lower_limit (torch.Tensor): Per-channel normalized minimum.
        mu (torch.Tensor): Per-channel normalization mean.
        std (torch.Tensor): Per-channel normalization std.
        epsilon (float): Max perturbation magnitude (default: 8/255).
        alpha (float): Step size for FGSM update (default: 8/255).
        k (float): Randomization factor for initial perturbations (default: 1.0).

    Returns:
        tuple:
            - torch.Tensor: Adversarial perturbation (`delta`).
            - torch.Tensor: ELLE linearity regularization term.
            - torch.Tensor: Gradient of loss wrt adversarial input from first step.
    """
    # Normalize perturbations
    eps = (epsilon / std).view(1, -1, 1, 1)
    alpha = (alpha / std).view(1, -1, 1, 1)

    # Generate adversarial example using FGSM-RS
    # Initialize random step
    eta = torch.empty_like(x).uniform_(-k, k) * eps
    eta = torch.clamp(eta, lower_limit - x, upper_limit - x)
    x_adv = copy.deepcopy(x) + eta

    x_adv.requires_grad=True
    outputs = model(x_adv)
    loss = F.cross_entropy(outputs, y)
    loss.backward(retain_graph=True)
    grads_input = copy.deepcopy(x_adv.grad)
    grad = x_adv.grad.detach()
    
    x_adv = x_adv + alpha * torch.sign(grads_input)
    x_adv = torch.clamp(x_adv, x - eps, x + eps).detach()
    model.zero_grad()

    x_adv.detach()

    bs = x.shape[0]
    x_ab = x.repeat([2,1,1,1]) 
    etaab = torch.empty_like(x_ab).uniform_(-k, k) * eps
    etaab = torch.clamp(etaab, lower_limit - x_ab, upper_limit - x_ab)
    x_ab = x_ab + etaab
    alphaa = torch.rand([bs,1,1,1],device = x.device)
    x_c = (1-alphaa)*x_ab[:bs] + alphaa*x_ab[bs:]
    alphaa = alphaa.squeeze()

    # Forward pass
    losses = F.cross_entropy(model(torch.cat((x_ab,x_c),dim=0)), y.repeat([3]), reduction='none')

    # Regularization term
    lin_err = F.mse_loss(losses[2*bs:], (1-alphaa)*losses[:bs] + alphaa*losses[bs:2*bs])

    delta = (x_adv - x).detach()
    return delta, lin_err, grad
