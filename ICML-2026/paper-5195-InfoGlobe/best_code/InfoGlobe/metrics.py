import torch
from torch import Tensor
from torch.nn import functional as F
from .constants import eps

# Part 1 of loss to control the similarity between high dimension data and low dimension data 
def euclidean(input: Tensor, target: Tensor) -> Tensor:
    r"""The `Euclidean distance

    \frac{1}{2} \sum_{n = 0}^{N - 1} (x_n - y_n)^2

    Args:
        input (Tensor): tensor of arbitrary shape
        target (Tensor): tensor of the same shape as input

    Returns:
        Tensor: single element tensor
    """

    return torch.mean((target - input) ** 2)


def kL_div(input: Tensor, target: Tensor) -> Tensor:
    r"""The generalized 'Kullback-Leibler divergence Loss' to measure the difference of 2 distributions
    
    \sum_{n=0}^{N-1} x_n log(\frac{x_n}{y_n}) - x_n + y_n

    Args:
        input(Tensor): tensor of arbitrary shape
        target(Tensor): tensor of the same shape as input
    
    Returns:
        Tensors: single element tensor
    """

    log_input = torch.log(input + eps)
    loss = F.kl_div(log_input.t(), target.t(), reduction='mean') # .sum(dim=1)
    
    return loss 

def fisher_rao_dis(input: Tensor, target: Tensor) -> Tensor:
    r"""Computes the Fisher-Rao distance between pairs of multinomial distributions represented by the columns of `input` and `target`. 

    arccos(\sqrt(x_n)@\sqrt(y_n))

    Args:
        input(Tensor): tensor of arbitrary shape(a multinomial distribution)
        target(Tensor): tensor of arbitrary shape(a multinomial distribution)
    
    Returns:
        Tensors: single element tensor
    """

    sqrt_input = torch.sqrt(input)
    sqrt_target = torch.sqrt(target)

    inner_product = torch.sum(sqrt_input * sqrt_target, dim=0)
    inner_product = torch.clip(inner_product, eps, 1.0-eps)
    
    loss = torch.mean(torch.acos(inner_product))
    
    return loss


def fisher_rao_dis_matrix_block(X, block=1024):
    """
    X: [D, N]
    return: [N, N] distance matrix (block computed)
    """
    device = X.device
    N = X.shape[1]

    sqrt_X = torch.sqrt(X + 1e-12)   # [D, N]
    result = torch.empty((N, N), device=device, dtype=X.dtype)

    for i in range(0, N, block):
        Xi = sqrt_X[:, i:i+block]           # [D, bi]
        for j in range(0, N, block):
            Xj = sqrt_X[:, j:j+block]       # [D, bj]
            inner = torch.matmul(Xi.t(), Xj)   # [bi, bj]
            dist = torch.acos(
                torch.clamp(inner, -1 + 1e-7, 1 - 1e-7)
            )
            result[i:i+Xi.shape[1], j:j+Xj.shape[1]] = dist

    return result

def fisher_rao_dis_matrix(input: Tensor) -> Tensor:
    r"""Computes the all-pairs Fisher-Rao distance matrix for a collection of multinomial distributions,
    where each column of the input matrix represents a distribution. This is an efficient batch version

    Args:
        input(Tensor): tensor of arbitrary shape

    Returns:
        Tensors: single element tensor
    """
    
    sqrt_input = torch.sqrt(input)
    
    inner_product = torch.matmul(sqrt_input.t(), sqrt_input)
    inner_product = torch.clamp(inner_product, eps, 1-eps)
    
    distance = torch.acos(inner_product)

    return distance 

# Part 2 of loss to make the distance of high dimension data and low dimension data same(like MDS but in fisher-rao distance)
def cosine_loss(input: Tensor, target: Tensor) -> Tensor:
    r""" cosine loss for 2 angles

    L(\phi, \theta) = \sqrt(2(1-cos(\phi-\theta)))

    Args:
        input(Tensor): tensor of arbitrary shape
        target(Tensor): tensor of arbitrary shape

    Returns:
        Tensor: single element tensor
    """

    return torch.mean(torch.sqrt(2*(1+eps-torch.cos(torch.abs(target-input)))))

def angle_mse_loss(input: Tensor, target: Tensor) -> Tensor:
    r""" mse loss for 2 angles first item of the taylor spansion of cosine loss

    L(\phi, \theta) = \sqrt(2(1-cos(\phi-\theta)))

    Args:
        input(Tensor): tensor of arbitrary shape
        target(Tensor): tensor of arbitrary shape

    Returns:
        Tensor: single element tensor
    """

    return torch.mean((target-input)**2)

def angle_mae_loss(input: Tensor, target: Tensor) -> Tensor:
    r""" mse loss for 2 angles first item of the taylor spansion of cosine loss

    L(\phi, \theta) = \sqrt(2(1-cos(\phi-\theta)))

    Args:
        input(Tensor): tensor of arbitrary shape
        target(Tensor): tensor of arbitrary shape

    Returns:
        Tensor: single element tensor
    """

    return torch.mean(torch.abs(target-input))

def elastic_net_loss(input, target, alpha=0.2, lam=1.0):
    diff = target - input
    l1 = torch.mean(torch.abs(diff))
    l2 = torch.mean(diff ** 2)
    return lam * (alpha * l1 + (1 - alpha) * l2)

# Part 3 to make the factors sparse
def orthogonal_loss(input: Tensor) -> Tensor:
    r""" if each column of input is orthogonal, then the elements of A^T@A which is not in diagonal is supposed to be zero

    A^T@A = Diag(\lambda_1, \lambda_2, ..., \lambda_n)

    Args:
        input(Tensor): tensor of arbitrary shape

    Return:
        Tensor: single element tensor
    """

    inner_product = torch.matmul(input.t(), input)
    
    n_cols = input.shape[1]
    eye = torch.eye(n_cols, device=input.device)
    
    off_diagonal = inner_product * (1 - eye)
    loss = torch.mean(off_diagonal) 
    
    return loss

def entropy_loss(input: Tensor) -> Tensor:
    r""" entropy of a given distribution

    p*logp

    Args:
        input(Tensor): tensor of arbitrary shape

    Return:
        Tensor: single element tensor
    """

    log_input = torch.log(input + eps)
    entropy = input * log_input
    entropy_total = torch.sum(entropy, dim=0)

    loss = torch.mean(-entropy_total)

    return loss