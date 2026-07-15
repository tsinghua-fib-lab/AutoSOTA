import copy
import torch
from typing import Dict, Any


def laplace_mechanism_output_perturb(
    model: torch.nn.Module, sensitivity: float, epsilon: float
) -> Dict[str, Any]:
    """
    Differential privacy for output perturbation based on Laplacian distribution.
    The output perturbation adds Laplacian noise with zero mean to ``model.named_parameters()``.
    Variance is equal to `2*(scale_value)^2`, and `scale_value = sensitivity / epsilon`,
    where `sensitivity` is determined by data and algorithm.
    :param model: torch.nn.Module
    :param sensitivity: sensitivity
    :param epsilon: privacy budget
    :return: model state dictionary with Laplacian noise added
    """
    scale_value = sensitivity / epsilon
    state_dict = copy.deepcopy(model.state_dict())
    for name, param in model.named_parameters():
        mean = torch.zeros_like(param.data)
        scale = torch.zeros_like(param.data) + scale_value
        m = torch.distributions.laplace.Laplace(mean, scale)
        state_dict[name] += m.sample()
    return state_dict


def gaussian_mechanism_output_perturb(
    model: torch.nn.Module, sensitivity: float, epsilon: float, delta: float = 1e-5
) -> Dict[str, Any]:
    """
    Gaussian mechanism for DP.
    Adds Gaussian noise proportional to sensitivity/epsilon.
    """
    sigma = (
        sensitivity * torch.sqrt(2 * torch.log(torch.tensor(1.25 / delta))) / epsilon
    )
    state_dict = copy.deepcopy(model.state_dict())
    for name, param in model.named_parameters():
        noise = torch.normal(
            mean=0.0, std=sigma, size=param.data.size(), device=param.data.device
        )
        state_dict[name] += noise
    return state_dict
