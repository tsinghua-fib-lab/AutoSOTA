# third party
import torch


def sqrt_PEHE(po: torch.Tensor, hat_te: torch.Tensor) -> torch.Tensor:
    """
    Precision in Estimation of Heterogeneous Effect(PyTorch version).
    PEHE reflects the ability to capture individual variation in treatment effects.
    Args:
        po: expected outcome.
        hat_te: estimated outcome.
    """
    po = torch.Tensor(po)
    hat_te = torch.Tensor(hat_te)
    return torch.sqrt(torch.mean(((po[:, 1] - po[:, 0]) - hat_te) ** 2))


def abs_error_ATE(po: torch.Tensor, hat_te: torch.Tensor) -> torch.Tensor:
    """
    Average Treatment Effect.
    ATE measures what is the expected causal effect of the treatment across all individuals in the population.
    Args:
        po: expected outcome.
        hat_te: estimated outcome.
    """
    po = torch.Tensor(po)
    hat_te = torch.Tensor(hat_te)
    return torch.abs(torch.mean(po[:, 1] - po[:, 0]) - torch.mean(hat_te))
