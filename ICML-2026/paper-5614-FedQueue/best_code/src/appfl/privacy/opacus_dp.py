import torch
from opacus import PrivacyEngine
from typing import Tuple


def make_private_with_opacus(
    privacy_engine,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.0,
    device: str = "cpu",
) -> Tuple[
    torch.nn.Module, torch.optim.Optimizer, torch.utils.data.DataLoader, PrivacyEngine
]:
    """
    Apply DP-SGD using Opacus.
    Wraps the model, optimizer, and dataloader with Opacus' PrivacyEngine.
    """
    # privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    return model.to(device), optimizer, data_loader
