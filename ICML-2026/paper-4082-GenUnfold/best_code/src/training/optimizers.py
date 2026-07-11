# src/training/optimizers.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler, StepLR, ReduceLROnPlateau, CosineAnnealingLR
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_optimizer(model: nn.Module, optimizer_name: str, learning_rate: float, weight_decay: float = 0.0) -> optim.Optimizer:
    """
    Returns a specified PyTorch optimizer for the model parameters.

    Args:
        model (nn.Module): The model whose parameters will be optimized.
        optimizer_name (str): Name of the optimizer ('adam', 'adamw').
        learning_rate (float): The learning rate.
        weight_decay (float): The weight decay (L2 penalty).

    Returns:
        optim.Optimizer: The initialized PyTorch optimizer.

    Raises:
        ValueError: If the optimizer_name is not supported.
    """
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        # AdamW is often preferred for models with weight decay
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Add other optimizers if needed (e.g., SGD, Adadelta, etc.)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Choose from 'adam', 'adamw'.")

    logging.info(f"Initialized optimizer '{optimizer_name}' with lr={learning_rate}, weight_decay={weight_decay}.")
    return optimizer

def get_lr_scheduler(optimizer: optim.Optimizer, scheduler_name: str, **kwargs) -> _LRScheduler:
    """
    Returns a specified PyTorch learning rate scheduler.

    Args:
        optimizer (optim.Optimizer): The optimizer to which the scheduler will be attached.
        scheduler_name (str): Name of the scheduler ('step', 'reduceonplateau', 'cosine', 'none').
        **kwargs: Keyword arguments specific to the chosen scheduler.

    Returns:
        _LRScheduler: The initialized PyTorch learning rate scheduler, or None if 'none'.

    Raises:
        ValueError: If the scheduler_name is not supported or required kwargs are missing.
    """
    scheduler_name = scheduler_name.lower()

    if scheduler_name == 'none':
        logging.info("No learning rate scheduler will be used.")
        return None
    elif scheduler_name == 'step':
        if 'step_size' not in kwargs or 'gamma' not in kwargs:
            raise ValueError("StepLR requires 'step_size' and 'gamma' keyword arguments.")
        scheduler = StepLR(optimizer, step_size=kwargs['step_size'], gamma=kwargs['gamma'])
        logging.info(f"Initialized StepLR scheduler with step_size={kwargs['step_size']}, gamma={kwargs['gamma']}.")
    elif scheduler_name == 'reduceonplateau':
        # Requires a metric to monitor (e.g., validation loss)
        if 'mode' not in kwargs or 'factor' not in kwargs or 'patience' not in kwargs:
             raise ValueError("ReduceLROnPlateau requires 'mode', 'factor', and 'patience'.")
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=kwargs['mode'],       # 'min' or 'max'
            factor=kwargs['factor'],   # Factor by which the learning rate will be reduced
            patience=kwargs['patience'] # Number of epochs with no improvement after which learning rate will be reduced
            # Add other optional parameters like 'threshold', 'cooldown', etc.
        )
        logging.info(f"Initialized ReduceLROnPlateau scheduler with mode='{kwargs['mode']}', factor={kwargs['factor']}, patience={kwargs['patience']}.")
    elif scheduler_name == 'cosine':
        if 'T_max' not in kwargs:
            raise ValueError("CosineAnnealingLR requires 'T_max' (maximum number of training epochs).")
        scheduler = CosineAnnealingLR(optimizer, T_max=kwargs['T_max'], eta_min=kwargs.get('eta_min', 0)) # eta_min is optional
        logging.info(f"Initialized CosineAnnealingLR scheduler with T_max={kwargs['T_max']}, eta_min={kwargs.get('eta_min', 0)}.")

    # Add other schedulers if needed
    else:
        raise ValueError(f"Unsupported LR scheduler: {scheduler_name}. Choose from 'step', 'reduceonplateau', 'cosine', 'none'.")

    return scheduler

# Example Usage
if __name__ == "__main__":
    print("--- Testing optimizers.py ---")

    # Create a dummy model for testing
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        def forward(self, x):
            return self.linear(x)

    dummy_model = DummyModel()

    # Test get_optimizer
    print("\n--- Testing get_optimizer ---")
    try:
        adam_optimizer = get_optimizer(dummy_model, 'adam', learning_rate=1e-3, weight_decay=1e-5)
        print("Adam optimizer created:", adam_optimizer)

        adamw_optimizer = get_optimizer(dummy_model, 'adamw', learning_rate=5e-4)
        print("AdamW optimizer created:", adamw_optimizer)

    except Exception as e:
        print(f"An error occurred during optimizer test: {e}")

    # Test get_lr_scheduler
    print("\n--- Testing get_lr_scheduler ---")
    try:
        # Test StepLR
        step_scheduler = get_lr_scheduler(adam_optimizer, 'step', step_size=30, gamma=0.1)
        print("StepLR scheduler created:", step_scheduler)

        # Test ReduceLROnPlateau
        reduce_scheduler = get_lr_scheduler(adamw_optimizer, 'reduceonplateau', mode='min', factor=0.5, patience=10)
        print("ReduceLROnPlateau scheduler created:", reduce_scheduler)

        # Test CosineAnnealingLR
        cosine_scheduler = get_lr_scheduler(adam_optimizer, 'cosine', T_max=100)
        print("CosineAnnealingLR scheduler created:", cosine_scheduler)

        # Test None scheduler
        none_scheduler = get_lr_scheduler(adamw_optimizer, 'none')
        print("None scheduler created:", none_scheduler)

    except ValueError as e:
        print(f"Scheduler configuration error: {e}")
    except Exception as e:
        print(f"An error occurred during scheduler test: {e}")