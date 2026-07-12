"""
Utility functions
"""

from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

import math


# copied from huggingface
def get_cosine_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
                                    num_cycles: float = 0.5, last_epoch: int = -1):
    """
    Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step + 1) / float(max(1, num_warmup_steps + 1))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_openai_lr(model: Module):
    num_params = sum(p.numel() for p in model.parameters())
    return 0.003239 - 0.0001395 * math.log(num_params)


def get_model_size(model: Module):
    param_size = sum(param.nelement() * param.element_size() for param in model.parameters())
    buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())
    param_counts = sum(param.nelement() for param in model.parameters())
    buffer_counts = sum(buffer.nelement() for buffer in model.buffers())
    return (param_size + buffer_size) / 1024 ** 2, param_counts + buffer_counts   # Return size in MB
