"""Fixed CosineWarmup that handles epoch-based LR stepping.

The original CosineWarmup starts lr_lambda(0)=0, which wastes the first epoch
when used with epoch-based stepping (lr_scheduler.step() at epoch end).
This version shifts the schedule so that epoch 1 gets a non-zero LR.
"""
import math
from functools import partial
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

class CosineWarmupFixed(LambdaLR):
    def __init__(self, optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1):
        lr_lambda = partial(
            self._get_cosine_schedule_with_warmup_lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
        )
        super().__init__(optimizer, lr_lambda, last_epoch)

    @staticmethod
    def _get_cosine_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float):
        # current_step is 0-indexed. Shift so epoch 1 gets > 0 LR.
        if num_warmup_steps > 0:
            if current_step < num_warmup_steps:
                return float(current_step + 1) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


class CosineWarmupRestartsFixed(LambdaLR):
    def __init__(self, optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1):
        lr_lambda = partial(
            self._get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
        )
        super().__init__(optimizer, lr_lambda, last_epoch)

    @staticmethod
    def _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda(
        current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: int
    ):
        if num_warmup_steps > 0:
            if current_step < num_warmup_steps:
                return float(current_step + 1) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))
