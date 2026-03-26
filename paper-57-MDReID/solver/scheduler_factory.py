""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler


def create_scheduler(cfg, optimizer):
    num_epochs = cfg.SOLVER.MAX_EPOCHS
    lr_min = 0.001 * cfg.SOLVER.BASE_LR
    warmup_t = cfg.SOLVER.WARMUP_ITERS
    if cfg.MODEL.TRANSFORMER_TYPE == 'ViT-B-16':
        warmup_lr_init = 0.1 * cfg.SOLVER.BASE_LR
        noise_range = (0, num_epochs)
    else:
        warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR
        noise_range = None

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_epochs,
        lr_min=lr_min,
        t_mul=1.,
        decay_rate=0.1,
        warmup_lr_init=warmup_lr_init,
        warmup_t=warmup_t,
        cycle_limit=1,
        t_in_epochs=True,
        noise_range_t=noise_range,
        noise_pct=0.67,
        noise_std=1.,
        noise_seed=42,
    )

    return lr_scheduler
