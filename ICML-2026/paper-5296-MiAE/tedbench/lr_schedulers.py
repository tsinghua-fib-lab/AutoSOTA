import math
import torch


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_factor: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine decay LR schedule with a linear warm-up phase.

    The learning rate ramps linearly from ~0 to the base LR over
    ``warmup_steps``, then decays following a cosine curve to
    ``min_factor * base_lr`` at ``max_steps``.

    Args:
        optimizer: The optimizer whose LR is scheduled.
        warmup_steps: Number of linear warm-up steps.
        max_steps: Total number of training steps (end of cosine decay).
        min_factor: Minimum LR as a fraction of the base LR (default 0.0).

    Returns:
        :class:`torch.optim.lr_scheduler.LambdaLR` instance.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-06, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(progress, 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return coeff * (1.0 - min_factor) + min_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_constant_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    max_epochs: int | None = None,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Constant LR schedule with a linear warm-up phase.

    Args:
        optimizer: The optimizer whose LR is scheduled.
        warmup_epochs: Number of linear warm-up epochs (or steps).
        max_epochs: Unused; accepted for API symmetry.

    Returns:
        :class:`torch.optim.lr_scheduler.LambdaLR` instance.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(1e-06, epoch / max(1.0, warmup_epochs))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_layerwise_lr_decay(
    model: torch.nn.Module,
    lr: float,
    lr_decay_factor: float = 0.75,
) -> list[dict]:
    """Build per-layer parameter groups with exponentially decaying learning rates.

    Lower layers receive smaller learning rates, following the layer-wise LR
    decay (LLRD) strategy used in MAE fine-tuning (Clark et al., 2020).
    The paper uses ``lr_decay_factor=0.8`` for fine-tuning MiAE on TEDBench
    (Table 6b).

    Args:
        model: A :class:`~tedbench.model.MiAEEncoder` instance with a
            ``num_layers`` property and a ``get_layer_id_by_param_name``
            method.
        lr: Base learning rate for the top (classification head) layer.
        lr_decay_factor: Multiplicative decay per layer from top to bottom.
            A value of 1.0 disables LLRD (uniform LR).

    Returns:
        List of parameter-group dicts suitable for passing to an optimizer.
    """
    num_layers = model.num_layers

    layer_scales = list(
        lr_decay_factor ** (num_layers - i) for i in range(num_layers + 1)
    )

    param_groups = {}

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        g_decay = "no_decay" if p.dim() < 2 else "decay"
        layer_id = model.get_layer_id_by_param_name(n)
        group_name = f"layer_{layer_id}_{g_decay}"

        if group_name not in param_groups:
            lr_scale = layer_scales[layer_id]
            param_groups[group_name] = {"lr": lr * lr_scale, "params": []}
            if g_decay == "no_decay":
                param_groups[group_name]["weight_decay"] = 0.0

        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())
