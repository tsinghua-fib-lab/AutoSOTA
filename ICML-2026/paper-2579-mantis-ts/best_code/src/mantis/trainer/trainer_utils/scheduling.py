import math


def adjust_learning_rate(epochs, optimizer, loader, step, original_lr):
    """

    Parameters
    ----------
    epochs: int
        Number of epochs.
    optimizer: class object from torch.optim
        Optimizer.
    loader: DataLoader
        Data loader.
    step: int
        Step corresponds to the total number of backpropagation updates has been made to this moment.
        At the end of each epoch, step is equal to the number of passed epochs multiplied by the number of batches.
    original_lr: float
        Original learning rate.
    Returns
    -------

    """
    max_steps = epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = 1
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        eps = 1e-15
        q = 0.5 * (1 + math.cos(math.pi * step / (max_steps + eps)))
        end_lr = base_lr * 0
        lr = base_lr * q + end_lr * (1 - q)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * original_lr
