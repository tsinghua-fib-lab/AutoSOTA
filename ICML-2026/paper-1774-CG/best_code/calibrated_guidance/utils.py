import torch


def broadcast_time(t: float | torch.Tensor, x: torch.Tensor) -> float | torch.Tensor:
    """
    Broadcasts time tensor t to match the shape of tensor x.

    Float inputs and scalar tensors are returned unchanged.

    :param t: A tensor of shape (batch_size,) or (batch_size, 1)
    :param x: A tensor of shape (batch_size, channels, height, width) or similar
    :return: A tensor of shape (batch_size, 1, 1, 1) or compatible with x for broadcasting
    """
    if not torch.is_tensor(t) or t.shape == ():
        return t
    while t.dim() < x.dim():
        t = t.unsqueeze(-1)
    return t
