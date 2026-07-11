import torch

from dyllm.utils.context import get_context


def gather_rows(x: torch.Tensor, row_idx: torch.Tensor) -> torch.Tensor:
    row_idx = row_idx.long()
    B, K = row_idx.shape
    expand_shape = [B, K] + [1] * (x.dim() - 2)

    idx = row_idx.view(expand_shape)

    expand_sizes = [B, K] + list(x.shape[2:])
    idx = idx.expand(expand_sizes)

    return torch.gather(x, dim=1, index=idx)


def gather_rows_2D(x: torch.Tensor, row_idx: torch.Tensor) -> torch.Tensor:
    """
    x: [L, D]
    row_idx: [K]
    return: [K, D]
    """
    row_idx = row_idx.long()
    K = row_idx.size(0)
    L, D = x.shape

    expand_shape = [K, 1]

    idx = row_idx.view(expand_shape)

    expand_sizes = [K, D]
    idx = idx.expand(expand_sizes)

    return torch.gather(x, dim=0, index=idx)


def scatter_update(original: torch.Tensor, x: torch.Tensor, row_idx: torch.Tensor) -> torch.Tensor:
    B, K = row_idx.shape
    _, L, D = original.shape
    context = get_context()

    idx = row_idx.unsqueeze(-1).expand(B, K, D)
    original.scatter_(dim=1, index=idx, src=x)

    return original


def scatter_update_2D(original: torch.Tensor, x: torch.Tensor, row_idx: torch.Tensor) -> torch.Tensor:
    K, D = x.shape
    idx = row_idx[:, None].expand(K, D)
    original.scatter_(dim=0, index=idx, src=x)

    return original
