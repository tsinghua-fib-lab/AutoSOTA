import math
import torch
from torch.autograd.function import InplaceFunction
import torch.nn.functional as F


class QClamp(InplaceFunction):
    @staticmethod
    def forward(ctx, input: torch.Tensor, min_val, max_val):
        ctx.min_val = min_val
        ctx.max_val = max_val
        return input.clamp(min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class QRound(InplaceFunction):
    @staticmethod
    def forward(ctx, input: torch.Tensor, mode="nearest"):
        ctx.mode = mode
        match mode:
            case "nearest":
                return torch.round(input)
            case "truncate":
                return torch.trunc(input)
            case "ceil":
                return torch.ceil(input)
            case "floor":
                return torch.floor(input)
            case _:
                raise ValueError(f"Invalid mode: {mode}")

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def q_clamp(input: torch.Tensor | float, min_val: float, max_val: float):
    if isinstance(input, float):
        return min(max(input, min_val), max_val)
    elif isinstance(input, torch.Tensor):
        return QClamp.apply(input, min_val, max_val)
    else:
        raise ValueError(f"Invalid input type: {type(input)}")


def q_round(input: torch.Tensor | float, mode="nearest"):
    assert mode in ["nearest", "truncate", "ceil", "floor"]
    if isinstance(input, float):
        match mode:
            case "nearest":
                return round(input)
            case "truncate":
                return math.trunc(input)
            case "ceil":
                return math.ceil(input)
            case "floor":
                return math.floor(input)
    elif isinstance(input, torch.Tensor):
        return QRound.apply(input, mode)
    else:
        raise ValueError(f"Invalid input type: {type(input)}")


# --------------------------------
# Block and unblock
# --------------------------------


def _infer_block_shape(x_shape: list[int], block_shape: list[int]):
    """
    Infer a reasonable block shape.
    - right align block_shape with x_shape,
        1. If len(block_shape) > len(x_shape), truncate redundant block_shape dims.
        2. If block_shape.ndim < x_shape.ndim, prepend -1 to block_shape until block_shape.ndim == x_shape.ndim
        - if block_shape[i] < x_shape[i], inferred_block_shape[i] = block_shape[i]
        - if block_shape[i] >= x_shape[i], inferred_block_shape[i] = x_shape[i]
        - if block_shape[i] == -1, inferred_block_shape[i] = x_shape[i]
    """
    x_ndim = len(x_shape)
    block_ndim = len(block_shape)

    if block_ndim >= x_ndim:
        inferred_block_shape = block_shape[-x_ndim:]
    else:
        inferred_block_shape = [-1] * (x_ndim - block_ndim) + block_shape
    for dim_i in range(x_ndim):
        if inferred_block_shape[dim_i] == -1 or inferred_block_shape[dim_i] > x_shape[dim_i]:
            inferred_block_shape[dim_i] = x_shape[dim_i]
        else:
            inferred_block_shape[dim_i] = inferred_block_shape[dim_i]
    return inferred_block_shape


def _infer_padding_shape(x_shape: list[int], block_shape: list[int]):
    """
    Calculate paddings to make x_shape[i] divisable by block_shape[i]
    """
    pad_diff = []
    for x_shape_dim_i, block_shape_dim_i in zip(x_shape, block_shape):
        if block_shape_dim_i == -1 or x_shape_dim_i < block_shape_dim_i:
            pad_diff += [0, 0]
        else:
            num_blocks_dim_i = math.ceil(x_shape_dim_i / block_shape_dim_i)
            new_x_dim_i = num_blocks_dim_i * block_shape_dim_i
            pad_diff += [new_x_dim_i - x_shape_dim_i, 0]
    pad_diff = pad_diff[::-1]
    return pad_diff


def _block_1d_bias(x: torch.Tensor, block_shape: list[int]):
    """
    bias shape: [output_features] -> [num_blocks, block_size]

    The bias of nn.Linear, nn.Conv1d, and nn.Conv2d are all 1D tensors

    ---
    x: a bias with bias.ndim == 1
    """
    assert x.ndim == 1
    x_shape = [i for i in x.shape]
    block_shape = _infer_block_shape(x_shape, block_shape)
    pad_diff = _infer_padding_shape(x_shape, block_shape)
    padded_x = F.pad(x, pad_diff)
    padded_x_shape = torch.tensor(padded_x.shape, dtype=torch.int)
    blocked_x = padded_x.reshape(padded_x_shape[0] // block_shape[0], block_shape[0])
    per_block_max = torch.abs(blocked_x).max(dim=1, keepdim=True)[0]

    return blocked_x, per_block_max, padded_x_shape, block_shape


def _unblock_to_1d_bias(
    blocked_x: torch.Tensor,
    x_shape_before_blocking: list[int],
):
    """
    blocked bias shape: [num_blocks, block_size] -> [output_features]

    ---
    blocked x: blocked bias with blocked_bias.ndim == 2
    """
    x = blocked_x.flatten()

    indexes = []
    for i in range(len(x_shape_before_blocking)):
        indexes.append(slice(None, x_shape_before_blocking[i]))
    # print(f"indexes: {indexes}")
    x = x[indexes]
    return x


def _block_2d_activation(x: torch.Tensor, block_shape: list[int]):
    """
    [batch_size, hidden_size] -> [batch_size, num_blocks, block_size[-1]]
    """
    assert x.ndim == 2
    x_shape = [i for i in x.shape]
    one_batch_shape = [1, x_shape[1]]
    block_shape = _infer_block_shape(one_batch_shape, block_shape=block_shape)
    pad_diff = _infer_padding_shape(x_shape, block_shape=block_shape)
    padded_x = F.pad(x, pad_diff)
    padded_x_shape = torch.tensor(padded_x.shape, dtype=torch.int)
    # [batch_size, hidden_size] -> [batch_size, num_blocks, block_size[-1]]
    blocked_x = padded_x.reshape(x_shape[0], padded_x_shape[1] // block_shape[-1], block_shape[-1])
    per_block_max = torch.abs(blocked_x).max(dim=2, keepdim=True)[0]

    return blocked_x, per_block_max, padded_x_shape, block_shape


def _unblock_to_2d_activation(blocked_x: torch.Tensor, x_shape_before_blocking: list[int]):
    """
    [batch_size, num_blocks, block_size] -> [batch_size, hidden_size]
    """
    x = blocked_x.flatten(1)

    indexes = []
    for i in range(len(x_shape_before_blocking)):
        indexes.append(slice(None, x_shape_before_blocking[i]))
    # print(f"indexes: {indexes}")
    x = x[indexes]
    return x


def _block_2d_weight(x: torch.Tensor, block_shape: list[int]):
    """
    [in_features, out_features] -> [block_size_0 * block_size_1, num_blocks]

    """
    assert x.ndim == 2
    x_shape = [i for i in x.shape]
    block_shape = _infer_block_shape(x_shape, block_shape)
    pad_diff = _infer_padding_shape(x_shape, block_shape)
    padded_x = F.pad(x, pad_diff)
    padded_x_shape = torch.tensor(padded_x.shape, dtype=torch.int)

    padded_x = padded_x.unsqueeze(0).unsqueeze(0)
    # [1, 1, in_features, out_features] -> [1, block_size_0 * block_size_1, num_blocks]
    blocked_x = F.unfold(padded_x, kernel_size=block_shape, dilation=1, padding=0, stride=block_shape)

    # [1, block_size_0 * block_size_1, num_blocks] -> [block_size_0 * block_size_1, num_blocks]
    blocked_x = blocked_x.squeeze(0)
    per_block_max = torch.abs(blocked_x).max(dim=0, keepdim=True)[0]

    return blocked_x, per_block_max, padded_x_shape, block_shape


def _unblock_to_2d_weight(blocked_x: torch.Tensor, x_shape_before_blocking, padded_x_shape, block_shape):
    """
    [block_size_0 * block_size_1, num_blocks] -> [in_features, out_features]
    """
    # [block_size_0 * block_size_1, num_blocks] -> [1, padded_x_shape[0], padded_x_shape[1]]
    x = F.fold(
        blocked_x,
        output_size=padded_x_shape,  # [padded_in_features, padded_out_features]
        kernel_size=block_shape,  # [block_shape_0, block_shape_1]
        dilation=1,
        padding=0,
        stride=block_shape,
    )

    x = x.squeeze(0)
    indexes = []
    for i in range(len(x_shape_before_blocking)):
        indexes.append(slice(None, x_shape_before_blocking[i]))
    x = x[indexes]
    # print(f"indexes: {indexes}")
    return x


def _block_3d_activation(x: torch.Tensor, block_shape: list[int]):
    """
    [batch_size, hidden_dim_0, hidden_dim_1] -> [batch_size, block_size_0 * block_size_1, num_blocks]

    ---
    Return blocked_x, per_block_max, padded_x_shape, block_shape
    """
    assert x.ndim == 3
    x_shape = [i for i in x.shape]
    one_batch_shape = [1, *x_shape[1:]]
    block_shape = _infer_block_shape(one_batch_shape, block_shape)  # [1, ...]
    pad_diff = _infer_padding_shape(one_batch_shape, block_shape)
    padded_x = F.pad(x, pad_diff)
    padded_x_shape = torch.tensor(padded_x.shape, dtype=torch.int)
    padded_x = padded_x.unsqueeze(1)
    # [batch_size, 1, num_tokens, hidden_size] -> [batch_size, block_size_0 * block_size_1, num_blocks]
    blocked_x = F.unfold(
        padded_x,
        kernel_size=block_shape[1:],
        dilation=1,
        padding=0,
        stride=block_shape[1:],
    )

    per_block_max = torch.abs(blocked_x).max(dim=1, keepdim=True)[0]

    return blocked_x, per_block_max, padded_x_shape, block_shape


def _unblock_to_3d_activation(blocked_x: torch.Tensor, x_shape_before_blocking, padded_x_shape, block_shape):
    # [batch_size, block_size_0 * block_size_1, num_blocks] -> [batch_size, 1, padded_x_shape_1, padded_x_shape_2]
    x = F.fold(
        blocked_x,
        output_size=padded_x_shape[1:],
        kernel_size=block_shape[1:],
        dilation=1,
        padding=0,
        stride=block_shape[1:],
    )
    x = x.squeeze(1)
    indexes = []
    for i in range(len(x_shape_before_blocking)):
        indexes.append(slice(None, x_shape_before_blocking[i]))
    x = x[indexes]
    # print(f"indexes: {indexes}")
    return x


def block(x: torch.Tensor, block_shape: list[int], skip_first_dim: bool = False):
    """
    - skip_first_dim (bool): If True, block_shape[0] will always take 1.

    ---
    Return (blocked_x, per_block_max, padded_x_shape, block_shape)
    """
    if x.ndim == 1:
        assert skip_first_dim is False, "skip_first_dim must be False for bias to be blocked"
        return _block_1d_bias(x, block_shape)
    elif x.ndim == 2:
        if skip_first_dim:
            return _block_2d_activation(x, block_shape)
        else:
            return _block_2d_weight(x, block_shape)
    elif x.ndim == 3:
        if skip_first_dim:
            return _block_3d_activation(x, block_shape)
        else:
            raise NotImplementedError("block 3d weight is not supported.")
    else:
        raise RuntimeError(f"Unsupported x.ndim = {x.ndim}")


def unblock(
    blocked_x: torch.Tensor,
    x_shape_before_blocking: list[int],
    padded_x_shape,
    block_shape: list[int],
    skipped_first_dim_when_blocking: bool = True,
):
    if len(x_shape_before_blocking) == 1:
        assert skipped_first_dim_when_blocking is False, "first dim of bias can not have been skipped in blocking"
        return _unblock_to_1d_bias(blocked_x, x_shape_before_blocking)
    elif len(x_shape_before_blocking) == 2:
        if skipped_first_dim_when_blocking:
            return _unblock_to_2d_activation(blocked_x, x_shape_before_blocking)
        else:
            return _unblock_to_2d_weight(
                blocked_x,
                x_shape_before_blocking,
                padded_x_shape,
                block_shape,
            )
    elif len(x_shape_before_blocking) == 3:
        if skipped_first_dim_when_blocking:
            return _unblock_to_3d_activation(blocked_x, x_shape_before_blocking, padded_x_shape, block_shape)
        else:
            raise NotImplementedError("unblock to 3d weight is not supported")
    else:
        raise RuntimeError("Unsupported n.dims ({}) to unblock back".format(len(x_shape_before_blocking)))
