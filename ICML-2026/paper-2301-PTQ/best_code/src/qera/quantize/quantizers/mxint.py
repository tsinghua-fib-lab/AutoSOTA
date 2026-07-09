import torch


def group_tensor(x: torch.Tensor, block_size: int, block_axis: int) -> tuple[torch.Tensor, tuple, tuple]:
    """Group the elements into blocks along the specified axis.
    - Only support 1D, 2D, or 3D tensor.
    - When x is 3D tensor, cannot group along batch axis (block_axis=0).
    - Use the view and permute to restore grouped x to the original shape.

    :param torch.Tensor x: 1D, 2D, or 3D tensor
    :param int block_size: number of elements in each block
    :param int block_axis: Group the elements into blocks along the specified axis
    :raises ValueError: illegal block_axis
    :raises NotImplementedError: illegal tensor dimension and shape
    :return tuple[torch.Tensor, tuple, tuple]: grouped tensor, view_args, permute_args

    .. code-block:: python

        >>> x = torch.arange(12).reshape(3, 4)
        >>> block_size = 2
        >>> block_axis = -1
        >>> print(x)
        tensor([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> x, view_args, permute_args = _group_tensor(x, block_size, block_axis)
        >>> print(x)
        tensor([[ 0,  1],
                [ 2,  3],
                [ 4,  5],
                [ 6,  7],
                [ 8,  9],
                [10, 11]])
        >>> print(x.view(view_args).permute(permute_args))
        tensor([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
    """

    if block_axis < 0:
        block_axis = x.ndim + block_axis

    ori_shape = x.size()
    if x.ndim == 1:
        return x.reshape(-1, block_size), ori_shape, (0,)
    elif x.ndim == 2:
        if block_axis == 0:
            permute_args = (1, 0)
            x = x.permute(1, 0)
            view_args = x.size()
            x = x.contiguous()
            return x.reshape(-1, block_size), view_args, permute_args
        else:
            permute_args = (0, 1)
            return x.view(-1, block_size), ori_shape, permute_args
    elif x.ndim == 3:
        if block_axis == 1:
            permute_args = (0, 2, 1)
            x = x.permute(0, 2, 1)
            view_args = x.size()
            x = x.contiguous()
            return x.reshape(-1, block_size), view_args, permute_args
        elif block_axis == 2:
            permute_args = (0, 1, 2)
            view_args = x.size()
            return x.reshape(-1, block_size), view_args, permute_args
        else:
            raise ValueError("cannot group along batch axis for 3D tensor")
    else:
        raise NotImplementedError("Only support 1D, 2D tensor, and 3D activation tensor")


def pad_zeros_if_necessary(x: torch.Tensor, block_size: int, block_axis: int) -> torch.Tensor:
    """Append zeros to x if the number of elements along block_axis is not a multiple of block_size, else return x.

    :param torch.Tensor x: input tensor
    :param int block_size: number of elements in each block
    :param int block_axis: group the elements into blocks along the specified axis
    :return torch.Tensor: padded tensor
    """

    if x.shape[block_axis] % block_size == 0:
        return x

    pad_size = block_size - x.shape[block_axis] % block_size
    pad_shape = list(x.shape)
    pad_shape[block_axis] = pad_size
    pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    x = torch.cat([x, pad], dim=block_axis)
    return x


def _check_shape_mxint(x: torch.Tensor, block_size: int, block_axis: int):
    assert x.ndim >= 1, "x must have at least 1 dimension"
    # assert (
    #     x.shape[block_axis] % block_size == 0
    # ), f"block_size (={block_size}) must divide the number of elements along block_axis (= {x.shape[block_axis]})"

    if x.ndim == 1:
        assert block_axis in [0, -1], "block_axis must be 0 or -1 for 1D tensor"
    elif x.ndim == 2:
        assert block_axis in [0, 1, -1, -2], "block_axis must be 0, 1, -1, or -2 for 2D tensor"
    elif x.ndim == 3:
        assert block_axis != 0, "cannot group along batch axis for 3D tensor"
        assert block_axis in [1, 2, -2, -1], "block_axis must be 1, 2, -2, or -1 for 3D tensor"
    else:
        raise NotImplementedError("Only support 1D, 2D tensor, and 3D activation tensor")
    return True


def _mxint_quantizer(x: torch.Tensor, width: int, block_size: int, block_axis: int) -> torch.Tensor:
    ori_type = x.dtype
    assert ori_type in [torch.float32, torch.float16, torch.bfloat16, torch.float64]
    x = x.to(torch.float32)
    assert width <= 8 and width >= 2
    assert _check_shape_mxint(x, block_size, block_axis)

    ori_shape = x.size()
    # group the elements into blocks along the specified axis
    x = pad_zeros_if_necessary(x, block_size, block_axis)
    x, view_args, permute_args = group_tensor(x, block_size, block_axis)

    sign = x < 0

    # set subnormal numbers to 0
    x = x.abs()
    is_normal = x >= torch.finfo(torch.bfloat16).smallest_normal
    x = torch.where(is_normal, x, 0.0)

    is_zeros = torch.all(x == 0.0, dim=1, keepdim=True)
    # extract exponent
    exponent = (x.view(dtype=torch.int32) >> 23) & 0xFF

    # use the max exponent as the shared scale
    group_max_exp = exponent.max(dim=1, keepdim=True).values
    group_max_exp = torch.where(is_zeros, 1, group_max_exp)
    group_max_exp = (group_max_exp << 23).view(torch.float32)

    # elements after the shared scale is extracted
    x = x / group_max_exp

    # round the elements to the nearest fixed-point number
    # note that the element of the MXINT has 1 sign bit, and (width - 1) bits for the mantissa
    # the radix point is after the first bit of the mantissa.
    # for example, the mantissa of MXINT8 follows the form:  _ . _ _ _ _ _ _ , i.e., 1-bit before the radix point, 6-bit after the radix point
    x = x * (2 ** (width - 2))
    x = x.round().clamp(0, 2 ** (width - 1) - 1)

    x = x * group_max_exp / (2 ** (width - 2))
    x = torch.where(sign, -x, x)

    # restore x to the original shape
    x = x.view(view_args).permute(permute_args)
    # if len(ori_shape) == n, then slice x to ori_shape by x[:ori_shape[0], :ori_shape[1], ..., :ori_shape[n-1]]
    x = x[tuple(slice(ori_shape[i]) for i in range(len(ori_shape)))]

    x = x.to(ori_type)
    return x


class MXINTQuantize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        width: int,
        block_size: int,
        block_axis: int,
    ):
        return _mxint_quantizer(x, width, block_size, block_axis)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None


def mxint_quantizer(x: torch.Tensor, width: int, block_size: int, block_axis: int):
    """Emulated quantizer from bfloat16 to mxint8.

    :param torch.Tensor x: torch.bfloat16 tensor
    :param int block_size: number of elements in each block
    :param int block_axis: group the elements into blocks along the specified axis
    :return torch.Tensor: emulated mxint tensor with the same shape as x, dtype=torch.bfloat16
    """
    return MXINTQuantize.apply(x, width, block_size, block_axis)
