import torch
from torch import Tensor

from .utils import block, q_clamp, q_round, unblock


def _block_fp_quantize(
    x: Tensor,
    width: int = 12,
    exponent_width: int = 8,
    exponent_bias: int = None,
    block_size: list[int] = [16],
    skip_first_dim: bool = False,
):
    """
    - Convert IEEE FP32/64 to Microsoft floating point (MSFP), where an exponent is shared over all elements in a block.
    - `e_shared x [(-1)^s1 x mantissa1, (-1)^s2 x mantissa2, ...]`
    - See https://proceedings.neurips.cc/paper/2020/file/747e32ab0fea7fbd2ad9ec03daa3f840-Paper.pdf

    ---
    - forward: convert IEEE FP32/64 to MSFP
    - backward: STE

    ---
    - `width`: The number of mantissa bits + 1 (the sign bit)
    - `exponent_width`: the number of exponent bits, which is shared over a block
    - `exponent_bias`: the exponent bias, if None, `2**(exponent_bits-1)-1` will be used
    - `block_size`: a list of integers where each integer is the block size on that dimension. See function `block`.

    """
    if isinstance(block_size, int):
        block_size = [block_size]
    # separate x into blocks
    x_shape_before_blocking = [i for i in x.shape]
    blocked_x, per_block_max, padded_x_shape, block_shape = block(
        x, block_shape=block_size, skip_first_dim=skip_first_dim
    )

    # fill zeros to avoid log2(0) = -inf
    if torch.all(per_block_max == 0):
        # all elements in zero-initialized bias can be 0 thus per_block_max is 0
        per_block_max = torch.ones_like(per_block_max)
    else:
        per_block_max[per_block_max == 0] = per_block_max[per_block_max != 0].min()
    # minifloat_denorm_quantizer on each block over which a exponent is shared
    mantissa_bits = width - 1
    if exponent_bias is None:
        exponent_bias = 2 ** (exponent_width - 1) - 1

    exponent_max = 2**exponent_width - 1 - exponent_bias
    exponent_min = -exponent_bias

    mantissa_integer_max = 2**mantissa_bits - 1
    # sign
    per_block_sign = torch.sign(blocked_x + 1e-9)
    # exponent
    per_block_value = torch.abs(blocked_x) + 1e-9
    per_block_exponent = torch.ceil(torch.log2(per_block_max))
    per_block_exponent = q_clamp(per_block_exponent, exponent_min, exponent_max)
    # mantissa
    per_block_mantissa = per_block_value / 2**per_block_exponent
    shift = 2**mantissa_bits
    per_block_mantissa_integer = q_clamp(q_round(per_block_mantissa * shift), 0, mantissa_integer_max)
    per_block_mantissa = per_block_mantissa_integer / shift

    per_block_msfp = per_block_sign * (2**per_block_exponent) * per_block_mantissa
    msfp_x = unblock(
        per_block_msfp,
        x_shape_before_blocking=x_shape_before_blocking,
        padded_x_shape=padded_x_shape,
        block_shape=block_shape,
        skipped_first_dim_when_blocking=skip_first_dim,
    )

    # fmt: off
    # this `is_close_to_0` helps the grad keeps 1 if input x is 0, or the zero-initialized value will be trapped in 0
    is_close_to_0 = torch.isclose(x, torch.tensor([0.0], dtype=x.dtype, device=x.device))
    msfp_x = (~is_close_to_0) * msfp_x + (is_close_to_0) * x
    # fmt: on
    return msfp_x


class BlockFPQuantize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        width: int = 12,
        exponent_width: int = 8,
        exponent_bias: int = None,
        block_size: list[int] = [16],
        skip_first_dim: bool = False,
    ):
        return _block_fp_quantize(
            x,
            width=width,
            exponent_width=exponent_width,
            exponent_bias=exponent_bias,
            block_size=block_size,
            skip_first_dim=skip_first_dim,
        )

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None, None


def block_fp_quantizer(
    x: Tensor,
    width: int = 12,
    exponent_width: int = 8,
    exponent_bias: int = None,
    block_size: list[int] = [16],
    skip_first_dim: bool = False,
):
    """
    - Convert IEEE FP32/64 to Microsoft floating point (MSFP), where an exponent is shared over all elements in a block.
    - `e_shared x [(-1)^s1 x mantissa1, (-1)^s2 x mantissa2, ...]`
    - See https://proceedings.neurips.cc/paper/2020/file/747e32ab0fea7fbd2ad9ec03daa3f840-Paper.pdf

    ---
    - forward: convert IEEE FP32/64 to MSFP
    - backward: STE

    ---
    - `width`: The number of mantissa bits + 1 (the sign bit)
    - `exponent_width`: the number of exponent bits, which is shared over a block
    - `exponent_bias`: the exponent bias, if None, `2**(exponent_bits-1)-1` will be used
    - `block_size`: a list of integers where each integer is the block size on that dimension. See function `block`.

    """
    return BlockFPQuantize.apply(
        x,
        width,
        exponent_width,
        exponent_bias,
        block_size,
        skip_first_dim,
    )
