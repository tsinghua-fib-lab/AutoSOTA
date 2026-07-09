import logging
import torch

from .utils import q_round, q_clamp


logger = logging.getLogger(__name__)


def int_find_scale(fp_min: float, fp_max: float, n_bits: int, is_affine):
    if is_affine:
        int_min = 0
        int_max = 2**n_bits - 1
    else:
        int_min = -(2 ** (n_bits - 1))
        int_max = 2 ** (n_bits - 1) - 1
        fp_max = max(abs(fp_min), abs(fp_max))
        fp_min = -fp_max

    alpha = fp_min
    beta = fp_max
    alpha_q = int_min
    beta_q = int_max

    scale = (beta - alpha) / (beta_q - alpha_q)

    zero_point = q_round((beta * alpha_q - alpha * beta_q) / (beta - alpha), mode="nearest")

    return (
        scale,
        zero_point,
        int_min,
        int_max,
    )


def int_find_fp_min_max(x: torch.Tensor, quantile: float):
    x = x.flatten()

    n = x.numel()
    k = min(int(n * quantile), n - 1)

    x_sorted, _ = x.sort()

    return x_sorted[k].item(), x_sorted[-k].item()


def int_quantize(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    int_min: int,
    int_max: int,
):
    input_q = q_clamp(
        q_round(input / scale + zero_point, mode="nearest"),
        min_val=int_min,
        max_val=int_max,
    )

    return input_q


def int_dequantize(
    input_q: torch.Tensor,
    scale: float,
    zero_point: int,
):
    input_deq = scale * (input_q - zero_point)
    return input_deq


def _int_quantizer(
    input: torch.Tensor,
    fp_min: float,
    fp_max: float,
    n_bits: int,
    is_affine: bool,
):
    scale, zero_point, int_min, int_max = int_find_scale(fp_min, fp_max, n_bits, is_affine)

    input_q = int_quantize(
        input,
        scale,
        zero_point,
        int_min,
        int_max,
    )

    input_deq = int_dequantize(
        input_q,
        scale,
        zero_point,
    )

    return input_deq


class IntQuantize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        fp_min: float,
        fp_max: float,
        n_bits: int,
        is_affine: bool,
    ):
        return _int_quantizer(
            x,
            fp_min,
            fp_max,
            n_bits,
            is_affine,
        )

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None


def int_quantizer(
    x: torch.Tensor,
    fp_min: float,
    fp_max: float,
    n_bits: int,
    is_affine: bool,
):
    return IntQuantize.apply(
        x,
        fp_min,
        fp_max,
        n_bits,
        is_affine,
    )
