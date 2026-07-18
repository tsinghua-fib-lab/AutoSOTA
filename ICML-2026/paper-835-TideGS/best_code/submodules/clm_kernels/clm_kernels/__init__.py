from typing import NamedTuple
import torch.nn as nn
import torch
from clm_kernels._C import (
    set_signal,
    compute_cnt_h,
    extract_ffs,
    scatter_to_bit,
    send_shs2gpu_stream,
    send_shs2cpu_grad_buffer_stream,
    send_shs2gpu_stream_retention,
    send_shs2cpu_grad_buffer_stream_retention,
    fusedssim,
    fusedssim_backward,
    selective_adam_update,
    compute_sh_bwd_inplace,
)
from typing import Optional

allowed_padding = ["same", "valid"]


class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2, padding="same", train=True):
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim(
            C1, C2, img1, img2, train
        )

        if padding == "valid":
            ssim_map = ssim_map[:, :, 5:-5, 5:-5]

        ctx.save_for_backward(img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.C1 = C1
        ctx.C2 = C2
        ctx.padding = padding

        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        C1, C2, padding = ctx.C1, ctx.C2, ctx.padding
        dL_dmap = opt_grad
        if padding == "valid":
            dL_dmap = torch.zeros_like(img1)
            dL_dmap[:, :, 5:-5, 5:-5] = opt_grad
        grad = fusedssim_backward(
            C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12
        )
        return None, None, grad, None, None, None


def fused_ssim(img1, img2, padding="same", train=True):
    C1 = 0.01**2
    C2 = 0.03**2

    assert padding in allowed_padding

    map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)
    return map.mean()


@torch.no_grad()
def spherical_harmonics_bwd_inplace(
    degrees_to_use: int,
    dirs: torch.Tensor,
    coeffs: torch.Tensor,
    v_coeffs: torch.Tensor,
    v_colors: torch.Tensor,
    masks: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    num_bases = coeffs.shape[-2]
    v_dirs = compute_sh_bwd_inplace(
        num_bases,
        degrees_to_use,
        dirs,
        coeffs,
        v_coeffs,
        masks,
        v_colors,
        True,
    )
    return v_dirs


__all__ = [
    "set_signal",
    "compute_cnt_h",
    "extract_ffs",
    "scatter_to_bit",
    "send_shs2gpu_stream",
    "send_shs2cpu_grad_buffer_stream",
    "send_shs2gpu_stream_retention",
    "send_shs2cpu_grad_buffer_stream_retention",
    "fused_ssim",
    "selective_adam_update",
    "spherical_harmonics_bwd_inplace",
]
