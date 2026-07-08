import torch.nn.functional as F


def downsample_pixels_by_bicubic_4x(pixels_nchw_m1to1):
    target_h = pixels_nchw_m1to1.shape[-2] // 4
    target_w = pixels_nchw_m1to1.shape[-1] // 4
    return F.interpolate(
        pixels_nchw_m1to1,
        size=(target_h, target_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
