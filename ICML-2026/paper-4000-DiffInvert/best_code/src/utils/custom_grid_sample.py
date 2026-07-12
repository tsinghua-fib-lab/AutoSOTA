import torch
import torch.nn.functional as F


def grid_sample_border(image, optical):
    """https://github.com/pytorch/pytorch/issues/34704#issuecomment-878940122"""
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    # convert [-1, 1] to pixel indices [0, IH-1] and [0, IW-1]
    ix = ((ix + 1) / 2) * (IW-1)
    iy = ((iy + 1) / 2) * (IH-1)

    # corner pixel indices (floors and floors + 1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    # bilinear interpolation weights
    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)

    # clamp indices for gather (padding_mode='border')
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)

        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    # reshape image for gather
    image = image.view(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    # combine using bilinear weights
    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val


def grid_sample_zeros(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    # convert [-1, 1] to pixel indices [0, IH-1] and [0, IW-1]
    ix = ((ix + 1) / 2) * (IW-1)
    iy = ((iy + 1) / 2) * (IH-1)

    # corner pixel indices (floors and floors + 1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    # bilinear interpolation weights
    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)

    # zero masks
    mask_nw = (
        (ix_nw >= 0) & (ix_nw < IW) &
        (iy_nw >= 0) & (iy_nw < IH)
    ).float()
    mask_ne = (
        (ix_ne >= 0) & (ix_ne < IW) &
        (iy_ne >= 0) & (iy_ne < IH)
    ).float()
    mask_sw = (
        (ix_sw >= 0) & (ix_sw < IW) &
        (iy_sw >= 0) & (iy_sw < IH)
    ).float()
    mask_se = (
        (ix_se >= 0) & (ix_se < IW) &
        (iy_se >= 0) & (iy_se < IH)
    ).float()

    # clamp indices for gather
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)

        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    # reshape image for gather
    image = image.view(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    nw_val = nw_val * mask_nw.view(N, 1, H * W)
    ne_val = ne_val * mask_ne.view(N, 1, H * W)
    sw_val = sw_val * mask_sw.view(N, 1, H * W)
    se_val = se_val * mask_se.view(N, 1, H * W)

    # combine using bilinear weights
    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val


if __name__ == "__main__":
    image = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).view(1, 3, 1, 3)
    optical = torch.Tensor([0.9, 0.5, 0.6, -0.7]).view(1, 1, 2, 2)

    custom_impl_border = grid_sample_border(image, optical)
    ref_impl_border = F.grid_sample(image, optical, align_corners=True, mode='bilinear', padding_mode='border')
    print(custom_impl_border)
    print(ref_impl_border)

    custom_impl_zeros = grid_sample_zeros(image, optical)
    ref_impl_zeros = F.grid_sample(image, optical, align_corners=True, mode='bilinear', padding_mode='zeros')
    print(custom_impl_zeros)
    print(ref_impl_zeros)

    image = torch.randn(1, 3, 28, 28)
    optical = torch.randn(1, 28, 28, 2)

    custom_impl_border = grid_sample_border(image, optical)
    ref_impl_border = F.grid_sample(image, optical, align_corners=True, mode='bilinear', padding_mode='border')

    custom_impl_zeros = grid_sample_zeros(image, optical)
    ref_impl_zeros = F.grid_sample(image, optical, align_corners=True, mode='bilinear', padding_mode='zeros')

    print(f"relative error (border): {torch.norm(custom_impl_border - ref_impl_border) / torch.norm(ref_impl_border)}")
    print(f"relative error (zeros): {torch.norm(custom_impl_zeros - ref_impl_zeros) / torch.norm(ref_impl_zeros)}")
    print(f"relative error (border-zeros): {torch.norm(custom_impl_border - ref_impl_zeros) / torch.norm(ref_impl_zeros)}")
    print(f"relative error (zeros-border): {torch.norm(custom_impl_zeros - ref_impl_border) / torch.norm(ref_impl_border)}")

    print(f"max error (border): {torch.max(torch.abs(custom_impl_border - ref_impl_border))}")
    print(f"max error (zeros): {torch.max(torch.abs(custom_impl_zeros - ref_impl_zeros))}")
    print(f"max error (border-zeros): {torch.max(torch.abs(custom_impl_border - ref_impl_zeros))}")
    print(f"max error (zeros-border): {torch.max(torch.abs(custom_impl_zeros - ref_impl_border))}")
