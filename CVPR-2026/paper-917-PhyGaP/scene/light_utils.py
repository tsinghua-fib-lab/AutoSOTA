import os
import numpy as np

import torch
import nvdiffrast.torch as dr


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)


def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x


def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN


def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)


def cube_to_dir(s, x, y):
    if s == 0:   rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1: rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2: rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3: rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4: rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5: rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)


def latlong_to_cubemap(latlong_map, res, device='cuda'):
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device=device)
    for s in range(6):
        gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=device), 
                                torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=device)
                                )
        v = safe_normalize(cube_to_dir(s, gx, gy))

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')[0]
    return cubemap


def cubemap_to_latlong(cubemap, res, device='cuda'):
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=device), 
                            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=device)
                            )
    
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi     = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    
    reflvec = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    return dr.texture(cubemap[None, ...], reflvec[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')[0]


def rotate_latlong(latlong_map: torch.Tensor, yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> torch.Tensor:
    """
    Rotate an equirectangular (latlong) HDR image by yaw (horizontal) and pitch (vertical) angles.

    Inputs:
    - latlong_map: torch.Tensor of shape (H, W, C), float32, on any device (preferably CUDA for speed)
    - yaw_deg:   rotation around Y axis in degrees (horizontal rotation)
    - pitch_deg: rotation around X axis in degrees (vertical rotation)

    Output:
    - rotated latlong map with the same shape (H, W, C) and device/dtype as input

    Notes:
    - The rotation is applied as g(d) = f(R^{-1} d), where f is the original environment and g is rotated.
      This corresponds to sampling the original map at direction d_in = R^{-1} d_out.
    - Horizontal wrap (U) is handled by modulo; V is clamped to [0,1].
    """
    if latlong_map.ndim != 3 or latlong_map.shape[-1] not in (1, 3, 4):
        raise ValueError(f"latlong_map must have shape (H, W, C) with C in [1,3,4], got {tuple(latlong_map.shape)}")

    device = latlong_map.device
    dtype = latlong_map.dtype
    H, W, C = latlong_map.shape

    # Create output pixel center grid
    gy, gx = torch.meshgrid(
        torch.linspace(0.0 + 1.0 / H, 1.0 - 1.0 / H, H, device=device, dtype=dtype),
        torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device=device, dtype=dtype)
    )

    # Convert output grid to direction vectors d_out (same convention as cubemap_to_latlong)
    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)
    d_out = torch.stack((
        sintheta * sinphi,
        costheta,
        -sintheta * cosphi
    ), dim=-1)

    # Build rotation R = R_yaw @ R_pitch; then sample original at d_in = R^{-1} d_out = d_out @ R
    yaw = torch.as_tensor(yaw_deg, device=device, dtype=dtype) * (np.pi / 180.0)
    pitch = torch.as_tensor(pitch_deg, device=device, dtype=dtype) * (np.pi / 180.0)

    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cx, sx = torch.cos(pitch), torch.sin(pitch)

    R_yaw = torch.stack([
        torch.stack([cy, torch.zeros((), device=device, dtype=dtype), sy]),
        torch.stack([torch.zeros((), device=device, dtype=dtype), torch.ones((), device=device, dtype=dtype), torch.zeros((), device=device, dtype=dtype)]),
        torch.stack([-sy, torch.zeros((), device=device, dtype=dtype), cy])
    ])
    R_pitch = torch.stack([
        torch.stack([torch.ones((), device=device, dtype=dtype), torch.zeros((), device=device, dtype=dtype), torch.zeros((), device=device, dtype=dtype)]),
        torch.stack([torch.zeros((), device=device, dtype=dtype), cx, -sx]),
        torch.stack([torch.zeros((), device=device, dtype=dtype), sx, cx])
    ])
    R = R_yaw @ R_pitch  # (3,3)

    # d_in = d_out @ R  (row-vector application of R^{-1})
    d_in = torch.matmul(d_out, R)  # (H,W,3)
    d_in = d_in / torch.clamp(torch.norm(d_in, dim=-1, keepdim=True), min=1e-8)

    # Convert directions to (u,v) in [0,1]
    tu = torch.atan2(d_in[..., 0], -d_in[..., 2]) / (2 * np.pi) + 0.5
    tv = torch.acos(torch.clamp(d_in[..., 1], min=-1.0, max=1.0)) / np.pi
    tu = tu % 1.0
    tv = torch.clamp(tv, 0.0, 1.0)
    texcoord = torch.stack((tu, tv), dim=-1)

    # Sample original latlong map with linear filtering
    rotated = dr.texture(latlong_map[None, ...], texcoord[None, ...].contiguous(), filter_mode='linear')[0]
    return rotated


class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        return torch.nn.functional.avg_pool2d(cubemap.permute(0, 3, 1, 2), (2, 2)).permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device=dout.device)
        for s in range(6):
            gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device=dout.device), 
                                    torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device=dout.device)
                                   )
            v = safe_normalize(cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        return out        