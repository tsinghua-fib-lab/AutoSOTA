#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

# #
def getProjectionMatrixCorrect(znear, zfar, H, W, K):

  top = (K[1,2])/K[1,1] * znear
  bottom = -(H - K[1,2])/K[1,1] * znear
  right = (K[0,2])/K[0,0] * znear
  left = -(W - K[0,2])/K[0,0] * znear

  P = torch.zeros(4, 4)

  z_sign = 1.0

  P[0, 0] = 2.0 * znear / (right - left)
  P[1, 1] = 2.0 * znear / (top - bottom)
  P[0, 2] = (right + left) / (right - left)
  P[1, 2] = (top + bottom) / (top - bottom)
  P[3, 2] = z_sign
  P[2, 2] = z_sign * zfar / (zfar - znear)
  P[2, 3] = -(zfar * znear) / (zfar - znear)
  return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def linaer_to_srgb_np(linear,eps=None):
    """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    if eps is None:
        eps = np.finfo(linear.dtype).eps
        # eps = 1e-3

    srgb0 = 323 / 25 * linear
    srgb1 = (211 * np.clip(linear, eps, None) ** (5 / 12) - 11) / 200
    return np.where(linear <= 0.0031308, srgb0, srgb1)
    
def linear_to_srgb(linear, eps=None):
    """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    if eps is None:
        eps = torch.finfo(linear.dtype).eps
        # eps = 1e-3

    srgb0 = 323 / 25 * linear
    srgb1 = (211 * linear.clamp_min(eps) ** (5 / 12) - 11) / 200
    return torch.where(linear <= 0.0031308, srgb0, srgb1)


def srgb_to_linear(srgb, eps=None):
    """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    if eps is None:
        eps = torch.finfo(srgb.dtype).eps
    linear0 = 25 / 323 * srgb
    linear1 = ((200 * srgb + 11) / (211)).clamp_min(eps) ** (12 / 5)
    return torch.where(srgb <= 0.04045, linear0, linear1)

def rotation_between_z(vec):
    """
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    Args:
        vec: [..., 3]

    Returns:
        R: [..., 3, 3]

    """
    v1 = -vec[..., 1]
    v2 = vec[..., 0]
    v3 = torch.zeros_like(v1)
    v11 = v1 * v1
    v22 = v2 * v2
    v33 = v3 * v3
    v12 = v1 * v2
    v13 = v1 * v3
    v23 = v2 * v3
    cos_p_1 = (vec[..., 2] + 1).clamp_min(1e-7)
    R = torch.zeros(vec.shape[:-1] + (3, 3,), dtype=torch.float32, device="cuda")
    R[..., 0, 0] = 1 + (-v33 - v22) / cos_p_1
    R[..., 0, 1] = -v3 + v12 / cos_p_1
    R[..., 0, 2] = v2 + v13 / cos_p_1
    R[..., 1, 0] = v3 + v12 / cos_p_1
    R[..., 1, 1] = 1 + (-v33 - v11) / cos_p_1
    R[..., 1, 2] = -v1 + v23 / cos_p_1
    R[..., 2, 0] = -v2 + v13 / cos_p_1
    R[..., 2, 1] = v1 + v23 / cos_p_1
    R[..., 2, 2] = 1 + (-v22 - v11) / cos_p_1
    R = torch.where((vec[..., 2] + 1 > 0)[..., None, None], R,
                    -torch.eye(3, dtype=torch.float32, device="cuda").expand_as(R))
    return R

def quaternion_product(p, q):
    p_r = p[..., [0]]
    p_i = p[..., 1:]
    q_r = q[..., [0]]
    q_i = q[..., 1:]

    out_r = p_r * q_r - (p_i * q_i).sum(dim=-1)
    out_i = p_r * q_i + q_r * p_i + torch.linalg.cross(p_i, q_i, dim=-1)

    return torch.cat([out_r, out_i], dim=-1)

def quaternion_inverse(p):
    p_r = p[..., [0]]
    p_i = -p[..., 1:]

    return torch.cat([p_r, p_i], dim=-1)

def quaternion_rotate(p, q):
    q_inv = quaternion_inverse(q)

    qp = quaternion_product(q, p)
    out = quaternion_product(qp, q_inv)
    return out

def build_q(vec, angle):
    out_r = torch.cos(angle / 2)
    out_i = torch.sin(angle / 2) * vec

    return torch.cat([out_r, out_i], dim=-1)

def spherical2cartesian(theta, phi):
    x = torch.cos(phi) * torch.sin(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(theta)

    return [x, y, z]

def cartesian2quaternion(x):
    zeros_ = x.new_zeros([*x.shape[:-1], 1])
    return torch.cat([zeros_, x], dim=-1)

def init_predefined_omega(n_theta, n_phi):
    theta_list = torch.arange(n_theta) * 0.5 * np.pi / n_theta + 0.5 * np.pi / (2 * n_theta)
    phi_list = torch.arange(n_phi) * 2 * np.pi / n_phi + 2 * np.pi / (2 * n_phi)

    out_omega = []
    out_omega_lambda = []
    out_omega_mu = []

    for i in range(n_theta):
        theta = theta_list[i].view(1, 1)

        for j in range(n_phi):
            phi = phi_list[j].view(1, 1)

            omega = spherical2cartesian(theta, phi)
            omega = torch.stack(omega, dim=-1).view(1, 3)

            omega_lambda = spherical2cartesian(theta + np.pi / 2, phi)
            omega_lambda = torch.stack(omega_lambda, dim=-1).view(1, 3)

            p = cartesian2quaternion(omega_lambda)
            q = build_q(omega, torch.tensor(np.pi / 2).view(1, 1))
            omega_mu = quaternion_rotate(p, q)[..., 1:]

            out_omega.append(omega)
            out_omega_lambda.append(omega_lambda)
            out_omega_mu.append(omega_mu)

    out_omega = torch.cat(out_omega, dim=0)
    out_omega_lambda = torch.cat(out_omega_lambda, dim=0)
    out_omega_mu = torch.cat(out_omega_mu, dim=0)

    return out_omega.cuda(), out_omega_lambda.cuda(), out_omega_mu.cuda()