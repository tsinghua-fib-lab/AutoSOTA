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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import get_args, get_log_file
import utils.general_utils as utils
import time
import math

SPACE_SORT_KEY_DIM = -1


def set_space_sort_key_dim(dim):
    global SPACE_SORT_KEY_DIM
    # import pdb; pdb.set_trace()
    SPACE_SORT_KEY_DIM = dim


def get_space_sort_key_dim():
    global SPACE_SORT_KEY_DIM
    if SPACE_SORT_KEY_DIM == -1:
        raise ValueError(
            "SPACE_SORT_KEY_DIM is not set. Please set it using set_space_sort_key_dim(dim)."
        )
    return SPACE_SORT_KEY_DIM


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        gt_alpha_mask,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        offload=False,
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.device = "cpu" if offload else "cuda"

        args = get_args()
        log_file = get_log_file()

        if args.time_image_loading:
            start_time = time.time()

        # Single GPU mode - always load
        # load to cpu
        self.original_image_backup = image.contiguous()
        self.image_width = self.original_image_backup.shape[2]
        self.image_height = self.original_image_backup.shape[1]

        if args.time_image_loading:
            log_file.write(f"Image processing in {time.time() - start_time} seconds\n")

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale))
            .transpose(0, 1)
            .to(self.device)
        )
        self.world_view_transform_backup = self.world_view_transform.clone().detach()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .to(self.device)
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.camera_center_cpu = self.camera_center.cpu().tolist()  # [x,y,z]

        # self.K = self.create_k_on_gpu()

    def create_k_on_gpu(self):
        # Set up rasterization configuration
        image_width = int(self.image_width)
        image_height = int(self.image_height)
        tanfovx = math.tan(self.FoVx * 0.5)
        tanfovy = math.tan(self.FoVy * 0.5)
        focal_length_x = self.image_width / (2 * tanfovx)
        focal_length_y = self.image_height / (2 * tanfovy)
        K = torch.tensor(
            [
                [focal_length_x, 0, self.image_width / 2.0],
                [0, focal_length_y, self.image_height / 2.0],
                [0, 0, 1],
            ],
            device="cuda",
        )
        return K

    def get_camera2world(self):
        return self.world_view_transform_backup.t().inverse()

    def update(self, dx, dy, dz):
        # Update the position of this camera pose. TODO: support updating rotation of camera pose.
        with torch.no_grad():
            c2w = self.get_camera2world()
            c2w[0, 3] += dx
            c2w[1, 3] += dy
            c2w[2, 3] += dz

            t_prime = c2w[:3, 3]
            self.T = (-c2w[:3, :3].t() @ t_prime).cpu().numpy()
            # import pdb; pdb.set_trace()

            self.world_view_transform = (
                torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale))
                .transpose(0, 1)
                .to(self.device)
            )
            self.projection_matrix = (
                getProjectionMatrix(
                    znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
                )
                .transpose(0, 1)
                .to(self.device)
            )
            self.full_proj_transform = (
                self.world_view_transform.unsqueeze(0).bmm(
                    self.projection_matrix.unsqueeze(0)
                )
            ).squeeze(0)
            self.camera_center = self.world_view_transform.inverse()[3, :3]


class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
