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
import sys
from datetime import datetime
import numpy as np
import random
import os
import time
from argparse import Namespace
import psutil

ARGS = None
LOG_FILE = None
CUR_ITER = None
TIMERS = None
DENSIFY_ITER = 0


def set_args(args):
    global ARGS
    ARGS = args


def get_args():
    global ARGS
    return ARGS


def set_log_file(log_file):
    global LOG_FILE
    LOG_FILE = log_file


def get_log_file():
    global LOG_FILE
    return LOG_FILE


def set_cur_iter(cur_iter):
    global CUR_ITER
    CUR_ITER = cur_iter


def get_cur_iter():
    global CUR_ITER
    return CUR_ITER


def set_timers(timers):
    global TIMERS
    TIMERS = timers


def get_timers():
    global TIMERS
    return TIMERS


BLOCK_X, BLOCK_Y = 16, 16
ONE_DIM_BLOCK_SIZE = 256
IMG_H, IMG_W = None, None
TILE_Y, TILE_X = None, None


def set_block_size(x, y, z):
    global BLOCK_X, BLOCK_Y, ONE_DIM_BLOCK_SIZE
    BLOCK_X, BLOCK_Y, ONE_DIM_BLOCK_SIZE = x, y, z


def set_img_size(h, w):
    global IMG_H, IMG_W, TILE_Y, TILE_X
    IMG_H, IMG_W = h, w
    TILE_Y = (IMG_H + BLOCK_Y - 1) // BLOCK_Y
    TILE_X = (IMG_W + BLOCK_X - 1) // BLOCK_X


def get_img_size():
    global IMG_H, IMG_W
    return IMG_H, IMG_W


def get_img_width():
    global IMG_W
    return IMG_W


def get_img_height():
    global IMG_H
    return IMG_H


def get_num_pixels():
    global IMG_H, IMG_W
    return IMG_H * IMG_W


def get_denfify_iter():
    global DENSIFY_ITER
    return DENSIFY_ITER


def inc_densify_iter():
    global DENSIFY_ITER
    DENSIFY_ITER += 1


def print_rank_0(str):
    print(str)


def check_enable_python_timer():
    args = get_args()
    iteration = get_cur_iter()
    return args.enable_timer and check_update_at_this_iter(
        iteration, args.bsz, args.log_interval, 1
    )


def check_update_at_this_iter(iteration, bsz, update_interval, update_residual):
    residual_l = iteration % update_interval
    residual_r = residual_l + bsz
    # residual_l <= update_residual < residual_r
    if residual_l <= update_residual and update_residual < residual_r:
        return True
    # residual_l <= update_residual+update_interval < residual_r
    if (
        residual_l <= update_residual + update_interval
        and update_residual + update_interval < residual_r
    ):
        return True
    return False


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def check_initial_gpu_memory_usage(prefix):
    if get_cur_iter() not in [0, 1]:
        return
    args = get_args()
    log_file = get_log_file()
    if (
        hasattr(args, "check_gpu_memory")
        and args.check_gpu_memory
        and log_file is not None
    ):
        log_file.write(
            "check_gpu_memory["
            + prefix
            + "]: Memory usage: {} GB. Max Memory usage: {} GB. Now reserved memory: {} GB. Max reserved memory: {} GB\n".format(
                torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
                torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024,
                torch.cuda.memory_reserved() / 1024 / 1024 / 1024,
                torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024,
            )
        )


def gaussian_report(gaussians):
    iteration = get_cur_iter()
    args = get_args()
    log_file = get_log_file()

    if (iteration % args.log_interval) == 1:
        log_file.write(
            "### Iteration {}, num of gaussians = {} ###\n".format(
                iteration, gaussians.get_xyz.shape[0]
            )
        )


def memory_report(prefix):
    iteration = get_cur_iter()
    args = get_args()
    log_file = get_log_file()
    p = psutil.Process()

    if (iteration % args.log_interval) == 1:
        log_file.write(
            "[Memory Report] {}\n".format(prefix)
            + " -> [CPU] Memory Usage: {:.3f} GB. Available Memory: {:.3f} GB. Total memory: {:.3f} GB.\n".format(
                psutil.virtual_memory().used / 1024 / 1024 / 1024,
                psutil.virtual_memory().available / 1024 / 1024 / 1024,
                psutil.virtual_memory().total / 1024 / 1024 / 1024,
            )
            + " -> [CPU more] rss: {:.3f} GB, vms: {:.3f} GB, shared: {:.3f} GB, text: {:.3f} GB, lib: {:.3f} GB, data: {:.3f} GB, dirty: {:.3f} GB.\n".format(
                p.memory_info().rss / 1024 / 1024 / 1024,
                p.memory_info().vms / 1024 / 1024 / 1024,
                p.memory_info().shared / 1024 / 1024 / 1024,
                p.memory_info().text / 1024 / 1024 / 1024,
                p.memory_info().lib / 1024 / 1024 / 1024,
                p.memory_info().data / 1024 / 1024 / 1024,
                p.memory_info().dirty / 1024 / 1024 / 1024,
            )
            + " -> [GPU] Memory usage: {:.3f} GB. Max Memory usage: {:.3f} GB. Now reserved memory: {:.3f} GB. Max reserved memory: {:.3f} GB\n".format(
                torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
                torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024,
                torch.cuda.memory_reserved() / 1024 / 1024 / 1024,
                torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024,
            )
        )


def check_memory_usage(log_file, args, iteration, gaussians, before_densification_stop):
    p = psutil.Process()

    memory_usage = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    max_memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
    max_reserved_memory = torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024
    now_reserved_memory = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
    now_pinned_memory = p.memory_info().shared / 1024 / 1024 / 1024
    log_str = ""
    log_str += "iteration[{},{}) {}Now num of 3dgs: {}. Now Memory usage: {} GB. Max Memory usage: {} GB. Max Reserved Memory: {} GB. Now Reserved Memory: {} GB. Now Pinned Memory: {} GB\n".format(
        iteration,
        iteration + args.bsz,
        "densify_and_prune. " if not before_densification_stop else "",
        gaussians.get_xyz.shape[0],
        memory_usage,
        max_memory_usage,
        max_reserved_memory,
        now_reserved_memory,
        now_pinned_memory,
    )
    if args.log_memory_summary:
        log_str += "Memory Summary: {} GB \n".format(torch.cuda.memory_summary())

    if args.check_gpu_memory:
        log_file.write(log_str)


def PILtoTorch(pil_image, resolution, args, log_file, decompressed_image=None):
    if decompressed_image is not None:
        return decompressed_image
    pil_image.load()
    resized_image_PIL = pil_image.resize(resolution)
    if args.time_image_loading:
        start_time = time.time()
    resized_image = torch.from_numpy(np.array(resized_image_PIL))
    if args.time_image_loading:
        log_file.write(f"pil->numpy->torch in {time.time() - start_time} seconds\n")
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def safe_state(silent, log_file=None):
    old_f = sys.stdout

    class F:
        def __init__(self, silent, log_file=None):
            self.silent = silent
            self.log_file = log_file

        def write(self, x):
            if self.silent:
                target = self.log_file or get_log_file()
                if target is not None and not target.closed:
                    target.write(x)
                    if x.endswith("\n"):
                        target.flush()
                return
            else:
                if x.endswith("\n"):
                    old_f.write(
                        x.replace(
                            "\n",
                            " [{}]\n".format(
                                str(datetime.now().strftime("%d/%m %H:%M:%S"))
                            ),
                        )
                    )
                else:
                    old_f.write(x)

        def flush(self):
            if self.silent:
                target = self.log_file or get_log_file()
                if target is not None and not target.closed:
                    target.flush()
            else:
                old_f.flush()

    sys.stdout = F(silent, log_file)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda", 0))


def prepare_output_and_logger(args):
    # Set up output folder
    print_rank_0("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(
        os.path.join(args.model_path, "cfg_args"), "w"
    ) as cfg_log_f:  # TODO: I want to delete cfg_args file.
        cfg_log_f.write(str(Namespace(**vars(args))))


def log_cpu_memory_usage(position_str):
    args = get_args()
    p = psutil.Process()

    if not args.check_cpu_memory:
        return
    LOG_FILE.write(
        "[Check CPU Memory]"
        + position_str
        + " ->  Memory Usage: {:.3f} GB. Available Memory: {:.3f} GB. Total memory: {:.3f} GB.\n".format(
            psutil.virtual_memory().used / 1024 / 1024 / 1024,
            psutil.virtual_memory().available / 1024 / 1024 / 1024,
            psutil.virtual_memory().total / 1024 / 1024 / 1024,
        )
        + " -> rss: {:.3f} GB, vms: {:.3f} GB, shared: {:.3f} GB, text: {:.3f} GB, lib: {:.3f} GB, data: {:.3f} GB, dirty: {:.3f} GB.\n".format(
            p.memory_info().rss / 1024 / 1024 / 1024,
            p.memory_info().vms / 1024 / 1024 / 1024,
            p.memory_info().shared / 1024 / 1024 / 1024,
            p.memory_info().text / 1024 / 1024 / 1024,
            p.memory_info().lib / 1024 / 1024 / 1024,
            p.memory_info().data / 1024 / 1024 / 1024,
            p.memory_info().dirty / 1024 / 1024 / 1024,
        )
    )


def drop_duplicate_gaussians(model_params, drop_duplicate_gaussians_coeff):
    if drop_duplicate_gaussians_coeff == 1.0:
        return model_params

    active_sh_degree = model_params[0]
    xyz = model_params[1]
    features_dc = model_params[2]
    features_rest = model_params[3]
    scaling = model_params[4]
    rotation = model_params[5]
    opacity = model_params[6]
    max_radii2D = model_params[7]
    xyz_gradient_accum = model_params[8]
    denom = model_params[9]
    opt_dict = None
    spatial_lr_scale = model_params[11]

    all_indices = torch.arange(
        int(xyz.shape[0] * drop_duplicate_gaussians_coeff), device=xyz.device
    )
    keep_indices = all_indices % xyz.shape[0]

    return (
        active_sh_degree,
        nn.Parameter(xyz[keep_indices].requires_grad_(True)),
        nn.Parameter(features_dc[keep_indices].requires_grad_(True)),
        nn.Parameter(features_rest[keep_indices].requires_grad_(True)),
        nn.Parameter(scaling[keep_indices].requires_grad_(True)),
        nn.Parameter(rotation[keep_indices].requires_grad_(True)),
        nn.Parameter(opacity[keep_indices].requires_grad_(True)),
        max_radii2D[keep_indices],
        xyz_gradient_accum[keep_indices],
        denom[keep_indices],
        opt_dict,
        spatial_lr_scale,
    )


def load_checkpoint(args):
    # Single GPU mode - load checkpoint from single file
    if args.start_checkpoint[-1] != "/":
        args.start_checkpoint += "/"

    # Find the checkpoint file in the directory
    checkpoint_files = [
        f for f in os.listdir(args.start_checkpoint) if f.endswith(".pth")
    ]
    assert len(checkpoint_files) > 0, "No checkpoint files found in the directory"

    # Use the first checkpoint file (assuming single GPU checkpoint)
    file_name = args.start_checkpoint + checkpoint_files[0]
    (model_params, start_from_this_iteration) = torch.load(
        file_name, map_location=torch.device("cpu")
    )

    if args.drop_duplicate_gaussians_coeff != 1.0:
        model_params = drop_duplicate_gaussians(
            model_params, args.drop_duplicate_gaussians_coeff
        )

    return model_params, start_from_this_iteration
