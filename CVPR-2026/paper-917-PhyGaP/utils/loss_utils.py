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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from kornia.filters import spatial_gradient
from .image_utils import psnr
from utils.image_utils import erode
from utils.refl_utils import sample_camera_rays
import numpy as np
import os
from torchvision.utils import save_image
from scene.linear_polarizer import LinearPolarizer
from lpipsPyTorch import lpips

def linear_polar_simple(stokes,_phi):
    # used for debugging the linear polarizer, currently not in use
    phi = _phi.to(stokes.device)
    cos2p = torch.cos(2*phi)
    sin2p = torch.sin(2*phi)
    # import pdb;pdb.set_trace()  
    return 0.5*(stokes[...,0] + cos2p * stokes[...,1] + sin2p * stokes[...,2])

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def first_order_edge_aware_loss(data, img):
    return (spatial_gradient(data[None], order=1)[0].abs() * torch.exp(-spatial_gradient(img[None], order=1)[0].abs())).sum(1).mean()



def calculate_loss(viewpoint_camera, pc, render_pkg, opt, iteration):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    
    if iteration <= opt.init_until_iter or iteration <= opt.volume_render_until_iter:
        gt_image = viewpoint_camera.original_image.cuda().clamp(0.,1.)
        rendered_image = render_pkg["render"].clamp(0.,1.)
    else:
        gt_image = viewpoint_camera.original_image.cuda()
        if viewpoint_camera.is_linear == 1:
            pred_LPstokes = render_pkg["linear_stokes"]
            rendered_image = pred_LPstokes[..., 0]  # Assuming pred_stokes is in the format [s0, s1, s2]
            pred_stokes = pred_LPstokes
        elif viewpoint_camera.is_linear == 0:
            # import pdb;pdb.set_trace()
            pred_rgb = render_pkg["render"]
            rendered_image = pred_rgb
            pred_stokes = render_pkg["stokes_combined"]
    # rendered_opacity = render_pkg["rend_alpha"]
    rendered_depth = render_pkg["surf_depth"]
    rendered_normal = render_pkg["rend_normal"] #if iteration <opt.normal_follow_rend else render_pkg["rend_normal"].detach()
    visibility_filter = render_pkg["visibility_filter"]
    rend_dist = render_pkg["rend_dist"]
    surf_normal = render_pkg["surf_normal"]
    rendered_alpha = render_pkg["rend_alpha"]

    # if iteration <= opt.init_until_iter or iteration <= opt.volume_render_until_iter:
    
    gt_mask = viewpoint_camera.gt_alpha_mask.cuda() if viewpoint_camera.gt_alpha_mask is not None else None
    # gt_stokes = viewpoint_camera.original_stokes.cuda()
    # gt_image = gt_stokes[...,0]
    # import pdb;pdb.set_trace()
    Ll1 = l1_loss(rendered_image, gt_image)
    ssim_val = ssim(rendered_image, gt_image)
    loss0 = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)
    loss = torch.zeros_like(loss0)
    tb_dict["loss_l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    tb_dict["loss0"] = loss0.item()
    loss += loss0
    if iteration % 1000 == 0:
        print(f"Iteration {iteration}, Loss:{loss.item()}, L1: {Ll1.item()}, SSIM: {ssim_val.item()}")
    # else:
    #     gt_stokes = viewpoint_camera.original_stokes.cuda()
    #     gt_45 = (gt_stokes[...,0] + gt_stokes[...,2]) * 0.5
    #     gt_135 = (gt_stokes[...,0] - gt_stokes[...,2]) * 0.5
    #     stokes_45 = linear_polar_simple(render_pkg["stokes_combined"], torch.deg2rad(torch.tensor(45.0)))
    #     stokes_135 = linear_polar_simple(render_pkg["stokes_combined"], torch.deg2rad(torch.tensor(135.0)))
    #     # Ll1 = (l1_loss(stokes_45,gt_45) + l1_loss(stokes_135,gt_135))*2
    #     pred_stokes = torch.stack([render_pkg["stokes_combined"][...,0],
    #                                     torch.zeros_like(render_pkg["stokes_combined"][...,0]),
    #                                   stokes_45 - stokes_135], dim=-1)
    #     Ll1 = (l1_loss(gt_45,stokes_45) + l1_loss(gt_135, stokes_135))*0.5
    #     ssim_val = (ssim(gt_135,stokes_135)+ssim(gt_45, stokes_45))*0.5# ssim(gt_45+gt_135, stokes_45+stokes_135)
    #     loss0 = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)+10*l1_loss(gt_45, stokes_45) + 10*l1_loss(gt_135, stokes_135)
    #     loss = torch.zeros_like(loss0)
    #     tb_dict["loss_l1"] = Ll1.item() 
    #     tb_dict["psnr"] = psnr(stokes_45,gt_45).mean().item()
    #     tb_dict["ssim"] = ssim_val.item()
    #     tb_dict["loss0"] = loss0.item()
    #     loss += loss0
    #     # if iteration ==5002:
    #     #     # import pdb;pdb.set_trace()
    #     if iteration % 1000 == 3:
    #         print(f"Iteration {iteration}, MODIFIED LOSS, Loss:{loss.item()}, L1: {Ll1.item()}, SSIM: {ssim_val.item()}")
    #         print(f"SSIM_S0 {ssim(gt_45+gt_135, stokes_45+stokes_135)}")
    #         print(f"SSIM_45 {ssim(gt_45, stokes_45)}")
    #         print(f"SSIM_135 {ssim(gt_135, stokes_135)}")

    if opt.lambda_normal_render_depth > 0 and iteration > opt.normal_loss_start:
        loss_normal_render_depth = (1 - (rendered_normal * surf_normal).sum(dim=0))[None]
        loss_normal_render_depth = loss_normal_render_depth.mean()
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth
    else:
        tb_dict["loss_normal_render_depth"] = torch.zeros_like(loss)
    # import pdb;pdb.set_trace()
    if opt.lambda_mask > 0 and gt_mask is not None:
        # import pdb;pdb.set_trace()

        # gt_mask: (H, W) or (C, H, W), ensure it's (1, 1, H, W)
        if gt_mask.dim() == 2:
            mask = gt_mask.unsqueeze(0).unsqueeze(0)
        elif gt_mask.dim() == 3:
            mask = gt_mask.unsqueeze(0)
        else:
            mask = gt_mask

        # Create a 3x3 kernel for erosion
        kernel = torch.ones((1, 1, 3, 3), device=mask.device, dtype=mask.dtype)
        # Use min pooling to perform erosion
        # gt_mask_erode = -F.max_pool2d(-mask.float(), kernel_size=3, stride=1, padding=1)
        # gt_mask_erode = gt_mask_erode[0, 0]
        mask_loss = l2_loss(rendered_alpha, gt_mask)
        tb_dict["loss_mask"] = mask_loss.item()
        loss += opt.lambda_mask * mask_loss

    if opt.lambda_dist > 0 and iteration > opt.dist_loss_start:
        dist_loss = opt.lambda_dist * rend_dist.mean()
        tb_dict["loss_dist"] = dist_loss
        loss += dist_loss
    else:
        tb_dict["loss_dist"] = torch.zeros_like(loss)

    if opt.lambda_normal_smooth > 0 and iteration > opt.normal_smooth_from_iter and iteration < opt.normal_smooth_until_iter:
        loss_normal_smooth = first_order_edge_aware_loss(rendered_normal, gt_image)
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        lambda_normal_smooth = opt.lambda_normal_smooth
        loss = loss + lambda_normal_smooth * loss_normal_smooth
    else:
        tb_dict["loss_normal_smooth"] = torch.zeros_like(loss)
    
    if opt.lambda_depth_smooth > 0 and iteration > 3000:
        loss_depth_smooth = first_order_edge_aware_loss(rendered_depth, gt_image)
        tb_dict["loss_depth_smooth"] = loss_depth_smooth.item()
        lambda_depth_smooth = opt.lambda_depth_smooth
        loss = loss + lambda_depth_smooth * loss_depth_smooth
    else:
        tb_dict["loss_depth_smooth"] = torch.zeros_like(loss)
    
    
    if opt.lambda_stokes > 0 and iteration > opt.init_until_iter \
            and iteration > opt.volume_render_until_iter and not opt.use_LP: # and iteration < opt.volume_render_until_iter:
        gt_stokes = viewpoint_camera.original_stokes.cuda()
        loss_s1 = l1_loss(pred_stokes[...,1], gt_stokes[...,1])
        loss_s2 = l1_loss(pred_stokes[...,2], gt_stokes[...,2])
        stokes_loss = loss_s1 + loss_s2
        tb_dict["loss_stokes"] = stokes_loss.item()
        if iteration % 1000 == 0:
            print(f"Iteration {iteration}, Loss:{loss.item()}, Stokes Loss: {stokes_loss.item()}, S1 Loss: {loss_s1.item()}, S2 Loss: {loss_s2.item()}")
        # Adaptive weighting: scale stokes loss by per-pixel DoLP
        # DoLP = sqrt(S1^2 + S2^2) / S0, clamped to [0, 1]
        s0 = gt_stokes[..., 0]
        s1_sq = gt_stokes[..., 1] ** 2
        s2_sq = gt_stokes[..., 2] ** 2
        dolp = torch.sqrt(s1_sq + s2_sq) / (s0 + 1e-6)
        dolp = torch.clamp(dolp, 0.0, 1.0)
        # Scale: strong polarization gets up to 1.5x, weak gets 0.5x
        adaptive_weight = 0.5 + dolp.mean()
        loss += opt.lambda_stokes * stokes_loss * adaptive_weight
    else:
        tb_dict["loss_stokes"] = torch.zeros_like(loss)

    if opt.use_LP and iteration > opt.init_until_iter \
            and iteration > opt.volume_render_until_iter:
            stokes_loss = l1_loss(rendered_image, gt_image)
            loss += opt.lambda_stokes * stokes_loss
            tb_dict["loss_stokes"] = stokes_loss.item()
    else:
        tb_dict["loss_stokes"] = torch.zeros_like(loss)
    # print("stokes loss", tb_dict["loss_stokes"], "aolp dop loss", tb_dict["loss_aolp_dop"],"total loss", loss.item())
    # import pdb;pdb.set_trace()  
    tb_dict["loss"] = loss.item()
    
    return loss, tb_dict
