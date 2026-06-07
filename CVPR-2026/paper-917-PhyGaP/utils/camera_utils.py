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

from scene.cameras import Camera
import numpy as np
import os, cv2, torch
from utils.general_utils import PILtoTorch, ArrayImagetoTorch
from utils.graphics_utils import fov2focal
# from utils.stokes_utils import calc_aolp_dop NOTE will cause circular import
from utils.stokes_io_utils import load_aolp_dop
import skimage
from skimage.transform import rescale

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    # import pdb;pdb.set_trace()
    orig_w, orig_h = cam_info.image.shape[1], cam_info.image.shape[0]

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        scale = float(resolution_scale * args.resolution)
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600 and False:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    HWK = None  # #
    if cam_info.K is not None:
        K = cam_info.K.copy()
        K[:2] = K[:2] / scale
        HWK = (resolution[1], resolution[0], K)

    # print("Resolution", resolution, "Scale", scale, "HWK", HWK)
    # import pdb;pdb.set_trace()
    if cam_info.image.shape[2] > 3:
        resized_image_rgb = torch.cat([ArrayImagetoTorch(cam_info.image[..., i], resolution) 
                                    for i in range(3)], dim=0)
        loaded_mask = ArrayImagetoTorch(cam_info.image[...,3], resolution)
        gt_image = resized_image_rgb
    else:
        # print("Image Shape", cam_info.image.shape)
        resized_image_rgb = ArrayImagetoTorch(cam_info.image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb
    # #
    # print("[Check stokes]",cam_info.stokes)

    if cam_info.stokes is not None:
        gt_stokes = []
        # print("STOKES SHAPE", cam_info.stokes.shape)
        
        s0 = skimage.img_as_float32(cam_info.stokes[...,0].cpu().numpy())
        s1 = skimage.img_as_float32(cam_info.stokes[...,1].cpu().numpy())
        s2 = skimage.img_as_float32(cam_info.stokes[...,2].cpu().numpy())
        resized_s0 = rescale(s0, 1./scale, anti_aliasing=False,channel_axis=-1)
        resized_s1 = rescale(s1, 1./scale, anti_aliasing=False,channel_axis=-1)
        resized_s2 = rescale(s2, 1./scale, anti_aliasing=False,channel_axis=-1)
        # print("Resized Stokes Shapes", resized_s0.shape, resized_s1.shape, resized_s2.shape)
        gt_stokes = torch.stack([torch.Tensor(resized_s0), torch.Tensor(resized_s1), torch.Tensor(resized_s2)], dim=-1)
        aolp, dop = load_aolp_dop(gt_stokes)
        # print("GT STOKES SHAPE",gt_stokes.shape)
        # print(gt_image.shape, gt_stokes.shape)
        return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                    image=gt_image, gt_alpha_mask=loaded_mask,
                    image_name=cam_info.image_name, uid=id, 
                    data_device=args.data_device, HWK=HWK, gt_refl_mask=None,
                    stokes=gt_stokes.permute(2,0,1,3),aolp=aolp, dop=dop, is_linear=cam_info.is_linear,LP = cam_info.LP)
    else:
        gt_stokes = None
        aolp, dop = None, None
        # print(gt_image.shape, gt_stokes.shape)
        return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                    image=gt_image, gt_alpha_mask=loaded_mask,
                    image_name=cam_info.image_name, uid=id, 
                    data_device=args.data_device, HWK=HWK, gt_refl_mask=None,
                    stokes=None,aolp=aolp, dop=dop, is_linear=cam_info.is_linear,LP = cam_info.LP)

    

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []
    # import pdb; pdb.set_trace()
    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry