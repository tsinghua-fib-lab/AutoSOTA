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

import os
import sys
import torch
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import math
import json, cv2
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.stokes_io_utils import read_stokes, read_single_linear_stokes, load_rgb_exr
from scene.linear_polarizer import LinearPolarizer
from glob import glob
import imageio
import skimage

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    K: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    stokes:np.array = None
    aolp: np.array = None
    dop: np.array = None
    is_linear: int = 0 #NOTE: means linear polarized, not means whether or not in linear color space
    LP: LinearPolarizer = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        # print(extr)
        # Image(id=1, qvec=array(...), 
        #       tvec=array(...),
        #        camera_id=4, name='04.jpg', 
        #       xys=array(...), point3D_ids=array(...))
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec)) #qvec2rotmat(extr.qvec) get R_w2c, R = R_w2c.T = **R_c2w**
        T = np.array(extr.tvec) #w2c
        # Convert COLMAP world convention (Y-down) to Y-up convention used by
        # env-map / Mitsuba pipeline: apply F=diag(1,-1,-1) to world, which
        # negates rows 1 and 2 of R_c2w.  T_w2c is mathematically unchanged.
        R[1:3, :] *= -1
        # print(-R @ T) #C
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            K = np.array([
                [focal_length_x, 0, intr.params[1]],
                [0, focal_length_x, intr.params[2]],
                [0, 0, 1],
            ])
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            K = np.array([
                [focal_length_x, 0, intr.params[2]],
                [0, focal_length_y, intr.params[3]],
                [0, 0, 1],
            ])
        elif intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            K = np.array([
                [focal_length_x, 0, intr.params[1]],
                [0, focal_length_x, intr.params[2]],
                [0, 0, 1],
            ])
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path.replace('.JPG', '.jpg'))
        image = np.array(image)

        mask_path  = os.path.join(os.path.dirname(images_folder), "masks", os.path.basename(extr.name).replace('.png', '.jpg'))
        mask = Image.open(mask_path.replace('.JPG', '.jpg')) if os.path.exists(mask_path.replace('.JPG', '.jpg')) else None
        # import pdb;pdb.set_trace()
        if mask is not None:
            if mask.mode != 'L':
                mask = mask.convert('L')
            mask_array = np.array(mask)
            if mask_array.shape != image.shape[:2]:
                mask_array = cv2.resize(mask_array, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_array = mask_array[..., np.newaxis]
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1)
            image = np.concatenate([image, mask_array], axis=-1)
        # import pdb; pdb.set_trace()
        
        #if intr.model=="SIMPLE_RADIAL":
        #    image = cv2.undistort(np.array(image), K, np.array([intr.params[3], 0,0,0]))
        #    image = Image.fromarray(image.astype('uint8')).convert('RGB')

        real_im_scale = image.shape[1] / width
        # print("Real image scale", real_im_scale)
        K[:2] *=  real_im_scale

        # print("STOKES PATH",os.path.join(os.path.dirname(images_folder), "images_stokes", image_name + "_s0.hdr"))
        if os.path.exists(os.path.join(os.path.dirname(images_folder), "images_stokes", image_name + "_s0.hdr")):
            gt_stokes = read_stokes(images_folder, image_name, method="pandora", downscale=real_im_scale)
            # NOTE: the rgb images in pandora dataset are in srgb color space,
            # So the color is a bit different. Since in our work we use rgb to represent the intensity, 
            # we need to use linear color space. So here the image readin is aborted if the s0 image exists
            # import pdb;pdb.set_trace()
            
            if mask is not None:
                gt_stokes[...,0] = torch.from_numpy(gt_stokes[...,0].numpy() * (mask_array/255.0))
                gt_stokes[...,1] = torch.from_numpy(gt_stokes[...,1].numpy() * (mask_array/255.0))
                gt_stokes[...,2] = torch.from_numpy(gt_stokes[...,2].numpy() * (mask_array/255.0))
            if gt_stokes is not None:
                s0 = gt_stokes[..., 0].numpy()
                if s0.ndim == 3:
                    s0 = (s0 * 255).astype(np.int16)
                    image[..., :3] = s0
                else:
                    print(f"Unsupported Stokes image shape: {s0.shape}")
                    pass

        else:
            def srgb_to_linear(srgb):
                srgb = np.asarray(srgb)
                threshold = 0.04045
                below = srgb <= threshold
                above = ~below
                linear = np.zeros_like(srgb)
                linear[below] = srgb[below] / 12.92
                linear[above] = ((srgb[above] + 0.055) / 1.055) ** 2.4
                return linear
            gt_stokes = None
            image[...,:3] = srgb_to_linear(image[...,:3]/255.0)*255.0
        
        cam_info = CameraInfo(uid=uid, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,stokes=gt_stokes)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readSingleLPColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, LP=None):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        # import pdb;pdb.set_trace()
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec)) #qvec2rotmat(extr.qvec) get R_w2c, R = R_w2c.T = **R_c2w**
        T = np.array(extr.tvec) #w2c
        # Convert COLMAP world convention (Y-down) to Y-up convention used by
        # env-map / Mitsuba pipeline: apply F=diag(1,-1,-1) to world, which
        # negates rows 1 and 2 of R_c2w.  T_w2c is mathematically unchanged.
        R[1:3, :] *= -1
        # print(-R @ T) #C
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            K = np.array([
                [focal_length_x, 0, intr.params[1]],
                [0, focal_length_x, intr.params[2]],
                [0, 0, 1],
            ])
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            K = np.array([
                [focal_length_x, 0, intr.params[2]],
                [0, focal_length_y, intr.params[3]],
                [0, 0, 1],
            ])
        elif intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            K = np.array([
                [focal_length_x, 0, intr.params[1]],
                [0, focal_length_x, intr.params[2]],
                [0, 0, 1],
            ])
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path.replace('.JPG', '.jpg'))
        if LP is not None:
            image = np.array(image).astype(np.float32) * 0.5
        else:
            image = np.array(image).astype(np.float32)

        mask_path  = os.path.join(os.path.dirname(images_folder), "masks", os.path.basename(extr.name))
        mask = Image.open(mask_path.replace('.JPG', '.jpg')) if os.path.exists(mask_path.replace('.JPG', '.jpg')) else None
        if mask is not None:
            if mask.mode != 'L':
                mask = mask.convert('L')
            mask_array = np.array(mask)
            if mask_array.shape != image.shape[:2]:
                mask_array = cv2.resize(mask_array, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_array = mask_array[..., np.newaxis]
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1)
            image = np.concatenate([image, mask_array], axis=-1)
        
        #if intr.model=="SIMPLE_RADIAL":
        #    image = cv2.undistort(np.array(image), K, np.array([intr.params[3], 0,0,0]))
        #    image = Image.fromarray(image.astype('uint8')).convert('RGB')

        real_im_scale = image.shape[1] / width
        # print("Real image scale", real_im_scale)
        K[:2] *=  real_im_scale
        if LP is not None:
            can_info = CameraInfo(uid=uid, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                                    image_path=image_path, image_name=image_name, width=width, height=height,
                                    stokes=None, aolp=None, dop=None, is_linear = 1, LP=LP)
        else:
            can_info = CameraInfo(uid=uid, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                                    image_path=image_path, image_name=image_name, width=width, height=height,
                                    stokes=None, aolp=None, dop=None, is_linear = 0, LP=None)
        cam_infos.append(can_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        print('Load Ply color and normals failed, random init')
        colors = np.random.rand(*positions.shape) / 255.0
        normals = np.random.rand(*positions.shape)
        normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def createPcd(ply_path, cam_infos, num_pts = 500_000):
    num_pts = 500_000
    print(f"Generating random point cloud ({num_pts})...")
    # 获取所有相机的中心坐标
    cam_centers = []
    for cam in cam_infos:
        # W2C = [R | T], C2W = inv(W2C)
        # W2C = getWorld2View2(cam.R, cam.T)
        # C2W = np.linalg.inv(W2C)
        # cam_centers.append(C2W[:3, 3])
        R = cam.R  # Note that here the R is already transposed cam.R -> c2w
        T = cam.T
        
        cam_centers.append(-R @ T)  # C = -R.T @ T
    cam_centers = np.array(cam_centers)
    cam_dirs = []
    for cam in cam_infos:
        cam_dir = cam.R @ np.array([0, 0, 1])
        cam_dirs.append(cam_dir / np.linalg.norm(cam_dir))
    cam_dirs = np.array(cam_dirs)
    cam_centers = np.array(cam_centers)

    # print("Camera world positions and directions:")
    # for i in range(len(cam_centers)):
    #     print(f"Camera {i}: position={cam_centers[i]}, direction={cam_dirs[i]}")

    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(cam_centers, cam_dirs):
        d = d.reshape(3, 1)
        I = np.eye(3)
        A += I - d @ d.T
        b += (I - d @ d.T) @ o
    center = np.linalg.solve(A, b)

    # print(f"Converged point (sphere center): {center}")
    max_radius = np.max(np.linalg.norm(cam_centers - center, axis=1))

    radius = max_radius * 1.1
    xyz = np.random.normal(size=(num_pts, 3))
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    scales = np.random.uniform(0, radius, size=(num_pts, 1))
    xyz = xyz * scales + center
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd_tmp = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    pcd = pcd_tmp
    # import pdb;pdb.set_trace()
    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    return pcd

def readMitsubaCamInfos(cam_folder,img_folder = "images" , is_linear = 0, eval = True, llffhold = 8):
    path = cam_folder

    npz_path = os.path.join(cam_folder, "camera_params.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Camera parameters file not found at {npz_path}. Please ensure the file exists.")
    data = np.load(npz_path)
    K = data['intrinsics']
    fx, fy = K[0,0], K[1,1]
    all_extrinsics = data['extrinsics']
    width = int(data['image_width'])
    height = int(data['image_height'])
    FovX = focal2fov(fx, width)
    FovY = focal2fov(fy, height)

    cam_infos = []
    num_cameras = all_extrinsics.shape[0]
    # import pdb;pdb.set_trace()
    for i in range(num_cameras):
        extrinsic_matrix = all_extrinsics[i]
        
        R = extrinsic_matrix[:3, :3].T #c2w
        T = extrinsic_matrix[:3, 3] #w2c
        # print(-R @ T)
        image_path = os.path.join(path, f"images_full/{i:03d}.exr")
        image_name = f"{i:03d}"
        # import pdb;pdb.set_trace()
        image_folder_abs = os.path.join(path,"images_full")
        if not os.path.exists(image_path):
            print(f"Image {image_path} does not exist, skipping camera {i}.")
            continue
        image = load_rgb_exr(image_path, downscale=1)
        # import pdb;pdb.set_trace()
        real_im_scale = image.shape[1] / width
        K[:2] *=  real_im_scale
        
        gt_stokes = read_stokes(image_folder_abs, image_name, method="mitsuba", downscale=real_im_scale)
        if gt_stokes is not None:
            s0 = gt_stokes[..., 0].numpy()
            if s0.ndim == 3:
                s0 = (s0 * 255).astype(np.int16)
                image = s0
            else:
                print(f"Unsupported Stokes image shape: {s0.shape}")
                pass
        # import pdb; pdb.set_trace()
        mask_path  = os.path.join(os.path.dirname(image_folder_abs), "masks", f"{i:03d}.jpg")
        mask = Image.open(mask_path.replace('.JPG', '.jpg')) if os.path.exists(mask_path.replace('.JPG', '.jpg')) else None
        if mask is not None:
            if mask.mode != 'L':
                mask = mask.convert('L')
            mask_array = np.array(mask)
            if mask_array.shape != image.shape[:2]:
                mask_array = cv2.resize(mask_array, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_array = mask_array[..., np.newaxis]
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1)
            image = np.concatenate([image, mask_array], axis=-1)
        # import pdb; pdb.set_trace()
        cam_info = CameraInfo(uid=i, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, stokes=gt_stokes,is_linear = is_linear)
        cam_infos.append(cam_info)
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    # import pdb; pdb.set_trace()
    return train_cam_infos,test_cam_infos

def readSMVP3DCamInfos(cam_folder,img_folder = "image" , is_linear = 0, eval = True, llffhold = 8):#(path, white_background, name):
    # import pdb;pdb.set_trace()

    def load_mask_SMVP3D(path):
        alpha = imageio.imread(path, pilmode='F')
        alpha = skimage.img_as_float32(alpha) / 255
        return alpha

    def glob_imgs(path):
        imgs = []
        for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
            imgs.extend(glob(os.path.join(path, ext)))
        return imgs

    image_dir = os.path.join(cam_folder, img_folder)
    image_paths = sorted(glob_imgs(image_dir))
    cam_file = os.path.join(cam_folder,'cameras.npz')

    n_images = len(image_paths)

    camera_dict = np.load(cam_file)
    try:
        scale_mats = [camera_dict['scale_mat_%02d' % idx].astype(np.float32) for idx in range(1,n_images+1)]
        world_mats = [camera_dict['world_mat_%02d' % idx].astype(np.float32) for idx in range(1,n_images+1)]
    except:
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    pose_all = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat
        pose_all.append(P)


    azimuth_path = sorted(os.listdir(os.path.join(cam_folder, "input_azimuth_maps")))

    cam_infos = []
    for i in range(n_images):
        P = pose_all[i]
        K, R, t = cv2.decomposeProjectionMatrix(P[:3, :4])[:3]
        t = t[:3, :] / t[3:, :]
        K = K / K[2, 2]

        T = -R @ t  
        T = T[:, 0]
        R = R.T
        R[:2,:3] *= -1 # the difference between opengl and opencv camera


        image_path = image_paths[i]
        image_name = image_path.split('.')[0].split('/')[-1]
        uid = int(image_name.split('/')[-1])

        
        gt_stokes = read_stokes(cam_folder,f"{i:04d}", method="SMVP3D", downscale=1)

        azimuth_file= os.path.join(os.path.join(cam_folder, "input_azimuth_maps"), azimuth_path[i])
        azimuth = imageio.imread(azimuth_file)
        azimuth = skimage.img_as_float32(azimuth)
        if azimuth.ndim == 2:
            # (H, W) -> (H, W, 1)
            azimuth = azimuth[...,None]
        mask = azimuth[...,[-1]].astype(np.float32)
        mask = (mask*255).astype(np.uint8)
        
        # import pdb; pdb.set_trace()
        image = (gt_stokes[..., 0].numpy() * 255).astype(np.uint8)
        
        FovY = focal2fov(K[1, 1], image.shape[0])
        FovX = focal2fov(K[0, 0], image.shape[1])
        
        if mask is not None:
            mask_array = np.array(mask)
            if mask_array.shape != image.shape[:2]:
                mask_array = cv2.resize(mask_array, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1)
            if image.ndim == 4:
                image = image[...,:3]
            # import pdb; pdb.set_trace()
            image = np.concatenate([image, mask_array[...,None]], axis=-1)
            # import pdb;pdb.set_trace()
        # image = Image.fromarray(np.array(image, dtype=np.byte), "RGB")

        cam_infos.append(CameraInfo(uid=uid, R=R, T=T, K=K,  FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, image_name=image_name, width=image.shape[0], height=image.shape[1], 
                         stokes=gt_stokes))
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    return train_cam_infos,test_cam_infos

def readRMVP3DCamInfos(cam_folder,img_folder = "image" , is_linear = 0, eval = True, llffhold = 8):#(path, white_background, name):
    def load_mask_SMVP3D(path):
        alpha = imageio.imread(path, pilmode='F')
        alpha = skimage.img_as_float32(alpha) / 255
        return alpha
    def glob_imgs(path):
        imgs = []
        for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
            imgs.extend(glob(os.path.join(path, ext)))
        return imgs
    image_dir = os.path.join(cam_folder, img_folder)
    image_paths = sorted(glob_imgs(image_dir))
    cam_file = os.path.join(cam_folder,'cameras.npz')

    n_images = len(image_paths)

    camera_dict = np.load(cam_file)
    try:
        scale_mats = [camera_dict['scale_mat_%02d' % idx].astype(np.float32) for idx in range(1,n_images+1)]
        world_mats = [camera_dict['world_mat_%02d' % idx].astype(np.float32) for idx in range(1,n_images+1)]
    except:
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    pose_all = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat
        pose_all.append(P)



    azimuth_path = sorted(os.listdir(os.path.join(cam_folder, "input_azimuth_maps")))

    cam_infos = []
    for i in range(n_images):
        P = pose_all[i]
        K, R, t = cv2.decomposeProjectionMatrix(P[:3, :4])[:3]
        t = t[:3, :] / t[3:, :]
        K = K / K[2, 2]

        T = -R @ t  
        T = T[:, 0]
        R = R.T
        # R[:2,:3] *= -1 # the difference between opengl and opencv camera

        image_path = image_paths[i]
        image_name = image_path.split('.')[0].split('/')[-1]
        uid = int(image_name.split('/')[-1])
        # image = (rgb_images[i].transpose([1, 2, 0]) * 0.5 + 0.5) * 255

        gt_stokes = read_stokes(cam_folder,f"{i:04d}", method="RMVP3D", downscale=1) 
        image = (gt_stokes[..., 0].numpy() * 255).astype(np.uint8)
        mask_path  = os.path.join(cam_folder, "mask", f'{i:04d}.png') 
        mask = Image.open(mask_path) if os.path.exists(mask_path) else None
        # import pdb;pdb.set_trace()
        mask = (np.array(mask)).astype(np.uint8) if mask is not None else None
        FovY = focal2fov(K[1, 1], image.shape[0])
        FovX = focal2fov(K[0, 0], image.shape[1])
        
        if mask is not None:
            mask_array = np.array(mask)
            if mask_array.shape != image.shape[:2]:
                mask_array = cv2.resize(mask_array, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1)
            if image.ndim == 4:
                image = image[...,:3]
            # import pdb; pdb.set_trace()
            mask_array = mask_array[...,None]
            image = (image*(mask_array/255)).astype(np.uint8)
            image = np.concatenate([image, mask_array], axis=-1)
            # import pdb;pdb.set_trace()
        # image = Image.fromarray(np.array(image, dtype=np.byte), "RGB")

        cam_infos.append(CameraInfo(uid=uid, R=R, T=T, K=K,  FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, image_name=image_name, width=image.shape[0], height=image.shape[1], 
                         stokes=gt_stokes))
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    return train_cam_infos,test_cam_infos


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    # print("[Check stokes dataset_readers.py-readColmapSceneInfo]", cam_infos[0].stokes)
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    spc_ply_path = os.path.join(path, "sparse/0/points_spc.ply")
    if os.path.exists(spc_ply_path):
        ply_path = spc_ply_path
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    # Apply the same Y/Z flip as the cameras so the initial point cloud is
    # aligned with the corrected world coordinate system.
    if pcd is not None:
        xyz_flipped = pcd.points.copy()
        xyz_flipped[:, 1:3] *= -1
        pcd = BasicPointCloud(points=xyz_flipped, colors=pcd.colors, normals=pcd.normals)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readColmapSceneInfoDoubleLP(path, images, eval, llffhold=8, LP=None):
    cam_folders = ['camera_0', 'camera_1']
    all_train_cam_infos = []
    all_test_cam_infos = []
    for idx,cam_folder in enumerate(cam_folders):
        cam_path = os.path.join(path, cam_folder)
        try:
            cameras_extrinsic_file = os.path.join(cam_path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(cam_path, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(cam_path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(cam_path, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        reading_dir = "images" if images == None else images
        LP_cam = LP[idx] if LP is not None else None
        cam_infos_unsorted = readSingleLPColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(cam_path, reading_dir),LP=LP_cam)
        cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
        if eval:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []

        all_train_cam_infos.extend(train_cam_infos)
        all_test_cam_infos.extend(test_cam_infos)
    reference_cam_path = cam_folders[0]
    nerf_normalization = getNerfppNorm(all_train_cam_infos)
    ply_path = os.path.join(path, reference_cam_path, "sparse/0/points3D.ply")
    spc_ply_path = os.path.join(path, reference_cam_path, "sparse/0/points_spc.ply")
    if os.path.exists(spc_ply_path):
        ply_path = spc_ply_path
    bin_path = os.path.join(path, reference_cam_path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, reference_cam_path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    # Apply the same Y/Z flip as the cameras so the initial point cloud is
    # aligned with the corrected world coordinate system.
    if pcd is not None:
        xyz_flipped = pcd.points.copy()
        xyz_flipped[:, 1:3] *= -1
        pcd = BasicPointCloud(points=xyz_flipped, colors=pcd.colors, normals=pcd.normals)

    scene_info = SceneInfo(point_cloud=pcd,
                            train_cameras=all_train_cam_infos,
                             test_cameras=all_test_cam_infos,
                             nerf_normalization=nerf_normalization,
                             ply_path=ply_path)
    return scene_info

def readColmapSceneInfoSingleLP(path, images, eval, llffhold=8, LP=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readSingleLPColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir),LP=LP[0])
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    spc_ply_path = os.path.join(path, "sparse/0/points_spc.ply")
    if os.path.exists(spc_ply_path):
        ply_path = spc_ply_path
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    # Apply the same Y/Z flip as the cameras so the initial point cloud is
    # aligned with the corrected world coordinate system.
    if pcd is not None:
        xyz_flipped = pcd.points.copy()
        xyz_flipped[:, 1:3] *= -1
        pcd = BasicPointCloud(points=xyz_flipped, colors=pcd.colors, normals=pcd.normals)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            ### NOTE !!!!!
            # Here R has been transposed, R = w2c.T
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            image = np.array(image)
            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            image = np.array(image)
            fo = fov2focal(fovx, image.size[0])

            W,H = image.size[0], image.size[1]
            K = np.array([
                [fo, 0, W/2],
                [0, fo, H/2],
                [0, 0, 1],
            ])

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            # For blender datasets, we consider its camera center offset is zero (ideal camera)
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readMitsubaSceneInfo(path, img_folder, eval,  llffhold=8):
    camfolder = path
    train_cam_infos, test_cam_infos = readMitsubaCamInfos(camfolder,img_folder,0, eval, llffhold)
    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "points3D.ply")
    pcd = None
    # import pdb;pdb.set_trace()
    if not os.path.exists(ply_path):
        pcd = createPcd(ply_path, train_cam_infos, num_pts = 500_000)
    else:
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = createPcd(ply_path, train_cam_infos, num_pts = 500_000)
    # import pdb;pdb.set_trace()  
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                            test_cameras=test_cam_infos,
                            nerf_normalization=nerf_normalization,
                            ply_path=ply_path)
    return scene_info

def readMitsubaSceneInfoDoubleLP(path, img_folder, eval,  llffhold=8, LP=None):
    cam0_folder = os.path.join(path, "camera_0")
    camLP_folder = os.path.join(path, "camera_1")
    cam0_infos_train, cam0_infos_test = readMitsubaCamInfos(cam0_folder,img_folder,0,eval,llffhold)
    camLP_infos_train, camLP_infos_test = readMitsubaCamInfos(camLP_folder,img_folder,0,eval,llffhold)
    # import pdb;pdb.set_trace()
    if LP is not None:
        # import pdb; pdb.set_trace()
        camLP_infos_train = applyLP2MitsubaCaminfos(camLP_infos_train, LP=LP[0])
        camLP_infos_test = applyLP2MitsubaCaminfos(camLP_infos_test, LP=LP[0])
        cam0_infos_train = applyLP2MitsubaCaminfos2(cam0_infos_train, LP=LP[1])
        cam0_infos_test = applyLP2MitsubaCaminfos2(cam0_infos_test, LP=LP[1])
    
    
    train_cam_infos = cam0_infos_train + camLP_infos_train
    test_cam_infos = cam0_infos_test + camLP_infos_test
    # import pdb;pdb.set_trace()
    
    nerf_normalization = getNerfppNorm(train_cam_infos) #NOTE: Here, the cam infos contains 2 different cameras, so the nerf normalization is not accurate, but it is ok for now, representing an average effect of all cameras.
                                                        #TODO: If anything went wrong, see here. But theoretically nerf normalization only used in the initial process of Snene.
    ply_path = os.path.join(path, "points3D.ply")
    pcd = None
    # import pdb;pdb.set_trace()
    if not os.path.exists(ply_path):
        pcd = createPcd(ply_path, train_cam_infos, num_pts = 500_000)
    else:
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = createPcd(ply_path, train_cam_infos, num_pts = 500_000)
    # import pdb;pdb.set_trace()  
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                            test_cameras=test_cam_infos,
                            nerf_normalization=nerf_normalization,
                            ply_path=ply_path)
    return scene_info



def applyLP2MitsubaCaminfos(cam_infos,LP=None):
    new_cam_infos = [] 
    for cam in cam_infos:
        images = []
        mask = cam.image[...,3:] if cam.image.shape[-1]==4 else None
        image = ((cam.stokes[...,0]+cam.stokes[...,2])*0.5).clamp(0,1).cpu().numpy()#45
        image = (image * 255).astype(np.int16)
        if mask is not None:
            image = np.concatenate([image, mask], axis=-1)
        # import pdb;pdb.set_trace()
        new_cam_infos.append(CameraInfo(
            uid=cam.uid, R=cam.R, T=cam.T, K=cam.K, FovY=cam.FovY, FovX=cam.FovX, 
            image=image, image_path=cam.image_path, image_name=cam.image_name, 
            width=cam.width, height=cam.height, stokes=cam.stokes, aolp=None, dop=None, is_linear = 1, LP=LP))
    return new_cam_infos
def applyLP2MitsubaCaminfos2(cam_infos,LP=None):
    new_cam_infos = [] 
    for cam in cam_infos:
        mask = cam.image[...,3:] if cam.image.shape[-1]==4 else None
        image = ((cam.stokes[...,0]-cam.stokes[...,2])*0.5).clamp(0,1).cpu().numpy()#135
        image = (image * 255).astype(np.int16)
        if mask is not None:
            image = np.concatenate([image, mask], axis=-1)
        new_cam_infos.append(CameraInfo(
            uid=cam.uid, R=cam.R, T=cam.T, K=cam.K, FovY=cam.FovY, FovX=cam.FovX, 
            image=image, image_path=cam.image_path, image_name=cam.image_name, 
            width=cam.width, height=cam.height, stokes=cam.stokes, aolp=None, dop=None, is_linear = 1, LP=LP))
    return new_cam_infos

def readMitsubaSceneInfoDoubleViews(path, img_folder, eval,  llffhold=8, LP=None):
    return readMitsubaSceneInfoDoubleLP(path, img_folder, eval,  llffhold=8, LP=None)

def readColmapSceneInfoDoubleViews(path, images, eval, llffhold=8,LP=None):
    return readColmapSceneInfoDoubleLP(path, images, eval, llffhold=8,LP=None)  

def readSMVP3DSceneInfo(path, img_folder, eval,  llffhold=8,LP=None):
    camfolder = path
    train_cam_infos, test_cam_infos = readSMVP3DCamInfos(camfolder,img_folder,0, eval, llffhold)
    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "points3D.ply")
    pcd = None
    # import pdb;pdb.set_trace()
    if not os.path.exists(ply_path):
        pcd = createPcd(ply_path, train_cam_infos, num_pts = 500_000)
    else:
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = createPcd(ply_path, train_cam_infos, num_pts = 500_000)
    # import pdb;pdb.set_trace()
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                            test_cameras=test_cam_infos,
                            nerf_normalization=nerf_normalization,
                            ply_path=ply_path)
    return scene_info

def readRMVP3DSceneInfo(path, img_folder, eval,  llffhold=8,LP=None):
    camfolder = path
    # import pdb;pdb.set_trace()
    train_cam_infos, test_cam_infos = readRMVP3DCamInfos(camfolder,img_folder,0, eval, llffhold)
    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "points3D.ply")
    pcd = None
    # import pdb;pdb.set_trace()
    if not os.path.exists(ply_path):
        pcd = createPcd(ply_path, train_cam_infos, num_pts = 500_000)
    else:
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = createPcd(ply_path, train_cam_infos, num_pts = 500_000)
    # import pdb;pdb.set_trace()
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                            test_cameras=test_cam_infos,
                            nerf_normalization=nerf_normalization,
                            ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "ColmapSingleLP": readColmapSceneInfoSingleLP,  
    "ColmapDoubleLP": readColmapSceneInfoDoubleLP, 
    "ColmapDoubleViews": readColmapSceneInfoDoubleViews,  
    "Mitsuba": readMitsubaSceneInfo,
    "MitsubaDoubleViews": readMitsubaSceneInfoDoubleViews,
    "MitsubaSingleLP": None,  
    "MitsubaDoubleLP":readMitsubaSceneInfoDoubleLP,  
    "RMVP3D": readRMVP3DSceneInfo, 
    "SMVP3D": readSMVP3DSceneInfo, 
}