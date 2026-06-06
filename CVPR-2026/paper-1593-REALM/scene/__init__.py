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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera, MiniCam, LiteCamera
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import pdb
from tqdm import tqdm  # 别忘了在文件顶部导入 tqdm
from copy import deepcopy
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import torch

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.object_path, n_views=args.n_views, random_init=args.random_init, train_split=args.train_split)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            if isinstance(self.loaded_iter,str):
                print("edit load path", self.loaded_iter)
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud"+self.loaded_iter,
                                                            "point_cloud.ply"))
            else:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]


    def getRenderCameras(self, scale=1.0, interp_per_pair=5):
        """
        对 train cameras 做插值，生成更多 LiteCamera 实例用于渲染，避免爆显存。
        :param scale: 相机分辨率 scale
        :param interp_per_pair: 每对相机之间插值数（默认9倍插值）
        :return: List[LiteCamera]
        """
        base_cameras = self.train_cameras[scale]
        render_cameras = []

        uid_counter = 0
        print(f"[Info] Interpolating path using LiteCamera with interp_per_pair={interp_per_pair}...")

        for i in tqdm(range(len(base_cameras) - 1), desc="Interpolating Camera Pairs"):
            cam0 = base_cameras[i]
            cam1 = base_cameras[i + 1]

            R0, T0 = cam0.R, cam0.T
            R1, T1 = cam1.R, cam1.T

            # 添加起点相机（lite）
            render_cameras.append(LiteCamera(
                colmap_id=-1,
                R=R0,
                T=T0,
                FoVx=cam0.FoVx,
                FoVy=cam0.FoVy,
                image_name=f"orig_{uid_counter:05d}",
                uid=uid_counter,
                trans=cam0.trans,
                scale=cam0.scale,
                device=cam0.data_device,
                image_width=cam0.image_width,
                image_height=cam0.image_height
            ))
            uid_counter += 1

            # 中间插值相机
            for t in np.linspace(0, 1, interp_per_pair + 2)[1:-1]:  # 去掉端点
                R_interp = (1 - t) * R0 + t * R1
                U, _, Vt = np.linalg.svd(R_interp)
                R_interp = U @ Vt
                T_interp = (1 - t) * T0 + t * T1

                render_cameras.append(LiteCamera(
                    colmap_id=-1,
                    R=R_interp,
                    T=T_interp,
                    FoVx=cam0.FoVx,
                    FoVy=cam0.FoVy,
                    image_name=f"interp_{uid_counter:05d}",
                    uid=uid_counter,
                    trans=cam0.trans,
                    scale=cam0.scale,
                    device=cam0.data_device,
                    image_width=cam0.image_width,
                    image_height=cam0.image_height
                ))
                print(render_cameras[-1].image_width, render_cameras[-1].image_height)

                uid_counter += 1

        # 添加最后一个相机
        cam_last = base_cameras[-1]
        render_cameras.append(LiteCamera(
            colmap_id=-1,
            R=cam_last.R,
            T=cam_last.T,
            FoVx=cam_last.FoVx,
            FoVy=cam_last.FoVy,
            image_name=f"orig_{uid_counter:05d}",
            uid=uid_counter,
            trans=cam_last.trans,
            scale=cam_last.scale,
            device=cam_last.data_device,
            image_width=cam0.image_width,
            image_height=cam0.image_height
        ))
        print(render_cameras[-1].image_width, render_cameras[-1].image_height)
        return render_cameras

    def create_cameras_from_path(self, json_path, device="cuda", scale=1.0, lookat_target=None):
        pdb.set_trace()
        import json
        import numpy as np
        from scene.cameras import LiteCamera

        def blender_to_colmap(c2w):
            # Blender -> COLMAP 坐标系转换矩阵
            blender_to_colmap_mat = np.array([
                [1,  0,  0, 0],
                [0, -1,  0, 0],
                [0,  0, -1, 0],
                [0,  0,  0, 1],
            ])
            return c2w @ blender_to_colmap_mat

        def look_at(eye, center, up=np.array([0, 1, 0])):
            # 标准 LookAt 实现：从 eye 看向 center
            forward = center - eye
            forward /= np.linalg.norm(forward)

            right = np.cross(up, forward)
            right /= np.linalg.norm(right)

            up_corrected = np.cross(forward, right)

            R = np.stack([right, up_corrected, forward], axis=1)
            T = eye
            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = T
            return c2w

        # Step 1: 加载 JSON 文件
        with open(json_path, 'r') as f:
            data = json.load(f)

        camera_path = data["camera_path"]
        render_w = data.get("render_width", 800)
        render_h = data.get("render_height", 800)

        # Step 2: 如果未指定目标点，则使用训练相机的平均中心作为注视点
        if lookat_target is None:
            train_cams = self.getTestCameras(scale)
            lookat_target = np.mean([cam.T for cam in train_cams], axis=0)

        # Step 3: 构建每一帧相机
        cams = []
        for idx, cam_dict in enumerate(camera_path):
            mat_flat = cam_dict["camera_to_world"]
            assert len(mat_flat) == 16
            c2w = np.array(mat_flat, dtype=np.float64).reshape(4, 4)

            # 转换到 COLMAP 坐标系
            c2w = blender_to_colmap(c2w)

            # 获取相机位置
            cam_pos = c2w[:3, 3]

            # 计算新的相机方向（看向 lookat_target）
            new_c2w = look_at(cam_pos, lookat_target)

            # 替换旋转和平移部分
            c2w[:3, :3] = new_c2w[:3, :3]
            c2w[:3, 3] = new_c2w[:3, 3]

            # 拆出旋转和平移
            R = c2w[:3, :3]
            T = c2w[:3, 3]
            # 添加一个绕 Y 轴旋转 180 度的旋转矩阵
            y_rot_180 = np.array([
                [-1,  0,  0],
                [ 0,  1,  0],
                [ 0,  0, 1],
            ])

            # 在设置 R 后应用这个旋转（注意顺序）
            R = c2w[:3, :3] @ y_rot_180  # 或者使用 R = y_rot_180 @ R 看你需要的是局部旋转还是全局旋转
            # FOV 计算
            fov_deg = cam_dict.get("fov", 50)
            aspect = cam_dict.get("aspect", render_w / render_h)
            fovy = np.deg2rad(fov_deg)
            fovx = 2 * np.arctan(np.tan(fovy / 2) * aspect)

            # 创建 LiteCamera 对象
            cam = LiteCamera(
                colmap_id=-1,
                R=R,
                T=T,
                FoVx=fovx,
                FoVy=fovy,
                image_name=f"path_{idx:04d}",
                uid=idx,
                trans=np.array([0.0, 0.0, 0.0]),
                scale=scale,
                device=device,
                image_width=render_w,
                image_height=render_h
            )
            cams.append(cam)

        return cams