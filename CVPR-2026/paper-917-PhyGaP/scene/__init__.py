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
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], LP = None):
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
        if args.double_view == False:
            if LP is not None:
                print("Single view mode enabled, loading scene with LP cameras")
                scene_info = sceneLoadTypeCallbacks["ColmapSingleLP"](args.source_path, args.images, args.eval, LP=LP)
            else:
                if os.path.exists(os.path.join(args.source_path, "sparse")):
                    print("Found sparse folder, assuming Colmap dataset!")
                    scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
                elif os.path.exists(os.path.join(args.source_path, "input_azimuth_maps")):
                    print("Found input_azimuth_maps folder, assuming RMVP3D or SMVP3D dataset!")
                    if os.path.exists(os.path.join(args.source_path, "images_stokes")):
                        print("Found images_stokes folder, assuming RMVP3D dataset!")
                        scene_info = sceneLoadTypeCallbacks["RMVP3D"](args.source_path, "image", args.eval)
                    elif os.path.exists(os.path.join(args.source_path, "s0")):
                        print("Assuming SMVP3D dataset!")
                        scene_info = sceneLoadTypeCallbacks["SMVP3D"](args.source_path, "image", args.eval)
                    else:
                        assert False, "Could not recognize RMVP3D or SMVP3D dataset!"
                elif os.path.exists(os.path.join(args.source_path, "camera_params.npz")):
                    print("Found camera_params.npz file, assuming Mitsuba dataset!")
                    scene_info = sceneLoadTypeCallbacks["Mitsuba"](args.source_path, args.images, args.eval)
                else:
                    print(args.source_path)
                    assert False, "Could not recognize scene type!"
        else:
            if LP is not None:
                print("Double view mode enabled, loading scene with LP cameras")
                all_subfolders = [os.path.join(args.source_path, d) for d in os.listdir(args.source_path) if os.path.isdir(os.path.join(args.source_path, d))]
                # import pdb;pdb.set_trace()  
                if all(os.path.exists(os.path.join(subfolder, "camera_params.npz")) for subfolder in all_subfolders):
                    print("Found camera_params.npz file, assuming Mitsuba data set!")
                    scene_info = sceneLoadTypeCallbacks["MitsubaDoubleLP"](args.source_path, args.images, args.eval, LP=LP)
                elif all(os.path.exists(os.path.join(subfolder, "sparse")) for subfolder in all_subfolders):
                    print("Found sparse folder, assuming Colmap data set!")
                    scene_info = sceneLoadTypeCallbacks["ColmapDoubleLP"](args.source_path, args.images, args.eval, LP=LP)
                else:
                    assert False, "DoubleView, Could not recognize scene type!"
            else:
                print("Double view mode enabled, loading scene without LP cameras")
                all_subfolders = [os.path.join(args.source_path, d) for d in os.listdir(args.source_path) if os.path.isdir(os.path.join(args.source_path, d))]
                if all(os.path.exists(os.path.join(subfolder, "camera_params.npz")) for subfolder in all_subfolders):
                    print("Found camera_params.npz file, assuming Mitsuba data set!")
                    scene_info = sceneLoadTypeCallbacks["MitsubaDoubleViews"](args.source_path, args.images, args.eval, LP=LP)
                elif all(os.path.exists(os.path.join(subfolder, "sparse")) for subfolder in all_subfolders):
                    print("Found sparse folder, assuming Colmap data set!")
                    scene_info = sceneLoadTypeCallbacks["ColmapDoubleViews"](args.source_path, args.images, args.eval, LP=LP)
                else:
                    assert False, "DoubleView mode but not mitsuba requires LP argument!"

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
            if args.relight:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply"), relight=True, args=args)
            else:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply"), args=args)        
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, args)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]