#
# Copyright (C) 2023, Inria GRAPHDECO research group
# Copyright (C) 2025, New York University
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE file.
#
# Original 3D Gaussian Splatting code from:
# https://github.com/graphdeco-inria/gaussian-splatting
#
# CLM-GS modifications by NYU Systems Group
# https://github.com/nyu-systems/CLM-GS
#

import os
import random
import json
from random import randint
from torch.utils.data import Dataset
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from strategies.base_gaussian_model import BaseGaussianModel
from utils.camera_utils import (
    cameraList_from_camInfos,
    camera_to_JSON,
    loadCam,
    predecode_dataset_to_disk,
    clean_up_disk,
    loadCam_raw_from_disk,
)
import utils.general_utils as utils
import psutil
from scene.cameras import set_space_sort_key_dim


class Scene:

    gaussians: BaseGaussianModel

    def __init__(
        self,
        args,
        gaussians: BaseGaussianModel,
        load_iteration=None,
        shuffle=True,
        only_for_rendering=False,
    ):

        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.args = args
        self.train_cameras_info = None
        self.test_cameras_info = None
        log_file = utils.get_log_file()

        utils.log_cpu_memory_usage("before loading images meta data")

        if os.path.exists(
            os.path.join(args.source_path, "sparse")
        ):  # This is the format from colmap.
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval, args.llffhold
            )
        else:  # NOTE: we only support colmap format and matrixcity format dataset for now.
            scene_info = sceneLoadTypeCallbacks["City"](
                args.source_path,
                args.random_background,
                args.white_background,
                llffhold=args.llffhold,
                is_debug=args.debug,
            )

        if not self.loaded_iter:
            # with open(scene_info.ply_path, "rb") as src_file, open(
            #     os.path.join(self.model_path, "input.ply"), "wb"
            # ) as dest_file:
            # dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        # zch closed shufle
        if shuffle: 
            # random.shuffle(
            #     scene_info.train_cameras
            # )  # Multi-res consistent random shuffling
            random.shuffle(
                scene_info.test_cameras
            )  # Multi-res consistent random shuffling

        scene_info.train_cameras.sort(key=lambda x: x.image_name)

        utils.log_cpu_memory_usage("before decoding images")

        self.cameras_extent = scene_info.nerf_normalization["radius"] # 是一个全局缩放系数 (Scale Factor), 1)动态缩放空间位置的学习率 2)作为高斯球分裂的尺度参考, 使得能够适应各种尺度的场景
        self.scene_info = scene_info  # For torch dataloader, save scene_info

        # Set image size to global varaible. In case not all image sizes are identical, choose the minimum.
        orig_w, orig_h = (
            min(
                [
                    camera.width
                    for camera in scene_info.train_cameras + scene_info.test_cameras
                ]
            ),
            min(
                [
                    camera.height
                    for camera in scene_info.train_cameras + scene_info.test_cameras
                ]
            ),
        )
        utils.set_img_size(orig_h, orig_w)
        # Dataset size in GB
        if args.num_train_cameras > 0:
            assert (
                args.num_test_cameras > 0
            ), "Should set both `num_train_cameras` and `num_test_cameras`"
            assert args.num_train_cameras <= len(
                scene_info.train_cameras
            ) and args.num_test_cameras <= len(
                scene_info.test_cameras
            ), "Can not config more cameras than dataset size"
            dataset_size_in_GB = (
                1.0
                * (args.num_train_cameras + args.num_test_cameras)
                * orig_w
                * orig_h
                * 3
                / 1e9
            )
        else:
            dataset_size_in_GB = ( # rubble-colmap 79.409718984 GB
                1.0
                * (len(scene_info.train_cameras) + len(scene_info.test_cameras))
                * orig_w
                * orig_h
                * 3
                / 1e9
            )
        log_file.write(f"Dataset size: {dataset_size_in_GB} GB\n")

        # Preprocess dataset
        # Train on original resolution, no downsampling in our implementation.

        # Predecode dataset as raw files to local disk
        # If decode_dataset_path does not exist, run predecode
        if args.decode_dataset_path == "":
            args.decode_dataset_path = os.path.join(
                args.source_path, f"decoded_{args.images}"
            )
            os.makedirs(args.decode_dataset_path, exist_ok=True)
            print("create folder: ", args.decode_dataset_path)
            log_file.write(f"create folder: {args.decode_dataset_path}\n")

        self.decode_dataset_path = os.path.join(args.decode_dataset_path, "dataset_raw")

        if not os.path.isdir(self.decode_dataset_path):
            os.makedirs(self.decode_dataset_path)
            statvfs = os.statvfs(args.decode_dataset_path)
            available_space_in_GB = 1.0 * statvfs.f_frsize * statvfs.f_bavail / 1e9
            assert (
                available_space_in_GB >= dataset_size_in_GB
            ), f"Not enough space in disk for decompressed dataset. avail: {available_space_in_GB}. need: {dataset_size_in_GB}"
            log_file.write(
                f"[NOTE]: Pre-decoding dataset({dataset_size_in_GB}GB) to disk dir: {self.decode_dataset_path}\n"
            )
            do_decode = True
        else:
            log_file.write(
                f"[NOTE]: Reusing decoded dataset({dataset_size_in_GB}GB) in disk dir: {self.decode_dataset_path}\n"
            )
            utils.print_rank_0(
                f"Reusing decoded dataset on disk: {self.decode_dataset_path}"
            )
            do_decode = False

        self.train_cameras = None
        self.test_cameras = None
        if args.num_train_cameras >= 0:
            train_cameras = scene_info.train_cameras[: args.num_train_cameras]
        else:
            train_cameras = scene_info.train_cameras
        if do_decode:
            utils.print_rank_0("Decoding Training Cameras To Disk")
            predecode_dataset_to_disk(train_cameras, args)
        self.train_cameras_info = train_cameras

        if len(train_cameras) > 0:
            log_file.write("Train Image size: {}x{}\n".format(orig_h, orig_w))

        if args.eval:
            if args.num_test_cameras >= 0:
                test_cameras = scene_info.test_cameras[: args.num_test_cameras]
            else:
                test_cameras = scene_info.test_cameras
            if do_decode:
                utils.print_rank_0("Decoding Test Cameras To Disk")
                predecode_dataset_to_disk(test_cameras, args)
            self.test_cameras_info = test_cameras

            if len(test_cameras) > 0:
                log_file.write("Test Image size: {}x{}\n".format(orig_h, orig_w))

        utils.check_initial_gpu_memory_usage("after Loading all images")
        utils.log_cpu_memory_usage("after decoding images")

        if args.load_pt_path != "":
            self.gaussians.load_tensors(args.load_pt_path)
        elif args.load_ply_path != "":
            self.gaussians.load_ply(args.load_ply_path)
        elif load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter)
                )
            )
        else:
            pure_ssd_resume_manifest = getattr(args, "_pure_ssd_resume_manifest", None)
            pure_ssd_prebuilt_manifest = getattr(args, "_pure_ssd_prebuilt_manifest", None)
            if pure_ssd_prebuilt_manifest is not None:
                self.gaussians.prepare_pure_ssd_prebuilt_manifest(
                    pure_ssd_prebuilt_manifest,
                    self.cameras_extent,
                )
            elif pure_ssd_resume_manifest is not None:
                self.gaussians.prepare_pure_ssd_checkpoint_resume(
                    pure_ssd_resume_manifest,
                    self.cameras_extent,
                )
            elif (
                getattr(args, "pure_ssd_offload", False)
                and getattr(args, "pure_ssd_init_backend", "auto") == "streaming"
            ):
                if scene_info.ply_path == "":
                    raise ValueError("pure SSD streaming init requires scene_info.ply_path")
                self.gaussians.prepare_streaming_ply_init(
                    scene_info.ply_path,
                    self.cameras_extent,
                )
            else:
                # Three model types use create_from_pcd
                self.gaussians.create_from_pcd(
                    scene_info.point_cloud,
                    self.cameras_extent,
                    subsample_ratio=args.initial_point_cloud_downsampled_ratio,
                )

        utils.check_initial_gpu_memory_usage("after initializing point cloud")
        utils.log_cpu_memory_usage("after loading initial 3dgs points")

        # get the longest axis in self.gaussians.  Streaming pure-SSD init has
        # not materialized full _xyz in RAM, so use a harmless default until
        # the storage adapter computes block bounds from the streamed PLY.
        if (
            getattr(self.gaussians, "_streaming_ply_init_pending", False)
            or getattr(self.gaussians, "_pure_ssd_resume_pending", False)
            or getattr(self.gaussians, "_pure_ssd_prebuilt_pending", False)
        ):
            longest_axis = 0
            log_file.write(
                "[PURE SSD INIT] _xyz is not RAM-resident during Scene init; using longest_axis=0 placeholder\n"
            )
        else:
            longest_axis = (
                (self.gaussians._xyz.max(0)[0] - self.gaussians._xyz.min(0)[0])
                .argmax()
                .item()
            )
        set_space_sort_key_dim(longest_axis)

    def save_tensors(self, iteration):
        parent_path = os.path.join(
            self.model_path, f"saved_tensors/iteration_{iteration}"
        )
        self.gaussians.save_tensors(parent_path)

    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        avail_ram_bytes = psutil.virtual_memory().available
        N = self.gaussians._xyz.shape[0]
        required_bytes = 16 * N * 59 * 4

        # Check if the available ram can fit all attributes for cat and map with 20% redundency.
        if avail_ram_bytes * 0.8 > required_bytes:
            # Save in one ply file.
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        else:
            # Save in multiple sub files.
            n_split = int(
                (required_bytes + avail_ram_bytes * 0.8 - 1) // (avail_ram_bytes * 0.8)
            )
            split_size = (N + n_split - 1) // n_split
            utils.print_rank_0(
                f"Requires {required_bytes / 1024 / 1024 / 1024:.3f} GB RAM for saving. Avail {avail_ram_bytes / 1024 / 1024 / 1024:.3f} GB. Split: {N} -> {n_split} x {split_size}"
            )
            self.gaussians.save_sub_plys(
                os.path.join(point_cloud_path, "point_cloud.ply"), n_split, split_size
            )

    def getTrainCameras(self):
        return self.train_cameras

    def getTrainCamerasInfo(self):
        return self.train_cameras_info

    def getTestCameras(self):
        return self.test_cameras

    def getTestCamerasInfo(self):
        return self.test_cameras_info

    def log_scene_info_to_file(self, log_file, prefix_str=""):
        if prefix_str:
            log_file.write(f"{prefix_str}\n")

        def _write_tensor_shape(name, tensor):
            if tensor is None:
                log_file.write(f"{name} shape: <released>\n")
            else:
                log_file.write(f"{name} shape: {tensor.shape}\n")

        # In paper out-of-core mode the full tensors can be released before
        # training starts, so log a readable marker instead of dereferencing
        # None.
        _write_tensor_shape("xyz", self.gaussians._xyz)
        _write_tensor_shape("f_dc", self.gaussians._features_dc)
        _write_tensor_shape("f_rest", self.gaussians._features_rest)
        _write_tensor_shape("opacity", self.gaussians._opacity)
        _write_tensor_shape("scaling", self.gaussians._scaling)
        _write_tensor_shape("rotation", self.gaussians._rotation)

        if getattr(self.gaussians, "_paper_unified_params_freed", False):
            num_total = getattr(self.gaussians, "_paper_unified_params_num_total", None)
            param_width = getattr(self.gaussians, "_paper_unified_params_param_width", None)
            log_file.write("paper out-of-core mode: True\n")
            if num_total is not None:
                log_file.write(f"paper total gaussians: {num_total}\n")
            if param_width is not None:
                log_file.write(f"paper param width: {param_width}\n")

            gpu_ws = getattr(self.gaussians, "gpu_working_set_manager", None)
            if gpu_ws is not None:
                num_total_ws = getattr(gpu_ws, "num_total_gaussians", None)
                block_size = getattr(gpu_ws, "block_size", None)
                if num_total_ws is not None:
                    log_file.write(f"gpu working-set total gaussians: {num_total_ws}\n")
                if block_size is not None:
                    log_file.write(f"gpu working-set block size: {block_size}\n")

    def clean_up(self):
        pass
        # Remove the predecoded dataset from disk
        # if self.args.decode_dataset_to_disk and not self.args.reuse_decoded_dataset:
        #     clean_up_disk(self.args)
        #     utils.print_rank_0("Cleaned up decoded dataset on disk.")


class SceneDataset:
    def __init__(self, cameras, cameras_info=None):
        self.cameras = cameras
        self.cameras_info = cameras_info
        self.camera_size = (
            len(self.cameras) if self.cameras is not None else len(self.cameras_info)
        )

        self.cur_epoch_cameras = []
        self.cur_iteration = 0

        self.iteration_loss = []
        self.epoch_loss = []

        self.log_file = utils.get_log_file()
        self.args = utils.get_args()

        self.last_time_point = None
        self.epoch_time = []
        self.epoch_n_sample = []

    @property
    def cur_epoch(self):
        return len(self.epoch_loss)

    @property
    def cur_iteration_in_epoch(self):
        return len(self.iteration_loss)

    def get_one_camera(self, batched_cameras_uid):
        args = utils.get_args()
        if len(self.cur_epoch_cameras) == 0:
            # start a new epoch
            self.cur_epoch_cameras = list(range(self.camera_size))
            random.shuffle(self.cur_epoch_cameras) # zch closed shufle

        self.cur_iteration += 1

        if args.decode_dataset_to_disk:
            idx = 0
            while self.cur_epoch_cameras[idx] in batched_cameras_uid:
                idx += 1
            camera_idx = self.cur_epoch_cameras.pop(idx)
            viewpoint_cam = loadCam_raw_from_disk(
                args, camera_idx, self.cameras_info[camera_idx], to_gpu=True
            )
        else:
            idx = 0
            while self.cameras[self.cur_epoch_cameras[idx]].uid in batched_cameras_uid:
                idx += 1
            camera_idx = self.cur_epoch_cameras.pop(idx)
            viewpoint_cam = self.cameras[camera_idx]
        return camera_idx, viewpoint_cam

    def get_batched_cameras(self, batch_size):
        assert (
            batch_size <= self.camera_size
        ), "Batch size is larger than the number of cameras in the scene."
        batched_cameras = []
        batched_cameras_uid = []
        for i in range(batch_size):
            _, camera = self.get_one_camera(batched_cameras_uid)
            batched_cameras.append(camera)
            batched_cameras_uid.append(camera.uid)

        return batched_cameras

    def get_batched_cameras_idx(self, batch_size):
        assert (
            batch_size <= self.camera_size
        ), "Batch size is larger than the number of cameras in the scene."
        batched_cameras_idx = []
        batched_cameras_uid = []
        for i in range(batch_size):
            idx, camera = self.get_one_camera(batched_cameras_uid)
            batched_cameras_uid.append(camera.uid)
            batched_cameras_idx.append(idx)

        return batched_cameras_idx

    def get_batched_cameras_from_idx(self, idx_list):
        return [self.cameras[i] for i in idx_list]

    def update_losses(self, losses):
        for loss in losses:
            self.iteration_loss.append(loss)
            if len(self.iteration_loss) % self.camera_size == 0:
                self.epoch_loss.append(
                    sum(self.iteration_loss[-self.camera_size :]) / self.camera_size
                )
                self.log_file.write(
                    "epoch {} loss: {}\n".format(
                        len(self.epoch_loss), self.epoch_loss[-1]
                    )
                )
                self.iteration_loss = []


def load_scene_info_for_rendering(args):
    """
    Load scene information (camera poses, intrinsics) WITHOUT decoding images to disk.
    This is a lightweight version for rendering-only workflows (e.g., trajectory rendering).

    Args:
        args: Arguments containing source_path, images, eval, llffhold, etc.

    Returns:
        tuple: (scene_info, cameras_extent) containing camera metadata and scene radius
    """
    # Load scene metadata based on dataset type
    if os.path.exists(os.path.join(args.source_path, "sparse")):
        # COLMAP format
        scene_info = sceneLoadTypeCallbacks["Colmap"](
            args.source_path, args.images, args.eval, args.llffhold
        )
    elif "matrixcity" in args.source_path:
        # MatrixCity format
        scene_info = sceneLoadTypeCallbacks["City"](
            args.source_path,
            args.random_background,
            args.white_background,
            llffhold=args.llffhold,
        )
    else:
        raise ValueError("No valid dataset found in the source path")

    # Get scene extent (radius for normalization)
    cameras_extent = scene_info.nerf_normalization["radius"]

    return scene_info, cameras_extent


def custom_collate_fn(batch):
    return batch


class OffloadSceneDataset(Dataset):
    def __init__(self, cameras_info):
        self.cameras_info = cameras_info
        self.camera_size = len(self.cameras_info)

        self.cur_epoch_cameras = []
        self.cur_iteration = 0

        self.iteration_loss = []
        self.epoch_loss = []

        self.log_file = utils.get_log_file()
        self.args = utils.get_args()

        self.last_time_point = None
        self.epoch_time = []
        self.epoch_n_sample = []

    def __len__(self):
        return self.camera_size

    def __getitem__(self, id):
        return loadCam_raw_from_disk(
            self.args,
            id,
            self.cameras_info[id],
        )

    @property
    def cur_epoch(self):
        return len(self.epoch_loss)

    @property
    def cur_iteration_in_epoch(self):
        return len(self.iteration_loss)

    def update_losses(self, losses):
        for loss in losses:
            self.iteration_loss.append(loss)
            if len(self.iteration_loss) % self.camera_size == 0:
                self.epoch_loss.append(
                    sum(self.iteration_loss[-self.camera_size :]) / self.camera_size
                )
                self.log_file.write(
                    "epoch {} loss: {}\n".format(
                        len(self.epoch_loss), self.epoch_loss[-1]
                    )
                )
                self.iteration_loss = []
