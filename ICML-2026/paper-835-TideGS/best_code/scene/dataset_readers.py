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
import glob
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import (
    read_extrinsics_text,
    read_intrinsics_text,
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
    read_points3D_text,
)
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import utils.general_utils as utils
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.graphics_utils import BasicPointCloud
import torch


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    cam_center: np.array = None


class PureSSDPlyInitError(RuntimeError):
    pass


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def _sample_frames_for_debug(frames, max_cameras: int, mode: str):
    """Subsample camera frames for faster smoke/debug runs."""
    if max_cameras is None or max_cameras <= 0 or len(frames) <= max_cameras:
        return frames

    args = utils.get_args()
    sample_mode = str(getattr(args, "debug_camera_sample_mode", "linspace")).lower()
    sample_start = int(getattr(args, "debug_camera_sample_start", 0))

    if sample_mode == "linspace":
        indices = np.linspace(0, len(frames) - 1, num=max_cameras, dtype=int)
        indices = np.unique(indices)
    elif sample_mode == "contiguous":
        indices = np.arange(0, max_cameras, dtype=int)
    elif sample_mode == "window":
        start = min(sample_start, max(0, len(frames) - max_cameras))
        indices = np.arange(start, start + max_cameras, dtype=int)
    else:
        raise ValueError(
            f"Invalid debug_camera_sample_mode={sample_mode!r}; expected linspace, contiguous, or window"
        )

    sampled_frames = [frames[idx] for idx in indices.tolist()]
    utils.print_rank_0(
        f"[DEBUG DATASET] Sampling {len(sampled_frames)}/{len(frames)} {mode} cameras "
        f"for fast validation (sample_mode={sample_mode}, start={int(indices[0])}, end={int(indices[-1])})"
    )
    return sampled_frames


def _get_city_frame_ref(frame):
    if "file_name" in frame and frame["file_name"]:
        return frame["file_name"]
    if "file_path" in frame and frame["file_path"]:
        return frame["file_path"]
    raise KeyError("City frame must contain either 'file_name' or 'file_path'")


def _resolve_city_frame_image(path, transformsfile, frame, mode):
    frame_ref = _get_city_frame_ref(frame)
    frame_ref_str = str(frame_ref)
    transforms_abs = os.path.realpath(os.path.join(path, transformsfile))
    transforms_dir = os.path.dirname(transforms_abs)

    candidates = []
    if "file_name" in frame and frame["file_name"]:
        candidates.extend([
            os.path.join(path, "../..", mode, frame_ref_str),
            os.path.join(transforms_dir, "../..", mode, frame_ref_str),
            os.path.join(path, frame_ref_str),
            os.path.join(transforms_dir, frame_ref_str),
        ])
    else:
        candidates.extend([
            os.path.join(transforms_dir, frame_ref_str),
            os.path.join(path, frame_ref_str),
        ])

    normalized_candidates = []
    seen = set()
    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if normalized not in seen:
            normalized_candidates.append(normalized)
            seen.add(normalized)

    for candidate in normalized_candidates:
        if os.path.exists(candidate):
            return candidate, os.path.basename(frame_ref_str)

    return normalized_candidates[0], os.path.basename(frame_ref_str)


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
    args = utils.get_args()
    cam_infos = []
    utils.print_rank_0("Loading cameras from disk...")
    for idx, key in tqdm(
        enumerate(cam_extrinsics),
        total=len(cam_extrinsics),
    ):

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            # we're ignoring the 4 distortion
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(
            image_path
        )  # this is a lazy load, the image is not loaded yet
        width, height = image.size

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=None,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
        )

        # release memory
        image.close()
        image = None

        cam_infos.append(cam_info)
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    try:
        colors = (
            np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
        )
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    try:
        normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    except:
        normals = np.random.rand(positions.shape[0], positions.shape[1])
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def read_ply_vertex_count(path):
    """Read only the PLY header and return the declared vertex count."""
    with open(path, "rb") as f:
        for raw_line in f:
            line = raw_line.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return None


def should_stream_pure_ssd_ply():
    args = utils.get_args()
    return bool(
        getattr(args, "pure_ssd_offload", False)
        and getattr(args, "pure_ssd_init_backend", "auto") == "streaming"
    )


def guard_pure_ssd_ply_fetch(path):
    """Prevent accidental full in-memory PLY loads for billion-scale pure SSD runs."""
    args = utils.get_args()
    if not getattr(args, "pure_ssd_offload", False):
        return

    limit = int(getattr(args, "pure_ssd_max_inmemory_init_points", 50_000_000))
    if limit < 0:
        return

    vertex_count = read_ply_vertex_count(path)
    if vertex_count is None:
        print(
            f"[PURE SSD PREFLIGHT] Could not read vertex count from {path}; "
            "continuing with regular PLY loader"
        )
        return

    print(
        f"[PURE SSD PREFLIGHT] PLY vertices={vertex_count:,}, "
        f"in-memory init guard={limit:,}"
    )
    if vertex_count > limit:
        raise PureSSDPlyInitError(
            "[PURE SSD PREFLIGHT] Refusing to load this PLY fully into RAM in "
            f"pure SSD mode: vertices={vertex_count:,} > limit={limit:,}. "
            "Use a streaming PLY-to-SSD initializer for this scale, or set "
            "--pure_ssd_max_inmemory_init_points -1 only for debugging on a "
            "machine with enough RAM."
        )


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=10):
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
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    args = utils.get_args()
    if args.dense_ply_file == "":
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
    else:
        ply_path = args.dense_ply_file
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print(
            "Converting point3d.bin to .ply, will happen only the first time you open the scene."
        )
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    if args.load_pt_path != "":
        pcd = None
    elif should_stream_pure_ssd_ply():
        utils.print_rank_0(
            f"[STREAMING PLY INIT] Deferring PLY load to SSD streaming initializer: {ply_path}"
        )
        pcd = None
    else:
        try:
            guard_pure_ssd_ply_fetch(ply_path)
            pcd = fetchPly(ply_path)
        except PureSSDPlyInitError:
            raise
        except:
            pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCamerasFromTransformsCity(
    path,
    transformsfile,
    random_background,
    white_background,
    extension=".png",
    undistorted=False,
    is_debug=False,
    mode="train",
):
    args = utils.get_args()
    cam_infos = []
    if undistorted:
        print("Undistortion the images!!!")
        # TODO: Support undistortion here. Please refer to octree-gs implementation.

    if args.matrixcity_ocean_mask:
        transforms_ocean_json_path = os.path.join(
            path, transformsfile.replace(".json", "_ocean_info.json")
        )

        print(f"Loading ocean info from {transforms_ocean_json_path}")
        with open(transforms_ocean_json_path, "r") as ocean_json_file:
            transforms_ocean = json.load(ocean_json_file)

        print(f"Loading original transforms from {os.path.join(path, transformsfile)}")
        with open(os.path.join(path, transformsfile)) as json_file:
            transforms = json.load(json_file)

        transforms_frames = transforms["frames"]
        transforms_ocean_frames = transforms_ocean["frames"]
        assert len(transforms_frames) == len(
            transforms_ocean_frames
        ), "Ocean info does not match the original frames"

        new_frames = []
        for i in range(len(transforms_frames)):
            # assert transforms_frames[i]["file_path"] suffix is transforms_ocean_frames[i]["file_name"]
            src_name = os.path.basename(_get_city_frame_ref(transforms_frames[i]))
            ocean_name = os.path.basename(_get_city_frame_ref(transforms_ocean_frames[i]))
            assert (
                src_name[-len(ocean_name):] == ocean_name
            ), f"Ocean info does not match the original frames at index {i}. Filename: {src_name} vs {ocean_name}"
            if not transforms_ocean_frames[i]["is_ocean"]:
                new_frames.append(transforms_frames[i])
        transforms["frames"] = new_frames

    else:
        with open(os.path.join(path, transformsfile)) as json_file:
            transforms = json.load(json_file)

    # contents = json.load(json_file)
    try:
        fovx = transforms["camera_angle_x"]
    except:
        fovx = None

    frames = transforms["frames"]
    max_debug_cameras = (
        args.debug_max_train_cameras if mode == "train" else args.debug_max_test_cameras
    )
    frames = _sample_frames_for_debug(frames, max_debug_cameras, mode)
    # check if filename already contain postfix
    # if frames[0]["file_name"].split(".")[-1] in ["jpg", "jpeg", "JPG", "png"]:
    #     extension = ""

    c2ws = np.array([frame["transform_matrix"] for frame in frames])

    Ts = c2ws[:, :3, 3] # camera centers 

    ct = 0

    progress_bar = tqdm(
        frames,
        desc="Loading dataset",
    )

    for idx, frame in enumerate(frames):
        cam_path, cam_name = _resolve_city_frame_image(path, transformsfile, frame, mode)
        if not os.path.exists(cam_path):
            print(f"File {cam_path} not found, skipping...")
            continue
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])

        if idx % 10 == 0:
            progress_bar.set_postfix({"num": f"{ct}/{len(frames)}"})
            progress_bar.update(10)
        if idx == len(frames) - 1:
            progress_bar.close()

        ct += 1

        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3 ])
        T = w2c[:3, 3]

        # # ====== debug ======
        # c2w = np.array(frame["transform_matrix"])

        # # ===============================================================
        # # [FIX] 重构坐标系 (Coordinate System Re-alignment)
        # # ===============================================================
        # # 根据 Debug 观测: 
        # # 1. 原始数据的 X 轴 (Col 0) 是 Forward (指向物体)
        # # 2. 原始数据的 Z 轴 (Col 2) 是 Up (指向天空)
        # #
        # # 目标 Colmap 格式: X=Right, Y=Down, Z=Forward
        # # ---------------------------------------------------------------
        
        # # 1. 提取原始轴
        # raw_x_axis = c2w[:3, 0] # 目前的 Forward
        # raw_y_axis = c2w[:3, 1] 
        # raw_z_axis = c2w[:3, 2] # 目前的 Up

        # # 2. 构建新轴
        # # Step A: 确定 Forward (Z)
        # # 既然 X 轴对着物体，那它就是我们的新 Z 轴
        # new_z_axis = raw_x_axis 
        # new_z_axis = new_z_axis / np.linalg.norm(new_z_axis) # 归一化

        # # Step B: 确定 Down (Y)
        # # 既然原始 Z 轴朝上 (Up)，那我们的新 Y 轴 (Down) 应该是它的反方向
        # new_y_axis = -raw_z_axis
        # new_y_axis = new_y_axis / np.linalg.norm(new_y_axis)

        # # Step C: 确定 Right (X)
        # # 利用右手定则: Right = Down x Forward (或者 Forward x Up)
        # # Cross(New_Y, New_Z)
        # new_x_axis = np.cross(new_y_axis, new_z_axis)
        # new_x_axis = new_x_axis / np.linalg.norm(new_x_axis)

        # # 3. 组装回 c2w
        # c2w[:3, 0] = new_x_axis
        # c2w[:3, 1] = new_y_axis
        # c2w[:3, 2] = new_z_axis
        # # ===============================================================

        # # get the world-to-camera transform and set R, T
        # w2c = np.linalg.inv(c2w)

        # # [CRITICAL] 这里的转置必须保留！
        # # 此时 w2c 是标准的 Row-Major。
        # # 我们需要转置 R 部分，以便它在内存中变成 Column-Major 格式
        # # 这样在后续 viewmat.transpose(0, 1) 时（或者不转置时）才能对上 gsplat
        # R = np.transpose(w2c[:3, :3]) 
        # T = w2c[:3, 3]

        # # # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        # # c2w[:3, 1:3] *= -1

        # # # get the world-to-camera transform and set R, T
        # # w2c = np.linalg.inv(c2w)

        # # R = np.transpose(
        # #     w2c[:3, :3]
        # # )  # R is stored transposed due to 'glm' in CUDA code

        # # T = w2c[:3, 3]

        # # ====== debug ======

        image_path = cam_path
        image_name = cam_name[-17:]  # Path(cam_name).stem
        image = Image.open(image_path)

        if fovx is not None:
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx
        else:
            # given focal in pixel unit
            FovY = focal2fov(frame["fl_y"], image.size[1])
            FovX = focal2fov(frame["fl_x"], image.size[0])

        cam_infos.append(
            CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=None,
                image_path=image_path,
                image_name=image_name,
                width=image.size[0],
                height=image.size[1],
                cam_center=Ts[idx],
            )
        )

        # release memory
        image.close()
        image = None

        if is_debug and idx > 128:
            break

    return cam_infos


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
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
                1 - norm_data[:, :, 3:4]
            )
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                )
            )

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension
    )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension
    )

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
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCityInfo(
    path,
    random_background,
    white_background,
    extension=".tif",
    llffhold=8,
    undistorted=False,
    is_debug=False,
):
    args = utils.get_args()

    train_json_path = os.path.join(path, f"transforms_train.json")
    test_json_path = os.path.join(path, f"transforms_test.json")
    print(
        "Reading Training Transforms from {} {}".format(train_json_path, test_json_path)
    )

    train_cam_infos = readCamerasFromTransformsCity(
        path,
        train_json_path,
        random_background,
        white_background,
        extension,
        undistorted,
        is_debug=is_debug,
        mode="train",
    )

    if args.eval:
        test_cam_infos = readCamerasFromTransformsCity(
            path,
            test_json_path,
            random_background,
            white_background,
            extension,
            undistorted,
            mode="test",
        )
    else:
        test_cam_infos = []
    print("Load Cameras(train, test): ", len(train_cam_infos), len(test_cam_infos))

    nerf_normalization = getNerfppNorm(train_cam_infos)

    pure_ssd_resume_manifest = getattr(args, "_pure_ssd_resume_manifest", None)
    pure_ssd_prebuilt_manifest = getattr(args, "_pure_ssd_prebuilt_manifest", None)
    pure_ssd_metadata_manifest = pure_ssd_prebuilt_manifest or pure_ssd_resume_manifest
    resume_source_ply = (pure_ssd_metadata_manifest or {}).get("source_ply", "")

    if args.dense_ply_file == "" and not resume_source_ply:
        ply_candidates = glob.glob(os.path.join(path, "*.ply"))
        if len(ply_candidates) == 0:
            raise FileNotFoundError(
                f"No ply file found in {path}. "
                "Pass --dense_ply_file to specify the point cloud explicitly."
            )
        ply_path = ply_candidates[0]
    else:
        ply_path = args.dense_ply_file or resume_source_ply

    if pure_ssd_prebuilt_manifest is not None:
        utils.print_rank_0(
            f"[PURE SSD PREBUILT] Skipping PLY load; using prebuilt SSD metadata for {ply_path}"
        )
        pcd = None
    elif pure_ssd_resume_manifest is not None:
        utils.print_rank_0(
            f"[PURE SSD RESUME] Skipping PLY load; using checkpoint snapshot metadata for {ply_path}"
        )
        pcd = None
    elif os.path.exists(ply_path):
        if should_stream_pure_ssd_ply():
            utils.print_rank_0(
                f"[STREAMING PLY INIT] Deferring PLY load to SSD streaming initializer: {ply_path}"
            )
            pcd = None
        else:
            try:
                guard_pure_ssd_ply_fetch(ply_path)
                pcd = fetchPly(ply_path)
            except PureSSDPlyInitError:
                raise
            except:
                raise ValueError("must have tiepoints!")
    else:
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "City": readCityInfo,
}
