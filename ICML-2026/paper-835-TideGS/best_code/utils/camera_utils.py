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
import shutil

from numba.cuda.cudadrv.runtime import Runtime
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch, get_args, get_log_file
import utils.general_utils as utils
from tqdm import tqdm
from utils.graphics_utils import fov2focal
import time
import multiprocessing
from multiprocessing import shared_memory
import torch
from PIL import Image


def loadCam(args, id, cam_info, decompressed_image=None, return_image=False):
    # orig_w, orig_h = cam_info.width, cam_info.height
    # assert (
    #     orig_w == utils.get_img_width() and orig_h == utils.get_img_height()
    # ), f"All images should have the same size. Found {orig_w}, {utils.get_img_width()}, {orig_h}, and {utils.get_img_height()}."

    args = get_args()
    log_file = get_log_file()
    # resolution = orig_w, orig_h
    resolution = utils.get_img_width(), utils.get_img_height()
    # NOTE: we do not support downsampling here.

    # Single GPU mode - always load
    if args.time_image_loading:
        start_time = time.time()
    image = Image.open(cam_info.image_path)
    resized_image_rgb = PILtoTorch(
        image, resolution, args, log_file, decompressed_image=decompressed_image
    )
    if args.time_image_loading:
        log_file.write(f"PILtoTorch image in {time.time() - start_time} seconds\n")

    # assert resized_image_rgb.shape[0] == 3, "Image should have exactly 3 channels!"
    gt_image = resized_image_rgb[:3, ...].contiguous()
    loaded_mask = None

    # Free the memory: because the PIL image has been converted to torch tensor, we don't need it anymore. And it takes up lots of cpu memory.
    image.close()
    image = None

    if return_image:
        return gt_image

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=id,
    )


def loadCam_raw_from_disk(args, id, cam_info, to_gpu=False):
    # orig_w, orig_h = cam_info.width, cam_info.height
    # assert (
    #     orig_w == utils.get_img_width() and orig_h == utils.get_img_height()
    # ), "All images should have the same size. "
    orig_h, orig_w = utils.get_img_size()

    # Get dececoded gt_image from disk
    try:
        with open(
            os.path.join(
                args.decode_dataset_path,
                "dataset_raw",
                (cam_info.image_name.lstrip("/") + ".raw"),
            ),
            "rb",
        ) as raw_file:
            raw_data = raw_file.read()
    except RuntimeError as e:
        raise RuntimeError(e)

    raw_np = np.frombuffer(raw_data, dtype=np.uint8)
    if to_gpu:
        image_tensor_host = (
            torch.tensor(raw_np, dtype=torch.uint8)
            .view(orig_h, orig_w, -1)
            .permute(2, 0, 1)
            .pin_memory()
        )
        image_tensor = image_tensor_host.to("cuda")
    else:
        image_tensor = (
            torch.tensor(raw_np, dtype=torch.uint8)
            .view(orig_h, orig_w, -1)
            .permute(2, 0, 1)
        )

    # assert image_tensor.shape[0] == 3, image_tensor.shape[0]
    image_tensor = image_tensor[:3, ...].contiguous()

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=image_tensor,
        gt_alpha_mask=None,
        image_name=cam_info.image_name,
        uid=id,
        offload=True,
    )


def load_decompressed_image(params):
    args, id, cam_info = params
    return loadCam(args, id, cam_info, decompressed_image=None, return_image=True)


# Modify this code to support shared_memory.SharedMemory to make inter-process communication faster
def decompressed_images_from_camInfos_multiprocess(cam_infos, args):
    args = get_args()
    decompressed_images = []
    total_cameras = len(cam_infos)

    # Create a pool of processes
    with multiprocessing.Pool(processes=2) as pool:
        # Prepare data for processing
        tasks = [(args, id, cam_info) for id, cam_info in enumerate(cam_infos)]

        # Map load_camera_data to the tasks
        # results = pool.map(load_decompressed_image, tasks)
        results = list(
            tqdm(
                pool.imap(load_decompressed_image, tasks),
                total=total_cameras,
            )
        )

        for id, result in enumerate(results):
            decompressed_images.append(result)

    return decompressed_images


def decompress_and_scale_image(cam_info):
    pil_image = cam_info.image
    resolution = cam_info.image.size  # (w, h)
    # print("cam_info.image.size: ", cam_info.image.size)
    pil_image.load()
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = np.array(resized_image_PIL)  # (h, w, 3)
    # print("resized_image.shape: ", resized_image.shape)
    if len(resized_image.shape) == 3:
        return resized_image.transpose(2, 0, 1)
    else:
        return resized_image[..., np.newaxis].transpose(2, 0, 1)


def load_decompressed_image_shared(params):
    shared_mem_name, args, id, cam_info, resolution_scale = params
    # Retrieve the shared memory block
    existing_shm = shared_memory.SharedMemory(name=shared_mem_name)

    # Assume each image will be stored as a flat array in shared memory
    # Example: using numpy for manipulation; adjust size and dtype as needed
    resolution_width, resolution_height = cam_info.image.size
    image_shape = (3, resolution_height, resolution_width)  # Set appropriate values
    dtype = np.uint8  # Adjust as per your image data type

    # Calculate the offset for this particular image
    offset = id * np.prod(image_shape)
    np_image_array = np.ndarray(
        image_shape, dtype=dtype, buffer=existing_shm.buf, offset=offset
    )

    # Decompress image into the numpy array directly
    decompressed_image = decompress_and_scale_image(cam_info)  # Implement this
    np_image_array[:] = decompressed_image

    # Clean up
    existing_shm.close()


def decompressed_images_from_camInfos_multiprocess_sharedmem(
    cam_infos, resolution_scale, args
):
    args = get_args()
    decompressed_images = []
    total_cameras = len(cam_infos)

    # Assume each image shape and dtype
    resolution_width, resolution_height = cam_infos[0].image.size
    image_shape = (
        3,
        resolution_height,
        resolution_width,
    )  # Define these as per your data
    dtype = np.uint8
    image_size = np.prod(image_shape) * np.dtype(dtype).itemsize

    # Create shared memory
    total_size = image_size * total_cameras
    shm = shared_memory.SharedMemory(create=True, size=total_size)

    # Create a pool of processes
    with multiprocessing.Pool(16) as pool:
        # Prepare data for processing
        tasks = [
            (shm.name, args, id, cam_info, resolution_scale)
            for id, cam_info in enumerate(cam_infos)
        ]

        # print("Start Parallel loading...")
        # Map load_camera_data to the tasks
        list(
            tqdm(pool.imap(load_decompressed_image_shared, tasks), total=total_cameras)
        )

    # Read images from shared memory
    decompressed_images = []
    for id in range(total_cameras):
        offset = id * np.prod(image_shape)
        np_image_array = np.ndarray(
            image_shape, dtype=dtype, buffer=shm.buf, offset=offset
        )
        decompressed_images.append(
            torch.from_numpy(np_image_array)
        )  # Make a copy if necessary

    # Clean up shared memory
    shm.close()
    shm.unlink()

    return decompressed_images


def _process_single_image_for_predecode(params):
    """Helper function to process a single image for predecoding (used by multiprocessing)"""
    cam_info, decode_dataset_path, orig_w, orig_h = params
    try:
        img = Image.open(cam_info.image_path)
        img = img.crop(
            (0, 0, orig_w, orig_h)
        )  # crop the image to the minimum size in dataset
        raw_data = img.tobytes()
        raw_data_path = os.path.join(
            decode_dataset_path,
            "dataset_raw",
            (cam_info.image_name.lstrip("/") + ".raw"),
        )
        os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
        with open(raw_data_path, "wb+") as raw_file:
            raw_file.write(raw_data)
        img.close()
        return True
    except Exception as e:
        print(f"Error processing {cam_info.image_path}: {e}")
        return False


def predecode_dataset_to_disk_multiprocess(cam_infos, args, num_workers=None):
    """
    Multiprocessing version of predecode_dataset_to_disk for faster processing.

    Args:
        cam_infos: List of camera info objects
        args: Arguments object
        num_workers: Number of worker processes. If None, uses min(cpu_count, 16)
    """
    args = get_args()
    orig_h, orig_w = utils.get_img_size()

    # Create tasks for multiprocessing
    tasks = [(c, args.decode_dataset_path, orig_w, orig_h) for c in cam_infos]

    # Use multiprocessing with reasonable number of workers
    # Cap at 16 to avoid too many open files and resource contention
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 16)

    print(f"Using {num_workers} worker processes for image predecoding")

    with multiprocessing.Pool(processes=num_workers) as pool:
        list(
            tqdm(
                pool.imap(_process_single_image_for_predecode, tasks),
                total=len(cam_infos),
                desc="Predecoding images to disk (multiprocess)",
            )
        )


def predecode_dataset_to_disk(cam_infos, args):
    """Original single-threaded version"""
    args = get_args()
    if args.multiprocesses_decode_dataset_to_disk:
        # Use multiprocessing to decode dataset to disk
        predecode_dataset_to_disk_multiprocess(cam_infos, args)
        return

    orig_h, orig_w = utils.get_img_size()
    for id, c in tqdm(enumerate(cam_infos), total=len(cam_infos)):
        img = Image.open(c.image_path)
        img = img.crop(
            (0, 0, orig_w, orig_h)
        )  # crop the image to the minimum size in dataset
        raw_data = img.tobytes()
        raw_data_path = os.path.join(
            args.decode_dataset_path, "dataset_raw", (c.image_name.lstrip("/") + ".raw")
        )
        os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
        with open(raw_data_path, "wb+") as raw_file:
            raw_file.write(raw_data)


def clean_up_disk(args):
    args = get_args()
    shutil.rmtree(os.path.join(args.decode_dataset_path, "dataset_raw"))


def cameraList_from_camInfos(cam_infos, args):
    args = get_args()

    if args.multiprocesses_image_loading:
        decompressed_images = decompressed_images_from_camInfos_multiprocess(
            cam_infos, args
        )
        # decompressed_images = decompressed_images_from_camInfos_multiprocess_sharedmem(cam_infos, resolution_scale, args)
    else:
        decompressed_images = [None for _ in cam_infos]

    camera_list = []
    for id, c in tqdm(enumerate(cam_infos), total=len(cam_infos)):
        camera_list.append(
            loadCam(
                args,
                id,
                c,
                decompressed_image=decompressed_images[id],
                return_image=False,
            )
        )

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }
    return camera_entry
