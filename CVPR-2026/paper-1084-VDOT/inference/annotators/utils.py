# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import io
import os
import tempfile
import torch
import numpy as np
import cv2
import imageio
from PIL import Image
import pycocotools.mask as mask_utils

import subprocess

def single_mask_to_rle(mask):
    rle = mask_utils.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def single_rle_to_mask(rle):
    mask = np.array(mask_utils.decode(rle)).astype(np.uint8)
    return mask

def single_mask_to_xyxy(mask):
    bbox = np.zeros((4), dtype=int)
    rows, cols = np.where(np.array(mask))
    if len(rows) > 0 and len(cols) > 0:
        x_min, x_max = np.min(cols), np.max(cols)
        y_min, y_max = np.min(rows), np.max(rows)
        bbox[:] = [x_min, y_min, x_max, y_max]
    return bbox.tolist()

def get_mask_box(mask, threshold=255):
    locs = np.where(mask >= threshold)
    if len(locs) < 1 or locs[0].shape[0] < 1 or locs[1].shape[0] < 1:
        return None
    left, right = np.min(locs[1]), np.max(locs[1])
    top, bottom = np.min(locs[0]), np.max(locs[0])
    return [left, top, right, bottom]

def convert_to_numpy(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    elif isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    elif isinstance(image, np.ndarray):
        image = image.copy()
    else:
        raise f'Unsurpport datatype{type(image)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'
    return image

def convert_to_pil(image):
    if isinstance(image, Image.Image):
        image = image.copy()
    elif isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        image = Image.fromarray(image.astype('uint8'))
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    else:
        raise TypeError(f'Unsupported data type {type(image)}, only supports np.ndarray, torch.Tensor, Pillow Image.')
    return image

def convert_to_torch(image):
    if isinstance(image, Image.Image):
        image = torch.from_numpy(np.array(image)).float()
    elif isinstance(image, torch.Tensor):
        image = image.clone()
    elif isinstance(image, np.ndarray):
        image = torch.from_numpy(image.copy()).float()
    else:
        raise f'Unsurpport datatype{type(image)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'
    return image

def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(
        input_image, (W, H),
        interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img, k


def resize_image_ori(h, w, image, k):
    img = cv2.resize(
        image, (w, h),
        interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


def save_one_video(file_path, videos, fps=8, quality=8, macro_block_size=None):
    try:
        video_writer = imageio.get_writer(file_path, fps=fps, codec='libx264', quality=quality, macro_block_size=macro_block_size)
        for frame in videos:
            video_writer.append_data(frame)
        video_writer.close()
        return True
    except Exception as e:
        print(f"Video save error: {e}")
        return False


def save_one_video_s3(file_path, videos, fps=8, quality=8, macro_block_size=None):

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        video_writer = imageio.get_writer(tmp_path, fps=fps, codec='libx264', quality=quality, macro_block_size=macro_block_size)
        for frame in videos:
            video_writer.append_data(frame)
        video_writer.close()

        # transport tmp_path video to s3 path 
        result = subprocess.run(
            ["aws", "s3", 
                "cp",
                "--endpoint-url=http://d-ceph-ssd-inside.pjlab.org.cn:80",
                tmp_path,
                file_path], stderr=None,
            stdout=None)

        return True
    except Exception as e:
        print(f"Video save error: {e}")
        return False


def save_one_image(file_path, image, use_type='cv2'):
    try:
        if use_type == 'cv2':
            cv2.imwrite(file_path, image)
        elif use_type == 'pil':
            image = Image.fromarray(image)
            image.save(file_path)
        else:
            raise ValueError(f"Unknown image write type '{use_type}'")
        return True
    except Exception as e:
        print(f"Image save error: {e}")
        return False

def read_image(image_path, use_type='cv2', is_rgb=True, info=False):
    image = None
    width, height = None, None

    if use_type == 'cv2':
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("Image not found or path is incorrect.")
            if is_rgb:
                image = image[..., ::-1]
            height, width = image.shape[:2]
        except Exception as e:
            print(f"OpenCV read error: {e}")
            return None
    elif use_type == 'pil':
        try:
            image = Image.open(image_path)
            if is_rgb:
                image = image.convert('RGB')
            width, height = image.size
            image = np.array(image)
        except Exception as e:
            print(f"PIL read error: {e}")
            return None
    else:
        raise ValueError(f"Unknown image read type '{use_type}'")

    if info:
        return image, width, height
    else:
        return image


def read_mask(mask_path, use_type='cv2', info=False):
    mask = None
    width, height = None, None

    if use_type == 'cv2':
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise Exception("Mask not found or path is incorrect.")
            height, width = mask.shape
        except Exception as e:
            print(f"OpenCV read error: {e}")
            return None
    elif use_type == 'pil':
        try:
            mask = Image.open(mask_path).convert('L')
            width, height = mask.size
            mask = np.array(mask)
        except Exception as e:
            print(f"PIL read error: {e}")
            return None
    else:
        raise ValueError(f"Unknown mask read type '{use_type}'")

    if info:
        return mask, width, height
    else:
        return mask

def read_video_frames(video_path, use_type='cv2', is_rgb=True, info=False):
    frames = []
    if use_type == "decord":
        import decord
        decord.bridge.set_bridge("native")
        try:
            cap = decord.VideoReader(video_path)
            total_frames = len(cap)
            fps = cap.get_avg_fps()
            height, width, _ = cap[0].shape
            frames = [cap[i].asnumpy() for i in range(len(cap))]
        except Exception as e:
            print(f"Decord read error: {e}")
            return None
    elif use_type == "cv2":
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if is_rgb:
                    frames.append(frame[..., ::-1])
                else:
                    frames.append(frame)
            cap.release()
            total_frames = len(frames)
        except Exception as e:
            print(f"OpenCV read error: {e}")
            return None
    else:
        raise ValueError(f"Unknown video type {use_type}")
    if info:
        return frames, fps, width, height, total_frames
    else:
        return frames



def read_video_one_frame(video_path, use_type='cv2', is_rgb=True):
    image_first = None
    if use_type == "decord":
        import decord
        decord.bridge.set_bridge("native")
        try:
            cap = decord.VideoReader(video_path)
            image_first = cap[0].asnumpy()
        except Exception as e:
            print(f"Decord read error: {e}")
            return None
    elif use_type == "cv2":
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if is_rgb:
                image_first = frame[..., ::-1]
            else:
                image_first = frame
            cap.release()
        except Exception as e:
            print(f"OpenCV read error: {e}")
            return None
    else:
        raise ValueError(f"Unknown video type {use_type}")
    return image_first


def read_video_last_frame(video_path, use_type='cv2', is_rgb=True):
    image_last = None
    if use_type == "decord":
        import decord
        decord.bridge.set_bridge("native")
        try:
            cap = decord.VideoReader(video_path)
            if len(cap) > 0:  # Check if video has at least one frame
                image_last = cap[-1].asnumpy()  # Get last frame using negative index
        except Exception as e:
            print(f"Decord read error: {e}")
            return None
    elif use_type == "cv2":
        try:
            cap = cv2.VideoCapture(video_path)
            # Get total frame count
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                # Set position to last frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                ret, frame = cap.read()
                if ret:  # Check if frame was read successfully
                    if is_rgb:
                        image_last = frame[..., ::-1]
                    else:
                        image_last = frame
            cap.release()
        except Exception as e:
            print(f"OpenCV read error: {e}")
            return None
    else:
        raise ValueError(f"Unknown video type {use_type}")
    return image_last


def align_frames(first_frame, last_frame):
    h1, w1 = first_frame.shape[:2]
    h2, w2 = last_frame.shape[:2]
    if (h1, w1) == (h2, w2):
        return last_frame
    ratio = min(w1 / w2, h1 / h2)
    new_w = int(w2 * ratio)
    new_h = int(h2 * ratio)
    resized = cv2.resize(last_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    aligned = np.ones((h1, w1, 3), dtype=np.uint8) * 255
    x_offset = (w1 - new_w) // 2
    y_offset = (h1 - new_h) // 2
    aligned[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return aligned


def save_sam2_video(video_path, video_segments, output_video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    obj_mask_map = {}
    for frame_idx, segments in video_segments.items():
        for obj_id, info in segments.items():
            seg = single_rle_to_mask(info['mask'])[None, ...].squeeze(0).astype(bool)
            if obj_id not in obj_mask_map:
                obj_mask_map[obj_id] = [seg]
            else:
                obj_mask_map[obj_id].append(seg)

    for obj_id, segs in obj_mask_map.items():
        output_obj_video_path = os.path.join(output_video_path, f"{obj_id}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for saving the video
        video_writer = cv2.VideoWriter(output_obj_video_path, fourcc, fps, (width * 2, height))

        for i, (frame, seg) in enumerate(zip(frames, segs)):
            print(obj_id, i, np.sum(seg), seg.shape)
            left_frame = frame.copy()
            left_frame[seg] = 0
            right_frame = frame.copy()
            right_frame[~seg] = 255
            frame_new = np.concatenate([left_frame, right_frame], axis=1)
            video_writer.write(frame_new)
        video_writer.release()


def get_annotator_instance(anno_cfg):
    import vace.annotators as annotators
    anno_cfg = copy.deepcopy(anno_cfg)
    class_name = anno_cfg.pop("NAME")
    input_params = anno_cfg.pop("INPUTS")
    output_params = anno_cfg.pop("OUTPUTS")
    anno_ins = getattr(annotators, class_name)(cfg=anno_cfg)
    return {"inputs": input_params, "outputs": output_params, "anno_ins": anno_ins}

def get_annotator(config_type='', config_task='', return_dict=True):
    anno_dict = None
    from vace.configs import VACE_CONFIGS
    if config_type in VACE_CONFIGS:
        task_configs = VACE_CONFIGS[config_type]
        if config_task in task_configs:
            anno_dict = get_annotator_instance(task_configs[config_task])
        else:
            raise ValueError(f"Unknown config task {config_task}")
    else:
        for cfg_type, cfg_dict in VACE_CONFIGS.items():
            if config_task in cfg_dict:
                for task_name, task_cfg in cfg_dict[config_task].items():
                    anno_dict = get_annotator_instance(task_cfg)
            else:
                raise ValueError(f"Unknown config type {config_type}")
    if return_dict:
        return anno_dict
    else:
        return anno_dict['anno_ins']

