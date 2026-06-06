# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import cv2
import torch
import numpy as np

import os
import copy
import time
import inspect
import argparse
import importlib
from scipy.spatial.distance import cdist
from tqdm import tqdm

from .dwpose import util
from .dwpose.wholebody import Wholebody, HWC3, resize_image
from .utils import convert_to_numpy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def draw_pose(pose, H, W, use_hand=False, use_body=False, use_face=False):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if use_body:
        canvas = util.draw_bodypose(canvas, candidate, subset)
    if use_hand:
        canvas = util.draw_handpose(canvas, hands)
    if use_face:
        canvas = util.draw_facepose(canvas, faces)

    return canvas


class PoseAnnotator2:
    def __init__(self, cfg, device=None):
        onnx_det = cfg['DETECTION_MODEL']
        onnx_pose = cfg['POSE_MODEL']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.pose_estimation = Wholebody(onnx_det, onnx_pose, device=self.device)
        self.resize_size = cfg.get("RESIZE_SIZE", 1024)
        self.use_body = cfg.get('USE_BODY', True)
        self.use_face = cfg.get('USE_FACE', True)
        self.use_hand = cfg.get('USE_HAND', True)
        #print(f"device: {self.device}")

    @torch.no_grad()
    @torch.inference_mode
    def forward(self, image):
        image = convert_to_numpy(image)
        input_image = HWC3(image[..., ::-1])
        return self.process(resize_image(input_image, self.resize_size), image.shape[:2])

    def process(self, ori_img, ori_shape):
        ori_h, ori_w = ori_shape
        ori_img = ori_img.copy()
        H, W, C = ori_img.shape
        with torch.no_grad():
            candidate, subset, det_result = self.pose_estimation(ori_img)
            #print(f"candidate size: {candidate.shape}")
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:, :18].copy()
            body = body.reshape(nums * 18, locs)
            #print(f"body size: {body.shape}")
            score = subset[:, :18]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18 * i + j)
                    else:
                        score[i][j] = -1

            un_visible = subset < 0.3
            candidate[un_visible] = -1

            body_ = candidate[:, :18].copy()
            body_ = body_.reshape(nums * 18, locs)

            foot = candidate[:, 18:24]

            faces = candidate[:, 24:92]

            hands = candidate[:, 92:113]
            hands = np.vstack([hands, candidate[:, 113:]])

            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            ret_data = {}
            # if self.use_body:
            #     detected_map_body = draw_pose(pose, H, W, use_body=True)
            #     detected_map_body = cv2.resize(detected_map_body[..., ::-1], (ori_w, ori_h),
            #                                    interpolation=cv2.INTER_LANCZOS4 if ori_h * ori_w > H * W else cv2.INTER_AREA)
            #     ret_data["detected_map_body"] = detected_map_body

            # if self.use_face:
            #     detected_map_face = draw_pose(pose, H, W, use_face=True)
            #     detected_map_face = cv2.resize(detected_map_face[..., ::-1], (ori_w, ori_h),
            #                                    interpolation=cv2.INTER_LANCZOS4 if ori_h * ori_w > H * W else cv2.INTER_AREA)
            #     ret_data["detected_map_face"] = detected_map_face

            if self.use_body and self.use_face:
                detected_map_bodyface = draw_pose(pose, H, W, use_body=True, use_face=True)
                detected_map_bodyface = cv2.resize(detected_map_bodyface[..., ::-1], (ori_w, ori_h),
                                                   interpolation=cv2.INTER_LANCZOS4 if ori_h * ori_w > H * W else cv2.INTER_AREA)
                ret_data["detected_map_bodyface"] = detected_map_bodyface

            # if self.use_hand and self.use_body and self.use_face:
            #     detected_map_handbodyface = draw_pose(pose, H, W, use_hand=True, use_body=True, use_face=True)
            #     detected_map_handbodyface = cv2.resize(detected_map_handbodyface[..., ::-1], (ori_w, ori_h),
            #                                            interpolation=cv2.INTER_LANCZOS4 if ori_h * ori_w > H * W else cv2.INTER_AREA)
            #     ret_data["detected_map_handbodyface"] = detected_map_handbodyface

            # convert_size
            if det_result.shape[0] > 0:
                w_ratio, h_ratio = ori_w / W, ori_h / H
                det_result[..., ::2] *= h_ratio
                det_result[..., 1::2] *= w_ratio
                det_result = det_result.astype(np.int32)
            return ret_data, det_result, nums, body_


class PoseBodyFaceAnnotator2(PoseAnnotator2):
    def __init__(self, cfg, device=None):
        super().__init__(cfg, device)
        self.use_body, self.use_face, self.use_hand = True, True, False
    @torch.no_grad()
    @torch.inference_mode
    def forward(self, image):
        ret_data, det_result, nums, body = super().forward(image)
        return ret_data['detected_map_bodyface'], nums, body


def calculate_pose_similarity(keypoints1, keypoints2, threshold=0.3):
    """计算两个姿态关键点的相似度
    
    参数:
        keypoints1, keypoints2: 形状为(nums*18, 2)的数组，nums*18为关键点数量，每行为(x, y, 置信度)
        threshold: 置信度阈值，低于此值的关键点将被忽略
    
    返回:
        similarity: 姿态相似度得分（0-1之间，值越高表示越相似）
    """
    if keypoints1 is None or keypoints2 is None:
        return 0.0

    count_method1 = np.count_nonzero(keypoints1 == -1)
    #print(f"count 1: {count_method1}")
    nums1 = keypoints1.shape[0] * 2 
    
    count_method2 = np.count_nonzero(keypoints2 == -1)
    #print(f"count 2: {count_method2}")
    nums2 = keypoints2.shape[0] * 2 

    if count_method1 == nums1 and count_method2 == nums2: 
        return 0.0

    # # 过滤低置信度的关键点
    # valid1 = keypoints1[:, 2] > threshold
    # valid2 = keypoints2[:, 2] > threshold
    # valid_points = np.logical_and(valid1, valid2)
    
    # if not np.any(valid_points):
    #     return 0.0
    
    # 提取有效关键点的坐标
    points1 = keypoints1 #[valid_points, :2]
    points2 = keypoints2 #[valid_points, :2]
    
    # 归一化处理（减去重心并缩放）
    center1 = np.mean(points1, axis=0)
    center2 = np.mean(points2, axis=0)
    points1_norm = points1 - center1
    points2_norm = points2 - center2
    
    # 缩放至相同尺度
    scale1 = np.mean(np.linalg.norm(points1_norm, axis=1))
    scale2 = np.mean(np.linalg.norm(points2_norm, axis=1))
    
    if scale1 > 0:
        points1_norm /= scale1
    if scale2 > 0:
        points2_norm /= scale2
    
    # 计算关键点间的平均欧氏距离
    distances = cdist(points1_norm, points2_norm, 'euclidean')
    # 使用对角线距离（假设关键点顺序一致）
    avg_distance = np.mean(np.diag(distances))
    
    # 归一化距离到相似度得分
    max_possible_distance = np.sqrt(2)  # 归一化空间中两点的最大距离
    similarity = max(0, 1 - avg_distance / max_possible_distance)
    
    return similarity


class PoseBodyFaceVideoAnnotator2(PoseBodyFaceAnnotator2):
    def forward(self, frames):
        ret_frames = []
        prev_nums = None 
        prev_body = None 
        sim = []
        for frame in frames:
            anno_frame, nums, body = super().forward(np.array(frame))
            if len(ret_frames) != 0:
                if prev_nums == nums: 
                    sim_ = calculate_pose_similarity(body, prev_body)
                else:
                    sim_ = 0.0
            else: 
                sim_ = 0.0
            #print(f'nums: {nums}, sim_: {sim_}')
            sim.append(sim_)
            prev_nums = nums 
            prev_body = body
            ret_frames.append(anno_frame)
        return ret_frames, np.mean(sim)

class PoseBodyAnnotator2(PoseAnnotator2):
    def __init__(self, cfg, device=None):
        super().__init__(cfg, device)
        self.use_body, self.use_face, self.use_hand = True, False, False
    @torch.no_grad()
    @torch.inference_mode
    def forward(self, image):
        ret_data, det_result = super().forward(image)
        return ret_data['detected_map_body']


class PoseBodyVideoAnnotator2(PoseBodyAnnotator2):
    def forward(self, frames):
        ret_frames = []
        for frame in frames:
            anno_frame = super().forward(np.array(frame))
            ret_frames.append(anno_frame)
        return ret_frames


