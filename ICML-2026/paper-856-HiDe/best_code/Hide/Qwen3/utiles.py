import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from scipy.ndimage import zoom
import os
import pandas as pd
import pyarrow.parquet as pq
import ast
import json
from typing import List, Dict
from itertools import combinations
import base64
from PIL import Image
from io import BytesIO
import io
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter
from scipy.ndimage import median_filter
import torch

def process(dicts, start_k, end_k, attention, inputs, img_start, img_end, sig):
    accept_att = {}
    noise_token_num = 8
    noise_mean = [[0 for k in range(noise_token_num)] for i in range(len(inputs["image_grid_thw"]))]
    for k in range(start_k,end_k-4):
        max_att_sum = 0
        per_img_attention = []
        for img_idx in range(len(inputs["image_grid_thw"])):
            image_grid_thw = inputs["image_grid_thw"][img_idx]
            start = img_start[img_idx]
            end = img_end[img_idx]
            if start_k < end:
                start_k = end+1
            layer_sum = []
            layer_mean = []
            for i in range(len(attention)):
                k_att_map = np.array([row[k] for row in attention[i][0]])
                att_map = k_att_map[:,start:end].reshape(-1, image_grid_thw[1]//2,image_grid_thw[2]//2).mean(axis=0)
                layer_mean.append(att_map)
            per_img_attention.append(np.array(layer_mean).mean(axis=0,keepdims=True))
        max_att_get = 0
        for i in range(len(per_img_attention)):
            sum_per_img_att = per_img_attention[i].max()
            if sum_per_img_att > max_att_get:
                max_att_get = sum_per_img_att
                img_idx = i
            if k < start_k+noise_token_num:
                per_att = per_img_attention[i]
                # per_att = np.array(maxpooling(torch.from_numpy(per_att)))
                if sig > 0:
                    per_att = gaussian_filter(per_att, sigma=sig)
                per_att = per_att - per_att.min()
                per_att = per_att / per_att.max()
                noise_mean[i][k-start_k] = per_att
        if k < start_k+noise_token_num: continue
        if not img_idx in accept_att:
            accept_att[img_idx] = {}
        # accept_att[img_idx][k] = per_img_attention[img_idx]
        accept_s = per_img_attention[img_idx]
        if sig > 0:
            accept_s = gaussian_filter(accept_s, sigma=sig)
        accept_s = accept_s - accept_s.min()
        accept_s = accept_s / accept_s.max()
        if noise_token_num > 0:
            # plt.imshow(accept_s[0])
            # plt.savefig(f'{img_idx}_{k}_{dicts[inputs["input_ids"][0][k].cpu().item()].replace(r"/",r"[]")}.png')
            # plt.close()
            accept_s -= np.array(noise_mean[img_idx]).mean(axis=0)
            # plt.imshow(accept_s[0])
            # plt.savefig(f'{img_idx}_{k}_{dicts[inputs["input_ids"][0][k].cpu().item()].replace(r"/",r"[]")}_clearn.png')
            # plt.close()
            accept_s[accept_s<0] = 0
        if accept_s.max() <= 0: continue
        accept_s = accept_s - accept_s.min()
        accept_s = accept_s / accept_s.max()
        accept_att[img_idx][k]=accept_s
    return accept_att

def create_directory(path):
    """
    创建给定路径的目录，包括所有必要的父目录。

    :param path: 完整的目录路径字符串
    """
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Directory created successfully at {path}")
    except Exception as e:
        print(f"Failed to create directory at {path}: {e}")

def load_json_to_list(json_path: str) -> List[Dict]:
    """
    加载 JSON 文件并返回一个由字典组成的列表
    
    参数:
        json_path (str): JSON 文件路径
    
    返回:
        List[Dict]: 列表中的每个元素都是一个字典
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON 文件内容不是一个列表")

    return data

def serialize_dict(my_dict, file_path):
    """
    将一个字典序列化为一行 JSON，追加写入到 .jsonl 文件。
    
    每次调用写入一行，不换行嵌套，符合 JSONL 标准。
    
    参数:
        my_dict: 要写入的字典（可能包含 ndarray、np.int64 等）
        file_path: 输出的 .jsonl 文件路径
    """
    def serialize_obj(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: serialize_obj(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [serialize_obj(item) for item in obj]
        else:
            return obj

    # 序列化整个字典
    serialized_dict = serialize_obj(my_dict)

    # 追加写入一行 JSON
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(serialized_dict, ensure_ascii=False, indent=4) + '\n')

def image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_str = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image;base64,{encoded_str}"

def pil_to_base64(pil_img, format="PNG"):
    buffered = BytesIO()
    # 如果 pil_img.format 不存在，使用指定的默认格式
    img_format = pil_img.format if pil_img.format else format
    pil_img.save(buffered, format=img_format)  # 使用指定格式保存图像到内存
    encoded_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image;base64,{encoded_str}"

def swap_and_rebuild_dict(nested_dict):
    """
    将两层嵌套字典的内外层 key 对调。
    
    输入:
        nested_dict: 形如 {outer_key: {inner_key: value}}
    输出:
        new_dict: 形如 {inner_key: {outer_key: value}}
    """
    new_dict = {}

    for outer_key, inner_dict in nested_dict.items():
        for inner_key, value in inner_dict.items():
            if inner_key not in new_dict:
                new_dict[inner_key] = {}
            new_dict[inner_key][outer_key] = value
            
    return dict(sorted(new_dict.items()))

def detect_concentrated_regions_with_merge(matrix, k=3, merge_distance_ratio=0.1):
    """
    自动检测集中区域，并合并距离较近的区域。
    如果一个小区域被一个大区域完全包含，则只保留外层的大区域。
    
    参数:
        matrix (np.ndarray): NxN 受力矩阵
        k (float): 控制灵敏度的倍数，默认为 2
        merge_distance_ratio (float): 合并距离阈值（相对于图像对角线的比例）
    
    返回:
        List[List[int]]: 每个元素是一个 bounding box [x1, y1, x2, y2]
    """
    H, W = matrix.shape
    diag_length = np.sqrt(H**2 + W**2)
    merge_distance_threshold = diag_length * merge_distance_ratio  # 转换为实际像素距离

    # Step 1: 原有方法提取所有原始区域
    mean = np.mean(matrix)
    std = np.std(matrix)
    threshold = mean + k * std
    binary = matrix > threshold
    labeled_matrix, num_features = ndimage.label(binary)

    regions = []
    for label_id in range(1, num_features + 1):
        coords = np.column_stack(np.where(labeled_matrix == label_id))
        regions.append(coords)

    if not regions:
        return []

    # Step 2: 获取每个区域的 bounding box
    boxes = []
    for coords in regions:
        y_min, x_min = np.min(coords, axis=0)
        y_max, x_max = np.max(coords, axis=0)
        boxes.append([x_min, y_min, x_max, y_max])  # [x1,y1,x2,y2]

    # Step 3: 构建区域之间的距离图
    n = len(boxes)
    to_merge = []

    def box_center(box):
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    # 判断哪些框可以合并
    for i, j in combinations(range(n), 2):
        c1 = box_center(boxes[i])
        c2 = box_center(boxes[j])
        dist = np.linalg.norm(c1 - c2)
        if dist < merge_distance_threshold:
            to_merge.append((i, j))


    # Step 4: 合并逻辑（使用并查集）
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px

    for i, j in to_merge:
        union(i, j)

    # Step 5: 收集合并后的区域
    merged_boxes = {}
    for i in range(n):
        root = find(i)
        if root not in merged_boxes:
            merged_boxes[root] = boxes[i]
        else:
            x1 = min(merged_boxes[root][0], boxes[i][0])
            y1 = min(merged_boxes[root][1], boxes[i][1])
            x2 = max(merged_boxes[root][2], boxes[i][2])
            y2 = max(merged_boxes[root][3], boxes[i][3])
            merged_boxes[root] = [x1, y1, x2, y2]

    # Step 6: 转换为 list 格式返回
    final_boxes = [list(map(int, box)) for box in merged_boxes.values()]

    # Step 7: 去除被完全包含的小区域
    def remove_nested_boxes(boxes):
        if not boxes:
            return []

        # 按面积从大到小排序
        def area(box):
            return (box[2] - box[0]) * (box[3] - box[1])

        boxes_sorted = sorted(boxes, key=area, reverse=True)
        result = []

        for current in boxes_sorted:
            x1, y1, x2, y2 = current
            contained = False
            for other in result:
                ox1, oy1, ox2, oy2 = other
                if ox1 <= x1 and oy1 <= y1 and ox2 >= x2 and oy2 >= y2:
                    contained = True
                    break
            if not contained:
                result.append(current)

        return result
    H, W = matrix.shape
    final_boxes = remove_nested_boxes(final_boxes)
    return final_boxes

def load_dataset_Vstar_json(path):
    Vstar_list = []
    with open(path, 'r', encoding='utf-8') as f:
        Vstar_list = json.load(f)
    mmetype_Vstarbench = []
    for i in range(len(Vstar_list)):
        # if Vstar_list[i]["category"] == "direct_attributes": continue
        dict_i = {}
        dict_i["id"] = Vstar_list[i]["id"]
        dict_i["Text"] = Vstar_list[i]["question"].replace("\nAnswer with the option's letter from the given choices directly.","")
        # dict_i["Choices"] = "\n".join(Vstar_list[i]["text"].split("\n")[1:-1])
        dict_i["Ground truth"] = Vstar_list[i]["labels"]
        dict_i["image"] = Vstar_list[i]["image_path"]
        if "box_json" in Vstar_list[i]:
            dict_i["box_json"] = Vstar_list[i]["box_json"]
        dict_i["category"] = Vstar_list[i]["category"]
        mmetype_Vstarbench.append(dict_i)
    return mmetype_Vstarbench

def load_dataset_hrbench_json(path):
    Vstar_list = []
    with open(path, 'r', encoding='utf-8') as f:
        Vstar_list = json.load(f)
    mmetype_Vstarbench = []
    for i in range(len(Vstar_list)):
        dict_i = {}
        dict_i["id"] = Vstar_list[i]["id"]
        dict_i["Text"] = Vstar_list[i]["question"]
        # dict_i["Text"] = Vstar_list[i]["question"] + "\nAnswer with the option's letter from the given choices directly."
        # dict_i["Choices"] = "\n".join(Vstar_list[i]["text"].split("\n")[1:-1])
        dict_i["Ground truth"] = Vstar_list[i]["labels"]
        dict_i["image"] = Vstar_list[i]["image_path"]
        dict_i["Category"] = Vstar_list[i]["Category"]
        mmetype_Vstarbench.append(dict_i)
    return mmetype_Vstarbench

def load_dataset_hrbench(path):
    hrbench = pd.read_csv(path, sep='\t')
    mmetype_hrbench = []
    for i in hrbench.index:
        # if str(hrbench["cycle_category"][i]) != "0": continue
        dict_i = {}
        dict_i["Text"] = hrbench["question"][i] + "\n(A) "+hrbench["A"][i] + "\n(B) "+hrbench["B"][i]+"\n(C) "+hrbench["C"][i]+ "\n(D) "+hrbench["D"][i]
        # dict_i["Answer choices"] = ["(A) "+hrbench["A"][i], "(B) "+hrbench["B"][i], "(C) "+hrbench["C"][i], "(D) "+hrbench["D"][i]]
        dict_i["Ground truth"] = hrbench["answer"][i]
        dict_i["image"] = r"data:image;base64,"+hrbench["image"][i]
        dict_i["Category"] = hrbench["category"][i]
        dict_i["cycle_category"] = str(hrbench["cycle_category"][i])
        mmetype_hrbench.append(dict_i)
    return mmetype_hrbench
