"""
Text-guided segmentation code for SAVVY pipeline stage1 - part of "SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing" 
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Mingfei Chen, Zijun Cui and Xiulong Liu.
"""


import os
import cv2
import gc
import sys
import math
import torch
import json
import base64
import argparse
import numpy as np
import open3d as o3d
import random
import matplotlib
import matplotlib.cm
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
import shutil
import time

sys.path.append("../third_party/ZoeDepth")
sys.path.append("../third_party/efficientvit")

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from transformers import SamModel, SamProcessor

from transformers import AutoProcessor, CLIPSegForImageSegmentation



device = "cuda:0" if torch.cuda.is_available() else "cpu"
RUN_DEMO = True

root = "../data/det_aea_clipseg_sdv2_gemini25"
context_folder = "../data/sdv2_context_json"
video_root = "../data/videos"
if RUN_DEMO:
    root = "../data/det_aea_clipseg_sdv2_gemini25_demo"
    os.makedirs(root, exist_ok=True)
    context_folder = "../data/context_json/demo_sdv2"
    video_root = "../data/demo_videos"



conf = get_config("zoedepth", "infer")
depth_model = build_model(conf)
clipseg_processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

# import pdb; pdb.set_trace()

def depth(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    depth_model.to(device)
    depth = depth_model.infer_pil(img)
    raw_depth = Image.fromarray((depth*256).astype('uint16'))
    return raw_depth



def find_medoid_and_closest_points(points, num_closest=5):
    """
    Find the medoid from a collection of points and the closest points to the medoid.

    Parameters:
    points (np.array): A numpy array of shape (N, D) where N is the number of points and D is the dimensionality.
    num_closest (int): Number of closest points to return.

    Returns:
    np.array: The medoid point.
    np.array: The closest points to the medoid.
    """
    distances = np.sqrt(((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2).sum(axis=-1))
    distance_sums = distances.sum(axis=1)
    medoid_idx = np.argmin(distance_sums)
    medoid = points[medoid_idx]
    sorted_indices = np.argsort(distances[medoid_idx])
    closest_indices = sorted_indices[1:num_closest + 1]
    return medoid, points[closest_indices]


def sample_points_from_heatmap(heatmap, height, width, original_size, num_points=5, percentile=0.95):
    """
    Sample points from the given heatmap, focusing on areas with higher values.
    """
    threshold = np.percentile(heatmap.numpy(), percentile)
    masked_heatmap = torch.where(heatmap > threshold, heatmap, torch.tensor(0.0))
    probabilities = torch.softmax(masked_heatmap.flatten(), dim=0)

    attn = torch.sigmoid(heatmap)
    w = attn.shape[0]
    sampled_indices = torch.multinomial(torch.tensor(probabilities.ravel()), num_points, replacement=True)

    sampled_coords = np.array(np.unravel_index(sampled_indices, attn.shape)).T
    medoid, sampled_coords = find_medoid_and_closest_points(sampled_coords)
    pts = []
    attention_scores = []
    for pt in sampled_coords.tolist():
        x, y = pt
        attention_score = attn[x, y].item()
        attention_scores.append(attention_score)
        x = height * x / w
        y = width * y / w
        pts.append([y, x])
    return pts, attention_scores


### visualization
def apply_mask_to_image(image, mask):
    """
    Apply a binary mask to an image. The mask should be a binary array where the regions to keep are True.
    """
    masked_image = image.copy()
    for c in range(masked_image.shape[2]):
        masked_image[:, :, c] = masked_image[:, :, c] * mask
    return masked_image
###


step1_time_list = []
step2_time_list = []
read_time_list = []
write_time_list = []
depth_time_list = []
def get_det(total_frames, context_data, det_json_path="detection.json", n_samples = 128):
    if "sounding_object" in context_data and "description" in context_data["sounding_object"]:
        sounding_object_text = [context_data["sounding_object"]["description"]]
    else:
        sounding_object_text = []
    static_object_text = []
    if "exo" in context_data["qid"]:
        if "reference_object" in context_data and "description" in context_data["reference_object"]:
            static_object_text.append(context_data["reference_object"]["description"])
        if "distance" not in context_data["qid"]:
            if "facing_object" in context_data and "description" in context_data["facing_object"]:
                static_object_text.append(context_data["facing_object"]["description"])
        return
    full_object_text = sounding_object_text.copy() + static_object_text.copy()

    frames_dir = os.path.join("data/det_aea_clipseg_frames", context_data["video_id"], f"rgbd_frames_uni")
    os.makedirs(frames_dir, exist_ok=True)
    
    outjson_path = det_json_path

    start_time = time.time()
    # step1: extract key frames
    extracted_frames = []
    frame_indices = np.linspace(0, total_frames-1, n_samples, dtype=int)        
    for idx, frame_idx in enumerate(frame_indices):
        img_path = os.path.join(frames_dir, f"{frame_idx:05d}_rgb.jpg")
        depth_path = os.path.join(frames_dir, os.path.basename(img_path).replace('_rgb.jpg', '_depth.png'))
        
        if True:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                write_time_s = time.time()
                img.save(img_path)
                write_time_list.append(time.time()-write_time_s)
                
        
        if not os.path.exists(img_path):
            continue
        # Step 2: Depth estimation
        original_image = Image.open(img_path)
        depth_time_s = time.time()
        depth_image = depth(original_image)
        torch.cuda.synchronize()
        depth_time_cur = time.time() - depth_time_s
        depth_time_list.append(depth_time_cur)
        depth_image.save(depth_path)
            
        # # Visualize depth map
        # depth_viz = np.array(depth_image) /  np.array(depth_image).max()
        # colormap = plt.get_cmap('inferno')
        # depth_color = colormap(depth_viz)[:, :, :3] * 256
        # depth_color = depth_color.astype(np.uint8)
        # depth_color_img = Image.fromarray(depth_color)
        # depth_viz_path = os.path.join(frames_dir, os.path.basename(img_path).replace('_rgb.jpg', '_depth_viz.jpg'))
        # depth_color_img.save(depth_viz_path)

        extracted_frames.append(img_path)
    video.release()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    step1_time = time.time()-start_time
    step1_time_list.append(step1_time)
    print("step1-2: ", np.mean(step1_time_list))
    print("write time: ", np.mean(write_time_list))
    print("depth time: ", np.mean(depth_time_list))
    
    # Step 3 : Clipseg
    start_time2 = time.time()
    detect_dict = {}
    for img_path in sorted(extracted_frames):
        frame_idx = os.path.basename(img_path).split("_")[0]
        text_descriptions = full_object_text
        if len(text_descriptions) == 0:
            continue
        read_time_s = time.time()
        original_image = Image.open(img_path)
        read_time_list.append(time.time()-read_time_s)
        

        width, height = original_image.size
        inputs = clipseg_processor(text=text_descriptions, images=[original_image] * len(text_descriptions), padding=True, return_tensors="pt", truncation = True)
        for key in inputs:
            inputs[key] = inputs[key].to(clipseg_model.device)
        outputs = clipseg_model(**inputs)
        logits = outputs.logits
        preds = logits.detach().unsqueeze(1)

        sampled_points = []
        original_image_cv = cv2.imread(img_path)
        original_image_cv = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2RGB)
        original_size = original_image_cv.shape[:2][::-1]
        all_scores = []
        for idx in range(preds.shape[0]):
            cur_sampled_pointset, scores = sample_points_from_heatmap(preds[idx][0].data.cpu(), height, width, original_size, num_points=10)
            all_scores.append(scores)
            if np.max(scores) > 0.5:
                sampled_points.append(cur_sampled_pointset)
            else:
                sampled_points.append([])

        sam_masks = []
        all_sam_scores = []
        for idx in range(preds.shape[0]):
            if len(sampled_points[idx]) > 0:
                sam_inputs = sam_processor(original_image, input_points=[sampled_points[idx]], return_tensors="pt").to(device)
                with torch.no_grad():
                    sam_outputs = sam_model(**sam_inputs)

                sam_masks.append(sam_processor.image_processor.post_process_masks(
                    sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"].cpu(), sam_inputs["reshaped_input_sizes"].cpu()
                    ))
                all_sam_scores.append(sam_outputs.iou_scores.data.cpu().numpy().reshape(-1).tolist())
            else:
                sam_masks.append([])
                all_sam_scores.append([0.0])
        
        # vis seg on rgb and depth
        # Step 4: Get segmented pointcloud
        depth_path = os.path.join(frames_dir, os.path.basename(img_path).replace('_rgb.jpg', '_depth.png'))
        depth_image_cv = cv2.imread(depth_path)
        for i, mask_tensor in enumerate(sam_masks):
            if len(mask_tensor) == 0:
                continue
            mask = cv2.cvtColor(255 * mask_tensor[0].numpy().squeeze().transpose((1, 2, 0)).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            mask_binary = mask > 0

            masked_depth = apply_mask_to_image(depth_image_cv, mask_binary)

            y_coords, x_coords = np.where(mask_binary)
            if len(x_coords) > 0 and len(y_coords) > 0:
                center_x = np.mean(x_coords)
                
                # Calculate direction relative to image center
                img_center_x = mask_binary.shape[1] / 2
                
                # Calculate horizontal offset from center (positive = right, negative = left)
                x_offset = center_x - img_center_x
                
                # Convert to angle (-90 to 90 degrees)
                # Normalize x_offset to range [-1, 1] based on half image width
                normalized_x = x_offset / (mask_binary.shape[1] / 2)
                
                # Clamp to [-1, 1] range to avoid values outside valid range
                normalized_x = max(min(normalized_x, 1.0), -1.0)
    
                # Map to angle: -1 -> -90°, 0 -> 0°, 1 -> 90°
                angle = normalized_x * 90
            
            
            detect_dict.setdefault(frame_idx, {})
            cur_info = {
                "direction": round(angle, 2),
                "distance": round(masked_depth[masked_depth>0].mean(), 2),
                "det_conf": all_scores[i],
            }
            detect_dict[frame_idx][text_descriptions[i]] = cur_info
                 
            json.dump(detect_dict, open(outjson_path, "w"), indent=4)


    torch.cuda.synchronize()
    step2_time = time.time()-start_time2
    step2_time_list.append(step2_time)
    print("step2: ", np.mean(step2_time_list))
    print("read: ", np.mean(read_time_list))

if __name__ == "__main__":
    text_dict = {}
    valid_list = []
    total_frames_dict = {}
    for contextfile in tqdm(os.listdir(context_folder)):
        context_json = os.path.join(context_folder, contextfile)
        if not contextfile.endswith(".json"):
            continue
        with open(context_json, "r") as fin:
            context_data = json.load(fin)
        if not isinstance(context_data, dict):
            continue
        video_id = context_data["video_id"]
        qid = context_data["qid"]
        det_json_path = f"{root}/{qid}/detection.json"

        if not os.path.exists(det_json_path) or not os.path.exists(f"{root}/{qid}/{contextfile}"):
            valid_list.append(contextfile)
        if video_id not in total_frames_dict:
            video_path = os.path.join(video_root, video_id+".mp4")
            video = cv2.VideoCapture(video_path)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames_dict[video_id] = total_frames

    random.shuffle(valid_list)
    for contextfile in tqdm(valid_list):
        context_json = os.path.join(context_folder, contextfile)
        context_data = json.load(open(context_json, "r"))
        video_id = context_data["video_id"]
        qid = context_data["qid"]
        os.makedirs(f"{root}/{qid}", exist_ok=True)
        det_json_path = f"{root}/{qid}/detection.json"
        get_det(total_frames_dict[video_id], context_data, det_json_path)
        shutil.copy(context_json, f"{root}/{qid}/{contextfile}")