import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import zoom
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from utiles import *
import json
import torch.multiprocessing as mp
import multiprocessing
from joblib import Parallel, delayed
import time
import random
from PIL import Image
import io
import numpy as np
import base64
import gc
import base64
import multiprocessing
from multiprocessing import Pool
import cv2
import numpy as np
import base64
from io import BytesIO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_filter

from skimage.filters import threshold_otsu

def get_inputs(messages,processor,model):
    text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages,return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs
    )
    inputs = inputs.to(model.device)
    return text,image_inputs,video_inputs,inputs,video_kwargs

def messages2out(model,processor,inputs):
    inputs = inputs.to(model.device)
    end_ques = len(inputs['input_ids'][0])

    with torch.no_grad():
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    del inputs,generated_ids
    torch.cuda.empty_cache()
    return output_text,end_ques
    
def messages2att(model,processor,inputs):
    inputs = inputs.to(model.device)
    end_ques = len(inputs['input_ids'][0])
    img_start = []
    img_end = []
    idx2word_dicts = {}
    need_2_att_w = []
    for i in range(len(inputs['input_ids'][0])):
        words = processor.post_process_image_text_to_text(
        torch.tensor([inputs['input_ids'][0][i]]), skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        idx2word_dicts[inputs['input_ids'][0][i].cpu().item()] = words
        if inputs['input_ids'][0][i].cpu().item() == 151652:
            img_start.append(i+1)
        if inputs['input_ids'][0][i].cpu().item() == 151653:
            img_end.append(i)
    for i in range(len(inputs['input_ids'][0])):
        if i>img_end[-1]:
            need_2_att_w.append(i)
    # print(len(need_2_att_w))
    with torch.no_grad():
        out = model(**inputs, output_attentions=True,target_indices=torch.tensor(need_2_att_w))  # logits,past_key_values,Attention
    # del inputs
    # torch.cuda.empty_cache()
    attention = []
    for i in range(len(out['attentions'])):
        if out['attentions'][i] is None:
            continue
        attention.append(out['attentions'][i])
    del inputs,out
    torch.cuda.empty_cache()
    return attention,idx2word_dicts,img_start,img_end


def place_on_center(canvas_bgra, content_bgra):
    """一个辅助函数，将 content 图像(BGRA)居中放置在 canvas 画布(BGRA)上"""
    canvas_h, canvas_w, _ = canvas_bgra.shape
    content_h, content_w, _ = content_bgra.shape

    if content_h > canvas_h or content_w > canvas_w:
        scale = min(canvas_h / content_h, canvas_w / content_w)
        new_h, new_w = int(content_h * scale), int(content_w * scale)
        content_bgra = cv2.resize(content_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)
        content_h, content_w = new_h, new_w

    paste_x = (canvas_w - content_w) // 2
    paste_y = (canvas_h - content_h) // 2
    
    # 使用Alpha通道作为蒙版来粘贴
    alpha_mask = content_bgra[:, :, 3] / 255.0
    
    # 遍历每个颜色通道
    for c in range(0, 3):
        canvas_bgra[paste_y:paste_y+content_h, paste_x:paste_x+content_w, c] = \
            alpha_mask * content_bgra[:, :, c] + \
            (1 - alpha_mask) * canvas_bgra[paste_y:paste_y+content_h, paste_x:paste_x+content_w, c]
            
    # 更新画布的alpha通道
    canvas_bgra[paste_y:paste_y+content_h, paste_x:paste_x+content_w, 3] = \
        np.maximum(canvas_bgra[paste_y:paste_y+content_h, paste_x:paste_x+content_w, 3], content_bgra[:, :, 3])
        
    return canvas_bgra

def decompose_bbox_by_alpha(image_bgra, bbox, alpha_threshold=10):
    """
    将单个BBox根据其Alpha通道分解为多个不包含透明区域的子BBox。

    Args:
        image_bgra (np.array): 4通道的BGRA格式图像。
        bbox (list or tuple): 单个边界框 [x0, y0, x1, y1]。
        alpha_threshold (int): 用于判断像素是否透明的阈值。
                               高于此值的Alpha被认为是不透明的。

    Returns:
        list: 一个包含多个子BBox [x, y, w, h] 的列表。
              如果BBox内没有不透明区域，则返回空列表。
    """
    x0, y0, x1, y1 = bbox
    img_h, img_w, _ = image_bgra.shape

    # 确保BBox坐标在图像范围内
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img_w, x1), min(img_h, y1)

    if x0 >= x1 or y0 >= y1:
        return []

    # 1. 提取BBox内的区域，并获取其Alpha通道
    roi = image_bgra[y0:y1, x0:x1]
    alpha_channel = roi[:, :, 3]  # BGRA格式的Alpha通道在索引3

    # 2. 二值化Alpha通道
    # 使用cv2.THRESH_BINARY，高于阈值的像素变为255，否则为0
    _, mask = cv2.threshold(alpha_channel, alpha_threshold, 255, cv2.THRESH_BINARY)

    # 3. 寻找轮廓
    # cv2.RETR_EXTERNAL 只检测最外层的轮廓，这正是我们需要的
    # cv2.CHAIN_APPROX_SIMPLE 压缩轮廓，节省内存
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. 将轮廓转换为BBox
    sub_bboxes = []
    for contour in contours:
        # 计算轮廓的边界框 (x, y, w, h)
        sub_x, sub_y, sub_w, sub_h = cv2.boundingRect(contour)
        
        # 将子BBox的相对坐标转换回原图的绝对坐标
        abs_x0 = x0 + sub_x
        abs_y0 = y0 + sub_y
        abs_x1 = abs_x0 + sub_w
        abs_y1 = abs_y0 + sub_h
        
        sub_bboxes.append([abs_x0, abs_y0, abs_x1, abs_y1])
        
    return sub_bboxes

def merge_overlapping_bboxes(bboxes):
    """
    合并列表中所有重叠的BBox。

    Args:
        bboxes (list): 一个包含多个BBox [x0, y0, x1, y1] 的列表。

    Returns:
        list: 一个新的BBox列表，其中所有重叠的BBox已被合并。
    """
    if not bboxes:
        return []

    # 使用索引来操作，避免在迭代时修改列表
    bboxes = [list(b) for b in bboxes] # 确保是可修改的列表

    while True:
        merged_one = False
        i = 0
        while i < len(bboxes):
            j = i + 1
            while j < len(bboxes):
                box1 = bboxes[i]
                box2 = bboxes[j]

                # 检查是否重叠
                # 如果一个box的右边在另一个的左边之外，或者上边在下边之外，则不重叠
                is_overlapping = not (box1[2] < box2[0] or  # box1在box2左侧
                                      box1[0] > box2[2] or  # box1在box2右侧
                                      box1[3] < box2[1] or  # box1在box2上方
                                      box1[1] > box2[3])   # box1在box2下方

                if is_overlapping:
                    # 合并两个BBox
                    new_x0 = min(box1[0], box2[0])
                    new_y0 = min(box1[1], box2[1])
                    new_x1 = max(box1[2], box2[2])
                    new_y1 = max(box1[3], box2[3])
                    
                    # 用合并后的大BBox替换第一个，并删除第二个
                    bboxes[i] = [new_x0, new_y0, new_x1, new_y1]
                    bboxes.pop(j)
                    
                    # 因为我们合并了，需要从头开始重新检查
                    merged_one = True
                    break # 跳出内层j循环
                else:
                    j += 1
            
            if merged_one:
                break # 跳出外层i循环，重新开始while True
            else:
                i += 1
        
        # 如果完整遍历一次后没有任何合并发生，则结束
        if not merged_one:
            break
            
    return bboxes

def compact_and_center_with_relative_pos(imgidx, img_nums, image, normalized_bboxes, n=1):
    """
    将 BBox 区域紧凑排列（保留相对位置），然后居中放置在透明背景上。
    
    新增功能:
    - n (int): 一个阈值。找出所有被至少 n 个 BBox 覆盖的区域，
              然后返回所有与这些区域有交集的原始 BBox 的并集。

    Args:
        image (str): 原始图像的路径或base64字符串。
        normalized_bboxes (list of lists): Bounding box 列表。
        n (int): 重叠阈值，默认为1。
    """
    # ✅ 1. 解码并统一转换为4通道BGRA格式
    if not image.startswith('data:image;base64,'):
        image64 = image_to_base64(image).split(',')[1]
    elif ',' in image:
        image64 = image.split(',')[1]
    image_data = base64.b64decode(image64)
    pil_img = Image.open(io.BytesIO(image_data)).convert("RGBA")
    img_cv_bgra = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
    img_h, img_w, _ = img_cv_bgra.shape

    # ✅ 2. 修复空BBox情况的返回值
    if not normalized_bboxes:
        return None, []
        
    # --- 坐标转换 ---
    initial_pixel_bboxes = []
    for n_box in normalized_bboxes:
        nx0, ny0, nx1, ny1 = n_box
        x0, y0 = int(nx0 * img_w), int(ny0 * img_h)
        x1, y1 = int(nx1 * img_w), int(ny1 * img_h)
        initial_pixel_bboxes.append([x0, y0, x1, y1])

    # --- 修正后的逻辑：基于重叠阈值 n 筛选贡献BBox的并集 ---
    
    # 步骤 1: 创建重叠计数图，识别高置信度像素区域
    overlap_map = np.zeros((img_h, img_w), dtype=np.uint16)
    for bbox in initial_pixel_bboxes:
        x0, y0, x1, y1 = bbox
        if x0 < x1 and y0 < y1:
            overlap_map[y0:y1, x0:x1] += 1
    
    # 根据阈值 n 创建高置信度区域的二元掩码
    threshold_mask = (overlap_map >= n)
    
    # 如果没有任何区域满足阈值，则返回空
    if not np.any(threshold_mask):
         return None, []

    # 步骤 2: 查找所有与高置信度区域有交集的原始BBox
    contributing_bboxes = []
    for bbox in initial_pixel_bboxes:
        x0, y0, x1, y1 = bbox
        if x0 < x1 and y0 < y1 and np.any(threshold_mask[y0:y1, x0:x1]):
            contributing_bboxes.append(bbox)
            
    if not contributing_bboxes:
        return None, []

    # 步骤 3: 计算所有贡献BBox的并集
    final_merged_bboxes = merge_overlapping_bboxes(contributing_bboxes)
    
    # 使用这个最终合并后的BBox列表进行后续操作
    bboxes = np.array(final_merged_bboxes, dtype=int)
    # --- 筛选逻辑结束 ---

    # --- 新增的分解步骤 (可以保留，作用于最终的并集区域) ---
    decomposed_bboxes = []
    for bbox in bboxes:
        # 对每个初始BBox进行分解
        sub_bboxes = decompose_bbox_by_alpha(img_cv_bgra, bbox)
        decomposed_bboxes.extend(sub_bboxes)
    
    if not decomposed_bboxes:
        return None,[]

    # 使用分解后的BBox列表进行后续操作
    bboxes = np.array(decomposed_bboxes, dtype=int)
    # --- 分解结束 ---

    # ✅ 3. 您的 "bbox_only_region.png" 逻辑，现在作用于最终筛选出的区域
    masked_img_bgra = np.zeros_like(img_cv_bgra) 
    for x0, y0, x1, y1 in bboxes:
        x0_c, y0_c = max(0, x0), max(0, y0)
        x1_c, y1_c = min(img_w, x1), min(img_h, y1)
        if x0_c < x1_c and y0_c < y1_c:
            masked_img_bgra[y0_c:y1_c, x0_c:x1_c] = img_cv_bgra[y0_c:y1_c, x0_c:x1_c]
            
    masked_img_rgba = cv2.cvtColor(masked_img_bgra, cv2.COLOR_BGRA2RGBA)
    pil_result_masked = Image.fromarray(masked_img_rgba)
    pil_result_masked.save(f"case_SPAR_{img_nums}_{imgidx}_result_transparent_bg.png")

    # --- 紧凑排列逻辑 (逻辑不变) ---
    x_coords = sorted(list(set(bboxes[:, [0, 2]].flatten())))
    y_coords = sorted(list(set(bboxes[:, [1, 3]].flatten())))

    x_map, new_x = {}, 0
    for i in range(len(x_coords) - 1):
        x_map[x_coords[i]] = new_x
        start_x, end_x = x_coords[i], x_coords[i+1]
        if any(b[0] < end_x and b[2] > start_x for b in bboxes):
            new_x += (end_x - start_x)
    x_map[x_coords[-1]] = new_x
    new_total_width = new_x

    y_map, new_y = {}, 0
    for i in range(len(y_coords) - 1):
        y_map[y_coords[i]] = new_y
        start_y, end_y = y_coords[i], y_coords[i+1]
        if any(b[1] < end_y and b[3] > start_y for b in bboxes):
            new_y += (end_y - start_y)
    y_map[y_coords[-1]] = new_y
    new_total_height = new_y

    # ✅ 4. 创建4通道透明画布，并从4通道源图粘贴
    composite_image_bgra = np.zeros((new_total_height, new_total_width, 4), dtype=np.uint8)
    for x0, y0, x1, y1 in bboxes:
        y0_c, y1_c = max(0, y0), min(img_h, y1)
        x0_c, x1_c = max(0, x0), min(img_w, x1)
        if y0_c >= y1_c or x0_c >= x1_c: continue
        
        roi = img_cv_bgra[y0_c:y1_c, x0_c:x1_c]
        paste_x, paste_y = x_map[x0], y_map[y0]
        h, w, _ = roi.shape
        composite_image_bgra[paste_y : paste_y + h, paste_x : paste_x + w] = roi

    # ✅ 5. 创建最终的4通道透明画布，并居中粘贴
    final_canvas_bgra = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    final_img_bgra = place_on_center(final_canvas_bgra, composite_image_bgra)
    
    final_img_rgba = cv2.cvtColor(final_img_bgra, cv2.COLOR_BGRA2RGBA)
    pil_result_centered = Image.fromarray(final_img_rgba)
    pil_result_centered.save(f"case_SPAR_{img_nums}_{imgidx}_result_transparent_bg_center.png")

    # ✅ 6. 对紧凑图进行缩放和返回
    composite_image_rgba = cv2.cvtColor(composite_image_bgra, cv2.COLOR_BGRA2RGBA)
    pil_result = Image.fromarray(composite_image_rgba)

    up_sclae = 1
    new_size = (round(pil_result.width * up_sclae), round(pil_result.height * up_sclae))
    pil_result = pil_result.resize(new_size, Image.Resampling.BILINEAR)
    pil_result.save(f"case_SPAR_{img_nums}_{imgidx}_result.png")
    
    return_bboxes = []
    for x0, y0, x1, y1 in bboxes:
        return_bboxes.append([x0/img_w, y0/img_h, x1/img_w, y1/img_h])
    return [pil_to_base64(pil_result)], return_bboxes

def find_top_n_attended_regions(norm_att, n, threshold=0.5):
    """
    从注意力图中找到前n个最受关注的连通区域。

    这个函数通过以下步骤工作：
    1. 使用阈值对注意力图进行二值化，以识别高关注度区域。
    2. 对二值化图进行连通域分析，找到所有独立的区域。
    3. 为每个区域计算一个“关注度分数”（区域内所有注意力值的总和）。
    4. 根据分数对所有区域进行降序排序。
    5. 返回排名前n的区域的边界框。如果总区域数小于n，则返回所有区域。

    参数:
    att_map (np.ndarray): 二维的注意力图，值通常在0到1之间。
    n (int): 需要寻找的顶部区域的数量。
    threshold (float, optional): 用于二值化的阈值。默认为 0.5。

    返回:
    list: 一个包含边界框的列表。每个边界框格式为 [x_min, y_min, x_max, y_max]。
          列表按关注度分数降序排列。
    """
    # 1. 二值化处理和连通域分析（与原代码相同）
    att_map = np.array(norm_att)
    # map_area = att_map.shape[0] * att_map.shape[1]
    binarized_map = (att_map >= threshold)
    # binarized_map = binarize_with_otsu(att_map)
    if not np.any(binarized_map):  # 如果阈值化后没有任何区域，直接返回空列表
        return [],0
        
    labeled_map = label(binarized_map, connectivity=2)
    regions = regionprops(labeled_map)

    # 2. 为每个区域计算分数并存储
    scored_regions = []
    for region in regions:
        # 创建一个与att_map同样大小的掩码，其中只有当前区域为True
        mask = (labeled_map == region.label)
        # 计算该区域内所有像素在原始att_map上的注意力值总和作为分数
        score = np.sum(att_map[mask])
        scored_regions.append({
            'score': score,
            'bbox': region.bbox  # bbox格式为 (y0, x0, y1, x1)
        })
        # if 0 == region.bbox[0] and 0 == region.bbox[1]: return [],0

    # 3. 根据分数对区域进行降序排序
    sorted_regions = sorted(scored_regions, key=lambda r: r['score'], reverse=True)

    # # 4. 选择前n个区域（如果不够n个，则全选）
    # if n > len(sorted_regions):
    #     n = len(sorted_regions)
    # top_n_regions = sorted_regions[:n]

    final_boxes = []
    # 5. 提取并格式化边界框
    get_num = 0
    for region in sorted_regions:
        y0, x0, y1, x1 = region['bbox']
        # 转换为 [x_min, y_min, x_max, y_max] 格式
        box_area = (y1-y0) * (x1-x0)
        # if 0 == x0 and 0 == y0 and box_area/map_area < 0.1:
        #     continue
        get_num += 1
        final_boxes.append([x0, y0, x1, y1])
        # if 0 == x0 and 0 == y0: return [],0

    # final_boxes = []
    # for region in sorted_regions:
    #     y0, x0, y1, x1 = region['bbox']
    #     final_boxes.append([x0, y0, x1, y1])

    return final_boxes, len(final_boxes)

def from_img_and_att_get_cropbox(inputs,attention, dicts, img_url, img_start, img_end,sig,thre):
    tmp_att = []
    for i in range(len(attention)):
        if attention[i] is None:
            continue
        tmp_att.append(attention[i])
    attention = tmp_att
    start_k = img_end[-1]+1
    end_k = len(inputs['input_ids'][0])
    results = {}
    for s in sig:
        for t in thre:
            accept_att = process(dicts, start_k, end_k, attention, inputs, img_start, img_end,s)
            # print(accept_att)
            imgs_words_att_box = {}
            for img_idx in accept_att:
                accept_word_att = accept_att[img_idx]
                words_att_box = {}
                for word in accept_word_att:
                    att_map = accept_word_att[word][0]
                    boxs, rigion_nums = find_top_n_attended_regions(att_map, 100, t)
                    total_attention = np.sum(att_map)
                    img_height, img_width = att_map.shape
                    total_area = img_width * img_height
                    save_boxs = []
                    if boxs:
                        H, W = att_map.shape
                        words_att_box[word] = []
                        for box in boxs:
                            x0,y0,x1,y1 = box
                            # 计算 box 面积
                            bbox_norm = (x0 / W, y0 / H, (x1) / W, (y1) / H)
                            x0,y0,x1,y1 = bbox_norm
                            region = att_map[int(y0*H):int(y1*H), int(x0*W):int(x1*W)]
                            region_sum = np.sum(region)
                            words_att_box[word].append(bbox_norm)
                            save_boxs.append(box)
                imgs_words_att_box[img_idx] = words_att_box

            for img_idx in imgs_words_att_box:
                max_word_idx = 0
                for words_idx in imgs_words_att_box[img_idx]:
                    max_word_idx = max(max_word_idx,words_idx)
            img_merged_boxes = swap_and_rebuild_dict(imgs_words_att_box)

            words_lines = {}
            get_words = ""
            # print(start_k,end_k)
            for i in range(start_k,end_k):
                token_idx = inputs['input_ids'][0][i].cpu().item()
                # print(i,dicts[token_idx],end="||")
                if token_idx < 151643:
                    get_words+=dicts[token_idx]
                for word in img_merged_boxes:
                    if i == word+1:
                        words_lines[word] = get_words
                        get_words = ''
            for word in img_merged_boxes:
                if i == word:
                    words_lines[word] = get_words
                    get_words = ''
            words_lines[-1] = get_words
            get_words = ''
            crop_list = {}
            bounding_boxes = {}
            highlight_imgs = []
            for word in img_merged_boxes:
                if not word in crop_list:
                    crop_list[word] = {}
                for imgidx in img_merged_boxes[word]:
                    if not imgidx in bounding_boxes: bounding_boxes[imgidx] = []
                    for boxid in range(len(img_merged_boxes[word][imgidx])):
                        bounding_boxes[imgidx].append(img_merged_boxes[word][imgidx][boxid])
            for imgidx in bounding_boxes:
                img,bboxs = compact_and_center_with_relative_pos(imgidx,len(img_url),img_url[imgidx],bounding_boxes[imgidx])
                if img:
                    bounding_boxes[imgidx] = bboxs
                    for im in img:
                        highlight_imgs.append(im)
                # highlight_imgs.append(compact_and_center_with_relative_pos_in_ori(image_list[imgidx],bounding_boxes[imgidx]))
            if not str(s) in results:results[str(s)] = {}
            # if not str(i) in results[s]:results[s][t] = {}
            results[str(s)][str(t)] = [img_merged_boxes,crop_list,words_lines,highlight_imgs,bounding_boxes]
    return results
