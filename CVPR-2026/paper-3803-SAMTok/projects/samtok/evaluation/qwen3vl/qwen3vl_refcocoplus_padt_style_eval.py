import argparse
import copy
import math
import os
import torch
import torchvision
import tqdm
from pycocotools import mask as mask_utils
import numpy as np
import random
import re
from PIL import Image
import json
import uuid
import hydra
from collections import defaultdict

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from qwen_vl_utils import process_vision_info
from projects.samtok.models import VQ_SAM2, VQ_SAM2Config, SAM2Config, DirectResize

def parse_args():
    parser = argparse.ArgumentParser(description='RefCoco+')
    parser.add_argument(
        '--model_path',
        default="zhouyik/Qwen3-VL-4B-SAMTok-co",
        help='hf model path.')
    parser.add_argument(
        '--vq_sam2_path',
        default="zhouyik/Qwen3-VL-4B-SAMTok-co/mask_tokenizer_256x2.pth",
        help='vq-sam2 model path.')
    parser.add_argument(
        '--dataset',
        default='./data/PaDT-MLLM/RefCOCO/refcoco+_val.json',
        help='Specify a ref dataset')
    parser.add_argument('--task_id', '--task-id', type=int, default=0)
    args = parser.parse_args()
    return args

IMAGE_FOLDER = './data/glamm_data/images/coco2014/train2014/'

def extract_mt_token_ids(text):
    pattern = r"<\|mt_(\d{4})\|>"
    return [int(x) for x in re.findall(pattern, text)]

def find_first_index(arr, value):
    """
    在NumPy数组中找到第一个指定值的第一个出现的索引
    
    参数:
        arr: NumPy数组
        value: 要查找的值
        
    返回:
        第一个匹配值的索引，如果没有找到则返回-1
    """
    # 使用where找到所有匹配值的索引
    indices = np.where(arr == value)[0]
    
    # 返回第一个索引，如果没有找到则返回-1
    return indices[0] if len(indices) > 0 else -1

def mask_iou(mask1, mask2):
    mask1 = mask1.unsqueeze(1).char() # n, 1, h, w
    mask2 = mask2.unsqueeze(0).char() # 1, n, h, w

    intersection = (mask1 & mask2)
    union = (mask1 + mask2 - intersection).sum(-1).sum(-1)
    intersection = intersection.sum(-1).sum(-1)

    return intersection / union

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(mask_utils.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle

def rle_to_mask(rle):
    mask = []
    for r in rle:
        m = mask_utils.decode(r)
        m = np.uint8(m)
        mask.append(m)
    mask = np.stack(mask, axis=0)
    return mask

def decode_mask(object_masks, ori_height, ori_width):
    binary_masks = []
    for object_mask in object_masks:
        if isinstance(object_mask, dict):
            if isinstance(object_mask["counts"], list):
                # convert to compressed RLE
                object_mask = mask_utils.frPyObjects(object_mask, ori_height, ori_width)
            m = mask_utils.decode(object_mask)
            m = m.astype(np.uint8).squeeze()
        elif object_mask:
            rles = mask_utils.frPyObjects(object_mask, ori_height, ori_width)
            rle = mask_utils.merge(rles)
            m = mask_utils.decode(rle).astype(np.uint8).squeeze()
        else:
            m = np.zeros((ori_height, ori_width), dtype=np.uint8)
        binary_masks.append(m)
    return binary_masks


def calculate_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x1_prime, y1_prime, w1_prime, h1_prime = bbox2

    bbox1_coords = [x1, y1, x1 + w1, y1 + h1]
    bbox2_coords = [x1_prime, y1_prime, x1_prime + w1_prime, y1_prime + h1_prime]

    inter_x1 = max(bbox1_coords[0], bbox2_coords[0])
    inter_y1 = max(bbox1_coords[1], bbox2_coords[1])
    inter_x2 = min(bbox1_coords[2], bbox2_coords[2])
    inter_y2 = min(bbox1_coords[3], bbox2_coords[3])

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)

    inter_area = inter_width * inter_height

    bbox1_area = w1 * h1
    bbox2_area = w1_prime * h1_prime

    union_area = bbox1_area + bbox2_area - inter_area

    if union_area == 0:
        return 0.0

    iou = inter_area / union_area
    return iou


def calculate_ciou(pred: np.ndarray, gt: np.ndarray):
    i = np.logical_and(pred, gt).sum()
    u = np.logical_or(pred, gt).sum()
    return i/u if u>0 else 0.0

def metric():
    gt_dict = defaultdict(list)
    accuracy = defaultdict(int)
    mask_cious = defaultdict(float)
    pred_dict = defaultdict(list)

    for json_file in os.listdir('./temp_save/refcoco_plus'):
        with open(os.path.join('./temp_save/refcoco_plus', json_file), 'r') as f:
            data_dict = json.load(f)
        
        bbox_name = data_dict['bbox_name']
        gt_masks = data_dict['gt_masks']
        pred_masks = data_dict['prediction_masks']
        assert len(gt_masks) == len(pred_masks) == 1
        gt_masks = rle_to_mask(gt_masks)
        pred_masks = rle_to_mask(pred_masks)

        gt_masks_tensor = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in gt_masks])
        pred_masks_tensor = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in pred_masks])

        try:
            gt_boxes = torchvision.ops.masks_to_boxes(gt_masks_tensor).squeeze().cpu().numpy().tolist()
            x1, y1, x2, y2 = gt_boxes
            w = x2 - x1
            h = y2 - y1
            gt_bbox = [x1, y1, w, h]
        except:
            gt_bbox = [0, 0, 0, 0]
        
        try:
            pred_boxes = torchvision.ops.masks_to_boxes(pred_masks_tensor).squeeze().cpu().numpy().tolist()
            x1, y1, x2, y2 = pred_boxes
            w = x2 - x1
            h = y2 - y1
            pred_bbox = [x1, y1, w, h]
        except:
            pred_bbox = [0, 0, 0, 0]

        gt_dict[bbox_name] = [gt_bbox, gt_masks]
        accuracy[bbox_name] = 0.

        ciou = calculate_ciou(pred_masks > 0, gt_masks > 0)
        iou = calculate_iou(gt_bbox, pred_bbox)

        if ciou < 0.7:
            print(f"======>>>ciou: {ciou}, bbox_name: {bbox_name}")

        accuracy[bbox_name] = max(iou, accuracy[bbox_name])
        mask_cious[bbox_name] = max(ciou, mask_cious[bbox_name])
        pred_dict[bbox_name] = [pred_bbox, pred_masks]

    all_ious = np.array([i for i in accuracy.values()])
    all_mask_cious = np.array([i for i in mask_cious.values()])
    ap = (all_ious >= 0.5).mean()
    mean_cious = all_mask_cious.mean()
    print('The results using our validation set.')
    print('REC AP_50:', ap, '| RES CIoU:', mean_cious)

    # align to VLM-R1
    vlm_eval_ap = []
    vlm_eval_ciou = []
    vlm_json_files = ['./data/PaDT-MLLM/RefCOCO/rec_jsons_processed/refcocop_val.json']
    with open(vlm_json_files[0], 'r') as f:
        items = json.load(f)
        for idx, item in enumerate(items):
            image_id = int(item['image'].split('_')[-1].split('.')[0])
            category = item['normal_caption']
            vlm_eval_ap.append(accuracy['%d_%s' % (image_id, category)] >= 0.5)
            vlm_eval_ciou.append(mask_cious['%d_%s' % (image_id, category)])

    print('\nThe results using VLM-R1 validation set. [The results present in our paper]')
    print('REC AP_50:', np.array(vlm_eval_ap).mean().item(), '| RES CIoU:', np.array(vlm_eval_ciou).mean().item())


def main():
    args = parse_args()

    # build qwen3vl model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto"
    ).cuda().eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    # build vq-sam2 model
    CODEBOOK_SIZE = 256
    CODEBOOK_DEPTH = 2
    sam2_config = SAM2Config(
        ckpt_path=args.sam2_path,
    )
    
    vq_sam2_config = VQ_SAM2Config(
        sam2_config=sam2_config,
        codebook_size=CODEBOOK_SIZE,
        codebook_depth=CODEBOOK_DEPTH,
        shared_codebook=False,
        latent_dim=256,
    )

    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()

    state = torch.load(args.vq_sam2_path, map_location="cpu")
    vq_sam2.load_state_dict(state)

    sam2_image_processor = DirectResize(1024)


    # dataset
    all_data_dict = []
    case_id = 0
    with open(args.dataset, 'r') as f:
        for line in f:
            # Skip empty lines
            if line.strip():
                item = json.loads(line)
                item.update({'case_id': case_id})
                all_data_dict.append(item)
                case_id += 1

    rows = len(all_data_dict)
    # chunk_size = (rows+7) // 8
    chunk_size = (rows+0) // 1
    _start_ = args.task_id * chunk_size
    _end_ = _start_ + chunk_size
    _end_ = rows if _end_ > rows else _end_

    for data_dict in tqdm.tqdm(all_data_dict[_start_:_end_]):
        image_file = data_dict['image']
        image_path = os.path.join(IMAGE_FOLDER, image_file)
        rle = data_dict['objects'][0]['rle']
        phrase = data_dict['objects'][0]['label']
        case_id = data_dict['case_id']
        image_id = data_dict['id']
        question = f"Please segment {phrase} in this image."

        bbox_name = '%d_%s' % (image_id, phrase)

        if os.path.exists(f"./temp_save/refcoco_plus/{case_id}.json"):
            print("file exists.............")
            continue
        
        batch_size = 1
        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size
        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
        sam2_pixel_values = sam2_pixel_values.repeat(batch_size, 1, 1, 1)

        gt_mask = decode_mask([rle], ori_height, ori_width)[0]
        gt_mask = gt_mask[np.newaxis, :, :]
        gt_mask = mask_to_rle(gt_mask)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=128,
            do_sample=False,  # 关闭采样，使用贪婪解码
            top_p=1.0,  # 配合do_sample=False使用
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print("Assistant: ", output_text)

        quant_ids = extract_mt_token_ids(output_text[0])
        if len(quant_ids) == 0:
            zero_mask = np.zeros((1, ori_height, ori_width)).astype(np.uint8)
            zero_mask = mask_to_rle(zero_mask)
            prediction = {'image_id': image_id, 'phrase': phrase, 'bbox_name': bbox_name, 'gt_masks': gt_mask, 'prediction_masks': zero_mask}
            with open(f"./temp_save/refcoco_plus/{case_id}.json", 'w') as f:
                json.dump(prediction, f)
            continue


        batch_size = 1
        remap_quant_ids = np.array([-1 for _ in range(CODEBOOK_DEPTH)])
        for quant_id in quant_ids:
            depth_idx = quant_id // CODEBOOK_SIZE
            remap_quant_ids[depth_idx] = quant_id % CODEBOOK_SIZE

        truncated_idx = find_first_index(remap_quant_ids, -1)
        if truncated_idx != -1:
            remap_quant_ids[truncated_idx:] = -1
        if remap_quant_ids[0] == -1:
            zero_mask = np.zeros((1, ori_height, ori_width)).astype(np.uint8)
            zero_mask = mask_to_rle(zero_mask)
            prediction = {'image_id': image_id, 'phrase': phrase, 'bbox_name': bbox_name, 'gt_masks': gt_mask, 'prediction_masks': zero_mask}
            with open(f"./temp_save/refcoco_plus/{case_id}.json", 'w') as f:
                json.dump(prediction, f)
            continue
        quant_ids = torch.LongTensor(remap_quant_ids).to(vq_sam2.device).unsqueeze(0)

        _pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids)
        _pred_masks = torch.nn.functional.interpolate(_pred_masks, size=(ori_height, ori_width), mode='bilinear')
        _pred_masks = _pred_masks > 0.5
        _pred_masks = _pred_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)

        _pred_masks = mask_to_rle(_pred_masks)
        prediction = {'image_id': image_id, 'phrase': phrase, 'bbox_name': bbox_name, 'gt_masks': gt_mask, 'prediction_masks': _pred_masks}
        with open(f"./temp_save/refcoco_plus/{case_id}.json", 'w') as f:
            json.dump(prediction, f)

        
if __name__ == '__main__':
    main()
    print(metric())