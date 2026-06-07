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

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from qwen_vl_utils import process_vision_info
from projects.samtok.models import VQ_SAM2, VQ_SAM2Config, SAM2Config, DirectResize


def parse_args():
    parser = argparse.ArgumentParser(description='MRREFCOCOg')
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
        default='./data/segllm_data/conversation_folder/all_data_mix_val/mr_refcocog_val_sampled.json',
        help='Specify a ref dataset')
    parser.add_argument('--task_id', '--task-id', type=int, default=0)
    parser.add_argument('--round_id', '--round-id', type=int, default=0)
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

def metric():
    from projects.samtok.evaluation.utils import REFER, Summary, AverageMeter, intersectionAndUnionGPU, master_only
    
    trackers = {
        "intersection": AverageMeter("Intersec", ":6.3f", Summary.SUM),
        "union": AverageMeter("Union", ":6.3f", Summary.SUM),
        "gIoU": AverageMeter("gIoU", ":6.3f", Summary.SUM)
    }
    for json_file in os.listdir('./temp_save/mr_refcocog'):
        with open(os.path.join('./temp_save/mr_refcocog', json_file), 'r') as f:
            data_dict = json.load(f)

        intersection, union, accuracy_iou = 0.0, 0.0, 0.0
        masks = data_dict['prediction_masks']
        _masks = []
        for mask in masks:
            if mask is not None:
                mask = rle_to_mask([mask])
            _masks.append(mask)
        targets = data_dict['gt_masks']
        _targets = rle_to_mask(targets)

        for i_item, _mask in enumerate(_masks):
            if _mask is None:
                continue

            _target = _targets[i_item: i_item+1]
            for prediction, target in zip(_mask, _target):
                prediction = torch.from_numpy(prediction).int().cuda()
                target = torch.from_numpy(target).int().cuda()
                intersect, union_, _ = intersectionAndUnionGPU(
                    prediction.contiguous().clone(), target.contiguous(), 2, ignore_index=255
                )
                intersection += intersect
                union += union_
                accuracy_iou += intersect / (union_ + 1e-5)
                accuracy_iou[union_ == 0] += 1.0
        try:
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            accuracy_iou = accuracy_iou.cpu().numpy() / _targets.shape[0]
        except:
            accuracy_iou = accuracy_iou / _targets.shape[0]
        trackers["intersection"].update(intersection)
        trackers["union"].update(union)
        trackers["gIoU"].update(accuracy_iou, n=_targets.shape[0])

    cur_results = {'pixel_intersection': trackers["intersection"].sum[1],
                    'pixel_union': trackers["union"].sum[1],
                    'gIoU': trackers["gIoU"].avg[1],
                    'mask_counts': trackers["gIoU"].count,
                    }
    class_iou = cur_results['pixel_intersection'] / (cur_results['pixel_union'] + 1e-10)
    global_iou = cur_results['gIoU']

    print('============================================', 'current')
    print('CIoU: {}, GIoU: {}'.format(class_iou, global_iou), 'current')
    print('============================================', 'current')
    return {'Acc': class_iou}



def main():
    args = parse_args()

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
        json_data = json.load(f)
        round_items = json_data[str(args.round_id)]
        for item in round_items:
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
        image_path = data_dict['image']
        history_chat = data_dict['history']
        target_masks = data_dict['target_masks']
        case_id = data_dict['case_id']

        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

        gt_mask = decode_mask(target_masks, ori_height, ori_width)[0]
        gt_mask = gt_mask[np.newaxis, :, :]
        gt_mask = mask_to_rle(gt_mask)

        messages = []
        for turn_id, turn in enumerate(history_chat):
            role = turn["from"]
            content = turn["value"]
            if role == "human":
                if turn_id == 0:
                    assert "<image>\n" in content
                    content = content.replace("<image>\n", "").strip()
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image_path,
                                },
                                {"type": "text", "text": content},
                            ],
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": content},
                            ],
                        }
                    )
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": content},
                        ],
                    }
                )

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

        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

        quant_ids = extract_mt_token_ids(output_text[0])
        if len(quant_ids) == 0:
            zero_mask = np.zeros((1, ori_height, ori_width)).astype(np.uint8)
            zero_mask = mask_to_rle(zero_mask)
            prediction = {'image_id': case_id, 'gt_masks': gt_mask, 'prediction_masks': zero_mask}
            with open(f"./temp_save/mr_refcocog/{case_id}.json", 'w') as f:
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
            prediction = {'image_id': case_id, 'gt_masks': gt_mask, 'prediction_masks': zero_mask}
            with open(f"./temp_save/mr_refcocog/{case_id}.json", 'w') as f:
                json.dump(prediction, f)
            continue
        quant_ids = torch.LongTensor(remap_quant_ids).to(vq_sam2.device).unsqueeze(0)

        _pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids)
        _pred_masks = torch.nn.functional.interpolate(_pred_masks, size=(ori_height, ori_width), mode='bilinear')
        _pred_masks = _pred_masks > 0.5
        _pred_masks = _pred_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)

        _pred_masks = mask_to_rle(_pred_masks)
        prediction = {'image_id': case_id, 'gt_masks': gt_mask, 'prediction_masks': _pred_masks}
        with open(f"./temp_save/mr_refcocog/{case_id}.json", 'w') as f:
            json.dump(prediction, f)

        
if __name__ == '__main__':
    main()
    print(metric())