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
from typing import Callable, Optional, Tuple, TypedDict
import hydra

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from xtuner.model.utils import guess_load_checkpoint

from qwen_vl_utils import process_vision_info

from projects.samtok.models import VQ_SAM2, VQ_SAM2Config, SAM2Config, DirectResize
from projects.samtok.evaluation.grefer import G_REFER


def load_dataset(split='val'):
    refer_api = G_REFER('./data/ref_seg/grefs/coco2014/train2014', './data/ref_seg/grefs/grefs(unc).json', './data/ref_seg/grefs/instances.json')
    
    ref_ids_val = refer_api.getRefIds(split=split)
    images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
    refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
    refer_seg_ds = {}
    refer_seg_ds["images"] = []
    loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
    for item in loaded_images:
        item = item.copy()
        item["file_name"] = os.path.join('./data/ref_seg/grefs/coco2014/train2014', item["file_name"])
        refer_seg_ds["images"].append(item)
    refer_seg_ds["annotations"] = refer_api.Anns  # anns_val
    img2refs = {}
    for ref in refs_val:
        image_id = ref["image_id"]
        img2refs[image_id] = img2refs.get(image_id, []) + [ref]
    refer_seg_ds["img2refs"] = img2refs


    all_items = []
    for index in range(len(refer_seg_ds["images"])):
        image_info = refer_seg_ds["images"][index]
        image_path = image_info["file_name"]
        image_id = image_info["id"]
        image_size = image_info["width"], image_info["height"]

        refs = img2refs[image_id]
        if len(refs) == 0:
            continue

        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                sents.append(sent["sent"].strip().lower())
                ann_ids.append(ref["ann_id"])
        sampled_sents = sents
        sampled_ann_ids = ann_ids

        anno_masks = []
        for i, ann_id in enumerate(sampled_ann_ids):
            no_target = ann_id == [-1]
            if no_target:  # no target
                m = np.zeros((image_info["height"], image_info["width"], 1))
            elif len(ann_id) > 1:  # multi target / already merged ?
                m = []
                for sub_ann_id in ann_id:
                    sub_mask_info = refer_seg_ds["annotations"][sub_ann_id]["segmentation"]
                    if len(sub_mask_info) == 0:
                        sub_m = np.zeros((image_info["height"], image_info["width"], 1))
                    else:
                        if isinstance(sub_mask_info, dict):
                            if isinstance(sub_mask_info["counts"], list):
                                # convert to compressed RLE
                                rle = mask_utils.frPyObjects(sub_mask_info, image_info["height"], image_info["width"])
                        else:
                            # filter out invalid polygons (< 3 points)
                            polygons = [poly for poly in sub_mask_info if len(poly) % 2 == 0 and len(poly) >= 6]
                            if len(polygons) == 0:
                                continue  # ignore this instance
                            rle = mask_utils.frPyObjects(polygons, image_info["height"], image_info["width"])
                        sub_m = mask_utils.decode(rle)
                        if sub_m.ndim < 3:
                            assert sub_m.ndim == 2
                            sub_m = sub_m[..., np.newaxis]
                    sub_m = np.sum(sub_m, axis=2)
                    m.append(sub_m)
                m = np.sum(m, axis=0)[..., np.newaxis]
            else:
                assert len(ann_id) == 1 and ann_id[0] != -1
                mask_info = refer_seg_ds["annotations"][ann_id[0]]["segmentation"]
                if len(mask_info) == 0:
                    m = np.zeros((image_info["height"], image_info["width"], 1))
                else:
                    if isinstance(mask_info, dict):
                        if isinstance(mask_info["counts"], list):
                            # convert to compressed RLE
                            rle = mask_utils.frPyObjects(mask_info, image_info["height"], image_info["width"])
                    else:
                        # filter out invalid polygons (< 3 points)
                        polygons = [poly for poly in mask_info if len(poly) % 2 == 0 and len(poly) >= 6]
                        if len(polygons) == 0:
                            continue  # ignore this instance
                        rle = mask_utils.frPyObjects(polygons, image_info["height"], image_info["width"])
                    m = mask_utils.decode(rle)
                    if m.ndim < 3:
                        assert m.ndim == 2
                        m = m[..., np.newaxis]
            m = np.sum(m, axis=2)
            anno_masks.append(m)   

        for sent, binary_mask in zip(sents, anno_masks):
            assert len(binary_mask.shape) == 2
            binary_mask = (binary_mask > 0).astype(np.uint8)
            rle = mask_utils.encode(np.array(binary_mask[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")

            all_items.append({
                "image": image_path,
                "phrase": sent,
                "segmentation": rle,
            })
    
    with open(f'./data/PaDT-MLLM/RefCOCO/grefcoco_{split}.json', 'w') as f:
        json.dump(all_items, f)
    print(f"Saved at ./data/PaDT-MLLM/RefCOCO/grefcoco_{split}.json")

def parse_args():
    parser = argparse.ArgumentParser(description='GRES')
    parser.add_argument(
        '--model_path',
        default="zhouyik/Qwen2.5-VL-3B-SAMTok-gres-rl",
        help='hf model path.')
    parser.add_argument(
        '--vq_sam2_path',
        default="zhouyik/Qwen2.5-VL-3B-SAMTok-gres-rl/mask_tokenizer_256x2.pth",
        help='vq-sam2 model path.')
    parser.add_argument(
        '--dataset',
        default='./data/PaDT-MLLM/RefCOCO/grefcoco_val.json',
        help='Specify a ref dataset')
    parser.add_argument('--task_id', '--task-id', type=int, default=0)
    args = parser.parse_args()
    return args


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


def extract_mt_token_ids_v1(text):
    pattern = r"<\|mt_(\d{4})\|>"
    return [int(x) for x in re.findall(pattern, text)]

def extract_mt_token_ids_v2(text):
    pattern = re.compile(r'<\|mt_start\|><\|mt_(\d{4})\|><\|mt_(\d{4})\|><\|mt_end\|>')
    matches = pattern.findall(text)
    ret_list = []
    for num1, num2 in matches:
        ret_list.append(int(num1))
        ret_list.append(int(num2))
    return ret_list

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

def fix_mt_format_comprehensive(text):
    """
    全面修正 <|mt_...> 格式的函数。
    它会处理以下几种情况：
    1. 标记太少 (1个): <|mt_start|><|mt_0198|><|mt_end|> -> <|mt_start|><|mt_0198|><|mt_-1|><|mt_end|>
    2. 标记太少 (1个, 无end): <|mt_start|><|mt_0198|> -> <|mt_start|><|mt_0198|><|mt_-1|><|mt_end|>
    3. 标记太多 (3个或以上): <|mt_start|><|mt_0186|><|mt_0410|><|mt_0186|><|mt_end|> -> <|mt_start|><|mt_0186|><|mt_0410|><|mt_end|>
    4. 正确格式: <|mt_start|><|mt_0044|><|mt_0442|><|mt_end|> -> 不变
    """
    # 规则 1: 处理标记太多的情况 (3个或以上)
    # 捕获前两个，匹配掉多余的，然后用前两个重构
    pattern_too_many = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_\d+\|>)(?:<\|mt_\d+\|>)+<\|mt_end\|>'
    replacement_too_many = r'\1\2\3<|mt_end|>'
    text = re.sub(pattern_too_many, replacement_too_many, text)
    # 规则 2: 处理标记太少的情况 (只有1个，且有<|mt_end|>)
    pattern_too_few_with_end = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_end\|>)'
    replacement_too_few = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern_too_few_with_end, replacement_too_few, text)
    # 规则 3: 处理标记太少的情况 (只有1个，且没有<|mt_end|>)
    # 使用负向前瞻确保后面不是另一个mt_token
    pattern_too_few_no_end = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(?!<\|mt_)'
    replacement_too_few_no_end = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern_too_few_no_end, replacement_too_few_no_end, text)
    return text

def extract_think_and_answer_robust(response: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts the content between <think> and <answer> tags from a string,
    regardless of their order or position, as long as the tags exist.
    Args:
        response (str): The input string, potentially containing <think> and <answer> tags.
    Returns:
        Tuple[Optional[str], Optional[str]]: 
            A tuple containing (think_content, answer_content).
            Each element will be a string if found, or None if the corresponding tag is not found.
    """
    think_content = None
    answer_content = None
    # Pattern for <think> tag content
    # re.DOTALL allows '.' to match any character, including newlines.
    # Non-greedy match (.*?) ensures it stops at the first </think>.
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    # Pattern for <answer> tag content
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    # Search for <think> content
    think_match = think_pattern.search(response)
    if think_match:
        think_content = think_match.group(1) # group(1) gets the content of the first capture group
    # Search for <answer> content
    answer_match = answer_pattern.search(response)
    if answer_match:
        answer_content = answer_match.group(1) # group(1) gets the content of the first capture group
    
    if answer_content is None or think_content is None:
        if '<answer>' in response:
            head, tail = response.split('<answer>', 1)
            if think_content is None:
                think_content = head
            if answer_content is None:
                answer_content = tail
        elif '</think>' in response:
            head, tail = response.split('</think>', 1)
            if think_content is None:
                think_content = head
            if answer_content is None:
                answer_content = tail

    return think_content, answer_content

from projects.samtok.evaluation.utils import Summary, AverageMeter, intersectionAndUnionGPU
def metric():
    inter_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    g_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    nt_tp_meter = AverageMeter("NT_TP", ":6.3f", Summary.SUM)
    nt_tn_meter = AverageMeter("NT_TN", ":6.3f", Summary.SUM)
    nt_fp_meter = AverageMeter("NT_FP", ":6.3f", Summary.SUM)
    nt_fn_meter = AverageMeter("NT_FN", ":6.3f", Summary.SUM)

    for json_file in os.listdir("./temp_save/grefcoco"):
        json_file_path = os.path.join("./temp_save/grefcoco", json_file)
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
        
        pred_mask = rle_to_mask([json_data["pred_masks"]])[0]
        gt_mask = rle_to_mask([json_data["gt_masks"]])[0]

        pred_mask = torch.from_numpy(pred_mask).int().cuda()
        gt_mask = torch.from_numpy(gt_mask).int().cuda()
        # pred_mask = gt_mask
    

        if gt_mask.sum() < 1.0:  # empty target
            if pred_mask.sum() < 1.0:
                # true positive
                nt_tp_meter.update(1.0)
                g_iou_meter.update(1.0)
            else:
                inter_i, union_i, _ = intersectionAndUnionGPU(pred_mask.contiguous().clone(), gt_mask.contiguous().clone(), K=2, ignore_index=255)
                inter_i = inter_i.cpu().numpy()
                union_i = union_i.cpu().numpy()
                nt_fn_meter.update(1.0)
                g_iou_meter.update(0.0)
                union_meter.update(union_i)
        else:
            if pred_mask.sum() < 1.0:
                nt_fp_meter.update(1.0)
            else:
                nt_tn_meter.update(1.0)
            try:
                inter_i, union_i, _ = intersectionAndUnionGPU(pred_mask.contiguous().clone(), gt_mask.contiguous().clone(), K=2, ignore_index=255)
            except:
                print("pred_mask.shape: ", pred_mask.shape)
                print("gt_mask.shape: ", gt_mask.shape)
                continue
                # exit(0)
            inter_i = inter_i.cpu().numpy()
            union_i = union_i.cpu().numpy()
            this_giou = inter_i / (union_i + 1e-8)
            inter_meter.update(inter_i)
            union_meter.update(union_i)
            g_iou_meter.update(this_giou)
        
    # inter_meter.all_reduce()
    # union_meter.all_reduce()
    # g_iou_meter.all_reduce()
    # nt_tp_meter.all_reduce()
    # nt_tn_meter.all_reduce()
    # nt_fp_meter.all_reduce()
    # nt_fn_meter.all_reduce()

    N_acc = nt_tp_meter.sum / (nt_tp_meter.sum + nt_fn_meter.sum)  # for gt is empty, pred is empty
    T_acc = nt_tn_meter.sum / (nt_tn_meter.sum + nt_fp_meter.sum)  # for gt is target, pred is target
    g_iou = g_iou_meter.avg[1]
    c_iou = (inter_meter.sum / (union_meter.sum + 1e-10))[1]
    log_stats = {}
    log_stats["N_acc"] = round(N_acc * 100, 2)
    log_stats["T_acc"] = round(T_acc * 100, 2)
    log_stats["g_iou"] = round(g_iou * 100, 2)
    log_stats["c_iou"] = round(c_iou * 100, 2)
    print(log_stats)
        
def main():
    args = parse_args()

    # build qwen25vl model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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

    all_data_dict = []
    case_id = 0
    with open(args.dataset, 'r') as f:
        json_data = json.load(f)
        for item in json_data:
            item.update({'case_id': case_id})
            all_data_dict.append(item)
            case_id += 1
    
    rows = len(all_data_dict)
    # chunk_size = (rows+23) // 24
    chunk_size = (rows+0) // 1

    _start_ = args.task_id * chunk_size
    _end_ = _start_ + chunk_size
    _end_ = rows if _end_ > rows else _end_

    count = 0
    for data_dict in tqdm.tqdm(all_data_dict[_start_:_end_]):
        image_path = data_dict['image']
        phrase = data_dict['phrase']
        rle = data_dict['segmentation']
        case_id = data_dict['case_id']

        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

        if rle['size'][0] != ori_height or rle['size'][1] != ori_width:
            print("skip this cases!!!!!!!!!!!!!!!!!!!!!!!!")
            continue

        # gt_masks = rle_to_mask([rle])
        # if len(gt_masks) < 3:
        #     continue
        # output_image = visualize(image, gt_masks, ["gt"])
        # output_image.save('./CVPR2026/grefcoco_gt.jpg')
        # print("GT RLE: ", rle)

        question = f"Please segment {phrase} in this image. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"
        # question = f"Please segment {phrase} in this image."
        
        if os.path.exists(f"./temp_save/grefcoco/{case_id}.json"):
            print("file exists.............")
            continue

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
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # generated_ids = model.generate(
        #     **inputs, 
        #     max_new_tokens=2048,
        #     do_sample=True,             # 启用采样，这是关键！
        #     temperature=0.7,            # 调整温度，增加随机性。可以尝试0.5到1.0之间的值
        #     top_k=50,                   # 考虑概率最高的50个词元
        #     top_p=0.95,                 # 考虑累积概率达到95%的词元集合
        # )
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=1024,
            do_sample=False,  # 关闭采样，使用贪婪解码
            top_p=1.0,  # 配合do_sample=False使用
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        # print("User: ", phrase)
        print("Assistant: ", output_text)
        # exit(0)

        thinking_content, answer_content = extract_think_and_answer_robust(output_text[0])
        if answer_content is None:
            quant_ids = extract_mt_token_ids_v1(output_text[0])
            if len(quant_ids) == 0:
                zero_mask = np.zeros((1, ori_height, ori_width)).astype(np.uint8)
                zero_mask = mask_to_rle(zero_mask)
                prediction = {'gt_masks': rle, 'pred_masks': zero_mask[0]}
                with open(f"./temp_save/grefcoco/{case_id}.json", 'w') as f:
                    json.dump(prediction, f)
                continue

            if len(quant_ids) % CODEBOOK_DEPTH != 0:
                print("FORMAT ERROR: ", output_text)
                output_text = [fix_mt_format_comprehensive(output_text[0])]
                print("FIXED OUTPUT TEXT: ", output_text)
                quant_ids = extract_mt_token_ids_v2(output_text[0])
            # assert len(quant_ids) % CODEBOOK_DEPTH == 0
            if len(quant_ids) % CODEBOOK_DEPTH != 0:
                zero_mask = np.zeros((1, ori_height, ori_width)).astype(np.uint8)
                zero_mask = mask_to_rle(zero_mask)
                prediction = {'gt_masks': rle, 'pred_masks': zero_mask[0]}

                with open(f"./temp_save/grefcoco/{case_id}.json", 'w') as f:
                    json.dump(prediction, f)
                continue
        else:
            quant_ids = extract_mt_token_ids_v1(answer_content)
            if len(quant_ids) % CODEBOOK_DEPTH != 0:
                answer_content = fix_mt_format_comprehensive(answer_content)
                quant_ids = extract_mt_token_ids_v2(answer_content)
            if len(quant_ids) == 0:
                zero_mask = np.zeros((1, ori_height, ori_width)).astype(np.uint8)
                zero_mask = mask_to_rle(zero_mask)
                prediction = {'gt_masks': rle, 'pred_masks': zero_mask[0]}
                with open(f"./temp_save/grefcoco/{case_id}.json", 'w') as f:
                    json.dump(prediction, f)
                continue

        batch_size = len(quant_ids) // CODEBOOK_DEPTH
        if batch_size > 10:
            quant_ids = quant_ids[:2]
            batch_size = len(quant_ids) // CODEBOOK_DEPTH
        remap_quant_ids = []
        for bs_id in range(batch_size):
            chunk_quant_ids = quant_ids[bs_id*CODEBOOK_DEPTH:(bs_id+1)*CODEBOOK_DEPTH]
            remap_chunk_quant_ids = [quant_id - book_id*CODEBOOK_SIZE for book_id, quant_id in enumerate(chunk_quant_ids)]
            code1 = remap_chunk_quant_ids[0]
            code2 = remap_chunk_quant_ids[1]
            if not (code1 >= 0 and code1 < CODEBOOK_SIZE):
                continue
            if not (code2 >= 0 and code2 < CODEBOOK_SIZE):
                code2 = -1
            remap_chunk_quant_ids_error_handle = [code1, code2]
            remap_quant_ids.append(remap_chunk_quant_ids_error_handle)

        batch_size = len(remap_quant_ids)
        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
        sam2_pixel_values = sam2_pixel_values.repeat(batch_size, 1, 1, 1)

        quant_ids = torch.LongTensor(remap_quant_ids).to(vq_sam2.device)

        with torch.no_grad():
            _pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids)
        _pred_masks = torch.nn.functional.interpolate(_pred_masks, size=(ori_height, ori_width), mode='bilinear')
        _pred_masks = _pred_masks > 0.5
        _pred_masks = _pred_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)
        _pred_masks = np.sum(_pred_masks, axis=0).astype(np.uint8)[np.newaxis, :, :]
        _pred_masks = (_pred_masks > 0).astype(np.uint8)

        # output_image = visualize(image, _pred_masks, tags=['pred'])
        # output_image.save("grefcoco_pred.jpg")
        # exit(0)

        _pred_masks = mask_to_rle(_pred_masks)
        prediction = {'gt_masks': rle, 'pred_masks': _pred_masks[0]}
        with open(f"./temp_save/grefcoco/{case_id}.json", 'w') as f:
            json.dump(prediction, f)


if __name__ == "__main__":
    main()
    metric()
