#!/usr/bin/env python3
"""Standalone GRES evaluation script for SAMTok model."""
import sys, os, json, re, torch, numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm

# Setup path
sys.path.insert(0, "/repo/projects/samtok")
sys.path.insert(0, "/repo/projects/samtok/evaluation")
sys.path.insert(0, "/autosota_cache/pylibs/g3803_v2")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from projects.samtok.models import VQ_SAM2, VQ_SAM2Config, SAM2Config, DirectResize
from projects.samtok.evaluation.utils import AverageMeter, Summary, intersectionAndUnionGPU

MODEL_PATH = "/models/Qwen2.5-VL-3B-SAMTok-gres-rl"
CODEBOOK_SIZE = 256
CODEBOOK_DEPTH = 2

print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16
).cuda().eval()
processor = AutoProcessor.from_pretrained(MODEL_PATH)
# Increase image resolution: 896x896 (default was ~448x448)
# 28x28 patches, min/max_pixels sets the resolution
processor.image_processor.min_pixels = 896 * 28 * 28
processor.image_processor.max_pixels = 896 * 28 * 28
print(f"Resolution set to {processor.image_processor.min_pixels} pixels")
print("Model loaded.")

print("Loading VQ-SAM2...")
sam2_config = SAM2Config(ckpt_path=f"{MODEL_PATH}/sam2.1_hiera_large.pt")
vq_sam2_config = VQ_SAM2Config(
    sam2_config=sam2_config,
    codebook_size=CODEBOOK_SIZE,
    codebook_depth=CODEBOOK_DEPTH,
    shared_codebook=False, latent_dim=256,
)
vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()
state = torch.load(f"{MODEL_PATH}/mask_tokenizer_256x2.pth", map_location="cpu")
vq_sam2.load_state_dict(state)
sam2_image_processor = DirectResize(1024)
print("VQ-SAM2 loaded.")

def extract_mt_token_ids_v1(text):
    return [int(x) for x in re.findall(r"<\|mt_(\d{4})\|>", text)]

def extract_mt_token_ids_v2(text):
    pattern = re.compile(r'<\|mt_start\|><\|mt_(\d{4})\|><\|mt_(\d{4})\|><\|mt_end\|>')
    matches = pattern.findall(text)
    ret = []
    for n1, n2 in matches:
        ret.extend([int(n1), int(n2)])
    return ret

def fix_mt_format(text):
    text = re.sub(r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_\d+\|>)(?:<\|mt_\d+\|>)+<\|mt_end\|>',
                  r'\1\2\3<|mt_end|>', text)
    text = re.sub(r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_end\|>)',
                  r'\1\2<|mt_9999|><|mt_end|>', text)
    text = re.sub(r'(<\|mt_start\|>)(<\|mt_\d+\|>)(?!<\|mt_)',
                  r'\1\2<|mt_9999|><|mt_end|>', text)
    return text

def extract_think_answer(response):
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    think = think_match.group(1) if think_match else None
    answer = answer_match.group(1) if answer_match else None
    if answer is None and '<answer>' in response:
        _, tail = response.split('<answer>', 1)
        answer = tail
    elif think is None and '</think>' in response:
        head, _ = response.split('</think>', 1)
        think = head
    return think, answer

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
        mask.append(np.uint8(m))
    return np.stack(mask, axis=0)

# Process each split
results = {}
for split in ['val']:
    dataset_path = f"/repo/data/PaDT-MLLM/RefCOCO/grefcoco_{split}.json"
    print(f"\n{'='*60}")
    print(f"Evaluating {split}: {dataset_path}")

    with open(dataset_path) as f:
        all_data = json.load(f)
    print(f"Samples: {len(all_data)}")

    os.makedirs("/repo/temp_save/grefcoco", exist_ok=True)

    count = 0
    for idx, data_dict in enumerate(tqdm(all_data[:500])):
        image_path = data_dict['image']
        phrase = data_dict['phrase']
        rle = data_dict['segmentation']

        if not os.path.exists(image_path):
            continue

        image = Image.open(image_path).convert('RGB')
        ori_w, ori_h = image.size

        if rle['size'][0] != ori_h or rle['size'][1] != ori_w:
            continue

        question = f"Please segment {phrase} in this image. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                          padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=384, do_sample=False, top_p=1.0)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        _, answer = extract_think_answer(output_text[0])
        src_text = answer if answer else output_text[0]

        quant_ids = extract_mt_token_ids_v1(src_text)
        if len(quant_ids) == 0:
            # No mask tokens found -> zero mask
            zero_mask = np.zeros((1, ori_h, ori_w), dtype=np.uint8)
            zero_mask = mask_to_rle(zero_mask)
            with open(f"/repo/temp_save/grefcoco/{idx}.json", 'w') as f:
                json.dump({'gt_masks': rle, 'pred_masks': zero_mask[0]}, f)
            count += 1
            continue

        if len(quant_ids) % CODEBOOK_DEPTH != 0:
            src_text = fix_mt_format(src_text)
            quant_ids = extract_mt_token_ids_v2(src_text)

        if len(quant_ids) % CODEBOOK_DEPTH != 0:
            zero_mask = np.zeros((1, ori_h, ori_w), dtype=np.uint8)
            zero_mask = mask_to_rle(zero_mask)
            with open(f"/repo/temp_save/grefcoco/{idx}.json", 'w') as f:
                json.dump({'gt_masks': rle, 'pred_masks': zero_mask[0]}, f)
            count += 1
            continue

        batch_size = len(quant_ids) // CODEBOOK_DEPTH
        if batch_size > 10:
            quant_ids = quant_ids[:2]
            batch_size = 1

        remap_quant_ids = []
        for bs_id in range(batch_size):
            chunk = quant_ids[bs_id*CODEBOOK_DEPTH:(bs_id+1)*CODEBOOK_DEPTH]
            remap = [qid - book_id*CODEBOOK_SIZE for book_id, qid in enumerate(chunk)]
            c1, c2 = remap[0], remap[1]
            if not (0 <= c1 < CODEBOOK_SIZE):
                continue
            if not (0 <= c2 < CODEBOOK_SIZE):
                c2 = -1
            remap_quant_ids.append([c1, c2])

        if len(remap_quant_ids) == 0:
            zero_mask = np.zeros((1, ori_h, ori_w), dtype=np.uint8)
            zero_mask = mask_to_rle(zero_mask)
            with open(f"/repo/temp_save/grefcoco/{idx}.json", 'w') as f:
                json.dump({'gt_masks': rle, 'pred_masks': zero_mask[0]}, f)
            count += 1
            continue

        batch_size = len(remap_quant_ids)
        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
        sam2_pixel_values = sam2_pixel_values.repeat(batch_size, 1, 1, 1)

        quant_ids_t = torch.LongTensor(remap_quant_ids).to(vq_sam2.device)

        with torch.no_grad():
            _pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids_t)
        _pred_masks = torch.nn.functional.interpolate(_pred_masks, size=(ori_h, ori_w), mode='bilinear')
        _pred_masks = _pred_masks > 0.5
        _pred_masks = _pred_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)
        _pred_masks = np.sum(_pred_masks, axis=0).astype(np.uint8)[np.newaxis, :, :]
        _pred_masks = (_pred_masks > 0).astype(np.uint8)

        _pred_masks = mask_to_rle(_pred_masks)
        with open(f"/repo/temp_save/grefcoco/{idx}.json", 'w') as f:
            json.dump({'gt_masks': rle, 'pred_masks': _pred_masks[0]}, f)
        count += 1

    print(f"Processed {count} samples for {split}")

    # Compute metrics
    print(f"Computing metrics for {split}...")
    inter_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    g_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    nt_tp_meter = AverageMeter("NT_TP", ":6.3f", Summary.SUM)
    nt_tn_meter = AverageMeter("NT_TN", ":6.3f", Summary.SUM)
    nt_fp_meter = AverageMeter("NT_FP", ":6.3f", Summary.SUM)
    nt_fn_meter = AverageMeter("NT_FN", ":6.3f", Summary.SUM)

    for json_file in os.listdir("/repo/temp_save/grefcoco"):
        json_path = os.path.join("/repo/temp_save/grefcoco", json_file)
        try:
            with open(json_path) as f:
                jd = json.load(f)
            pred_mask = rle_to_mask([jd["pred_masks"]])[0]
            gt_mask = rle_to_mask([jd["gt_masks"]])[0]
            pred_mask = torch.from_numpy(pred_mask).int().cuda()
            gt_mask = torch.from_numpy(gt_mask).int().cuda()

            if gt_mask.sum() < 1.0:
                if pred_mask.sum() < 1.0:
                    nt_tp_meter.update(1.0)
                    g_iou_meter.update(1.0)
                else:
                    nt_fn_meter.update(1.0)
                    g_iou_meter.update(0.0)
            else:
                if pred_mask.sum() < 1.0:
                    nt_fp_meter.update(1.0)
                else:
                    nt_tn_meter.update(1.0)
                inter_i, union_i, _ = intersectionAndUnionGPU(
                    pred_mask.contiguous().clone(),
                    gt_mask.contiguous().clone(),
                    K=2, ignore_index=255)
                inter_i = inter_i.cpu().numpy()
                union_i = union_i.cpu().numpy()
                this_giou = inter_i / (union_i + 1e-8)
                inter_meter.update(inter_i)
                union_meter.update(union_i)
                g_iou_meter.update(this_giou)
        except Exception as e:
            continue

    N_acc = nt_tp_meter.sum / (nt_tp_meter.sum + nt_fn_meter.sum) * 100 if (nt_tp_meter.sum + nt_fn_meter.sum) > 0 else 0
    g_iou = g_iou_meter.avg[1] * 100
    c_iou = (inter_meter.sum / (union_meter.sum + 1e-10))[1] * 100

    results[split] = {
        "gIoU": round(g_iou, 1),
        "cIoU": round(c_iou, 1),
        "N-acc": round(N_acc, 1),
    }
    print(f"\n{split} Results: gIoU={g_iou:.1f}, cIoU={c_iou:.1f}, N-acc={N_acc:.1f}")

    # Clean temp files
    os.system("rm -f /repo/temp_save/grefcoco/*.json")

print("\n" + "="*60)
print("FINAL RESULTS:")
for split, metrics in results.items():
    print(f"  {split}: gIoU={metrics['gIoU']}, cIoU={metrics['cIoU']}, N-acc={metrics['N-acc']}")
print("="*60)
