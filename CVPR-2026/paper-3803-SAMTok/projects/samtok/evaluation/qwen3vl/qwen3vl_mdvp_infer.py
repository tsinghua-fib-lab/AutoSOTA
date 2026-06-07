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
import base64
import io

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from projects.samtok.models import VQ_SAM2, VQ_SAM2Config, SAM2Config, DirectResize


def parse_args():
    parser = argparse.ArgumentParser(description='MDVP')
    parser.add_argument(
        '--model_path',
        default="zhouyik/Qwen3-VL-4B-SAMTok-dam",
        help='hf model path.')
    parser.add_argument(
        '--vq_sam2_path',
        default="zhouyik/Qwen3-VL-4B-SAMTok-dam/mask_tokenizer_256x2.pth",
        help='vq-sam2 model path.')
    args = parser.parse_args()
    return args

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

def main():
    args = parse_args()

    MT_START_TOKEN = '<|mt_start|>'
    MT_END_TOKEN = '<|mt_end|>'
    MT_CONTEXT_TOKEN = '<|mt_{}|>'

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

    # download from github repo: Grasp-Any-Region
    with open('./data/Grasp-Any-Region/evaluation/MDVP-Bench/annotations/mdvp_caption_mask.json', 'r') as f:
        eval_samples = json.load(f)
    
    model_outputs = []
    for eval_sample in eval_samples:
        image_file = eval_sample['image_path']
        image_path = os.path.join('./data/Grasp-Any-Region/evaluation/MDVP-Bench/data/', image_file)
        
        mask_anno = eval_sample['mask_rle']

        image = Image.open(image_path).convert('RGB')
        ori_width, ori_height = image.size

        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

        binary_masks = decode_mask([mask_anno], ori_height, ori_width)

        masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in binary_masks])

        boxes = torchvision.ops.masks_to_boxes(masks)
        x1, y1, x2, y2 = boxes.squeeze().cpu().numpy().tolist()
        boxes_w = x2 - x1
        boxes_h = y2 - y1
        boxes_area = boxes_h * boxes_w
        image_area = ori_height * ori_width
        boxes_occupied_ratio = boxes_area / image_area

        whwh = torch.as_tensor([[ori_width, ori_height, ori_width, ori_height]])
        boxes = boxes / whwh
        boxes = boxes.to(vq_sam2.device)
        masks = [m.unsqueeze(0).to(vq_sam2.device) for m in masks]
        
        with torch.no_grad():
            vq_sam2_output = vq_sam2(
                sam2_pixel_values,
                masks,
                boxes,
                reconstruct_mask=False,
            )

        quant_codes = vq_sam2_output.quant_codes.squeeze().cpu().numpy().astype(np.int32).tolist()
        remap_quant_codes = [depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(quant_codes)]
        quant_codes = remap_quant_codes
        global_mask_tokens_str = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in quant_codes]) + MT_END_TOKEN

        if boxes_occupied_ratio < 0.2:
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            if bbox_w < 140:
                x1 = x1 - (140 - bbox_w) // 2
                x2 = x2 + (140 - bbox_w) // 2
            if bbox_h < 140:
                y1 = y1 - (140 - bbox_h) // 2
                y2 = y2 + (140 - bbox_h) // 2
            x1 = int(max(0, x1))
            x2 = int(min(ori_width, x2))
            y1 = int(max(0, y1))
            y2 = int(min(ori_height, y2))

            cropped_image = image.crop((x1, y1, x2, y2))
            crop_width, crop_height = cropped_image.size

            
            if crop_width > crop_height and crop_width < 280:
                ratio = 280 / crop_height
                new_height = 280
                new_width = int(crop_width * ratio)
                resized_crop_image = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            elif crop_height > crop_width and crop_height < 280:
                ratio = 280 / crop_width
                new_width = 280
                new_height = int(crop_height * ratio)
                resized_crop_image = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            elif crop_height == crop_width and crop_width < 280:
                ratio = 280 / crop_height
                new_height = 280
                new_width = int(crop_width * ratio)
                resized_crop_image = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                new_height = new_width = None
                resized_crop_image = None
                # continue

            if resized_crop_image is None:
                cropped_sam2_image = np.array(cropped_image)
                cropped_sam2_image = sam2_image_processor.apply_image(cropped_sam2_image)
                cropped_sam2_pixel_values = torch.from_numpy(cropped_sam2_image).permute(2, 0, 1).contiguous()
                cropped_sam2_pixel_values = cropped_sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
            else:
                cropped_sam2_image = np.array(resized_crop_image)
                cropped_sam2_image = sam2_image_processor.apply_image(cropped_sam2_image)
                cropped_sam2_pixel_values = torch.from_numpy(cropped_sam2_image).permute(2, 0, 1).contiguous()
                cropped_sam2_pixel_values = cropped_sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

            cropped_masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy()[y1:y2, x1:x2])) for x in binary_masks])
            assert cropped_masks.shape[-2] == crop_height and cropped_masks.shape[-1] == crop_width

            if resized_crop_image is not None:
                resized_crop_masks = torch.nn.functional.interpolate(cropped_masks.unsqueeze(0), size=(new_height, new_width), mode='bilinear')
                resized_crop_masks = resized_crop_masks[0] > 0.5
                cropped_masks = resized_crop_masks
            crop_height, crop_width = cropped_masks.shape[-2:]
            cropped_boxes = torchvision.ops.masks_to_boxes(cropped_masks)
            crop_whwh = torch.as_tensor([[crop_width, crop_height, crop_width, crop_height]])
            cropped_boxes = cropped_boxes / crop_whwh
            cropped_boxes = cropped_boxes.to(vq_sam2.device)
            cropped_masks = [m.unsqueeze(0).to(vq_sam2.device) for m in cropped_masks]

            with torch.no_grad():
                cropped_vq_sam2_output = vq_sam2(
                    cropped_sam2_pixel_values,
                    cropped_masks,
                    cropped_boxes,
                    reconstruct_mask=True,
                )
            
            crop_quant_codes = cropped_vq_sam2_output.quant_codes.squeeze().detach().cpu().numpy().astype(np.int32).tolist()
            remap_crop_quant_codes = [depth_idx*CODEBOOK_SIZE+quant_code for depth_idx, quant_code in enumerate(crop_quant_codes)]
            crop_quant_codes = remap_crop_quant_codes
            zoom_in_mask_tokens_str = MT_START_TOKEN + ''.join([MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in crop_quant_codes]) + MT_END_TOKEN
            question = "Given a detailed description of this region {SEG}. Zoom in with the perspective as ".format(SEG=global_mask_tokens_str)
            buffer = io.BytesIO()
            if resized_crop_image is None:
                cropped_image.save(buffer, format='JPEG')
            else:
                resized_crop_image.save(buffer, format='JPEG')
            buffer.seek(0)
            b64 = base64.b64encode(buffer.read()).decode("utf-8")

            with open(image_path, "rb") as f:
                global_b64 = base64.b64encode(f.read()).decode()

            print("USING ZOOM IN...")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"data:image/jpeg;base64,{global_b64}",
                        },
                        {"type": "text", "text": question},
                        {
                            "type": "image",
                            "image": f"data:image/jpeg;base64,{b64}",
                        },
                        {"type": "text", "text": f", {zoom_in_mask_tokens_str}."},
                    ],
                }
            ]
        else:
            question = "Given a detailed description of this region {SEG}.".format(SEG=global_mask_tokens_str)
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"data:image/jpeg;base64,{b64}",
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
            max_new_tokens=1024,
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

        pred_caption = output_text[0].replace('<|im_end|>', '')

        model_outputs.append(
            {
                "image_path": image_path,
                "caption": pred_caption,
                "gt": eval_sample["caption"],
            }
        )

    with open(f"result_qwen3vl_mdvp.json", "w") as file:
        json.dump(model_outputs, file, indent=4, ensure_ascii=False)

if __name__ == '__main__':     
    main()
