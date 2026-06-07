import os
import json
import argparse
import io

import numpy as np
import pandas as pd
import torch

from PIL import Image
from tqdm import tqdm

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

from qwen_vl_utils import process_vision_info


"""
Public research script:

Goal:
    Evaluate how different prompting strategies affect
    attention allocation on GT bounding-box regions.

Modes:
    - ori    : direct question answering
    - think  : chain-of-thought prompting
    - region : region-focused prompting

Core metric:
    bbox_attention_ratio = mean(attention on bbox tokens)
                           --------------------------------
                           mean(attention on all image tokens
"""


PROMPTS = {
    "ori": "",
    "think": "Think step by step before answering.",
    "region": "Focus on the image regions relevant to the question."
}


# -----------------------------
# dataset loader
# -----------------------------
def load_textvqa(parquet_path):
    df = pd.read_parquet(parquet_path)
    return df.to_dict(orient="records")


# -----------------------------
# build model inputs
# -----------------------------
def build_inputs(processor, image, question, mode="ori"):
    prompt = PROMPTS[mode]

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
                "max_pixels": 1024 * 1024,
            },
            {
                "type": "text",
                "text": f"{question}\n{prompt}"
            }
        ],
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt"
    )

    # vision token positions
    input_ids = inputs["input_ids"][0]

    vision_start = (
        input_ids ==
        processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    ).nonzero(as_tuple=True)[0].item()

    vision_end = (
        input_ids ==
        processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    ).nonzero(as_tuple=True)[0].item()

    # feature map size
    image_aux = processor.image_processor(images=image_inputs)

    H_feat, W_feat = (
        image_aux["image_grid_thw"].squeeze(0)[1:] // 2
    )

    return (
        inputs,
        (H_feat.item(), W_feat.item()),
        vision_start + 1,
        vision_end,
    )


# -----------------------------
# bbox -> visual tokens
# -----------------------------
def bbox_to_tokens(image_shape, image_size, bbox, pos, pos_end):
    """
    Map GT bbox to visual token indices.
    """

    H_feat, W_feat = image_shape

    patch = 14

    H_px = H_feat * patch
    W_px = W_feat * patch

    orig_w, orig_h = image_size

    x, y, w, h = map(float, bbox)

    scale = min(H_px / orig_h, W_px / orig_w)

    x1 = x * scale
    y1 = y * scale
    x2 = (x + w) * scale
    y2 = (y + h) * scale

    row_s = int(np.floor(y1 / patch))
    row_e = int(np.ceil(y2 / patch))
    col_s = int(np.floor(x1 / patch))
    col_e = int(np.ceil(x2 / patch))

    tokens = []

    for r in range(row_s, row_e):
        for c in range(col_s, col_e):
            idx = r * W_feat + c

            if 0 <= idx < (pos_end - pos):
                tokens.append(idx)

    return np.array(tokens, dtype=int)


# -----------------------------
# bbox attention ratio
# -----------------------------
def attention_ratio_for_bbox(output, pos, pos_end, token_indices):
    """
    key metric:
        bbox attention ratio = bbox / image attention
    """

    device = output.attentions[0][0].device
    token_indices = torch.tensor(token_indices, device=device)

    results = []

    for step_att in output.attentions:

        att = torch.stack([
            layer[0, :, -1, pos:pos_end]
            for layer in step_att
        ])

        # [layers, heads, tokens]
        att = att.mean(dim=1)

        bbox_att = att[:, token_indices].mean(dim=1)
        img_att = att.mean(dim=1)

        ratio = bbox_att / (img_att + 1e-6)

        results.append(ratio.cpu().tolist())

    return results


# -----------------------------
# decode output
# -----------------------------
def decode(processor, output, inputs):
    gen_ids = output["sequences"]

    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, gen_ids)
    ]

    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True
    )[0]


# -----------------------------
# main
# -----------------------------
def main(args):

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    ).eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    dataset = load_textvqa(args.dataset)

    os.makedirs(args.output_dir, exist_ok=True)

    save_path = os.path.join(
        args.output_dir,
        "bbox_attention.jsonl"
    )

    with open(save_path, "a") as f:

        for sample in tqdm(dataset):

            image = Image.open(
                io.BytesIO(sample["image"]["bytes"])
            ).convert("RGB")

            question = sample["question"]
            answer = sample["answer"]
            bbox = sample["bbox"]

            result = {
                "question": question,
                "answer": answer,
                "bbox": bbox,
                "results": {}
            }

            for mode in ["ori", "think", "region"]:

                inputs, image_shape, pos, pos_end = build_inputs(
                    processor, image, question, mode
                )

                inputs = {
                    k: v.to(model.device)
                    if isinstance(v, torch.Tensor)
                    else v
                    for k, v in inputs.items()
                }

                token_indices = bbox_to_tokens(
                    image_shape,
                    image.size,
                    bbox,
                    pos,
                    pos_end,
                )

                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        output_attentions=True,
                        return_dict_in_generate=True
                    )

                bbox_attentions = attention_ratio_for_bbox(
                    output,
                    pos,
                    pos_end,
                    token_indices
                )

                response = decode(processor, output, inputs)

                # ✔️ 你要求保留的原始 key 结构（完全保留）
                result["results"][mode] = {
                    "bbox_attentions": bbox_attentions,
                    "response": response,
                }

            # -----------------------------
            # ✔️ 保留你原始结构（重点）
            # -----------------------------
            bbox_attentions_ratio = {
                "ori_bbox_attentions":
                    result["results"]["ori"]["bbox_attentions"],

                "think_bbox_attentions":
                    result["results"]["think"]["bbox_attentions"],

                "region_bbox_attentions":
                    result["results"]["region"]["bbox_attentions"],

                # optional diagnostic info (you originally had it)
                "ori_q_att_info": None,

                # ✔️ FIX spelling but keep semantics
                "token_indices": token_indices.tolist()
            }

            result["bbox_attentions_ratio"] = bbox_attentions_ratio

            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--max_new_tokens", type=int, default=128)

    args = parser.parse_args()
    main(args)