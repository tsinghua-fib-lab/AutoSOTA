# Modified from https://huggingface.co/spaces/PolyU-ChenLab/UniPixel/blob/main/app.py
import os
from pathlib import Path
import random
import re
import colorsys
from PIL import Image
import matplotlib as mpl
import numpy as np
import uuid
import imageio.v3 as iio
import base64
import io
import re

import torch
import torchvision
from torchvision.transforms.functional import to_pil_image
from huggingface_hub import hf_hub_download

import spaces
import gradio as gr

from transformers import SamModel, SamProcessor
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from visualizer import sample_color, draw_mask

class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))

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
    indices = np.where(arr == value)[0]
    
    return indices[0] if len(indices) > 0 else -1

def fix_mt_format_comprehensive(text):
    pattern_too_many = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_\d+\|>)(?:<\|mt_\d+\|>)+<\|mt_end\|>'
    replacement_too_many = r'\1\2\3<|mt_end|>'
    text = re.sub(pattern_too_many, replacement_too_many, text)

    pattern_too_few_with_end = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_end\|>)'
    replacement_too_few = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern_too_few_with_end, replacement_too_few, text)

    pattern_too_few_no_end = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(?!<\|mt_)'
    replacement_too_few_no_end = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern_too_few_no_end, replacement_too_few_no_end, text)
    return text

MODEL = 'zhouyik/Qwen3-VL-8B-SAMTok'

TITLE = 'SAMTok: Representing Any Mask with Two Words'

HEADER = """
<h2 align="center">SAMTok: Representing Any Mask with Two Words</h3>
<div style="display: flex; justify-content: center; gap: 5px;">
    <a href="https://github.com/bytedance/Sa2VA/tree/main/projects/samtok" target="_blank"><img src="https://img.shields.io/badge/arXiv-2509.18094-red"></a>
    <a href="https://github.com/bytedance/Sa2VA/tree/main/projects/samtok" target="_blank"><img src="https://img.shields.io/badge/Project-Page-brightgreen"></a>
    <a href="https://huggingface.co/collections/zhouyik/samtok" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue"></a>
    <a href="https://github.com/bytedance/Sa2VA" target="_blank"><img src="https://img.shields.io/github/stars/bytedance/Sa2VA"></a>
</div>
<p style="margin-top: 1em;">SAMTok provides a unified mask-token interface for MLLMs.</p>
"""

JS = """
function init() {
    if (window.innerWidth >= 1536) {
        document.querySelector('main').style.maxWidth = '1536px'
    }
    document.getElementById('query_1').addEventListener('keydown', function f1(e) { if (e.key === 'Enter') { document.getElementById('submit_1').click() } })
}
window.addEventListener('load', init);
"""

MT_START_TOKEN = '<|mt_start|>'
MT_END_TOKEN = '<|mt_end|>'
MT_CONTEXT_TOKEN = '<|mt_{}|>'

# build vq-sam2 model
vq_sam2 = None
sam2_image_processor = DirectResize(1024)
sam2_ckpt_local = hf_hub_download(repo_id=MODEL, filename="sam2.1_hiera_large.pt")
mask_tokenizer_local = hf_hub_download(repo_id=MODEL, filename="mask_tokenizer_256x2.pth")
CODEBOOK_SIZE = 256
CODEBOOK_DEPTH = 2
def load_vq_sam2():
    global vq_sam2

    if vq_sam2 is not None:
        return vq_sam2
    
    if hasattr(torch, "set_default_device"):
        torch.set_default_device("cpu")
    
    sam2_config = SAM2Config(
        ckpt_path=sam2_ckpt_local,
    )
    vq_sam2_config = VQ_SAM2Config(
        sam2_config=sam2_config,
        codebook_size=CODEBOOK_SIZE,
        codebook_depth=CODEBOOK_DEPTH,
        shared_codebook=False,
        latent_dim=256,
    )

    vq_sam2 = VQ_SAM2(vq_sam2_config)
    state = torch.load(mask_tokenizer_local, map_location="cpu")
    vq_sam2.load_state_dict(state)

    vq_sam2 = vq_sam2.cuda().eval()
    return vq_sam2

processor = AutoProcessor.from_pretrained(MODEL)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

_qwen = None
_sam = None

def get_qwen():
    """Must be called only inside @spaces.GPU function."""
    global _qwen
    if _qwen is None:
        _qwen = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype="auto").to("cuda").eval()
    return _qwen

def get_sam():
    """Must be called only inside @spaces.GPU function."""
    global _sam
    if _sam is None:
        _sam = SamModel.from_pretrained("facebook/sam-vit-huge").to("cuda").eval()
    return _sam

colors = sample_color()
color_map = {f'Target {i + 1}': f'#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}' for i, c in enumerate(colors * 255)}
color_map_light = {
    f'Target {i + 1}': f'#{int(c[0] * 127.5 + 127.5):02x}{int(c[1] * 127.5 + 127.5):02x}{int(c[2] * 127.5 + 127.5):02x}'
    for i, c in enumerate(colors)
}

def enable_btns():
    return (gr.update(interactive=True), ) * 4

def disable_btns():
    return (gr.update(interactive=False), ) * 4

def reset_seg():
    return 16, gr.update(interactive=False)

def reset_reg():
    return 1, gr.update(interactive=False)

def new_mu_state():
    return {
        "image_path": None,
        "ori_size": None,                 # (w, h)
        "original_sizes": None,           # e.g. [h, w]
        "reshaped_input_sizes": None,     # e.g. [h', w']
        "image_embeddings": None,         # numpy array on CPU
        "points": [],
        "labels": [],
        "cur_mask": None,                 # np.uint8 (H,W)
        "regions": {},
        "next_region_id": 1,
    }

@spaces.GPU
def mu_on_upload_image(media_path, mu_state):
    if not media_path:
        return new_mu_state(), None, None

    sam_model = get_sam()  # GPU-side

    img = Image.open(media_path).convert("RGB")
    w, h = img.size

    inputs = sam_processor(img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        emb = sam_model.get_image_embeddings(inputs["pixel_values"])  # CUDA tensor

    st = new_mu_state()
    st["image_path"] = media_path
    st["ori_size"] = (w, h)

    # store sizes as python lists (not tensors)
    st["original_sizes"] = inputs["original_sizes"][0].detach().cpu().tolist()
    st["reshaped_input_sizes"] = inputs["reshaped_input_sizes"][0].detach().cpu().tolist()

    # store embeddings as CPU numpy (picklable)
    st["image_embeddings"] = emb[0].detach().cpu().to(torch.float16).numpy()  # (256,64,64)

    return st, media_path, None

def mu_predict_mask_from_state(mu_state):
    if mu_state["image_embeddings"] is None or mu_state["image_path"] is None:
        return None
    if len(mu_state["points"]) == 0:
        return None

    sam_model = get_sam()

    img = Image.open(mu_state["image_path"]).convert("RGB")

    prompt_inputs = sam_processor(
        img,
        input_points=[mu_state["points"]],
        input_labels=[mu_state["labels"]],
        return_tensors="pt",
    ).to("cuda")

    # restore embedding to CUDA tensor, shape (1,256,64,64)
    emb = torch.from_numpy(mu_state["image_embeddings"]).to("cuda")
    emb = emb.unsqueeze(0)

    with torch.no_grad():
        outputs = sam_model(
            image_embeddings=emb,
            input_points=prompt_inputs["input_points"],
            input_labels=prompt_inputs["input_labels"],
            multimask_output=False,
        )

    # postprocess needs lists/tensors on CPU
    original_sizes = torch.tensor([mu_state["original_sizes"]], dtype=torch.long)
    reshaped_sizes = torch.tensor([mu_state["reshaped_input_sizes"]], dtype=torch.long)
    masks = sam_processor.post_process_masks(
        outputs.pred_masks.detach().cpu(),
        original_sizes,
        reshaped_sizes,
    )
    mask = masks[0][0][0].numpy()
    mask = (mask > 0).astype(np.float32)
    return mask

@spaces.GPU
def mu_add_point(evt: gr.SelectData, mu_state, is_positive: bool):
    if mu_state["image_path"] is None:
        return mu_state, None

    x, y = evt.index
    mu_state["points"].append([float(x), float(y)])
    mu_state["labels"].append(1 if is_positive else 0)

    mask = mu_predict_mask_from_state(mu_state)
    mu_state["cur_mask"] = mask
    return mu_state, mask

@spaces.GPU
def mu_add_point_xy(xy, mu_state, is_positive: bool):
    if mu_state["image_path"] is None:
        return mu_state, None

    if xy is None:
        return mu_state, mu_state.get("cur_mask")

    x, y = xy  # xy is a tuple/list of two ints
    mu_state["points"].append([float(x), float(y)])
    mu_state["labels"].append(1 if is_positive else 0)

    mask = mu_predict_mask_from_state(mu_state)
    mu_state["cur_mask"] = mask
    return mu_state, mask

def mu_evt_to_xy(evt: gr.SelectData):
    # return plain python types only (picklable)
    x, y = evt.index
    return (int(x), int(y))

def mu_clear_prompts(mu_state):
    mu_state["points"] = []
    mu_state["labels"] = []
    mu_state["cur_mask"] = None
    return mu_state, None

@spaces.GPU
def mu_save_region(mu_state):
    if mu_state["cur_mask"] is None:
        return mu_state, gr.update(choices=[], value=None)

    rid = f"region{mu_state['next_region_id']}"
    mu_state["next_region_id"] += 1

    reg = {"mask": mu_state["cur_mask"], "token_str": None, "zoom_in_token_str": None, "zoom_in_image": None}

    vq_sam2 = load_vq_sam2()

    image = Image.open(mu_state["image_path"]).convert('RGB')
    ori_width, ori_height = image.size

    sam2_image = np.array(image)
    sam2_image = sam2_image_processor.apply_image(sam2_image)
    sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
    sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)

    masks = torch.stack([torch.from_numpy(np.ascontiguousarray(mu_state["cur_mask"].copy()))])

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

    reg["token_str"] = global_mask_tokens_str

    if boxes_occupied_ratio < 0.3:
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

        cropped_masks = torch.stack([torch.from_numpy(np.ascontiguousarray(mu_state["cur_mask"].copy()[y1:y2, x1:x2]))])
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

        buffer = io.BytesIO()
        if resized_crop_image is None:
            cropped_image.save(buffer, format='JPEG')
        else:
            resized_crop_image.save(buffer, format='JPEG')
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")

        reg["zoom_in_token_str"] = zoom_in_mask_tokens_str
        reg["zoom_in_image"] = b64

    mu_state["regions"][rid] = reg
    choices = list(mu_state["regions"].keys())
    return mu_state, gr.update(choices=choices, value=rid)

def replace_region_all(text: str, rid: str, token_str: str) -> str:
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(rid)}(?![A-Za-z0-9_])")
    return pattern.sub(f"{rid} {token_str}", text)

def short_tag_from_codes(code_a: int, code_b: int) -> str:
    return f"<{code_a:04d}-{code_b:04d}>"

@spaces.GPU
def infer_understanding(mu_media, mu_query, mu_state):
    model = get_qwen()

    if not mu_media:
        gr.Warning("Please upload an image")
        return ""
    if not mu_query:
        gr.Warning("Please provide a text prompt.")
        return ""
    
    raw_query = mu_query

    # 1) find which regions are referenced in the ORIGINAL query
    used = []
    for rid in mu_state["regions"].keys():
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(rid)}(?![A-Za-z0-9_])", raw_query):
            used.append(rid)

    # 2) replace ALL occurrences for each used rid
    for rid in used:
        reg = mu_state["regions"][rid]
        token_str = reg.get("token_str")
        if token_str:
            mu_query = replace_region_all(mu_query, rid, token_str)

    content = [{"type": "image", "image": mu_media}]
    content.append({"type": "text", "text": mu_query})

    # 3) zoom-in blocks only for used regions
    for rid in used:
        reg = mu_state["regions"][rid]
        zoom_in_image = reg.get("zoom_in_image")
        zoom_in_token_str = reg.get("zoom_in_token_str")
        if zoom_in_image and zoom_in_token_str:
            content.append({"type": "text", "text": f" Zoom in {rid}: "})
            content.append({"type": "image", "image": f"data:image/jpeg;base64,{zoom_in_image}"})
            content.append({"type": "text", "text": f", {zoom_in_token_str}."})
    
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.8,
        top_k=20,
        temperature=0.7,
        repetition_penalty=1.0,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

@spaces.GPU
def infer_seg(media, query):
    model = get_qwen()
    vq_sam2 = load_vq_sam2()

    if not media:
        gr.Warning('Please upload an image')
        return None, None, None

    if not query:
        gr.Warning('Please provide a text prompt.')
        return None, None, None

    image = Image.open(media).convert('RGB')
    ori_width, ori_height = image.size
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": media,
                },
                {"type": "text", "text": query},
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

    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.8,
        top_k=20,
        temperature=0.7,
        repetition_penalty=1.0,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    quant_ids = extract_mt_token_ids_v1(output_text)
    if len(quant_ids) == 0:
        # only show model response; hide masks & download
        answer = dict(text=output_text, entities=[])
        return (
            answer,
            gr.update(value=None, visible=False),                      # hide AnnotatedImage
            gr.update(value=None, interactive=False, visible=False),   # hide DownloadButton
        )
    
    if len(quant_ids) % CODEBOOK_DEPTH != 0:
        output_text = fix_mt_format_comprehensive(output_text)
        quant_ids = extract_mt_token_ids_v2(output_text)

    if len(quant_ids) == 0 or (len(quant_ids) % CODEBOOK_DEPTH != 0):
        answer = dict(text=output_text, entities=[])
        return (
            answer,
            gr.update(value=None, visible=False),
            gr.update(value=None, interactive=False, visible=False),
        )

    batch_size = len(quant_ids) // CODEBOOK_DEPTH
    remap_quant_ids = []
    tags = []
    for bs_id in range(batch_size):
        chunk_quant_ids = quant_ids[bs_id*CODEBOOK_DEPTH:(bs_id+1)*CODEBOOK_DEPTH]
        tags.append(f'<|mt_{str(chunk_quant_ids[0]).zfill(4)}|><|mt_{str(chunk_quant_ids[1]).zfill(4)}|>')
        remap_chunk_quant_ids = [quant_id - book_id*CODEBOOK_SIZE for book_id, quant_id in enumerate(chunk_quant_ids)]
        code1 = remap_chunk_quant_ids[0]
        code2 = remap_chunk_quant_ids[1]
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
    _pred_masks = _pred_masks.long().unsqueeze(2).cpu() # n, 1, 1, h, w 

    tag_to_mask_idx = {}
    for i, tag in enumerate(tags):
        if tag not in tag_to_mask_idx:
            tag_to_mask_idx[tag] = i
    unique_tags = list(tag_to_mask_idx.keys())

    entities = []
    for tag in unique_tags:
        for m in re.finditer(re.escape(tag), output_text):
            entities.append(dict(entity=tag, start=m.start(), end=m.end()))

    answer = dict(text=output_text, entities=entities)

    frames = torch.from_numpy(np.array(image)).unsqueeze(0)
    imgs = draw_mask(frames, _pred_masks, colors=colors)

    path = f"/tmp/{uuid.uuid4().hex}.png"
    iio.imwrite(path, imgs, duration=100, loop=0)

    mask_items = []
    entity_names = unique_tags
    for i, tag in enumerate(unique_tags):
        m = _pred_masks[tag_to_mask_idx[tag]][0, 0].numpy()
        mask_items.append((m, entity_names[i]))
    masks_value = (media, mask_items)

    # return answer, masks, path
    return (
        gr.update(value=answer, visible=True),  # ans_1
        gr.update(value=masks_value, visible=True),   # msk_1
        gr.update(value=path, interactive=True, visible=True),                 # download
    )

def build_demo():
    with gr.Blocks(title=TITLE, js=JS, theme=gr.themes.Soft()) as demo:
        gr.HTML(HEADER)

        with gr.Tab('Mask Generation'):
            download_btn_1 = gr.DownloadButton(label='📦 Download', interactive=False, render=False)
            msk_1 = gr.AnnotatedImage(label='De-tokenized 2D masks', color_map=color_map, render=False)
            ans_1 = gr.HighlightedText(
                label='Model Response', color_map=color_map_light, show_inline_category=False, render=False)
            with gr.Row():
                with gr.Column():
                    media_1 = gr.Image(type='filepath')

                    sample_frames_1 = gr.Slider(1, 32, value=16, step=1, visible=False)

                    query_1 = gr.Textbox(
                        label='Text Prompt',
                        placeholder='Please segment the...',
                        lines=3,
                        max_lines=12,
                        elem_id='query_1'
                    )

                    with gr.Row():
                        random_btn_1 = gr.Button(value='🔮 Random', visible=False)

                        reset_btn_1 = gr.ClearButton([media_1, query_1, msk_1, ans_1], value='🗑️ Reset')
                        reset_btn_1.click(reset_seg, None, [sample_frames_1, download_btn_1])

                        download_btn_1.render()

                        submit_btn_1 = gr.Button(value='🚀 Submit', variant='primary', elem_id='submit_1')
                
                with gr.Column():
                    msk_1.render()
                    ans_1.render()

            ctx_1 = submit_btn_1.click(disable_btns, None, [random_btn_1, reset_btn_1, download_btn_1, submit_btn_1])
            ctx_1 = ctx_1.then(infer_seg, [media_1, query_1], [ans_1, msk_1, download_btn_1])
            ctx_1.then(enable_btns, None, [random_btn_1, reset_btn_1, download_btn_1, submit_btn_1])

            EXAMPLES = [
                ["examples/example1.jpeg", "Locate the tissue box in this image and response with its segmentation mask."],
                ["examples/example2.jpg", "Could you please give me a detail description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer."],
                ["examples/example3.png", "Find all the people who are currently standing and response with segmentation masks."],
                ["examples/example4.jpg", "Segment every instance that belongs to the following categories: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush, banner, blanket, bridge, cardboard, counter, curtain, door-stuff, floor-wood, flower, fruit, gravel, house, light, mirror-stuff, net, pillow, platform, playingfield, railroad, river, road, roof, sand, sea, shelf, snow, stairs, tent, towel, wall-brick, wall-stone, wall-tile, wall-wood, water-other, window-blind, window-other, tree-merged, fence-merged, ceiling-merged, sky-other-merged, cabinet-merged, table-merged, floor-other-merged, pavement-merged, mountain-merged, grass-merged, dirt-merged, paper-merged, food-other-merged, building-other-merged, rock-merged, wall-other-merged, rug-merged"],
                ["examples/example5.jpg", "Generate a scene graph for this image. Identify the main objects and describe their relationships to each other."],
                ["examples/example6.jpg", "What item for sale indicates that the primary product is also offered in a ready-to-eat form? A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"]
            ]
            gr.Markdown("## Examples")
            gr.Examples(
                examples=EXAMPLES,
                inputs=[media_1, query_1],
                label="Click an example to load the image and prompt",
            )
        with gr.Tab("Mask Understanding"):
            MU_INSTRUCTIONS = """
            ### Mask Understanding — Instructions

            1. **Upload an image.**
            2. **Create a region mask**
            - Click **Clear Prompts**
            - Click **Positive Point**, then click on the target region in the image.
            - The **Current Mask** preview updates after each click. Add more clicks to refine the mask.
            - Click **Save Region** to store the current mask. A new region ID (e.g., `region1`) will be created.
            3. *(Optional)* Repeat Step 2 to add more regions.
            4. **Enter a text prompt.** When referring to a saved region, use its exact auto-generated ID (e.g., `region1`), e.g. `Given a detailed description of region1.`
            You can reference multiple regions, e.g. `Compare region1 and region2 and describe their differences.`

            **Tips:** Use **Negative Point** to remove unwanted parts; use **Clear Prompts** to reset points.
            """
            with gr.Accordion("Instructions (click to expand)", open=False):
                gr.Markdown(MU_INSTRUCTIONS)
            mu_click_xy = gr.State(None)
            mu_state = gr.State(new_mu_state())
            mu_point_is_pos = gr.State(True)

            with gr.Row():
                with gr.Column():
                    mu_media = gr.Image(type="filepath", label="Upload Image")
                    mu_click_img = gr.Image(label="Click to add points", interactive=True)

                    with gr.Row():
                        mu_pos_btn = gr.Button("Positive Point")
                        mu_neg_btn = gr.Button("Negative Point")
                        mu_clear_btn = gr.Button("Clear Prompts")
                        mu_save_btn = gr.Button("Save Region")

                    mu_region_dd = gr.Dropdown(label="Saved Regions", choices=[], interactive=True)

                    mu_query = gr.Textbox(label="Text Prompt", lines=3, max_lines=12)
                    mu_submit = gr.Button("Submit", variant="primary")

                with gr.Column():
                    mu_mask_preview = gr.Image(label="Current Mask")
                    mu_answer = gr.Textbox(label="Model Response", lines=12)

            mu_media.change(
                fn=mu_on_upload_image,
                inputs=[mu_media, mu_state],
                outputs=[mu_state, mu_click_img, mu_mask_preview],
            )

            mu_pos_btn.click(lambda: True, None, mu_point_is_pos)
            mu_neg_btn.click(lambda: False, None, mu_point_is_pos)

            # mu_click_img.select(
            #     fn=mu_add_point,
            #     inputs=[mu_state, mu_point_is_pos],
            #     outputs=[mu_state, mu_mask_preview],
            # )

            mu_click_img.select(
                fn=mu_evt_to_xy,
                inputs=None,
                outputs=mu_click_xy,
                queue=False,
            ).then(
                fn=mu_add_point_xy,
                inputs=[mu_click_xy, mu_state, mu_point_is_pos],
                outputs=[mu_state, mu_mask_preview],
            )

            mu_clear_btn.click(mu_clear_prompts, [mu_state], [mu_state, mu_mask_preview])

            mu_save_btn.click(mu_save_region, [mu_state], [mu_state, mu_region_dd])

            mu_submit.click(
                fn=infer_understanding,
                inputs=[mu_media, mu_query, mu_state],
                outputs=[mu_answer],
            )

            EXAMPLES_MU = [
                ["examples/example1.jpeg"],
                ["examples/example2.jpg"],
                ["examples/example3.png"],
                ["examples/example4.jpg"],
                ["examples/example5.jpg"],
                ["examples/example6.jpg"],
            ]

            gr.Markdown("## Examples")
            gr.Examples(
                examples=EXAMPLES_MU,
                inputs=[mu_media],  # only load image
                label="Click an example to load the image",
            )

    return demo

if __name__ == '__main__':
    demo = build_demo()

    demo.queue()
    demo.launch()