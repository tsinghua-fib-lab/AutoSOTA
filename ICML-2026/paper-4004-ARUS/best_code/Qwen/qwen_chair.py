#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_qwenvl_chair.py — Qwen-VL CHAIR 实验脚本 (带CSV结果汇总)

使用 Qwen2.5-VL 作为多模态模型，对 COCO val2014 跑 CHAIR benchmark：
- Baseline caption
- CARD + Beta Gating
- CARD + Simple Add (Ablation)

假定：
- 只用 image 输入，不使用 video；
- COCO 图片放在 COCO_IMG_DIR，instances/captions JSON 在 COCO_INSTANCES_JSON 同目录。
"""

import os, json, argparse, math, random, re, sys, pickle, csv
from typing import List, Dict, Tuple
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    AutoProcessor,
    AutoModelForVision2Seq,
)

# ====== config & methods ======
from config import *
from methods_decodings_qwen import (
    set_global_seed,
    mask_carrier,
    BayesianGatingHookMaskedDynamic,
    get_qwenvl_self_attn,  # 在 methods_decodings_qwen 里定义好的 Qwen 层路径函数
)

# ------------------------------------------------------------------------------------
# 兼容性 & 缺省值
# ------------------------------------------------------------------------------------
if not hasattr(mask_carrier, "get"):
    def _mc_get(): return getattr(mask_carrier, "mask", None)
    mask_carrier.get = _mc_get

RESULTS_DIR_CHAIR   = globals().get("RESULTS_DIR_CHAIR", os.path.join("results", "chair_qwenvl_final"))
CACHE_DIR_CHAIR     = globals().get("CACHE_DIR_CHAIR", os.path.join("cache", "chair_evaluator"))
BATCH_SIZE          = globals().get("BATCH_SIZE", 4)
NUM_WORKERS         = globals().get("NUM_WORKERS", 0)
SEEDS               = globals().get("SEEDS", [42])
EGR_POOLINGS        = globals().get("EGR_POOLINGS", ["attn"])
INJECTION_LAYERS    = globals().get("INJECTION_LAYERS", [24,25,26])
BETA_ALPHA_MAX      = globals().get("BETA_ALPHA_MAX", [5.0, 8.0])
BETA_K              = globals().get("BETA_K", [3.0, 5.0])
BETA_C              = globals().get("BETA_C", [0.5, 1.0])
ADD_ALPHA           = globals().get("ADD_ALPHA", [50])
GATE_CLAMP          = globals().get("GATE_CLAMP", (0.05, 1))
CAP_MAX_NEW_TOKENS  = globals().get("MAX_NEW_TOKENS_CAP", 512)

# 🔴 模型 ID：可改为你自己的本地目录
MODEL_ID            = globals().get("MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")

CACHE_DIR           = globals().get("CACHE_DIR", None)
DEVICE              = globals().get("DEVICE", "cuda:0")
DTYPE               = globals().get("DTYPE", "bf16")
COCO_IMG_DIR        = globals().get("COCO_IMG_DIR", "val2014")
COCO_INSTANCES_JSON = globals().get("COCO_INSTANCES_JSON", os.path.join("dataset", "instances_val2014.json"))

# ------------------------------------------------------------------------------------
# CSV 日志记录模块
# ------------------------------------------------------------------------------------
CSV_HEADER = [
    "timestamp", "method", "seed", "decoding", "decoding_params", "limit",
    "layer", "pooling", "alpha_max", "k", "c", "alpha",
    "CHAIRs", "CHAIRi", "n_images", "n_mentions"
]

def log_to_csv(csv_filepath: str, result_data: Dict):
    """将单次实验的结果和参数追加写入到CSV文件中。"""
    file_exists = os.path.isfile(csv_filepath)
    with open(csv_filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        row_to_write = {header: result_data.get(header, '') for header in CSV_HEADER}
        writer.writerow(row_to_write)

# ------------------------------------------------------------------------------------
# Hybrid CHAIR Evaluator
# ------------------------------------------------------------------------------------
class HybridCHAIREvaluator:
    def __init__(self, instances_json_path: str, cache_dir: str):
        self.cache_path = os.path.join(cache_dir, "hybrid_chair_gt_cache.pkl")
        self.annotation_dir = os.path.dirname(instances_json_path)
        self._setup_mention_extractor()

        if os.path.exists(self.cache_path):
            print(f"Loading Hybrid CHAIR ground truth from cache: {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                cached_data = pickle.load(f)
            self.imgid_to_objects = cached_data["imgid_to_objects"]
            self.fname_to_imgid = cached_data["fname_to_imgid"]
        else:
            print("Building Hybrid CHAIR ground truth for the first time (will be cached)...")
            self.imgid_to_objects = defaultdict(set)
            self._build_gt_from_segments()
            self._build_gt_from_captions()
            with open(instances_json_path, "r") as f:
                coco_instances = json.load(f)
            self.fname_to_imgid = {img['file_name']: img['id'] for img in coco_instances['images']}
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(
                    {"imgid_to_objects": self.imgid_to_objects, "fname_to_imgid": self.fname_to_imgid}, f
                )
            print(f"Ground truth built and cached to {self.cache_path}")

    def _setup_mention_extractor(self):
        raw = {
            "person":["person","people","man","woman","boy","girl"],
            "bicycle":["bicycle","bike"],
            "car":["car","cars","automobile","auto"],
            "motorcycle":["motorcycle","motorbike"],
            "airplane":["airplane","plane","jet","aircraft"],
            "bus":["bus","coach"],
            "train":["train","locomotive"],
            "truck":["truck","lorry"],
            "boat":["boat","ship","vessel"],
            "traffic light":["traffic light","stoplight","signal light"],
            "fire hydrant":["fire hydrant","hydrant"],
            "stop sign":["stop sign"],
            "parking meter":["parking meter"],
            "bench":["bench"],
            "bird":["bird"],
            "cat":["cat","kitty","kitten"],
            "dog":["dog","puppy"],
            "horse":["horse","pony"],
            "sheep":["sheep","lamb"],
            "cow":["cow","cattle"],
            "elephant":["elephant"],
            "bear":["bear","teddy bear","teddy"],
            "zebra":["zebra"],
            "giraffe":["giraffe"],
            "backpack":["backpack","pack"],
            "umbrella":["umbrella","brolly"],
            "handbag":["handbag","purse","bag"],
            "tie":["tie","necktie"],
            "suitcase":["suitcase","luggage"],
            "frisbee":["frisbee","flying disc"],
            "skis":["skis","ski"],
            "snowboard":["snowboard"],
            "sports ball":["sports ball","ball"],
            "kite":["kite"],
            "baseball bat":["baseball bat","bat"],
            "baseball glove":["baseball glove","mitt"],
            "skateboard":["skateboard"],
            "surfboard":["surfboard"],
            "tennis racket":["tennis racket","racket","racquet"],
            "bottle":["bottle"],
            "wine glass":["wine glass","goblet"],
            "cup":["cup","mug"],
            "fork":["fork"],
            "knife":["knife"],
            "spoon":["spoon"],
            "bowl":["bowl"],
            "banana":["banana"],
            "apple":["apple"],
            "sandwich":["sandwich"],
            "orange":["orange"],
            "broccoli":["broccoli"],
            "carrot":["carrot"],
            "hot dog":["hot dog"],
            "pizza":["pizza"],
            "donut":["donut","doughnut"],
            "cake":["cake"],
            "chair":["chair","seat"],
            "couch":["couch","sofa"],
            "potted plant":["potted plant","plant pot"],
            "bed":["bed"],
            "dining table":["dining table","table"],
            "toilet":["toilet","wc","restroom"],
            "tv":["tv","television","monitor","tv monitor"],
            "laptop":["laptop","notebook computer"],
            "mouse":["mouse","computer mouse"],
            "remote":["remote","remote control"],
            "keyboard":["keyboard"],
            "cell phone":["cell phone","mobile phone","phone","smartphone"],
            "microwave":["microwave","microwave oven"],
            "oven":["oven","stove oven"],
            "toaster":["toaster"],
            "sink":["sink","basin"],
            "refrigerator":["refrigerator","fridge"],
            "book":["book","books"],
            "clock":["clock"],
            "vase":["vase"],
            "scissors":["scissors","shears"],
            "hair drier":["hair drier","hair dryer"],
            "toothbrush":["toothbrush","tooth brush"],
        }
        alias2canon = {}
        for canon, aliases in raw.items():
            s = set()
            for a in aliases:
                s.add(a.lower())
                if not a.endswith("s"):
                    s.add((a + "s").lower())
            for alias in s:
                alias2canon[alias] = canon
        aliases = sorted(alias2canon.keys(), key=lambda s: (-len(s.split()), -len(s)))
        self.regex_patterns = []
        for a in aliases:
            p = r"\b" + re.escape(a).replace(r"\ ", r"\s+") + r"\b"
            self.regex_patterns.append((re.compile(p, flags=re.IGNORECASE), alias2canon[a]))

    def _mentions_from_caption(self, caption: str) -> set:
        cap_lower = (caption or "").lower()
        found = set()
        for rgx, canon in self.regex_patterns:
            if rgx.search(cap_lower):
                found.add(canon)
        return found
        
    def _build_gt_from_segments(self):
        path = os.path.join(self.annotation_dir, 'instances_val2014.json')
        with open(path, "r") as f:
            coco_segments = json.load(f)
        id_to_name = {cat['id']: cat['name'] for cat in coco_segments['categories']}
        for ann in tqdm(coco_segments['annotations'], desc="  - GT from segments", ncols=100, leave=False):
            self.imgid_to_objects[ann['image_id']].add(
                id_to_name.get(ann['category_id'], "").lower()
            )

    def _build_gt_from_captions(self):
        path = os.path.join(self.annotation_dir, 'captions_val2014.json')
        with open(path, "r") as f:
            coco_caps = json.load(f)
        for ann in tqdm(coco_caps['annotations'], desc="  - GT from captions", ncols=100, leave=False):
            self.imgid_to_objects[ann['image_id']].update(
                self._mentions_from_caption(ann['caption'])
            )

    def evaluate(self, file_names: List[str], captions: List[str]) -> Dict:
        total_mentions, total_hallu, sent_hallu = 0, 0, 0
        for fn, cap in zip(file_names, captions):
            img_id = self.fname_to_imgid.get(fn)
            if img_id is None:
                continue
            M = self._mentions_from_caption(cap)
            G = self.imgid_to_objects.get(img_id, set())
            H = M - G
            total_mentions += len(M)
            total_hallu += len(H)
            if len(H) > 0:
                sent_hallu += 1
        n_img = len(file_names)
        return {
            "n_images": n_img,
            "n_mentions": total_mentions,
            "hallucinated_mentions": total_hallu,
            "images_with_hallucination": sent_hallu,
            "CHAIRi": total_hallu / max(1, total_mentions),
            "CHAIRs": sent_hallu / max(1, n_img),
        }

# ------------------------------------------------------------------------------------
# Qwen-VL CARD: same prompts, Qwen self-attn
# ------------------------------------------------------------------------------------
def build_caption_prompt() -> str:
    """
    按 Qwen2.5-VL 官方推荐，手写 chat 模板 + image 占位符：
    - system：设定角色
    - user：包含视觉 token <|vision_start|><|image_pad|><|vision_end|> + 指令
    - assistant：让模型从这里开始生成
    """
    system_prompt = "You are a helpful assistant that describes images in detail."
    user_instruction = "Please describe the image in detail."
    return (
        "<|im_start|>system\n"
        f"{system_prompt}"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        f"{user_instruction}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

@torch.no_grad()
def card_from_same_prompts_qwenvl(
    model,
    processor,
    images,
    prompts,
    layer_idx: int = 24,
    pooling: str = "attn",
):
    """
    使用与生成时相同的 caption prompt，在 Qwen-VL 某层上抽取 CARD 向量。
    这里只使用 image，不处理 video。
    """
    # ⚠️ 注意：images 是一层 list[Image]，不能嵌套成 [[im]]
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    tgt = get_qwenvl_self_attn(model, layer_idx)

    before = after = None

    def _pre(_m, args, kwargs):
        nonlocal before
        hs = None
        if kwargs is not None:
            hs = kwargs.get("hidden_states", None)
        if hs is None and args:
            hs = args[0]
        if hs is not None:
            before = hs.detach()

    def _post(_m, args, output):
        nonlocal after
        out0 = output[0] if isinstance(output, (tuple, list)) else output
        after = out0.detach()

    h_pre = tgt.register_forward_pre_hook(_pre, with_kwargs=True)
    h_post = tgt.register_forward_hook(_post)

    try:
        _ = model(**inputs)
    finally:
        h_pre.remove()
        h_post.remove()

    if before is None or after is None:
        raise RuntimeError("CARD capture failed: before or after tensor is None.")

    delta = after - before  # [B, T, H]
    mask = torch.ones_like(delta[..., :1])
    attn_mask = inputs.get("attention_mask", None)
    if attn_mask is not None and attn_mask.size(1) == delta.size(1):
        mask = attn_mask.unsqueeze(-1).to(delta.dtype)

    if pooling == "mean":
        v = (delta * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
    else:
        w = delta.norm(p=2, dim=-1, keepdim=True) * mask
        v = (delta * w).sum(1) / w.sum(1).clamp(min=1e-6)

    return F.normalize(v, p=2, dim=-1).detach()

# ------------------------------------------------------------------------------------
# Simple Add Hook
# ------------------------------------------------------------------------------------
class SimpleAddHook:
    def __init__(self, alpha: float, v_batch: torch.Tensor, mask_full: torch.Tensor, num_beams: int = 1):
        self.alpha = float(alpha)
        self.v_batch = v_batch
        self.mask_full = mask_full
        self.num_beams = max(1, int(num_beams))

    def __call__(self, module, args, out):
        attn_out, *rest = out
        Bq, q_len, H = attn_out.shape

        ao32 = attn_out.float()
        v = F.normalize(self.v_batch.float(), p=2, dim=-1).to(ao32.device)
        if self.num_beams > 1 and Bq == v.size(0) * self.num_beams:
            v = v.repeat_interleave(self.num_beams, dim=0)
        v = v.unsqueeze(1).expand(Bq, q_len, H)

        m = self.mask_full.to(dtype=torch.float32, device=ao32.device)
        if self.num_beams > 1 and Bq == m.size(0) * self.num_beams:
            m = m.repeat_interleave(self.num_beams, dim=0)

        T_full = m.size(1)
        if T_full >= q_len:
            m_cur = m[:, -q_len:, :]
        else:
            m_cur = torch.cat([m, m[:, -1:, :].expand(Bq, q_len - T_full, 1)], dim=1)

        delta = self.alpha * v * m_cur
        max_token_norm = 20.0
        dnorm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True) + 1e-6
        delta = delta * torch.clamp(max_token_norm / dnorm, max=1.0)

        out32 = ao32 + delta
        out32 = torch.nan_to_num(out32, nan=0.0, posinf=0.0, neginf=0.0)

        return (out32.to(attn_out.dtype), *rest)

# ------------------------------------------------------------------------------------
# Dataset & Collate
# ------------------------------------------------------------------------------------
class CHAIRImageDataset(Dataset):
    def __init__(self, image_dir: str, instances_json: str, limit: int = 0, limit_seed: int = 42):
        with open(instances_json, "r") as f:
            coco = json.load(f)
        annotated_img_ids = {ann['image_id'] for ann in coco['annotations']}
        rows = []
        for img_info in coco['images']:
            if img_info['id'] in annotated_img_ids:
                path = os.path.join(image_dir, img_info['file_name'])
                if os.path.exists(path):
                    rows.append({"file_name": img_info['file_name'], "path": path})
        if limit and limit > 0 and len(rows) > limit:
            random.seed(limit_seed)
            rows = random.sample(rows, limit)
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(r["path"]).convert("RGB")
        return {"image": img, "file_name": r["file_name"]}

def chair_collate(batch):
    return [b["image"] for b in batch], [b["file_name"] for b in batch]

# ------------------------------------------------------------------------------------
# Baseline caption
# ------------------------------------------------------------------------------------
@torch.no_grad()
def run_once_caption(model, processor, loader, gen_kwargs: dict):
    all_caps, all_files = [], []
    for images, files in tqdm(loader, desc="  [Baseline Generation]", ncols=100, leave=False):
        prompts = [build_caption_prompt() for _ in images]
        inputs = processor(
            text=prompts,
            images=images,          # ⚠️ 一层 list[Image]，不嵌套，避免走 video 分支
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outs_ids = model.generate(**inputs, **gen_kwargs)
        # 截掉 prompt 部分，只 decode 新生成的 token
        out_txt = processor.batch_decode(
            outs_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        all_caps.extend([o.strip() for o in out_txt])
        all_files.extend(files)
    return all_caps, all_files

# ------------------------------------------------------------------------------------
# CARD + Beta Gating
# ------------------------------------------------------------------------------------
@torch.no_grad()
def run_once_beta(model, processor, loader, layer, alpha_max, k, c, pooling: str, gen_kwargs: dict):
    tgt_layer = get_qwenvl_self_attn(model, layer)
    hook = BayesianGatingHookMaskedDynamic(
        max_alpha=alpha_max,
        sensitivity=k,
        concentration=c,
        carrier=mask_carrier,
        clamp=GATE_CLAMP,
    )
    handle = tgt_layer.register_forward_hook(hook)
    hook.disable()

    all_caps, all_files = [], []
    try:
        for images, files in tqdm(
            loader,
            desc=f"  [Beta L{layer} A{alpha_max} K{k} C{c}]",
            ncols=100,
            leave=False,
        ):
            prompts = [build_caption_prompt() for _ in images]

            # 1) 抽取 CARD
            hook.disable()
            v_batch = card_from_same_prompts_qwenvl(
                model, processor, images, prompts,
                layer_idx=layer,
                pooling=pooling,
            )

            # 2) 准备生成输入（同样是 text + images）
            inputs = processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # 3) 使用 attention_mask 作为注入 mask（只在非 padding 位置注入）
            m = inputs["attention_mask"].unsqueeze(-1).to(dtype=torch.float32)
            mask_carrier.set(m.to(model.device))

            # 4) 运行带 BetaGate 的生成
            hook.set_vector(v_batch)
            hook.enabled = True

            outs_ids = model.generate(**inputs, **gen_kwargs)
            outs_txt = processor.batch_decode(
                outs_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            all_caps.extend([o.strip() for o in outs_txt])
            all_files.extend(files)
    finally:
        handle.remove()
        mask_carrier.clear()
    return all_caps, all_files

# ------------------------------------------------------------------------------------
# CARD + Simple Add
# ------------------------------------------------------------------------------------
@torch.no_grad()
def run_once_add(model, processor, loader, layer, alpha, pooling: str, gen_kwargs: dict):
    tgt_layer = get_qwenvl_self_attn(model, layer)
    all_caps, all_files = [], []
    num_beams = gen_kwargs.get("num_beams", 1)

    for images, files in tqdm(
        loader,
        desc=f"  [Add L{layer} A{alpha}]",
        ncols=100,
        leave=False,
    ):
        prompts = [build_caption_prompt() for _ in images]
        v_batch = card_from_same_prompts_qwenvl(
            model, processor, images, prompts,
            layer_idx=layer,
            pooling=pooling,
        )
        inputs = processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 同样用 attention_mask 作为 mask_full
        m = inputs["attention_mask"].unsqueeze(-1).to(dtype=torch.float32)

        add_hook = SimpleAddHook(
            alpha=alpha,
            v_batch=v_batch,
            mask_full=m.to(model.device),
            num_beams=num_beams,
        )
        handle = tgt_layer.register_forward_hook(add_hook)
        outs_ids = model.generate(**inputs, **gen_kwargs)
        handle.remove()

        outs_txt = processor.batch_decode(
            outs_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        all_caps.extend([o.strip() for o in outs_txt])
        all_files.extend(files)

    return all_caps, all_files

# ------------------------------------------------------------------------------------
# Decoding helpers
# ------------------------------------------------------------------------------------
def make_generate_kwargs(decoding: str, **kwargs) -> dict:
    common = {
        "max_new_tokens": int(kwargs.get("max_new_tokens", 512)),
        "repetition_penalty": 1.0,
    }
    if decoding == "greedy":
        return {"do_sample": False, "num_beams": 1, **common}
    if decoding == "beam":
        return {
            "do_sample": False,
            "num_beams": int(kwargs.get("num_beams", 5)),
            "length_penalty": float(kwargs.get("length_penalty", 1.0)),
            "early_stopping": bool(kwargs.get("early_stopping", False)),
            **common,
        }
    if decoding == "nucleus":
        nuc = {
            "do_sample": True,
            "temperature": float(kwargs.get("temperature", 0.9)),
            "top_p": float(kwargs.get("top_p", 0.95)),
            "num_beams": 1,
            **common,
        }
        if (top_k := kwargs.get("top_k", 0)) > 0:
            nuc["top_k"] = int(top_k)
        return nuc
    raise ValueError(f"Unknown decoding mode: {decoding}")

def decoding_tag(decoding: str, **gen_kwargs) -> str:
    d = decoding.lower()
    if d == "greedy":
        return "decG"
    if d == "beam":
        return (
            f'decB{gen_kwargs.get("num_beams",1)}'
            f'LP{gen_kwargs.get("length_penalty",1.0):g}'
            f'ES{int(gen_kwargs.get("early_stopping",False))}'
        )
    if d == "nucleus":
        base = f'decN_p{gen_kwargs.get("top_p",1.0):g}_T{gen_kwargs.get("temperature",1.0):g}'
        if gen_kwargs.get("top_k", 0) > 0:
            base += f'_k{gen_kwargs.get("top_k")}'
        return base
    return "decUNK"

# ------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Qwen-VL CHAIR experiments script with CSV logging.")
    ap.add_argument("--limit", type=int, default=0, help="Subset of images for quick testing (0=all)")
    ap.add_argument("--limit_seed", type=int, default=42)
    ap.add_argument("--run_beta", action="store_true", help="Run CARD+Beta Gating experiments")
    ap.add_argument("--run_add", action="store_true", help="Run CARD+Simple Addition ablation")
    ap.add_argument("--decoding", type=str, default="greedy", choices=["greedy", "beam", "nucleus"])
    ap.add_argument("--max_new_tokens", type=int, default=CAP_MAX_NEW_TOKENS)
    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--length_penalty", type=float, default=1.08)
    ap.add_argument("--early_stopping", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--pool", type=str, default="attn", choices=["attn", "mean"], help="Pooling for CARD vector")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR_CHAIR, exist_ok=True)
    os.makedirs(CACHE_DIR_CHAIR, exist_ok=True)

    csv_log_path = os.path.join(RESULTS_DIR_CHAIR, "experiments_summary.csv")
    print(f"Logging results to: {csv_log_path}")

    # ---- 加载 Qwen2.5-VL 模型 ----
    print(f"⏳ Loading Qwen-VL model: {MODEL_ID}")
    dtype = torch.bfloat16 if str(DTYPE).lower() == "bf16" else torch.float16

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,   # 用 torch_dtype 兼容旧版 transformers
        cache_dir=CACHE_DIR,
    ).to(DEVICE).eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
    )

    # 确保 tokenizer 有 pad_token 且使用 left padding（方便截断解码）
    if getattr(processor, "tokenizer", None) is not None:
        tok = processor.tokenizer
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"

    print("✅ Qwen-VL model & processor ready.")

    print("⏳ Initializing Hybrid CHAIR Evaluator...")
    evaluator = HybridCHAIREvaluator(
        instances_json_path=COCO_INSTANCES_JSON,
        cache_dir=CACHE_DIR_CHAIR,
    )
    print("✅ Evaluator ready.")

    # 只把有用的 decoding 参数传给 make_generate_kwargs，避免非法 flag
    arg_dict = vars(args).copy()
    arg_dict.pop("decoding", None)
    gen_kwargs = make_generate_kwargs(args.decoding, **arg_dict)
    dec_tag = decoding_tag(args.decoding, **gen_kwargs)

    base_params = {
        "decoding": args.decoding,
        "decoding_params": dec_tag,
        "limit": args.limit if args.limit > 0 else "all",
        "pooling": args.pool,
    }

    for seed in SEEDS:
        set_global_seed(seed)
        ds = CHAIRImageDataset(
            COCO_IMG_DIR,
            COCO_INSTANCES_JSON,
            limit=args.limit,
            limit_seed=args.limit_seed,
        )
        loader = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=chair_collate,
        )
        print(f"\n--- Running Seed: {seed} on {len(ds)} images ---")
        sub_tag = f"_sub{len(ds)}" if args.limit > 0 else ""

        # ---------- Baseline ----------
        base_name = f"CHAIR_QwenVL_Baseline_seed{seed}_{dec_tag}{sub_tag}"
        pred_path = os.path.join(RESULTS_DIR_CHAIR, f"pred_{base_name}.json")
        met_path  = os.path.join(RESULTS_DIR_CHAIR, f"metrics_{base_name}.json")

        if os.path.exists(met_path):
            print(f"⏭️  Skipping Baseline Run (metrics exist): {base_name}")
            with open(met_path, 'r') as f:
                met = json.load(f)
        else:
            caps, files = run_once_caption(
                model, processor, loader,
                gen_kwargs=gen_kwargs,
            )
            with open(pred_path, "w", encoding="utf-8") as f:
                json.dump(
                    [{"file_name": fn, "caption": cap} for fn, cap in zip(files, caps)],
                    f,
                    indent=2,
                )
            met = evaluator.evaluate(files, caps)
            with open(met_path, "w") as f:
                json.dump(met, f, indent=2)

        print(f"✅ Baseline: CHAIRs={met['CHAIRs']:.4f}, CHAIRi={met['CHAIRi']:.4f}")
        log_data = {
            **base_params,
            "method": "Baseline",
            "seed": seed,
            "CHAIRs": met["CHAIRs"],
            "CHAIRi": met["CHAIRi"],
            "n_images": met["n_images"],
            "n_mentions": met["n_mentions"],
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        log_to_csv(csv_log_path, log_data)

        # ---------- CARD + Beta ----------
        if args.run_beta:
            print("\n--- Starting CARD+Beta Gating Experiments (Qwen-VL) ---")
            for L in INJECTION_LAYERS:
                for amax in BETA_ALPHA_MAX:
                    for kk in BETA_K:
                        for cc in BETA_C:
                            name = (
                                f"CHAIR_QwenVL_Beta_seed{seed}_L{L}_{args.pool}"
                                f"_A{amax}_K{kk}_C{cc}_{dec_tag}{sub_tag}"
                            )
                            met_o = os.path.join(
                                RESULTS_DIR_CHAIR, f"metrics_{name}.json"
                            )
                            if os.path.exists(met_o):
                                print(f"⏭️  Skipping Beta (metrics exist): {name}")
                                with open(met_o, 'r') as f:
                                    met = json.load(f)
                            else:
                                pred_o = os.path.join(
                                    RESULTS_DIR_CHAIR, f"pred_{name}.json"
                                )
                                caps, files = run_once_beta(
                                    model,
                                    processor,
                                    loader,
                                    layer=L,
                                    alpha_max=amax,
                                    k=kk,
                                    c=cc,
                                    pooling=args.pool,
                                    gen_kwargs=gen_kwargs,
                                )
                                with open(pred_o, "w", encoding="utf-8") as f:
                                    json.dump(
                                        [
                                            {"file_name": fn, "caption": cap}
                                            for fn, cap in zip(files, caps)
                                        ],
                                        f,
                                        indent=2,
                                    )
                                met = evaluator.evaluate(files, caps)
                                with open(met_o, "w") as f:
                                    json.dump(met, f, indent=2)

                            print(
                                f"✅ Result (Beta): CHAIRs={met['CHAIRs']:.4f}, "
                                f"CHAIRi={met['CHAIRi']:.4f} | Params: "
                                f"L={L}, A{amax}, K{kk}, C{cc}"
                            )
                            log_data = {
                                **base_params,
                                "method": "Beta",
                                "seed": seed,
                                "layer": L,
                                "alpha_max": amax,
                                "k": kk,
                                "c": cc,
                                "CHAIRs": met["CHAIRs"],
                                "CHAIRi": met["CHAIRi"],
                                "n_images": met["n_images"],
                                "n_mentions": met["n_mentions"],
                                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            }
                            log_to_csv(csv_log_path, log_data)

        # ---------- CARD + Simple Add ----------
        if args.run_add:
            print("\n--- Starting CARD+Simple Addition Experiments (Qwen-VL) ---")
            for L in INJECTION_LAYERS:
                for a in ADD_ALPHA:
                    name = (
                        f"CHAIR_QwenVL_Add_seed{seed}_L{L}_{args.pool}"
                        f"_A{a}_{dec_tag}{sub_tag}"
                    )
                    met_o = os.path.join(
                        RESULTS_DIR_CHAIR, f"metrics_{name}.json"
                    )
                    if os.path.exists(met_o):
                        print(f"⏭️  Skipping Add (metrics exist): {name}")
                        with open(met_o, 'r') as f:
                            met = json.load(f)
                    else:
                        pred_o = os.path.join(
                            RESULTS_DIR_CHAIR, f"pred_{name}.json"
                        )
                        caps, files = run_once_add(
                            model,
                            processor,
                            loader,
                            layer=L,
                            alpha=a,
                            pooling=args.pool,
                            gen_kwargs=gen_kwargs,
                        )
                        with open(pred_o, "w", encoding="utf-8") as f:
                            json.dump(
                                [
                                    {"file_name": fn, "caption": cap}
                                    for fn, cap in zip(files, caps)
                                ],
                                f,
                                indent=2,
                            )
                        met = evaluator.evaluate(files, caps)
                        with open(met_o, "w") as f:
                            json.dump(met, f, indent=2)

                    print(
                        f"✅ Result (Add): CHAIRs={met['CHAIRs']:.4f}, "
                        f"CHAIRi={met['CHAIRi']:.4f} | Params: L={L}, A{a}"
                    )
                    log_data = {
                        **base_params,
                        "method": "Add",
                        "seed": seed,
                        "layer": L,
                        "alpha": a,
                        "CHAIRs": met["CHAIRs"],
                        "CHAIRi": met["CHAIRi"],
                        "n_images": met["n_images"],
                        "n_mentions": met["n_mentions"],
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    }
                    log_to_csv(csv_log_path, log_data)

    print(f"\n✅ All experiments finished. Summary saved in: {csv_log_path}")

if __name__ == "__main__":
    main()
