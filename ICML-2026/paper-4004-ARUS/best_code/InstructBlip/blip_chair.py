#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run_chair_instructblip_final.py
#
# InstructBLIP + COCO-CHAIR 的完整实验脚本 (最终修复版, 仅 beam / nucleus 解码)
# - Baseline, CARD+Beta, 简单加法消融
# - 运行时安全修复（pad/left padding/OOV/嵌入钩子）
# - 修正了 hook 启用/禁用与 beam 维度对齐
# - 新增 --decode {beam,nucleus}，分别配置生成参数

import os, json, argparse, math, random, re
from typing import List, Dict, Tuple, Optional

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

from config_hal import *
from methods import (
    set_global_seed,
    mask_carrier,
    build_answer_mask_from_prompts,
    BayesianGatingHookMaskedDynamic,
)

# -------- 健壮性默认值 --------
RESULTS_DIR_CHAIR   = globals().get("RESULTS_DIR_CHAIR", os.path.join("results", "chair"))
BATCH_SIZE          = globals().get("BATCH_SIZE", 2)
NUM_WORKERS         = globals().get("NUM_WORKERS", 0)
SEEDS               = globals().get("SEEDS", [42])
EGR_POOLINGS        = globals().get("EGR_POOLINGS", ["attn"])
INJECTION_LAYERS    = globals().get("INJECTION_LAYERS", [1, 8, 16, 24, 28, 31])
BETA_ALPHA_MAX      = globals().get("BETA_ALPHA_MAX", [2.0, 4.0, 8.0])
BETA_K              = globals().get("BETA_K", [3.0, 5.0])
BETA_C              = globals().get("BETA_C", [0.5, 1.0])
GATE_CLAMP          = globals().get("GATE_CLAMP", (0.0, 0.8))
ADD_ALPHA           = globals().get("ADD_ALPHA", [4.5])

CAP_MAX_NEW_TOKENS  = globals().get("CAP_MAX_NEW_TOKENS", 512)
IMAGE_DIR           = globals().get("IMAGE_DIR", "images")
COCO_INSTANCES_JSON = globals().get("COCO_INSTANCES_JSON", "annotations/instances_val2014.json")
MODEL_ID            = globals().get("MODEL_ID", "Salesforce/instructblip-flan-t5-xl")
CACHE_DIR           = globals().get("CACHE_DIR", None)
DEVICE              = globals().get("DEVICE", "cuda")
DTYPE               = globals().get("DTYPE", "bf16")

# ==============================================================================
# 运行时安全修复逻辑
# ==============================================================================
def ensure_safe_tokenizer_and_model(model, processor, verbose=True):
    tok = processor.tokenizer
    if verbose: print("\n--- 运行时安全检查与修复 ---")
    tok.padding_side = "left"
    if tok.pad_token is None or tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        if verbose: print(f"[Fix] pad_token <- eos_token (id={tok.pad_token_id})")
    pad_id = tok.pad_token_id

    if hasattr(model.config, "pad_token_id"):
        model.config.pad_token_id = pad_id
    if hasattr(model, "language_model") and hasattr(model.language_model.config, "pad_token_id"):
        model.language_model.config.pad_token_id = pad_id

    try:
        lm = model.language_model
        current_embed_size = lm.get_input_embeddings().weight.size(0)
        vocab_size = len(tok)
        if current_embed_size != vocab_size:
            if verbose: print(f"[Fix] resize embeddings: {current_embed_size} -> {vocab_size}")
            lm.resize_token_embeddings(vocab_size)
            model.tie_weights()
    except Exception as e:
        if verbose: print(f"[Fix] 调整词嵌入矩阵失败: {e}")
    if verbose:
        print(f"[sanity] tokenizer={len(tok)}, pad_id={pad_id}, eos_id={tok.eos_token_id}")
        print("--- 安全检查完成 ---\n")

def install_embedding_sanitizer(lm, vocab_size: int, replace_id: int):
    emb_module = lm.get_input_embeddings()
    if emb_module is None: return None
    def _pre_hook(module, args, kwargs):
        input_ids = args[0] if len(args) > 0 else kwargs.get("input_ids", None)
        if isinstance(input_ids, torch.Tensor):
            bad = (input_ids < 0) | (input_ids >= vocab_size)
            if torch.any(bad):
                input_ids = torch.where(bad, torch.full_like(input_ids, replace_id), input_ids)
                return (input_ids,), kwargs
        return args, kwargs
    handle = emb_module.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    return handle

def safe_batch_decode(processor, gen_ids: torch.Tensor, cut_from: int):
    tok = processor.tokenizer
    vocab_size = len(tok)
    replace_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if gen_ids.dtype != torch.long:
        gen_ids = gen_ids.long()
    if cut_from >= gen_ids.size(1):
        new_tokens = gen_ids.new_empty((gen_ids.size(0), 0))
    else:
        new_tokens = gen_ids[:, cut_from:]
    new_tokens = torch.where(new_tokens < 0, torch.full_like(new_tokens, replace_id), new_tokens)
    new_tokens = torch.where(new_tokens >= vocab_size, torch.full_like(new_tokens, replace_id), new_tokens)
    return [s.strip() for s in tok.batch_decode(new_tokens.tolist(), skip_special_tokens=True)]

# ==============================================================================
# 数据加载与评测
# ==============================================================================
def _fallback_aliases() -> Dict[str, List[str]]:
    raw = {
        "person": ["person","people","man","woman","boy","girl"],
        "bicycle": ["bicycle","bike"],
        "car": ["car","cars","automobile","auto"],
        "motorcycle": ["motorcycle","motorbike"],
        "airplane": ["airplane","plane","jet","aircraft"],
        "bus": ["bus","coach"],
        "train": ["train","locomotive"],
        "truck": ["truck","lorry"],
        "boat": ["boat","ship","vessel"],
        "traffic light": ["traffic light","stoplight","signal light"],
        "fire hydrant": ["fire hydrant","hydrant"],
        "stop sign": ["stop sign"],
        "parking meter": ["parking meter"],
        "bench": ["bench"],
        "bird": ["bird"],
        "cat": ["cat","kitty","kitten"],
        "dog": ["dog","puppy"],
        "horse": ["horse","pony"],
        "sheep": ["sheep","lamb"],
        "cow": ["cow","cattle"],
        "elephant": ["elephant"],
        "bear": ["bear","teddy bear","teddy"],
        "zebra": ["zebra"],
        "giraffe": ["giraffe"],
        "backpack": ["backpack","pack"],
        "umbrella": ["umbrella","brolly"],
        "handbag": ["handbag","purse","bag"],
        "tie": ["tie","necktie"],
        "suitcase": ["suitcase","luggage"],
        "frisbee": ["frisbee","flying disc"],
        "skis": ["skis","ski"],
        "snowboard": ["snowboard"],
        "sports ball": ["sports ball","ball"],
        "kite": ["kite"],
        "baseball bat": ["baseball bat","bat"],
        "baseball glove": ["baseball glove","mitt"],
        "skateboard": ["skateboard"],
        "surfboard": ["surfboard"],
        "tennis racket": ["tennis racket","racket","racquet"],
        "bottle": ["bottle"],
        "wine glass": ["wine glass","goblet"],
        "cup": ["cup","mug"],
        "fork": ["fork"],
        "knife": ["knife"],
        "spoon": ["spoon"],
        "bowl": ["bowl"],
        "banana": ["banana"],
        "apple": ["apple"],
        "sandwich": ["sandwich"],
        "orange": ["orange"],
        "broccoli": ["broccoli"],
        "carrot": ["carrot"],
        "hot dog": ["hot dog"],
        "pizza": ["pizza"],
        "donut": ["donut","doughnut"],
        "cake": ["cake"],
        "chair": ["chair","seat"],
        "couch": ["couch","sofa"],
        "potted plant": ["potted plant","plant pot"],
        "bed": ["bed"],
        "dining table": ["dining table","table"],
        "toilet": ["toilet","wc","restroom"],
        "tv": ["tv","television","monitor","tv monitor"],
        "laptop": ["laptop","notebook computer"],
        "mouse": ["mouse","computer mouse"],
        "remote": ["remote","remote control"],
        "keyboard": ["keyboard"],
        "cell phone": ["cell phone","mobile phone","phone","smartphone"],
        "microwave": ["microwave","microwave oven"],
        "oven": ["oven","stove oven"],
        "toaster": ["toaster"],
        "sink": ["sink","basin"],
        "refrigerator": ["refrigerator","fridge"],
        "book": ["book","books"],
        "clock": ["clock"],
        "vase": ["vase"],
        "scissors": ["scissors","shears"],
        "hair drier": ["hair drier","hair dryer"],
        "toothbrush": ["toothbrush","tooth brush"],
    }
    out = {}
    for canon, aliases in raw.items():
        s = set()
        for a in aliases:
            s.add(a.lower())
            if not a.endswith("s"):
                s.add((a + "s").lower())
        out[canon] = sorted(s)
    return out

def load_aliases() -> Dict[str, str]:
    alias2canon = {}
    path = globals().get("COCO_VOCAB_JSON", None)
    if path and os.path.exists(path):
        with open(path, "r") as f: obj = json.load(f)
        for canon, aliases in obj.items():
            for a in aliases:
                alias2canon[str(a).lower()] = canon
    else:
        fb = _fallback_aliases()
        for canon, aliases in fb.items():
            for a in aliases:
                alias2canon[a.lower()] = canon
    return alias2canon

def load_coco_gt(instances_json: str) -> Tuple[Dict[int,str], Dict[str, set]]:
    with open(instances_json, "r") as f:
        coco = json.load(f)
    id2name = {c["id"]: c["name"].lower() for c in coco["categories"]}
    imgid2fname = {im["id"]: im["file_name"] for im in coco["images"]}
    fname2cats = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]; cat_id = ann["category_id"]
        fname = imgid2fname.get(img_id)
        if fname is None: continue
        s = fname2cats.setdefault(fname, set())
        s.add(id2name.get(cat_id, ""))
    return id2name, fname2cats

class CHAIRImageDataset(Dataset):
    def __init__(self, image_dir: str, instances_json: str, limit: int = 0, limit_seed: int = 42):
        _, fname2cats = load_coco_gt(instances_json)
        rows = []
        for fname, cats in fname2cats.items():
            path = os.path.join(image_dir, fname)
            if os.path.exists(path):
                rows.append({"file_name": fname, "path": path, "gt": cats})
        if limit and limit > 0 and len(rows) > limit:
            random.seed(limit_seed)
            random.shuffle(rows)
            rows = rows[:limit]
        self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(r["path"]).convert("RGB")
        return {"image": img, "file_name": r["file_name"], "gt": r["gt"]}

def chair_collate(batch):
    images, fns, gts = [], [], []
    for b in batch:
        images.append(b["image"]); fns.append(b["file_name"]); gts.append(b["gt"])
    return images, fns, gts

def build_alias_regex(alias2canon: Dict[str,str]):
    aliases = sorted(alias2canon.keys(), key=lambda s: (-len(s.split()), -len(s)))
    patterns = []
    for a in aliases:
        p = r"\b" + re.escape(a).replace(r"\ ", r"\s+") + r"\b"
        patterns.append((re.compile(p, flags=re.IGNORECASE), alias2canon[a]))
    return patterns

def mentions_from_caption(caption: str, patterns) -> set:
    cap = caption.lower()
    found = set()
    for rgx, canon in patterns:
        if rgx.search(cap):
            found.add(canon)
    return found

def wilson_center_half(p: float, n: int, z: float=1.96) -> Tuple[float,float]:
    if n <= 0: return float("nan"), float("nan")
    denom = 1.0 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half = (z/denom) * math.sqrt((p*(1-p))/n + (z*z)/(4*n*n))
    return center, half

def bootstrap_chairi(images_info: List[Tuple[set,set]], B: int=2000, seed: int=2025, alpha: float=0.05):
    import numpy as np
    rng = np.random.default_rng(seed)
    n = len(images_info)
    idx = np.arange(n)
    vals = []
    for _ in range(B):
        bs = rng.choice(idx, size=n, replace=True)
        m_sum = 0; h_sum = 0
        for i in bs:
            M, H = images_info[i]
            m_sum += len(M)
            h_sum += len(H)
        vals.append( h_sum / max(1, m_sum) )
    lo, hi = float(np.percentile(vals, 100*alpha/2)), float(np.percentile(vals, 100*(1-alpha/2)))
    return lo, hi

def evaluate_chair(file_names: List[str], captions: List[str], gts: List[set], alias2canon: Dict[str,str],
                   ci_alpha=0.05, boot_B=2000) -> Dict[str, float]:
    patterns = build_alias_regex(alias2canon)
    n_img = len(captions)
    total_mentions = 0
    total_hallu = 0
    sent_hallu = 0
    per_image_info = []
    for fn, cap, gt in zip(file_names, captions, gts):
        M = mentions_from_caption(cap, patterns)
        G = set([a.lower() for a in gt])
        H = {m for m in M if m not in G}
        total_mentions += len(M)
        total_hallu    += len(H)
        if len(H) > 0:
            sent_hallu += 1
        per_image_info.append((M, H))
    CHAIRi = total_hallu / max(1, total_mentions)
    CHAIRs = sent_hallu  / max(1, n_img)
    z = 1.959963984540054
    s_center, s_half = wilson_center_half(CHAIRs, n_img, z)
    i_lo, i_hi = bootstrap_chairi(per_image_info, B=boot_B, seed=2025, alpha=ci_alpha)
    return {
        "n_images": n_img,
        "n_mentions": total_mentions,
        "hallucinated_mentions": total_hallu,
        "images_with_hallucination": sent_hallu,
        "CHAIRi": CHAIRi,
        "CHAIRi_ci_low": i_lo,
        "CHAIRi_ci_high": i_hi,
        "CHAIRs": CHAIRs,
        "CHAIRs_center": s_center,
        "CHAIRs_halfwidth": s_half,
    }

def build_caption_prompt() -> str:
    return (
        "Instruction: Describe the image in rich detail.\n"
        "Image: <image>\n"
        "Answer:"
    )

# ==============================================================================
# 核心干预逻辑 (适配 InstructBLIP)
# ==============================================================================
def get_decoder_self_attn_module(model, layer_idx: int):
    lm = model.language_model
    if hasattr(lm, "decoder") and hasattr(lm.decoder, "block"):
        layers = lm.decoder.block
        if not (0 <= layer_idx < len(layers)):
            raise IndexError(f"T5 decoder 层数={len(layers)}，传入 L={layer_idx} 不合法。")
        return layers[layer_idx].layer[0]  # cross/self-attn所在层索引
    if hasattr(lm, "model") and hasattr(lm.model, "layers"):
        layers = lm.model.layers
        if not (0 <= layer_idx < len(layers)):
            raise IndexError(f"LLaMA decoder 层数={len(layers)}，传入 L={layer_idx} 不合法。")
        return layers[layer_idx].self_attn
    raise RuntimeError("未能定位到语言模型的 decoder layers/blocks。")

@torch.no_grad()
def compute_card_vector_batch_blip(model, processor, images, prompts, layer_idx: int, pooling: str):
    """
    简化版 CARD：直接基于编码后的 text embeddings 做加权/平均池化（稳定且足够一致）。
    """
    device = next(model.parameters()).device
    enc = processor(images=images, text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    lm = model.language_model
    # LLaMA/Vicuna 路径
    if hasattr(lm, "model") and hasattr(lm.model, "embed_tokens"):
        emb = lm.model.embed_tokens(enc.input_ids)           # [B, T, H]
        mask = enc.attention_mask.unsqueeze(-1).to(emb.dtype)  # [B, T, 1]
        if pooling == 'attn':
            w = (emb.pow(2).sum(-1, keepdim=True).sqrt() + 1e-9) * mask
            v = (emb * w).sum(1) / w.sum(1).clamp_min(1e-9)
        else:
            v = (emb * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
        return F.normalize(v, p=2, dim=-1)
    # T5 路径
    if hasattr(lm, "decoder") and hasattr(lm.decoder, "block"):
        emb = lm.get_input_embeddings()(enc.input_ids)       # [B, T, H]
        mask = enc.attention_mask.unsqueeze(-1).to(emb.dtype)
        if pooling == 'attn':
            w = (emb.pow(2).sum(-1, keepdim=True).sqrt() + 1e-9) * mask
            v = (emb * w).sum(1) / w.sum(1).clamp_min(1e-9)
        else:
            v = (emb * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
        return F.normalize(v, p=2, dim=-1)
    raise RuntimeError("不支持的 language_model 结构，无法计算 CARD 向量。")

# ====================== 简单加法 PreHook（在 self_attn 前注入） ======================
class SimpleAddPreHook:
    """hs <- hs + alpha * expand(v) * mask（在 self_attn 前注入，带鲁棒的 mask 对齐逻辑）"""
    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.v_batch: Optional[torch.Tensor] = None
        self.h = None

    def set_vector(self, v_batch: torch.Tensor):
        self.v_batch = v_batch

    def register(self, target_layer):
        self.h = target_layer.register_forward_pre_hook(self, with_kwargs=True)
        return self.h

    def remove(self):
        if self.h: self.h.remove(); self.h = None

    def __call__(self, module, args, kwargs):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        if hs is None or self.v_batch is None:
            return args, kwargs

        B, T_current, H = hs.size()
        v = self.v_batch.to(device=hs.device, dtype=hs.dtype)

        # 适配 beam search 等导致的 batch 扩大
        if v.size(0) != B:
            num_beams = B // v.size(0)
            v = v.repeat_interleave(num_beams, dim=0)

        vT = v.unsqueeze(1).expand(B, T_current, H)

        # 掩码获取与对齐
        mask_from_carrier = getattr(mask_carrier, "mask", None)
        if mask_from_carrier is None:
            aligned_mask = hs.new_ones(B, T_current, 1)
        else:
            mask = mask_from_carrier.to(device=hs.device, dtype=hs.dtype)
            if mask.size(0) != B:
                num_beams = B // mask.size(0)
                mask = mask.repeat_interleave(num_beams, dim=0)
            if mask.size(1) != T_current:
                aligned_mask = mask[:, -1:, :].expand(B, T_current, 1)
            else:
                aligned_mask = mask

        hs_new = hs + self.alpha * vT * aligned_mask

        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = hs_new
        else:
            args = (hs_new,) + args[1:]
        return args, kwargs

# ==============================================================================
# 实验运行器（统一支持 beam / nucleus）
# ==============================================================================
@torch.no_grad()
def run_experiment(model, processor, loader, exp_type: str, max_new_tokens: int,
                   decoding_params: dict, intervention_params: dict = None):
    all_caps, all_files = [], []
    device = next(model.parameters()).device

    hook, handle = None, None
    if intervention_params:
        layer_idx = intervention_params['layer']
        target_layer = get_decoder_self_attn_module(model, layer_idx)
        if exp_type == "beta":
            hook = BayesianGatingHookMaskedDynamic(**intervention_params['hook_args'])
            handle = target_layer.register_forward_hook(hook)
        elif exp_type == "add":
            hook = SimpleAddPreHook(alpha=intervention_params['alpha'])
            handle = hook.register(target_layer)

    # 估计安全的文本预算
    try:
        max_pos = getattr(model.language_model.config, "max_position_embeddings", 2048)
        num_q = getattr(model.config, "num_query_tokens", 32)
        max_text_len = max_pos - num_q - max_new_tokens - 10
        if max_text_len <= 0: raise ValueError("文本预算不足")
    except Exception:
        max_text_len = 2048 - 32 - max_new_tokens - 10

    num_beams = decoding_params.get("num_beams", 1)
    is_beam = (not decoding_params.get("do_sample", False)) and (num_beams and num_beams > 1)

    try:
        for images, files, _ in tqdm(loader, desc=f"  [{exp_type}]", ncols=100, leave=False):
            prompts = [build_caption_prompt() for _ in images]
            inputs = processor(
                text=prompts, images=images, padding="longest",
                truncation=True, max_length=max_text_len, return_tensors="pt"
            ).to(device)

            if hook:
                # 1) 先禁用（如有）以免计算 CARD 过程中干预
                if hasattr(hook, "disable") and callable(hook.disable):
                    hook.disable()

                # 2) 计算 CARD 向量（与当前 batch 同一 prompts/images）
                v_batch = compute_card_vector_batch_blip(
                    model, processor, images, prompts,
                    layer_idx=intervention_params['layer'],
                    pooling=intervention_params.get('pool', 'attn')
                )

                # 3) 构造 Answer 段掩码，并在 beam 下提前展开
                m = build_answer_mask_from_prompts(processor.tokenizer, prompts, inputs["input_ids"])  # [B, T, 1]
                if "attention_mask" in inputs and m.size(1) == inputs["attention_mask"].size(1):
                    m = m * inputs["attention_mask"].unsqueeze(-1).to(m.dtype)

                if is_beam and m is not None:
                    m = m.repeat_interleave(num_beams, dim=0)  # 预展开以匹配解码时 B*num_beams

                mask_carrier.set(m.to(device))

                # 4) 注入向量，并开启 hook（注意某些实现里 enable/enabled 是 bool，而非可调用函数）
                if hasattr(hook, "set_vector") and callable(hook.set_vector):
                    hook.set_vector(v_batch)
                if hasattr(hook, "enable") and not callable(getattr(hook, "enable")):
                    hook.enable = True
                if hasattr(hook, "enabled") and not callable(getattr(hook, "enabled")):
                    hook.enabled = True

            # 5) 生成
            gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, **decoding_params)

            cut_from = inputs["input_ids"].size(1)
            outs = safe_batch_decode(processor, gen_ids, cut_from)
            all_caps.extend(outs)
            all_files.extend(files)

            if hook:
                mask_carrier.clear()
    finally:
        if handle and hasattr(handle, "remove"):
            handle.remove()

    return all_caps, all_files

# ==============================================================================
# 主程序入口
# ==============================================================================
def main():
    ap = argparse.ArgumentParser("CHAIR on InstructBLIP: Baseline, CARD-Beta, Simple-Add (beam/nucleus only)")
    ap.add_argument("--limit", type=int, default=60, help="子样本图片数 (<=0 全量)")
    # 解码选择：仅支持 beam 或 nucleus
    ap.add_argument("--decode", type=str, default="nucleus", choices=["beam","nucleus"], help="解码方式：beam 或 nucleus")
    # beam 参数
    ap.add_argument("--num_beams", type=int, default=5, help="beam search 的 beam 数")
    ap.add_argument("--length_penalty", type=float, default=1.0, help="beam 的长度惩罚")
    ap.add_argument("--early_stopping", action="store_true", help="beam 提前停止")
    # nucleus 参数
    ap.add_argument("--top_p", type=float, default=0.9, help="nucleus 采样的 top_p")
    ap.add_argument("--temperature", type=float, default=1.0, help="nucleus 采样温度")
    # 共同参数
    ap.add_argument("--max_new_tokens", type=int, default=CAP_MAX_NEW_TOKENS, help="生成最大新 token 数")
    ap.add_argument("--repetition_penalty", type=float, default=1.0, help="重复惩罚（两种解码都可用）")

    # 哪些实验
    ap.add_argument("--run_baseline", action="store_true", help="运行 Baseline")
    ap.add_argument("--run_beta", action="store_true", help="运行 CARD+Beta 网格搜索")
    ap.add_argument("--run_add", action="store_true", help="运行简单加法消融")

    args = ap.parse_args()

    # 解析 dtype
    dtype = torch.bfloat16 if DTYPE=="bf16" else (torch.float16 if DTYPE=="fp16" else torch.float32)

    os.makedirs(RESULTS_DIR_CHAIR, exist_ok=True)
    print(f"⏳ Loading model: {MODEL_ID}")
    model = InstructBlipForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, cache_dir=CACHE_DIR, device_map={"": DEVICE}
    )
    processor = InstructBlipProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

    ensure_safe_tokenizer_and_model(model, processor, verbose=True)

    lm = model.language_model
    vocab_size = len(processor.tokenizer)
    replace_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id
    embed_hook_handle = install_embedding_sanitizer(lm, vocab_size, replace_id)

    model.eval(); print("✅ Model ready.")
    alias2canon = load_aliases()

    # ===== 构造解码参数（仅 beam / nucleus） =====
    if args.decode == "beam":
        if args.num_beams < 2:
            raise ValueError("beam 解码要求 num_beams >= 2")
        decoding_params = dict(
            do_sample=False,
            num_beams=int(args.num_beams),
            length_penalty=float(args.length_penalty),
            early_stopping=bool(args.early_stopping),
            repetition_penalty=float(args.repetition_penalty),
        )
    else:  # nucleus
        decoding_params = dict(
            do_sample=True,
            top_p=float(args.top_p),
            temperature=float(args.temperature),
            num_beams=1,
            repetition_penalty=float(args.repetition_penalty),
        )

    for seed in SEEDS:
        set_global_seed(seed)
        ds = CHAIRImageDataset(IMAGE_DIR, COCO_INSTANCES_JSON, limit=args.limit, limit_seed=seed)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=chair_collate)
        print(f"📦 CHAIR subset = {len(ds)} images for seed {seed}")
        sub_tag = f"_sub{len(ds)}"

        # ---------- Baseline ----------
        if args.run_baseline:
            base_name = f"CHAIR_InstructBLIP_baseline_{args.decode}_seed{seed}{sub_tag}"
            pred_path = os.path.join(RESULTS_DIR_CHAIR, f"pred_{base_name}.json")
            met_path  = os.path.join(RESULTS_DIR_CHAIR, f"metrics_{base_name}.json")

            if os.path.exists(pred_path) and os.path.exists(met_path):
                print(f"⏭️  Skip baseline exists: {pred_path}")
            else:
                print(f"\n▶️ Running Baseline ({args.decode}): {base_name}")
                caps, files = run_experiment(
                    model, processor, loader, "baseline",
                    max_new_tokens=int(args.max_new_tokens),
                    decoding_params=decoding_params,
                    intervention_params=None
                )
                with open(pred_path, "w") as f:
                    json.dump([{"file_name": fn, "caption": cap} for fn, cap in zip(files, caps)], f, indent=2, ensure_ascii=False)
                gts = [r["gt"] for r in ds.rows]
                met = evaluate_chair(files, caps, gts, alias2canon)
                with open(met_path, "w") as f:
                    json.dump(met, f, indent=2)
                print(f"✅ Baseline saved. CHAIRs={met['CHAIRs']:.3f}  CHAIRi={met['CHAIRi']:.3f}")

        # ---------- CARD+Beta ----------
        if args.run_beta:
            for L in INJECTION_LAYERS:
                for pool in EGR_POOLINGS:
                    for amax in BETA_ALPHA_MAX:
                        for kk in BETA_K:
                            for cc in BETA_C:
                                name = f"CHAIR_InstructBLIP_beta_{args.decode}_seed{seed}_L{L}_{pool}_A{amax}_K{kk}_C{cc}{sub_tag}"
                                pred_o = os.path.join(RESULTS_DIR_CHAIR, f"pred_{name}.json")
                                met_o  = os.path.join(RESULTS_DIR_CHAIR, f"metrics_{name}.json")
                                if os.path.exists(pred_o) and os.path.exists(met_o):
                                    print(f"⏭️  Skip exists: {pred_o}")
                                    continue
                                print(f"\n▶️ Running Beta ({args.decode}): {name}")
                                params = {
                                    'layer': L,
                                    'pool': pool,
                                    'hook_args': {
                                        'max_alpha': amax,
                                        'sensitivity': kk,
                                        'concentration': cc,
                                        'carrier': mask_carrier,
                                        'clamp': GATE_CLAMP
                                    }
                                }
                                caps, files = run_experiment(
                                    model, processor, loader, "beta",
                                    max_new_tokens=int(args.max_new_tokens),
                                    decoding_params=decoding_params,
                                    intervention_params=params
                                )
                                with open(pred_o, "w") as f:
                                    json.dump([{"file_name": fn, "caption": cap} for fn, cap in zip(files, caps)], f, indent=2, ensure_ascii=False)
                                gts = [r["gt"] for r in ds.rows]
                                met = evaluate_chair(files, caps, gts, alias2canon)
                                with open(met_o, "w") as f:
                                    json.dump(met, f, indent=2)
                                print(f"✅ Beta saved. CHAIRs={met['CHAIRs']:.3f}  CHAIRi={met['CHAIRi']:.3f}")

        # ---------- 简单加法（消融） ----------
        if args.run_add:
            for L in INJECTION_LAYERS:
                for pool in EGR_POOLINGS:
                    for a in ADD_ALPHA:
                        name = f"CHAIR_InstructBLIP_add_{args.decode}_seed{seed}_L{L}_{pool}_A{a}{sub_tag}"
                        pred_o = os.path.join(RESULTS_DIR_CHAIR, f"pred_{name}.json")
                        met_o  = os.path.join(RESULTS_DIR_CHAIR, f"metrics_{name}.json")
                        if os.path.exists(pred_o) and os.path.exists(met_o):
                            print(f"⏭️  Skip exists: {pred_o}")
                            continue
                        print(f"\n▶️ Running Add ({args.decode}): {name}")
                        params = {'layer': L, 'pool': pool, 'alpha': a}
                        caps, files = run_experiment(
                            model, processor, loader, "add",
                            max_new_tokens=int(args.max_new_tokens),
                            decoding_params=decoding_params,
                            intervention_params=params
                        )
                        with open(pred_o, "w") as f:
                            json.dump([{"file_name": fn, "caption": cap} for fn, cap in zip(files, caps)], f, indent=2, ensure_ascii=False)
                        gts = [r["gt"] for r in ds.rows]
                        met = evaluate_chair(files, caps, gts, alias2canon)
                        with open(met_o, "w") as f:
                            json.dump(met, f, indent=2)
                        print(f"✅ Add saved. CHAIRs={met['CHAIRs']:.3f}  CHAIRi={met['CHAIRi']:.3f}")

    if embed_hook_handle:
        embed_hook_handle.remove()

    print(f"\n✅ All experiments finished. Results are in: {RESULTS_DIR_CHAIR}")

if __name__ == "__main__":
    # 解析 dtype 需要在 main 内部，但这里保留结构一致性
    main()
