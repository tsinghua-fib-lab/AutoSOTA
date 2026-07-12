#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run_chair.py — COCO-CHAIR caption hallucination (baseline + CARD-Beta + simple-add ablation)
# 支持 greedy / beam / nucleus；early_stopping 可调；加入防 prompt-echo 的可选 bad_words_ids

import os, json, argparse, math, random, re
from typing import List, Dict, Tuple

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, Idefics2ForConditionalGeneration

# ====== config & methods ======
from config_hal import *  # will be partially overridden by safe defaults below
from methods import (
    set_global_seed,
    mask_carrier,
    build_answer_mask_from_prompts,
    BayesianGatingHookMaskedDynamic,
)

# -------- mask_carrier 兼容（旧版无 get()）--------
if not hasattr(mask_carrier, "get"):
    def _mc_get():
        return getattr(mask_carrier, "mask", None)
    mask_carrier.get = _mc_get  # type: ignore[attr-defined]

# -------- safe defaults (if config_hal misses anything) --------
RESULTS_DIR_CHAIR      = globals().get("RESULTS_DIR_CHAIR", os.path.join("results","chair"))
BATCH_SIZE             = globals().get("BATCH_SIZE", 8)
NUM_WORKERS            = globals().get("NUM_WORKERS", 0)
SEEDS                  = globals().get("SEEDS", [42])
EGR_POOLINGS           = globals().get("EGR_POOLINGS", ["attn"])
INJECTION_LAYERS       = globals().get("INJECTION_LAYERS", [26])
#BETA_ALPHA_MAX         = globals().get("BETA_ALPHA_MAX", [5.0, 6.0])  # 如未在 config 定义可取消注释
BETA_K                 = globals().get("BETA_K", [3.0, 5.0])
BETA_C                 = globals().get("BETA_C", [0.5, 1.0])
ADD_ALPHA              = globals().get("ADD_ALPHA", [6.0, 8.0])
CAP_MAX_NEW_TOKENS     = globals().get("CAP_MAX_NEW_TOKENS", 512)
GATE_CLAMP             = globals().get("GATE_CLAMP", (0.0, 1.0))
IMAGE_DIR              = globals().get("IMAGE_DIR", "images")
MODEL_ID               = globals().get("MODEL_ID", "HuggingFaceM4/idefics2-8b")
CACHE_DIR              = globals().get("CACHE_DIR", None)
DEVICE                 = globals().get("DEVICE", "cuda")
DTYPE                  = globals().get("DTYPE", "bf16")
#COCO_INSTANCES_JSON    = globals().get("COCO_INSTANCES_JSON", os.path.join("annotations","instances_val2014.json"))

# -------- hook on/off 兼容封装 --------
def _hook_off(hook):
    if hasattr(hook, "enable") and isinstance(getattr(hook, "enable"), bool):
        hook.enable = False
    elif hasattr(hook, "disable") and callable(getattr(hook, "disable")):
        hook.disable()
    setattr(hook, "enabled", False)

def _hook_on(hook):
    if hasattr(hook, "enable") and isinstance(getattr(hook, "enable"), bool):
        hook.enable = True
    elif hasattr(hook, "enable") and callable(getattr(hook, "enable")):
        hook.enable()
    setattr(hook, "enabled", True)

# ====================== Prompt ======================
def build_caption_prompt() -> str:
    return (
        "Instruction: Describe the image in rich detail.\n"
        "Image: <image>\n"
        "Answer:"
    )

# ====================== COCO-80 别名 ======================
def _fallback_aliases() -> Dict[str, List[str]]:
    raw = {
        "person": ["person","people","man","woman","boy","girl"],
        "bicycle": ["bicycle","bike"], "car": ["car","cars","automobile","auto"],
        "motorcycle": ["motorcycle","motorbike"], "airplane": ["airplane","plane","jet","aircraft"],
        "bus": ["bus","coach"], "train": ["train","locomotive"], "truck": ["truck","lorry"],
        "boat": ["boat","ship","vessel"], "traffic light": ["traffic light","stoplight","signal light"],
        "fire hydrant": ["fire hydrant","hydrant"], "stop sign": ["stop sign"],
        "parking meter": ["parking meter"], "bench": ["bench"], "bird": ["bird"],
        "cat": ["cat","kitty","kitten"], "dog": ["dog","puppy"], "horse": ["horse","pony"],
        "sheep": ["sheep","lamb"], "cow": ["cow","cattle"], "elephant": ["elephant"],
        "bear": ["bear","teddy bear","teddy"], "zebra": ["zebra"], "giraffe": ["giraffe"],
        "backpack": ["backpack","pack"], "umbrella": ["umbrella","brolly"],
        "handbag": ["handbag","purse","bag"], "tie": ["tie","necktie"], "suitcase": ["suitcase","luggage"],
        "frisbee": ["frisbee","flying disc"], "skis": ["skis","ski"], "snowboard": ["snowboard"],
        "sports ball": ["sports ball","ball"], "kite": ["kite"], "baseball bat": ["baseball bat","bat"],
        "baseball glove": ["baseball glove","mitt"], "skateboard": ["skateboard"], "surfboard": ["surfboard"],
        "tennis racket": ["tennis racket","racket","racquet"], "bottle": ["bottle"],
        "wine glass": ["wine glass","goblet"], "cup": ["cup","mug"], "fork": ["fork"], "knife": ["knife"],
        "spoon": ["spoon"], "bowl": ["bowl"], "banana": ["banana"], "apple": ["apple"],
        "sandwich": ["sandwich"], "orange": ["orange"], "broccoli": ["broccoli"], "carrot": ["carrot"],
        "hot dog": ["hot dog"], "pizza": ["pizza"], "donut": ["donut","doughnut"], "cake": ["cake"],
        "chair": ["chair","seat"], "couch": ["couch","sofa"], "potted plant": ["potted plant","plant pot"],
        "bed": ["bed"], "dining table": ["dining table","table"], "toilet": ["toilet","wc","restroom"],
        "tv": ["tv","television","monitor","tv monitor"], "laptop": ["laptop","notebook computer"],
        "mouse": ["mouse","computer mouse"], "remote": ["remote","remote control"], "keyboard": ["keyboard"],
        "cell phone": ["cell phone","mobile phone","phone","smartphone"], "microwave": ["microwave","microwave oven"],
        "oven": ["oven","stove oven"], "toaster": ["toaster"], "sink": ["sink","basin"],
        "refrigerator": ["refrigerator","fridge"], "book": ["book","books"], "clock": ["clock"], "vase": ["vase"],
        "scissors": ["scissors","shears"], "hair drier": ["hair drier","hair dryer"], "toothbrush": ["toothbrush","tooth brush"],
    }
    out = {}
    for canon, aliases in raw.items():
        s = set()
        for a in aliases:
            s.add(a.lower())
            if not a.endswith("s"):
                s.add((a+"s").lower())
        out[canon] = sorted(s)
    return out

def load_aliases() -> Dict[str, str]:
    alias2canon = {}
    path = globals().get("COCO_VOCAB_JSON", None)
    if path and os.path.exists(path):
        with open(path, "r") as f:
            obj = json.load(f)
        for canon, aliases in obj.items():
            for a in aliases:
                alias2canon[str(a).lower()] = canon
    else:
        fb = _fallback_aliases()
        for canon, aliases in fb.items():
            for a in aliases:
                alias2canon[a.lower()] = canon
    return alias2canon

# ====================== COCO instances → GT 类别集合 ======================
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

# ====================== Dataset ======================
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

# ====================== mention extraction ======================
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

# ====================== CHAIR metrics（正确 + 置信区间） ======================
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

# ====================== CARD 捕获 ======================
class _CardCaptureRP:
    def __init__(self, target_layer):
        self.layer = target_layer
        self.h_pre = None; self.h_post = None
        self.before = None; self.after = None
    def __enter__(self):
        def pre(_m, args, kwargs):
            hs = kwargs.get("hidden_states", None) if kwargs is not None else None
            if hs is None and len(args)>0: hs = args[0]
            if hs is not None: self.before = hs.detach()
        def post(_m, args, output):
            out0 = output[0] if isinstance(output, (tuple,list)) else output
            self.after = out0.detach()
        self.h_pre  = self.layer.register_forward_pre_hook(pre,  with_kwargs=True)
        self.h_post = self.layer.register_forward_hook(post, with_kwargs=False)
        return self
    def __exit__(self, exc_type, exc, tb):
        if self.h_pre:  self.h_pre.remove()
        if self.h_post: self.h_post.remove()

@torch.no_grad()
def _card_from_same_prompts(model, processor, images, prompts, layer_idx=24, pooling="attn"):
    inputs = processor(text=prompts, images=[[im] for im in images], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    tgt = model.model.text_model.layers[layer_idx].self_attn
    with _CardCaptureRP(tgt) as cap:
        _ = model(**inputs, output_hidden_states=False, return_dict=True)
    before, after = cap.before, cap.after  # [B,T,H]
    delta = after - before
    attn_mask = inputs["attention_mask"].unsqueeze(-1).to(delta.dtype)
    if pooling == "mean":
        v = (delta * attn_mask).sum(dim=1) / (attn_mask.sum(dim=1) + 1e-6)
    else:
        w = (delta.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6) * attn_mask
        v = (delta * w).sum(dim=1) / (w.sum(dim=1) + 1e-6)
    return F.normalize(v, p=2, dim=-1).detach()

# ====================== 简单加法 Hook ======================
'''
class SimpleAddHook:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.v_batch = None
        self.mask = None
    def set_vector(self, v): self.v_batch = v
    def set_mask(self, m):   self.mask = m
    def __call__(self, module, args, out):
        if self.v_batch is None or self.mask is None:
            return out
        attn_out, *rest = out
        B, q_len, H = attn_out.shape
        v = F.normalize(self.v_batch, p=2, dim=-1).to(attn_out.dtype).to(attn_out.device)
        v = v.unsqueeze(1).expand(B, q_len, H)
        m = self.mask.to(attn_out.dtype).to(attn_out.device)
        T_full = m.size(1)
        if T_full == q_len:
            m_cur = m
        elif T_full > q_len:
            m_cur = m[:, -q_len:, :]
        else:
            pad = m[:, -1:, :].expand(B, q_len - T_full, 1)
            m_cur = torch.cat([m, pad], dim=1)
        attn_out = attn_out + self.alpha * v * m_cur
        return (attn_out, *rest)
'''
# ====================== 简单加法 Hook（支持 beam 扩批 & 自回归对齐） ======================
class SimpleAddHook:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.v_batch = None  # [B,H] 或 [B,T0,H]
        self.mask = None     # [B,T,1]

    def set_vector(self, v): self.v_batch = v
    def set_mask(self, m):   self.mask = m

    def __call__(self, module, args, out):
        # out: (attn_output, attn_weights, present_kv) 或 Tensor
        if self.v_batch is None or self.mask is None:
            return out

        # 1) 取当前层输出 hidden（注意 generate 时是增量解码，q_len 可能=1）
        if isinstance(out, tuple):
            h, *rest = out
        else:
            h, rest = out, []
        if h is None:
            return out
        B_eff, q_len, H = h.shape  # B_eff=原B或B*num_beams

        v = self.v_batch.to(h.device, dtype=h.dtype)  # [B0,H] 或 [B0,T0,H]
        m = self.mask.to(h.device, dtype=h.dtype)     # [B,T_full,1]

        # 2) 工具：把 [B0,*] 重复到 B_eff（beam 扩批）
        def _repeat_to_batch(x, target_B):
            B0 = x.size(0)
            if B0 == target_B:
                return x
            if target_B % B0 == 0:
                r = target_B // B0
                return x.repeat_interleave(r, dim=0)
            # 兜底：tile 后截断
            reps = (target_B + B0 - 1) // B0
            x = x.repeat((reps,) + (1,) * (x.dim() - 1))
            return x[:target_B]

        # 3) 时间对齐：把 token 维右对齐到当前 q_len（过长截尾，过短尾部补）
        def _align_time(x, T_target):
            if x.dim() < 3:  # [B,H]
                return x
            T0 = x.size(1)
            if T0 == T_target:
                return x
            if T0 > T_target:
                return x[:, -T_target:, :]
            pad = x.new_zeros(x.size(0), T_target - T0, x.size(2))
            return torch.cat([x, pad], dim=1)

        # 4) 处理 v：支持 [B0,H] 或 [B0,T0,H]
        if v.dim() == 2:
            v = F.normalize(v, p=2, dim=-1)
            v = _repeat_to_batch(v, B_eff)          # [B_eff,H]
            v = v.unsqueeze(1).expand(B_eff, q_len, H)
        elif v.dim() == 3:
            v = F.normalize(v, p=2, dim=-1)
            v = _repeat_to_batch(v, B_eff)          # [B_eff,T0,H]
            v = _align_time(v, q_len)               # [B_eff,q_len,H]
        else:
            # 形状不支持，直接返回原输出
            return out

        # 5) 处理 mask：对齐到 beam 后的 batch 和当前步长度
        if m.dim() == 2:  # [B,T] -> [B,T,1]
            m = m.unsqueeze(-1)
        m = _repeat_to_batch(m, B_eff)              # [B_eff,T_full,1]
        if m.size(1) != q_len:
            if m.size(1) > q_len:
                m = m[:, -q_len:, :]
            else:
                # 掩码不够长时，用最后一位（通常是 Answer 段的 1）补齐
                last = m[:, -1:, :].expand(B_eff, q_len - m.size(1), 1)
                m = torch.cat([m, last], dim=1)     # [B_eff,q_len,1]

        # 6) 注入并返回
        h_out = h + self.alpha * v * m
        if isinstance(out, tuple):
            return (h_out, *rest)
        else:
            return h_out
# ====================== 生成参数构造 & 防回声 ======================
def parse_early_stopping(s: str):
    s = (s or "").strip().lower()
    if s in {"true","t","1","yes","y"}:   return True
    if s in {"false","f","0","no","n"}:   return False
    if s in {"never"}:                    return "never"
    return False

def make_bad_words_ids(tokenizer, phrases):
    toks = tokenizer(phrases, add_special_tokens=False).input_ids
    # 过滤空 token 序列（避免 generate 报错）
    toks = [t for t in toks if isinstance(t, list) and len(t) > 0]
    return toks if len(toks) > 0 else None

def build_gen_kwargs(decoding: str, temperature: float, top_p: float,
                     num_beams: int, length_penalty: float,
                     early_stopping_arg, max_new_tokens: int,
                     eos_token_id: int, pad_token_id: int,
                     no_repeat_ngram_size: int, repetition_penalty: float,
                     bad_words_ids=None):
    decoding = decoding.lower()
    early_stopping = parse_early_stopping(early_stopping_arg)
    base = dict(
        max_new_tokens=max_new_tokens,
        length_penalty=float(length_penalty),
        early_stopping=early_stopping,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        no_repeat_ngram_size=int(no_repeat_ngram_size),
        repetition_penalty=float(repetition_penalty),
        remove_invalid_values=True,
    )
    if bad_words_ids is not None:
        base["bad_words_ids"] = bad_words_ids

    if decoding == "greedy":
        base.update(dict(do_sample=False, num_beams=1))
    elif decoding == "beam":
        base.update(dict(do_sample=False, num_beams=int(num_beams)))
    elif decoding == "nucleus":
        base.update(dict(do_sample=True, top_p=float(top_p), temperature=float(temperature), num_beams=1))
    else:
        raise ValueError(f"Unknown decoding: {decoding}")
    return base

# ====================== 生成（baseline / beta / add） ======================
@torch.no_grad()
def run_once_caption(model, processor, loader, gen_kwargs):
    all_caps, all_files = [], []
    for images, files, _gts in tqdm(loader, desc="  [baseline]", ncols=100, leave=False):
        prompts = [build_caption_prompt() for _ in images]
        inputs = processor(text=prompts, images=[[im] for im in images], return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        gen_ids = model.generate(**inputs, **gen_kwargs)
        outs = processor.batch_decode(gen_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
        # 轻量清洗：去掉可能残留的 'Answer:' 回显
        outs = [re.sub(r"^(?:\s*Answer:\s*)+", "", o).strip() for o in outs]
        all_caps.extend([o for o in outs]); all_files.extend(files)
    return all_caps, all_files

@torch.no_grad()
def run_once_beta(model, processor, loader, layer, alpha_max, k, c, pooling, gen_kwargs):
    tgt_layer = model.model.text_model.layers[layer].self_attn
    hook = BayesianGatingHookMaskedDynamic(
        max_alpha=alpha_max, sensitivity=k, concentration=c, carrier=mask_carrier,
        clamp=GATE_CLAMP, rms_match=False, record=True
    )
    stats = hook.fetch_last_stats()
    if stats and ("gate" in stats):
        g_mean = float(stats["gate"].mean())
        print(f"[Beta] gate_mean={g_mean:.3f} (Amax={alpha_max}, k={k}, c={c}, clamp={GATE_CLAMP})")
    handle = tgt_layer.register_forward_hook(hook)
    _hook_off(hook)

    all_caps, all_files = [], []
    try:
        for images, files, _gts in tqdm(loader, desc="  [beta]", ncols=100, leave=False):
            prompts = [build_caption_prompt() for _ in images]

            _hook_off(hook)
            v_batch = _card_from_same_prompts(model, processor, images, prompts, layer_idx=layer, pooling=pooling)

            inputs = processor(text=prompts, images=[[im] for im in images], return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            m = build_answer_mask_from_prompts(processor.tokenizer, prompts, inputs["input_ids"])
            attn = inputs["attention_mask"].unsqueeze(-1).to(m.dtype)
            m = m * attn
            mask_carrier.set(m.to(model.device))
            if hasattr(hook, "set_vector"): hook.set_vector(v_batch)
            
            with torch.no_grad():
                _hook_off(hook)                                   # 关：基线 logits
                out0 = model(**inputs, return_dict=True)
                _hook_on(hook)                                    # 开：注入后的 logits
                out1 = model(**inputs, return_dict=True)
            delta = (out1.logits.to(torch.float32) - out0.logits.to(torch.float32)).abs().mean().item()
            print(f"[sanity] mean|Δlogits| = {delta:.6f}")

            # （可选）看一下门控均值
            stats = hook.fetch_last_stats()
            if stats and ("gate" in stats):
                g_mean = float(stats["gate"].mean())
                print(f"[sanity] gate_mean = {g_mean:.3f}")

            _hook_on(hook)
            gen_ids = model.generate(**inputs, **gen_kwargs)
            _hook_off(hook)
            mask_carrier.clear()

            outs = processor.batch_decode(gen_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
            outs = [re.sub(r"^(?:\s*Answer:\s*)+", "", o).strip() for o in outs]
            all_caps.extend([o for o in outs]); all_files.extend(files)
    finally:
        handle.remove()
    return all_caps, all_files

@torch.no_grad()
def run_once_add(model, processor, loader, layer, alpha, pooling, gen_kwargs):
    tgt_layer = model.model.text_model.layers[layer].self_attn
    all_caps, all_files = [], []
    try:
        for images, files, _gts in tqdm(loader, desc="  [card_add]", ncols=100, leave=False):
            prompts = [build_caption_prompt() for _ in images]
            v_batch = _card_from_same_prompts(model, processor, images, prompts, layer_idx=layer, pooling=pooling)

            inputs = processor(text=prompts, images=[[im] for im in images], return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            m = build_answer_mask_from_prompts(processor.tokenizer, prompts, inputs["input_ids"])
            attn = inputs["attention_mask"].unsqueeze(-1).to(m.dtype)
            m = m * attn

            add_hook = SimpleAddHook(alpha=alpha)
            add_hook.set_vector(v_batch)
            add_hook.set_mask(m.to(model.device))
            handle = tgt_layer.register_forward_hook(add_hook)

            gen_ids = model.generate(**inputs, **gen_kwargs)
            handle.remove()

            outs = processor.batch_decode(gen_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
            outs = [re.sub(r"^(?:\s*Answer:\s*)+", "", o).strip() for o in outs]
            all_caps.extend([o for o in outs]); all_files.extend(files)
    finally:
        pass
    return all_caps, all_files

# ====================== 主程序 ======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100, help="子样本图片数（<=0=全量）")
    ap.add_argument("--limit_seed", type=int, default=None, help="子样本 seed（默认用外层 SEED）")

    # decoding 相关
    ap.add_argument("--decoding", type=str, default="greedy", choices=["greedy","beam","nucleus"])
    ap.add_argument("--num_beams", type=int, default=5, help="beam search 的 beam 数（decoding=beam 时生效）")
    ap.add_argument("--length_penalty", type=float, default=1.0, help="生成长度惩罚（beam/greedy均可用）")
    ap.add_argument("--early_stopping", type=str, default="false",
                    help="early_stopping={false|true|never}")

    # 采样参数（nucleus 时生效）
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--max_new_tokens", type=int, default=CAP_MAX_NEW_TOKENS)

    # 防复述
    ap.add_argument("--no_repeat_ngram_size", type=int, default=6)
    ap.add_argument("--repetition_penalty", type=float, default=1.08)
    ap.add_argument("--ban_echo", action="store_true",
                    help="启用 bad_words_ids，禁止复述 Instruction/Image/Question/Answer/Describe the image 等锚词")

    # which experiments
    ap.add_argument("--run_beta", action="store_true")
    ap.add_argument("--run_add", action="store_true")
    ap.add_argument("--pool", type=str, default="attn", choices=["attn","mean"])
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR_CHAIR, exist_ok=True)

    # model
    dtype = torch.bfloat16 if DTYPE=="bf16" else (torch.float16 if DTYPE=="fp16" else torch.float32)
    print(f"⏳ Loading model: {MODEL_ID}")
    model = Idefics2ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, cache_dir=CACHE_DIR, device_map={'': DEVICE}
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    tok = processor.tokenizer
    model.eval(); print("✅ Model ready.")

    alias2canon = load_aliases()

    # bad_words_ids（可选）
    bad_words_ids = None
    if args.ban_echo:
        ban_phrases = [
            "Instruction:", "Image:", "Question:", "Answer:",
            "Describe the image", "describe the image", "Describe the image in rich detail"
        ]
        bad_words_ids = make_bad_words_ids(tok, ban_phrases)

    # 统一生成参数
    gen_kwargs = build_gen_kwargs(
        decoding=args.decoding,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        early_stopping_arg=args.early_stopping,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
        bad_words_ids=bad_words_ids
    )
    print(f"[Gen] decoding={args.decoding} num_beams={gen_kwargs.get('num_beams')} "
          f"do_sample={gen_kwargs.get('do_sample', False)} top_p={gen_kwargs.get('top_p', None)} "
          f"temp={gen_kwargs.get('temperature', None)} early_stopping={gen_kwargs.get('early_stopping')} "
          f"len_penalty={gen_kwargs.get('length_penalty')} ngram={gen_kwargs.get('no_repeat_ngram_size')} "
          f"rep_pen={gen_kwargs.get('repetition_penalty')} ban_echo={args.ban_echo}")

    for seed in SEEDS:
        set_global_seed(seed)
        ds = CHAIRImageDataset(
            IMAGE_DIR, COCO_INSTANCES_JSON,
            limit=(args.limit if (args.limit is None or args.limit > 0) else 0),
            limit_seed=(seed if args.limit_seed is None else args.limit_seed)
        )
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=chair_collate)
        print(f"📦 CHAIR subset = {len(ds)} images")
        sub_tag = f"_sub{len(ds)}_{args.decoding}"

        # ---------- Baseline ----------
        base_name = f"CHAIR_ICL_seed{seed}{sub_tag}"
        pred_path = os.path.join(RESULTS_DIR_CHAIR, f"pred_{base_name}.json")
        met_path  = os.path.join(RESULTS_DIR_CHAIR, f"metrics_{base_name}.json")
        if os.path.exists(pred_path) and os.path.exists(met_path):
            print(f"⏭️  Skip baseline exists: {pred_path}")
        else:
            caps, files = run_once_caption(model, processor, loader, gen_kwargs)
            with open(pred_path, "w") as f:
                json.dump([{"file_name": fn, "caption": cap} for fn, cap in zip(files, caps)], f, ensure_ascii=False, indent=2)
            gts = [r["gt"] for r in ds.rows]
            met = evaluate_chair(files, caps, gts, alias2canon, ci_alpha=0.05, boot_B=2000)
            with open(met_path, "w") as f: json.dump(met, f, indent=2)
            print(f"✅ Baseline: CHAIRs={met['CHAIRs']:.3f}  CHAIRi={met['CHAIRi']:.3f}")

        # ---------- CARD+Beta ----------
        if args.run_beta:
            for L in INJECTION_LAYERS:
                for amax in BETA_ALPHA_MAX:
                    for kk in BETA_K:
                        for cc in BETA_C:
                            name = f"CHAIR_CARD_Beta_seed{seed}_L{L}_{args.pool}_A{amax}_K{kk}_C{cc}{sub_tag}"
                            pred_o = os.path.join(RESULTS_DIR_CHAIR, f"pred_{name}.json")
                            met_o  = os.path.join(RESULTS_DIR_CHAIR, f"metrics_{name}.json")
                            if os.path.exists(pred_o) and os.path.exists(met_o):
                                print(f"⏭️  Skip exists: {pred_o}"); continue
                            print(f"\n▶️ {name}")
                            loader2 = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                                 num_workers=NUM_WORKERS, collate_fn=chair_collate)
                            caps, files = run_once_beta(
                                model, processor, loader2, L, amax, kk, cc,
                                pooling=args.pool, gen_kwargs=gen_kwargs
                            )
                            with open(pred_o, "w") as f:
                                json.dump([{"file_name": fn, "caption": cap} for fn, cap in zip(files, caps)], f, ensure_ascii=False, indent=2)
                            gts = [r["gt"] for r in ds.rows]
                            met = evaluate_chair(files, caps, gts, alias2canon, ci_alpha=0.05, boot_B=2000)
                            with open(met_o, "w") as f: json.dump(met, f, indent=2)
                            print(f"✅ Saved: CHAIRs={met['CHAIRs']:.3f}  CHAIRi={met['CHAIRi']:.3f}")

        # ---------- 简单加法（消融） ----------
        if args.run_add:
            for L in INJECTION_LAYERS:
                for a in ADD_ALPHA:
                    name = f"CHAIR_CARD_Add_seed{seed}_L{L}_{args.pool}_A{a}{sub_tag}"
                    pred_o = os.path.join(RESULTS_DIR_CHAIR, f"pred_{name}.json")
                    met_o  = os.path.join(RESULTS_DIR_CHAIR, f"metrics_{name}.json")
                    if os.path.exists(pred_o) and os.path.exists(met_o):
                        print(f"⏭️  Skip exists: {pred_o}"); continue
                    print(f"\n▶️ {name}")
                    loader2 = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                         num_workers=NUM_WORKERS, collate_fn=chair_collate)
                    caps, files = run_once_add(
                        model, processor, loader2, L, a, pooling=args.pool, gen_kwargs=gen_kwargs
                    )
                    with open(pred_o, "w") as f:
                        json.dump([{"file_name": fn, "caption": cap} for fn, cap in zip(files, caps)], f, ensure_ascii=False, indent=2)
                    gts = [r["gt"] for r in ds.rows]
                    met = evaluate_chair(files, caps, gts, alias2canon, ci_alpha=0.05, boot_B=2000)
                    with open(met_o, "w") as f: json.dump(met, f, indent=2)
                    print(f"✅ Saved: CHAIRs={met['CHAIRs']:.3f}  CHAIRi={met['CHAIRi']:.3f}")

    print("\n✅ CHAIR done. Results in:", RESULTS_DIR_CHAIR)

if __name__ == "__main__":
    main()
