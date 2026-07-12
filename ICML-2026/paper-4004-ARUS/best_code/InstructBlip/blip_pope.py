# -*- coding: utf-8 -*-
# blip_pope_card_firstlogit_decoding.py
#
# InstructBLIP + POPE：baseline / CARD-Beta / 简单加法
# 仅支持 beam / nucleus 解码；判别基于第一步 logits（不做字符串解码）

import os, json, argparse, math, random, csv
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
)

# ======= 你的工程配置 / 工具 =======
from config_hal import (
    DATA_DIR, IMAGE_DIR,
    POPE_DIR, POPE_SPLITS,
    RESULTS_DIR_POPE,
    MODEL_ID, CACHE_DIR, DEVICE, DTYPE,
    BATCH_SIZE, NUM_WORKERS,
    SEEDS,
    INJECTION_LAYERS, EGR_POOLINGS,
    BETA_ALPHA_MAX, BETA_K, BETA_C, GATE_CLAMP,
    MAX_NEW_TOKENS_POPE,  # 未直接使用（第一步 logits），仅保留导入
)
from methods import (
    set_global_seed,
    mask_carrier,
    build_answer_mask_from_prompts,
    BayesianGatingHookMaskedDynamic,
)

# ====================== dtype 解析 ======================
def _torch_dtype_from_str(s: str):
    s = (s or "fp32").lower()
    if s == "bf16": return torch.bfloat16
    if s == "fp16": return torch.float16
    return torch.float32

# ====================== yes/no 归一化（用于指标一致性） ======================
def norm_yesno(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.split("</s>")[0]
    for sep in ["\n", ".", ",", "!", "?", "  "]:
        s = s.split(sep)[0]
    s = s.strip().strip('"').strip("'")
    yes_set = {"yes","y","yeah","yep","true","1","correct","of course","sure","affirmative"}
    no_set  = {"no","n","nope","false","0","incorrect","negative","not"}
    if s.startswith("yes"): return "yes"
    if s.startswith("no"):  return "no"
    if s in yes_set: return "yes"
    if s in no_set:  return "no"
    tok = s.split()
    if tok:
        if tok[0] in yes_set: return "yes"
        if tok[0] in no_set:  return "no"
    if "yes" in s: return "yes"
    if "no"  in s: return "no"
    return "no"
def _should_skip(out_pred_path: str, out_met_path: str) -> bool:
    # 仅根据 metrics 文件是否存在且非空来跳过；也可以加更严格校验（JSON 可读且含 'accuracy'）
    if os.path.exists(out_met_path) and os.path.getsize(out_met_path) > 0:
        try:
            with open(out_met_path, "r", encoding="utf-8") as f:
                j = json.load(f)
            return isinstance(j, dict) and ("accuracy" in j or "f1" in j)
        except Exception:
            return False
    return False
# ====================== Wilson & Bootstrap CI（与原表头兼容） ======================
def _conf_wilson(p: float, n: int, z: float=1.959963984540054) -> Tuple[float,float]:
    if n <= 0: return float("nan"), float("nan")
    denom = 1.0 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half = (z/denom) * math.sqrt((p*(1-p))/n + (z*z)/(4*n*n))
    return center, half

def _counts(y_true: List[int], y_pred: List[int]) -> Tuple[int,int,int,int]:
    tp = sum(1 for yp, yt in zip(y_pred, y_true) if yp==1 and yt==1)
    tn = sum(1 for yp, yt in zip(y_pred, y_true) if yp==0 and yt==0)
    fp = sum(1 for yp, yt in zip(y_pred, y_true) if yp==1 and yt==0)
    fn = sum(1 for yp, yt in zip(y_pred, y_true) if yp==0 and yt==1)
    return tp, fp, tn, fn

def _f1_from_counts(tp, fp, fn):
    prec = tp / max(1, (tp+fp))
    rec  = tp / max(1, (tp+fn))
    if prec + rec == 0: return 0.0, prec, rec
    return 2*prec*rec/(prec+rec), prec, rec

def _bootstrap_f1_ci(y_true: List[int], y_pred: List[int], B: int=2000, seed: int=2025, alpha: float=0.05):
    import numpy as np
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = np.arange(n)
    f1s = []
    for _ in range(B):
        bs = rng.choice(idx, size=n, replace=True)
        tp, fp, tn, fn = _counts([y_true[i] for i in bs], [y_pred[i] for i in bs])
        f1, _, _ = _f1_from_counts(tp, fp, fn)
        f1s.append(f1)
    lo, hi = float(np.percentile(f1s, 100*alpha/2)), float(np.percentile(f1s, 100*(1-alpha/2)))
    return lo, hi

def metrics_with_ci(preds: List[str], gts: List[str], ci_alpha=0.05, boot_B=2000) -> Dict[str, float]:
    y_pred = [1 if norm_yesno(p) == "yes" else 0 for p in preds]
    y_true = [1 if norm_yesno(g) == "yes" else 0 for g in gts]
    tp, fp, tn, fn = _counts(y_true, y_pred)
    n = tp + fp + tn + fn
    acc = (tp + tn) / max(1, n)
    prec = tp / max(1, (tp+fp))
    rec  = tp / max(1, (tp+fn))
    f1, _, _ = _f1_from_counts(tp, fp, fn)
    acc_c, acc_h = _conf_wilson(acc, n)
    prec_c, prec_h = _conf_wilson(prec, tp+fp) if (tp+fp)>0 else (float("nan"), float("nan"))
    rec_c,  rec_h  = _conf_wilson(rec,  tp+fn) if (tp+fn)>0 else (float("nan"), float("nan"))
    f1_lo, f1_hi = _bootstrap_f1_ci(y_true, y_pred, B=boot_B, seed=2025, alpha=ci_alpha)
    return {
        "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": acc, "acc_center": acc_c, "acc_halfwidth": acc_h,
        "precision": prec, "prec_center": prec_c, "prec_halfwidth": prec_h, "prec_n": tp+fp,
        "recall": rec, "rec_center": rec_c, "rec_halfwidth": rec_h, "rec_n": tp+fn,
        "f1": f1, "f1_ci_low": f1_lo, "f1_ci_high": f1_hi
    }

# ====================== 数据集（POPE） ======================
class POPEDataset(Dataset):
    def __init__(self, split: str, limit: int = -1, seed: int = 42):
        self.rows = []
        candidates = [
            os.path.join(POPE_DIR, f"pope_{split}.jsonl"),
            os.path.join(POPE_DIR, f"pope_{split}.json"),
            os.path.join(POPE_DIR, f"coco_pope_{split}.jsonl"),
            os.path.join(POPE_DIR, f"coco_pope_{split}.json"),
            os.path.join(POPE_DIR, f"{split}.jsonl"),
            os.path.join(POPE_DIR, f"{split}.json"),
        ]
        file_path = next((p for p in candidates if os.path.exists(p)), None)
        if file_path is None:
            raise FileNotFoundError(f"POPE split='{split}' 未找到。候选: {candidates}")

        records = []
        if file_path.endswith(".jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    records.append(json.loads(line))
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            try:
                obj = json.loads(txt)
            except json.JSONDecodeError:
                for line in txt.splitlines():
                    line = line.strip()
                    if not line: continue
                    records.append(json.loads(line))
            else:
                if isinstance(obj, list):
                    records = obj
                else:
                    for k in ("data","samples","items","rows","entries","questions","annotations","results", split):
                        v = obj.get(k)
                        if isinstance(v, list): records = v; break
                    if not records:
                        best = None; L = -1
                        for v in obj.values():
                            if isinstance(v, list) and len(v)>L:
                                best, L = v, len(v)
                        if best is not None: records = best

        def _parse_label(v):
            if v is None: return None
            s = str(v).strip().lower()
            if s in {"yes","y","true","1"}: return "yes"
            if s in {"no","n","false","0"}:  return "no"
            return None

        for i, d in enumerate(records):
            q = d.get("text") or d.get("question") or d.get("prompt")
            lab = _parse_label(d.get("label") or d.get("answer") or d.get("gt") or d.get("gold"))
            img_name = d.get("image") or d.get("image_file") or d.get("file_name") \
                       or d.get("filename") or d.get("image_path") or d.get("path")
            if not (q and lab and img_name): continue
            img_path = img_name if os.path.isabs(img_name) else os.path.join(IMAGE_DIR, img_name)
            if not os.path.exists(img_path): continue
            self.rows.append({"question": q, "label": lab, "image_path": img_path, "qid": i+1})

        if limit > 0 and len(self.rows) > limit:
            random.seed(seed)
            self.rows = random.sample(self.rows, limit)

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        img = Image.open(r["image_path"]).convert("RGB")
        return {"image": img, "question": r["question"], "label": r["label"], "qid": r["qid"]}

def collate_fn(batch):
    return {
        "images":    [b["image"] for b in batch],
        "questions": [b["question"] for b in batch],
        "labels":    [b["label"] for b in batch],
        "qids":      [b["qid"] for b in batch],
    }

# ====================== Prompt ======================
def build_pope_prompt(q: str) -> str:
    # 必须包含 "Answer:" 以便 methods.build_answer_mask_from_prompts 正确定位 Answer 段
    return f"Question: {q}\nAnswer:"

# ====================== yes/no 首 token 组（含空格/换行/大小写） ======================
def collect_yes_no_first_token_groups(tokenizer):
    def first_id(text: str) -> Optional[int]:
        ids = tokenizer(text, add_special_tokens=False).input_ids
        return int(ids[0]) if ids else None
    def uniq(xs): return sorted(set([x for x in xs if x is not None]))
    yes_vars, no_vars = [], []
    for base in ["yes", "Yes"]:
        for pref in ["", " ", "\n"]:
            yes_vars.append(first_id(pref + base))
    for base in ["no", "No"]:
        for pref in ["", " ", "\n"]:
            no_vars.append(first_id(pref + base))
    yids, nids = uniq(yes_vars), uniq(no_vars)
    if not yids or not nids:
        raise ValueError("收集 yes/no 首 token 失败，请检查 tokenizer 配置。")
    return yids, nids

# ====================== 折叠 beam 形状 ======================
def _collapse_beam_logits(step0_logits: torch.Tensor, batch_size: int, num_beams: int) -> torch.Tensor:
    """
    输入: [B * num_beams, V]  -> 输出: [B, V]（按 beam 维取 max）
    """
    if num_beams <= 1:
        return step0_logits
    V = step0_logits.size(-1)
    step0_logits = step0_logits.view(batch_size, num_beams, V)
    return torch.max(step0_logits, dim=1).values  # [B, V]

# ====================== 取第一步 logits（支持 beam / nucleus） ======================
@torch.no_grad()
def first_step_logits(model, inputs, decoding: str, num_beams: int, top_p: float, temperature: float, top_k: int,
                      batch_size: int) -> torch.Tensor:
    decoding = decoding.lower()
    if decoding == "beam":
        gen_kwargs = dict(do_sample=False, num_beams=max(2, int(num_beams)))
    elif decoding == "nucleus":
        gen_kwargs = dict(do_sample=True, top_p=float(top_p), temperature=float(temperature), top_k=int(top_k), num_beams=1)
    else:
        raise ValueError("只支持 'beam' 或 'nucleus'。")

    out = model.generate(
        **inputs,
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_scores=True,
        **gen_kwargs,
    )
    step0 = out.scores[0]  # beam: [B*num_beams, V] ; nucleus: [B, V]
    if decoding == "beam":
        step0 = _collapse_beam_logits(step0, batch_size=batch_size, num_beams=gen_kwargs["num_beams"])
    return step0

def yesno_from_logits(next_logits: torch.Tensor, yes_ids: List[int], no_ids: List[int],
                      no_token_penalty: float = 0.0) -> List[str]:
    if no_token_penalty > 0 and len(no_ids) > 0:
        next_logits[:, no_ids] -= no_token_penalty
    yes_max, _ = torch.max(next_logits[:, yes_ids], dim=1)
    no_max,  _ = torch.max(next_logits[:, no_ids],  dim=1)
    return ["yes" if y.item() > n.item() else "no" for y, n in zip(yes_max, no_max)]

# ====================== CARD 向量（文本侧近似） ======================
@torch.no_grad()
def compute_card_vector_batch_blip(model, processor, images, questions, layer_idx: int, pooling: str = "attn") -> torch.Tensor:
    device = next(model.parameters()).device
    texts = [build_pope_prompt(q) for q in questions]
    enc = processor(images=images, text=texts, return_tensors="pt", padding=True, truncation=True)
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    lm = model.language_model

    # T5 encoder
    if hasattr(lm, "encoder"):
        out = lm.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hs_all = out.hidden_states  # [emb, h1, h2, ...]
        idx = max(0, min(layer_idx, len(hs_all)-1))
        h = hs_all[idx]  # [B, T, H]
        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            if pooling == "attn":
                w = (h.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6) * m
                v = (h * w).sum(dim=1) / (w.sum(dim=1) + 1e-6)
            else:
                v = (h * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))
        else:
            v = h.mean(dim=1)
        return F.normalize(v, p=2, dim=-1)

    # LLaMA/Vicuna decoder-only
    if hasattr(lm, "model") and hasattr(lm.model, "embed_tokens"):
        emb = lm.model.embed_tokens(input_ids)  # [B, T, H]
        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).to(emb.dtype)
            if pooling == "attn":
                w = (emb.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6) * m
                v = (emb * w).sum(dim=1) / (w.sum(dim=1) + 1e-6)
            else:
                v = (emb * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))
        else:
            v = emb.mean(dim=1)
        return F.normalize(v, p=2, dim=-1)

    raise RuntimeError("不支持的 InstructBLIP language_model 结构，无法计算 CARD 向量。")

# ====================== 定位自注意力层（InstructBLIP） ======================
def get_decoder_self_attn_module(model, layer_idx: int):
    lm = model.language_model
    # T5：decoder.block[i].layer[0]
    if hasattr(lm, "decoder") and hasattr(lm.decoder, "block"):
        blocks = lm.decoder.block
        L = len(blocks)
        if not (0 <= layer_idx < L):
            raise IndexError(f"T5 decoder 层数={L}，传入 L={layer_idx} 不合法。")
        return blocks[layer_idx].layer[0]

    # LLaMA/Vicuna：model.layers[i].self_attn
    if hasattr(lm, "model") and hasattr(lm.model, "layers"):
        layers = lm.model.layers
        L = len(layers)
        if not (0 <= layer_idx < L):
            raise IndexError(f"LLaMA decoder 层数={L}，传入 L={layer_idx} 不合法。")
        return layers[layer_idx].self_attn

    raise RuntimeError("未能定位到语言模型的自注意力子层。")

# ====================== 简单加法 PreHook ======================
class SimpleAddPreHook:
    """hs <- hs + alpha * expand(v) * mask（在 self_attn 前注入）"""
    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.v_batch: Optional[torch.Tensor] = None
        self.h = None

    def set_vector(self, v_batch: torch.Tensor):
        self.v_batch = v_batch

    def register(self, target_layer):
        self.h = target_layer.register_forward_pre_hook(self, with_kwargs=True)

    def remove(self):
        if self.h: self.h.remove(); self.h = None
    '''
    def __call__(self, module, args, kwargs):
        hs = kwargs.get("hidden_states", None)
        if hs is None and len(args) > 0:
            hs = args[0]
        if hs is None or self.v_batch is None:
            return (args, kwargs)

        B, T, H = hs.size()
        v = self.v_batch.to(device=hs.device, dtype=hs.dtype)
        B0 = v.size(0)
        if B0 != B:
            if B % B0 == 0:
                v = v.repeat_interleave(B // B0, dim=0)
            else:
                reps = (B + B0 - 1) // B0
                v = v.repeat(reps, 1)[:B, :]
        vT = v.unsqueeze(1).expand(B, T, H)

        # Answer 段 mask（若未设置则全 1）
        m = getattr(mask_carrier, "mask", None)
        if m is None:
            mask = hs.new_ones(B, T, 1)
        else:
            mask = m.to(hs.device, dtype=hs.dtype)
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            if mask.size(1) != T:
                if mask.size(1) > T:
                    mask = mask[:, -T:, :]
                else:
                    pad = mask.new_zeros(B, T - mask.size(1), 1)
                    mask = torch.cat([mask, pad], dim=1)

        hs_new = hs + self.alpha * vT * mask

        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = hs_new
        else:
            new_args = list(args)
            if new_args:
                new_args[0] = hs_new
            args = tuple(new_args)
        return (args, kwargs)
    '''
    def __call__(self, module, args, kwargs):
        # 1) 取 hidden_states
        hs = kwargs.get("hidden_states", None)
        if hs is None and len(args) > 0:
            hs = args[0]
        if hs is None or self.v_batch is None:
            return (args, kwargs)

        B, T, H = hs.size()  # 注意：在 beam 下这里已经是 B*num_beams
        # 2) 对齐 v 的 batch 维
        v = self.v_batch.to(device=hs.device, dtype=hs.dtype)
        B0 = v.size(0)
        if B0 != B:
            if B % B0 == 0:
                v = v.repeat_interleave(B // B0, dim=0)
            else:
                reps = (B + B0 - 1) // B0
                v = v.repeat(reps, 1)[:B, :]
        vT = v.unsqueeze(1).expand(B, T, H)

        # 3) 取 Answer 段 mask（若无则全 1），并先把 batch 对齐到 B（考虑 beam 扩维）
        m = getattr(mask_carrier, "mask", None)
        if m is None:
            mask = hs.new_ones(B, T, 1)
        else:
            mask = m.to(hs.device, dtype=hs.dtype)
            # 3.1 对齐 batch 维
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)  # [B0, Tm, 1]
            Bm, Tm = mask.size(0), mask.size(1)
            if Bm != B:
                if B % Bm == 0:
                    mask = mask.repeat_interleave(B // Bm, dim=0)
                else:
                    reps = (B + Bm - 1) // Bm
                    mask = mask.repeat(reps, 1, 1)[:B, :, :]

            # 3.2 对齐时间维（左填充 tokenizer 也能兼容）
            if Tm != T:
                if Tm > T:
                    mask = mask[:, -T:, :]            # 截到右侧 T
                else:
                    pad = mask.new_zeros(B, T - Tm, 1)
                    mask = torch.cat([mask, pad], dim=1)  # 右侧补零

        # 4) 应用加法注入
        hs_new = hs + self.alpha * vT * mask

        # 5) 写回 kwargs/args
        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = hs_new
        else:
            new_args = list(args)
            if new_args:
                new_args[0] = hs_new
            args = tuple(new_args)
        return (args, kwargs)
# ====================== 单 batch 预测（首步打分 + 解码策略） ======================
@torch.no_grad()
def predict_batch_firstlogit(model, processor, images, questions, yes_ids, no_ids,
                             decoding: str, num_beams: int, top_p: float, temperature: float, top_k: int,
                             no_token_penalty: float, set_mask: bool = True):
    device = next(model.parameters()).device
    prompts = [build_pope_prompt(q) for q in questions]
    enc = processor(images=images, text=prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in enc.items()}

    # Answer 段 mask（供 BetaHook / AddHook 使用）
    if set_mask:
        m = build_answer_mask_from_prompts(processor.tokenizer, prompts, inputs["input_ids"])
        if "attention_mask" in inputs and m.size(1) == inputs["attention_mask"].size(1):
            m = m * inputs["attention_mask"].unsqueeze(-1).to(m.dtype)
        mask_carrier.set(m.to(device))

    B = len(images)
    # 取第 1 步 logits（beam/nucleus）
    logits = first_step_logits(model, inputs, decoding=decoding, num_beams=num_beams,
                               top_p=top_p, temperature=temperature, top_k=top_k, batch_size=B)

    # nucleus 极端回退：若某样本 yes/no 两组都被 top-p 过滤为 -inf，则回退到 greedy 的第一步 logits
    if decoding == "nucleus":
        with torch.no_grad():
            yes_finite = torch.isfinite(logits[:, yes_ids]).any(dim=1)
            no_finite  = torch.isfinite(logits[:, no_ids]).any(dim=1)
            bad_mask = ~(yes_finite | no_finite)  # 两组都没 finite
            if bad_mask.any():
                out_g = model.generate(
                    **inputs, max_new_tokens=1,
                    return_dict_in_generate=True, output_scores=True,
                    do_sample=False, num_beams=1
                )
                logits_g = out_g.scores[0]
                logits[bad_mask] = logits_g[bad_mask]

    preds  = yesno_from_logits(logits, yes_ids, no_ids, no_token_penalty=no_token_penalty)
    mask_carrier.clear()
    return preds

# ====================== Baseline / Beta / Add 实验循环 ======================
'''
@torch.no_grad()
def run_experiment_baseline(model, processor, loader,
                            yes_ids, no_ids, decoding, num_beams, top_p, temperature, top_k,
                            no_token_penalty, summary_csv_path, split, seed, subset_size):
    name = f"POPE_{split}_firstlogit_baseline_{decoding}_seed{seed}"
    out_pred = os.path.join(RESULTS_DIR_POPE, f"pred_{name}.json")
    out_met  = os.path.join(RESULTS_DIR_POPE, f"metrics_{name}.json")

    preds, gts = [], []
    for batch in tqdm(loader, ncols=100, leave=False, desc=f"[Baseline-{decoding}]"):
        images, questions, labels = batch["images"], batch["questions"], batch["labels"]
        batch_preds = predict_batch_firstlogit(
            model, processor, images, questions, yes_ids, no_ids,
            decoding=decoding, num_beams=num_beams, top_p=top_p, temperature=temperature, top_k=top_k,
            no_token_penalty=no_token_penalty, set_mask=False  # baseline不需要mask
        )
        preds.extend(batch_preds); gts.extend(labels)

    met = metrics_with_ci(preds, gts, ci_alpha=0.05, boot_B=2000)
    os.makedirs(RESULTS_DIR_POPE, exist_ok=True)
    with open(out_pred, "w") as f: json.dump(preds, f, ensure_ascii=False)
    with open(out_met,  "w") as f: json.dump(met,   f, ensure_ascii=False, indent=2)
    print(f"✅ Saved(Baseline-{decoding}): acc={met['accuracy']:.4f} f1={met['f1']:.4f}")

    with open(summary_csv_path, "a", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow([
            os.path.basename(out_met), f"baseline_firstlogit_{decoding}", split, seed, 0, "-", "-", "-", "-", "-", subset_size,
            met["accuracy"],
            met["acc_center"]-met["acc_halfwidth"], met["acc_center"]+met["acc_halfwidth"],
            met["precision"],
            (met["prec_center"]-met["prec_halfwidth"]) if not math.isnan(met["prec_halfwidth"]) else "",
            (met["prec_center"]+met["prec_halfwidth"]) if not math.isnan(met["prec_halfwidth"]) else "",
            met["recall"],
            (met["rec_center"]-met["rec_halfwidth"]) if not math.isnan(met["rec_halfwidth"]) else "",
            (met["rec_center"]+met["rec_halfwidth"]) if not math.isnan(met["rec_halfwidth"]) else "",
            met["f1"], met["f1_ci_low"], met["f1_ci_high"]
        ])

@torch.no_grad()
def run_experiment_beta(model, processor, loader, layers, pools,
                        alphas, ks, cs, yes_ids, no_ids,
                        decoding, num_beams, top_p, temperature, top_k,
                        no_token_penalty, summary_csv_path,
                        split, seed, subset_size):
    for L in layers:
        target = get_decoder_self_attn_module(model, L)
        for pool in pools:
            for A in alphas:
                for K in ks:
                    for C in cs:
                        name = f"POPE_{split}_firstlogit_CARD_Beta_{decoding}_seed{seed}_L{L}_{pool}_A{A}_K{K}_C{C}"
                        out_pred = os.path.join(RESULTS_DIR_POPE, f"pred_{name}.json")
                        out_met  = os.path.join(RESULTS_DIR_POPE, f"metrics_{name}.json")

                        preds, gts = [], []
                        hook = BayesianGatingHookMaskedDynamic(
                            max_alpha=A, sensitivity=K, concentration=C,
                            carrier=mask_carrier, clamp=GATE_CLAMP, rms_match=False, record=False
                        )
                        handle = target.register_forward_hook(hook)
                        try:
                            for batch in tqdm(loader, ncols=100, leave=False, desc=f"[Beta-{decoding}] L{L}|{pool}|A{A}|K{K}|C{C}"):
                                images, questions, labels = batch["images"], batch["questions"], batch["labels"]

                                v_batch = compute_card_vector_batch_blip(
                                    model, processor, images, questions, layer_idx=L, pooling=pool
                                )
                                hook.set_vector(v_batch)

                                batch_preds = predict_batch_firstlogit(
                                    model, processor, images, questions, yes_ids, no_ids,
                                    decoding=decoding, num_beams=num_beams, top_p=top_p, temperature=temperature, top_k=top_k,
                                    no_token_penalty=no_token_penalty, set_mask=True
                                )
                                preds.extend(batch_preds); gts.extend(labels)
                        finally:
                            handle.remove()

                        met = metrics_with_ci(preds, gts, ci_alpha=0.05, boot_B=2000)
                        os.makedirs(RESULTS_DIR_POPE, exist_ok=True)
                        with open(out_pred, "w") as f: json.dump(preds, f, ensure_ascii=False)
                        with open(out_met,  "w") as f: json.dump(met,   f, ensure_ascii=False, indent=2)
                        print(f"✅ Saved(Beta-{decoding}): acc={met['accuracy']:.4f} f1={met['f1']:.4f}")

                        with open(summary_csv_path, "a", newline="") as fcsv:
                            w = csv.writer(fcsv)
                            w.writerow([
                                os.path.basename(out_met), f"beta_firstlogit_{decoding}", split, seed, 0, L, pool, A, K, C, subset_size,
                                met["accuracy"],
                                met["acc_center"]-met["acc_halfwidth"], met["acc_center"]+met["acc_halfwidth"],
                                met["precision"],
                                (met["prec_center"]-met["prec_halfwidth"]) if not math.isnan(met["prec_halfwidth"]) else "",
                                (met["prec_center"]+met["prec_halfwidth"]) if not math.isnan(met["prec_halfwidth"]) else "",
                                met["recall"],
                                (met["rec_center"]-met["rec_halfwidth"]) if not math.isnan(met["rec_halfwidth"]) else "",
                                (met["rec_center"]+met["rec_halfwidth"]) if not math.isnan(met["rec_halfwidth"]) else "",
                                met["f1"], met["f1_ci_low"], met["f1_ci_high"]
                            ])

@torch.no_grad()
def run_experiment_add(model, processor, loader, layers, pools,
                       add_alphas, yes_ids, no_ids,
                       decoding, num_beams, top_p, temperature, top_k,
                       no_token_penalty, summary_csv_path,
                       split, seed, subset_size):
    for L in layers:
        target = get_decoder_self_attn_module(model, L)
        for pool in pools:
            for A in add_alphas:
                name = f"POPE_{split}_firstlogit_CARD_Add_{decoding}_seed{seed}_L{L}_{pool}_A{A}"
                out_pred = os.path.join(RESULTS_DIR_POPE, f"pred_{name}.json")
                out_met  = os.path.join(RESULTS_DIR_POPE, f"metrics_{name}.json")

                preds, gts = [], []
                add_hook = SimpleAddPreHook(alpha=A)
                add_hook.register(target)
                try:
                    for batch in tqdm(loader, ncols=100, leave=False, desc=f"[Add-{decoding}] L{L}|{pool}|A{A}"):
                        images, questions, labels = batch["images"], batch["questions"], batch["labels"]

                        v_batch = compute_card_vector_batch_blip(
                            model, processor, images, questions, layer_idx=L, pooling=pool
                        )
                        add_hook.set_vector(v_batch)

                        batch_preds = predict_batch_firstlogit(
                            model, processor, images, questions, yes_ids, no_ids,
                            decoding=decoding, num_beams=num_beams, top_p=top_p, temperature=temperature, top_k=top_k,
                            no_token_penalty=no_token_penalty, set_mask=True
                        )
                        preds.extend(batch_preds); gts.extend(labels)
                finally:
                    add_hook.remove()

                met = metrics_with_ci(preds, gts, ci_alpha=0.05, boot_B=2000)
                os.makedirs(RESULTS_DIR_POPE, exist_ok=True)
                with open(out_pred, "w") as f: json.dump(preds, f, ensure_ascii=False)
                with open(out_met,  "w") as f: json.dump(met,   f, ensure_ascii=False, indent=2)
                print(f"✅ Saved(Add-{decoding}):  acc={met['accuracy']:.4f} f1={met['f1']:.4f}")

                with open(summary_csv_path, "a", newline="") as fcsv:
                    w = csv.writer(fcsv)
                    w.writerow([
                        os.path.basename(out_met), f"add_firstlogit_{decoding}", split, seed, 0, L, pool, A, "-", "-", subset_size,
                        met["accuracy"],
                        met["acc_center"]-met["acc_halfwidth"], met["acc_center"]+met["acc_halfwidth"],
                        met["precision"],
                        (met["prec_center"]-met["prec_halfwidth"]) if not math.isnan(met["prec_halfwidth"]) else "",
                        (met["prec_center"]+met["prec_halfwidth"]) if not math.isnan(met["prec_halfwidth"]) else "",
                        met["recall"],
                        (met["rec_center"]-met["rec_halfwidth"]) if not math.isnan(met["rec_halfwidth"]) else "",
                        (met["rec_center"]+met["rec_halfwidth"]) if not math.isnan(met["rec_halfwidth"]) else "",
                        met["f1"], met["f1_ci_low"], met["f1_ci_high"]
                    ])
'''
@torch.no_grad()
def run_experiment_baseline(model, processor, loader,
                            yes_ids, no_ids, decoding, num_beams, top_p, temperature, top_k,
                            no_token_penalty, summary_csv_path, split, seed, subset_size,
                            skip_existing: bool):
    name = f"POPE_{split}_firstlogit_baseline_{decoding}_seed{seed}"
    out_pred = os.path.join(RESULTS_DIR_POPE, f"pred_{name}.json")
    out_met  = os.path.join(RESULTS_DIR_POPE, f"metrics_{name}.json")

    # ==== 已有结果则跳过 ====
    if skip_existing and _should_skip(out_pred, out_met):
        print(f"⏭️ Skip(Baseline-{decoding}): {os.path.basename(out_met)} 已存在。")
        return

    preds, gts = [], []
    for batch in tqdm(loader, ncols=100, leave=False, desc=f"[Baseline-{decoding}]"):
        images, questions, labels = batch["images"], batch["questions"], batch["labels"]
        batch_preds = predict_batch_firstlogit(
            model, processor, images, questions, yes_ids, no_ids,
            decoding=decoding, num_beams=num_beams, top_p=top_p, temperature=temperature, top_k=top_k,
            no_token_penalty=no_token_penalty, set_mask=False  # baseline不需要mask
        )
        preds.extend(batch_preds); gts.extend(labels)

    met = metrics_with_ci(preds, gts, ci_alpha=0.05, boot_B=2000)
    os.makedirs(RESULTS_DIR_POPE, exist_ok=True)
    with open(out_pred, "w") as f: json.dump(preds, f, ensure_ascii=False)
    with open(out_met,  "w") as f: json.dump(met,   f, ensure_ascii=False, indent=2)
    print(f"✅ Saved(Baseline-{decoding}): acc={met['accuracy']:.4f} f1={met['f1']:.4f}")

    with open(summary_csv_path, "a", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow([
            os.path.basename(out_met), f"baseline_firstlogit_{decoding}", split, seed, 0, "-", "-", "-", "-", "-", subset_size,
            met["accuracy"],
            met["acc_center"]-met["acc_halfwidth"], met["acc_center"]+met["acc_halfwidth"],
            met["precision"],
            (met["prec_center"]-met["prec_halfwidth"]) if not math.isnan(met["prec_halfwidth"]) else "",
            (met["prec_center"]+met["prec_halfwidth"]) if not math.isnan(met["prec_halfwidth"]) else "",
            met["recall"],
            (met["rec_center"]-met["rec_halfwidth"]) if not math.isnan(met["rec_halfwidth"]) else "",
            (met["rec_center"]+met["rec_halfwidth"]) if not math.isnan(met["rec_halfwidth"]) else "",
            met["f1"], met["f1_ci_low"], met["f1_ci_high"]
        ])
@torch.no_grad()
def run_experiment_beta(model, processor, loader, layers, pools,
                        alphas, ks, cs, yes_ids, no_ids,
                        decoding, num_beams, top_p, temperature, top_k,
                        no_token_penalty, summary_csv_path,
                        split, seed, subset_size,
                        skip_existing: bool):
    for L in layers:
        target = get_decoder_self_attn_module(model, L)
        for pool in pools:
            for A in alphas:
                for K in ks:
                    for C in cs:
                        name = f"POPE_{split}_firstlogit_CARD_Beta_{decoding}_seed{seed}_L{L}_{pool}_A{A}_K{K}_C{C}"
                        out_pred = os.path.join(RESULTS_DIR_POPE, f"pred_{name}.json")
                        out_met  = os.path.join(RESULTS_DIR_POPE, f"metrics_{name}.json")

                        # ==== 已有结果则跳过此组合 ====
                        if skip_existing and _should_skip(out_pred, out_met):
                            print(f"⏭️ Skip(Beta-{decoding}): L{L}|{pool}|A{A}|K{K}|C{C} 已存在。")
                            continue

                        preds, gts = [], []
                        hook = BayesianGatingHookMaskedDynamic(
                            max_alpha=A, sensitivity=K, concentration=C,
                            carrier=mask_carrier, clamp=GATE_CLAMP, rms_match=False, record=False
                        )
                        handle = target.register_forward_hook(hook)
                        try:
                            for batch in tqdm(loader, ncols=100, leave=False,
                                              desc=f"[Beta-{decoding}] L{L}|{pool}|A{A}|K{K}|C{C}"):
                                images, questions, labels = batch["images"], batch["questions"], batch["labels"]

                                v_batch = compute_card_vector_batch_blip(
                                    model, processor, images, questions, layer_idx=L, pooling=pool
                                )
                                hook.set_vector(v_batch)

                                batch_preds = predict_batch_firstlogit(
                                    model, processor, images, questions, yes_ids, no_ids,
                                    decoding=decoding, num_beams=num_beams, top_p=top_p, temperature=temperature, top_k=top_k,
                                    no_token_penalty=no_token_penalty, set_mask=True
                                )
                                preds.extend(batch_preds); gts.extend(labels)
                        finally:
                            handle.remove()

                        met = metrics_with_ci(preds, gts, ci_alpha=0.05, boot_B=2000)
                        os.makedirs(RESULTS_DIR_POPE, exist_ok=True)
                        with open(out_pred, "w") as f: json.dump(preds, f, ensure_ascii=False)
                        with open(out_met,  "w") as f: json.dump(met,   f, ensure_ascii=False, indent=2)
                        print(f"✅ Saved(Beta-{decoding}): acc={met['accuracy']:.4f} f1={met['f1']:.4f}")

                        with open(summary_csv_path, "a", newline="") as fcsv:
                            w = csv.writer(fcsv)
                            w.writerow([
                                os.path.basename(out_met), f"beta_firstlogit_{decoding}", split, seed, 0, L, pool, A, K, C, subset_size,
                                met["accuracy"],
                                met["acc_center"]-met["acc_halfwidth"], met["acc_center"]+met["acc_halfwidth"],
                                met["precision"],
                                (met["prec_center"]-met["prec_halfwidth"]) if not math.isnan(met["prec_halfwidth"]) else "",
                                (met["prec_center"]+met["prec_halfwidth"]) if not math.isnan(met["prec_halfwidth"]) else "",
                                met["recall"],
                                (met["rec_center"]-met["rec_halfwidth"]) if not math.isnan(met["rec_halfwidth"]) else "",
                                (met["rec_center"]+met["rec_halfwidth"]) if not math.isnan(met["rec_halfwidth"]) else "",
                                met["f1"], met["f1_ci_low"], met["f1_ci_high"]
                            ])
@torch.no_grad()
def run_experiment_add(model, processor, loader, layers, pools,
                       add_alphas, yes_ids, no_ids,
                       decoding, num_beams, top_p, temperature, top_k,
                       no_token_penalty, summary_csv_path,
                       split, seed, subset_size,
                       skip_existing: bool):
    for L in layers:
        target = get_decoder_self_attn_module(model, L)
        for pool in pools:
            for A in add_alphas:
                name = f"POPE_{split}_firstlogit_CARD_Add_{decoding}_seed{seed}_L{L}_{pool}_A{A}"
                out_pred = os.path.join(RESULTS_DIR_POPE, f"pred_{name}.json")
                out_met  = os.path.join(RESULTS_DIR_POPE, f"metrics_{name}.json")

                # ==== 已有结果则跳过此组合 ====
                if skip_existing and _should_skip(out_pred, out_met):
                    print(f"⏭️ Skip(Add-{decoding}): L{L}|{pool}|A{A} 已存在。")
                    continue

                preds, gts = [], []
                add_hook = SimpleAddPreHook(alpha=A)
                add_hook.register(target)
                try:
                    for batch in tqdm(loader, ncols=100, leave=False, desc=f"[Add-{decoding}] L{L}|{pool}|A{A}"):
                        images, questions, labels = batch["images"], batch["questions"], batch["labels"]

                        v_batch = compute_card_vector_batch_blip(
                            model, processor, images, questions, layer_idx=L, pooling=pool
                        )
                        add_hook.set_vector(v_batch)

                        batch_preds = predict_batch_firstlogit(
                            model, processor, images, questions, yes_ids, no_ids,
                            decoding=decoding, num_beams=num_beams, top_p=top_p, temperature=temperature, top_k=top_k,
                            no_token_penalty=no_token_penalty, set_mask=True
                        )
                        preds.extend(batch_preds); gts.extend(labels)
                finally:
                    add_hook.remove()

                met = metrics_with_ci(preds, gts, ci_alpha=0.05, boot_B=2000)
                os.makedirs(RESULTS_DIR_POPE, exist_ok=True)
                with open(out_pred, "w") as f: json.dump(preds, f, ensure_ascii=False)
                with open(out_met,  "w") as f: json.dump(met,   f, ensure_ascii=False, indent=2)
                print(f"✅ Saved(Add-{decoding}):  acc={met['accuracy']:.4f} f1={met['f1']:.4f}")

                with open(summary_csv_path, "a", newline="") as fcsv:
                    w = csv.writer(fcsv)
                    w.writerow([
                        os.path.basename(out_met), f"add_firstlogit_{decoding}", split, seed, 0, L, pool, A, "-", "-", subset_size,
                        met["accuracy"],
                        met["acc_center"]-met["acc_halfwidth"], met["acc_center"]+met["acc_halfwidth"],
                        met["precision"],
                        (met["prec_center"]-met["prec_halfwidth"]) if not math.isnan(met["prec_halfwidth"]) else "",
                        (met["prec_center"]+met["prec_halfwidth"]) if not math.isnan(met["prec_halfwidth"]) else "",
                        met["recall"],
                        (met["rec_center"]-met["rec_halfwidth"]) if not math.isnan(met["rec_halfwidth"]) else "",
                        (met["rec_center"]+met["rec_halfwidth"]) if not math.isnan(met["rec_halfwidth"]) else "",
                        met["f1"], met["f1_ci_low"], met["f1_ci_high"]
                    ])
# ====================== 主程序 ======================
def main():
    parser = argparse.ArgumentParser("POPE — baseline / CARD-Beta / Add（first-step logits, beam|nucleus）")
    parser.add_argument("--split", type=str, default=None, choices=POPE_SPLITS, help="POPE split（默认跑全量 split 列表）")
    parser.add_argument("--limit", type=int, default=200, help="每个 split 的样本上限（-1 全量）")
    parser.add_argument("--skip_existing", action="store_true", help="若目标 metrics 文件已存在则跳过该实验")
    # 解码策略（仅支持 beam / nucleus）
    parser.add_argument("--decoding", type=str, default="beam", choices=["beam", "nucleus"], help="解码策略")
    parser.add_argument("--num_beams", type=int, default=4, help="beam search 的 beam 数（>=2）")
    parser.add_argument("--top_p", type=float, default=0.9, help="nucleus 的 top_p")
    parser.add_argument("--temperature", type=float, default=1.0, help="nucleus 的 temperature")
    parser.add_argument("--top_k", type=int, default=0, help="nucleus 的 top_k（0=不启用截断）")

    # 运行开关
    parser.add_argument("--baseline", action="store_true", help="运行 baseline（不注入，不门控）")
    parser.add_argument("--no_beta", action="store_true", help="不跑 Beta 网格")
    parser.add_argument("--abl_add", action="store_true", help="运行简单加法消融")
    parser.add_argument("--add_alphas", type=str, default="4.5", help="简单加法强度（逗号分隔）")

    # 其他
    parser.add_argument("--no_token_penalty", type=float, default=0.0, help="对 'no' 组加惩罚（logit 级别）")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR_POPE, exist_ok=True)
    summary_csv = os.path.join(RESULTS_DIR_POPE, "pope_sampling_summary.csv")
    if not os.path.exists(summary_csv):
        with open(summary_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "file","type","split","seed","resample_id","layer","pool","A","K","C","subset",
                "accuracy","acc_lo","acc_hi",
                "precision","prec_lo","prec_hi",
                "recall","rec_lo","rec_hi",
                "f1","f1_lo","f1_hi"
            ])

    # ====== 模型 ======
    dtype = _torch_dtype_from_str(DTYPE)
    print(f"⏳ Loading model: {MODEL_ID}")
    model = InstructBlipForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, cache_dir=CACHE_DIR, device_map={'': DEVICE}
    )
    processor = InstructBlipProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    tok = processor.tokenizer
    if getattr(tok, "pad_token_id", None) in (None, -1):
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    model.eval(); print("✅ Model ready.")

    # yes/no 首 token 组
    yes_ids, no_ids = collect_yes_no_first_token_groups(processor.tokenizer)

    splits = [args.split] if args.split else POPE_SPLITS
    for seed in SEEDS:
        set_global_seed(seed)
        for split in splits:
            print(f"\n================ POPE[{split}] SEED {seed} | decoding={args.decoding} ================")
            ds = POPEDataset(split=split, limit=(args.limit if (args.limit is None or args.limit > 0) else -1), seed=seed)
            subset_size = len(ds)
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)
            print(f"📦 subset size = {subset_size}")

            # Baseline
            if args.baseline:
                run_experiment_baseline(
                    model, processor, loader,
                    yes_ids, no_ids,
                    decoding=args.decoding, num_beams=args.num_beams, top_p=args.top_p, temperature=args.temperature, top_k=args.top_k,
                    no_token_penalty=args.no_token_penalty,
                    summary_csv_path=summary_csv, split=split, seed=seed, subset_size=subset_size,skip_existing=args.skip_existing
                )

            # Beta
            if not args.no_beta:
                run_experiment_beta(
                    model, processor, loader,
                    layers=INJECTION_LAYERS, pools=EGR_POOLINGS,
                    alphas=BETA_ALPHA_MAX, ks=BETA_K, cs=BETA_C,
                    yes_ids=yes_ids, no_ids=no_ids,
                    decoding=args.decoding, num_beams=args.num_beams, top_p=args.top_p, temperature=args.temperature, top_k=args.top_k,
                    no_token_penalty=args.no_token_penalty,
                    summary_csv_path=summary_csv, split=split, seed=seed, subset_size=subset_size,skip_existing=args.skip_existing
                )

            # Add
            if args.abl_add:
                add_as = [float(x) for x in (args.add_alphas.split(",") if args.add_alphas else []) if x.strip()]
                run_experiment_add(
                    model, processor, loader,
                    layers=INJECTION_LAYERS, pools=EGR_POOLINGS,
                    add_alphas=add_as,
                    yes_ids=yes_ids, no_ids=no_ids,
                    decoding=args.decoding, num_beams=args.num_beams, top_p=args.top_p, temperature=args.temperature, top_k=args.top_k,
                    no_token_penalty=args.no_token_penalty,
                    summary_csv_path=summary_csv, split=split, seed=seed, subset_size=subset_size,skip_existing=args.skip_existing
                )

    print("\n✅ POPE baseline / CARD-Beta / Add（beam|nucleus）实验完成。")
    print(f"📄 Summary CSV appended: {summary_csv}")

if __name__ == "__main__":
    main()