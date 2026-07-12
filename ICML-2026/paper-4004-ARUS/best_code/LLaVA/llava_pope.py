# run_pope.py — POPE 抽样 + 误差条 + Beta门控 + 简单加法（LLaVA-1.5-7B）
import os, json, argparse, math, random, csv
from typing import List, Dict, Tuple

from transformers import AutoTokenizer, AutoImageProcessor
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

from config import *
from methods_decoding import (
    set_global_seed,
    mask_carrier,
    build_answer_mask_from_prompts,
    compute_card_vector_batch,
    BayesianGatingHookMaskedDynamic,
    llava_singleturn_prompt,
    get_llava_self_attn,
)

# ====================== yes/no 归一化 ======================
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

# ====================== Wilson & Bootstrap CI ======================
def _conf_wilson(p: float, n: int, z: float=1.96) -> Tuple[float,float]:
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
    z = 1.959963984540054
    acc_c, acc_h = _conf_wilson(acc, n, z)
    prec_c, prec_h = _conf_wilson(prec, tp+fp, z) if (tp+fp)>0 else (float("nan"), float("nan"))
    rec_c,  rec_h  = _conf_wilson(rec,  tp+fn, z) if (tp+fn)>0 else (float("nan"), float("nan"))
    f1_lo, f1_hi = _bootstrap_f1_ci(y_true, y_pred, B=boot_B, seed=2025, alpha=ci_alpha)
    return {
        "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": acc, "acc_center": acc_c, "acc_halfwidth": acc_h,
        "precision": prec, "prec_center": prec_c, "prec_halfwidth": prec_h, "prec_n": tp+fp,
        "recall": rec, "rec_center": rec_c, "rec_halfwidth": rec_h, "rec_n": tp+fn,
        "f1": f1, "f1_ci_low": f1_lo, "f1_ci_high": f1_hi
    }

# ====================== POPE 数据 ======================
class POPEDataset(Dataset):
    """支持 jsonl/json 多字段名；limit<=0 用全量；正负均衡抽样。"""
    def __init__(self, split_name: str, limit_per_split: int = 200, limit_seed: int = 42):
        from config import POPE_DIR, IMAGE_DIR
        self.rows: List[Dict] = []

        cand = []
        for base in (f"coco_pope_{split_name}", f"pope_coco_{split_name}", f"pope_{split_name}", f"{split_name}"):
            cand.append(os.path.join(POPE_DIR, base + ".jsonl"))
            cand.append(os.path.join(POPE_DIR, base + ".json"))

        pope_file = None
        for p in cand:
            if os.path.exists(p):
                pope_file = p; break
        if pope_file is None:
            raise FileNotFoundError(f"POPE json/jsonl for split '{split_name}' not found in {POPE_DIR}")

        def _parse_label(v):
            if v is None: return None
            s = str(v).strip().lower()
            if s in {"yes","y","true","1"}: return "yes"
            if s in {"no","n","false","0"}:  return "no"
            return None

        records = []
        if pope_file.endswith(".jsonl"):
            with open(pope_file, "r", encoding="utf-8-sig") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("//"): continue
                    rec = json.loads(line)
                    records.append(rec)
        else:
            with open(pope_file, "r", encoding="utf-8-sig") as f:
                txt = f.read().strip()
            try:
                obj = json.loads(txt)
            except json.JSONDecodeError:
                for line_no, line in enumerate(txt.splitlines(), 1):
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("//"): continue
                    rec = json.loads(line)
                    records.append(rec)
            else:
                if isinstance(obj, list):
                    records = obj
                elif isinstance(obj, dict):
                    for k in ("data","samples","items","rows","entries","questions","annotations","results", split_name):
                        v = obj.get(k)
                        if isinstance(v, list): records = v; break
                    if not records:
                        best = None; best_len = -1
                        for v in obj.values():
                            if isinstance(v, list) and len(v)>best_len:
                                best, best_len = v, len(v)
                        if best is not None: records = best

        missing = 0
        for i, d in enumerate(records):
            qid = d.get("question_id") or d.get("qid") or (i + 1)
            q = d.get("text") or d.get("question") or d.get("prompt")
            if not q:
                obj = d.get("object") or d.get("obj") or d.get("category") or d.get("cat") or d.get("name")
                if obj: q = f"Is there a {obj} in the image?"
            if not q: continue
            lab = _parse_label(d.get("label") or d.get("answer") or d.get("gt") or d.get("gold"))

            img_name = d.get("image") or d.get("image_file") or d.get("file_name") \
                       or d.get("filename") or d.get("image_path") or d.get("path")
            if not img_name: continue
            img_path = img_name if os.path.isabs(img_name) else os.path.join(IMAGE_DIR, img_name)
            if not os.path.exists(img_path):
                missing += 1; continue

            self.rows.append({
                "qid": int(qid),
                "question": q,
                "label": lab,
                "image_path": img_path,
            })

        print(f"✅ POPE[{split_name}] total usable items: {len(self.rows)} (missing skipped: {missing})")

        # 子采样（limit<=0 => 全量）
        if (limit_per_split is not None) and (limit_per_split > 0) and (len(self.rows) > limit_per_split):
            random.seed(limit_seed)
            yes = [r for r in self.rows if (r.get("label") or "").lower() == "yes"]
            no  = [r for r in self.rows if (r.get("label") or "").lower() == "no"]
            if len(yes) + len(no) >= limit_per_split // 2:
                random.shuffle(yes); random.shuffle(no)
                n_yes = min(len(yes), limit_per_split // 2)
                n_no  = min(len(no),  limit_per_split - n_yes)
                sub = yes[:n_yes] + no[:n_no]
                random.shuffle(sub)
                self.rows = sub
            else:
                random.shuffle(self.rows)
                self.rows = self.rows[:limit_per_split]
            print(f"📦 Subsampled POPE[{split_name}] to {len(self.rows)} items (balanced where possible).")

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        img = Image.open(r["image_path"]).convert("RGB")
        return {"question_id": r["qid"], "question": r["question"], "label": r["label"], "image": img}

def collate_fn(batch):
    images, questions, answers, qids = [], [], [], []
    for b in batch:
        if b is None: continue
        images.append(b["image"])
        questions.append(b["question"])
        answers.append(b.get("label") or "")
        qids.append(b["question_id"])
    return images, questions, answers, qids

# ====================== prompt & 禁止串（LLaVA 模板） ======================
def build_pope_prompt(q: str) -> str:
    return llava_singleturn_prompt(q)

def build_bad_words_ids(tokenizer):
    stops = [
        "Question:", "Image:", "Instruction:",
        "Neutral answer:", "Reasoned answer:",
        "Assistant:", "User:",
        "ASSISTANT:", "USER:"
    ]
    ids = []
    for s in stops:
        toks = tokenizer(s, add_special_tokens=False).input_ids
        if toks: ids.append(toks)
    return ids

# ====================== 统一解码配置 ======================
def build_generate_kwargs(args):
    """
    根据命令行开关构造 HuggingFace generate() 的解码参数。
    三种模式：
      - greedy: do_sample=False, num_beams=1
      - nucleus: do_sample=True, top_p/temperature/(可选top_k)
      - beam: do_sample=False, num_beams>1, 可带 length_penalty/early_stopping
    """
    mode = getattr(args, "decoding", "greedy")
    kw = dict(
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.0
    )
    if mode == "greedy":
        pass
    elif mode == "nucleus":
        kw.update(
            do_sample=True,
            num_beams=1,
            top_p=getattr(args, "top_p", 0.9),
            temperature=getattr(args, "temperature", 0.7),
        )
        top_k = getattr(args, "top_k", 0)
        if top_k and top_k > 0:
            kw["top_k"] = int(top_k)
    elif mode == "beam":
        kw.update(
            do_sample=False,
            num_beams=int(getattr(args, "num_beams", 4)),
            length_penalty=float(getattr(args, "length_penalty", 1.0)),
        )
        if getattr(args, "early_stopping", False):
            kw["early_stopping"] = True
    else:
        raise ValueError(f"Unknown decoding mode: {mode}")
    return kw

def decoding_tag(args) -> str:
    """在输出文件名里追加一个简短标签，避免不同解码策略混淆/覆盖。"""
    mode = getattr(args, "decoding", "greedy")
    if mode == "greedy":
        return "_decG"
    if mode == "nucleus":
        tp = str(getattr(args, "top_p", 0.9)).rstrip('0').rstrip('.')
        tt = str(getattr(args, "temperature", 0.7)).rstrip('0').rstrip('.')
        tk = getattr(args, "top_k", 0)
        return f"_decP{tp}T{tt}" + (f"K{tk}" if tk and tk>0 else "")
    if mode == "beam":
        nb = int(getattr(args, "num_beams", 4))
        lp = str(getattr(args, "length_penalty", 1.0)).rstrip('0').rstrip('.')
        return f"_decB{nb}LP{lp}"
    return "_decG"

# ====================== Beta 门控 ======================
@torch.no_grad()
def run_once_bayes(model, processor, loader, layer, alpha, k, c, pooling="attn", local=False, gen_args=None):
    results_pred, results_gt = [], []
    bad_words_ids = build_bad_words_ids(processor.tokenizer)
    gen_kwargs = build_generate_kwargs(gen_args) if gen_args is not None else {}

    target_layer = get_llava_self_attn(model, layer)
    hook = BayesianGatingHookMaskedDynamic(
        max_alpha=alpha, sensitivity=k, concentration=c, carrier=mask_carrier,
        clamp=GATE_CLAMP, rms_match=False, record=False
    )
    handle = target_layer.register_forward_hook(hook)

    try:
        for batch in tqdm(loader, leave=False, ncols=100, desc="  [card_bayes]"):
            if batch[0] is None: continue
            images, questions, answers, _ = batch

            prompts = [build_pope_prompt(q) for q in questions]
            inputs = processor(text=prompts, images=[[im] for im in images],
                               return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # 计算 CARD（禁用 hook）
            hook.disable()
            v_batch = compute_card_vector_batch(
                model, processor, images=images, questions=questions,
                layer_idx=layer, pooling=pooling, local=local
            )

            # Answer 段掩码
            m = build_answer_mask_from_prompts(processor.tokenizer, prompts, inputs["input_ids"])
            m = m * inputs["attention_mask"].unsqueeze(-1).to(m.dtype)
            mask_carrier.set(m.to(model.device))

            # 启用 hook
            hook.set_vector(v_batch)

            gen_ids = model.generate(
                **inputs,
                max_new_tokens=getattr(gen_args, "max_new_tokens_pope", MAX_NEW_TOKENS_POPE),
                bad_words_ids=bad_words_ids,
                **gen_kwargs
            )
            outs = processor.batch_decode(gen_ids[:, inputs['input_ids'].size(1):], skip_special_tokens=True)
            results_pred.extend([norm_yesno(o) for o in outs])
            results_gt.extend([norm_yesno(g) for g in answers])

            mask_carrier.clear()
    finally:
        handle.remove()

    return results_pred, results_gt

# ====================== 简单加法消融：self_attn 前置 PreHook ======================
class SimpleAddPreHook:
    """在 self_attn 的 forward 前：hs <- hs + alpha * expand(v_batch) * mask（只改 hidden_states）"""
    def __init__(self, alpha: float, carrier):
        self.alpha = float(alpha)
        self.carrier = carrier
        self.v_batch = None
        self.h = None

    def set_vector(self, v_batch: torch.Tensor):
        self.v_batch = v_batch

    def register(self, target_layer):
        self.h = target_layer.register_forward_pre_hook(self, with_kwargs=True)

    def remove(self):
        if self.h is not None:
            self.h.remove(); self.h = None
    '''
    def __call__(self, module, args, kwargs):
        hs = kwargs.get("hidden_states", None)
        if hs is None and len(args) > 0:
            hs = args[0]
        if hs is None or self.v_batch is None:
            return (args, kwargs)

        B, T, H = hs.size()
        v = self.v_batch.to(device=hs.device, dtype=hs.dtype)
        vT = v.unsqueeze(1).expand(B, T, H)

        from methods import _align_mask_to_seq
        m = _align_mask_to_seq(self.carrier.mask, hs)
        if m is None:
            mask = hs.new_ones(B, T, 1)
        else:
            mask = m.to(device=hs.device, dtype=hs.dtype)

        hs_new = hs + self.alpha * vT * mask

        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = hs_new
            return (args, kwargs)
        else:
            new_args = list(args)
            if len(new_args) > 0:
                new_args[0] = hs_new
            return (tuple(new_args), kwargs)
    '''
    def __call__(self, module, args, kwargs):
        hs = kwargs.get("hidden_states", None)
        if hs is None and len(args) > 0:
            hs = args[0]
        if hs is None or self.v_batch is None:
            return (args, kwargs)

        B, T, H = hs.size()
        v = self.v_batch.to(device=hs.device, dtype=hs.dtype)  # [B0,H]

        # === 对齐 batch（beam search 支持）===
        if v.size(0) != B:
            if B % v.size(0) == 0:
                rep = B // v.size(0)
                v = v.repeat_interleave(rep, dim=0)
            else:
                rep = math.ceil(B / v.size(0))
                v = v.repeat_interleave(rep, dim=0)[:B]

        vT = v.unsqueeze(1).expand(B, T, H)

        from methods import _align_mask_to_seq
        m = _align_mask_to_seq(self.carrier.mask, hs)
        if m is not None and m.size(0) != B:
            if B % m.size(0) == 0:
                rep = B // m.size(0)
                m = m.repeat_interleave(rep, dim=0)
            else:
                rep = math.ceil(B / m.size(0))
                m = m.repeat_interleave(rep, dim=0)[:B]

        mask = hs.new_ones(B, T, 1) if m is None else m.to(device=hs.device, dtype=hs.dtype)
        hs_new = hs + self.alpha * vT * mask

        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = hs_new
            return (args, kwargs)
        else:
            new_args = list(args)
            if len(new_args) > 0:
                new_args[0] = hs_new
            return (tuple(new_args), kwargs)
@torch.no_grad()
def run_once_simple_add(model, processor, loader, layer, alpha_add, pooling="attn", local=False, gen_args=None):
    if local:
        print("[WARN] simple-add ablation ignores `local=True` and uses GLOBAL vector.")
    results_pred, results_gt = [], []
    bad_words_ids = build_bad_words_ids(processor.tokenizer)
    gen_kwargs = build_generate_kwargs(gen_args) if gen_args is not None else {}

    target_layer = get_llava_self_attn(model, layer)
    add_hook = SimpleAddPreHook(alpha=alpha_add, carrier=mask_carrier)
    add_hook.register(target_layer)

    try:
        for batch in tqdm(loader, leave=False, ncols=100, desc="  [card_add]"):
            images, questions, answers, _ = batch
            prompts = [build_pope_prompt(q) for q in questions]
            inputs = processor(text=prompts, images=[[im] for im in images],
                               return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # 全局 CARD（与生成 prompt 对齐）
            v_batch = compute_card_vector_batch(
                model, processor, images=images, questions=questions,
                layer_idx=layer, pooling=pooling, local=False
            )

            # Answer 掩码
            m = build_answer_mask_from_prompts(processor.tokenizer, prompts, inputs["input_ids"])
            m = m * inputs["attention_mask"].unsqueeze(-1).to(m.dtype)
            mask_carrier.set(m.to(model.device))

            add_hook.set_vector(v_batch)

            # 生成
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=getattr(gen_args, "max_new_tokens_pope", MAX_NEW_TOKENS_POPE),
                bad_words_ids=bad_words_ids,
                **gen_kwargs
            )
            outs = processor.batch_decode(gen_ids[:, inputs['input_ids'].size(1):], skip_special_tokens=True)
            results_pred.extend([norm_yesno(o) for o in outs])
            results_gt.extend([norm_yesno(a) for a in answers])

            mask_carrier.clear()
    finally:
        add_hook.remove()

    return results_pred, results_gt

def load_llava_processor(model_id: str, cache_dir: str):
    """
    尝试用 AutoProcessor 加载；若因 LlavaProcessor 参数不兼容（image_token）而报 TypeError，
    则回退到手工组合 AutoTokenizer + AutoImageProcessor 的简易封装，接口与 AutoProcessor 一致。
    """
    try:
        proc = AutoProcessor.from_pretrained(
            model_id, cache_dir=cache_dir, use_fast=False, legacy=True
        )
        # 统一 padding & pad_token
        if getattr(proc, "tokenizer", None) is not None:
            if proc.tokenizer.pad_token is None:
                proc.tokenizer.pad_token = proc.tokenizer.eos_token
            proc.tokenizer.padding_side = "left"
        return proc
    except TypeError as e:
        if "image_token" not in str(e):
            raise

        # ---- 回退路径：手工拼装 ----
        tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, use_fast=False, legacy=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        img_proc = AutoImageProcessor.from_pretrained(model_id, cache_dir=cache_dir)

        class _Proc:
            def __init__(self, tok, img):
                self.tokenizer = tok
                self.image_processor = img
            def __call__(self, text=None, images=None, return_tensors="pt", padding=True, **kwargs):
                # text -> tokenizer
                text = text if text is not None else [""]
                t_inputs = self.tokenizer(text, return_tensors=return_tensors, padding=padding)

                # images 可能是 [[PIL], [PIL], ...]，拍平成每样本一张图
                flat_imgs = []
                if images is not None:
                    for it in images:
                        if isinstance(it, (list, tuple)) and len(it) > 0:
                            flat_imgs.append(it[0])
                        else:
                            flat_imgs.append(it)
                i_inputs = self.image_processor(images=flat_imgs, return_tensors=return_tensors) if images is not None else {}

                # 合并
                out = {**t_inputs, **i_inputs}
                return out
            def batch_decode(self, *a, **k):
                return self.tokenizer.batch_decode(*a, **k)

        return _Proc(tok, img_proc)

# ====================== 主程序 ======================
def _parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default=None, choices=POPE_SPLITS, help="Which POPE split to run.")
    parser.add_argument("--experiment", type=str, default=None, help="Run only the exact-named experiment.")
    parser.add_argument("--limit", type=int, default=200, help="Limit #samples per split (<=0 = full).")
    parser.add_argument("--limit_seed", type=int, default=None, help="Subsample seed (None => use outer SEED).")
    parser.add_argument("--resamples", type=int, default=1, help="重复多少个不同子集。")
    parser.add_argument("--force_baseline", action="store_true", help="即使有缓存也重跑 baseline。")
    parser.add_argument("--ci_alpha", type=float, default=0.05, help="CI 显著性；默认 0.05 (95% CI)。")
    parser.add_argument("--boot_B", type=int, default=2000, help="F1 bootstrap 轮数。")
    parser.add_argument("--local", action="store_true", help="token-level CARD（慎用；默认关）。")
    parser.add_argument("--no_beta", action="store_true", help="不跑 Beta 门控网格。")
    parser.add_argument("--abl_add", action="store_true", help="运行无门控的简单加法消融。")
    parser.add_argument("--add_alphas", type=str, default="3.0,5.0,7.0", help="简单加法强度列表，逗号分隔。")

    # ---- 解码策略开关 ----
    parser.add_argument("--decoding", type=str, choices=["greedy","nucleus","beam"], default="greedy",
                        help="选择解码：greedy / nucleus(top-p) / beam")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus Sampling 的 top-p")
    parser.add_argument("--top_k", type=int, default=0, help="（可选）与 top-p 同用；0 表示禁用 top-k")
    parser.add_argument("--temperature", type=float, default=1.0, help="Nucleus Sampling 的温度")
    parser.add_argument("--num_beams", type=int, default=4, help="Beam Search 的 beam 数")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Beam Search 的长度惩罚")
    parser.add_argument("--early_stopping", action="store_true", help="Beam Search 启用 early_stopping")
    parser.add_argument("--max_new_tokens_pope", type=int, default=MAX_NEW_TOKENS_POPE,
                        help="覆盖 POPE 单步最大新 token 数（默认取配置里的 MAX_NEW_TOKENS_POPE）")

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

    # 模型
    dtype = torch.bfloat16 if DTYPE=="bf16" else (torch.float16 if DTYPE=="fp16" else torch.float32)
    print(f"⏳ Loading model: {MODEL_ID}")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, cache_dir=CACHE_DIR
    ).to(DEVICE)

    processor = load_llava_processor(MODEL_ID, CACHE_DIR)
    if getattr(processor, "tokenizer", None) is not None:
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "left"
    processor.tokenizer.padding_side = "left"  # HF docs tip

    model.eval(); print("✅ Model ready.")

    splits = [args.split] if args.split else POPE_SPLITS

    for seed in SEEDS:
        set_global_seed(seed)
        for split in splits:
            for rep in range(args.resamples):
                limit_seed = args.limit_seed if args.limit_seed is not None else (seed + rep*9973)

                print(f"\n================ POPE[{split}] SEED {seed} REP {rep} ================")
                ds = POPEDataset(
                    split,
                    limit_per_split=(args.limit if (args.limit is None or args.limit > 0) else 0),
                    limit_seed=limit_seed
                )
                loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=NUM_WORKERS, collate_fn=collate_fn)
                sub_tag = f"_sub{len(ds)}"
                rep_tag = f"_rep{rep}" if args.resamples > 1 else ""

                # ---------- baseline ----------
                base_name = f"POPE_{split}_ICL_seed{seed}{sub_tag}{rep_tag}"
                base_name = base_name + decoding_tag(args)
                base_pred = os.path.join(RESULTS_DIR_POPE, f"pred_{base_name}.json")
                base_met  = os.path.join(RESULTS_DIR_POPE, f"metrics_{base_name}.json")

                need_run_base = args.force_baseline or (not (os.path.exists(base_pred) and os.path.exists(base_met)))
                if need_run_base:
                    preds, gts = [], []
                    bad_words_ids = build_bad_words_ids(processor.tokenizer)
                    for batch in tqdm(loader, leave=False, ncols=100, desc="  [baseline]"):
                        images, questions, answers, _ = batch
                        prompts = [build_pope_prompt(q) for q in questions]
                        inputs = processor(text=prompts, images=[[im] for im in images],
                                           return_tensors="pt", padding=True)
                        inputs = {k: v.to(model.device) for k, v in inputs.items()}
                        gen_ids = model.generate(
                            **inputs,
                            max_new_tokens=getattr(args, "max_new_tokens_pope", MAX_NEW_TOKENS_POPE),
                            bad_words_ids=bad_words_ids,
                            **build_generate_kwargs(args)
                        )
                        outs = processor.batch_decode(gen_ids[:, inputs['input_ids'].size(1):], skip_special_tokens=True)
                        preds.extend([norm_yesno(o) for o in outs]); gts.extend([norm_yesno(a) for a in answers])
                    met = metrics_with_ci(preds, gts, ci_alpha=args.ci_alpha, boot_B=args.boot_B)
                    with open(base_pred, "w") as f: json.dump(preds, f)
                    with open(base_met,  "w") as f: json.dump(met,   f, indent=2)
                    print(f"✅ Baseline: acc={met['accuracy']:.4f} f1={met['f1']:.4f}  (95%CI on file)")
                else:
                    with open(base_met, "r") as f: met = json.load(f)
                    print(f"⏭️  Skip baseline exists: {base_pred}")

                # 写入 CSV
                with open(summary_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        os.path.basename(base_met), "baseline", split, seed, rep, "-", "-", "-", "-", "-", len(ds),
                        met["accuracy"], met["acc_center"]-met["acc_halfwidth"], met["acc_center"]+met["acc_halfwidth"],
                        met["precision"],
                        (met["prec_center"]-met["prec_halfwidth"]) if not math.isnan(met["prec_halfwidth"]) else "",
                        (met["prec_center"]+met["prec_halfwidth"]) if not math.isnan(met["prec_halfwidth"]) else "",
                        met["recall"],
                        (met["rec_center"]-met["rec_halfwidth"]) if not math.isnan(met["rec_halfwidth"]) else "",
                        (met["rec_center"]+met["rec_halfwidth"]) if not math.isnan(met["rec_halfwidth"]) else "",
                        met["f1"], met["f1_ci_low"], met["f1_ci_high"]
                    ])

                # ---------- Beta 网格 ----------
                if not args.no_beta:
                    for L in INJECTION_LAYERS:
                        for pool in EGR_POOLINGS:
                            for amax in BETA_ALPHA_MAX:
                                for kk in BETA_K:
                                    for cc in BETA_C:
                                        name = f"POPE_{split}_CARD_Beta_seed{seed}_L{L}_{pool}_A{amax}_K{kk}_C{cc}{sub_tag}{rep_tag}" + decoding_tag(args)
                                        out_pred = os.path.join(RESULTS_DIR_POPE, f"pred_{name}.json")
                                        out_met  = os.path.join(RESULTS_DIR_POPE, f"metrics_{name}.json")

                                        if args.experiment and name != args.experiment:
                                            continue
                                        if os.path.exists(out_pred) and os.path.exists(out_met):
                                            print(f"⏭️  Skip exists: {out_pred}")
                                            with open(out_met, "r") as f: met_o = json.load(f)
                                            with open(summary_csv, "a", newline="") as fcsv:
                                                w = csv.writer(fcsv)
                                                w.writerow([
                                                    os.path.basename(out_met), "beta", split, seed, rep, L, pool, amax, kk, cc, len(ds),
                                                    met_o["accuracy"], met_o["acc_center"]-met_o["acc_halfwidth"], met_o["acc_center"]+met_o["acc_halfwidth"],
                                                    met_o["precision"],
                                                    (met_o["prec_center"]-met_o["prec_halfwidth"]) if not math.isnan(met_o["prec_halfwidth"]) else "",
                                                    (met_o["prec_center"]+met_o["prec_halfwidth"]) if not math.isnan(met_o["prec_halfwidth"]) else "",
                                                    met_o["recall"],
                                                    (met_o["rec_center"]-met_o["rec_halfwidth"]) if not math.isnan(met_o["rec_halfwidth"]) else "",
                                                    (met_o["rec_center"]+met_o["rec_halfwidth"]) if not math.isnan(met_o["rec_halfwidth"]) else "",
                                                    met_o["f1"], met_o["f1_ci_low"], met_o["f1_ci_high"]
                                                ])
                                            continue

                                        print(f"\n▶️ {name}")
                                        loader_b = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                                              num_workers=NUM_WORKERS, collate_fn=collate_fn)
                                        preds_o, gts_o = run_once_bayes(
                                            model, processor, loader_b, L, amax, kk, cc,
                                            pooling=pool, local=args.local, gen_args=args
                                        )
                                        met_o = metrics_with_ci(preds_o, gts_o, ci_alpha=args.ci_alpha, boot_B=args.boot_B)
                                        with open(out_pred,"w") as f: json.dump(preds_o, f)
                                        with open(out_met,"w")  as f: json.dump(met_o,  f, indent=2)
                                        print(f"✅ Saved(Beta): acc={met_o['accuracy']:.4f} f1={met_o['f1']:.4f}")

                                        with open(summary_csv, "a", newline="") as fcsv:
                                            w = csv.writer(fcsv)
                                            w.writerow([
                                                os.path.basename(out_met), "beta", split, seed, rep, L, pool, amax, kk, cc, len(ds),
                                                met_o["accuracy"], met_o["acc_center"]-met_o["acc_halfwidth"], met_o["acc_center"]+met_o["acc_halfwidth"],
                                                met_o["precision"],
                                                (met_o["prec_center"]-met_o["prec_halfwidth"]) if not math.isnan(met_o["prec_halfwidth"]) else "",
                                                (met_o["prec_center"]+met_o["prec_halfwidth"]) if not math.isnan(met_o["prec_halfwidth"]) else "",
                                                met_o["recall"],
                                                (met_o["rec_center"]-met_o["rec_halfwidth"]) if not math.isnan(met_o["rec_halfwidth"]) else "",
                                                (met_o["rec_center"]+met_o["rec_halfwidth"]) if not math.isnan(met_o["rec_halfwidth"]) else "",
                                                met_o["f1"], met_o["f1_ci_low"], met_o["f1_ci_high"]
                                            ])

                # ---------- 简单加法消融 ----------
                if args.abl_add:
                    add_alphas = _parse_float_list(args.add_alphas)
                    for L in INJECTION_LAYERS:
                        for pool in EGR_POOLINGS:
                            for aA in add_alphas:
                                name = f"POPE_{split}_CARD_Add_seed{seed}_L{L}_{pool}_A{aA}{sub_tag}{rep_tag}" + decoding_tag(args)
                                out_pred = os.path.join(RESULTS_DIR_POPE, f"pred_{name}.json")
                                out_met  = os.path.join(RESULTS_DIR_POPE, f"metrics_{name}.json")

                                if args.experiment and name != args.experiment:
                                    continue
                                if os.path.exists(out_pred) and os.path.exists(out_met):
                                    print(f"⏭️  Skip exists: {out_pred}")
                                    with open(out_met, "r") as f: met_o = json.load(f)
                                    with open(summary_csv, "a", newline="") as fcsv:
                                        w = csv.writer(fcsv)
                                        w.writerow([
                                            os.path.basename(out_met), "add", split, seed, rep, L, pool, aA, "-", "-", len(ds),
                                            met_o["accuracy"], met_o["acc_center"]-met_o["acc_halfwidth"], met_o["acc_center"]+met_o["acc_halfwidth"],
                                            met_o["precision"],
                                            (met_o["prec_center"]-met_o["prec_halfwidth"]) if not math.isnan(met_o["prec_halfwidth"]) else "",
                                            (met_o["prec_center"]+met_o["prec_halfwidth"]) if not math.isnan(met_o["prec_halfwidth"]) else "",
                                            met_o["recall"],
                                            (met_o["rec_center"]-met_o["rec_halfwidth"]) if not math.isnan(met_o["rec_halfwidth"]) else "",
                                            (met_o["rec_center"]+met_o["rec_halfwidth"]) if not math.isnan(met_o["rec_halfwidth"]) else "",
                                            met_o["f1"], met_o["f1_ci_low"], met_o["f1_ci_high"]
                                        ])
                                    continue

                                print(f"\n▶️ {name}")
                                loader_a = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                                      num_workers=NUM_WORKERS, collate_fn=collate_fn)
                                preds_o, gts_o = run_once_simple_add(
                                    model, processor, loader_a, L, aA, pooling=pool, local=args.local, gen_args=args
                                )
                                met_o = metrics_with_ci(preds_o, gts_o, ci_alpha=args.ci_alpha, boot_B=args.boot_B)
                                with open(out_pred,"w") as f: json.dump(preds_o, f)
                                with open(out_met,"w")  as f: json.dump(met_o,  f, indent=2)
                                print(f"✅ Saved(Add):  acc={met_o['accuracy']:.4f} f1={met_o['f1']:.4f}")

                                with open(summary_csv, "a", newline="") as fcsv:
                                    w = csv.writer(fcsv)
                                    w.writerow([
                                        os.path.basename(out_met), "add", split, seed, rep, L, pool, aA, "-", "-", len(ds),
                                        met_o["accuracy"], met_o["acc_center"]-met_o["acc_halfwidth"], met_o["acc_center"]+met_o["acc_halfwidth"],
                                        met_o["precision"],
                                        (met_o["prec_center"]-met_o["prec_halfwidth"]) if not math.isnan(met_o["prec_halfwidth"]) else "",
                                        (met_o["prec_center"]+met_o["prec_halfwidth"]) if not math.isnan(met_o["prec_halfwidth"]) else "",
                                        met_o["recall"],
                                        (met_o["rec_center"]-met_o["rec_halfwidth"]) if not math.isnan(met_o["rec_halfwidth"]) else "",
                                        (met_o["rec_center"]+met_o["rec_halfwidth"]) if not math.isnan(met_o["rec_halfwidth"]) else "",
                                        met_o["f1"], met_o["f1_ci_low"], met_o["f1_ci_high"]
                                    ])

    print("\n✅ POPE sampling + error-bars done.")
    print(f"📄 Summary CSV appended: {summary_csv}")

if __name__ == "__main__":
    main()