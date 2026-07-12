# -*- coding: utf-8 -*-
"""
run_qwen_pope_decodings.py

Qwen2.5-VL + POPE evaluation script for:
  1) baseline
  2) RUDDER-Beta / CARD-Beta
  3) RUDDER-Add / fixed additive steering

This script is adapted from the existing LLaVA/Idefics2/InstructBLIP POPE scripts,
and uses methods_decodings_qwen.py for Qwen-specific CARD extraction and hooks.

Example:
python run_qwen_pope_decodings.py \
  --pope_dir /path/to/pope \
  --image_dir /path/to/coco/val2014 \
  --split random \
  --decoding greedy \
  --limit -1 \
  --layers 1 \
  --beta_alpha_max 8.0 \
  --beta_k 5.0
"""

import os
import json
import csv
import math
import random
import argparse
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, LogitsProcessor, LogitsProcessorList

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None

from methods_decodings_qwen import (
    set_global_seed,
    mask_carrier,
    compute_card_vector_batch,
    BayesianGatingHookMaskedDynamic,
    get_qwenvl_self_attn,
    qwenvl_singleturn_prompt,
)


# ====================== yes/no normalization ======================
def norm_yesno(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.split("</s>")[0]
    s = s.split("<|im_end|>")[0]
    s = s.split("<|endoftext|>")[0]
    for sep in ["\n", ".", ",", "!", "?", ";", ":", "  "]:
        s = s.split(sep)[0]
    s = s.strip().strip('"').strip("'")

    yes_set = {"yes", "y", "yeah", "yep", "true", "1", "correct", "of course", "sure", "affirmative"}
    no_set = {"no", "n", "nope", "false", "0", "incorrect", "negative", "not"}

    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    if s in yes_set:
        return "yes"
    if s in no_set:
        return "no"

    toks = s.split()
    if toks:
        if toks[0] in yes_set:
            return "yes"
        if toks[0] in no_set:
            return "no"

    if "yes" in s and "no" not in s:
        return "yes"
    if "no" in s and "yes" not in s:
        return "no"
    return "no"


# ====================== Metrics + CI ======================
def _conf_wilson(p: float, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))
    return center, half


def _counts(y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
    tp = sum(1 for yp, yt in zip(y_pred, y_true) if yp == 1 and yt == 1)
    tn = sum(1 for yp, yt in zip(y_pred, y_true) if yp == 0 and yt == 0)
    fp = sum(1 for yp, yt in zip(y_pred, y_true) if yp == 1 and yt == 0)
    fn = sum(1 for yp, yt in zip(y_pred, y_true) if yp == 0 and yt == 1)
    return tp, fp, tn, fn


def _f1_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    if prec + rec == 0:
        return 0.0, prec, rec
    return 2 * prec * rec / (prec + rec), prec, rec


def _bootstrap_f1_ci(
    y_true: List[int],
    y_pred: List[int],
    B: int = 2000,
    seed: int = 2025,
    alpha: float = 0.05,
) -> Tuple[float, float]:
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
    lo = float(np.percentile(f1s, 100 * alpha / 2))
    hi = float(np.percentile(f1s, 100 * (1 - alpha / 2)))
    return lo, hi


def metrics_with_ci(preds: List[str], gts: List[str], ci_alpha: float = 0.05, boot_B: int = 2000) -> Dict[str, float]:
    y_pred = [1 if norm_yesno(p) == "yes" else 0 for p in preds]
    y_true = [1 if norm_yesno(g) == "yes" else 0 for g in gts]
    tp, fp, tn, fn = _counts(y_true, y_pred)
    n = tp + fp + tn + fn

    acc = (tp + tn) / max(1, n)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1, _, _ = _f1_from_counts(tp, fp, fn)

    acc_c, acc_h = _conf_wilson(acc, n)
    prec_c, prec_h = _conf_wilson(prec, tp + fp) if (tp + fp) > 0 else (float("nan"), float("nan"))
    rec_c, rec_h = _conf_wilson(rec, tp + fn) if (tp + fn) > 0 else (float("nan"), float("nan"))
    f1_lo, f1_hi = _bootstrap_f1_ci(y_true, y_pred, B=boot_B, seed=2025, alpha=ci_alpha)

    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": acc,
        "acc_center": acc_c,
        "acc_halfwidth": acc_h,
        "precision": prec,
        "prec_center": prec_c,
        "prec_halfwidth": prec_h,
        "prec_n": tp + fp,
        "recall": rec,
        "rec_center": rec_c,
        "rec_halfwidth": rec_h,
        "rec_n": tp + fn,
        "f1": f1,
        "f1_ci_low": f1_lo,
        "f1_ci_high": f1_hi,
    }


# ====================== POPE Dataset ======================
class POPEDataset(Dataset):
    """
    Supports common POPE json/jsonl formats:
      - coco_pope_random.json / .jsonl
      - pope_coco_random.json / .jsonl
      - pope_random.json / .jsonl
      - random.json / .jsonl
    """

    def __init__(self, split_name: str, pope_dir: str, image_dir: str, limit: int = -1, seed: int = 42):
        self.rows: List[Dict] = []
        self.image_dir = image_dir

        candidates = []
        for base in (
            f"coco_pope_{split_name}",
            f"pope_coco_{split_name}",
            f"pope_{split_name}",
            f"{split_name}",
        ):
            candidates.append(os.path.join(pope_dir, base + ".jsonl"))
            candidates.append(os.path.join(pope_dir, base + ".json"))

        pope_file = next((p for p in candidates if os.path.exists(p)), None)
        if pope_file is None:
            raise FileNotFoundError(f"POPE file for split='{split_name}' not found. Candidates: {candidates}")

        records = self._load_records(pope_file, split_name)

        missing = 0
        for i, d in enumerate(records):
            qid = d.get("question_id") or d.get("qid") or (i + 1)
            q = d.get("text") or d.get("question") or d.get("prompt")
            if not q:
                obj = d.get("object") or d.get("obj") or d.get("category") or d.get("cat") or d.get("name")
                if obj:
                    q = f"Is there a {obj} in the image?"
            lab = self._parse_label(d.get("label") or d.get("answer") or d.get("gt") or d.get("gold"))

            img_name = (
                d.get("image")
                or d.get("image_file")
                or d.get("file_name")
                or d.get("filename")
                or d.get("image_path")
                or d.get("path")
            )

            if not (q and lab and img_name):
                continue

            img_path = img_name if os.path.isabs(img_name) else os.path.join(image_dir, img_name)
            if not os.path.exists(img_path):
                # Some POPE files store COCO ids; try common COCO val2014 naming.
                stem = str(img_name)
                if stem.isdigit():
                    alt = os.path.join(image_dir, f"COCO_val2014_{int(stem):012d}.jpg")
                    if os.path.exists(alt):
                        img_path = alt
                    else:
                        missing += 1
                        continue
                else:
                    missing += 1
                    continue

            self.rows.append(
                {
                    "qid": int(qid) if str(qid).isdigit() else i + 1,
                    "question": q,
                    "label": lab,
                    "image_path": img_path,
                }
            )

        print(f"✅ POPE[{split_name}] usable items: {len(self.rows)}; missing skipped: {missing}")

        if limit is not None and limit > 0 and len(self.rows) > limit:
            random.seed(seed)
            yes = [r for r in self.rows if r["label"] == "yes"]
            no = [r for r in self.rows if r["label"] == "no"]
            random.shuffle(yes)
            random.shuffle(no)
            n_yes = min(len(yes), limit // 2)
            n_no = min(len(no), limit - n_yes)
            sub = yes[:n_yes] + no[:n_no]
            if len(sub) < limit:
                rest = [r for r in self.rows if r not in sub]
                random.shuffle(rest)
                sub += rest[: limit - len(sub)]
            random.shuffle(sub)
            self.rows = sub
            print(f"📦 Subsampled POPE[{split_name}] to {len(self.rows)} items.")

    @staticmethod
    def _parse_label(v) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip().lower()
        if s in {"yes", "y", "true", "1"}:
            return "yes"
        if s in {"no", "n", "false", "0"}:
            return "no"
        return None

    @staticmethod
    def _load_records(path: str, split_name: str) -> List[Dict]:
        records = []
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("//"):
                        continue
                    records.append(json.loads(line))
            return records

        with open(path, "r", encoding="utf-8-sig") as f:
            txt = f.read().strip()

        try:
            obj = json.loads(txt)
        except json.JSONDecodeError:
            for line in txt.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("//"):
                    continue
                records.append(json.loads(line))
            return records

        if isinstance(obj, list):
            return obj

        if isinstance(obj, dict):
            for k in ("data", "samples", "items", "rows", "entries", "questions", "annotations", "results", split_name):
                v = obj.get(k)
                if isinstance(v, list):
                    return v

            best = None
            best_len = -1
            for v in obj.values():
                if isinstance(v, list) and len(v) > best_len:
                    best, best_len = v, len(v)
            if best is not None:
                return best

        return records

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        img = Image.open(r["image_path"]).convert("RGB")
        return {
            "question_id": r["qid"],
            "question": r["question"],
            "label": r["label"],
            "image": img,
        }


def collate_fn(batch):
    images, questions, labels, qids = [], [], [], []
    for b in batch:
        images.append(b["image"])
        questions.append(b["question"])
        labels.append(b["label"])
        qids.append(b["question_id"])
    return images, questions, labels, qids


# ====================== Prompt / processor helpers ======================
def build_pope_prompt(q: str) -> str:
    return qwenvl_singleturn_prompt(q)


def prepare_qwen_inputs(processor, images: List[Image.Image], questions: List[str], device):
    prompts = [build_pope_prompt(q) for q in questions]

    # methods_decodings_qwen.py uses images=[[PIL], ...], so keep that as first choice.
    try:
        inputs = processor(text=prompts, images=[[im] for im in images], return_tensors="pt", padding=True)
    except Exception:
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)

    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    return prompts, inputs


def build_generation_mask(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    For POPE prompts there are no answer tokens in the prefill stage.
    We set only the final non-pad token to 1:
      - prevents broad prompt-side steering,
      - enables steering on incremental decode steps through right-aligned mask behavior.
    """
    attn = inputs["attention_mask"]
    B, T = attn.shape
    m = torch.zeros((B, T, 1), dtype=torch.float32, device=attn.device)
    lengths = attn.sum(dim=1).long()
    # Handles both right padding and left padding by using the last non-pad index.
    for b in range(B):
        nonpad = torch.where(attn[b] > 0)[0]
        if len(nonpad) > 0:
            m[b, int(nonpad[-1].item()), 0] = 1.0
        elif lengths[b] > 0:
            m[b, int(lengths[b].item()) - 1, 0] = 1.0
    return m


def decode_new_tokens(processor, gen_ids: torch.Tensor, prompt_len: int) -> List[str]:
    new_ids = gen_ids[:, prompt_len:]
    return processor.batch_decode(new_ids, skip_special_tokens=True)


# ====================== Yes/No first token processor for nucleus ======================
class YesNoFirstTokenProcessor(LogitsProcessor):
    def __init__(self, tokenizer, start_len: int, mode: str = "argmax", bias: float = 5.0):
        self.start_len = int(start_len)
        self.mode = mode
        self.bias = float(bias)

        def first_id(text: str):
            ids = tokenizer(text, add_special_tokens=False).input_ids
            return ids[0] if ids else None

        variants = []
        for base in ["yes", "Yes", "no", "No"]:
            for pref in ["", " ", "\n"]:
                tid = first_id(pref + base)
                if tid is not None:
                    variants.append(tid)

        self.allow_ids = sorted(set(int(x) for x in variants))
        if len(self.allow_ids) < 2:
            raise ValueError("Could not collect yes/no first-token ids.")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        step = input_ids.shape[1] - self.start_len
        if step != 0:
            return scores

        allow = torch.tensor(self.allow_ids, device=scores.device, dtype=torch.long)
        sub = scores.index_select(dim=1, index=allow)

        if self.mode == "argmax":
            best = allow[sub.argmax(dim=1)]
            new_scores = scores.new_full(scores.shape, float("-inf"))
            new_scores.scatter_(1, best.unsqueeze(1), 0.0)
            return new_scores

        if self.mode == "hard":
            mask = torch.full_like(scores, float("-inf"))
            mask.scatter_(1, allow.unsqueeze(0).expand(scores.size(0), -1), 0.0)
            return scores + mask

        scores.scatter_add_(1, allow.unsqueeze(0).expand(scores.size(0), -1), torch.full_like(sub, self.bias))
        return scores


# ====================== Decoding kwargs ======================
def build_gen_kwargs(args) -> Dict:
    mode = args.decoding.lower()
    if mode == "greedy":
        return dict(do_sample=False, num_beams=1)
    if mode == "beam":
        return dict(
            do_sample=False,
            num_beams=max(2, int(args.num_beams)),
            length_penalty=float(args.length_penalty),
            early_stopping=bool(args.early_stopping),
        )
    if mode == "nucleus":
        kw = dict(
            do_sample=True,
            num_beams=1,
            top_p=float(args.top_p),
            temperature=float(args.temperature),
        )
        if args.top_k and args.top_k > 0:
            kw["top_k"] = int(args.top_k)
        return kw
    raise ValueError(f"Unknown decoding mode: {args.decoding}")


def decoding_tag(args) -> str:
    if args.decoding == "greedy":
        return "_decG"
    if args.decoding == "beam":
        lp = str(args.length_penalty).rstrip("0").rstrip(".")
        return f"_decB{args.num_beams}LP{lp}"
    if args.decoding == "nucleus":
        tp = str(args.top_p).rstrip("0").rstrip(".")
        tt = str(args.temperature).rstrip("0").rstrip(".")
        tk = f"K{args.top_k}" if args.top_k and args.top_k > 0 else ""
        return f"_decP{tp}T{tt}{tk}"
    return "_decG"


def generation_extra(processor, inputs, args) -> Dict:
    extra = {}
    if args.decoding == "nucleus" and args.force_yesno_first_token:
        start_len = inputs["input_ids"].size(1)
        extra["logits_processor"] = LogitsProcessorList(
            [YesNoFirstTokenProcessor(processor.tokenizer, start_len, mode=args.yesno_processor_mode)]
        )
    return extra


# ====================== Baseline ======================
@torch.no_grad()
def run_once_baseline(model, processor, loader, args, gen_kwargs: Dict):
    preds, gts = [], []
    for batch in tqdm(loader, leave=False, ncols=100, desc="  [baseline]"):
        images, questions, answers, _ = batch
        prompts, inputs = prepare_qwen_inputs(processor, images, questions, model.device)
        extra = generation_extra(processor, inputs, args)

        gen_ids = model.generate(
            **inputs,
            **gen_kwargs,
            **extra,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        )
        outs = decode_new_tokens(processor, gen_ids, inputs["input_ids"].size(1))
        preds.extend([norm_yesno(o) for o in outs])
        gts.extend([norm_yesno(a) for a in answers])
    return preds, gts


# ====================== RUDDER-Beta ======================
@torch.no_grad()
def run_once_bayes(model, processor, loader, layer: int, alpha: float, k: float, c: float, args, gen_kwargs: Dict):
    preds, gts = [], []
    target_layer = get_qwenvl_self_attn(model, layer)
    hook = BayesianGatingHookMaskedDynamic(
        max_alpha=alpha,
        sensitivity=k,
        concentration=c,
        carrier=mask_carrier,
        clamp=(args.gate_min, args.gate_max),
        rms_match=args.rms_match,
        record=False,
    )
    handle = target_layer.register_forward_hook(hook)

    try:
        for batch in tqdm(loader, leave=False, ncols=100, desc="  [card_beta]"):
            images, questions, answers, _ = batch
            prompts, inputs = prepare_qwen_inputs(processor, images, questions, model.device)

            # Compute CARD with steering hook disabled.
            hook.disable()
            v_batch = compute_card_vector_batch(
                model,
                processor,
                images=images,
                questions=questions,
                layer_idx=layer,
                pooling=args.pooling,
                local=args.local,
            )

            mask_carrier.set(build_generation_mask(inputs).to(model.device))
            hook.set_vector(v_batch)

            extra = generation_extra(processor, inputs, args)
            gen_ids = model.generate(
                **inputs,
                **gen_kwargs,
                **extra,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
            )
            outs = decode_new_tokens(processor, gen_ids, inputs["input_ids"].size(1))
            preds.extend([norm_yesno(o) for o in outs])
            gts.extend([norm_yesno(a) for a in answers])

            mask_carrier.clear()
    finally:
        handle.remove()
        mask_carrier.clear()

    return preds, gts


# ====================== RUDDER-Add ======================
class SimpleAddPreHook:
    def __init__(self, alpha: float, carrier):
        self.alpha = float(alpha)
        self.carrier = carrier
        self.v_batch = None
        self.handle = None

    def set_vector(self, v_batch: torch.Tensor):
        self.v_batch = v_batch

    def register(self, target_layer):
        self.handle = target_layer.register_forward_pre_hook(self, with_kwargs=True)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def __call__(self, module, args, kwargs):
        hs = kwargs.get("hidden_states", None)
        if hs is None and len(args) > 0:
            hs = args[0]
        if hs is None or self.v_batch is None:
            return (args, kwargs)

        B, T, H = hs.size()
        v = self.v_batch.to(device=hs.device, dtype=hs.dtype)
        if v.dim() == 3:
            v = v.mean(dim=1)

        B0 = v.size(0)
        if B0 != B:
            if B % B0 == 0:
                v = v.repeat_interleave(B // B0, dim=0)
            else:
                reps = math.ceil(B / B0)
                v = v.repeat(reps, 1)[:B]

        v = torch.nn.functional.normalize(v, p=2, dim=-1)
        vT = v.unsqueeze(1).expand(B, T, H)

        m = getattr(self.carrier, "mask", None)
        if m is None:
            mask = hs.new_ones(B, T, 1)
        else:
            mask = m.to(device=hs.device, dtype=hs.dtype)
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)

            if mask.size(0) != B:
                Bm = mask.size(0)
                if B % Bm == 0:
                    mask = mask.repeat_interleave(B // Bm, dim=0)
                else:
                    reps = math.ceil(B / Bm)
                    mask = mask.repeat(reps, 1, 1)[:B]

            if mask.size(1) > T:
                mask = mask[:, -T:, :]
            elif mask.size(1) < T:
                # Incremental decoding: if original last prompt position is enabled,
                # keep the generated one-token step enabled.
                last = mask[:, -1:, :]
                if T == 1:
                    mask = torch.where(last > 0, last, torch.ones_like(last))
                else:
                    pad = mask.new_zeros(B, T - mask.size(1), 1)
                    mask = torch.cat([mask, pad], dim=1)

        hs_new = hs + self.alpha * vT * mask
        hs_new = torch.nan_to_num(hs_new, nan=0.0, posinf=1e4, neginf=-1e4)

        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = hs_new
            return (args, kwargs)

        new_args = list(args)
        if len(new_args) > 0:
            new_args[0] = hs_new
        return (tuple(new_args), kwargs)


@torch.no_grad()
def run_once_simple_add(model, processor, loader, layer: int, alpha_add: float, args, gen_kwargs: Dict):
    if args.local:
        print("[WARN] RUDDER-Add ignores --local and uses global CARD.")

    preds, gts = [], []
    target_layer = get_qwenvl_self_attn(model, layer)
    add_hook = SimpleAddPreHook(alpha=alpha_add, carrier=mask_carrier)
    add_hook.register(target_layer)

    try:
        for batch in tqdm(loader, leave=False, ncols=100, desc="  [card_add]"):
            images, questions, answers, _ = batch
            prompts, inputs = prepare_qwen_inputs(processor, images, questions, model.device)

            v_batch = compute_card_vector_batch(
                model,
                processor,
                images=images,
                questions=questions,
                layer_idx=layer,
                pooling=args.pooling,
                local=False,
            )
            add_hook.set_vector(v_batch)
            mask_carrier.set(build_generation_mask(inputs).to(model.device))

            extra = generation_extra(processor, inputs, args)
            gen_ids = model.generate(
                **inputs,
                **gen_kwargs,
                **extra,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
            )
            outs = decode_new_tokens(processor, gen_ids, inputs["input_ids"].size(1))
            preds.extend([norm_yesno(o) for o in outs])
            gts.extend([norm_yesno(a) for a in answers])

            mask_carrier.clear()
    finally:
        add_hook.remove()
        mask_carrier.clear()

    return preds, gts


# ====================== Utils ======================
def parse_int_list(s: str) -> List[int]:
    return [int(x) for x in str(s).split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in str(s).split(",") if x.strip()]


def write_summary_row(summary_csv: str, file_name: str, typ: str, split: str, seed: int, rep: int,
                      layer, pool, A, K, C, subset: int, met: Dict):
    with open(summary_csv, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                file_name,
                typ,
                split,
                seed,
                rep,
                layer,
                pool,
                A,
                K,
                C,
                subset,
                met["accuracy"],
                met["acc_center"] - met["acc_halfwidth"],
                met["acc_center"] + met["acc_halfwidth"],
                met["precision"],
                (met["prec_center"] - met["prec_halfwidth"]) if not math.isnan(met["prec_halfwidth"]) else "",
                (met["prec_center"] + met["prec_halfwidth"]) if not math.isnan(met["prec_halfwidth"]) else "",
                met["recall"],
                (met["rec_center"] - met["rec_halfwidth"]) if not math.isnan(met["rec_halfwidth"]) else "",
                (met["rec_center"] + met["rec_halfwidth"]) if not math.isnan(met["rec_halfwidth"]) else "",
                met["f1"],
                met["f1_ci_low"],
                met["f1_ci_high"],
            ]
        )


def load_model_and_processor(args):
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "auto": "auto",
    }[args.dtype]

    print(f"⏳ Loading Qwen model: {args.model_id}")

    if Qwen2_5_VLForConditionalGeneration is not None:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            cache_dir=args.cache_dir if args.cache_dir else None,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation if args.attn_implementation else None,
        )
    elif AutoModelForVision2Seq is not None:
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            cache_dir=args.cache_dir if args.cache_dir else None,
            device_map=args.device_map,
        )
    else:
        raise ImportError("Neither Qwen2_5_VLForConditionalGeneration nor AutoModelForVision2Seq is available.")

    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=args.cache_dir if args.cache_dir else None)

    if hasattr(processor, "tokenizer"):
        if processor.tokenizer.pad_token_id is None and processor.tokenizer.eos_token is not None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        # Decoder-only generation should use left padding.
        processor.tokenizer.padding_side = "left"

    model.eval()
    print("✅ Model ready.")
    return model, processor


# ====================== Main ======================
def main():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--pope_dir", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="./results_qwen_pope")

    # Runtime
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--attn_implementation", type=str, default=None,
                        help="Optional: flash_attention_2 / sdpa / eager. Leave None if unsure.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--resamples", type=int, default=1)
    parser.add_argument("--limit", type=int, default=-1, help="Samples per split. <=0 means full split.")
    parser.add_argument("--limit_seed", type=int, default=None)

    # Dataset split
    parser.add_argument("--split", type=str, default=None, choices=["random", "popular", "adversarial"])
    parser.add_argument("--splits", type=str, default="random,popular,adversarial")

    # Decoding
    parser.add_argument("--decoding", type=str, default="greedy", choices=["greedy", "beam", "nucleus"])
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--force_yesno_first_token", action="store_true",
                        help="For nucleus decoding, force the first generated token into yes/no candidates.")
    parser.add_argument("--yesno_processor_mode", type=str, default="argmax", choices=["argmax", "hard", "bias"])

    # RUDDER config
    parser.add_argument("--layers", type=str, default="1")
    parser.add_argument("--pooling", type=str, default="attn", choices=["attn", "mean"])
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--gate_min", type=float, default=0.05)
    parser.add_argument("--gate_max", type=float, default=1.0)
    parser.add_argument("--rms_match", action="store_true")

    # Experiment switches
    parser.add_argument("--force_baseline", action="store_true")
    parser.add_argument("--no_beta", action="store_true")
    parser.add_argument("--abl_add", action="store_true")
    parser.add_argument("--beta_alpha_max", type=str, default="8.0")
    parser.add_argument("--beta_k", type=str, default="5.0")
    parser.add_argument("--beta_c", type=str, default="1.0")
    parser.add_argument("--add_alphas", type=str, default="3.0,5.0,7.0")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Run only exact experiment name after baseline.")

    # Metrics
    parser.add_argument("--ci_alpha", type=float, default=0.05)
    parser.add_argument("--boot_B", type=int, default=2000)

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    summary_csv = os.path.join(args.results_dir, "pope_qwen_summary.csv")
    if not os.path.exists(summary_csv):
        with open(summary_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "file", "type", "split", "seed", "resample_id", "layer", "pool", "A", "K", "C", "subset",
                    "accuracy", "acc_lo", "acc_hi",
                    "precision", "prec_lo", "prec_hi",
                    "recall", "rec_lo", "rec_hi",
                    "f1", "f1_lo", "f1_hi",
                ]
            )

    model, processor = load_model_and_processor(args)
    gen_kwargs = build_gen_kwargs(args)
    dec_tag = decoding_tag(args)

    layers = parse_int_list(args.layers)
    seeds = parse_int_list(args.seeds)
    beta_alphas = parse_float_list(args.beta_alpha_max)
    beta_ks = parse_float_list(args.beta_k)
    beta_cs = parse_float_list(args.beta_c)
    add_alphas = parse_float_list(args.add_alphas)

    splits = [args.split] if args.split else [x.strip() for x in args.splits.split(",") if x.strip()]

    print(f"🧪 Decoding: {args.decoding} | gen_kwargs={gen_kwargs}")
    print(f"🧪 Splits: {splits} | layers={layers} | pooling={args.pooling}")

    for seed in seeds:
        set_global_seed(seed)

        for split in splits:
            for rep in range(args.resamples):
                limit_seed = args.limit_seed if args.limit_seed is not None else (seed + rep * 9973)

                print(f"\n================ POPE[{split}] SEED {seed} REP {rep} ================")
                ds = POPEDataset(
                    split_name=split,
                    pope_dir=args.pope_dir,
                    image_dir=args.image_dir,
                    limit=args.limit,
                    seed=limit_seed,
                )
                loader = DataLoader(
                    ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=collate_fn,
                )

                sub_tag = f"_sub{len(ds)}"
                rep_tag = f"_rep{rep}" if args.resamples > 1 else ""

                # ---------- baseline ----------
                base_name = f"POPE_{split}{dec_tag}_Qwen_seed{seed}{sub_tag}{rep_tag}"
                base_pred = os.path.join(args.results_dir, f"pred_{base_name}.json")
                base_met = os.path.join(args.results_dir, f"metrics_{base_name}.json")

                need_base = args.force_baseline or not (os.path.exists(base_pred) and os.path.exists(base_met))
                if need_base:
                    preds, gts = run_once_baseline(model, processor, loader, args, gen_kwargs)
                    met = metrics_with_ci(preds, gts, ci_alpha=args.ci_alpha, boot_B=args.boot_B)
                    with open(base_pred, "w") as f:
                        json.dump(preds, f)
                    with open(base_met, "w") as f:
                        json.dump(met, f, indent=2)
                    print(f"✅ Baseline: acc={met['accuracy']:.4f} f1={met['f1']:.4f}")
                else:
                    with open(base_met, "r") as f:
                        met = json.load(f)
                    print(f"⏭️  Skip baseline exists: {base_met}")

                write_summary_row(
                    summary_csv, os.path.basename(base_met), "baseline", split, seed, rep,
                    "-", "-", "-", "-", "-", len(ds), met,
                )

                # ---------- RUDDER-Beta ----------
                if not args.no_beta:
                    for L in layers:
                        for amax in beta_alphas:
                            for kk in beta_ks:
                                for cc in beta_cs:
                                    name = (
                                        f"POPE_{split}{dec_tag}_Qwen_CARD_Beta_seed{seed}"
                                        f"_L{L}_{args.pooling}_A{amax}_K{kk}_C{cc}{sub_tag}{rep_tag}"
                                    )
                                    if args.experiment and name != args.experiment:
                                        continue

                                    out_pred = os.path.join(args.results_dir, f"pred_{name}.json")
                                    out_met = os.path.join(args.results_dir, f"metrics_{name}.json")

                                    if os.path.exists(out_pred) and os.path.exists(out_met):
                                        with open(out_met, "r") as f:
                                            met_o = json.load(f)
                                        print(f"⏭️  Skip exists: {out_met}")
                                    else:
                                        print(f"\n▶️ {name}")
                                        loader_b = DataLoader(
                                            ds,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers,
                                            collate_fn=collate_fn,
                                        )
                                        preds_o, gts_o = run_once_bayes(
                                            model, processor, loader_b, L, amax, kk, cc, args, gen_kwargs
                                        )
                                        met_o = metrics_with_ci(preds_o, gts_o, ci_alpha=args.ci_alpha, boot_B=args.boot_B)
                                        with open(out_pred, "w") as f:
                                            json.dump(preds_o, f)
                                        with open(out_met, "w") as f:
                                            json.dump(met_o, f, indent=2)
                                        print(f"✅ Saved(Beta): acc={met_o['accuracy']:.4f} f1={met_o['f1']:.4f}")

                                    write_summary_row(
                                        summary_csv, os.path.basename(out_met), "beta", split, seed, rep,
                                        L, args.pooling, amax, kk, cc, len(ds), met_o,
                                    )

                # ---------- RUDDER-Add ----------
                if args.abl_add:
                    for L in layers:
                        for aA in add_alphas:
                            name = (
                                f"POPE_{split}{dec_tag}_Qwen_CARD_Add_seed{seed}"
                                f"_L{L}_{args.pooling}_A{aA}{sub_tag}{rep_tag}"
                            )
                            if args.experiment and name != args.experiment:
                                continue

                            out_pred = os.path.join(args.results_dir, f"pred_{name}.json")
                            out_met = os.path.join(args.results_dir, f"metrics_{name}.json")

                            if os.path.exists(out_pred) and os.path.exists(out_met):
                                with open(out_met, "r") as f:
                                    met_o = json.load(f)
                                print(f"⏭️  Skip exists: {out_met}")
                            else:
                                print(f"\n▶️ {name}")
                                loader_a = DataLoader(
                                    ds,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    collate_fn=collate_fn,
                                )
                                preds_o, gts_o = run_once_simple_add(
                                    model, processor, loader_a, L, aA, args, gen_kwargs
                                )
                                met_o = metrics_with_ci(preds_o, gts_o, ci_alpha=args.ci_alpha, boot_B=args.boot_B)
                                with open(out_pred, "w") as f:
                                    json.dump(preds_o, f)
                                with open(out_met, "w") as f:
                                    json.dump(met_o, f, indent=2)
                                print(f"✅ Saved(Add): acc={met_o['accuracy']:.4f} f1={met_o['f1']:.4f}")

                            write_summary_row(
                                summary_csv, os.path.basename(out_met), "add", split, seed, rep,
                                L, args.pooling, aA, "-", "-", len(ds), met_o,
                            )

    print("\n✅ Qwen POPE decoding done.")
    print(f"📄 Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
