# methods.py  — CARD + Beta（仅此方法）+ 评测辅助
import os, json, random
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from PIL import Image

# ====================== 随机性控制 ======================
def set_global_seed(seed: int):
    import numpy as np, random, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ====================== 掩码载体（Answer段） ======================
class _MaskCarrier:
    """一次 batch 使用的 Answer 段掩码载体。mask 形状: [B, T, 1]，1=开启注入。"""
    def __init__(self): self.mask = None
    def set(self, mask): self.mask = mask
    def clear(self): self.mask = None
mask_carrier = _MaskCarrier()

def _align_mask_to_seq(m: torch.Tensor, h: torch.Tensor) -> Optional[torch.Tensor]:
    '''
    """将 [B,T,1] 的 mask 对齐到当前层 hidden 的 [B,q_len,H]。"""
    if m is None: return None
    if m.size(1) == h.size(1): return m
    return m[:, -1:, :]
    '''
    """
    m: [B, T, 1]  基于输入构造的掩码
    h: [B, q_len, H]  当前层 hidden
    若 q_len==T：直接返回 m
    若 q_len==1：表示增量解码步（新token），此时应视为位于 Answer 段，返回 1
    """
    if m is None:
        return None
    if m.size(1) == h.size(1):
        return m
    # 增量解码阶段：取末位，但若末位==0，则提升为1（新token属于Answer段）
    last = m[:, -1:, :]
    # 若你更保守，也可以直接: last = torch.ones_like(last)
    last = torch.where(last > 0, last, torch.ones_like(last))
    return last
# ====================== Answer 段定位（稳健版） ======================
import re
_PUNCT_SPLIT = re.compile(r"[,\.;:\n\?\!\t]")

def clean_answer(text: str) -> str:
    t = (text or "").strip()
    if not t: return t
    t = t.split("</s>")[0]
    t = _PUNCT_SPLIT.split(t)[0]
    t = t.strip().strip('"').strip("'")
    for pref in ["answer is","the answer is","it is","it's","it was","this is",
                 "the color is","color is","number is"]:
        if t.lower().startswith(pref): t = t[len(pref):].strip()
    if t.lower().startswith("a "):  t = t[2:]
    if t.lower().startswith("an "): t = t[3:]
    return t

def build_answer_mask_from_prompts(tokenizer, prompts: List[str], input_ids: torch.Tensor) -> torch.Tensor:
    """
    基于原始 prompt 文本定位最后一个 'Answer:'，用相同 tokenizer 重新分词得到准确 token 边界。
    返回 mask [B, T, 1]：前缀之后=1（目标回答段），其余=0。
    """
    key = "Answer:"
    prefixes = []
    for p in prompts:
        pos = p.rfind(key)
        prefixes.append(p if pos == -1 else p[:pos + len(key)])
    tok = tokenizer(prefixes, add_special_tokens=False, padding=True, return_tensors="pt")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    lens = (tok.input_ids != pad_id).sum(dim=1)  # [B]
    bsz, seqlen = input_ids.size(0), input_ids.size(1)
    mask = input_ids.new_zeros((bsz, seqlen, 1), dtype=torch.float32)
    for b in range(bsz):
        L = int(lens[b].item())
        if 0 < L <= seqlen:
            mask[b, L:, 0] = 1.0
    return mask

# ====================== Prompt 构造（固定 ICD 池） ======================
def build_prompt_and_images_fixed(shots: List[Dict], target_q: str) -> Tuple[str, List[Image.Image]]:
    """
    固定的 ICD 池：每个 shot: Image + Question + Neutral + Reasoned
    """
    ex_lines, ex_imgs = [], []
    for s in shots:
        ex_lines.append(
            "Image: <image>\n"
            f"Question: {s['question']}\n"
            f"Neutral answer: {s['neutral']}\n"
            f"Reasoned answer: {s['reasoned']}"
        )
        ex_imgs.append(Image.open(s["image_path"]).convert("RGB"))
    prompt = (
        "Instruction: provide an answer to the question. Use the image to answer.\n" +
        "\n\n".join(ex_lines) + "\n\n" +
        "Image: <image>\n" +
        f"Question: {target_q}\nAnswer:"
    )
    return prompt, ex_imgs

def _build_min_prefix(q: str) -> str:
    return ("Instruction: provide an answer to the question. Use the image to answer.\n"
            "Image: <image>\n"
            f"Question: {q}\n"
            "Answer:")

# ====================== CARD 向量（批量） ======================
'''
class _CardCapture:
    """捕获目标层 self_attn 的前后 hidden，用于 delta=after-before。"""
    def __init__(self, target_layer):
        self.target_layer = target_layer
        self.before = None; self.after = None
        self._pre = None; self._post = None
    def __enter__(self):
        def pre(_m, inp):
            self.before = inp[0].detach()
        def post(_m, inp, out):
            self.after = out[0].detach()
        self._pre  = self.target_layer.register_forward_pre_hook(pre)
        self._post = self.target_layer.register_forward_hook(post)
        return self
    def __exit__(self, exc_type, exc, tb):
        if self._pre:  self._pre.remove()
        if self._post: self._post.remove()
'''
class _CardCapture:
    """捕获目标层 self_attn 的前后 hidden，用于 delta=after-before。兼容 kwargs 调用。"""
    def __init__(self, target_layer):
        self.target_layer = target_layer
        self.before = None
        self.after = None
        self._pre = None
        self._post = None

    def __enter__(self):
        # 注意：with_kwargs=True 才能拿到 kwargs 里的 hidden_states
        def pre(_m, args, kwargs):
            hs = None
            # 绝大多数 HF 模型通过 kwargs 传 hidden_states
            if kwargs is not None:
                hs = kwargs.get("hidden_states", None)
            # 兜底：如果有人用位置参数
            if hs is None and len(args) > 0:
                hs = args[0]
            # 再兜底：避免 None 继续向下
            if hs is not None:
                self.before = hs.detach()
            else:
                self.before = None  # 后面会做检查并报更友好的错误

        def post(_m, args, output):
            # output 可能是 Tensor 或 tuple(Tensor, attn_weights, kv)
            out0 = output[0] if isinstance(output, (tuple, list)) else output
            self.after = out0.detach()

        # 关键：带 with_kwargs=True
        self._pre = self.target_layer.register_forward_pre_hook(pre, with_kwargs=True)
        self._post = self.target_layer.register_forward_hook(post, with_kwargs=False)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._pre:  self._pre.remove()
        if self._post: self._post.remove()
@torch.no_grad()
def compute_card_vector_batch(model, processor, images: List[Image.Image], questions: List[str],
                              layer_idx=20, pooling="attn", local=False) -> torch.Tensor:
    """
    返回:
      - local=False: [B, H] 的 v_card
      - local=True:  [B, T, H] 的 token-specific v_card
    """
    texts = [_build_min_prefix(q) for q in questions]
    imgs  = [[img] for img in images]
    inputs = processor(text=texts, images=imgs, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    target_layer = model.model.text_model.layers[layer_idx].self_attn
    with _CardCapture(target_layer) as cap:
        _ = model(**inputs, output_hidden_states=False, return_dict=True)

    before, after = cap.before, cap.after  # [B,T,H]
    if before is None or after is None:
        raise RuntimeError(
            "CARD capture failed: 'before' or 'after' is None. "
            "Likely because the attention layer was called with kwargs and the hook "
            "did not use with_kwargs=True. Please ensure _CardCapture uses with_kwargs=True."
        )
    delta = after - before
    attn_mask = inputs["attention_mask"].unsqueeze(-1).to(delta.dtype)

    if local:
        v_tok = F.normalize(delta, p=2, dim=-1) * attn_mask
        return v_tok

    if pooling == "mean":
        v = (delta * attn_mask).sum(dim=1) / (attn_mask.sum(dim=1) + 1e-6)
    elif pooling == "attn":
        # 无显式 cross-attn 权重时，用 |delta| 的范数作为权重代理
        w = (delta.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6) * attn_mask
        v = (delta * w).sum(dim=1) / (w.sum(dim=1) + 1e-6)
    else:
        v = (delta * attn_mask).sum(dim=1) / (attn_mask.sum(dim=1) + 1e-6)

    return F.normalize(v, p=2, dim=-1).detach()

# ====================== Beta 门控（动态批向量 + 记录） ======================
class BayesianGatingHookMaskedDynamic:
    """
    支持 v_batch: [B,H] 或 token-specific: [B,q_len,H]
    仅在 Answer 段注入；记录 sim/gate 以供可视化。
    """
    def __init__(self, max_alpha=2, sensitivity=5.0, concentration=1.0,
                 carrier=mask_carrier, clamp=(0.0,0.8), rms_match=False, record=True):
        self.enable = True
        self.enabled = True
        self.v_batch = None
        self.max_alpha = max_alpha; self.k = sensitivity; self.c = concentration
        self.carrier = carrier
        self.clamp = clamp
        self.rms_match = rms_match
        self.record = record
        # self._last = {}
        self._last = None
    '''
    def set_vector(self, v_batch: torch.Tensor):
        # ← 修正变量名，并规范化到单位范数
        self.v_batch = F.normalize(v_batch, p=2, dim=-1)
        self.enabled = True
        for attr in ["_dbg_seen", "_dbg_printed"]:
            if hasattr(self, attr): delattr(self, attr)

    
    def disable(self):
        self.enabled = False
        self.v_batch = self.v_batch
    '''
    def set_vector(self, v_batch: torch.Tensor):
        self.v_batch = F.normalize(v_batch, p=2, dim=-1)
        self.enable = True
        self.enabled = True

    def disable(self):
        self.enable = False
        self.enabled = False
    def fetch_last_stats(self):
        return self._last
    '''
    def __call__(self, module, module_in, module_out):
        if (not self.enabled) or (self.v_batch is None):
            return module_out
        h = module_out[0]  # [B,q_len,H]
        if self.v_batch is None:
            return module_out
        v = self.v_batch.to(h.device, dtype=h.dtype)
        if v.dim() == 2:
            v = v.unsqueeze(1).expand_as(h)
        elif v.size(1) == 1:
            v = v.expand_as(h)

        h_n = F.normalize(h, p=2, dim=-1)
        v_n = F.normalize(v, p=2, dim=-1)
        sim = F.cosine_similarity(h_n, v_n, dim=-1, eps=1e-6).unsqueeze(-1)  # [B,q_len,1]
        alpha = F.softplus(self.k * sim + self.c)
        beta  = F.softplus(-self.k * sim + self.c)
        gate  = (alpha / (alpha + beta)).clamp(self.clamp[0], self.clamp[1])

        m = _align_mask_to_seq(self.carrier.mask, h)
        if m is not None: gate = gate * m.to(h.dtype)
        # ===  DEBUG 打印（只打印一次）===
        if not hasattr(self, "_dbg_seen"):
            print("[DEBUG] hook invoked once")
            self._dbg_seen = True

        if not hasattr(self, "_dbg_printed"):
            g_mean = gate.mean().item()
            s_mean = sim.mean().item()
            mask_sum = float(m.sum().item()) if m is not None else -1.0
            print(f"[DEBUG] gate_mean={g_mean:.3f}  sim_mean={s_mean:.3f}  mask_sum={mask_sum:.1f}")
            self._dbg_printed = True
        # ======================================
        if self.record:
            self._last = {
                "sim":  sim.detach().to(torch.float32).cpu(),
                "gate": gate.detach().to(torch.float32).cpu()
            }
        return (h + self.max_alpha * gate * v,) + module_out[1:]
    '''
    def __call__(self, module, module_in, module_out):
    # 关/无向量则不改输出
        if (not getattr(self, "enable", True)) or (self.v_batch is None):
            return module_out

        # 1) 取 hidden_states（module_out 可能是 tensor 或 tuple）
        h = module_out[0] if isinstance(module_out, tuple) else module_out   # [B_eff, T, H]
        if h is None:
            return module_out
        B_eff, T, H = h.shape

        # 2) 取 CARD 向量并与当前 batch/time 对齐（适配 beam 扩批）
        v = self.v_batch.to(h.device, dtype=h.dtype)  # [B0, H] 或 [B0, T0, H]

        def _repeat_to_batch(x, target_B):
            B0 = x.size(0)
            if B0 == target_B:
                return x
            if target_B % B0 == 0:
                r = target_B // B0
                return x.repeat_interleave(r, dim=0)
            # 非整倍数兜底：tile 后截断
            reps = (target_B + B0 - 1) // B0
            x = x.repeat((reps,) + (1,) * (x.dim() - 1))
            return x[:target_B]

        def _align_time(x, T_target):
            # 右对齐到当前序列长度 T：过长截尾，过短尾部补零
            if x.dim() < 3:
                return x
            T0 = x.size(1)
            if T0 == T_target:
                return x
            if T0 > T_target:
                return x[:, -T_target:, :]
            pad = x.new_zeros(x.size(0), T_target - T0, x.size(2))
            return torch.cat([x, pad], dim=1)

        if v.dim() == 2:
            # [B0, H] -> [B_eff, T, H]
            v = _repeat_to_batch(v, B_eff)
            v = v.unsqueeze(1).expand(B_eff, T, H)
        elif v.dim() == 3:
            # [B0, T0, H] -> [B_eff, T, H]
            v = _repeat_to_batch(v, B_eff)
            v = _align_time(v, T)
        else:
            # 未知形状，放弃本轮
            return module_out

        # 3) 归一化并计算相似度/门控
        h_n = F.normalize(h, p=2, dim=-1)
        v_n = F.normalize(v, p=2, dim=-1)
        sim = F.cosine_similarity(h_n, v_n, dim=-1, eps=1e-6).unsqueeze(-1)  # [B_eff, T, 1]
        alpha = F.softplus(self.k * sim + self.c)
        beta  = F.softplus(-self.k * sim + self.c)
        gate  = (alpha / (alpha + beta)).clamp(self.clamp[0], self.clamp[1])  # [B_eff, T, 1]

        # 4) 对齐并应用 Answer 段 mask（支持 carrier.mask 或 carrier._m）
        m_raw = getattr(self.carrier, "mask", None)
        if m_raw is None:
            m_raw = getattr(self.carrier, "_m", None)
        if m_raw is not None:
            m = m_raw.to(h.device)
            if m.dim() == 2:  # [B, T] -> [B, T, 1]
                m = m.unsqueeze(-1)
            # batch 对齐
            m = _repeat_to_batch(m, B_eff)
            # 时间对齐
            if m.size(1) != T:
                if m.size(1) > T:
                    m = m[:, -T:, :]
                else:
                    pad = m.new_zeros(m.size(0), T - m.size(1), 1)
                    m = torch.cat([m, pad], dim=1)
            gate = gate * m.to(h.dtype)

        # 5) 计算增量并返回同结构输出（支持 record / rms_match）
        if getattr(self, "rms_match", False):
            den = (h.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6)
            update = self.max_alpha * gate * (v_n / den)
        else:
            update = self.max_alpha * gate * v_n

        out0 = h + update

        if getattr(self, "record", False):
            self._last = {
                "sim":  sim.detach().to(torch.float32).cpu(),
                "gate": gate.detach().to(torch.float32).cpu()
            }

        if isinstance(module_out, tuple):
            return (out0,) + module_out[1:]
        else:
            return out0
# ====================== 可视化记录器 ======================
class GatingLogger:
    def __init__(self, out_dir: str):
        import numpy as np, pandas as pd
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.rows = []         # per-q stats
        self.hist_gate = []    # all tokens
        self.hist_sim  = []
        self.np = __import__("numpy")
        self.pd = __import__("pandas")

    def log_batch(self, name: str, qids: List[int], stats: Dict, mask_ans: torch.Tensor):
        # sim = stats["sim"].numpy()    # [B,q_len,1]
        # gate= stats["gate"].numpy()
        sim  = stats["sim"].to(torch.float32).numpy()
        gate = stats["gate"].to(torch.float32).numpy()
        m   = mask_ans.cpu().numpy()
        for i, qid in enumerate(qids):
            sel = m[i,:,0] > 0.5
            sim_i = sim[i,sel,0]; gate_i = gate[i,sel,0]
            if sim_i.size==0: continue
            self.rows.append({
                "exp": name, "qid": int(qid),
                "sim_mean": float(sim_i.mean()),
                "sim_p95":  float(self.np.percentile(sim_i,95)),
                "gate_mean":float(gate_i.mean()),
                "gate_p95": float(self.np.percentile(gate_i,95)),
                "n_tokens": int(sim_i.size)
            })
            self.hist_sim.extend(sim_i.tolist()); self.hist_gate.extend(gate_i.tolist())
class GatingLogger:
    def __init__(self, out_dir: str):
        import numpy as np, pandas as pd
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.rows = []
        self.hist_gate = []
        self.hist_sim  = []
        self.np = __import__("numpy")
        self.pd = __import__("pandas")

    def log_batch(self, name, qids, stats, mask_ans):
        # 防御：有些 batch 可能没有统计
        if stats is None or ("sim" not in stats) or ("gate" not in stats) \
        or (stats["sim"] is None) or (stats["gate"] is None):
            return
        import numpy as np
        # 强制到 float32，避免 bf16 -> numpy 报错
        sim  = stats["sim"].to(torch.float32).cpu().numpy()    # [B, q_len, 1]
        gate = stats["gate"].to(torch.float32).cpu().numpy()   # [B, q_len, 1]
        m    = mask_ans.detach().to(torch.float32).cpu().numpy()  # [B, T, 1]

        # ★ 关键：对齐掩码到 sim 的序列长度
        if m.shape[1] != sim.shape[1]:
            # 增量解码场景，sim 的 q_len=1，只取最后一个位置的掩码
            m = m[:, -1:, :]

        for i, qid in enumerate(qids):
            sel = m[i, :, 0] > 0.5  # [q_len]
            # 兜底：如果没有选中 token（极少数异常），就用全部位置
            if not sel.any():
                sel = np.ones(sim.shape[1], dtype=bool)

            sim_i  = sim[i, sel, 0]
            gate_i = gate[i, sel, 0]
            if sim_i.size == 0:
                continue

            self.rows.append({
                "exp": name, "qid": int(qid),
                "sim_mean": float(sim_i.mean()),
                "sim_p95":  float(self.np.percentile(sim_i,95)),
                "gate_mean": float(gate_i.mean()),
                "gate_p95":  float(self.np.percentile(gate_i,95)),
                "n_tokens": int(sim_i.size)
            })
            self.hist_sim.extend(sim_i.tolist())
            self.hist_gate.extend(gate_i.tolist())
    def flush(self, tag: str):
        import numpy as np
        df = self.pd.DataFrame(self.rows)
        df.to_csv(os.path.join(self.out_dir, f"{tag}_gating_per_qid.csv"), index=False)
        np.save(os.path.join(self.out_dir, f"{tag}_hist_gate.npy"), np.array(self.hist_gate))
        np.save(os.path.join(self.out_dir, f"{tag}_hist_sim.npy"),  np.array(self.hist_sim))

# ====================== VQA Soft-Acc ======================
def norm_text(s: str) -> str:
    s = str(s).lower().strip()
    s = s.replace("&"," and ").replace("/"," ")
    for ch in [",",".","\"","'",";",";",":","?","!","(",")","-","–","—","\n","\t"]:
        s = s.replace(ch," ")
    s = " ".join([w for w in s.split() if w not in {"a","an","the"}])
    num = {"none":"0","zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6",
           "seven":"7","eight":"8","nine":"9","ten":"10"}
    s = " ".join([num.get(w,w) for w in s.split()])
    return " ".join(s.split())

def vqa_soft_acc(preds: List[Dict], gts: Dict[int, List[str]]) -> float:
    import numpy as np
    def acc_one(p, g_list):
        p_norm = norm_text(p)
        matches = sum(1 for g in g_list if norm_text(g) == p_norm)
        return min(1.0, matches / 3.0)
    return float(np.mean([acc_one(r['answer'], gts.get(r['question_id'], [])) for r in preds]))

# ====================== 固定 ICD 池（按 seed 固化并落盘） ======================
class FixedICDPool:
    """
    读取 VQA Q/A，构造 (image, question, neutral, reasoned) 样本池；
    然后按 seed 固定抽取 K 条，作为全局 few-shot 示例（每个样本均使用相同示例池）。
    """
    def __init__(self, questions_file: str, annotations_file: str, image_dir: str):
        with open(questions_file, 'r', encoding='utf-8') as f:
            q_list = json.load(f)['questions']
        with open(annotations_file, 'r', encoding='utf-8') as f:
            anns = json.load(f)['annotations']
        q_by_id = {q['question_id']: q for q in q_list}

        def reasoned(q: str, a: str) -> str:
            ql = q.strip().rstrip("?"); low = ql.lower(); a = a.strip()
            if ("how many" in low) or ("number" in low) or ("count" in low):
                return f"By counting the relevant objects for \"{ql}\", the answer is {a}."
            if ("color" in low) or ("colour" in low):
                return f"Observing the color indicated in \"{ql}\", the answer is {a}."
            if any(p in low for p in ["is there","are there","does the","do the","is the"]):
                return f"Checking the presence/condition in the image for \"{ql}\", the answer is {a}."
            return f"Considering the visual evidence for \"{ql}\", the answer is {a}."

        items = []
        for ann in anns:
            qid = ann["question_id"]
            qa = q_by_id.get(qid)
            if qa is None: continue
            answers = [norm_text(a["answer"]) for a in ann["answers"]]
            if not answers: continue
            maj = max(set(answers), key=answers.count)
            img_id = qa["image_id"]
            img_path = os.path.join(image_dir, f"COCO_val2014_{str(img_id).zfill(12)}.jpg")
            if not os.path.exists(img_path): continue
            items.append({
                "qid": qid, "image_id": img_id, "image_path": img_path,
                "question": qa["question"], "neutral": maj, "reasoned": reasoned(qa["question"], maj),
                "answer_type": ann.get("answer_type","other")
            })
        self.pool = items

    def build_or_load(self, out_path: str, k_shots: int, seed: int) -> List[Dict]:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if os.path.exists(out_path):
            with open(out_path, "r") as f: return json.load(f)
        rng = random.Random(seed)
        cand = self.pool.copy()
        rng.shuffle(cand)
        shots = cand[:k_shots]
        with open(out_path, "w") as f: json.dump(shots, f)
        return shots
