import os, json, random, re
from typing import List, Dict, Tuple, Optional

import math
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
    """
    将 [B, T, 1] 的 mask 对齐到当前层 hidden 的 [B, q_len, H]。
    q_len==T：直接返回；q_len==1（增量解码步）视为位于 Answer 段。
    """
    if m is None:
        return None
    if m.size(1) == h.size(1):
        return m
    last = m[:, -1:, :]
    last = torch.where(last > 0, last, torch.ones_like(last))
    return last

# ====================== 文本清理 & VQA soft-acc 等 ======================
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

# ====================== 聊天模板相关（通用 / Qwen-VL 用） ======================
def build_answer_mask_from_prompts(tokenizer, prompts: List[str], input_ids: torch.Tensor) -> torch.Tensor:
    """
    基于原始 prompt 文本定位最后一个 'ASSISTANT:'，用相同 tokenizer 重新分词得到 token 边界。
    返回 mask [B, T, 1]：'ASSISTANT:' 之后=1（目标回答段），其余=0。
    适用于使用类似 "USER: ... ASSISTANT:" 这种双角色模板的场景。
    """
    key = "ASSISTANT:"
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

def qwenvl_singleturn_prompt(q: str) -> str:
    """
    为 Qwen-VL 这类多模态对话模型构造一个简单的单轮问答模板：
      USER: <image>\nQuestion: ...\nASSISTANT:
    如果是 VQA/POPE，可强制 yes/no 回答；如果只跑 CHAIR，可以不用这个函数。
    """
    return (
        "USER: <image>\n"
        f"Answer yes or no based on the image.\nQuestion: {q}\n"
        "ASSISTANT:"
    )

# ====================== Qwen-VL 层路径工具 ======================
'''
def get_qwenvl_self_attn(model, layer_idx: int):
    """
    返回 Qwen-VL / 类 Qwen 架构中第 layer_idx 个 decoder block 的 self_attn 模块。

    兼容多种实现：
      - model.transformer.h[layer_idx].self_attn  （Qwen/Qwen-VL-Chat 风格）
      - model.model.layers[layer_idx].self_attn   （部分 HF 封装）
      - model.layers[layer_idx].self_attn         （兜底）
    如果某些版本使用 'attn' 而不是 'self_attn'，也会尝试兼容。
    """
    # 候选的「层列表」路径
    layer_lists = [
        getattr(getattr(model, "transformer", None), "h", None),
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
    ]
    for layers in layer_lists:
        if layers is None:
            continue
        try:
            layer = layers[layer_idx]
        except Exception:
            continue
        # self_attn 或 attn
        sa = getattr(layer, "self_attn", None)
        if sa is None:
            sa = getattr(layer, "attn", None)
        if sa is not None:
            return sa

    raise AttributeError(
        "Unable to locate Qwen-VL self attention layer; "
        "please check the model architecture and update get_qwenvl_self_attn."
    )
'''
def get_qwenvl_self_attn(model, layer_idx: int):
    """
    返回 Qwen2.5-VL 文本塔中第 layer_idx 个 decoder 层的 self-attn 模块。
    """
    # 1. 找到 text tower
    text_model = getattr(model, "model", None)
    if text_model is None:
        raise AttributeError("model.model not found in Qwen2.5-VL model")

    lm = getattr(text_model, "language_model", None)
    if lm is None:
        raise AttributeError("model.model.language_model not found; please inspect model structure.")

    layers = getattr(lm, "layers", None)
    if layers is None:
        raise AttributeError("language_model.layers not found; please inspect model structure.")

    if not (0 <= layer_idx < len(layers)):
        raise IndexError(f"layer_idx {layer_idx} out of range [0, {len(layers)-1}]")

    layer = layers[layer_idx]

    # 2. 在这个 layer 里找 self-attn (名字可能是 self_attn / self_attention / attn 等)
    for attn_name in ("self_attn", "self_attention", "attn", "mixer"):
        if hasattr(layer, attn_name):
            return getattr(layer, attn_name)

    raise AttributeError(
        f"Layer {layer_idx} found, but no self-attention attr in it. "
        f"Available attrs: {dir(layer)}"
    )

# ====================== CARD 向量（批量） ======================
class _CardCapture:
    """捕获目标层 self_attn 的前后 hidden，用于 delta=after-before。兼容 kwargs 调用。"""
    def __init__(self, target_layer):
        self.target_layer = target_layer
        self.before = None
        self.after = None
        self._pre = None
        self._post = None

    def __enter__(self):
        def pre(_m, args, kwargs):
            hs = None
            if kwargs is not None:
                hs = kwargs.get("hidden_states", None)
            if hs is None and len(args) > 0:
                hs = args[0]
            self.before = hs.detach() if hs is not None else None

        def post(_m, args, output):
            out0 = output[0] if isinstance(output, (tuple, list)) else output
            self.after = out0.detach()

        self._pre = self.target_layer.register_forward_pre_hook(pre, with_kwargs=True)
        self._post = self.target_layer.register_forward_hook(post, with_kwargs=False)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._pre:  self._pre.remove()
        if self._post: self._post.remove()

@torch.no_grad()
def compute_card_vector_batch(
    model,
    processor,
    images: List[Image.Image],
    questions: List[str],
    layer_idx: int = 27,
    pooling: str = "attn",
    local: bool = False,
) -> torch.Tensor:
    """
    在 Qwen-VL 上计算 CARD 向量。

    默认使用一个简单的对话模板：
      USER: <image> ...
      ASSISTANT:

    参数:
      - model: 已经在 GPU 上的 Qwen-VL 模型
      - processor: 与模型配套的 AutoProcessor（或等价类）
      - images: PIL Image 列表
      - questions: 文本问题列表
      - layer_idx: 抽取的层索引
      - pooling: "attn" 或 "mean"
      - local:
          - False -> 返回 [B, H]
          - True  -> 返回 [B, T, H]
    """
    texts = [qwenvl_singleturn_prompt(q) for q in questions]
    imgs  = [[img] for img in images]
    inputs = processor(text=texts, images=imgs, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    target_layer = get_qwenvl_self_attn(model, layer_idx)
    with _CardCapture(target_layer) as cap:
        _ = model(**inputs, output_hidden_states=False, return_dict=True)

    before, after = cap.before, cap.after  # [B,T,H]
    if before is None or after is None:
        raise RuntimeError("CARD capture failed: before/after is None; check hooks and model forward arguments.")

    delta = after - before
    B, q_len, H = delta.shape
    mask = delta.new_ones((B, q_len, 1))  # 默认全 1

    # 使用 attention_mask 推断左侧 padding，用于屏蔽 pad token 的 delta
    if "attention_mask" in inputs:
        pad_left = (inputs["attention_mask"] == 0).sum(dim=1)   # [B]
        pos = torch.arange(q_len, device=delta.device).unsqueeze(0).expand(B, q_len)  # [B, q_len]
        left_pad = (pos < pad_left.unsqueeze(1)).unsqueeze(-1).to(delta.dtype)        # [B, q_len, 1]
        mask = mask * (1.0 - left_pad)

    if local:
        v_tok = F.normalize(delta, p=2, dim=-1) * mask
        return v_tok

    if pooling == "mean":
        v = (delta * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
    else:
        # attn 代理：用 |delta| 的范数加权
        w = (delta.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6) * mask
        v = (delta * w).sum(dim=1) / (w.sum(dim=1) + 1e-6)

    return F.normalize(v, p=2, dim=-1).detach()

# ====================== Beta 门控（动态批向量 + 记录） ======================
class BayesianGatingHookMaskedDynamic:
    """
    支持 v_batch: [B,H] 或 token-specific: [B,q_len,H]
    仅在 Answer 段注入；记录 sim/gate 以供可视化。
    """
    def __init__(self, max_alpha=2, sensitivity=5.0, concentration=1.0,
                 carrier=mask_carrier, clamp=(0.0,0.8), rms_match=False, record=True):
        self.enabled = True
        self.v_batch = None
        self.max_alpha = max_alpha; self.k = sensitivity; self.c = concentration
        self.carrier = carrier
        self.clamp = clamp
        self.rms_match = rms_match
        self.record = record
        self._last = None

    def set_vector(self, v_batch: torch.Tensor):
        self.v_batch = F.normalize(v_batch, p=2, dim=-1)
        self.enabled = True
        for attr in ["_dbg_seen", "_dbg_printed"]:
            if hasattr(self, attr): delattr(self, attr)

    def disable(self):
        self.enabled = False

    def fetch_last_stats(self):
        return self._last

    def __call__(self, module, module_in, module_out):
        if (not self.enabled) or (self.v_batch is None):
            return module_out

        h = module_out[0]  # [B,q_len,H]，注意 beam search 时 B = 原B * num_beams
        v = self.v_batch.to(h.device, dtype=h.dtype)  # [B0,H] 或 [B0,Tv,H]

        # === (A) 对齐 batch 维（支持 beam search）===
        if v.size(0) != h.size(0):
            if h.size(0) % v.size(0) == 0:
                rep = h.size(0) // v.size(0)
                v = v.repeat_interleave(rep, dim=0)
            else:
                rep = math.ceil(h.size(0) / v.size(0))
                v = v.repeat_interleave(rep, dim=0)[:h.size(0)]

        # === (B) 对齐序列维 ===
        if v.dim() == 2:  # [B,H] -> [B,T,H]
            v = v.unsqueeze(1).expand(-1, h.size(1), -1)
        elif v.dim() == 3:
            if v.size(1) == 1:
                v = v.expand(-1, h.size(1), -1)
            elif v.size(1) > h.size(1):
                v = v[:, -h.size(1):, :]
            elif v.size(1) < h.size(1):
                pad = v.new_zeros(v.size(0), h.size(1) - v.size(1), v.size(2))
                v = torch.cat([v, pad], dim=1)

        # 归一化与相似度
        h_n = F.normalize(h, p=2, dim=-1)
        v_n = F.normalize(v, p=2, dim=-1)
        sim = F.cosine_similarity(h_n, v_n, dim=-1, eps=1e-6).unsqueeze(-1)  # [B,q_len,1]
        alpha = F.softplus(self.k * sim + self.c)
        beta  = F.softplus(-self.k * sim + self.c)
        gate  = (alpha / (alpha + beta)).clamp(self.clamp[0], self.clamp[1])

        # 掩码（也要对齐 batch）
        m = _align_mask_to_seq(self.carrier.mask, h)
        if m is not None:
            if m.size(0) != h.size(0):
                if h.size(0) % m.size(0) == 0:
                    rep = h.size(0) // m.size(0)
                    m = m.repeat_interleave(rep, dim=0)
                else:
                    rep = math.ceil(h.size(0) / m.size(0))
                    m = m.repeat_interleave(rep, dim=0)[:h.size(0)]
            gate = gate * m.to(h.dtype)

        # 记录（可选）
        if self.record:
            self._last = {
                "sim":  sim.detach().to(torch.float32).cpu(),
                "gate": gate.detach().to(torch.float32).cpu()
            }

        # === RMS 匹配（启用时，按 hidden 的 RMS 缩放更新量） ===
        update_vec = v_n
        if self.rms_match:
            den = (h.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6)
            update_vec = update_vec / den

        # 应用更新 + 数值消毒
        out0 = h + self.max_alpha * gate * update_vec
        out0 = torch.nan_to_num(out0, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.record:
            self._last = {
                "sim":  sim.detach().to(torch.float32).cpu(),
                "gate": gate.detach().to(torch.float32).cpu()
            }
        return (out0,) + module_out[1:]

# ====================== 可视化记录器（可选，用不到可忽略） ======================
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
        if stats is None or ("sim" not in stats) or ("gate" not in stats) \
           or (stats["sim"] is None) or (stats["gate"] is None):
            return
        import numpy as np
        sim  = stats["sim"].to(torch.float32).cpu().numpy()
        gate = stats["gate"].to(torch.float32).cpu().numpy()
        m    = mask_ans.detach().to(torch.float32).cpu().numpy()
        if m.shape[1] != sim.shape[1]:
            m = m[:, -1:, :]
        for i, qid in enumerate(qids):
            sel = m[i, :, 0] > 0.5
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

# ====================== VQA Soft-Acc（保留） ======================
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
