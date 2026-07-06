from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt


Tensor = torch.Tensor

# === ECOC ===

def _to_device(t: Tensor, ref: Tensor | torch.device | None) -> Tensor:
    if isinstance(ref, torch.device):
        return t.to(ref)
    if isinstance(ref, torch.Tensor):
        return t.to(ref.device)
    return t


def pairwise_hamming(a: Tensor, b: Tensor) -> Tensor:
    """Pairwise Hamming distances between binary {0,1} row‑vectors.
    a: [Na, L], b: [Nb, L]  →  out: [Na, Nb]
    """
    # XOR then sum bits
    return torch.cdist(a.float(), b.float(), p=1)  # L1 equals Hamming for {0,1}


def distance_euclidean(x: Tensor, y: Tensor) -> Tensor:
    """Pairwise Euclidean (L2) distances between real row‑vectors x and y.
    x: [N, L], y: [M, L] → [N, M]
    """
    diff2 = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(dim=-1)
    return torch.sqrt(diff2 + 1e-6)


# ======================================
# Build Hamming blocks & codebook select
# ======================================

def _hamming_block_m(num_classes: int) -> int:
    """Choose the smallest Hamming(n=2^m−1, k=n−m) such that the number of
    codewords 2^k covers `num_classes` and is as close as possible from above.
    Examples: C≤16 → m=3 (7,4); 17≤C≤2048 → m=4 (15,11); 2049≤C≤2^26 → m=5 (31,26).
    """
    import math
    m = 2
    while True:
        n = 2 ** m - 1
        k = n - m
        codewords = 2 ** k
        if codewords >= num_classes:
            return m
        m += 1


def _hamming_parity_and_generator(m: int, device: torch.device) -> Tuple[Tensor, Tensor]:
    """Return (H, G) for Hamming(n=2^m−1, k=n−m) in systematic form over GF(2).
    Construct columns as all non‑zero m‑bit vectors; then permute so last m
    columns are the identity, i.e., H = [P | I_m], G = [I_k | P^T].
    """
    n = 2 ** m - 1
    # integers 1..n label the non‑zero columns
    cols = torch.arange(1, n + 1, dtype=torch.int64, device=device)
    # bits: [m, n], column j is the m‑bit binary of cols[j]
    bits = (((cols.unsqueeze(0)) >> torch.arange(m, device=device).unsqueeze(1)) & 1).to(torch.uint8)

    # find indices of identity columns (powers of two)
    powers = (1 << torch.arange(m, device=device)).to(torch.int64)
    id_idx = [(cols == p).nonzero(as_tuple=False).item() for p in powers]
    mask = torch.ones(n, dtype=torch.bool, device=device)
    mask[id_idx] = False
    nonid_idx = mask.nonzero(as_tuple=False).squeeze(1)
    perm = torch.cat([nonid_idx, torch.tensor(id_idx, device=device)])

    H = bits[:, perm]  # [m, n], last m columns == I_m
    P = H[:, : n - m]  # [m, n−m]
    k = n - m
    G = torch.cat([torch.eye(k, dtype=torch.uint8, device=device), P.T], dim=1)  # [k, n]
    return H, G


def _sample_codewords(G: Tensor, num_classes: int, *, sample_size: int = 4096, seed: Optional[int] = None) -> Tensor:
    """Pick `num_classes` Hamming codewords (rows) maximizing minimum pairwise distance.
    Vectorized farthest‑first from a random pool of messages.
    Returns: [C, n] binary {0,1} tensor.
    """
    if seed is not None:
        torch.manual_seed(seed)
    device = G.device
    k, n = G.shape

    # ---- GPU‑safe GF(2) matmul (no addmm on Long): use bitwise‑AND + popcount mod 2 ----
    pool = torch.randint(0, 2, (min(sample_size, 1 << min(k, 16)), k), device=device, dtype=torch.uint8)
    Gu8 = G.to(torch.uint8)
    # pool: [P,k], Gu8: [k,n] → (pool[:,:,None] & Gu8[None,:,:]).sum(dim=1) % 2 → [P,n]
    cand_bits = (pool.unsqueeze(2) & Gu8.unsqueeze(0)).sum(dim=1) & 1
    cand = cand_bits.to(torch.float32)

    # Start with the medoid in L2 sense to stabilize selection
    dist = distance_euclidean(cand, cand)
    start_idx = torch.argmin(dist.mean(dim=1))
    selected = [int(start_idx)]

    # Farthest‑first: iterate C−1 times; inner distances are fully vectorized
    for _ in range(num_classes - 1):
        idx = torch.as_tensor(selected, device=device, dtype=torch.long)
        ref = cand.index_select(0, idx)  # [k, n] even when k==1
        dmin = torch.min(pairwise_hamming(cand, ref), dim=1).values  # [P]
        dmin.index_fill_(0, idx, -1.0)
        nxt = torch.argmax(dmin)
        selected.append(int(nxt))

    idx = torch.as_tensor(selected, device=device, dtype=torch.long)
    codebook = cand.index_select(0, idx).contiguous()
    return codebook


def _concat_blocks(blocks: list[Tensor]) -> Tensor:
    return torch.cat(blocks, dim=1)


# ======================
# ECOC Head (one‑touch)
# ======================

@dataclass
class ECOCHead:
    num_classes: int
    dataset_hint: Optional[str] = None
    extension_steps: int = 1
    m: Optional[int] = None
    bit_values: Tuple[float, float] = (0.0, 1.0)
    seed: Optional[int] = None
    device: Optional[torch.device] = None

    # populated after build()
    codebook: Optional[Tensor] = None  # [C, L], binary in {0,1} mapped to bit_values

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build()

    @property
    def code_length(self) -> int:
        assert self.codebook is not None
        return int(self.codebook.shape[1])

    # ---------- build ----------
    def build(self, class_averages: Optional[Tensor] = None):
        """Build codebook: concatenate `extension_steps` Hamming blocks and optionally
        align classes using class‑average similarity.
        class_averages: [C, ...] tensor used for order optimization (optional).
        """
        C = self.num_classes
        dev = self.device
        m = self.m or _hamming_block_m(C)
        _, G = _hamming_parity_and_generator(m, dev)

        blocks: list[Tensor] = []
        for s in range(self.extension_steps):
            blk = _sample_codewords(G, C, sample_size=max(4096, 16 * C), seed=None if self.seed is None else self.seed + s)
            blocks.append(blk)
        code = _concat_blocks(blocks)  # [C, L]

        # Optional lightweight order optimization using cosine between class averages
        if class_averages is not None:
            code = self._order_by_correlation(code, class_averages)

        # Map bits to desired values
        lo, hi = self.bit_values
        code = code * (hi - lo) + lo
        self.codebook = code.to(dev)

    def _order_by_correlation(self, code: Tensor, class_averages: Tensor) -> Tensor:
        """Approximate ordering: align leading principal axes of data & codes.
        Fully vectorized (no Python swap loops)."""
        dev = self.device
        C = self.num_classes
        X = class_averages.view(C, -1).float().to(dev)
        X = X - X.mean(dim=1, keepdim=True)
        # data embedding = first left singular vector
        u_data = torch.linalg.svdvals(X @ X.T).unsqueeze(0)  # magnitude only (ordering surrogate)
        # code embedding via {−1,+1} transform then PCA‑1
        B = (code * 2 - 1).float()
        u_code = torch.linalg.svdvals(B @ B.T).unsqueeze(0)
        # simple monotone sort alignment: sort classes & rows by their row sums
        order_data = torch.argsort(X.sum(dim=1))
        order_code = torch.argsort(B.sum(dim=1))
        inv = torch.empty_like(order_code)
        inv[order_code] = torch.arange(C, device=dev)
        return code[inv[order_data]]

    # ---------- training utils ----------
    def encode(self, targets: Tensor) -> Tensor:
        """Map integer labels → ECOC code rows. targets: [N] int64."""
        assert self.codebook is not None, "build() not called"
        return self.codebook.index_select(0, targets.long())

    def _to01(self, code: Tensor) -> Tensor:
        lo, hi = self.bit_values
        return (code - lo) / (hi - lo + 1e-12)

    def loss_mse(self, outputs: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
        """MSE(outputs, ECOC(targets)). outputs: [N, L]."""
        tgt = self.encode(targets)
        loss = (outputs - tgt).pow(2).mean(dim=1)
        return loss.mean() if reduction == "mean" else loss

    def loss_bce(self, outputs: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
        """
        BCEWithLogitsLoss(outputs, target_bits∈{0,1}).
        bit_values가 (-1,+1)여도 내부에서 (0,1)로 정규화해서 사용.
        """
        tgt_bits01 = self._to01(self.encode(targets))  # [N, L] in {0,1}
        return F.binary_cross_entropy_with_logits(outputs, tgt_bits01, reduction=reduction)

    def loss_ce(self,
                outputs: Tensor,
                targets: Tensor,
                metric: str = "euclidean",
                temp: float = 1.0,
                squared: bool = True,
                reduction: str = "mean") -> Tensor:
        """
        logits_k = - distance(outputs, codebook[k]) / temp
        metric: 'euclidean' 또는 'hamming'
        squared=True면 제곱거리 사용(권장: 더 부드러운 로그-소프트맥스).
        """
        assert self.codebook is not None
        if metric == "hamming":
            lo, hi = self.bit_values
            thr = (lo + hi) / 2
            bx = (outputs > thr).float()
            d = pairwise_hamming(bx, self.codebook)  # [N, C]
            logits = -d / (temp + 1e-12)
        else:
            # Euclidean
            diff2 = (outputs.unsqueeze(1) - self.codebook.unsqueeze(0)).pow(2).sum(dim=-1)  # [N, C]
            if squared:
                logits = -diff2 / (temp + 1e-12)
            else:
                logits = -torch.sqrt(diff2 + 1e-6) / (temp + 1e-12)
        return F.cross_entropy(logits, targets, reduction=reduction)

    # ---------- inference ----------
    def decode(self, outputs: Tensor, metric: str = "euclidean") -> Tensor:
        """Vectorized decoding: pick class with smallest distance to each row of outputs.
        metric ∈ {"euclidean", "hamming"}.
        """
        assert self.codebook is not None
        m = (metric or "euclidean").lower()
        if m == "hamming":
            lo, hi = self.bit_values
            thr = (lo + hi) / 2
            bx = (outputs > thr).float()
            d = pairwise_hamming(bx, self.codebook)
        elif m == "euclidean":
            d = distance_euclidean(outputs, self.codebook)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        return torch.argmin(d, dim=1)

    # ---------- model patch (optional) ----------
    def patch_last_linear(self, model: nn.Module) -> nn.Module:
        """Replace the last nn.Linear with out_features==num_classes by a new Linear
        with out_features==code_length. Returns the patched model.
        """
        L = self.code_length
        C = self.num_classes

        last_linear: Optional[nn.Linear] = None
        last_name: Optional[str] = None
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) and getattr(m, "out_features", None) == C:
                last_linear = m
                last_name = name
        if last_linear is None:
            return model  # nothing to patch

        new_head = nn.Linear(last_linear.in_features, L, bias=getattr(last_linear, "bias", None) is not None)
        new_head = new_head.to(next(model.parameters()).device)

        # Install the new head by walking attributes
        def _set_by_name(root: nn.Module, dotted: str, value: nn.Module):
            parts = dotted.split(".")
            parent = root
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], value)

        _set_by_name(model, last_name, new_head)
        return model

def ecoc_one_touch(model: nn.Module, outputs: Tensor, targets: Optional[Tensor], *,
                   head: ECOCHead) -> Tuple[Optional[Tensor], Tensor]:
    """If `targets` is given → return (loss, decoded_pred). If None → (None, pred).
    This keeps training/eval code one‑liner‑ish while staying explicit.
    """
    if targets is not None:
        loss = head.loss(outputs, targets)
        pred = head.decode(outputs)
        return loss, pred
    else:
        return None, head.decode(outputs)

# === SoftSNN ===
class SoftSNNBounder:
    """
    SoftSNN Bound-and-Protect (지연 스냅샷 지원)
    - attach(model) 또는 model.apply(bounder)로 pre-hook 설치(초기엔 '비활성')
    - 훈련 에폭 1회 후: capture_snapshot(model) → activate() 하면 그때부터 바운딩 시작
    - 모드:
        'bnp1': |w| >= th → 0
        'bnp2': |w| >= th → th (캡핑)
        'bnp3': |w| >= th → hp(분포 대표값, 기본=중앙값)
        'range': [-max_val, max_val] 하드 클립 (스냅샷 불필요)
    - per: 'layer' (스칼라) / 'channel' (출력 채널별)
    """
    def __init__(
            self,
            mode: str = "bnp2",
            per: str = "layer",
            hp_quantile: float = 0.5,
            symmetric: bool = True,
            include: Tuple[type, ...] = (nn.Conv2d, nn.Linear, nn.Conv1d, nn.Conv3d),
            exclude_bias: bool = True,
            min_val: Optional[float] = None,    # range 모드용
            max_val: Optional[float] = None,
    ):
        assert mode in ("bnp1", "bnp2", "bnp3", "range")
        assert per in ("layer", "channel")
        if mode == "range":
            assert (max_val is not None) or (min_val is not None), "range 모드는 min/max 필요"

        self.mode = mode
        self.per = per
        self.hp_quantile = hp_quantile
        self.symmetric = symmetric
        self.include = include
        self.exclude_bias = exclude_bias
        self.min_val = min_val
        self.max_val = max_val

        self._handles = []
        self._armed = False          # 활성화( enforce ) 여부
        self._snapshotted = False    # 스냅샷 보유 여부

    # ---------------- 내부 유틸 ----------------
    @staticmethod
    def _is_supported(m: nn.Module, include) -> bool:
        return any(isinstance(m, t) for t in include) and hasattr(m, "weight")

    def _stats_from_weight(self, w: torch.Tensor):
        base = w.detach().abs() if self.symmetric else w.detach()
        if self.per == "layer":
            th = base.max().to(w)
            hp = torch.quantile(base.flatten(), self.hp_quantile).to(w)
        else:
            # per-channel (out-dim=0)
            red_dims = tuple(range(1, w.dim()))
            th = base.amax(dim=red_dims)                               # [C_out]
            hp = base.flatten(start_dim=1).quantile(self.hp_quantile, dim=1)  # [C_out]
            view_shape = [w.shape[0]] + [1]*(w.dim()-1)
            th = th.view(*view_shape); hp = hp.view(*view_shape)
        return th, hp

    def _ensure_attrs(self, m: nn.Module):
        # 스냅샷 저장 버퍼(혹은 속성) 존재 확인
        if not hasattr(m, "_softsnn_th"):
            setattr(m, "_softsnn_th", None)
            setattr(m, "_softsnn_hp", None)

    def _hook_fn(self, m: nn.Module, _):
        # 1) range 모드는 항상 즉시 클립
        if self.mode == "range":
            w = m.weight.data
            # 대칭 범위 기본값
            lo = -self.max_val if (self.max_val is not None and self.min_val is None and self.symmetric) else self.min_val
            hi =  self.max_val if (self.max_val is not None) else (max(self.min_val, 0.0) if self.min_val is not None else None)
            if lo is not None and hi is not None:
                w.clamp_(lo, hi)
            elif lo is not None:
                w.clamp_min_(lo)
            elif hi is not None:
                w.clamp_max_(hi)
            return

        # 2) BnP 모드: 활성화 전/스냅샷 전이면 아무 것도 안 함
        if not self._armed:
            return
        self._ensure_attrs(m)
        if (m._softsnn_th is None) or (m._softsnn_hp is None):
            return

        w = m.weight.data
        th = m._softsnn_th
        hp = m._softsnn_hp

        if self.symmetric:
            mag = w.abs()
            sgn = w.sign()
        else:
            mag = w
            sgn = torch.ones_like(w)

        mask = (mag >= th)
        if self.mode == "bnp1":
            repl = torch.zeros_like(mag)
        elif self.mode == "bnp2":
            repl = th.expand_as(mag)
        else:  # 'bnp3'
            repl = hp.expand_as(mag)

        w.copy_(torch.where(mask, repl * sgn, w))

    # ---------------- public API ----------------
    def __call__(self, m: nn.Module):
        """ model.apply(bounder)에서 호출됨: pre-hook만 설치(비활성 상태) """
        if not self._is_supported(m, self.include):
            return
        if hasattr(m, "_softsnn_bound_handle"):
            return
        h = m.register_forward_pre_hook(self._hook_fn, with_kwargs=False)
        m._softsnn_bound_handle = h
        self._handles.append(h)

    def attach(self, model: nn.Module):
        """원터치 설치: pre-hook만 심고 비활성 상태로 둠"""
        model.apply(self)
        return self

    @torch.no_grad()
    def capture_snapshot(self, model: nn.Module):
        """현재 가중치로 임계치/대표값 스냅샷 저장 (에폭 1회 후 호출 권장)"""
        self._snapshotted = True
        for m in model.modules():
            if not self._is_supported(m, self.include):
                continue
            self._ensure_attrs(m)
            th, hp = self._stats_from_weight(m.weight)
            # 장치/정밀도 맞춰 저장
            m._softsnn_th = th.to(device=m.weight.device, dtype=m.weight.dtype)
            m._softsnn_hp = hp.to(device=m.weight.device, dtype=m.weight.dtype)

    def clear_snapshot(self, model: nn.Module):
        """스냅샷 제거(재스냅샷용)"""
        self._snapshotted = False
        for m in model.modules():
            if self._is_supported(m, self.include):
                if hasattr(m, "_softsnn_th"):
                    m._softsnn_th = None
                if hasattr(m, "_softsnn_hp"):
                    m._softsnn_hp = None

    def activate(self):
        """바운딩 활성화(스냅샷이 있어야 효과)"""
        self._armed = True

    def deactivate(self):
        """바운딩 비활성화(후크는 유지)"""
        self._armed = False

    def remove(self):
        """후크 제거"""
        for h in self._handles:
            try: h.remove()
            except: pass
        self._handles.clear()

def install_softsnn(model: nn.Module, **kwargs) -> SoftSNNBounder:
    """
    예) bounder = install_softsnn(net, mode='bnp2', per='channel')
    (지연 스냅샷이므로 즉시 바운딩은 안 됨: 에폭 뒤 capture_snapshot+activate 필요)
    """
    b = SoftSNNBounder(**kwargs)
    return b.attach(model)

# === Fault routing ===

# ────────────────────────── 공통 헬퍼 ──────────────────────────
_SUPPORTED = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)

def _is_supported(m: nn.Module) -> bool:
    return isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)) and hasattr(m, "weight")

def _build_consecutive_pairs(model: nn.Module) -> List[Tuple[str, str]]:
    """
    지원 레이어들(Conv/Linear)을 나열해서, 인접한 쌍 중
    'consumer.in_channels == producer.out_channels'인 경우만 페어로 묶는다.
    VGG/ResNet 등 일반적 구조에서 안전하게 동작하는 fallback.
    """
    nm = dict(model.named_modules())
    names = [n for n, m in model.named_modules() if _is_supported(m)]
    pairs: List[Tuple[str, str]] = []
    for a, b in zip(names[:-1], names[1:]):
        wa = nm[a].weight
        wb = nm[b].weight
        if wb.shape[1] == wa.shape[0]:
            pairs.append((a, b))
    return pairs

# 전용 빌더가 이미 있으면 그걸 쓰고, 없으면 fallback 사용
def build_pairs_vgg(model: nn.Module) -> List[Tuple[str, str]]:
    return _build_consecutive_pairs(model)

def build_pairs_resnet(model: nn.Module) -> List[Tuple[str, str]]:
    return _build_consecutive_pairs(model)

def build_pairs_mlp(model: nn.Module) -> List[Tuple[str, str]]:
    nm = dict(model.named_modules())
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    pairs: List[Tuple[str, str]] = []
    for a, b in zip(names[:-1], names[1:]):
        la: nn.Linear = nm[a]  # type: ignore
        lb: nn.Linear = nm[b]  # type: ignore
        if lb.in_features == la.out_features:
            pairs.append((a, b))
    return pairs

def attach_slot_activity_tracker(model: nn.Module, beta: float = 0.9):
    """
    각 consumer 레이어(Conv/Linear)의 입력 텐서를 forward-pre-hook으로 받아
    채널별 |activation| 평균을 EMA로 추적하여 m._slot_activity에 저장.
    - 그래프 보존 방지: no_grad + detach 사용
    - 메모리 절약: in-place EMA 갱신
    반환: (handles, module_name_list)
    """
    handles = []
    names = dict(model.named_modules())

    def _reduce_per_channel(x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            # 안전장치
            return torch.zeros(0, device=x.device, dtype=torch.float32)
        dims = tuple(d for d in range(x.dim()) if d != 1)
        # 그래프 끊기 + FP32로 집계
        with torch.no_grad():
            r = x.detach().abs().mean(dim=dims).to(dtype=torch.float32)
        return r

    for name, m in model.named_modules():
        if not _is_supported(m):
            continue

        def _pre_hook(mod: nn.Module, inputs):
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return

            # act는 항상 leaf + no grad
            act = _reduce_per_channel(x)
            if act.numel() == 0:
                return

            dev = mod.weight.device
            with torch.no_grad():
                act = act.to(device=dev, dtype=torch.float32)
                if not hasattr(mod, "_slot_activity"):
                    mod._slot_activity = act.clone()
                    mod._slot_beta = float(beta)
                else:
                    b = float(getattr(mod, "_slot_beta", beta))
                    # in-place EMA: m._slot_activity = b*m._slot_activity + (1-b)*act
                    mod._slot_activity.mul_(b).add_(act, alpha=(1.0 - b))

        h = m.register_forward_pre_hook(_pre_hook, with_kwargs=False)
        handles.append(h)

    return handles, list(names.keys())

def _apply_perm_inplace_input_channel(consumer: nn.Module, perm_full: torch.Tensor):
    """
    입력 채널 축을 IN-PLACE로 치환한다(사이클 분해). 거대한 index_select/contiguous 복사를 없애서
    피크 메모리를 거의 0으로 만든다. Linear, Conv{1,2,3}d 지원.
    """
    with torch.no_grad():
        W = consumer.weight.data
        dev = W.device
        perm = perm_full.to(device=dev)
        K = int(perm.numel())

        def _cycle_apply(get_slice, set_slice):
            visited = torch.zeros(K, dtype=torch.bool, device=dev)
            for start in range(K):
                if visited[start]:
                    continue
                nxt = int(perm[start])
                if nxt == start:
                    visited[start] = True
                    continue
                tmp = get_slice(start).clone()
                cur = start
                while True:
                    nxt = int(perm[cur])
                    if nxt == start:
                        break
                    set_slice(cur, get_slice(nxt))
                    visited[cur] = True
                    cur = nxt
                set_slice(cur, tmp)
                visited[cur] = True

        if isinstance(consumer, nn.Linear):           # W: [Cout, Cin]
            _cycle_apply(
                get_slice=lambda j: W[:, j],
                set_slice=lambda j, val: W[:, j].copy_(val),
            )
        else:                                          # Conv: [Cout, Cin, kH, kW]
            _cycle_apply(
                get_slice=lambda j: W[:, j, ...],
                set_slice=lambda j, val: W[:, j, ...].copy_(val),
            )

# --------- ChannelRouter ---------
class ChannelRouter:
    """
    최소 코어 라우터.
    - 모델과 stuck_map(dict: 모듈명 -> weight와 동일 shape의 BoolTensor; True = 고장)을 보유
    - consumer '입력 채널' 축만 재배치(permute)하여, 더 '건강한' producer 출력 채널을 우선 연결
    - Linear/Conv(1/2/3d)만 지원, groups!=1인 Conv는 스킵 (안전)
    """
    def __init__(self, model: nn.Module, stuck_map: Dict[str, torch.Tensor],
                 alpha: float = 0.5, soft_beta: float = 0.3, swap_frac: float = 0.25):
        """
        alpha: S = alpha*|W|열L1 + (1-alpha)*EMA활성도
        soft_beta: 소프트 라우팅 강도 (0~1). 최종 scale = soft_beta + (1-soft_beta)*H
        swap_frac: 퍼뮤테이션 예산(M = swap_frac*K)
        """
        self.model = model
        self.stuck_map = stuck_map
        self.alpha = float(alpha)
        self.soft_beta = float(soft_beta)
        self.swap_frac = float(max(0.0, min(1.0, swap_frac)))
        self._routed_once = set()

    @staticmethod
    def _health_from_mask(mask: torch.Tensor, dim_out: int = 0) -> torch.Tensor:
        out = mask.shape[dim_out]
        flat = mask.view(out, -1).float()
        stuck_ratio = flat.mean(dim=1)
        return (1.0 - stuck_ratio).clamp(0.0, 1.0)  # [Cout]

    @staticmethod
    def _slot_importance(cons: nn.Module, alpha: float = 0.5) -> torch.Tensor:
        W = cons.weight
        if W.dim() == 2:  # [Cout, Cin]
            S_w = W.abs().sum(dim=0)
        else:  # [Cout, Cin, ...]
            S_w = W.abs().view(W.shape[0], W.shape[1], -1).sum(dim=(0, 2))
        S_w = S_w.to(dtype=torch.float32)
        S_w = S_w / (S_w.mean() + 1e-8)

        if hasattr(cons, "_slot_activity"):
            S_a = cons._slot_activity.to(dtype=torch.float32, device=W.device)
            if S_a.numel() != W.shape[1]:
                # 크기 불일치시 무시
                S_a = None
        else:
            S_a = None

        if S_a is not None:
            S_a = S_a / (S_a.mean() + 1e-8)
            S = alpha * S_w + (1.0 - alpha) * S_a
        else:
            S = S_w
        return S  # [Cin]

    def _soft_attenuation(self, consumer: nn.Module, H: torch.Tensor):
        Cin = consumer.weight.shape[1]
        Cout = consumer.weight.shape[0]
        K = min(Cin, H.numel())
        scale = self.soft_beta + (1.0 - self.soft_beta) * H[:K]  # [K]
        scale = scale.to(device=consumer.weight.device, dtype=consumer.weight.dtype)

        if isinstance(consumer, nn.Linear):
            consumer.weight.data[:, :K] *= scale.view(1, K)
        else:
            # Conv: [Cout, Cin, kH, kW]
            consumer.weight.data[:, :K, ...] *= scale.view(1, K, *([1] * (consumer.weight.dim() - 2)))

    @torch.no_grad()
    def route_pair(self,
                   producer_name: str,
                   consumer_name: str,
                   mode: str = "desc",
                   strategy: str = "consumer") -> bool:
        if strategy != "consumer":
            return False

        nm = dict(self.model.named_modules())
        if producer_name not in nm or consumer_name not in nm:
            return False
        cons = nm[consumer_name]

        if not _is_supported(cons) or not hasattr(cons, "weight"):
            return False
        if isinstance(cons, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if getattr(cons, "groups", 1) != 1:
                return False

        if producer_name not in self.stuck_map:
            return False

        # H: producer 출력채널 건강도
        mask_cpu = self.stuck_map[producer_name].to("cpu", non_blocking=True)
        H_cpu = self._health_from_mask(mask_cpu)  # [Cout] on CPU
        H = H_cpu.to(cons.weight.device, dtype=cons.weight.dtype, non_blocking=True)

        if H.numel() == 0:
            return False

        Cin = cons.weight.shape[1]
        Cout = H.shape[0]
        K = min(Cin, Cout)
        if K == 0:
            return False

        # 1) 소프트 라우팅(attenuation) 먼저: 손상 채널 영향 완화
        self._soft_attenuation(cons, H)

        # 2) 중요도 기반 부분 퍼뮤테이션
        #    동일 페어 1회만 적용(지터 방지)
        done_key = (producer_name, consumer_name)
        if done_key in self._routed_once:
            return False

        S_full = self._slot_importance(cons, alpha=self.alpha)  # [Cin]
        S = S_full[:K]
        desc = (mode == "desc")
        # 상위 매칭 예산
        M = max(1, int(self.swap_frac * K))

        # 중요한 입력 슬롯 상위 M, 건강한 채널 상위 M
        j_sorted = torch.argsort(S, descending=desc)[:M]
        i_sorted = torch.argsort(H[:K], descending=desc)[:M]

        # 기대 이득(라이트 추정): sum S[j]*(H[i]-H[j])
        expected_gain = float((S.index_select(0, j_sorted) *
                               (H.index_select(0, i_sorted) - H.index_select(0, j_sorted))).sum().item())
        if expected_gain <= 0.0:
            # 소프트만 적용하고 종료
            self._routed_once.add(done_key)
            return True  # 소프트는 했으니 True

        # perm_full 구성: 우선 identity
        device = cons.weight.device
        perm_full = torch.arange(Cin, device='cpu')

        used_i = set()
        for jj, ii in zip(j_sorted.tolist(), i_sorted.tolist()):
            perm_full[jj] = int(ii)
            used_i.add(int(ii))

        # 남는 j/i 채워서 완전한 perm 구성
        remaining_j = [j for j in range(K) if j not in set(j_sorted.tolist())]
        remaining_i = [i for i in range(K) if i not in used_i]
        for j_idx, i_idx in zip(remaining_j, remaining_i):
            perm_full[j_idx] = int(i_idx)

        # 실제 적용: consumer 입력축 permute
        _apply_perm_inplace_input_channel(cons, perm_full)

        self._routed_once.add(done_key)
        return True

    @torch.no_grad()
    def route_many(self,
                   pairs: List[Tuple[str, str]],
                   mode: str = "desc",
                   strategy: str = "consumer") -> int:
        """
        여러 (producer, consumer) 페어에 대해 순차 적용.
        반환: 실제 permute를 적용한 페어 수
        """
        applied = 0
        for a, b in pairs:
            ok = self.route_pair(a, b, mode=mode, strategy=strategy)
            if ok:
                applied += 1
        return applied

# --------- Simple install & use (mask 기반) ---------
def install_channel_router(model: nn.Module,
                           stuck_map: Dict[str, torch.Tensor]) -> ChannelRouter:
    """
    ChannelRouter 인스턴스 생성. (탐지 없음: 외부에서 제공한 stuck_map만 사용)
    """
    return ChannelRouter(model, stuck_map)

def install_router_from_mask(model: nn.Module,
                             stuck_map: Dict[str, torch.Tensor],
                             arch: str = "mlp"
                             ) -> Tuple[ChannelRouter, List[Tuple[str, str]]]:

    router = install_channel_router(model, stuck_map)
    al = arch.lower()
    if al.startswith("mlp"):
        pairs = build_pairs_mlp(model)
    elif al.startswith("vgg"):
        pairs = build_pairs_vgg(model)
    elif al.startswith("resnet"):
        pairs = build_pairs_resnet(model)
    else:
        has_conv = any(isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) for m in model.modules())
        pairs = build_pairs_resnet(model) if has_conv else build_pairs_mlp(model)
    return router, pairs

@torch.no_grad()
def autoroute_with_mask(router: ChannelRouter,
                        pairs: List[Tuple[str, str]],
                        stuck_map: Dict[str, torch.Tensor],
                        mode: str = "desc",
                        strategy: str = "consumer") -> int:

    router.stuck_map = stuck_map

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return router.route_many(pairs, mode=mode, strategy=strategy)

# === Astrocyte ===
def _per_out_sum(W: torch.Tensor, mode: str = "pos") -> torch.Tensor:
    """
    W: [Cout, Cin, ...] or [Cout, Cin]
    return S: [Cout]  (PR 합 유사치)
    mode='pos' : ReLU로 PR 유사치(음수 억제)
    mode='abs' : |W|
    """
    if mode == "abs":
        V = W.abs()
    else:
        V = W.clamp(min=0)
    return V.view(W.shape[0], -1).sum(dim=1)  # [Cout]

class AstroRepairManager:
    """
    Astromorphic Self-Repair (gradient-friendly)
    - epoch0(or warmup) 종료 후 capture_baseline() 호출 → w0 스냅샷
    - stuck_map 등록 시 뉴런별 결함비 z를 w0에서 로컬하게 추정 → q 계산
    - loss_astro()로 L = λ * || W - q*W0 ||^2 (healthy만) 반환
      (논문 q≈1/z 또는 1.03/(z+0.04) 선택)  :contentReference[oaicite:1]{index=1}
    """
    def __init__(self,
                 model: nn.Module,
                 include: Tuple[type,...]=_SUPPORTED,
                 sum_mode: str = "pos",         # 'pos' or 'abs'
                 q_mode: str = "inverse",       # 'inverse' or 'empirical'
                 q_clip: float = 4.0,           # q 상한(과보정 방지)
                 mask_missing_as_healthy: bool = True):
        self.model = model
        self.include = include
        self.sum_mode = sum_mode
        assert q_mode in ("inverse", "empirical")
        self.q_mode = q_mode
        self.q_clip = q_clip
        self.mask_missing_as_healthy = mask_missing_as_healthy

        # 레이어별 버퍼
        self._w0: Dict[str, torch.Tensor] = {}
        self._stuck: Dict[str, torch.Tensor] = {}   # Bool same-shape (True=stuck/frozen)
        self._active = False

    def set_stuck_map(self, stuck_map: Dict[str, torch.Tensor]):
        """stuck_map[name] = Bool(weight.shape)  (True=완전 고정 시냅스)"""
        self._stuck = stuck_map or {}

    @torch.no_grad()
    def capture_baseline(self):
        """현재 가중치를 baseline(w0)으로 스냅샷."""
        self._w0.clear()
        for name, m in self.model.named_modules():
            if _is_supported(m) and m.weight is not None:
                self._w0[name] = m.weight.detach().to("cpu").clone()

    def activate(self, flag: bool=True):
        self._active = bool(flag)

    def _compute_q_per_neuron(self, name: str, W0: torch.Tensor, device) -> torch.Tensor:

        dev_w0 = W0.device
        dtype = W0.dtype
        Cout = W0.shape[0]

        # 1) stuck mask를 W0 디바이스로 강제 이동 + bool化
        if name in self._stuck:
            M = self._stuck[name]
            if M.dtype is not torch.bool:
                M = (M != 0)
            M = M.to(device=dev_w0, dtype=torch.bool)
            assert M.shape == W0.shape, f"[Astro] stuck_map shape mismatch at {name}: {M.shape} vs {W0.shape}"
            ok_mask = ~M
        else:
            if getattr(self, "mask_missing_as_healthy", True):
                ok_mask = torch.ones_like(W0, dtype=torch.bool, device=dev_w0)
            else:
                # stuck 정보 없으면 q=1 그대로 (출력 디바이스로 반환)
                return torch.ones(Cout, device=device, dtype=dtype)

        # 2) 모든 합/마스킹은 W0의 디바이스에서 수행
        S_all = _per_out_sum(W0, mode=self.sum_mode) + 1e-12
        S_ok = _per_out_sum(W0.masked_fill(~ok_mask, 0), mode=self.sum_mode) + 1e-12

        # 3) z, q 계산 (W0 디바이스)
        z = (S_ok / S_all).clamp(min=1e-3)
        if self.q_mode == "empirical":
            q = (1.03 / (z + 0.04))
        else:
            q = 1.0 / z
        q = q.clamp(max=self.q_clip).to(dtype=dtype)

        # 4) 최종적으로 호출자가 원하는 디바이스로만 이동
        return q.to(device=device)
    def loss_astro(self,
                   lam: float = 1e-3,
                   reduction: str = "mean",
                   relative: bool = True,
                   per_layer_normalize: bool = True,
                   eps: float = 1e-6) -> torch.Tensor:
        """
        L_astro (정규화 버전):
          - relative=True  : ((W - qW0)^2) / (W0^2 + eps)
          - per_layer_normalize=True : 건강 위치 개수로 평균
          - reduction='mean' : 레이어 평균
        """
        if not self._active or not self._w0:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        losses = []
        for name, m in self.model.named_modules():
            if not _is_supported(m) or name not in self._w0:
                continue

            W = m.weight
            W0 = self._w0[name].to(W.device, W.dtype)

            q_vec = self._compute_q_per_neuron(name, W0, W.device)  # [Cout]
            view = [W.shape[0]] + [1] * (W.dim() - 1)
            Wtgt = (q_vec.view(*view) * W0)

            # healthy mask (고장 제외)
            if name in self._stuck:
                mask = ~self._stuck[name].to(W.device, dtype=torch.bool)
            else:
                mask = torch.ones_like(W, dtype=torch.bool, device=W.device)

            diff2 = (W - Wtgt).pow(2)
            if relative:
                denom = (W0.pow(2) + eps)
                diff2 = diff2 / denom

            diff2 = diff2.masked_fill(~mask, 0.0)

            if per_layer_normalize:
                denom_count = mask.sum().clamp(min=1).to(diff2.dtype)
                l = diff2.sum() / denom_count
            else:
                l = diff2.sum()

            losses.append(l)

        if not losses:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        total = torch.stack(losses).mean() if reduction == "mean" else torch.stack(losses).sum()
        return lam * total

# --- 재사용: 지원 레이어 판별 ---
def _is_supported(m: nn.Module) -> bool:
    return isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)) and hasattr(m, "weight")

# --- 1) 스파이크/활성 레이트 트래커 + 항상성 손실 ---
def attach_spike_rate_tracker(model: nn.Module, beta: float = 0.9):
    handles = []
    def _hook(mod: nn.Module, _inp, out):
        if not torch.is_tensor(out) or out.dim() < 2:
            return
        dims = tuple(d for d in range(out.dim()) if d != 1)
        with torch.no_grad():
            r = out.detach().abs().mean(dim=dims).to(dtype=torch.float32)
            if not hasattr(mod, "_rate_ema"):
                mod._rate_ema = r.clone(); mod._rate_beta = float(beta)
            else:
                b = float(getattr(mod, "_rate_beta", beta))
                mod._rate_ema.mul_(b).add_(r, alpha=(1.0 - b))
    for m in model.modules():
        if _is_supported(m):
            handles.append(m.register_forward_hook(_hook))

    return handles

def capture_rate_baseline(model: nn.Module):
    for m in model.modules():
        if hasattr(m, "_rate_ema"): m._rate_ref = m._rate_ema.detach().clone()

def loss_rate_homeostasis(model: nn.Module, lam: float = 1e-4, eps: float = 1e-6):
    dev = next(model.parameters()).device
    losses = []
    for m in model.modules():
        if hasattr(m, "_rate_ema") and hasattr(m, "_rate_ref"):
            r  = m._rate_ema.to(dev); r0 = m._rate_ref.to(dev)
            losses.append(((r - r0) ** 2 / (r0 ** 2 + eps)).mean())
    if not losses: return torch.tensor(0.0, device=dev)
    return lam * torch.stack(losses).mean()

# --- 2) Astro 게이트(뉴런별 출력 스케일 파라미터 g) ---
class AstroGateManager:
    def __init__(self, model: nn.Module, q_clip: Tuple[float, float]=(0.5, 2.0), q_mode: str="empirical"):
        self.model = model
        self.q: Dict[str, torch.Tensor] = {}
        self._handles: List = []
        self.q_clip = q_clip
        self.q_mode = q_mode

    def set_stuck_map(self, stuck_map: Dict[str, torch.Tensor], eps: float = 0.04, a: float = 1.03):
        self.q = {}
        nm = dict(self.model.named_modules())
        for name, mask in (stuck_map or {}).items():
            if name not in nm or not _is_supported(nm[name]): continue
            H = 1.0 - mask.view(mask.shape[0], -1).float().mean(dim=1)  # [Cout]
            z = (1.0 - H).clamp(0.0, 1.0)
            if self.q_mode == "inverse":
                q = 1.0 / (z + 1e-6)
            else:
                q = a / (z + eps)
            self.q[name] = q.clamp(self.q_clip[0], self.q_clip[1])

    def attach(self):
        for name, m in self.model.named_modules():
            if not _is_supported(m) or hasattr(m, "_astro_gate"): continue
            g = nn.Parameter(torch.ones(m.weight.shape[0], dtype=m.weight.dtype, device=m.weight.device))
            m.register_parameter("_astro_gate", g)
            def _hook(mod: nn.Module, _inp, out):
                g = mod._astro_gate
                if out.dim() == 2:
                    return out * g.view(1, -1)
                shape = [1, -1] + [1]*(out.dim()-2)
                return out * g.view(*shape)
            self._handles.append(m.register_forward_hook(_hook))

    def loss_gate(self, lam: float = 5e-4):
        dev = next(self.model.parameters()).device
        terms = []
        for name, m in self.model.named_modules():
            if hasattr(m, "_astro_gate") and name in self.q:
                q = self.q[name].to(device=dev, dtype=m._astro_gate.dtype)
                terms.append((m._astro_gate - q).pow(2).mean())
        if not terms: return torch.tensor(0.0, device=dev)
        return lam * torch.stack(terms).mean()

# --- 3) Astro(가중치) 정규화 손실(astro-lite) ---
def loss_astro_lite(
        astro_mgr,
        lam: float = 1e-4,
        eps: float = 1e-6,
        chunk_out: int = 32,              # 더 작게(기본 32) → 피크↓
        move_w0_to_cpu: bool = True,
        use_amp: bool = True,             # astro만 AMP 적용
        amp_dtype: str = "bf16",
        use_checkpoint: bool = True,      # 청크별 재계산로 메모리↓
        layer_stride: int = 1,            # 레이어 서브샘플링
        layer_phase: int = 0,             # (idx % stride == phase)일 때만 계산
        scale_for_stride: bool = True,    # 기대값 보존 스케일링
):
    """
    메모리 친화형 astro loss:
      - Cout 청크 + (선택) 체크포인트 → 저장 activation 최소화
      - AMP(bf16/fp16)로 astro 부분만 저정밀
      - 레이어 서브샘플링으로 per-step 부담 분산 (기대값 보존 스케일링 옵션)
    """
    model = astro_mgr.model
    device = next(model.parameters()).device
    if not getattr(astro_mgr, "_active", False) or not getattr(astro_mgr, "_w0", {}):
        return torch.tensor(0.0, device=device)

    # amp 설정
    if use_amp and torch.cuda.is_available():
        dtype = torch.bfloat16 if amp_dtype.lower() == "bf16" else torch.float16
    else:
        dtype = None

    layer_losses = []
    layer_count_used = 0

    def _chunk_num_fn(Wc, qc, W0c, mask_f):
        # 청크 분자(healthy 가중 평균 제곱상대오차의 "합")만 반환 → 합쳐서 /den_total
        view = [Wc.shape[0]] + [1] * (Wc.dim() - 1)
        Wtgt = qc.view(*view) * W0c
        diff = Wc - Wtgt
        # 마스크를 float로 곱해 선택( masked_select 대신 ) → 별도 텐서 복사 최소화
        denom = (W0c * W0c + eps)
        val = ((diff * diff) / denom) * mask_f
        return val.sum()

    for li, (name, m) in enumerate(model.named_modules()):
        if not (_is_supported(m) and name in astro_mgr._w0):
            continue
        if layer_stride > 1 and (li % layer_stride) != layer_phase:
            continue  # 이번 스텝은 이 레이어 스킵

        W = m.weight                               # GPU 파라미터 (grad 필요)
        Cout = W.shape[0]

        # W0: 가능하면 CPU에 두고 청크만 GPU로
        W0 = astro_mgr._w0[name]
        if move_w0_to_cpu and W0.is_cuda:
            with torch.no_grad():
                astro_mgr._w0[name] = W0.detach().cpu()
            W0 = astro_mgr._w0[name]

        # q(뉴런 스케일) — 결과는 GPU로만 받음(내부 계산은 W0 디바이스)
        q_full = astro_mgr._compute_q_per_neuron(name, W0, W.device).to(device=W.device, dtype=W.dtype)

        # healthy mask (GPU)
        if name in getattr(astro_mgr, "_stuck", {}):
            healthy_full = (~astro_mgr._stuck[name]).to(W.device, dtype=torch.bool)
        else:
            healthy_full = torch.ones_like(W, dtype=torch.bool, device=W.device)

        # 총 분모(건강 위치 개수): grad 불필요
        den_total = healthy_full.sum().clamp(min=1).to(dtype=W.dtype)

        num_total = W.new_zeros((), dtype=W.dtype)  # grad 유지 스칼라

        step = max(1, min(int(chunk_out), Cout))
        for i0 in range(0, Cout, step):
            i1 = min(Cout, i0 + step); s = slice(i0, i1)

            # 준비: 청크 텐서
            Wc   = W[s]                                        # requires_grad=True
            W0c  = W0[s].to(device=W.device, dtype=W.dtype, non_blocking=True)
            qc   = q_full[s]
            mask = healthy_full[s]
            mask_f = mask.to(dtype=W.dtype)

            # AMP 컨텍스트 + (선택) 체크포인트
            ctx = torch.amp.autocast(enabled=(dtype is not None), dtype=dtype, device_type="cuda")
            with ctx:
                if use_checkpoint and Wc.requires_grad:
                    # checkpoint는 분자 합(스칼라)만 그래프에 남김 → 메모리↓
                    num_chunk = ckpt.checkpoint(
                        _chunk_num_fn, Wc, qc, W0c, mask_f,
                        use_reentrant=False
                    )
                else:
                    num_chunk = _chunk_num_fn(Wc, qc, W0c, mask_f)

            num_total = num_total + num_chunk

            # 임시 텐서 정리
            del W0c, qc, mask, mask_f, num_chunk

        layer_loss = num_total / den_total
        layer_losses.append(layer_loss)
        layer_count_used += 1

    if layer_count_used == 0:
        return torch.tensor(0.0, device=device)

    total = torch.stack(layer_losses).mean()

    # 레이어 서브샘플링 보정(기대값 유지)
    if scale_for_stride and layer_stride > 1:
        total = total * layer_stride

    return lam * total

# --- 4) 원터치 오토 매니저 ---
def _astro_supported(m: nn.Module) -> bool:
    return isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)) and hasattr(m, "weight")

def _q_from_mask(mask: torch.Tensor, mode: str="empirical",
                 clip: float=2.0, eps: float=0.04, a: float=1.03) -> torch.Tensor:
    # mask: True=stuck, shape=[Cout, Cin, ...] -> q per out-channel
    H = 1.0 - mask.view(mask.shape[0], -1).float().mean(dim=1)  # [Cout]
    z = (1.0 - H).clamp(0.0, 1.0)
    if mode == "inverse":
        q = 1.0 / (z + 1e-6)
    else:
        q = a / (z + eps)
    return q.clamp(1.0/clip, clip)

class _AstroGradHook:
    """
    손실 그래프 없이 grad hook으로만 astro 정규화를 주입:
      L ~ mean( ((W - q*W0)^2) / (W0^2 + eps) )  ==> dL/dW = 2*(W - qW0)/(W0^2+eps)/N
    - W0, q, inv_denom, healthy 마스크는 CPU에 저장 → VRAM 추가 0
    - 매 backward마다 weight.grad에 청크 단위로 in-place 더함 → OOM 차단
    - astro(epoch)는 램프만 업데이트하고 항상 0.0 반환(메인 수정 불필요)
    """
    def __init__(self, model: nn.Module, fault_mgr=None, start_epoch: int=0,
                 lam: float=1e-4, ramp_epochs: int=3, q_mode: str="empirical",
                 q_clip: float=2.0, eps: float=1e-6, chunk_out: int=32):
        self.model = model
        self.fault_mgr = fault_mgr
        self.start_epoch = int(start_epoch)
        self.lam_base = float(lam)
        self.ramp_epochs = int(max(0, ramp_epochs))
        self.q_mode = q_mode
        self.q_clip = float(q_clip)
        self.eps = float(eps)
        self.chunk_out = int(max(1, chunk_out))

        self._armed = False
        self._handles = []
        # per-layer CPU caches
        self.W0: Dict[str, torch.Tensor] = {}
        self.q:  Dict[str, torch.Tensor] = {}
        self.inv_denom: Dict[str, torch.Tensor] = {}
        self.healthy: Dict[str, torch.Tensor] = {}
        self.norm: Dict[str, float] = {}
        self._lam_now = 0.0

    def _get_fault_map(self) -> Dict[str, torch.Tensor]:
        try:
            from fault_injection import get_fault_map
            if self.fault_mgr is not None:
                m = get_fault_map(self.fault_mgr, include_bias=False)
                if m is not None:
                    return {k: v.detach().to("cpu") for k, v in m.items()}  # <-- CPU 강제
            return {}
        except Exception:
            pass
        return {}

    def _capture_W0_q(self):
        stuck_map = self._get_fault_map()
        for name, m in self.model.named_modules():
            if not _astro_supported(m):
                continue
            W = m.weight.detach().to("cpu")
            self.W0[name] = W.clone()
            if name in stuck_map:
                q = _q_from_mask(stuck_map[name], mode=self.q_mode, clip=self.q_clip).to(dtype=W.dtype)
            else:
                q = torch.ones(W.shape[0], dtype=W.dtype)
            self.q[name] = q  # CPU

            denom = (self.W0[name] * self.W0[name]) + self.eps
            self.inv_denom[name] = denom.reciprocal()             # CPU
            if name in stuck_map:
                healthy = (~stuck_map[name]).to(dtype=torch.bool).to("cpu")
            else:
                healthy = torch.ones_like(self.W0[name], dtype=torch.bool, device="cpu")
            self.healthy[name] = healthy
            N = int(healthy.sum().item())
            self.norm[name] = 1.0 / max(1, N)

    def _hook_factory(self, name: str, mod: nn.Module):
        # grad: [Cout, Cin, ...] on device
        def _hook(grad: torch.Tensor):
            lam = self._lam_now
            if lam <= 0.0:
                return grad
            with torch.no_grad():
                W = mod.weight
                Cout = W.shape[0]
                step = min(self.chunk_out, Cout)

                q_cpu  = self.q[name]         # [Cout]    CPU
                W0_cpu = self.W0[name]        # [Cout,...]CPU
                inv_cpu= self.inv_denom[name] #           CPU
                H_cpu  = self.healthy[name]   # bool      CPU
                scale = 2.0 * lam * self.norm[name]

                for i0 in range(0, Cout, step):
                    i1 = min(Cout, i0 + step); s = slice(i0, i1)
                    # 필요한 청크만 GPU로
                    q  = q_cpu[s].to(device=W.device, dtype=W.dtype, non_blocking=True)
                    W0 = W0_cpu[s].to(device=W.device, dtype=W.dtype, non_blocking=True)
                    inv= inv_cpu[s].to(device=W.device, dtype=W.dtype, non_blocking=True)
                    H  = H_cpu[s].to(device=W.device, dtype=torch.bool, non_blocking=True)

                    view = [W0.shape[0]] + [1]*(W0.dim()-1)
                    target = q.view(*view) * W0
                    delta = (W[s] - target) * inv
                    delta.masked_fill_(~H, 0)

                    grad[s].add_(scale * delta)

                    del q, W0, inv, H, target, delta
            return grad
        return _hook

    def _attach_hooks(self):
        for name, m in self.model.named_modules():
            if not _astro_supported(m):
                continue
            h = m.weight.register_hook(self._hook_factory(name, m))
            self._handles.append(h)

    def _bootstrap(self):
        if self._armed:
            return
        self._capture_W0_q()
        self._attach_hooks()
        self._armed = True

    def __call__(self, epoch: int) -> torch.Tensor:
        # lazy init at start_epoch
        if (not self._armed) and (int(epoch) >= self.start_epoch):
            self._bootstrap()

        # ramp λ
        if self.ramp_epochs > 0:
            r = max(0.0, min(1.0, (int(epoch) - self.start_epoch) / float(self.ramp_epochs)))
        else:
            r = 1.0
        self._lam_now = self.lam_base * r

        # zero loss (정규화는 grad hook에서 주입)
        dev = next(self.model.parameters()).device
        return torch.tensor(0.0, device=dev)

# drop-in replacement: 메인 코드를 바꾸지 않고 그대로 사용 가능
def install_astro_auto(model: nn.Module,
                       fault_mgr=None,
                       start_epoch: int = 0,
                       lam_astro: float = 1e-4,
                       ramp_epochs: int = 3,
                       q_mode: str = "empirical",
                       q_clip_weight: float = 2.0,
                       chunk_out: int = 32):
    """
    사용법 (메인 그대로):
        from benchmarks import install_astro_auto
        astro = install_astro_auto(net, fault_mgr, start_epoch=fault_start_epoch)
        ...
        loss = main_loss + astro(epoch)   # astro(epoch) == 0 (grad hook이 정규화 주입)
    """
    return _AstroGradHook(model, fault_mgr, start_epoch,
                          lam=lam_astro, ramp_epochs=ramp_epochs,
                          q_mode=q_mode, q_clip=q_clip_weight,
                          chunk_out=chunk_out)

# === FalVolt ===
from dataclasses import dataclass

@dataclass
class FalVoltConfig:
    start_epoch: int = 0                  # begin learning V_l at / after this epoch
    clamp: tuple[float, float] = (0.2, 3.0)  # valid range of thresholds
    include_bias: bool = False            # usually thresholds are per-neuron; bias zeroing not needed
    verbose: bool = True                  # print once when activated

class FalVoltAuto:
    def __init__(self, model: nn.Module, fault_mgr, cfg: FalVoltConfig):
        self.model = model
        self.fault_mgr = fault_mgr
        self.cfg = cfg
        self._activated = False           # whether thresholds have been made trainable
        self._last_epoch_zeroed = -1

        # resolve spiking classes lazily to avoid import-time dependency
        from spikingjelly.activation_based import neuron as sj_neuron
        self._spike_classes = (
            sj_neuron.LIFNode,
        )
        # Safe keep for restore or logging
        self._original_vths: dict[int, float] = {}

    # ---------- public API (to mirror Soft/Astro one‑touch style) ----------
    def __call__(self, epoch: int) -> torch.Tensor:
        """Return 0.0 but perform one‑time activation at cfg.start_epoch."""
        if (not self._activated) and (epoch >= self.cfg.start_epoch):
            self._activate_trainable_thresholds()
            # if self.cfg.verbose:
            #     print(f"[FalVolt] Activated trainable thresholds at epoch {epoch}. "
            #           f"Trainable params += {self._count_vth_params()}")
        # optionally do a once‑per‑epoch zeroing at the *beginning* of the epoch
        if epoch != self._last_epoch_zeroed and epoch >= self.cfg.start_epoch:
            # zeroing at epoch begin is equivalent to zeroing at previous epoch end in practice,
            # since an optimizer.step() must have happened previously. We also re‑zero every epoch
            # to keep faulty weights at zero.
            self._zero_fault_mapped_weights()
            self._last_epoch_zeroed = epoch
        # return a scalar Tensor to keep "+" semantics in training loss
        return torch.tensor(0.0, device=next(self.model.parameters()).device)

    def post_step(self):
        """Call this right after optimizer.step(): clamps trainable thresholds and re‑zeroes pruned weights."""
        # Clamp trainable thresholds
        lo, hi = self.cfg.clamp
        for m in self._iter_spiking_modules():
            vth = getattr(m, "v_threshold", None)
            if isinstance(vth, torch.nn.Parameter):
                vth.data.clamp_(lo, hi)
        # Re‑zero fault‑mapped weights so any optimizer update does not resurrect them
        self._zero_fault_mapped_weights()

    def on_epoch_end(self):
        """Optional: enforce Algorithm‑1 line 13 zeroing at the end of each retraining epoch."""
        self._zero_fault_mapped_weights()

    # ---------- internals ----------
    def _iter_spiking_modules(self):
        for m in self.model.modules():
            if isinstance(m, self._spike_classes):
                yield m

    def _activate_trainable_thresholds(self):
        """Convert LIFNode.v_threshold to nn.Parameter so it participates in autograd."""
        device = next(self.model.parameters()).device
        dtype  = next(self.model.parameters()).dtype
        for m in self._iter_spiking_modules():
            vth = getattr(m, "v_threshold", 1.0)
            if isinstance(vth, torch.nn.Parameter):
                continue  # already converted
            # Remember original for debugging
            try:
                self._original_vths[id(m)] = float(vth)
            except Exception:
                self._original_vths[id(m)] = float(torch.as_tensor(vth).item())
            # Convert
            p = torch.nn.Parameter(torch.tensor(float(vth), device=device, dtype=dtype), requires_grad=True)
            setattr(m, "v_threshold", p)

    def _count_vth_params(self) -> int:
        n = 0
        for m in self._iter_spiking_modules():
            if isinstance(getattr(m, "v_threshold", None), torch.nn.Parameter):
                n += 1
        return n

    @torch.no_grad()
    def _zero_fault_mapped_weights(self):
        """Set weights mapped to faulty PEs to ZERO (Alg. 1 lines 2 and 13)."""
        if self.fault_mgr is None:
            return
        try:
            from fault_injection import get_fault_map
        except Exception as e:
            print(f"[FalVolt] WARNING: fault_injection.get_fault_map import failed ({e}); skipping zeroing.")
            return
        stuck_map = get_fault_map(self.fault_mgr, include_bias=self.cfg.include_bias)
        # stick to modules that have weights
        name2mod = dict(self.model.named_modules())
        for name, mask in stuck_map.items():
            m = name2mod.get(name, None)
            if m is None or getattr(m, "weight", None) is None:
                continue
            W = m.weight.data
            mask_t = mask.to(device=W.device)
            # two cases: shape‑matched tensor mask OR channel‑wise mask
            if mask_t.shape == W.shape:
                W[mask_t] = 0
            else:
                # try to broadcast along out_channels dimension
                # Conv: [Cout, Cin, kH, kW]; Linear: [Cout, Cin]
                if mask_t.dim() == 1 and mask_t.numel() == W.shape[0]:
                    view = [W.shape[0]] + [1] * (W.dim() - 1)
                    W[mask_t.view(*view).expand_as(W)] = 0
                else:
                    # fallback: if incompatible, log once
                    print(f"[FalVolt] WARNING: mask shape {tuple(mask_t.shape)} not compatible with "
                          f"weight {name} shape {tuple(W.shape)}; skipping.")
        # Also zero bias if requested and available
        if self.cfg.include_bias:
            for name, mask in stuck_map.items():
                m = name2mod.get(name, None)
                if m is None or getattr(m, "bias", None) is None:
                    continue
                b = m.bias.data
                mask_t = mask.to(device=b.device)
                if mask_t.shape == b.shape:
                    b[mask_t] = 0

def install_falvolt_auto(model: nn.Module,
                         fault_mgr,
                         start_epoch: int = 0,
                         clamp: tuple[float, float] = (0.2, 3.0),
                         include_bias: bool = False,
                         verbose: bool = True) -> FalVoltAuto:
    """
    One‑touch FalVolt installer.
    - Converts LIF thresholds into trainable nn.Parameters and starts learning them at `start_epoch`.
    - Enforces 'prune-by-fault' zeroing after each optimizer step and once per epoch.
    Returns a controller that you should call per‑batch (like Astro) and after each optimizer step.
    """
    cfg = FalVoltConfig(start_epoch=start_epoch, clamp=clamp, include_bias=include_bias, verbose=verbose)
    return FalVoltAuto(model, fault_mgr, cfg)

@dataclass
class LIFAConfig:
    start_epoch: int = 0           # LIFA 시작 에폭 (fault 주입 직후 추천)
    lam: float = 1e-4              # astro-style 정규화 강도
    ramp_epochs: int = 3           # 램프업 기간
    tau_n: float = 4.0             # 뉴런 활동(입력 슬롯) 시정수
    tau_g: float = 8.0             # Ca2+ (astro) 시정수
    tau_p: float = 16.0            # 글리오전달물질 저장고 시정수
    fault_inject: float = 1.0      # (1-H) 주입 계수 (고장 심할수록 더 세게 보정)
    ema_beta: float = 0.9          # slot activity EMA(attach_slot_activity_tracker용)
    q_clip: tuple[float, float] = (0.25, 4.0)  # q 동적 범위
    arch: str = "resnet"           # 'mlp' | 'vgg' | 'resnet'
    chunk_out: int = 32            # grad hook 청크 크기 (메모리 세이브)

class LIFAAuto:
    """
    LIFA = 활동(consumer 입력 채널) → vN ↘︎ → vG ↘︎ → g  (leaky integrator 3단)
         + 고장도(1-H) 주입 → 동적 게인 q_dyn
         → astro-style grad hook: (W - q*W0) / (W0^2 + eps) 를 grad에 주입
    - q = q_mask(정적, 고장 마스크 기반) * q_dyn(동적, 활동/글리오 기반)
    - W0, inv_denom, healthy, q_mask, 상태(vN, vG, g)는 CPU에 둬서 VRAM 추가 0
    - __call__(epoch)은 0.0을 반환(메인 학습 로직 손대지 않음)
    """
    def __init__(self, model: nn.Module, fault_mgr=None, cfg: LIFAConfig = LIFAConfig()):
        self.model = model
        self.fault_mgr = fault_mgr
        self.cfg = cfg

        self._armed = False
        self._handles: list = []
        self._slot_handles: list = []
        self._lam_now = 0.0

        # per-layer CPU 캐시
        self.W0: Dict[str, torch.Tensor] = {}
        self.inv_denom: Dict[str, torch.Tensor] = {}
        self.healthy: Dict[str, torch.Tensor] = {}
        self.q_mask: Dict[str, torch.Tensor] = {}
        self.q_dyn: Dict[str, torch.Tensor] = {}
        self.q_now: Dict[str, torch.Tensor] = {}
        self.norm: Dict[str, float] = {}

        # 동적 상태 (producer의 out-채널 기준)
        self.state_vN: Dict[str, torch.Tensor] = {}
        self.state_vG: Dict[str, torch.Tensor] = {}
        self.state_g:  Dict[str, torch.Tensor] = {}

        # producer → [consumer modules] 매핑
        self._consumers_of: Dict[str, list[nn.Module]] = {}

    # ---------- 내부 유틸 ----------
    def _get_fault_map(self) -> Dict[str, torch.Tensor]:
        try:
            from fault_injection import get_fault_map
            if self.fault_mgr is not None:
                m = get_fault_map(self.fault_mgr, include_bias=False)
                return m if m is not None else {}
        except Exception:
            pass
        return {}

    def _build_pairs_and_consumers(self):
        arch = (self.cfg.arch or "resnet").lower()
        if arch.startswith("mlp"):
            pairs = build_pairs_mlp(self.model)
        elif arch.startswith("vgg"):
            pairs = build_pairs_vgg(self.model)
        else:
            pairs = build_pairs_resnet(self.model)

        nm = dict(self.model.named_modules())
        consumers_of: Dict[str, list[nn.Module]] = {}
        for prod_name, cons_name in pairs:
            if prod_name not in nm or cons_name not in nm:
                continue
            consumers_of.setdefault(prod_name, []).append(nm[cons_name])
        self._consumers_of = consumers_of

    def _capture_W0_and_masks(self):
        stuck_map = self._get_fault_map()
        for name, m in self.model.named_modules():
            if not _astro_supported(m) or getattr(m, "weight", None) is None:
                continue

            W = m.weight.detach().to("cpu")
            self.W0[name] = W.clone()

            # q/healthy는 CPU 고정
            if name in stuck_map:
                msk = stuck_map[name].detach().to("cpu")
                q = _q_from_mask(msk, mode="empirical", clip=max(self.cfg.q_clip)).to(
                    dtype=torch.float32, device="cpu"
                )
                healthy = (~msk).to(dtype=torch.bool, device="cpu")
            else:
                q = torch.ones(W.shape[0], dtype=torch.float32, device="cpu")
                healthy = torch.ones_like(W, dtype=torch.bool, device="cpu")

            self.q_mask[name] = q
            self.healthy[name] = healthy

            denom = (W * W) + 1e-6
            self.inv_denom[name] = denom.reciprocal()

            N = int(healthy.sum().item())
            self.norm[name] = 1.0 / max(1, N)

            Cout = W.shape[0]
            self.state_vN[name] = torch.zeros(Cout, dtype=torch.float32, device="cpu")
            self.state_vG[name] = torch.zeros(Cout, dtype=torch.float32, device="cpu")
            self.state_g[name] = torch.ones(Cout, dtype=torch.float32, device="cpu")
            self.q_dyn[name] = torch.ones(Cout, dtype=torch.float32, device="cpu")
            self.q_now[name] = (self.q_mask[name] * self.q_dyn[name]).clamp(*self.cfg.q_clip)

        # ---- 보정: 훅 달렸는데 캐시 누락된 레이어가 있으면 기본값으로 메움
        for name, m in self.model.named_modules():
            if not _astro_supported(m) or getattr(m, "weight", None) is None:
                continue
            if name not in self.norm:
                W = m.weight.detach().to("cpu")
                Cout = W.shape[0]
                self.W0.setdefault(name, W.clone())
                self.inv_denom.setdefault(name, ((W * W) + 1e-6).reciprocal())
                self.healthy.setdefault(name, torch.ones_like(W, dtype=torch.bool))
                self.q_mask.setdefault(name, torch.ones(Cout, dtype=torch.float32))
                self.q_dyn.setdefault(name, torch.ones(Cout, dtype=torch.float32))
                self.q_now.setdefault(name, torch.ones(Cout, dtype=torch.float32))
                self.norm[name] = 1.0

    def _attach_grad_hooks(self):
        # 기존 훅들 제거(중복 방지)
        for h in getattr(self, "_handles", []):
            try:
                h.remove()
            except:
                pass
        self._handles = []

        def _hook_factory(name: str, m: nn.Module):
            step = int(max(1, self.cfg.chunk_out))

            def _hook(grad: torch.Tensor):
                # 캐시 누락이면 아무것도 안 함(안전 가드)
                if (name not in self.W0) or (name not in self.inv_denom) or (name not in self.healthy):
                    return grad

                W = m.weight
                Cout = W.shape[0]

                # lam은 스칼라로 고정 (lambda 지움)
                lam_scalar = 2.0 * float(self._lam_now) * float(self.norm.get(name, 1.0))

                W0_cpu = self.W0[name]
                inv_cpu = self.inv_denom[name]
                H_cpu = self.healthy[name]
                # q가 없으면 전부 1로
                q_cpu = self.q_now.get(
                    name, torch.ones(Cout, dtype=W0_cpu.dtype, device=W0_cpu.device)
                )

                for i0 in range(0, Cout, step):
                    i1 = min(Cout, i0 + step);
                    s = slice(i0, i1)
                    W0 = W0_cpu[s].to(device=W.device, dtype=W.dtype, non_blocking=True)
                    inv = inv_cpu[s].to(device=W.device, dtype=W.dtype, non_blocking=True)
                    H = H_cpu[s].to(device=W.device, dtype=torch.bool, non_blocking=True)
                    q = q_cpu[s].to(device=W.device, dtype=W.dtype, non_blocking=True)

                    view = [W0.shape[0]] + [1] * (W0.dim() - 1)
                    target = q.view(*view) * W0
                    delta = (W[s] - target) * inv
                    delta.masked_fill_(~H, 0)

                    grad[s].add_(lam_scalar * delta)

                    del W0, inv, H, q, target, delta
                return grad

            return _hook

        for name, m in self.model.named_modules():
            if not _astro_supported(m) or getattr(m, "weight", None) is None:
                continue
            h = m.weight.register_hook(_hook_factory(name, m))
            self._handles.append(h)

    def _attach_slot_activity(self):
        # consumer 입력 채널 EMA 추적 (CPU 상태 갱신은 __call__에서)
        hs, _ = attach_slot_activity_tracker(self.model, beta=self.cfg.ema_beta)
        self._slot_handles = hs

    # 동적 상태 갱신: consumer 입력 채널 activity → producer out-채널로 역매핑
    @torch.no_grad()
    def _update_dynamic_q(self):
        if not self._consumers_of:
            return
        stuck_map = self._get_fault_map()
        for prod_name, consumers in self._consumers_of.items():
            # activity 수집 (여러 consumer 있으면 평균)
            acts = []
            for cons in consumers:
                a = getattr(cons, "_slot_activity", None)
                if a is None or a.numel() == 0:
                    continue
                acts.append(a.detach().to("cpu", dtype=torch.float32))
            if not acts:
                continue
            a = torch.stack(acts, dim=0).mean(dim=0)  # [Cin] ~= producer Cout
            # 안전 길이 맞춤
            Cout = int(self.W0[prod_name].shape[0])
            K = min(Cout, int(a.numel()))
            if K == 0:
                continue
            a = a[:K]
            # 정규화된 활동 (평균 1.0 근처)
            a_norm = a / (a.mean() + 1e-6)

            # 건강도 H (per-out 채널, 0~1) → (1-H) 주입
            if prod_name in stuck_map:
                H = 1.0 - stuck_map[prod_name].view(Cout, -1).float().mean(dim=1).to("cpu")
            else:
                H = torch.ones(Cout, dtype=torch.float32)
            H = H[:K]
            inj = self.cfg.fault_inject * (1.0 - H).clamp(0.0, 1.0)

            # 3-stage leaky integration (Euler/EMA)
            def ema(x_prev, x_in, tau):
                alpha = float(max(1e-3, 1.0 / tau))
                return (1.0 - alpha) * x_prev + alpha * x_in

            vN = self.state_vN[prod_name][:K]
            vG = self.state_vG[prod_name][:K]
            g  = self.state_g [prod_name][:K]

            vN = ema(vN, a_norm, self.cfg.tau_n)
            vG = ema(vG, vN,    self.cfg.tau_g)
            g  = ema(g,  vG + inj, self.cfg.tau_p)

            self.state_vN[prod_name][:K].copy_(vN)
            self.state_vG[prod_name][:K].copy_(vG)
            self.state_g [prod_name][:K].copy_(g)

            # q_dyn: g를 평균 1.0으로 센터링한 스케일
            g_norm = g / (float(g.mean().item()) + 1e-6)
            q_dyn = g_norm.clamp(self.cfg.q_clip[0], self.cfg.q_clip[1])

            # 최종 q = q_mask * q_dyn  (정적･동적 결합)
            q = self.q_mask[prod_name].to(torch.float32).clone()
            q[:K].mul_(q_dyn)
            q.clamp_(self.cfg.q_clip[0], self.cfg.q_clip[1])

            self.q_dyn[prod_name] = torch.cat([q_dyn, torch.ones_like(q)[K:]], dim=0) if K < q.numel() else q_dyn
            self.q_now[prod_name] = q

    # ---------- 수명 주기 ----------
    def _bootstrap(self):
        if self._armed:
            return
        self._build_pairs_and_consumers()
        self._capture_W0_and_masks()
        self._attach_grad_hooks()
        self._attach_slot_activity()
        self._armed = True

    def __call__(self, epoch: int) -> torch.Tensor:
        # lazy init
        if (not self._armed) and (int(epoch) >= int(self.cfg.start_epoch)):
            self._bootstrap()

        # 램프업
        if self.cfg.ramp_epochs > 0:
            r = max(0.0, min(1.0, (int(epoch) - int(self.cfg.start_epoch)) / float(self.cfg.ramp_epochs)))
        else:
            r = 1.0
        self._lam_now = float(self.cfg.lam) * r

        # 동적 q 갱신 (forward 끝난 직후, backward 전에 호출되도록 메인에서 lifa(epoch) 더함)
        if self._armed:
            self._update_dynamic_q()

        # 손실값은 0.0 (정규화는 grad hook으로 주입)
        dev = next(self.model.parameters()).device
        return torch.tensor(0.0, device=dev)

def install_lifa_auto(model: nn.Module,
                      fault_mgr=None,
                      start_epoch: int = 0,
                      lam_lifa: float = 1e-4,
                      ramp_epochs: int = 3,
                      arch: str = "mlp",
                      tau_n: float = 4.0, tau_g: float = 8.0, tau_p: float = 16.0,
                      fault_inject: float = 1.0,
                      ema_beta: float = 0.9,
                      q_clip: tuple[float,float] = (0.25, 4.0),
                      chunk_out: int = 32) -> LIFAAuto:
    """
        lifa = install_lifa_auto(net, fault_mgr, start_epoch=fault_start_epoch, arch="resnet")
        loss = main_loss + lifa(epoch)
    """
    cfg = LIFAConfig(
        start_epoch=start_epoch, lam=lam_lifa, ramp_epochs=ramp_epochs,
        arch=arch, tau_n=tau_n, tau_g=tau_g, tau_p=tau_p,
        fault_inject=fault_inject, ema_beta=ema_beta,
        q_clip=q_clip, chunk_out=chunk_out
    )
    return LIFAAuto(model, fault_mgr, cfg)