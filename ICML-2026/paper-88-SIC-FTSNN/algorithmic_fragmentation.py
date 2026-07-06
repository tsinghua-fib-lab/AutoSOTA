from __future__ import annotations
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F
import torch.nn as nn
import math

def _gauss2d_kernel(sigma: torch.Tensor, ks: int, *, device, dtype):
    # sigma: [...] (broadcastable)
    r = ks // 2
    y = torch.arange(-r, r+1, device=device, dtype=dtype).view(1, ks, 1)
    x = torch.arange(-r, r+1, device=device, dtype=dtype).view(1, 1, ks)
    r2 = y**2 + x**2                                # [1,ks,ks]
    sig2 = (sigma.view(-1, 1, 1) + 1e-12)**2        # [S,1,1]
    g = torch.exp(-0.5 * r2 / sig2)                 # [S,ks,ks]
    g = g / (g.sum(dim=(1,2), keepdim=True) + 1e-12)
    return g                                        # [S,ks,ks]

def _log2d_bank(sigmas: torch.Tensor, ks: int, *, device, dtype):
    # LoG ∝ ((r^2 - 2 σ^2)/σ^4) * exp(-r^2/(2σ^2)) ; mean-zero, L1-normalize
    r = ks // 2
    y = torch.arange(-r, r+1, device=device, dtype=dtype).view(1, ks, 1)
    x = torch.arange(-r, r+1, device=device, dtype=dtype).view(1, 1, ks)
    r2 = y**2 + x**2                                # [1,ks,ks]
    sig2 = (sigmas.view(-1,1,1) + 1e-12)**2         # [S,1,1]
    logk = ((r2 - 2*sig2) / (sig2**2)) * torch.exp(-0.5 * r2 / sig2)  # [S,ks,ks]
    logk = logk - logk.mean(dim=(1,2), keepdim=True)                   # zero-mean
    logk = logk / (logk.abs().sum(dim=(1,2), keepdim=True) + 1e-12)    # L1 norm
    return logk                                                        # [S,ks,ks]

def _sobel_mag(gray_bchw: torch.Tensor):
    # gray_bchw: [B,1,H,W]
    device, dtype = gray_bchw.device, gray_bchw.dtype
    sobx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]], device=device, dtype=dtype).view(1,1,3,3)
    soby = sobx.transpose(-1,-2)
    gx = F.conv2d(gray_bchw, sobx, padding=1)
    gy = F.conv2d(gray_bchw, soby, padding=1)
    return torch.sqrt(gx*gx + gy*gy + 1e-12)

def _local_var(gray_bchw: torch.Tensor, sigma: float, ks: int):
    # Gaussian window로 로컬 분산(텍스처 세기) 계산, 전부 conv2d
    device, dtype = gray_bchw.device, gray_bchw.dtype
    gk = _gauss2d_kernel(torch.tensor([sigma], device=device, dtype=dtype), ks, device=device, dtype=dtype)[0]
    gk = gk.view(1,1,ks,ks)
    mu  = F.conv2d(gray_bchw, gk, padding=ks//2)
    mu2 = F.conv2d(gray_bchw*gray_bchw, gk, padding=ks//2)
    var = (mu2 - mu*mu).clamp_min(0.0)
    return torch.sqrt(var + 1e-12)

def _percentile_norm(x: torch.Tensor, p: float = 0.95):
    q = torch.quantile(x.flatten(), p)
    return (x / (q + 1e-12)).clamp(0, 1)

def _first_module(model: nn.Module, types):
    for m in model.modules():
        if isinstance(m, types):
            return m
    return None

@torch.no_grad()
def _wmap_from_conv(conv: nn.Conv2d, H: int, W: int, device, dtype):
    Wabs = conv.weight.detach().abs()                            # [Cout,Cin,kh,kw]
    # 대충 입력과 같은 해상도로 역투영(보통 stride/pad 맞추면 H,W로 나옴)
    ones = torch.ones((1, Wabs.size(0), H, W), device=device, dtype=dtype)
    wmap = F.conv_transpose2d(ones, Wabs, stride=conv.stride,
                              padding=conv.padding, dilation=conv.dilation,
                              groups=conv.groups).sum(1, keepdim=True)     # [1,1,?,?]
    # 크기 안 맞으면 center-crop 또는 interpolate
    if wmap.shape[-2:] != (H, W):
        wmap = F.interpolate(wmap, size=(H, W), mode='bilinear', align_corners=False)
    return _percentile_norm(wmap, 0.95)                                    # [1,1,H,W]

@torch.no_grad()
def _wmap_from_linear(lin: nn.Linear, C: int, H: int, W: int, device, dtype):
    # 입력을 [C*H*W]로 바로 flatten한다고 가정한 안전 fallback
    # |W| 총합을 입력 차원별로 모아 [1,1,H,W]로 reshape
    Wabs = lin.weight.detach().abs()                       # [Fout, Fin]
    v = Wabs.sum(dim=0)                                    # [Fin]
    if v.numel() != C*H*W:
        # 입력이 flatten 전에 다른 stem을 거치면 정확 매핑 어려움 → saliency-only로 대체
        return None
    v = v.view(1, 1, C, H, W).sum(dim=2)                   # 채널 합 → [1,1,H,W]
    return _percentile_norm(v, 0.95)

def _saliency_map(model, images_bchw, layer):
    # ALWAYS enable grad here (works even under no_grad; not under inference_mode)

    with torch.enable_grad():
        x = images_bchw.clone().detach().requires_grad_(True)  # [B,1,H,W]
        if isinstance(layer, torch.nn.Conv2d):
            z = layer(x)
            J = z.abs().sum()
        elif isinstance(layer, torch.nn.Linear):
            x_flat = x.view(x.size(0), -1)  # 🔧 Linear이면 평탄화
            z = layer(x_flat)  # fc pre-act 기준의 J
            J = z.abs().sum()
        else:
            x_in = x.view(x.size(0), -1) if x.dim() == 4 else x  # 🔧 안전빵 평탄화
            y = model(x_in)
            if isinstance(y, (tuple, list)): y = y[0]
            J = y.float().abs().sum()
        g, = torch.autograd.grad(J, x, retain_graph=False, create_graph=False)
        S = g.norm(p=2, dim=1, keepdim=True)

    S = S.mean(dim=0, keepdim=True)

    return _percentile_norm(S, 0.95)            # [1,1,H,W] in [0,1]

def _auto_weakness_maps(model: nn.Module, images_bchw: torch.Tensor,
                        layer: nn.Module | None = None):
    """
    sens, wmap 자동 생성. layer 우선, 없으면 모델에서 Conv→Linear 순으로 탐색.
    Conv 없음 + Linear 매핑 불가면 wmap=None (saliency만 사용).
    """
    B, C, H, W = images_bchw.shape
    device, dtype = images_bchw.device, images_bchw.dtype

    # 레이어 자동 선택
    chosen = layer
    if chosen is None:
        chosen = _first_module(model, (nn.Conv2d,))
        if chosen is None:
            chosen = _first_module(model, (nn.Linear,))

    # saliency
    sens = _saliency_map(model, images_bchw[:min(8, B)], chosen)  # 소배치로 빠르게

    # wmap
    wmap = None
    if isinstance(chosen, nn.Conv2d):
        wmap = _wmap_from_conv(chosen, H, W, device, dtype)
    elif isinstance(chosen, nn.Linear):
        wmap = _wmap_from_linear(chosen, C, H, W, device, dtype)
        # Linear 매핑 불가(입력 형상이 안 맞음)면 None 유지 → saliency-only

    return sens, wmap  # wmap은 None일 수 있음

def _importance_combo(gray_bchw: torch.Tensor, *, cfg: dict | None):
    """
    멀티스케일 LoG + Sobel + 로컬분산을 결합하고,
    선택적으로 saliency/|w|-맵으로 가중 (전부 벡터 연산).
    반환: [B,1,H,W]
    """
    device, dtype = gray_bchw.device, gray_bchw.dtype
    B, _, H, W = gray_bchw.shape

    # ---- 기본 하이퍼 (디폴트) ----
    default = dict(
        # LoG bank
        sigmas=torch.tensor([1.0, 2.0, 4.0], device=device, dtype=dtype),
        alpha =torch.tensor([0.5, 0.3, 0.2], device=device, dtype=dtype),  # 합=1
        log_kernel_size = 9,  # ks=9
        w_log=1.0,
        # Sobel
        w_sobel=0.3,
        # Local variance
        var_sigma=1.5,
        var_kernel_size= 9,
        w_var=0.2,
        # Model-aware gains
        sens=None,
        wmap=None,
        lambda_sens=0.1,   # 0.1
        lambda_wmap=0.1,   # 0.1
    )
    if cfg is None: cfg = {}
    # python 3.10+: | merge
    D = {**default, **cfg}

    # ---- 멀티스케일 LoG (bank conv) ----
    S = D["sigmas"].numel()
    ks_log = int(D["log_kernel_size"])
    bank = _log2d_bank(D["sigmas"], ks_log, device=device, dtype=dtype)    # [S,ks,ks]
    Wlog = bank.view(S,1,ks_log,ks_log)                                    # [S,1,ks,ks]
    # group conv trick: out_ch=S, in_ch=1
    log_resp = F.conv2d(gray_bchw, Wlog, padding=ks_log//2)                # [B,S,H,W]
    log_feat = (log_resp.abs() * D["alpha"].view(1,S,1,1)).sum(dim=1, keepdim=True)  # [B,1,H,W]

    # ---- Sobel magnitude ----
    sobel = _sobel_mag(gray_bchw)                                          # [B,1,H,W]

    # ---- Local variance ----
    var = _local_var(gray_bchw, float(D["var_sigma"]), int(D["var_kernel_size"]))    # [B,1,H,W]

    # ---- 선형 결합 ----
    I = float(D["w_log"])  * log_feat \
        + float(D["w_sobel"])* sobel \
        + float(D["w_var"])  * var      # [B,1,H,W]

    # ---- 모델-인식 가중(옵션): (1 + λ_g*sens + λ_w*wmap) ----
    gain = torch.ones((B,1,H,W), device=device, dtype=dtype)
    for key, lam in (("sens","lambda_sens"), ("wmap","lambda_wmap")):
        M = D[key]
        if M is not None:
            if M.dim() == 2:
                M = M.to(device=device, dtype=dtype).view(1,1,H,W).expand(B,1,H,W)
            elif M.dim() == 3:  # [1,H,W]
                M = M.to(device=device, dtype=dtype).view(1,1,H,W).expand(B,1,H,W)
            else:               # [B,1,H,W] 기대
                M = M.to(device=device, dtype=dtype)
            gain = gain + float(D[lam]) * M
    I = (I * gain).clamp_min(0.0)                                          # [B,1,H,W]
    return I

def choose_best_angle(I: torch.Tensor, n_angles: int = 180) -> float:
    """
    STRICT version:
    - Angle candidates are exact integers: 0,1,...,n_angles-1 (deg)
    - Per-angle bin length n_i is respected in Gini; no bias from padded zeros
    - Fully vectorized (no Python loops)
    """
    assert I.dim() == 2, "I must be [H,W]"
    H, W = I.shape
    A = int(n_angles)
    device, dtype = I.device, I.dtype

    # Integer-degree angles [0..A-1]
    angles = torch.arange(A, device=device, dtype=dtype)
    rad = angles * (math.pi / 180.0)
    cos = rad.cos().view(A, 1, 1)
    sin = rad.sin().view(A, 1, 1)

    yi = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)
    xi = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)

    # Continuous coordinate per angle → shift to 0
    s = cos * yi + sin * xi                                 # [A,H,W]
    s_min = s.amin(dim=(1, 2), keepdim=True)
    s = s - s_min                                           # [A,H,W]
    s_max = s.amax(dim=(1, 2), keepdim=True)
    n_i = s_max.floor().long().view(A) + 1                  # per-angle bin length (>=1)
    Lmax = int(n_i.max().item())

    idx = s.floor().long().clamp_(min=0, max=Lmax - 1)      # [A,H,W]

    # Profiles via scatter_add
    prof = torch.zeros(A, Lmax, device=device, dtype=dtype) # [A,Lmax]
    I_flat = I.view(1, -1).expand(A, -1)                    # [A,HW]
    idx_flat = idx.view(A, -1)
    prof.scatter_add_(1, idx_flat, I_flat)

    # Sort ascending per angle
    x_sorted, _ = torch.sort(prof, dim=1)                   # [A,Lmax]

    # Build tail selector to keep last n_i entries only (exclude padded zeros)
    r_full = torch.arange(1, Lmax + 1, device=device, dtype=dtype).view(1, Lmax)
    # position inside the tail (1..n_i) or <=0 for head
    pos = r_full - (Lmax - n_i.view(-1, 1))
    mask = (pos > 0).to(dtype)

    # Gini over the last n_i bins only
    # numerator = sum_{k=1..n_i} (2k - n_i - 1) * x_sorted_tail[k]
    coef = (2 * pos - n_i.view(-1, 1) - 1) * mask
    num = (coef * x_sorted).sum(dim=1)
    den = (n_i.to(dtype) * (x_sorted * mask).sum(dim=1)).clamp(min=1e-12)
    g_all = num / den

    best_idx = int(torch.argmin(g_all).item())
    return float(angles[best_idx].item())

def dynamic_fragments(
        image: torch.Tensor,
        method: str = 'laplacian',
        n_steps: int = 8,
        n_angles: int = 180,
        overlap: bool = False,
        kernel_size: int = 11,
        overlap_iter: int = 2,
        importance_cfg: Optional[Dict[str, Any]] = None) -> list:
    """
    STRICT vectorized dynamic fragmentation.
    Returns: list of [C,H,W] tensors.
    """
    img = image.unsqueeze(0) if image.dim() == 2 else image  # [C,H,W]
    C, H, W = img.shape
    device, dtype = img.device, img.dtype

    gray_bchw = img.mean(0, keepdim=True).unsqueeze(0)       # [1,1,H,W]

    # ---- importance (vectorized) ----
    if method == 'laplacian':
        ker = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=device, dtype=dtype)
        I = F.conv2d(gray_bchw, ker.view(1,1,3,3), padding=1).abs()[0,0]
    elif method == 'grad':
        sobx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]], device=device, dtype=dtype).view(1,1,3,3)
        soby = sobx.transpose(-1,-2)
        gx = F.conv2d(gray_bchw, sobx, padding=1)[0,0]
        gy = F.conv2d(gray_bchw, soby, padding=1)[0,0]
        I = torch.sqrt(gx*gx + gy*gy + 1e-12)
    elif method in ('combo', 'calibrated'):
        I = _importance_combo(gray_bchw, cfg=importance_cfg)[0,0]          # [H,W]
    else:
        raise ValueError(f"Unsupported method: {method}")

    # ---- 이하 동일 (각도 선택, 분위수 컷, 마스크 생성) ----
    angle = choose_best_angle(I, n_angles)
    theta = math.radians(angle)

    yi = torch.arange(H, device=device).view(H, 1)
    xi = torch.arange(W, device=device).view(1, W)
    s = yi*math.cos(theta) + xi*math.sin(theta)
    s = s - s.min()
    idx = s.floor().long()

    L = int(idx.max().item() + 1)
    prof = torch.zeros(L, device=device, dtype=dtype)
    prof.scatter_add_(0, idx.view(-1), I.view(-1))
    cum = prof.cumsum(0)
    total = cum[-1].clamp_min(1e-12)
    cuts = (total * torch.arange(1, n_steps, device=device) / n_steps)
    boundaries = torch.searchsorted(cum, cuts, right=False)

    seg_id = torch.bucketize(idx.view(-1), boundaries, right=False).view(H, W).clamp(max=n_steps-1)
    masks = F.one_hot(seg_id, num_classes=n_steps).permute(2, 0, 1).to(img.dtype)

    if overlap:
        masks = _dilate_same(masks, ks=kernel_size, iters=overlap_iter)

    frags = (img.unsqueeze(0) * masks.unsqueeze(1))
    return list(frags)

def batch_dynamic_fragments(
        images: torch.Tensor,
        n_steps: int = 8,
        n_angles: int = 180,
        method: str = 'laplacian',
        overlap: bool = False,
        kernel_size : int = 11,
        overlap_iter : int = 2,
        per_image: bool = False,
        power_norm: Optional[Dict[str, Any]] = None,
        importance_cfg: Optional[Dict[str, Any]] = None,
        *,
        weak_model: Optional[nn.Module] = None,
        weak_layer: Optional[nn.Module] = None) -> torch.Tensor:
    """
    auto_weak=True & weak_model 제공 시, sens/wmap을 자동 계산해 importance_cfg에 주입.
    Conv가 맨 앞에 없어도 동작(Conv→Linear→saliency-only 순으로 fallback).
    """
    B, C, H, W = images.shape
    device, dtype = images.device, images.dtype

    # ---- auto weakness injection (combo일 때만 필요) ----
    if method in ('combo', 'calibrated') and weak_model is not None:
        need_sens = (importance_cfg is None) or ('sens' not in importance_cfg or importance_cfg['sens'] is None)
        need_wmap = (importance_cfg is None) or ('wmap' not in importance_cfg)  # None 허용
        if need_sens or need_wmap:
            sens, wmap = _auto_weakness_maps(weak_model, images, weak_layer)
            base = {} if importance_cfg is None else dict(importance_cfg)
            if need_sens:
                base['sens'] = sens.to(device=device, dtype=dtype)
                base.setdefault('lambda_sens', 0.1)
            if need_wmap:
                if wmap is not None:
                    base['wmap'] = wmap.to(device=device, dtype=dtype)
                    base.setdefault('lambda_wmap', 0.1)
            # 디폴트 LoG/Sobel/Var 하이퍼 채워주기(없을 때만)
            base.setdefault('sigmas', torch.tensor([1.0,2.0,4.0], device=device, dtype=dtype))
            base.setdefault('alpha',  torch.tensor([0.4,0.3,0.3], device=device, dtype=dtype))
            base.setdefault('log_kernel_size', 9)
            base.setdefault('w_log', 1.0)
            base.setdefault('w_sobel', 1.0)
            base.setdefault('var_sigma', 1.5)
            base.setdefault('var_kernel_size', 9)
            base.setdefault('w_var', 1.0)
            importance_cfg = base

    # ---- 이하 기존 로직 동일 (중요도 맵 만들고 마스크 추출) ----
    if not per_image:
        gray = images.mean(dim=1, keepdim=True)  # [B,1,H,W]

        if method == 'laplacian':
            ker = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=device, dtype=dtype).view(1,1,3,3)
            I = F.conv2d(gray, ker, padding=1).abs()
        elif method == 'grad':
            sobx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]], device=device, dtype=dtype).view(1,1,3,3)
            soby = sobx.transpose(-1,-2)
            gx = F.conv2d(gray, sobx, padding=1)
            gy = F.conv2d(gray, soby, padding=1)
            I = torch.sqrt(gx*gx + gy*gy + 1e-12)
        elif method in ('combo','calibrated'):
            I = _importance_combo(gray, cfg=importance_cfg)  # [B,1,H,W]
        else:
            raise ValueError(f"Unsupported method: {method}")

        # 공통 마스크 추출을 위해 배치 평균 중요도 사용
        avg_imp = I.mean(dim=0, keepdim=False)[0]  # [H,W]
        base_frags = dynamic_fragments(
            avg_imp.unsqueeze(0).repeat(C, 1, 1),
            method=method, n_steps=n_steps, n_angles=n_angles,
            overlap=overlap, kernel_size=kernel_size, overlap_iter=overlap_iter,
            importance_cfg=importance_cfg
        )
        masks = torch.stack([(frag.abs().sum(0) > 0).to(dtype) for frag in base_frags], dim=0)  # [T,H,W]
        frags = images.unsqueeze(1) * masks.unsqueeze(0).unsqueeze(2)                           # [B,T,C,H,W]

        if power_norm:
            mask_bt1 = masks.unsqueeze(0).unsqueeze(2).expand(frags.size(0), -1, 1, frags.size(-2), frags.size(-1))
            frags, _ = power_normalize_frags(frags, mask=mask_bt1, **power_norm)

        return frags

def manual_fragments(image: torch.Tensor,
                     n_steps: int,
                     direction: str = 'horizontal',
                     overlap: bool = False,
                     kernel_size : int = 3,
                     overlap_iter : int = 1) -> list:

    C, H, W = image.shape
    device = image.device

    if direction == 'horizontal':
        idx = torch.arange(H, device=device).view(H, 1).expand(H, W)
        step = math.ceil(H / n_steps)
        seg_id = (idx // step).clamp(max=n_steps-1)
    elif direction == 'vertical':
        idx = torch.arange(W, device=device).view(1, W).expand(H, W)
        step = math.ceil(W / n_steps)
        seg_id = (idx // step).clamp(max=n_steps-1)
    elif direction in ('diag-left-right', 'diag'):
        yi = torch.arange(H, device=device).view(H,1)
        xi = torch.arange(W, device=device).view(1,W)
        idx = yi + xi
        max_idx = int(idx.max().item()) + 1
        step = math.ceil(max_idx / n_steps)
        seg_id = (idx // step).clamp(max=n_steps-1)
    elif direction in ('diag-right-left', 'anti-diag'):
        yi = torch.arange(H, device=device).view(H,1)
        xi = torch.arange(W, device=device).view(1,W)
        idx = yi + (W - 1 - xi)
        max_idx = int(idx.max().item()) + 1
        step = math.ceil(max_idx / n_steps)
        seg_id = (idx // step).clamp(max=n_steps-1)
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    masks = F.one_hot(seg_id, num_classes=n_steps).permute(2,0,1).to(image.dtype)

    if overlap:
        masks = _dilate_same(masks, ks=kernel_size, iters=overlap_iter)

    frags = image.unsqueeze(0) * masks.unsqueeze(1)

    return list(frags)

def batch_manual_fragments(images: torch.Tensor,
                           time_steps: int,
                           direction: str = 'horizontal',
                           overlap: bool = False,
                           kernel_size : int = 11,
                           overlap_iter : int = 2,
                           power_norm: Optional[Dict[str, Any]] = None) -> torch.Tensor:

    sample = manual_fragments(images[0], time_steps, direction, overlap=overlap,
                              kernel_size=kernel_size, overlap_iter=overlap_iter)

    masks = torch.stack([(frag.abs().sum(0) > 0).to(images.dtype) for frag in sample], dim=0)

    frags = images.unsqueeze(1) * masks.unsqueeze(0).unsqueeze(2)  # [B,T,C,H,W]

    if power_norm:
        try:
            mask_bt1 = masks.unsqueeze(0).unsqueeze(2).expand(frags.size(0), -1, 1, frags.size(-2), frags.size(-1))
        except NameError:
            mask_bt1 = None
        frags, _ = power_normalize_frags(frags, mask=mask_bt1, **power_norm)

    return frags

def _dilate_same(masks: torch.Tensor, ks: int = 3, iters: int = 1) -> torch.Tensor:
    if ks < 1:
        return masks

    m = masks.unsqueeze(1).float()        # [T,1,H,W]
    pad_l = ks // 2
    pad_r = ks - 1 - pad_l
    pad_t = ks // 2
    pad_b = ks - 1 - pad_t
    for _ in range(iters):
        m = F.pad(m, (pad_l, pad_r, pad_t, pad_b), mode="constant", value=0.0)
        m = F.max_pool2d(m, kernel_size=ks, stride=1, padding=0)

    return (m > 0).to(masks.dtype).squeeze(1)  # [T,H,W] (크기 동일)

def _frag_mask_from_values(frags: torch.Tensor) -> torch.Tensor:
    """
    값 기반 활성 마스크 생성: [B,T,1,H,W]
    채널합의 절댓값이 0 초과인 위치를 활성으로 본다.
    """
    assert frags.dim() == 5, f"frags must be [B,T,C,H,W], got {tuple(frags.shape)}"
    return (frags.abs().sum(dim=2, keepdim=True) > 0)

def power_normalize_frags(
        frags: torch.Tensor,                 # [B,T,C,H,W]
        *,
        mode: str = "rms",                   # 'rms' | 'l2' | 'l1' | 'std'
        target: float = 1.0,                 # 목표 파워 (예: RMS=1.0)
        per_channel: bool = False,           # True면 (B,T,C)별, False면 (B,T)별
        use_mask: bool = True,               # True면 활성 픽셀만 통계
        mask: Optional[torch.Tensor] = None, # [B,T,1,H,W] (옵션)
        max_gain: float = 10.0,              # 과증폭 방지
        eps: float = 1e-8,
        center: bool = False,                # True면 평균 0 보정 후 'std'/'l2' 등 계산
        clip: Optional[Tuple[float,float]] = None,  # (lo, hi) 값 클립
        detach_stats: bool = True            # 통계/게인 그래프 분리(메모리/안정성↑)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    각 프래그(또는 채널별) 파워를 target으로 스케일.
    반환: (scaled_frags, gain)  # gain: [B,T,(1|C),1,1]
    """
    assert frags.dim() == 5, f"frags must be [B,T,C,H,W], got {tuple(frags.shape)}"
    B,T,C,H,W = frags.shape
    device, dtype = frags.device, frags.dtype

    if use_mask:
        if mask is None:
            mask = _frag_mask_from_values(frags)  # [B,T,1,H,W]
        m = mask.to(dtype)
    else:
        m = torch.ones((B,T,1,H,W), device=device, dtype=dtype)

    mC = m.expand(B,T,C,H,W)  # 채널로 브로드캐스트
    x  = frags

    # 통계 계산 함수
    def _compute_stat(x_: torch.Tensor) -> torch.Tensor:
        xC = x_
        if center:
            denom = mC.sum(dim=(3,4), keepdim=True).clamp_min(1.0)
            mean  = (xC * mC).sum(dim=(3,4), keepdim=True) / denom
            xC    = (xC - mean) * mC
        if mode in ("rms", "l2"):
            num = (xC.pow(2) * mC).sum(dim=(3,4), keepdim=True)       # [B,T,C,1,1]
            N   = mC.sum(dim=(3,4), keepdim=True).clamp_min(1.0)
            stat = torch.sqrt((num / N) + eps) if mode=="rms" else torch.sqrt(num + eps)
        elif mode == "l1":
            num = (xC.abs() * mC).sum(dim=(3,4), keepdim=True)
            N   = mC.sum(dim=(3,4), keepdim=True).clamp_min(1.0)
            stat = num / N
        elif mode == "std":
            num = ((xC - 0.0)**2 * mC).sum(dim=(3,4), keepdim=True)
            N   = mC.sum(dim=(3,4), keepdim=True).clamp_min(1.0)
            stat = torch.sqrt(num / N + eps)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        if not per_channel:
            stat = stat.mean(dim=2, keepdim=True)                     # [B,T,1,1,1]
        return stat  # [B,T,(C|1),1,1]

    # 통계/게인 분리 옵션
    if detach_stats:
        with torch.no_grad():
            stat = _compute_stat(x)                                   # [B,T,(C|1),1,1]
            gain = (float(target) / (stat + eps)).clamp(max=max_gain) # [B,T,(C|1),1,1]
            inactive = (m.sum(dim=(2,3,4), keepdim=True) == 0)        # [B,T,1,1,1]
            if per_channel:
                inactive = inactive.expand(B,T,C,1,1)
            gain = torch.where(inactive, torch.ones_like(gain), gain)
        y = x * gain                                                  # 게인은 detatched
    else:
        stat = _compute_stat(x)
        gain = (float(target) / (stat + eps)).clamp(max=max_gain)
        inactive = (m.sum(dim=(2,3,4), keepdim=True) == 0)
        if per_channel:
            inactive = inactive.expand(B,T,C,1,1)
        gain = torch.where(inactive, torch.ones_like(gain), gain)
        y = x * gain

    if clip is not None:
        lo, hi = float(clip[0]), float(clip[1])
        y = y.clamp(min=lo, max=hi)

    return y, gain

def agg_conf_logits(
        logits_t: torch.Tensor,
        tau: float = 2.0,
        *,
        time_major: bool = True,
        eps: float = 1e-8,
) -> torch.Tensor:
    if logits_t.dim() != 3:
        raise ValueError(f"logits_t must be 3D, got {tuple(logits_t.shape)}")
    if not time_major:
        logits_t = logits_t.permute(1, 0, 2)  # [B,T,K] -> [T,B,K]

    p_t = torch.softmax(logits_t, dim=-1)                           # [T,B,K]
    ent = -(p_t * (p_t.clamp_min(eps)).log()).sum(dim=-1)           # [T,B]

    w = torch.softmax(-tau * ent, dim=0)                            # [T,B]

    logits_seq = (logits_t * w.unsqueeze(-1)).sum(dim=0)            # [B,K]
    return logits_seq

def agg_poe_logits(logits_t, tau=1.0, time_major=True):
    if not time_major:
        logits_t = logits_t.permute(1,0,2)           # [T,B,K]
    logp_t = torch.log_softmax(logits_t, dim=-1)     # [T,B,K]
    logits_seq = (tau * logp_t).sum(dim=0)         # [B,K]
    return logits_seq

def agg_owa_logits(logits_t, tau=0.5, time_major=True):
    if not time_major:
        logits_t = logits_t.permute(1,0,2)         # [T,B,K]
    T,B,K = logits_t.shape
    p = torch.softmax(logits_t, dim=-1)
    ent = -(p * p.clamp_min(1e-8).log()).sum(-1)   # [T,B]
    idx = torch.argsort(ent, dim=0)                # [T,B]
    keep = int(max(1, round(T*tau)))
    pick = idx[:keep, torch.arange(B)]             # [keep,B]
    logits_sel = logits_t.gather(0, pick.unsqueeze(-1).expand(-1,-1,K))  # [keep,B,K]
    return logits_sel.mean(dim=0)                  # [B,K]

def _one_hot(targets, num_classes, dtype):
    y = F.one_hot(targets, num_classes=num_classes).to(dtype)
    return y

def _brier_rmse_from_logits(logits_bk, targets, *, mode="brier", temperature=1.0, label_smoothing=0.0,
                            reduction="mean", eps=1e-8):
    # logits_bk: [B,K]
    B, K = logits_bk.shape
    p = torch.softmax(logits_bk / temperature, dim=-1)        # [B,K]
    y = _one_hot(targets, K, p.dtype)
    if label_smoothing > 0.0:
        y = (1.0 - label_smoothing) * y + label_smoothing / K
    mse_per = (p - y).pow(2).mean(dim=-1)                     # [B]
    if mode == "brier":
        out = mse_per
    else:  # rmse
        out = torch.sqrt(mse_per + eps)
    if reduction == "mean": return out.mean()
    if reduction == "sum":  return out.sum()
    return out  # 'none'

def _ce_or_focal_from_logits(logits_bk, targets, *, label_smoothing=0.0, temperature=1.0,
                             reduction="mean", focal_gamma=0.0, focal_alpha=None, eps=1e-8):
    # 표준 CE 또는 Focal(γ>0)
    if focal_gamma <= 0.0 and focal_alpha is None:
        return F.cross_entropy(logits_bk / temperature, targets,
                               label_smoothing=label_smoothing, reduction=reduction)
    # Focal CE
    B, K = logits_bk.shape
    logp = F.log_softmax(logits_bk / temperature, dim=-1)
    y_one = _one_hot(targets, K, logp.dtype)
    if label_smoothing > 0.0:
        y_sm = (1.0 - label_smoothing) * y_one + label_smoothing / K
    else:
        y_sm = y_one
    ce_per = -(y_sm * logp).sum(dim=-1)                       # [B]
    with torch.no_grad():
        p = logp.exp()
        p_y = p.gather(dim=-1, index=targets.view(-1,1)).squeeze(1)  # [B]
        mod = (1.0 - p_y).clamp_min(0).pow(focal_gamma)               # (1-pt)^γ
        if isinstance(focal_alpha, (float, int)):
            alpha_w = torch.full_like(mod, float(focal_alpha))
        elif isinstance(focal_alpha, torch.Tensor):
            alpha_w = focal_alpha.to(mod.dtype).to(mod.device)
            if alpha_w.numel() == K:
                alpha_w = alpha_w[targets]
        else:
            alpha_w = torch.ones_like(mod)
        w = mod * alpha_w
    loss = (w * ce_per)
    if reduction == "mean": return loss.mean()
    if reduction == "sum":  return loss.sum()
    return loss

def _linear_tail_weights(T, warmup_frac=0.3, device=None, dtype=None, normalize=True):
    """처음 warmup_frac*T 스텝은 0, 그 뒤로 선형 증가(후반 스텝에 집중)."""
    w = torch.linspace(0.0, 1.0, steps=T, device=device, dtype=dtype)
    warm = int(round(T * float(warmup_frac)))
    if warm > 0:
        w[:warm] = 0.0
    if normalize:
        s = w.sum()
        w = w / (s + 1e-12)
    return w  # [T]

def fragmentation_loss(logits: torch.Tensor,
                       targets: torch.Tensor,
                       *,
                       # --- 기존 인자 (그대로 유지) ---
                       mode: str = "ce",                 # 'ce' | 'brier' | 'rmse'
                       temperature: float = 1.0,
                       label_smoothing: float = 0.0,
                       reduction: str = "mean",
                       step_agg: str = "mean",           # 3D일 때만 사용: 'mean'|'sum'|'last'
                       step_weights: torch.Tensor | None = None,  # [T]
                       time_major: bool = True,          # logits=[T,B,K]이면 True, [B,T,K]면 False
                       eps: float = 1e-8,
                       # --- 새 옵션: 성능 강화 ---
                       # 1) 타임스텝 일관성 정규화(스텝 분포들이 비슷하게)
                       lambda_cons: float = 0.0,
                       cons_type: str = "mse",           # 'mse' | 'kl' (symmetric)
                       cons_detach_anchor: bool = True,  # 앵커(집계 분포) stop_grad 권장
                       # 2) 후반 스텝 가중 보조 손실 (per-step 분류 보조)
                       lambda_aux: float = 0.0,
                       aux_warmup_frac: float = 0.3,     # 초기 T*frac 스텝은 0 가중
                       # 3) Focal CE (선택)
                       focal_gamma: float = 0.0,
                       focal_alpha: float | torch.Tensor | None = None,
                       ) -> torch.Tensor:
    """
    강건한 프래그 시퀀스 분류 손실:
      main(CE/Brier/RMSE) + lambda_cons*Consistency + lambda_aux*AuxPerStep
    - logits: [B,K] 또는 [T,B,K] / [B,T,K]
    - targets: [B]
    """
    # 1) 입력 정규화 (2D or 3D)
    is_seq = (logits.dim() == 3)
    if not is_seq and logits.dim() != 2:
        raise ValueError(f"logits must be 2D or 3D, got {tuple(logits.shape)}")

    if is_seq and not time_major:
        logits = logits.permute(1, 0, 2)  # [B,T,K] -> [T,B,K]
    if is_seq:
        T, B, K = logits.shape
    else:
        B, K = logits.shape

    # 2) 집계 로짓 (없으면 내부에서)
    if is_seq:
        if step_weights is not None:
            if step_weights.dim() != 1 or step_weights.numel() != T:
                raise ValueError(f"step_weights must be [T]={T}, got {tuple(step_weights.shape)}")
            w = step_weights.to(device=logits.device, dtype=logits.dtype)
            w = w / (w.sum() + 1e-12)
            logits_seq = (logits * w.view(T,1,1)).sum(dim=0)    # [B,K]
        else:
            if step_agg == "mean":
                logits_seq = logits.mean(dim=0)
            elif step_agg == "sum":
                logits_seq = logits.sum(dim=0)
            elif step_agg == "last":
                logits_seq = logits[-1]
            else:
                raise ValueError(f"Unsupported step_agg: {step_agg}")
    else:
        logits_seq = logits  # [B,K]

    # 3) 메인 분류 손실
    if mode == "ce":
        main_loss = _ce_or_focal_from_logits(
            logits_seq, targets,
            label_smoothing=label_smoothing, temperature=temperature,
            reduction=reduction, focal_gamma=focal_gamma, focal_alpha=focal_alpha
        )
    elif mode in ("brier", "rmse"):
        main_loss = _brier_rmse_from_logits(
            logits_seq, targets, mode=mode,
            temperature=temperature, label_smoothing=label_smoothing,
            reduction=reduction, eps=eps
        )
    else:
        raise ValueError(f"mode must be 'ce' | 'brier' | 'rmse', got {mode}")

    total_loss = main_loss

    # 4) 타임스텝 일관성 정규화
    if is_seq and lambda_cons > 0.0:
        # 스텝별 확률 & 앵커(집계) 확률
        p_t = torch.softmax(logits / temperature, dim=-1)              # [T,B,K]
        p_anchor = torch.softmax(logits_seq / temperature, dim=-1)     # [B,K]
        if cons_detach_anchor:
            p_anchor = p_anchor.detach()
        if cons_type == "mse":
            cons = ((p_t - p_anchor.unsqueeze(0))**2).mean()
        elif cons_type == "kl":
            # 대칭 KL
            p_t_cl = (p_t + eps).clamp_max(1.0)
            pa_cl = (p_anchor + eps).clamp_max(1.0)
            logp_t = p_t_cl.log()
            logpa = pa_cl.log()
            kl1 = (p_t_cl * (logp_t - logpa.unsqueeze(0))).mean()
            kl2 = (pa_cl.unsqueeze(0) * (logpa.unsqueeze(0) - logp_t)).mean()
            cons = 0.5 * (kl1 + kl2)
        else:
            raise ValueError("cons_type must be 'mse' or 'kl'")
        total_loss = total_loss + lambda_cons * cons

    # 5) 후반 스텝 가중 보조 손실(per-step)
    if is_seq and lambda_aux > 0.0:
        w_aux = _linear_tail_weights(T, warmup_frac=aux_warmup_frac,
                                     device=logits.device, dtype=logits.dtype, normalize=True)  # [T]
        aux_losses = []
        for t in range(T):
            step_logits = logits[t]  # [B,K]
            if mode == "ce":
                loss_t = _ce_or_focal_from_logits(
                    step_logits, targets,
                    label_smoothing=label_smoothing, temperature=temperature,
                    reduction="mean", focal_gamma=0.0, focal_alpha=None  # 보조는 표준 CE 권장
                )
            else:
                loss_t = _brier_rmse_from_logits(
                    step_logits, targets, mode=("brier" if mode=="brier" else "rmse"),
                    temperature=temperature, label_smoothing=label_smoothing,
                    reduction="mean", eps=eps
                )
            aux_losses.append(loss_t * w_aux[t])
        aux_loss = torch.stack(aux_losses).sum()
        total_loss = total_loss + lambda_aux * aux_loss

    return total_loss


class FragNorm(nn.Module):
    """
    BN2d의 time(프래그) 축 확장 버전.
    - 입력 shape 지원:
        [B, T, C, H, W]  (권장)  ← 프래그/타임축 포함
        [B, C, H, W]     (T=1로 간주)
        [B, T, F]        (F를 C로 보고 H=W=1)
        [B, F]           (T=1, H=W=1)
    - time_aggregate=True  → 통계에 T도 포함(즉, N=B*T로 간주)  [기본값]
      time_aggregate=False → 각 time-step별 통계를 따로 사용(러닝스탯은 평균 집계)
    - track_running_stats=True  → eval 모드에서 러닝스탯 사용 (BatchNorm과 동일)
      False                    → eval에서도 미니배치 통계 사용 (InstanceNorm 느낌)
    - affine=True  → per-channel γ/β 학습 (BatchNorm과 동일)
    """
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            time_aggregate: bool = True,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        self.track_running_stats = bool(track_running_stats)
        self.time_aggregate = bool(time_aggregate)

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias   = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var',  torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var',  None)
            self.register_buffer('num_batches_tracked', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    # ---- 유틸: 어떤 입력이 와도 [B,T,C,H,W]로 바꿨다가 다시 복원 ----
    def _as_5d(self, x):
        orig_dim = x.dim()
        if orig_dim == 5:                     # [B,T,C,H,W]
            return x, orig_dim
        elif orig_dim == 4:                   # [B,C,H,W] → T=1
            x = x.unsqueeze(1)
            return x, orig_dim
        elif orig_dim == 3:                   # [B,T,F] → [B,T,C=F,H=1,W=1]
            x = x.unsqueeze(-1).unsqueeze(-1)
            return x, orig_dim
        elif orig_dim == 2:                   # [B,F] → [B,T=1,C=F,H=1,W=1]
            x = x.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            return x, orig_dim
        else:
            raise ValueError(f"FragNorm expects input dim in {{2,3,4,5}}, got {orig_dim}")

    def _restore_shape(self, y, orig_dim):
        if orig_dim == 5:   # [B,T,C,H,W]
            return y
        elif orig_dim == 4: # [B,C,H,W]
            return y.squeeze(1)
        elif orig_dim == 3: # [B,T,F]
            return y.squeeze(-1).squeeze(-1)
        elif orig_dim == 2: # [B,F]
            return y.squeeze(1).squeeze(-1).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x5, orig_dim = self._as_5d(x)  # [B,T,C,H,W]
        B, T, C, H, W = x5.shape
        assert C == self.num_features, f"num_features={self.num_features}, but input C={C}"

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1

        # ----- 통계 계산 -----
        if self.time_aggregate:
            # BN2d와 동일하게 N=B*T로 보고 평균/분산 계산
            reduce_dims = (0, 1, 3, 4)  # N,Both spatial
            batch_mean = x5.mean(dim=reduce_dims, keepdim=True)       # [1,1,C,1,1]
            batch_var  = x5.var(dim=reduce_dims, keepdim=True, unbiased=False)
            ch_mean_for_running = x5.mean(dim=(0,1,3,4))              # [C]
            ch_var_for_running  = x5.var(dim=(0,1,3,4), unbiased=False)
        else:
            # time-step별 통계(각 t마다 B,H,W로 평균/분산) → 적용은 per-step로,
            # 러닝스탯 업데이트는 t 평균으로 집계.
            batch_mean = x5.mean(dim=(0, 3, 4), keepdim=True)         # [1,T,C,1,1]
            batch_var  = x5.var(dim=(0, 3, 4), keepdim=True, unbiased=False)
            ch_mean_for_running = x5.mean(dim=(0,3,4)).mean(dim=0)    # [C]  ← T 평균
            ch_var_for_running  = x5.var(dim=(0,3,4), unbiased=False).mean(dim=0)

        # ----- 러닝스탯 업데이트/사용 -----
        if self.track_running_stats:
            if self.training:
                with torch.no_grad():
                    self.running_mean.mul_(1 - self.momentum).add_(self.momentum * ch_mean_for_running)
                    self.running_var.mul_(1 - self.momentum).add_(self.momentum * ch_var_for_running)

            # eval일 때 러닝스탯 사용
            if not self.training:
                mean_use = self.running_mean.view(1,1,C,1,1)
                var_use  = self.running_var.view(1,1,C,1,1)
            else:
                mean_use = batch_mean
                var_use  = batch_var
        else:
            # 트래킹 안 하면 eval에서도 배치 통계 사용 (IN 스타일)
            mean_use = batch_mean
            var_use  = batch_var

        # ----- 정규화 + 어파인 -----
        y = (x5 - mean_use) / torch.sqrt(var_use + self.eps)

        if self.affine:
            w = self.weight.view(1,1,C,1,1)
            b = self.bias.view(1,1,C,1,1)
            y = y * w + b

        return self._restore_shape(y, orig_dim)
