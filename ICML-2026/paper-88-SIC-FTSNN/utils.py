import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.transforms as T
import torchvision.datasets as tvd

import fnmatch
import math
from typing import List, Optional, Tuple, Dict

class TDBatchNorm(nn.Module):
    def __init__(self, num_features, init_threshold=1.0, momentum=0.1, epsilon=1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon

        # learnable scale/shift
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta  = nn.Parameter(torch.zeros(num_features))

        # buffers
        self.register_buffer("threshold", torch.ones(num_features) * init_threshold)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # tdBN 수식: gamma * ((x-μ)/sqrt(σ²+ε)) / threshold + beta
        # ↓ weight=gamma/threshold, bias=beta 로 접어서 최적화된 batch_norm 커널 사용
        weight = self.gamma / self.threshold
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            weight,
            self.beta,
            self.training,
            self.momentum,
            self.epsilon,
        )

# ----- Label-invariant Cutout -----
class Cutout:
    """라벨 불변 Cutout. transforms.Compose 안에서 바로 사용."""
    def __init__(self, size=8, p=0.5):
        self.size, self.p = int(size), float(p)
    def __call__(self, img: torch.Tensor):
        # img: Tensor [C,H,W] (ToTensor 이후, Normalize 전/후 상관없음)
        if torch.rand(()) > self.p:
            return img
        _, H, W = img.shape
        y = torch.randint(0, H, (1,)).item()
        x = torch.randint(0, W, (1,)).item()
        y1 = max(0, y - self.size // 2); y2 = min(H, y1 + self.size)
        x1 = max(0, x - self.size // 2); x2 = min(W, x1 + self.size)
        img[:, y1:y2, x1:x2] = 0
        return img

# ----- Transforms preset (label-invariant only) -----
def build_cifar_transforms(dataset: str = "cifar10",
                           preset: str = "strong",
                           normalize_for_poisson: bool = True,
                           cutout_size: int = 8,
                           cutout_p: float = 0.5,
                           random_erasing_p: float = 0.25):
    ds = dataset.lower()
    assert ds in {"cifar10", "cifar100"}

    # Poisson 인코더 호환: 값범위 유지 (0~1)
    mean, std = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) if normalize_for_poisson else \
        ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)) if ds == "cifar10" else \
            ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    # 정책: AutoAugment(또는 RandAugment로 바꿔도 OK)
    policy = T.AutoAugment(T.AutoAugmentPolicy.CIFAR10)

    if preset == "none":
        train_tf = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    elif preset == "base":
        train_tf = T.Compose([
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
            Cutout(size=cutout_size, p=cutout_p),
        ])
    else:
        # strong (기본)
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            policy,
            T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.5),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.1),
            T.RandomApply([T.RandomAffine(degrees=10, translate=(0.1, 0.1))], p=0.3),
            T.RandomPerspective(distortion_scale=0.1, p=0.1),
            T.ToTensor(),
            T.Normalize(mean, std),
            Cutout(size=cutout_size, p=cutout_p),
            T.RandomErasing(p=random_erasing_p),
        ])

    test_tf = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tf, test_tf

# ----- One-liner: loaders + meta -----
def make_cifar_loaders(data_root: str,
                       dataset: str = "cifar10",
                       batch_size: int = 128,
                       num_workers: int = 4,
                       preset: str = "strong",
                       normalize_for_poisson: bool = True,
                       shuffle: bool = True,
                       drop_last: bool = True):
    ds = dataset.lower()
    assert ds in {"cifar10", "cifar100"}
    train_tf, test_tf = build_cifar_transforms(ds, preset, normalize_for_poisson)

    if ds == "cifar10":
        train_set = tvd.CIFAR10(data_root, train=True, download=True, transform=train_tf)
        test_set  = tvd.CIFAR10(data_root, train=False, download=True, transform=test_tf)
        num_classes = 10
    else:
        train_set = tvd.CIFAR100(data_root, train=True, download=True, transform=train_tf)
        test_set  = tvd.CIFAR100(data_root, train=False, download=True, transform=test_tf)
        num_classes = 100

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=True, drop_last=drop_last)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    meta = {"in_ch": 3, "H": 32, "W": 32, "num_classes": num_classes}
    return train_loader, test_loader, meta


class ZBiasAdder:
    """
    Z-bias forward hook with:
      - Layer selection by name pattern(s) OR apply_to_all toggle
      - Fractional neuron/channel targeting (apply_fraction)
      - Outlier bias for a subset of neurons (outlier_bias / outlier_fraction)
      - Start-epoch gating
      - Optional per-forward resampling (stochastic mask)
      - Backward compatibility for legacy args: spike_bias/spike_fraction/spike_mode

    Terminology:
      - "Neuron":
          Linear  -> out_features
          ConvNd  -> out_channels (channel-wise, broadcast over spatial dims)

    Example:
      adder = ZBiasAdder(
          base_bias=5.0,
          apply_fraction=1.0,             # 70%만 하려면 0.7
          outlier_bias=50.0,
          outlier_fraction=0.10,          # 전체 뉴런의 10%에 큰 값
          outlier_mode='override',        # 'override' or 'add'
          start_epoch=5,
          target_patterns=['fc1', 'layer3.*'],  # 이름 패턴
          apply_to_all=False,              # True면 모든 모듈에 적용(제외 패턴은 유효)
          resample_every_forward=False,    # True면 매 forward 때마다 무작위 재샘플
          seed=123
      )
      adder.attach(model)
      for epoch in range(epochs):
          adder.set_epoch(epoch)
          ...
      adder.detach()
    """

    def __init__(
            self,
            base_bias: float = 5.0,
            apply_fraction: float = 1.0,
            outlier_bias: Optional[float] = None,
            outlier_fraction: float = 0.0,
            outlier_mode: str = "override",           # 'override' or 'add'
            start_epoch: int = 0,
            target_patterns: Optional[List[str]] = None,
            exclude_patterns: Optional[List[str]] = None,
            layer_types: Tuple[type, ...] = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d),
            resample_every_forward: bool = False,
            last_only: bool = False,
            apply_to_all: bool = False,
            seed: Optional[int] = None,
            **legacy_kwargs,
    ):
        # --- core params ---
        self.base_bias = float(base_bias)
        self.apply_fraction = float(apply_fraction)
        self.outlier_bias = None if outlier_bias is None else float(outlier_bias)
        self.outlier_fraction = float(outlier_fraction)
        assert outlier_mode in ("override", "add"), "outlier_mode must be 'override' or 'add'"
        self.outlier_mode = outlier_mode

        self.start_epoch = int(start_epoch)
        self.current_epoch = 0

        self.target_patterns = target_patterns or []  # empty => match all (unless apply_to_all False with exclude only)
        self.exclude_patterns = exclude_patterns or []
        self.layer_types = layer_types

        self.resample_every_forward = bool(resample_every_forward)
        self.last_only = bool(last_only)
        self.apply_to_all = bool(apply_to_all)

        # --- RNG ---
        self.seed = seed
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

        # --- runtime holders ---
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._modules: List[nn.Module] = []
        self._bias_vec_cache: Dict[int, torch.Tensor] = {}
        self._attached = False

        # --- Backward-compat: map legacy spike_* kwargs to outlier_* ---
        if "spike_bias" in legacy_kwargs and outlier_bias is None:
            self.outlier_bias = float(legacy_kwargs["spike_bias"])
            print("[ZBiasAdder] DEPRECATION: 'spike_bias' -> use 'outlier_bias'")
        if "spike_fraction" in legacy_kwargs and outlier_fraction == 0.0:
            self.outlier_fraction = float(legacy_kwargs["spike_fraction"])
            print("[ZBiasAdder] DEPRECATION: 'spike_fraction' -> use 'outlier_fraction'")
        if "spike_mode" in legacy_kwargs and outlier_mode == "override":
            self.outlier_mode = str(legacy_kwargs["spike_mode"])
            print("[ZBiasAdder] DEPRECATION: 'spike_mode' -> use 'outlier_mode' ('override'|'add')")

    # ---------- Public API ----------
    def attach(self, model: nn.Module, verbose: bool = True):
        if self._attached:
            if verbose:
                print("[ZBiasAdder] Already attached; call detach() first.")
            return

        candidates = []
        for name, m in model.named_modules():
            if not isinstance(m, self.layer_types):
                continue
            if self._is_excluded(name):
                continue
            if self.apply_to_all or self._is_target(name):
                candidates.append((name, m))

        if self.last_only and candidates:
            candidates = [candidates[-1]]

        for name, m in candidates:
            h = m.register_forward_hook(self._hook)
            self._handles.append(h)
            self._modules.append(m)
            if verbose:
                print(f"[ZBiasAdder] attached to: {name} ({m.__class__.__name__})")

        self._attached = True
        if not candidates and verbose:
            print("[ZBiasAdder] No layers matched. Check patterns or set apply_to_all=True.")

    def detach(self, verbose: bool = True):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        self._modules.clear()
        self._bias_vec_cache.clear()
        self._attached = False
        if verbose:
            print("[ZBiasAdder] detached")

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

    # ---------- Internals ----------
    def _is_target(self, name: str) -> bool:
        if not self.target_patterns:
            # If no patterns provided, treat as "match all" only when apply_to_all is False?
            # We'll allow match-all if empty => True to keep simple usage.
            return True
        return any(fnmatch.fnmatch(name, pat) for pat in self.target_patterns)

    def _is_excluded(self, name: str) -> bool:
        return any(fnmatch.fnmatch(name, pat) for pat in self.exclude_patterns)

    @torch.no_grad()
    def _make_bias_vector_for_module(self, m: nn.Module, device, dtype) -> torch.Tensor:
        if isinstance(m, nn.Linear):
            N = int(m.out_features)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            N = int(m.out_channels)
        else:
            N = getattr(m, "out_features", None) or getattr(m, "out_channels", None)
            if N is None:
                raise RuntimeError(f"Unsupported module type for bias vector: {type(m)}")

        # clamp fractions to [0, 1]
        apply_frac = min(max(self.apply_fraction, 0.0), 1.0)
        outlier_frac = min(max(self.outlier_fraction, 0.0), 1.0)

        n_outlier = int(round(outlier_frac * N))
        n_base_total = int(round(apply_frac * N))

        if n_outlier > N:
            n_outlier = N

        perm = torch.randperm(N, generator=self._rng, device='cpu').to(device)
        outlier_idx = perm[:n_outlier]
        remain = perm[n_outlier:]
        n_base_only = min(n_base_total, remain.numel())
        base_idx = remain[:n_base_only]

        bias_vec = torch.zeros(N, device=device, dtype=dtype)

        # base bias (for "base only" set)
        if n_base_only > 0 and self.base_bias != 0.0:
            bias_vec[base_idx] += self.base_bias

        # outlier bias
        if n_outlier > 0 and (self.outlier_bias is not None) and (self.outlier_bias != 0.0):
            if self.outlier_mode == "override":
                bias_vec[outlier_idx] = self.outlier_bias
            else:  # 'add'
                bias_vec[outlier_idx] += self.outlier_bias

        # Note:
        # - If apply_fraction == 1.0, everyone gets base unless chosen as outlier.
        # - Outliers either override or add on top of base depending on outlier_mode,
        #   but here outliers come from a disjoint set (we didn't give base to them).
        #   If you want outliers to ALSO include base by default, switch order or use outlier_mode='add'.
        return bias_vec

    def _get_bias_vector(self, m: nn.Module, out: torch.Tensor) -> torch.Tensor:
        key = id(m)
        if self.resample_every_forward:
            return self._make_bias_vector_for_module(m, out.device, out.dtype)

        if key not in self._bias_vec_cache:
            self._bias_vec_cache[key] = self._make_bias_vector_for_module(m, out.device, out.dtype)
        else:
            vec = self._bias_vec_cache[key]
            if vec.device != out.device or vec.dtype != out.dtype:
                self._bias_vec_cache[key] = vec.to(device=out.device, dtype=out.dtype)
        return self._bias_vec_cache[key]

    def _hook(self, m: nn.Module, inputs, out: torch.Tensor):
        if self.current_epoch < self.start_epoch:
            return out

        bias_vec = self._get_bias_vector(m, out)

        if isinstance(m, nn.Linear):
            bias = bias_vec.view(1, -1)          # [1, F]
        elif isinstance(m, nn.Conv1d):
            bias = bias_vec.view(1, -1, 1)       # [1, C, 1]
        elif isinstance(m, nn.Conv2d):
            bias = bias_vec.view(1, -1, 1, 1)    # [1, C, 1, 1]
        elif isinstance(m, nn.Conv3d):
            bias = bias_vec.view(1, -1, 1, 1, 1) # [1, C, 1, 1, 1]
        else:
            shape = [1] * out.ndim
            shape[1] = bias_vec.numel()
            bias = bias_vec.view(*shape)

        return out + bias


# ---- Auto-fold: 1D sequence → 2D image-like layout ----

def _choose_hw_autofold(eff_len: int, min_side: int = 15, aspect_max: float = 4.0) -> Tuple[int, int]:
    """Pick (H, W) close to sqrt(eff_len) with H <= aspect_max * W and W <= aspect_max * H."""
    best_h, best_w = 1, eff_len
    best_err = abs(best_h - best_w)
    for h in range(min_side, int(math.sqrt(eff_len * aspect_max)) + 2):
        w = math.ceil(eff_len / h)
        if w < min_side:
            continue
        if h / w > aspect_max or w / h > aspect_max:
            continue
        err = abs(h - w)
        if err < best_err:
            best_err = err
            best_h, best_w = h, w
    return best_h, best_w


def seq1d_to_2d_autofold(
    data: torch.Tensor,
    H: int,
    W: int,
    out_len: Optional[int] = None,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Fold [B, C, L] (or [B, L]) into [B, 1, H, W].

    If out_len is given, only that many elements are used; the rest is padded.
    """
    if data.dim() == 2:
        data = data.unsqueeze(1)  # [B, 1, L]
    B, C, L = data.shape
    eff = L if out_len is None else min(L, out_len)
    total = H * W
    if eff < total:
        pad_amt = total - eff
        out = F.pad(data[:, :, :eff], (0, pad_amt), value=pad_value)
    else:
        out = data[:, :, :total]
    return out.view(B, 1, H, W)


class FragmentEnergyTracker:
    """Tracks per-fragment statistics (RMS, spike count, etc.)."""

    def __init__(self, layout: str = "btc"):
        self.layout = layout
        self.history: List[Dict[str, float]] = []

    def update(self, fragments: torch.Tensor):
        """fragments: [B, T, C, H, W] or [B, T, C] (layout 'btc')."""
        if self.layout == "btc" and fragments.dim() >= 3:
            t_dim = 1
            rms = fragments.pow(2).mean(dim=[d for d in range(fragments.dim()) if d != t_dim]).sqrt()
            self.history.append({"rms": float(rms.mean().item()), "t": len(self.history)})

    def format(self, title: Optional[str] = None, joiner: str = ", ") -> str:
        if not self.history:
            return ""
        last = self.history[-1]
        prefix = f"{title}: " if title else ""
        return prefix + joiner.join(f"{k}={v:.4f}" for k, v in last.items())

