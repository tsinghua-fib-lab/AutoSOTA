"""FerretNet backbone for forensic feature extraction in Ferret-SAM.
Refined with Instance Normalization and SE-Blocks for better OOD robustness.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class SELayer(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SeparableConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 0
    ):
        super().__init__()
        self.depth_wise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding=padding, groups=in_channels, bias=False
        )
        self.point_wise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False
        )

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.point_wise(x)
        return x


class DilatedConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            dilation=1,
            bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=2,
            groups=in_channels,
            dilation=2,
            bias=False
        )
        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            dilation=1,
            bias=False
        )

    def forward(self, x):
        x = self.conv1(x) + self.conv2(x)
        x = self.conv3(x)
        return x


class DSBlock(nn.Module):
    """Deep Separable Block with Instance Norm and SE Attention."""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int
    ):
        super().__init__()
        
        # 使用 InstanceNorm2d 替代 BatchNorm2d 以提高跨域鲁棒性
        self.layers = nn.Sequential(
            DilatedConv2d(in_channels, out_channels, stride),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            SeparableConv2d(out_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm2d(out_channels, affine=True)
        )
        
        # 引入 SE Block 增强关键伪造特征通道
        self.se = SELayer(out_channels, reduction=8)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=True),
                nn.InstanceNorm2d(out_channels, affine=True)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.layers(x)
        out = self.se(out)  # Apply Channel Attention
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class MLDC_v2(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # 1. 物理差分核
        self.register_buffer('diff_kernels', self._get_diff_kernels())
        
        # 2. 投影层 - 这里的 InstanceNorm 已经符合我们的设计思路
        self.projection = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(32, out_channels, kernel_size=1, bias=False),
        )

    def _get_diff_kernels(self):
        kernels = []
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1: continue
                k = torch.zeros(1, 1, 3, 3)
                k[0, 0, 1, 1] = 1
                k[0, 0, i, j] = -1
                kernels.append(k)
        kernels = torch.cat(kernels, dim=0)
        return kernels.repeat(3, 1, 1, 1)

    def forward(self, x):
        # MLDC 分支 (Learnable Detail)
        diffs = F.conv2d(x, self.diff_kernels, padding=1, groups=3)
        mldc_feat = self.projection(diffs)
        return mldc_feat


class LADOperator(nn.Module):
    """Local Adjacency Discrepancy map following the paper formulation.

    For each pixel, aggregate the bounded RGB-joint squared distance to the
    hollow 3x3 neighborhood:

        L(p) = mean_q tanh(||I(p) - I(q)||_2^2 / tau^2)

    The output is a single-channel naturalness / local energy map.
    """

    def __init__(self, kernel_size: int = 3, tau: float = 0.004):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric neighborhoods")
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.tau = float(tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ks = self.kernel_size
        patches = F.unfold(x, kernel_size=ks, padding=self.padding)
        patches = patches.view(b, c, ks * ks, h * w)

        center_idx = (ks * ks) // 2
        center = patches[:, :, center_idx, :].unsqueeze(2)
        neighbors = torch.cat(
            [patches[:, :, :center_idx, :], patches[:, :, center_idx + 1 :, :]],
            dim=2,
        )

        diff = center - neighbors
        diff_sq = (diff * diff).sum(dim=1)
        tau2 = self.tau * self.tau
        lad = torch.tanh(diff_sq / tau2)
        lad = lad.mean(dim=1, keepdim=True)
        return lad.view(b, 1, h, w)


class LADMultiOperator(nn.Module):
    """Stack LAD maps computed at multiple local-difference scales.

    A single LAD ``tau`` can saturate or invert dataset-specific foreground /
    background contrast.  Keeping several bounded LAD maps preserves the paper
    operator while giving the trainable Ferret stem access to multiple local
    contrast scales.
    """

    def __init__(
        self,
        taus: tuple[float, ...] | list[float] = (0.016, 0.032, 0.064, 0.128),
        kernel_size: int = 3,
    ):
        super().__init__()
        parsed_taus = tuple(float(tau) for tau in taus)
        if not parsed_taus:
            raise ValueError("LADMultiOperator requires at least one tau")
        self.taus = parsed_taus
        self.operators = nn.ModuleList(
            [LADOperator(kernel_size=kernel_size, tau=tau) for tau in self.taus]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([operator(x) for operator in self.operators], dim=1)


class AdaptiveTauFusion(nn.Module):
    """Identity-initialized spatial reweighting for multi-tau LAD maps.

    Static concatenation of several LAD ``tau`` maps improved MagicBrush/SID
    but slightly hurt AutoSplice/CoCoGLIDE.  This module keeps all tau channels
    available to the existing Ferret stem while letting training learn a local
    per-channel multiplier.  It initializes to an exact identity multiplier so
    a resumed single-/multi-tau checkpoint is not perturbed at step zero.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int | None = None,
        max_delta: float = 0.5,
    ):
        super().__init__()
        channels = int(channels)
        if channels <= 0:
            raise ValueError("AdaptiveTauFusion requires a positive channel count")
        hidden = int(hidden_channels or max(channels * 4, 8))
        self.channels = channels
        self.max_delta = float(max_delta)
        if self.max_delta <= 0.0:
            raise ValueError("AdaptiveTauFusion max_delta must be positive")
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(maps)
        weights = 1.0 + self.max_delta * torch.tanh(logits)
        return maps * weights, weights


class MLDC_v2(nn.Module):
    """Legacy multi-directional local difference convolution operator.

    This is kept as an optional fallback/operator for SAM3 diagnostics.  The
    default remains the paper-style LAD path, so existing LAD checkpoints keep
    their architecture unless ``forensic_operator="mldc"`` is requested.
    """

    def __init__(self, out_channels: int = 3):
        super().__init__()
        self.register_buffer("diff_kernels", self._get_diff_kernels())
        self.projection = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(32, int(out_channels), kernel_size=1, bias=False),
        )

    @staticmethod
    def _get_diff_kernels() -> torch.Tensor:
        kernels = []
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                kernel = torch.zeros(1, 1, 3, 3)
                kernel[0, 0, 1, 1] = 1
                kernel[0, 0, i, j] = -1
                kernels.append(kernel)
        return torch.cat(kernels, dim=0).repeat(3, 1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diffs = F.conv2d(x, self.diff_kernels, padding=1, groups=3)
        return self.projection(diffs)


class FerretBackbone(nn.Module):
    """FerretNet backbone for forensic feature extraction.
    Modified to use Instance Normalization and Learnable Mask Compression.
    """
    
    def __init__(
            self,
            in_channels: int = 3,
            dim: int = 96,
            depths: List[int] = [2,2],
            lad_tau: float = 0.004,
            coarse_prompt_head: str = "mask_compressor",
            coarse_prompt_hidden: int | None = None,
            coarse_prompt_dropout: float = 0.0,
            coarse_prompt_gate_init: float = 0.02,
            coarse_prompt_gate_max: float = 1.0,
            coarse_prompt_area_bias: bool = False,
            coarse_prompt_signed_residual_max_delta: float = 0.5,
            coarse_prompt_unet_gate_init: float | None = None,
            coarse_prompt_unet_gate_max: float | None = None,
            coarse_prompt_unet_signed_residual_max_delta: float | None = None,
            forensic_operator: str = "lad",
            lad_multi_taus: tuple[float, ...] | list[float] | None = None,
            mask_compressor_kernel_size: int = 3,
            mask_compressor_output: str = "logits",
            legacy_logit_head: bool = False,
    ):
        super().__init__()
        base_dim = int(dim)
        self.dim = base_dim
        self.coarse_prompt_head = str(coarse_prompt_head)
        self.coarse_prompt_dropout = float(coarse_prompt_dropout)
        self.coarse_prompt_gate_init = float(coarse_prompt_gate_init)
        self.coarse_prompt_gate_max = float(coarse_prompt_gate_max)
        self.coarse_prompt_area_bias = bool(coarse_prompt_area_bias)
        self.coarse_prompt_signed_residual_max_delta = float(coarse_prompt_signed_residual_max_delta)
        self.coarse_prompt_unet_gate_init = (
            float(coarse_prompt_gate_init)
            if coarse_prompt_unet_gate_init is None
            else float(coarse_prompt_unet_gate_init)
        )
        self.coarse_prompt_unet_gate_max = (
            float(coarse_prompt_gate_max)
            if coarse_prompt_unet_gate_max is None
            else float(coarse_prompt_unet_gate_max)
        )
        self.coarse_prompt_unet_signed_residual_max_delta = (
            float(coarse_prompt_signed_residual_max_delta)
            if coarse_prompt_unet_signed_residual_max_delta is None
            else float(coarse_prompt_unet_signed_residual_max_delta)
        )
        self.forensic_operator = str(forensic_operator).lower()
        self.mask_compressor_kernel_size = int(mask_compressor_kernel_size)
        self.mask_compressor_output = str(mask_compressor_output).lower()
        self.legacy_logit_head = bool(legacy_logit_head)
        if self.mask_compressor_kernel_size not in {1, 3}:
            raise ValueError("mask_compressor_kernel_size must be 1 or 3")
        if self.mask_compressor_output not in {"logits", "sigmoid"}:
            raise ValueError("mask_compressor_output must be 'logits' or 'sigmoid'")
        self._last_dense_prompt_gate = None
        self._last_dense_prompt_area_bias = None
        self._last_dense_prompt_fg_gate = None
        self._last_dense_prompt_bg_gate = None
        self._last_dense_prompt_core_gate = None
        self._last_dense_prompt_fg_residual = None
        self._last_dense_prompt_bg_residual = None
        self._last_dense_prompt_core_residual = None
        self._last_dense_prompt_small_gate = None
        self._last_dense_prompt_signed_delta = None
        self._last_dense_prompt_signed_gate = None
        self._last_dense_prompt_pre_unet = None
        self._last_dense_prompt_unet_delta = None
        self._last_dense_prompt_unet_gate = None
        self._last_lad_tau_weights = None
        self._last_hybrid_mldc_gate = None
        
        if self.forensic_operator == "lad":
            # Paper-style Local Adjacency Discrepancy extraction.
            self.lad = LADOperator(kernel_size=3, tau=lad_tau)
            operator_channels = 1
        elif self.forensic_operator in {"rgb", "image", "identity"}:
            # Legacy SAM2 checkpoints from early experiments used the raw RGB
            # image directly as the forensic stream input.
            operator_channels = in_channels
        elif self.forensic_operator == "lad_multi":
            taus = tuple(float(tau) for tau in (lad_multi_taus or (0.016, 0.032, 0.064, 0.128)))
            self.lad_multi = LADMultiOperator(kernel_size=3, taus=taus)
            operator_channels = len(taus)
        elif self.forensic_operator == "mldc":
            self.mldc = MLDC_v2(out_channels=3)
            operator_channels = 3
        elif self.forensic_operator == "lad_mldc_hybrid":
            # Keep LAD-multi as the checkpoint-compatible main stream.  MLDC
            # is injected later into the high-resolution feature tensor through
            # a zero-output projection, so an old LAD-multi checkpoint starts
            # with the same effective features while the MLDC projection can
            # receive gradients immediately through the non-zero sigmoid gate.
            taus = tuple(float(tau) for tau in (lad_multi_taus or (0.016, 0.032, 0.064, 0.128)))
            self.lad_multi = LADMultiOperator(kernel_size=3, taus=taus)
            self.hybrid_mldc = MLDC_v2(out_channels=3)
            operator_channels = len(taus)
            self.hybrid_mldc_high_proj = nn.Sequential(
                nn.Conv2d(3, base_dim // 2, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(base_dim // 2, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_dim // 2, base_dim // 2, kernel_size=1, bias=True),
            )
            hybrid_final = self.hybrid_mldc_high_proj[-1]
            nn.init.normal_(hybrid_final.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(hybrid_final.bias)
            self.hybrid_mldc_gate = nn.Conv2d(base_dim // 2, 1, kernel_size=3, padding=1)
            nn.init.zeros_(self.hybrid_mldc_gate.weight)
            nn.init.zeros_(self.hybrid_mldc_gate.bias)
        else:
            raise ValueError(f"Unsupported forensic_operator={forensic_operator!r}")

        if self.coarse_prompt_head in {
            "adaptive_tau_fusion_multiscale",
            "adaptive_detail_guided_signed_tribranch_multiscale",
            "precision_recall_adaptive_prompt_head",
            "uncertainty_guided_precision_recall_prompt_head",
            "contextual_highres_precision_recall_prompt_head",
            "fpn_highres_precision_recall_prompt_head",
            "direct_signed_highres_prompt_head",
            "unet_highres_prompt_head",
            "unet_residual_only_prompt_head",
        }:
            if self.forensic_operator not in {"lad_multi", "lad_mldc_hybrid"}:
                raise ValueError(
                    f"coarse_prompt_head={self.coarse_prompt_head!r} requires forensic_operator='lad_multi' or 'lad_mldc_hybrid'"
                )
            self.lad_tau_fusion = AdaptiveTauFusion(operator_channels)
        
        # Initial conv layers - Replace BN with IN
        self.cbr1 = nn.Sequential(
            nn.Conv2d(operator_channels, self.dim // 2, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(self.dim // 2, affine=True),
            nn.ReLU(inplace=True),
        )
        self.cbr2 = nn.Sequential(
            nn.Conv2d(self.dim // 2, self.dim, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(self.dim, affine=True),
            nn.ReLU(inplace=True),
        )
        
        # Feature extractor following original FerretNet architecture
        # But DSBlocks now internally use IN and SE
        depths = [2, 2] 
        self.feature = nn.Sequential()
        for depth in depths:
            blocks = nn.Sequential(
                DSBlock(self.dim, self.dim * 2, stride=2),
                *[DSBlock(self.dim * 2, self.dim * 2, stride=1) for _ in range(depth - 1)],
            )
            self.feature.append(blocks)
            self.dim = self.dim * 2
            
        # Final projection to align channels
        self.final_conv = nn.Conv2d(self.dim, self.dim, 1, 1, bias=False)
        self.output_dim = self.dim
        
        # [NEW] Learnable Mask Compressor
        # Replaces the old CAM/mean-based approach.
        # Outputs raw mask logits, not sigmoid probabilities.  SAM/SAM3 dense
        # mask prompts and BCEWithLogits supervision both expect logit-like
        # values; bounding for numerical safety is handled at the prompt call.
        mask_layers: list[nn.Module] = [
            nn.Conv2d(
                self.dim,
                self.dim // 4,
                kernel_size=self.mask_compressor_kernel_size,
                padding=self.mask_compressor_kernel_size // 2,
            ),
            nn.InstanceNorm2d(self.dim // 4, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 4, 1, 1),
        ]
        if self.mask_compressor_output == "sigmoid":
            mask_layers.append(nn.Sigmoid())
        self.mask_compressor = nn.Sequential(*mask_layers)

        if self.coarse_prompt_head == "mask_compressor":
            self.prompt_head = None
        elif self.coarse_prompt_head in {
            "multiscale",
            "split_multiscale",
            "gated_split_multiscale",
            "gated_split_multiscale_highres",
            "adaptive_tau_fusion_multiscale",
            "dual_branch_multiscale",
            "signed_tribranch_multiscale",
            "signed_tribranch_multiscale_highres",
            "detail_guided_signed_tribranch_multiscale",
            "adaptive_detail_guided_signed_tribranch_multiscale",
            "precision_recall_adaptive_prompt_head",
            "uncertainty_guided_precision_recall_prompt_head",
            "contextual_highres_precision_recall_prompt_head",
            "fpn_highres_precision_recall_prompt_head",
            "direct_signed_highres_prompt_head",
            "unet_highres_prompt_head",
            "unet_residual_only_prompt_head",
        }:
            hidden = int(coarse_prompt_hidden or max(self.output_dim // 4, 32))
            self.prompt_head_final_proj = nn.Sequential(
                nn.Conv2d(self.output_dim, hidden, kernel_size=1, bias=False),
                nn.InstanceNorm2d(hidden, affine=True),
                nn.ReLU(inplace=True),
            )
            self.prompt_head_mid_proj = nn.Sequential(
                nn.Conv2d(base_dim, hidden, kernel_size=1, bias=False),
                nn.InstanceNorm2d(hidden, affine=True),
            )
            self.prompt_head_high_proj = nn.Sequential(
                nn.Conv2d(base_dim // 2, hidden, kernel_size=1, bias=False),
                nn.InstanceNorm2d(hidden, affine=True),
            )
            if self.coarse_prompt_head in {
                "dual_branch_multiscale",
                "signed_tribranch_multiscale",
                "signed_tribranch_multiscale_highres",
                "detail_guided_signed_tribranch_multiscale",
                "adaptive_detail_guided_signed_tribranch_multiscale",
                "precision_recall_adaptive_prompt_head",
                "uncertainty_guided_precision_recall_prompt_head",
                "contextual_highres_precision_recall_prompt_head",
                "fpn_highres_precision_recall_prompt_head",
                "direct_signed_highres_prompt_head",
                "unet_highres_prompt_head",
                "unet_residual_only_prompt_head",
            }:
                if self.coarse_prompt_head in {
                    "detail_guided_signed_tribranch_multiscale",
                    "adaptive_detail_guided_signed_tribranch_multiscale",
                    "precision_recall_adaptive_prompt_head",
                    "uncertainty_guided_precision_recall_prompt_head",
                    "contextual_highres_precision_recall_prompt_head",
                    "fpn_highres_precision_recall_prompt_head",
                    "direct_signed_highres_prompt_head",
                    "unet_highres_prompt_head",
                    "unet_residual_only_prompt_head",
                }:
                    self.prompt_head_detail_proj = nn.Sequential(
                        nn.Conv2d(operator_channels, hidden, kernel_size=3, padding=1, bias=False),
                        nn.InstanceNorm2d(hidden, affine=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(hidden, hidden, kernel_size=1, bias=True),
                    )
                    detail_final = self.prompt_head_detail_proj[-1]
                    nn.init.zeros_(detail_final.weight)
                    nn.init.zeros_(detail_final.bias)
                self.prompt_head_fuse = nn.Sequential(
                    nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
                    nn.InstanceNorm2d(hidden, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=self.coarse_prompt_dropout),
                )
                if self.coarse_prompt_head in {
                    "precision_recall_adaptive_prompt_head",
                    "uncertainty_guided_precision_recall_prompt_head",
                    "contextual_highres_precision_recall_prompt_head",
                    "fpn_highres_precision_recall_prompt_head",
                    "direct_signed_highres_prompt_head",
                    "unet_highres_prompt_head",
                    "unet_residual_only_prompt_head",
                }:
                    # Keep the legacy signed-tribranch parameter names so an
                    # R170/R212 checkpoint loads the old recall/suppression
                    # endpoints and their gates exactly.  The new router then
                    # learns how to rebalance those endpoints locally instead
                    # of throwing away the best existing prompt generator.
                    self.prompt_head_fg = nn.Conv2d(hidden, 1, kernel_size=1)
                    self.prompt_head_bg = nn.Conv2d(hidden, 1, kernel_size=1)
                    self.prompt_head_core = nn.Conv2d(hidden, 1, kernel_size=1)
                    if self.coarse_prompt_gate_max <= 0.0:
                        raise ValueError("coarse_prompt_gate_max must be positive")
                    init_ratio = min(
                        max(self.coarse_prompt_gate_init / self.coarse_prompt_gate_max, 1e-6),
                        1.0 - 1e-6,
                    )
                    init_bias = torch.logit(torch.tensor(init_ratio)).item()
                    self.prompt_head_fg_gate = nn.Conv2d(hidden, 1, kernel_size=3, padding=1)
                    self.prompt_head_bg_gate = nn.Conv2d(hidden, 1, kernel_size=3, padding=1)
                    self.prompt_head_core_gate = nn.Conv2d(hidden, 1, kernel_size=3, padding=1)
                    for gate_head in (
                        self.prompt_head_fg_gate,
                        self.prompt_head_bg_gate,
                        self.prompt_head_core_gate,
                    ):
                        nn.init.zeros_(gate_head.weight)
                        if gate_head.bias is not None:
                            gate_head.bias.data.fill_(init_bias)
                    self.prompt_head_router_gate = nn.Conv2d(hidden, 1, kernel_size=3, padding=1)
                    nn.init.zeros_(self.prompt_head_router_gate.weight)
                    if self.prompt_head_router_gate.bias is not None:
                        self.prompt_head_router_gate.bias.data.zero_()
                    if self.coarse_prompt_head == "uncertainty_guided_precision_recall_prompt_head":
                        self.prompt_head_uncertainty_proj = nn.Sequential(
                            nn.Conv2d(hidden + 3, hidden, kernel_size=3, padding=1, bias=False),
                            nn.InstanceNorm2d(hidden, affine=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(hidden, hidden, kernel_size=1, bias=True),
                        )
                        uncertainty_final = self.prompt_head_uncertainty_proj[-1]
                        nn.init.zeros_(uncertainty_final.weight)
                        nn.init.zeros_(uncertainty_final.bias)
                        self.prompt_head_uncertainty_balance_gate = nn.Conv2d(
                            hidden, 1, kernel_size=3, padding=1
                        )
                        nn.init.zeros_(self.prompt_head_uncertainty_balance_gate.weight)
                        if self.prompt_head_uncertainty_balance_gate.bias is not None:
                            self.prompt_head_uncertainty_balance_gate.bias.data.zero_()
                    if self.coarse_prompt_head == "contextual_highres_precision_recall_prompt_head":
                        # Add raw-image and MLDC local-difference context to
                        # the high-resolution prompt feature tensor.  The
                        # final projections are zero-initialized so old
                        # R212/R216-style checkpoints resume with near-identical
                        # dense prompts, then training can learn whether RGB or
                        # MLDC detail helps MagicBrush/SID localization.
                        self.prompt_head_rgb_proj = nn.Sequential(
                            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
                            nn.InstanceNorm2d(hidden, affine=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(hidden, hidden, kernel_size=1, bias=True),
                        )
                        rgb_final = self.prompt_head_rgb_proj[-1]
                        nn.init.zeros_(rgb_final.weight)
                        nn.init.zeros_(rgb_final.bias)
                        self.prompt_head_mldc = MLDC_v2(out_channels=3)
                        self.prompt_head_mldc_proj = nn.Sequential(
                            nn.Conv2d(3, hidden, kernel_size=3, padding=1, bias=False),
                            nn.InstanceNorm2d(hidden, affine=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(hidden, hidden, kernel_size=1, bias=True),
                        )
                        mldc_final = self.prompt_head_mldc_proj[-1]
                        nn.init.zeros_(mldc_final.weight)
                        nn.init.zeros_(mldc_final.bias)
                    if self.coarse_prompt_head in {
                        "fpn_highres_precision_recall_prompt_head",
                        "direct_signed_highres_prompt_head",
                        "unet_highres_prompt_head",
                    }:
                        # High-resolution local refinement path for small
                        # masks.  Final projections are tiny so R244/R212
                        # checkpoints start near the old prompt generator, but
                        # gradients immediately reach all FPN/local modules.
                        self.prompt_head_fpn_mid_refine = nn.Sequential(
                            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
                            nn.InstanceNorm2d(hidden, affine=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(hidden, hidden, kernel_size=1, bias=True),
                        )
                        self.prompt_head_fpn_high_refine = nn.Sequential(
                            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
                            nn.InstanceNorm2d(hidden, affine=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(hidden, hidden, kernel_size=1, bias=True),
                        )
                        self.prompt_head_small_detail_proj = nn.Sequential(
                            nn.Conv2d(operator_channels, hidden, kernel_size=3, padding=1, bias=False),
                            nn.InstanceNorm2d(hidden, affine=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(hidden, hidden, kernel_size=1, bias=True),
                        )
                        for module in (
                            self.prompt_head_fpn_mid_refine,
                            self.prompt_head_fpn_high_refine,
                            self.prompt_head_small_detail_proj,
                        ):
                            final = module[-1]
                            nn.init.normal_(final.weight, mean=0.0, std=1e-4)
                            nn.init.zeros_(final.bias)
                        self.prompt_head_small_gate = nn.Conv2d(hidden, 1, kernel_size=3, padding=1)
                        nn.init.zeros_(self.prompt_head_small_gate.weight)
                        if self.prompt_head_small_gate.bias is not None:
                            self.prompt_head_small_gate.bias.data.fill_(init_bias)
                    if self.coarse_prompt_head == "direct_signed_highres_prompt_head":
                        # Direct signed dense-prompt correction.  Unlike the
                        # feature-only FPN small path, this branch predicts a
                        # bounded positive/negative logit delta and applies it
                        # directly to the dense prompt sent to SAM/SAM3.  The
                        # final delta projection starts tiny, not exactly zero:
                        # behavior is near identity, but gradients also reach
                        # the spatial gate from the first step.
                        self.prompt_head_signed_delta = nn.Sequential(
                            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
                            nn.InstanceNorm2d(hidden, affine=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
                        )
                        signed_delta_final = self.prompt_head_signed_delta[-1]
                        nn.init.normal_(signed_delta_final.weight, mean=0.0, std=1e-4)
                        nn.init.zeros_(signed_delta_final.bias)
                        self.prompt_head_signed_gate = nn.Conv2d(hidden, 1, kernel_size=3, padding=1)
                        nn.init.zeros_(self.prompt_head_signed_gate.weight)
                        if self.prompt_head_signed_gate.bias is not None:
                            self.prompt_head_signed_gate.bias.data.fill_(init_bias)
                    if self.coarse_prompt_head in {"unet_highres_prompt_head", "unet_residual_only_prompt_head"}:
                        # Stronger high-resolution dense-prompt decoder.  It
                        # keeps the old precision/recall/core endpoints for
                        # checkpoint compatibility, but additionally injects
                        # RGB and MLDC context into a U-Net-like local branch
                        # that predicts a bounded signed dense-logit residual.
                        self.prompt_head_unet_rgb_proj = nn.Sequential(
                            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
                            nn.InstanceNorm2d(hidden, affine=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(hidden, hidden, kernel_size=1, bias=True),
                        )
                        rgb_final = self.prompt_head_unet_rgb_proj[-1]
                        nn.init.normal_(rgb_final.weight, mean=0.0, std=1e-4)
                        nn.init.zeros_(rgb_final.bias)
                        self.prompt_head_unet_mldc = MLDC_v2(out_channels=3)
                        self.prompt_head_unet_mldc_proj = nn.Sequential(
                            nn.Conv2d(3, hidden, kernel_size=3, padding=1, bias=False),
                            nn.InstanceNorm2d(hidden, affine=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(hidden, hidden, kernel_size=1, bias=True),
                        )
                        mldc_final = self.prompt_head_unet_mldc_proj[-1]
                        nn.init.normal_(mldc_final.weight, mean=0.0, std=1e-4)
                        nn.init.zeros_(mldc_final.bias)
                        self.prompt_head_unet_refine = nn.Sequential(
                            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
                            nn.InstanceNorm2d(hidden, affine=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
                            nn.InstanceNorm2d(hidden, affine=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(hidden, hidden, kernel_size=1, bias=True),
                        )
                        refine_final = self.prompt_head_unet_refine[-1]
                        nn.init.normal_(refine_final.weight, mean=0.0, std=1e-4)
                        nn.init.zeros_(refine_final.bias)
                        self.prompt_head_unet_delta = nn.Sequential(
                            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
                            nn.InstanceNorm2d(hidden, affine=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
                        )
                        delta_final = self.prompt_head_unet_delta[-1]
                        nn.init.normal_(delta_final.weight, mean=0.0, std=1e-4)
                        nn.init.zeros_(delta_final.bias)
                        self.prompt_head_unet_gate = nn.Conv2d(hidden, 1, kernel_size=3, padding=1)
                        nn.init.zeros_(self.prompt_head_unet_gate.weight)
                        if self.prompt_head_unet_gate.bias is not None:
                            if self.coarse_prompt_unet_gate_max <= 0.0:
                                raise ValueError("coarse_prompt_unet_gate_max must be positive")
                            unet_init_ratio = min(
                                max(
                                    self.coarse_prompt_unet_gate_init
                                    / self.coarse_prompt_unet_gate_max,
                                    1e-6,
                                ),
                                1.0 - 1e-6,
                            )
                            unet_init_bias = torch.logit(torch.tensor(unet_init_ratio)).item()
                            self.prompt_head_unet_gate.bias.data.fill_(unet_init_bias)
                    branch_heads = [
                        self.prompt_head_fg,
                        self.prompt_head_bg,
                        self.prompt_head_core,
                    ]
                    if self.coarse_prompt_area_bias:
                        self.prompt_head_area_bias = nn.Linear(5, 1)
                        nn.init.zeros_(self.prompt_head_area_bias.weight)
                        nn.init.zeros_(self.prompt_head_area_bias.bias)
                    else:
                        self.prompt_head_area_bias = None
                else:
                    self.prompt_head_fg = nn.Conv2d(hidden, 1, kernel_size=1)
                    self.prompt_head_bg = nn.Conv2d(hidden, 1, kernel_size=1)
                    branch_heads = [self.prompt_head_fg, self.prompt_head_bg]
                if self.coarse_prompt_head in {
                    "signed_tribranch_multiscale",
                    "signed_tribranch_multiscale_highres",
                    "detail_guided_signed_tribranch_multiscale",
                    "adaptive_detail_guided_signed_tribranch_multiscale",
                }:
                    self.prompt_head_core = nn.Conv2d(hidden, 1, kernel_size=1)
                    branch_heads.append(self.prompt_head_core)
                    if self.coarse_prompt_gate_max <= 0.0:
                        raise ValueError("coarse_prompt_gate_max must be positive")
                    init_ratio = min(
                        max(self.coarse_prompt_gate_init / self.coarse_prompt_gate_max, 1e-6),
                        1.0 - 1e-6,
                    )
                    init_bias = torch.logit(torch.tensor(init_ratio)).item()
                    self.prompt_head_fg_gate = nn.Conv2d(hidden, 1, kernel_size=3, padding=1)
                    self.prompt_head_bg_gate = nn.Conv2d(hidden, 1, kernel_size=3, padding=1)
                    self.prompt_head_core_gate = nn.Conv2d(hidden, 1, kernel_size=3, padding=1)
                    for gate_head in (
                        self.prompt_head_fg_gate,
                        self.prompt_head_bg_gate,
                        self.prompt_head_core_gate,
                    ):
                        nn.init.zeros_(gate_head.weight)
                        if gate_head.bias is not None:
                            gate_head.bias.data.fill_(init_bias)
                    if self.coarse_prompt_area_bias:
                        self.prompt_head_area_bias = nn.Linear(5, 1)
                        nn.init.zeros_(self.prompt_head_area_bias.weight)
                        nn.init.zeros_(self.prompt_head_area_bias.bias)
                    else:
                        self.prompt_head_area_bias = None
                for branch_head in branch_heads:
                    nn.init.zeros_(branch_head.weight)
                    if branch_head.bias is not None:
                        nn.init.zeros_(branch_head.bias)
            else:
                self.prompt_head_fuse = nn.Sequential(
                    nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
                    nn.InstanceNorm2d(hidden, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=self.coarse_prompt_dropout),
                    nn.Conv2d(hidden, 1, kernel_size=1),
                )
                final_residual = self.prompt_head_fuse[-1]
                nn.init.zeros_(final_residual.weight)
                if final_residual.bias is not None:
                    nn.init.zeros_(final_residual.bias)
            if self.coarse_prompt_head in {
                "gated_split_multiscale",
                "gated_split_multiscale_highres",
                "adaptive_tau_fusion_multiscale",
            }:
                if self.coarse_prompt_gate_max <= 0.0:
                    raise ValueError("coarse_prompt_gate_max must be positive")
                gate_hidden = max(hidden // 2, 8)
                self.prompt_head_gate = nn.Sequential(
                    nn.Linear(self.output_dim + 5, gate_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(gate_hidden, 1),
                )
                with torch.no_grad():
                    # Preserve identity behavior through the final gate output
                    # while keeping the feature layer learnable from step one.
                    # The dense residual starts at zero, so a tiny non-zero
                    # final gate weight is still identity-preserving but lets
                    # the gate vary by sample as soon as the residual moves.
                    gate_final = self.prompt_head_gate[-1]
                    nn.init.normal_(gate_final.weight, mean=0.0, std=1e-3)
                    if gate_final.bias is not None:
                        nn.init.zeros_(gate_final.bias)
                    init_ratio = min(
                        max(self.coarse_prompt_gate_init / self.coarse_prompt_gate_max, 1e-6),
                        1.0 - 1e-6,
                    )
                    gate_final.bias.fill_(
                        torch.logit(torch.tensor(init_ratio)).item()
                    )
                if self.coarse_prompt_area_bias:
                    self.prompt_head_area_bias = nn.Linear(5, 1)
                    nn.init.zeros_(self.prompt_head_area_bias.weight)
                    nn.init.zeros_(self.prompt_head_area_bias.bias)
                else:
                    self.prompt_head_area_bias = None
            elif self.coarse_prompt_head == "dual_branch_multiscale":
                if self.coarse_prompt_gate_max <= 0.0:
                    raise ValueError("coarse_prompt_gate_max must be positive")
                gate_hidden = max(hidden // 2, 8)
                self.prompt_head_fg_gate = nn.Sequential(
                    nn.Linear(self.output_dim + 5, gate_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(gate_hidden, 1),
                )
                self.prompt_head_bg_gate = nn.Sequential(
                    nn.Linear(self.output_dim + 5, gate_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(gate_hidden, 1),
                )
                init_ratio = min(
                    max(self.coarse_prompt_gate_init / self.coarse_prompt_gate_max, 1e-6),
                    1.0 - 1e-6,
                )
                init_bias = torch.logit(torch.tensor(init_ratio)).item()
                with torch.no_grad():
                    for gate_net in (self.prompt_head_fg_gate, self.prompt_head_bg_gate):
                        gate_final = gate_net[-1]
                        nn.init.normal_(gate_final.weight, mean=0.0, std=1e-3)
                        if gate_final.bias is not None:
                            gate_final.bias.fill_(init_bias)
                if self.coarse_prompt_area_bias:
                    self.prompt_head_area_bias = nn.Linear(5, 1)
                    nn.init.zeros_(self.prompt_head_area_bias.weight)
                    nn.init.zeros_(self.prompt_head_area_bias.bias)
                else:
                    self.prompt_head_area_bias = None
            # Keep a simple marker for introspection without registering the
            # same submodules twice in state_dict/checkpoint missing-key logs.
            self.prompt_head = None
        else:
            raise ValueError(f"Unsupported coarse_prompt_head={self.coarse_prompt_head!r}")
        
        # Classification head (MLP on pooled features)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        hidden_dim = max(self.dim // 2, 64)
        if self.legacy_logit_head:
            self.logit = nn.Sequential(
                nn.Dropout(0.2, inplace=True),
                nn.Linear(self.dim, 1),
            )
        else:
            self.logit = nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1)
            )
        # self.logit = nn.Sequential(
        #     nn.Dropout(0.2, inplace=True),
        #     nn.Linear(self.dim, 1)
        # )
        
    def forward(
        self,
        x,
        return_dense_prompt: bool = False,
        return_forensic_pyramid: bool = False,
    ):
        """Forward pass for forensic feature extraction.
        
        Args:
            x: Input image tensor of shape [B, 3, H, W]
            
        Returns:
            forensic_features: Full forensic feature map [B, C, H/8, W/8]
            coarse_mask: Compressed 1-channel raw mask logits for SAM Prompt [B, 1, H/4, W/4]
            detection_logit: Global detection score [B, 1]
        """
        input_h, input_w = x.shape[-2:]
        image_context = x

        # 1. Compute forensic detail map.
        hybrid_mldc_map = None
        if self.forensic_operator == "lad":
            x = self.lad(x)
            self._last_lad_tau_weights = None
            self._last_hybrid_mldc_gate = None
        elif self.forensic_operator == "lad_multi":
            x = self.lad_multi(x)
            if hasattr(self, "lad_tau_fusion"):
                x, tau_weights = self.lad_tau_fusion(x)
                self._last_lad_tau_weights = tau_weights
            else:
                self._last_lad_tau_weights = None
            self._last_hybrid_mldc_gate = None
        elif self.forensic_operator == "mldc":
            x = self.mldc(x)
            self._last_lad_tau_weights = None
            self._last_hybrid_mldc_gate = None
        elif self.forensic_operator in {"rgb", "image", "identity"}:
            self._last_lad_tau_weights = None
            self._last_hybrid_mldc_gate = None
        elif self.forensic_operator == "lad_mldc_hybrid":
            hybrid_mldc_map = self.hybrid_mldc(image_context)
            x = self.lad_multi(x)
            if hasattr(self, "lad_tau_fusion"):
                x, tau_weights = self.lad_tau_fusion(x)
                self._last_lad_tau_weights = tau_weights
            else:
                self._last_lad_tau_weights = None
        else:
            raise RuntimeError(f"Unsupported forensic_operator={self.forensic_operator!r}")
        detail_map = x
        
        # 2. Initial processing
        high_feature = self.cbr1(x)
        if self.forensic_operator == "lad_mldc_hybrid":
            if hybrid_mldc_map is None:
                raise RuntimeError("lad_mldc_hybrid expected an MLDC detail map")
            hybrid_context = self.hybrid_mldc_high_proj(hybrid_mldc_map)
            if hybrid_context.shape[-2:] != high_feature.shape[-2:]:
                hybrid_context = F.interpolate(
                    hybrid_context,
                    size=high_feature.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            hybrid_gate = torch.sigmoid(self.hybrid_mldc_gate(high_feature))
            high_feature = F.relu(
                high_feature + hybrid_gate.to(dtype=high_feature.dtype) * hybrid_context.to(dtype=high_feature.dtype),
                inplace=False,
            )
            self._last_hybrid_mldc_gate = hybrid_gate.detach()
        mid_feature = self.cbr2(high_feature)
        
        # 3. Feature extraction through Backbone
        feature_map = self.feature(mid_feature)
        
        # fc_layer = self.logit[1]  # Second layer is Linear
        # weight = fc_layer.weight  # shape (num_classes, in_channels)
        
        # # Extract weight vector (for binary classification, num_classes=1)
        # class_weight = weight[0]  # shape (in_channels,)
        
        # # Compute weighted feature map: apply class weights to each channel
        # # feature_map: (B, C, H, W) * class_weight: (1, C, 1, 1) -> (B, C, H, W)
        # b, c, h, w = feature_map.shape
        
        # forensic_features = (class_weight.view(1, c, 1, 1) * feature_map)
        # forensic_features = F.relu(forensic_features)
        
        # forensic_features_mask = forensic_features.mean(dim=1)
        # forensic_features_mask = forensic_features_mask.unsqueeze(1)  # Add channel dimension: (B, 1, H, W)
        
        # prompt_h = 256
        # prompt_w = 256
        # forensic_features_mask = F.interpolate(
        #     forensic_features_mask,
        #     size=(prompt_h, prompt_w),
        #     mode='bilinear',
        #     align_corners=False
        # )
        
        # # ------------------------------
        # # Generate enhanced outputs
        # # ------------------------------
        
        # # 1. Generate detection logit
        # x_pool = self.avg_pool(feature_map)  # [B, C, 1, 1]
        # x_flat = x_pool.view(x_pool.size(0), -1)  # [B, C]
        # detection_logit = self.logit(x_flat)  # [B, 1]
        
        # return forensic_features, forensic_features_mask, detection_logit
        
        
        feature_map = self.final_conv(feature_map) # [B, 384, H/8, W/8] for default dim=96
        
        # 4. Prepare Forensic Features for Adapter
        # Direct usage of feature map, preserving all channel information
        forensic_features = feature_map 
        
        # 5. Generate coarse mask logits for supervision and, optionally, a
        # separate dense prompt logit map for SAM.  In split mode, the
        # supervised coarse mask remains the stable mask_compressor output
        # while the dense prompt can learn SAM-specific residual semantics.
        coarse_logits, dense_prompt_logits = self._make_coarse_and_dense_prompt_logits(
            feature_map=feature_map,
            high_feature=high_feature,
            mid_feature=mid_feature,
            detail_map=detail_map,
            image_context=image_context,
        ) # [B, 1, H/8, W/8] or higher-res depending prompt head
        
        prompt_h = 256
        prompt_w = 256
        forensic_features_mask = F.interpolate(
            coarse_logits,
            size=(prompt_h, prompt_w),
            mode='bilinear',
            align_corners=False
        )
        dense_prompt_mask = F.interpolate(
            dense_prompt_logits,
            size=(prompt_h, prompt_w),
            mode='bilinear',
            align_corners=False
        )
        
        # 6. Generate detection logit
        x_pool = self.avg_pool(feature_map)  # [B, C, 1, 1]
        x_flat = x_pool.view(x_pool.size(0), -1)  # [B, C]
        detection_logit = self.logit(x_flat)  # [B, 1]
        
        if return_forensic_pyramid:
            forensic_pyramid = [forensic_features, high_feature, mid_feature]
            if return_dense_prompt:
                return (
                    forensic_features,
                    forensic_features_mask,
                    detection_logit,
                    dense_prompt_mask,
                    forensic_pyramid,
                )
            return forensic_features, forensic_features_mask, detection_logit, forensic_pyramid
        if return_dense_prompt:
            return forensic_features, forensic_features_mask, detection_logit, dense_prompt_mask
        return forensic_features, forensic_features_mask, detection_logit
    
    def _make_coarse_and_dense_prompt_logits(
        self,
        feature_map: torch.Tensor,
        high_feature: torch.Tensor,
        mid_feature: torch.Tensor,
        detail_map: torch.Tensor | None = None,
        image_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate supervised coarse logits and SAM dense-prompt logits.

        The default path is the original R21-compatible mask compressor.  The
        ``multiscale`` path preserves the earlier experimental behavior where
        the residual changes both coarse supervision and SAM prompt logits.
        ``split_multiscale`` keeps coarse supervision on ``mask_compressor`` and
        applies the residual only to the dense prompt sent to SAM.
        """
        base_logits = self.mask_compressor(feature_map)
        if self.coarse_prompt_head == "mask_compressor":
            self._last_dense_prompt_gate = None
            self._last_dense_prompt_area_bias = None
            self._last_dense_prompt_fg_gate = None
            self._last_dense_prompt_bg_gate = None
            self._last_dense_prompt_core_gate = None
            self._last_dense_prompt_fg_residual = None
            self._last_dense_prompt_bg_residual = None
            self._last_dense_prompt_core_residual = None
            self._last_dense_prompt_small_gate = None
            self._last_dense_prompt_signed_delta = None
            self._last_dense_prompt_signed_gate = None
            self._last_dense_prompt_pre_unet = None
            self._last_dense_prompt_unet_delta = None
            self._last_dense_prompt_unet_gate = None
            return base_logits, base_logits

        final_prompt = self.prompt_head_final_proj(feature_map)
        prompt = F.interpolate(
            final_prompt,
            size=mid_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        prompt = F.relu(prompt + self.prompt_head_mid_proj(mid_feature), inplace=False)
        if self.coarse_prompt_head in {
            "fpn_highres_precision_recall_prompt_head",
            "direct_signed_highres_prompt_head",
            "unet_highres_prompt_head",
        }:
            prompt = F.relu(
                prompt + self.prompt_head_fpn_mid_refine(prompt),
                inplace=False,
            )
        prompt = F.interpolate(
            prompt,
            size=high_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        prompt = F.relu(prompt + self.prompt_head_high_proj(high_feature), inplace=False)
        if self.coarse_prompt_head in {
            "detail_guided_signed_tribranch_multiscale",
            "adaptive_detail_guided_signed_tribranch_multiscale",
            "precision_recall_adaptive_prompt_head",
            "uncertainty_guided_precision_recall_prompt_head",
            "contextual_highres_precision_recall_prompt_head",
            "fpn_highres_precision_recall_prompt_head",
            "direct_signed_highres_prompt_head",
            "unet_highres_prompt_head",
            "unet_residual_only_prompt_head",
        }:
            if detail_map is None:
                raise ValueError(f"{self.coarse_prompt_head} requires detail_map")
            detail_context = self.prompt_head_detail_proj(detail_map)
            if detail_context.shape[-2:] != prompt.shape[-2:]:
                detail_context = F.interpolate(
                    detail_context,
                    size=prompt.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            prompt = F.relu(prompt + detail_context, inplace=False)
            if self.coarse_prompt_head in {
                "fpn_highres_precision_recall_prompt_head",
                "direct_signed_highres_prompt_head",
                "unet_highres_prompt_head",
            }:
                small_context = self.prompt_head_small_detail_proj(detail_map)
                if small_context.shape[-2:] != prompt.shape[-2:]:
                    small_context = F.interpolate(
                        small_context,
                        size=prompt.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                small_gate = (
                    torch.sigmoid(self.prompt_head_small_gate(prompt))
                    * self.coarse_prompt_gate_max
                )
                prompt = F.relu(
                    prompt + small_gate.to(dtype=prompt.dtype) * small_context.to(dtype=prompt.dtype),
                    inplace=False,
                )
                prompt = F.relu(
                    prompt + self.prompt_head_fpn_high_refine(prompt),
                    inplace=False,
                )
                self._last_dense_prompt_small_gate = small_gate
            else:
                self._last_dense_prompt_small_gate = None
        if self.coarse_prompt_head == "contextual_highres_precision_recall_prompt_head":
            if image_context is None:
                raise ValueError(f"{self.coarse_prompt_head} requires image_context")
            rgb_context = self.prompt_head_rgb_proj(image_context)
            if rgb_context.shape[-2:] != prompt.shape[-2:]:
                rgb_context = F.interpolate(
                    rgb_context,
                    size=prompt.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            mldc_context = self.prompt_head_mldc(image_context)
            mldc_context = self.prompt_head_mldc_proj(mldc_context)
            if mldc_context.shape[-2:] != prompt.shape[-2:]:
                mldc_context = F.interpolate(
                    mldc_context,
                    size=prompt.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            prompt = F.relu(
                prompt
                + rgb_context.to(dtype=prompt.dtype)
                + mldc_context.to(dtype=prompt.dtype),
                inplace=False,
            )
        if self.coarse_prompt_head == "unet_highres_prompt_head":
            if image_context is None:
                raise ValueError(f"{self.coarse_prompt_head} requires image_context")
            rgb_context = self.prompt_head_unet_rgb_proj(image_context)
            if rgb_context.shape[-2:] != prompt.shape[-2:]:
                rgb_context = F.interpolate(
                    rgb_context,
                    size=prompt.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            unet_mldc_context = self.prompt_head_unet_mldc(image_context)
            unet_mldc_context = self.prompt_head_unet_mldc_proj(unet_mldc_context)
            if unet_mldc_context.shape[-2:] != prompt.shape[-2:]:
                unet_mldc_context = F.interpolate(
                    unet_mldc_context,
                    size=prompt.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            prompt = F.relu(
                prompt
                + rgb_context.to(dtype=prompt.dtype)
                + unet_mldc_context.to(dtype=prompt.dtype),
                inplace=False,
            )
            prompt = F.relu(
                prompt + self.prompt_head_unet_refine(prompt),
                inplace=False,
            )
        prompt_features = self.prompt_head_fuse(prompt)
        if self.coarse_prompt_head in {
            "precision_recall_adaptive_prompt_head",
            "uncertainty_guided_precision_recall_prompt_head",
            "contextual_highres_precision_recall_prompt_head",
            "fpn_highres_precision_recall_prompt_head",
            "direct_signed_highres_prompt_head",
            "unet_highres_prompt_head",
            "unet_residual_only_prompt_head",
        }:
            if self.coarse_prompt_head == "uncertainty_guided_precision_recall_prompt_head":
                base_context = F.interpolate(
                    base_logits,
                    size=prompt_features.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                base_prob = torch.sigmoid(base_context)
                base_uncertainty = base_prob * (1.0 - base_prob)
                if detail_map is None:
                    raise ValueError(f"{self.coarse_prompt_head} requires detail_map")
                detail_summary = detail_map.float().mean(dim=1, keepdim=True)
                if detail_summary.shape[-2:] != prompt_features.shape[-2:]:
                    detail_summary = F.interpolate(
                        detail_summary,
                        size=prompt_features.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                detail_summary = detail_summary.to(dtype=prompt_features.dtype)
                detail_dims = tuple(range(1, detail_summary.ndim))
                detail_summary = detail_summary - detail_summary.mean(dim=detail_dims, keepdim=True)
                uncertainty_context = torch.cat(
                    [
                        prompt_features,
                        base_context.to(dtype=prompt_features.dtype),
                        base_uncertainty.to(dtype=prompt_features.dtype),
                        detail_summary,
                    ],
                    dim=1,
                )
                prompt_features = F.relu(
                    prompt_features + self.prompt_head_uncertainty_proj(uncertainty_context),
                    inplace=False,
                )
            core_residual = torch.tanh(self.prompt_head_core(prompt_features)) * 2.0
            recall_residual = self._nonnegative_residual_magnitude(
                self.prompt_head_fg(prompt_features)
            )
            precision_residual = self._nonnegative_residual_magnitude(
                self.prompt_head_bg(prompt_features)
            )
            fg_gate = torch.sigmoid(self.prompt_head_fg_gate(prompt_features)) * self.coarse_prompt_gate_max
            bg_gate = torch.sigmoid(self.prompt_head_bg_gate(prompt_features)) * self.coarse_prompt_gate_max
            core_gate = torch.sigmoid(self.prompt_head_core_gate(prompt_features)) * self.coarse_prompt_gate_max
            router_logits = self.prompt_head_router_gate(prompt_features)
            if self.coarse_prompt_head == "uncertainty_guided_precision_recall_prompt_head":
                router_logits = router_logits + self.prompt_head_uncertainty_balance_gate(prompt_features)
            router_gate = torch.sigmoid(router_logits)
            recall_gate = fg_gate * (0.5 + router_gate)
            precision_gate = bg_gate * (1.5 - router_gate)

            stats = self._coarse_logit_stats(base_logits)
            area_bias = recall_residual.new_zeros((recall_residual.shape[0], 1, 1, 1))
            if self.prompt_head_area_bias is not None:
                area_bias = self.prompt_head_area_bias(stats.to(dtype=recall_residual.dtype)).view(-1, 1, 1, 1)

            core_for_dense = core_residual
            recall_for_dense = recall_residual
            precision_for_dense = precision_residual
            recall_gate_for_dense = recall_gate
            precision_gate_for_dense = precision_gate
            core_gate_for_dense = core_gate
            highres_contextual_prompt = (
                self.coarse_prompt_head == "contextual_highres_precision_recall_prompt_head"
                or self.coarse_prompt_head == "fpn_highres_precision_recall_prompt_head"
                or self.coarse_prompt_head == "direct_signed_highres_prompt_head"
            )
            dense_size = (
                recall_for_dense.shape[-2:]
                if highres_contextual_prompt
                else base_logits.shape[-2:]
            )
            if core_for_dense.shape[-2:] != dense_size:
                core_for_dense = F.interpolate(
                    core_for_dense,
                    size=dense_size,
                    mode="bilinear",
                    align_corners=False,
                )
            if recall_for_dense.shape[-2:] != dense_size:
                recall_for_dense = F.interpolate(
                    recall_for_dense,
                    size=dense_size,
                    mode="bilinear",
                    align_corners=False,
                )
            if precision_for_dense.shape[-2:] != dense_size:
                precision_for_dense = F.interpolate(
                    precision_for_dense,
                    size=dense_size,
                    mode="bilinear",
                    align_corners=False,
                )
            if recall_gate_for_dense.shape[-2:] != dense_size:
                recall_gate_for_dense = F.interpolate(
                    recall_gate_for_dense,
                    size=dense_size,
                    mode="bilinear",
                    align_corners=False,
                )
            if precision_gate_for_dense.shape[-2:] != dense_size:
                precision_gate_for_dense = F.interpolate(
                    precision_gate_for_dense,
                    size=dense_size,
                    mode="bilinear",
                    align_corners=False,
                )
            if core_gate_for_dense.shape[-2:] != dense_size:
                core_gate_for_dense = F.interpolate(
                    core_gate_for_dense,
                    size=dense_size,
                    mode="bilinear",
                    align_corners=False,
                )
            dense_base = base_logits
            if dense_base.shape[-2:] != dense_size:
                dense_base = F.interpolate(
                    dense_base,
                    size=dense_size,
                    mode="bilinear",
                    align_corners=False,
                )
            dense_logits = (
                dense_base
                + core_for_dense * core_gate_for_dense
                + recall_for_dense * recall_gate_for_dense
                - precision_for_dense * precision_gate_for_dense
                + area_bias
            )
            if self.coarse_prompt_head == "direct_signed_highres_prompt_head":
                signed_delta = (
                    torch.tanh(self.prompt_head_signed_delta(prompt_features))
                    * self.coarse_prompt_signed_residual_max_delta
                )
                signed_gate = (
                    torch.sigmoid(self.prompt_head_signed_gate(prompt_features))
                    * self.coarse_prompt_gate_max
                )
                signed_delta_for_dense = signed_delta
                signed_gate_for_dense = signed_gate
                if signed_delta_for_dense.shape[-2:] != dense_logits.shape[-2:]:
                    signed_delta_for_dense = F.interpolate(
                        signed_delta_for_dense,
                        size=dense_logits.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                if signed_gate_for_dense.shape[-2:] != dense_logits.shape[-2:]:
                    signed_gate_for_dense = F.interpolate(
                        signed_gate_for_dense,
                        size=dense_logits.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                dense_logits = dense_logits + signed_delta_for_dense * signed_gate_for_dense
                self._last_dense_prompt_signed_delta = signed_delta
                self._last_dense_prompt_signed_gate = signed_gate
            else:
                self._last_dense_prompt_signed_delta = None
                self._last_dense_prompt_signed_gate = None
            self._last_dense_prompt_pre_unet = (
                dense_logits if self.coarse_prompt_head in {"unet_highres_prompt_head", "unet_residual_only_prompt_head"} else None
            )
            if self.coarse_prompt_head in {"unet_highres_prompt_head", "unet_residual_only_prompt_head"}:
                unet_features = prompt_features
                if self.coarse_prompt_head == "unet_residual_only_prompt_head":
                    if image_context is None:
                        raise ValueError(f"{self.coarse_prompt_head} requires image_context")
                    rgb_context = self.prompt_head_unet_rgb_proj(image_context)
                    if rgb_context.shape[-2:] != unet_features.shape[-2:]:
                        rgb_context = F.interpolate(
                            rgb_context,
                            size=unet_features.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    unet_mldc_context = self.prompt_head_unet_mldc(image_context)
                    unet_mldc_context = self.prompt_head_unet_mldc_proj(unet_mldc_context)
                    if unet_mldc_context.shape[-2:] != unet_features.shape[-2:]:
                        unet_mldc_context = F.interpolate(
                            unet_mldc_context,
                            size=unet_features.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    unet_features = F.relu(
                        unet_features
                        + rgb_context.to(dtype=unet_features.dtype)
                        + unet_mldc_context.to(dtype=unet_features.dtype),
                        inplace=False,
                    )
                    unet_features = F.relu(
                        unet_features + self.prompt_head_unet_refine(unet_features),
                        inplace=False,
                    )
                unet_delta = (
                    torch.tanh(self.prompt_head_unet_delta(unet_features))
                    * self.coarse_prompt_unet_signed_residual_max_delta
                )
                unet_gate = (
                    torch.sigmoid(self.prompt_head_unet_gate(unet_features))
                    * self.coarse_prompt_unet_gate_max
                )
                unet_delta_for_dense = unet_delta
                unet_gate_for_dense = unet_gate
                if unet_delta_for_dense.shape[-2:] != dense_logits.shape[-2:]:
                    unet_delta_for_dense = F.interpolate(
                        unet_delta_for_dense,
                        size=dense_logits.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                if unet_gate_for_dense.shape[-2:] != dense_logits.shape[-2:]:
                    unet_gate_for_dense = F.interpolate(
                        unet_gate_for_dense,
                        size=dense_logits.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                dense_logits = dense_logits + unet_delta_for_dense * unet_gate_for_dense
                self._last_dense_prompt_unet_delta = unet_delta
                self._last_dense_prompt_unet_gate = unet_gate
            else:
                self._last_dense_prompt_pre_unet = None
                self._last_dense_prompt_unet_delta = None
                self._last_dense_prompt_unet_gate = None
            self._last_dense_prompt_gate = router_gate
            self._last_dense_prompt_area_bias = area_bias.detach()
            self._last_dense_prompt_fg_gate = recall_gate
            self._last_dense_prompt_bg_gate = precision_gate
            self._last_dense_prompt_core_gate = core_gate
            self._last_dense_prompt_fg_residual = recall_residual
            self._last_dense_prompt_bg_residual = precision_residual
            self._last_dense_prompt_core_residual = core_residual
            return base_logits, dense_logits

        if self.coarse_prompt_head in {
            "signed_tribranch_multiscale",
            "signed_tribranch_multiscale_highres",
            "detail_guided_signed_tribranch_multiscale",
            "adaptive_detail_guided_signed_tribranch_multiscale",
        }:
            core_residual = torch.tanh(self.prompt_head_core(prompt_features)) * 2.0
            fg_residual = self._nonnegative_residual_magnitude(self.prompt_head_fg(prompt_features))
            bg_residual = self._nonnegative_residual_magnitude(self.prompt_head_bg(prompt_features))
            fg_gate = torch.sigmoid(self.prompt_head_fg_gate(prompt_features)) * self.coarse_prompt_gate_max
            bg_gate = torch.sigmoid(self.prompt_head_bg_gate(prompt_features)) * self.coarse_prompt_gate_max
            core_gate = torch.sigmoid(self.prompt_head_core_gate(prompt_features)) * self.coarse_prompt_gate_max
            highres_signed_prompt = self.coarse_prompt_head == "signed_tribranch_multiscale_highres"
            if (
                not highres_signed_prompt
                and core_residual.shape[-2:] != base_logits.shape[-2:]
            ):
                core_residual = F.interpolate(
                    core_residual,
                    size=base_logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                fg_residual = F.interpolate(
                    fg_residual,
                    size=base_logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                bg_residual = F.interpolate(
                    bg_residual,
                    size=base_logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                fg_gate = F.interpolate(
                    fg_gate,
                    size=base_logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                bg_gate = F.interpolate(
                    bg_gate,
                    size=base_logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                core_gate = F.interpolate(
                    core_gate,
                    size=base_logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            stats = self._coarse_logit_stats(base_logits)
            area_bias = fg_residual.new_zeros((fg_residual.shape[0], 1, 1, 1))
            if self.prompt_head_area_bias is not None:
                area_bias = self.prompt_head_area_bias(stats.to(dtype=fg_residual.dtype)).view(-1, 1, 1, 1)
            dense_base = base_logits
            if dense_base.shape[-2:] != fg_residual.shape[-2:]:
                dense_base = F.interpolate(
                    dense_base,
                    size=fg_residual.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            dense_logits = (
                dense_base
                + core_residual * core_gate
                + fg_residual * fg_gate
                - bg_residual * bg_gate
                + area_bias
            )
            self._last_dense_prompt_gate = core_gate
            self._last_dense_prompt_area_bias = area_bias.detach()
            self._last_dense_prompt_fg_gate = fg_gate
            self._last_dense_prompt_bg_gate = bg_gate
            self._last_dense_prompt_core_gate = core_gate
            self._last_dense_prompt_fg_residual = fg_residual
            self._last_dense_prompt_bg_residual = bg_residual
            self._last_dense_prompt_core_residual = core_residual
            self._last_dense_prompt_small_gate = None
            self._last_dense_prompt_signed_delta = None
            self._last_dense_prompt_signed_gate = None
            self._last_dense_prompt_pre_unet = None
            self._last_dense_prompt_unet_delta = None
            self._last_dense_prompt_unet_gate = None
            return base_logits, dense_logits

        if self.coarse_prompt_head == "dual_branch_multiscale":
            fg_residual = self._nonnegative_residual_magnitude(self.prompt_head_fg(prompt_features))
            bg_residual = self._nonnegative_residual_magnitude(self.prompt_head_bg(prompt_features))
            if fg_residual.shape[-2:] != base_logits.shape[-2:]:
                fg_residual = F.interpolate(
                    fg_residual,
                    size=base_logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                bg_residual = F.interpolate(
                    bg_residual,
                    size=base_logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            stats = self._coarse_logit_stats(base_logits)
            pooled = self.avg_pool(feature_map).flatten(1)
            gate_input = torch.cat([pooled, stats.to(dtype=pooled.dtype)], dim=1)
            fg_gate = torch.sigmoid(self.prompt_head_fg_gate(gate_input)).view(-1, 1, 1, 1)
            bg_gate = torch.sigmoid(self.prompt_head_bg_gate(gate_input)).view(-1, 1, 1, 1)
            fg_gate = fg_gate * self.coarse_prompt_gate_max
            bg_gate = bg_gate * self.coarse_prompt_gate_max
            area_bias = fg_residual.new_zeros((fg_residual.shape[0], 1, 1, 1))
            if self.prompt_head_area_bias is not None:
                area_bias = self.prompt_head_area_bias(stats.to(dtype=fg_residual.dtype)).view(-1, 1, 1, 1)
            dense_logits = base_logits + fg_residual * fg_gate - bg_residual * bg_gate + area_bias
            self._last_dense_prompt_gate = None
            self._last_dense_prompt_area_bias = area_bias.detach()
            self._last_dense_prompt_fg_gate = fg_gate
            self._last_dense_prompt_bg_gate = bg_gate
            self._last_dense_prompt_core_gate = None
            self._last_dense_prompt_fg_residual = fg_residual
            self._last_dense_prompt_bg_residual = bg_residual
            self._last_dense_prompt_core_residual = None
            self._last_dense_prompt_small_gate = None
            self._last_dense_prompt_signed_delta = None
            self._last_dense_prompt_signed_gate = None
            self._last_dense_prompt_pre_unet = None
            self._last_dense_prompt_unet_delta = None
            self._last_dense_prompt_unet_gate = None
            return base_logits, dense_logits

        residual_logits = prompt_features
        highres_dense_prompt = self.coarse_prompt_head == "gated_split_multiscale_highres"
        if (
            not highres_dense_prompt
            and residual_logits.shape[-2:] != base_logits.shape[-2:]
        ):
            residual_logits = F.interpolate(
                residual_logits,
                size=base_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if self.coarse_prompt_head in {
            "gated_split_multiscale",
            "gated_split_multiscale_highres",
            "adaptive_tau_fusion_multiscale",
        }:
            stats = self._coarse_logit_stats(base_logits)
            pooled = self.avg_pool(feature_map).flatten(1)
            gate_input = torch.cat([pooled, stats.to(dtype=pooled.dtype)], dim=1)
            gate = torch.sigmoid(self.prompt_head_gate(gate_input)).view(-1, 1, 1, 1)
            gate = gate * self.coarse_prompt_gate_max
            area_bias = residual_logits.new_zeros((residual_logits.shape[0], 1, 1, 1))
            if self.prompt_head_area_bias is not None:
                area_bias = self.prompt_head_area_bias(stats.to(dtype=residual_logits.dtype)).view(-1, 1, 1, 1)
            dense_base = base_logits
            if dense_base.shape[-2:] != residual_logits.shape[-2:]:
                dense_base = F.interpolate(
                    dense_base,
                    size=residual_logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            dense_logits = dense_base + residual_logits * gate + area_bias
            # Keep the gate tensor attached to the graph.  Training losses such
            # as ``prompt_gate_supervision_loss(gate_sources="dense")`` use
            # this handle to teach the dense-prompt policy when to expand or
            # suppress.  Callers that only log/probe diagnostics can detach at
            # the consumption site or run under ``torch.no_grad()``.
            self._last_dense_prompt_gate = gate
            self._last_dense_prompt_area_bias = area_bias.detach()
            self._last_dense_prompt_fg_gate = None
            self._last_dense_prompt_bg_gate = None
            self._last_dense_prompt_core_gate = None
            self._last_dense_prompt_fg_residual = None
            self._last_dense_prompt_bg_residual = None
            self._last_dense_prompt_core_residual = None
            self._last_dense_prompt_small_gate = None
            self._last_dense_prompt_signed_delta = None
            self._last_dense_prompt_signed_gate = None
            self._last_dense_prompt_pre_unet = None
            self._last_dense_prompt_unet_delta = None
            self._last_dense_prompt_unet_gate = None
            return base_logits, dense_logits
        dense_logits = base_logits + residual_logits
        self._last_dense_prompt_gate = None
        self._last_dense_prompt_area_bias = None
        self._last_dense_prompt_fg_gate = None
        self._last_dense_prompt_bg_gate = None
        self._last_dense_prompt_core_gate = None
        self._last_dense_prompt_fg_residual = None
        self._last_dense_prompt_bg_residual = None
        self._last_dense_prompt_core_residual = None
        self._last_dense_prompt_small_gate = None
        self._last_dense_prompt_signed_delta = None
        self._last_dense_prompt_signed_gate = None
        self._last_dense_prompt_pre_unet = None
        self._last_dense_prompt_unet_delta = None
        self._last_dense_prompt_unet_gate = None
        if self.coarse_prompt_head == "split_multiscale":
            return base_logits, dense_logits
        return dense_logits, dense_logits

    @property
    def prompt_head_recall(self) -> nn.Module:
        """Semantic alias for the legacy foreground expansion endpoint."""
        if not hasattr(self, "prompt_head_fg"):
            raise AttributeError("prompt_head_recall is only available for dual/signed prompt heads")
        return self.prompt_head_fg

    @property
    def prompt_head_precision(self) -> nn.Module:
        """Semantic alias for the legacy background suppression endpoint."""
        if not hasattr(self, "prompt_head_bg"):
            raise AttributeError("prompt_head_precision is only available for dual/signed prompt heads")
        return self.prompt_head_bg

    @staticmethod
    def _coarse_logit_stats(logits: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, logits.ndim))
        logits_f = logits.float()
        return torch.stack(
            [
                logits_f.mean(dim=dims),
                logits_f.std(dim=dims, unbiased=False),
                logits_f.amin(dim=dims),
                logits_f.amax(dim=dims),
                torch.sigmoid(logits_f).mean(dim=dims),
            ],
            dim=1,
        )

    @staticmethod
    def _nonnegative_residual_magnitude(raw: torch.Tensor) -> torch.Tensor:
        """Non-negative residual magnitude with straight-through gradients.

        Dual-branch prompt semantics require both branch residual magnitudes to
        be non-negative; otherwise the background-suppression branch can flip
        into an expansion branch.  A plain ReLU would be identity-preserving but
        dead at the zero-initialized branch heads.  The straight-through term
        keeps the forward value clamped to ``relu(raw)`` while allowing useful
        gradients when raw values are non-positive.
        """
        relu = torch.relu(raw)
        negative_or_zero = (raw <= 0).to(dtype=raw.dtype)
        return relu + (raw - raw.detach()) * negative_or_zero
    
    def load_pretrained_weights(self, checkpoint_path):
        """Load pretrained weights from FerretNet checkpoint.
        
        Note: Because we replaced BN with IN and added SE blocks, 
        many weights will be skipped. This is expected. The loaded weights
        will serve as a good initialization for the Convolutional layers.
        """
        from collections import OrderedDict
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model' in checkpoint:
                pretrained_dict = checkpoint['model']
            else:
                pretrained_dict = checkpoint
                
            model_dict = self.state_dict()
            
            new_pretrained_dict = OrderedDict()
            skipped_keys = []
            loaded_keys = []
            
            for k, v in pretrained_dict.items():
                # Filter out BN keys since we use IN now (running_mean/var mismatch)
                if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
                    continue
                    
                if k in model_dict:
                    # Only load if shapes match
                    if v.shape == model_dict[k].shape:
                        new_pretrained_dict[k] = v
                        loaded_keys.append(k)
                    else:
                        skipped_keys.append(f"{k} (shape mismatch)")
                else:
                    skipped_keys.append(k)
            
            # Update model state dict
            model_dict.update(new_pretrained_dict)
            self.load_state_dict(model_dict)
            
            print(f"\n[FerretBackbone] Loaded {len(loaded_keys)} weights from {checkpoint_path}")
            # print(f"Skipped {len(skipped_keys)} weights due to architecture changes (BN->IN, SE blocks)")
            
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights: {e}")
            print("Continuing with random initialization.")
