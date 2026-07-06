"""Paper-faithful SEW-ResNet implementation for static-image SNNs.

This module follows the architecture proposed in:
    Wei Fang et al., "Deep Residual Learning in Spiking Neural Networks," NeurIPS 2021.

Design choices in this implementation:
- The SEW residual block uses the paper's spike-element-wise combination g(A, S), where
  A is the residual spike tensor and S is the shortcut spike tensor.
- The downsample shortcut contains Conv-BN-SN, matching the SEW downsample block.
- Zero-initialization is adapted to the selected SEW connection function:
    * ADD / IAND: last BN weight = 0, bias = 0  -> A = 0 -> identity mapping
    * AND:       last BN weight = 0, bias = v_threshold -> A = 1 -> identity mapping
- The model is written in single-step style. Run it in an outer time loop, or use
  `forward_sequence` for convenience.

The code targets SpikingJelly's `activation_based` API when available.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

try:
    from torchvision.models import resnet as tv_resnet
except Exception:  # pragma: no cover - torchvision may be absent in minimal envs
    tv_resnet = None

try:  # pragma: no cover - optional dependency in this execution environment
    from spikingjelly.activation_based import functional as sj_functional
    from spikingjelly.activation_based import neuron as sj_neuron
except Exception:  # pragma: no cover
    sj_functional = None
    sj_neuron = None


__all__ = [
    "SEWConnection",
    "combine_spikes",
    "SEWBasicBlock",
    "SEWBottleneck",
    "SEWResNet",
    "sew_resnet18",
    "sew_resnet34",
    "sew_resnet50",
    "sew_resnet101",
    "sew_resnet152",
    "safe_reset_net",
]


class SEWConnection:
    ADD = "ADD"
    AND = "AND"
    IAND = "IAND"

    @classmethod
    def normalize(cls, value: str) -> str:
        out = str(value).strip().upper()
        if out not in {cls.ADD, cls.AND, cls.IAND}:
            raise ValueError(f"Unsupported SEW connection: {value!r}")
        return out



def combine_spikes(residual: Tensor, shortcut: Tensor, cnf: str) -> Tensor:
    """Paper-faithful spike-element-wise operator g(A, S).

    The paper defines g(A, S) using:
      ADD  : A + S
      AND  : A * S
      IAND : (1 - A) * S
    where A is the residual spike tensor and S is the shortcut spike tensor.
    """
    cnf = SEWConnection.normalize(cnf)
    if cnf == SEWConnection.ADD:
        return residual + shortcut
    if cnf == SEWConnection.AND:
        return residual * shortcut
    return (1.0 - residual) * shortcut



def _conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )



def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



def _clone_neuron(neuron_ctor: Callable[..., nn.Module], neuron_kwargs: Dict[str, Any]) -> nn.Module:
    return neuron_ctor(**deepcopy(neuron_kwargs))



def safe_reset_net(module: nn.Module) -> None:
    """Reset spiking states if the backend exposes reset hooks.

    SpikingJelly exposes `functional.reset_net`. As a fallback, we recursively call
    `reset()` on submodules that implement it.
    """
    if sj_functional is not None:
        try:
            sj_functional.reset_net(module)
            return
        except Exception:
            pass

    for m in module.modules():
        reset_fn = getattr(m, "reset", None)
        if callable(reset_fn):
            reset_fn()


@dataclass
class SequenceOutput:
    logits: Tensor  # [T, B, K]


class SEWBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        cnf: str = SEWConnection.ADD,
        spiking_neuron: Optional[Callable[..., nn.Module]] = None,
        neuron_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if spiking_neuron is None:
            raise ImportError("SpikingJelly is required to instantiate SEW blocks. Pass a spiking_neuron constructor.")
        if groups != 1 or base_width != 64:
            raise ValueError("SEWBasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 is not supported in SEWBasicBlock")

        neuron_kwargs = {} if neuron_kwargs is None else dict(neuron_kwargs)
        self.cnf = SEWConnection.normalize(cnf)

        self.conv1 = _conv3x3(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = _clone_neuron(spiking_neuron, neuron_kwargs)

        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = _clone_neuron(spiking_neuron, neuron_kwargs)

        self.downsample = downsample
        self.downsample_sn = _clone_neuron(spiking_neuron, neuron_kwargs) if downsample is not None else None
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            shortcut = self.downsample_sn(shortcut)

        return combine_spikes(residual=out, shortcut=shortcut, cnf=self.cnf)


class SEWBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        cnf: str = SEWConnection.ADD,
        spiking_neuron: Optional[Callable[..., nn.Module]] = None,
        neuron_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if spiking_neuron is None:
            raise ImportError("SpikingJelly is required to instantiate SEW blocks. Pass a spiking_neuron constructor.")

        neuron_kwargs = {} if neuron_kwargs is None else dict(neuron_kwargs)
        width = int(planes * (base_width / 64.0)) * groups
        self.cnf = SEWConnection.normalize(cnf)

        self.conv1 = _conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.sn1 = _clone_neuron(spiking_neuron, neuron_kwargs)

        self.conv2 = _conv3x3(width, width, stride=stride, groups=groups, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.sn2 = _clone_neuron(spiking_neuron, neuron_kwargs)

        self.conv3 = _conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn3 = _clone_neuron(spiking_neuron, neuron_kwargs)

        self.downsample = downsample
        self.downsample_sn = _clone_neuron(spiking_neuron, neuron_kwargs) if downsample is not None else None
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.sn3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            shortcut = self.downsample_sn(shortcut)

        return combine_spikes(residual=out, shortcut=shortcut, cnf=self.cnf)


class SEWResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[SEWBasicBlock, SEWBottleneck]],
        layers: Sequence[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[Sequence[bool]] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        cnf: str = SEWConnection.ADD,
        spiking_neuron: Optional[Callable[..., nn.Module]] = None,
        neuron_kwargs: Optional[Dict[str, Any]] = None,
        stem: str = "imagenet",
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if spiking_neuron is None:
            if sj_neuron is None:
                raise ImportError(
                    "SpikingJelly is not installed. Install it, or pass a custom spiking_neuron constructor."
                )
            spiking_neuron = sj_neuron.IFNode
        neuron_kwargs = {} if neuron_kwargs is None else dict(neuron_kwargs)

        self._norm_layer = norm_layer
        self.cnf = SEWConnection.normalize(cnf)
        self.spiking_neuron = spiking_neuron
        self.neuron_kwargs = neuron_kwargs
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.stem = str(stem).strip().lower()
        self.in_channels = int(in_channels)

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element sequence")

        if self.stem == "imagenet":
            self.conv1 = nn.Conv2d(self.in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.sn1 = _clone_neuron(spiking_neuron, neuron_kwargs)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif self.stem == "cifar":
            self.conv1 = nn.Conv2d(self.in_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.sn1 = _clone_neuron(spiking_neuron, neuron_kwargs)
            self.maxpool = nn.Identity()
        else:
            raise ValueError("stem must be 'imagenet' or 'cifar'")

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()
        if zero_init_residual:
            self._apply_paper_zero_init(block=block)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[Union[SEWBasicBlock, SEWBottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        previous_dilation = self.dilation
        downsample = None

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                norm_layer(planes * block.expansion),
            )

        layers: List[nn.Module] = []
        layers.append(
            block(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                norm_layer=norm_layer,
                cnf=self.cnf,
                spiking_neuron=self.spiking_neuron,
                neuron_kwargs=self.neuron_kwargs,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    inplanes=self.inplanes,
                    planes=planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    cnf=self.cnf,
                    spiking_neuron=self.spiking_neuron,
                    neuron_kwargs=self.neuron_kwargs,
                )
            )
        return nn.Sequential(*layers)

    def _identity_bias_for_and(self) -> float:
        # Paper: for IF neurons, setting the last BN bias to V_th gives A == 1.
        return float(self.neuron_kwargs.get("v_threshold", 1.0))

    def _apply_paper_zero_init(self, block: Type[Union[SEWBasicBlock, SEWBottleneck]]) -> None:
        if block is SEWBottleneck:
            target_bn_name = "bn3"
            target_block_type = SEWBottleneck
        else:
            target_bn_name = "bn2"
            target_block_type = SEWBasicBlock

        and_bias = self._identity_bias_for_and()
        for m in self.modules():
            if isinstance(m, target_block_type):
                bn = getattr(m, target_bn_name)
                if getattr(bn, "weight", None) is not None:
                    nn.init.constant_(bn.weight, 0.0)
                if getattr(bn, "bias", None) is not None:
                    if self.cnf == SEWConnection.AND:
                        nn.init.constant_(bn.bias, and_bias)
                    else:
                        nn.init.constant_(bn.bias, 0.0)

    def load_from_torchvision_resnet(self, strict: bool = False) -> None:
        """Optionally initialize Conv/BN/FC weights from torchvision ANN ResNet.

        This is useful when one wants the same pretrained-ANN warm-start style supported by
        the official SEW-ResNet implementation. Only layers with matching names/shapes are loaded.

        The helper is intentionally tolerant to torchvision API differences:
        recent releases use ``weights=...`` while older ones use ``pretrained=True``.
        """
        if tv_resnet is None:
            raise ImportError("torchvision is required for ANN-pretrained initialization")

        if isinstance(self.layer1[0], SEWBottleneck):
            depth_to_ctor = {
                (3, 4, 6, 3): tv_resnet.resnet50,
                (3, 4, 23, 3): tv_resnet.resnet101,
                (3, 8, 36, 3): tv_resnet.resnet152,
            }
        else:
            depth_to_ctor = {
                (2, 2, 2, 2): tv_resnet.resnet18,
                (3, 4, 6, 3): tv_resnet.resnet34,
            }
        key = tuple(len(stage) for stage in [self.layer1, self.layer2, self.layer3, self.layer4])
        if key not in depth_to_ctor:
            raise ValueError(f"No torchvision counterpart for stage layout {key}")
        if self.stem == "cifar":
            raise ValueError("Torchvision pretrained weights are ImageNet-stem only; use stem='imagenet'.")

        ctor = depth_to_ctor[key]
        ann = None
        last_err = None
        for kwargs in ({"weights": "DEFAULT"}, {"pretrained": True}, {}):
            try:
                ann = ctor(**kwargs)
                break
            except TypeError as exc:
                last_err = exc
                continue
        if ann is None:
            raise RuntimeError("Failed to instantiate the torchvision ResNet counterpart") from last_err

        try:
            incompatible = self.load_state_dict(ann.state_dict(), strict=strict)
        except RuntimeError as exc:
            raise RuntimeError(
                "Torchvision state_dict loading failed. This usually means the backbone depth/stem/classifier "
                "shape does not match the selected torchvision counterpart."
            ) from exc

        if strict:
            return

        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        if unexpected:
            raise RuntimeError(f"Unexpected torchvision keys when loading ANN weights: {unexpected}")
        # Missing keys correspond to spiking neurons and optional shortcut neurons.

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.fc(x)
        return x

    def forward_sequence(self, x_seq: Tensor, reset_after: bool = True) -> SequenceOutput:
        """Run a sequence of inputs [T, B, C, H, W] through the single-step SNN."""
        if x_seq.dim() != 5:
            raise ValueError(f"Expected [T, B, C, H, W], got {tuple(x_seq.shape)}")
        logits: List[Tensor] = []
        try:
            for t in range(x_seq.size(0)):
                logits.append(self.forward(x_seq[t]))
        finally:
            if reset_after:
                safe_reset_net(self)
        return SequenceOutput(logits=torch.stack(logits, dim=0))



def _build_sew_resnet(
    block: Type[Union[SEWBasicBlock, SEWBottleneck]],
    layers: Sequence[int],
    **kwargs: Any,
) -> SEWResNet:
    return SEWResNet(block=block, layers=layers, **kwargs)



def sew_resnet18(**kwargs: Any) -> SEWResNet:
    return _build_sew_resnet(SEWBasicBlock, [2, 2, 2, 2], **kwargs)



def sew_resnet34(**kwargs: Any) -> SEWResNet:
    return _build_sew_resnet(SEWBasicBlock, [3, 4, 6, 3], **kwargs)



def sew_resnet50(**kwargs: Any) -> SEWResNet:
    return _build_sew_resnet(SEWBottleneck, [3, 4, 6, 3], **kwargs)



def sew_resnet101(**kwargs: Any) -> SEWResNet:
    return _build_sew_resnet(SEWBottleneck, [3, 4, 23, 3], **kwargs)



def sew_resnet152(**kwargs: Any) -> SEWResNet:
    return _build_sew_resnet(SEWBottleneck, [3, 8, 36, 3], **kwargs)
