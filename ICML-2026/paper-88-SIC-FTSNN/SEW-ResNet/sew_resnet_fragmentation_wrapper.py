"""Convenience wrapper: SEW-ResNet + learnable fragmentation.

This file packages the two paper ideas together:
1) SEW-ResNet (NeurIPS 2021)
2) learnable fragmentation with optional dynamic fragment-count selection (uploaded ICML paper)

The main goal is ease-of-use: one object takes images, applies fragmentation, optionally encodes
per-step inputs, runs the SEW-ResNet in a time loop, and returns decoded logits plus auxiliary terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

try:
    from .learnable_fragmentation_addon import (
        DynamicLearnableFragmentation,
        EntropyTimeDecoder,
        FragmentationOutput,
        LearnableLineFragmentation,
        MeanTimeDecoder,
    )
    from .sew_resnet_paper import SEWConnection, SEWResNet, safe_reset_net, sew_resnet18, sew_resnet34, sew_resnet50, sew_resnet101, sew_resnet152
except ImportError:  # pragma: no cover
    from learnable_fragmentation_addon import (
        DynamicLearnableFragmentation,
        EntropyTimeDecoder,
        FragmentationOutput,
        LearnableLineFragmentation,
        MeanTimeDecoder,
    )
    from sew_resnet_paper import SEWConnection, SEWResNet, safe_reset_net, sew_resnet18, sew_resnet34, sew_resnet50, sew_resnet101, sew_resnet152

try:  # pragma: no cover - optional in this environment
    from spikingjelly.activation_based import neuron as sj_neuron
    from spikingjelly.activation_based import surrogate as sj_surrogate
except Exception:  # pragma: no cover
    sj_neuron = None
    sj_surrogate = None


__all__ = [
    "ExpectedPoissonEncoder",
    "FragmentedSEWOutput",
    "FragmentedSEWResNet",
    "build_fragmented_sew_resnet",
]


class ExpectedPoissonEncoder(nn.Module):
    """Simple differentiable rate encoder.

    For learnable fragmentation, a sampled Bernoulli encoder can weaken gradients into the
    fragmentation module. The uploaded example code uses a differentiable surrogate/expected
    encoder for that reason. This module therefore defaults to returning the expected firing rate.

    Modes:
    - expected: return x clamped to [0, 1]
    - ste_bernoulli: hard Bernoulli forward with straight-through gradients
    - bernoulli: plain hard Bernoulli sampling
    """

    def __init__(self, mode: str = "expected") -> None:
        super().__init__()
        mode = str(mode).strip().lower()
        if mode not in {"expected", "ste_bernoulli", "bernoulli"}:
            raise ValueError("mode must be one of {'expected', 'ste_bernoulli', 'bernoulli'}")
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        rate = x.clamp(0.0, 1.0)
        if self.mode == "expected":
            return rate
        hard = torch.bernoulli(rate)
        if self.mode == "ste_bernoulli":
            return hard.detach() - rate.detach() + rate
        return hard


@dataclass
class FragmentedSEWOutput:
    logits: Tensor                 # [B, K]
    step_logits: Tensor            # [B, T, K]
    balance_loss: Tensor
    selector_probs: Optional[Tensor] = None
    selected_t: Optional[int] = None
    fragmentation: Optional[FragmentationOutput] = None


class FragmentedSEWResNet(nn.Module):
    """A thin wrapper that makes SEW-ResNet consume fragmented inputs conveniently."""

    def __init__(
        self,
        backbone: SEWResNet,
        fragmenter: Optional[Union[LearnableLineFragmentation, DynamicLearnableFragmentation]] = None,
        *,
        input_encoder: Optional[Callable[[Tensor], Tensor]] = None,
        decoder: Optional[nn.Module] = None,
        dynamic_train_mode: str = "mix",
        dynamic_eval_mode: str = "selected",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.fragmenter = fragmenter
        self.input_encoder = input_encoder
        self.decoder = decoder if decoder is not None else EntropyTimeDecoder(gamma=1.0, time_dim=1)
        self.dynamic_train_mode = str(dynamic_train_mode).strip().lower()
        self.dynamic_eval_mode = str(dynamic_eval_mode).strip().lower()

    def _encode_step(self, x_t: Tensor) -> Tensor:
        if self.input_encoder is None:
            return x_t
        return self.input_encoder(x_t)

    def _make_plain_sequence(self, images: Tensor, num_steps: int) -> FragmentationOutput:
        seq = images.unsqueeze(1).repeat(1, int(num_steps), 1, 1, 1)
        return FragmentationOutput(sequence=seq, balance_loss=images.new_tensor(0.0), selected_t=int(num_steps))

    def forward(
        self,
        images: Tensor,
        *,
        return_aux: bool = True,
        return_fragmentation_state: bool = False,
        plain_num_steps: int = 4,
        sample_selector: bool = True,
    ) -> Union[Tensor, FragmentedSEWOutput]:
        if images.dim() != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(images.shape)}")

        if self.fragmenter is None:
            frag_out = self._make_plain_sequence(images, plain_num_steps)
        elif isinstance(self.fragmenter, DynamicLearnableFragmentation):
            mode = self.dynamic_train_mode if self.training else self.dynamic_eval_mode
            frag_out = self.fragmenter(images, mode=mode, sample_selector=sample_selector)
        else:
            frag_out = self.fragmenter(images)

        seq = frag_out.sequence  # [B, T, C, H, W]
        step_logits = []
        try:
            for t in range(seq.size(1)):
                x_t = self._encode_step(seq[:, t])
                step_logits.append(self.backbone(x_t))
        finally:
            safe_reset_net(self.backbone)

        step_logits_t = torch.stack(step_logits, dim=1)  # [B, T, K]
        logits = self.decoder(step_logits_t)

        if not return_aux:
            return logits

        fragmentation_state = frag_out if return_fragmentation_state else None
        return FragmentedSEWOutput(
            logits=logits,
            step_logits=step_logits_t,
            balance_loss=frag_out.balance_loss,
            selector_probs=frag_out.selector_probs,
            selected_t=frag_out.selected_t,
            fragmentation=fragmentation_state,
        )



def _default_spiking_components(neuron_name: str = "if") -> Tuple[Callable[..., nn.Module], Dict[str, Any]]:
    if sj_neuron is None or sj_surrogate is None:
        raise ImportError("SpikingJelly is required to build the default SEW-ResNet configuration.")

    neuron_name = str(neuron_name).strip().lower()
    if neuron_name == "if":
        neuron_ctor = sj_neuron.IFNode
    elif neuron_name == "lif":
        neuron_ctor = sj_neuron.LIFNode
    else:
        raise ValueError("neuron_name must be 'if' or 'lif'")

    kwargs: Dict[str, Any] = dict(v_threshold=1.0, surrogate_function=sj_surrogate.ATan(), detach_reset=True)
    if neuron_name == "lif":
        kwargs["tau"] = 2.0
    return neuron_ctor, kwargs



def build_fragmented_sew_resnet(
    *,
    depth: int = 18,
    num_classes: int = 10,
    image_size: Tuple[int, int] = (32, 32),
    in_channels: int = 3,
    stem: str = "cifar",
    cnf: str = SEWConnection.ADD,
    neuron_name: str = "if",
    zero_init_residual: bool = True,
    use_expected_poisson: bool = False,
    fixed_num_fragments: Optional[int] = None,
    dynamic_candidates: Optional[Sequence[int]] = None,
    init_direction: str = "horizontal",
    mask_scale: float = 1.0,
    decoder: str = "entropy",
    entropy_gamma: float = 1.0,
) -> FragmentedSEWResNet:
    """Build a ready-to-train fragmented SEW-ResNet.

    This is the main convenience entry point for the user.
    """
    neuron_ctor, neuron_kwargs = _default_spiking_components(neuron_name=neuron_name)

    if depth == 18:
        backbone = sew_resnet18(
            num_classes=num_classes,
            cnf=cnf,
            spiking_neuron=neuron_ctor,
            neuron_kwargs=neuron_kwargs,
            stem=stem,
            in_channels=in_channels,
            zero_init_residual=zero_init_residual,
        )
    elif depth == 34:
        backbone = sew_resnet34(
            num_classes=num_classes,
            cnf=cnf,
            spiking_neuron=neuron_ctor,
            neuron_kwargs=neuron_kwargs,
            stem=stem,
            in_channels=in_channels,
            zero_init_residual=zero_init_residual,
        )
    elif depth == 50:
        backbone = sew_resnet50(
            num_classes=num_classes,
            cnf=cnf,
            spiking_neuron=neuron_ctor,
            neuron_kwargs=neuron_kwargs,
            stem=stem,
            in_channels=in_channels,
            zero_init_residual=zero_init_residual,
        )
    elif depth == 101:
        backbone = sew_resnet101(
            num_classes=num_classes,
            cnf=cnf,
            spiking_neuron=neuron_ctor,
            neuron_kwargs=neuron_kwargs,
            stem=stem,
            in_channels=in_channels,
            zero_init_residual=zero_init_residual,
        )
    elif depth == 152:
        backbone = sew_resnet152(
            num_classes=num_classes,
            cnf=cnf,
            spiking_neuron=neuron_ctor,
            neuron_kwargs=neuron_kwargs,
            stem=stem,
            in_channels=in_channels,
            zero_init_residual=zero_init_residual,
        )
    else:
        raise ValueError("depth must be one of {18, 34, 50, 101, 152}")

    if fixed_num_fragments is not None and dynamic_candidates is not None:
        raise ValueError("Choose either fixed_num_fragments or dynamic_candidates, not both.")

    fragmenter = None
    if fixed_num_fragments is not None:
        fragmenter = LearnableLineFragmentation(
            image_size=image_size,
            num_fragments=int(fixed_num_fragments),
            init_direction=init_direction,
            hard_forward=True,
            mask_scale=mask_scale,
        )
    elif dynamic_candidates is not None:
        fragmenter = DynamicLearnableFragmentation(
            image_size=image_size,
            candidates=tuple(int(t) for t in dynamic_candidates),
            init_direction=init_direction,
            hard_forward=True,
            mask_scale=mask_scale,
            selector_hard=False,
            gumbel_tau=1.0,
        )

    if decoder == "entropy":
        decoder_module = EntropyTimeDecoder(gamma=float(entropy_gamma), time_dim=1)
    elif decoder == "mean":
        decoder_module = MeanTimeDecoder(time_dim=1)
    else:
        raise ValueError("decoder must be 'entropy' or 'mean'")

    input_encoder = ExpectedPoissonEncoder(mode="expected") if use_expected_poisson else None

    return FragmentedSEWResNet(
        backbone=backbone,
        fragmenter=fragmenter,
        input_encoder=input_encoder,
        decoder=decoder_module,
    )
