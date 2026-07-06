"""BlockMixed-compatible pTProxy variant.

This implementation is used by the response/new-search-space experiments. It
scores encoded backbones with one batched forward over clean and perturbed
inputs, and supports multiple activation types beyond ReLU.
"""

import math
import time
from functools import partial
from typing import List, Optional, Tuple

import torch
from torch import nn

from proxies.ptproxy import (
    _assert_has_attr,
    _linearize,
    _nonlinearize,
    _weighted_score_traj_only,
    _weighted_score_traj_width,
    _weighted_score_width_only,
)


SUPPORTED_ACTIVATIONS = (
    nn.ReLU,
    nn.GELU,
    nn.SiLU,
    nn.LeakyReLU,
    nn.ELU,
)


class BatchedActivationHook:
    """Collect clean and perturbed activations from one batched forward pass."""

    def __init__(self, clean_batch_size: int):
        self.clean_batch_size = clean_batch_size
        self.originals: List[torch.Tensor] = []
        self.perturbations: List[torch.Tensor] = []
        self.Vs: List[torch.Tensor] = []
        self.clean_activation_map = {}

    def forward_hook(self, module: nn.Module, inputs, output: torch.Tensor):
        if output.shape[0] != 2 * self.clean_batch_size:
            raise ValueError(
                "ptproxy_blockmixed expects a concatenated batch with shape[0] == 2 * clean_batch_size."
            )

        clean_output, perturbed_output = torch.split(
            output, [self.clean_batch_size, self.clean_batch_size], dim=0
        )
        self.originals.append(clean_output)
        self.perturbations.append(perturbed_output)

        self.clean_activation_map[id(module)] = clean_output
        output.register_hook(partial(self.backward_hook, module=module))

    def backward_hook(self, grad: torch.Tensor, module: nn.Module):
        act = self.clean_activation_map[id(module)]
        grad_clean = grad[: self.clean_batch_size]
        self.Vs.append(act * grad_clean.abs())

    def trajectory_lengths(self, epsilon: float) -> List[torch.Tensor]:
        return [
            (p - o).abs().norm() / epsilon
            for o, p in zip(self.originals, self.perturbations)
        ]

    def clear(self):
        self.originals.clear()
        self.perturbations.clear()
        self.Vs.clear()
        self.clean_activation_map.clear()


def _ptproxy_blockmixed_activation_score(
    arch: nn.Module,
    batch_data: torch.Tensor,
    device: str = "cpu",
    *,
    use_wo_embedding: bool = False,
    linearize_target: Optional[nn.Module] = None,
    epsilon: float = 1e-5,
    weight_mode: str = "traj_width",
    use_fp64: bool = False,
    activation_types: Tuple[type[nn.Module], ...] = SUPPORTED_ACTIVATIONS,
    respect_input: bool = False,
) -> Tuple[float, float]:
    """Compute the BlockMixed-compatible pTProxy score."""

    assert isinstance(arch, nn.Module)
    if not isinstance(batch_data, torch.Tensor):
        raise TypeError("ptproxy_blockmixed_score expects batch_data to be a torch.Tensor.")

    arch = arch.to(device)

    x = batch_data.to(device) if respect_input else torch.ones_like(batch_data).to(device)
    dtype = torch.float64 if use_fp64 else torch.float32

    if use_wo_embedding:
        _assert_has_attr(arch, "forward_wo_embedding")
        fwd = arch.forward_wo_embedding
    else:
        fwd = arch.forward

    target = linearize_target if linearize_target is not None else arch

    arch.eval()
    arch.zero_grad(set_to_none=True)
    signs = _linearize(target)

    hook_obj = BatchedActivationHook(clean_batch_size=x.shape[0])
    handles: List[torch.utils.hooks.RemovableHandle] = []
    try:
        for module in arch.modules():
            if isinstance(module, activation_types):
                handles.append(module.register_forward_hook(hook_obj.forward_hook))

        x = x.to(dtype)
        delta_x = torch.randn_like(x) * epsilon
        batched_x = torch.cat([x, x + delta_x], dim=0)

        if "cuda" in device:
            torch.cuda.synchronize()
        t0 = time.time()

        out_batched = fwd(batched_x)
        out = out_batched[: x.shape[0]]

        traj = hook_obj.trajectory_lengths(epsilon)
        torch.sum(out).backward()

        if weight_mode == "traj":
            total = _weighted_score_traj_only(traj, hook_obj.Vs)
        elif weight_mode == "width":
            total = _weighted_score_width_only(traj, hook_obj.Vs)
        else:
            total = _weighted_score_traj_width(traj, hook_obj.Vs)

        if "cuda" in device:
            torch.cuda.synchronize()
        t1 = time.time()

        score = float(total.detach().item())
        if not math.isfinite(score):
            score = 1e8 if score > 0 else -1e8
        return score, (t1 - t0)

    finally:
        for handle in handles:
            handle.remove()
        hook_obj.clear()
        _nonlinearize(target, signs)


def ptproxy_blockmixed_score(
    arch: nn.Module,
    batch_data: torch.Tensor,
    batch_labels: Optional[torch.Tensor] = None,
    device: str = "cpu",
    respect_input: bool = False,
) -> Tuple[float, float]:
    del batch_labels
    return _ptproxy_blockmixed_activation_score(
        arch=arch,
        batch_data=batch_data,
        device=device,
        use_wo_embedding=True,
        linearize_target=None,
        epsilon=1e-5,
        weight_mode="traj_width",
        use_fp64=False,
        respect_input=respect_input,
    )
