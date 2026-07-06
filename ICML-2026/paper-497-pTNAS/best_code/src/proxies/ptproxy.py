import time
import math
from functools import partial
from typing import List, Tuple, Optional

import torch
from torch import nn


class IntegratedHook:
    """
    Collects (1) ReLU outputs for original & perturbed inputs,
            (2) V = activation * |grad| for each ReLU during backward.
    """

    def __init__(self):
        self.originals: List[torch.Tensor] = []
        self.perturbations: List[torch.Tensor] = []
        self.Vs: List[torch.Tensor] = []
        self.clean_activation_map = {}
        self.is_perturbed = False

    def forward_hook(self, module: nn.Module, inputs, output: torch.Tensor):
        if isinstance(module, nn.ReLU):
            if self.is_perturbed:
                self.perturbations.append(output)
            else:
                self.originals.append(output)
                self.clean_activation_map[id(module)] = output
        output.register_hook(partial(self.backward_hook, module=module))

    def backward_hook(self, grad: torch.Tensor, module: nn.Module):
        act = self.clean_activation_map[id(module)]
        V = act * grad.abs()
        self.Vs.append(V)

    def trajectory_lengths(self, epsilon: float) -> List[torch.Tensor]:
        # in case some nets have fewer ReLUs on one pass, zip limits to the shortest
        return [(p - o).abs().norm() / epsilon
                for o, p in zip(self.originals, self.perturbations)]

    def clear(self):
        self.originals.clear()
        self.perturbations.clear()
        self.Vs.clear()
        self.clean_activation_map.clear()


@torch.no_grad()
def _linearize(model: nn.Module) -> dict:
    """store signs; make weights non-negative in-place."""
    signs = {}
    for name, p in model.named_parameters():  # ← only parameters
        signs[name] = torch.sign(p)
        p.abs_()
    return signs


@torch.no_grad()
def _nonlinearize(model: nn.Module, signs: dict):
    for name, p in model.named_parameters():
        p.mul_(signs[name])


def _weighted_score_traj_width(traj: List[torch.Tensor], Vs: List[torch.Tensor]) -> torch.Tensor:
    # deeper layers get smaller weights via inverse trajectory length; also scale by width
    if not traj or not Vs:
        return torch.tensor(0.0, device=Vs[0].device) if Vs else torch.tensor(0.0)

    traj_rev = list(reversed(traj))
    inv = [1.0 / (t + 1e-6) for t in traj_rev]
    inv_sum = sum(inv)
    weights = [i / inv_sum for i in inv]
    total = sum(w * V.flatten().sum() * V.shape[1] for w, V in zip(weights, Vs))
    return total


def _weighted_score_traj_only(traj: List[torch.Tensor], Vs: List[torch.Tensor]) -> torch.Tensor:
    if not traj or not Vs:
        return torch.tensor(0.0, device=Vs[0].device) if Vs else torch.tensor(0.0)
    traj_rev = list(reversed(traj))
    inv = [1.0 / (t + 1e-6) for t in traj_rev]
    inv_sum = sum(inv)
    weights = [i / inv_sum for i in inv]
    return sum(w * V.flatten().sum() for w, V in zip(weights, Vs))


def _weighted_score_width_only(traj: List[torch.Tensor], Vs: List[torch.Tensor]) -> torch.Tensor:
    if not Vs:
        return torch.tensor(0.0)
    return sum(V.flatten().sum() * V.shape[1] for V in Vs) / 10.0


@torch.no_grad()
def _assert_has_attr(obj, name: str):
    if not hasattr(obj, name):
        raise ValueError(f"need {name}, but the object {obj} does not have it.")


def ptproxy_score(
        arch: nn.Module,
        batch_data: torch.Tensor,
        device: str = "cpu",
        *,
        use_wo_embedding: bool = False,  # True: arch.forward_wo_embedding; False: arch.forward
        linearize_target: Optional[nn.Module] = None,  # Specify which submodule to linearize; None means linearize the entire arch
        epsilon: float = 1e-5,
        weight_mode: str = "traj_width",
        use_fp64: bool = False,
        respect_input: bool = False,
) -> Tuple[float, float]:
    """
    Compute pTProxy score (zero-cost proxy).
    Differences from the old version:
      1) No longer guesses based on space_name, fully controlled by use_wo_embedding / linearize_target explicitly;
      2) Other processes and weight modes remain consistent.
    Returns: (score, elapsed_seconds)
    """
    assert isinstance(arch, nn.Module)
    arch = arch.to(device)

    x = batch_data.to(device) if respect_input else torch.ones_like(batch_data).to(device)
    # eps_base = x.detach().abs().median().item()
    # epsilon = 1e-3 if eps_base == 0 else 1e-2 * eps_base

    dtype = torch.float64 if use_fp64 else torch.float32

    # Select forward function (explicit switch)
    if use_wo_embedding:
        _assert_has_attr(arch, "forward_wo_embedding")
        fwd = arch.forward_wo_embedding
    else:
        fwd = arch.forward

    # Select linearization target (explicitly specify submodule; default is to linearize the entire model)
    target = linearize_target if linearize_target is not None else arch

    arch.eval()  # disable Dropout; BN uses running stats
    arch.zero_grad(set_to_none=True)

    # 1) Linearize: take absolute value of parameters and save signs to avoid positive/negative cancellation
    signs = _linearize(target)

    # 2) Register ReLU hooks: record original/perturbed forward activations & backward |grad|, forming V = act * |grad|
    hook_obj = IntegratedHook()
    handles: List[torch.utils.hooks.RemovableHandle] = []
    try:
        for m in arch.modules():
            if isinstance(m, nn.ReLU):
                handles.append(m.register_forward_hook(hook_obj.forward_hook))

        # 3) Two forward passes: original x and perturbed x+δ (δ ~ N(0, I)*epsilon)
        x = x.to(dtype)
        delta_x = torch.randn_like(x) * epsilon

        if "cuda" in device:
            torch.cuda.synchronize()
        t0 = time.time()

        hook_obj.is_perturbed = False
        out = fwd(x)  # Original input, collect a(x)

        hook_obj.is_perturbed = True
        _ = fwd(x + delta_x)  # Perturbed input, collect a(x+δ)

        # 4) Trajectory length tau: ||a(x+δ) - a(x)|| / epsilon
        traj = hook_obj.trajectory_lengths(epsilon)

        # 5) Backward pass once to ReLU outputs, hooks collect V = act * |grad|
        torch.sum(out).backward()

        # 6) Aggregate weights: three modes (default traj_width)
        if weight_mode == "traj":
            total = _weighted_score_traj_only(traj, hook_obj.Vs)
        elif weight_mode == "width":
            total = _weighted_score_width_only(traj, hook_obj.Vs)
        else:  # "traj_width"
            total = _weighted_score_traj_width(traj, hook_obj.Vs)

        if "cuda" in device:
            torch.cuda.synchronize()
        t1 = time.time()

        # 7) Numerical cleanup and return
        score = float(total.detach().item())
        if not math.isfinite(score):
            score = 1e8 if score > 0 else -1e8

        return score, (t1 - t0)

    finally:
        for h in handles:
            h.remove()
        hook_obj.clear()
        _nonlinearize(target, signs)
