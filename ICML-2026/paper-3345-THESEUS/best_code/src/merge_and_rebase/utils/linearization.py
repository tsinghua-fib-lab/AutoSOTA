from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from torch.func import functional_call, jvp
from torch.nn.attention import SDPBackend, sdpa_kernel


@contextmanager
def forward_ad_safe_attention_context(device: torch.device):
    old_mha_fastpath: bool | None = None
    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "get_fastpath_enabled"):
        old_mha_fastpath = bool(torch.backends.mha.get_fastpath_enabled())
        torch.backends.mha.set_fastpath_enabled(False)

    try:
        if device.type == "cuda":
            with sdpa_kernel([SDPBackend.MATH], set_priority=True):
                yield
        else:
            with nullcontext():
                yield
    finally:
        if old_mha_fastpath is not None:
            torch.backends.mha.set_fastpath_enabled(old_mha_fastpath)


def _snapshot_named_tensors(named: list[tuple[str, torch.Tensor]]) -> tuple[list[str], tuple[torch.Tensor, ...]]:
    names = [n for n, _ in named]
    values = tuple(t.detach().clone() for _, t in named)
    return names, values


class LinearizedModule:
    """
    First-order linearization helper around a frozen reference module.
    """

    def __init__(self, ref_module: nn.Module, *, param_names: list[str] | None = None) -> None:
        self.ref_module = ref_module
        named_params = list(ref_module.named_parameters())
        if param_names is not None:
            keep = set(param_names)
            named_params = [(n, p) for n, p in named_params if n in keep]
            missing = [n for n in param_names if n not in {name for name, _ in named_params}]
            if missing:
                raise ValueError(f"Linearization param_names not found on reference module: {missing[:10]}")
        self.param_names, self.theta0 = _snapshot_named_tensors(named_params)
        self.buffer_names, self.buffer_values = _snapshot_named_tensors(list(ref_module.named_buffers()))

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        *,
        device: torch.device | None = None,
        copy_module: bool = True,
        param_names: list[str] | None = None,
    ) -> LinearizedModule:
        ref_module = deepcopy(module) if copy_module else module
        if device is not None:
            ref_module = ref_module.to(device)
        ref_module.eval()
        for p in ref_module.parameters():
            p.requires_grad = False
        return cls(ref_module, param_names=param_names)

    def forward(
        self,
        *,
        current_module: nn.Module | None = None,
        current_params: Mapping[str, torch.Tensor] | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        output_transform: Callable[[Any], torch.Tensor] | None = None,
        post_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        pre_post_transform_callback: Callable[[torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        if current_module is None and current_params is None:
            raise ValueError("LinearizedModule.forward requires current_module or current_params.")
        if current_module is not None:
            self.ref_module.train(current_module.training)
        params_now = dict(current_params) if current_params is not None else dict(current_module.named_parameters())
        tangents = tuple(params_now[n] - p0 for n, p0 in zip(self.param_names, self.theta0, strict=True))
        call_kwargs = dict(kwargs or {})

        def _f(*primals: torch.Tensor) -> Any:
            param_map = {n: p for n, p in zip(self.param_names, primals, strict=True)}
            buffer_map = {n: b for n, b in zip(self.buffer_names, self.buffer_values, strict=True)}
            out = functional_call(self.ref_module, (param_map, buffer_map), args=args, kwargs=call_kwargs, strict=False)
            return output_transform(out) if output_transform is not None else out

        first_tensor = next((arg for arg in args if isinstance(arg, torch.Tensor)), None)
        if first_tensor is None:
            first_tensor = next((value for value in call_kwargs.values() if isinstance(value, torch.Tensor)), None)
        if first_tensor is None:
            raise ValueError("LinearizedModule.forward requires at least one tensor input to determine the device.")

        with forward_ad_safe_attention_context(first_tensor.device):
            f0, f_jvp = jvp(_f, self.theta0, tangents)
        out = f0 + f_jvp
        if pre_post_transform_callback is not None:
            pre_post_transform_callback(out)
        if post_transform is not None:
            out = post_transform(out)
        return out
