from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from types import MethodType

import torch
import torch.nn as nn
from torch.func import functional_call


class _DeltaModule(nn.Module):
    def __init__(self, names: list[str], params: list[nn.Parameter]) -> None:
        super().__init__()
        self.names = tuple(names)
        self.params = nn.ParameterList(params)

    def by_name(self) -> dict[str, nn.Parameter]:
        return {name: param for name, param in zip(self.names, self.params, strict=True)}


def _sync_runtime_feature_cache(*, target: nn.Module, source: nn.Module) -> None:
    for attr in ("_last_visual_features", "_last_image_features", "_last_logits"):
        setattr(target, attr, getattr(source, attr, None))

    target_classifier = getattr(target, "clip_model", None)
    source_classifier = getattr(source, "clip_model", None)
    if target_classifier is None or source_classifier is None:
        return
    for attr in ("_last_visual_features", "_last_image_features", "_last_logits"):
        setattr(target_classifier, attr, getattr(source_classifier, attr, None))


def _materialized_param_map(
    *,
    named_params: list[tuple[str, nn.Parameter]],
    base_params: Mapping[str, torch.Tensor],
    delta_module: _DeltaModule,
) -> dict[str, torch.Tensor]:
    deltas = delta_module.by_name()
    param_map: dict[str, torch.Tensor] = {}
    for name, param in named_params:
        if name in deltas:
            delta = deltas[name]
            base = base_params[name].to(device=delta.device, dtype=delta.dtype)
            param_map[name] = base + delta
        else:
            param_map[name] = param
    return param_map


def _materialized_state_dict(
    *,
    model: nn.Module,
    named_params: list[tuple[str, nn.Parameter]],
    base_params: Mapping[str, torch.Tensor],
    delta_module: _DeltaModule,
) -> dict[str, torch.Tensor]:
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for key in [key for key in state if key.startswith("_delta_module.")]:
        state.pop(key, None)
    deltas = delta_module.by_name()
    for name, _param in named_params:
        if name not in deltas or name not in state or name not in base_params:
            continue
        delta = deltas[name]
        base = base_params[name].to(device=delta.device, dtype=delta.dtype)
        state[name] = (base + delta).detach().cpu()
    return state


def bind_delta_parameterization(
    *,
    model: nn.Module,
    named_params: list[tuple[str, nn.Parameter]],
    target_names: set[str],
    device: torch.device,
) -> list[nn.Parameter]:
    if not target_names:
        raise RuntimeError("strategy.params.parameterization='delta' selected zero parameters.")

    base_params = {name: param.detach().to(device=device).clone() for name, param in named_params if name in target_names}
    delta_names: list[str] = []
    delta_params: list[nn.Parameter] = []
    for name, param in named_params:
        if name not in target_names:
            continue
        param.requires_grad_(False)
        delta_names.append(name)
        delta_params.append(nn.Parameter(torch.zeros_like(param, dtype=param.dtype, device=device)))

    ref_model = deepcopy(model).to(device)
    delta_module = _DeltaModule(delta_names, delta_params).to(device)

    def _current_param_map() -> dict[str, torch.Tensor]:
        return _materialized_param_map(named_params=named_params, base_params=base_params, delta_module=delta_module)

    def _materialized_state() -> dict[str, torch.Tensor]:
        return _materialized_state_dict(
            model=model,
            named_params=named_params,
            base_params=base_params,
            delta_module=delta_module,
        )

    def _delta_forward(self, *args, **kwargs):
        out = functional_call(ref_model, _current_param_map(), args=args, kwargs=kwargs, strict=False)
        _sync_runtime_feature_cache(target=self, source=ref_model)
        return out

    model._delta_module = delta_module  # type: ignore[attr-defined]
    model._current_param_map = _current_param_map  # type: ignore[attr-defined]
    model._materialized_state_dict = _materialized_state  # type: ignore[attr-defined]
    model.forward = MethodType(_delta_forward, model)  # type: ignore[method-assign]
    return list(delta_module.params)
