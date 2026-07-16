"""Kernel specifications for GPyTorch regressors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import gpytorch
import numpy as np
import torch

KernelParamValue = (
    float
    | np.ndarray
    | torch.Tensor
    | Sequence[float]
    | Sequence[Sequence[float]]
    | Sequence[Sequence[Sequence[float]]]
)


def _as_float(value: float | np.ndarray | torch.Tensor) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0])
    return float(value)


def _resolve_module_attr(
    kernel: gpytorch.kernels.Kernel, name: str
) -> tuple[torch.nn.Module, str]:
    if "." in name:
        parts = name.split(".")
        module: torch.nn.Module = kernel
        for part in parts[:-1]:
            if not hasattr(module, part):
                raise ValueError(f"Kernel parameter '{name}' not found.")
            module = getattr(module, part)
        attr = parts[-1]
        if not hasattr(module, attr):
            raise ValueError(f"Kernel parameter '{name}' not found.")
        return module, attr
    if hasattr(kernel, "base_kernel") and hasattr(kernel.base_kernel, name):
        return kernel.base_kernel, name
    if hasattr(kernel, name):
        return kernel, name
    raise ValueError(f"Kernel parameter '{name}' not found.")


def _coerce_param_value(
    module: torch.nn.Module, attr: str, value: KernelParamValue
) -> float | torch.Tensor:
    if isinstance(value, (float, int)):
        return float(value)
    if torch.is_tensor(value):
        target = getattr(module, attr, None)
        if torch.is_tensor(target):
            return value.to(device=target.device, dtype=target.dtype)
        return value
    arr = np.asarray(value, dtype=float)
    target = getattr(module, attr, None)
    if torch.is_tensor(target):
        return torch.as_tensor(arr, dtype=target.dtype, device=target.device)
    return torch.as_tensor(arr)


def _initialize_parameter(
    kernel: gpytorch.kernels.Kernel, name: str, value: KernelParamValue
) -> None:
    module, attr = _resolve_module_attr(kernel, name)
    value = _coerce_param_value(module, attr, value)
    try:
        module.initialize(**{attr: value})
    except (
        Exception
    ) as exc:  # pragma: no cover - defensive for unexpected gpytorch errors
        raise ValueError(f"Failed to initialize kernel parameter '{name}'.") from exc


def _parameter_alias_map(kernel: gpytorch.kernels.Kernel) -> dict[str, str]:
    alias: dict[str, str] = {}
    for raw_name, _ in kernel.named_parameters():
        alias[raw_name] = raw_name
        parts = raw_name.split(".")
        tail = parts[-1]
        if tail.startswith("raw_"):
            friendly_tail = tail[4:]
            friendly_full = (
                ".".join(parts[:-1] + [friendly_tail]) if parts[:-1] else friendly_tail
            )
            alias[friendly_full] = raw_name
            alias[friendly_tail] = raw_name
        if raw_name.startswith("base_kernel."):
            alias[raw_name[len("base_kernel.") :]] = raw_name
    return alias


def _configure_trainable(
    kernel: gpytorch.kernels.Kernel, trainable_parameters: Sequence[str] | None
) -> None:
    if trainable_parameters is None:
        return
    alias_map = _parameter_alias_map(kernel)
    trainable_raw: set[str] = set()
    for name in trainable_parameters:
        if name not in alias_map:
            known = ", ".join(sorted(set(alias_map.keys())))
            raise ValueError(
                f"Unknown kernel parameter '{name}'. Known parameters: {known}"
            )
        trainable_raw.add(alias_map[name])
    for param_name, param in kernel.named_parameters():
        param.requires_grad_(param_name in trainable_raw)


def _read_parameter(kernel: gpytorch.kernels.Kernel, name: str) -> float:
    module, attr = _resolve_module_attr(kernel, name)
    value = getattr(module, attr)
    if value is None:
        raise ValueError(f"Kernel parameter '{name}' is None.")
    return _as_float(value)


def _median_lengthscale(times: np.ndarray) -> float | None:
    if times.size < 2:
        return None
    diffs = np.abs(times[:, None] - times[None, :])
    upper = diffs[np.triu_indices(times.size, k=1)]
    upper = upper[upper > 0]
    if upper.size == 0:
        return None
    return float(np.median(upper))


@dataclass(frozen=True)
class KernelSpec(ABC):
    """Specification for building and configuring GPyTorch kernels."""

    trainable_parameters: Sequence[str] | None = None
    track_parameters: Sequence[str] | None = None
    use_scale: bool = True

    @abstractmethod
    def _build_base_kernel(self) -> gpytorch.kernels.Kernel:
        """Build the unscaled base kernel."""

    def _init_parameters(self) -> dict[str, KernelParamValue]:
        return {}

    def parameter_overrides_from_inputs(
        self, times: np.ndarray
    ) -> dict[str, KernelParamValue] | None:
        return None

    def build(
        self,
        *,
        outputscale: float | None = None,
        parameter_overrides: dict[str, KernelParamValue] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> gpytorch.kernels.Kernel:
        kernel = self._build_base_kernel()
        if self.use_scale:
            kernel = gpytorch.kernels.ScaleKernel(kernel)
        if device is not None or dtype is not None:
            kernel = kernel.to(device=device, dtype=dtype)
        init_params = dict(self._init_parameters())
        if parameter_overrides:
            init_params.update(parameter_overrides)
        if outputscale is not None:
            if not self.use_scale:
                raise ValueError(
                    "outputscale provided but kernel spec disables ScaleKernel"
                )
            init_params["outputscale"] = float(outputscale)
        for name, value in init_params.items():
            _initialize_parameter(kernel, name, value)
        _configure_trainable(kernel, self.trainable_parameters)
        return kernel

    def snapshot(self, kernel: gpytorch.kernels.Kernel) -> dict[str, float]:
        track = self.track_parameters
        if track is None:
            track = list(self._init_parameters().keys()) + [
                "lengthscale",
                "outputscale",
            ]
        snapshot: dict[str, float] = {}
        for name in track:
            try:
                snapshot[name] = _read_parameter(kernel, name)
            except ValueError:
                continue
        return snapshot

    def supports_parameter(self, name: str) -> bool:
        kernel = self.build()
        try:
            _resolve_module_attr(kernel, name)
        except ValueError:
            return False
        return True


@dataclass(frozen=True)
class RBFKernelSpec(KernelSpec):
    """RBF kernel with optional lengthscale initialization."""

    lengthscale: float = 1.0

    def _build_base_kernel(self) -> gpytorch.kernels.Kernel:
        return gpytorch.kernels.RBFKernel()

    def _init_parameters(self) -> dict[str, KernelParamValue]:
        return {"lengthscale": float(self.lengthscale)}

    def parameter_overrides_from_inputs(
        self, times: np.ndarray
    ) -> dict[str, KernelParamValue] | None:
        median_ls = _median_lengthscale(times)
        if median_ls is None or median_ls <= 0:
            return None
        return {"lengthscale": median_ls}


@dataclass(frozen=True)
class MaternKernelSpec(KernelSpec):
    """Matern kernel with optional lengthscale initialization."""

    nu: float = 2.5
    lengthscale: float = 1.0

    def _build_base_kernel(self) -> gpytorch.kernels.Kernel:
        return gpytorch.kernels.MaternKernel(nu=self.nu)

    def _init_parameters(self) -> dict[str, KernelParamValue]:
        return {"lengthscale": float(self.lengthscale)}

    def parameter_overrides_from_inputs(
        self, times: np.ndarray
    ) -> dict[str, KernelParamValue] | None:
        median_ls = _median_lengthscale(times)
        if median_ls is None or median_ls <= 0:
            return None
        return {"lengthscale": median_ls}
