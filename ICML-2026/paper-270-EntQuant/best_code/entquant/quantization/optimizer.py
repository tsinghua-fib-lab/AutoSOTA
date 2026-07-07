"""Scale optimizers for quantization.

Provides entropy-aware LBFGS scale optimization
(:class:`SymmetricEntropyOptimizer`) and a simple absmax baseline
(:class:`WrappedAbsmaxOptimizer`), both compatible with the
``optimum.quanto.Optimizer`` interface.
"""

import logging
from typing import Callable, Literal

import torch
from optimum.quanto import AbsmaxOptimizer, qtype
from torch import dtype, Tensor

from ..utils import get_device
from .utils import entropy, LpNormDistance

logger = logging.getLogger(__name__)

__all__ = ["SymmetricEntropyOptimizer", "WrappedAbsmaxOptimizer"]


class Round(torch.autograd.Function):
    """Round with straight-through estimator (STE) gradient."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx, gy: Tensor) -> Tensor:
        return gy


class FakeFloatCast(torch.autograd.Function):
    """Cast to float qtype and back with straight-through estimator (STE) gradient."""

    @staticmethod
    def forward(ctx, x: Tensor, weight_dtype) -> Tensor:
        return x.to(weight_dtype).to(x.dtype)

    @staticmethod
    def backward(ctx, gy: Tensor) -> tuple[Tensor, None]:
        return gy, None


class STClamp(torch.autograd.Function):
    """Clamp with straight-through estimator (STE) gradient."""

    @staticmethod
    def forward(ctx, x: Tensor, xmin, xmax) -> Tensor:
        return torch.clamp(x, xmin, xmax)

    @staticmethod
    def backward(ctx, gy: Tensor) -> tuple[Tensor, None, None]:
        return gy, None, None


class SymmetricQuantizer(torch.nn.Module):
    """Differentiable symmetric quantizer for scale optimization.

    Parameterizes scales in log2-space for stable LBFGS optimization.
    Supports both integer and floating-point qtypes.
    """

    def __init__(self, scale: Tensor, weight_dtype: dtype, weight_qtype: qtype):
        super().__init__()
        self.weight_dtype = weight_dtype
        self.weight_qtype = weight_qtype
        self.log_scale = torch.nn.Parameter(torch.log2(scale.float()))

    @property
    def scale(self) -> Tensor:
        """Compute per-channel scale from log2 parameters."""
        scale = self.log_scale.to(self.weight_dtype)
        scale = torch.ones_like(scale).ldexp(scale)
        return scale

    def dequantize(self, x: Tensor) -> Tensor:
        """Simulate dequantization: multiply quantized values by scale."""
        return x.to(self.weight_dtype) * self.scale

    def quantize(self, x: Tensor) -> Tensor:
        """Quantize weights using differentiable round/clamp/cast."""
        assert len(x.shape) == 2, "Only Linear Layers are supported"

        # x = x / self.scale  # This would lead to optimization issues
        x = x.ldexp(-self.log_scale)  # This line is essential for optimization
        if not self.weight_qtype.is_floating_point:
            x = Round.apply(x)
        x = STClamp.apply(x, self.weight_qtype.qmin, self.weight_qtype.qmax)
        if self.weight_qtype.is_floating_point:
            x = FakeFloatCast.apply(x, self.weight_qtype.dtype)
        return x

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return (dequantized, quantized) representations."""
        x_quant = self.quantize(x)
        x_deq = self.dequantize(x_quant)
        return x_deq, x_quant


def l1_reg_fun(x: Tensor, weight_qtype: qtype) -> Tensor:
    """L1 entropy regularizer scaled by qtype range."""
    if weight_qtype.is_floating_point:
        return (torch.abs(x) * 4.0).mean()
    else:
        return (torch.abs(x) / 128.0).mean()


class SymmetricEntropyOptimizer(AbsmaxOptimizer):
    """Entropy-aware LBFGS scale optimizer (core of EntQuant).

    Minimizes reconstruction error + entropy regularization via
    differentiable quantization in log2-scale space. Defaults match
    the EntQuant paper; ``reg_param`` corresponds to lambda.
    """

    def __init__(
        self,
        norm_type: Literal["absolute", "relative", "relative_entrywise", "mean"] = "relative",
        norm_p: float = 1.0,
        reg_fun: Callable[[Tensor, qtype], Tensor] = l1_reg_fun,
        reg_param: float = 3.9,  # corresponds to ~4-bit
        lr: float = 1.0,
        maxiters: int = 500,
        history_size: int = 100,
        device_compute: str | torch.device | int | None = None,
        verbose: bool = True,
    ) -> None:
        """Initialize the entropy optimizer.

        Args:
            norm_type: Distance metric type for reconstruction loss.
            norm_p: Lp norm exponent used in the distance metric.
            reg_fun: Entropy regularization function.
            reg_param: Regularization strength (lambda in the paper).
            lr: LBFGS learning rate.
            maxiters: Maximum LBFGS iterations.
            history_size: LBFGS history size for Hessian approximation.
            device_compute: Device for optimization (None = same as
                input tensor).
            verbose: Whether to log start/end errors and entropy.
        """
        self.dist_fun = LpNormDistance(norm_type=norm_type, p=norm_p)
        self.dist_fun_eval = LpNormDistance(norm_type="relative", p=2.0)
        self.reg_fun = reg_fun
        self.reg_param = reg_param
        self.lr = lr
        self.maxiters = maxiters
        self.history_size = history_size
        self.verbose = verbose

        if device_compute is None:
            self.device_compute = None
        else:
            self.device_compute = get_device(device_compute)

    def optimize(
        self,
        base: torch.Tensor,
        weight_qtype: qtype,
        axis: int | None = None,
        lr: float | None = None,  # override for retries
    ) -> Tensor:
        """Optimize per-channel scales for a weight tensor.

        Args:
            base: Original weight tensor to quantize.
            weight_qtype: Target quantization type.
            axis: Quantization axis (unused, kept for API compat).
            lr: Learning rate override (used internally for retries).

        Returns:
            Optimized per-channel scale tensor.
        """
        device_host = base.device
        device_compute = self.device_compute if self.device_compute is not None else base.device

        # Get initial scales from standard AbsmaxOptimizer
        scale_orig = super(SymmetricEntropyOptimizer, self).optimize(base.to(device_compute), weight_qtype, axis)
        _base = base.clone().detach().to(device_compute)

        lr = self.lr if lr is None else lr

        quantizer = SymmetricQuantizer(scale_orig.clone().detach(), weight_dtype=_base.dtype, weight_qtype=weight_qtype)
        quantizer = quantizer.to(device_compute)

        if self.verbose:
            with torch.no_grad():
                deq_base, quant_base = quantizer(_base)
                error = self.dist_fun((deq_base - _base), _base)
                error_eval = self.dist_fun_eval((deq_base - _base), _base)
                logger.info(
                    f"[{str(device_compute)}] Start error: "
                    f"{error:.3f}, Start l2: {error_eval:.3f}, "
                    f"Start entropy: "
                    f"{entropy(quant_base).item():.3f}"
                )

        optimizer = torch.optim.LBFGS(
            quantizer.parameters(),
            lr=lr,
            max_iter=self.maxiters,
            history_size=self.history_size,
            line_search_fn="strong_wolfe",
        )

        def closure():
            """Loss function optimized by EntQuant."""
            optimizer.zero_grad()
            _deq_base, _quant_base = quantizer(_base)

            rec_loss = self.dist_fun((_deq_base - _base), _base)
            reg_loss = self.reg_param * self.reg_fun(_quant_base, weight_qtype)
            loss = rec_loss + reg_loss
            loss.backward()
            if loss.isnan():
                raise ValueError("Loss is NaN!")
            return loss

        try:
            optimizer.step(closure)
        except ValueError as e:
            logger.warning(f"Optimization failed: {e}\n Retry with lr={lr * 0.5}")
            return self.optimize(base, weight_qtype, axis, lr=lr * 0.5)

        if self.verbose:
            with torch.no_grad():
                deq_base, quant_base = quantizer(_base)
                error = self.dist_fun((deq_base - _base), _base)
                error_eval = self.dist_fun_eval((deq_base - _base), _base)
                logger.info(
                    f"[{str(device_compute)}] End error: "
                    f"{error:.3f}, End l2: {error_eval:.3f}, "
                    f"End entropy: "
                    f"{entropy(quant_base).item():.3f}"
                )

        scale = quantizer.scale.clone().detach()
        return scale.to(device_host)


class WrappedAbsmaxOptimizer(AbsmaxOptimizer):
    """Absmax optimizer with configurable compute device.

    Thin wrapper that moves tensors to ``device_compute`` before
    calling the parent absmax logic, then quantizes the scale to
    power-of-two for consistency with :class:`SymmetricEntropyOptimizer`.
    """

    def __init__(
        self,
        device_compute: str | torch.device | int | None = None,
    ) -> None:
        """Initialize the absmax optimizer.

        Args:
            device_compute: Device for scale computation (None = same
                as input tensor).
        """
        if device_compute is None:
            self.device_compute = None
        else:
            self.device_compute = get_device(device_compute)

    def optimize(
        self,
        base: torch.Tensor,
        weight_qtype: qtype,
        axis: int | None = None,
    ) -> Tensor:
        """Compute absmax scales on the configured device.

        Args:
            base: Original weight tensor.
            weight_qtype: Target quantization type.
            axis: Quantization axis (unused, kept for API compat).

        Returns:
            Optimized per-channel scale tensor.
        """
        device_compute = self.device_compute if self.device_compute is not None else base.device
        scale = (
            super(WrappedAbsmaxOptimizer, self).optimize(base.to(device_compute), weight_qtype, axis).to(base.device)
        )
        # Mimics SymmetricEntropyOptimizer behavior for consistency
        scale = torch.log2(scale)
        return torch.ones_like(scale).ldexp(scale)
