from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class _SurrogateHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = float(alpha)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        alpha = float(ctx.alpha)
        sig = torch.sigmoid(alpha * x)
        grad = alpha * sig * (1.0 - sig)
        return grad_output * grad, None


class NativeMultiStepLIFNode(nn.Module):
    """Pure PyTorch multi-step LIF node."""

    def __init__(
        self,
        *,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        detach_reset: bool = True,
        surrogate_alpha: float = 4.0,
    ) -> None:
        super().__init__()
        if tau <= 1.0:
            raise ValueError(f"tau must be > 1, got {tau}")
        self.tau = float(tau)
        self.v_threshold = float(v_threshold)
        self.v_reset = None if v_reset is None else float(v_reset)
        self.detach_reset = bool(detach_reset)
        self.surrogate_alpha = float(surrogate_alpha)
        self.register_buffer("_last_v", torch.tensor(0.0), persistent=False)

    def reset(self) -> None:
        self._last_v = torch.tensor(0.0, device=self._last_v.device, dtype=self._last_v.dtype)

    def _step(self, x_t: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.v_reset is None:
            v = v + (x_t - v) / self.tau
        else:
            v = v + (x_t - (v - self.v_reset)) / self.tau

        spike = _SurrogateHeaviside.apply(v - self.v_threshold, self.surrogate_alpha)
        reset_spike = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            v = v - reset_spike * self.v_threshold
        else:
            v = v * (1.0 - reset_spike) + self.v_reset * reset_spike
        return spike, v

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.dim() < 2:
            raise ValueError(f"Expected x_seq with time as first dim, got {tuple(x_seq.shape)}")
        t = x_seq.shape[0]
        v = torch.zeros_like(x_seq[0])
        spikes = []
        for i in range(t):
            s, v = self._step(x_seq[i], v)
            spikes.append(s)
        self._last_v = v.detach()
        return torch.stack(spikes, dim=0)


class _SpikingJellyLIFWrapper(nn.Module):
    def __init__(
        self,
        *,
        tau: float,
        v_threshold: float,
        v_reset: Optional[float],
        detach_reset: bool,
        surrogate_alpha: float,
        backend: str,
    ) -> None:
        super().__init__()
        from spikingjelly.activation_based import neuron, surrogate

        sj_backend = backend
        if sj_backend not in {"torch", "cupy", "triton"}:
            sj_backend = "torch"

        self.node = neuron.LIFNode(
            tau=float(tau),
            v_threshold=float(v_threshold),
            v_reset=v_reset,
            detach_reset=bool(detach_reset),
            surrogate_function=surrogate.Sigmoid(alpha=float(surrogate_alpha)),
            step_mode="m",
            backend=sj_backend,
        )

    def reset(self) -> None:
        self.node.reset()

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        return self.node(x_seq)


def _has_spikingjelly() -> bool:
    try:
        import spikingjelly  # noqa: F401
        return True
    except Exception:
        return False


def make_multistep_lif_node(
    *,
    spike_backend: str = "native",
    tau: float = 2.0,
    v_threshold: float = 1.0,
    v_reset: Optional[float] = 0.0,
    detach_reset: bool = True,
    surrogate_alpha: float = 4.0,
    backend: str = "torch",
) -> nn.Module:
    mode = str(spike_backend).strip().lower()
    if mode not in {"native", "spikingjelly", "auto"}:
        raise ValueError(f"Unknown spike_backend={spike_backend!r}")

    use_sj = mode == "spikingjelly" or (mode == "auto" and _has_spikingjelly())
    if use_sj:
        if not _has_spikingjelly():
            if mode == "spikingjelly":
                raise ImportError("spikingjelly is not installed, but spike_backend='spikingjelly' was requested")
        else:
            try:
                return _SpikingJellyLIFWrapper(
                    tau=tau,
                    v_threshold=v_threshold,
                    v_reset=v_reset,
                    detach_reset=detach_reset,
                    surrogate_alpha=surrogate_alpha,
                    backend=backend,
                )
            except Exception:
                if mode == "spikingjelly":
                    raise

    return NativeMultiStepLIFNode(
        tau=tau,
        v_threshold=v_threshold,
        v_reset=v_reset,
        detach_reset=detach_reset,
        surrogate_alpha=surrogate_alpha,
    )


__all__ = ["NativeMultiStepLIFNode", "make_multistep_lif_node"]
