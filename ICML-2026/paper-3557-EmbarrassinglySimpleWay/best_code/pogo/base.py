import math
import numpy as np
import torch


__all__ = ["BaseOptimizer", "SGD", "VectorAdam", "Muon"]


class BaseOptimizer:
    defaults: dict

    def __call__(self, point: torch.Tensor, grad: torch.Tensor, state, group) -> torch.Tensor:
        pass

    def get_defaults(self) -> dict:
        return self.defaults


class SGD(BaseOptimizer):
    def __init__(self, momentum=0, nesterov=False, dampening=0):
        self.defaults = dict(momentum=momentum, nesterov=nesterov, dampening=dampening)

    def __call__(self, point, grad, state, group):
        momentum = group["momentum"]
        dampening = group["dampening"]
        
        if (group['step'] == 1) and (momentum > 0):
            state["momentum_buffer"] = grad.clone()
        
        if momentum > 0:
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(
                grad, alpha=1 - dampening
            )
            grad = grad.add(buf, alpha=momentum) if group["nesterov"] else buf
            
        return grad

class VectorAdam(BaseOptimizer):
    def __init__(self, betas=(0.9, 0.999), eps=1e-08):
        self.defaults = dict(betas=betas, eps=eps)
        
    def __call__(self, point, grad, state, group):
        betas = group["betas"]
        eps = group["eps"]
                        
        # State initialization
        if group['step'] == 1:
            state["exp_avg"] = torch.zeros_like(grad)
            state["exp_avg_sq"] = torch.zeros(*grad.shape[:-2], device=grad.device, dtype=grad.dtype)
        
        # Compute VectorAdam update (TODO denominator?)
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        exp_avg.lerp_(grad, 1 - betas[0])
        exp_avg_sq.mul_(betas[1]).add_(torch.linalg.norm(grad, ord='fro', dim=(-1, -2)).pow(2), alpha=1 - betas[1])  # Vector Adam

        bias_correction1 = 1 - betas[0] ** group['step']
        bias_correction2 = 1 - betas[1] ** group['step']
        bias_correction2_sqrt = bias_correction2 ** 0.5
        
        step_size = 1. / bias_correction1

        denom = (
            exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size)
        ).add_(eps / step_size)
        denom = denom.view(-1, 1, 1)
    
        return exp_avg / denom


@torch.compile
def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G / (G.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(SGD):
    def __call__(self, point, grad, state, group):
        grad = super().__call__(point, grad, state, group)
        return zeropower_via_newtonschulz5(grad.reshape(len(grad), -1)).view(grad.shape) # whiten the update

