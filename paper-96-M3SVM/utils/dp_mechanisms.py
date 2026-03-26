import math
import torch
from torch import nn
from torch.autograd import functional as F

# Utility: convert between (eps, delta)-DP and rho-zCDP

def eps_to_zcdp(eps: float, delta: float) -> float:
    """
    Convert (ε,δ)-DP to ρ-zCDP.
    ρ = ε² / (4·log(1/δ))
    """
    return eps * eps / (4 * math.log(1 / delta))


def zcdp_to_eps(rho: float, delta: float) -> float:
    """
    Convert ρ-zCDP to (ε,δ)-DP.
    ε = ρ + 2·sqrt(ρ·log(1/δ))
    """
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))


class TorchLRWrapper:
    """
    Wrapper for a PyTorch nn.Linear multiclass classifier,
    providing flat-parameter gradient and Hessian for DP mechanisms.

    X : torch.Tensor of shape [n, d]
    y : torch.LongTensor of shape [n] (labels 0..C-1)
    p : int, norm degree for regularization
    lam : float, regularization strength
    """
    def __init__(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor, p: int, lam: float):
        self.model = model
        self.X = X
        self.y = y
        self.p = p
        self.lam = lam
        # number of samples
        self.n = X.size(0)
        # total number of parameters
        self.d = sum(param.numel() for param in model.parameters())

    def get_flat_params(self) -> torch.Tensor:
        """Flatten model.parameters() into a single vector."""
        return torch.cat([param.data.view(-1) for param in self.model.parameters()])

    def set_flat_params(self, w: torch.Tensor):
        """Distribute flat vector w into model.parameters()."""
        offset = 0
        for param in self.model.parameters():
            num = param.numel()
            chunk = w[offset:offset+num].view_as(param)
            param.data.copy_(chunk.to(param.device, param.dtype))
            offset += num

    def loss(self, w: torch.Tensor) -> torch.Tensor:
        """Compute loss at flattened parameter w."""
        self.set_flat_params(w)
        logits = self.model(self.X)
        ce = torch.nn.functional.cross_entropy(logits, self.y, reduction='mean')
        # l_p-norm regularization
        # assume a single Linear layer named `.fc` or first parameter
        try:
            W = getattr(self.model, 'fc').weight
        except AttributeError:
            W = next(self.model.parameters())
        reg = self.lam * torch.norm(W, p=self.p) / self.n
        return ce + reg

    def grad(self, w: torch.Tensor) -> torch.Tensor:
        """Compute gradient of loss at w."""
        self.model.zero_grad()
        L = self.loss(w)
        L.backward()
        return torch.cat([param.grad.view(-1) for param in self.model.parameters()])

    def hess(self, w: torch.Tensor) -> torch.Tensor:
        """Compute full Hessian matrix of loss at w (costly)."""
        return F.hessian(lambda v: self.loss(v), w)


class DoubleNoiseMech:
    """
    DP Algorithm 3: Double-Noise Mechanism (add/clip, Hessian/UB)
    Works on flat-parameter vector w (torch.Tensor).
    Requires lr_wrapper with methods:
      - grad(w): returns gradient torch.Tensor [d]
      - hess(w): returns Hessian torch.Tensor [d,d]
      - n (int): number of samples
    """
    def __init__(self, lr_wrapper: TorchLRWrapper, type_reg='add', hyper_tuning=False, curvature_info='hessian'):
        self.lr = lr_wrapper
        self.type_reg = type_reg
        self.hyper_tuning = hyper_tuning
        # choose second-order information function
        if curvature_info == 'hessian':
            self.H = lr_wrapper.hess
        else:
            self.H = lr_wrapper.hess  # no separate upperbound implemented

    def _hessian_clip(self, Hmat: torch.Tensor, min_eval: float) -> torch.Tensor:
        evals, evecs = torch.linalg.eigh(Hmat)
        clipped = torch.clamp(evals, min=min_eval)
        return (evecs * clipped) @ evecs.T

    def update_rule(self, w: torch.Tensor, lr_wrapper: TorchLRWrapper, step: int, total_steps: int, pb: dict) -> torch.Tensor:
        rho_total = pb['total']
        rho1 = pb['grad_frac'] * rho_total / total_steps
        rho2 = (1 - pb['grad_frac']) * rho_total / total_steps
        # gradient privatization
        grad = lr_wrapper.grad(w)
        sc1 = (1.0 / lr_wrapper.n) * math.sqrt(0.5 / rho1)
        noisy_grad = grad + torch.randn_like(grad) * sc1
        # second-order info
        Hmat = self.H(w)
        # min_eval selection
        if self.hyper_tuning:
            min_eval = self.find_opt_reg_wop(w, lr_wrapper, noisy_grad, rho2)
        else:
            trace = torch.trace(Hmat)
            sc_trace = (0.25 / lr_wrapper.n) * math.sqrt(0.5 / (pb['trace_frac'] * rho2))
            noisy_trace = max((trace + torch.randn(()) * sc_trace).item(), 0.0)
            min_eval = max((noisy_trace / ((lr_wrapper.n**2) * (1-pb['trace_frac']) * rho2))**(1/3), 1.0/lr_wrapper.n)
        # modify Hessian
        if self.type_reg == 'add':
            Hmod = Hmat + min_eval * torch.eye(Hmat.size(0), device=w.device)
        else:
            Hmod = self._hessian_clip(Hmat, min_eval)
        # solve direction
        direction = torch.linalg.solve(Hmod, noisy_grad)
        # direction privatization
        grad_norm = torch.norm(noisy_grad)
        m = 0.25
        if self.type_reg == 'add':
            sens2 = grad_norm * m / (lr_wrapper.n * min_eval**2 + m * min_eval)
        else:
            sens2 = grad_norm * m / (lr_wrapper.n * min_eval**2 - m * min_eval)
        noisy_dir = direction + torch.randn_like(direction) * (sens2 * math.sqrt(0.5 / rho2))
        return w - noisy_dir

    def update_rule_stochastic(self, w: torch.Tensor, lr_wrapper: TorchLRWrapper, step: int, total_steps: int, pb: dict) -> torch.Tensor:
        # minibatch gradient
        p1 = pb['batchsize_grad'] / lr_wrapper.n
        mask1 = (torch.rand(lr_wrapper.n, device=w.device) < p1)
        grad_mb = lr_wrapper.grad(w)  # ideally select subset inside wrapper
        noisy_grad = grad_mb + torch.randn_like(grad_mb) * pb['noise_multiplier_grad']
        # minibatch Hessian
        p2 = pb['batchsize_hess'] / lr_wrapper.n
        Hmat = lr_wrapper.hess(w)
        if self.type_reg == 'add':
            Hmod = Hmat + pb['min_eval'] * torch.eye(Hmat.size(0), device=w.device)
        else:
            Hmod = self._hessian_clip(Hmat, pb['min_eval'])
        direction = torch.linalg.solve(Hmod, noisy_grad)
        grad_norm = torch.norm(noisy_grad)
        m = 0.25
        if self.type_reg == 'add':
            sens2 = grad_norm * m / ((lr_wrapper.n*p2) * pb['min_eval']**2 + m * pb['min_eval'])
        else:
            sens2 = grad_norm * m / ((lr_wrapper.n*p2) * pb['min_eval']**2 - m * pb['min_eval'])
        noisy_dir = direction + torch.randn_like(direction) * pb['noise_multiplier_hess']
        return w - noisy_dir
