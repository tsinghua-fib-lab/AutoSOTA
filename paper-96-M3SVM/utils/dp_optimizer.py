import torch
import math
import numpy as np
from scipy.optimize import fsolve
from opacus.accountants import GaussianAccountant


def zcdp_to_eps(rho: float, delta: float) -> float:
    return rho + math.sqrt(4 * rho * math.log(math.sqrt(math.pi * rho) / delta))


def eps_to_zcdp(eps: float, delta: float) -> float:
    func = lambda x: zcdp_to_eps(x, delta) - eps
    root = fsolve(func, x0=np.array([0.001]))[0]
    return root


def calibrateAnalyticGaussianMechanism(eps: float, delta: float, clip_norm: float) -> float:
    rho = eps_to_zcdp(eps, delta)
    return clip_norm * math.sqrt(2 * rho)


class DoubleNoiseMech:
    """Double Noise Mechanism for DP Newton-like Update on nn.Linear."""
    def __init__(
        self,
        eps: float,
        delta: float,
        clip_norm: float,
        grad_frac: float = 0.5,
        trace_frac: float = 0.2,
        trace_coeff: float = 1.0,
        type_reg: str = 'add',
        hyper_tuning: bool = False
    ):
        self.eps = eps
        self.delta = delta
        self.clip_norm = clip_norm
        self.grad_frac = grad_frac
        self.trace_frac = trace_frac
        self.trace_coeff = trace_coeff
        self.type_reg = type_reg
        self.hyper_tuning = hyper_tuning
        self.accountant = GaussianAccountant()

    def update_rule(
        self,
        model: torch.nn.Linear,
        X: torch.Tensor,
        y: torch.Tensor,
        i: int,
        iters: int
    ) -> torch.nn.Linear:
        """
        Perform one DP Newton-like update step on nn.Linear parameters.
        Prints spent ε after the step.
        """
        device = X.device
        N, D = X.shape
        C = model.bias.numel()

        # Prepare flat parameters requiring gradients
        bias = model.bias.data.clone().view(-1)
        weight = model.weight.data.clone().flatten()
        flat_params = torch.cat((bias, weight)).to(device)
        flat_params.requires_grad_(True)

        # Forward using flat_params
        b = flat_params[:C]
        W = flat_params[C:].view(C, D)
        logits = X @ W.T + b
        # Convert targets
        if y.dim() > 1:
            target = y.argmax(dim=1)
        else:
            target = y.long()
        loss = torch.nn.functional.cross_entropy(logits, target, reduction='mean')

        # Gradient w.r.t flat_params
        grad_vec = torch.autograd.grad(loss, flat_params, create_graph=True)[0]

        # Hessian w.r.t flat_params
        def flat_loss(params):
            b_ = params[:C]
            W_ = params[C:].view(C, D)
            logits_ = X @ W_.T + b_
            return torch.nn.functional.cross_entropy(logits_, target, reduction='mean')
        H = torch.autograd.functional.hessian(flat_loss, flat_params)

        # Noise on gradient
        rho_total = eps_to_zcdp(self.eps, self.delta)
        rho_grad = self.grad_frac * rho_total / iters
        sc1 = (1.0 / N) * math.sqrt(0.5 / rho_grad)
        noise1 = torch.randn_like(grad_vec) * sc1
        noisy_grad = grad_vec + noise1
        g_norm = noisy_grad.norm()

        # Noise on Hessian and clip
        rho_H = (1.0 - self.grad_frac) * rho_total / iters
        sc2 = (0.25 / N) * math.sqrt(0.5 / rho_H)
        H_noise = torch.randn_like(H) * sc2
        H_noise = 0.5 * (H_noise + H_noise.T)
                # Add noise to Hessian
        H_mod = H + H_noise

        # Ensure symmetry, contiguous, and use double precision for eigen-decomposition
        H_mod = 0.5 * (H_mod + H_mod.transpose(-1, -2))
        H_mod = H_mod.contiguous().double()

        # Eigen-decomposition with fallback to CPU if required
        try:
            evals, evecs = torch.linalg.eigh(H_mod, UPLO='U')
        except RuntimeError:
            H_cpu = H_mod.cpu()
            evals, evecs = torch.linalg.eigh(H_cpu, UPLO='U')
            evals = evals.to(device)
            evecs = evecs.to(device)

        # Clamp eigenvalues and reconstruct Hessian
        min_eval = 1.0 / N
        evals_clamped = torch.clamp(evals, min=min_eval)
        H_clipped = evecs @ torch.diag(evals_clamped) @ evecs.T
        H_clipped = H_clipped.float()
