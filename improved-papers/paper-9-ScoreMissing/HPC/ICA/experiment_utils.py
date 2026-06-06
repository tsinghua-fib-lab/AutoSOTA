from abc import abstractmethod
from functools import partial
import torch
from torch import nn, optim
import numpy as np
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from MSM.utils import hmc  # noqa: E402
from MSM.models import density_models as models  # noqa: E402


def ica_log_prob(X, Theta):
    Z = X**2
    return -0.5 * np.sum((Z @ Theta) * Z, axis=-1)


def ica_grad_log_prob(X, Theta):
    X = torch.tensor(X, requires_grad=True, dtype=torch.float32)
    Theta = torch.tensor(Theta, dtype=torch.float32)
    Z = X**2
    log_prob = -0.5 * torch.sum((Z @ Theta) * Z)
    log_prob.backward()
    return X.grad.numpy()


class MyModel:
    def __init__(self, log_prob, grad_log_prob):
        self.log_prob = log_prob
        self.grad_log_prob = grad_log_prob

    def lnprob(self, x):
        return self.log_prob(x)

    def lnprob_grad(self, x):
        return self.grad_log_prob(x)


def sample_ica(theta: torch.Tensor, iterations, initial_val=None, thin=1, burn=0, use_map=False, **kwargs):
    dim = theta.shape[0]
    if use_map:
        true_model = ICAModel(dim=dim, Theta=theta)
        # Train to find MAP X value w.r.t. this model
        x_start = torch.randn(dim, requires_grad=True)
        temp_opt = optim.SGD([x_start], lr=0.01)
        log_probs = []
        for _ in range(1000):
            temp_opt.zero_grad()
            # Calculate log prob
            neg_log_prob = - true_model.log_prob(x_start)
            log_probs.append(-neg_log_prob.item())
            neg_log_prob.backward()
            temp_opt.step()
        initial_val = x_start.detach().numpy()

    iterations = iterations * thin + burn
    if isinstance(theta, torch.Tensor):
        theta = theta.numpy()
    my_model = MyModel(partial(ica_log_prob, Theta=theta), partial(ica_grad_log_prob, Theta=theta))
    sampler = hmc.BasicHMC(my_model, verbose=False)
    for _ in range(1000):
        initial = np.random.normal(size=(dim,)) if initial_val is None else initial_val
        try:
            _ = sampler.sample(initial, iterations=iterations, **kwargs)
        except ValueError as e:
            if e.args[0] == 'alpha is 0':
                continue
            else:
                raise e
        if hasattr(sampler, 'chain'):
            if np.mean(sampler.accepted) > 0.6:
                break
    sample = torch.tensor(sampler.chain[burn::thin], dtype=torch.float32)
    return sample, sampler


class ICAModel(models.UDensity):
    def __init__(self, dim: None, Theta: torch.Tensor = None):
        super().__init__()
        if (dim is None) and (Theta is None):
            raise ValueError("Must specify either dim or W")
        if (Theta is not None) and (Theta.shape[0] != Theta.shape[1]):
            raise ValueError("W must be square")
        if Theta is None:
            self.dim = dim
            Theta = torch.randn(dim, dim)
        else:
            self.dim = Theta.shape[0]
        self.Theta = nn.Parameter(Theta.detach().clone())

    def log_prob(self, X: torch.Tensor):
        Z = X**2
        return -0.5 * torch.sum((Z @ self.Theta) * Z, dim=-1)


class ICAParamModel(ICAModel):
    @abstractmethod
    def update_params(self):
        pass

    def log_prob(self, X: torch.Tensor):
        self.update_params()
        return super().log_prob(X)

    def forward(self, X: torch.Tensor):
        self.update_params()
        return super().forward(X)


class ICACholeskyModel(ICAParamModel):
    def __init__(self, dim: None, Theta: torch.Tensor = None):
        nn.Module.__init__(self)
        if (dim is None) and (Theta is None):
            raise ValueError("Must specify either dim or W")
        if (Theta is not None) and (Theta.shape[0] != Theta.shape[1]):
            raise ValueError("W must be square")
        if Theta is None:
            self.dim = dim
            Theta = torch.randn(dim, dim)
        else:
            self.dim = Theta.shape[0]

        chol_Theta = torch.linalg.cholesky(Theta)
        self.chol_Theta = nn.Parameter(chol_Theta.detach().clone())
        self.Theta: torch.Tensor
        self.register_buffer("Theta", Theta.detach().clone())

    def update_params(self):
        self.Theta = self.chol_Theta @ self.chol_Theta.T


def create_strong_cov(dim, connection_strength=0.5):
    sub_dim = dim - 1
    # Create sub covariance
    orthonormal = torch.linalg.qr(torch.randn(sub_dim, sub_dim))[0]
    sub_cov = orthonormal @ torch.diag(torch.rand(sub_dim) + 0.5) @ orthonormal.T
    # Create additional covariance
    sub_cov2 = torch.zeros((dim, dim))
    sub_cov2[:sub_dim, :][:, :sub_dim] = sub_cov
    sub_cov2[sub_dim, sub_dim] = sub_cov[0, 0]
    # Create relationship matrix
    A = torch.eye(dim)
    A[sub_dim, 0] = connection_strength
    A[sub_dim, sub_dim] = 1-connection_strength
    # Construct full covariance
    temp_cov = A @ sub_cov2 @ A.T
    return temp_cov


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
