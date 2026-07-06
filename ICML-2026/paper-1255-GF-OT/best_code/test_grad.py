import torch
import numpy as np
from src.loss_funcs import quota_loss
from src.datagen import get_gaussian_mixture
import ot

print("Testing torch.func.grad...")
(X, Y), (S_X, S_Y) = get_gaussian_mixture(d=2, n_x=250, n_y=25, scale=0.2, p_x0=0.5, p_y0=0.5, rng=42)
eps = 1.0
F_target = torch.tensor([[0.2, 0.3], [0.28, 0.22]])

C = ot.dist(X.numpy(), Y.numpy(), metric="sqeuclidean")
C = torch.from_numpy(C).float()
a = torch.ones(250) / 250
b = torch.ones(25) / 25
T0 = ot.sinkhorn(a.numpy(), b.numpy(), C.numpy(), eps)
T0 = torch.from_numpy(T0).float()

print("T0 shape:", T0.shape)
loss_val = quota_loss(T0, S_X, S_Y, F_target)
print("quota_loss value:", loss_val.item())

print("Computing gradient...")
grad_fn = torch.func.grad(quota_loss, argnums=0)
grad_val = grad_fn(T0, S_X, S_Y, F_target)
print("gradient shape:", grad_val.shape)
print("gradient norm:", torch.norm(grad_val).item())
print("Success!")
