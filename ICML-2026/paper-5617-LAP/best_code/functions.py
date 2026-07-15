# MIT License
#
# Copyright (c) 2024 Antonio Terpin, Nicolas Lanzetti, Martin Gadea, Florian Dörfler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
This module provides a collection of energy landscape functions commonly used for optimization,
testing, and benchmarking purposes. These functions have been translated to PyTorch.
"""

import torch
import math


def double_exp(v):
    v = v.squeeze()
    return torch.exp(-torch.sum((v - 2.) ** 2)) + torch.exp(-torch.sum((v + 2.) ** 2))


def rotational(v):
    v = v.squeeze()
    alpha_0 = 1.
    alpha_1 = 0.5
    u_0 = torch.tensor([1., 0.], device=v.device, dtype=v.dtype)
    u_1 = torch.tensor([0., 1.], device=v.device, dtype=v.dtype)
    return alpha_0 * torch.sum(v * u_0) ** 2 + alpha_1 * torch.sum(v * u_1) ** 2


def relu(v):
    v = v.squeeze()
    norm = torch.linalg.norm(v)
    return torch.where(norm < 1., 0., norm - 1.)


def wavy_plateau(v):
    v = v.squeeze()
    return torch.sum(torch.sin(v))


def friedman(v):
    v = v.squeeze()
    return 10 * torch.sin(math.pi * v[0] * v[1]) + 20 * (v[2] - 0.5) ** 2 + 10 * v[3] + 5 * v[4]


def watershed(v):
    v = v.squeeze()
    return torch.sum(v ** 2) - 0.1 * torch.sum(torch.cos(5 * math.pi * v))


def ishigami(v):
    v = v.squeeze()
    return torch.sin(v[0]) + 7 * torch.sin(v[1]) ** 2 + 0.1 * v[2] ** 4 * torch.sin(v[0])


def flowers(v):
    v = v.squeeze()
    return torch.linalg.norm(v) + torch.sin(4 * torch.atan2(v[1], v[0]))


def bohachevsky(v):
    v = v.squeeze()
    return v[0] ** 2 + 2 * v[1] ** 2 - 0.3 * torch.cos(3 * math.pi * v[0]) - 0.4 * torch.cos(4 * math.pi * v[1]) + 0.7


def holder_table(v):
    v = v.squeeze()
    return -torch.abs(torch.sin(v[0]) * torch.cos(v[1]) * torch.exp(torch.abs(1 - torch.linalg.norm(v) / math.pi)))


def zigzag_ridge(v):
    v = v.squeeze()
    return torch.sum(v ** 2) - 0.1 * torch.sum(torch.cos(10 * math.pi * v))


def oakley_ohagan(v):
    v = v.squeeze()
    return torch.sum(v) + torch.sum(v ** 2) + torch.sum(v[1:] * v[:-1])


def sphere(v):
    v = v.squeeze()
    return torch.sum(v ** 2)


def poly(v):
    v = v.squeeze()
    return 0.5 * torch.sum(v ** 2)


def styblinski_tang(v):
    v = v.squeeze()
    return 0.5 * torch.sum(v ** 4 - 16 * v ** 2 + 5 * v)


def flat_disk_moat(v):
    v = v.squeeze()
    r = torch.linalg.norm(v)
    return torch.where(r < 1., 0., (r - 1.) ** 2)


def flat(v):
    return torch.tensor(0., device=v.device, dtype=v.dtype)


potentials_all = {
    'double_exp': double_exp,
    'rotational': rotational,
    'relu': relu,
    'flat': flat,
    'wavy_plateau': wavy_plateau,
    'friedman': friedman,
    'watershed': watershed,
    'ishigami': ishigami,
    'flowers': flowers,
    'bohachevsky': bohachevsky,
    'holder_table': holder_table,
    'zigzag_ridge': zigzag_ridge,
    'oakley_ohagan': oakley_ohagan,
    'sphere': sphere,
    'poly': poly,
    'styblinski_tang': styblinski_tang,
    'flat_disk_moat': flat_disk_moat
}


# Bridge potentials to PyTorch Gradients directly
def get_potential_grad_as_torch(pot_name):
    """
    Returns a PyTorch function that computes \nabla \Psi(x) using torch.autograd.
    """
    if pot_name not in potentials_all:
        raise ValueError(f"Unknown potential: {pot_name}. Available options are: {list(potentials_all.keys())}")

    torch_pot = potentials_all[pot_name]

    def grad_func_torch(x_torch, t=0.0):
        # x_torch shape is expected to be (N, d)
        # We need to compute the gradient for each point in the batch.
        # Since the potentials are written to process a single point v,
        # and autograd computes the sum of gradients if we sum the output,
        # we can just apply the potential to each row and sum the results.

        # Ensure requires_grad is True so autograd can track it
        x_in = x_torch.detach().requires_grad_(True)

        # Compute potential for the entire batch.
        # Because our custom potentials use operations like v.squeeze() and v[0],
        # we compute them point-by-point via vmap to handle batched inputs cleanly.
        pot_vals = torch.vmap(torch_pot)(x_in)

        # Summing the potentials allows us to get the gradient for each independent input point
        loss = pot_vals.sum()

        # Compute gradient
        grad = torch.autograd.grad(loss, x_in, create_graph=False)[0]

        # The gradient is pointing towards increasing potential.
        # Usually, gradient flow goes down the potential landscape, so we return the gradient.
        # (If your system expects the negative gradient, you might need to return -grad)
        return grad

    return grad_func_torch