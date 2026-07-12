# pylint: disable=no-value-for-parameter,missing-module-docstring,missing-function-docstring
import argparse
from typing import Tuple
import math
import tqdm
import yaml
from easydict import EasyDict
import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor

from src import groups


class SO(groups.ConnectedMatrixLieGroup):  # pylint: disable=abstract-method
    """Special orthogonal group SO(n)"""
    def __init__(self, n: int):
        assert n >= 2, "n must be at least 2"
        basis = torch.zeros(n * (n - 1) // 2, n, n)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                basis[idx, i, j] = 1.
                basis[idx, j, i] = -1.
                idx += 1
        super().__init__(basis)

    @property
    def adjoint_trace(self) -> Tensor:
        return torch.zeros(self.num_generators, device=self.device)

    @property
    def scales(self) -> Tensor:
        return torch.ones(self.num_generators, device=self.device)


class SOEnergy(nn.Module):
    """Energy function for SO(n) group."""
    def __init__(self, group: SO):
        super().__init__()
        self.group = group

    def forward(self, g: Tensor) -> Tensor:
        """
        Energy function used in Section 6 of https://arxiv.org/abs/2403.12012
        U(X) = -10 X_11 ** 2
        Args:
            g: [bsize, n, n] group element
        Return:
            energy: [bsize,]
        """
        return -10. * g[:, 0, 0] ** 2


class GroupSampler(nn.Module):
    """Base class for sampling from a group using an energy function."""
    def __init__(self, energy: SOEnergy, temperature: float):
        super().__init__()
        self.group = energy.group
        self.energy = energy
        self.temperature = temperature

    def forward(self, bsize: int) -> Tensor:
        """
        Sample from the group using the energy function.
        Return:
            samples: [bsize, steps, n, n] group elements
        """
        raise NotImplementedError


class KineticLangevinSampler(GroupSampler):
    """Kinetic Langevin sampler for sampling from a group using an energy function."""
    def __init__(
        self,
        energy: SOEnergy,
        temperature: float,
        step_size: float,
        steps: int,
        friction: float,
        init_scale: float
    ):
        super().__init__(energy, temperature)
        self.step_size = step_size
        self.steps = steps
        self.friction = friction
        self.init_scale = init_scale
        self.dtype = torch.float64

        self.momentum_scale = math.exp(-friction * step_size)
        self.grad_scale = (1 - math.exp(-friction * step_size)) / friction
        self.noise_scale = math.sqrt(1 - math.exp(-2 * friction * step_size))

    @torch.enable_grad()
    @torch.inference_mode(False)
    def trivialized_grad(self, g: Tensor) -> Tensor:
        """
        Compute trivialized gradient
        Args:
            g: [bsize, n, n] group element
        Return:
            grad: [bsize, num_generators] trivialized gradient
        """
        bsize = g.shape[0]

        # prepare trivialized gradient computation
        zeros = torch.zeros(bsize, self.group.num_generators, device=g.device, dtype=self.dtype)
        zeros.requires_grad_(True)
        id_ = self.group.exp(zeros)
        g_ = self.group.compose(g.clone(), id_.clone())

        # compute energy
        energy = self.energy(g_.float()) / self.temperature

        # compute trivialized gradient
        grad, = torch.autograd.grad(energy.sum(), zeros, create_graph=False)
        return grad

    def forward(self, bsize: int) -> Tensor:
        """
        Algorithm 1 of https://proceedings.mlr.press/v247/kong24a/kong24a.pdf
        """
        # initialize group elements and momenta
        g = self.group.exp(torch.randn(bsize, self.group.num_generators, device=self.group.device, dtype=self.dtype))
        m = torch.zeros(bsize, self.group.num_generators, device=self.group.device, dtype=self.dtype)

        # prepare trajectory storage
        g_traj = torch.empty(bsize, self.steps, *g.shape[1:], device=g.device, dtype=g.dtype)

        for i in tqdm.tqdm(range(self.steps), bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'):
            # compute group update
            dg = self.group.exp(self.step_size * m)

            # compute trivialized gradient
            grad = self.trivialized_grad(g)

            # update momenta
            m = self.momentum_scale * m \
                - self.grad_scale * grad \
                + self.noise_scale * torch.randn_like(m)

            # update group elements
            g = self.group.compose(g, dg)

            # store trajectory
            g_traj[:, i] = g

        return g_traj


class DiffusionSampler(GroupSampler):
    """Diffusion-based sampler for sampling from a group using an energy function."""
    timesteps: Tensor
    gamma: Tensor

    def __init__(
        self,
        energy: SOEnergy,
        temperature: float,
        steps: int,
        noise_min: float,
        noise_max: float,
        num_mc: int
    ):
        super().__init__(energy, temperature)
        self.steps = steps
        self.step_size = 1 / steps
        self.num_mc = num_mc
        self.dtype = torch.float64

        # need correction?
        self.adjoint_trace_is_zero = (self.group.adjoint_trace == 0.).all()

        # diffusion parameters
        # gamma: noise_min at t = 0, noise_max at t = 1
        self.register_buffer('timesteps', torch.linspace(0, 1, steps))
        self.register_buffer('gamma', noise_min * ((noise_max / noise_min) ** self.timesteps))
        self.drift_scale = self.step_size
        self.diffusion_scale = self.step_size ** 0.5

    @staticmethod
    def _repeat_and_flatten(x: Tensor, repeats: int) -> Tensor:
        bsize = x.shape[0]
        if repeats == 1:
            return x.clone()
        return x[:, None].expand(bsize, repeats, *x.shape[1:]).flatten(0, 1)

    @staticmethod
    def _deflatten(x: Tensor, bsize: int, repeats: int) -> Tensor:
        return x.view(bsize, repeats, *x.shape[1:])

    @torch.no_grad()
    def forward_process(self, bsize: int, t: int) -> Tuple[Tensor, Tensor]:
        """
        Forward diffusion process
        Args:
            bsize: number of samples
            t: int in {1, ..., steps}
        Return:
            w: [bsize, ...]
            log_modular_w: [bsize,]
        """
        # steps in Lie algebra
        dw_coeffs = torch.randn(bsize * t, self.group.num_generators, device=self.group.device, dtype=self.dtype)
        dw_coeffs = self._deflatten(dw_coeffs, bsize, t)
        dw_coeffs *= self.gamma[None, :t].view(1, t, *([1] * (dw_coeffs.ndim - 2)))
        dw_coeffs *= self.diffusion_scale

        # steps in group
        dws = self.group.exp(dw_coeffs.flatten(0, 1))
        dws = self._deflatten(dws, bsize, t)

        # compose steps
        w = dws[:, 0]
        for step in range(1, t):
            w = self.group.compose(w, dws[:, step])

        # log modular function
        if self.adjoint_trace_is_zero:
            log_modular_w = torch.zeros(bsize, device=self.group.device)
        else:
            log_modular_w = torch.einsum(
                'bti,i->bt', -dw_coeffs.float(), self.group.adjoint_trace).cumsum(1)[:, -1]

        return w, log_modular_w

    @torch.enable_grad()
    @torch.inference_mode(False)
    def estimate_score(self, g: Tensor, t: int) -> Tensor:
        """
        Estimate trivialized score at given timesteps
        """
        bsize = g.shape[0]

        # forward process
        w, log_modular_w = self.forward_process(self.num_mc, t)

        # prepare trivialized gradient computation
        zeros = torch.zeros(bsize, self.group.num_generators, device=g.device)
        zeros.requires_grad_(True)
        id_ = self.group.exp(zeros)

        # [bsize * num_mc, ...]
        w_inv = self.group.inverse(w)
        w_inv = w_inv[None].expand(bsize, self.num_mc, *w_inv.shape[1:]).flatten(0, 1)
        id_ = self._repeat_and_flatten(id_, self.num_mc)
        g_ = self._repeat_and_flatten(g, self.num_mc)

        # evaluation points
        g_ = self.group.compose(g_, id_)
        g_w_inv = self.group.compose(g_, w_inv.float())

        # compute energy
        energy = self.energy(g_w_inv).view(bsize, self.num_mc) / self.temperature

        # logits
        logits = - energy - log_modular_w[None].expand(bsize, self.num_mc)

        # estimate scores
        logsumexp = torch.logsumexp(logits, dim=1)

        score, = torch.autograd.grad(logsumexp.sum(), zeros, create_graph=False, retain_graph=True)
        return score

    def init(self, bsize: int) -> Tensor:
        """
        Initialize group elements using forward process at the last timestep
        """
        w, _ = self.forward_process(bsize, self.steps)
        return w

    def forward(self, bsize: int) -> Tensor:
        """
        Reverse diffusion
        """
        # initialize group elements using forward process
        g = self.init(bsize)

        # prepare trajectory storage
        g_traj = torch.empty(bsize, self.steps, *g.shape[1:], device=g.device, dtype=g.dtype)

        for i in tqdm.tqdm(range(self.steps), bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'):
            t = self.steps - i

            score = self.estimate_score(g.float(), t)
            noise = torch.randn(bsize, self.group.num_generators, device=self.group.device, dtype=self.dtype)

            step = self.gamma[t - 1].pow(2) * self.drift_scale * score \
                + self.gamma[t - 1] * self.diffusion_scale * noise

            dg = self.group.exp(step.to(self.dtype))
            g = self.group.compose(g, dg)

            # store trajectory
            g_traj[:, i] = g

        return g_traj


def main(config):
    device = torch.device("cuda:0")

    # setup group
    group = SO(config.n).to(device)

    # setup energy
    energy = SOEnergy(group).to(device)

    # setup sampler
    if config.sampler == 'langevin':
        sampler = KineticLangevinSampler(
            energy,
            temperature=config.temperature,
            step_size=config.step_size,
            steps=config.steps,
            friction=config.friction,
            init_scale=config.init_scale
        ).eval().to(device)
    elif config.sampler == 'diffusion':
        sampler = DiffusionSampler(
            energy,
            temperature=config.temperature,
            steps=config.steps,
            noise_min=config.noise_min,
            noise_max=config.noise_max,
            num_mc=config.num_mc
        ).eval().to(device)
    else:
        raise ValueError(f"Unknown sampler: {config.sampler}")

    # sample from the group
    g_traj = sampler(config.bsize)

    if config.sampler == 'langevin':
        # visualize results
        file_id = f'so{config.n}_langevin_stepsize{config.step_size}_friction{config.friction}_steps{config.steps}'

        # plot the histogram of the first element
        g_11 = g_traj[:, :, 0, 0].cpu().numpy().flatten()
        plt.figure(figsize=(8, 6))
        plt.hist(g_11, bins=100, density=True)
        plt.title('Histogram of $g_{11}$')
        plt.xlabel('$g_{11}$')
        plt.ylabel('Density')
        plt.grid()
        plt.savefig(f'_hist_{file_id}.png')
        plt.close()

        # plot the trajectory of the first element
        plt.figure(figsize=(8, 6))
        plt.plot(g_traj[0, :, 0, 0].cpu().numpy())
        plt.title('Trajectory of $g_{11}$')
        plt.xlabel('Step')
        plt.ylabel('$g_{11}$')
        plt.grid()
        plt.savefig(f'_traj_{file_id}.png')
        plt.close()

        # plot the mean of ensemble over time
        g_mean = g_traj[:, :, 0, 0].mean(dim=0).abs().cpu().numpy()
        plt.figure(figsize=(8, 6))
        plt.plot(g_mean)
        plt.title('Mean of $g_{11}$ Over Time')
        plt.xlabel('Step')
        plt.ylabel('$g_{11}$')
        plt.yscale('log')
        plt.ylim(1e-3, 1)
        plt.grid()
        plt.savefig(f'_mean_{file_id}.png')
        plt.close()

    elif config.sampler == 'diffusion':
        # visualize results
        file_id = f'so{config.n}_diffusion_steps{config.steps}_noise_[{config.noise_min},{config.noise_max}]_mc{config.num_mc}'

        # plot the histogram of the first element
        g_11 = g_traj[:, -1, 0, 0].cpu().numpy().flatten()
        plt.figure(figsize=(8, 6))
        plt.hist(g_11, bins=100, density=True)
        plt.title('Histogram of $g_{11}$')
        plt.xlabel('$g_{11}$')
        plt.ylabel('Density')
        plt.grid()
        plt.savefig(f'_hist_{file_id}.png')
        plt.close()

        # plot the trajectory of the first element
        plt.figure(figsize=(8, 6))
        plt.plot(g_traj[0, :, 0, 0].cpu().numpy())
        plt.title('Trajectory of $g_{11}$')
        plt.xlabel('Step')
        plt.ylabel('$g_{11}$')
        plt.grid()
        plt.savefig(f'_traj_{file_id}.png')
        plt.close()

        # plot the mean of ensemble over time
        g_mean = g_traj[:, :, 0, 0].mean(dim=0).abs().cpu().numpy()
        plt.figure(figsize=(8, 6))
        plt.plot(g_mean)
        plt.title('Mean of $g_{11}$ Over Time')
        plt.xlabel('Step')
        plt.ylabel('$g_{11}$')
        plt.yscale('log')
        plt.ylim(1e-3, 1)
        plt.grid()
        plt.savefig(f'_mean_{file_id}.png')
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    with open(args.config, 'r', encoding="utf-8") as f:
        config_ = EasyDict(yaml.safe_load(f))

    main(config_)
