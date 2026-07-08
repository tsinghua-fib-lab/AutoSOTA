import torch
from tqdm import tqdm
from .base import Algo

import wandb

# ----------------------------------------------------------------------------------------
# Paper: Ensemble kalman diffusion guidance: A derivative-free method for inverse problems
# Official implementation: https://github.com/devzhk/enkg-pytorch
# ----------------------------------------------------------------------------------------

class EnKG(Algo):
    def __init__(self, 
                 net, 
                 forward_op,
                 guidance_scale, 
                 num_steps, 
                 num_updates, 
                 sigma_max,
                 sigma_min,
                 num_samples=1024,
                 threshold=0.1,
                 batch_size=128,
                 lr_min_ratio=0.0,
                 rho: int=7, 
                 factor: int=4):
        super(EnKG, self).__init__(net, forward_op)
        self.rho = rho
        self.num_steps = num_steps
        self.num_updates = num_updates
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.batch_size = batch_size
        self.guidance_scale = guidance_scale
        self.threshold = threshold
        self.num_samples = num_samples
        self.lr_min_ratio = lr_min_ratio
        self.factor = factor

    @torch.no_grad()
    def inference(self, observation, num_samples=1):
        device = self.forward_op.device
        x_initial = torch.randn(self.num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device) * self.sigma_max
        # Time step discretization.
        step_indices = torch.arange(self.num_steps, dtype=torch.float32, device=device)

        t_steps = (
            self.sigma_max ** (1 / self.rho)
            + step_indices
            / (self.num_steps - 1)
            * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        t_steps = torch.cat(
            [self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        )  # t_N = 0

        num_batches = x_initial.shape[0] // self.batch_size
            # Main sampling loop.
        x_next = x_initial
        denoised = torch.zeros_like(x_initial)

        for i, (t_cur, t_next) in tqdm(
            enumerate(zip(t_steps[:-1], t_steps[1:]))
        ):  # 0, ..., N-1
            x_cur = x_next

            # # Update the ensemble particles
            if i < (self.num_steps - int(0.5 * self.threshold)) and i > self.threshold:
                x_hat, t_hat = self.update_particles(
                    x_cur,
                    observation,
                    num_steps=min(1 + (self.num_steps - i) // self.factor, 20),
                    sigma_start=t_cur,
                    guidance_scale=self.get_lr(i),
                )
            else:
                t_hat = t_cur
                x_hat = x_cur

            # batched netwrok forward
            for j in range(num_batches):
                start = j * self.batch_size
                end = (j + 1) * self.batch_size
                denoised[start:end] = self.net(x_hat[start:end], t_hat)

            # Euler step
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
        return x_next
    
    def get_lr(self, i):
        if self.lr_min_ratio > 0.0:
            return self.guidance_scale * (1 - self.lr_min_ratio) * (self.num_steps - i) / self.num_steps + self.lr_min_ratio
        else:
            return self.guidance_scale
        
    @torch.no_grad()
    def update_particles(self, particles, observation, num_steps, sigma_start, guidance_scale=1.0):
        x0s = torch.zeros_like(particles)  # (N, C, H, W), x0 of each particle
        num_batchs = particles.shape[0] // self.batch_size  # number of batches
        N, *spatial = particles.shape
        t_hat = sigma_start

        for j in range(self.num_updates):
            # get x0 for each particle
            for i in range(num_batchs):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                x0s[start:end] = ode_sampler(
                    self.net,
                    particles[start:end],
                    num_steps=num_steps,
                    sigma_start=sigma_start,
                )
            # get measurement for each particle
            ys = self.forward_op.forward(x0s)

            # difference from the mean
            xs_diff = particles - particles.mean(dim=0, keepdim=True)
            ys_diff = ys - ys.mean(dim=0, keepdim=True)
            ys_err = 0.5 * self.forward_op.gradient_m(ys, observation)
            # ys_err = ys - observation

            coef = (
                torch.matmul(
                    ys_err.reshape(ys_err.shape[0], -1),
                    ys_diff.reshape(ys_diff.shape[0], -1).T,
                )
                / particles.shape[0]
            )
            dxs = coef @ xs_diff.reshape(N, -1)  # (N, C*H*W)
            lr = guidance_scale / torch.linalg.matrix_norm(coef)
            particles = particles - lr * dxs.reshape(N, *spatial)

            if wandb.run is not None:
                abs_ys = torch.abs(ys_err)
                abs_err = torch.mean(abs_ys)
                max_err = torch.max(abs_ys)
                # coefficient for updating particles
                std = torch.std(particles, dim=0, keepdim=True)
                avg_std = torch.mean(std)
                wandb.log(
                    {
                        "EnKG/abs error": abs_err.item(),
                        'EnKG/max error': max_err.item(),
                        "EnKG/averaged norm of updates": torch.mean(
                            torch.linalg.vector_norm(dxs, dim=1)
                        ).item(),
                        "EnKG/lr": lr,
                        "EnKG/std": avg_std.item(),
                    }
                )
        return particles, t_hat


# ----------- deterministic sampler ------------#
# Generate x_0 from x_t for any t.


@torch.no_grad()
def ode_sampler(
    net,
    x_initial,
    num_steps=18,
    sigma_start=80.0,
    sigma_eps=0.002,
    rho=7,
):
    if num_steps == 1:
        denoised = net(x_initial, sigma_start)
        return denoised
    last_sigma = sigma_eps
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=x_initial.device)

    t_steps = (
        sigma_start ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (last_sigma ** (1 / rho) - sigma_start ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0

    # Main sampling loop.
    x_next = x_initial
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        t_hat = t_cur
        x_hat = x_cur

        # Euler step.
        denoised = net(x_hat, t_hat)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

    return x_next

