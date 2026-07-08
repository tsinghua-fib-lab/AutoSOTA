import os
import pickle
import sys

import torch

from calibrated_guidance.diffusion_posterior.base import DiffusionPosterior, SampleOutput


def load_meanflow_net(ckpt_path, device, easy_meanflow_root=None):
    """
    Load a MeanFlow network. Accepts either:
      - .pkl network snapshot (recommended; uses 'ema').
      - .pt training state (uses 'net' — live, non-EMA weights).
    The persistence-pickled module class is reconstructed from source bytes
    in the pickle, so `training.networks_mf` does not need to be importable.
    `dnnlib` and `torch_utils` (already present at the repo root) must be.
    """
    if easy_meanflow_root is not None and easy_meanflow_root not in sys.path:
        sys.path.insert(0, easy_meanflow_root)

    if ckpt_path.endswith('.pkl'):
        with open(ckpt_path, 'rb') as f:
            obj = pickle.load(f)
        net = obj.get('ema', obj.get('net'))
    else:
        obj = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        net = obj.get('ema', obj.get('net'))

    return net.to(device).eval()


class MeanFlowDiffusionPosterior(DiffusionPosterior):
    """
    Single-step MeanFlow replacement for the 25-step inner EDM-SDE loop in
    `algo.reinforce.InverseBenchModel`. Same `sample(xt, t, K)` contract:
    returns `[1, K, C, H, W]` candidate x0 estimates.

    Coordinate conversion (VP-EDM outer -> rectified-flow MF):
        sigma   = scheduler.sigma_steps[start_step]
        scaling = scheduler.scaling_steps[start_step]
        tau     = sigma / (1 + sigma)
        z_tau   = xt / (scaling * (1 + sigma))   # ~= (1-tau) y + tau n

    Sibling re-noising for diverse K candidates (mirrors approxposterior's
    PMFOneShotSiblingReinforceDiffusionPosterior):
        1. anchor x_hat0 = z_tau - (tau - sigma_min) * mf_net(z_tau, tau, h=tau-sigma_min)
        2. broadcast anchor to K replicas, draw K i.i.d. fresh noises n_i,
           rebuild z_i = (1 - renoise_tau) * anchor + renoise_tau * n_i
        3. K-batched MF jump: x0_i = z_i - (renoise_tau - sigma_min) * mf_net(z_i, renoise_tau, h=...)
    Default `sibling_renoise_t = None` -> renoise to the same `tau` as the outer step.
    """

    def __init__(self, scheduler, mf_net, output_path,
                 sigma_min=0.002, sibling_renoise_t=None, enable_grad=False):
        self.scheduler = scheduler
        self.mf_net = mf_net
        self.output_path = output_path
        self.sigma_min = sigma_min
        self.sibling_renoise_t = sibling_renoise_t
        self.enable_grad = enable_grad

    MAX_INNER_BATCH = 256

    def _mf_one_step(self, z, tau_scalar):
        B = z.shape[0]
        h = max(tau_scalar - self.sigma_min, 0.0)
        outs = []
        grad_ctx = torch.enable_grad() if self.enable_grad else torch.no_grad()
        for start in range(0, B, self.MAX_INNER_BATCH):
            end = min(start + self.MAX_INNER_BATCH, B)
            chunk = z[start:end]
            cb = end - start
            tau_t = torch.full((cb, 1, 1, 1), tau_scalar, device=z.device, dtype=z.dtype)
            h_t = torch.full((cb, 1, 1, 1), h, device=z.device, dtype=z.dtype)
            augment_labels = torch.zeros(cb, 9, device=z.device, dtype=z.dtype)
            with grad_ctx:
                v = self.mf_net(chunk, tau_t, class_labels=None,
                                h=h_t, augment_labels=augment_labels)
            outs.append(chunk - h * v)
        return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]

    def mean(self, xt, t):
        start_step = int((1.0 - float(t)) * self.scheduler.num_steps)
        start_step = min(max(start_step, 0), self.scheduler.num_steps - 1)
        sigma = float(self.scheduler.sigma_steps[start_step])
        scaling = float(self.scheduler.scaling_steps[start_step])
        tau = sigma / (1.0 + sigma)
        with torch.no_grad():
            z_tau = xt.detach() / (scaling * (1.0 + sigma))
            tau_t = torch.full((z_tau.shape[0], 1, 1, 1), tau, device=z_tau.device, dtype=z_tau.dtype)
            h = max(tau - self.sigma_min, 0.0)
            h_t = torch.full_like(tau_t, h)
            augment_labels = torch.zeros(z_tau.shape[0], 9, device=z_tau.device, dtype=z_tau.dtype)
            v = self.mf_net(z_tau, tau_t, class_labels=None, h=h_t, augment_labels=augment_labels)
            return z_tau - h * v

    def sample(self, xt, t, num_samples_per_step):
        B, C, H, W = xt.shape
        assert B == 1, "Only batch size 1 is supported in MeanFlowDiffusionPosterior"

        start_step = int((1.0 - t) * self.scheduler.num_steps)
        start_step = min(max(start_step, 0), self.scheduler.num_steps - 1)
        sigma = float(self.scheduler.sigma_steps[start_step])
        scaling = float(self.scheduler.scaling_steps[start_step])

        tau = sigma / (1.0 + sigma)
        z_tau = xt / (scaling * (1.0 + sigma))

        anchor_x0 = self._mf_one_step(z_tau, tau)

        renoise_tau = self.sibling_renoise_t if self.sibling_renoise_t is not None else tau
        renoise_tau = float(max(min(renoise_tau, 1.0), 1e-3))

        K = num_samples_per_step
        expanded_anchor = anchor_x0.expand(K, C, H, W).contiguous()
        fresh_noise = torch.randn_like(expanded_anchor)
        renoised = (1.0 - renoise_tau) * expanded_anchor + renoise_tau * fresh_noise

        siblings_x0 = self._mf_one_step(renoised, renoise_tau)

        result = siblings_x0.reshape(1, K, C, H, W)
        return SampleOutput(result)
