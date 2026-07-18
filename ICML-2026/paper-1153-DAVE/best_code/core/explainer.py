import torch
from torch import Tensor

from pathlib import Path
from typing import List, Any

from core.config import DAVEConfig
from core.utils.detach_mode import (
    detach_gelu,
    detach_attention,
    detach_layer_norm,
    attach_gelu,
    attach_attention,
    attach_layer_norm,   
)


class DAVEExplainer:
    def __init__(
        self, 
        model_cfg_path: Path,
        device: torch.device,
        eps: float = 1e-8
    ):
        self.eps = eps
        self._op_variation_removed: bool = False
        
        cfg = DAVEConfig.load_from_yaml(
            path=model_cfg_path, 
            eps=eps,
        )

        self.model_name = cfg.model_name
        self.model = cfg.model.to(device)
        self.aug = cfg.aug.to(device)
        self.post_proc = cfg.post_proc.to(device)
        self.input_transform = cfg.input_transform

    def explain(
        self,
        x: Tensor,
        y: Tensor,
        num_steps: int,
        post_proc: bool = True,
        seed: int = None,
        detach_layer_range: tuple = None,
    ) -> Tensor:
        """
        Performs DAVE attribution.
        detach_layer_range: (start, end) 0-indexed block range.
        None = all layers. (0, 7) = layers 1-8 only.
        """
        self.model.eval()
        self.aug.train()
        self.remove_operator_variation(layer_range=detach_layer_range)

        t_schedule = self.get_noise_schedule(
            num_steps=num_steps,
        ).to(x.device)

        buffer = []
        for step_idx in range(num_steps):
            t = t_schedule[step_idx]
            step_seed = seed + step_idx if seed is not None else None
            c = self.effective_transform(x=x, y=y, t=t, seed=step_seed)
            c = c.detach() * x.detach()
            buffer.append(c)

        c = self._aggregate_maps(buffer)

        if post_proc:
            c = self.post_proc(c)

        self.restore_operator_variation()
        return c

    def effective_transform(
        self, x: Tensor, y: Tensor, t: Tensor,
        seed: int = None,
    ) -> Tensor:
        """
        Computes Effective Transformation,
        assuming removed operator variation
        via remove_operator_variation().
        """
        assert self._op_variation_removed, (
            "Call remove_operator_variation() first!"
        )

        x = self._clone_input(x)
        z = self.pred_batch(x, y, t, seed)

        # After calling remove_operator_variation(), 
        # grad becomes effective transform;
        w_eff = torch.autograd.grad(
            outputs=z,
            inputs=[x],
            grad_outputs=torch.ones_like(z),
            retain_graph=False,
        )[0]
        return w_eff.detach()

    def pred_batch(
        self, x: Tensor, y: Tensor, t: Tensor,
        seed: int = None,
    ) -> Tensor:
        """
        Predicts image batch with: 
        - spatial augmentations (for equivariant transform) 
        - noise addition (for low-pass filter). 
        """
        self._check_batch_shapes(x, y, t)

        x = self.aug(x, seed=seed)
        x = self.add_noise(x, t, seed=seed)
        z = self.model(x)

        y = y.unsqueeze(-1).long()
        z = z.gather(dim=1, index=y)
        return z

    def get_noise_schedule(self, num_steps: int) -> Tensor:
        schedule_type = getattr(self.aug.cfg, 'noise_schedule', 'linear')
        if schedule_type == 'cosine':
            return self._cosine_noise_schedule(num_steps)
        else:
            return torch.linspace(
                0.0, self.aug.noise_alpha, steps=num_steps,
            )

    def _cosine_noise_schedule(self, num_steps: int) -> Tensor:
        """Cosine-warped noise schedule within [0, noise_alpha].
        More samples at intermediate noise levels where
        attribution information is richest."""
        noise_alpha = self.aug.noise_alpha
        steps = torch.arange(num_steps, dtype=torch.float32)
        # Cosine warping: t in [0, 1] with cosine distribution
        t_linear = steps / max(num_steps - 1, 1)
        t_cosine = 0.5 * (1.0 - torch.cos(torch.pi * t_linear))
        return noise_alpha * t_cosine

    def add_noise(self, x: Tensor, t: Tensor, seed: int = None) -> Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        else:
            torch.seed()
        noise = torch.randn_like(x, device=x.device)
        x = (1.0 - t) * x + torch.sqrt(1.0 - (1.0 - t)**2) * noise
        return x

    def remove_operator_variation(self, layer_range: tuple = None):
        """
        Converts model grad to effective transform.
        layer_range: (start, end) 0-indexed block range to detach.
        None means all layers.
        """
        detach_gelu(self.model, layer_range=layer_range)
        detach_attention(self.model, layer_range=layer_range)
        detach_layer_norm(self.model, layer_range=layer_range)
        self._op_variation_removed = True

    def restore_operator_variation(self):
        """
        Converts model grad back to gradient.
        """
        attach_gelu(self.model)
        attach_attention(self.model)
        attach_layer_norm(self.model)
        self._op_variation_removed = False

    def _aggregate_maps(self, maps: List[Tensor]) -> Tensor:
        x = torch.stack(maps, dim=0)
        weights = self._mad_weights(x)
        x = (x * weights).sum(dim=0)
        den = weights.sum(dim=0).clamp(min=1e-8)
        return x / den

    def _mad_weights(self, x: Tensor) -> Tensor:
        """
        Soft MAD-based weights for robust aggregation.
        Uses Gaussian decay instead of hard thresholding
        to preserve outlier information while down-weighting.
        """
        med = x.median(dim=0).values
        mad = (x - med).abs().median(dim=0).values
        scale = 1.4826 * mad + self.eps
        dev = (x - med).abs() / scale.clamp(min=self.eps)
        weights = torch.exp(-0.5 * (dev / 2.5) ** 2)
        return weights

    def _clone_input(self, x: Tensor) -> Tensor:
        return x.clone().detach().requires_grad_(True)

    def _check_batch_shapes(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
    ):
        assert x.ndim == 4, "Expected batch of image samples!"
        assert t.ndim == 0, "Expected batch of noise levels!"
        assert y.ndim == 1, "Expected batch of sample labels!"
