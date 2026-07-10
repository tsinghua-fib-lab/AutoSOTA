# edm.py
from __future__ import annotations

import math
import pickle
import inspect
from typing import Any, Dict, List, Optional, Union

import torch

from methods import register_baseline_model
from methods.base import BaseDenoiser
from tqdm import tqdm


@register_baseline_model("baseline_edm_unconditional")
class EDM_Unconditional(BaseDenoiser):
    """
    EDM sampler wrapper (Algorithm 2 from "Elucidating the Design Space...").

    - Follows the same project structure as BaselineUNet(BaseDenoiser).
    - Overrides _sample_batch() to perform EDM sampling (rho discretization + churn + Euler + 2nd-order correction).
    - denoise(): calls the underlying EDM model and returns pred_x0.

    Expected model behavior:
      model(x, sigma, *optional conditioning*) -> pred_x0   (preferred)
    If your model returns epsilon or v, convert it to pred_x0 inside denoise().
    """

    prediction_type: str = "sample"  # keep consistent with BaseDenoiser

    def __init__(
        self,
        resolution: int,
        device: str,
        num_steps: int,
        model_path: str,
        dataset_name: str = "cifar10",
        in_channels: int = 3,
        **kwargs: Any,
    ) -> None:
        # Base fields (device, channels, resolution, etc.)
        super().__init__(
            resolution=resolution,
            device=device,
            num_steps=num_steps,
            dataset_name=dataset_name,
            in_channels=in_channels,
            **kwargs,
        )

        # ---- EDM sampling hyperparams (defaults aligned with NVIDIA reference) ----
        self.sigma_min: float = float(kwargs.get("sigma_min", 0.002))
        self.sigma_max: float = float(kwargs.get("sigma_max", 80.0))
        self.rho: float = float(kwargs.get("rho", 7.0))

        self.S_churn: float = float(kwargs.get("S_churn", 0.0))
        self.S_min: float = float(kwargs.get("S_min", 0.0))
        self.S_max: float = float(kwargs.get("S_max", float("inf")))
        self.S_noise: float = float(kwargs.get("S_noise", 1.0))

        # Integration dtype
        self.use_fp64: bool = bool(kwargs.get("use_fp64", True))

        # Optional: class conditioning can be set externally if needed
        self.class_labels = kwargs.get("class_labels", None)

        # ---- Load EDM network ----
        self.model_path = model_path
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        # If the loaded model provides sigma_min/sigma_max/round_sigma, use them
        self._net_sigma_min = getattr(self.model, "sigma_min", None)
        self._net_sigma_max = getattr(self.model, "sigma_max", None)
        self._net_round_sigma = getattr(self.model, "round_sigma", None)

    # -------------------------
    # Model loading
    # -------------------------

    def _load_model(self, path: str) -> torch.nn.Module:
        """
        Supports:
        1) NVIDIA EDM pickle: pickle.load(...)-> dict with key 'ema'
        2) A raw torch module saved via torch.save(model) (less common)
        3) A state_dict checkpoint (dict of tensors) -> requires your code to build the model elsewhere
        """
        try:
            # Try EDM-style pickle
            with open(path, "rb") as f:
                obj = pickle.load(f)

            # NVIDIA EDM format: {'ema': net, ...}
            if isinstance(obj, dict) and "ema" in obj:
                return obj["ema"]

            # Sometimes directly saved module
            if isinstance(obj, torch.nn.Module):
                return obj

            # If it's a dict but not 'ema', we can't safely build the net here
            raise ValueError(
                "Unsupported checkpoint format: dict without key 'ema'. "
                "If this is a state_dict, please provide the EDM network constructor."
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load EDM model from {path}: {e}")

    # -------------------------
    # EDM utilities
    # -------------------------

    def round_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        """Use model.round_sigma if available; otherwise identity."""
        if callable(self._net_round_sigma):
            return self._net_round_sigma(sigma)
        return sigma

    def _clamp_sigmas(self) -> tuple[float, float]:
        """Clamp sigma_min/max to what the model supports (if exposed)."""
        sigma_min = self.sigma_min
        sigma_max = self.sigma_max
        if self._net_sigma_min is not None:
            sigma_min = max(sigma_min, float(self._net_sigma_min))
        if self._net_sigma_max is not None:
            sigma_max = min(sigma_max, float(self._net_sigma_max))
        if not (sigma_min > 0 and sigma_max > 0 and sigma_min < sigma_max):
            raise ValueError(f"Invalid sigma range: sigma_min={sigma_min}, sigma_max={sigma_max}")
        return sigma_min, sigma_max

    def _build_t_steps(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """EDM rho discretization + append 0."""
        N = int(self.num_steps)
        sigma_min, sigma_max = self._clamp_sigmas()

        if N <= 0:
            raise ValueError("num_steps must be positive")
        if N == 1:
            t = torch.tensor([sigma_max], device=device, dtype=dtype)
        else:
            i = torch.arange(N, device=device, dtype=dtype)
            inv_rho = 1.0 / float(self.rho)
            t0 = sigma_max ** inv_rho
            t1 = sigma_min ** inv_rho
            t = (t0 + i / (N - 1) * (t1 - t0)) ** float(self.rho)

        t = self.round_sigma(t)
        t = torch.cat([t, torch.zeros_like(t[:1])])  # t_N = 0
        return t

    def prepare_latents(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Override BaseDenoiser.prepare_latents:
        EDM starts from N(0,1), initial scaling by sigma_max is done in _sample_batch.
        """
        shape = (batch_size, self.n_channels, self.resolution, self.resolution)
        return torch.randn(shape, generator=generator, device=self.device)

    def _randn_like(self, x: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
        if generator is None:
            return torch.randn_like(x)
        return torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)

    # -------------------------
    # Denoise
    # -------------------------

    def denoise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        *,
        generator: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Here `timestep` is continuous sigma (shape [] or [B]).
        We call the underlying model and return pred_x0.

        This function is written to be tolerant to different forward signatures:
        - model(x, sigma)
        - model(x, sigma, class_labels)
        - model(x, sigma, **kwargs)
        """
        x = latents.to(self.device)

        # make sigma shape compatible
        B = x.shape[0]
        if isinstance(timestep, (int, float)):
            sigma = torch.full((B,), float(timestep), device=self.device, dtype=torch.float32)
        elif timestep.ndim == 0:
            sigma = timestep.expand(B).to(self.device).float()
        else:
            sigma = timestep.to(self.device).float()

        # Try to call with (x, sigma, class_labels) if possible
        cls = self.class_labels
        try:
            # inspect whether forward likely accepts 3 args (excluding self)
            sig = inspect.signature(self.model.forward)
            n_params = len(sig.parameters)
            if cls is not None and n_params >= 3:
                out = self.model(x, sigma, cls)
            else:
                out = self.model(x, sigma)
        except TypeError:
            # fallback: try with kwargs
            out = self.model(x, sigma, **kwargs)

        # Assume out is pred_x0 (EDM networks usually output denoised sample)
        pred_x0 = out
        return pred_x0

    # -------------------------
    # EDM sampling loop
    # -------------------------

    @torch.no_grad()
    def _sample_batch(
        self,
        *,
        batch_size: int,
        generator: Optional[torch.Generator],
        return_intermediates: bool,
    ):
        latents = self.prepare_latents(batch_size, generator=generator)

        integ_dtype = torch.float64 if self.use_fp64 else torch.float32
        t_steps = self._build_t_steps(device=latents.device, dtype=integ_dtype)

        # x = z * t0
        x_next = latents.to(integ_dtype) * t_steps[0]

        trajectory_xt: Optional[List[torch.Tensor]] = [] if return_intermediates else None
        trajectory_x0: Optional[List[torch.Tensor]] = [] if return_intermediates else None
        timesteps_list: Optional[List[int]] = [] if return_intermediates else None

        N = int(self.num_steps)

        pbar = tqdm(range(N), total=N, desc="EDM Sampling", unit="step")
        last_x0: Optional[torch.Tensor] = None

        for i in pbar:
            t_cur = t_steps[i]
            t_next = t_steps[i + 1]
            x_cur = x_next

            # churn
            t_cur_val = float(t_cur.item())
            gamma = 0.0
            if self.S_min <= t_cur_val <= self.S_max:
                gamma = min(self.S_churn / max(N, 1), math.sqrt(2.0) - 1.0)

            t_hat = self.round_sigma(t_cur + gamma * t_cur)

            add_std = torch.sqrt(torch.clamp(t_hat * t_hat - t_cur * t_cur, min=0.0))
            x_hat = x_cur + add_std * float(self.S_noise) * self._randn_like(x_cur, generator)

            # Euler
            pred_x0 = self.denoise(
                x_hat.to(torch.float32),
                t_hat.to(torch.float32),
                generator=generator,
            )
            if not isinstance(pred_x0, torch.Tensor):
                raise TypeError("denoise() must return torch.Tensor")

            denoised = pred_x0.to(integ_dtype)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # 2nd order correction (Heun-like)
            if i < N - 1:
                pred_x0_2 = self.denoise(
                    x_next.to(torch.float32),
                    t_next.to(torch.float32),
                    generator=generator,
                )
                denoised_2 = pred_x0_2.to(integ_dtype)
                d_prime = (x_next - denoised_2) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
                last_x0 = denoised_2
            else:
                last_x0 = denoised

            if return_intermediates:
                if trajectory_xt is not None:
                    trajectory_xt.append(x_next.detach().cpu())
                if trajectory_x0 is not None:
                    trajectory_x0.append(last_x0.detach().cpu())
                if timesteps_list is not None:
                    timesteps_list.append(i)  # EDM records by step index (not sigma)

        images = x_next.to(torch.float32).detach().cpu()

        return self.build_sample_output(
            images=images,
            trajectory_xt=trajectory_xt if return_intermediates else None,
            trajectory_x0=trajectory_x0 if return_intermediates else None,
            timesteps=timesteps_list if return_intermediates else None,
        )

    def train(self, dataset: Any) -> None:
        # Pre-trained model, nothing to do.
        pass
