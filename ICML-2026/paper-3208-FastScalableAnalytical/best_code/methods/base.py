from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import DDIMScheduler
from tqdm import tqdm
import torch.distributed as dist

LOGGER = logging.getLogger(__name__)

def is_main_process() -> bool:
    """Check if the current process is the main distributed process."""
    return (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)


@dataclass
class SamplingOutput:
    images: torch.Tensor
    timesteps: Optional[List[int]]
    trajectory_xt: Optional[List[torch.Tensor]]
    trajectory_x0: Optional[List[torch.Tensor]]



def _validate_and_normalize_shapes(
    x0_batch: torch.Tensor, logits: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Helper to ensure inputs are [B, K, N] and match dimensions.
    Returns normalized (x0, logits) on the correct device.
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected logits shape [B, K, N], got {tuple(logits.shape)}")

    B, K, N = logits.shape
    
    # Normalize x0_batch to [B, K, N]
    if x0_batch.ndim == 2:
        if x0_batch.shape != (K, N):
            raise ValueError(f"Expected x0 2D shape {(K, N)}, got {tuple(x0_batch.shape)}")
        x0_out = x0_batch.unsqueeze(0).expand(B, -1, -1)
    elif x0_batch.ndim == 3:
        if x0_batch.shape != (B, K, N):
            raise ValueError(f"Expected x0 3D shape {(B, K, N)}, got {tuple(x0_batch.shape)}")
        x0_out = x0_batch
    else:
        raise ValueError(f"Expected x0 to be 2D or 3D, got {x0_batch.ndim}D")

    return x0_out, logits

class WeightedStreamingSoftmax:
    """
    Averages per-batch local Softmax weights.
    Final weight = local_softmax / total_batches.
    """

    def __init__(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-8,
        store_weights: bool = False,  # <--- toggle
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.eps = eps
        self.sum_weighted: Optional[torch.Tensor] = None
        self.sum_weights: Optional[torch.Tensor] = None
        
        # storage list
        self.store_weights = store_weights
        self._stored_local_weights = []

    def add(self, x0_batch: torch.Tensor, logits: torch.Tensor) -> None:
        x0, logits = _validate_and_normalize_shapes(x0_batch, logits)
        x0 = x0.to(device=self.device, dtype=self.dtype)
        logits = logits.to(device=self.device, dtype=self.dtype)

        # 1. compute local per-batch softmax
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits_exp = torch.exp(logits - logits_max)
        # weights: [B, K] (or [B, K, 1] for the N=1 simplification)
        weights = logits_exp / (logits_exp.sum(dim=1, keepdim=True) + self.eps)

        # 2. if storage is enabled, save the weights for this batch
        if self.store_weights:
            # only store first-sample weights to save memory: [B, K] -> [K]
            # must move to CPU
            self._stored_local_weights.append(weights[0].detach().cpu())

        # 3. normal accumulation logic
        current_weighted_sum = torch.einsum("bkn,bkn->bn", weights, x0)
        current_weight_sum = weights.sum(dim=1) 

        if self.sum_weighted is None:
            self.sum_weighted = current_weighted_sum
            self.sum_weights = current_weight_sum
        else:
            self.sum_weighted += current_weighted_sum
            self.sum_weights += current_weight_sum

    def get_average(self) -> Optional[torch.Tensor]:
        if self.sum_weighted is None or self.sum_weights is None:
            return None
        return self.sum_weighted / (self.sum_weights + self.eps)

    def get_all_weights(self) -> torch.Tensor:
        """
        Return the final true weights of every sample.
        Returns: Tensor shape [Total_Dataset_Samples]
        """
        if not self._stored_local_weights:
            return torch.tensor([])
        
        # 1. concatenate all per-batch weights
        all_local_weights = torch.cat(self._stored_local_weights, dim=0) # [N_total]
        
        # 2. get the denominator (sum of all weights)
        # sum_weights has shape [B, N]; pick the normaliser for the first sample
        total_normalization = self.sum_weights[0].detach().cpu()
        
        # 3. compute final per-sample contribution: local / global_sum
        return all_local_weights / (total_normalization + self.eps)


class StreamingSoftmax:
    """
    Mathematically exact global Softmax (numerically stable).
    Final weight = exp(logit - global_max) / global_sum_exp.
    """

    def __init__(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-8,
        store_weights: bool = False, # <--- toggle
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.eps = eps
        
        # Running statistics
        self.m: Optional[torch.Tensor] = None # Global Max
        self.s: Optional[torch.Tensor] = None # Global Sum (relative to m)
        self.y: Optional[torch.Tensor] = None # Numerator
        self._batch_shape: Optional[Tuple[int, int]] = None

        # storage list
        self.store_weights = store_weights
        self._stored_logits = []

    @torch.no_grad()
    def add(self, x0_batch: torch.Tensor, logits: torch.Tensor) -> None:
        x0, logits = _validate_and_normalize_shapes(x0_batch, logits)
        x0 = x0.to(device=self.device, dtype=self.dtype)
        logits = logits.to(device=self.device, dtype=self.dtype)

        # 1. if storage is enabled, save raw logits
        if self.store_weights:
            # only store first-sample data: [B, K] -> [K]
            self._stored_logits.append(logits.detach().cpu())

        # ... (standard online-softmax update logic below) ...
        B, _, N = logits.shape
        if self._batch_shape is None: self._batch_shape = (B, N)
        
        block_max = logits.max(dim=1).values

        if self.m is None:
            self.m = block_max
            exp_block = torch.exp(logits - self.m.unsqueeze(1))
            self.s = exp_block.sum(dim=1)
            self.y = (exp_block * x0).sum(dim=1)
            return

        m_prev = self.m
        m_new = torch.maximum(m_prev, block_max)
        scale_prev = torch.exp(m_prev - m_new)
        exp_block = torch.exp(logits - m_new.unsqueeze(1))
        s_block = exp_block.sum(dim=1)
        y_block = (exp_block * x0).sum(dim=1)

        self.m = m_new
        self.s = self.s * scale_prev + s_block
        self.y = self.y * scale_prev + y_block

    def get_average(self) -> Optional[torch.Tensor]:
        if self.y is None or self.s is None:
            return None
        return self.y / (self.s + self.eps)

    def get_all_weights(self) -> torch.Tensor:
        """
        Use the streaming final global max (m) and global sum (s)
        to compute the true softmax weight of every sample.
        """
        if not self._stored_logits:
            return torch.tensor([])
        
        # 1. concatenate all logits
        all_logits = torch.cat(self._stored_logits, dim=0) # [N_total]
        
        # 2. fetch the final global statistics (move to CPU)
        # self.m / self.s have shape [B, N] (one entry per latent dim)
        # simplification: use the first-sample statistics for analysis
        # Note: logits are normally already reduced over N (or N=1); for shape [B, K] the code above is correct.
        final_m = self.m[0].detach().cpu() # Global Max
        final_s = self.s[0].detach().cpu() # Global Sum Exp (relative to final_m)

        # 3. compute final weights
        # Formula: w_i = exp(logit_i - M_final) / S_final
        # broadcasting is used here
        numerator = torch.exp(all_logits - final_m)
        final_weights = numerator / (final_s + self.eps)
        
        return final_weights
    


class BaseDenoiser(torch.nn.Module):
    """Base diffusion interface shared by analytic and learned models."""

    prediction_type: str = "sample"

    def __init__(
        self,
        resolution: int,
        device: str,
        num_steps: int,
        *args,
        beta_1: float = 0.0001,
        beta_T: float = 0.02,
        dataset_name: str = "cifar10",
        scheduler_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.n_channels = kwargs.get("in_channels", 3)
        self.img_resolution = resolution
        self.resolution = resolution
        self.dataset_name = dataset_name
        self.num_steps = num_steps

        scheduler_kwargs = scheduler_kwargs or {}
        self.scheduler = DDIMScheduler(
            beta_start=beta_1,
            beta_end=beta_T,
            beta_schedule="linear",
            prediction_type=self.prediction_type,
            **scheduler_kwargs,
        )
        self.scheduler.set_timesteps(num_steps)

    # ---------------------------------------------------------------------
    # Init helpers (reduce duplicate __init__ code in subclasses)
    # ---------------------------------------------------------------------
    @staticmethod
    def init_kwargs_from_dataset(dataset: Any, device: str, num_steps: int, **kwargs: Any) -> Dict[str, Any]:
        """Generate common super().__init__(...) kwargs from a DatasetBundle-like object."""
        return dict(
            resolution=dataset.resolution,
            device=device,
            num_steps=num_steps,
            in_channels=dataset.in_channels,
            dataset_name=dataset.name,
            **kwargs,
        )

    def _init_shape_helpers(self) -> None:
        self.img_shape = (self.resolution, self.resolution)
        self.spatial_dim = self.resolution * self.resolution
        self.feature_dim = self.n_channels * self.spatial_dim
        self.n_data = 0

    def _init_wiener_path(self, dataset: Any, params: Dict[str, object], *, key: str = "wiener_path") -> None:
        wiener_path = params.get(key, None)
        if wiener_path is None:
            default_root = Path("services/base_models/wiener")
            self.wiener_path = default_root / f"{dataset.name}_{dataset.resolution}"
        else:
            self.wiener_path = Path(str(wiener_path))

    def _init_knn_params(
        self,
        params: Dict[str, object],
        *,
        k_key: str = "k_min",
        m_key: str = "k_max",
        low_key: str = "low_res_dim",
        coarse_key: str = "coarse_chunk_size",
        fine_key: str = "fine_chunk_size",
        default_m: int = 10000,
        default_low: int = 0,
        default_coarse: int = 1024,
        default_fine: int = 256,
    ) -> None:
        # Schedule bounds: at every denoising step the actual k_t lies in
        # [k_min, k_max] (Eq. 6) and m_t in [k_max, N] (Sec. 3.4).
        # Accept both new names (k_min / k_max) and legacy (k_neighbors / m_coarse_candidates).
        k = params.get(k_key, params.get("k_neighbors", None))
        self.k_neighbors: Optional[int] = int(k) if k is not None else None
        self.use_knn_screening: bool = self.k_neighbors is not None

        self.m_coarse_candidates: int = int(
            params.get(m_key, params.get("m_coarse_candidates", default_m))
        )
        self.low_res_dim: int = int(params.get(low_key, default_low))

        self.coarse_chunk_size: int = int(params.get(coarse_key, default_coarse))
        self.fine_chunk_size: int = int(params.get(fine_key, default_fine))

        # Proxy upsample: create d > D proxy space (e.g., 2 for d=2D)
        # proxy_repeat_factor: multiplier for proxy dim (1 = no upsample)
        # proxy_upsample_mode: 'repeat' (tile features) or 'bilinear' (interpolate spatially)
        self.proxy_repeat_factor: int = int(params.get("proxy_repeat_factor", 1))
        self.proxy_upsample_mode: str = str(params.get("proxy_upsample_mode", "repeat"))

        self.n_pre_sample_min: int = 0
        self.n_pre_sample_max: int = 0

        # Ablation flags for schedule control
        self.dynamic_m: bool = bool(params.get("dynamic_m", True))
        self.dynamic_k: bool = bool(params.get("dynamic_k", True))
        self.random_retrieval: bool = bool(params.get("random_retrieval", False))
        # Non-uniform timestep schedule (ALGO-03)
        self.timestep_schedule: str = str(params.get("timestep_schedule", "uniform"))
        # Static override values (default None = use midpoint of dynamic range)
        _static_m = params.get("static_m", None)
        self.static_m: Optional[int] = int(_static_m) if _static_m is not None else None
        _static_k = params.get("static_k", None)
        self.static_k: Optional[int] = int(_static_k) if _static_k is not None else None

    def _init_knn_buffers(self) -> None:
        self.register_buffer("x0_database", torch.empty(0), persistent=False)          # [N, D]
        self.register_buffer("x0_db_norm_sq", torch.empty(0), persistent=False)       # [N]
        self.register_buffer("x0_db_low_res", torch.empty(0), persistent=False)       # [N, D_low]
        self.register_buffer("x0_db_low_res_norm_sq", torch.empty(0), persistent=False)

    # ---------------------------------------------------------------------
    # Basic interface
    # ---------------------------------------------------------------------
    def denoise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        raise NotImplementedError

    def build_sample_output(
        self,
        images: torch.Tensor,
        trajectory_xt: Optional[List[torch.Tensor]],
        trajectory_x0: Optional[List[torch.Tensor]],
        timesteps: Optional[List[int]],
    ) -> SamplingOutput:
        return SamplingOutput(
            images=images,
            trajectory_xt=trajectory_xt,
            trajectory_x0=trajectory_x0,
            timesteps=timesteps,
        )

    def train(self, dataset):  # NOTE: project-specific (overrides nn.Module.train)
        raise NotImplementedError

    def set_timesteps(self, num_steps: int) -> None:
        """Set DDIM timesteps with optional non-uniform schedule (ALGO-03).

        Supported schedules:
        - uniform: standard linear spacing (default)
        - quadratic: t_i = T * (i/N)^2 — more steps at low noise
        - cosine: t_i = T * (1 - cos(pi*i/(2*N))) — smoother allocation
        """
        self.scheduler.set_timesteps(num_steps)
        self.num_steps = num_steps

        schedule = getattr(self, 'timestep_schedule', 'uniform')
        if schedule == 'uniform':
            return  # scheduler already set uniform timesteps

        T = self.scheduler.config.num_train_timesteps
        N = num_steps

        if schedule == 'quadratic':
            # t_i = T * (1 - i/N)^2 — more steps at low noise
            # Spacing decreases as i increases (cleaner steps are denser)
            indices = np.arange(N)
            timesteps_float = (T - 1) * (1.0 - indices / (N - 1)) ** 2
            timesteps = np.round(timesteps_float).astype(np.int64)
            timesteps = np.clip(timesteps, 0, T - 1)
        elif schedule == 'cosine':
            # t_i = T * (1 - cos(pi * (N-1-i) / (2*(N-1))))
            # More steps at low noise (cos is steepest near pi/2)
            indices = np.arange(N)
            timesteps_float = (T - 1) * (1.0 - np.cos(np.pi * (N - 1 - indices) / (2.0 * (N - 1))))
            timesteps = np.round(timesteps_float).astype(np.int64)
            timesteps = np.clip(timesteps, 0, T - 1)
        elif schedule == 'snr':
            # SNR-weighted: allocate steps based on log-SNR
            # This puts more steps where SNR changes rapidly (mid-noise regime)
            log_snr = np.log(
                np.array(self.scheduler.alphas_cumprod) / (1.0 - np.array(self.scheduler.alphas_cumprod) + 1e-8)
            )
            # Find N timesteps that evenly divide the log-SNR range
            log_snr_min, log_snr_max = log_snr[-1], log_snr[0]
            target_log_snr = np.linspace(log_snr_max, log_snr_min, N)
            # Find closest timesteps
            timesteps = []
            for target in target_log_snr:
                idx = np.argmin(np.abs(np.array(log_snr) - target))
                timesteps.append(idx)
            timesteps = np.array(timesteps, dtype=np.int64)
            timesteps = np.clip(timesteps, 0, T - 1)
        else:
            return  # unknown schedule, keep uniform

        self.scheduler.timesteps = torch.from_numpy(timesteps).to(self.device)
        # Reset the step index counter for the scheduler's internal state
        self.scheduler._step_index = None

    # ---------------------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------------------
    def prepare_latents(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        shape = (batch_size, self.n_channels, self.resolution, self.resolution)
        latents = torch.randn(shape, generator=generator, device=self.device)
        return latents * self.scheduler.init_noise_sigma

    def compute_noise_from_x0(
        self,
        x_t: torch.Tensor,
        pred_x0: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        t = int(timestep.item() if isinstance(timestep, torch.Tensor) else timestep)
        alpha_prod = self.scheduler.alphas_cumprod[t].to(x_t.device)
        beta_prod = 1 - alpha_prod
        sqrt_alpha = torch.sqrt(alpha_prod)
        sqrt_beta = torch.sqrt(beta_prod + 1e-8)
        return (x_t - sqrt_alpha * pred_x0) / sqrt_beta

    @torch.no_grad()
    def sample(
        self,
        *,
        num_samples: int,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
        return_intermediates: bool = False,
    ) -> SamplingOutput:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        batches: List[SamplingOutput] = []
        total_generated = 0
        while total_generated < num_samples:
            current_batch = min(batch_size, num_samples - total_generated)
            batch_result = self._sample_batch(
                batch_size=current_batch,
                generator=generator,
                return_intermediates=return_intermediates,
            )
            batches.append(batch_result)
            total_generated += current_batch

        images = torch.cat([b.images for b in batches], dim=0)

        trajectory_xt: Optional[List[torch.Tensor]] = None
        trajectory_x0: Optional[List[torch.Tensor]] = None
        timesteps: Optional[List[int]] = None

        if return_intermediates:
            for batch in batches:
                if batch.trajectory_xt is not None:
                    if trajectory_xt is None:
                        trajectory_xt = [tensor.clone() for tensor in batch.trajectory_xt]
                    else:
                        for idx, tensor in enumerate(batch.trajectory_xt):
                            trajectory_xt[idx] = torch.cat([trajectory_xt[idx], tensor], dim=0)

                if batch.trajectory_x0 is not None:
                    if trajectory_x0 is None:
                        trajectory_x0 = [tensor.clone() for tensor in batch.trajectory_x0]
                    else:
                        for idx, tensor in enumerate(batch.trajectory_x0):
                            trajectory_x0[idx] = torch.cat([trajectory_x0[idx], tensor], dim=0)

                if batch.timesteps is not None and timesteps is None:
                    timesteps = list(batch.timesteps)

        return self.build_sample_output(
            images=images,
            trajectory_xt=trajectory_xt,
            trajectory_x0=trajectory_x0,
            timesteps=timesteps,
        )

    def _sample_batch(
        self,
        *,
        batch_size: int,
        generator: Optional[torch.Generator],
        return_intermediates: bool,
    ) -> SamplingOutput:
        latents = self.prepare_latents(batch_size, generator=generator)

        trajectory_xt: Optional[List[torch.Tensor]] = [] if return_intermediates else None
        trajectory_x0: Optional[List[torch.Tensor]] = [] if return_intermediates else None
        timesteps_list: Optional[List[int]] = [] if return_intermediates else None
        last_pred_x0: Optional[torch.Tensor] = None

        timesteps_iter = tqdm(
            enumerate(self.scheduler.timesteps),
            total=len(self.scheduler.timesteps),
            desc="Sampling",
            unit="step",
        )
        for _, timestep in timesteps_iter:
            pred_x0 = self.denoise(latents, timestep, generator=generator)
            if not isinstance(pred_x0, torch.Tensor):
                raise TypeError("denoise must return a torch.Tensor")

            step_output = self.scheduler.step(
                model_output=pred_x0,
                timestep=timestep,
                sample=latents,
            )
            last_pred_x0 = pred_x0
            latents = step_output.prev_sample

            if return_intermediates:
                if trajectory_xt is not None:
                    trajectory_xt.append(latents.detach().cpu())
                if trajectory_x0 is not None:
                    trajectory_x0.append(pred_x0.detach().cpu())
                if timesteps_list is not None:
                    timestep_value = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
                    timesteps_list.append(timestep_value)

        if last_pred_x0 is None:
            raise RuntimeError("Sampling loop did not execute any timesteps.")

        return SamplingOutput(
            images=last_pred_x0.detach().cpu(),
            trajectory_xt=trajectory_xt if return_intermediates else None,
            trajectory_x0=trajectory_x0 if return_intermediates else None,
            timesteps=timesteps_list if return_intermediates else None,
        )

    # ---------------------------------------------------------------------
    # Common dataset / buffer helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _images_from_batch(batch: Any) -> torch.Tensor:
        if isinstance(batch, (tuple, list)):
            if not batch:
                raise ValueError("Empty batch (tuple/list).")
            return batch[0]
        if isinstance(batch, torch.Tensor):
            return batch
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    @torch.no_grad()
    def _build_flat_database(
        self,
        dataloader,
        *,
        dtype: torch.dtype = torch.float32,
        desc: str = "Build flat DB",
        leave: bool = False,
        to_01: bool = False,
    ) -> torch.Tensor:
        all_flat: list[torch.Tensor] = []
        for batch in tqdm(dataloader, desc=desc, leave=leave):
            imgs = self._images_from_batch(batch).to(self.device, dtype=dtype)
            if to_01:
                imgs = ((imgs + 1.0) / 2.0).clamp(0, 1)
            all_flat.append(imgs.reshape(imgs.shape[0], -1))
        if not all_flat:
            raise RuntimeError("Dataloader produced no batches.")
        return torch.cat(all_flat, dim=0).contiguous()

    def _replace_buffer(self, name: str, tensor: torch.Tensor, *, persistent: bool = False) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"tensor must be a torch.Tensor, got {type(tensor)}")
        if name in self._buffers:
            self._buffers[name] = tensor
        else:
            self.register_buffer(name, tensor, persistent=persistent)

    # ---------------------------------------------------------------------
    # Wiener utilities
    # ---------------------------------------------------------------------
    def _compute_linear_mix(self, g, l, ratio: float = 0.8):
        return ratio * g + (1 - ratio) * l
    
    def _load_or_compute_wiener_svd(
        self,
        global_wiener_path: Path,
        wiener_path: Path,
        *,
        dataset_name: str,
        dataloader,
        resolution: int,
        n_channels: int,
        logger: Optional[logging.Logger] = None,
        ratio: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        log = logger or LOGGER
        try:
            from services.wiener import load_wiener_filter, compute_wiener_filter, save_wiener_filter
        except Exception as e:  # noqa: BLE001
            raise ImportError("services.wiener is not available. Cannot load/compute Wiener filter.") from e
        
        is_imagenet = (dataset_name == "imagenet_1k")

        def load(path: Path):
            U, LA, Vh, mean = load_wiener_filter(path, device=self.device)
            if not is_imagenet or path == global_wiener_path:
                return U, LA, Vh, mean
            Ug, LAg, Vhg, mg = load_wiener_filter(global_wiener_path, device=self.device)
            mix = lambda g, l: self._compute_linear_mix(g, l, ratio)
            return mix(Ug, U), mix(LAg, LA), mix(Vhg, Vh), mix(mg, mean)
    
        try:
            return load(wiener_path)
        except FileNotFoundError:
            log.info("Wiener filter not found at %s. Computing from dataset...", wiener_path)
            S, mean = compute_wiener_filter(
            dataloader=dataloader,
            device=self.device,
            resolution=resolution,
            n_channels=n_channels,
        )
            U, LA, Vh = torch.linalg.svd(S)
            save_wiener_filter(U, LA, Vh, mean, wiener_path)
            log.info("Computed and saved Wiener filter to %s", wiener_path)

            return load(wiener_path) 

    def _register_wiener_buffers(self, U: torch.Tensor, LA: torch.Tensor, Vh: torch.Tensor, mean: torch.Tensor) -> None:
        self._replace_buffer("U", U.to(self.device), persistent=False)
        self._replace_buffer("LA", LA.to(self.device), persistent=False)
        self._replace_buffer("Vh", Vh.to(self.device), persistent=False)
        self._replace_buffer("mean", mean.to(self.device), persistent=False)

    # ---------------------------------------------------------------------
    # Shared hierarchical KNN
    # ---------------------------------------------------------------------
    def _choose_low_res_dim(self, resolution: int, preferred: int) -> int:
        def best_default(res: int) -> int:
            target = max(1, res // 2)
            candidates = [d for d in range(4, target + 1) if res % d == 0]
            if candidates:
                return candidates[-1]
            candidates2 = [d for d in range(1, target + 1) if res % d == 0]
            return candidates2[-1] if candidates2 else res

        if preferred <= 0:
            return best_default(resolution)
        if resolution % preferred == 0:
            return preferred
        divisors = [d for d in range(1, preferred + 1) if resolution % d == 0]
        if divisors:
            return divisors[-1]
        return best_default(resolution)

    def _setup_hierarchical_knn(self) -> None:
        self.use_knn_screening = getattr(self, "k_neighbors", None) is not None
        if not self.use_knn_screening:
            return

        if self.k_neighbors is None:
            raise RuntimeError("use_knn_screening=True but k_neighbors is None.")
        if int(getattr(self, "n_data", 0)) <= 0:
            raise RuntimeError("n_data must be set before _setup_hierarchical_knn().")

        if self.k_neighbors > self.n_data:
            self.k_neighbors = self.n_data

        if self.m_coarse_candidates < self.k_neighbors:
            self.m_coarse_candidates = self.k_neighbors

        if self.m_coarse_candidates > self.n_data:
            self.m_coarse_candidates = self.n_data

        self.n_pre_sample_min = max(self.k_neighbors, self.m_coarse_candidates)
        self.n_pre_sample_max = self.n_data

        if self.proxy_repeat_factor > 1:
            # Upsample mode: no spatial downsampling, proxy is built by repeating features
            self.low_res_dim = self.resolution
            self.downsample_factor = 1
        else:
            self.low_res_dim = self._choose_low_res_dim(self.resolution, int(self.low_res_dim))
            if self.resolution % self.low_res_dim != 0:
                raise ValueError(f"low_res_dim ({self.low_res_dim}) must divide resolution ({self.resolution}).")
            self.downsample_factor = self.resolution // self.low_res_dim

    def _apply_proxy_transform(self, img: torch.Tensor) -> torch.Tensor:
        """Apply downsample or upsample transform for proxy space.
        Input: [N, C, H, W]. Output: [N, d]."""
        import math
        n = img.shape[0]
        if self.downsample_factor > 1:
            img = torch.nn.functional.avg_pool2d(
                img, kernel_size=self.downsample_factor, stride=self.downsample_factor
            )
        if self.proxy_repeat_factor > 1:
            if self.proxy_upsample_mode == "bilinear":
                scale = math.sqrt(self.proxy_repeat_factor)
                img = torch.nn.functional.interpolate(
                    img, scale_factor=scale, mode="bilinear", align_corners=False
                )
            else:
                flat = img.reshape(n, -1)
                return flat.repeat(1, self.proxy_repeat_factor)
        return img.reshape(n, -1)

    def _downsample_db(self, db_flat: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "downsample_factor"):
            raise RuntimeError("downsample_factor not set. Call _setup_hierarchical_knn() first.")
        n_data = db_flat.shape[0]
        H = W = self.resolution
        db_img = db_flat.view(n_data, self.n_channels, H, W)
        return self._apply_proxy_transform(db_img)

    def _downsample_batch(self, x_batch_flat: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "downsample_factor"):
            raise RuntimeError("downsample_factor not set. Call _setup_hierarchical_knn() first.")
        n_samples = x_batch_flat.shape[0]
        H = W = self.resolution
        x_img = x_batch_flat.view(n_samples, self.n_channels, H, W)
        return self._apply_proxy_transform(x_img)

    def _precompute_multiscale_db(self) -> None:
        if getattr(self, "x0_database", None) is None or self.x0_database.numel() == 0:
            raise RuntimeError("x0_database is empty. Build database before calling _precompute_multiscale_db().")

        x0_db_low_res = self._downsample_db(self.x0_database).contiguous()
        self._replace_buffer("x0_db_low_res", x0_db_low_res, persistent=False)

        x0_db_low_res_norm_sq = torch.sum(self.x0_db_low_res ** 2, dim=1)
        self._replace_buffer("x0_db_low_res_norm_sq", x0_db_low_res_norm_sq, persistent=False)

    def _get_knn_subset_and_distances(
        self,
        x_t_batch: torch.Tensor,
        alpha_t: Union[float, torch.Tensor],
        noise_level: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "k_neighbors", None) is None:
            raise RuntimeError("K-NN requested but k_neighbors is None.")
        if getattr(self, "x0_db_low_res", None) is None or self.x0_db_low_res.numel() == 0:
            raise RuntimeError("x0_db_low_res is empty. Call _precompute_multiscale_db() first.")
        if getattr(self, "x0_db_norm_sq", None) is None or self.x0_db_norm_sq.numel() == 0:
            raise RuntimeError("x0_db_norm_sq is empty. Build database first.")

        n_samples = x_t_batch.shape[0]
        n_data = int(self.n_data)
        k_final = int(self.k_neighbors)
        ## 10000
        m_candidates = int(self.m_coarse_candidates)

        noise_level = float(min(1.0, max(0.0, float(noise_level))))
        locality_factor = 1.0 - noise_level

        # m_t schedule (coarse candidate pool size)
        if self.dynamic_m:
            n_pre_sample_float = self.n_pre_sample_min + (self.n_pre_sample_max - self.n_pre_sample_min) * locality_factor
            n_pre_sample = min(self.n_pre_sample_max, max(self.n_pre_sample_min, int(round(n_pre_sample_float))))
        else:
            if self.static_m is not None:
                n_pre_sample = min(n_data, self.static_m)
            else:
                n_pre_sample = (self.n_pre_sample_min + self.n_pre_sample_max) // 2

        # k_t schedule (aggregation budget)
        k_min = int(self.k_neighbors)
        k_max = int(self.m_coarse_candidates)

        if self.dynamic_k:
            k_dynamic = int(round(k_max - (k_max - k_min) * locality_factor))
            k_final = max(k_min, k_dynamic)
        else:
            if self.static_k is not None:
                k_final = min(k_max, self.static_k)
            else:
                k_final = (k_min + k_max) // 2

        K = k_final

        # Random retrieval: skip coarse-to-fine, randomly select K samples
        if self.random_retrieval:
            rand_vals = torch.rand(n_samples, n_data, device=self.device)
            _, best_idx = torch.topk(rand_vals, K, dim=1, largest=False)
            flat_best = best_idx.reshape(-1)
            x0_final_subset = self.x0_database.index_select(0, flat_best).view(n_samples, K, int(self.feature_dim))
            sq_dist_final = torch.full((n_samples, K), 0.0, device=self.device, dtype=x_t_batch.dtype)
            return x0_final_subset, sq_dist_final

        # ---- pre-sampling ----
        rand_vals = torch.rand(n_samples, n_data, device=self.device)
        _, pre_sample_indices = torch.topk(rand_vals, n_pre_sample, dim=1, largest=False)  # [S, N_pre]

        # ---- coarse (low-res) ----
        m_to_find = min(m_candidates, n_pre_sample)
        x_t_low_res = self._downsample_batch(x_t_batch)
        x_t_low_res_norm_sq = (x_t_low_res ** 2).sum(dim=1)

        S = x_t_low_res.shape[0]
        N_pre = pre_sample_indices.shape[1]
        M = m_to_find
        chunk_size = int(getattr(self, "coarse_chunk_size", 1024))

        best_dist = torch.full((S, M), float("inf"), device=self.device, dtype=x_t_low_res.dtype)
        best_pos = torch.full((S, M), -1, device=self.device, dtype=torch.long)

        for start in range(0, N_pre, chunk_size):
            end = min(start + chunk_size, N_pre)
            idx_chunk = pre_sample_indices[:, start:end]
            C = idx_chunk.shape[1]

            flat = idx_chunk.reshape(-1)
            x0_flat = self.x0_db_low_res.index_select(0, flat)
            x0_chunk = x0_flat.view(S, C, -1)

            dot = torch.einsum("sd,scd->sc", x_t_low_res, x0_chunk)
            x0_norm = self.x0_db_low_res_norm_sq.index_select(0, flat).view(S, C)

            dist = x_t_low_res_norm_sq[:, None] + (alpha_t * alpha_t) * x0_norm - 2.0 * alpha_t * dot

            pos_chunk = torch.arange(start, end, device=self.device, dtype=torch.long)[None, :].expand(S, -1)
            merged_dist = torch.cat([best_dist, dist], dim=1)
            merged_pos = torch.cat([best_pos, pos_chunk], dim=1)

            best_dist, sel = torch.topk(merged_dist, M, dim=1, largest=False)
            best_pos = merged_pos.gather(1, sel)

        top_m_indices_in_pre_sample = best_pos

        pad_size = m_candidates - m_to_find
        if pad_size > 0:
            top_m_indices_in_pre_sample = torch.nn.functional.pad(
                top_m_indices_in_pre_sample, (0, pad_size), mode="replicate"
            )

        # ---- fine (full-res) ----
        db_idx_m = pre_sample_indices.gather(1, top_m_indices_in_pre_sample)  # [S, M]
        x_t_norm_sq = (x_t_batch ** 2).sum(dim=1)

        S = x_t_batch.shape[0]
        M = db_idx_m.shape[1]

        chunk_size = int(getattr(self, "fine_chunk_size", 256))

        best_dist = torch.full((S, K), float("inf"), device=x_t_batch.device, dtype=x_t_batch.dtype)
        best_idx = torch.full((S, K), -1, device=x_t_batch.device, dtype=torch.long)

        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            idx_chunk = db_idx_m[:, start:end]
            C = idx_chunk.shape[1]

            flat = idx_chunk.reshape(-1)
            x0_flat = self.x0_database.index_select(0, flat)
            x0_chunk = x0_flat.view(S, C, int(self.feature_dim))

            dot = torch.einsum("sd,scd->sc", x_t_batch, x0_chunk)
            x0_norm = self.x0_db_norm_sq.index_select(0, flat).view(S, C)

            dist = x_t_norm_sq[:, None] + (alpha_t * alpha_t) * x0_norm - 2.0 * alpha_t * dot

            merged_dist = torch.cat([best_dist, dist], dim=1)
            merged_idx = torch.cat([best_idx, idx_chunk], dim=1)

            best_dist, pos = torch.topk(merged_dist, K, dim=1, largest=False)
            best_idx = merged_idx.gather(1, pos)

        flat_best = best_idx.reshape(-1)
        x0_final_subset = self.x0_database.index_select(0, flat_best).view(S, K, int(self.feature_dim))
        sq_dist_final = best_dist
        return x0_final_subset, sq_dist_final



        # --------- new/replace: randomly pick K candidates ---------
        # rand_vals = torch.rand(n_samples, n_data, device=self.device)
        # _, best_idx = torch.topk(rand_vals, K, dim=1, largest=False)   # [S, K] random indices over full DB

        # flat_best = best_idx.reshape(-1)
        # x0_final_subset = self.x0_database.index_select(0, flat_best).view(n_samples, K, int(self.feature_dim))

