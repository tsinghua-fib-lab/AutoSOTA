from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist

from data_src import DatasetBundle
from methods import register_model
from methods.base import BaseDenoiser, StreamingSoftmax, WeightedStreamingSoftmax


LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def is_main_process() -> bool:
    """Check if the current process is the main distributed process."""
    return (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)


# -----------------------------------------------------------------------------
# Model (KNN-only)
# -----------------------------------------------------------------------------

@register_model("ours")
class OursDenoiser(BaseDenoiser):
    """
    Analytical diffusion model using PCA-based KNN screening (KNN-only version).
    """

    def __init__(
        self,
        dataset: DatasetBundle,
        device: str,
        num_steps: int,
        *,
        params: Optional[Dict[str, Any]] = None,
        class_id: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        params = params or {}
        super().__init__(**BaseDenoiser.init_kwargs_from_dataset(dataset, device, num_steps, **kwargs))

        # --- Parameters ---
        self.temperature = float(params.get("temperature", 1.0))
        self.mask_threshold = float(params.get("mask_threshold", 0.02))
        self.weight_method = str(params.get("weight_method", "ss"))  # 'wss' or 'ss'; paper uses 'ss'
        self.class_id = class_id
        self.eps = 1e-6

        # --- Profiling accumulators (CODE-01) ---
        self._profile_enabled = True
        self._profile_data: dict = {
            'mask_compute_ms': [],
            'knn_retrieval_ms': [],
            'projection_einsum_ms': [],
            'mask_memory_mb': [],
            'knn_memory_mb': [],
            'total_memory_mb': [],
        }

        # --- Helpers Initialization ---
        self._init_knn_params(params, default_coarse=1024, default_fine=256)
        self._init_knn_buffers()
        self.use_knn_screening = True

        ##add
        # self.m_coarse_candidates_min = self.k_neighbors
        # self.m_coarse_candidates_max = 10000

        # self.k_neighbors_min = 1000
        # self.k_neighbors_max = 3000

        # --- Path Setup ---
        self._init_wiener_path(dataset, params, key="wiener_path")
        self.global_wiener_path = self.wiener_path

        # Specific handling for ImageNet class subdirectories
        if dataset.name == "imagenet_1k" and self.class_id is not None:
            self.global_wiener_path = self.wiener_path / "global"
            self.wiener_path = self.wiener_path / f"{int(self.class_id):04d}"
            LOGGER.info(f"[Ours] Global path: {self.global_wiener_path}")
            LOGGER.info(f"[Ours] Local path:  {self.wiener_path}")

        self.dataset: Optional[DatasetBundle] = None

    def train(self, dataset: DatasetBundle):
        """
        Prepare the model: Load SVD stats and build/cache the x0 database.
        KNN-only: build DB for KNN indexing and setup hierarchical structures.
        """
        # 1. Load PCA/Wiener Statistics
        U, LA, Vh, mean = self._load_or_compute_wiener_svd(
            self.global_wiener_path,
            self.wiener_path,
            dataset_name=dataset.name,
            dataloader=dataset.dataloader,
            resolution=self.resolution,
            n_channels=self.n_channels,
            logger=LOGGER,
        )
        self._register_wiener_buffers(U, LA, Vh, mean)
        self.dataset = dataset

        # 2. Build flattened x0 database (for KNN)
        LOGGER.info("Building x0 database (KNN-only)...")
        x0_db = self._build_flat_database(
            dataset.dataloader,
            dtype=torch.float32,
            desc="Build x0 DB",
            leave=False,
            to_01=False,
        )

        # 3. Register KNN Buffers + Build KNN Index
        self.n_data = int(x0_db.shape[0])
        self.feature_dim = int(x0_db.shape[1])
        LOGGER.info(f"[Ours] n_data={self.n_data}, feature_dim={self.feature_dim}")

        self._replace_buffer("x0_database", x0_db.to(self.device).contiguous(), persistent=False)
        self._replace_buffer("x0_db_norm_sq", torch.sum(self.x0_database ** 2, dim=1), persistent=False)

        self._setup_hierarchical_knn()
        self._precompute_multiscale_db()

        return self

    def _compute_projection_mask(
        self, timestep_index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the PCA locality mask M = (U * shrink * V^h) / diag
        """
        if not all(hasattr(self, attr) for attr in ["U", "LA", "Vh"]):
            raise RuntimeError("Model not trained. Missing SVD components.")

        alpha_prod_t = self.scheduler.alphas_cumprod[timestep_index].to(self.device)
        beta_prod_t = 1 - alpha_prod_t

        shrink_factors = alpha_prod_t * self.LA / (beta_prod_t + alpha_prod_t * self.LA)
        LLt = (self.U * shrink_factors.unsqueeze(0)) @ self.Vh

        denom = torch.diagonal(LLt).unsqueeze(1)
        denom[denom.abs() < self.eps] = 1.0
        mask = LLt / denom

        if self.mask_threshold > 0:
            threshold = mask.abs().max() * self.mask_threshold
            mask = torch.where(mask.abs() >= threshold, torch.ones_like(mask), torch.zeros_like(mask))

        return mask, alpha_prod_t, beta_prod_t

    @torch.no_grad()
    def denoise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        KNN-only: always uses _denoise_knn.
        Includes per-stage CUDA event profiling (CODE-01).
        """
        if self.dataset is None:
            raise RuntimeError("Model not trained. Call model.train(dataset) first.")

        self.t_idx = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        do_profile = self._profile_enabled and torch.cuda.is_available()

        # --- Stage 1: Compute projection mask ---
        if do_profile:
            mem_before = torch.cuda.max_memory_allocated() / (1024 ** 2)
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()

        mask, alpha_prod_t, beta_prod_t = self._compute_projection_mask(self.t_idx)

        if do_profile:
            end_ev.record()
            torch.cuda.synchronize()
            mask_ms = start_ev.elapsed_time(end_ev)
            mask_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) - mem_before
            self._profile_data['mask_compute_ms'].append(float(mask_ms))
            self._profile_data['mask_memory_mb'].append(max(0.0, float(mask_mem)))

        sqrt_alpha = torch.sqrt(alpha_prod_t).to(latents.device, dtype=latents.dtype)
        xt_flat = latents.flatten(start_dim=1)  # [B, D]

        # --- Stage 2: KNN subset retrieval ---
        if do_profile:
            mem_before = torch.cuda.max_memory_allocated() / (1024 ** 2)
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()

        noise_level = float(torch.sqrt(torch.clamp(beta_prod_t, min=0.0)).item())
        x0_subset, _ = self._get_knn_subset_and_distances(
            x_t_batch=xt_flat,
            alpha_t=sqrt_alpha,
            noise_level=noise_level,
        )

        if do_profile:
            end_ev.record()
            torch.cuda.synchronize()
            knn_ms = start_ev.elapsed_time(end_ev)
            knn_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) - mem_before
            self._profile_data['knn_retrieval_ms'].append(float(knn_ms))
            self._profile_data['knn_memory_mb'].append(max(0.0, float(knn_mem)))

        # --- Stage 3: Projection einsum + streaming softmax ---
        if do_profile:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()

        x0_mean = self._denoise_knn(xt_flat, sqrt_alpha, beta_prod_t, mask, x0_subset)

        if do_profile:
            end_ev.record()
            torch.cuda.synchronize()
            proj_ms = start_ev.elapsed_time(end_ev)
            self._profile_data['projection_einsum_ms'].append(float(proj_ms))
            self._profile_data['total_memory_mb'].append(
                float(torch.cuda.max_memory_allocated() / (1024 ** 2))
            )

        if x0_mean is None:
            raise RuntimeError("Failed to compute softmax average (result is None).")

        return x0_mean.view_as(latents)

    def _denoise_knn(
        self,
        xt: torch.Tensor,
        sqrt_alpha: torch.Tensor,
        beta_prod_t: torch.Tensor,
        mask: torch.Tensor,
        x0_subset: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Denoising using Hierarchical KNN to select a subset of x0 candidates.

        Args:
            xt: Noisy latents [B, D].
            sqrt_alpha: sqrt(alpha_prod_t).
            beta_prod_t: 1 - alpha_prod_t.
            mask: PCA projection mask [D, D].
            x0_subset: Pre-computed KNN subset [B, K, D]. If None, computed here.
        """
        do_store = False  # keep as your original code

        if x0_subset is None:
            noise_level = float(torch.sqrt(torch.clamp(beta_prod_t, min=0.0)).item())
            # Retrieve subset: [B, K, D]
            x0_subset, _ = self._get_knn_subset_and_distances(
                x_t_batch=xt,
                alpha_t=sqrt_alpha,
                noise_level=noise_level,
            )

        # Compute PCA-Masked Distance
        delta = (xt.unsqueeze(1) - sqrt_alpha * x0_subset) ** 2  # [B, K, N]
        # Use torch.matmul instead of einsum — dispatches to optimized cuBLAS GEMM.
        # Equivalent to einsum("bkn,nm->bkm", delta, mask).
        ds_chunk = torch.matmul(delta, mask)
        logits = -ds_chunk / (2 * beta_prod_t * self.temperature)

        # KNN subset in memory => global online softmax
        if self.weight_method == "wss":
            aggregator = WeightedStreamingSoftmax(device=xt.device, dtype=xt.dtype, store_weights=do_store)
        else:  # 'ss' (paper default for ours: unbiased streaming softmax over the golden subset)
            aggregator = StreamingSoftmax(device=xt.device, dtype=xt.dtype, store_weights=do_store)
        aggregator.add(x0_subset, logits)

        if do_store:
            all_sample_weights = aggregator.get_all_weights()
            torch.save(all_sample_weights, f"weights_Ours_step_{self.t_idx}.pt")
            LOGGER.info(f"Saved weights_Ours_step_{self.t_idx}.pt")

        return aggregator.get_average()

    def get_profile_summary(self) -> dict:
        """Return per-stage profiling summary averaged across all denoising steps (CODE-01)."""
        if not getattr(self, '_profile_enabled', False):
            return {'profile_enabled': False}
        d = self._profile_data
        if len(d.get('mask_compute_ms', [])) == 0:
            return {'profile_enabled': True, 'num_steps_profiled': 0}

        def safe_mean(lst):
            return float(sum(lst) / len(lst)) if lst else 0.0

        n = len(d['mask_compute_ms'])
        mask_avg = safe_mean(d['mask_compute_ms'])
        knn_avg = safe_mean(d['knn_retrieval_ms'])
        proj_avg = safe_mean(d['projection_einsum_ms'])

        return {
            'profile_enabled': True,
            'num_steps_profiled': n,
            'mask_compute_ms_avg': round(mask_avg, 3),
            'mask_compute_pct': round(100.0 * mask_avg / max(0.001, mask_avg + knn_avg + proj_avg), 1),
            'knn_retrieval_ms_avg': round(knn_avg, 3),
            'knn_retrieval_pct': round(100.0 * knn_avg / max(0.001, mask_avg + knn_avg + proj_avg), 1),
            'projection_einsum_ms_avg': round(proj_avg, 3),
            'projection_einsum_pct': round(100.0 * proj_avg / max(0.001, mask_avg + knn_avg + proj_avg), 1),
            'total_per_step_ms_avg': round(mask_avg + knn_avg + proj_avg, 3),
            'peak_gpu_memory_mb': round(max(d['total_memory_mb']), 1) if d['total_memory_mb'] else 0.0,
        }
