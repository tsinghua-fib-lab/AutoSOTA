import datetime
import os
import sys

import fire
import numpy as np
import torch
import torchvision.utils as tvu
import wandb


_THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_APPROXPOSTERIOR_ROOT = os.path.dirname(_THIS_FILE_DIR)
_SIBLING_CALIBRATED_GUIDANCE_ROOT = os.path.abspath(os.path.join(_APPROXPOSTERIOR_ROOT, "..", "Calibrated-Guidance"))
for extra_path in (_APPROXPOSTERIOR_ROOT, _SIBLING_CALIBRATED_GUIDANCE_ROOT):
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)

from pixel_space_inverse_problems.imagenet_test_images import load_imagenet_test_images_and_class_labels  # noqa: E402
from pixel_space_inverse_problems.pixel_mean_flow_adapter import (  # noqa: E402
    DEFAULT_PMF_B16_CHECKPOINT, PixelMeanFlowOneStepModel, build_pmf_module,
)
from pixel_space_inverse_problems.pixel_colorization_likelihood import (  # noqa: E402
    PixelSpaceColorizationLikelihood,
    build_tiled_pixel_colorization_log_likelihood,
)
from pixel_space_inverse_problems.pixel_superresolution_likelihood import PixelSpaceSuperResolutionLikelihood  # noqa: E402
from pixel_space_inverse_problems.superresolution_likelihood import downsample_pixels_by_bicubic_4x  # noqa: E402

from calibrated_guidance.diffusion_posterior.base import DiffusionPosterior, SampleOutput  # noqa: E402
from calibrated_guidance.guidance import build_reinforce_mean_fn, build_reparam_mean_fn  # noqa: E402
from calibrated_guidance.inference import flow_matching  # noqa: E402


DEFAULT_TEST_IMAGES_DIR = os.path.join(_THIS_FILE_DIR, "test_images")
_RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_OUTPUT_DIR = os.path.join(_THIS_FILE_DIR, "results_jan", f"pmf_pixel_space_run_{_RUN_TIMESTAMP}")


class MultiDevicePixelMeanFlowOneStepModel:
    # Wraps N PixelMeanFlowOneStepModel replicas on N CUDA devices. predict_x0_one_step
    # chunks the input batch along dim 0, dispatches each chunk to its replica with
    # async cross-device copies, and gathers outputs back to the primary device. GPU
    # compute on distinct devices overlaps via CUDA's async execution; Python-side
    # dispatch is sequential but cheap relative to the DiT forward. When the batch is
    # smaller than the replica count we fall back to the primary replica to avoid
    # empty chunks (e.g. mean(x_t, t) with batch_size=1).
    def __init__(self, replicas, devices):
        assert len(replicas) == len(devices) and len(replicas) > 0
        self._replicas = list(replicas)
        self._devices = [torch.device(d) for d in devices]
        self._primary = self._devices[0]

    @property
    def img_size(self):
        return self._replicas[0].img_size

    @property
    def img_channels(self):
        return self._replicas[0].img_channels

    @property
    def num_classes(self):
        return self._replicas[0].num_classes

    @property
    def noise_scale(self):
        return self._replicas[0].noise_scale

    def predict_x0_one_step(self, x_t, t_scalar, labels):
        n_replicas = len(self._replicas)
        if x_t.shape[0] < n_replicas:
            return self._replicas[0].predict_x0_one_step(
                x_t.to(self._primary), t_scalar, labels.to(self._primary),
            )
        x_chunks = torch.chunk(x_t, n_replicas, dim=0)
        label_chunks = torch.chunk(labels, n_replicas, dim=0)
        per_device_outputs = []
        for replica, device, x_chunk, label_chunk in zip(self._replicas, self._devices, x_chunks, label_chunks):
            x_on_device = x_chunk.to(device, non_blocking=True)
            labels_on_device = label_chunk.to(device, non_blocking=True)
            per_device_outputs.append(replica.predict_x0_one_step(x_on_device, t_scalar, labels_on_device))
        gathered = [out.to(self._primary, non_blocking=True) for out in per_device_outputs]
        return torch.cat(gathered, dim=0)


class PMFOneShotSiblingReinforceDiffusionPosterior(DiffusionPosterior):
    # mean(x_t, t): one-step pMF prediction of x_0.
    # sample(x_t, t, N): predict an anchor x_0_hat via one-step pMF, re-noise the anchor to
    # `sibling_renoise_t` with N independent noises, then one-step re-predict to obtain N
    # diverse x_0 siblings. With sibling_renoise_t = t the siblings sit at the same noise
    # level as x_t.
    def __init__(self, pmf_one_step_model: PixelMeanFlowOneStepModel, class_labels_batch: torch.Tensor, sibling_renoise_t=None):
        self._model = pmf_one_step_model
        self._labels = class_labels_batch
        self._sibling_renoise_t = sibling_renoise_t

    def mean(self, x_t, t):
        t_scalar = float(t.item()) if torch.is_tensor(t) else float(t)
        return self._model.predict_x0_one_step(x_t, t_scalar, self._labels).clamp(-1.0, 1.0)

    def sample(self, x_t, t, num_samples_per_step):
        t_scalar = float(t.item()) if torch.is_tensor(t) else float(t)
        anchor_x0 = self._model.predict_x0_one_step(x_t, t_scalar, self._labels).clamp(-1.0, 1.0)
        batch_size = anchor_x0.shape[0]
        spatial_shape = anchor_x0.shape[1:]

        renoise_t = self._sibling_renoise_t if self._sibling_renoise_t is not None else t_scalar
        renoise_t = float(max(min(renoise_t, 1.0), 1e-3))

        expanded_anchor = anchor_x0.unsqueeze(1).expand(batch_size, num_samples_per_step, *spatial_shape).contiguous()
        noise_scale = self._model.noise_scale
        fresh_noise = torch.randn_like(expanded_anchor) * noise_scale
        renoised_xt = (1.0 - renoise_t) * expanded_anchor + renoise_t * fresh_noise

        flat_xt = renoised_xt.view(batch_size * num_samples_per_step, *spatial_shape)
        flat_labels = self._labels.unsqueeze(1).expand(batch_size, num_samples_per_step).reshape(batch_size * num_samples_per_step)
        flat_clean = self._model.predict_x0_one_step(flat_xt, renoise_t, flat_labels).clamp(-1.0, 1.0)
        sampled_x0 = flat_clean.view(batch_size, num_samples_per_step, *spatial_shape)
        return SampleOutput(samples=sampled_x0)


class AdaptiveGuidanceScaleLikelihood:
    def __init__(self, inner_likelihood, target_max_weight=0.5, num_binary_search_steps=20):
        self._inner = inner_likelihood
        self._target = target_max_weight
        self._num_search_steps = num_binary_search_steps
        self.last_adaptive_scale = None
        self.last_raw_log_likelihoods = None

    def __call__(self, x0_samples_bnchw):
        raw_log_likelihoods = self._inner(x0_samples_bnchw)
        self.last_raw_log_likelihoods = raw_log_likelihoods.detach()
        scale = self._find_scale_for_target_max_weight(raw_log_likelihoods)
        self.last_adaptive_scale = scale
        return scale * raw_log_likelihoods

    def _find_scale_for_target_max_weight(self, raw_ll):
        hi = 1.0
        for _ in range(50):
            if torch.softmax(hi * raw_ll, dim=1).max().item() > self._target:
                break
            hi *= 2.0
        lo = 0.0
        for _ in range(self._num_search_steps):
            mid = (lo + hi) / 2.0
            if torch.softmax(mid * raw_ll, dim=1).max().item() > self._target:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2.0


class TiledAdaptiveGuidanceScaleLikelihood:
    # Per-tile analogue of AdaptiveGuidanceScaleLikelihood. Expects inner likelihood to return
    # (B, N, tile_count) and finds a per-(B, tile) scale via vectorized binary search so that
    # each tile's max-particle softmax weight reaches `target_max_weight`. Returns scale * raw
    # of shape (B, N, tile_count). `last_adaptive_scale` exposes the mean-over-tiles scalar
    # (for plotter compat) and `last_adaptive_scale_per_tile` exposes the full (B, a) tensor.
    def __init__(self, inner_likelihood, target_max_weight=0.9, num_binary_search_steps=20):
        self._inner = inner_likelihood
        self._target = float(target_max_weight)
        self._num_search_steps = int(num_binary_search_steps)
        self.last_adaptive_scale = None
        self.last_adaptive_scale_per_tile = None
        self.last_raw_log_likelihoods = None

    def __call__(self, x0_samples_bnchw):
        raw_log_likelihoods = self._inner(x0_samples_bnchw)
        self.last_raw_log_likelihoods = raw_log_likelihoods.detach()
        scale_per_tile = self._find_scale_per_tile(raw_log_likelihoods)
        self.last_adaptive_scale_per_tile = scale_per_tile.detach()
        self.last_adaptive_scale = float(scale_per_tile.detach().mean().item())
        return scale_per_tile.unsqueeze(1) * raw_log_likelihoods

    def _find_scale_per_tile(self, raw_ll):
        # Calibration constant — detach so reparam-pool gradient flow only goes through
        # raw_ll, not through the binary search ops (which include non-differentiable
        # comparisons and sometimes raise "could not compute gradients" otherwise).
        raw_ll = raw_ll.detach()
        batch_size, _num_particles, tile_count = raw_ll.shape
        hi = torch.ones((batch_size, tile_count), device=raw_ll.device, dtype=raw_ll.dtype)
        for _ in range(50):
            max_per_tile = torch.softmax(hi.unsqueeze(1) * raw_ll, dim=1).max(dim=1).values
            needs_expand = max_per_tile <= self._target
            if not needs_expand.any():
                break
            hi = torch.where(needs_expand, hi * 2.0, hi)
        lo = torch.zeros_like(hi)
        for _ in range(self._num_search_steps):
            mid = (lo + hi) / 2.0
            max_per_tile = torch.softmax(mid.unsqueeze(1) * raw_ll, dim=1).max(dim=1).values
            above = max_per_tile > self._target
            hi = torch.where(above, mid, hi)
            lo = torch.where(above, lo, mid)
        return (lo + hi) / 2.0


def save_pixel_image_minus1_to_1(pixels_tensor, destination_path):
    normalized = (pixels_tensor.detach().clamp(-1, 1) + 1.0) / 2.0
    tvu.save_image(normalized, destination_path)


def to_wandb_image(pixels_chw_minus1to1, caption=None):
    normalized = ((pixels_chw_minus1to1.detach().clamp(-1, 1) + 1.0) / 2.0).cpu().numpy().transpose(1, 2, 0)
    return wandb.Image(normalized, caption=caption)


def build_time_steps_noise_to_clean(num_steps, eps=1e-3):
    return torch.linspace(1.0, eps, num_steps + 1)


def build_intermediate_particle_image_plotter(
    intermediate_results_dir, image_index, save_images=True, adaptive_likelihood=None,
    direction_save_dir=None, num_random_candidates: int = 0,
    save_spread_map: bool = False, spread_metric: str = "std", spread_scale: float = 1.0,
):
    step_counter = [0]

    def plot_intermediate_x0_particles(xt, x0_samples, log_likelihoods, t, direction_to_x0=None):
        step_index = step_counter[0]

        best_log_likelihood_image = None
        if log_likelihoods is not None:
            ll_detached = log_likelihoods.detach()
            best_ll_idx = ll_detached.argmax(dim=1)  # (B,)
            samples_bnchw = x0_samples.samples
            b, _, c, h, w = samples_bnchw.shape
            best_gather = best_ll_idx.view(b, 1, 1, 1, 1).expand(b, 1, c, h, w)
            best_log_likelihood_image = samples_bnchw.gather(dim=1, index=best_gather).squeeze(1).detach()

        if log_likelihoods is not None:
            ll_flat = log_likelihoods.detach().flatten()
            weights = torch.softmax(log_likelihoods.detach() + x0_samples.log_weights.detach(), dim=1).flatten()
            num_particles = weights.shape[0]
            uniform_weight = 1.0 / num_particles
            kl_from_uniform = (weights * (weights / uniform_weight).clamp(min=1e-12).log()).sum().item()
            sorted_weights = weights.sort(descending=True).values
            top_weights_dict = {f"top_weights/rank_{rank + 1}": sorted_weights[rank].item() for rank in range(min(5, num_particles))}
            t_scalar = float(t) if not torch.is_tensor(t) else float(t.item())
            metrics = {
                "log_likelihood/min": ll_flat.min().item(), "log_likelihood/max": ll_flat.max().item(),
                "log_likelihood/mean": ll_flat.mean().item(), "log_likelihood/median": ll_flat.median().item(),
                "weight_diagnostics/kl_from_uniform": kl_from_uniform, **top_weights_dict,
                "image_index": image_index, "diffusion_step": step_index, "t": t_scalar,
            }
            if adaptive_likelihood is not None and adaptive_likelihood.last_adaptive_scale is not None:
                metrics["adaptive_guidance_scale"] = adaptive_likelihood.last_adaptive_scale
            per_tile_scale = getattr(adaptive_likelihood, "last_adaptive_scale_per_tile", None) if adaptive_likelihood is not None else None
            if per_tile_scale is not None:
                per_tile_scale_flat = per_tile_scale.flatten()
                metrics["adaptive_guidance_scale_per_tile/min"] = per_tile_scale_flat.min().item()
                metrics["adaptive_guidance_scale_per_tile/max"] = per_tile_scale_flat.max().item()
                metrics["adaptive_guidance_scale_per_tile/mean"] = per_tile_scale_flat.mean().item()
            if adaptive_likelihood is not None and adaptive_likelihood.last_raw_log_likelihoods is not None:
                raw_ll_flat = adaptive_likelihood.last_raw_log_likelihoods.flatten()
                metrics["raw_log_likelihood/min"] = raw_ll_flat.min().item()
                metrics["raw_log_likelihood/max"] = raw_ll_flat.max().item()
                metrics["raw_log_likelihood/mean"] = raw_ll_flat.mean().item()
                metrics["raw_log_likelihood/median"] = raw_ll_flat.median().item()
            if direction_to_x0 is not None:
                metrics["direction_to_x0"] = to_wandb_image(direction_to_x0[0], caption=f"image {image_index} step {step_index}")
            if xt is not None:
                metrics["x_t"] = to_wandb_image(xt[0], caption=f"image {image_index} step {step_index} t={t_scalar:.3f}")
            if best_log_likelihood_image is not None:
                metrics["best_log_likelihood_image"] = to_wandb_image(
                    best_log_likelihood_image[0],
                    caption=f"image {image_index} step {step_index} ll={ll_flat.max().item():.3f}",
                )
            wandb.log(metrics)
        elif direction_to_x0 is not None or xt is not None:
            log_payload = {"image_index": image_index, "diffusion_step": step_index}
            if direction_to_x0 is not None:
                log_payload["direction_to_x0"] = to_wandb_image(direction_to_x0[0], caption=f"image {image_index} step {step_index}")
            if xt is not None:
                log_payload["x_t"] = to_wandb_image(xt[0], caption=f"image {image_index} step {step_index}")
            wandb.log(log_payload)

        if save_images:
            step_directory = os.path.join(intermediate_results_dir, f"image_{image_index:03d}", f"step_{step_index:03d}")
            os.makedirs(step_directory, exist_ok=True)
            particles_bnchw = x0_samples.samples
            num_particles = particles_bnchw.shape[1]
            spatial_shape = particles_bnchw.shape[2:]
            flat_particle_pixels = particles_bnchw.reshape(particles_bnchw.shape[0] * num_particles, *spatial_shape).clamp(-1, 1)
            for particle_index in range(num_particles):
                save_pixel_image_minus1_to_1(flat_particle_pixels[particle_index], os.path.join(step_directory, f"particle_{particle_index:02d}.png"))
            if direction_to_x0 is not None:
                save_pixel_image_minus1_to_1(direction_to_x0[0], os.path.join(step_directory, "direction_to_x0.png"))
            if xt is not None:
                save_pixel_image_minus1_to_1(xt[0], os.path.join(step_directory, "x_t.png"))
            if best_log_likelihood_image is not None:
                save_pixel_image_minus1_to_1(best_log_likelihood_image[0], os.path.join(step_directory, "best_log_likelihood_image.png"))

        if direction_save_dir is not None:
            per_image_dir = os.path.join(direction_save_dir, f"{image_index:04d}")
            os.makedirs(per_image_dir, exist_ok=True)
            if direction_to_x0 is not None:
                save_pixel_image_minus1_to_1(
                    direction_to_x0[0],
                    os.path.join(per_image_dir, f"step_{step_index:02d}_direction.png"),
                )
            if xt is not None:
                save_pixel_image_minus1_to_1(
                    xt[0],
                    os.path.join(per_image_dir, f"step_{step_index:02d}_xt.png"),
                )
            if num_random_candidates > 0 and x0_samples is not None:
                particles = x0_samples.samples[0]  # (N, C, H, W)
                num_particles = particles.shape[0]
                k = min(int(num_random_candidates), num_particles)
                chosen_idx = torch.randperm(num_particles, device=particles.device)[:k]
                for j, p_idx in enumerate(chosen_idx.tolist()):
                    save_pixel_image_minus1_to_1(
                        particles[p_idx],
                        os.path.join(per_image_dir, f"step_{step_index:02d}_sample_{j}.png"),
                    )
            if save_spread_map and x0_samples is not None:
                from pixel_space_inverse_problems.particle_spread_visualization import (
                    save_per_channel_spread_png,
                )
                save_per_channel_spread_png(
                    x0_samples.samples[0],
                    os.path.join(per_image_dir, f"step_{step_index:02d}_{spread_metric}.png"),
                    metric=spread_metric, scale=spread_scale,
                )

        step_counter[0] += 1

    return plot_intermediate_x0_particles


def build_tiled_pixel_sr_log_likelihood(observation_low_res_m1to1, noise_std, tile_count):
    # Returns a callable (B, N, C, H, W) -> (B, N, tile_count) of per-tile Gaussian SR log-likelihoods.
    # Downsamples each whole sample once (avoiding boundary artifacts that arise from per-tile bicubic),
    # then sums the squared-error map over a row-major k x k = tile_count grid in observation space.
    # By per-pixel-Gaussian independence, sum_over_tiles(log_lik_per_tile) == whole-image SR log-lik.
    k = int(round(tile_count ** 0.5))
    assert k * k == tile_count, f"tile_count={tile_count} must be a perfect square"
    noise_std_value = float(noise_std)

    def _tiled_log_likelihoods(samples_bnchw):
        batch_size, num_particles, channels, height, width = samples_bnchw.shape
        flat_samples = samples_bnchw.reshape(batch_size * num_particles, channels, height, width).clamp(-1.0, 1.0)
        predicted_obs_flat = downsample_pixels_by_bicubic_4x(flat_samples)  # (B*N, C, obs_h, obs_w)
        _, _, obs_h, obs_w = predicted_obs_flat.shape
        assert obs_h % k == 0 and obs_w % k == 0, f"observation {obs_h}x{obs_w} not divisible by k={k}"
        obs_tile_h, obs_tile_w = obs_h // k, obs_w // k

        sq_err = (predicted_obs_flat - observation_low_res_m1to1).pow(2)  # broadcasts over batch
        sq_err_tiled = sq_err.reshape(batch_size * num_particles, channels, k, obs_tile_h, k, obs_tile_w)
        sq_err_tiled = sq_err_tiled.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(batch_size * num_particles, tile_count, channels * obs_tile_h * obs_tile_w)
        sq_err_per_tile = sq_err_tiled.sum(dim=-1)
        return (-0.5 * sq_err_per_tile / (noise_std_value ** 2)).reshape(batch_size, num_particles, tile_count)

    return _tiled_log_likelihoods


def build_tiled_reinforce_mean_fn(diffusion_posterior, tile_log_likelihood_fn, num_samples_per_step, tile_count, guidance_scale=1.0, plot_samples=None):
    # REINFORCE mean estimator on a k x k grid of tiles: per-tile softmax weights pick the best
    # sibling for each tile independently, then tiles are stitched back to a full image. Reduces
    # to standard whole-image REINFORCE at tile_count=1 (but takes the new path in this case too;
    # use the upstream build_reinforce_mean_fn if you want exact equivalence).
    k = int(round(tile_count ** 0.5))
    assert k * k == tile_count, f"tile_count={tile_count} must be a perfect square"

    def tiled_reinforce_mean_fn(xt, t):
        with torch.no_grad():
            x0_samples = diffusion_posterior.sample(xt, t, num_samples_per_step)
            samples = x0_samples.samples
            batch_size, num_particles, channels, height, width = samples.shape
            tile_h, tile_w = height // k, width // k

            log_lik_per_tile = guidance_scale * tile_log_likelihood_fn(samples)
            log_weights_b_n_a = x0_samples.log_weights.unsqueeze(-1).expand(batch_size, num_particles, tile_count)
            weights_b_n_a = torch.softmax(log_lik_per_tile + log_weights_b_n_a, dim=1)

            samples_tiles = samples.reshape(batch_size, num_particles, channels, k, tile_h, k, tile_w)
            samples_tiles = samples_tiles.permute(0, 1, 3, 5, 2, 4, 6).contiguous().reshape(batch_size, num_particles, tile_count, channels, tile_h, tile_w)
            mixed_tiles = (weights_b_n_a.reshape(batch_size, num_particles, tile_count, 1, 1, 1) * samples_tiles).sum(dim=1)
            mixed_full = mixed_tiles.reshape(batch_size, k, k, channels, tile_h, tile_w).permute(0, 3, 1, 4, 2, 5).contiguous().reshape(batch_size, channels, height, width)

        if plot_samples is not None:
            plot_samples(xt, x0_samples, log_lik_per_tile.mean(dim=-1), t, direction_to_x0=mixed_full)
        return mixed_full

    return tiled_reinforce_mean_fn


def run_single_image_reinforce_guided_sampling(
    pmf_one_step_model: PixelMeanFlowOneStepModel, observation_low_res, class_label, num_outer_steps, num_particles,
    noise_std, guidance_scale, sibling_renoise_t, device, estimator: str = "reinforce", tile_count: int = 1,
    intermediate_results_dir=None, image_index=0, save_intermediate_images=True,
    task: str = "superresolution",
    direction_save_dir=None,
    num_random_candidates_per_step: int = 0,
    save_spread_map: bool = False,
    spread_metric: str = "std",
    spread_scale: float = 1.0,
):
    # `observation_low_res` is the task observation: bicubic-downsampled RGB
    # (B, 3, H/4, W/4) for superresolution or grayscale (B, 1, H, W) for
    # colorization. The variable name is kept for backward compat.
    use_tiled = estimator == "reinforce" and tile_count > 1
    if task == "superresolution":
        if use_tiled:
            raw_likelihood = build_tiled_pixel_sr_log_likelihood(observation_low_res, noise_std, tile_count)
        else:
            raw_likelihood = PixelSpaceSuperResolutionLikelihood(observation_low_res_m1to1=observation_low_res, noise_std=noise_std)
    elif task == "colorization":
        if use_tiled:
            raw_likelihood = build_tiled_pixel_colorization_log_likelihood(observation_low_res, noise_std, tile_count)
        else:
            raw_likelihood = PixelSpaceColorizationLikelihood(observation_gray_m1to1=observation_low_res, noise_std=noise_std)
    else:
        raise ValueError(f"unknown task {task!r}; expected 'superresolution' or 'colorization'")
    if guidance_scale is None:
        adaptive_cls = TiledAdaptiveGuidanceScaleLikelihood if use_tiled else AdaptiveGuidanceScaleLikelihood
        adaptive_likelihood = adaptive_cls(raw_likelihood)
        effective_likelihood = adaptive_likelihood
        effective_guidance_scale = 1.0
    else:
        adaptive_likelihood = None
        effective_likelihood = raw_likelihood
        effective_guidance_scale = guidance_scale

    intermediate_plotter = build_intermediate_particle_image_plotter(
        intermediate_results_dir=intermediate_results_dir or "", image_index=image_index,
        save_images=(save_intermediate_images and intermediate_results_dir is not None),
        adaptive_likelihood=adaptive_likelihood,
        direction_save_dir=direction_save_dir,
        num_random_candidates=num_random_candidates_per_step,
        save_spread_map=save_spread_map,
        spread_metric=spread_metric,
        spread_scale=spread_scale,
    )

    posterior = PMFOneShotSiblingReinforceDiffusionPosterior(
        pmf_one_step_model=pmf_one_step_model, class_labels_batch=class_label,
        sibling_renoise_t=sibling_renoise_t,
    )
    if use_tiled:
        main_mean_function = build_tiled_reinforce_mean_fn(
            diffusion_posterior=posterior, tile_log_likelihood_fn=effective_likelihood,
            num_samples_per_step=num_particles, tile_count=tile_count,
            guidance_scale=effective_guidance_scale, plot_samples=intermediate_plotter,
        )
    else:
        if estimator == "reinforce":
            mean_function_builder = build_reinforce_mean_fn
        elif estimator == "reparam":
            mean_function_builder = build_reparam_mean_fn
        else:
            raise ValueError(f"unknown estimator {estimator!r}; expected 'reinforce' or 'reparam'")
        main_mean_function = mean_function_builder(
            diffusion_posterior=posterior, log_likelihood_fn=effective_likelihood,
            num_samples_per_step=num_particles, guidance_scale=effective_guidance_scale, plot_samples=intermediate_plotter,
        )

    mean_function = main_mean_function

    time_steps = build_time_steps_noise_to_clean(num_outer_steps).to(device)
    img_size = pmf_one_step_model.img_size
    img_channels = pmf_one_step_model.img_channels
    final_pixels = flow_matching(
        mean_function=mean_function, time_steps=time_steps, shape=(img_channels, img_size, img_size),
        n_samples=1, device=device, use_tqdm=True,
    )
    return final_pixels.clamp(-1.0, 1.0)


def _pool_shared_mean(x0_samples, effective_likelihood, effective_guidance_scale, tile_count, tile_grid_k):
    # Compute the shared x0_hat target the pool sampler will broadcast as drift to all P
    # particles. Mirrors `build_tiled_reinforce_mean_fn`: at tile_count=1 it's a whole-image
    # softmax; at tile_count>1 it's per-tile softmax + tile stitching. Returns the shared
    # mean shaped (1, C, H, W) plus a (1, K) log-likelihood tensor for the plotter (mean
    # over tiles in the tiled case, matching the tiled-reinforce mean fn).
    samples = x0_samples.samples  # (1, K, C, H, W)
    batch_size, num_particles, channels, height, width = samples.shape
    if tile_count <= 1:
        log_likelihoods = effective_guidance_scale * effective_likelihood(samples)  # (1, K)
        weights = torch.softmax(log_likelihoods + x0_samples.log_weights, dim=1)
        shared_mean = (weights.view(batch_size, num_particles, 1, 1, 1) * samples).sum(dim=1)
        return shared_mean, log_likelihoods
    log_lik_per_tile = effective_guidance_scale * effective_likelihood(samples)  # (1, K, A)
    log_weights_b_n_a = x0_samples.log_weights.unsqueeze(-1).expand(batch_size, num_particles, tile_count)
    weights_b_n_a = torch.softmax(log_lik_per_tile + log_weights_b_n_a, dim=1)
    tile_h, tile_w = height // tile_grid_k, width // tile_grid_k
    samples_tiles = samples.reshape(batch_size, num_particles, channels, tile_grid_k, tile_h, tile_grid_k, tile_w)
    samples_tiles = samples_tiles.permute(0, 1, 3, 5, 2, 4, 6).contiguous().reshape(batch_size, num_particles, tile_count, channels, tile_h, tile_w)
    mixed_tiles = (weights_b_n_a.reshape(batch_size, num_particles, tile_count, 1, 1, 1) * samples_tiles).sum(dim=1)
    shared_mean = mixed_tiles.reshape(batch_size, tile_grid_k, tile_grid_k, channels, tile_h, tile_w).permute(0, 3, 1, 4, 2, 5).contiguous().reshape(batch_size, channels, height, width)
    return shared_mean, log_lik_per_tile.mean(dim=-1)


def run_single_image_pool_guided_sampling(
    pmf_one_step_model: PixelMeanFlowOneStepModel, observation_low_res, class_label,
    num_outer_steps, pool_subset_k, noise_std, guidance_scale,
    pool_noise_eta, pool_stop_at_t, pool_last_hop, device,
    intermediate_results_dir=None, image_index=0, save_intermediate_images=True,
    direction_save_dir=None, num_random_candidates_per_step: int = 0,
    save_spread_map: bool = False, spread_metric: str = "std", spread_scale: float = 1.0,
    tile_count: int = 1, pool_estimator: str = "reinforce", task: str = "superresolution",
):
    # Virtual-pool sampler with effectively infinite population. Maintains the running
    # marginal (mean, variance) of the pool, samples K=pool_subset_k fresh particles
    # from it each step, runs MF on those, and computes a shared direction.
    #
    # pool_estimator="reinforce": softmax-weighted x0 across K candidates. Tiled target
    #   matches the sibling-reinforce path (per-tile softmax + stitched x0_hat).
    # pool_estimator="reparam":  Tweedie-corrected x0 using the gradient of
    #   logsumexp(log p(y|MF(m + sqrt(v)*eps_k, t))) w.r.t. m. No renoising chain, so
    #   the gradient is much lower-variance than the existing reparam mean fn.
    if pool_estimator not in ("reinforce", "reparam"):
        raise ValueError(f"unknown pool_estimator {pool_estimator!r}; expected 'reinforce' or 'reparam'")
    if task not in ("superresolution", "colorization"):
        raise ValueError(f"unknown task {task!r}; expected 'superresolution' or 'colorization'")
    use_tiled = int(tile_count) > 1
    if use_tiled:
        tile_grid_k = int(round(int(tile_count) ** 0.5))
        assert tile_grid_k * tile_grid_k == int(tile_count), f"tile_count={tile_count} must be a perfect square"
        if task == "superresolution":
            raw_likelihood = build_tiled_pixel_sr_log_likelihood(observation_low_res, noise_std, int(tile_count))
        else:
            raw_likelihood = build_tiled_pixel_colorization_log_likelihood(observation_low_res, noise_std, int(tile_count))
    else:
        if task == "superresolution":
            raw_likelihood = PixelSpaceSuperResolutionLikelihood(
                observation_low_res_m1to1=observation_low_res, noise_std=noise_std,
            )
        else:
            raw_likelihood = PixelSpaceColorizationLikelihood(
                observation_gray_m1to1=observation_low_res, noise_std=noise_std,
            )
    if guidance_scale is None:
        adaptive_cls = TiledAdaptiveGuidanceScaleLikelihood if use_tiled else AdaptiveGuidanceScaleLikelihood
        adaptive_likelihood = adaptive_cls(raw_likelihood)
        effective_likelihood = adaptive_likelihood
        effective_guidance_scale = 1.0
    else:
        adaptive_likelihood = None
        effective_likelihood = raw_likelihood
        effective_guidance_scale = float(guidance_scale)

    intermediate_plotter = build_intermediate_particle_image_plotter(
        intermediate_results_dir=intermediate_results_dir or "", image_index=image_index,
        save_images=(save_intermediate_images and intermediate_results_dir is not None),
        adaptive_likelihood=adaptive_likelihood,
        direction_save_dir=direction_save_dir,
        num_random_candidates=num_random_candidates_per_step,
        save_spread_map=save_spread_map, spread_metric=spread_metric, spread_scale=spread_scale,
    )

    img_size = pmf_one_step_model.img_size
    img_channels = pmf_one_step_model.img_channels
    noise_scale = pmf_one_step_model.noise_scale
    K = int(pool_subset_k)

    time_steps = build_time_steps_noise_to_clean(num_outer_steps).to(device)
    # Virtual-pool marginal: x_t ~ N(running_mean, running_variance * I), broadcast across particles.
    # The DDIM-style update x_{t+1} = a*x_t + b*direction + c*eps preserves Gaussianity given
    # i.i.d. per-particle eps, so we never materialize the pool: at each step we sample K fresh
    # particles from the current marginal, run MF/REINFORCE on those, then propagate the
    # marginal forward analytically. P is effectively infinite at O(K) memory cost.
    running_mean = torch.zeros((1, img_channels, img_size, img_size), device=device)
    running_variance = float(noise_scale) ** 2  # initial: x_0 = noise_scale * eps -> Var=noise_scale^2
    pool_labels = class_label.expand(K).contiguous()

    last_t_for_readout = float(time_steps[-1].item())

    for step_index in range(len(time_steps) - 1):
        t = float(time_steps[step_index].item())
        t_next = float(time_steps[step_index + 1].item())
        if pool_stop_at_t is not None and t < float(pool_stop_at_t):
            last_t_for_readout = t
            break

        if pool_estimator == "reinforce":
            with torch.no_grad():
                x_t_subset = torch.randn(K, img_channels, img_size, img_size, device=device).mul_(running_variance ** 0.5).add_(running_mean)
                x0_candidates = pmf_one_step_model.predict_x0_one_step(
                    x_t_subset, t, pool_labels,
                ).clamp(-1.0, 1.0)
                x0_samples = SampleOutput(samples=x0_candidates.unsqueeze(0))
                shared_mean, log_likelihoods_for_plot = _pool_shared_mean(
                    x0_samples=x0_samples, effective_likelihood=effective_likelihood,
                    effective_guidance_scale=effective_guidance_scale,
                    tile_count=int(tile_count), tile_grid_k=tile_grid_k if use_tiled else 1,
                )
        else:
            # pool_estimator == "reparam": Tweedie correction with gradient through MF
            # on K i.i.d. samples from N(m, v*I). No renoising chain, so the gradient is
            # one MF forward+backward per candidate (vs two with renoising). When tiled,
            # we sum logsumexp-per-tile over tiles -> per-region gradient signal.
            mean_grad = running_mean.detach().clone().requires_grad_(True)
            eps_K = torch.randn(K, img_channels, img_size, img_size, device=device)
            x_t_subset = mean_grad + (running_variance ** 0.5) * eps_K
            # No clamp on x0 — would zero gradients at saturation; matches build_reparam_mean_fn.
            x0_candidates = pmf_one_step_model.predict_x0_one_step(x_t_subset, t, pool_labels)
            log_likelihoods = effective_guidance_scale * effective_likelihood(x0_candidates.unsqueeze(0))
            # Shape: whole-image -> (1, K); tiled -> (1, K, A). Plotter wants (1, K).
            log_likelihoods_for_plot = log_likelihoods if log_likelihoods.dim() == 2 else log_likelihoods.mean(dim=-1)
            unguided_mean = x0_candidates.detach().mean(dim=0, keepdim=True)
            if t >= 1.0 - 1e-6:
                # At t≈1, a_t=1-t→0 so the Tweedie correction blows up. Skip the gradient
                # at the boundary and use the unguided pool mean (matches build_reparam_mean_fn).
                shared_mean = unguided_mean
            else:
                # For tiled (1, K, A): sum over tiles of logsumexp over K -> per-tile gradient.
                # For whole-image (1, K): single logsumexp over K.
                logp_y_x = torch.logsumexp(log_likelihoods, dim=1).sum()
                gradient = torch.autograd.grad(logp_y_x, mean_grad)[0]
                a_t = 1.0 - t
                b_t = t
                shared_mean = unguided_mean + (b_t * b_t / a_t) * gradient.detach()
            x0_samples = SampleOutput(samples=x0_candidates.detach().unsqueeze(0))
            log_likelihoods_for_plot = log_likelihoods_for_plot.detach()
            x_t_subset = x_t_subset.detach()
            shared_mean = shared_mean.detach()

        intermediate_plotter(
            x_t_subset[:1], x0_samples, log_likelihoods_for_plot, t,
            direction_to_x0=shared_mean,
        )

        # DDIM-style step (same formula as the explicit pool); propagate the marginal.
        eta = float(pool_noise_eta) if (pool_noise_eta > 0.0 and step_index < len(time_steps) - 2) else 0.0
        a = (t_next / t) * ((1.0 - eta * eta) ** 0.5)
        b = (1.0 - t_next) - a * (1.0 - t)
        c = t_next * noise_scale * eta
        running_mean = a * running_mean + b * shared_mean
        running_variance = a * a * running_variance + c * c

    if pool_last_hop:
        with torch.no_grad():
            x_t_subset = torch.randn(K, img_channels, img_size, img_size, device=device).mul_(running_variance ** 0.5).add_(running_mean)
            x0_candidates = pmf_one_step_model.predict_x0_one_step(
                x_t_subset, last_t_for_readout, pool_labels,
            ).clamp(-1.0, 1.0)
            x0_samples = SampleOutput(samples=x0_candidates.unsqueeze(0))
            result, _ = _pool_shared_mean(
                x0_samples=x0_samples, effective_likelihood=effective_likelihood,
                effective_guidance_scale=effective_guidance_scale,
                tile_count=int(tile_count), tile_grid_k=tile_grid_k if use_tiled else 1,
            )
    else:
        # P -> infinity pool average converges to the marginal mean (variance term vanishes).
        result = running_mean
    return result.clamp(-1.0, 1.0)


def run_sibling_reinforce_pixel_mean_flow_superresolution_experiment(
    checkpoint: str = DEFAULT_PMF_B16_CHECKPOINT, model_name: str = "pmfDiT_B_16",
    test_images_dir: str = DEFAULT_TEST_IMAGES_DIR, output_dir: str = DEFAULT_OUTPUT_DIR,
    num_outer_steps: int = 32, num_particles: int = 256, noise_std: float = 0.05,
    guidance_scale=None, sibling_renoise_t=None, unconditional: bool = False, estimator: str = "reinforce", tile_count: int = 64 * 64,
    cfg_omega: float = 7.5, interval_min: float = 0.1, interval_max: float = 0.8,
    seed: int = 1, device: str = "cuda", num_gpus=None, save_intermediate_particles: bool = True,
    wandb_project: str = "pmf_pixel_superresolution",
    pool_subset_k: int = 0, pool_noise_eta: float = 0.5,
    pool_stop_at_t=None, pool_last_hop: bool = False,
    pool_estimator: str = "reinforce",
    save_per_image_summary: bool = False, num_random_candidates_per_step: int = 0,
    save_spread_map: bool = False, spread_metric: str = "std", spread_scale: float = 1.0,
    task: str = "superresolution",
):
    os.makedirs(output_dir, exist_ok=True)
    intermediate_results_dir = os.path.join(output_dir, "intermediate_results")
    if save_intermediate_particles:
        os.makedirs(intermediate_results_dir, exist_ok=True)
    summary_dir = os.path.join(output_dir, "summary") if save_per_image_summary else None
    if summary_dir is not None:
        os.makedirs(summary_dir, exist_ok=True)
    needs_step_files = (
        save_per_image_summary or save_spread_map or num_random_candidates_per_step > 0
    )
    direction_save_dir = os.path.join(output_dir, "direction_to_x0") if needs_step_files else None
    if direction_save_dir is not None:
        os.makedirs(direction_save_dir, exist_ok=True)

    if unconditional:
        cfg_omega, interval_min, interval_max = 1.0, 0.0, 1.0

    wandb.init(
        project=wandb_project,
        config={
            "num_outer_steps": num_outer_steps, "num_particles": num_particles, "noise_std": noise_std,
            "guidance_scale": guidance_scale, "sibling_renoise_t": sibling_renoise_t, "unconditional": unconditional,
            "estimator": estimator, "tile_count": tile_count,
            "cfg_omega": cfg_omega, "interval_min": interval_min, "interval_max": interval_max,
            "model_name": model_name, "checkpoint": os.path.basename(checkpoint), "seed": seed,
            "pool_subset_k": pool_subset_k, "pool_noise_eta": pool_noise_eta,
            "pool_stop_at_t": pool_stop_at_t, "pool_last_hop": pool_last_hop,
            "pool_estimator": pool_estimator,
        },
        reinit=True, settings=wandb.Settings(start_method="fork"),
    )

    requested_device = torch.device(device)
    cuda_device_count = torch.cuda.device_count() if requested_device.type == "cuda" else 0
    if num_gpus is None:
        effective_num_gpus = max(1, cuda_device_count)
    else:
        effective_num_gpus = int(num_gpus)
        if cuda_device_count > 0 and effective_num_gpus > cuda_device_count:
            raise ValueError(f"requested num_gpus={effective_num_gpus} but only {cuda_device_count} CUDA devices are visible")
    if cuda_device_count == 0:
        replica_devices = [requested_device]
    else:
        replica_devices = [torch.device(f"cuda:{gpu_index}") for gpu_index in range(effective_num_gpus)]
    torch_device = replica_devices[0]
    print(f"using {len(replica_devices)} device(s) for proposal generation: {[str(d) for d in replica_devices]}")

    state_dict = torch.load(checkpoint, map_location="cpu")
    replicas = []
    for replica_device in replica_devices:
        module = build_pmf_module(model_name)
        module.load_state_dict(state_dict, strict=False)
        module = module.to(replica_device).eval()
        replicas.append(PixelMeanFlowOneStepModel(
            pmf_module=module, cfg_omega=cfg_omega,
            interval_min=interval_min, interval_max=interval_max,
        ))
    del state_dict

    torch.manual_seed(seed)
    np.random.seed(seed)

    pmf_one_step_model = (
        replicas[0] if len(replicas) == 1
        else MultiDevicePixelMeanFlowOneStepModel(replicas=replicas, devices=replica_devices)
    )

    ground_truth_pixels, class_labels = load_imagenet_test_images_and_class_labels(directory=test_images_dir, device=torch_device)
    if unconditional:
        class_labels = torch.full_like(class_labels, pmf_one_step_model.num_classes)
    if task == "superresolution":
        low_resolution_observations = downsample_pixels_by_bicubic_4x(ground_truth_pixels)
    elif task == "colorization":
        from pixel_space_inverse_problems.pixel_colorization_likelihood import rgb_to_grayscale_m1to1
        low_resolution_observations = rgb_to_grayscale_m1to1(ground_truth_pixels)
    else:
        raise ValueError(f"unknown task {task!r}; expected 'superresolution' or 'colorization'")

    for image_index in range(ground_truth_pixels.shape[0]):
        single_ground_truth = ground_truth_pixels[image_index : image_index + 1]
        single_observation = low_resolution_observations[image_index : image_index + 1]
        single_label = class_labels[image_index : image_index + 1]

        if pool_subset_k and int(pool_subset_k) > 0:
            guided_pixels = run_single_image_pool_guided_sampling(
                pmf_one_step_model=pmf_one_step_model, observation_low_res=single_observation, class_label=single_label,
                num_outer_steps=num_outer_steps, pool_subset_k=int(pool_subset_k),
                noise_std=noise_std, guidance_scale=guidance_scale,
                pool_noise_eta=float(pool_noise_eta), pool_stop_at_t=pool_stop_at_t, pool_last_hop=bool(pool_last_hop),
                device=torch_device, intermediate_results_dir=intermediate_results_dir, image_index=image_index,
                save_intermediate_images=save_intermediate_particles,
                direction_save_dir=direction_save_dir,
                num_random_candidates_per_step=int(num_random_candidates_per_step),
                save_spread_map=bool(save_spread_map), spread_metric=spread_metric, spread_scale=float(spread_scale),
                tile_count=int(tile_count), pool_estimator=str(pool_estimator), task=task,
            )
        else:
            guided_pixels = run_single_image_reinforce_guided_sampling(
                pmf_one_step_model=pmf_one_step_model, observation_low_res=single_observation, class_label=single_label,
                num_outer_steps=num_outer_steps, num_particles=num_particles, noise_std=noise_std,
                guidance_scale=guidance_scale, sibling_renoise_t=sibling_renoise_t, device=torch_device, estimator=estimator, tile_count=tile_count,
                intermediate_results_dir=intermediate_results_dir, image_index=image_index,
                save_intermediate_images=save_intermediate_particles,
                direction_save_dir=direction_save_dir,
                num_random_candidates_per_step=int(num_random_candidates_per_step),
                save_spread_map=bool(save_spread_map), spread_metric=spread_metric, spread_scale=float(spread_scale),
                task=task,
            )

        save_pixel_image_minus1_to_1(single_ground_truth[0], os.path.join(output_dir, f"{image_index:03d}_ground_truth.png"))
        save_pixel_image_minus1_to_1(single_observation[0], os.path.join(output_dir, f"{image_index:03d}_observation_64.png"))
        save_pixel_image_minus1_to_1(guided_pixels[0], os.path.join(output_dir, f"{image_index:03d}_guided.png"))

        if save_per_image_summary and direction_save_dir is not None:
            from pixel_space_inverse_problems.per_image_summary_plot import build_per_image_summary
            t_values = build_time_steps_noise_to_clean(num_outer_steps).tolist()[:num_outer_steps]
            per_image_dir = os.path.join(direction_save_dir, f"{image_index:04d}")
            summary_path = os.path.join(summary_dir, f"{image_index:03d}_summary.png")
            build_per_image_summary(
                gt_m1to1_chw=single_ground_truth[0], lr_m1to1_chw=single_observation[0],
                per_image_dir=per_image_dir, output_path=summary_path,
                num_outer_steps=num_outer_steps, num_candidates=int(num_random_candidates_per_step),
                t_values=t_values, spread_metric=spread_metric,
            )
            wandb.log({"results/summary": wandb.Image(summary_path, caption=f"image {image_index}"),
                       "image_index": image_index})
            import shutil
            shutil.rmtree(per_image_dir, ignore_errors=True)

        wandb.log({
            "results/ground_truth": to_wandb_image(single_ground_truth[0], caption=f"image {image_index}"),
            "results/observation_64": to_wandb_image(single_observation[0], caption=f"image {image_index}"),
            "results/guided": to_wandb_image(guided_pixels[0], caption=f"image {image_index}"),
            "image_index": image_index,
        })
        print(f"saved outputs for image {image_index}")

    if direction_save_dir is not None:
        try:
            os.rmdir(direction_save_dir)
        except OSError:
            pass

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(run_sibling_reinforce_pixel_mean_flow_superresolution_experiment)
