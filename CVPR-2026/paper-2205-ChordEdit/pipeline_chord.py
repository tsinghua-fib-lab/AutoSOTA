from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

# Optimization: enable TF32 on Ampere+ GPUs for ~20% faster matmuls
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
from PIL import Image, ImageOps
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.utils import BaseOutput
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPTextModel

DEFAULT_SEED = 42
DEFAULT_COMPUTE_DTYPE = torch.float32
DEFAULT_SAFETY_CHECKER_ID = "CompVis/stable-diffusion-safety-checker"

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline output container
# ---------------------------------------------------------------------------


@dataclass
class ChordEditPipelineOutput(BaseOutput):
    images: List[Image.Image] | torch.Tensor
    latents: torch.Tensor


class _CenterSquareCropTransform:
    """Center-crop the shorter image dimension before resizing."""

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width == height:
            return image
        target = min(width, height)
        try:
            resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover
            resample = Image.LANCZOS
        return ImageOps.fit(
            image,
            (target, target),
            method=resample,
            centering=(0.5, 0.5),
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# ChordEdit Pipeline
# ---------------------------------------------------------------------------


class ChordEditPipeline(DiffusionPipeline):
    """Standalone pipeline that wires up diffusers modules with the Chord editor."""

    def __init__(
        self,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        vae: AutoencoderKL,
        tokenizer,
        text_encoder: CLIPTextModel,
        default_edit_config: Optional[Dict[str, Any]] = None,
        image_size: int = 512,
        device: Optional[str | torch.device] = None,
        compute_dtype: torch.dtype = DEFAULT_COMPUTE_DTYPE,
        use_attention_mask: bool = False,
        use_center_crop: bool = True,
        use_safety_checker: bool = False,
        safety_checker_id: Optional[str] = DEFAULT_SAFETY_CHECKER_ID,
    ) -> None:
        super().__init__()
        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )
        self._device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._compute_dtype = compute_dtype
        self._use_attention_mask = bool(use_attention_mask)
        self.to(self._device)
        self._set_compute_precision()
        # Enable PyTorch SDPA (FlashAttention-2/Mem-Efficient backend) for UNet attention
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0
            self.unet.set_attn_processor(AttnProcessor2_0())
            LOGGER.info('UNet attention: SDPA (AttnProcessor2_0) enabled.')
        except Exception as exc:
            LOGGER.warning('SDPA attention setup failed, using default: %s', exc)

        self.default_edit_config = default_edit_config or {}
        self.image_size = int(image_size)
        self._use_center_crop = bool(use_center_crop)
        self._vae_transform = self._build_vae_transform()
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()
        self._max_unet_timestep = self.scheduler.config.num_train_timesteps - 1
        self._vae_scale = float(getattr(self.vae.config, "scaling_factor", 1.0))
        # Pre-cache alphas_cumprod as float32 for faster _get_alpha_sigma
        self._alphas_cumprod_f32 = self.scheduler.alphas_cumprod.to(dtype=torch.float32)
        self._use_safety_checker = bool(use_safety_checker)
        self._safety_checker_id = safety_checker_id
        self._safety_checker: Optional[StableDiffusionSafetyChecker] = None
        self._safety_feature_extractor: Optional[CLIPImageProcessor] = None
        if self._use_safety_checker:
            self._init_safety_checker()

    def _set_compute_precision(self) -> None:
        modules = (self.unet, self.vae, self.text_encoder)
        for module in modules:
            if module is not None:
                module.to(device=self._device, dtype=self._compute_dtype)
    def _init_safety_checker(self) -> None:
        if not self._safety_checker_id:
            LOGGER.warning("Safety checker requested but no identifier provided; disabling safety checks.")
            self._use_safety_checker = False
            return
        try:
            self._safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                self._safety_checker_id,
                torch_dtype=self._compute_dtype,
            ).to(self._device)
            self._safety_feature_extractor = CLIPImageProcessor.from_pretrained(self._safety_checker_id)
        except Exception as exc:  # pragma: no cover - runtime dependency
            LOGGER.warning("Failed to initialize safety checker (%s). Safety checks disabled.", exc)
            self._safety_checker = None
            self._safety_feature_extractor = None
            self._use_safety_checker = False

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_local_weights(
        cls,
        component_paths: Dict[str, str],
        *,
        default_edit_config: Optional[Dict[str, Any]] = None,
        device: Optional[str | torch.device] = None,
        torch_dtype: torch.dtype = torch.float32,
        image_size: int = 512,
        use_center_crop: bool = True,
        compute_dtype: torch.dtype = DEFAULT_COMPUTE_DTYPE,
        use_attention_mask: bool = False,
        use_safety_checker: bool = False,
        safety_checker_id: Optional[str] = DEFAULT_SAFETY_CHECKER_ID,
    ) -> "ChordEditPipeline":
        """Instantiate the pipeline from individual component checkpoints."""

        unet = UNet2DConditionModel.from_pretrained(
            component_paths["unet_path"],
            torch_dtype=torch_dtype,
        )
        scheduler = DDPMScheduler.from_pretrained(component_paths["scheduler_path"])
        vae = AutoencoderKL.from_pretrained(component_paths["vae_path"], torch_dtype=torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(component_paths["tokenizer_path"])
        text_encoder = CLIPTextModel.from_pretrained(
            component_paths["text_encoder_path"],
            torch_dtype=torch_dtype,
        )
        return cls(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            default_edit_config=default_edit_config,
            image_size=image_size,
            device=device,
            compute_dtype=compute_dtype,
            use_attention_mask=use_attention_mask,
            use_center_crop=use_center_crop,
            use_safety_checker=use_safety_checker,
            safety_checker_id=safety_checker_id,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def __call__(
        self,
        image: Image.Image | torch.Tensor,
        *,
        source_prompt: str,
        target_prompt: str,
        edit_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        output_type: str = "pil",
    ) -> ChordEditPipelineOutput:
        """Run ChordEdit once on a single image."""

        cfg = dict(self.default_edit_config)
        if edit_config:
            cfg.update(edit_config)
        required_keys = ["noise_samples", "n_steps", "t_start", "t_end", "t_delta", "step_scale"]
        missing = [k for k in required_keys if k not in cfg]
        if missing:
            raise ValueError(f"edit_config is missing required keys: {missing}")

        pixel_values = self._prepare_image_tensor(image)
        latents = self._encode_image_to_latent(pixel_values)
        src_embed = self.encode_prompt([source_prompt])
        tgt_embed = self.encode_prompt([target_prompt])

        # IDEA-014: Auto-tune step_scale based on prompt embedding similarity
        sim_scale = float(cfg.get("sim_scale", 0.5))
        with torch.no_grad():
            src_norm = src_embed / (src_embed.norm(dim=-1, keepdim=True) + 1e-8)
            tgt_norm = tgt_embed / (tgt_embed.norm(dim=-1, keepdim=True) + 1e-8)
            cos_sim = (src_norm * tgt_norm).sum(dim=-1).mean().item()
            # For similar prompts (high sim), reduce step_scale; for different, increase
            # adjusted = base * (1.0 + sim_scale * (0.5 - cos_sim))
            adjusted_factor = 1.0 + sim_scale * (0.5 - cos_sim)
            adjusted_factor = max(0.4, min(2.0, adjusted_factor))
            cfg["step_scale"] = float(cfg["step_scale"]) * adjusted_factor

        output_latents: List[torch.Tensor] = []
        decoded_batches: List[torch.Tensor] = []

        edit_params = self._prepare_edit_params(cfg)
        seed_value = int(seed) if seed is not None else DEFAULT_SEED

        noise_list = self._prepare_noise_list(
            latents=latents,
            seed_value=seed_value,
            num_noises=edit_params["noise_samples"],
        )

        x0_pred = self._run_edit(
            x_src=latents,
            src_embed=src_embed,
            edit_embed=tgt_embed,
            noise=noise_list,
            params=edit_params,
        )

        decoded = self._decode_latent_to_image(x0_pred)
        decoded, _ = self._apply_safety_checker(decoded)
        output_latents.append(x0_pred.detach().cpu())
        decoded_batches.append(decoded.detach().cpu())

        images_tensor = torch.cat(decoded_batches, dim=0)
        latents_tensor = torch.cat(output_latents, dim=0)
        images = self._tensor_to_pil(images_tensor) if output_type == "pil" else images_tensor

        return ChordEditPipelineOutput(
            images=images,
            latents=latents_tensor,
        )

    def encode_prompt(self, prompts: Sequence[str]) -> torch.Tensor:
        """Public helper mirroring diffusers pipelines for text encoding."""
        return self._encode_text(prompts)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _prepare_image_tensor(self, image: Image.Image | torch.Tensor) -> torch.Tensor:
        if isinstance(image, Image.Image):
            vae_tensor = self._vae_transform(image)
        elif torch.is_tensor(image):
            tensor = image.float()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            tensor = tensor * 2.0 - 1.0
            vae_tensor = tensor
        else:
            raise TypeError("image must be a PIL.Image or a torch.Tensor.")

        if vae_tensor.ndim == 3:
            vae_tensor = vae_tensor.unsqueeze(0)
        if self._use_center_crop and vae_tensor.ndim == 4:
            _, _, height, width = vae_tensor.shape
            if height != width:
                side = min(height, width)
                top = (height - side) // 2
                left = (width - side) // 2
                vae_tensor = vae_tensor[:, :, top : top + side, left : left + side]
        return vae_tensor.to(device=self._device, dtype=self._compute_dtype)

    def _encode_image_to_latent(self, pixel_values: torch.Tensor) -> torch.Tensor:
        scaling_factor = self._vae_scale
        pixel_values = pixel_values.to(device=self._device, dtype=self._compute_dtype)
        latents = self.vae.encode(pixel_values).latent_dist.mode()
        latents = latents * scaling_factor
        return latents.to(device=self._device, dtype=self._compute_dtype)

    def _decode_latent_to_image(self, latents: torch.Tensor) -> torch.Tensor:
        scaling_factor = self._vae_scale
        latents = latents.to(device=self._device, dtype=self._compute_dtype)
        decoded = self.vae.decode(latents / scaling_factor).sample
        decoded = (decoded.clamp(-1.0, 1.0) + 1.0) / 2.0
        return decoded.to(dtype=self._compute_dtype)

    def _apply_safety_checker(self, images: torch.Tensor) -> tuple[torch.Tensor, List[bool]]:
        batch = images.shape[0]
        if (
            not self._use_safety_checker
            or self._safety_checker is None
            or self._safety_feature_extractor is None
            or batch == 0
        ):
            return images, [False] * batch

        images_clamped = images.detach().clamp(0.0, 1.0)
        pil_images = self._tensor_to_pil(images_clamped)
        try:
            clip_input = self._safety_feature_extractor(images=pil_images, return_tensors="pt").to(self._device)
            images_np = np.stack([np.array(img).astype(np.float32) / 255.0 for img in pil_images], axis=0)
            images_np = images_np * 2.0 - 1.0
            _, has_nsfw_concept = self._safety_checker(
                images=images_np,
                clip_input=clip_input.pixel_values.to(self._device),
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.warning("Safety checker failed (%s). Skipping safety checks.", exc)
            return images, [False] * batch

        if isinstance(has_nsfw_concept, torch.Tensor):
            has_nsfw = has_nsfw_concept.detach().cpu().to(dtype=torch.bool).tolist()
        else:
            has_nsfw = [bool(flag) for flag in has_nsfw_concept]

        if any(has_nsfw):
            for idx, flagged in enumerate(has_nsfw):
                if flagged:
                    images[idx] = torch.zeros_like(images[idx])
        return images, has_nsfw

    def _encode_text(self, prompts: Sequence[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            list(prompts),
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(self._device)
        attn_mask = inputs.attention_mask.to(self._device) if self._use_attention_mask else None
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs[0]
        return hidden.to(device=self._device, dtype=self._compute_dtype)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> List[Image.Image]:
        tensor = tensor.detach().cpu().clamp(0.0, 1.0)
        to_pil = transforms.ToPILImage()
        return [to_pil(sample) for sample in tensor]

    def _build_vae_transform(self) -> transforms.Compose:
        """Create image->latent preprocessing transform."""
        ops: List[Any] = []
        if self._use_center_crop:
            ops.append(_CenterSquareCropTransform())
            resize_interp = InterpolationMode.LANCZOS
        else:
            resize_interp = InterpolationMode.BILINEAR
        ops.append(
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=resize_interp,
            )
        )
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        return transforms.Compose(ops)

    def _prepare_edit_params(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(cfg)
        params["noise_samples"] = int(max(1, params["noise_samples"]))
        params["n_steps"] = int(max(1, params["n_steps"]))
        params["t_start"] = float(max(0.0, min(1.0, params["t_start"])))
        params["t_end"] = float(max(0.0, min(params["t_start"], params["t_end"])))
        t_delta = float(max(0.0, min(1.0, params["t_delta"])))
        if t_delta >= params["t_start"]:
            safe_max = max(1, self._max_unet_timestep)
            t_delta = max(0.0, params["t_start"] - 1.0 / safe_max)
        params["t_delta"] = t_delta
        params["step_scale"] = float(params["step_scale"])
        params["cleanup"] = bool(params.get("cleanup", False))
        return params

    def _prepare_noise_list(
        self,
        latents: torch.Tensor,
        seed_value: int,
        num_noises: int,
    ) -> List[torch.Tensor]:
        gen = torch.Generator(device=latents.device).manual_seed(seed_value)
        noise_list = [
            torch.randn(latents.shape, device=latents.device, dtype=self._compute_dtype, generator=gen)
            for _ in range(num_noises)
        ]
        return noise_list

    def _time_to_index(self, batch: int, t_scalar: float, device, dtype=torch.long):
        idx = round(self._max_unet_timestep * float(t_scalar))
        idx = max(0, min(self._max_unet_timestep, idx))
        return torch.full((batch,), idx, device=device, dtype=dtype)

    def _get_alpha_sigma(self, tensor: torch.Tensor, timesteps: torch.Tensor):
        alphas_cumprod = self._alphas_cumprod_f32.to(device=tensor.device)
        alpha_t = alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)
        sigma_t = (1 - alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
        alpha_t = alpha_t.to(dtype=tensor.dtype, device=tensor.device)
        sigma_t = sigma_t.to(dtype=tensor.dtype, device=tensor.device)
        eps = torch.finfo(alpha_t.dtype).eps
        alpha_t = alpha_t.clamp(min=eps)
        return alpha_t, sigma_t

    def _pred_x0(self, x_anchor, timesteps, cond, noise):
        alpha_t, sigma_t = self._get_alpha_sigma(x_anchor, timesteps)
        z_t = alpha_t * x_anchor + sigma_t * noise
        noise_pred = self.unet(
            sample=z_t,
            timestep=timesteps,
            encoder_hidden_states=cond,
            return_dict=False,
        )[0]
        x0_pred = (z_t - sigma_t * noise_pred) / alpha_t
        return x0_pred

    def _u_estimate(self, x_anchor, src_embed, edit_embed, noise, t_s: float, delta: float):
        batch, device = x_anchor.shape[0], x_anchor.device
        t_idx_s = self._time_to_index(batch, t_s, device=device)
        t_idx_s0 = self._time_to_index(batch, max(0.0, t_s - delta), device=device)

        noises = noise if isinstance(noise, (list, tuple)) else [noise]

        alpha_s, sigma_s = self._get_alpha_sigma(x_anchor, t_idx_s)
        alpha_prev, sigma_prev = self._get_alpha_sigma(x_anchor, t_idx_s0)

        num_noises = len(noises)
        noise_stack = torch.stack(noises, dim=0)

        x_anchor_b = x_anchor.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)
        alpha_s_b = alpha_s.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)
        alpha_prev_b = alpha_prev.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)
        sigma_s_b = sigma_s.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)
        sigma_prev_b = sigma_prev.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)

        z_s = alpha_s_b * x_anchor_b + sigma_s_b * noise_stack
        z_prev = alpha_prev_b * x_anchor_b + sigma_prev_b * noise_stack

        samples = torch.stack([z_s, z_s, z_prev, z_prev], dim=1)
        samples = samples.reshape(num_noises * 4 * batch, *x_anchor.shape[1:])

        conds = torch.cat([src_embed, edit_embed, src_embed, edit_embed], dim=0)
        repeat_dims = [num_noises] + [1] * (conds.dim() - 1)
        conds = conds.repeat(*repeat_dims)

        timesteps = torch.cat([t_idx_s, t_idx_s, t_idx_s0, t_idx_s0], dim=0)
        timesteps = timesteps.repeat(num_noises)

        alpha_cat = torch.stack(
            [alpha_s_b, alpha_s_b, alpha_prev_b, alpha_prev_b],
            dim=1,
        ).reshape(num_noises * 4 * batch, 1, 1, 1)
        sigma_cat = torch.stack(
            [sigma_s_b, sigma_s_b, sigma_prev_b, sigma_prev_b],
            dim=1,
        ).reshape(num_noises * 4 * batch, 1, 1, 1)

        noise_pred = self.unet(
            sample=samples,
            timestep=timesteps,
            encoder_hidden_states=conds,
            return_dict=False,
        )[0]

        x0_all = (samples - sigma_cat * noise_pred) / alpha_cat
        x0_all = x0_all.reshape(num_noises, 4, batch, *x_anchor.shape[1:])
        x_src_p_s, x_tar_p_s, x_src_p_s0, x_tar_p_s0 = x0_all.unbind(dim=1)

        dv_s = (x_tar_p_s - x_src_p_s).sum(dim=0) / float(num_noises)
        dv_s0 = (x_tar_p_s0 - x_src_p_s0).sum(dim=0) / float(num_noises)

        denom = (t_s + delta)
        if denom <= 1e-6:
            return dv_s
        return (delta * dv_s + t_s * dv_s0) / denom

    def _run_edit(
        self,
        x_src: torch.Tensor,
        src_embed: torch.Tensor,
        edit_embed: torch.Tensor,
        noise: List[torch.Tensor],
        params: Dict[str, Any],
    ) -> torch.Tensor:
        device = x_src.device
        if params["n_steps"] == 1:
            t_grid = [params["t_start"]]
        else:
            t_grid = torch.linspace(
                params["t_start"],
                params["t_end"],
                steps=params["n_steps"],
                device=device,
            ).tolist()

        x_curr = x_src
        for t_s in t_grid:
            u_hat = self._u_estimate(
                x_curr,
                src_embed,
                edit_embed,
                noise,
                float(t_s),
                params["t_delta"],
            )
            x_curr = x_curr + params["step_scale"] * u_hat

        if params["cleanup"]:
            t_end_idx = self._time_to_index(x_src.shape[0], params["t_end"], device=device)
            x_cleanup = self._pred_x0(x_curr, t_end_idx, edit_embed, noise[0])
            alpha = float(params.get("cleanup_alpha", 0.7))
            x_curr = alpha * x_cleanup + (1.0 - alpha) * x_curr

        return x_curr
