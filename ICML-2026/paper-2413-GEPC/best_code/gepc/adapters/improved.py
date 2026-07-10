# gepc/adapters/improved.py
# -*- coding: utf-8 -*-
"""
Adapter for OpenAI 'improved-diffusion' checkpoints.

Goals:
- Locate the improved-diffusion repo (IMPROVED_DIFFUSION_DIR or common local paths) and add it to PYTHONPATH.
- Build (model, diffusion) with the exact flags for official OpenAI checkpoints when detected by filename.
- Auto-detect learn_sigma from checkpoint output channels (3 vs 6).
- Expose:
    - model (UNet)
    - diffusion (GaussianDiffusion)
    - alphas_cumprod, sqrt_ab, sqrt_one_minus_ab
    - sigma_t(t) consistent with GEPC (sqrt(1 - alpha_bar_t))
"""
import os
import sys
from typing import Any, Dict

import numpy as np
import torch

# ------------------------------------------------------------------
# Locate improved-diffusion and prepend to sys.path
# ------------------------------------------------------------------
_here = os.path.abspath(os.path.dirname(__file__))
_candidates = [
    os.environ.get("IMPROVED_DIFFUSION_DIR"),
    os.path.abspath(os.path.join(_here, "../../..", "repos", "improved-diffusion")),
    os.path.abspath(os.path.join(_here, "../../..", "improved-diffusion")),
    os.path.abspath(os.path.join(_here, "..", "..", "repos", "improved-diffusion")),
]
for c in _candidates:
    if c and os.path.isdir(c) and c not in sys.path:
        sys.path.insert(0, c)

from improved_diffusion import dist_util
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _parse_mult_list(s):
    if s is None:
        return [1, 2, 3, 4]
    if isinstance(s, (list, tuple)):
        return [int(x) for x in s]
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _detect_learn_sigma(sd: Dict[str, torch.Tensor]) -> bool | None:
    """Infer learn_sigma from output conv channels (3 vs 6)."""
    for k, v in sd.items():
        if k.endswith("out.2.weight") and isinstance(v, torch.Tensor) and v.dim() == 4:
            return v.shape[0] == 6
    for k, v in sd.items():
        if "out_layers" in k and k.endswith(".weight") and isinstance(v, torch.Tensor) and v.dim() == 4:
            if v.shape[0] in (3, 6):
                return v.shape[0] == 6
    return None


def _validate_width(eff: Dict[str, Any]) -> None:
    """Ensure UNet channel widths are divisible by 32 (GroupNorm32)."""
    base = int(eff.get("num_channels", 128))
    mults = _parse_mult_list(eff.get("channel_mult", "1,2,3,4"))
    bad = [base * m for m in mults if (base * m) % 32 != 0]
    if bad:
        raise ValueError(
            f"[ImprovedAdapter] Invalid UNet width: num_channels={base} with channel_mult={mults} "
            f"produces channels {bad} not divisible by 32."
        )


# ------------------------------------------------------------------
# Adapter
# ------------------------------------------------------------------

class ImprovedDiffusionAdapter:
    """
    Args expected in `args`:
      - model_path: checkpoint path
      - image_size: backbone size
      - improved_args: optional overrides (num_channels, num_res_blocks, channel_mult, etc.)
      - device: GPU id (used by dist_util.setup_dist fallback)
    """

    def __init__(self, args):
        self.args = args

        # setup distributed + device
        try:
            dist_util.setup_dist()
        except TypeError:
            try:
                dev_id = getattr(args, "device", 0)
                dist_util.setup_dist(dev_id)
            except TypeError:
                dist_util.setup_dist()

        self.device = dist_util.dev()

        # load checkpoint
        ckpt_path = str(getattr(self.args, "model_path", ""))
        if not ckpt_path:
            raise ValueError("[ImprovedAdapter] args.model_path must be set.")

        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
        else:
            state = dist_util.load_state_dict(ckpt_path, map_location="cpu")

        state_dict = state["model"] if (isinstance(state, dict) and "model" in state) else state
        forced_learn_sigma = _detect_learn_sigma(state_dict)

        # build effective config
        cfg_defaults = model_and_diffusion_defaults()

        user: Dict[str, Any] = {}
        if hasattr(self.args, "improved_args") and self.args.improved_args:
            user = dict(self.args.improved_args)

        if "model_channels" in user and "num_channels" not in user:
            user["num_channels"] = user.pop("model_channels")

        ckpt_name = os.path.basename(ckpt_path).lower()
        preset: Dict[str, Any] = {}

        # official presets by filename
        if "cifar10_uncond_50m_500k" in ckpt_name or (
            "cifar10" in ckpt_name and "uncond" in ckpt_name and "50m_500k" in ckpt_name
        ):
            preset = {
                "image_size": 32,
                "num_channels": 128,
                "num_res_blocks": 3,
                "learn_sigma": True,
                "dropout": 0.3,
                "diffusion_steps": 4000,
                "noise_schedule": "cosine",
            }
            print("[ImprovedAdapter] Detected official CIFAR-10 uncond checkpoint; using OpenAI flags.")

        elif "imagenet64_uncond_100m_1500k" in ckpt_name or (
            "imagenet64" in ckpt_name and "uncond" in ckpt_name
        ):
            preset = {
                "image_size": 64,
                "num_channels": 128,
                "num_res_blocks": 3,
                "learn_sigma": True,
                "diffusion_steps": 4000,
                "noise_schedule": "cosine",
            }
            print("[ImprovedAdapter] Detected official ImageNet-64 uncond checkpoint; using OpenAI flags.")

        elif "lsun_uncond_100m_2400k" in ckpt_name or "lsun" in ckpt_name:
            preset = {
                "image_size": 256,
                "num_channels": 128,
                "num_res_blocks": 2,
                "num_heads": 1,
                "learn_sigma": True,
                "use_scale_shift_norm": False,
                "attention_resolutions": "16",
                "diffusion_steps": 1000,
                "noise_schedule": "linear",
                "rescale_learned_sigmas": False,
                "rescale_timesteps": False,
            }
            print("[ImprovedAdapter] Detected LSUN 256x256 uncond checkpoint; using OpenAI flags.")
        else:
            # fallback by image_size (kept for backwards compatibility)
            backbone_size = getattr(self.args, "image_size", None)
            if backbone_size is None:
                backbone_size = user.get("image_size", cfg_defaults["image_size"])
            backbone_size = int(backbone_size)

            if backbone_size == 32:
                preset = {
                    "image_size": 32,
                    "num_channels": 128,
                    "num_res_blocks": 3,
                    "learn_sigma": True,
                    "dropout": 0.3,
                    "diffusion_steps": 4000,
                    "noise_schedule": "cosine",
                }
            elif backbone_size == 64:
                preset = {
                    "image_size": 64,
                    "num_channels": 128,
                    "num_res_blocks": 3,
                    "learn_sigma": True,
                    "diffusion_steps": 4000,
                    "noise_schedule": "cosine",
                }
            elif backbone_size in (224, 256):
                preset = {
                    "image_size": backbone_size,
                    "num_channels": 128,
                    "num_res_blocks": 2,
                    "learn_sigma": True,
                    "diffusion_steps": 1000,
                    "noise_schedule": "linear",
                }

        eff: Dict[str, Any] = dict(cfg_defaults)
        eff.update(preset)
        eff.update(user)

        if getattr(self.args, "image_size", None) is not None:
            eff["image_size"] = int(self.args.image_size)

        # protect against num_channels confusion (width vs in_channels)
        if int(eff.get("num_channels", 128)) <= 4:
            nc = int(eff["num_channels"])
            if "in_channels" not in eff:
                eff["in_channels"] = nc
            eff["num_channels"] = int(preset.get("num_channels", 128) if preset else 128)
            print(
                f"[ImprovedAdapter][fix] Detected num_channels={nc} (likely image channels). "
                f"Using in_channels={eff['in_channels']} and width num_channels={eff['num_channels']}."
            )

        if forced_learn_sigma is not None:
            eff["learn_sigma"] = bool(forced_learn_sigma)

        _validate_width(eff)

        def _gv(k, d=None):
            return eff.get(k, d)

        print(
            "[ImprovedAdapter] Effective UNet config -> "
            f"image_size={_gv('image_size')}, width(num_channels)={_gv('num_channels')}, "
            f"num_res_blocks={_gv('num_res_blocks')}, channel_mult={_gv('channel_mult')}, "
            f"attn={_gv('attention_resolutions')}, heads={_gv('num_heads')}, "
            f"learn_sigma={_gv('learn_sigma')}, class_cond={_gv('class_cond')}"
        )

        # create model + diffusion
        cfg_obj = type("Config", (object,), eff)()
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(cfg_obj, model_and_diffusion_defaults().keys())
        )

        # load weights strictly (preserves exact behavior)
        r = model.load_state_dict(state_dict, strict=True)
        print(f"[ImprovedAdapter] state_dict loaded. missing={len(r.missing_keys)} unexpected={len(r.unexpected_keys)}")

        self.model = model.to(self.device).eval()
        self.diffusion = diffusion

        # cache schedule
        self.betas = np.asarray(self.diffusion.betas, dtype=np.float32)
        alphas = 1.0 - self.betas
        a_bar = np.cumprod(alphas, axis=0)
        self.alphas_cumprod = torch.from_numpy(a_bar).float().to(self.device)
        self.sqrt_ab = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_ab = torch.sqrt(1.0 - self.alphas_cumprod)

        raw_ut = getattr(self.diffusion, "use_timesteps", None)
        if raw_ut is None:
            self.use_timesteps = np.arange(len(self.betas), dtype=np.int64)
        else:
            self.use_timesteps = np.array(sorted(list(raw_ut)), dtype=np.int64)
        self.inner_steps = np.arange(len(self.betas), dtype=np.int64)

    def n_steps(self) -> int:
        return len(self.betas)

    def sigma_t(self, t_idx: int) -> torch.Tensor:
        """sigma_t = sqrt(1 - alpha_bar_t)."""
        return self.sqrt_one_minus_ab[int(t_idx)]
