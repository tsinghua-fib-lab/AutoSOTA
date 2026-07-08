"""
DAVE (DC Attenuation for diVersity Enhancement) for SANA pipeline.
Paper: "Breaking the Lock-in" (ICML 2026)

v2: Added per-block alpha scale_map support for multi-block profiles.
"""
import torch
from diffusers import SanaPipeline
from typing import List, Optional, Union, Dict


def apply_dave_dc(hidden_states: torch.Tensor, scale: float) -> torch.Tensor:
    """Apply DC attenuation: out = scale*DC + (x - DC)"""
    x = hidden_states.float()
    dc = x.mean(dim=1, keepdim=True)
    out = x + (scale - 1.0) * dc
    return out.to(dtype=hidden_states.dtype)


def gaussian_alpha_profile(
    target_blocks: List[int],
    center: int = 13,
    sigma: float = 2.0,
    alpha_max: float = 0.1,
) -> Dict[int, float]:
    """Compute per-block alpha values using a Gaussian profile.

    alpha_i = alpha_max * exp(-(i - center)^2 / (2 * sigma^2))

    Args:
        target_blocks: list of block indices to include
        center: center block (peak of Gaussian)
        sigma: standard deviation (narrower = fewer blocks affected)
        alpha_max: maximum alpha at the center

    Returns:
        dict mapping block_idx -> alpha value
    """
    import math
    scale_map = {}
    for bidx in target_blocks:
        weight = math.exp(-((bidx - center) ** 2) / (2 * sigma ** 2))
        scale_map[bidx] = alpha_max * weight
    return scale_map


def create_dave_sana_pipeline(
    model_path: str,
    target_blocks: Union[int, List[int]] = 13,
    dave_scale: float = 0.2,
    tau: float = 0.2,
    guidance_scale: float = 4.5,
    device: str = "cuda",
    **pipeline_kwargs,
):
    """
    Create SANA pipeline with DAVE diversity enhancement.

    Paper parameters for SANA1.5 (Table 8):
        target_blocks=[13] (L=13, fixed_block)
        dave_scale=0.2 (alpha)
        tau=0.2 (first 20% steps)
        guidance_scale=4.5 (omega_CFG)
    """
    if isinstance(target_blocks, int):
        target_blocks = [target_blocks]

    pipe = SanaPipeline.from_pretrained(model_path, **pipeline_kwargs)
    pipe = pipe.to(device)

    pipe._dave_config = {
        "target_blocks": target_blocks,
        "dave_scale": dave_scale,
        "tau": tau,
        "guidance_scale": guidance_scale,
    }

    _patch_transformer_blocks(pipe.transformer, target_blocks)
    _patch_pipeline_call(pipe)

    return pipe


def _patch_transformer_blocks(transformer, target_blocks):
    """Patch target blocks to support DAVE DC attenuation."""
    for bidx, block in enumerate(transformer.transformer_blocks):
        if bidx not in target_blocks:
            continue

        orig_fn = block.forward

        def make_patched(orig, idx):
            def patched_forward(
                self,
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                timestep=None,
                height=None,
                width=None,
            ):
                hidden_states = orig(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    height=height,
                    width=width,
                )
                if getattr(self, "_dave_active", False):
                    scale = getattr(self, "_dave_scale", 0.2)
                    if isinstance(scale, (list, tuple)):
                        scale = scale[idx] if idx < len(scale) else 1.0
                    hidden_states = apply_dave_dc(hidden_states, scale)
                return hidden_states
            return patched_forward

        block.forward = make_patched(orig_fn, bidx).__get__(block, type(block))
        block._dave_active = False
        block._dave_scale = 0.2


def _patch_pipeline_call(pipe):
    """Patch pipeline __call__ to control DAVE per denoising step.

    Uses callback_on_step_end, setting DAVE state for the NEXT step.
    DAVE is pre-enabled for step 0 if its in the active step list.

    Supports dave_scale_map (dict) for per-block alpha profiles.
    """
    original_call = pipe.__class__.__call__

    def patched_call(
        self,
        prompt=None,
        use_dave: bool = True,
        dave_scale: Optional[float] = None,
        dave_scale_map: Optional[Dict[int, float]] = None,
        tau: Optional[float] = None,
        **kwargs,
    ):
        cfg = getattr(self, "_dave_config", {})
        target_blocks = cfg.get("target_blocks", [13])
        default_scale = cfg.get("dave_scale", 0.2)
        default_tau = cfg.get("tau", 0.2)
        default_guidance = cfg.get("guidance_scale", 4.5)

        scale = dave_scale if dave_scale is not None else default_scale
        scale_map = dave_scale_map
        t = tau if tau is not None else default_tau
        if "guidance_scale" not in kwargs:
            kwargs["guidance_scale"] = default_guidance

        if not use_dave:
            _set_dave_state(self.transformer, target_blocks, False)
            return original_call(self, prompt=prompt, **kwargs)

        num_steps = kwargs.get("num_inference_steps", 20)
        dave_steps_count = int(num_steps * t)
        dave_steps = list(range(dave_steps_count))

        # Pre-enable DAVE for step 0
        _set_dave_state(self.transformer, target_blocks, 0 in dave_steps, scale, scale_map)

        # Store state for callback
        self._dave_active_steps = dave_steps
        self._dave_target_blocks = target_blocks
        self._dave_scale = scale
        self._dave_scale_map = scale_map

        original_cb = kwargs.pop("callback_on_step_end", None)

        def dave_callback(pipe, step_idx, timestep, callback_kwargs):
            # Set DAVE state for NEXT step
            next_step = step_idx + 1
            active = next_step in self._dave_active_steps
            _set_dave_state(pipe.transformer, target_blocks, active, scale, scale_map)
            if original_cb is not None:
                return original_cb(pipe, step_idx, timestep, callback_kwargs)
            return {}

        kwargs["callback_on_step_end"] = dave_callback
        result = original_call(self, prompt=prompt, **kwargs)
        _set_dave_state(self.transformer, target_blocks, False)
        return result

    pipe.__class__.__call__ = patched_call


def _set_dave_state(transformer, target_blocks, active: bool, scale=0.2, scale_map=None):
    """Enable/disable DAVE on target blocks.

    Args:
        scale: default scale for all target blocks
        scale_map: optional dict {block_idx: scale} for per-block overrides
    """
    for bidx, block in enumerate(transformer.transformer_blocks):
        if bidx in target_blocks:
            block._dave_active = active
            if scale_map is not None and bidx in scale_map:
                block._dave_scale = scale_map[bidx]
            else:
                block._dave_scale = scale


if __name__ == "__main__":
    import time

    print("Creating DAVE-SANA pipeline...")
    pipe = create_dave_sana_pipeline(
        "/models/SANA1.5_1.6B_1024px_diffusers",
        target_blocks=[13],
        dave_scale=0.2,
        tau=0.2,
        guidance_scale=4.5,
        torch_dtype=torch.bfloat16,
    )

    prompt = "a photo of a cat"
    seed = 42

    print(f"\nGenerating baseline (no DAVE, seed={seed})...")
    t0 = time.time()
    result_base = pipe(prompt, use_dave=False, num_inference_steps=20,
                       generator=torch.Generator("cuda").manual_seed(seed))
    dt_base = time.time() - t0
    print(f"  Time: {dt_base:.1f}s")
    result_base.images[0].save("/tmp/sana_baseline.png")

    print(f"\nGenerating with DAVE (seed={seed})...")
    t0 = time.time()
    result_dave = pipe(prompt, use_dave=True, num_inference_steps=20,
                       generator=torch.Generator("cuda").manual_seed(seed))
    dt_dave = time.time() - t0
    print(f"  Time: {dt_dave:.1f}s")
    result_dave.images[0].save("/tmp/sana_dave.png")

    print(f"\nDone! Baseline: {dt_base:.1f}s, DAVE: {dt_dave:.1f}s")
    print("Images saved to /tmp/sana_baseline.png and /tmp/sana_dave.png")
