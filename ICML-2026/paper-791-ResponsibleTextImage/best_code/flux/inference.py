#!/usr/bin/env python3
"""
External Heads Guided Image Generation for FLUX.1-dev
=====================================================

"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional

import torch
from diffusers import FluxPipeline

# =====================================================================
# CONFIG
# =====================================================================

MODEL_ID = "black-forest-labs/FLUX.1-dev"

# Prompts
PROMPT="a photo of a doctor"
NEGATIVE_PROMPT: Optional[str] = None

GUIDANCE_SCALE = 2.5
NUM_INFERENCE_STEPS = 30

# Seeds
BASE_SEED = 42
NUM_SEEDS = 1

# External heads checkpoint (PixArt-style final)
EXTERNAL_HEADS_PATH = "./external_heads.pt"

# Target layers must match training `cfg.target_layers` (0..18)
TARGET_LAYERS: List[int] = list(range(19))

# Optionally restrict to certain heads; None = all heads
# e.g., TARGET_HEADS = [7, 8] if you want just two heads.
TARGET_HEADS: Optional[List[int]] = [3, 12, 13, 18]

# Coefficients to sweep (positive = same direction as training)
COEFFICIENT_LIST = [0, 10]
# Generation resolution (must match training, which used 512)
IMAGE_RESOLUTION = 512

# Output
OUTPUT_DIR = "images_out"

# Device / dtype
DEVICE = "cuda"
DTYPE = torch.bfloat16  # training used bf16


# =====================================================================
# LoadedExternalHeads: utility to load & serve heads per layer
# =====================================================================

class LoadedExternalHeads(torch.nn.Module):
    """
    Wraps the saved external_heads tensor for generation.

    - Expects a tensor of shape [L, H, S, D]
    - target_layers: list of global layer indices (same as training)
    - Provides get_for_layer() to return [B, H, N, D] for a given layer
    """

    def __init__(self, tensor: torch.Tensor, target_layers: List[int]):
        super().__init__()
        self.target_layers_sorted = sorted(target_layers)
        self.layer_to_index = {l: i for i, l in enumerate(self.target_layers_sorted)}

        L, H, S, D = tensor.shape
        if L != len(self.target_layers_sorted):
            raise ValueError(
                f"Checkpoint has {L} layers but target_layers has "
                f"{len(self.target_layers_sorted)} entries."
            )

        self.num_heads = H
        self.seq_len = S
        self.head_dim = D

        # Register as a buffer (no gradients needed)
        self.register_buffer("external_heads", tensor)

    def get_for_layer(
        self,
        layer_idx: int,
        seq_len_current: int,
        batch_size: int,
        target_heads: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if layer_idx not in self.layer_to_index:
            # Layer not trained: return zeros
            return torch.zeros(
                batch_size,
                self.num_heads,
                seq_len_current,
                self.head_dim,
                device=self.external_heads.device,
                dtype=self.external_heads.dtype,
            )

        if seq_len_current > self.seq_len:
            raise ValueError(
                f"Runtime seq_len ({seq_len_current}) > checkpoint seq_len ({self.seq_len}) "
                f"for layer {layer_idx}."
            )

        l_local = self.layer_to_index[layer_idx]
        # [H, S, D] -> slice to current seq_len
        layer_heads = self.external_heads[l_local, :, :seq_len_current, :]  # [H, N, D]

        if target_heads is not None:
            # Build head mask
            mask = torch.zeros(self.num_heads, device=self.external_heads.device)
            for h in target_heads:
                if 0 <= h < self.num_heads:
                    mask[h] = 1.0
            # [H] -> [H,1,1]
            mask = mask.view(self.num_heads, 1, 1)
            layer_heads = layer_heads * mask  # zero out unselected heads

        # [H, N, D] -> [B, H, N, D]
        layer_heads = layer_heads.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return layer_heads


# =====================================================================
# Hook installation for FLUX transformer attention
# =====================================================================

def install_flux_external_heads_hooks(
    pipe: FluxPipeline,
    loaded_heads: LoadedExternalHeads,
    target_layers: List[int],
    coefficient: float,
    target_heads: Optional[List[int]] = None,
):
    """
    Patches Flux transformer attention (attn.forward) to add external heads.

    Returns:
        baseline_forwards: dict[layer_idx] -> original attn.forward
    """
    print(f"\nInstalling external-heads hooks (coef={coefficient}) on FLUX...")
    tr = pipe.transformer
    baseline_forwards: Dict[int, callable] = {}

    for layer_idx in target_layers:
        block = tr.transformer_blocks[layer_idx]
        attn = block.attn
        orig_forward = attn.forward

        def make_wrapper(orig_fwd, attn_module, layer_index):
            def wrapped_forward(hidden_states, *args, **kwargs):
                # Call original FLUX attention
                out = orig_fwd(hidden_states, *args, **kwargs)

                # Unpack tensor/tuple
                if isinstance(out, tuple):
                    attn_hidden = out[0]  # [B, N, C]
                    extra = out[1:]
                else:
                    attn_hidden = out
                    extra = None

                B, N, C = attn_hidden.shape
                H = loaded_heads.num_heads
                D = loaded_heads.head_dim

                if H * D != C:
                    raise RuntimeError(
                        f"[Shape mismatch] layer {layer_index}: "
                        f"C={C}, but H*D={H*D} (H={H}, D={D})"
                    )

                # [B, H, N, D]
                ext = loaded_heads.get_for_layer(
                    layer_index,
                    seq_len_current=N,
                    batch_size=B,
                    target_heads=target_heads,
                )
                # dtype/device align
                ext = ext.to(device=attn_hidden.device, dtype=attn_hidden.dtype)

                # [B, H, N, D] -> [B, N, H*D]
                ext = ext.permute(0, 2, 1, 3).reshape(B, N, H * D)

                # Add scaled external heads
                attn_hidden = attn_hidden + coefficient * ext

                if extra is not None:
                    return (attn_hidden, *extra)
                return attn_hidden

            return wrapped_forward

        attn.forward = make_wrapper(orig_forward, attn, layer_idx)
        baseline_forwards[layer_idx] = orig_forward
        print(f"  ✓ Patched FLUX attention forward for layer {layer_idx}")

    print("Done installing FLUX hooks.\n")
    return baseline_forwards


def reset_flux_external_heads_hooks(
    pipe: FluxPipeline,
    target_layers: List[int],
    baseline_forwards: Dict[int, callable],
):
    """
    Restore original attention.forward for all target layers.
    """
    tr = pipe.transformer
    for layer_idx in target_layers:
        block = tr.transformer_blocks[layer_idx]
        attn = block.attn
        if layer_idx in baseline_forwards:
            attn.forward = baseline_forwards[layer_idx]
    print("✓ Restored original FLUX attention forwards.\n")


# =====================================================================
# External heads loading
# =====================================================================

def load_external_heads_flux(
    checkpoint_path: str,
    target_layers: List[int],
    device: str,
    dtype: torch.dtype,
) -> LoadedExternalHeads:
    """
    Load external_heads_full.pt saved by `save_heads_pixart_style` in the FLUX trainer.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"External heads checkpoint not found: {checkpoint_path}")

    print(f"\n📦 Loading external heads from: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")

    if "external_heads" not in state:
        keys = ", ".join(state.keys())
        raise KeyError(
            f"'external_heads' not found in checkpoint. Available keys: {keys}"
        )

    ext_tensor = state["external_heads"]  # [L, H, S, D]
    print(
        f"  external_heads tensor shape = {tuple(ext_tensor.shape)} "
        "(layers, heads, seq_len, head_dim)"
    )

    loaded_heads = LoadedExternalHeads(ext_tensor.to(device=device, dtype=dtype), target_layers)

    print(
        f"  num_layers={len(loaded_heads.target_layers_sorted)}, "
        f"num_heads={loaded_heads.num_heads}, "
        f"seq_len={loaded_heads.seq_len}, "
        f"head_dim={loaded_heads.head_dim}"
    )
    print("✓ External heads loaded for FLUX.\n")
    return loaded_heads


# =====================================================================
# Image generation
# =====================================================================

def generate_image_flux(
    pipe: FluxPipeline,
    prompt: str,
    negative_prompt: Optional[str],
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
):
    """
    Generate a single image with FluxPipeline.
    Forces height/width to match training resolution (512x512).
    """
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        height=height,
        width=width,
    )
    return out.images[0]


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 80)
    print("FLUX.1-dev EXTERNAL HEADS GUIDED IMAGE GENERATION")
    print("=" * 80)
    print(f"Model: {MODEL_ID}")
    print(f"Prompt: {PROMPT!r}")
    print(f"Negative prompt: {NEGATIVE_PROMPT!r}")
    print(f"Target layers: {TARGET_LAYERS}")
    print(f"Target heads: {TARGET_HEADS if TARGET_HEADS is not None else 'All heads'}")
    print(f"Base seed: {BASE_SEED}")
    print(f"Num seeds: {NUM_SEEDS}")
    print(f"Coefficients: {COEFFICIENT_LIST}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------------
    # Load FLUX pipeline
    # ---------------------------
    print("\nLoading FLUX pipeline...")
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    print("HF_TOKEN visible:", "YES" if hf_token else "NO")

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        token=hf_token,
    )
    pipe.to(DEVICE)

    print(f"✓ Loaded FLUX model: {MODEL_ID} on {DEVICE} with dtype={DTYPE}")

    # ---------------------------
    # Load external heads
    # ---------------------------
    loaded_heads = load_external_heads_flux(
        EXTERNAL_HEADS_PATH,
        TARGET_LAYERS,
        device=DEVICE,
        dtype=DTYPE,
    )

    total_images = 0

    # ---------------------------
    # Loop over seeds
    # ---------------------------
    for i in range(NUM_SEEDS):
        seed = BASE_SEED + i
        seed_dir = os.path.join(OUTPUT_DIR, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"SEED {seed} ({i+1}/{NUM_SEEDS})")
        print("=" * 80)

        # Baseline image: no hooks installed
        print("  Generating baseline (no external heads)...")
        try:
            img_base = generate_image_flux(
                pipe,
                PROMPT,
                NEGATIVE_PROMPT,
                seed,
                NUM_INFERENCE_STEPS,
                GUIDANCE_SCALE,
                IMAGE_RESOLUTION,
                IMAGE_RESOLUTION,
            )
            base_path = os.path.join(seed_dir, "baseline_coef_0.0.png")
            img_base.save(base_path)
            print(f"  ✓ Saved baseline: {base_path}")
            total_images += 1
        except Exception as e:
            print(f"  ✗ Baseline generation failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Generation with external heads
        print("\n  Generating images with external heads...")

        for coef in COEFFICIENT_LIST:
            if coef == 0.0:
                # Already did baseline
                continue

            baseline_forwards = {}
            try:
                # Install FLUX hooks for this coefficient
                baseline_forwards = install_flux_external_heads_hooks(
                    pipe,
                    loaded_heads,
                    TARGET_LAYERS,
                    coefficient=coef,
                    target_heads=TARGET_HEADS,
                )

                img = generate_image_flux(
                    pipe,
                    PROMPT,
                    NEGATIVE_PROMPT,
                    seed,
                    NUM_INFERENCE_STEPS,
                    GUIDANCE_SCALE,
                    IMAGE_RESOLUTION,
                    IMAGE_RESOLUTION,
                )

                img_name = f"image_coef_{coef:.2f}.png"
                img_path = os.path.join(seed_dir, img_name)
                img.save(img_path)
                print(f"  ✓ Saved: {img_path}")
                total_images += 1

            except Exception as e:
                print(f"  ✗ ERROR for coef={coef}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Always restore original forwards
                try:
                    reset_flux_external_heads_hooks(
                        pipe,
                        TARGET_LAYERS,
                        baseline_forwards,
                    )
                except Exception as e2:
                    print(f"  ✗ ERROR restoring FLUX hooks: {e2}")

        print(
            f"  ➜ Done with seed {seed}: "
            f"{1 + max(0, len(COEFFICIENT_LIST) - 1)} images generated."
        )

    # ---------------------------
    # Summary file
    # ---------------------------
    summary_path = os.path.join(OUTPUT_DIR, "generation_summary_flux.txt")
    try:
        with open(summary_path, "w") as f:
            f.write("FLUX External Heads Guided Generation\n")
            f.write("=" * 80 + "\n\n")
            f.write("Configuration:\n")
            f.write(f"  Model: {MODEL_ID}\n")
            f.write(f"  Prompt: {PROMPT!r}\n")
            f.write(f"  Negative prompt: {NEGATIVE_PROMPT!r}\n")
            f.write(f"  Guidance scale: {GUIDANCE_SCALE}\n")
            f.write(f"  Inference steps: {NUM_INFERENCE_STEPS}\n")
            f.write(f"  Image resolution: {IMAGE_RESOLUTION}x{IMAGE_RESOLUTION}\n")
            f.write(f"  Base seed: {BASE_SEED}\n")
            f.write(f"  Num seeds: {NUM_SEEDS}\n")
            f.write(f"  Seeds used: {BASE_SEED} .. {BASE_SEED + NUM_SEEDS - 1}\n")
            f.write(f"  Target layers: {TARGET_LAYERS}\n")
            f.write(
                f"  Target heads: {TARGET_HEADS if TARGET_HEADS is not None else 'All heads'}\n"
            )
            f.write(f"  Coefficients: {COEFFICIENT_LIST}\n")
            f.write(f"  External heads path: {EXTERNAL_HEADS_PATH}\n")
            f.write(f"  Device: {DEVICE}\n")
            f.write(f"  DType: {DTYPE}\n\n")

            f.write("Approach:\n")
            f.write("  - Generate baseline image for each seed with no hooks.\n")
            f.write("  - For each coefficient, patch FLUX attention with external heads\n")
            f.write("    and generate another image, then restore original attention.\n\n")

            f.write("Directory structure:\n")
            for i in range(NUM_SEEDS):
                s = BASE_SEED + i
                f.write(f"  seed_{s}/\n")
                f.write(f"    - baseline_coef_0.0.png\n")
                for coef in COEFFICIENT_LIST:
                    if coef == 0.0:
                        continue
                    f.write(f"    - image_coef_{coef:.2f}.png\n")
                f.write("\n")

            f.write(
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
    except Exception as e:
        print(f"✗ ERROR writing summary file: {e}")

    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total images generated: {total_images}")
    print(f"Seeds: {BASE_SEED} .. {BASE_SEED + NUM_SEEDS - 1}")
    print(f"Coefficients: {COEFFICIENT_LIST}")
    print("✓ All done!")


if __name__ == "__main__":
    main()
