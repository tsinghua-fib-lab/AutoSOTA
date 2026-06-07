"""
ELIT Multi-Budget Inference Script.

Generates a single image at all possible inference budgets for an ELIT model,
measures FLOPs, and produces a budget vs FLOPs figure and an image grid.

Usage:
  python elit_multibudget_inference.py \
      --train-config experiments/train/elit_sit_xl_256.yaml \
      --ckpt path/to/ckpt.pt \
      --class-label 207 \
      --output-dir multibudget_results

  # With eval config overrides:
  python elit_multibudget_inference.py \
      --train-config experiments/train/elit_sit_xl_256.yaml \
      --eval-config  experiments/generation/elit_sit_xl_256.yaml \
      --ckpt path/to/ckpt.pt \
      --class-label 207
"""

import os
import json
import argparse

import yaml
import torch
import torch.distributed as dist
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from diffusers.models import AutoencoderKL
from torch.utils.flop_counter import FlopCounterMode

# Import model registries
from models.sit_elit import SiT_models as _elit_models
from utils import load_legacy_checkpoints


def _ensure_distributed():
    """Initialize a single-process distributed group if not already initialized.

    The ELIT masking strategy calls dist.get_rank() during training-mode
    forward passes.  Even though inference_budget != None bypasses that
    code path, we still need an initialized process group because the
    masking strategy object is created with synchronized_budget_sampling=True
    and other dist-dependent defaults.  This helper makes the script work
    with both plain `python` and `torchrun` launches.
    """
    if dist.is_initialized():
        return
    # Set minimal env vars for a single-process group when launched with
    # plain `python` (torchrun already sets these).
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ELIT Multi-Budget Inference: generate a single image at "
        "every budget, measure FLOPs, and produce plots.",
    )

    # Config files
    parser.add_argument("--train-config", type=str, default=None,
                        help="Path to a training YAML config (model architecture settings).")
    parser.add_argument("--eval-config", type=str, default=None,
                        help="Path to an evaluation YAML config (sampling settings). CLI args override.")

    # Model
    parser.add_argument("--model", type=str, default="ELIT-SiT-XL/2",
                        choices=list(_elit_models.keys()))
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to model checkpoint (.pt).")
    parser.add_argument("--resolution", type=int, default=256,
                        choices=[256, 512])
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action="store_true", default=False)
    parser.add_argument("--qk-norm", action="store_true", default=False)

    # REPA
    parser.add_argument("--enable-repa", action="store_true", default=False)
    parser.add_argument("--projector-embed-dims", type=str, default="768")

    # ELIT-specific
    parser.add_argument("--elit-max-mask-prob", type=float, default=0.0, help="Maximum masking probability for ELIT during training (0.0 = no masking)")
    parser.add_argument("--elit-min-mask-prob", type=float, default=None)
    parser.add_argument("--elit-group-size", type=int, default=4)
    parser.add_argument("--elit-read-depth", type=int, default=1, help="Depth of the read (cross-attention) layer")
    parser.add_argument("--elit-write-depth", type=int, default=1, help="Depth of the write (cross-attention) layer")

    # VAE
    parser.add_argument("--vae", type=str, default="ema", choices=["ema", "mse"])

    # Sampling
    parser.add_argument("--mode", type=str, default="ode", choices=["ode", "sde"])
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--path-type", type=str, default="linear",
                        choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action="store_true", default=False)
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=1.0)

    # Legacy
    parser.add_argument("--legacy", action="store_true", default=False)

    # Generation settings
    parser.add_argument("--class-label", type=int, default=263,
                        help="ImageNet class label for generation (default: 207 = golden retriever).")
    parser.add_argument("--seed", type=int, default=42)

    # Budgets
    parser.add_argument("--budgets", type=float, nargs="+", default=None,
                        help="List of inference budgets to evaluate. "
                             "If not set, uses np.arange(0.1, 1.05, 0.1).")

    # Output
    parser.add_argument("--output-dir", type=str, default="multibudget_results")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)

    # ── Parse with YAML support ──
    # Priority: CLI args > eval-config > train-config > defaults
    preliminary, _ = parser.parse_known_args()

    TRAIN_KEYS_TO_IMPORT = {
        'model', 'encoder_depth', 'resolution', 'num_classes',
        'fused_attn', 'qk_norm',
        'enable_repa', 'projector_embed_dims',
        'elit_max_mask_prob', 'elit_min_mask_prob', 'elit_group_size',
        'elit_read_depth', 'elit_write_depth',
    }

    defaults = {a.dest: a.default for a in parser._actions if a.dest != "help"}

    if preliminary.train_config is not None:
        with open(preliminary.train_config, "r") as f:
            train_cfg = yaml.safe_load(f) or {}
        train_cfg = {k.replace("-", "_"): v for k, v in train_cfg.items()}
        for k, v in train_cfg.items():
            if k in TRAIN_KEYS_TO_IMPORT:
                defaults[k] = v

    if preliminary.eval_config is not None:
        with open(preliminary.eval_config, "r") as f:
            eval_cfg = yaml.safe_load(f) or {}
        eval_cfg = {k.replace("-", "_"): v for k, v in eval_cfg.items()}
        defaults.update(eval_cfg)

    parser.set_defaults(**defaults)
    args = parser.parse_args()

    if args.budgets is None:
        g = args.elit_group_size
        n_levels = g * g
        args.budgets = (np.arange(2, n_levels + 1) / n_levels).tolist()

    return args


def build_model(args, device):
    """Build the ELIT model."""
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    latent_size = args.resolution // 8

    enable_repa = getattr(args, "enable_repa", False)
    z_dims = None
    if enable_repa and args.projector_embed_dims:
        z_dims = [int(d) for d in args.projector_embed_dims.split(",")]

    model_kwargs = dict(
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=True,
        enable_repa=enable_repa,
        z_dims=z_dims,
        encoder_depth=args.encoder_depth,
        enable_elit=True,
        elit_max_mask_prob=args.elit_max_mask_prob,
        elit_min_mask_prob=args.elit_min_mask_prob,
        group_size=args.elit_group_size,
        elit_read_depth=args.elit_read_depth,
        elit_write_depth=args.elit_write_depth,
        **block_kwargs,
    )

    model = _elit_models[args.model](**model_kwargs).to(device)
    return model


def load_checkpoint(model, args, device):
    """Load checkpoint into model."""
    state_dict = torch.load(args.ckpt, map_location=device, weights_only=False)["ema"]
    if args.legacy:
        state_dict = load_legacy_checkpoints(
            state_dict=state_dict, encoder_depth=args.encoder_depth
        )
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"  Warning: Missing keys ({len(result.missing_keys)})")
    if result.unexpected_keys:
        print(f"  Warning: Unexpected keys ({len(result.unexpected_keys)})")
    model.eval()
    return model


@torch.no_grad()
def sample_single(model, z, y, args, device, inference_budget):
    """Run the full ODE/SDE sampling loop for a single image with a given budget."""
    from samplers import euler_sampler, euler_maruyama_sampler

    sampling_kwargs = dict(
        model=model,
        latents=z,
        y=y,
        num_steps=args.num_steps,
        heun=args.heun,
        cfg_scale=args.cfg_scale,
        guidance_low=args.guidance_low,
        guidance_high=args.guidance_high,
        path_type=args.path_type,
        inference_budget=inference_budget,
    )
    if args.mode == "sde":
        samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
    elif args.mode == "ode":
        samples = euler_sampler(**sampling_kwargs).to(torch.float32)
    else:
        raise NotImplementedError(f"Unknown mode: {args.mode}")
    return samples


@torch.no_grad()
def decode_latent(vae, samples, device):
    """Decode latent samples to pixel space."""
    latents_scale = torch.tensor([0.18215] * 4).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor([0.0] * 4).view(1, 4, 1, 1).to(device)
    images = vae.decode((samples - latents_bias) / latents_scale).sample
    images = (images + 1) / 2.0
    images = torch.clamp(255.0 * images, 0, 255)
    images = images.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    return images


@torch.no_grad()
def measure_flops_single_forward(model, x_dummy, t_dummy, y_dummy, inference_budget):
    """Measure FLOPs for a single forward pass of the model (not full sampling loop)."""
    with FlopCounterMode(display=False) as counter:
        model(x_dummy, t_dummy, y=y_dummy, inference_budget=inference_budget)
    return counter.get_total_flops()


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    assert torch.cuda.is_available(), "This script requires a GPU."

    # Ensure distributed is initialized (needed by ELIT masking strategy)
    _ensure_distributed()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    latent_size = args.resolution // 8

    # Build model and load checkpoint
    print(f"Building model: {args.model}")
    model = build_model(args, device)
    model = load_checkpoint(model, args, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Load VAE
    print(f"Loading VAE (sd-vae-ft-{args.vae})...")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Prepare inputs (batch size 1)
    y = torch.tensor([args.class_label], device=device)

    os.makedirs(args.output_dir, exist_ok=True)

    budgets = sorted(args.budgets)
    print(f"\nBudgets to evaluate: {budgets}")

    rng_state = torch.cuda.get_rng_state()
    results = []

    for budget in budgets:
        budget_val = round(budget, 4)
        print(f"── Budget {budget_val:.2f} ──")

        # ── Measure FLOPs (single model forward, not full sampling loop) ──
        x_dummy = torch.randn(1, model.in_channels, latent_size, latent_size, device=device)
        t_dummy = torch.tensor([0.5], device=device, dtype=torch.float32)
        if args.cfg_scale > 1.0:
            x_flop = torch.cat([x_dummy] * 2, dim=0)
            y_flop = torch.cat([y, torch.tensor([1000], device=device)], dim=0)
        else:
            x_flop = x_dummy
            y_flop = y
        flops = measure_flops_single_forward(model, x_flop, t_dummy.expand(x_flop.shape[0]), y_flop, budget_val)
        gflops = flops / 1e9

        # ── Generate and decode ──
        torch.cuda.set_rng_state(rng_state)
        z = torch.randn(1, model.in_channels, latent_size, latent_size, device=device)
        samples = sample_single(model, z, y, args, device, inference_budget=budget_val)
        pixel_images = decode_latent(vae, samples, device)

        entry = {
            "budget": budget_val,
            "gflops_per_forward": round(gflops, 2),
            "flops_per_forward": flops,
            "image": Image.fromarray(pixel_images[0]),
        }
        results.append(entry)
        print(f"  GFLOPs/fwd: {gflops:.2f}")

    # ── Save results JSON ──
    json_results = [{k: v for k, v in r.items() if k != "image"} for r in results]
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ── Create FLOPs figure ──
    budgets_arr = np.array([r["budget"] for r in results])
    gflops_arr = np.array([r["gflops_per_forward"] for r in results])

    fig_flops, ax_flops = plt.subplots(figsize=(7, 5))
    ax_flops.plot(budgets_arr, gflops_arr, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax_flops.set_xlabel("Inference Budget (fraction of latent tokens)", fontsize=12)
    ax_flops.set_ylabel("GFLOPs per forward pass", fontsize=12)
    ax_flops.set_title("Budget vs. FLOPs", fontsize=14, fontweight="bold")
    ax_flops.grid(True, alpha=0.3)
    ax_flops.set_xlim(0, 1.05)
    for b, g in zip(budgets_arr, gflops_arr):
        ax_flops.annotate(f"{g:.0f}", (b, g), textcoords="offset points",
                          xytext=(0, 10), ha="center", fontsize=8)
    fig_flops_path = os.path.join(args.output_dir, "budget_vs_flops.png")
    fig_flops.savefig(fig_flops_path, dpi=150, bbox_inches="tight")
    plt.close(fig_flops)
    print(f"FLOPs figure saved to {fig_flops_path}")

    # ── Create image grid ──
    print("\nCreating image grid...")
    n_imgs = len(results)
    img_w, img_h = results[0]["image"].size
    cols = min(n_imgs, 5)
    rows = (n_imgs + cols - 1) // cols
    label_height = 30
    grid_w = cols * img_w
    grid_h = rows * (img_h + label_height)
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(grid)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()
    except ImportError:
        draw = None
        font = None

    for idx, r in enumerate(results):
        img = r["image"]
        row = idx // cols
        col = idx % cols
        x_off = col * img_w
        y_off = row * (img_h + label_height)
        grid.paste(img, (x_off, y_off))
        if draw is not None:
            label = f"b={r['budget']:.3f} | {r['gflops_per_forward']:.0f} GF"
            draw.text((x_off + 5, y_off + img_h + 5), label, fill=(0, 0, 0), font=font)

    grid_path = os.path.join(args.output_dir, "image_grid.png")
    grid.save(grid_path)
    print(f"Image grid saved to {grid_path}")

    print("\nDone!")

    # Cleanup distributed
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
