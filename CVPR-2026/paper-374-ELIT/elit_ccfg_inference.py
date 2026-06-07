"""
ELIT Cheap-CFG (CCFG) and AutoGuidance Inference Comparison Script.

Compares three guidance strategies across multiple guidance scales:
  - CFG:   both conditional and unconditional paths at the same budget.
  - CCFG:  unconditional path at a lower budget (e.g. 1/16), condition dropped.
  - AutoG: unconditional path at a lower budget but the real class label is
           kept — guidance comes from the capacity gap, not from dropping the
           condition.

For each cfg_scale the script generates one image per strategy, measures
per-step FLOPs for each, and produces comparison plots and an image grid.

Usage:
  python elit_ccfg_inference.py \
      --train-config experiments/train/elit_sit_xl_256.yaml \
      --ckpt path/to/ckpt.pt \
      --inference-budget 1.0 \
      --unconditional-inference-budget 0.0625 \
      --cfg-scales 1 2 3 4 5 \
      --class-label 207 \
      --output-dir ccfg_results
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

from models.sit_elit import SiT_models as _elit_models
from utils import load_legacy_checkpoints


def _ensure_distributed():
    """Initialize a single-process distributed group if not already initialized."""
    if dist.is_initialized():
        return
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ELIT CFG vs CCFG comparison across guidance scales.",
    )

    # Config files
    parser.add_argument("--train-config", type=str, default=None)
    parser.add_argument("--eval-config", type=str, default=None)

    # Model
    parser.add_argument("--model", type=str, default="ELIT-SiT-XL/2",
                        choices=list(_elit_models.keys()))
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=256, choices=[256, 512])
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action="store_true", default=False)
    parser.add_argument("--qk-norm", action="store_true", default=False)

    # REPA
    parser.add_argument("--enable-repa", action="store_true", default=False)
    parser.add_argument("--projector-embed-dims", type=str, default="768")

    # ELIT-specific
    parser.add_argument("--elit-max-mask-prob", type=float, default=0.0)
    parser.add_argument("--elit-min-mask-prob", type=float, default=None)
    parser.add_argument("--elit-group-size", type=int, default=4)
    parser.add_argument("--elit-read-depth", type=int, default=1)
    parser.add_argument("--elit-write-depth", type=int, default=1)

    # VAE
    parser.add_argument("--vae", type=str, default="ema", choices=["ema", "mse"])

    # Sampling
    parser.add_argument("--mode", type=str, default="ode", choices=["ode", "sde"])
    parser.add_argument("--path-type", type=str, default="linear",
                        choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action="store_true", default=False)
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=1.0)

    # Budgets
    parser.add_argument("--inference-budget", type=float, default=1.0,
                        help="Budget for the conditional path (and standard CFG).")
    parser.add_argument("--unconditional-inference-budget", type=float, default=0.0625,
                        help="Budget for the unconditional path in CCFG (default: 1/16).")

    # CFG scales to sweep
    parser.add_argument("--cfg-scales", type=float, nargs="+",
                        default=[1.0, 2.0, 3.0, 4.0, 5.0])

    # Legacy
    parser.add_argument("--legacy", action="store_true", default=False)

    # Generation settings
    parser.add_argument("--class-label", type=int, default=263)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output-dir", type=str, default="ccfg_results")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)

    # ---- YAML support (same pattern as other scripts) ----
    preliminary, _ = parser.parse_known_args()

    TRAIN_KEYS = {
        'model', 'encoder_depth', 'resolution', 'num_classes',
        'fused_attn', 'qk_norm', 'enable_repa', 'projector_embed_dims',
        'elit_max_mask_prob', 'elit_min_mask_prob', 'elit_group_size',
        'elit_read_depth', 'elit_write_depth',
    }
    defaults = {a.dest: a.default for a in parser._actions if a.dest != "help"}

    if preliminary.train_config is not None:
        with open(preliminary.train_config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        cfg = {k.replace("-", "_"): v for k, v in cfg.items()}
        for k, v in cfg.items():
            if k in TRAIN_KEYS:
                defaults[k] = v

    if preliminary.eval_config is not None:
        with open(preliminary.eval_config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        cfg = {k.replace("-", "_"): v for k, v in cfg.items()}
        defaults.update(cfg)

    parser.set_defaults(**defaults)
    args = parser.parse_args()
    return args


# ---- model helpers (shared with elit_multibudget_inference) ----

def build_model(args, device):
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    latent_size = args.resolution // 8
    enable_repa = getattr(args, "enable_repa", False)
    z_dims = None
    if enable_repa and args.projector_embed_dims:
        z_dims = [int(d) for d in args.projector_embed_dims.split(",")]

    model = _elit_models[args.model](
        input_size=latent_size, num_classes=args.num_classes, use_cfg=True,
        enable_repa=enable_repa, z_dims=z_dims,
        encoder_depth=args.encoder_depth, enable_elit=True,
        elit_max_mask_prob=args.elit_max_mask_prob,
        elit_min_mask_prob=args.elit_min_mask_prob,
        group_size=args.elit_group_size,
        elit_read_depth=args.elit_read_depth,
        elit_write_depth=args.elit_write_depth,
        **block_kwargs,
    ).to(device)
    return model


def load_checkpoint(model, args, device):
    state_dict = torch.load(args.ckpt, map_location=device, weights_only=False)["ema"]
    if args.legacy:
        state_dict = load_legacy_checkpoints(
            state_dict=state_dict, encoder_depth=args.encoder_depth)
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"  Warning: Missing keys ({len(result.missing_keys)})")
    if result.unexpected_keys:
        print(f"  Warning: Unexpected keys ({len(result.unexpected_keys)})")
    model.eval()
    return model


@torch.no_grad()
def sample_single(model, z, y, args, device, cfg_scale,
                  inference_budget, unconditional_inference_budget=None,
                  autoguidance=False):
    from samplers import euler_sampler, euler_maruyama_sampler

    kw = dict(model=model, latents=z, y=y, num_steps=args.num_steps,
              heun=args.heun, cfg_scale=cfg_scale,
              guidance_low=args.guidance_low, guidance_high=args.guidance_high,
              path_type=args.path_type, inference_budget=inference_budget,
              unconditional_inference_budget=unconditional_inference_budget,
              autoguidance=autoguidance)
    if args.mode == "sde":
        return euler_maruyama_sampler(**kw).to(torch.float32)
    return euler_sampler(**kw).to(torch.float32)


@torch.no_grad()
def decode_latent(vae, samples, device):
    scale = torch.tensor([0.18215] * 4).view(1, 4, 1, 1).to(device)
    bias = torch.tensor([0.0] * 4).view(1, 4, 1, 1).to(device)
    images = vae.decode((samples - bias) / scale).sample
    images = torch.clamp(255.0 * (images + 1) / 2.0, 0, 255)
    return images.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()


@torch.no_grad()
def measure_flops(model, x, t, y, budget):
    with FlopCounterMode(display=False) as counter:
        model(x, t, y=y, inference_budget=budget)
    return counter.get_total_flops()


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    assert torch.cuda.is_available(), "GPU required."
    _ensure_distributed()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    latent_size = args.resolution // 8

    print(f"Building model: {args.model}")
    model = build_model(args, device)
    model = load_checkpoint(model, args, device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"Loading VAE (sd-vae-ft-{args.vae})...")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    y = torch.tensor([args.class_label], device=device)
    y_null = torch.tensor([1000], device=device)

    os.makedirs(args.output_dir, exist_ok=True)

    cond_budget = args.inference_budget
    uncond_budget = args.unconditional_inference_budget
    cfg_scales = sorted(args.cfg_scales)

    print(f"\nCFG scales: {cfg_scales}")
    print(f"Conditional budget: {cond_budget}")
    print(f"Unconditional budget (CCFG): {uncond_budget}")

    # ---- Measure per-forward FLOPs at each budget ----
    x_dummy = torch.randn(1, model.in_channels, latent_size, latent_size, device=device)
    t_dummy = torch.tensor([0.5], device=device, dtype=torch.float32)
    flops_cond = measure_flops(model, x_dummy, t_dummy, y, cond_budget) / 1e9
    flops_uncond_full = measure_flops(model, x_dummy, t_dummy, y_null, cond_budget) / 1e9
    flops_uncond_cheap = measure_flops(model, x_dummy, t_dummy, y_null, uncond_budget) / 1e9

    flops_cheap_cond = measure_flops(model, x_dummy, t_dummy, y, uncond_budget) / 1e9

    cfg_gflops = flops_cond + flops_uncond_full
    ccfg_gflops = flops_cond + flops_uncond_cheap
    autog_gflops = flops_cond + flops_cheap_cond
    print(f"GFLOPs per step  - CFG: {cfg_gflops:.1f}  CCFG: {ccfg_gflops:.1f}  "
          f"AutoG: {autog_gflops:.1f}  "
          f"(CCFG saving {(1 - ccfg_gflops / cfg_gflops) * 100:.1f}%)\n")

    rng_state = torch.cuda.get_rng_state()
    results = []

    # (mode_name, unconditional_budget_override, autoguidance_flag, gflops_value)
    modes = [
        ("CFG",   None,          False, cfg_gflops),
        ("CCFG",  uncond_budget, False, ccfg_gflops),
        ("AutoG", uncond_budget, True,  autog_gflops),
    ]

    for cfg_s in cfg_scales:
        print(f"---- cfg_scale={cfg_s:.1f} ----")
        for mode_name, uncond_b, is_autog, gf in modes:
            if uncond_b is not None and cfg_s <= 1.0:
                continue

            torch.cuda.set_rng_state(rng_state)
            z = torch.randn(1, model.in_channels, latent_size, latent_size, device=device)
            samples = sample_single(model, z, y, args, device, cfg_scale=cfg_s,
                                    inference_budget=cond_budget,
                                    unconditional_inference_budget=uncond_b,
                                    autoguidance=is_autog)
            pix = decode_latent(vae, samples, device)

            entry = dict(cfg_scale=cfg_s, mode=mode_name,
                         gflops_per_step=round(gf, 2),
                         image=Image.fromarray(pix[0]))
            results.append(entry)
            print(f"  {mode_name}: {gf:.1f} GF/step")

    # ---- Save JSON ----
    json_results = [{k: v for k, v in r.items() if k != "image"} for r in results]
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(json_results, f, indent=2)

    # ---- GFLOPs comparison bar chart ----
    fig, ax = plt.subplots(figsize=(10, 5))
    cfg_entries = [r for r in results if r["mode"] == "CFG" and r["cfg_scale"] > 1.0]
    ccfg_entries = [r for r in results if r["mode"] == "CCFG"]
    autog_entries = [r for r in results if r["mode"] == "AutoG"]
    if cfg_entries and ccfg_entries:
        scales_plot = [r["cfg_scale"] for r in cfg_entries]
        x_pos = np.arange(len(scales_plot))
        n_bars = 3 if autog_entries else 2
        w = 0.8 / n_bars
        offset = np.linspace(-(n_bars - 1) * w / 2, (n_bars - 1) * w / 2, n_bars)

        bar_groups = [
            (cfg_entries,   "CFG",   "#2196F3"),
            (ccfg_entries,  "CCFG",  "#FF5722"),
        ]
        if autog_entries:
            bar_groups.append((autog_entries, "AutoG", "#4CAF50"))

        all_bars = []
        for idx, (entries, label, color) in enumerate(bar_groups):
            bars = ax.bar(x_pos + offset[idx],
                          [r["gflops_per_step"] for r in entries],
                          w, label=label, color=color)
            all_bars.append(bars)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{s:.0f}" for s in scales_plot])
        ax.set_xlabel("CFG Scale", fontsize=12)
        ax.set_ylabel("GFLOPs per denoising step", fontsize=12)
        ax.set_title("CFG vs CCFG vs AutoGuidance: FLOPs per step",
                      fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        for bars in all_bars:
            for bar in bars:
                ax.annotate(f"{bar.get_height():.0f}",
                            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            textcoords="offset points", xytext=(0, 5),
                            ha="center", fontsize=8)
        fig.savefig(os.path.join(args.output_dir, "cfg_vs_ccfg_flops.png"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Image grid: rows = [CFG, CCFG, AutoG], cols = cfg scales ----
    try:
        from PIL import ImageDraw, ImageFont
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()
    except ImportError:
        font = None

    row_modes = ["CFG", "CCFG", "AutoG"]
    guided_scales = [s for s in cfg_scales if s > 1.0]
    if guided_scales:
        first_img = results[0]["image"]
        img_w, img_h = first_img.size
        label_h = 30
        cols = len(guided_scales)
        rows = len(row_modes)
        grid = Image.new("RGB",
                         (cols * img_w, rows * (img_h + label_h)),
                         (255, 255, 255))
        draw = ImageDraw.Draw(grid) if font else None

        for row_idx, mode_name in enumerate(row_modes):
            for col_idx, s in enumerate(guided_scales):
                entry = next((r for r in results
                              if r["cfg_scale"] == s and r["mode"] == mode_name), None)
                if entry is None:
                    continue
                x_off = col_idx * img_w
                y_off = row_idx * (img_h + label_h)
                grid.paste(entry["image"], (x_off, y_off))
                if draw:
                    label = f"{mode_name} s={s:.0f} | {entry['gflops_per_step']:.0f} GF"
                    draw.text((x_off + 5, y_off + img_h + 5),
                              label, fill=(0, 0, 0), font=font)

        grid_path = os.path.join(args.output_dir, "cfg_vs_ccfg_grid.png")
        grid.save(grid_path)
        print(f"\nImage grid saved to {grid_path}")

    print(f"Results saved to {args.output_dir}/")
    print("Done!")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
