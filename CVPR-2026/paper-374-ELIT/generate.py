# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Unified sampling script for SiT, DFM-SiT, and ELIT-SiT models using DDP.
Saves a .npz file that can be used to compute FID and other evaluation metrics.

Supports configuration via:
  1. Training YAML:    --train-config path/to/train.yaml   (model architecture)
  2. Evaluation YAML:  --eval-config  path/to/eval.yaml    (sampling & output)
  3. CLI arguments:    --cfg-scale 1.5 ...
  Priority: CLI args > eval-config > train-config > defaults.

Usage examples:
  # Train config (model) + eval config (sampling) + checkpoint
  torchrun --nproc_per_node=8 generate.py \
      --train-config experiments/train/elit_sit_xl_256.yaml \
      --eval-config  experiments/generation/elit_sit_xl_256.yaml \
      --ckpt exps/elit-sit-xl-2-256px/checkpoints/0400000.pt

  # Only train config (uses default eval settings)
  torchrun --nproc_per_node=8 generate.py \
      --train-config experiments/train/sit_xl_2_256.yaml \
      --ckpt path/to/ckpt.pt

  # Pure CLI (no config files)
  torchrun --nproc_per_node=8 generate.py \
      --model ELIT-SiT-B/2 --ckpt path/to/ckpt.pt --inference-budget 0.5
"""

import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
import yaml

# Import all model registries
from models.sit import SiT_models as _sit_models
from models.sit_dfm import SiT_models as _dfm_models
from models.sit_elit import SiT_models as _elit_models
from utils import load_legacy_checkpoints, download_model

# Combined model registry
ALL_MODELS = {}
ALL_MODELS.update(_sit_models)
ALL_MODELS.update(_dfm_models)
ALL_MODELS.update(_elit_models)


def get_model_type(model_name):
    """Determine model type from model name prefix."""
    if model_name.startswith('DFM-'):
        return 'dfm'
    elif model_name.startswith('ELIT-'):
        return 'elit'
    return 'sit'


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    Detects and removes corrupted images, returning their indices.
    """
    samples = []
    corrupted_indices = []

    for i in tqdm(range(num), desc="Building .npz file from samples"):
        image_path = f"{sample_dir}/{i:06d}.png"
        try:
            sample_pil = Image.open(image_path)
            sample_np = np.asarray(sample_pil).astype(np.uint8)
            samples.append(sample_np)
        except Exception as e:
            print(f"\nWarning: Corrupted image at index {i} ({image_path}): {e}")
            corrupted_indices.append(i)
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Deleted corrupted file: {image_path}")

    if corrupted_indices:
        print(f"\n{'='*60}")
        print(f"Found {len(corrupted_indices)} corrupted images!")
        print(f"Corrupted indices: {corrupted_indices[:10]}{'...' if len(corrupted_indices) > 10 else ''}")
        print(f"Corrupted files have been deleted.")
        print(f"Please re-run the script to regenerate these images.")
        print(f"{'='*60}\n")
        return None

    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def build_model(args, device):
    """Build the model based on model type and args."""
    model_type = get_model_type(args.model)
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    latent_size = args.resolution // 8

    # REPA is only enabled when projector_embed_dims is set and enable_repa flag is true
    enable_repa = getattr(args, 'enable_repa', False)
    z_dims = None
    if enable_repa and args.projector_embed_dims:
        z_dims = [int(z_dim) for z_dim in args.projector_embed_dims.split(',')]

    model_kwargs = dict(
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=True,
        enable_repa=enable_repa,
        z_dims=z_dims,
        encoder_depth=args.encoder_depth,
        **block_kwargs,
    )

    if model_type == 'elit':
        model_kwargs.update(dict(
            enable_elit=True,
            elit_max_mask_prob=args.elit_max_mask_prob,
            elit_min_mask_prob=args.elit_min_mask_prob,
            group_size=args.elit_group_size,
            elit_read_depth=args.elit_read_depth,
            elit_write_depth=args.elit_write_depth,
        ))

    model = ALL_MODELS[args.model](**model_kwargs).to(device)
    return model, model_type


def load_checkpoint(model, args, device):
    """Load checkpoint into model."""
    ckpt_path = args.ckpt
    if ckpt_path is None:
        args.ckpt = 'SiT-XL-2-256x256.pt'
        assert args.model == 'SiT-XL/2'
        state_dict = download_model('last.pt')
    else:
        state_dict = torch.load(ckpt_path, map_location=f'cuda:{device}', weights_only=False)['ema']
    if args.legacy:
        state_dict = load_legacy_checkpoints(
            state_dict=state_dict, encoder_depth=args.encoder_depth
        )
    # strict=False allows loading checkpoints that were trained with REPA
    # into models without projectors (and vice versa)
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"  ⚠  Missing keys ({len(result.missing_keys)}):")
        for k in result.missing_keys:
            print(f"      - {k}")
    if result.unexpected_keys:
        print(f"  ⚠  Unexpected keys ({len(result.unexpected_keys)}):")
        for k in result.unexpected_keys:
            print(f"      - {k}")
    if not result.missing_keys and not result.unexpected_keys:
        print("  ✓  All checkpoint keys matched perfectly.")
    model.eval()
    return model


def sample_sit_or_elit(model, z, y, args, device):
    """Sample from a standard SiT or ELIT-SiT model."""
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
        inference_budget=args.inference_budget,
        unconditional_inference_budget=getattr(args, 'unconditional_inference_budget', None),
        autoguidance=getattr(args, 'autoguidance', False),
        cfg_schedule=getattr(args, 'cfg_schedule', 'constant'),
        cfg_beta_a=getattr(args, 'cfg_beta_a', 3),
        cfg_beta_b=getattr(args, 'cfg_beta_b', 3),
        timestep_schedule=getattr(args, 'timestep_schedule', 'uniform'),
        timestep_rho=getattr(args, 'timestep_rho', 7),
    )
    if args.mode == "sde":
        samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
    elif args.mode == "ode":
        samples = euler_sampler(**sampling_kwargs).to(torch.float32)
    else:
        raise NotImplementedError(f"Unknown sampling mode: {args.mode}")
    return samples


def sample_dfm(model, n, latent_size, y, args, device):
    """Sample from a DFM-SiT model with multiscale sampling."""
    from dfm_utils.samplers_dfm import dfm_euler_sampler
    from dfm_utils.laplacian_decomposer import LaplacianDecomposer2D

    # Create multiscale noise
    multiscale_z = {}
    for stage in range(args.stages_count):
        stage_latent_size = latent_size // (2 ** (args.stages_count - 1 - stage))
        multiscale_z[stage] = torch.randn(
            (n, 4, stage_latent_size, stage_latent_size), device=device
        )

    multiscale_samples = dfm_euler_sampler(
        model=model,
        latents_dict=multiscale_z,
        y=y,
        heun=args.heun,
        cfg_scale=args.cfg_scale,
        guidance_low=args.guidance_low,
        guidance_high=args.guidance_high,
        path_type=args.path_type,
        num_steps_per_scale=args.num_steps_per_scale,
        stage_thresholds=args.stage_sampling_thresholds,
    )

    # Recompose multiscale samples back to single latent
    laplacian_decomposer = LaplacianDecomposer2D(stages_count=args.stages_count)
    samples = laplacian_decomposer.recompose(multiscale_samples).to(torch.float32)
    return samples


def main(args):
    """Run sampling."""
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU."
    torch.set_grad_enabled(False)

    # Setup DDP
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    model_type = get_model_type(args.model)
    latent_size = args.resolution // 8

    # Build and load model
    model, model_type = build_model(args, device)
    model = load_checkpoint(model, args, device)

    # Load VAE
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale should be >= 1.0"

    # Latent scale/bias for VAE decoding
    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
    ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
    ).view(1, 4, 1, 1).to(device)

    # Create folder to save samples under results/{exp_name}/{run_id}/
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    run_id = (
        f"ckpt-{ckpt_string_name}-cfg-{args.cfg_scale}-"
        f"steps-{args.num_steps}-seed-{args.global_seed}-{args.mode}"
    )
    sample_folder_dir = os.path.join(args.sample_dir, args.exp_name, run_id)
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
        print(f"Model type: {model_type}")
        print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        if model.projectors is not None:
            print(f"Projector Parameters: {sum(p.numel() for p in model.projectors.parameters()):,}")
        else:
            print(f"REPA disabled (no projectors)")
    dist.barrier()

    # Figure out how many samples we need per GPU
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar

    # Count existing images to support resuming
    if rank == 0:
        existing_images = (
            len([f for f in os.listdir(sample_folder_dir) if f.endswith('.png')])
            if os.path.exists(sample_folder_dir) else 0
        )
        print(f"Found {existing_images} existing images, "
              f"will generate {args.num_fid_samples - existing_images} more")
    else:
        existing_images = 0

    existing_images_tensor = torch.tensor(existing_images, device=device)
    dist.broadcast(existing_images_tensor, src=0)
    existing_images = existing_images_tensor.item()

    # Skip generation if all images already exist
    if existing_images >= args.num_fid_samples:
        if rank == 0:
            print(f"All {args.num_fid_samples} images already exist. Skipping generation.")
        dist.barrier()
        if rank == 0:
            npz_path = create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
            if npz_path is None:
                print("\nError: Some images were corrupted. Please re-run to regenerate.")
                dist.barrier()
                dist.destroy_process_group()
                exit(1)
            print("Done.")
        dist.barrier()
        dist.destroy_process_group()
        return

    # Calculate completed iterations for resume
    completed_iterations = 0
    if existing_images > 0:
        for iter_idx in range(iterations):
            iter_start = iter_idx * global_batch_size
            all_exist = True
            for i in range(n):
                index = i * dist.get_world_size() + rank + iter_start
                if index >= args.num_fid_samples:
                    break
                image_path = f"{sample_folder_dir}/{index:06d}.png"
                if not os.path.exists(image_path):
                    all_exist = False
                    break
            if all_exist:
                completed_iterations += 1
            else:
                break
    if rank == 0 and completed_iterations > 0:
        print(f"Resuming from iteration {completed_iterations}/{iterations}")

    # ── Main generation loop ──
    for iteration_idx in pbar:
        if iteration_idx < completed_iterations:
            continue

        total = iteration_idx * global_batch_size
        y = torch.randint(0, args.num_classes, (n,), device=device)

        with torch.no_grad():
            if model_type == 'dfm':
                samples = sample_dfm(model, n, latent_size, y, args, device)
            else:
                z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
                samples = sample_sit_or_elit(model, z, y, args, device)

            # Decode latents with VAE
            samples = vae.decode((samples - latents_bias) / latents_scale).sample
            samples = (samples + 1) / 2.
            samples = torch.clamp(
                255. * samples, 0, 255
            ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples to disk
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                if index >= args.num_fid_samples:
                    continue
                image_path = f"{sample_folder_dir}/{index:06d}.png"
                if not os.path.exists(image_path):
                    Image.fromarray(sample).save(image_path)

    # Finalize
    dist.barrier()
    if rank == 0:
        npz_path = create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        if npz_path is None:
            print("\nError: Some images were corrupted. Please re-run to regenerate.")
            dist.barrier()
            dist.destroy_process_group()
            exit(1)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
#  Argument parsing with YAML + CLI support
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    # seed
    global_seed=0,
    # precision
    tf32=True,
    # logging / saving
    ckpt=None,
    sample_dir="results",
    exp_name=None,
    # model
    model="SiT-XL/2",
    num_classes=1000,
    encoder_depth=8,
    resolution=256,
    fused_attn=False,
    qk_norm=False,
    # REPA (disabled by default)
    enable_repa=False,
    projector_embed_dims="768",
    # vae
    vae="ema",
    # number of samples
    per_proc_batch_size=32,
    num_fid_samples=50_000,
    # sampling
    mode="ode",
    cfg_scale=1.5,
    path_type="linear",
    num_steps=50,
    heun=False,
    guidance_low=0.0,
    guidance_high=1.0,
    # legacy
    legacy=False,
    # ELIT
    elit_max_mask_prob=0.5,
    elit_min_mask_prob=None,
    elit_group_size=4,
    elit_read_depth=1,
    elit_write_depth=1,
    inference_budget=None,
    unconditional_inference_budget=None,
    autoguidance=False,
    # DFM
    stages_count=2,
    stage_sampling_thresholds=[0.1],
    num_steps_per_scale=[30, 10],
)


def build_parser():
    """Build argument parser."""
    d = DEFAULTS
    parser = argparse.ArgumentParser(
        description="Unified generation for SiT / DFM-SiT / ELIT-SiT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train config + eval config
  torchrun --nproc_per_node=8 generate.py \\
      --train-config experiments/train/elit_sit_xl_256.yaml \\
      --eval-config  experiments/generation/elit_sit_xl_256.yaml \\
      --ckpt path/to/ckpt.pt

  # Train config only (default eval settings)
  torchrun --nproc_per_node=8 generate.py \\
      --train-config experiments/train/sit_xl_2_256.yaml \\
      --ckpt path/to/ckpt.pt --cfg-scale 1.8

  # Pure CLI
  torchrun --nproc_per_node=8 generate.py \\
      --model ELIT-SiT-B/2 --ckpt path/to/ckpt.pt --inference-budget 0.5
""",
    )

    # Config files
    parser.add_argument("--train-config", type=str, default=None,
                        help="Path to a training YAML config. Extracts model architecture "
                             "settings (model, resolution, exp_name, ELIT/DFM/REPA params, etc.).")
    parser.add_argument("--eval-config", type=str, default=None,
                        help="Path to an evaluation YAML config. Sets sampling and output "
                             "settings (cfg_scale, num_steps, inference_budget, etc.). "
                             "CLI args override.")

    # seed
    parser.add_argument("--global-seed", type=int, default=d["global_seed"])

    # precision
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=d["tf32"],
                        help="Use TF32 matmuls (fast, slight numerical differences).")

    # logging / saving
    parser.add_argument("--ckpt", type=str, default=d["ckpt"],
                        help="Path to a checkpoint (.pt).")
    parser.add_argument("--sample-dir", type=str, default=d["sample_dir"],
                        help="Root directory for results (default: results/).")
    parser.add_argument("--exp-name", type=str, default=d["exp_name"],
                        help="Experiment name. Used as sub-folder under --sample-dir. "
                             "If not set, derived from model name.")

    # model
    parser.add_argument("--model", type=str, default=d["model"],
                        choices=list(ALL_MODELS.keys()),
                        help="Model name. Prefix determines type: SiT-*/DFM-SiT-*/ELIT-SiT-*")
    parser.add_argument("--num-classes", type=int, default=d["num_classes"])
    parser.add_argument("--encoder-depth", type=int, default=d["encoder_depth"])
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=d["resolution"])
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction,
                        default=d["fused_attn"])
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction,
                        default=d["qk_norm"])

    # REPA (disabled by default – projectors are dead weight during sampling)
    parser.add_argument("--enable-repa", action=argparse.BooleanOptionalAction,
                        default=d["enable_repa"],
                        help="Enable REPA projectors in the model (only needed if ckpt was trained with REPA).")
    parser.add_argument("--projector-embed-dims", type=str, default=d["projector_embed_dims"],
                        help="Comma-separated projector embed dims (only used when --enable-repa).")

    # vae
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default=d["vae"])

    # number of samples
    parser.add_argument("--per-proc-batch-size", type=int, default=d["per_proc_batch_size"])
    parser.add_argument("--num-fid-samples", type=int, default=d["num_fid_samples"])

    # sampling
    parser.add_argument("--mode", type=str, default=d["mode"], choices=["ode", "sde"])
    parser.add_argument("--cfg-scale", type=float, default=d["cfg_scale"])
    parser.add_argument("--path-type", type=str, default=d["path_type"],
                        choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=d["num_steps"])
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=d["heun"])
    parser.add_argument("--guidance-low", type=float, default=d["guidance_low"])
    parser.add_argument("--guidance-high", type=float, default=d["guidance_high"])

    # legacy
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=d["legacy"])

    # ELIT-specific
    parser.add_argument("--elit-max-mask-prob", type=float, default=d["elit_max_mask_prob"],
                        help="[ELIT] Maximum masking probability (training; not used in inference).")
    parser.add_argument("--elit-min-mask-prob", type=float, default=d["elit_min_mask_prob"],
                        help="[ELIT] Minimum masking probability. Defaults to max (single budget).")
    parser.add_argument("--elit-group-size", type=int, default=d["elit_group_size"],
                        help="[ELIT] Group size for token masking.")
    parser.add_argument("--elit-read-depth", type=int, default=d["elit_read_depth"],
                        help="[ELIT] Depth of the read (cross-attention) layer.")
    parser.add_argument("--elit-write-depth", type=int, default=d["elit_write_depth"],
                        help="[ELIT] Depth of the write (cross-attention) layer.")
    parser.add_argument("--inference-budget", type=float, default=d["inference_budget"],
                        help="[ELIT] Budget for ELIT masking at inference time.")
    parser.add_argument("--unconditional-inference-budget", type=float,
                        default=d["unconditional_inference_budget"],
                        help="[ELIT/CCFG] Budget for the unconditional CFG path. "
                             "When set, enables Cheap CFG: the unconditional path "
                             "runs at this (lower) budget while the conditional path "
                             "uses --inference-budget.")
    parser.add_argument("--autoguidance", action="store_true", default=d["autoguidance"],
                        help="[ELIT] Autoguidance: keep the real class label on the "
                             "'unconditional' path instead of dropping the condition. "
                             "Guidance comes from the capacity gap (budget difference) "
                             "rather than from classifier-free conditioning.")

    # DFM-specific
    parser.add_argument("--stages-count", type=int, default=d["stages_count"],
                        help="[DFM] Number of multiscale stages.")
    parser.add_argument("--stage-sampling-thresholds", type=float, nargs='+',
                        default=d["stage_sampling_thresholds"],
                        help="[DFM] Thresholds for activating each scale (len = stages_count-1).")
    parser.add_argument("--num-steps-per-scale", type=int, nargs='+',
                        default=d["num_steps_per_scale"],
                        help="[DFM] Sampling steps for each scale (len = stages_count).")

    return parser


def _normalize_key(key):
    """Convert between YAML underscore keys and argparse hyphen keys."""
    return key.replace('-', '_')


def parse_args(input_args=None):
    """
    Parse arguments with YAML + CLI support.
    Priority: CLI args  >  eval config  >  train config  >  defaults.
    """
    parser = build_parser()

    # First, do a preliminary parse to get --train-config and --eval-config
    preliminary, _ = parser.parse_known_args(input_args)

    merged = dict(DEFAULTS)

    # Layer 1: train config (lowest priority after defaults)
    # Extracts model-related settings: model, exp_name, encoder_depth, resolution, etc.
    TRAIN_KEYS_TO_IMPORT = {
        'model', 'exp_name', 'encoder_depth', 'resolution', 'num_classes',
        'fused_attn', 'qk_norm', 'path_type',
        # REPA
        'enable_repa', 'proj_coeff', 'enc_type', 'projector_embed_dims',
        # ELIT
        'elit_max_mask_prob', 'elit_min_mask_prob', 'elit_group_size',
        'elit_read_depth', 'elit_write_depth',
        # DFM
        'stages_count', 'stage_weights',
    }
    if preliminary.train_config is not None:
        with open(preliminary.train_config, 'r') as f:
            train_config = yaml.safe_load(f) or {}
        train_config = {_normalize_key(k): v for k, v in train_config.items()}
        for k, v in train_config.items():
            if k in TRAIN_KEYS_TO_IMPORT:
                merged[k] = v

    # Layer 2: eval config (overrides train config for eval-related keys)
    if preliminary.eval_config is not None:
        with open(preliminary.eval_config, 'r') as f:
            eval_yaml = yaml.safe_load(f) or {}
        eval_yaml = {_normalize_key(k): v for k, v in eval_yaml.items()}
        merged.update(eval_yaml)

    parser.set_defaults(**merged)

    # Layer 3: CLI args (highest priority — handled by argparse)
    args = parser.parse_args(input_args)

    # Derive exp_name if not set
    if args.exp_name is None:
        args.exp_name = args.model.replace("/", "-")

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
