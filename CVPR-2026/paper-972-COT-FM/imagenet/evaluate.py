import os
import argparse
import json
import numpy as np
import math
from tqdm import tqdm
from PIL import Image

import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
import torch_fidelity

from sit import SiT_models
from rf_sampler import rf_sampler

gaussians = torch.load("./mean_std_per_class_cond.pt")


def main(args):
    """
    Run sampling and evaluation.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU"
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    block_kwargs = {"fused_attn": False, "qk_norm": False}
    latent_size = args.resolution // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = True,
        **block_kwargs,
    ).to(device)
    
    try:
        state_dict = torch.load(args.ckpt, map_location=f'cuda:{device}', weights_only=False)['ema']
        model.load_state_dict(state_dict)
    except:
        state_dict = torch.load(args.ckpt, map_location=f'cuda:{device}', weights_only=False)
        model.load_state_dict(state_dict)
    model.eval()

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale should be >= 1.0"

    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"class-wise-cotfm-{model_string_name}-{ckpt_string_name}-size-{args.resolution}-" \
                  f"cfg-{args.cfg_scale}-steps-{args.num_steps}-seed-{args.global_seed}"
    eval_fid_dir = f"{args.sample_dir}/{folder_name}"
    img_folder = eval_fid_dir+'/img_dir'
    if rank == 0:
        os.makedirs(eval_fid_dir, exist_ok=True)            
        os.makedirs(img_folder, exist_ok=True)          
        print(f"Saving .png samples at {eval_fid_dir}")
    dist.barrier()

    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Using {args.num_steps}-step sampling")
    
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)

    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    
    for _ in pbar:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)
        means = gaussians["means"][y.cpu()].to(device)
        stds = gaussians["stds"][y.cpu()].to(device)
        z = torch.distributions.Normal(means, stds).sample()
        z = z.view(n, model.in_channels, latent_size, latent_size)
    
        with torch.no_grad():
            samples = rf_sampler(
                model=model, 
                latents=z,
                y=y,
                cfg_scale=args.cfg_scale,
                num_steps=args.num_steps,
            ).to(torch.float32)
            latents_scale = torch.tensor(
                [0.18125, 0.18125, 0.18125, 0.18125]
                ).view(1, 4, 1, 1).to(device)
            latents_bias = torch.tensor(
                [0., 0., 0., 0.]
                ).view(1, 4, 1, 1).to(device)
            samples = vae.decode((samples -  latents_bias) / latents_scale).sample
            samples = (samples + 1) / 2.
            samples = torch.clamp(
                255. * samples, 0, 255
                ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{eval_fid_dir}/img_dir/{index:06d}.png")
        total += global_batch_size

    dist.barrier()

    if rank == 0 and args.compute_metrics:
        print(f"Computing evaluation metrics...")
        
        metrics_dict = {}
        metrics_args = {
            'input1': img_folder,
            'cuda': True,
            'isc': True,
            'fid': True,
            'kid': False,
            'prc': False,
            'verbose': True,
        }
        if args.resolution == 256:
            metrics_args['input2'] = None
            metrics_args['fid_statistics_file'] = args.fid_statistics_file
        else:
            raise NotImplementedError

        metrics_dict = torch_fidelity.calculate_metrics(**metrics_args)
        
        fid = metrics_dict.get('frechet_inception_distance', None)
        is_mean = metrics_dict.get('inception_score_mean', None)
        is_std = metrics_dict.get('inception_score_std', None)
        
        print(f"\n===== Evaluation Results =====")
        if fid is not None:
            print(f"FID: {fid:.2f}")
        if is_mean is not None:
            print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")
            
        metrics_file = os.path.join(eval_fid_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"Metrics saved to {metrics_file}")
        
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--global-seed", type=int, default=0)

    parser.add_argument("--ckpt", type=str, required=True, help="Path to a RF checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="samples")

    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)

    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--num-steps", type=int, default=1, help="Number of sampling steps")
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    
    parser.add_argument("--compute-metrics", action="store_true", help="Compute FID and IS after sampling")
    parser.add_argument("--fid-statistics-file", type=str, default="", 
                       help="Path to pre-computed FID statistics file")

    args = parser.parse_args()
    
    main(args)

