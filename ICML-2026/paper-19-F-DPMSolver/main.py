import os
import argparse
from tqdm import tqdm
import torch
import PIL.Image
from contextlib import nullcontext
from torch_utils import distributed as dist
from sampler import load_sampler
from utils.load_model import load_model

import sys
sys.path.append("..")
sys.path.append("./")

def parse_int_list(s: str):
    """
    Parse a comma-separated list of ints and int ranges.
    """
    s = s.strip()
    if not s:
        return []
    out = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            a, b = int(a), int(b)
            if a <= b:
                out.extend(range(a, b + 1))
            else:
                out.extend(range(a, b - 1, -1))
        else:
            out.append(int(part))
    return out

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subdirs", type=str, default="samples", help="Where to save the output images")
    parser.add_argument("--seeds", type=parse_int_list, default="0-63", help="Random seeds (e.g. 1,2,5-10)")
    parser.add_argument("--NFE", type=int, default=5, help="Key parameter: neural function evaluation (NFE)")
    parser.add_argument("--order", type=int, default=1, help="Key parameter: order of ODESolver")
    parser.add_argument("--batch", type=int, default=64, help="Maximum batch size")
    parser.add_argument("--algorithm_name", type=str, default="F-DDIM", help="Name of algorithm to sample")
    parser.add_argument("--model_name", type=str, default="CIFAR10-cond", help="Name of dataset to sample")
    parser.add_argument("--rho", type=float, default=7.0, help="EDM schedule rho parameter")
    parser.add_argument("--sigma_min", type=float, default=0.002, help="EDM schedule sigma_min")
    parser.add_argument("--sigma_max", type=float, default=80.0, help="EDM schedule sigma_max")
    parser.add_argument("--gamma", type=float, default=0.0, help="Bidirectional correction weight (0=disabled, 0.15-0.25 recommended)")
    args = parser.parse_args()
    device = torch.device('cuda')

    # init and divide seeds to each gpu
    dist.init()
    seeds = args.seeds
    max_batch_size = args.batch
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network
    if args.model_name == "ImageNet64-S" or args.model_name == "ImageNet64-L" or args.model_name == "ImageNet512-XS" or args.model_name == "ImageNet512-XXL":
        model, encoder = load_model(dist, args.model_name)
    else:
        model = load_model(dist, args.model_name)

    # Other ranks follow
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    
    # use edm/edm2 or latent diffusion
    sample_dir = os.path.join("sample", args.subdirs)
    use_ldm = args.model_name == "LSUN" or args.model_name == "FFHQ"
    ctx = model.ema_scope("Plotting") if use_ldm else nullcontext()
    
    # load sampler
    sampler_class = load_sampler(args.algorithm_name, args.order, args.NFE, model, use_ldm, device, rho=args.rho, sigma_min=args.sigma_min, sigma_max=args.sigma_max)
    sampler_class.gamma = args.gamma
    sampler = sampler_class.get_sampler()

    # start sampling
    with ctx:
        for batch_seeds in tqdm(rank_batches, unit='max_batch_size', disable=(dist.get_rank() != 0)):
            torch.distributed.barrier()
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue
            rnd = StackedRandomGenerator(device, batch_seeds)
            if use_ldm:
                noise = rnd.randn([batch_size, model.model.diffusion_model.in_channels, model.model.diffusion_model.image_size, model.model.diffusion_model.image_size], device=device)
            else:
                noise = rnd.randn([batch_size, model.img_channels, model.img_resolution, model.img_resolution], device=device)
                
            class_labels = None
            if not use_ldm:
                if model.label_dim:
                    class_labels = torch.eye(model.label_dim, device=device)[rnd.randint(model.label_dim, size=[batch_size], device=device)]
            
            sampledimgs = sampler(noise, class_labels)
            
            if args.model_name == "CIFAR10-cond" or args.model_name == "CIFAR10-uncond":
                sampledimgs = sampledimgs * 0.5 + 0.5
                images_np = (sampledimgs * 255).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            elif args.model_name == "ImageNet64-S" or args.model_name == "ImageNet64-L" or args.model_name == "ImageNet512-XS" or args.model_name == "ImageNet512-XXL":
                sampledimgs = encoder.decode(sampledimgs)
                images_np = sampledimgs.permute(0, 2, 3, 1).cpu().numpy()
            elif args.model_name == "FFHQ" or args.model_name == "LSUN":
                sampledimgs = model.decode_first_stage(sampledimgs)
                sampledimgs = sampledimgs * 0.5 + 0.5
                x = sampledimgs.detach().cpu().numpy()
                images_np = (sampledimgs * 255).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            else:
                raise ValueError(f"No existing dataset {args.model_name}!")
            
            # Save images.
            for seed, image_np in zip(batch_seeds, images_np):
                image_dir = os.path.join(sample_dir, f'{seed-seed%1000:06d}')
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f'{seed:06d}.png')
                if image_np.shape[2] == 1:
                    PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
                else:
                    PIL.Image.fromarray(image_np, 'RGB').save(image_path)

    # Done.
    dist.print0('Done.')
