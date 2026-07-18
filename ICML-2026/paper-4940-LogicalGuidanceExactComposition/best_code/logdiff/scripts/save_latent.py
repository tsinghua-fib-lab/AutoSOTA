# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


import numpy as np
import argparse
import os
from diffusers.models import AutoencoderKL
from tqdm import tqdm

from transformers import CLIPModel
from torchvision import transforms
from PIL import Image

from datasets.celeba import CelebADataset



def transformations(encoder_type:str,image_size:int,dataset:str):
    if encoder_type == "vae":
        transform_list = [
            transforms.Resize((image_size,image_size)), 
            transforms.ToTensor()
        ]
        if dataset == "celeba":
            transform_list += [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        transform = transforms.Compose(transform_list)
    elif encoder_type == "clip":
        transform = transforms.Compose([
                transforms.Resize(224, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                lambda image: image.convert("RGB"),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711), inplace=True)
            ])
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    return transform



def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")



    # Setup a feature folder:
    if rank == 0:
        os.makedirs(args.features_path, exist_ok=True)
        for split in ['train', 'val']:
            os.makedirs(os.path.join(args.features_path, f"{args.encoder}_{split}_features"), exist_ok=True)

    if args.encoder == "vae":
        encoder = AutoencoderKL.from_pretrained(f"black-forest-labs/FLUX.1-schnell",cache_dir='checkpoints',subfolder='vae').to(device)
    elif args.encoder == "clip":
        encoder = CLIPModel.from_pretrained("openai/clip-vit-large-patch14",cache_dir='checkpoints',device_map = device)
    
    transform = transformations(args.encoder,args.image_size,args.dataset)


    if args.dataset == "celeba":
        train_dataset = CelebADataset(args.data_path, split='train',transforms=transform)
        val_dataset = CelebADataset(args.data_path, split='val',transforms=transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    #Create a config for the dataloader and the sampler
    sampler_config = dict(
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    dataloader_config = dict(
        batch_size = args.global_batch_size // dist.get_world_size(),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    val_sampler = DistributedSampler(val_dataset,**sampler_config)
    val_loader = DataLoader(val_dataset,sampler=val_sampler, **dataloader_config)

    total = 0

    for split,_dataset in [("train",train_dataset),("val",val_dataset)]:
        dataset_sampler = DistributedSampler(_dataset,**sampler_config)
        dataset_loader = DataLoader(_dataset,sampler=dataset_sampler, **dataloader_config)
        for batch in tqdm(dataset_loader):
            x, y,idx  = batch['X'].to(device),batch['label'].to(device),batch['idx']
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                if args.encoder == "clip":
                    x = encoder.get_image_features(pixel_values=x)
                else:
                   x = encoder.encode(x).latent_dist.sample().mul_(encoder.config.scaling_factor)
            x = x.detach().cpu().numpy()    # (1, 4, 16, 16)
            y = y.detach().cpu().numpy()   # (1,)
            for i in range(idx.size(0)):
                np.save(f'{args.features_path}/{args.encoder}_{split}_features/{idx[i]:06d}.npy', x[i])

if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    """
    CUDA_VISIBLE_DEVICES=1 torchrun --master_port=25670 logdiff/scripts/save_latent.py --encoder=vae --dataset=celeba --data-path=/research/hal-datastore/datasets/original/ --features-path=data/celeba
    CUDA_VISIBLE_DEVICES=2 torchrun --master_port=25671 logdiff/scripts/save_latent.py --encoder=clip --dataset=celeba --data-path=/research/hal-datastore/datasets/original/ --features-path=data/celeba
    CUDA_VISIBLE_DEVICES=3 torchrun --master_port=25672 logdiff/scripts/save_latent.py --encoder=clip --dataset=waterbirds --data-path=/research/hal-gaudisac/Diffusion/controllable-generation/ --features-path=data/waterbirds
    CUDA_VISIBLE_DEVICES=4 torchrun --master_port=25673 logdiff/scripts/save_latent.py --encoder=vae --dataset=waterbirds --data-path=/research/hal-gaudisac/Diffusion/controllable-generation/ --features-path=data/waterbirds
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str,default='/research/hal-datastore/datasets/original/')
    parser.add_argument("--features-path", type=str, default="data/celeba")
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, choices=[64,128,256, 512], default=64)
    parser.add_argument("--encoder", type=str, choices=["vae", "clip"], default="vae")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dataset", type=str, choices=["celeba", "waterbirds"], default="celeba")
    args = parser.parse_args()
    main(args)
