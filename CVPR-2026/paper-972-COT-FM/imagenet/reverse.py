#!/usr/bin/env python3
import argparse
import os
import math
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.serialization
from torch.utils.data import DataLoader

from diffusers.models import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from tqdm import tqdm

from sit import SiT_models
from rf_sampler import rf_sampler
from dataset import LMDBLatentsDataset


def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def main():
    parser = argparse.ArgumentParser("RF 64x64 Sampler (8-GPU Parallel)")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-dir", default="samples64")
    parser.add_argument("--model", choices=list(SiT_models.keys()), default="SiT-L/2")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--stop-class", type=int, default=1000)
    args = parser.parse_args()

    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    latent_size = args.resolution // 8

    # ===== Model =====
    block_kwargs = {"fused_attn": False, "qk_norm": False}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=True,
        **block_kwargs,
    ).to(device)

    torch.serialization.add_safe_globals([argparse.Namespace])
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "ema" in state:
        model.load_state_dict(state["ema"])
    else:
        model.load_state_dict(state)
    model.eval()

    # ===== VAE =====
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()

    latents_scale = torch.tensor([0.18125] * 4, device=device).view(1, 4, 1, 1)
    latents_bias = torch.zeros_like(latents_scale)

    # ===== Dataset =====
    dataset = LMDBLatentsDataset(args.data_dir, flip_prob=0.5)
    label2index = torch.load("label2index.pt")

    all_classes = sorted(label2index.keys())[: args.stop_class]

    classes_per_rank = math.ceil(len(all_classes) / world_size)
    start = rank * classes_per_rank
    end = min(start + classes_per_rank, len(all_classes))
    local_classes = all_classes[start:end]

    if rank == 0:
        print(f"[INFO] Total classes: {len(all_classes)}")
    print(f"[RANK {rank}] Processing classes {start} ~ {end - 1}")

    local_means = []
    local_stds = []
    local_class_ids = []

    for cls in tqdm(local_classes, disable=(rank != 0)):
        items = []
        for i in label2index[cls]:
            items.append(dataset.get_item(i, flip=True))
            items.append(dataset.get_item(i, flip=False))
        moments, labels = zip(*items)

        moments = torch.stack(moments).to(device, non_blocking=True)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        posterior = DiagonalGaussianDistribution(moments)

        x = posterior.mode()
        x = x * latents_scale + latents_bias

        x0 = rf_sampler(
            model=model,
            latents=x,
            y=labels,
            cfg_scale=args.cfg_scale,
            num_steps=args.num_steps,
            reverse=True,
        )

        x0_flat = x0.view(x0.shape[0], -1)
        mean = x0_flat.mean(dim=0).cpu()
        std = x0_flat.std(dim=0, unbiased=False).cpu()

        local_means.append(mean)
        local_stds.append(std)
        local_class_ids.append(cls)

    local_result = {
        "class_ids": local_class_ids,
        "means": local_means,
        "stds": local_stds,
    }

    gathered = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(local_result, gathered, dst=0)

    if rank == 0:
        all_means = {}
        all_stds = {}

        for part in gathered:
            for cid, m, s in zip(part["class_ids"], part["means"], part["stds"]):
                all_means[cid] = m
                all_stds[cid] = s

        ordered_ids = sorted(all_means.keys())
        means = torch.stack([all_means[i] for i in ordered_ids])
        stds = torch.stack([all_stds[i] for i in ordered_ids])

        save_path = f"mean_std_per_class_from_mean.pt"
        torch.save({"means": means, "stds": stds}, save_path)
        print(f"[INFO] Saved to {save_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
