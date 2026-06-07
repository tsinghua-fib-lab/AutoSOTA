# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import gc
import logging
import math
import os
from argparse import Namespace
from pathlib import Path
from typing import Iterable

import PIL.Image

import torch
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision.utils import save_image, make_grid
from training import distributed_mode
import models.rng as rng
from models.cifar_tsne import cluster_dataloader
import torchvision
from tqdm import tqdm
from training.ppo_utils import sample_noise
from training.optimal_transport import wasserstein
from torchvision.utils import save_image
logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 10
import random
import numpy as np
def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def eval_model(
    model: DistributedDataParallel,
    net_ema: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    args: Namespace,
    suffix: str = "",
    noise_sampler = None,
    rates = None,
):
    set_seed(42)
    gc.collect()
    model.train(False)
    target_samples = []
    if args.distributed:
        data_loader.sampler.set_epoch(0)

    assert args.fid_samples <= len(data_loader.dataset), (
        f"In this interface, dataset size ({len(data_loader.dataset)}) must be larger than FID samples ({args.fid_samples})."
    )
    fid_samples = math.ceil(args.fid_samples / distributed_mode.get_world_size())

    fid_metric = FrechetInceptionDistance(normalize=True).to(device=device, non_blocking=True)
    kid_metric = KernelInceptionDistance(subset_size=1000, feature=2048, normalize=True).to(device=device, non_blocking=True)
    inception_score_metric = InceptionScore(normalize=True).to(device=device, non_blocking=True)

    num_synthetic = 0
    snapshots_saved = False
    if args.output_dir:
        (Path(args.output_dir) / "snapshots").mkdir(parents=True, exist_ok=True)
    synthetic = []
    sz = 0
    for data_iter_step, (samples, _) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        sz += samples.shape[0]
        target_samples.append(samples.cpu())
        fid_metric.update(samples, real=True)  # real is always on the entire dataset
        kid_metric.update(samples, real=True)
        if num_synthetic < fid_samples:          
            model_without_ddp = model.module if isinstance(model, DistributedDataParallel) else model  

            with torch.random.fork_rng(devices=[device]):
                #per node and per step seed
                torch.manual_seed(rng.fold_in(args.seed, rng.get_rank(), data_iter_step, epoch))
                with torch.amp.autocast('cuda', enabled=False), torch.no_grad():
                    if rates is None:
                        random_class_label = torch.randint(0, 100, (samples.shape[0],), device=device)
                    else:
                        random_class_label = torch.multinomial(rates, samples.shape[0], replacement=True)
                    noise = sample_noise(noise_sampler, random_class_label, "cuda", True).view_as(samples)
                    synthetic_samples = model_without_ddp.sample(samples_shape=samples.shape, net=net_ema, device=device, e=noise)
            torch.cuda.synchronize()

            # Scaling to [0, 1] from [-1, 1]
            synthetic_samples = torch.clamp(
                synthetic_samples * 0.5 + 0.5, min=0.0, max=1.0
            )
            synthetic_samples = torch.floor(synthetic_samples * 255)

            synthetic_samples = synthetic_samples.to(torch.float32) / 255.0
            synthetic.append(synthetic_samples.cpu())
            if num_synthetic + synthetic_samples.shape[0] > fid_samples:
                synthetic_samples = synthetic_samples[: fid_samples - num_synthetic]
            fid_metric.update(synthetic_samples, real=False)
            inception_score_metric.update(synthetic_samples)
            kid_metric.update(synthetic_samples, real=False)
            num_synthetic += synthetic_samples.shape[0]
            if not snapshots_saved and args.output_dir:
                save_image(
                    synthetic_samples,
                    fp=Path(args.output_dir)
                    / "snapshots"
                    / f"{epoch}_{data_iter_step}{suffix}.png",
                )
                snapshots_saved = True

            if args.save_fid_samples and args.output_dir:

                images_np = (
                    (synthetic_samples * 255.0)
                    .clip(0, 255)
                    .to(torch.uint8)
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )
                for batch_index, image_np in enumerate(images_np):
                    image_dir = Path(args.output_dir) / f"fid_samples{suffix}"
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = (
                        image_dir
                        / f"{distributed_mode.get_rank()}_{data_iter_step}_{batch_index}.png"
                    )
                    PIL.Image.fromarray(image_np, "RGB").save(image_path)

        if not args.compute_fid:
            return {}

        if (data_iter_step + 1) % PRINT_FREQUENCY == 0 or data_iter_step == len(data_loader) - 1:
            # Sync fid metric to ensure that the processes dont deviate much.
            gc.collect()
            running_fid = fid_metric.compute()
            logger.info(f"Evaluating: current batch {samples.shape[0]},  [{num_synthetic}/{fid_samples}], running fid {running_fid}")
        if args.test_run:
            break
    metrics = {"fid": float(fid_metric.compute().detach().cpu())}
    return metrics
