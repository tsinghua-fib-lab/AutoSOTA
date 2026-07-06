# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Script for calculating Frechet Inception Distance (FID)."""

import os
import click
import tqdm
import pickle
import numpy as np
import scipy.linalg
import torch
import sys
sys.path.append("..")
sys.path.append("./")

import dnnlib
import argparse
from torch_utils import distributed as dist
from torch_utils import dataset

#----------------------------------------------------------------------------

def calculate_inception_stats(
    image_path, num_expected=None, seed=0, max_batch_size=64,
    num_workers=3, prefetch_factor=2, device=torch.device('cuda'),
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load Inception-v3 model (torchvision-based fallback when NVIDIA CDN is unavailable).
    dist.print0('Loading Inception-v3 model...')
    try:
        detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        with dnnlib.util.open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
            detector_net = pickle.load(f).to(device)
    except Exception as e:
        dist.print0(f'NVIDIA detector unavailable ({e}), using torchvision InceptionV3...')
        from torchvision.models import inception_v3
        import torch.nn as nn
        class InceptionFeatureExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = inception_v3(weights='IMAGENET1K_V1', transform_input=False)
                self.model.eval()
            def forward(self, x, return_features=False, **kwargs):
                if x.dtype != torch.float32:
                    x = x.to(torch.float32)
                if x.shape[-1] != 299:
                    x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
                if x.min() < 0:
                    x = (x + 1) / 2
                x = x - x.new_tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
                x = x / x.new_tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
                if return_features:
                    x = self.model.Conv2d_1a_3x3(x)
                    x = self.model.Conv2d_2a_3x3(x)
                    x = self.model.Conv2d_2b_3x3(x)
                    x = self.model.maxpool1(x)
                    x = self.model.Conv2d_3b_1x1(x)
                    x = self.model.Conv2d_4a_3x3(x)
                    x = self.model.maxpool2(x)
                    x = self.model.Mixed_5b(x)
                    x = self.model.Mixed_5c(x)
                    x = self.model.Mixed_5d(x)
                    x = self.model.Mixed_6a(x)
                    x = self.model.Mixed_6b(x)
                    x = self.model.Mixed_6c(x)
                    x = self.model.Mixed_6d(x)
                    x = self.model.Mixed_6e(x)
                    x = self.model.Mixed_7a(x)
                    x = self.model.Mixed_7b(x)
                    x = self.model.Mixed_7c(x)
                    x = self.model.avgpool(x)
                    x = torch.flatten(x, 1)
                    return x
                return self.model(x)
        detector_net = InceptionFeatureExtractor().to(device)
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048

    # List images.
    dist.print0(f'Loading images from "{image_path}"...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)
    if num_expected is not None and len(dataset_obj) < num_expected:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_expected}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    # if dist.get_rank() == 0:
    #     torch.distributed.barrier()

    # Divide images into batches.
    num_batches = ((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor)

    # Accumulate statistics.
    dist.print0(f'Calculating statistics for {len(dataset_obj)} images...')
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    for images, _labels in tqdm.tqdm(data_loader, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features

    # Calculate grand totals.
    torch.distributed.all_reduce(mu)
    torch.distributed.all_reduce(sigma)
    mu /= len(dataset_obj)
    sigma -= mu.ger(mu) * len(dataset_obj)
    sigma /= len(dataset_obj) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()

#----------------------------------------------------------------------------

def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------

def calc(image_path, ref_path, num_expected, seed, batch):
    """Calculate FID for a given set of images."""
    # torch.multiprocessing.set_start_method('spawn')
    # if not dist.is_initialized():
    #     dist.init()

    dist.print0(f'Loading dataset reference statistics from "{ref_path}"...')
    ref = None
    if dist.get_rank() == 0:
        with dnnlib.util.open_url(ref_path) as f:
            ref = dict(np.load(f))

    mu, sigma = calculate_inception_stats(image_path=image_path, num_expected=num_expected, seed=seed, max_batch_size=batch)
    dist.print0('Calculating FID...')
    if dist.get_rank() == 0:
        fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
        # print(f'{fid:g}')
    torch.distributed.barrier()
    return fid

#----------------------------------------------------------------------------

def ref(dataset_path, dest_path, batch):
    """Calculate dataset reference statistics needed by 'calc'."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    mu, sigma = calculate_inception_stats(image_path=dataset_path, max_batch_size=batch)
    dist.print0(f'Saving dataset reference statistics to "{dest_path}"...')
    if dist.get_rank() == 0:
        if os.path.dirname(dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        np.savez(dest_path, mu=mu, sigma=sigma)

    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subdirs", type=str, default="samples", help="Where to save the output images")
    parser.add_argument("--ref_path", type=str, required=True, help="Reference path for fid")
    args = parser.parse_args()
        
    dist.init()
    ref_path = args.ref_path
    sample_dir = os.path.join("sample", args.subdirs)
    print(sample_dir, ref_path)
    fid_score_ref = calc(sample_dir, ref_path, num_expected=50000, seed=0, batch=64)
    print('fid: ', fid_score_ref)

#----------------------------------------------------------------------------
