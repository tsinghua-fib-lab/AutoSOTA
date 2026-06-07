import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# ============================================================================
# Configuration
# ============================================================================

class PPOConfig:
    # Model parameters
    num_classes = 100
    gaussian = False
    std = False
    gaussians_path = f"./reverse_stats_cov_{num_classes}.pth"

    data_path = f"./cifar10_{num_classes}_cluster.pt"
ppo_config = PPOConfig()
gaussians = torch.load(ppo_config.gaussians_path)


# ============================================================================
# Resource Loading
# ============================================================================

def load_ppo_resources(data_path: str = '',
                       device: str = 'cuda') -> Dict:
    """Load resources for PPO training using classifier"""
    logging.info("Loading PPO resources...")
    data_path = ppo_config.data_path
    ppo_data = None
    class_to_indices = {}
    if os.path.exists(data_path):
        ppo_data = torch.load(data_path, map_location='cpu')

    resources = {
        'ppo_data': ppo_data,
        'class_to_indices': ppo_data['class_to_indices'],
        'device': device
    }
    logging.info("PPO resources loaded successfully")
    return resources


# ============================================================================
# Sample noise
# ============================================================================
dists = [torch.distributions.MultivariateNormal(mean, (cov+ 1e-4*torch.eye(cov.shape[0]))) for mean, cov in zip(gaussians['means'], gaussians['covs'])]
ppo_resources = load_ppo_resources()
rates = [len(ppo_resources['class_to_indices'][class_id]) for class_id in range(ppo_config.num_classes)]
rates = np.array(rates) / sum(rates)
def sample_noise(class_labels: torch.Tensor=None, device: str='cuda', batch_size: int = 64) -> torch.Tensor:
    if class_labels is None:
        random_class_labels = np.random.choice(np.arange(ppo_config.num_classes), size=(batch_size,), p=rates)
        class_labels = torch.from_numpy(random_class_labels).to(device).long() 
    class_labels = class_labels.cpu()
    if ppo_config.gaussian:
        noise = torch.randn(class_labels.shape[0], 3*32*32).to(device)
    elif ppo_config.std:
        means, stds = gaussians["means"][class_labels], gaussians["stds"][class_labels]
        dist = Normal(means.to(device), stds.to(device))
        noise = dist.rsample()
    else:
        noise_list = []
        for cls in class_labels:
            dist = dists[cls.item()]
            noise_sample = dist.sample().to(device)
            noise_list.append(noise_sample.unsqueeze(0))
        noise = torch.cat(noise_list, dim=0)
    return noise.view(-1, 3, 32, 32)