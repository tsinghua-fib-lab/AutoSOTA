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
    noise_dim = 3072  # 3 * 32 * 32
    hidden_dim = 256
    num_classes = 100
    embedding_dim = 64

    # PPO training parameters
    learning_rate = 1e-4
    batch_size = 1024
    sample_times = 1
    ppo_epochs = 10
    clip_eps = 0.2
    entropy_coef = 0.0

    # Reward scale
    reward_scale = 20.0

    # Warm-up
    warmup_steps = 50  # Gradually transition from N(0,1)

    # Regularization: keep distribution near N(0,1)
    kl_coeff = 1e-3

    # Training parameters
    max_rounds_per_class = 50
    device = 'cuda'

    # Logging
    log_interval = 5
    save_interval = 10

    # Diversity calculation
    min_class_samples = 3
    use_adaptive_diversity = False

    # Std clamp for stability
    min_std = 0.0001
    max_std = 0.3
    min_stdstd = 0.0001
    max_stdstd = 0.3


    num_per_distribution = 128
    steps_integral = 1
    
    gaussian_path = "noise_distributions.pth"
    # gaussian_path = "no"
ppo_config = PPOConfig()


# ============================================================================
# Classifier Loading
# ============================================================================

def load_cifar10_classifier(device: str = 'cuda') -> nn.Module:
    """Load CIFAR-10 trained classifier for reward computation"""
    model = models.resnet18(num_classes=10)
    checkpoint_path = './assets/cifar10_resnet18.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        logging.info(f"Loaded CIFAR-10 classifier from {checkpoint_path}")
    else:
        logging.warning(f"No pre-trained classifier found at {checkpoint_path}")
        logging.warning("Using untrained classifier - please train one first!")
        logging.warning("Run: python train_classifier.py")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ============================================================================
# Resource Loading
# ============================================================================

def load_ppo_resources(data_path: str = '/path/to/cifar10_ppo_data.pt',
                       device: str = 'cuda') -> Dict:
    """Load resources for PPO training using classifier"""
    logging.info("Loading PPO resources...")
    classifier = load_cifar10_classifier(device)

    ppo_data = None
    class_to_indices = {}
    if os.path.exists(data_path):
        ppo_data = torch.load(data_path, map_location='cpu')

    resources = {
        'classifier': classifier,
        'ppo_data': ppo_data,
        'class_to_indices': ppo_data['class_to_indices'],
        'device': device
    }
    logging.info("PPO resources loaded successfully")
    return resources

if os.path.exists(ppo_config.gaussian_path):
    gaussians = torch.load(ppo_config.gaussian_path)
    dists = [torch.distributions.MultivariateNormal(gaussians["means"][i], gaussians["covs"][i]) for i in range(ppo_config.num_classes)]
else:
    gaussians = None
noises = torch.load("/path/to/noises_ours.pth")
def sample_noise(noise_sampler: NoiseSampler, class_labels: torch.Tensor, device: str, deterministic: bool = False):
    if gaussians is None:
        return noises.to(device)
    class_labels = class_labels.cpu()
    noise = []
    for class_label in class_labels:
        noise.append(dists[class_label].rsample().to(device))
    noise = torch.stack(noise, dim=0)

    return noise


from torchvision.utils import save_image
def _save_grid(imgs: torch.Tensor, path: str, nrow: int = 4):
    """Save a grid image; expects imgs as BCHW, any float range."""
    save_image(imgs, path, nrow=nrow, normalize=True, value_range=(0, 1))

def generate_samples(model, args, rates):
    net = model.net_ema
    samples = []
    for class_id in tqdm(range(ppo_config.num_classes)):
        noises = sample_noise(None, torch.full((args.batch_size,), class_id, dtype=torch.long), args.device, deterministic=True)
        noises = noises.view(noises.shape[0], 3, 32, 32)
        sample = model.sample(samples_shape=(noises.shape[0], 3, 32, 32), net=net, device=args.device, e=noises)
        sample = torch.clamp(sample * 0.5 + 0.5, min=0.0, max=1.0)
        samples.append(sample)
    samples = torch.cat(samples, dim=0)
    _save_grid(samples, "./1_mf_ours.png", nrow=16)





