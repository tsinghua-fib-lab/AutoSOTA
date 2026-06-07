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
    num_classes_test = 100
    embedding_dim = 32
    num_reflow_samples = 600000

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
    max_std = 0.5
    min_stdstd = 0.01
    max_stdstd = 0.5


    num_per_distribution = 128
    steps_integral = 1
    # single_class_id = 1

    reflow_data_path_0 = "./ImageGeneration/logs/1_rectified_flow/eval/ckpt_6_reflow_data_merged_0.pth"
    flow_model_path = "./ImageGeneration/logs/1_rectified_flow/checkpoints-meta/checkpoint_ot_no_cluster_longer.pth"
    gaussian = False
    std = False
    gaussians_path = "./ImageGeneration/logs/1_rectified_flow/eval/reverse_stats_cov_100_5.pth"
    # gaussians_path = "./ImageGeneration/logs/1_rectified_flow/eval/reverse_stats.pth"

    data_path = f"./cifar10_{num_classes}_cluster.pt"
ppo_config = PPOConfig()
gaussians = torch.load(ppo_config.gaussians_path)

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

def load_ppo_resources(data_path: str = './cifar10_ppo_data.pt',
                       device: str = 'cuda') -> Dict:
    """Load resources for PPO training using classifier"""
    logging.info("Loading PPO resources...")
    classifier = load_cifar10_classifier(device)
    data_path = ppo_config.data_path
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

dists = [torch.distributions.MultivariateNormal(gaussians["means"][i], gaussians["covs"][i]) for i in range(ppo_config.num_classes)]
def sample_noise(noise_sampler: NoiseSampler, class_labels: torch.Tensor, device: str, deterministic: bool = False):
    class_labels = class_labels.cpu()
    if ppo_config.gaussian:
        noise = torch.randn(class_labels.shape[0], 3*32*32).to(device)
    elif ppo_config.std:
        means, stds = gaussians["means"][class_labels], gaussians["stds"][class_labels]
        dist = Normal(means.to(device), stds.to(device))
        noise = dist.rsample()
    else:
        noise = []
        for class_label in class_labels:
            noise.append(dists[class_label.item()].rsample().to(device))
        noise = torch.stack(noise, dim=0)
    return noise


# ============================================================================
# Utility Functions
# ============================================================================

def log_training_progress(round_idx: int,
                          target_class: int,
                          metrics: Dict[str, float],
                          config: PPOConfig):
    if round_idx % config.log_interval == 0:
        logging.info(
            f"Round {round_idx:3d} | Class {target_class} | "
            f"Reward: {metrics['avg_reward']:6.2f} ± {metrics['reward_std']:5.2f} | "
            f"Loss: {metrics['policy_loss']:8.4f} | "
            f"Adv: {metrics['avg_advantage']:6.3f}"
        )

def save_checkpoint(policy: NoiseSampler,
                    round_idx: int,
                    target_class: int,
                    save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f'noise_sampler_round_{round_idx}_class_{target_class}.pth')
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'round': round_idx,
        'class': target_class
    }, checkpoint_path)
    logging.info(f"Checkpoint saved: {checkpoint_path}")




