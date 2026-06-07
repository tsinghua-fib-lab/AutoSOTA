# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models."""

import gc
import io
import os
import time
import random

import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image

# Model imports
from models import ddpm, ncsnv2, ncsnpp
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from tqdm import tqdm
# Training imports
import losses
import sampling
import datasets
import evaluation
import likelihood
import sde_lib
import ppo_utils
from my_utils import save_checkpoint, restore_checkpoint


from absl import flags
FLAGS = flags.FLAGS


# ============================================================================
# Helper Functions
# ============================================================================

def setup_directories(workdir, eval_folder=None):
    """Create necessary directories for training/evaluation."""
    dirs = {}
    if eval_folder:
        dirs['eval'] = os.path.join(workdir, eval_folder)
        tf.io.gfile.makedirs(dirs['eval'])
    else:
        dirs['samples'] = os.path.join(workdir, "samples")
        dirs['tensorboard'] = os.path.join(workdir, "tensorboard")
        dirs['checkpoints'] = os.path.join(workdir, "checkpoints")
        dirs['checkpoints_meta'] = os.path.join(workdir, "checkpoints-meta")
        for dir_path in dirs.values():
            tf.io.gfile.makedirs(dir_path)
    return dirs


def initialize_model_state(config):
    """Initialize model, optimizer, EMA, and state dict."""
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    return state, score_model, ema, optimizer


def setup_sde(config, noise_sampler=None):
    """Configure SDE based on config settings."""
    sde_type = config.training.sde.lower()
    if sde_type == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min,
                            beta_max=config.model.beta_max,
                            N=config.model.num_scales)
        sampling_eps = 1e-3
    elif sde_type == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min,
                               beta_max=config.model.beta_max,
                               N=config.model.num_scales)
        sampling_eps = 1e-3
    elif sde_type == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min,
                            sigma_max=config.model.sigma_max,
                            N=config.model.num_scales)
        sampling_eps = 1e-5
    elif sde_type == 'rectified_flow':
        sde = sde_lib.RectifiedFlow(
            init_type=config.sampling.init_type,
            noise_scale=config.sampling.init_noise_scale,
            use_ode_sampler=config.sampling.use_ode_sampler,
            sigma_var=getattr(config.sampling, 'sigma_variance', None),
            ode_tol=getattr(config.sampling, 'ode_tol', None),
            sample_N=getattr(config.sampling, 'sample_N', None),
            noise_sampler=noise_sampler
        )
        sampling_eps = 1e-3
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")
    return sde, sampling_eps


def wait_for_checkpoint(checkpoint_path, max_retries=3):
    """Wait for checkpoint file and load with retries."""
    waiting_message_printed = False
    while not tf.io.gfile.exists(checkpoint_path):
        if not waiting_message_printed:
            logging.warning(f"Waiting for checkpoint: {checkpoint_path}")
            waiting_message_printed = True
        time.sleep(60)
    wait_times = [60, 60, 120]
    for i in range(max_retries):
        try:
            return restore_checkpoint(checkpoint_path, state, device=config.device)
        except:
            if i < max_retries - 1:
                time.sleep(wait_times[i])
            else:
                raise


# ---- New tiny utilities to avoid duplicated inverse-scaling on images ---- #

def _ensure_01(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is in [0,1]; if it looks like [-1,1], map back to [0,1]."""
    if x.min() < -0.1 or x.max() > 1.1:
        x = (x + 1) / 2
    return x.clamp(0.0, 1.0)


def _save_grid(imgs: torch.Tensor, path: str, nrow: int = 4):
    """Save a grid image; expects imgs as BCHW, any float range."""
    imgs01 = _ensure_01(imgs)
    grid = make_grid(imgs01, nrow=nrow, padding=0)
    save_image(grid, path, nrow=nrow, normalize=True, value_range=(0, 1))




# ============================================================================
# Training Dataset Function
# ============================================================================


from optimal_transport import OTPlanSampler
import torchvision.transforms.functional as TF
otplansampler = OTPlanSampler(method='exact')
def create_training_dataset(ppo_config, config, ppo_resources, sde, gaussians=None, REFLOW=False):
    """Create training dataset with OT + noise_sampler."""
    if REFLOW:
        p = np.random.random()
        reflow_data = torch.load(ppo_config.reflow_data_path_0)
        dataset = torch.utils.data.TensorDataset(reflow_data[0][1], reflow_data[0][0])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, drop_last=True)
        return dataloader
    datasets = []
    sde.deterministic = True
    logging.info("Creating training dataset with OT + noise_sampler")
    for class_id in range(ppo_config.num_classes):
        sde.set_target_class(class_id)
        
        if not ppo_config.gaussian:
            # target = ppo_resources['ppo_data']['images'][ppo_resources['class_to_indices'][class_id]].to(config.device)
            target = ppo_resources['ppo_data']['images'][ppo_resources['class_to_indices'][class_id]].to(config.device)
            target = torch.stack([
                TF.hflip(img) if torch.rand(1).item() < 0.5 else img
                for img in target
            ])
            if ppo_config.std:
                dist = torch.distributions.Normal(gaussians['means'][class_id].to(config.device), gaussians['stds'][class_id].to(config.device))
                noise = dist.sample((len(target), )).view(-1, 3, 32, 32)
            # gaussians['covs'][class_id] += torch.eye(gaussians['covs'][class_id].size(0)) * 1e-5
            else:
                dist = torch.distributions.MultivariateNormal(gaussians['means'][class_id].to(config.device), gaussians['covs'][class_id].to(config.device))
                noise = dist.sample((len(target), )).view(-1, 3, 32, 32)
        else:
            idx = torch.randint(0, len(ppo_resources['ppo_data']['images']), (500,))
            target = ppo_resources['ppo_data']['images'][idx].to(config.device)
            noise = torch.randn_like(target)
        # idx = torch.randint(0, 50000, (len(target) * 4, ))
        # noise = noises[idx].to(config.device).view(-1, 3, 32, 32)
        # noise, target = otplansampler.sample_plan_with_scipy(noise, target)
        # noise, target = noise.reshape(-1, 3, 32, 32), target.reshape(-1, 3, 32, 32)
        noise, target = otplansampler.sample_plan(noise, target, sample_size=500)
        # print(noise.shape, target.shape)
        # exit()
        dataset = torch.utils.data.TensorDataset(target.cpu(), noise.cpu())
        datasets.append(dataset)
    datasets = torch.utils.data.ConcatDataset(datasets)
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=config.training.batch_size, shuffle=True, drop_last=True)
    sde.deterministic = False
    return dataloader

# ============================================================================
# Training Function
# ============================================================================

def train(config, workdir):
    """Runs the training pipeline."""
    dirs = setup_directories(workdir)
    writer = tensorboard.SummaryWriter(dirs['tensorboard'])

    state, score_model, ema, optimizer = initialize_model_state(config)
    
    
    ppo_resources = ppo_utils.load_ppo_resources()
    ppo_config = _setup_ppo_config(config)
    state = _load_training_checkpoint(config, workdir, state, ppo_config=ppo_config)
    initial_step = int(state['step'])
    eval_dir = '/path/to/eval_ppo_training'
    train_ds, eval_ds, _ = datasets.get_dataset(
        config, uniform_dequantization=config.data.uniform_dequantization
    )
    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    scaler = datasets.get_data_scaler(config) # e.g. map to [-1,1]
    inverse_scaler = datasets.get_data_inverse_scaler(config)  # kept for other parts

    noise_sampler = _initialize_noise_sampler(ppo_config, config)
    sde, sampling_eps = setup_sde(config, noise_sampler)

    train_step_fn, eval_step_fn = _setup_step_functions(config, sde)

    sampling_fn = None
    if config.training.snapshot_sampling:
        sampling_shape = (16, config.data.num_channels,
                          config.data.image_size, config.data.image_size)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape,
                                               inverse_scaler, sampling_eps)

    logging.info(f"Starting training loop at step {initial_step}")
    gaussians = torch.load(ppo_config.gaussians_path)
    # train_ds = create_training_dataset(ppo_config, config, ppo_resources, sde, gaussians=gaussians, REFLOW=False)
    for step in tqdm(range(initial_step, config.training.n_iters + 1)):
        # _generate_class_grid_samples(eval_dir, ppo_config, sde, sampling_fn, score_model, inverse_scaler, step)
        # debug(ppo_config, config, sde, noise_sampler, sampling_fn, score_model)
        # batch = _prepare_batch(next(train_iter), config.device, scaler)
        train_ds = create_training_dataset(ppo_config, config, ppo_resources, sde, gaussians=gaussians, REFLOW=False)
        loss = 0.0
        sz = 0
        
        if step != initial_step:
            for batch in tqdm(train_ds): # batch should be (target, noise) with OT + noise_sampler
                loss += train_step_fn(state, batch)
                sz += 1
            loss /= sz
            if step % config.training.log_freq == 0:
                logging.info(f"step: {step}, training_loss: {loss.item():.5e}")
                writer.add_scalar("training_loss", loss, step)


        if step == initial_step:
                _generate_class_grid_samples(eval_dir, ppo_config, sde, sampling_fn, score_model, inverse_scaler, step)
            
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            _generate_class_grid_samples(eval_dir, ppo_config, sde, sampling_fn, score_model, inverse_scaler, step)
            
            checkpoint_meta_path = os.path.join(dirs['checkpoints_meta'], "checkpoint_cov_6_longer.pth")
            save_checkpoint(checkpoint_meta_path, state)

        if step != 0 and (step % config.training.snapshot_freq == 0 or
                        step == config.training.n_iters):
            _save_checkpoint_and_samples(step, state, score_model, ema, config,
                                        dirs, sampling_fn)


def _load_training_checkpoint(config, workdir, state, ppo_config=None):
    """Load existing checkpoint or resume from meta checkpoint."""
    checkpoint_meta_path = os.path.join(workdir, "checkpoints-meta", "best.pth")
    existing_checkpoint_path = './logs/1_rectified_flow/checkpoints/checkpoint_8.pth'
    if ppo_config is not None:
        state = restore_checkpoint(ppo_config.flow_model_path, state, config.device)
        logging.info(f"Loaded pre-trained flow model from {ppo_config.flow_model_path}")
        return state
    if os.path.exists(existing_checkpoint_path) and not os.path.exists(checkpoint_meta_path):
        state = restore_checkpoint(existing_checkpoint_path, state, config.device)
        logging.info(f"Loaded existing checkpoint from {existing_checkpoint_path}")
    else:
        state = restore_checkpoint(checkpoint_meta_path, state, config.device)
        logging.info(f"Loaded checkpoint from {checkpoint_meta_path}")
    return state


def _setup_step_functions(config, sde):
    """Create training and evaluation step functions."""
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting

    train_step_fn = losses.get_step_fn(
        sde, train=True, optimize_fn=optimize_fn,
        reduce_mean=reduce_mean, continuous=continuous,
        likelihood_weighting=likelihood_weighting
    )
    eval_step_fn = losses.get_step_fn(
        sde, train=False, optimize_fn=optimize_fn,
        reduce_mean=reduce_mean, continuous=continuous,
        likelihood_weighting=likelihood_weighting
    )
    return train_step_fn, eval_step_fn


def _prepare_batch(batch_data, device, scaler):
    """Convert and normalize batch data."""
    batch = torch.from_numpy(batch_data['image']._numpy()).to(device).float()
    batch = batch.permute(0, 3, 1, 2) #(B, H, W, C) -> (B, C, H, W)
    batch = scaler(batch)
    return batch


def _save_checkpoint_and_samples(step, state, score_model, ema, config, dirs, sampling_fn):
    """Save checkpoint and optionally generate samples."""
    save_step = step // config.training.snapshot_freq
    checkpoint_path = os.path.join(dirs['checkpoints'], f'checkpoint_{save_step}.pth')
    save_checkpoint(checkpoint_path, state)

    if config.training.snapshot_sampling and sampling_fn:
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        sample, n = sampling_fn(score_model)
        ema.restore(score_model.parameters())

        sample_dir = os.path.join(dirs['samples'], f"iter_{step}")
        tf.io.gfile.makedirs(sample_dir)

        nrow = int(np.sqrt(sample.shape[0]))
        image_grid = make_grid(_ensure_01(sample), nrow, padding=2)

        with tf.io.gfile.GFile(os.path.join(sample_dir, "sample.png"), "wb") as fout:
            # save_image accepts file-like; image_grid is already grid tensor in [0,1]
            save_image(image_grid, fout)

        sample_np = np.clip(_ensure_01(sample).permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        with tf.io.gfile.GFile(os.path.join(sample_dir, "sample.np"), "wb") as fout:
            np.save(fout, sample_np)


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate(config, workdir, eval_folder="eval"):
    """Main evaluation function with mode selection."""
    if config.sampling.init_type == 'noise_sampler':
        logging.info("Running PPO training mode")
        return evaluate_with_ppo(config, workdir, eval_folder)
    else:
        logging.info("Running original evaluation mode")
        return evaluate_original(config, workdir, eval_folder)


def find_reverse(sde, ppo_config, ppo_resources, score_model):
    means = []
    covs = []
    t = ppo_config.num_classes
    for target_class in tqdm(range(t)):
        images = ppo_resources['ppo_data']['images'][ppo_resources['class_to_indices'][target_class]].to(ppo_config.device)
        noises_flat = []
        for i in tqdm(range(0, len(images), 32)):
            batch = images[i:i+32]
            noises = sde.ode(batch, score_model, reverse=True)  # (N, C, H, W)
            noises_flat.append(noises.view(noises.size(0), -1).detach().cpu())  # (N, D)
        noises_flat = torch.cat(noises_flat, dim=0)
        mean = noises_flat.mean(dim=0)
        
        cov = torch.cov(noises_flat.T)  
        cov += torch.eye(cov.size(0)) * 1e-5
        means.append(mean)
        covs.append(cov)
        
        
    means = torch.stack(means, dim=0)
    covs = torch.stack(covs, dim=0)
    print("means shape:", means.shape)
    print("cov shape:", covs[0].shape)

    torch.save({'means': means, 'covs': covs}, f'./logs/1_rectified_flow/eval/reverse_stats_cov_100_6.pth')
    exit()
# ============================================================================
# PPO Evaluation
# ============================================================================

def evaluate_with_ppo(config, workdir, eval_folder="eval"):
    """Evaluate with PPO training for noise sampler."""
    EVAL_ONLY = False
    REFLOW = False
    EVAL_TEST_DATASET_FID = False
    GENERATE = False
    REVERSE = False

    eval_dir = os.path.join(workdir, eval_folder)
    tf.io.gfile.makedirs(eval_dir)

    logging.info("=== EVALUATION ONLY MODE ===" if EVAL_ONLY else "=== Starting PPO Training Mode ===")

    ppo_resources = ppo_utils.load_ppo_resources()
    ppo_config = _setup_ppo_config(config)
    noise_sampler = _initialize_noise_sampler(ppo_config, config)
    if not EVAL_ONLY:
        ppo_trainer = ppo_utils.PPOTrainer(noise_sampler, ppo_config)

    state, score_model, ema = _setup_model_for_eval(config, workdir)
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)  # kept for API compatibility

    sde, sampling_eps = setup_sde(config, noise_sampler)
    sampling_fn = _setup_sampling_function(config, sde, inverse_scaler, sampling_eps)

    inception_model = None

    begin_ckpt = config.eval.begin_ckpt
    logging.info(f"Begin checkpoint: {begin_ckpt}")
    gaussians = torch.load(ppo_config.gaussians_path)

    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
        state = _load_training_checkpoint(config, workdir, state, ppo_config=ppo_config)
        ema.copy_to(score_model.parameters())
        logging.info(f"Loaded checkpoint {ckpt}")
        if config.eval.enable_sampling:
            if EVAL_ONLY:
                _compute_train_dataset_fid(
                    config, ppo_config, sampling_fn, score_model,
                    sde, noise_sampler, ppo_resources
                )
            elif REFLOW:
                _collect_reflow_data(
                    config, eval_dir, ckpt, noise_sampler, sde,
                    sampling_fn, score_model, ppo_config, ppo_resources,
                )
            elif EVAL_TEST_DATASET_FID:
                _compute_test_dataset_fid(
                    config, ppo_config, sampling_fn, score_model,
                    sde, noise_sampler
                )
            elif GENERATE:
                _generate_class_grid_samples(eval_dir, ppo_config, sde, sampling_fn,
                                 score_model, inverse_scaler, f"1_rf_{config.sampling.sample_N}")
            elif REVERSE:
                find_reverse(sde, ppo_config, ppo_resources, score_model)
            else:
                raise NotImplementedError("No evaluation mode is selected.")

            _generate_final_evaluation_samples(
                eval_dir, ppo_config, sde, sampling_fn,
                score_model, inverse_scaler
            )


def _collect_reflow_data(config, eval_dir, ckpt, noise_sampler, sde,
                         sampling_fn, score_model, ppo_config, ppo_resources):
    round = ppo_config.num_reflow_samples // ppo_config.batch_size
    noises_images = []
    torch.manual_seed(8)
    torch.cuda.manual_seed(9)
    a = torch.randint(0, 1, (10,))
    for _ in tqdm(range(round)):
        random_class_labels = torch.randint(0, ppo_config.num_classes, (ppo_config.batch_size,), device=ppo_config.device)
        with torch.no_grad():
            noise_sample = ppo_utils.sample_noise(noise_sampler, random_class_labels, "cuda", deterministic=True)
            noise_for_sampling = noise_sample.view(config.eval.batch_size, 3, 32, 32)
            sde.set_noise(noise_for_sampling)
            samples, n = sampling_fn(score_model, z=noise_for_sampling)
            noises_images.append((noise_for_sampling.cpu(), samples.cpu()))
    noises_images = [(torch.cat([x[0] for x in noises_images], dim=0),
                      torch.cat([x[1] for x in noises_images], dim=0))]
    torch.save(noises_images, os.path.join(eval_dir, f"ckpt_{ckpt}_reflow_data_8.pth"))
    logging.info(f"Saved reflow data for checkpoint {ckpt} at {eval_dir}")
    exit()

def _setup_ppo_config(config):
    """Initialize PPO configuration."""
    ppo_config = ppo_utils.PPOConfig()
    ppo_config.batch_size = config.eval.batch_size
    ppo_config.device = config.device
    ppo_config.single_class_id = getattr(config.sampling, 'single_class_id', None)
    return ppo_config


def _initialize_noise_sampler(ppo_config, config):
    """Create and optionally load noise sampler."""
    noise_sampler = ppo_utils.NoiseSampler(ppo_config).to(config.device)
    checkpoint_path = ppo_config.noise_sampler_path
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        noise_sampler.load_state_dict(checkpoint['policy_state_dict'])
        noise_sampler.eval()
        logging.info(f"Loaded pre-trained noise sampler from {checkpoint_path}")
    return noise_sampler


def _setup_model_for_eval(config, workdir):
    """Initialize model components for evaluation."""
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    return state, score_model, ema


def _setup_sampling_function(config, sde, inverse_scaler, sampling_eps):
    """Create sampling function for evaluation."""
    if not config.eval.enable_sampling:
        return None
    sampling_shape = (config.eval.batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    return sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _generate_class_grid_samples(eval_dir, ppo_config, sde, sampling_fn,
                                 score_model, inverse_scaler, total_rounds):
    # set_seed(42)
    """Generate sample grid for all classes (no inverse-scaling)."""
    logging.info("Generating sample batch for all classes...")
    sample_grid_dir = os.path.join(eval_dir, "sample_grids")
    tf.io.gfile.makedirs(sample_grid_dir)
    all_class_samples = []
    sde.deterministic = True
    for test_class in range(ppo_config.num_classes):
        sde.set_target_class(test_class)
        
        with torch.no_grad():
            test_samples, noise = sampling_fn(score_model)
            all_class_samples.append(test_samples[:10])  # keep in [0,1]
    sde.deterministic = False
    all_class_samples = torch.cat(all_class_samples, dim=0)
    grid_path = os.path.join(sample_grid_dir, f"all_classes_round_{total_rounds}.png")
    _save_grid(all_class_samples, grid_path, nrow=10)
    logging.info(f"Saved all-class grid to {grid_path}")


def _generate_final_evaluation_samples(eval_dir, ppo_config, sde, sampling_fn,
                                       score_model, inverse_scaler):
    """Generate final evaluation samples for all classes (no inverse-scaling)."""
    logging.info("Generating final evaluation samples...")
    final_samples_dir = os.path.join(eval_dir, "final_samples")
    tf.io.gfile.makedirs(final_samples_dir)

    for eval_class in range(ppo_config.num_classes):
        sde.set_target_class(eval_class)
        with torch.no_grad():
            eval_samples, _ = sampling_fn(score_model)
            class_path = os.path.join(final_samples_dir, f"class_{eval_class}_samples.png")
            _save_grid(eval_samples[:64], class_path, nrow=8)

    logging.info("Final evaluation completed!")

from torchvision.transforms import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
def _compute_test_dataset_fid(config, ppo_config, sampling_fn, score_model, sde, noise_sampler):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=ppo_config.batch_size, shuffle=False, num_workers=4)
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(config.device)
    
    for target_images, _ in tqdm(test_dataloader):
        target_images = target_images.to(config.device)
        fid_metric.update(target_images, real=True)
        
        with torch.no_grad():
            random_class_labels = torch.randint(0, ppo_config.num_classes,
                                                (target_images.size(0),),
                                                device=config.device, dtype=torch.long)
            noise_sample = ppo_utils.sample_noise(noise_sampler, random_class_labels, "cuda", deterministic=True)
            noise_for_sampling = noise_sample.view(target_images.size(0), 3, 32, 32)
            sde.set_noise(noise_for_sampling)
            generated_samples, _ = sampling_fn(score_model, z=noise_for_sampling)
        generated_samples = torch.clamp(generated_samples * 0.5 + 0.5, 0, 1)
        fid_metric.update(generated_samples, real=False)
    fid_score = fid_metric.compute()
    logging.info(f"Test Dataset FID: {fid_score.item():.6e}")
    exit()
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from optimal_transport import wasserstein
from plot_tsne import plot_tsne_inception
def _compute_train_dataset_fid(config, ppo_config, sampling_fn, score_model, sde, noise_sampler, ppo_resources):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=ppo_config.batch_size, shuffle=False, num_workers=4)
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(config.device)
    kid_metric = KernelInceptionDistance(subset_size=1000, feature=2048, normalize=True).to(config.device)
    is_metric = InceptionScore(normalize=True).to(config.device)
    rates = [len(ppo_resources['class_to_indices'][class_id]) for class_id in range(ppo_config.num_classes)]
    rates = np.array(rates) / sum(rates)
    samples = []
    target_samples = []
    sz = 0
    for target_images, _ in tqdm(train_dataloader):
        target_images = target_images.to(config.device)
        fid_metric.update(target_images, real=True)
        kid_metric.update(target_images, real=True)
        
        with torch.no_grad():
            random_class_labels = np.random.choice(np.arange(ppo_config.num_classes), size=(target_images.size(0),), p=rates) 
            random_class_labels = torch.from_numpy(random_class_labels).to(config.device).long()
            if ppo_config.gaussian:
                noise_sample = torch.randn_like(target_images)
            else:
                noise_sample = ppo_utils.sample_noise(noise_sampler, random_class_labels, "cuda", deterministic=True)
            noise_for_sampling = noise_sample.view(target_images.size(0), 3, 32, 32)
            sde.set_noise(noise_for_sampling)
            generated_samples, _ = sampling_fn(score_model, z=noise_for_sampling)
        generated_samples = torch.clamp(generated_samples * 0.5 + 0.5, 0, 1)
        fid_metric.update(generated_samples, real=False)
        kid_metric.update(generated_samples, real=False)
        is_metric.update(generated_samples)
    fid_score = fid_metric.compute()
    kid_mean, kid_std = kid_metric.compute()
    is_mean, is_std = is_metric.compute()
    logging.info(f"Test Dataset FID: {fid_score.item():.6e}")
    logging.info(f"Test Dataset KID: {kid_mean.item():.6e} ± {kid_std.item():.6e}")
    logging.info(f"Test Dataset IS: {is_mean.item():.6e} ± {is_std.item():.6e}")
    exit()