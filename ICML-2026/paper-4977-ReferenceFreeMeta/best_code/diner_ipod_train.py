"""
DINER-IPOD Meta-Learning Training Script
Based on Reptile algorithm with DINER backbone.
Adapted from SIREN_IPOD_train.py for the DINER architecture.

Paper hyperparameters (DINER backbone):
- Architecture: Hash encoding + MLP (2 hidden layers, 16 neurons, ReLU)
- Meta learning rate: 5e-4
- Inner loop learning rate: 2e-2
- Inner steps: 300
- Meta batch size (tasks_per_epoch): 15
- Meta training epochs: 2500
- Optimizer: Adam
- Total variation weight: 2
"""

import os
import sys
import time
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import h5py
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add model path
sys.path.insert(0, '/repo')
from model_diner import DinerModel
from utils import build_coordinate_train, MYTVLoss

# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize01(img):
    if len(img.shape) == 3:
        nimg = len(img)
    else:
        nimg = 1
        r, c = img.shape
        img = np.reshape(img, (nimg, r, c))
    img2 = np.empty(img.shape, dtype=img.dtype)
    for i in range(nimg):
        denom = img[i].ptp()
        if denom == 0:
            denom = 1
        img2[i] = np.divide(img[i] - img[i].min(), denom, out=np.zeros_like(img[i]), where=denom != 0)
    return np.squeeze(img2).astype(img.dtype)


def calculate_psnr(pred, target, data_range=None):
    pred_abs = normalize01(np.abs(pred))
    target_abs = normalize01(np.abs(target))
    if data_range is None:
        data_range = np.max(target_abs)
    mse = np.mean((pred_abs - target_abs) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range / np.sqrt(mse))


# ============================================================================
# Simple Dataset for Processed fastMRI
# ============================================================================

class FastMRIDataset:
    """Simple dataset that loads pre-processed fastMRI tasks."""

    def __init__(self, data_dir, preload=True):
        self.data_dir = data_dir
        self.task_dirs = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('task_')
        ])
        print("Found %d tasks in %s" % (len(self.task_dirs), data_dir))

        self.task_metadata = {}
        for task in self.task_dirs:
            task_path = os.path.join(data_dir, task)
            sample_files = sorted([f for f in os.listdir(task_path) if f.endswith('.h5')])
            if sample_files:
                full_paths = [os.path.join(task_path, f) for f in sample_files]
                # Read task params from first sample
                try:
                    with h5py.File(full_paths[0], 'r') as hf:
                        params = {key: hf.attrs[key] for key in hf.attrs.keys()}
                except:
                    params = {}
                self.task_metadata[task] = {
                    'task_id': task,
                    'task_params': params,
                    'file_paths': full_paths,
                }

        self.task_dirs = list(self.task_metadata.keys())
        print("Valid tasks: %d" % len(self.task_dirs))

    def __len__(self):
        return len(self.task_dirs)

    def __getitem__(self, idx):
        task_id = self.task_dirs[idx]
        return self.task_metadata[task_id]

    def load_sample(self, file_path):
        """Load a single sample from disk."""
        with h5py.File(file_path, 'r') as hf:
            # Handle data format differences
            mask_raw = hf['mask'][:]
            csmp_raw = hf['csmp'][:]
            forward_fft_raw = hf['forward_fft'][:]
            img_full_raw = hf['img_full'][:]

            nRow, nCol = img_full_raw.shape

            # The original code expects transposed data
            # mask: (h, w) -> keep as (h, w)
            # csmp: (1, h, w) -> needs to be (h, w, 1) or similar
            # forward_fft: (h, w) -> needs to be (h, w, 1)

            # For singlecoil: csmp is (1, h, w), we need (h, w, 1)
            if len(csmp_raw.shape) == 3 and csmp_raw.shape[0] == 1:
                csmp_transposed = csmp_raw.transpose(1, 2, 0)  # (h, w, 1)
            else:
                csmp_transposed = csmp_raw.transpose(1, 2, 0)

            # mask: (h, w) -> (h, w, 1)
            mask_transposed = np.expand_dims(mask_raw, axis=-1)

            # forward_fft: (h, w) -> (h, w, 1)
            gt_ksp_transposed = np.expand_dims(forward_fft_raw, axis=-1)

            # Generate coordinates
            coordinates = build_coordinate_train(L_RO=nRow, L_PE=nCol)

            return {
                'mask': mask_raw,
                'mask_transposed': mask_transposed.astype(np.float32),
                'csmp': csmp_raw,
                'csmp_transposed': csmp_transposed.astype(np.complex64),
                'forward_fft': forward_fft_raw,
                'gt_ksp_transposed': gt_ksp_transposed.astype(np.complex64),
                'gt_img': img_full_raw,
                'coordinates': coordinates,
                'selected_caption': None,
            }

    def get_samples(self, task_idx, num_samples=1):
        """Get samples for a task."""
        task_id = self.task_dirs[task_idx]
        file_paths = self.task_metadata[task_id]['file_paths']

        # Select random samples
        if num_samples >= len(file_paths):
            selected = file_paths
        else:
            selected = random.sample(file_paths, num_samples)

        samples = []
        for fp in selected:
            try:
                sample = self.load_sample(fp)
                samples.append(sample)
            except Exception as e:
                print("Error loading %s: %s" % (fp, e))

        return {
            'task_id': task_id,
            'task_params': self.task_metadata[task_id]['task_params'],
            'samples': samples,
        }


# ============================================================================
# DINER Reptile Trainer
# ============================================================================

class DinerReptileTrainer:
    """Reptile meta-learning trainer for DINER backbone."""

    def __init__(
        self,
        encoding_config=None,
        network_config=None,
        inner_lr=2e-2,
        meta_lr=5e-4,
        inner_steps=300,
        samples_per_task=10,
        device='cuda:0',
        load_checkpoint=None,
    ):
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        self.samples_per_task = samples_per_task
        self.device = device
        self.start_epoch = 0

        # Create DINER model
        if encoding_config is None:
            encoding_config = {
                "otype": "Grid", "type": "Hash",
                "n_levels": 16, "n_features_per_level": 2,
                "log2_hashmap_size": 19, "base_resolution": 12,
                "per_level_scale": 2, "interpolation": "Linear"
            }
        if network_config is None:
            network_config = {
                "otype": "FullyFusedMLP", "activation": "ReLU",
                "output_activation": "None", "n_neurons": 16,
                "n_hidden_layers": 2
            }

        self.encoding_config = encoding_config
        self.network_config = network_config

        self.model = DinerModel(
            encoding_config=encoding_config,
            network_config=network_config
        ).to(device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print("DINER model created with %d parameters" % total_params)

        # Load checkpoint if provided
        if load_checkpoint:
            checkpoint = torch.load(load_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = checkpoint.get('epoch', 0)
            print("Loaded checkpoint from epoch %d" % self.start_epoch)

        # Meta optimizer (Adam)
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=meta_lr
        )

        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.meta_optimizer, step_size=500, gamma=0.5
        )

        # Loss functions
        self.mae_loss = nn.L1Loss()
        self.tv_loss = MYTVLoss()

    def inner_loop_adaptation(self, task_samples):
        """Inner loop: adapt model to a specific task."""
        adapted_model = DinerModel(
            encoding_config=self.encoding_config,
            network_config=self.network_config
        ).to(self.device)
        adapted_model.load_state_dict(self.model.state_dict())

        inner_optimizer = torch.optim.Adam(
            adapted_model.parameters(), lr=self.inner_lr
        )
        adapted_model.train()

        for step_ind in range(self.inner_steps):
            sample = random.choice(task_samples)

            mask = torch.tensor(sample['mask_transposed']).to(self.device)
            csmp = torch.tensor(sample['csmp_transposed']).to(self.device).to(torch.complex64)
            gt_ksp = torch.tensor(sample['gt_ksp_transposed']).to(self.device).to(torch.complex64)
            coordinates = torch.tensor(sample['coordinates']).to(self.device).float()

            nRow, nCol = sample['gt_img'].shape

            # Forward pass
            pre_intensity_mag, pre_intensity_phi = adapted_model(coordinates.view(-1, 2))
            pre_intensity = torch.complex(
                pre_intensity_mag.view(nRow, nCol, 1),
                pre_intensity_phi.view(nRow, nCol, 1)
            )

            # Apply coil sensitivity (for singlecoil, csmp is all ones)
            pre_intensity_multi = pre_intensity * csmp

            # FFT to k-space
            fft_pre_intensity = torch.fft.fftshift(
                torch.fft.fft2(
                    torch.fft.fftshift(pre_intensity_multi, dim=(0, 1)),
                    dim=(0, 1)
                ),
                dim=(0, 1)
            )

            # Loss: data consistency at sampled locations
            mae_ksp_loss = self.mae_loss(
                torch.view_as_real(fft_pre_intensity[mask == 1]).float(),
                torch.view_as_real(gt_ksp[mask == 1]).float()
            )

            # TV regularization
            TV_loss = self.tv_loss(pre_intensity_mag.view(nRow, nCol, 1)) + \
                      self.tv_loss(pre_intensity_phi.view(nRow, nCol, 1))

            loss = mae_ksp_loss + 2 * TV_loss

            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()

        final_loss = loss.item()
        return adapted_model, final_loss

    def adaptive_reptile_update(self, adapted_models, task_losses):
        """Weighted Reptile meta-update."""
        inverse_losses = [1.0 / (loss + 1e-8) for loss in task_losses]
        total_inverse = sum(inverse_losses)
        weights = [inv / total_inverse for inv in inverse_losses]

        with torch.no_grad():
            for adapted_model, weight in zip(adapted_models, weights):
                weighted_lr = self.meta_lr * weight
                for p_meta, p_adapted in zip(self.model.parameters(), adapted_model.parameters()):
                    update = weighted_lr * (p_adapted.data - p_meta.data)
                    p_meta.data.add_(update)

        return weights

    def train(
        self,
        dataset,
        epochs,
        save_dir='./checkpoints_diner',
        eval_interval=50,
        tasks_per_epoch=15,
    ):
        """Main meta-training loop."""
        os.makedirs(save_dir, exist_ok=True)

        stats = {
            'meta_losses': [],
            'eval_psnrs': [],
            'epoch_times': [],
        }

        best_psnr = 0
        total_tasks = len(dataset)

        print("Starting DINER meta-training: %d epochs, %d tasks/epoch" % (epochs, tasks_per_epoch))
        print("Inner lr: %e, Meta lr: %e, Inner steps: %d" % (self.inner_lr, self.meta_lr, self.inner_steps))

        for epoch in range(self.start_epoch, epochs):
            t_start = time.time()

            self.model.train()

            # Select tasks
            task_indices = random.sample(range(total_tasks), min(tasks_per_epoch, total_tasks))

            batch_task_losses = []
            batch_adapted_models = []

            for task_idx in task_indices:
                task_data = dataset.get_samples(task_idx, num_samples=self.samples_per_task)
                task_samples = task_data['samples']

                if len(task_samples) == 0:
                    continue

                # Inner loop adaptation
                adapted_model, task_loss = self.inner_loop_adaptation(task_samples)
                batch_task_losses.append(task_loss)
                batch_adapted_models.append(adapted_model)

            if len(batch_adapted_models) == 0:
                continue

            # Meta update
            weights = self.adaptive_reptile_update(batch_adapted_models, batch_task_losses)
            self.scheduler.step()

            epoch_meta_loss = np.mean(batch_task_losses)
            stats['meta_losses'].append(epoch_meta_loss)
            stats['epoch_times'].append(time.time() - t_start)

            current_lr = self.scheduler.get_last_lr()[0]
            print("Epoch %d/%d | Loss: %.6f | LR: %.2e | Time: %.1fs" % (
                epoch + 1, epochs, epoch_meta_loss, current_lr, stats['epoch_times'][-1]
            ))

            # Periodic evaluation
            if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
                eval_psnr = self.evaluate(dataset, num_tasks=5)
                stats['eval_psnrs'].append(eval_psnr)
                print("  Eval PSNR: %.2f dB" % eval_psnr)

                # Save checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.meta_optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'meta_loss': epoch_meta_loss,
                    'eval_psnr': eval_psnr,
                    'encoding_config': self.encoding_config,
                    'network_config': self.network_config,
                }
                torch.save(checkpoint, os.path.join(save_dir, 'model_epoch_%d.pth' % (epoch + 1)))

                if eval_psnr > best_psnr:
                    best_psnr = eval_psnr
                    torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
                    print("  New best PSNR: %.2f dB" % best_psnr)

        return stats

    def evaluate(self, dataset, num_tasks=5):
        """Quick evaluation on a few tasks."""
        self.model.eval()
        psnrs = []

        task_indices = random.sample(range(len(dataset)), min(num_tasks, len(dataset)))

        for task_idx in task_indices:
            task_data = dataset.get_samples(task_idx, num_samples=1)
            samples = task_data['samples']

            if len(samples) == 0:
                continue

            # Inner loop adaptation for evaluation
            adapted_model, _ = self.inner_loop_adaptation(samples)
            adapted_model.eval()

            sample = samples[0]
            mask = torch.tensor(sample['mask_transposed']).to(self.device)
            csmp = torch.tensor(sample['csmp_transposed']).to(self.device).to(torch.complex64)
            coordinates = torch.tensor(sample['coordinates']).to(self.device).float()
            gt_img = sample['gt_img']
            nRow, nCol = gt_img.shape

            with torch.no_grad():
                pre_intensity_mag, pre_intensity_phi = adapted_model(coordinates.view(-1, 2))
                pre_intensity = torch.complex(
                    pre_intensity_mag.view(nRow, nCol, 1),
                    pre_intensity_phi.view(nRow, nCol, 1)
                )

                pred_img = pre_intensity.squeeze().cpu().numpy()
                psnr = calculate_psnr(pred_img, gt_img)
                psnrs.append(psnr)

        return np.mean(psnrs) if psnrs else 0


# ============================================================================
# Main
# ============================================================================

def main():
    set_seed(35236)

    data_dir = '/datasets/fastmri_processed'
    save_dir = '/repo/checkpoints_diner'

    # Paper hyperparameters for DINER
    config = {
        'inner_lr': 2e-2,        # Paper: 2e-2 for DINER
        'meta_lr': 5e-4,         # Paper: 5e-4 base meta lr
        'inner_steps': 300,       # Paper: 300 inner steps
        'epochs': 2500,           # Paper: 2500 meta epochs
        'tasks_per_epoch': 15,    # Paper: 15 tasks per epoch
        'samples_per_task': 5,    # Number of samples to load per task
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    dataset = FastMRIDataset(data_dir, preload=False)

    trainer = DinerReptileTrainer(
        inner_lr=config['inner_lr'],
        meta_lr=config['meta_lr'],
        inner_steps=config['inner_steps'],
        samples_per_task=config['samples_per_task'],
        device=device,
    )

    stats = trainer.train(
        dataset=dataset,
        epochs=config['epochs'],
        save_dir=save_dir,
        eval_interval=100,
        tasks_per_epoch=config['tasks_per_epoch'],
    )

    print("\nTraining completed!")
    print("Best eval PSNR: %.2f dB" % (max(stats['eval_psnrs']) if stats['eval_psnrs'] else 0))


if __name__ == '__main__':
    main()
