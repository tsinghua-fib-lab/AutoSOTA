import json
import math
import os
from abc import ABC, abstractmethod

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import Callback
from matplotlib.colors import LogNorm
from torchvision import utils

from sed.utils import num_to_groups


class SvaeLogger(Callback, ABC):
    def __init__(self, num_samples: int = 25, num_reconstructed: int = 25, batch_size: int = 64, sample_every: int = 10000, sampled_dir: str = None, reconstr_dir: str = None):
        super().__init__()
        self.num_samples = num_samples
        self.num_reconstr = num_reconstructed
        assert self.num_reconstr <= batch_size
        self.batch_size = batch_size
        self.sample_every = sample_every
        self.sampled_dir = sampled_dir
        self.reconstr_dir = reconstr_dir

    def on_validation_batch_end(self, trainer, pl_module, outputs,
                                batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            in_positions, in_values, mu = outputs
            self.log_samples([(in_positions[:self.num_reconstr, :],
                               in_values[:self.num_reconstr, :])], trainer, pl_module, self.reconstr_dir, "original")
            out_positions, out_values = pl_module.sample(self.batch_size, mu)
            self.log_samples(
                [(out_positions[:self.num_reconstr, :], out_values[:self.num_reconstr, :])], trainer, pl_module,  self.reconstr_dir, "reconstructed")
            all_samples_list = self.sample(trainer, pl_module)
            self.log_samples(all_samples_list, trainer,
                             pl_module, self.sampled_dir, "sampled")

    def sample(self, trainer, pl_module):
        with torch.no_grad():
            batches = num_to_groups(
                self.num_samples, self.batch_size)
            all_samples_list = list(
                map(lambda n: pl_module.sample(batch_size=n), batches))
            return all_samples_list

    @abstractmethod
    def log_samples(self, all_samples_list, trainer, pl_module, dir, filename):
        pass


class ImageLogger(SvaeLogger):
    def log_samples(self, all_samples_list, trainer, pl_module, dir, filename):
        start_pos = trainer.datamodule.train_dataset.start_position
        end_pos = trainer.datamodule.train_dataset.end_position
        pad_pos = trainer.datamodule.train_dataset.pad_position
        data_dimensions = pl_module.data_dimensions
        for batch_idx, batch in enumerate(all_samples_list):
            images = get_sparse_images(
                batch, start_pos, end_pos, pad_pos, data_dimensions)
            utils.save_image(images.unsqueeze(1), os.path.join(
                dir, f'{filename}-{pl_module.global_step}-{batch_idx}.png'), nrow=int(math.sqrt(self.num_samples)))


def get_sparse_images(batch, start_pos, end_pos, pad_pos, data_dimensions):
    pos, values = batch[0], batch[1]
    batch_size, number_of_pos = pos.shape
    valid_pos, valid_values, valid_mask = get_valid_pos_values(
        batch, start_pos, end_pos, pad_pos)

    image_height = int(math.sqrt(data_dimensions))
    images = torch.zeros(
        (batch_size, image_height, image_height), dtype=values.dtype).to(values.device)
    batch_indices = torch.arange(batch_size).to(values.device).unsqueeze(
        1).expand(-1, number_of_pos)[valid_mask]

    row_indices = valid_pos // image_height
    col_indices = valid_pos % image_height

    images[batch_indices, row_indices, col_indices] = valid_values
    return images


class scrnaLogger(SvaeLogger):
    def log_samples(self, all_samples_list, trainer, pl_module, dir, filename):
        start_pos = trainer.datamodule.train_dataset.dataset.start_position
        end_pos = trainer.datamodule.train_dataset.dataset.end_position
        pad_pos = trainer.datamodule.train_dataset.dataset.pad_position
        data_dimensions = pl_module.data_dimensions
        for batch_idx, batch in enumerate(all_samples_list):
            cells = get_sparse_cells(
                batch, start_pos, end_pos, pad_pos, data_dimensions)
            adata = ad.AnnData(cells.clamp(min=0.0, max=1.0).cpu().numpy())
            adata.write(os.path.join(dir,
                        f'{filename}-{pl_module.global_step}-{batch_idx}.h5ad'))


def get_sparse_cells(batch, start_pos, end_pos, pad_pos, data_dimensions):
    pos, values = batch[0], batch[1]
    batch_size, number_of_pos = pos.shape
    valid_pos, valid_values, valid_mask = get_valid_pos_values(
        batch, start_pos, end_pos, pad_pos)

    cells = torch.zeros(
        (batch_size, data_dimensions), dtype=values.dtype).to(values.device)
    batch_indices = torch.arange(batch_size).to(values.device).unsqueeze(
        1).expand(-1, number_of_pos)[valid_mask]

    cells[batch_indices, valid_pos] = valid_values
    return cells


class CaloImageLogger(SvaeLogger):
    def log_samples(self, all_samples_list, trainer, pl_module, dir, filename):
        start_pos = trainer.datamodule.train_dataset.start_position
        end_pos = trainer.datamodule.train_dataset.end_position
        pad_pos = trainer.datamodule.train_dataset.pad_position
        data_dimensions = pl_module.data_dimensions
        scaling_value = trainer.datamodule.max_value

        for batch_idx, batch in enumerate(all_samples_list):
            images = get_sparse_images(
                batch, start_pos, end_pos, pad_pos, data_dimensions)

            # Create a 5x5 grid of subplots
            fig, axes = plt.subplots(int(math.sqrt(self.num_samples)), int(
                math.sqrt(self.num_samples)), figsize=(10, 10))
            for idx, ax in enumerate(axes.flat):
                ax.imshow(images[idx].cpu().numpy(), norm=LogNorm(
                    vmin=1e-6, vmax=1e2), cmap='viridis')
                ax.set_xticks([])
                ax.set_yticks([])
            plt.tight_layout()

            # Save the entire grid as a single image file
            plt.savefig(os.path.join(
                dir, f'{filename}-{pl_module.global_step}-{batch_idx}.png'), bbox_inches='tight')
            plt.close()
            np.save(os.path.join(
                    dir, f'{filename}-{pl_module.global_step}-{batch_idx}.npy'), (images*scaling_value).cpu().numpy())


def get_valid_pos_values(batch, start_pos, end_pos, pad_pos):
    pos, values = batch[0], batch[1]
    batch_size, number_of_pos = pos.shape
    # Create a mask for end_position
    is_end = (pos == end_pos)
    # Get the first occurrence index for each sample (set to number_of_pos if not found)
    first_end_idx = torch.where(is_end.any(dim=1), is_end.float().argmax(
        dim=1), number_of_pos * torch.ones(batch_size, dtype=torch.long, device=pos.device))
    # Mask: True for positions before the first end_position, False for first and after
    position_indices = torch.arange(
        number_of_pos, device=pos.device).unsqueeze(0).expand(batch_size, -1)
    before_end_mask = position_indices < first_end_idx.unsqueeze(1)
    valid_mask = torch.logical_and(torch.logical_and((pos != pad_pos), (
        pos != start_pos)), before_end_mask)
    valid_pos = pos[valid_mask]
    valid_values = values[valid_mask]
    return valid_pos, valid_values, valid_mask


class SaveTestMetricsCallback(Callback):
    def __init__(self, save_path=None):
        super().__init__()
        self.filepath = os.path.join(save_path, "final_test_metrics.json")

    def on_test_end(self, trainer, pl_module):
        # Access the final test metrics
        metrics = trainer.callback_metrics
        # Convert tensors to floats for JSON serialization
        metrics_to_save = {k: float(v) for k, v in metrics.items()}
        # Save to file
        with open(self.filepath, "w") as f:
            json.dump(metrics_to_save, f, indent=4)
