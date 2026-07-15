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

from sed.models.callbacks.svae_logging import (get_sparse_cells,
                                               get_sparse_images)
from sed.utils import num_to_groups


class SEDLogger(Callback, ABC):
    def __init__(self, num_samples: int = 25, batch_size: int = 64, sample_every: int = 10000, sampled_dir: str = None):
        super().__init__()
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.sample_every = sample_every
        self.sampled_dir = sampled_dir

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if pl_module.global_step % self.sample_every == 0:
            self.sample(trainer, pl_module)

    def sample(self, trainer, pl_module):
        trainer.ema_callback._swap_models(pl_module)
        is_train = pl_module.diffusion_model.training
        if is_train:
            pl_module.diffusion_model.eval()

        with torch.no_grad():
            batches = num_to_groups(
                self.num_samples, self.batch_size)
            all_samples_list = list(
                map(lambda n: pl_module.sample(batch_size=n), batches))
            self.log_samples(all_samples_list, trainer,
                             pl_module, self.sampled_dir, "sampled")

        if is_train:
            pl_module.diffusion_model.train()
        trainer.ema_callback._swap_models(pl_module)

    @abstractmethod
    def log_samples(self, all_samples_list, trainer, pl_module, dir, filename):
        pass


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


class SparseImageLogger(SEDLogger):
    def log_samples(self, all_samples_list, trainer, pl_module, dir, filename):
        start_pos = trainer.datamodule.train_dataset.start_position
        end_pos = trainer.datamodule.train_dataset.end_position
        pad_pos = trainer.datamodule.train_dataset.pad_position
        data_dimensions = pl_module.vae.data_dimensions
        for batch_idx, batch in enumerate(all_samples_list):
            images = get_sparse_images(
                batch, start_pos, end_pos, pad_pos, data_dimensions)
            utils.save_image(images.unsqueeze(1), os.path.join(
                dir, f'{filename}-{pl_module.global_step}-{batch_idx}.png'), nrow=int(math.sqrt(self.num_samples)))


class SparseScrnaLogger(SEDLogger):
    def log_samples(self, all_samples_list, trainer, pl_module, dir, filename):
        start_pos = trainer.datamodule.train_dataset.dataset.start_position
        end_pos = trainer.datamodule.train_dataset.dataset.end_position
        pad_pos = trainer.datamodule.train_dataset.dataset.pad_position
        data_dimensions = pl_module.vae.data_dimensions
        for i, batch in enumerate(all_samples_list):
            cells = get_sparse_cells(
                batch, start_pos, end_pos, pad_pos, data_dimensions)
            adata = ad.AnnData(cells.clamp(min=0.0, max=1.0).cpu().numpy())
            adata.write(os.path.join(dir,
                        f'{filename}-{pl_module.global_step}-{i}.h5ad'))


class SparseCaloImageLogger(SEDLogger):
    def log_samples(self, all_samples_list, trainer, pl_module, dir, filename):
        start_pos = trainer.datamodule.train_dataset.start_position
        end_pos = trainer.datamodule.train_dataset.end_position
        pad_pos = trainer.datamodule.train_dataset.pad_position
        data_dimensions = pl_module.vae.data_dimensions
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
