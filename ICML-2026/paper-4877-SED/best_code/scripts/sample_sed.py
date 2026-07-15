import os
import re
import time

import anndata as ad
import numpy as np
import torch
from torchvision import utils

from sed.data.physics import SparseCaloImageDataModule
from sed.data.scrna import SparseCellDataModule
from sed.data.vision import SparseImageDataModule
from sed.models.callbacks.svae_logging import (get_sparse_cells,
                                               get_sparse_images)
from sed.sed_main import SedCLI
from sed.utils import num_to_groups


class SedSampleCLI(SedCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_argument(
            "-ns",
            "--num_samples",
            type=int,
            default=50000,
            help="number of samples",
        )
        parser.add_argument(
            "-ddim",
            "--use_ddim",
            type=bool,
            default=True,
            help="Use ddim or ddpm",
            nargs='+'
        )
        parser.add_argument(
            "-ldv",
            "--log_dim_value",
            type=bool,
            default=False,
            help="log dim value",
        )
        parser.add_argument(
            "-sample_ic",
            "--sample_intermediate_checkpoints",
            type=bool,
            default=False,
            help="sample_intermediate_checkpoints",
        )

    def before_instantiate_classes(self):
        pass

    def after_instantiate_classes(self):
        pass


def cli_main():
    cli = SedSampleCLI(run=False)
    exp_dir = os.path.dirname(cli.config.config[0])
    ckpts_dir = os.path.join(exp_dir, "ckpt")
    if cli.config.sample_intermediate_checkpoints:
        for file in os.listdir(ckpts_dir):
            if "step" in file:
                model_ckpt_dir = os.path.join(ckpts_dir, file)
                step = re.search(r'step=(\d+)\.ckpt', file).group(1)
                samples_dir = os.path.join(
                    exp_dir, "sampled", "trained", "samples_"+step)
                setup_sampling_strategies(cli, model_ckpt_dir, samples_dir)
    else:
        model_ckpt_dir = os.path.join(ckpts_dir, "last.ckpt")
        samples_dir = os.path.join(
            exp_dir, "sampled", "trained", "samples")
        setup_sampling_strategies(cli, model_ckpt_dir, samples_dir)


def setup_sampling_strategies(cli, model_ckpt_dir, samples_dir):
    sampling_strategies = [
        "DDIM"] if True in cli.config.use_ddim else []
    if False in cli.config.use_ddim:
        sampling_strategies.append("DDPM")
    for sampling_strategy in sampling_strategies:
        sample_ss_dir = os.path.join(samples_dir, sampling_strategy)
        os.makedirs(sample_ss_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = cli.model.__class__.load_from_checkpoint(model_ckpt_dir)
    model.to(device)
    for use_ddim in cli.config.use_ddim:
        model.use_ddim = use_ddim
        sample(cli, model, samples_dir)


def sample(cli, model, samples_dir):
    batches = num_to_groups(cli.config.num_samples,
                            cli.config.data.init_args.batch_size)
    model.eval()
    glob_idx = 0
    sampling_strategy = "DDIM" if model.use_ddim else "DDPM"
    if cli.model.vae.input_mode == 'image':
        image_data_module = SparseImageDataModule.load_from_checkpoint(
            os.path.join(
                os.path.dirname(cli.config.config[0]), "ckpt", "last.ckpt"))
        start_pos = image_data_module.start_position
        end_pos = image_data_module.end_position
        pad_pos = image_data_module.pad_position
    if cli.model.vae.input_mode == 'calo_image':
        calo_image_data_module = SparseCaloImageDataModule.load_from_checkpoint(
            os.path.join(
                os.path.dirname(cli.config.config[0]), "ckpt", "last.ckpt"))
        scaling_value = calo_image_data_module.max_value
        start_pos = calo_image_data_module.start_position
        end_pos = calo_image_data_module.end_position
        pad_pos = calo_image_data_module.pad_position
    if cli.model.vae.input_mode == 'scrna':
        cell_data_module = SparseCellDataModule.load_from_checkpoint(
            os.path.join(
                os.path.dirname(cli.config.config[0]), "ckpt", "last.ckpt"))
        start_pos = cell_data_module.start_position
        end_pos = cell_data_module.end_position
        pad_pos = cell_data_module.pad_position

    data_dimensions = model.vae.data_dimensions
    start = time.time()
    with torch.no_grad():
        for i, batch_size in enumerate(batches):
            sampled_batch = model.sample(
                batch_size=batch_size)
            if cli.model.vae.input_mode == 'image' or cli.model.vae.input_mode == 'calo_image':
                if cli.model.vae.input_mode == 'image':
                    images = get_sparse_images(
                        sampled_batch, start_pos, end_pos, pad_pos, data_dimensions).unsqueeze(1)
                    for image in images:
                        utils.save_image(image, os.path.join(
                            samples_dir, sampling_strategy, f"image_{glob_idx}.png"))
                        glob_idx += 1
                elif cli.model.vae.input_mode == 'calo_image':
                    images = get_sparse_images(
                        sampled_batch, start_pos, end_pos, pad_pos, data_dimensions)
                    images = images.squeeze()
                    images = images * scaling_value
                    np.save(os.path.join(samples_dir, sampling_strategy,
                            f'sample-{i}.npy'), images.cpu().numpy())
            elif cli.model.vae.input_mode == 'scrna':
                cells = get_sparse_cells(
                    sampled_batch, start_pos, end_pos, pad_pos, data_dimensions)
                adata = ad.AnnData(cells.cpu().numpy())
                adata.write(os.path.join(
                    samples_dir, sampling_strategy, f'sample-{i}.h5ad'))
    end = time.time()
    with open(os.path.join(
            samples_dir, sampling_strategy, f'execution_time.txt'), 'w') as out_file:
        out_file.write(
            f"Elapsed: {(end-start)/cli.config.num_samples} s per sample for {cli.config.num_samples} samples")


if __name__ == "__main__":
    cli_main()
