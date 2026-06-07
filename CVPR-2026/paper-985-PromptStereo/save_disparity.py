import os
import hydra
import torch
import numpy as np
import imageio.v3 as imageio
from glob import glob
from tqdm import tqdm
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from accelerate import load_checkpoint_and_dispatch
from accelerate.logging import get_logger
from model import fetch_model
from dataset import fetch_dataloader
from util.padder import InputPadder

@hydra.main(version_base=None, config_path='config', config_name='save_disparity')
def main(cfg):
    logger = get_logger(__name__)
    accelerator = instantiate(cfg.accelerator)

    dataloader = fetch_dataloader(cfg, cfg.dataset, cfg.dataloader, logger)
    model = fetch_model(cfg, logger)
    model = load_checkpoint_and_dispatch(model, cfg.checkpoint)
    logger.info(f'Loading checkpoint from {cfg.checkpoint}.')

    model = accelerator.prepare_model(model)
    for name in dataloader:
        dataloader[name] = accelerator.prepare_data_loader(dataloader[name])

    for name in dataloader:
        model.eval()
        for data in tqdm(dataloader[name], dynamic_ncols=True, disable=not accelerator.is_main_process):
            filename, left, right, disp_gt, valid = [x for x in data]
            padder = InputPadder(left.shape, divis_by=32)
            left, right = padder.pad(left, right)

            with torch.no_grad():
                disp_pred = model(left, right, cfg.model.valid_iters, test_mode=True)
                disp_pred = padder.unpad(disp_pred)

            out = (disp_gt - disp_pred).abs() > cfg.dataset[name].outlier
            
            if cfg.max_disp:
                valid = (valid >= 0.5) & (disp_gt < cfg.max_disp)
            else:
                valid = (valid >= 0.5)

            out[~valid] = 0
            out = out * 255

            if cfg.base_index != -1:
                file_dir = os.path.join(cfg.disp_dir, filename[0].split('/')[cfg.base_index])
            else:
                file_dir = cfg.disp_dir

            os.makedirs(file_dir, exist_ok=True)
            plt.imsave(f'{file_dir}/{filename[0].split('/')[-1].split('.')[0]}.png', disp_pred.squeeze().cpu().numpy(), cmap='jet')

            out_dir = os.path.join(file_dir, 'outlier')
            os.makedirs(out_dir, exist_ok=True)
            plt.imsave(f'{out_dir}/{filename[0].split('/')[-1].split('.')[0]}.png', out.squeeze().cpu().numpy(), cmap='grey')

    accelerator.end_training()

if __name__ == '__main__':
    main()