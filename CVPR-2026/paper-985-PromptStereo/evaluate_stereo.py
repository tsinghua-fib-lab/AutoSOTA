import hydra
import torch
from tqdm import tqdm
from hydra.utils import instantiate
from accelerate import load_checkpoint_and_dispatch
from accelerate.logging import get_logger
from model import fetch_model
from dataset import fetch_dataloader
from util.padder import InputPadder

@hydra.main(version_base=None, config_path='config', config_name='evaluate_stereo')
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
        total_elem, total_epe, total_out = 0, 0, 0
        for data in tqdm(dataloader[name], dynamic_ncols=True, disable=not accelerator.is_main_process):
            _, left, right, disp_gt, valid = [x for x in data]
            padder = InputPadder(left.shape, divis_by=32)
            left, right = padder.pad(left, right)

            with torch.no_grad():
                disp_pred = model(left, right, cfg.model.valid_iters, test_mode=True)
                disp_pred = padder.unpad(disp_pred)

            epe = torch.abs(disp_pred - disp_gt)
            out = (epe > cfg.dataset[name].outlier).float()

            if cfg.max_disp:
                valid = (valid >= 0.5) & (disp_gt < cfg.max_disp)
            else:
                valid = (valid >= 0.5)

            epe, out = accelerator.gather_for_metrics((epe[valid >= 0.5].mean(), out[valid >= 0.5].mean()))
            
            if len(cfg.gpus.split(',')) == 1:
                total_elem += 1
            else:
                total_elem += epe.shape[0]
                
            total_epe += epe.sum().item()
            total_out += out.sum().item()
            
        accelerator.print(f'{name}/EPE: {total_epe / total_elem:.2f}, {name}/Bad {cfg.dataset[name].outlier}px: {100 * total_out / total_elem:.2f}')

    accelerator.end_training()

if __name__ == '__main__':
    main()