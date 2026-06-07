import os
import hydra
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from hydra.utils import instantiate
from accelerate import load_checkpoint_and_dispatch
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from model import fetch_model
from dataset import fetch_dataloader
from util.loss import sequence_loss
from util.padder import InputPadder

@hydra.main(version_base=None, config_path='config', config_name='train_sceneflow')
def main(cfg):
    set_seed(cfg.seed)
    logger = get_logger(__name__)
    gpu_num = len(cfg.gpus.split(','))
    accelerator = instantiate(cfg.accelerator, _partial_=True)(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    accelerator.init_trackers(project_name=cfg.tracker.project_name, config=OmegaConf.to_container(cfg, resolve=True), init_kwargs=cfg.tracker.init_kwargs)

    train_loader = fetch_dataloader(cfg, cfg.train_set, cfg.train_loader, logger)
    valid_loader = fetch_dataloader(cfg, cfg.valid_set, cfg.valid_loader, logger)
    model = fetch_model(cfg, logger)
    optimizer = instantiate(cfg.optimizer, _partial_=True)(model.parameters())
    scheduler = instantiate(cfg.scheduler, _partial_=True)(optimizer)

    if cfg.checkpoint:
        model = load_checkpoint_and_dispatch(model, cfg.checkpoint)
        logger.info(f'Loading checkpoint from {cfg.checkpoint}.')

    train_loader, model, optimizer, scheduler = accelerator.prepare(train_loader, model, optimizer, scheduler)
    for name in valid_loader:
        valid_loader[name] = accelerator.prepare_data_loader(valid_loader[name])

    set_seed(cfg.seed, device_specific=True)
    
    step = 0
    should_keep_training = True
    while should_keep_training:
        model.train()
        model.module.freeze_bn()
        for data in tqdm(train_loader, dynamic_ncols=True, disable=not accelerator.is_main_process):
            _, left, right, disp_gt, valid = [x for x in data]

            with accelerator.autocast():
                init_disp, disp_pred = model(left, right, iters=cfg.model.train_iters)

            loss, metric = sequence_loss(init_disp, disp_pred, disp_gt, valid, cfg.max_disp)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), cfg.max_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step += 1
            loss = accelerator.reduce(loss.detach(), reduction='mean')
            metric = accelerator.reduce(metric, reduction='mean')
            accelerator.log({'train/loss': loss, 'train/learning_rate': optimizer.param_groups[0]['lr']}, step)
            accelerator.log(metric, step)

            if (step > 0) and (step % cfg.save_freq == 0):
                accelerator.save_state(os.path.join(cfg.save_path, str(step)))

            if (step > 0) and (step % cfg.valid_freq == 0):
                for name in valid_loader:
                    model.eval()
                    total_elem, total_epe, total_out = 0, 0, 0
                    for data in tqdm(valid_loader[name], dynamic_ncols=True, disable=not accelerator.is_main_process):
                        _, left, right, disp_gt, valid = [x for x in data]
                        padder = InputPadder(left.shape, divis_by=32)
                        left, right = padder.pad(left, right)

                        with torch.no_grad():
                            disp_pred = model(left, right, iters=cfg.model.valid_iters, test_mode=True)
                            disp_pred = padder.unpad(disp_pred)
                            
                        epe = torch.abs(disp_pred - disp_gt)
                        out = (epe > cfg.valid_set[name].outlier).float()

                        if cfg.max_disp:
                            valid = (valid >= 0.5) & (disp_gt < cfg.max_disp)
                        else:
                            valid = (valid >= 0.5)

                        epe, out = accelerator.gather_for_metrics((torch.nan_to_num(epe[valid >= 0.5].mean()), torch.nan_to_num(out[valid >= 0.5].mean())))

                        total_elem += epe.shape[0]
                        total_epe += epe.sum().item()
                        total_out += out.sum().item()
                    accelerator.log({f'valid/{name}/EPE': total_epe / total_elem, f'valid/{name}/BP-{cfg.valid_set[name].outlier}': 100 * total_out / total_elem}, step)

                model.train()
                model.module.freeze_bn()

            if step == cfg.scheduler.total_steps:
                should_keep_training = False
                break

    accelerator.save_model(model, os.path.join(cfg.save_path, 'final'))

    accelerator.end_training()

if __name__ == '__main__':
    main()