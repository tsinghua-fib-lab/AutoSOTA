import logging
import time

import torch

from graphgym.checkpoint import clean_ckpt, load_ckpt, save_ckpt
from graphgym.config import cfg
from graphgym.loss import compute_loss
from graphgym.utils.epoch import is_ckpt_epoch, is_eval_epoch


def train_epoch(logger, loader, model, optimizer, scheduler):
    model.train()
    time_start = time.time()
    for batch in loader:
        batch.split = 'train'
        optimizer.zero_grad()
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        loss.backward()
        optimizer.step()
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()

    scheduler.step()


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=loss.item(),
                            lr=0,
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()


def train(loggers, loaders, model, optimizer, scheduler):
    r"""
    The core training pipeline

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    best_val_perf = None
    epochs_without_improvement = 0
    patience = cfg.train.get("early_stop_patience", 5)
    train_stats = None
    stats_list = None
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler)
        train_stats = loggers[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            valid = True
            stats_list = []
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model,
                           split=split_names[i - 1])
                if valid:
                    valid_stats = loggers[i].write_epoch(cur_epoch)
                    #val_perf = valid_stats.get('auc', valid_stats.get('accuracy', valid_stats.get('mse')))
                    val_perf = valid_stats.get('accuracy', valid_stats.get('mse'))
                    stats_list.append(valid_stats)
                    valid = False
                else:
                    test_stats = loggers[i].write_epoch(cur_epoch)
                    stats_list.append(test_stats)
            
            if best_val_perf is None or val_perf > best_val_perf:
                best_val_perf = val_perf
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                logging.info(f'No improvement for {epochs_without_improvement} epochs')

            if epochs_without_improvement >= patience:
                logging.info(f'Early stopping at epoch {cur_epoch} due to no improvement')
                break

        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)
    
    # Fill remaining epochs with last epoch's stats if early stopping occurred
    if cur_epoch < cfg.optim.max_epoch - 1:
        logging.info(f'Filling stats for the remaining epoch with last valid stats')
        for remaining_epoch in range(cur_epoch + 1, cfg.optim.max_epoch):
            loggers[0].write_epoch(remaining_epoch, train_stats)
            if is_eval_epoch(remaining_epoch):
                for i in range(1, num_splits):
                    loggers[i].write_epoch(remaining_epoch, stats_list[i - 1])

    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))
