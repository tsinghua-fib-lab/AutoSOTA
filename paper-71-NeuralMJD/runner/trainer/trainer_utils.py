import copy
import logging
import os
import time

import torch
import numpy as np
from typing import Dict

from utils.dist_training import get_ddp_save_flag, dist_save_model


def create_logging_dict(epoch: int) -> Dict[str, dict]:
    """
    Create a nested dictionary to track learning status for train/val/test.
    """
    _loss_status = {
        'time_start': None,
        'time_elapsed': None,
        'training_loss': [],
        'cond_mean_loss': [],
        'likelihood_loss': [],
        'mae_loss': [],
        'output': [],
        'output_winner': [],
        'output_prob': [],
        'target': [],
        'brands_factorize': [],
        'top_cat_factorize': [],
        'ticker_factorize': [],
    }

    loss_status_ls = [copy.deepcopy(_loss_status) for _ in range(3)]

    logger_dict = {'train': loss_status_ls[0],
                   'val': loss_status_ls[1],
                   'test': loss_status_ls[2],
                   'epoch': epoch,
                   'lr': 0.0}
    return logger_dict


def compute_metrics(output: torch.Tensor, target: torch.Tensor, adjust_r2: bool = False) -> Dict[str, torch.Tensor]:
    """
    Compute aggregate metrics for the revenue prediction task.

    Args:
        output: [B, Future] predictions.
        target: [B, Future] ground truth.
        adjust_r2: Whether to compute adjusted R² in addition to per-day R².

    Returns:
        Dictionary of tensors for loss/accuracy-style metrics.
    """
    with torch.no_grad():
        # Identify zero and non-zero targets.
        flag_tgt_zeros = target.isclose(torch.zeros_like(target))   # [B, Future]
        flag_tgt_non_zeros = ~flag_tgt_zeros                        # [B, Future]
        target_non_zeros = target[flag_tgt_non_zeros]               # [Y]

        # Compute element-wise absolute and squared errors.
        mae = torch.nn.functional.l1_loss(output, target, reduction='none')     # [B, Future]
        mse = torch.nn.functional.mse_loss(output, target, reduction='none')    # [B, Future]

        # Compute per-day metrics by averaging over the batch.
        mae_per_day = mae.mean(dim=0)   # [Future]
        mse_per_day = mse.mean(dim=0)   # [Future]

        # Errors on days where the target was zero or non-zero.
        mae_zeros = mae[flag_tgt_zeros]
        mae_non_zeros = mae[flag_tgt_non_zeros]
        mse_zeros = mse[flag_tgt_zeros]
        mse_non_zeros = mse[flag_tgt_non_zeros]

        eps = 1e-8
        if adjust_r2:
            # adjusted R² calculation
            r2_reg = 1.0 - mse.sum() / ((target - target.mean()).square().sum() + eps)

            # Use actual number of predictors
            k = 70.0
            n = target.size(0)
            p = (k - 1) / k * (n - 1)
            r2 = 1 - (1 - r2_reg) * (n - 1) / (n - p - 1)

            # Compute per-day R².
            r2_per_day = (1.0 - mse.sum(dim=0) / ((target - target.mean(dim=0)).square().sum(dim=0) + 1e-8)).mean()
        else:
            # vanilla R² score
            r2 = 1.0 - mse.sum() / ((target - target.mean()).square().sum() + eps)

            # r2 = (1.0 - mse.sum(dim=1) / ((target - target.mean(dim=1, keepdim=True)).square().sum(dim=1) + eps)).mean()

            # Compute per-day R².
            r2_per_day = 1.0 - mse.sum(dim=0) / ((target - target.mean(dim=0)).square().sum(dim=0) + eps)

        # Compute MAPE metrics, using clamp to avoid division by very small numbers.
        mape_non_zeros = (mae_non_zeros / target_non_zeros.clamp(min=1e-3)).mean()
        mape_avg_non_zeros = (mae_non_zeros / target_non_zeros.mean()).mean()

    metrics = {
        'mae': mae,
        'mae_per_day': mae_per_day,
        'mae_zeros': mae_zeros,
        'mae_non_zeros': mae_non_zeros,
        'mse': mse,
        'mse_per_day': mse_per_day,
        'mse_zeros': mse_zeros,
        'mse_non_zeros': mse_non_zeros,
        'r2': r2,
        'r2_per_day': r2_per_day,
        'mape_non_zeros': mape_non_zeros,
        'mape_avg_non_zeros': mape_avg_non_zeros
    }

    return metrics


def update_logging_dict(logger_dict, mode, training_loss=None, cond_mean_loss=None, likelihood_loss=None, mae_loss=None, 
                        output=None, output_winner=None, output_prob=None, target=None, 
                        brands_factorize=None, top_cat_factorize=None, ticker_factorize=None):
    """
    Update the per-epoch logging dictionary with the current iteration's results.
    """
    assert mode in ['train', 'val', 'test']

    with torch.no_grad():
        logger_dict[mode]['training_loss'].append(training_loss.view(-1).cpu().numpy())
        logger_dict[mode]['mae_loss'].append(mae_loss.view(-1).cpu().numpy())
        logger_dict[mode]['output'].append(output.cpu().numpy())
        logger_dict[mode]['target'].append(target.cpu().numpy())

        if output_winner is not None:
            logger_dict[mode]['output_winner'].append(output_winner.cpu().numpy())
        if output_prob is not None:
            logger_dict[mode]['output_prob'].append(output_prob.cpu().numpy())

        if cond_mean_loss is not None:
            logger_dict[mode]['cond_mean_loss'].append(cond_mean_loss.view(-1).cpu().numpy())
        if likelihood_loss is not None:
            logger_dict[mode]['likelihood_loss'].append(likelihood_loss.view(-1).cpu().numpy())

        if brands_factorize is not None:
            logger_dict[mode]['brands_factorize'].append(brands_factorize.cpu().numpy())
            logger_dict[mode]['top_cat_factorize'].append(top_cat_factorize.cpu().numpy())

        if ticker_factorize is not None:
            logger_dict[mode]['ticker_factorize'].append(ticker_factorize.cpu().numpy())

    if logger_dict[mode]['time_start'] is None:
        logger_dict[mode]['time_start'] = time.time()
    else:
        # update at each iteration for convenience, but only the last timestamp is useful
        logger_dict[mode]['time_elapsed'] = time.time() - logger_dict[mode]['time_start']
    return logger_dict


def print_epoch_learning_status(epoch_logger, writer, adjust_r2: bool = False) -> None:
    """
    Summarize and log metrics for an epoch across train/val/test.
    """
    epoch = epoch_logger['epoch']

    def _write_to_file_handler(np_array_data, file_handler, line_sampling_freq):
        """
        Helper function to write data into file handler.
        """
        for i_line, line in enumerate(np_array_data):
            if i_line % line_sampling_freq == 0:
                line_str = np.array2string(line, formatter={'float_kind': lambda x: "%.2f" % x}, separator=" ")
                file_handler.write(line_str[1:-1] + '\n')
        file_handler.flush()

    for mode in ['train', 'val', 'test']:

        """Init"""
        flag_empty = len(epoch_logger[mode]['training_loss']) == 0

        if flag_empty:
            continue

        time_elapsed = epoch_logger[mode]['time_elapsed']  # scalar
        training_loss = torch.from_numpy(np.concatenate(epoch_logger[mode]['training_loss']))  # array, [B * Future]
        output = torch.from_numpy(np.concatenate(epoch_logger[mode]['output']))                # array, [B, Future]
        target = torch.from_numpy(np.concatenate(epoch_logger[mode]['target']))                # array, [B, Future]

        metrics = compute_metrics(output, target, adjust_r2)

        if len(epoch_logger[mode]['output_winner']):
            output_winner = torch.from_numpy(np.concatenate(epoch_logger[mode]['output_winner']))  # array, [B, Future]
            metrics_winner = compute_metrics(output_winner, target, adjust_r2)

        if len(epoch_logger[mode]['output_prob']):
            output_prob = torch.from_numpy(np.concatenate(epoch_logger[mode]['output_prob']))  # array, [B, Future]
            metrics_prob = compute_metrics(output_prob, target, adjust_r2)

        """Logging key metrics"""
        def _log_str(mode, epoch, time_elapsed, training_loss, metrics, prefix=None):
            if prefix:
                log_str = f'{"*" * 10} {prefix} {"*" * 10}\n'
            else:
                log_str = ''
            log_str += 'Mode: {:5s}, Epoch: {:05d}, Time: {:.2f}s, Training loss: {:.3f}, MAE: {:.3f}, MSE: {:.3f}, R2: {:.3f}, Non-zero MAPE: {:.3f}, Zero-tgt MAE: {:.3f}, Non-zero-tgt MAE: {:.3f}'.format(
                mode, epoch, time_elapsed, training_loss.mean().item(), 
                metrics['mae'].mean().item(), metrics['mse'].mean().item(), 
                metrics['r2'].mean().item(), metrics['mape_non_zeros'].mean().item(), 
                metrics['mae_zeros'].mean().item(), metrics['mae_non_zeros'].mean().item())
        
            str_mae_by_day = ', '.join([f'{x:.2f}' for x in metrics['mae_per_day']])
            str_mse_by_day = ', '.join([f'{x:.2e}' for x in metrics['mse_per_day']])
            str_mse_sci = "{:.2e}".format(metrics['mse'].mean().item()).replace("e+", "e")
            log_str += "\nMulti-day metrics: MAE-by-day: [{:s}], MSE-by-day: [{:s}]. Latex friendly result: {:.1f} & {:s} & {:.3f} & {:.3f} & {:.1f} & {:.1f}".format(
                str_mae_by_day, str_mse_by_day,
                metrics['mae'].mean().item(), str_mse_sci,
                metrics['r2'].mean().item(), metrics['mape_non_zeros'].mean().item(),
                metrics['mae_zeros'].mean().item(), metrics['mae_non_zeros'].mean().item())
        
            logging.info(log_str)

        _log_str(mode, epoch, time_elapsed, training_loss, metrics, prefix='Mean prediction')

        if len(epoch_logger[mode]['output_winner']):
            _log_str(mode, epoch, time_elapsed, training_loss, metrics_winner, prefix='Winner-take-all selection')
        if len(epoch_logger[mode]['output_prob']):
            _log_str(mode, epoch, time_elapsed, training_loss, metrics_prob, prefix='Probabilistic selection')

        """Log to tensorboard and txt file"""
        if get_ddp_save_flag() and writer is not None:
            # record epoch-wise training status into tensorboard

            def _log_metrics_to_tb(mode, epoch, metrics, prefix=''):
                writer.add_scalar("{:s}_epoch/{:s}MSE".format(mode, prefix), metrics['mse'].mean().item(), epoch)
                writer.add_scalar("{:s}_epoch/{:s}MAE".format(mode, prefix), metrics['mae'].mean().item(), epoch)
                writer.add_scalar("{:s}_epoch/{:s}R2".format(mode, prefix), metrics['r2'].mean().item(), epoch)
                writer.add_scalar("{:s}_epoch/{:s}MAPE_non_zeros".format(mode, prefix), metrics['mape_non_zeros'].mean().item(), epoch)
                writer.add_scalar("{:s}_epoch/{:s}MAE_zeros".format(mode, prefix), metrics['mae_zeros'].mean().item(), epoch)
                writer.add_scalar("{:s}_epoch/{:s}MAE_non_zeros".format(mode, prefix), metrics['mae_non_zeros'].mean().item(), epoch)

            _log_metrics_to_tb(mode, epoch, metrics)

            if len(epoch_logger[mode]['output_winner']):
                _log_metrics_to_tb(mode, epoch, metrics_winner, prefix='winner_')
            if len(epoch_logger[mode]['output_prob']):
                _log_metrics_to_tb(mode, epoch, metrics_prob, prefix='prob_')

            writer.add_scalar("{:s}_epoch/training_loss".format(mode), training_loss.mean().item(), epoch)

            if len(epoch_logger[mode]['cond_mean_loss']):
                writer.add_scalar("{:s}_epoch/cond_mean_loss".format(mode), np.concatenate(epoch_logger[mode]['cond_mean_loss']).mean(), epoch)
            if len(epoch_logger[mode]['likelihood_loss']):
                writer.add_scalar("{:s}_epoch/likelihood_loss".format(mode), np.concatenate(epoch_logger[mode]['likelihood_loss']).mean(), epoch)

            if mode == 'train':
                writer.add_scalar("{:s}_epoch/learning_rate".format(mode), epoch_logger['lr'], epoch)
            writer.flush()


def check_best_model(model, ema_helper, epoch_logger, best_model_status, save_interval, config, dist_helper):
    """
    Check and update the best model based on validation MAE, and save it.
    """
    if get_ddp_save_flag():
        lowest_loss = best_model_status["loss"]
        train_mae = np.concatenate(epoch_logger['train']['mae_loss']).mean()
        val_mae = np.concatenate(epoch_logger['val']['mae_loss']).mean()
        test_mae = np.concatenate(epoch_logger['test']['mae_loss']).mean()
        epoch = epoch_logger['epoch']
        if lowest_loss > val_mae: #  if lowest_loss > mean_val_loss and epoch > save_interval:
            best_model_status["epoch"] = epoch
            best_model_status["loss"] = val_mae
            to_save = get_ckpt_data(model, ema_helper, epoch, train_mae, val_mae, test_mae, config, dist_helper)

            # save to model checkpoint dir (many network weights)
            to_save_path = os.path.join(config.model_ckpt_dir, f"{config.dataset.name}_best.pth")
            dist_save_model(to_save, to_save_path)
            logging.info(f"epoch: {epoch:05d}| validation loss : {val_mae:.2f} | best model updated at {to_save_path:s}")


def save_ckpt_model(model, ema_helper, epoch_logger, config, dist_helper):
    """
    Save the epoch checkpoint weights.
    """
    if get_ddp_save_flag():
        train_mae = np.concatenate(epoch_logger['train']['mae_loss']).mean()
        val_mae = np.concatenate(epoch_logger['val']['mae_loss']).mean()
        test_mae = np.concatenate(epoch_logger['test']['mae_loss']).mean()
        epoch = epoch_logger['epoch']
        to_save = get_ckpt_data(model, ema_helper, epoch, train_mae, val_mae, test_mae, config, dist_helper)
        to_save_path = os.path.join(config.model_ckpt_dir, f"{config.dataset.name}_{epoch:05d}.pth")
        dist_save_model(to_save, to_save_path)


def get_ckpt_data(model, ema_helper, epoch, train_loss, val_loss, test_loss, config, dist_helper):
    """
    Bundle the model state and metadata for saving to disk.
    """
    to_save = {
        'model': model.state_dict(),
        'config': config.to_dict(),
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_loss': test_loss
    }

    if ema_helper is not None:
        for ema in ema_helper:
            beta = ema.beta
            to_save['model_ema_beta_{:.4f}'.format(beta)] = ema.ema_model.state_dict()

    return to_save