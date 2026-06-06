from typing import Tuple, Optional

import torch
import torch.nn as nn

from config import get_norm_method
from utils.misc import prepare_inputs


def forecast(
    cfg, 
    inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], 
    model: nn.Module,
    norm_module: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
    norm_method = get_norm_method(cfg)
    if norm_method == 'SAN':
        enc_window, statistics = norm_module.normalize(enc_window)
    elif norm_method == 'RevIN':
        enc_window = norm_module(enc_window, 'norm')
    elif norm_method == 'DishTS':
        enc_window, _ = norm_module(enc_window, 'forward')
    else:  # Normalization from Non-stationary Transformer
        means = enc_window.mean(1, keepdim=True).detach()
        enc_window = enc_window - means
        # stdev = torch.sqrt(torch.var(enc_window, dim=1, keepdim=True, unbiased=False) + 1e-5)
        stdev = torch.sqrt(torch.var(enc_window, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        enc_window /= stdev
    
    ground_truth = dec_window[:, -cfg.DATA.PRED_LEN:, cfg.DATA.TARGET_START_IDX:].float()
    dec_zeros = torch.zeros_like(dec_window[:, -cfg.DATA.PRED_LEN:, :]).float()
    dec_window = torch.cat([dec_window[:, :cfg.DATA.LABEL_LEN:, :], dec_zeros], dim=1).float().cuda()
    
    model_cfg = cfg.MODEL
    if model_cfg.output_attention:
        pred = model(enc_window, enc_window_stamp, dec_window, dec_window_stamp)[0]
    else:
        pred = model(enc_window, enc_window_stamp, dec_window, dec_window_stamp)
    
    pred = pred[:, -cfg.DATA.PRED_LEN:, cfg.DATA.TARGET_START_IDX:]
    
    if norm_method == 'SAN':
        pred = norm_module.de_normalize(pred, statistics)
    elif norm_method == 'RevIN':
        pred = norm_module(pred, 'denorm')
    elif norm_method == 'DishTS':
        pred = norm_module(pred, 'inverse')
    else:  # De-Normalization from Non-stationary Transformer
        pred = pred * (stdev[:, 0, :].unsqueeze(1).repeat(1, cfg.DATA.PRED_LEN, 1))
        pred = pred + (means[:, 0, :].unsqueeze(1).repeat(1, cfg.DATA.PRED_LEN, 1))
    
    return pred, ground_truth
