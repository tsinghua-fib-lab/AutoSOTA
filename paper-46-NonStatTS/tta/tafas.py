# adapted from https://github.com/DequanWang/tent/blob/master/tent.py

from typing import List
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.optimizer import get_optimizer
from models.forecast import forecast
from datasets.loader import get_test_dataloader
from utils.misc import prepare_inputs
from config import get_norm_method


class Adapter(nn.Module):
    def __init__(self, cfg, model: nn.Module, norm_module=None):
        super(Adapter, self).__init__()
        self.cfg = cfg
        self.model_cfg = cfg.MODEL
        self.model = model
        self.norm_method = get_norm_method(cfg)
        self.norm_module = norm_module
        self.test_loader = get_test_dataloader(cfg)
        self.test_data = self.test_loader.dataset.test

        if self.cfg.TTA.TAFAS.CALI_MODULE:
            self.cali = Calibration(cfg).cuda()
        
        self._freeze_all_model_params()
        self.named_modules_to_adapt = self._get_named_modules_to_adapt()
        self._unfreeze_modules_to_adapt()
        self.named_params_to_adapt = self._get_named_params_to_adapt()
        
        self.optimizer = get_optimizer(self.named_params_to_adapt.values(), cfg.TTA)
        
        self.model_state, self.optimizer_state = self._copy_model_and_optimizer()
        
        cfg.TEST.BATCH_SIZE = len(self.test_loader.dataset)
        self.test_loader = get_test_dataloader(cfg)
        self.cur_step = cfg.DATA.SEQ_LEN - 2
        self.pred_step_end_dict = {}
        self.inputs_dict = {}
        self.n_adapt = 0

        self.mse_all = []
        self.mae_all = []
        
    def forward(self, enc_window, enc_window_stamp, dec_window, dec_window_stamp):
        raise NotImplementedError
    
    def reset(self):
        self._load_model_and_optimizer()
    
    def _copy_model_and_optimizer(self):
        model_state = deepcopy(self.model.state_dict())
        optimizer_state = deepcopy(self.optimizer.state_dict())
        return model_state, optimizer_state

    def _load_model_and_optimizer(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
    
    def _get_all_models(self):
        models = [self.model]
        if self.norm_module is not None:
            models.append(self.norm_module)
        if self.cfg.TTA.TAFAS.CALI_MODULE:
            models.append(self.cali)
        return models

    def _freeze_all_model_params(self):
        for model in self._get_all_models():
            for param in model.parameters():
                param.requires_grad_(False)
    
    def _get_named_modules(self):
        named_modules = []
        for model in self._get_all_models():
            named_modules += list(model.named_modules())
        return named_modules
    
    def _get_named_modules_to_adapt(self) -> List[str]:
        named_modules = self._get_named_modules()
        if self.cfg.TTA.MODULE_NAMES_TO_ADAPT == 'all':
            return named_modules
        
        named_modules_to_adapt = []
        for module_name in self.cfg.TTA.MODULE_NAMES_TO_ADAPT.split(','):
            exact_match = '(exact)' in module_name
            module_name = module_name.replace('(exact)', '')
            if exact_match:
                named_modules_to_adapt += [(name, module) for name, module in named_modules if name == module_name]
            else:
                named_modules_to_adapt += [(name, module) for name, module in named_modules if module_name in name]

        assert len(named_modules_to_adapt) > 0
        return named_modules_to_adapt
    
    def _unfreeze_modules_to_adapt(self):
        for _, module in self.named_modules_to_adapt:
            module.requires_grad_(True)
    
    def _get_named_params_to_adapt(self):
        named_params_to_adapt = {}
        for model in self._get_all_models():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    named_params_to_adapt[name] = param
        return named_params_to_adapt
    
    def switch_model_to_train(self):
        for model in self._get_all_models():
            model.train()
    
    def switch_model_to_eval(self):
        for model in self._get_all_models():
            model.eval()
    
    @torch.enable_grad()
    def adapt_tafas(self):
        batch_start = 0
        batch_end = 0
        batch_idx = 0
        is_last = False
        test_len = len(self.test_loader.dataset)
            
        self.switch_model_to_eval()
        for idx, inputs in enumerate(self.test_loader):
            enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)
            while batch_end < len(enc_window_all):
                enc_window_first = enc_window_all[batch_start]
                
                if self.cfg.TTA.TAFAS.PAAS:
                    period, batch_size = self._calculate_period_and_batch_size(enc_window_first)
                else:
                    batch_size = self.cfg.TTA.TAFAS.BATCH_SIZE
                    period = batch_size - 1
                batch_end = batch_start + batch_size

                if batch_end > len(enc_window_all):
                    batch_end = len(enc_window_all)
                    batch_size = batch_end - batch_start
                    is_last = True

                self.cur_step += batch_size
    
                inputs = enc_window_all[batch_start:batch_end], enc_window_stamp_all[batch_start:batch_end], dec_window_all[batch_start:batch_end], dec_window_stamp_all[batch_start:batch_end]
                
                self.pred_step_end_dict[batch_idx] = self.cur_step + self.cfg.DATA.PRED_LEN
                self.inputs_dict[batch_idx] = inputs
                
                self._adapt_with_full_ground_truth_if_available()
                pred, ground_truth = self._adapt_with_partial_ground_truth(inputs, period, batch_size, batch_idx)
                
                if self.cfg.TTA.TAFAS.ADJUST_PRED:
                    pred, ground_truth = self._adjust_prediction(pred, inputs, batch_size, period)
                
                mse = F.mse_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                mae = F.l1_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                self.mse_all.append(mse)
                self.mae_all.append(mae)
                
                batch_start = batch_end
                batch_idx += 1
        
        assert self.cur_step == len(self.test_data) - self.cfg.DATA.PRED_LEN - 1
        
        self.mse_all = np.concatenate(self.mse_all)
        self.mae_all = np.concatenate(self.mae_all)
        assert len(self.mse_all) == len(self.test_loader.dataset)
        
        print('After TSF-TTA of TAFAS')
        print(f'Number of adaptations: {self.n_adapt}')
        print(f'Test MSE: {self.mse_all.mean():.4f}, Test MAE: {self.mae_all.mean():.4f}')
        print()

        if self.cfg.WANDB.ENABLE:
            wandb.log({
                'mse': self.mse_all.mean(), 
                'mae': self.mae_all.mean(),
                'n_adapt': self.n_adapt,
            })
        
        self.model.eval()
    
    def _calculate_period_and_batch_size(self, enc_window_first):
        fft_result = torch.fft.rfft(enc_window_first - enc_window_first.mean(dim=0), dim=0)
        amplitude = torch.abs(fft_result)
        power = torch.mean(amplitude ** 2, dim=0)
        try:
            period = enc_window_first.shape[0] // torch.argmax(amplitude[:, power.argmax()]).item()
        except:
            period = 24
        period *= self.cfg.TTA.TAFAS.PERIOD_N
        batch_size = period + 1
        return period, batch_size

    def _adapt_with_full_ground_truth_if_available(self):
        while self.cur_step >= self.pred_step_end_dict[min(self.pred_step_end_dict.keys())]:
            batch_idx_available = min(self.pred_step_end_dict.keys())
            inputs_history = self.inputs_dict.pop(batch_idx_available)
            for _ in range(self.cfg.TTA.TAFAS.STEPS):
                self.n_adapt += 1
                
                self.switch_model_to_train()

                if self.cfg.TTA.TAFAS.CALI_MODULE and self.cfg.MODEL.NAME != 'PatchTST':
                    inputs_history = self.cali.input_calibration(inputs_history)
                pred, ground_truth = forecast(self.cfg, inputs_history, self.model, self.norm_module)
                
                if self.cfg.TTA.TAFAS.CALI_MODULE:
                    pred = self.cali.output_calibration(pred)
                    
                loss = 0.5 * F.mse_loss(pred, ground_truth) + 0.5 * F.l1_loss(pred, ground_truth)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                self.switch_model_to_eval()
            
            self.pred_step_end_dict.pop(batch_idx_available)

    def _adapt_with_partial_ground_truth(self, inputs, period, batch_size, batch_idx):
        for _ in range(self.cfg.TTA.TAFAS.STEPS):
            self.n_adapt += 1
            
            current_inputs = inputs
            if self.cfg.TTA.TAFAS.CALI_MODULE and self.cfg.MODEL.NAME != 'PatchTST':
                current_inputs = self.cali.input_calibration(inputs)
            pred, ground_truth = forecast(self.cfg, current_inputs, self.model, self.norm_module)
            
            if self.cfg.TTA.TAFAS.CALI_MODULE:
                pred = self.cali.output_calibration(pred)
            
            # Sample-specific partial GT: sample i has (period - i) available GT steps
            # Weight samples equally by normalizing their individual losses
            loss_sum = None
            n_valid = 0
            for i in range(batch_size - 1):  # last sample has 0 GT steps
                avail_steps = period - i
                if avail_steps <= 0:
                    break
                pred_i = pred[i:i+1, :avail_steps, :]
                gt_i = ground_truth[i:i+1, :avail_steps, :]
                loss_i = 0.5 * F.mse_loss(pred_i, gt_i) + 0.5 * F.l1_loss(pred_i, gt_i)
                loss_sum = loss_i if loss_sum is None else loss_sum + loss_i
                n_valid += 1
            
            if loss_sum is not None and n_valid > 0:
                mse_partial = loss_sum / n_valid
            else:
                # Fallback
                mse_partial = 0.5 * F.mse_loss(pred[:, :period, :], ground_truth[:, :period, :]) + 0.5 * F.l1_loss(pred[:, :period, :], ground_truth[:, :period, :])
                
            self.optimizer.zero_grad()
            mse_partial.backward()
            self.optimizer.step()
        return pred, ground_truth

    @torch.no_grad()
    def _adjust_prediction(self, pred, inputs, batch_size, period):
        if self.cfg.TTA.TAFAS.CALI_MODULE:
            inputs = self.cali.input_calibration(inputs)
        pred_after_adapt, ground_truth = forecast(self.cfg, inputs, self.model, self.norm_module)
        if self.cfg.TTA.TAFAS.CALI_MODULE:
            pred_after_adapt = self.cali.output_calibration(pred_after_adapt)
        
        for i in range(batch_size-1):
            pred[i, period-i:] = pred_after_adapt[i, period-i:]
        
        return pred, ground_truth
    
    def adapt(self):
        self.adapt_tafas()

def build_adapter(cfg, model, norm_module=None):
    adapter = Adapter(cfg, model, norm_module)
    return adapter


class GCM(nn.Module):
    def __init__(self, window_len, n_var=1, hidden_dim=64, gating_init=0.01, var_wise=True):
        super(GCM, self).__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.var_wise = var_wise
        if var_wise:
            self.weight = nn.Parameter(torch.Tensor(window_len, window_len, n_var))
        else:
            self.weight = nn.Parameter(torch.Tensor(window_len, window_len))
        self.weight.data.zero_()
        self.gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.bias = nn.Parameter(torch.zeros(window_len, n_var))

    def forward(self, x):
        if self.var_wise:
            x = x + torch.tanh(self.gating) * (torch.einsum('biv,iov->bov', x, self.weight) + self.bias)
        else:
            x = x + torch.tanh(self.gating) * (torch.einsum('biv,io->bov', x, self.weight) + self.bias)
        return x


class Calibration(nn.Module):
    def __init__(self, cfg):
        super(Calibration, self).__init__()
        self.cfg = cfg
        self.seq_len = cfg.DATA.SEQ_LEN
        self.pred_len = cfg.DATA.PRED_LEN
        self.n_var = cfg.DATA.N_VAR
        self.hidden_dim = cfg.TTA.TAFAS.HIDDEN_DIM
        self.gating_init = cfg.TTA.TAFAS.GATING_INIT
        self.var_wise = cfg.TTA.TAFAS.GCM_VAR_WISE
        if cfg.MODEL.NAME == 'PatchTST':
            self.in_cali = GCM(self.seq_len, 1, self.hidden_dim, self.gating_init, self.var_wise)
            self.out_cali = GCM(self.pred_len, 1, self.hidden_dim, self.gating_init, self.var_wise)
        else:
            self.in_cali = GCM(self.seq_len, self.n_var, self.hidden_dim, self.gating_init, self.var_wise)
            self.out_cali = GCM(self.pred_len, self.n_var, self.hidden_dim, self.gating_init, self.var_wise)
        
    def input_calibration(self, inputs):
        enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
        enc_window = self.in_cali(enc_window)
        return enc_window, enc_window_stamp, dec_window, dec_window_stamp

    def output_calibration(self, outputs):
        return self.out_cali(outputs)
