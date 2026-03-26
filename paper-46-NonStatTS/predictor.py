import os
from typing import Dict, Optional
import pickle

import wandb
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets.loader import get_train_dataloader, get_test_dataloader
from utils.misc import prepare_inputs
from utils.misc import mkdir
from config import get_norm_method


class Predictor:
    def __init__(self, cfg, model, norm_module: Optional[torch.nn.Module] = None):
        self.cfg = cfg

        self.model = model
        self.norm_method = get_norm_method(cfg)
        self.norm_module = norm_module

        cfg.TRAIN.SHUFFLE, cfg.TRAIN.DROP_LAST = False, False
        self.train_loader = get_train_dataloader(cfg)
        self.test_loader = get_test_dataloader(cfg)
        
        self.test_errors, self.train_errors = self._get_test_errors(), self._get_train_errors()

    @torch.no_grad()
    def predict(self):
        self.model.eval()
        self.norm_module.requires_grad_(False).eval() if self.norm_module is not None else None
        log_dict = {}
        
        self.errors_all = {
            "test_mse": self.test_errors['mse'], 
            "test_mae": self.test_errors['mae'], 
            "train_mse": self.train_errors['mse'], 
            "train_mae": self.train_errors['mae'], 
        }

        results = self.get_results()  # {test_mse: , test_mae:, train_mse: train_mae: }
        self.save_results(results)
        
        self.save_to_npy(**self.errors_all)

        # log to W&B
        log_dict.update({f"Test/{metric}": value for metric, value in results.items()})
        if self.cfg.WANDB.ENABLE:
            wandb.log(log_dict)

    @torch.no_grad()
    def _get_errors_from_dataloader(self, dataloader, tta=False, split='test') -> Dict[str, np.ndarray]:
        self.model.eval()
        self.norm_module.requires_grad_(False).eval() if self.norm_module is not None else None
        mse_all = []
        mae_all = []
        
        for inputs in tqdm(dataloader, desc='Calculating Errors'):
            enc_window_raw, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
            if self.norm_method == 'SAN':
                enc_window, statistics_pred = self.norm_module.normalize(enc_window_raw)
            else:  # Normalization from Non-stationary Transformer
                means = enc_window_raw.mean(1, keepdim=True).detach()
                enc_window = enc_window_raw - means
                stdev = torch.sqrt(torch.var(enc_window, dim=1, keepdim=True, unbiased=False) + 1e-5)
                enc_window /= stdev
            
            ground_truth = dec_window[:, -self.cfg.DATA.PRED_LEN:, self.cfg.DATA.TARGET_START_IDX:].float()
            dec_zeros = torch.zeros_like(dec_window[:, -self.cfg.DATA.PRED_LEN:, :]).float()
            dec_window = torch.cat([dec_window[:, :self.cfg.DATA.LABEL_LEN:, :], dec_zeros], dim=1).float().cuda()
            
            model_cfg = self.cfg.MODEL
            pred = self.model(enc_window, enc_window_stamp, dec_window, dec_window_stamp)
            if model_cfg.output_attention:
                pred = pred[0]
            
            pred = pred[:, -self.cfg.DATA.PRED_LEN:, self.cfg.DATA.TARGET_START_IDX:]
            
            if self.norm_method == 'SAN':
                pred = self.norm_module.de_normalize(pred, statistics_pred)
            else:  # De-Normalization from Non-stationary Transformer
                pred = pred * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.cfg.DATA.PRED_LEN, 1))
                pred = pred + (means[:, 0, :].unsqueeze(1).repeat(1, self.cfg.DATA.PRED_LEN, 1))
            
            mse = F.mse_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1))
            mae = F.l1_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1))
            
            mse_all.append(mse)
            mae_all.append(mae)
                
        mse_all = torch.flatten(torch.concat(mse_all, dim=0)).cpu().numpy()
        mae_all = torch.flatten(torch.concat(mae_all, dim=0)).cpu().numpy()

        return {'mse': mse_all, 'mae': mae_all}

    def _get_train_errors(self):
        return self._get_errors_from_dataloader(self.train_loader, tta=False, split='train')

    def _get_test_errors(self):
        return self._get_errors_from_dataloader(self.test_loader, tta=self.cfg.TTA.ENABLE, split='test')

    def get_results(self) -> Dict[str, float]:
        test_mse = self.test_errors['mse'].mean().astype(float)
        test_mae = self.test_errors['mae'].mean().astype(float)
        train_mse = self.train_errors['mse'].mean().astype(float)
        train_mae = self.train_errors['mae'].mean().astype(float)
        
        return {"test_mse": test_mse, "test_mae": test_mae, "train_mse": train_mse, "train_mae": train_mae}

    def save_results(self, results):
        results_string = ", ".join([f"{metric}: {value:.04f}" for metric, value in results.items()])
        print("Results without TSF-TTA:")
        print(results_string)

        with open(os.path.join(mkdir(self.cfg.RESULT_DIR) / "test.txt"), "w") as f:
            f.write(results_string)

    def save_to_npy(self, **kwargs):
        for key, value in kwargs.items():
            np.save(os.path.join(self.cfg.RESULT_DIR, f"{key}.npy"), value)
