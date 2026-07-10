from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
import os
import time
import warnings
import numpy as np
import random
from models import *

from utils.metrics import metric
from utils.tools import EarlyStopping



warnings.filterwarnings("ignore")

class FreDFLoss(nn.Module):
    def __init__(self, temp_weight=0.2, freq_weight=0.8):
        super(FreDFLoss, self).__init__()
        self.temp_weight = temp_weight
        self.freq_weight = freq_weight
        
    def forward(self, pred, true):
        temp_mse = nn.MSELoss()(pred, true)
        freq_mse = nn.L1Loss()(torch.fft.rfft(pred, dim=1), torch.fft.rfft(true, dim=1))
        return self.temp_weight * temp_mse + self.freq_weight * freq_mse

class Exp_Forecast():
    def __init__(self, args):
        self.args = args
        
        self.model_dict = {
            "Transformer": Transformer,
        }
        self.model = self._build_model().to(args.device)
        
        self.fredf_loss = args.fredf_loss if hasattr(args, "fredf_loss") else 0
        self.horizon_weight_beta = args.horizon_weight_beta if hasattr(args, "horizon_weight_beta") else 0.0
        if self.horizon_weight_beta > 0:
            T = self.args.pred_len
            t = torch.arange(T).float()
            self.horizon_weights = torch.exp(-self.horizon_weight_beta * t / T)
            self.horizon_weights = self.horizon_weights / self.horizon_weights.mean()
    
    def _build_model(self):
        model = self.model_dict[self.args.model](self.args)
        return model
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim
    
    def _select_scheduler(self):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs, eta_min=self.args.min_learning_rate)
        return scheduler
    
    def _select_criterion(self, override=False):
        if self.fredf_loss and not override:
            criterion = FreDFLoss()
        else:
            criterion = nn.MSELoss()
        eval_criterion = nn.L1Loss()
        return criterion, eval_criterion
    
    def train(self):
        train_data, train_loader = self._get_data("train")
        val_data, val_loader = self._get_data("val")
        
        criterion, eval_criterion = self._select_criterion()
        optimizer = self._select_optimizer()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, checkpointing=False)
        
        for epoch in range(self.args.epochs):
            self.model.train()
            
            epoch_loss = 0.0
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.args.device)
                batch_y = batch_y.float().to(self.args.device)
                
                if self.args.sampling_rate < 1.0:
                    sampled_indices = np.random.choice(batch_x.shape[2], size=int(batch_x.shape[2] * self.args.sampling_rate), replace=False)
                    batch_x = batch_x[:, :, sampled_indices]
                    batch_y = batch_y[:, :, sampled_indices]
                
                if "PEMS" in self.args.data or "Solar" in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.args.device)
                    batch_y_mark = batch_y_mark.float().to(self.args.device)
                
                y_hat = self.model(batch_x, batch_x_mark)
                if self.horizon_weight_beta > 0:
                    hw = self.horizon_weights.to(batch_y.device)
                    mse_per_step = ((y_hat - batch_y) ** 2).mean(dim=-1)
                    weighted_mse = (mse_per_step * hw.view(1, -1)).mean()
                    if self.fredf_loss:
                        freq_loss = nn.L1Loss()(torch.fft.rfft(y_hat, dim=1), torch.fft.rfft(batch_y, dim=1))
                        loss = 0.2 * weighted_mse + 0.8 * freq_loss
                    else:
                        loss = weighted_mse
                else:
                    loss = criterion(y_hat, batch_y)
                epoch_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                optimizer.step()
            
            val_loss = self.val(val_loader, criterion)
            print(f"Epoch {epoch+1}, Loss (MSE): {epoch_loss / len(train_loader)}, Val Loss (MSE): {val_loss}, Time: {time.time() - epoch_time} sec")
            
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            adjust_learning_rate(optimizer, epoch+1, self.args)

        return self.model
    
    def val(self, val_loader, criterion):
        self.model.eval()
        
        total_loss = 0.0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
                batch_x = batch_x.float().to(self.args.device)
                batch_y = batch_y.float().to(self.args.device)
                
                if "PEMS" in self.args.data or "Solar" in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.args.device)
                    batch_y_mark = batch_y_mark.float().to(self.args.device)
                    
                y_hat = self.model(batch_x, batch_x_mark)
                if self.horizon_weight_beta > 0:
                    hw = self.horizon_weights.to(batch_y.device)
                    mse_per_step = ((y_hat - batch_y) ** 2).mean(dim=-1)
                    vloss = (mse_per_step * hw.view(1, -1)).mean()
                    if self.fredf_loss:
                        freq_loss = nn.L1Loss()(torch.fft.rfft(y_hat, dim=1), torch.fft.rfft(batch_y, dim=1))
                        vloss = 0.2 * vloss + 0.8 * freq_loss
                    total_loss += vloss.item()
                else:
                    total_loss += criterion(y_hat, batch_y).item()
                
        return total_loss / len(val_loader)
    
    def test(self):
        test_data, test_loader = self._get_data(flag="test")
        
        criterion, eval_criterion = self._select_criterion(override=True)
        
        total_loss_mse = 0.0
        total_loss_mae = 0.0
        
        if self.args.save_pred:
            all_predictions = []
            all_ground_truth = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.args.device)
                batch_y = batch_y.float().to(self.args.device)
                
                if "PEMS" in self.args.data or "Solar" in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.args.device)
                    batch_y_mark = batch_y_mark.float().to(self.args.device)
                
                y_hat = self.model(batch_x, batch_x_mark, save_attn=True if (self.args.save_pred and i == 0) else False)
                
                total_loss_mse += criterion(y_hat, batch_y).item()
                total_loss_mae += eval_criterion(y_hat, batch_y).item()
                
        return total_loss_mse / len(test_loader), total_loss_mae / len(test_loader)
    
    def predict(self):
        pred_data, pred_loader = self._get_data(flag="pred")
        
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i, (batch_x, batch_x_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.args.device)
                
                if "PEMS" in self.args.data or "Solar" in self.args.data:
                    batch_x_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.args.device)
                    
                y_hat = self.model(batch_x, batch_x_mark)
                preds.append(y_hat)
                
        return preds
