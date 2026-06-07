import torch
import torch.nn as nn
import time
import datetime
import pytorch_lightning as pl
import os
import numpy as np
import sys
import torch.nn.functional as F
import anndata as ann

sys.path.insert(0, "../")
from utils import pearsonr, compute_correlations


class TrainerModel(pl.LightningModule):

    def __init__(self, config, model, ):
        super().__init__()
        self.model = model
        self.config = config
        self.criterion = nn.MSELoss()
        self.automatic_optimization = False
        self.min_loss = float("inf")
        self.max_corr = float("-inf")
        self.max_eval_corr = float("-inf")
        self.min_eval_loss = float("inf")
        self.start_time = None
        self.last_saved = None
        self.validation_step_outputs = []
        self.test_outputs = []

    def correlationMetric(self, x, y):
        corr = 0
        for idx in range(x.size(1)):
            corr += pearsonr(x[:, idx], y[:, idx])
        corr /= (idx + 1)
        return (1 - corr).mean()

    def training_step(self, data, idx, val=True):

        if self.current_epoch == 0 and idx == 0:
            self.start_time = time.time()

        optimizer = self.optimizers()
        optimizer.zero_grad()
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        window_bgrl_loss, example_bgrl_loss, pred_count = self.model(x_dict, edge_index_dict, val)

        reg_loss = self.criterion(pred_count, data["target"]["y"])
        corr_loss = self.correlationMetric(pred_count, data["target"]["y"])

        total_loss = reg_loss + 0.5 * corr_loss + 0.2 * window_bgrl_loss + 0.1 * example_bgrl_loss  #


        self.manual_backward(total_loss)  # reg_loss + corr_loss * 0.5
        optimizer.step()
        self.produce_log(reg_loss.detach(), corr_loss.detach(), idx)  #

    def produce_log(self, reg_loss, corr, idx):

        train_loss = self.all_gather(reg_loss).mean().item()
        train_corr = self.all_gather(corr).mean().item()

        self.min_loss = min(self.min_loss, train_loss)

        if self.trainer.is_global_zero and reg_loss.device.index == 0 and idx % self.config.verbose_step == 0:
            current_lr = self.optimizers().param_groups[0]['lr']

            len_loader = 200

            batches_done = self.current_epoch * len_loader + idx + 1
            batches_left = self.trainer.max_epochs * len_loader - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - self.start_time) / batches_done)

            self.config.logfun(
                "[Epoch %d/%d] [Batch %d/%d] [reg_loss: %f, Corr: %f, lr: %f] [Min Loss: %f] ETA: %s" %
                (self.current_epoch,
                 self.trainer.max_epochs,
                 idx,
                 len_loader,
                 train_loss,
                 train_corr,
                 current_lr,
                 self.min_loss,
                 time_left
                 )

            )

    def validation_step(self, data, idx, val=False):
        pred_count = self.model(data.x_dict, data.edge_index_dict, val)
        self.validation_step_outputs = []
        self.validation_step_outputs.append(pred_count)
        self.validation_step_outputs.append(data["target"]["y"])
        return pred_count, data["target"]["y"]

    def on_validation_epoch_end(self):

        logfun = self.config.logfun

        pred_count = self.validation_step_outputs[0]
        count = self.validation_step_outputs[1]

        total_loss = self.criterion(pred_count, count).item()
        gene_corr = compute_correlations(count, pred_count, True)
        corr = np.mean(gene_corr)

        if self.trainer.is_global_zero:
            if corr > self.max_eval_corr:
                self.save(self.current_epoch, total_loss, corr)
            self.max_eval_corr = max(self.max_eval_corr, corr)
            self.min_eval_loss = min(self.min_eval_loss, total_loss)

            logfun("==" * 25)
            logfun(
                "[Corr :%f, Loss: %f] [Min Loss :%f, Max Corr: %f]" %
                (corr,
                 total_loss,
                 self.min_eval_loss,
                 self.max_eval_corr,
                 )
            )
            logfun("==" * 25)

    def save(self, epoch, loss, acc):

        self.config.logfun(self.last_saved)
        output_path = os.path.join(self.config.store_dir, "best.pt")
        self.last_saved = output_path
        torch.save(self.model.state_dict(), output_path)
        self.config.logfun("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
            betas=(0.9, 0.999),
            weight_decay=self.config.weight_decay,
        )

        return optimizer
