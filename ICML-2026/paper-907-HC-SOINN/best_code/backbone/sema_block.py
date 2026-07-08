import torch
from torch import nn
from typing import List
import copy
import logging
from backbone.sema_components import Adapter, AE, Records


class AdapterModule(nn.Module):    
    def __init__(self, config, adapter_id, writer, device=None):
        super().__init__()
        self.config = config
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.functional = Adapter(self.config, adapter_id, dropout=0.1, bottleneck=self.config.ffn_num,
                                init_option=self.config.ffn_adapter_init_option,
                                adapter_scalar=self.config.ffn_adapter_scalar,
                                adapter_layernorm_option=self.config.ffn_adapter_layernorm_option,
                                )
        layer_id = int(adapter_id.split('.')[0])
        self.not_addition_layer = layer_id < config.adapt_start_layer or layer_id > config.adapt_end_layer
        if self.not_addition_layer:
            self.rd = None
        else:
            self.rd = AE(self.config)
        self.activation = nn.ReLU()
        self.newly_added = True
        self.adapter_id = adapter_id
        self.writer = writer
        self.rd_loss_record = Records(max_len=config.buffer_size)

    def forward(self, x):
        func_out = self.functional(x)
        if self.not_addition_layer:
            rd_loss = torch.tensor(0.).to(self.device)
            return func_out, rd_loss, torch.zeros_like(rd_loss).to(self.device)
        else:
            rd_loss = self.rd.compute_reconstruction_loss(x)
        z_score = self.get_z_score_deviation(rd_loss)
        if self.training:
            self.add_z_score_record(rd_loss)
        return func_out, rd_loss, z_score

    def get_z_score_deviation(self, rd_loss):
        mean, stddev = self.rd_loss_record.mean, self.rd_loss_record.stddev
        if not self.rd_loss_record.length > 2:
            return torch.zeros_like(rd_loss).to(self.device)
        z_score = (rd_loss-mean)/stddev
        z_score = torch.abs(z_score)
        return z_score
    
    def add_z_score_record(self, rd_loss):
        self.rd_loss_record.add_record(rd_loss.detach().cpu())


class SEMAModules(nn.Module):
    def __init__(self, config, layer_id, writer, device=None):
        super().__init__()
        self.adapters: List[Adapter] = nn.ModuleList()
        self.config = config
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.act_func = nn.ReLU()
        self.layer_id = layer_id
        self.writer = writer
        self.newly_added = True
        self.added_for_task = True
        self.adapt_start_layer = config.adapt_start_layer
        self.adapt_end_layer = config.adapt_end_layer
        # initialize with one adapter
        self.add_adapter(initialize=True)
        self.added_adapter = 0

        self.router = nn.Linear(config.d_model, 1).to(self.device)
        self.new_router = None
        self.detecting_outlier = False
        
    @property
    def num_adapters(self):
        return len(self.adapters)

    def set_new_router(self):
        self.new_router = nn.Linear(self.config.d_model, 1).to(self.device)
       
    def fix_router(self):
        trained_router = nn.Linear(self.config.d_model, len(self.adapters)).to(self.device)
        old_router = self.router
        weight = copy.deepcopy(old_router.weight.data)
        new_weight = copy.deepcopy(self.new_router.weight.data)
        weight = torch.cat([weight, new_weight])
        trained_router.weight = nn.Parameter(weight)
        bias = copy.deepcopy(old_router.bias.data)
        new_bias = copy.deepcopy(self.new_router.bias.data)
        bias = torch.cat([bias, new_bias])
        trained_router.bias = nn.Parameter(bias)
        self.router = trained_router
        self.new_router = None
        

    def add_adapter(self, initialize=False):
        adapter_id = f"{self.layer_id}.{len(self.adapters)}"
        new_adapter = AdapterModule(self.config, adapter_id, self.writer, device=self.device).to(self.device)
        self.newly_added = True
        self.added_for_task = True
        self.adapters.append(new_adapter)
        if not initialize:
            self.set_new_router()
        logging.info(f"Adapter {adapter_id} added at block {self.layer_id}")

    def forward(self, x):
        rd_loss = torch.tensor(0.).to(self.device)

        added = False
        not_addition_layer = self.layer_id < self.adapt_start_layer or self.layer_id > self.adapt_end_layer
        if not_addition_layer:
            func_out, _, _= self.adapters[-1](x)
        else:
            func_outs, rd_losses, z_scores = [], [], []
            for adapter in self.adapters:
                func_out, rd_loss, z_score = adapter(x)
                func_outs.append(func_out)  
                rd_losses.append(rd_loss)   
                z_scores.append(z_score)

            func_outs = torch.stack(func_outs)
            rd_losses = torch.stack(rd_losses)
            z_scores = torch.stack(z_scores)
            
            addition_criteria = z_scores.mean(dim=1).min() > self.config.exp_threshold \
                and self.layer_id >= self.adapt_start_layer \
                and self.layer_id <= self.adapt_end_layer \
                and not self.added_for_task and self.detecting_outlier

            if addition_criteria:
                self.add_adapter()
                out = {"func_out": torch.zeros_like(func_outs[0]).to(self.device), "rd_loss": torch.tensor(0.).to(self.device), "added": True}
                return out
            else:
                logits = self.router(x.mean(dim=1))
                if self.new_router is not None:
                    new_logits = self.new_router(x.mean(dim=1))
                    logits = torch.cat([logits, new_logits], dim=1)
                mask = torch.softmax(logits, dim=1) 
                func_out = (func_outs * mask.transpose(0,1).unsqueeze(-1).unsqueeze(-1)).sum(dim=0)
                if self.adapters[-1].newly_added:
                    rd_loss = rd_losses[-1].mean()
                else:
                    rd_loss = torch.tensor(0.).to(self.device)

        out = {"func_out": func_out, "rd_loss": rd_loss, "added": added}
        return out

    def end_of_task_training(self):
        self.freeze_functional()
        self.freeze_rd()
        self.reset_newly_added_status()
        self.added_for_task = False
    

    def reset_newly_added_status(self):
        self.newly_added = False
        for adapter in self.adapters:
            adapter.newly_added = False

    def freeze_functional(self):
        adapter_ls = self.adapters
        for adapter in adapter_ls:
            for param in adapter.functional.parameters():
                param.requires_grad = False
                param._grad = None
        if self.new_router is not None:
            self.fix_router()
        for param in self.router.parameters():
            param.requires_grad = False
            param._grad = None

    def freeze_rd(self):
        adapter_ls = self.adapters
        for adapter in adapter_ls:
            if adapter.rd is not None:
                for param in adapter.rd.parameters():
                    param.requires_grad = False
                    param._grad = None
                adapter.rd_loss_record.updating = False








