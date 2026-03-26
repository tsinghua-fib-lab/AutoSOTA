# @Time : 2024/9/27 8:21 
# @Author : ltl
# @File : fedlesam.py
import torch
from .client import Client
from utils import *
from optimizer import *


class fedlesam(Client):
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):
        super(fedlesam, self).__init__(device, model_func, received_vecs, dataset, lr, args)

        # rebuild
        self.base_optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.args.weight_decay)
        self.optimizer = LESAM(self.model.parameters(), self.base_optimizer, rho=self.args.rho)

    def train(self):
        # local training
        self.model.train()
        if self.received_vecs['global_update'] != None:
            self.received_vecs['global_update'].to(self.device)

        for k in range(self.args.local_epochs):
            for i, (inputs, labels) in enumerate(self.dataset):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).long()
                self.optimizer.paras = [inputs, labels, self.loss, self.model]

                self.optimizer.step(self.received_vecs['global_update'])
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm)
                self.base_optimizer.step()

        last_state_params_list = get_mdl_params(self.model)
        self.comm_vecs['local_update_list'] = last_state_params_list - self.received_vecs['Params_list']
        self.comm_vecs['local_model_param_list'] = last_state_params_list

        return self.comm_vecs