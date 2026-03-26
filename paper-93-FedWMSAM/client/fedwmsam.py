import numpy as np

from utils import *
from .client import Client
from optimizer import *
from torch.nn.functional import cross_entropy


def loss_gamma(predictions, labels, param_list, momentum_list, alpha):
    return alpha * torch.nn.functional.cross_entropy(predictions, labels, reduction='mean') + torch.sum(
        param_list * momentum_list) * (1 - alpha)


class fedwmsam(Client):
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):
        super(fedwmsam, self).__init__(device, model_func, received_vecs, dataset, lr, args)
        self.target_model = param_to_vector(self.model).to(self.device)
        self.sam_optimizer = WMSAM(self.model.parameters(), self.optimizer, rho=self.args.rho)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.args.weight_decay)
        self.loss = loss_gamma

    def train(self):
        # local training
        self.model.train()
        momentum_list = self.received_vecs['Client_momentum'].to(self.device)
        for k in range(self.args.local_epochs):
            for i, (inputs, labels) in enumerate(self.dataset):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).long()
                self.sam_optimizer.paras = [inputs, labels, self.loss, self.model, momentum_list, self.args.alpha]
                differ_vector = self.target_model - param_to_vector(self.model).to(self.device)
                self.sam_optimizer.step(differ_vector)

                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm)
                self.optimizer.step()
                self.target_model += momentum_list

        last_state_params_list = get_mdl_params(self.model)
        self.comm_vecs['local_update_list'] = last_state_params_list - self.received_vecs['Params_list']
        self.comm_vecs['local_model_param_list'] = last_state_params_list

        return self.comm_vecs

