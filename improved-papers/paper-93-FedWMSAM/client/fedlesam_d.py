from utils import *
from .client import Client
from optimizer import *
import torch


def loss_dyn(predictions, labels, param_list, delta_list, lamb):
    return torch.nn.functional.cross_entropy(predictions, labels, reduction='mean') + torch.sum(
        param_list * delta_list) * lamb


class fedlesam_d(Client):
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):
        super(fedlesam_d, self).__init__(device, model_func, received_vecs, dataset, lr, args)

        self.base_optimizer = torch.optim.SGD(self.model.parameters(), lr=lr,
                                              weight_decay=self.args.weight_decay + self.args.lamb)

        self.optimizer = LESAM_D(self.model.parameters(), self.base_optimizer, rho=self.args.rho)
        self.loss = loss_dyn

    def train(self):
        # local training
        self.model.train()

        for k in range(self.args.local_epochs):
            for i, (inputs, labels) in enumerate(self.dataset):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).long()
                param_list = param_to_vector(self.model)
                delta_list = self.received_vecs['Local_dual_correction'].to(self.device)
                self.optimizer.paras = [inputs, labels, self.loss, self.model, delta_list, self.args.lamb]
                self.optimizer.step(self.received_vecs['global_update'])
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm)
                self.base_optimizer.step()

        last_state_params_list = get_mdl_params(self.model)
        self.comm_vecs['local_update_list'] = last_state_params_list - self.received_vecs['Params_list']
        self.comm_vecs['local_model_param_list'] = last_state_params_list

        return self.comm_vecs