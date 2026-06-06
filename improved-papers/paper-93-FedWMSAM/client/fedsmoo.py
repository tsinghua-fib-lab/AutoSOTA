import torch
from .client import Client
from utils import *
from optimizer import *


class fedsmoo(Client):
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):   
        super(fedsmoo, self).__init__(device, model_func, received_vecs, dataset, lr, args)
        
        # rebuild
        self.base_optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.args.weight_decay+self.args.lamb)
        self.optimizer = DRegSAM(self.model.parameters(), self.base_optimizer, rho=self.args.rho)

        self.dynamic_dual = None

        self.comm_vecs = {
            'local_update_list': None,
            'local_model_param_list': None,
            'local_dynamic_dual': None,
        }
    
    
    def train(self):
        # local training
        self.model.train()

        for k in range(self.args.local_epochs):
            for i, (inputs, labels) in enumerate(self.dataset):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).long()
                
                self.optimizer.paras = [inputs, labels, self.loss, self.model, self.received_vecs['Dynamic_dual_correction']]
                self.received_vecs['Dynamic_dual'] = self.optimizer.step(self.received_vecs['Dynamic_dual'])
                
                param_list = param_to_vector(self.model)
                delta_list = self.received_vecs['Local_dual_correction'].to(self.device)
                loss_correct = self.args.lamb * torch.sum(param_list * delta_list)
                
                loss_correct.backward()
                
                # Clip gradients to prevent exploding
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm)
                self.base_optimizer.step()
                
        last_state_params_list = get_mdl_params(self.model)
        self.comm_vecs['local_update_list'] = last_state_params_list - self.received_vecs['Params_list']
        self.comm_vecs['local_model_param_list'] = last_state_params_list
        self.comm_vecs['local_dynamic_dual'] = self.received_vecs['Dynamic_dual']

        return self.comm_vecs