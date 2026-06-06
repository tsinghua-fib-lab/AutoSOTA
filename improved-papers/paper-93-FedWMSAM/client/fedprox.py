from utils import *
from .client import Client


class fedprox(Client):
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):   
        super(fedprox, self).__init__(device, model_func, received_vecs, dataset, lr, args)
        
        # rebuild
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.args.weight_decay+self.args.lamb)
    
    def train(self):
        # local training
        self.model.train()
        
        for k in range(self.args.local_epochs):
            for i, (inputs, labels) in enumerate(self.dataset):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).long()
                
                predictions = self.model(inputs)
                loss_pred = self.loss(predictions, labels)
                
                param_list = param_to_vector(self.model)
                delta_list = self.received_vecs['Params_list'].to(self.device)
                loss_correct = torch.sum(param_list * delta_list) * -1.
                
                loss = loss_pred + self.args.lamb * loss_correct
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients to prevent exploding
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm) 
                self.optimizer.step()
                
        last_state_params_list = get_mdl_params(self.model)
        self.comm_vecs['local_update_list'] = last_state_params_list - self.received_vecs['Params_list']
        self.comm_vecs['local_model_param_list'] = last_state_params_list

        return self.comm_vecs