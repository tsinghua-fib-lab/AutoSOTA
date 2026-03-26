import torch
from client import *
from .server import Server


class MoFedSAM(Server):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):   
        super(MoFedSAM, self).__init__(device, model_func, init_model, init_par_list, datasets, method, args)
        # rebuild
        self.comm_vecs = {
            'Params_list': init_par_list.clone().detach(),
            'Client_momentum': torch.zeros((init_par_list.shape[0])),
        }
        self.Client = mofedsam
        self.local_iteration = self.args.local_epochs * (self.datasets.client_x[0].shape[0] / self.args.batchsize)
        
    
    def process_for_communication(self, client, Averaged_update):
        if not self.args.use_RI:
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list)
        else:
            # RI adopts the w(i,t) = w(t) + beta[w(t) - w(i,K,t-1)] as initialization
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list + self.args.beta\
                                    * (self.server_model_params_list - self.clients_params_list[client]))
        
        # last lr = current lr/lr_decay
        self.comm_vecs['Client_momentum'].copy_(Averaged_update / self.local_iteration / self.lr * self.args.lr_decay * -1.)
        
    
    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        # MoFedSAM (ServerOpt)
        # w(t+1) = w(t) + eta_g * Delta
        return self.server_model_params_list + self.args.global_learning_rate * Averaged_update