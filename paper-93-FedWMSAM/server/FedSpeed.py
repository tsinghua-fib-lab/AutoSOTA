import torch
from client import *
from .server import Server


class FedSpeed(Server):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):   
        super(FedSpeed, self).__init__(device, model_func, init_model, init_par_list, datasets, method, args)
        
        self.h_params_list = torch.zeros((args.total_client, init_par_list.shape[0]))
        print(" Dual Variable Param List  --->  {:d} * {:d}".format(
                self.clients_updated_params_list.shape[0], self.clients_updated_params_list.shape[1]))
        
        # rebuild
        self.comm_vecs = {
            'Params_list': init_par_list.clone().detach(),
            'Local_dual_correction': torch.zeros((init_par_list.shape[0])),
        }
        self.Client = fedspeed
    
    
    def process_for_communication(self, client, Averaged_update):
        if not self.args.use_RI:
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list)
        else:
            # RI adopts the w(i,t) = w(t) + beta[w(t) - w(i,K,t-1)] as initialization
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list + self.args.beta\
                                    * (self.server_model_params_list - self.clients_params_list[client]))
        
        # self.comm_vecs['Local_dual_correction'].copy_(self.h_params_list[client] - self.server_model_params_list)
        self.comm_vecs['Local_dual_correction'].copy_(self.h_params_list[client] - self.comm_vecs['Params_list'])

    
    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        # FedSpeed (ServerOpt)
        # w(t+1) = average_s[wi(t)] + average_c[h(t)]
        return Averaged_model + torch.mean(self.h_params_list, dim=0)
    
    
    def postprocess(self, client, received_vecs):
        self.h_params_list[client] += self.clients_updated_params_list[client]