# @Time : 2024/9/27 8:20 
# @Author : ltl
# @File : FedLESAM.py
import torch
from client import *
from .server import Server


class FedLESAM(Server):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):
        super(FedLESAM, self).__init__(device, model_func, init_model, init_par_list, datasets, method, args)
        # rebuild
        self.comm_vecs = {
            'Params_list': init_par_list.clone().detach(),
            'global_update': None
        }
        self.Client = fedlesam

    def process_for_communication(self, client, Averaged_update):
        if not self.args.use_RI:
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list)
        else:
            # RI adopts the w(i,t) = w(t) + beta[w(t) - w(i,K,t-1)] as initialization
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list + self.args.beta \
                                                * (self.server_model_params_list - self.clients_params_list[client]))

    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        # FedSAM (ServerOpt)
        # w(t+1) = w(t) + eta_g * Delta
        self.comm_vecs['global_update'] = (self.args.global_learning_rate * Averaged_update).clone().detach()
        return self.server_model_params_list + self.args.global_learning_rate * Averaged_update