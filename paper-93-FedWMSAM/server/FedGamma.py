import torch
from client import *
from .server import Server


class FedGamma(Server):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):   
        super(FedGamma, self).__init__(device, model_func, init_model, init_par_list, datasets, method, args)
        
        # initialize c_i and c in SCAFFOLD
        self.c_i_params_list = torch.zeros((args.total_client, init_par_list.shape[0]))
        print(" Var Reduction Param List  --->  {:d} * {:d}".format(
                self.clients_updated_params_list.shape[0], self.clients_updated_params_list.shape[1]))
        self.c_params_list = torch.zeros((init_par_list.shape[0]))
        
        # rebuild
        self.comm_vecs = {
            'Params_list': init_par_list.clone().detach(),
            'Local_VR_correction': torch.zeros((init_par_list.shape[0])),
        }
        self.Client = fedgamma
        self.local_iteration = self.args.local_epochs * (self.datasets.client_x[0].shape[0] / self.args.batchsize)
        self.delta_c = torch.zeros((init_par_list.shape[0]))
        
    
    def process_for_communication(self, client, Averaged_update):
        if not self.args.use_RI:
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list)
        else:
            # RI adopts the w(i,t) = w(t) + beta[w(t) - w(i,K,t-1)] as initialization
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list + self.args.beta *\
                                    (self.server_model_params_list - self.clients_params_list[client]))
        
        # combination of c_i and c
        # local gradient is g - c_i + c
        # Therefore, c - c_i is communicated as Local_VR_correction term
        self.comm_vecs['Local_VR_correction'].copy_(self.c_params_list - self.c_i_params_list[client])

    
    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        # SCAFFOLD (ServerOpt)
        # updated global c
        self.c_params_list += self.delta_c / self.args.total_client
        # zero delta_c for the training on the next communication round
        self.delta_c *= 0.
        # w(t+1) = w(t) + eta_g * Delta
        return self.server_model_params_list + self.args.global_learning_rate * Averaged_update
    
    
    def postprocess(self, client, received_vecs):
        updated_c_i = self.c_i_params_list[client] - self.c_params_list -\
                      self.clients_updated_params_list[client] / self.local_iteration / self.lr
        self.delta_c += updated_c_i - self.c_i_params_list[client]
        self.c_i_params_list[client] = updated_c_i