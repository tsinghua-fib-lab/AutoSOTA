import torch

from utils import *
from client import *
from .server import Server



class FedSMOO(Server):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):   
        super(FedSMOO, self).__init__(device, model_func, init_model, init_par_list, datasets, method, args)
        
        self.h_params_list = torch.zeros((args.total_client, init_par_list.shape[0]))
        print("    Dual Variable List     --->  {:d} * {:d}".format(
                self.h_params_list.shape[0], self.h_params_list.shape[1]))
        
        self.mu_params_list = torch.zeros((args.total_client, init_par_list.shape[0]))
        print("   Dyn-Dual Variable List  --->  {:d} * {:d}".format(
                self.mu_params_list.shape[0], self.mu_params_list.shape[1]))
        
        self.global_dynamic_dual = torch.zeros(init_par_list.shape[0])

        # rebuild
        self.comm_vecs = {
            'Params_list': init_par_list.clone().detach(),
            'Local_dual_correction': torch.zeros((init_par_list.shape[0])), # dual variable - global model
            'Dynamic_dual': None,
            'Dynamic_dual_correction': None,
        }
        # self.comm_vecs = {
        #     'Params_list': None,
        #     'Local_dual_correction': None,
        #     'Dynamic_dual': None,
        #     'Dynamic_dual_correction': None,
        # }
        self.Client = fedsmoo
    
    
    def process_for_communication(self, client, Averaged_update):
        if not self.args.use_RI:
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list)
        else:
            # RI adopts the w(i,t) = w(t) + beta[w(t) - w(i,K,t-1)] as initialization
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list + self.args.beta\
                                    * (self.server_model_params_list - self.clients_params_list[client]))
        
        # self.comm_vecs['Local_dual_correction'].copy_(self.h_params_list[client] - self.server_model_params_list)
        self.comm_vecs['Local_dual_correction'].copy_(self.h_params_list[client] - self.comm_vecs['Params_list'])
        self.comm_vecs['Dynamic_dual'] = get_params_list_with_shape(self.server_model, self.mu_params_list[client], self.device)
        self.comm_vecs['Dynamic_dual_correction'] = get_params_list_with_shape(self.server_model, self.global_dynamic_dual, self.device)

    
    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        # FedSMOO (ServerOpt)

        ### in this version we simplify the solution of global_dynamic_dual as
        ### ---> s / || average_s(mu_i) || * rho by ignoring the \hat{s}_{i,K} term (its norm is small)
        Averaged_dynamic_dual = torch.mean(self.mu_params_list[selected_clients], dim=0)
        _l2_ = torch.norm(Averaged_dynamic_dual, p=2, dim=0) + 1e-7
        self.global_dynamic_dual = Averaged_dynamic_dual / _l2_ * self.args.rho
        
        # w(t+1) = average_s[wi(t)] + average_c[h(t)]
        return Averaged_model + torch.mean(self.h_params_list, dim=0)
    
    
    def postprocess(self, client, received_vecs):
        self.h_params_list[client] += self.clients_updated_params_list[client]
        mu = []
        for _mu_ in received_vecs['local_dynamic_dual']:
            mu.append(_mu_.clone().detach().cpu().reshape(-1))
        self.mu_params_list[client] = torch.cat(mu)