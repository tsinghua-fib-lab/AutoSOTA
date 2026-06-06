import scipy.stats
import torch

from client import *
from .server import Server
import numpy as np


class FedWMSAM(Server):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):
        super(FedWMSAM, self).__init__(device, model_func, init_model, init_par_list, datasets, method, args)
        print(" Var Reduction Param List  --->  {:d} * {:d}".format(
            self.clients_updated_params_list.shape[0], self.clients_updated_params_list.shape[1]))
        # Initialize communication vectors
        self.comm_vecs = {
            'Params_list': init_par_list.clone().detach(),
            'Client_momentum': torch.zeros((init_par_list.shape[0])),
        }
        self.Client = fedwmsam
        self.local_iteration = self.args.local_epochs * (self.datasets.client_x[0].shape[0] / self.args.batchsize)
        self.value = []
        self.c_i_params_list = torch.zeros((args.total_client, init_par_list.shape[0]))
        self.c_params_list = torch.zeros((init_par_list.shape[0]))
        if args.mode == "dg":
            self.local_iteration = self.args.local_epochs * (
                    len(self.datasets.client_train_set[0]) / self.args.batchsize)
        else:
            self.local_iteration = self.args.local_epochs * (self.datasets.client_x[0].shape[0] / self.args.batchsize)
        self.delta_c = torch.zeros((init_par_list.shape[0]))
        self.momentum = torch.zeros((init_par_list.shape[0]))

    def process_for_communication(self, client, Averaged_update):
        if not self.args.use_RI:
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list)
        else:
            # RI adopts the w(i,t) = w(t) + beta[w(t) - w(i,K,t-1)] as initialization
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list + self.args.beta \
                                                * (self.server_model_params_list - self.clients_params_list[client]))

        local_vr_correction = self.c_params_list - self.c_i_params_list[client]
        # Compute client personalized momentum
        self.comm_vecs['Client_momentum'].copy_(
            self.momentum + local_vr_correction * self.args.alpha / (
                        1 - self.args.alpha)
        )

    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        value = np.mean(self.value)
        self.args.alpha = 0.99 * self.args.alpha + 0.01 * max(0.1, min([value, 0.9]))
        self.value = []
        # Update global model parameters
        # SCAFFOLD (ServerOpt)
        # updated global c
        self.c_params_list += self.delta_c / self.args.total_client
        # zero delta_c for the training on the next communication round
        self.delta_c *= 0.
        self.momentum = Averaged_update / self.local_iteration / self.lr * self.args.lr_decay * -1.
        return self.server_model_params_list + self.args.global_learning_rate * Averaged_update

    def postprocess(self, client, received_vecs):
        local_value = cosine_similarity(self.clients_updated_params_list[client],
                                        self.momentum)
        self.value.append(local_value)
        updated_c_i = self.c_i_params_list[client] - self.c_params_list - \
                      self.clients_updated_params_list[client] / self.local_iteration / self.lr
        self.delta_c += updated_c_i - self.c_i_params_list[client]
        self.c_i_params_list[client] = updated_c_i


def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0

    similarity = dot_product / (norm_a * norm_b)
    return similarity + 1