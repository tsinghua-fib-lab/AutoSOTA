import time
from flcore.clients.clientlesamd import clientLESAMD
from flcore.servers.serverbase import Server
from tqdm import tqdm
import torch
import os
import copy
from utils.mem_utils import get_gpu_memory_usage
import statistics
class FedLESAMD(Server):
    def __init__(self, args):
        super().__init__(args)

        self.set_clients(args, clientLESAMD)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        init_par_list = param_to_vector(self.global_model)
        self.dual_variable_list = torch.zeros((self.num_clients, init_par_list.shape[0])).to(self.device)
        self.global_update = torch.zeros_like((param_to_vector(args.model)))
    def train(self):
        for epoch in tqdm(range(1, self.global_rounds+1), desc='server-training'):
     
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            # loss_before,_ = self.sharpness()
            print()

            for client in self.selected_clients:
                client.learning_rate = self.learning_rate
                client.train()
                self.dual_variable_list[client.id] += client.local_update

            self._lr_scheduler_()
            self.receive_models()
            with torch.no_grad():
                old_para = param_to_vector(self.global_model)


                self.aggregate_parameters()
                global_para = param_to_vector(self.global_model)

                global_para += ((torch.mean(self.dual_variable_list, dim=0)))
                vector_to_param(global_para,self.global_model)

                self.global_update = get_params_list_with_shape(self.global_model,(old_para-global_para))


            self.empty_cache()
            print(f"\n-------------Round number: {epoch}-------------")
            print("\nEvaluate global model")           
            # loss_after,train_acc = self.sharpness()
            # self.loss_diff.append(loss_before-loss_after)

           
            self.evaluate()
            print('-'*25, 'This global round time cost', '-'*25, time.time() - s_t)
            if epoch%self.save_gap == 0:
                self.save_results(epoch)
            if epoch == self.global_rounds:
                print(f"Avg last 50 round acc:{sum(self.test_acc[-50:])/50}, std: {statistics.stdev(self.test_acc[-50:])}")

    def send_models(self):
        for client in self.selected_clients:
            client.model = copy.deepcopy(self.global_model)
            client.global_update = self.global_update
            #To save gpu space, we use the dual_variable on server. Note that dual_variable on client also can be computed locally.
            client.dual_variable = self.dual_variable_list[client.id]

    def empty_cache(self):
        for client in self.selected_clients:
            del client.model,  client.local_update
            client.model = None
            client.local_update = None
      
        allocated, reserved = get_gpu_memory_usage(self.device)
        print(f"allocated GPU space: {allocated:.2f} MB，reserved GPU space: {reserved:.2f} MB")


def param_to_vector(model):
    # model parameters ---> vector (same storage)
    
    vec = []
    for param in model.parameters():
        vec.append((param.reshape(-1).detach()))
    return torch.cat(vec)


def vector_to_param(vector, model):
    # vector ---> model parameters
    vector = vector.detach().clone()
    index = 0
    for param in model.parameters():
        param_size = param.numel()
        
        param.data = vector[index:index + param_size].view(param.shape)
        
        index += param_size

def get_params_list_with_shape(model, param_list):
    vec_with_shape = []
    idx = 0
    for param in model.parameters():
        length = param.numel()
        vec_with_shape.append(param_list[idx:idx + length].reshape(param.shape))
    return vec_with_shape