import time
from flcore.clients.clientgmt import clientGMT
from flcore.servers.serverbase import Server
from tqdm import tqdm
import torch
import os
import copy
from utils.mem_utils import get_gpu_memory_usage
import statistics
class FedGMT(Server):
    def __init__(self, args):
        super().__init__(args)

        self.set_clients(args, clientGMT)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.EMA_model = copy.deepcopy(args.model)

        init_par_list = param_to_vector(self.global_model)
        self.dual_variable_list = torch.zeros((self.num_clients, init_par_list.shape[0])).to(self.device)

        self.alpha = 0.99  # tuned from args.alpha=0.95
        # SWA: start averaging after round swa_start, apply to last 100 rounds
        self.swa_start = 400
        self.swa_model = copy.deepcopy(args.model)
        self.swa_n = 0  # number of models averaged so far

        
    def train(self):


        for epoch in tqdm(range(1, self.global_rounds+1), desc='server-training'):
     
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.current_epoch = epoch
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
               
                self.aggregate_parameters()
                global_para = param_to_vector(self.global_model)

                alpha= self.alpha
                EMA_para = param_to_vector(self.EMA_model)

               
                global_para += ((torch.mean(self.dual_variable_list, dim=0)))
                vector_to_param(global_para,self.global_model)


                EMA_para = EMA_para*alpha + global_para*(1-alpha)
                vector_to_param(EMA_para,self.EMA_model)

            # SWA: accumulate model average in last 100 rounds
            if epoch >= self.swa_start:
                with torch.no_grad():
                    self.swa_n += 1
                    swa_para = param_to_vector(self.swa_model)
                    global_para_cur = param_to_vector(self.global_model)
                    swa_para = swa_para * (self.swa_n - 1) / self.swa_n + global_para_cur / self.swa_n
                    vector_to_param(swa_para, self.swa_model)

            self.empty_cache()
            print(f"\n-------------Round number: {epoch}-------------")
            print("\nEvaluate global model")           
            # loss_after,train_acc = self.sharpness()
            # self.loss_diff.append(loss_before-loss_after)

            # Use SWA model for evaluation when available
            if epoch >= self.swa_start:
                orig_model = self.global_model
                self.global_model = self.swa_model
                self.evaluate()
                self.global_model = orig_model
            else:
                self.evaluate()
            print('-'*25, 'This global round time cost', '-'*25, time.time() - s_t)
            if epoch%self.save_gap == 0:
                self.save_results(epoch)
            if epoch == self.global_rounds:
                print(f"Avg last 50 round acc:{sum(self.test_acc[-50:])/50}, std: {statistics.stdev(self.test_acc[-50:])}")

    def send_models(self):
        # In late training, use SWA model as teacher for more stable guidance
        epoch = getattr(self, 'current_epoch', 0)
        teacher_model = self.swa_model if (epoch >= self.swa_start and self.swa_n > 0) else self.EMA_model
        for client in self.selected_clients:
            client.model = copy.deepcopy(self.global_model)
            client.EMA = copy.deepcopy(teacher_model)
            #To save gpu space, we use the dual_variable on server. Note that dual_variable on client also can be computed locally.
            client.dual_variable = self.dual_variable_list[client.id]

    def empty_cache(self):
        for client in self.selected_clients:
            del client.model, client.EMA, client.local_update
            client.model = None
            client.EMA = None
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

