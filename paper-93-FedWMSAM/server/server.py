import os
import time
import numpy as np

import torch
from utils import *
from dataset import Dataset
from torch.utils import data

from utils import *


class Server(object):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):
        # super(Server, self).__init__()
        self.args = args
        self.device = device
        self.datasets = datasets
        self.model_func = model_func
        
        self.server_model = init_model
        self.server_model_params_list = init_par_list 
        
        print("Initialize the Server      --->  {:s}".format(self.args.method))
        ### Generate Storage
        print("Initialize the Public Storage:")
        self.clients_params_list = init_par_list.repeat(args.total_client, 1)
        print("   Local Model Param List  --->  {:d} * {:d}".format(
                self.clients_params_list.shape[0], self.clients_params_list.shape[1]))
        
        self.clients_updated_params_list = torch.zeros((args.total_client, init_par_list.shape[0]))
        # self.clients_updated_params_list = np.expand_dims(init_par_list, axis=0).repeat(args.total_client, axis=0)
        print(" Local Updated Param List  --->  {:d} * {:d}".format(
                self.clients_updated_params_list.shape[0], self.clients_updated_params_list.shape[1]))

        ### Generate Log Storage : [[loss, acc]...] * T
        self.train_perf = np.zeros((self.args.comm_rounds, 2))
        self.test_perf = np.zeros((self.args.comm_rounds, 2))
        print("   Train/Test [loss, acc]  --->  {:d} * {:d}".format(self.train_perf.shape[0], self.train_perf.shape[1]))
        ### Generate Log Storage : [[E||wi - w||]...] * T
        self.divergence = np.zeros((args.comm_rounds))
        print("  Consistency (Divergence) --->  {:d}".format(self.divergence.shape[0]))
              
        self.time = np.zeros((args.comm_rounds))
        self.lr = self.args.local_learning_rate
        
        # transfer vectors (must be defined if use)
        self.comm_vecs = {
            'Params_list': None,
        }
        self.received_vecs = None
        self.Client = None
        
    
    def _see_the_watch_(self):
        # see time
        self.time.append(datetime.datetime.now())
        
    
    def _see_the_divergence_(self, selected_clients, t):
        # calculate the divergence
        self.divergence[t] = torch.norm(self.clients_params_list[selected_clients] -\
                                            self.server_model_params_list) ** 2 / len(selected_clients)
        
    
    def _activate_clients_(self, t):
        # select active clients ID
        inc_seed = 0
        while(True):
            np.random.seed(t + self.args.seed + inc_seed)
            act_list = np.random.uniform(size=self.args.total_client)
            act_clients = act_list <= self.args.active_ratio
            selected_clients = np.sort(np.where(act_clients)[0])
            inc_seed += 1
            if len(selected_clients) != 0:
                return selected_clients
            
            
    def _lr_scheduler_(self):
        self.lr *= self.args.lr_decay
        
        
    def _test_(self, t, selected_clients):
        # test
        # validation on train set
        loss, acc = self._validate_((np.concatenate(self.datasets.client_x, axis=0), np.concatenate(self.datasets.client_y, axis=0)))
        self.train_perf[t] = [loss, acc]
        print("   Train    ----    Loss: {:.4f},   Accuracy: {:.4f}".format(self.train_perf[t][0], self.train_perf[t][1]), flush = True)
        # validation on test set
        loss, acc = self._validate_((self.datasets.test_x, self.datasets.test_y))
        self.test_perf[t] = [loss, acc]
        print("    Test    ----    Loss: {:.4f},   Accuracy: {:.4f}".format(self.test_perf[t][0], self.test_perf[t][1]), flush = True)
        # calculate consistency
        self._see_the_divergence_(selected_clients, t)
        print("            ----    Divergence: {:.4f}".format(self.divergence[t]), flush = True)
        
    
    def _summary_(self):
        # print results summary
        print("##=============================================##")
        print("##                   Summary                   ##")
        print("##=============================================##")
        print("     Communication round   --->   T = {:d}       ".format(self.args.comm_rounds))
        print("    Average Time / round   --->   {:.2f}s        ".format(np.mean(self.time)))
        print("     Top-1 Test Acc (T)    --->   {:.2f}% ({:d}) ".format(np.max(self.test_perf[:,1])*100., np.argmax(self.test_perf[:,1])))
    
    
    def _validate_(self, dataset):
        self.server_model.eval()
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        testdataset = data.DataLoader(Dataset(dataset[0], dataset[1], train=False, dataset_name=self.args.dataset), batch_size=1000, shuffle=False)
        
        total_loss = 0
        total_acc = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(testdataset):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).long()
                
                predictions = self.server_model(inputs)
                loss = self.loss(predictions, labels)
                total_loss += loss.item()
                
                predictions = predictions.cpu().numpy()            
                predictions = np.argmax(predictions, axis=1).reshape(-1)
                labels = labels.cpu().numpy().reshape(-1).astype(np.int32)
                batch_correct = np.sum(predictions == labels)
                total_acc += batch_correct
        
        if self.args.weight_decay != 0.:
            # Add L2 loss
            total_loss += self.args.weight_decay / 2. * torch.sum(self.server_model_params_list * self.server_model_params_list)

        return total_loss/(i+1), total_acc/dataset[0].shape[0]
    
    
    def _save_results_(self):
        # save results.npy
        options = '' # write your own saving configs

        root = '{:s}/T={:d}'.format(self.args.out_file, self.args.comm_rounds)
        if not os.path.exists(root):
            os.makedirs(root)
        if not self.args.non_iid:
            root += '/{:s}-{:s}{:s}-{:d}'.format(self.args.dataset, 'IID', 
                                                 '', self.args.total_client)
        else:
            root += '/{:s}-{:s}{:s}-{:d}'.format(self.args.dataset, self.args.split_rule, 
                                                 str(self.args.split_coef), self.args.total_client)
        if not os.path.exists(root):
            os.makedirs(root)
        
        participation = str(self.args.active_ratio)
        root = root + '/active-' + participation
        
        if not os.path.exists(root):
            os.makedirs(root)
        
        # save [loss, acc] results
        perf_dir = root + '/Performance'
        if not os.path.exists(perf_dir):
            os.makedirs(perf_dir)
        train_file = perf_dir + '/trn-{:s}{:s}.npy'.format(self.args.method, options)
        test_file = perf_dir + '/tst-{:s}{:s}.npy'.format(self.args.method, options)
        np.save(train_file, self.train_perf)
        np.save(test_file, self.test_perf)
        
        # save [divergence, consistency] results
        divergence_dir = root + '/Divergence'
        if not os.path.exists(divergence_dir):
            os.makedirs(divergence_dir)
        divergence_file = divergence_dir + '/divergence-{:s}{:s}.npy'.format(self.args.method, options)
        np.save(divergence_file, self.divergence)
                
    
    def process_for_communication(self):
        pass
        
    
    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        pass
    
    
    def postprocess(self, client, received_vecs):
        pass
        
    
    def train(self):
        print("##=============================================##")
        print("##           Training Process Starts           ##")
        print("##=============================================##")
        
        Averaged_update = torch.zeros(self.server_model_params_list.shape)
        
        for t in range(self.args.comm_rounds):
            start = time.time()
            # select active clients list
            selected_clients = self._activate_clients_(t)
            print('============= Communication Round', t + 1, '=============', flush = True)
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clients])))
            
            for client in selected_clients:
                dataset = (self.datasets.client_x[client], self.datasets.client_y[client])
                self.process_for_communication(client, Averaged_update)
                _edge_device = self.Client(device=self.device, model_func=self.model_func, received_vecs=self.comm_vecs,
                                          dataset=dataset, lr=self.lr, args=self.args)
                self.received_vecs = _edge_device.train()
                self.clients_updated_params_list[client] = self.received_vecs['local_update_list']
                self.clients_params_list[client] = self.received_vecs['local_model_param_list']
                self.postprocess(client, self.received_vecs)
                
                # release the salloc
                del _edge_device
            
            # calculate averaged model
            Averaged_update = torch.mean(self.clients_updated_params_list[selected_clients], dim=0)
            Averaged_model  = torch.mean(self.clients_params_list[selected_clients], dim=0)
            
            self.server_model_params_list = self.global_update(selected_clients, Averaged_update, Averaged_model)
            set_client_from_params(self.device, self.server_model, self.server_model_params_list)
            
            
            self._test_(t, selected_clients)
            self._lr_scheduler_()
            
            # time
            end = time.time()
            self.time[t] = end-start
            print("            ----    Time: {:.2f}s".format(self.time[t]), flush = True)
    
            
        
        self._save_results_()
        self._summary_()