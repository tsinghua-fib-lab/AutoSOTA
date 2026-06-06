import torch
import os
import numpy as np
import csv
import copy
from utils.data_utils import read_data,read_client_json,read_client_data
from torch.utils.data import DataLoader

from utils.mem_utils import get_gpu_memory_usage

class Server(object):
    def __init__(self, args):
        # Set up the main attributes
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.lr_decay = args.lr_decay
        self.global_model = args.model
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.algorithm = args.algorithm

        self.rho = args.rho
        self.tau = args.tau
        self.alpha = args.alpha
        self.gama = args.gama
        self.beta = args.beta




        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.test_acc = []
        self.personal_acc = []
 
        self.current_acc = 0

        self.save_gap = args.save_gap

        self.data_test = read_data(self.dataset,idx=None,is_train=False)
        self.test_loader = DataLoader(self.data_test, 1000,pin_memory=True,num_workers=4)
        self.loss = torch.nn.CrossEntropyLoss()

        self.loss_diff = []





    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_samples = read_client_json(self.dataset, i)[0]
            client = clientObj(args, 
                            id=i, 
                            train_samples=train_samples)
            self.clients.append(client)


    def select_clients(self):
        selected_clients = list(np.random.choice(self.clients, self.num_join_clients, replace=False))
        
        return selected_clients

    def send_models(self,selected_clients,model):
        assert (len(self.clients) > 0)

        for client in selected_clients:
            client.model = copy.deepcopy(model)

            


    def receive_models(self):
        assert (len(self.selected_clients) > 0)
        receive_clients = self.selected_clients
        self.uploaded_ids = []
        self.uploaded_weights = []#num of samples
        self.uploaded_models = []
        tot_samples = 0
        for client in receive_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model.state_dict())
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        fedavg_global_params = self.global_model.state_dict()
        for name_param in self.uploaded_models[0]:
            list_values_param = []
            for dict_local_params, local_weight in zip(self.uploaded_models, self.uploaded_weights):
                list_values_param.append(dict_local_params[name_param] * local_weight)
            value_global_param = sum(list_values_param)
            fedavg_global_params[name_param] = value_global_param
        self.global_model.load_state_dict(fedavg_global_params)


        
    def save_results(self,r,save_model = False):
        algo = self.algorithm
        result_path = "../results/"+ self.dataset+"/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.test_acc)):
            algo = algo + "_test"
            file_path = result_path+ "{}-rho{}-alpha{}-gama{}-tau{}.csv".format(algo,self.rho,self.alpha,self.gama,self.tau)
            print("File path: " + file_path)
            my_list_2d = [[x] for x in self.test_acc]
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(my_list_2d)  

        if (len(self.loss_diff)):
            algo = algo + "_loss_diff"
            file_path = result_path + "{}.csv".format(algo)
            print("File path: " + file_path)
            my_list_2d = [[x] for x in self.loss_diff]
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(my_list_2d)

        
        if save_model:
            model_path = os.path.join(result_path,"model")
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_path = os.path.join(model_path, self.algorithm + "_server" +str(r)+ ".pt")
            torch.save(self.global_model.state_dict(), model_path)

    def save_model(self):
        result_path = "../results/"+ self.dataset+"/model/"
        result_path_per = "../results/"+ self.dataset+"/model/"+self.algorithm+"-per/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if not os.path.exists(result_path_per):
            os.makedirs(result_path_per)

        torch.save(self.global_model.state_dict(),result_path+self.algorithm+"best.pt")
        # for client in self.clients:
        #     torch.save(client.model.state_dict(),result_path_per+str(client.id)+".pt")



    # evaluate 
    def evaluate(self): 
 
        model = self.global_model
        model.eval()
        with torch.no_grad():
            num_corrects = {i: 0 for i in range(self.num_classes)}
            total = {i: 0 for i in range(self.num_classes)}
            total_corrects = 0
            total_samples = 0
            total_loss = 0
            for step,data_batch in enumerate(self.test_loader):
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = self.loss(outputs,labels)
                total_loss+=loss.item()*labels.size(0)

                _, predicts = torch.max(outputs, -1)
                for i in range(len(labels)):
                    total[labels[i].item()] += 1
                    total_samples += 1
                    if predicts[i] == labels[i]:
                        num_corrects[labels[i].item()] += 1
                        total_corrects += 1

            test_loss = total_loss/len(self.test_loader.dataset)
            print(f"test loss: {test_loss}")

            total_accuracy = total_corrects / total_samples
            self.test_acc.append(total_accuracy)
            print(f"Global acc:{self.test_acc}")
        return test_loss


    def _lr_scheduler_(self):
        self.learning_rate *= self.lr_decay


    def sharpness(self):
        model = self.global_model
        model.eval()
        with torch.no_grad():
            train_data = []
            for client in self.selected_clients:
                train_data += read_client_data(self.dataset, client.id)
            train_loader = DataLoader(train_data, 500, shuffle=True,pin_memory=True)
            total_loss = 0
            correct = 0.0
            for step,data_batch in enumerate(train_loader):
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                _, outputs = model(images)
                loss = self.loss(outputs,labels)
                total_loss+=loss.item()*labels.size(0)
                pred = outputs.data.argmax(1, keepdim=True)
                correct += pred.eq(labels.data.view_as(pred)).sum().item()
            return total_loss/len(train_loader.dataset),correct/len(train_loader.dataset)



    def empty_cache(self):
        for client in self.selected_clients:
            del client.model
            client.model = None
        
        allocated, reserved = get_gpu_memory_usage(self.device)
        print(f"allocated GPU space: {allocated:.2f} MB，reserved GPU space: {reserved:.2f} MB")
