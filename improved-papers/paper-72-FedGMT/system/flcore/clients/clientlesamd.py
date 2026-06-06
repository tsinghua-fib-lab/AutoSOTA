import time
from flcore.clients.clientbase import Client
from torchvision.transforms import transforms
import copy
import torch.nn.functional as F
import torch
import torch.nn as nn
from utils.lesam_utils import LESAM

class clientLESAMD(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)
        self.dual_variable = None
        self.beta= 1/args.beta
        self.rho = args.rho
        self.global_update = None

    def train(self):
        base_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5,momentum=self.momentum)
        optimizer = LESAM(self.model.parameters(), base_optimizer, rho=self.rho)

        trainloader = self.load_train_data()
        self.model.train()
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])

        with torch.no_grad():
            regular_params = param_to_vector(self.model).detach()
            
        start_time = time.time()

        for step in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                if self.dataset != "agnews":
                    x = transform_train(x)

                optimizer.first_step(self.global_update)
                output = self.model(x)
                loss = self.loss(output, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.second_step()
   
                #dyn
                local_params = param_to_vector(self.model)
                loss = self.beta/2 * torch.norm(local_params - regular_params, 2)
                loss -= torch.dot(local_params, -(self.beta)*self.dual_variable )

                loss.backward()
                base_optimizer.step()   

        print(f"client{self.id} train_samples:{self.train_samples} train cost time :{time.time() - start_time}")

        # DYN
        with torch.no_grad():
            local_params = param_to_vector(self.model).detach()
            self.local_update = (local_params-regular_params)


def param_to_vector(model):
    # model parameters ---> vector (same storage)
    
    vec = []
    for param in model.parameters():
        vec.append((param.reshape(-1)))
    return torch.cat(vec)



