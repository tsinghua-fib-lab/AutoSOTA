from torch import nn

import torch
import torch.nn.functional as F
import torch.utils.data as data_utils

from utils.EIPM import compute_EIPM

# NN ---------------------------------------------
# This network is passed a penalty function, for example 
# the Hirschfeld-Gebelein-Rényi coefficient, or rather its approximation from utils/hgr.py 
# to penalize in favour of fairness. Thus this can incooperate a variety of different approaches 
# but is used in the paper for the ''NN-HGR''.

class NeuralWrapper():
    def __init__(self, network_cls, network_params,num_epochs, penalty, batch_size=200):
        self.network = network_cls(**network_params)
        self.penalty = penalty
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def predict(self, X):
        return self.network(X)

    def train(self, X, y, p, lr=1e-3, lbd_reg = 0):
        # No regularization of protected = None
        X.to('cuda')
        y.to('cuda')
        p.to('cuda')
        dataset = data_utils.TensorDataset(X, y, p)
        dataset_loader = data_utils.DataLoader(dataset=dataset, batch_size=200, shuffle=True)

        # mse regression objective
        data_fitting_loss = nn.MSELoss()

        # stochastic optimizer
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=0.01)

        for j in range(self.num_epochs):
            if j % 25 == 0:
                print("Epoch ", str(j))
            for i, (x, y, z) in enumerate(dataset_loader):
                def closure():
                    optimizer.zero_grad()
                    outputs = self.network(x).flatten()
                    loss = data_fitting_loss(outputs, y)
                    if self.penalty is not None:
                        loss += lbd_reg*self.penalty(outputs, z, y)
                    loss.backward()
                    return loss

                optimizer.step(closure)


class ExampleNeuralNet(nn.Module):
    def __init__(self, input_size, output_size, size = 100):
        super(ExampleNeuralNet, self).__init__()
        self.first = nn.Linear(input_size, size)
        self.middle = nn.Linear(size, size-20)
        self.last = nn.Linear(size-20, output_size)

    def forward(self, x):
        out = self.hidden(x)
        out = self.last(out)
        return out
    
    def hidden(self, x):
        out = F.selu(self.first(x))
        out = F.selu(self.middle(out))
        return out

    def set_readout(self, W, b):
        # Unused for now
        self.last.weight = nn.Parameter(W)
        self.last.bias = nn.Parameter(b)
    


##########################################
# Implementation based on description in https://ieeexplore.ieee.org/document/10874180

class ExampleNN_FREM(nn.Module):
    def __init__(self, input_size, output_size):
        super(ExampleNN_FREM, self).__init__()
        size = 100
        self.first = nn.Linear(input_size, size)
        self.middle = nn.Linear(size, size-20)
        self.last = nn.Linear(size-20, output_size)


    def hidden(self, x):
        out = F.selu(self.first(x))
        out = F.selu(self.middle(out))
        return out
    
    def last_layer(self, x):
        return self.last(x)

    def forward(self,x):
        out = self.hidden(x)
        return self.last_layer(out)
    
class FREM:
    def __init__(self, network_cls, network_params,num_epochs,batch_size=200):
        self.network = network_cls(**network_params)
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def predict(self, X):
        return self.network(X)

    def train(self, X, y, p, lr=1e-3, lbd_reg = 0, ):
        # No regularization of protected = None

        gamma = 0.05 #1 / X.shape[0] # 0.05 is given in their paper 

        X.to('cuda')
        y.to('cuda')
        p.to('cuda')

        dataset = data_utils.TensorDataset(X, y, p)
        dataset_loader = data_utils.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        # mse regression objective
        data_fitting_loss = nn.MSELoss()

        # stochastic optimizer
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=0.01)

        for j in range(self.num_epochs):
            #if j % 1 == 0:
            print("Epoch ", str(j))
            for i, (x, y, z) in enumerate(dataset_loader):
               
                optimizer.zero_grad()
                hidden_outputs = self.network.hidden(x)

                reg_loss = lbd_reg * compute_EIPM(hidden_outputs, z, sigma = 1, gamma = gamma)

                outputs = self.network.last_layer(hidden_outputs).flatten()

                loss = data_fitting_loss(outputs, y)

                loss += reg_loss

                loss.backward()
                    
                optimizer.step()