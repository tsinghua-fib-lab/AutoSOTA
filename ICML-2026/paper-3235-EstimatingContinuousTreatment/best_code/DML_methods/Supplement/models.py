# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 23:04:09 2020
Last update Monday Jan 27 12:11 pm 2025

This file defines the neural network models used for both the simulations and
empirical application in Colangelo and Lee (2025). PyTorch is the package used
for all neural networks. We begin by defining a class "NeuralNet" which creates
a torch neural network that adheres to the requirements of the DDMLCT class.
That is, DDMLCT requires models which have both a .fit and .predict method,
so NeuralNet creates this for us. We also define class NeuralNetk as the base
class for the K neural network which utilizes the additional kernel function
in the loss function as specified in Colangelo and Lee (2025).
We then define a subclass of NeuralNet and NeuralNetk for each of the models used in the
simulations and empirical application. To define a new neural network model
which can be used with the DDMLCT class, you need only define a subclass of
NeuralNet or NeuralNetk and define the layers as has been done below, and
specify the criterion, optimizer, and number of epochs.

"""

import torch
import torch.nn as nn

torch.set_default_tensor_type("torch.DoubleTensor")


class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(20, 1))
        self.layer_1.add_module("R1", nn.SELU())

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        self.epochs = 100

    def weight_reset(self, m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()

    def forward(self, x):
        for i in range(1, len(vars(self)["_modules"])):
            x = getattr(self, ("layer_" + str(i)))(x)

        return x

    def fit(self, x, y):
        self.apply(self.weight_reset)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y).reshape(len(y), 1)

        for t in range(self.epochs):
            y_pred = self(x)
            loss = self.criterion(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return self

    def predict(self, x):
        x = torch.from_numpy(x)
        pred = self(x).detach().numpy()
        return pred[:, 0]


def my_loss(output, target, K):
    loss = torch.mean(((1 / 2) * (output - target) ** 2) * K)
    return loss


class NeuralNetk(torch.nn.Module):
    def __init__(self):
        super(NeuralNetk, self).__init__()

        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(20, 1))
        self.layer_1.add_module("R1", nn.SELU())

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        self.epochs = 300

    def weight_reset(self, m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()

    def forward(self, x):
        for i in range(1, len(vars(self)["_modules"])):
            x = getattr(self, ("layer_" + str(i)))(x)

        return x

    def fit(self, x, y, K):
        self.apply(self.weight_reset)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y).reshape(len(y), 1)
        K = torch.from_numpy(K)
        K2 = torch.sqrt(K).reshape(len(y), 1)
        K2y = torch.mul(y, K2)
        for t in range(self.epochs):
            y_pred = self(x)
            loss = self.criterion(torch.mul(y_pred, K2), K2y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return self

    def predict(self, x):
        x = torch.from_numpy(x)
        pred = self(x).detach().numpy()
        return pred[:, 0]


class NeuralNet1_n10000(NeuralNet):
    def __init__(self, k, lr=0.01, momentum=0.5, weight_decay=0.2, epochs=300):
        super(NeuralNet1_n10000, self).__init__()

        self.k = k

        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(self.k, self.k))
        self.layer_1.add_module("R1", nn.ReLU())

        self.layer_2 = torch.nn.Linear(self.k, 1)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        self.epochs = epochs


class NeuralNet1_n1000(NeuralNet):
    def __init__(self, k, lr=0.01, momentum=0.5, weight_decay=0.2, epochs=300):
        super(NeuralNet1_n1000, self).__init__()

        self.k = k

        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(self.k, 10))
        self.layer_1.add_module("R1", nn.ReLU())

        self.layer_2 = torch.nn.Linear(10, 1)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        self.epochs = epochs


class NeuralNet1_emp_app(NeuralNet):
    def __init__(self, k, lr=0.01, momentum=0.5, weight_decay=0.2, epochs=300):
        super(NeuralNet1_emp_app, self).__init__()

        self.k = k

        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(self.k, 25))
        self.layer_1.add_module("R1", nn.ReLU())

        self.layer_2 = torch.nn.Linear(25, 1)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        self.epochs = epochs


class NeuralNet1k_n1000(NeuralNetk):
    def __init__(self, k, lr=0.01, momentum=0.5, weight_decay=0.2, epochs=100):
        super(NeuralNet1k_n1000, self).__init__()

        self.k = k

        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(self.k, 10))
        self.layer_1.add_module("R1", nn.ReLU())

        self.layer_2 = torch.nn.Linear(10, 1)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        self.epochs = epochs


class NeuralNet1k_n10000(NeuralNetk):
    def __init__(self, k, lr=0.01, momentum=0.5, weight_decay=0.2, epochs=100):
        super(NeuralNet1k_n10000, self).__init__()

        self.k = k

        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(self.k, 100))
        self.layer_1.add_module("R1", nn.ReLU())

        self.layer_2 = torch.nn.Linear(100, 1)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        self.epochs = epochs


class NeuralNet1k_emp_app(NeuralNetk):
    def __init__(self, k, lr=0.01, momentum=0.5, weight_decay=0.2, epochs=100):
        super(NeuralNet1k_emp_app, self).__init__()

        self.k = k

        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(self.k, 25))
        self.layer_1.add_module("R1", nn.ReLU())

        self.layer_2 = torch.nn.Linear(25, 1)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        self.epochs = epochs


class NeuralNet2_n10000(NeuralNet):
    def __init__(self, k, lr=0.01, momentum=0.5, weight_decay=0.2, epochs=300):
        super(NeuralNet2_n10000, self).__init__()

        self.k = k

        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(self.k, 100))
        self.layer_1.add_module("R1", nn.ReLU())

        self.layer_2 = torch.nn.Linear(100, 1)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        self.epochs = epochs


class NeuralNet2_n1000(NeuralNet):
    def __init__(self, k, lr=0.01, momentum=0.5, weight_decay=0.2, epochs=300):
        super(NeuralNet2_n1000, self).__init__()

        self.k = k

        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(self.k, 10))
        self.layer_1.add_module("R1", nn.ReLU())

        self.layer_2 = torch.nn.Linear(10, 1)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        self.epochs = epochs


class NeuralNet2_emp_app(NeuralNet):
    def __init__(self, k, lr=0.01, momentum=0.5, weight_decay=0.2, epochs=300):
        super(NeuralNet2_emp_app, self).__init__()

        self.k = k

        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(self.k, 25))
        self.layer_1.add_module("R1", nn.ReLU())

        self.layer_2 = torch.nn.Linear(25, 1)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        self.epochs = epochs
