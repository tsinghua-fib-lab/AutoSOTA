import torch
import torch.nn as nn
import torch.nn.functional as F

class NTKMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, sigma_w = 1.0, sigma_b = 1.0):
        super(NTKMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=True)
        self.relu = nn.ReLU()
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize the weights of the network according 
        to the NTK parametrization
        """
        nn.init.normal_(self.fc1.weight, mean=0, std=self.sigma_w / (self.fc1.in_features ** 0.5))
        nn.init.normal_(self.fc1.bias, mean=0, std=self.sigma_b)
        nn.init.normal_(self.fc2.weight, mean=0, std=self.sigma_w / (self.fc2.in_features ** 0.5))
        nn.init.normal_(self.fc2.bias, mean=0, std=self.sigma_b)


    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

