import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('.')
from benchmark.dqs.torch.distribution import DistributionLinear
from benchmark.dqs.torch.loss import NegativeLogLikelihood


class MLP(nn.Module):
    def __init__(self, input_len, n_output, n_hidden_layers=6, hidden_units=128, 
                 dropout_rate=0.5, use_batch_norm=False):
        super(MLP, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        layers = []
        in_features = input_len
        
        # Create hidden layers
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(in_features, hidden_units))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_units
        
        # Sequential hidden part
        self.hidden = nn.Sequential(*layers)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_units, n_output)

    def forward(self, x):
        x = self.hidden(x)
        x = self.fc_out(x)
        return torch.softmax(x, dim=1)
    

if __name__ == '__main__':
    # prepare data
    x = np.random.rand(1000,3)
    y = np.random.rand(1000)
    e = (np.random.rand(1000) > 0.5)
    data_x = torch.from_numpy(x.astype(np.float32)).clone()
    data_y = torch.from_numpy(y.astype(np.float32)).clone()
    data_e = torch.from_numpy(e).clone()

    # prepare model
    boundaries = torch.linspace(0.0, 1.0, 5)
    dist = DistributionLinear(boundaries)
    loss_fn = NegativeLogLikelihood(dist, boundaries)
    mlp = MLP(3, 4)

    # train model
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
    for epoch in range(100):
        pred = mlp(data_x)
        loss = loss_fn.loss(pred, data_y, data_e)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('epoch=%d, loss=%f' % (epoch,loss))

    pred = mlp(data_x)
    F_ = loss_fn._compute_F(pred, data_y)
    S = 1 - F_
