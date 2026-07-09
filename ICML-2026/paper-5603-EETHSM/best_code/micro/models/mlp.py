import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLPLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = F.relu
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, mask=None):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x