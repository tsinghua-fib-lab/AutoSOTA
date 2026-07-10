import torch 
from torch import nn

class IdentiyEncoder(torch.nn.Module):
    def __init__(self, inputdim = 64):
        super(IdentiyEncoder, self).__init__()
        self.inputdim = inputdim
    
    def forward(self, y):
        # y: (batch_size, inputdim)
        return torch.sigmoid(-2*y)

class BadEncoder(torch.nn.Module):
    def __init__(self, inputdim = 64):
        super(BadEncoder, self).__init__()
        self.inputdim = inputdim
    
    def forward(self, y):
        # y: (batch_size, inputdim)
        # corrupt y by zeroing out half of its values
        batch_size = y.size(0)
        y = torch.sin(y * 2)
        return y
    
class NormalEncoder(torch.nn.Module):
    def __init__(self, enc_dim = 64, input_dim = 64, hiddn_dim=128):
        super(NormalEncoder, self).__init__()
        self.enc_dim = enc_dim
        self.input_dim = input_dim
        
        self.fc1 = torch.nn.Linear(input_dim, hiddn_dim)
        self.fc2 = torch.nn.Linear(hiddn_dim, hiddn_dim)
        self.fc3 = torch.nn.Linear(hiddn_dim, enc_dim)

    def forward(self, y):
        # y: (batch_size, inputdim)
        h = torch.relu(self.fc1(y))
        h = torch.relu(self.fc2(h))
        out = self.fc3(h)
        return out
    
# vector field neural network over time steps (CNN)
class VectorField2(torch.nn.Module):
    def __init__(self, inputdim = 64, hiddn_dim=1024, enc_dim=64):
        super(VectorField2, self).__init__()
        self.inputdim = inputdim
        
        self.fc1 = torch.nn.Linear(inputdim + enc_dim + inputdim + 1, hiddn_dim)
        self.fc2 = torch.nn.Linear(hiddn_dim, hiddn_dim)
        self.fc3 = torch.nn.Linear(hiddn_dim, inputdim)
        
    def forward(self, xt, enc_y, y, t):
        # xt: (batch_size, inputdim)
        # ency: (batch_size, inputdim)
        # y: (batch_size, inputdim)
        # t: (batch_size, 1)
        
        inp = torch.cat([xt, enc_y, y, t], dim=1) # Concatenate imputed data and time
        h = torch.relu(self.fc1(inp))
        h = torch.relu(self.fc2(h))
        out = self.fc3(h)
        return out
    
# vector field neural network over time steps (CNN)
class VectorField(torch.nn.Module):
    def __init__(self, inputdim = 64, hiddn_dim=1024):
        super(VectorField, self).__init__()
        self.inputdim = inputdim
        
        self.fc1 = torch.nn.Linear(inputdim + inputdim + inputdim + 1, hiddn_dim)
        self.fc2 = torch.nn.Linear(hiddn_dim, hiddn_dim)
        self.fc3 = torch.nn.Linear(hiddn_dim, inputdim)
        
    def forward(self, xt, enc_y, y, t):
        # xt: (batch_size, inputdim)
        # ency: (batch_size, inputdim)
        # y: (batch_size, inputdim)
        # t: (batch_size, 1)
        
        inp = torch.cat([xt, enc_y, y, t], dim=1) # Concatenate imputed data and time
        h = torch.relu(self.fc1(inp))
        h = torch.relu(self.fc2(h))
        out = self.fc3(h)
        return out
