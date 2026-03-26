import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange
import random
from torch.utils.data import TensorDataset
import warnings 
import pandas as pd
warnings.filterwarnings('ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mask_mono = np.array([1 if i in range(51, 55) or i in range(56, 60) else 0 for i in range(280)])

def split_input(inputs):
    return inputs[:, np.where(mask_mono==0)].squeeze(),         inputs[:, np.where(mask_mono!=0)].squeeze() * torch.tensor(mask_mono[np.where(mask_mono!=0)][None,:], dtype=torch.float32).to(device)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    losses = []
    val_losses = []
    for _ in trange(num_epochs, leave=False):
        model.train()
        total = 0
        losses_buffer = []
        for inputs, labels in train_loader:
            inputs_free, inputs_mono = split_input(inputs)
            optimizer.zero_grad()
            outputs = model(inputs_free.float(), inputs_mono.float())
            loss = criterion(outputs, labels.float())
            losses_buffer.append(loss)
            loss.backward()
            optimizer.step()
            total += labels.size(0)
        losses.append(np.mean([el.detach().cpu() for el in losses_buffer]))
        val_loss = validate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
    return losses, val_losses

def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs_free, inputs_mono = split_input(inputs)
            outputs = model(inputs_free, inputs_mono)
            loss = criterion(outputs, labels)
            val_loss += [loss.item()]
    return np.mean(val_loss)

class MonotonicLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, pre_activation=nn.Identity()):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.act = pre_activation
        
    def forward(self, x):
        w_pos = self.weight.clamp(min=0.0)
        w_neg = self.weight.clamp(max=0.0)
        x_pos = F.linear(self.act(x), w_pos, self.bias)
        x_neg = F.linear(self.act(-x), w_neg, self.bias)  
        return x_pos + x_neg

class MonoModel(torch.nn.Module):
    def __init__(self, input_size_mono, num_layers_mono, num_layers_pre_mono, num_neurons_mono, num_neurons_pre_mono, activation=nn.ReLU()):
        super().__init__()
        self.pre_mono = torch.nn.ModuleList([torch.nn.LazyLinear(num_neurons_pre_mono) for _ in range(num_layers_pre_mono)])
        self.mono = torch.nn.ModuleList([
            MonotonicLinear(input_size_mono + num_neurons_pre_mono, num_neurons_mono, pre_activation=nn.Identity()),
            *[MonotonicLinear(num_neurons_mono, num_neurons_mono, pre_activation=activation) for _ in range(num_layers_mono)],
            MonotonicLinear(num_neurons_mono, 1, pre_activation=activation),
        ])
    
    def forward(self, x, x_mono):
        for layer in self.pre_mono:
            x = torch.nn.functional.silu(layer(x))
        x = torch.cat((x, x_mono), dim=-1)
        for layer in self.mono:
            x = layer(x)
        return x

def run(SEED, lr=1e-3, activation=nn.ELU(), n1=3, n2=2, num_epochs=1500, batch_size=256, num_layers_mono=2, num_layers_pre_mono=2):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)

    df_train = pd.read_csv('data/train_blog.csv', header=None)
    df_train = df_train.dropna(axis=0)
    X_train = df_train.to_numpy()[:,:-1]
    y_train = df_train.to_numpy()[:,-1:]

    df_val = pd.read_csv('data/test_blog.csv', header=None)
    df_val = df_val.dropna(axis=0)
    X_val = df_val.to_numpy()[:,:-1]
    y_val = df_val.to_numpy()[:,-1:]

    X_train = torch.tensor(X_train).to(device).float()
    X_val = torch.tensor(X_val).to(device).float()
    y_train = torch.tensor(y_train).to(device).float()
    y_val = torch.tensor(y_val).to(device).float()

    train_loader = torch.utils.data.DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=True, drop_last=True)

    model = MonoModel((mask_mono!=0).sum(), num_layers_mono, num_layers_pre_mono, n1, n2, activation=activation).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)
    best_rmse = np.sqrt(np.min(val_losses))
    print(f'SEED {SEED}: {best_rmse:.5f}')
    return losses, val_losses

if __name__ == '__main__':
    activation = nn.ELU()
    train_losses, val_losses = [], []
    for seed in range(5):
        ltrain, lval = run(seed, activation=activation, lr=1e-3, n1=3, n2=2)
        train_losses.append(ltrain)
        val_losses.append(lval)
    
    mean_rmse = np.mean([np.min(np.sqrt(l)) for l in val_losses])
    std_rmse = np.std([np.min(np.sqrt(l)) for l in val_losses])
    print('---------------------------------')
    print(f'Mean: {mean_rmse:.5f}')
    print(f'Std: {std_rmse:.5f}')
    print('---------------------------------')
    print(f'TEST_RMSE={mean_rmse:.5f}')
