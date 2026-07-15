
import os
import copy
import random
import functools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch
from torch import nn

# %%
def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


class DNNModel(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dims=None, n_layers=3, dropout=0., bias=True, non_linearity='lrelu',
                 batch_norm=False, last_layer=None, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        if hidden_dims is None:
            hidden_dims = [64] * n_layers

        n_layers = len(hidden_dims)
        if n_layers == 0:
            self.intermediate_layer = nn.Sequential()
            modules = [nn.Linear(in_dim, out_dim, bias=bias)]

        else:
            modules = [nn.Linear(in_dim, hidden_dims[0], bias=bias)]
            if dropout > 0:
                modules += [nn.Dropout(dropout)]
            if batch_norm:
                modules += [nn.BatchNorm1d(hidden_dims[0])]
            modules += [get_non_linearity(non_linearity)()]

            for i in range(n_layers - 1):
                modules += [nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=bias)]
                if batch_norm:
                    modules += [nn.BatchNorm1d(hidden_dims[i + 1])]
                modules += [get_non_linearity(non_linearity)()]
                if dropout > 0:
                    modules += [nn.Dropout(dropout)]

            self.intermediate_layer = nn.Sequential(*modules)

            modules += [nn.Linear(hidden_dims[-1], out_dim, bias=bias)]

        if last_layer is not None:
            modules += [last_layer()]

        self.network = nn.Sequential(*modules)

    def forward(self, x):
        return self.network(x)

    def change_requires_grad(self, new_val):
        for p in self.parameters():
            p.requires_grad = new_val

    def freeze(self):
        self.change_requires_grad(True)

    def unfreeze(self):
        self.change_requires_grad(False)
    
    def get_alpha(self):
        return self.alpha


# %%
def batch_pinball_loss(quantile_level, quantile, y):
    diff = quantile - y.squeeze()
    mask = (diff.ge(0).float() - quantile_level).detach()

    return (mask * diff).mean(dim=0)

def compute_loss(x, y, model):
    alpha = model.get_alpha()
    pred = model(x).squeeze()
    y = y.squeeze()
    lower_loss = batch_pinball_loss(alpha/2, pred[:, 0], y)
    upper_loss = batch_pinball_loss(1-alpha/2, pred[:, 1], y)
    return (lower_loss + upper_loss) / 2

def train_model(model, x_train, y_train, epochs=1000, batch_size = 64, lr=1e-3, n_wait=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = x_train.shape[0]
    idx = np.random.permutation(n)
    train_ratio = 0.9
    train_idx = idx[:int(n*train_ratio)]
    val_idx = idx[int(n*train_ratio):]
    
    x_val, y_val = x_train[val_idx], y_train[val_idx]
    x_train, y_train = x_train[train_idx], y_train[train_idx]
    
    best_loss = np.inf
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    best_model = model
    for e in tqdm(range(epochs)):
        epoch_losses = []
        idx = np.random.permutation(len(x_train))
        x_train = x_train[idx]
        y_train = y_train[idx]
        model.train()
        for i in range(0, len(x_train), batch_size):
            start = i
            end = min(len(x_train), i+batch_size)
            x = x_train[start:end]
            y = y_train[start:end]
            x.requires_grad = True
            loss = compute_loss(x, y, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses += [loss.item()]
        train_losses += [np.mean(epoch_losses)]
        
        model.eval()
        with torch.no_grad():
            val_loss = compute_loss(x_val, y_val, model).item()
            val_losses += [val_loss]
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= n_wait:
                break
                
    plt.plot(range(len(train_losses)), train_losses, label='train')
    plt.plot(range(len(val_losses)), val_losses, label='validation')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def test_model(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        test_loss = compute_loss(x_test, y_test, model).item()
        print(f"number of test samples: {x_test.shape[0]}")
        print(f"Test loss: {test_loss}")


# %%
base_data_dir = "./meps_data"

df19 = pd.read_csv(f'{base_data_dir}/meps_19_reg.csv')
df20 = pd.read_csv(f'{base_data_dir}/meps_20_reg.csv')
df21 = pd.read_csv(f'{base_data_dir}/meps_21_reg.csv')

# Drop unused column
df19 = df19.drop(columns=['Unnamed: 0'])
df20 = df20.drop(columns=['Unnamed: 0'])
df21 = df21.drop(columns=['Unnamed: 0'])

# get only the columns that are in all three dataframes
common_columns = list(set(df19.columns) & set(df20.columns) & set(df21.columns))
df19 = df19[common_columns]
df20 = df20[common_columns]
df21 = df21[common_columns]

# Drop rows with missing values
df19 = df19.dropna()
df20 = df20.dropna()
df21 = df21.dropna()


# %%
df=df19

def df_to_x_y(df, test_size=0.1, random_state=26):
    df = df.dropna()

    # Separate features and target
    y = df['UTILIZATION_reg'].values
    X = df.drop(columns=['UTILIZATION_reg'])

    # Preprocessing: remove near-constant columns and scale features
    # low_variance_cols = X.columns[X.nunique() <= 1]
    # X = X.drop(columns=low_variance_cols)

    # Normalize continuous features and keep others as-is
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Log-transform the target to reduce skewness
    y = np.log1p(y)
    if test_size == 1.0 or test_size == 0.0:
        x_train = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)  # shape: [batch, 1, features]
        y_train = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
        x_test = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
        y_test = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
        return x_train, y_train, x_test, y_test
    else:
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

        # Convert to PyTorch tensors
        x_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # shape: [batch, 1, features]
        y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
        x_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
        y_test = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
        return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = df_to_x_y(df, test_size=0)


# %%
alpha = 0.3
model = DNNModel(x_train.shape[-1], 2, hidden_dims=[256, 128, 64, 32], dropout=0.3, batch_norm=False, alpha=alpha)
train_model(model, x_train, y_train, epochs=50, batch_size=128, lr=1e-4, n_wait=100)

# %%
test_model(model, x_train, y_train)

# %%
test_frac = 1.0
x_train,y_train, x_test, y_test = df_to_x_y(df20, test_frac)
x_test.shape,y_test.shape, x_train.shape,y_train.shape

# %%
test_model(model, x_test, y_test)

# %% [markdown]
# # Save DS

# %%
age_ranges = [(0,60),(40,60), (60,100)]
base_score_dir = "./meps_scores"
# %%
df_name = 'meps_19'
df = df19
test_frac = 1.0

base_dir = f'{base_score_dir}/alpha_{alpha}/{df_name}'
for age_range in age_ranges:
    lower_bound, upper_bound = age_range
    filtered_df = df[(df['AGE'] >= lower_bound) & (df['AGE'] < upper_bound)]
    print(f"number of samples in age range {age_range}: {len(filtered_df)}")
    x_train,y_train, x_test, y_test = df_to_x_y(filtered_df, test_frac)
    y_pred = model(x_test).detach().numpy()
    y_true = y_test.detach().numpy()
    folder_name = f"ages_{lower_bound}_{upper_bound}"
    out_dir = os.path.join(base_dir, folder_name)   
    if not os.path.exists(out_dir):
        print(f"Creating folder {out_dir}")
        os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'pred.npy'), y_pred)
    np.save(os.path.join(out_dir, 'true.npy'), y_true)

print(f"number of samples in age range all: {len(df)}")
x_train,y_train, x_test, y_test = df_to_x_y(df, test_frac)
y_pred = model(x_test).detach().numpy()
y_true = y_test.detach().numpy()
folder_name = f"ages_all"
out_dir = os.path.join(base_dir, folder_name)   
if not os.path.exists(out_dir):
    print(f"Creating folder {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
np.save(os.path.join(out_dir, 'pred.npy'), y_pred)
np.save(os.path.join(out_dir, 'true.npy'), y_true)


# %%
df_name = 'meps_20'
df = df20
test_frac = 1.0

base_dir = f'{base_score_dir}/alpha_{alpha}/{df_name}'
for age_range in age_ranges:
    lower_bound, upper_bound = age_range
    filtered_df = df[(df['AGE'] >= lower_bound) & (df['AGE'] < upper_bound)]
    print(f"number of samples in age range {age_range}: {len(filtered_df)}")
    x_train,y_train, x_test, y_test = df_to_x_y(filtered_df, test_frac)
    y_pred = model(x_test).detach().numpy()
    y_true = y_test.detach().numpy()
    folder_name = f"ages_{lower_bound}_{upper_bound}"
    out_dir = os.path.join(base_dir, folder_name)   
    if not os.path.exists(out_dir):
        print(f"Creating folder {out_dir}")
        os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'pred.npy'), y_pred)
    np.save(os.path.join(out_dir, 'true.npy'), y_true)

print(f"number of samples in age range all: {len(df)}")
x_train,y_train, x_test, y_test = df_to_x_y(df, test_frac)
y_pred = model(x_test).detach().numpy()
y_true = y_test.detach().numpy()
folder_name = f"ages_all"
out_dir = os.path.join(base_dir, folder_name)   
if not os.path.exists(out_dir):
    print(f"Creating folder {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
np.save(os.path.join(out_dir, 'pred.npy'), y_pred)
np.save(os.path.join(out_dir, 'true.npy'), y_true)


# %%
df_name = 'meps_21'
df = df21
test_frac = 1.0

base_dir = f'{base_score_dir}/alpha_{alpha}/{df_name}'
for age_range in age_ranges:
    lower_bound, upper_bound = age_range
    filtered_df = df[(df['AGE'] >= lower_bound) & (df['AGE'] < upper_bound)]
    print(f"number of samples in age range {age_range}: {len(filtered_df)}")
    x_train,y_train, x_test, y_test = df_to_x_y(filtered_df, test_frac)
    y_pred = model(x_test).detach().numpy()
    y_true = y_test.detach().numpy()
    folder_name = f"ages_{lower_bound}_{upper_bound}"
    out_dir = os.path.join(base_dir, folder_name)   
    if not os.path.exists(out_dir):
        print(f"Creating folder {out_dir}")
        os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'pred.npy'), y_pred)
    np.save(os.path.join(out_dir, 'true.npy'), y_true)

print(f"number of samples in age range all: {len(df)}")
x_train,y_train, x_test, y_test = df_to_x_y(df, test_frac)
y_pred = model(x_test).detach().numpy()
y_true = y_test.detach().numpy()
folder_name = f"ages_all"
out_dir = os.path.join(base_dir, folder_name)   
if not os.path.exists(out_dir):
    print(f"Creating folder {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
np.save(os.path.join(out_dir, 'pred.npy'), y_pred)
np.save(os.path.join(out_dir, 'true.npy'), y_true)



