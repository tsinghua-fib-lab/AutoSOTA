import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

####################################################################################

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def make_splits(df, train_size=1000, seed=0):    
    set_seed(seed)
    rng = np.random.default_rng(seed)

    # fixed test set
    test = df.tail(1000).copy()
    remaining = df.iloc[:-1000].copy()
    perm = rng.permutation(remaining.index.values)

    # sample split
    assert 0 < train_size and 2*train_size <= len(remaining)
    nuisance = remaining.loc[perm[:train_size]].copy()
    stage_two = remaining.loc[perm[train_size:2*train_size]].copy()

    def train_val_helper(training_pool, rng):
        train_size = len(training_pool)
        n_val = int(train_size * 0.2)
        
        # val set
        val_indices = rng.choice(training_pool.index, size=n_val, replace=False)
        val = training_pool.loc[val_indices].copy()
    
        # train set
        train = training_pool.drop(val_indices)
        return train, val

    # split samples
    nuisance_train, nuisance_val = train_val_helper(nuisance, rng)
    stage_two_train, stage_two_val = train_val_helper(stage_two, rng)
        
    return nuisance_train, nuisance_val, stage_two_train, stage_two_val, test

####################################################################################

class NuisanceDataset(Dataset):
    def __init__(self, df, confounders):
        self.df = df
        self.confounders = confounders
        self.T = 'T'
        self.Y = 'Y'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(row[self.confounders].astype(np.float32).to_numpy(), dtype=torch.float32)
        t = torch.tensor(float(row[self.T]), dtype=torch.float32)
        y = torch.tensor(float(row[self.Y]), dtype=torch.float32)
        return x, t, y


class CateDataset(Dataset):
    def __init__(self, df, confounders):
        self.df = df
        self.confounders = confounders
        self.DR = 'DR'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(row[self.confounders].astype(np.float32).to_numpy(), dtype=torch.float32)
        dr = torch.tensor(float(row[self.DR]), dtype=torch.float32)
        return x, dr


class RankerDataset(Dataset):    
    def __init__(self, df, confounders, kappa):
        # plug-in predictions
        df = df.copy()
        df["tau_hat"] = df["m1_hat"] - df["m0_hat"]

        # tensors
        self.X     = torch.as_tensor(df[confounders].to_numpy(), dtype=torch.float32)
        self.tau_hat   = torch.as_tensor(df["tau_hat"].to_numpy(), dtype=torch.float32)
        self.DR    = torch.as_tensor(df["DR"].to_numpy(), dtype=torch.float32)

        # other
        self.N = self.X.size(0)
        self.kappa = float(kappa)

    def __len__(self):
        return self.N * (self.N - 1)

    
    @staticmethod
    def k_to_ij(k: int, N: int):
        """
        k is index in [0, N*(N-1)], (i, j) is pair with i != j.
        """
        i = k // (N - 1)
        j = k % (N - 1)
        if j >= i:
            j += 1
        return i, j

    def __getitem__(self, idx):
        
        # get pair indices
        i, j = self.k_to_ij(idx, self.N)

        # features
        x_i = self.X[i]
        x_j = self.X[j]

        # soft label
        soft = torch.sigmoid((self.tau_hat[i] - self.tau_hat[j]) / self.kappa)

        # pseudo label
        weight = (1/self.kappa)*(soft * (1.0 - soft))
        delta = ((self.DR[i] - self.tau_hat[i]) - (self.DR[j] - self.tau_hat[j]))
        orth = torch.clamp(soft + weight * delta, 0, 1)
        
        # return
        return x_i, x_j, soft, orth
        
####################################################################################

def make_nuisance_loaders(train_df, val_df, confounders, batch_size):
    train_ds = NuisanceDataset(train_df, confounders)
    val_ds = NuisanceDataset(val_df, confounders)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def make_cate_loaders(train_df, val_df, confounders, batch_size):
    train_ds = CateDataset(train_df, confounders)
    val_ds = CateDataset(val_df, confounders)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def make_ranker_loaders(train_df, val_df, confounders, kappa, batch_size):
    train_ds = RankerDataset(train_df, confounders, kappa)
    val_ds = RankerDataset(val_df, confounders, kappa)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
    
####################################################################################

@torch.no_grad()
def score_propensity(df, confounders, model, device):
    """helper to get propensity predictions in dataframe."""
    batch_size=1024
    model.eval()
    xs = torch.tensor(df[confounders].astype(np.float32).values, dtype=torch.float32)
    probs = []
    for i in range(0, len(xs), batch_size):
        x = xs[i:i+batch_size].to(device)
        logits = model(x)
        probs.append(torch.sigmoid(logits).detach().cpu())
    return torch.cat(probs).numpy()

@torch.no_grad()
def score_response(df, confounders, model, device):
    """helper to get reponse surface predictions in dataframe."""
    batch_size=1024
    model.eval()
    xs = torch.tensor(df[confounders].astype(np.float32).values, dtype=torch.float32)
    preds = []
    for i in range(0, len(xs), batch_size):
        x = xs[i:i+batch_size].to(device)
        yhat = model(x)
        preds.append(yhat.detach().cpu())
    return torch.cat(preds).numpy()

def compute_dr_scores(df, confounders, prop_model, m0_model, m1_model, device):
    # score nuisances
    e_hat  = score_propensity(df, confounders, prop_model, device).astype(np.float32).ravel()
    m0_hat = score_response(df, confounders, m0_model, device).astype(np.float32).ravel()
    m1_hat = score_response(df, confounders, m1_model, device).astype(np.float32).ravel()

    # DR components
    T = df["T"].astype(np.float32).to_numpy()
    Y = df["Y"].astype(np.float32).to_numpy()

    # stabilize denominator
    eps = 1e-3
    denom_e   = np.clip(e_hat,   eps, 1.0 - eps)
    denom_1_e = np.clip(1.0 - e_hat, eps, 1.0 - eps)

    # compute DR scores
    term_treated = (T * (Y - m1_hat)) / denom_e
    term_control = ((1.0 - T) * (Y - m0_hat)) / denom_1_e
    dr = term_treated - term_control + (m1_hat - m0_hat)

    # store
    df["e_hat"]  = e_hat
    df["m0_hat"] = m0_hat
    df["m1_hat"] = m1_hat
    df["DR"] = dr.astype(np.float32)
    return df