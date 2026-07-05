import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from torchmetrics import AUROC, AveragePrecision
from model import _RealNVP
import time

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.targets = y
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), self.targets[idx]

def NRDE_run(train_data, train_labels, test_data, test_labels,
                   lr=0.001, grad_pun=0.1, n_epochs=100, bs=512, mid_dim=2048,
                   act=2, adam=True, PNAL='L_1sq', learn=False, mu=torch.zeros(1), sigma=torch.ones(1),
                   kl_weight=0.0, verbose=True):


    train_data = np.asarray(train_data, dtype=np.float32)
    test_data = np.asarray(test_data, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)
    
    input_dim = train_data.shape[1]
    
 
    mus = np.mean(train_data, axis=0)
    sds = np.std(train_data, axis=0)
    sds[sds == 0] = 1
    train_data_norm = (train_data - mus) / sds
    test_data_norm = (test_data - mus) / sds
    

    train_set = CustomDataset(train_data_norm, train_labels)
    test_set = CustomDataset(test_data_norm, test_labels)
    
    std = torch.diag(torch.tensor(1.0 / sds)).to(device).float()
    

    model = _RealNVP(
        input_dim=input_dim,
        mid_dim=mid_dim,
        masktype=0,
        act=act,
        mu=mu.to(device),
        sigma=sigma.to(device),
        learn=learn
    ).to(device)
    
 
    if adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8)
    
    best_auc = 0.0
    best_auprc = 0.0
    
    for epoch in range(n_epochs):
        if verbose:
            print(f"\n[EPOCH {epoch}]")
        torch.cuda.empty_cache()
        
        train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=2048, shuffle=False, num_workers=0)
        con_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)
        

        train_one_epoch(model, train_loader, optimizer, std, grad_pun, PNAL, kl_weight=kl_weight, verbose=verbose)

        top_idx = contribution_calculation(model, con_loader, std, verbose=verbose)
        new_test_set, _, _ = build_contributed_testset(model, test_loader, top_idx)
        new_test_loader = DataLoader(new_test_set, batch_size=2048, shuffle=False, num_workers=0)
        auc, prc = testing(model, new_test_loader)

        if auc > best_auc:
            best_auc = auc
            best_auprc = prc
        #if verbose:
            #print(f"Current AUROC: {auc:.3f}, AUPRC: {prc:.3f} | Best so far: {best_auc:.3f}")

    return best_auc, best_auprc


def NRDE_run_verbose(train_data, train_labels, test_data, test_labels,
                     lr=0.001, grad_pun=0.1, n_epochs=100, bs=512, mid_dim=2048,
                     act=2, adam=True, PNAL='L_1sq', learn=False, mu=torch.zeros(1), sigma=torch.ones(1),
                     kl_weight=0.0, verbose=False):
    """Like NRDE_run but logs per-epoch metrics and returns them for diagnostics."""

    train_data = np.asarray(train_data, dtype=np.float32)
    test_data = np.asarray(test_data, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)

    input_dim = train_data.shape[1]

    mus = np.mean(train_data, axis=0)
    sds = np.std(train_data, axis=0)
    sds[sds == 0] = 1
    train_data_norm = (train_data - mus) / sds
    test_data_norm = (test_data - mus) / sds

    train_set = CustomDataset(train_data_norm, train_labels)
    test_set = CustomDataset(test_data_norm, test_labels)

    std = torch.diag(torch.tensor(1.0 / sds)).to(device).float()

    model = _RealNVP(
        input_dim=input_dim,
        mid_dim=mid_dim,
        masktype=0,
        act=act,
        mu=mu.to(device),
        sigma=sigma.to(device),
        learn=learn
    ).to(device)

    if adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8)

    best_auc = 0.0
    best_auprc = 0.0
    epoch_log = []

    for epoch in range(n_epochs):
        torch.cuda.empty_cache()

        train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=2048, shuffle=False, num_workers=0)
        con_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)

        loss_val = train_one_epoch(model, train_loader, optimizer, std, grad_pun, PNAL, kl_weight=kl_weight, verbose=False)

        top_idx = contribution_calculation(model, con_loader, std, verbose=False)
        new_test_set, _, _ = build_contributed_testset(model, test_loader, top_idx)
        new_test_loader = DataLoader(new_test_set, batch_size=2048, shuffle=False, num_workers=0)
        auc, prc = testing(model, new_test_loader)

        if auc > best_auc:
            best_auc = auc
            best_auprc = prc

        epoch_log.append({
            "epoch": epoch,
            "loss": float(loss_val),
            "auroc": float(auc),
            "auprc": float(prc),
        })

        if verbose and (epoch % 10 == 0 or epoch < 5 or epoch >= n_epochs - 5):
            print(f"Epoch {epoch:3d}: loss={loss_val:.4f}, AUROC={auc:.4f}, AUPRC={prc:.4f}, best_AUROC={best_auc:.4f}")

    return best_auc, best_auprc, epoch_log


def NRDE_run_ensemble(train_data, train_labels, test_data, test_labels,
                      lr=0.001, grad_pun=0.1, n_epochs=100, bs=512, mid_dim=2048,
                      act=2, adam=True, PNAL='L_1sq', learn=False, mu=torch.zeros(1), sigma=torch.ones(1),
                      kl_weight=0.0, verbose=True):
    """Like NRDE_run but returns per-sample anomaly scores from best epoch for ensemble averaging."""

    train_data = np.asarray(train_data, dtype=np.float32)
    test_data = np.asarray(test_data, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)

    input_dim = train_data.shape[1]

    mus = np.mean(train_data, axis=0)
    sds = np.std(train_data, axis=0)
    sds[sds == 0] = 1
    train_data_norm = (train_data - mus) / sds
    test_data_norm = (test_data - mus) / sds

    train_set = CustomDataset(train_data_norm, train_labels)
    test_set = CustomDataset(test_data_norm, test_labels)

    std = torch.diag(torch.tensor(1.0 / sds)).to(device).float()

    model = _RealNVP(
        input_dim=input_dim,
        mid_dim=mid_dim,
        masktype=0,
        act=act,
        mu=mu.to(device),
        sigma=sigma.to(device),
        learn=learn
    ).to(device)

    if adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8)

    best_auc = 0.0
    best_auprc = 0.0
    best_preds = None
    best_targets = None

    for epoch in range(n_epochs):
        if verbose:
            print(f"\n[EPOCH {epoch}]")
        torch.cuda.empty_cache()

        train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=2048, shuffle=False, num_workers=0)
        con_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)

        train_one_epoch(model, train_loader, optimizer, std, grad_pun, PNAL, kl_weight=kl_weight, verbose=verbose)

        top_idx = contribution_calculation(model, con_loader, std, verbose=verbose)
        new_test_set, _, _ = build_contributed_testset(model, test_loader, top_idx)
        new_test_loader = DataLoader(new_test_set, batch_size=2048, shuffle=False, num_workers=0)
        auc, prc, preds, targets = testing_with_scores(model, new_test_loader)

        if auc > best_auc:
            best_auc = auc
            best_auprc = prc
            best_preds = preds
            best_targets = targets

    return best_auc, best_auprc, best_preds, best_targets


def testing_with_scores(model, test_loader):
    """Like testing() but also returns per-sample predictions and targets."""
    model = model.to(device)
    model.eval()
    preds, targets = [], []
    for batch in test_loader:
        x, y = batch
        x = x.to(device).float()
        y = y.to(device)
        outputs, sldj = model(x, sldj=0)
        log_likelihood = -0.5 * (torch.pow(outputs, 2))
        sample_likelihood = torch.sum(log_likelihood, dim=1)
        pred = -(sample_likelihood + sldj)
        preds.append(pred)
        targets.append(y)
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    auroc = AUROC(task="binary")
    ap = AveragePrecision(task="binary")
    return auroc(preds, targets), ap(preds, targets), preds, targets


def first_i_by_max_gap_ratio_1d(jac_sum: torch.Tensor):
    assert jac_sum.dim() == 1
    a_sorted, idx_sorted = torch.sort(jac_sum, descending=False)
    a_i = a_sorted[:-1]
    a_ip1 = a_sorted[1:]
    eps = 1e-12
    ratios = (a_ip1 - a_i) / (a_ip1 + eps)
    i_star0 = torch.argmax(ratios).item()
    i_star1 = i_star0 + 1
    return idx_sorted[:i_star1]

def contribution_calculation(model, train_loader, std, verbose=True):
    model = model.to(device)
    model.train()
    jac_sum = None
    num = 0
    loader = tqdm(train_loader) if verbose else train_loader
    for batch in loader:
        imgs, _ = batch
        imgs = imgs.float().to(device)
        imgs.requires_grad = True
        outputs, _ = model(imgs, 0)
        num += outputs.shape[0]
        outVector = torch.sum(outputs, 0).view(-1)
        outdim = outVector.size(0)
        jac = torch.stack(
            [torch.autograd.grad(outVector[i], imgs, retain_graph=True, create_graph=False)[0]
             for i in range(outdim)], dim=0)
        jac = jac.permute(1, 0, 2)          # (B, outdim, in_dim)
        jac = jac @ std
        jac = torch.abs(jac)
        if jac_sum is None:
            jac_sum = torch.sum(jac, dim=0)
        else:
            jac_sum += torch.sum(jac, dim=0)
    jac_sum = jac_sum / num
    jac_sum = torch.norm(jac_sum, dim=1)   # (outdim,)
    top_idx = first_i_by_max_gap_ratio_1d(jac_sum)
    #if verbose:
        #print(f"Selected {len(top_idx)} dimensions out of {outdim}")
    return top_idx

def build_contributed_testset(model, test_loader, top_idx):
    model = model.to(device)
    model.eval()
    new_x_list = []
    new_y_list = []
    if not torch.is_tensor(top_idx):
        top_idx = torch.tensor(top_idx, dtype=torch.long)
    top_idx = top_idx.long()
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(device).float()
            y = y.to(device)
            z, _ = model(x, sldj=0)
            if z.dim() == 2:
                flat_idx = top_idx.view(-1)
                mask = torch.zeros_like(z)
                mask[:, flat_idx] = 1.0
                z_masked = z * mask
            else:
                mask = torch.zeros_like(z)
                rows = top_idx[:, 0]
                cols = top_idx[:, 1]
                mask[:, rows, cols] = 1.0
                z_masked = z * mask
            x_new, _ = model.reverse(z_masked, sldj=0)
            new_x_list.append(x_new.detach().cpu())
            new_y_list.append(y.detach().cpu())
    new_test_data = torch.cat(new_x_list, dim=0).numpy()
    new_test_lab  = torch.cat(new_y_list, dim=0).numpy()
    return CustomDataset(new_test_data, new_test_lab), new_test_data, new_test_lab

def train_one_epoch(model, train_loader, optimizer, std, grad_pun, PNAL, kl_weight=0.0, verbose=True):
    model.train()
    total_loss = 0.0
    loader = tqdm(train_loader) if verbose else train_loader
    for batch in loader:
        optimizer.zero_grad()
        imgs, _ = batch
        imgs = imgs.to(device).float()
        imgs.requires_grad = True
        outputs, sldj = model(imgs, sldj=0)

        log_likelihood = -0.5 * (torch.pow(outputs, 2) + torch.log(torch.tensor(torch.pi * 2)))
        sample_likelihood = torch.sum(log_likelihood, dim=1)
        loss = torch.mean(-(sample_likelihood + sldj))

        jac = 0.0
        if grad_pun != 0:
            outVector = torch.sum(outputs, 0).view(-1)
            outdim = outVector.size(0)
            jac = torch.stack(
                [torch.autograd.grad(outVector[i], imgs, retain_graph=True, create_graph=True)[0]
                 for i in range(outdim)], dim=0)
            jac = jac.permute(1, 0, 2)
            jac = torch.matmul(jac, std)
            jac = torch.abs(jac)
            jac = torch.mean(jac, dim=0)
            if PNAL == 'L_1sq':
                jac = torch.sum(torch.sqrt(torch.sum(jac, dim=1)))
            else:
                jac = torch.sum(jac)
        loss += jac * grad_pun

        # KL divergence regularization: push latent toward N(0,I)
        kl_loss_val = 0.0
        if kl_weight > 0:
            z_mean = outputs.mean(dim=0)
            z_var = outputs.var(dim=0, unbiased=False)
            kl_loss_val = 0.5 * (z_mean.pow(2).sum() + z_var.sum() - outputs.size(1) - torch.log(z_var + 1e-8).sum())
            loss += kl_weight * kl_loss_val

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    if verbose:
        print(f"Train | avg loss = {avg_loss:.5f}")
    return avg_loss

def testing(model, test_loader):
    model = model.to(device)
    model.eval()
    preds, targets = [], []
    for batch in test_loader:
        x, y = batch
        x = x.to(device).float()
        y = y.to(device)
        outputs, sldj = model(x, sldj=0)
        log_likelihood = -0.5 * (torch.pow(outputs, 2))
        sample_likelihood = torch.sum(log_likelihood, dim=1)
        pred = -(sample_likelihood + sldj)
        preds.append(pred)
        targets.append(y)
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    auroc = AUROC(task="binary")
    ap = AveragePrecision(task="binary")
    return auroc(preds, targets), ap(preds, targets)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.targets = y
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), self.targets[idx]


def read_contaminated_data(file, normalization='z-score', seed=42,noise_rate=0):
    if file.endswith('.npz'):
        data = np.load(file, allow_pickle=True)
        x, y = data['X'], data['y']
        y = np.array(y, dtype=int)
    else:
        if file.endswith('pkl'):
            func = pd.read_pickle
        elif file.endswith('csv'):
            func = pd.read_csv
        else:
            raise NotImplementedError('')

        df = func(file)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        x = df.values[:, :-1]
        y = np.array(df.values[:, -1], dtype=int)

    # train-test splitting
    rng = np.random.RandomState(seed)
    idx = rng.permutation(np.arange(len(x)))
    #idx = np.random.permutation(np.arange(len(x)))
    #print(idx[0:10])
    x, y = x[idx], y[idx]

    norm_idx = np.where(y==0)[0]
    anom_idx = np.where(y==1)[0]
    split = int(0.5 * len(norm_idx))
    ab_train=int(split*noise_rate)
    max_ab_train=int(split*0.1)
    train_norm_idx, test_norm_idx = norm_idx[:split], norm_idx[split:]
    train_ab_idx, test_ab_idx = anom_idx[:ab_train], anom_idx[max_ab_train:]
    x_train = x[np.hstack([train_norm_idx,train_ab_idx])]
    y_train = y[np.hstack([train_norm_idx, train_ab_idx])]
    data_dim=x_train.shape[1]
    x_test = x[np.hstack([test_norm_idx, test_ab_idx])]
    y_test = y[np.hstack([test_norm_idx, test_ab_idx])]

    print(f'Original size: [{x.shape}], Normal/Anomaly: [{len(norm_idx)}/{len(anom_idx)}] \n'
          f'After splitting: training/testing [{len(x_train)}/{len(x_test)}]')
    #print(str(torch.rand(1)))
    sds=None
    # normalization
    if normalization == 'min-max':
        minmax_scaler = MinMaxScaler()
        minmax_scaler.fit(x_train)
        x_train = minmax_scaler.transform(x_train)
        x_test = minmax_scaler.transform(x_test)

    elif normalization == 'z-score':
        mus = np.mean(x_train, axis=0)
        sds = np.std(x_train, axis=0)
        sds[sds == 0] = 1
        x_train = np.array([(xx - mus) / sds for xx in x_train])
        x_test = np.array([(xx - mus) / sds for xx in x_test])

    elif normalization == 'scale':
        x_train = x_train / 255
        x_test = x_test / 255
    elif normalization =='ours':
        mean=np.mean(x_train,0)
        std=np.std(x_train,0)
        x_train=(x_train-mean)/ (std + 1e-4)
        x_test= (x_test - mean)/(std + 1e-4)

    return x_train, y_train, x_test, y_test,data_dim,sds


def read_gauss_noise_data(file, normalization='z-score', train_level=1,test_level=2,seed=42):
    if file.endswith('.npz'):
        data = np.load(file, allow_pickle=True)
        x, y = data['X'], data['y']
        y = np.array(y, dtype=int)
    else:
        if file.endswith('pkl'):
            func = pd.read_pickle
        elif file.endswith('csv'):
            func = pd.read_csv
        else:
            raise NotImplementedError('')

        df = func(file)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        x = df.values[:, :-1]
        y = np.array(df.values[:, -1], dtype=int)

    # train-test splitting
    rng = np.random.RandomState(seed)
    idx = rng.permutation(np.arange(len(x)))
    #idx = np.random.permutation(np.arange(len(x)))
    #print(idx[0:10])
    x, y = x[idx], y[idx]

    norm_idx = np.where(y==0)[0]
    anom_idx = np.where(y==1)[0]
    split = int(0.5 * len(norm_idx))
    train_norm_idx, test_norm_idx = norm_idx[:split], norm_idx[split:]
    x_train=x[train_norm_idx]
    x_test_norm= x[test_norm_idx]
    x_test_ab=x[anom_idx]
    if normalization == 'min-max':
        minmax_scaler = MinMaxScaler()
        minmax_scaler.fit(x_train)
        x_train = minmax_scaler.transform(x_train)
        x_test = minmax_scaler.transform(x_test)

    elif normalization == 'z-score':
        mus = np.mean(x_train, axis=0)
        sds = np.std(x_train, axis=0)
        sds[sds == 0] = 1
        x_train = np.array([(xx - mus) / sds for xx in x_train])
        x_test_norm = np.array([(xx - mus) / sds for xx in x_test_norm])
        x_test_ab = np.array([(xx - mus) / sds for xx in x_test_ab])

    elif normalization == 'scale':
        x_train = x_train / 255
        x_test = x_test / 255
    elif normalization =='ours':
        mean=np.mean(x_train,0)
        std=np.std(x_train,0)
        x_train=(x_train-mean)/ (std + 1e-4)
        x_test= (x_test - mean)/(std + 1e-4)
    data_dim=x_train.shape[1]
    y_train = y[train_norm_idx]
    noise_train= np.random.normal(
    loc=0.0,
    scale=np.sqrt(train_level),
    size=(len(train_norm_idx), x.shape[1])
)  
    noise_test_abnorm= np.random.normal(
    loc=0.0,
    scale=np.sqrt(test_level),
    size=(len(anom_idx), x.shape[1])
)  
    noise_test_norm= np.random.normal(
    loc=0.0,
    scale=np.sqrt(train_level),
    size=(len(test_norm_idx), x.shape[1])
)
    print(noise_test_norm)
    x_test_ab =x_test_ab +noise_test_abnorm
    x_test_norm=x_test_norm+noise_test_norm
    x_train=x_train+noise_train
    #x_train=x[train_norm_idx]
    x_test = np.concat([x_test_norm, x_test_ab])
    y_test = y[np.hstack([test_norm_idx, anom_idx])]

    print(f'Original size: [{x.shape}], Normal/Anomaly: [{len(norm_idx)}/{len(anom_idx)}] \n'
          f'After splitting: training/testing [{len(x_train)}/{len(x_test)}]')
    #print(str(torch.rand(1)))
    #sds=None
    # normalization

    return x_train, y_train, x_test, y_test,data_dim,sds

def read_data(file, normalization='z-score', seed=42):
    if file.endswith('.npz'):
        data = np.load(file, allow_pickle=True)
        x, y = data['X'], data['y']
        y = np.array(y, dtype=int)
    else:
        if file.endswith('pkl'):
            func = pd.read_pickle
        elif file.endswith('csv'):
            func = pd.read_csv
        else:
            raise NotImplementedError('')
        df = func(file)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        x = df.values[:, :-1]
        y = np.array(df.values[:, -1], dtype=int)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(np.arange(len(x)))
    x, y = x[idx], y[idx]

    norm_idx = np.where(y == 0)[0]
    anom_idx = np.where(y == 1)[0]
    split = int(0.5 * len(norm_idx))
    train_norm_idx, test_norm_idx = norm_idx[:split], norm_idx[split:]

    x_train = x[train_norm_idx]
    y_train = y[train_norm_idx]
    x_test = x[np.hstack([test_norm_idx, anom_idx])]
    y_test = y[np.hstack([test_norm_idx, anom_idx])]

    print(f'Original size: [{x.shape}], Normal/Anomaly: [{len(norm_idx)}/{len(anom_idx)}]')
    print(f'After splitting: training/testing [{len(x_train)}/{len(x_test)}]')

    if normalization == 'min-max':
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
    elif normalization == 'z-score':
        mus = np.mean(x_train, axis=0)
        sds = np.std(x_train, axis=0)
        sds[sds == 0] = 1
        x_train = (x_train - mus) / sds
        x_test = (x_test - mus) / sds
    elif normalization == 'scale':
        x_train = x_train / 255
        x_test = x_test / 255
    elif normalization == 'ours':
        mean = np.mean(x_train, 0)
        std = np.std(x_train, 0)
        x_train = (x_train - mean) / (std + 1e-4)
        x_test = (x_test - mean) / (std + 1e-4)

    return x_train, y_train, x_test, y_test, x_train.shape[1], sds





