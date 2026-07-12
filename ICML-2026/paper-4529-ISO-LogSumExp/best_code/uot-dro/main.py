import math
import os
import numpy as np
import urllib.request
import random
import json
from tqdm import tqdm, trange
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

from models import SimpleCNN


def prepare_dataset():
    train_path_cuda = "../data/train_set_cuda.pt"
    val_path, test_path = "../data/val_set.pt", "../data/test_set.pt"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load full training set (60k samples)
    print("Donwloading MNIST (train and val)")
    full_train_dataset = torchvision.datasets.MNIST(
        root='../data/',
        train=True,
        download=False,
        transform=transform
    )

    filename = "../data/feature-dependent_25_ytrain.npy"
    if not os.path.exists(filename):
        print("Donwloading noisy labels...", end=' ')
        url = "https://github.com/gorkemalgan/corrupting_labels_with_distillation/raw/refs/heads/master/noisylabels/mnist/feature-dependent_25_ytrain.npy"
        urllib.request.urlretrieve(url, filename)
        print('Done.')

    y_train_noisy = np.load(filename)
    full_train_dataset.targets = y_train_noisy.tolist()

    # Split into train (50k) and validation (10k)
    train_size = 50000
    val_size = 10000
    torch.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset,
        [train_size, val_size]
    )

    # Load test set (10k samples)
    test_dataset = torchvision.datasets.MNIST(
        root='/kaggle/working',
        train=False,
        download=False,
        transform=transform
    )

    # Convert and save training set (50k)
    X_train = torch.stack([img for img, _ in train_dataset])
    y_train = torch.tensor([full_train_dataset.targets[i] for i in train_dataset.indices])
    torch.save((X_train.to('cuda'), y_train.to('cuda')), train_path_cuda)

    # Convert and save validation set (10k)
    X_val = torch.stack([img for img, _ in val_dataset]).to('cuda')
    y_val = torch.tensor([full_train_dataset.targets[i] for i in val_dataset.indices]).to('cuda')
    torch.save((X_val, y_val), val_path)

    # Convert and save test set (10k)
    X_test = torch.stack([img for img, _ in test_dataset]).to('cuda')
    y_test = torch.tensor([label for _, label in test_dataset]).to('cuda')
    torch.save((X_test, y_test), test_path)


def inner_maximization(model, criterion, X, y, lam, lr=None, momentum=0.4, n_steps=5):
    if lam is None:
        lam = 1.
    if lr is None:
        lr = 0.1 / lam
    U = X.clone().requires_grad_(True)
    v = torch.zeros_like(U)

    for step in range(n_steps):
        U_ahead = U + momentum * v
        preds = model(U_ahead)
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            return None

        loss = criterion(preds, y) - lam * (X - U).pow(2).sum()
        grad, = torch.autograd.grad(loss, U, create_graph=False)
        v = momentum * v + lr * grad
        with torch.no_grad():
            U += v

    return U.detach()


def dro_loss(model, X, y, ce_sum_reduction, ce_no_reduction, lam_beta=1., lam=1., rho=0.1):
    model.freeze_weights()
    U_star = inner_maximization(model, ce_sum_reduction, X, y, lam)
    model.unfreeze_weights()
    if U_star is None:
        return None, None

    preds = model(U_star)
    losses = ce_no_reduction(preds, y)
    costs = (X - U_star).pow(2).sum(dim=(1, 2, 3))
    exponent = losses - lam * costs
    if rho is None:
        total = torch.exp(exponent / lam_beta).mean()
    else:
        adjusted_exponent = (exponent - model.alpha) / lam_beta + math.log(rho)
        total = (lam_beta / rho) * F.softplus(adjusted_exponent).mean() + model.alpha
    return total, exponent


def eval_loss(model, cuda_loader, lam_beta, lam, ce_sum_reduction, ce_no_reduction):
    model.freeze_weights().eval()
    exponents = []
    for X, y in cuda_loader:
        U_star = inner_maximization(model, ce_sum_reduction, X, y, lam)
        preds = model(U_star)
        losses = ce_no_reduction(preds, y)
        costs = (X - U_star).pow(2).sum(dim=(1, 2, 3))
        exponents.append(losses - lam * costs)

    exponents = torch.cat(exponents)
    objective = lam_beta * torch.logsumexp(exponents / lam_beta, dim=0) \
                - lam_beta * math.log(len(exponents))

    model.unfreeze_weights().train()
    return objective.item()


def train(lam_beta, lam, rho, lr, seed, batch_sz=64, n_epochs=20, save_weights=False, extra_checkpoints=[]):
    torch.manual_seed(seed)
    random.seed(seed)

    X_train_cuda, y_train_cuda = torch.load("../data/train_set_cuda.pt")
    cuda_loader = DataLoader(TensorDataset(X_train_cuda, y_train_cuda),
                             batch_size=batch_sz, shuffle=True)

    model = SimpleCNN(with_alpha=rho is not None).cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    ce_sum_reduction = nn.CrossEntropyLoss(reduction='sum')
    ce_no_reduction = nn.CrossEntropyLoss(reduction='none')

    state_dicts = []
    if save_weights:
        state_dicts.append(copy.deepcopy(model.state_dict()))

    for epoch in trange(n_epochs):
        for i, (X, y) in enumerate(cuda_loader):
            if save_weights and (epoch == 0) and (i in extra_checkpoints):
                state_dicts.append(copy.deepcopy(model.state_dict()))
            loss, exponent = dro_loss(model, X, y, ce_sum_reduction, ce_no_reduction,
                                      lam_beta=lam_beta, lam=lam, rho=rho)
            if loss is None:
                print('Floating point exception')
                return state_dicts if save_weights else None

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if save_weights:
            state_dicts.append(copy.deepcopy(model.state_dict()))

    if save_weights:
        return state_dicts

    obj_val = eval_loss(model, cuda_loader, lam_beta, lam, ce_sum_reduction, ce_no_reduction)
    return obj_val


if __name__ == '__main__':
    param_grid = [ # (lam_beta, lam, rho, lr)
        ### lam_beta = 1/5 ###
        (.2, .2, None, 1e-9),
        (.2, .2, 0.01, 1e-2),
        (.2, .2, 0.1 , 1e-2),
        (.2, .2, 1.  , 1e-2),

        (.2, 1., None, 1e-9),
        (.2, 1., 0.01, 1e-2),
        (.2, 1., 0.1 , 1e-2),
        (.2, 1., 1.  , 1e-2),

        (.2, 5., None, 1e-9),
        (.2, 5., 0.01, 1e-3),
        (.2, 5., 0.1 , 1e-2),
        (.2, 5., 1.  , 1e-2),

        ### lam_beta = 1 ###
        (1., .2, None, 1e-4),
        (1., .2, 0.01, 1e-2),
        (1., .2, 0.1 , 1e-2),
        (1., .2, 1.  , 1e-1),

        (1., 1., None, 1e-4),
        (1., 1., 0.01, 1e-2),
        (1., 1., 0.1 , 1e-2),
        (1., 1., 1.  , 1e-1),

        (1., 5., None, 1e-4),
        (1., 5., 0.01, 1e-2),
        (1., 5., 0.1 , 1e-2),
        (1., 5., 1.  , 1e-1),

        ### lam_beta = 5 ###
        (5., .2, None, 1.  ),
        (5., .2, 0.01, 1e-1),
        (5., .2, 0.1 , 1e-1),
        (5., .2, 1.  , 1e-1),

        (5., 1., None, 1.  ),
        (5., 1., 0.01, 1e-1),
        (5., 1., 0.1 , 1e-1),
        (5., 1., 1.  , 1e-1),

        (5., 5., None, 1e-1),
        (5., 5., 0.01, 1e-2),
        (5., 5., 0.1 , 1e-1),
        (5., 5., 1.  , 1e-1)
    ]
    seeds = list(range(5))

    # Computations for Table 2 in the paper
    print("Writing results to results.jsonl")
    with open("results.jsonl", "a") as f:
        for lam_beta, lam, rho, lr in param_grid:
            for seed in seeds:
                obj_val = train(lam_beta, lam, rho, lr, seed)
                if obj_val is None:
                    break

                record = {
                    "lam_beta": lam_beta,
                    "lam": lam,
                    "rho": rho,
                    "lr": lr,
                    "seed": seed,
                    "obj_val": obj_val,
                }

                f.write(json.dumps(record) + "\n")
                f.flush()

    print("Computing objective values for Figure 3 in the paper")
    lam_beta, lam = 1., 1.
    seed = 2
    X_train_cuda, y_train_cuda = torch.load("../data/train_set_cuda.pt")
    cuda_loader = DataLoader(TensorDataset(X_train_cuda, y_train_cuda),
                             batch_size=1000, shuffle=True)
    ce_sum_reduction = nn.CrossEntropyLoss(reduction='sum')
    ce_no_reduction = nn.CrossEntropyLoss(reduction='none')

    for rho, lr in [(None, 1e-4), (0.1, 1e-2), (None, 1e-3)]:
        if lr in [1e-4, 1e-2]:
            print("Baseline:" if rho is None else "Proposed approach:")
            state_dicts = train(lam_beta, lam, rho, lr, seed, save_weights=True)
        else:
            print("Baseline with larger lr (overflow is expected during iteration #615, but that may depend on your system):")
            state_dicts = train(lam_beta, lam, rho, lr, seed,
                                save_weights=True, extra_checkpoints=[300, 600, 610, 611, 612, 613, 614])

        model = SimpleCNN(with_alpha=rho is not None).cuda()
        losses = []
        for weights in tqdm(state_dicts):
            model.load_state_dict(weights)
            obj_val = eval_loss(model, cuda_loader, lam_beta, lam, ce_sum_reduction, ce_no_reduction)
            losses.append(obj_val)
        print(losses)
