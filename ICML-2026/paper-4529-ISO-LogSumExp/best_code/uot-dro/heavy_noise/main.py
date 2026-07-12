import math
import os
import numpy as np
import urllib.request
import random
from tqdm import trange

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

def prepare_dataset(use_cuda=True):
    train_path_cuda = "../../data/train_set_cuda85.pt"
    test_path_cuda = "../../data/test_set.pt"

    if not os.path.exists(train_path_cuda):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        print("Downloading MNIST (train and val)")
        full_train_dataset = torchvision.datasets.MNIST(
            root='../../data/',
            train=True,
            download=True,
            transform=transform
        )

        filename = "../../data/feature-dependent_85_ytrain.npy"
        if not os.path.exists(filename):
            print("Downloading noisy labels...", end=' ')
            url = "https://github.com/gorkemalgan/corrupting_labels_with_distillation/raw/refs/heads/master/noisylabels/mnist/feature-dependent_85_ytrain.npy"
            urllib.request.urlretrieve(url, filename)
            print('Done.')

        y_train_noisy = np.load(filename)
        full_train_dataset.targets = y_train_noisy.tolist()

        train_size = 50000
        val_size = 10000
        torch.manual_seed(42)
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset,
            [train_size, val_size]
        )

        X_train = torch.stack([img for img, _ in train_dataset])
        y_train = torch.tensor([full_train_dataset.targets[i] for i in train_dataset.indices])

        if use_cuda:
            torch.save((X_train.to('cuda'), y_train.to('cuda')), train_path_cuda)
        else:
            torch.save((X_train, y_train), train_path_cuda)

        print(f"Training set saved to {train_path_cuda}")

    if not os.path.exists(test_path_cuda):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        print("Loading MNIST test set...")
        test_dataset = torchvision.datasets.MNIST(
            root='../../data/',
            train=False,
            download=True,
            transform=transform
        )

        # Convert test set to tensors
        X_test = torch.stack([img for img, _ in test_dataset])
        y_test = test_dataset.targets

        if use_cuda:
            torch.save((X_test.to('cuda'), y_test.to('cuda')), test_path_cuda)
        else:
            torch.save((X_test, y_test), test_path_cuda)

        print(f"Test set saved to {test_path_cuda}")


prepare_dataset()


class SimpleCNN(nn.Module):
    def __init__(self, with_alpha=False):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [B, 32, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 32, 14, 14]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B, 64, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 64, 7, 7]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        if with_alpha:
            self.alpha = nn.Parameter(torch.randn(1))

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def freeze_weights(self):
        for p in self.parameters():
            p.requires_grad_(False)
        return self

    def unfreeze_weights(self):
        for p in self.parameters():
            p.requires_grad_(True)
        return self


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


def train(lam_beta, lam, rho, lr, seed, batch_sz=64, n_epochs=20):
    torch.manual_seed(seed)
    random.seed(seed)

    X_train_cuda, y_train_cuda = torch.load("../../data/train_set_cuda85.pt")
    cuda_loader = DataLoader(TensorDataset(X_train_cuda, y_train_cuda),
                             batch_size=batch_sz, shuffle=True)

    model = SimpleCNN(with_alpha=rho is not None).cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    ce_sum_reduction = nn.CrossEntropyLoss(reduction='sum')
    ce_no_reduction = nn.CrossEntropyLoss(reduction='none')

    for epoch in trange(n_epochs):
        for i, (X, y) in enumerate(cuda_loader):
            loss, exponent = dro_loss(model, X, y, ce_sum_reduction, ce_no_reduction,
                                      lam_beta=lam_beta, lam=lam, rho=rho)
            if loss is None:
                print('Floating point exception')
                return None

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


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


def get_metrics(lam_beta, lam, rho, lr, seed, model):
    X_train_cuda, y_train_cuda = torch.load("../../data/train_set_cuda85.pt")
    cuda_loader = DataLoader(TensorDataset(X_train_cuda, y_train_cuda),
                             batch_size=5000, shuffle=False)
    ce_sum_reduction = nn.CrossEntropyLoss(reduction='sum')
    ce_no_reduction = nn.CrossEntropyLoss(reduction='none')

    X_test, y_test = torch.load("../../data/test_set.pt")
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=5000, shuffle=False)

    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []
    all_losses = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())

            loss_per_sample = F.cross_entropy(outputs, batch_y, reduction='none')
            all_losses.append(loss_per_sample.cpu())

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(batch_y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_losses = torch.cat(all_losses).numpy()
    all_probs = torch.cat(all_probs).numpy()

    n_classes = 10

    overall_loss = np.mean(all_losses)

    loss_per_class = []
    for c in range(n_classes):
        class_mask = (all_labels == c)
        if class_mask.sum() > 0:
            class_loss = np.mean(all_losses[class_mask])
        else:
            class_loss = 0.0
        loss_per_class.append(class_loss)
    worst_class_loss = np.max(loss_per_class)
    worst_class_for_loss = np.argmin(loss_per_class)

    overall_accuracy = accuracy_score(all_labels, all_preds)

    accuracy_per_class = []
    for c in range(n_classes):
        class_mask = (all_labels == c)
        if class_mask.sum() > 0:
            class_acc = accuracy_score(all_labels[class_mask], all_preds[class_mask])
        else:
            class_acc = 1.0
        accuracy_per_class.append(class_acc)
    worst_class_accuracy = np.min(accuracy_per_class)
    worst_class_for_accuracy = np.argmin(accuracy_per_class)

    overall_f1_macro = f1_score(all_labels, all_preds, average='macro')

    f1_per_class = f1_score(all_labels, all_preds, average=None)
    worst_class_f1 = np.min(f1_per_class)
    worst_class_for_f1 = np.argmin(f1_per_class)

    aucs = roc_auc_score(all_labels, all_probs, multi_class='ovr', average=None)
    macro_roc_auc = np.mean(aucs)

    worst_roc_auc = np.min(aucs)
    worst_class_for_auc = np.argmin(aucs)

    obj_val = eval_loss(model, cuda_loader, lam_beta, lam, ce_sum_reduction, ce_no_reduction)

    results = {
        'lam_beta': lam_beta, 'lam': lam, 'rho': rho, 'lr': lr, 'seed': seed,
        'obj_val': obj_val,
        'overall_loss': float(overall_loss),
        'loss_on_worst_class': float(worst_class_loss),
        'worst_class_for_loss': int(worst_class_for_loss),
        'overall_accuracy': float(overall_accuracy),
        'accuracy_on_worst_class': float(worst_class_accuracy),
        'worst_class_for_accuracy': int(worst_class_for_accuracy),
        'overall_f1_macro': float(overall_f1_macro),
        'f1_on_worst_class': float(worst_class_f1),
        'worst_class_for_f1': int(worst_class_for_f1),
        'macro_roc_auc': float(macro_roc_auc),
        'worst_roc_auc': float(worst_roc_auc),
        'worst_class_for_auc': int(worst_class_for_auc)
    }

    print(results)

    return results

def train_erm(lr, seed, batch_sz=64, n_epochs=20):  # batch sz 128 too large
    torch.manual_seed(seed)
    random.seed(seed)

    X_train_cuda, y_train_cuda = torch.load("../../data/train_set_cuda85.pt")
    cuda_loader = DataLoader(TensorDataset(X_train_cuda, y_train_cuda),
                             batch_size=batch_sz, shuffle=True)

    model = SimpleCNN(with_alpha=False).cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    ce_sum_reduction = nn.CrossEntropyLoss(reduction='sum')

    for epoch in trange(n_epochs):
        epoch_loss = 0.
        for i, (X, y) in enumerate(cuda_loader):
            preds = model(X)
            loss = ce_sum_reduction(preds, y)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("loss", epoch_loss / X_train_cuda.shape[0])

    return model

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
    (1., 5., 0.01, 1e-3),  # lr was 1e-2
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

for lam_beta, lam, rho, lr in param_grid:
    for seed in seeds:
        model = train(lam_beta, lam, rho, lr, seed)
        if model is not None:
            get_metrics(lam_beta, lam, rho, lr, seed, model)