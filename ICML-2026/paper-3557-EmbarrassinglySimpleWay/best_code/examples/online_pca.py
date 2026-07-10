# Example adapted from https://github.com/andyjm3/RSDM to use the POGO optimizer.
# Software under the Apache-2.0 license.

import random
from time import time
from datetime import datetime

import torch
import numpy as np
from torch import nn, optim, linalg

from pogo import base, POGO


def pca_loss(X):
    return -torch.trace(X.transpose(-1,-2) @ A @ X)/(2)

def pca_optgap(X):
    return abs(pca_loss(X).item() - loss_star)/abs(loss_star)


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = int(datetime.now().timestamp())
    
    print('seed', seed)
    print(device)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    n = 2000
    p = 1500
    lr = 0.25
    n_epochs = 3000

    CN = 1000
    D = 10 * torch.diag(torch.logspace(-np.log10(CN), 0, n))
    [Q, R] = linalg.qr(torch.randn(n,n))
    A = Q @ D @ Q
    A = (A + A.t())/2

    A = A.to(device)
    init_weights = linalg.qr(torch.randn(n, p))[0]

    # Compute closed-form solution from svd, used for monitoring.
    [_, w_star] = torch.linalg.eigh(A/(2))
    w_star = w_star[:,-p:]
    loss_star = pca_loss(w_star)
    loss_star = loss_star.item()

    W = nn.Parameter(torch.empty(n, p))
    W.data = init_weights.clone().to(device)
    flatten_fn = lambda x: x.unsqueeze(0)

    optimizer = POGO([W], base.SGD(momentum=0.3), lr, flatten_fn=flatten_fn, rows=False)

    # init
    loss = pca_loss(W.data).cpu().item()
    dist2opt = pca_optgap(W.data)
    losses = [loss]
    time_epochs = [0]
    optgap_epochs = [dist2opt]
    print(
        "|POGO| epoch: %.1e sec, Loss: %.2e, optgap: %.2e"
        % (time_epochs[-1], loss, dist2opt)
    )

    for epoch in range(n_epochs):
        t0 = time()

        optimizer.zero_grad()
        loss = pca_loss(W)
        loss.backward()
        optimizer.step()

        loss = pca_loss(W.data).cpu().item()
        dist2opt = pca_optgap(W.data)
        time_epochs.append(time() - t0)
        losses.append(loss)
        optgap_epochs.append(dist2opt)
        print(
            "|POGO| epoch %i : %.1e sec, loss: %.2e, optgap: %.2e"
            % (epoch, time_epochs[-1], loss, dist2opt)
        )

        if dist2opt < 1e-6:
            print(f"Tolerance reached. Break at iter {epoch}!")
            break

    print(torch.cumsum(torch.tensor(time_epochs), dim=0)[-1])


