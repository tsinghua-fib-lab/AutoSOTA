"""
Copyright 2023 Daeho Um
SPDX-License-Identifier: Apache-2.0
"""
import torch

def train(model, x, train_pos_edge_index, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    loss.backward()
    optimizer.step()
    return float(loss)


def test(model, x, train_pos_edge_index, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)