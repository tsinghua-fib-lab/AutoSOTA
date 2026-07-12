"""
version 1.0
date 2021/02/04
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from numpy import mean, std
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim import Adam
from sklearn.metrics import f1_score
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EarlyStopping:
    def __init__(self, patience, verbose):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0
        self.state_dict = None

    def reset(self):
        self.counter = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0
        self.state_dict = None

    def check(self, evals, model, epoch):
        if evals['val_loss'] <= self.best_val_loss or evals['val_acc'] >= self.best_val_acc:
            if evals['val_loss'] <= self.best_val_loss and evals['val_acc'] >= self.best_val_acc:
                self.state_dict = deepcopy(model.state_dict())
            self.best_val_loss = min(self.best_val_loss, evals['val_loss'])
            self.best_val_acc = max(self.best_val_acc, evals['val_acc'])
            self.counter = 0
        else:
            self.counter += 1
        stop = False
        if self.counter >= self.patience:
            stop = True
            if self.verbose:
                print("Stop training, epoch:", epoch)
            model.load_state_dict(self.state_dict)
        return stop


class NodeClsTrainer:
    def __init__(self, data, model, params, niter=100, verbose=False):
        self.data = data
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        self.lr = params['lr']
        self.weight_decay = params['weight_decay']
        self.epochs = params['epochs']
        self.niter = niter
        self.verbose = verbose
        self.early_stopping = params['early_stopping']
        if self.early_stopping:
            self.stop_checker = EarlyStopping(params['patience'], verbose)

        self.data.to(device)

    def reset(self):
        self.model.to(device).reset_parameters()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.early_stopping:
            self.stop_checker.reset()

    def train(self):
        data, model, optimizer = self.data, self.model, self.optimizer
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output[data.train_mask], data.labels[data.train_mask])
        loss.backward()
        optimizer.step()

    def evaluate(self):
        data, model = self.data, self.model
        model.eval()

        with torch.no_grad():
            output = model(data)

        outputs = {}
        for key in ['train', 'val', 'test']:
            if key == 'train':
                mask = data.train_mask
            elif key == 'val':
                mask = data.val_mask
            else:
                mask = data.test_mask
            loss = F.nll_loss(output[mask], data.labels[mask]).item()
            pred = output[mask].max(dim=1)[1]

            acc = f1_score(
                y_true = data.labels[mask].cpu().numpy(),
                y_pred = pred.cpu().numpy(),
                average = 'macro'
            )
            # acc = pred.eq(data.labels[mask]).sum().item() / mask.sum().item()

            outputs['{}_loss'.format(key)] = loss
            outputs['{}_acc'.format(key)] = acc

        return outputs

    def print_verbose(self, epoch, evals):
        print('epoch: {: 5d}'.format(epoch),
              'train loss: {:.5f}'.format(evals['train_loss']),
              'train acc: {:.5f}'.format(evals['train_acc']),
              'val loss: {:.5f}'.format(evals['val_loss']),
              'val acc: {:.5f}'.format(evals['val_acc']))

    def run(self):
        val_acc_list = []
        test_acc_list = []

        for _ in tqdm(range(self.niter)):
            self.reset()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            for epoch in range(1, self.epochs + 1):
                self.train()
                evals = self.evaluate()

                if self.verbose:
                    self.print_verbose(epoch, evals)

                if self.early_stopping:
                    if self.stop_checker.check(evals, self.model, epoch):
                        break

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            evals = self.evaluate()
            if self.verbose:
                for met, val in evals.items():
                    print(met, val)

            val_acc_list.append(evals['val_acc'])
            test_acc_list.append(evals['test_acc'])

        print(mean(test_acc_list))
        print(std(test_acc_list))
        return {
            'val_acc': mean(val_acc_list),
            'test_acc': mean(test_acc_list),
            'test_acc_std': std(test_acc_list)
        }


class LinkPredTrainer:
    def __init__(self, data, model, params, niter=100, verbose=False):
        self.model = model
        self.data = data
        self.optimizer = Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        self.lr = params['lr']
        self.weight_decay = params['weight_decay']
        self.epochs = params['epochs']
        self.niter = niter
        self.verbose = verbose

        self.data.to(device)

    def reset(self):
        self.model.to(device).reset_parameters()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train(self):
        data, model, optimizer = self.data, self.model, self.optimizer
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = linkpred_loss(data, output)
        loss.backward()
        optimizer.step()

    def evaluate(self):
        data, model = self.data, self.model
        model.eval()

        with torch.no_grad():
            output = model(data)

        loss = linkpred_loss(data, output)
        val_auc, val_ap = linkpred_score(output["z"], data.val_edges, data.neg_val_edges)
        test_auc, test_ap = linkpred_score(output["z"], data.val_edges, data.neg_val_edges)
        return {
            'train_loss': loss,
            'val_auc': val_auc,
            'val_ap': val_ap,
            'test_auc': test_auc,
            'test_ap': test_ap
        }

    def print_verbose(self, epoch, evals):
        print("Epoch: {:4d}".format(epoch + 1),
              "Train loss: {:.5f}".format(evals['train_loss']),
              "Val AUC: {:.5f}".format(evals['val_auc']),
              "Val AP: {:.5f}".format(evals['val_ap']))

    def run(self):
        val_auc_list = []
        val_ap_list = []
        test_auc_list = []
        test_ap_list = []

        for _ in tqdm(range(self.niter)):
            self.reset()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            for epoch in range(self.epochs):
                self.train()
                evals = self.evaluate()

                if self.verbose:
                    self.print_verbose(epoch, evals)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            evals = self.evaluate()
            val_auc_list.append(evals['val_auc'])
            val_ap_list.append(evals['val_ap'])
            test_auc_list.append(evals['test_auc'])
            test_ap_list.append(evals['test_ap'])

        return {
            "val_auc": mean(val_auc_list),
            "val_ap": mean(val_ap_list),
            "test_auc": mean(test_auc_list),
            "test_auc_std": std(test_auc_list),
            "test_ap": mean(test_ap_list),
            "test_ap_std": std(test_ap_list),
        }


def reconstruction_loss(data, output):
    adj_recon = output['adj_recon']
    return data.norm * F.binary_cross_entropy_with_logits(adj_recon, data.adjmat, pos_weight=data.pos_weight)


def linkpred_loss(data, output):
    recon_loss = reconstruction_loss(data, output)
    mu, logvar = output['mu'], output['logvar']
    kl = - 1 / (2 * data.num_nodes) * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return recon_loss + kl


def linkpred_score(z, pos_edges, neg_edges):
    pos_score = torch.sigmoid(torch.sum(z[pos_edges[0]] * z[pos_edges[1]], dim=1))
    neg_score = torch.sigmoid(torch.sum(z[neg_edges[0]] * z[neg_edges[1]], dim=1))
    pred_score = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    true_score = np.hstack([np.ones(pos_score.size(0)), np.zeros(neg_score.size(0))])
    auc_score = roc_auc_score(true_score, pred_score)
    ap_score = average_precision_score(true_score, pred_score)
    return auc_score, ap_score
