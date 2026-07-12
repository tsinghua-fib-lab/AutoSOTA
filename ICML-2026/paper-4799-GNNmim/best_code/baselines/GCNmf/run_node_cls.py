"""
version 1.0
date 2021/02/04
"""

import argparse

from models import GCNmf
from train import NodeClsTrainer
from utils import NodeClsData, apply_mask, generate_mask
import scipy.sparse as sp
import torch
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    default='cora',
                    choices=['cora', 'citeseer', 'amacomp', 'amaphoto', 'synthetic'],
                    help='dataset name')
parser.add_argument('--type',
                    default='uniform',
                    choices=['uniform', 'bias', 'struct', 'MAR'],
                    help="uniform randomly missing, biased randomly missing, and structurally missing")
parser.add_argument('--rate', default=0.4, type=float, help='missing rate')
parser.add_argument('--nhid', default=32, type=int, help='the number of hidden units')
parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
parser.add_argument('--ncomp', default=10, type=int, help='the number of Gaussian components')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-2, type=float, help='weight decay')
parser.add_argument('--epoch', default=10000, type=int, help='the number of training epoch')
parser.add_argument('--patience', default=100, type=int, help='patience for early stopping')
parser.add_argument('--verbose', action='store_true', help='verbose')
parser.add_argument('--miss_type', type=str, default='mcar')
parser.add_argument('--miss_percentage', type=int, default=10)
parser.add_argument('--influential_features', type=int, default=1)
parser.add_argument('--dependancy', type=float, default=0.3)

args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'synthetic':
        data = torch.load('/home/ferrini/phd/gnn-missing-data/synthetic/pa_graph.pt')
        data = NodeClsData(args.dataset, data=data)
    else:
        data = NodeClsData(args.dataset)
    mask = generate_mask(data.features, args.rate, args.type)
    print("Percentage of newly generated missing values: ", (mask.sum()).numpy()/np.prod(mask.size())*100, " %")
    apply_mask(data.features, mask)
    print("Percentage of newly generated missing values: ", (mask.sum()).numpy()/np.prod(mask.size())*100, " %")
    print(data)

    model = GCNmf(data, nhid=args.nhid, dropout=args.dropout, n_components=args.ncomp)
    params = {
        'lr': args.lr,
        'weight_decay': args.wd,
        'epochs': args.epoch,
        'patience': args.patience,
        'early_stopping': True
    }
    trainer = NodeClsTrainer(data, model, params, niter=2, verbose=args.verbose)
    trainer.run()
