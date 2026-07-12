"""
version 1.0
date 2021/02/04
"""

import argparse
import random

import numpy as np
import optuna
import torch

from numpy import mean
from tqdm import tqdm
import json

from models import GCNmf
from train import NodeClsTrainer
from utils import NodeClsData, apply_mask, generate_mask

torch.set_num_threads(4) 

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    default='cora',
                    choices=['cora', 'citeseer', 'amacomp', 'amaphoto', 'synthetic'],
                    help='dataset name')
parser.add_argument('--type',
                    default='uniform',
                    choices=['uniform', 'bias', 'struct'],
                    help="uniform randomly missing, biased randomly missing, and structurally missing")
parser.add_argument('--rate', default=0.0, type=float, help='missing rate')
parser.add_argument('--nhid', default=16, type=int, help='the number of hidden units')
parser.add_argument('--ncomp', default=5, type=int, help='the number of Gaussian components')
parser.add_argument('--epoch', default=1000, type=int, help='the number of training epoch')
parser.add_argument('--patience', default=100, type=int, help='patience for early stopping')
parser.add_argument('--seed', default=17, type=int)

args = parser.parse_args()
TRIAL_SIZE = 100
TIMEOUT = 60 * 60 * 3

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

print(args.dataset, args.type, args.rate)
print("num of components:", args.ncomp)
print("nhid:", args.nhid)
print("epochs:", args.epoch)
print("patience:", args.patience)

# generate all masks for the experiment
d = torch.load('/home/ferrini/phd/gnn-missing-data/synthetic/graph.pt')
# tmpdata = NodeClsData(args.dataset, data=d)
# masks = [generate_mask(tmpdata.features, args.rate, args.type) for _ in range(5)]

def objective(trial):
    # Tune hyperparameters (dropout, weight decay, learning rate) using Optuna
    dropout = trial.suggest_uniform('dropout', 0.4, 0.8)
    lr = trial.suggest_loguniform('lr', 5e-4, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-1)

    # prepare data and model
    
    data = NodeClsData(args.dataset, data=d)
    apply_mask(data.features, masks[0])
    model = GCNmf(data, args.nhid, dropout, args.ncomp)

    # run model
    params = {
        'lr': lr,
        'weight_decay': weight_decay,
        'epochs': args.epoch,
        'patience': args.patience,
        'early_stopping': True
    }
    trainer = NodeClsTrainer(data, model, params, niter=10)
    result = trainer.run()
    return - result['val_acc']


def tune_hyperparams():
    study = optuna.create_study()
    study.optimize(objective, n_trials=TRIAL_SIZE, timeout=TIMEOUT)
    return study.best_params


def evaluate_model(hyperparams):
    means = []
    dropout = hyperparams['dropout']
    
    results_all = {}
    for mech_name in ['MAR', 'MCAR', 'MNAR_logistic', 'MNAR_selfmasked']:
        results_all[mech_name] = {}
        for key, mask in d.masks_gcnmf[mech_name].items():
            print("%missing: ", key)
            print(mask)
        # for mask in tqdm(masks):
            # generate missing data, model and trainer
            data = NodeClsData(args.dataset, data=d)
            apply_mask(data.features, mask)  # convert masked number to nan
            model = GCNmf(data, args.nhid, dropout, args.ncomp)
            params = {
                'lr': hyperparams['lr'],
                'weight_decay': hyperparams['weight_decay'],
                'epochs': args.epoch,
                'patience': args.patience,
                'early_stopping': True
            }
            
            trainer = NodeClsTrainer(data, model, params, niter=20)
            
            # run the model
            result = trainer.run()
            results_all[mech_name][key] = result['test_acc']
            means.append(result['test_acc'])
    with open("/home/ferrini/phd/gnn-missing-data/synthetic/gcnmf.json", "w", encoding="utf-8") as f:
        json.dump(results_all, f, ensure_ascii=False, indent=4)
    exit()
    return mean(means)


def main():
    # hyper_params = tune_hyperparams()
    hyper_params = {'lr': 0.01, 'weight_decay': 0.0005, 'dropout': 0.5}
    result = evaluate_model(hyper_params)
    print(result)


if __name__ == '__main__':
    main()
