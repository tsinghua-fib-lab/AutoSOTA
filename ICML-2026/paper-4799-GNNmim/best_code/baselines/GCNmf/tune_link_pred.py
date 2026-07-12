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

from models import VGAEmf
from train import LinkPredTrainer
from utils import LinkPredData, apply_mask, generate_mask


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    default='cora',
                    choices=['cora', 'citeseer'],
                    help='dataset name')
parser.add_argument('--type',
                    default='uniform',
                    choices=['uniform', 'bias', 'struct'],
                    help="uniform randomly missing, biased randomly missing, and structurally missing")
parser.add_argument('--rate', default=0.1, type=float, help='missing rate')
parser.add_argument('--nhid', default=32, type=int, help='the number of hidden units')
parser.add_argument('--latent_dim', default=16, type=int, help='the dimension of latent variables')
parser.add_argument('--ncomp', default=5, type=int, help='the number of Gaussian components')
parser.add_argument('--epoch', default=1000, type=int, help='the number of training epochs')
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

# generate all masks for the experiment
tmpdata = LinkPredData(args.dataset)
masks = [generate_mask(tmpdata.features, args.rate, args.type) for _ in range(5)]


def objective(trial):
    # Tune hyperparameters (dropout, weight decay, learning rate) using Optuna
    dropout = trial.suggest_uniform('dropout', 0., 0.1)
    lr = trial.suggest_loguniform('lr', 5e-4, 2e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)

    # prepare data and model
    data = LinkPredData(args.dataset, seed=args.seed)
    apply_mask(data.features, masks[0])
    model = VGAEmf(data, args.nhid, args.latent_dim, dropout, args.ncomp)

    # run model
    params = {
        'lr': lr,
        'weight_decay': weight_decay,
        'epochs': args.epoch,
    }
    trainer = LinkPredTrainer(data, model, params, niter=10)
    result = trainer.run()
    return - result['val_auc']


def tune_hyperparams():
    study = optuna.create_study()
    study.optimize(objective, n_trials=TRIAL_SIZE, timeout=TIMEOUT)
    return study.best_params


def evaluate_model(hyperparams):
    means = []
    for mask in tqdm(masks):
        # generate missing data, model and trainer
        data = LinkPredData(args.dataset, seed=args.seed)
        apply_mask(data.features, mask)  # convert masked number to nan
        model = VGAEmf(data, args.nhid, args.latent_dim, hyperparams['dropout'], args.ncomp)
        params = {
            'lr': hyperparams['lr'],
            'weight_decay': hyperparams['weight_decay'],
            'epochs': args.epoch,
        }
        trainer = LinkPredTrainer(data, model, params, niter=20)

        # run the model
        result = trainer.run()
        means.append(result['test_auc'])

    return mean(means)


def main():
    hyper_params = tune_hyperparams()
    result = evaluate_model(hyper_params)
    print(result)


if __name__ == '__main__':
    main()
