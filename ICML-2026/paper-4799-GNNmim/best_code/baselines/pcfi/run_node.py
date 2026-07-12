"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
Modified by Daeho Um (daehoum1@snu.ac.kr)
"""
import numpy as np
import argparse
import torch
from data_loading import get_dataset
from data_utils import set_train_val_test_split
from utils import get_missing_feature_mask
from models import get_model
from seeds import seeds
from evaluation import test
from train import train
import random
from pcfi import pcfi


parser = argparse.ArgumentParser("Setting for graphs with partially known features")
parser.add_argument(
    "--dataset_name",
    type=str,
    help="Name of dataset",
    default="Cora",
    choices=[
        "Cora",
        "CiteSeer",
        "PubMed",
        "OGBN-Arxiv",
        "Photo",
        "Computers",
    ],
)
parser.add_argument(
    "--mask_type", type=str, help="Type of missing feature mask", default="structural", choices=["uniform", "structural"],
)
parser.add_argument("--missing_rate", type=float, help="Rate of node features missing", default=0.5)
parser.add_argument(
    "--num_iterations", type=int, help="Number of diffusion iterations for feature reconstruction", default=100,
)
parser.add_argument("--alpha", type=float, default=0.9)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument(
    "--model",
    type=str,
    help="Type of model to make a prediction on the downstream task",
    default="gcn",
    choices=["gcn", "gat"],
)
parser.add_argument("--patience", type=int, help="Patience for early stopping", default=200)
parser.add_argument("--lr", type=float, help="Learning Rate", default=0.005)
parser.add_argument("--epochs", type=int, help="Max number of epochs", default=10000)
parser.add_argument("--n_runs", type=int, help="Max number of runs", default=10)
parser.add_argument("--hidden_dim", type=int, help="Hidden dimension of model", default=64)
parser.add_argument("--num_layers", type=int, help="Number of GNN layers", default=3)
parser.add_argument("--dropout", type=float, help="Feature dropout", default=0.5)
parser.add_argument("--jk", action="store_true", help="Whether to use the jumping knowledge scheme")
parser.add_argument("--gpu_idx", type=int, help="Indexes of gpu to run program on", default=0)

def run(args):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)

    device = torch.device(
        f"cuda:{args.gpu_idx}"
        if torch.cuda.is_available() else "cpu"
    )

    dataset, evaluator = get_dataset(name=args.dataset_name)
    split_idx = dataset.get_idx_split() if hasattr(dataset, "get_idx_split") else None
    n_nodes, n_features = dataset.data.x.shape
    test_accs, best_val_accs, train_times = [], [], []

    train_loader = None
    i = 0
    for seed in seeds[: args.n_runs]:
        num_classes = dataset.num_classes
        data = set_train_val_test_split(
            seed=seed, data=dataset.data, split_idx=split_idx, dataset_name=args.dataset_name,
        ).to(device)
        print(data)
        missing_feature_mask = get_missing_feature_mask(
            rate=args.missing_rate, n_nodes=n_nodes, n_features=n_features, seed = seed, type=args.mask_type,
        ).to(device)
        x = data.x.clone()
        print("x prima: ", x)
        print(missing_feature_mask)
        x[~missing_feature_mask] = float("nan")
        print("x dopo: ", x)
        filled_features = pcfi(data.edge_index, x, missing_feature_mask, args.num_iterations, args.mask_type, args.alpha, args.beta)
        print(filled_features)
        i += 1
        feature_num = data.num_features
        model = get_model(
            model_name=args.model,
            num_features=feature_num,
            num_classes=num_classes,
            edge_index=data.edge_index,
            x=x,
            mask=missing_feature_mask,
            args=args,
        ).to(device)
        params = list(model.parameters())
        print(model)
        exit()
        optimizer = torch.optim.Adam(params, lr=args.lr)
        criterion = torch.nn.NLLLoss()
        test_acc = 0
        val_accs = []
        for epoch in range(0, args.epochs):
            x = filled_features
            train(
                model, x, data, optimizer, criterion, train_loader=train_loader, device=device,
            )
            (train_acc, val_acc, tmp_test_acc), out = test(
                model, x=x, data=data, evaluator=evaluator,  device=device,
            )
            if epoch == 0 or val_acc > max(val_accs):
                test_acc = tmp_test_acc
            val_accs.append(val_acc)
            if epoch > args.patience and max(val_accs[-args.patience :]) <= max(val_accs[: -args.patience]):
                break
        best_val_accs.append(val_acc)
        test_accs.append(test_acc)
    test_acc_mean, test_acc_std = np.mean(test_accs), np.std(test_accs)
    print(f"Test Accuracy: {test_acc_mean * 100:.2f}% +- {test_acc_std * 100:.2f}")


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    run(args)