"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
"""
import numpy as np
from tqdm import tqdm
import argparse
import logging
import time
import yaml

import torch
from torch_geometric.data import NeighborSampler

from data_loading import get_dataset
from data_utils import set_train_val_test_split
from utils import get_missing_feature_mask
from models import get_model
from seeds import seeds
from filling_strategies import filling
from evaluation import test
from train import train

parser = argparse.ArgumentParser("GNN-Missing-Features")
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
        "OGBN-Products",
        "MixHopSynthetic",
    ],
)
parser.add_argument(
    "--mask_type", type=str, help="Type of missing feature mask", default="uniform", choices=["uniform", "structural"],
)
parser.add_argument(
    "--filling_method",
    type=str,
    help="Method to solve the missing feature problem",
    default="feature_propagation",
    choices=["random", "zero", "mean", "neighborhood_mean", "feature_propagation",],
)
parser.add_argument(
    "--model",
    type=str,
    help="Type of model to make a prediction on the downstream task",
    default="gcn",
    choices=["mlp", "sgc", "sage", "gcn", "gat", "gcnmf", "pagnn", "lp"],
)
parser.add_argument("--missing_rate", type=float, help="Rate of node features missing", default=0.99)
parser.add_argument("--patience", type=int, help="Patience for early stopping", default=200)
parser.add_argument("--lr", type=float, help="Learning Rate", default=0.005)
parser.add_argument("--epochs", type=int, help="Max number of epochs", default=10000)
parser.add_argument("--n_runs", type=int, help="Max number of runs", default=5)
parser.add_argument("--hidden_dim", type=int, help="Hidden dimension of model", default=64)
parser.add_argument("--num_layers", type=int, help="Number of GNN layers", default=2)
parser.add_argument(
    "--num_iterations", type=int, help="Number of diffusion iterations for feature reconstruction", default=40,
)
parser.add_argument("--lp_alpha", type=float, help="Alpha parameter of label propagation", default=0.9)
parser.add_argument("--dropout", type=float, help="Feature dropout", default=0.5)
parser.add_argument("--jk", action="store_true", help="Whether to use the jumping knowledge scheme")
parser.add_argument(
    "--batch_size", type=int, help="Batch size for models trained with neighborhood sampling", default=1024,
)
parser.add_argument(
    "--graph_sampling",
    help="Set if you want to use graph sampling (always true for large graphs)",
    action="store_true",
)
parser.add_argument(
    "--homophily", type=float, help="Level of homophily for synthetic datasets", default=None,
)
parser.add_argument("--gpu_idx", type=int, help="Indexes of gpu to run program on", default=0)
parser.add_argument(
    "--log", type=str, help="Log Level", default="INFO", choices=["DEBUG", "INFO", "WARNING"],
)


def run(args):
    logger.info(args)

    assert not (
        args.graph_sampling and args.model != "sage"
    ), f"{args.model} model does not support training with neighborhood sampling"
    assert not (args.graph_sampling and args.jk), "Jumping Knowledge is not supported with neighborhood sampling"

    device = torch.device(
        f"cuda:{args.gpu_idx}"
        if torch.cuda.is_available() and not (args.dataset_name == "OGBN-Products" and args.model == "lp")
        else "cpu"
    )
    device = torch.device('cpu')
    dataset, evaluator = get_dataset(name=args.dataset_name, homophily=args.homophily)
    data = dataset.data

    split_idx = dataset.get_idx_split() if hasattr(dataset, "get_idx_split") else None
    n_nodes, n_features = dataset.data.x.shape
    test_accs, best_val_accs, train_times = [], [], []

    train_loader = (
        NeighborSampler(
            dataset.data.edge_index,
            node_idx=split_idx["train"],
            sizes=[15, 10, 5][: args.num_layers],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=12,
        )
        if args.graph_sampling
        else None
    )
    # Setting `sizes` to -1 simply loads all the neighbors for each node. We can do this while evaluating
    # as we first compute the representation of all nodes after the first layer (in batches), then for the second layer, and so on
    inference_loader = (
        NeighborSampler(
            dataset.data.edge_index, node_idx=None, sizes=[-1], batch_size=4096, shuffle=False, num_workers=12,
        )
        if args.graph_sampling
        else None
    )

    for seed in tqdm(seeds[: args.n_runs]):
        num_classes = dataset.num_classes
        data = set_train_val_test_split(
            seed=seed, data=dataset.data, split_idx=split_idx, dataset_name=args.dataset_name,
        ).to(device)
        train_start = time.time()
        if args.model == "lp":
            model = get_model(
                model_name=args.model,
                num_features=data.num_features,
                num_classes=num_classes,
                edge_index=data.edge_index,
                x=None,
                args=args,
            ).to(device)
    
            logger.info("Starting Label Propagation")
            logits = model(y=data.y, edge_index=data.edge_index, mask=data.train_mask)
            (_, val_acc, test_acc), _ = test(model=None, x=None, data=data, logits=logits, evaluator=evaluator)
        else:
            missing_feature_mask = get_missing_feature_mask(
                rate=args.missing_rate, n_nodes=n_nodes, n_features=n_features, type=args.mask_type,
            ).to(device)
            x = data.x.clone()
            print("x ha nan? ", torch.isnan(x).any())
            x[~missing_feature_mask] = float("nan")
            print("missing mask ", missing_feature_mask)
            logger.debug("Starting feature filling")
            start = time.time()
            filled_features = (
                filling(args.filling_method, data.edge_index, x, missing_feature_mask, args.num_iterations,)
                if args.model not in ["gcnmf", "pagnn"]
                else torch.full_like(x, float("nan"))
            )
            print(filled_features)
            logger.debug(f"Feature filling completed. It took: {time.time() - start:.2f}s")

            model = get_model(
                model_name=args.model,
                num_features=data.num_features,
                num_classes=num_classes,
                edge_index=data.edge_index,
                x=x,
                mask=missing_feature_mask,
                args=args,
            ).to(device)
            params = list(model.parameters())

            optimizer = torch.optim.Adam(params, lr=args.lr)
            critereon = torch.nn.NLLLoss()

            test_acc = 0
            val_accs = []
            print(model)
    
            for epoch in range(0, args.epochs):
                start = time.time()
                x = torch.where(missing_feature_mask, data.x, filled_features)
                print("x ha nan? ", torch.isnan(x).any())
                exit()

                train(
                    model, x, data, optimizer, critereon, train_loader=train_loader, device=device,
                )
                (train_acc, val_acc, tmp_test_acc), out = test(
                    model, x=x, data=data, evaluator=evaluator, inference_loader=inference_loader, device=device,
                )
                if epoch == 0 or val_acc > max(val_accs):
                    test_acc = tmp_test_acc
                    y_soft = out.softmax(dim=-1)

                val_accs.append(val_acc)
                if epoch > args.patience and max(val_accs[-args.patience :]) <= max(val_accs[: -args.patience]):
                    break
                logger.debug(
                    f"Epoch {epoch + 1} - Train acc: {train_acc:.3f}, Val acc: {val_acc:.3f}, Test acc: {tmp_test_acc:.3f}. It took {time.time() - start:.2f}s"
                )

            (_, val_acc, test_acc), _ = test(model, x=x, data=data, logits=y_soft, evaluator=evaluator)
        best_val_accs.append(val_acc)
        test_accs.append(test_acc)
        train_times.append(time.time() - train_start)

    test_acc_mean, test_acc_std = np.mean(test_accs), np.std(test_accs)
    print(f"Test Accuracy: {test_acc_mean * 100:.2f}% +- {test_acc_std * 100:.2f}")


if __name__ == "__main__":
    args = parser.parse_args()
    hyperparam_file = '/home/ferrini/phd/gnn-missing-data/baselines/feature-propagation/hyperparameters.yaml'
    with open(hyperparam_file, "r") as f:
        hyperparams = yaml.safe_load(f)
        dataset = args.dataset_name
        if dataset in hyperparams:
            for k, v in hyperparams[dataset].items():
                setattr(args, k, v)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=getattr(logging, args.log.upper(), None))

    run(args)
