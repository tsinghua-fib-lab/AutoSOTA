import os
import torch
import argparse
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree, to_networkx

import utils
from utils import *
from models import *


# ---------------------- EXPERIMENT FUNCTION ----------------------
def run(data, model_names, mechanisms):
    results = {}

    probs = [round(p, 2) for p in np.arange(0.0, 1.0, 0.1)]
    probs += [0.99]
    prob_pairs = [(p, p) for p in probs]

    data.features = data.x.clone()
    data.adj = data.adj.to(device)
    data_clone = data.clone()

    mechanism_map = {
        "UMCAR": {"mecha": "UMCAR", "opt": None},
        "SMCAR": {"mecha": "SMCAR", "opt": None},
        "LDMCAR": {"mecha": "LDMCAR", "opt": None},
        "FDMNAR": {"mecha": "FDMNAR", "opt": None},
        "CDMNAR": {"mecha": "CDMNAR", "opt": None},
    }
    for mech_name in mechanisms:
        if mech_name not in mechanism_map:
            raise ValueError(f"Mechanism {mech_name} not implemented")

        params = mechanism_map[mech_name]
        results[mech_name] = {}

        for p_train, p_test in prob_pairs:
            k = str((p_train, p_test))
            p_miss_dict = {"train": p_train, "test": p_test}

            # generate missingness once for this pair
            data_masked = produce_NA_ood(data_clone.clone(), p_miss_dict, mech=params, seeds=seeds)
            real_p = torch.isnan(data_masked.masks[seeds[0]]["X_incomp"]).float().mean().item() * 100

            for model in model_names:
                dm = data_masked.clone()

                if model == "gcnmi":
                    metrics = evaluate_gcn(dm, method=None, mod=None)
                elif model == "gnnmim":
                    dm_tmp = dm.clone()
                    for seed in dm.masks.keys():
                        miss_flag = (~dm.masks[seed]['mask']).double().to(device)
                        x_incomp = dm_tmp.masks[seed]['X_incomp'].to(device)
                        dm_tmp.masks[seed]['X_incomp'] = torch.cat([x_incomp, miss_flag], dim=1)
                        dm_tmp.masks[seed]['X_incomp'] = torch.nan_to_num(dm_tmp.masks[seed]['X_incomp'], nan=0.0).float()
                    dm_tmp.num_features = dm_tmp.num_features * 2
                    metrics = evaluate_gcn(dm_tmp, method=None, mod=model)
                elif model == "gcnmf":
                    metrics = evaluate_gcnmf(dm, max_epochs=1000, patience=40)
                elif model == "fp":
                    metrics = evaluate_fp(dm, hidden_channels=64, max_epochs=1000, patience=40,
                                          seeds=seeds)
                elif model == "pcfi":
                    metrics = evaluate_pcfi(dm, hidden_channels=64, max_epochs=1000, patience=40,
                                            seeds=seeds)
                elif model == "fairac":
                    metrics = evaluate_fairac(dm, hidden_channels=64, max_epochs=1000, patience=40, seeds=seeds)
                elif model == "gspn":
                    metrics = evaluate_gspn(dm, hidden_channels=64, max_epochs=1000, patience=300, seeds=seeds)
                elif model == "goodie":
                    metrics = evaluate_goodie(dm, max_epochs=1000, patience=40)
                else:
                    raise ValueError(f"Unknown model: {model}")

                acc, loss, f1 = metrics[0], metrics[1], metrics[2]
                print(f"[{mech_name}] model={model}, p_train={p_train}, p_test={p_test}, "
                      f"real_missing={real_p:.2f}%, f1={f1:.3f}, ")

                results[mech_name].setdefault(model, {})[k] = {
                    "f1": f1,
                }

    return results


# ---------------------- DATASET LOADING ----------------------
def load_dataset(name, device):
    """Load dataset by name. All datasets are stored/loaded from ./data/."""

    base_dir = os.path.join(os.getcwd(), "data")        

    if name == "synthetic":
        data = torch.load(os.path.join(base_dir, "synthetic.pt"))

    elif name in ["Cora", "Citeseer", "PubMed"]:
        folder_name = os.path.join(base_dir, name)
        transform = T.NormalizeFeatures()
        dataset = Planetoid(root=folder_name, name=name, transform=transform)
        data = dataset[0]
        data.num_classes = len(torch.unique(data.y))

    elif name == "electric":
        data = torch.load(os.path.join(base_dir, "electric.pt"))
        deg = degree(data.edge_index[0], data.num_nodes).view(-1, 1)
        G = to_networkx(data, to_undirected=True)
        clust = torch.tensor([nx.clustering(G, i) for i in range(data.num_nodes)]).view(-1, 1)
        data.x = torch.cat([data.x, deg, clust], dim=1)
        data.num_features = data.x.size(1)

    elif name == "air":
        data = torch.load(os.path.join(base_dir, "air.pt"))
        data.num_features = data.x.size(1)

    elif name == "tadpole":
        data = torch.load(os.path.join(base_dir, "tadpole.pt"))
        data.num_features = data.x.size(1)
        data.num_classes = 3

    else:
        raise ValueError(f"Dataset {name} not supported.")

    # add adjacency matrix
    indices = data.edge_index
    values = torch.ones(indices.size(1))
    adj = torch.sparse_coo_tensor(
        indices.to(device), values.to(device), (data.num_nodes, data.num_nodes)
    )
    data.adj = adj

    return data



# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GNN missing data experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices= ["synthetic", "Cora", "Citeseer", "PubMed", "electric", "air", "tadpole"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["fairac"],
        help="List of models to evaluate"
    )
    parser.add_argument(
        "--mechanisms",
        type=str,
        nargs="+",
        default=["UMCAR"],
        choices=["UMCAR", "SMCAR", "LDMCAR", "FDMNAR", "CDMNAR"],
        help="Missingness mechanisms to use"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset(args.dataset, device).to(device)

    print(f"\nDataset loaded: {args.dataset}, num_nodes={data.num_nodes}, num_features={data.num_features}")

    # Run experiments (just prints results)
    _ = run(data, args.models, args.mechanisms)



# # Synthetic dataset, FairAC model, UMCAR mechanism
# python main.py --dataset synthetic --models fairac --mechanisms UMCAR

# # PubMed with multiple models and SMCAR + CDMNAR
# python main.py --dataset PubMed --models fairac gcnmi gcnmf --mechanisms SMCAR CDMNAR
