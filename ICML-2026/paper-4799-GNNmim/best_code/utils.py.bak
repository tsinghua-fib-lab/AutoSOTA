import os
import sys
import copy
import warnings
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import ks_2samp
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, NeighborLoader
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import (
    from_networkx,
    to_scipy_sparse_matrix,
    to_undirected,
    subgraph,
    k_hop_subgraph,
)
sys.path.append(os.path.join(os.path.dirname(__file__), "baselines", "graph-sum-product-networks"))





from gensim.models import Word2Vec
from tqdm import tqdm

from models import *
from GOODIE import *
from model import GSPN
from readout import ProbabilisticGraphReadoutNoLayerAttentionMLP

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric.typing")
warnings.filterwarnings("ignore")




seeds=[1, 43, 15, 118, 222]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fill_nan_with_col_mean_split(X, train_val_mask, test_mask):
    """
    Fill NaNs using column means from train+val for both train+val and test rows.
    
    Parameters
    ----------
    X : torch.Tensor
        Input 2D tensor (n_nodes x n_features).
    train_val_mask : torch.BoolTensor
        Mask indicating the rows belonging to train or val.
    test_mask : torch.BoolTensor
        Mask indicating the rows belonging to test.

    Returns
    -------
    X_filled : torch.Tensor
        A new tensor where NaNs are filled using column means computed from train+val only.
    """
    X_filled = X.clone()
    nan_mask = torch.isnan(X_filled)

    # Compute column means from train+val only
    col_means = []
    for j in range(X.shape[1]):
        col = X[train_val_mask, j]
        col_no_nan = col[~torch.isnan(col)]
        if len(col_no_nan) > 0:
            col_mean = col_no_nan.mean()
        else:
            col_mean = torch.tensor(0.0, device=X.device)
        col_means.append(col_mean)
    
    # Fill NaNs in both train+val and test using those means
    for j in range(X.shape[1]):
        X_filled[nan_mask[:, j], j] = col_means[j]

    return X_filled

def generate_deepwalk_embeddings(edge_index, num_nodes, embedding_dim=16, walk_length=10, num_walks=10, window_size=5):
    G_nx = nx.Graph()
    G_nx.add_nodes_from(range(num_nodes))
    edges = edge_index.t().tolist() 
    G_nx.add_edges_from(edges)

    walks = []
    for node in G_nx.nodes():
        for _ in range(num_walks):
            walk = [str(node)]
            current = node
            for _ in range(walk_length - 1):
                neighbors = list(G_nx.neighbors(current))
                if neighbors:
                    current = np.random.choice(neighbors)
                    walk.append(str(current))
                else:
                    break
            walks.append(walk)

    model = Word2Vec(sentences=walks, vector_size=embedding_dim, window=window_size, sg=1, workers=4, min_count=0, epochs=1)

    embeddings = np.zeros((num_nodes, embedding_dim), dtype=np.float32)
    for i in range(num_nodes):
        embeddings[i] = model.wv[str(i)]

    return torch.tensor(embeddings)

def evaluate_gcnmf(
    data,
    max_epochs=1000,
    patience=40,
    seeds=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    accs, losses, f1s = [], [], []
    seed_keys = list(data.masks.keys()) if seeds is None else list(seeds)

    for seed in seed_keys:
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_mask = data.masks[seed]["train_mask"].to(device)
        val_mask = data.masks[seed]["val_mask"].to(device)
        test_mask = data.masks[seed]["test_mask"].to(device)

        data = data.to(device)
        y = data.y
        x = data.masks[seed]["X_incomp"]
        edge_index = data.edge_index.to(device)
        adj = data.adj.to(device)

        model = GCNmf(data, nhid=16, dropout=0.0, n_components=5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

        best_val_f1 = 0.0
        best_weights = None
        patience_counter = 0
        model.reset_parameters()

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(x, adj, edge_index)
            loss = F.nll_loss(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = out[val_mask].argmax(dim=1)
                val_f1 = f1_score(
                    y[val_mask].cpu().numpy(),
                    val_pred.cpu().numpy(),
                    average="macro",
                )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        model.load_state_dict(best_weights)
        model.eval()

        with torch.no_grad():
            out = model(x, adj, edge_index)
            test_pred = out[test_mask].argmax(dim=1)
            test_acc = accuracy_score(y[test_mask].cpu(), test_pred.cpu())
            test_loss = F.cross_entropy(out[test_mask], y[test_mask]).item()
            test_f1 = f1_score(
                y[test_mask].cpu().numpy(),
                test_pred.cpu().numpy(),
                average="macro",
            )

            accs.append(test_acc)
            f1s.append(test_f1)
            losses.append(test_loss)

    return float(np.mean(accs)), float(np.mean(losses)), float(np.mean(f1s)), float(np.std(f1s))

def evaluate_goodie(data, max_epochs=1000, patience=30, seeds=seeds, device=device, x_original=None):
    accs = []
    f1s = []

    for seed in data.masks.keys():
        train_mask = data.masks[seed]['train_mask'].to(device)
        val_mask = data.masks[seed]['val_mask'].to(device)
        test_mask = data.masks[seed]['test_mask'].to(device)

        # ==========
        # Config as dict
        # ==========
        config = {
            'device': device,
            'dataset': 'custom',
            'missing_rate': 0.0,  # o quello che usi
            'embedder': 'goodie',
            'gnn': 'GCN',
            'n_feat': data.x.size(1),
            'n_nodes': data.x.size(0),
            'n_class': int(data.y.max()) + 1,
            'n_hid': 32,
            'lr': 0.01,
            'epochs': max_epochs,
            'patience': patience,
            'lamb': 0.3,  # loss bilanciamento
            'scaled': False,
            'leaky_alpha': 0.2,
            'temp': 0.07,
            'lp_alpha': 0.5,
            'num_iterations': 40,
            'n_runs': 1,
            'seed': seed,
            'mask_type': 'random',
            'data': data,
            'train_mask' : train_mask.to(device), 
            'val_mask' : val_mask.to(device), 
            'test_mask' : test_mask.to(device), 
            'edge_index': data.edge_index.to(device),
            'labels': data.y.to(device),
            'num_layers': 2,
            'num_heads': 4,
            'x_original': x_original
        }

        args = Namespace(**config)

        # ==========
        # Costruisci modello
        # ==========
        model = GOODIE(args)
        
        model.x = data.x.to(device)
        model.edge_index = data.edge_index.to(device)
        model.labels = data.y.to(device)
        model.train_mask = train_mask.to(device)
        model.val_mask = val_mask.to(device)
        model.test_mask = test_mask.to(device)
        model.missing_feature_mask = torch.isnan(data.x).to(device)
        model.evaluator = None  # o custom evaluator

        # ==========
        # Training
        # ==========
        mean_acc, mean_f1, x = model.training(seed=seed)  # se modifichi GOODIE per restituirli
        # print('differenza: ', (x_original.cpu() - x.cpu()).mean())
        accs.append(mean_acc)
        f1s.append(mean_f1)
    
    return np.mean(accs), None, np.mean(f1s), np.std(f1s)

def evaluate_gcn(
    data,
    hidden_channels=128,
    max_epochs=500,
    patience=50,
    seeds=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    method=None,
    mod=None,
):
    accs, losses, f1s = [], [], []
    seed_keys = list(data.masks.keys()) if seeds is None else list(seeds)

    for seed in seed_keys:
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_mask = data.masks[seed]["train_mask"].to(device)
        val_mask = data.masks[seed]["val_mask"].to(device)
        test_mask = data.masks[seed]["test_mask"].to(device)

        data = data.to(device)
        y = data.y.to(device)
        x = data.masks[seed]["X_incomp"]
        edge_index = data.edge_index.to(device)

        train_val_mask = train_mask | val_mask
        if torch.isnan(x).any():
            x = fill_nan_with_col_mean_split(x, train_val_mask, test_mask)

        if mod is None or mod == "gnnmim":
            lr = 0.01
            hidden_channels = 128
            model = GCNFull(
                x.size(1), hidden_channels, num_classes=data.num_classes, num_layers=2
            ).to(device)

        class_counts = torch.bincount(y[train_mask])
        weight = 1.0 / class_counts.float()
        weight = weight / weight.sum()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_f1 = 0.0
        best_weights = None
        patience_counter = 0

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = F.cross_entropy(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out_val = model(x, edge_index)
                val_pred = out_val[val_mask].argmax(dim=1)
                val_f1 = f1_score(
                    y[val_mask].cpu().numpy(),
                    val_pred.cpu().numpy(),
                    average="macro",
                )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        model.load_state_dict(best_weights)
        model.eval()

        with torch.no_grad():
            out_test = model(x, edge_index)
            valid_test_mask = test_mask & (y != -1)
            test_pred = out_test[valid_test_mask].argmax(dim=1)
            test_acc = accuracy_score(
                y[valid_test_mask].cpu(), test_pred.cpu()
            )
            test_loss = F.cross_entropy(
                out_test[valid_test_mask], y[valid_test_mask]
            ).item()
            test_f1 = f1_score(
                y[valid_test_mask].cpu().numpy(),
                test_pred.cpu().numpy(),
                average="macro",
            )

            accs.append(test_acc)
            losses.append(test_loss)
            f1s.append(test_f1)

    del model
    torch.cuda.empty_cache()

    return float(np.mean(accs)), float(np.mean(losses)), float(np.mean(f1s)), float(np.std(f1s))

def feature_propagation(edge_index, X, feature_mask, num_iterations, test_mask):
    propagation_model = FeaturePropagation(num_iterations=num_iterations)

    return propagation_model.propagate(x=X, edge_index=edge_index, mask=feature_mask, test_mask=test_mask)

def filling(filling_method, edge_index, X, feature_mask, num_iterations=None, test_mask=None):
    X_reconstructed = feature_propagation(edge_index, X, feature_mask, num_iterations, test_mask)
    return X_reconstructed

def evaluate_fp(
    data,
    hidden_channels=16,
    max_epochs=1000,
    patience=30,
    seeds=None,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    accs, losses, f1s = [], [], []

    seed_keys = list(data.masks.keys()) if seeds is None else list(seeds)

    for seed in seed_keys:
        torch.manual_seed(seed)
        np.random.seed(seed)

        x_incomp = data.masks[seed]["X_incomp"]
        missing_feature_mask = ~torch.isnan(x_incomp)

        filling_method = "feature_propagation"
        num_iterations = 40
        num_layers = 2
        dropout = 0.5

        filled_features = filling(
            filling_method,
            data.edge_index,
            x_incomp,
            missing_feature_mask,
            num_iterations,
            test_mask=data.masks[seed]["test_mask"].to(device),
        )

        model = GNN(
            num_features=data.x.shape[1],
            num_classes=len(torch.unique(data.y)),
            num_layers=num_layers,
            hidden_dim=hidden_channels,
            dropout=dropout,
            conv_type="gcn",
            jumping_knowledge=False,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.where(missing_feature_mask, x_incomp, filled_features).to(device)

        edge_index = data.edge_index.to(device)
        y = data.y.to(device)

        train_mask = data.masks[seed]["train_mask"].to(device)
        val_mask = data.masks[seed]["val_mask"].to(device)
        test_mask = data.masks[seed]["test_mask"].to(device)

        best_val_f1 = 0.0
        best_weights = None
        patience_counter = 0

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = F.nll_loss(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out_val = model(x, edge_index)
                val_pred = out_val[val_mask].argmax(dim=1)
                val_f1 = f1_score(
                    y[val_mask].cpu().numpy(),
                    val_pred.cpu().numpy(),
                    average="macro",
                )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        model.load_state_dict(best_weights)
        model.eval()

        with torch.no_grad():
            out_test = model(x, edge_index)
            test_pred = out_test[test_mask].argmax(dim=1)
            test_acc = accuracy_score(y[test_mask].cpu().numpy(), test_pred.cpu().numpy())
            test_loss = F.nll_loss(out_test[test_mask], y[test_mask]).item()
            test_f1 = f1_score(
                y[test_mask].cpu().numpy(),
                test_pred.cpu().numpy(),
                average="macro",
            )

            accs.append(test_acc)
            losses.append(test_loss)
            f1s.append(test_f1)

    return float(np.mean(accs)), float(np.mean(losses)), float(np.mean(f1s)), float(np.std(f1s))

def evaluate_pcfi(
    data,
    hidden_channels=16,
    max_epochs=1000,
    patience=30,
    seeds=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    x_original=None,
):
    accs, losses, f1s = [], [], []

    seed_keys = list(data.masks.keys()) if seeds is None else list(seeds)

    for seed in seed_keys:
        torch.manual_seed(seed)
        np.random.seed(seed)

        x_incomp = data.masks[seed]["X_incomp"]
        missing_feature_mask = ~torch.isnan(x_incomp)

        mask_type = "structural"
        num_iterations = 40
        alpha, beta = 0.9, 1.0
        dropout = 0.5

        filled_features = pcfi(
            data.edge_index,
            x_incomp,
            missing_feature_mask,
            num_iterations,
            mask_type,
            alpha,
            beta,
        )

        model = GNN(
            num_features=data.x.shape[1],
            num_classes=len(torch.unique(data.y)),
            num_layers=2,
            hidden_dim=hidden_channels,
            dropout=dropout,
            conv_type="gcn",
            jumping_knowledge=False,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        edge_index = data.edge_index.to(device)
        y = data.y.to(device)
        x = filled_features.to(device)

        train_mask = data.masks[seed]["train_mask"].to(device)
        val_mask = data.masks[seed]["val_mask"].to(device)
        test_mask = data.masks[seed]["test_mask"].to(device)

        best_val_f1 = 0.0
        best_weights = None
        patience_counter = 0

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = F.nll_loss(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out_val = model(x, edge_index)
                val_pred = out_val[val_mask].argmax(dim=1)
                val_f1 = f1_score(
                    y[val_mask].cpu().numpy(),
                    val_pred.cpu().numpy(),
                    average="macro",
                )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        model.load_state_dict(best_weights)
        model.eval()

        with torch.no_grad():
            out_test = model(x, edge_index)
            test_pred = out_test[test_mask].argmax(dim=1)
            test_acc = accuracy_score(y[test_mask].cpu().numpy(), test_pred.cpu().numpy())
            test_loss = F.nll_loss(out_test[test_mask], y[test_mask]).item()
            test_f1 = f1_score(
                y[test_mask].cpu().numpy(),
                test_pred.cpu().numpy(),
                average="macro",
            )

            accs.append(test_acc)
            losses.append(test_loss)
            f1s.append(test_f1)

            if x_original is not None:
                out_orig = model(x_original.to(device), edge_index)
                _ = out_orig[test_mask]  # Optional second evaluation (not aggregated)

    return float(np.mean(accs)), float(np.mean(losses)), float(np.mean(f1s)), float(np.std(f1s))

def evaluate_fairac(
    data,
    hidden_channels=16,
    transformed_feature_dim=16,
    max_epochs=600,
    patience=30,
    eval_every=5,
    seeds=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
):

    @torch.no_grad()
    def build_norm_adj(edge_index: torch.Tensor, num_nodes: int, device: str):
        ei = edge_index.to(device)
        if ei.dtype != torch.long:
            ei = ei.long()
        ei_undirected = torch.cat([ei, ei.flip(0)], dim=1)
        values = torch.ones(ei_undirected.size(1), device=device)
        adj = torch.sparse_coo_tensor(
            ei_undirected, values, size=(num_nodes, num_nodes), device=device
        ).coalesce()
        row, col = adj.indices()
        val = adj.values()
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = deg.clamp(min=1e-12).pow(-0.5)
        norm_vals = deg_inv_sqrt[row] * val * deg_inv_sqrt[col]
        adj_norm = torch.sparse_coo_tensor(
            adj.indices(), norm_vals, size=(num_nodes, num_nodes), device=device
        ).coalesce()
        return adj_norm

    num_classes = int(data.y.max().item()) + 1
    args = SimpleNamespace(
        num_hidden=hidden_channels, lr=0.001, weight_decay=1e-5,
        attn_vec_dim=128, dropout=0.3, num_heads=1, cuda=(device!="cpu"),
        num_sen_class=1, transformed_feature_dim=transformed_feature_dim,
        feat_drop_rate=0.0, lambda1=1.0, lambda2=1.0, model="GCN"
    )

    edge_index = data.edge_index.to(device)
    y = data.y.to(device).long()
    num_nodes = data.num_nodes
    adj_norm = build_norm_adj(edge_index, num_nodes, device)

    with torch.no_grad():
        emb_cpu = generate_deepwalk_embeddings(
            edge_index.detach().cpu(),
            num_nodes=num_nodes,
            embedding_dim=16
        )
    embedding = emb_cpu.to(device)

    seed_keys = list(data.masks.keys()) if seeds is None else list(seeds)
    accs, losses, f1s = [], [], []

    use_amp = (device != "cpu")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for seed in seed_keys:
        torch.manual_seed(seed)
        np.random.seed(seed)

        x = data.masks[seed]["X_incomp"].clone()
        x[torch.isnan(x)] = 0.0
        x = x.to(device)

        train_mask = data.masks[seed]["train_mask"].to(device)
        val_mask   = data.masks[seed]["val_mask"].to(device)
        test_mask  = data.masks[seed]["test_mask"].to(device)

        gnn = GNNf(nfeat=transformed_feature_dim, args=args, num_classes=num_classes).to(device)
        ac  = FairAC2(
            feature_dim=x.shape[1],
            transformed_feature_dim=transformed_feature_dim,
            emb_dim=embedding.shape[1],
            args=args
        ).to(device)

        optimizer_g  = torch.optim.Adam(gnn.parameters(), lr=0.005, weight_decay=5e-4)
        optimizer_ac = torch.optim.Adam(ac.parameters(),  lr=0.001, weight_decay=1e-4)

        best_val_f1 = -1.0
        best_gnn_w, best_ac_w = None, None
        patience_ctr = 0

        for epoch in range(1, max_epochs + 1):
            gnn.train(); ac.train()
            optimizer_g.zero_grad(set_to_none=True)
            optimizer_ac.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                transformed_feature = ac.feature_transform(x)
                ac_feature_re = ac.hgnn_ac(adj_norm, embedding, embedding, transformed_feature)
                _, logits = gnn(edge_index, ac_feature_re)
                loss = F.cross_entropy(logits[train_mask], y[train_mask])

            scaler.scale(loss).backward()
            scaler.step(optimizer_g)
            scaler.step(optimizer_ac)
            scaler.update()

            if epoch % eval_every == 0 or epoch == 1:
                gnn.eval(); ac.eval()
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                    tf = ac.feature_transform(x)
                    ac_feat_val = ac.hgnn_ac(adj_norm, embedding, embedding, tf)
                    _, logits_val = gnn(edge_index, ac_feat_val)
                    val_pred = logits_val[val_mask].argmax(dim=1)
                    val_f1 = f1_score(
                        y[val_mask].detach().cpu().numpy(),
                        val_pred.detach().cpu().numpy(),
                        average="macro"
                    )

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_gnn_w = {k: v.detach().clone() for k, v in gnn.state_dict().items()}
                    best_ac_w  = {k: v.detach().clone() for k, v in ac.state_dict().items()}
                    patience_ctr = 0
                else:
                    patience_ctr += eval_every
                    if patience_ctr >= patience:
                        break

        if best_gnn_w is not None: gnn.load_state_dict(best_gnn_w)
        if best_ac_w  is not None: ac.load_state_dict(best_ac_w)
        gnn.eval(); ac.eval()

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            tf = ac.feature_transform(x)
            ac_feat = ac.hgnn_ac(adj_norm, embedding, embedding, tf)
            _, logits_test = gnn(edge_index, ac_feat)
            test_pred = logits_test[test_mask].argmax(dim=1)
            test_acc  = accuracy_score(y[test_mask].cpu().numpy(), test_pred.cpu().numpy())
            test_loss = F.cross_entropy(logits_test[test_mask], y[test_mask]).item()
            test_f1   = f1_score(
                y[test_mask].cpu().numpy(),
                test_pred.cpu().numpy(),
                average="macro"
            )

        accs.append(test_acc)
        losses.append(test_loss)
        f1s.append(test_f1)

    return float(np.mean(accs)), float(np.mean(losses)), float(np.mean(f1s)), float(np.std(f1s))

def evaluate_gspn(data, seeds, hidden_channels=64, max_epochs=1000, patience=30, lr=0.005):
    accs, losses, f1s = [], [], []

    for seed in data.masks.keys():
        torch.manual_seed(seed)
        np.random.seed(seed)

        config = {
            "num_layers": 3,
            "num_mixtures": 10,
            "num_hidden_neurons": 128,
            "convolution_class": "model.GSPNBaseConv",
            "emission_class": "model.GSPNGaussianEmission",
            "avg_parameters_across_layers": True,
            "init_kmeans": True,
            "global_pooling": "mean",
            "graph_emission_class": "model.GSPNGaussianEmission",
            "seed": seed,
        }

        train_mask = data.masks[seed]["train_mask"].to(device)
        val_mask = data.masks[seed]["val_mask"].to(device)
        test_mask = data.masks[seed]["test_mask"].to(device)

        model = GSPN(
            dim_node_features=data.num_features,
            dim_edge_features=0,
            dim_target=int(data.y.max().item()) + 1,
            readout_class=ProbabilisticGraphReadoutNoLayerAttentionMLP,
            config=config,
        ).to(device)

        data = data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_f1 = 0.0
        best_weights = None
        patience_counter = 0

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data, seed)
            loss = F.cross_entropy(out[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out_val = model(data, seed)
                val_pred = out_val[val_mask].argmax(dim=1)
                val_f1 = f1_score(
                    data.y[val_mask].cpu().numpy(),
                    val_pred.cpu().numpy(),
                    average="macro",
                )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        model.load_state_dict(best_weights)
        model.eval()

        with torch.no_grad():
            out_test = model(data, seed)
            test_pred = out_test[test_mask].argmax(dim=1)
            test_acc = accuracy_score(data.y[test_mask].cpu(), test_pred.cpu())
            test_loss = F.cross_entropy(out_test[test_mask], data.y[test_mask])
            test_f1 = f1_score(
                data.y[test_mask].cpu().numpy(),
                test_pred.cpu().numpy(),
                average="macro",
            )
            accs.append(test_acc)
            losses.append(test_loss.item())
            f1s.append(test_f1)

    return np.mean(accs), np.mean(losses), np.mean(f1s), np.std(f1s)

def umcar(data, p_miss_dict, seeds):
    """
    Generates MCAR-masked feature matrices and masks for multiple seeds.
    
    Parameters
    ----------
    X : torch.Tensor of shape (n, d)
        Input node features.
    y : torch.Tensor of shape (n,)
        Node labels (used for stratified splitting).
    data : object with attribute `num_nodes`
        Typically a PyG data object.
    p_miss_dict : dict
        Dictionary with masking probabilities, e.g., {'train': 0.2, 'test': 0.3}.
    seeds : list of int
        List of seeds to iterate over.
    
    Returns
    -------
    masks : dict
        Dictionary keyed by seed. Each value is a dict with keys:
        'X_incomp' : torch.Tensor with NaNs in missing positions,
        'mask'     : torch.BoolTensor of same shape as X (True = missing).
    """
    masks = {}

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Split node indices
        idx = np.arange(data.num_nodes)
        idx_train_val, idx_test = train_test_split(
            idx, test_size=0.2, stratify=data.y.cpu().numpy(), random_state=seed
        )
        idx_train, idx_val = train_test_split(
            idx_train_val, test_size=0.125, stratify=data.y[idx_train_val].cpu().numpy(), random_state=seed
        )

        # Create boolean masks
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[idx_train] = True
        val_mask[idx_val] = True
        test_mask[idx_test] = True

        # Create initial MCAR mask (all False)
        n, d = data.x.shape
        mask = torch.zeros(n, d, dtype=torch.bool)

        # Apply MCAR separately to train/val and test using respective probabilities
        for mask_name, node_mask, p in [('train', train_mask | val_mask, p_miss_dict['train']),
                                        ('test', test_mask, p_miss_dict['test'])]:
            selected_nodes = torch.where(node_mask)[0]
            temp_mask = torch.rand(len(selected_nodes), d) < p
            mask[selected_nodes] = temp_mask

        # Ensure at least one observed value per column
        for j in range(d):
            if mask[:, j].all():
                i = torch.randint(0, n, (1,))
                mask[i, j] = False

        # Apply mask to create incomplete version of X
        X_incomp = data.x.clone()
        X_incomp[mask] = float('nan')

        # Store results
        masks[seed] = {
            'X_incomp': X_incomp,
            'mask': mask,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask
        }

    return masks

def ldmcar(data, p_miss_dict, seeds):
    """
    Label-Dependent MCAR (LD-MCAR).

    Generates MCAR masks that exactly match the target missingness rates,
    prioritizing the masking of the most label-informative features
    (according to mutual information with the labels).

    Parameters
    ----------
    data : PyG Data
        Graph data object with attributes 'x' (features) and 'y' (labels).
    p_miss_dict : dict
        Dictionary with target missing fractions per split:
        e.g. {"train": 0.2, "test": 0.3}.
    seeds : list[int]
        List of random seeds for reproducibility.

    Returns
    -------
    masks : dict
        Dict[seed] = {
            "X_incomp": torch.Tensor with NaNs for missing values,
            "mask": torch.BoolTensor indicating missing entries,
            "train_mask": BoolTensor for train nodes,
            "val_mask": BoolTensor for validation nodes,
            "test_mask": BoolTensor for test nodes,
        }
    """

    X = data.x.clone().cpu().numpy()
    y = data.y.clone().cpu().numpy()
    n, d = X.shape
    masks = {}

    # Compute mutual information per feature, sort descending
    mi_scores = mutual_info_classif(X, y, discrete_features=False)
    feat_ranking = np.argsort(mi_scores)[::-1]  # descending: most informative first

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Stratified split
        idx = np.arange(n)
        idx_train_val, idx_test = train_test_split(
            idx, test_size=0.2, stratify=y, random_state=seed
        )
        idx_train, idx_val = train_test_split(
            idx_train_val, test_size=0.125, stratify=y[idx_train_val], random_state=seed
        )

        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)
        train_mask[idx_train] = True
        val_mask[idx_val] = True
        test_mask[idx_test] = True

        mask = np.zeros((n, d), dtype=bool)

        for split_name, node_mask, p_miss in [
            ("train", train_mask | val_mask, p_miss_dict["train"]),
            ("test", test_mask, p_miss_dict["test"]),
        ]:
            nodes = np.where(node_mask.numpy())[0]
            n_mask_per_node = int(np.round(p_miss * d))

            for node in nodes:
                # Mask exactly n_mask_per_node features according to ranking
                masked_indices = feat_ranking[:n_mask_per_node]
                mask[node, masked_indices] = True

        # Ensure at least one observed feature per node
        for i in range(n):
            if mask[i].all():
                rand_j = np.random.randint(0, d)
                mask[i, rand_j] = False

        X_incomp = torch.tensor(X).clone()
        X_incomp[mask] = float("nan")

        masks[seed] = {
            "X_incomp": X_incomp.to(data.x.device),
            "mask": torch.tensor(mask).to(data.x.device),
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
        }

    return masks

def smcar(data, p_miss_dict, seeds):
    """
    Structured MCAR (S-MCAR).

    Generates missingness at the node level: entire nodes are masked,
    meaning all features of a selected node become missing.

    Parameters
    ----------
    data : PyG Data
        Graph data object with attributes 'num_nodes', 'x', and 'y'.
    p_miss_dict : dict
        Dictionary with target missing fractions per split:
        e.g. {"train": 0.2, "test": 0.3}.
    seeds : list[int]
        List of random seeds.

    Returns
    -------
    masks : dict
        Dict[seed] = {
            "X_incomp": torch.Tensor with NaNs in missing positions,
            "mask": BoolTensor of same shape as X (True = missing),
            "train_mask": BoolTensor for train nodes,
            "val_mask": BoolTensor for validation nodes,
            "test_mask": BoolTensor for test nodes,
        }
    """
    

    masks = {}

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Stratified split
        idx = np.arange(data.num_nodes)
        idx_train_val, idx_test = train_test_split(
            idx, test_size=0.2, stratify=data.y.cpu().numpy(), random_state=seed
        )
        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=0.125,
            stratify=data.y[idx_train_val].cpu().numpy(),
            random_state=seed,
        )

        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[idx_train] = True
        val_mask[idx_val] = True
        test_mask[idx_test] = True

        n, d = data.x.shape
        mask = torch.zeros(n, d, dtype=torch.bool)

        # Apply MCAR at node level for train/val and test
        for split_name, node_mask, p in [
            ("train", train_mask | val_mask, p_miss_dict["train"]),
            ("test", test_mask, p_miss_dict["test"]),
        ]:
            selected_nodes = torch.where(node_mask)[0]
            num_nodes_to_mask = int(np.round(len(selected_nodes) * p))
            if num_nodes_to_mask > 0:
                nodes_to_mask = np.random.choice(
                    selected_nodes.cpu().numpy(),
                    size=num_nodes_to_mask,
                    replace=False,
                )
                mask[nodes_to_mask] = True

        X_incomp = data.x.clone()
        X_incomp[mask] = float("nan")

        masks[seed] = {
            "X_incomp": X_incomp,
            "mask": mask,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
        }

    return masks

def _nanmean(x, dim=0, keepdim=False):
    mask = ~torch.isnan(x)
    x_filled = torch.where(mask, x, torch.zeros_like(x))
    count = mask.sum(dim=dim, keepdim=keepdim).clamp(min=1)
    return x_filled.sum(dim=dim, keepdim=keepdim) / count

def _nanstd(x, dim=0, keepdim=False, eps=1e-8):
    mean = _nanmean(x, dim=dim, keepdim=True)
    diff = torch.where(~torch.isnan(x), x - mean, torch.zeros_like(x))
    var = (diff ** 2).sum(dim=dim, keepdim=keepdim)
    count = (~torch.isnan(x)).sum(dim=dim, keepdim=keepdim).clamp(min=1)
    var = var / count
    std = torch.sqrt(var + eps)
    if not keepdim:
        std = std.squeeze(dim)
    return std

def _zscore_(X):
    mu = _nanmean(X, dim=0, keepdim=True)
    sd = _nanstd(X, dim=0, keepdim=True)
    sd = torch.where(sd == 0, torch.ones_like(sd), sd)  # avoid division by 0
    return (X - mu) / sd

def fdmnar(
    data,
    seeds,
    p_miss,                        # target % NaN on the whole matrix X (0..1)
    signal_feature=None,           # None=all; int=first k; list/tuple=explicit indices
    top_quantile=0.90,             # threshold to define the "signal" group
    which="high",                  # 'high' | 'low' | 'both'
    p_hi=0.95,                     # used to define ratio r (default) or as upper bound
    p_lo=0.65,                     # used only to define ratio r (default)
    mask_columns="self",           # 'self' = only the feature; 'row' = entire row (all features)
    calibration="ratio",           # 'ratio' (default) or 'lo_from_hi'
    clip_eps=1e-6,                 # epsilon for probability clipping
):
    """
    Feature-Dependent MNAR applied UNIFORMLY on ALL data (no label usage).

    - Defines a "signal group" for each selected feature (or rows if mask_columns="row")
      based on z-scores and quantiles across ALL nodes.
    - Calibrates probabilities (p_hi > p_lo) to match expected missingness p_miss.
      * calibration="ratio": preserves the ratio r = (p_hi / p_lo), solves p_lo.
      * calibration="lo_from_hi": fixes p_hi, solves p_lo = (p_miss - s*p_hi) / (1 - s).
    - Generates a Bernoulli mask according to calibrated probabilities.
    - Guarantees: no column is fully missing.
    Returns a dict per seed with X_incomp, mask, and metadata.
    """
    assert 0 <= p_miss <= 1, "p_miss must be in [0,1]"
    assert mask_columns in ("self", "row")
    assert which in ("high", "low", "both")
    assert calibration in ("ratio", "lo_from_hi")

    X = data.x.clone()
    n, d = X.shape
    device = X.device

    # compute z-scores (nan-safe)
    Xz = _zscore_(X)

    # select features
    if signal_feature is None:
        feats = list(range(d))
    elif isinstance(signal_feature, int):
        k = max(1, min(int(signal_feature), d))
        feats = list(range(k))
    else:
        feats = list(map(int, signal_feature))

    masks = {}

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        mask = torch.zeros(n, d, dtype=torch.bool, device=device)

        # stratified TRAIN/VAL/TEST split
        idx = np.arange(n)
        y_np = data.y.cpu().numpy() if hasattr(data, "y") else None

        idx_train_val, idx_test = train_test_split(
            idx, test_size=0.2, stratify=y_np, random_state=seed
        )
        y_tv = data.y[idx_train_val].cpu().numpy() if hasattr(data, "y") else None
        idx_train, idx_val = train_test_split(
            idx_train_val, test_size=0.125, stratify=y_tv, random_state=seed
        )

        train_mask_nodes = torch.zeros(n, dtype=torch.bool, device=device); train_mask_nodes[idx_train] = True
        val_mask_nodes   = torch.zeros(n, dtype=torch.bool, device=device); val_mask_nodes[idx_val]   = True
        test_mask_nodes  = torch.zeros(n, dtype=torch.bool, device=device); test_mask_nodes[idx_test] = True

        if mask_columns == "self":
            # per-feature calibration: s_j = fraction of rows in signal group for column j
            for j in feats:
                xj = Xz[:, j]
                q_hi = torch.quantile(xj, top_quantile)
                q_lo = torch.quantile(xj, 1.0 - top_quantile)

                cond_high = (xj >= q_hi)
                cond_low = (xj <= q_lo)

                if which == "high":
                    rows_sig = cond_high
                elif which == "low":
                    rows_sig = cond_low
                else:  # both
                    rows_sig = cond_high | cond_low

                s = rows_sig.float().mean().item()
                s = min(max(s, 0.0), 1.0)

                if s >= 1.0 - 1e-12:
                    # all signal
                    p_hi_j, p_lo_j = float(np.clip(p_miss, 0.0, 1.0)), 0.0
                elif s <= 1e-12:
                    # no signal
                    p_lo_j = float(np.clip(p_miss, 0.0, 1.0))
                    p_hi_j = max(p_lo_j + 1e-4, min(1.0, p_hi))
                else:
                    if calibration == "ratio":
                        r = np.inf if p_lo <= 0 else (p_hi / max(p_lo, 1e-12))
                        if not np.isfinite(r) or r <= 1.0:
                            r = 2.0
                        denom = (s * r + (1 - s))
                        p_lo_j = float(np.clip(p_miss / max(denom, 1e-12), 0.0, 1.0))
                        p_hi_j = float(np.clip(r * p_lo_j, 0.0, 1.0))
                        if p_hi_j <= p_lo_j:
                            p_hi_j = min(1.0, p_lo_j + 1e-3)
                    else:  # lo_from_hi
                        p_hi_j = float(np.clip(p_hi, 0.0, 1.0))
                        p_lo_j = float(np.clip((p_miss - s * p_hi_j) / max(1 - s, 1e-12), 0.0, 1.0))
                        if p_lo_j >= p_hi_j:
                            p_hi_j = min(1.0, max(p_lo_j + 1e-3, p_hi_j - 1e-3))

                # clip for numerical stability
                p_hi_j = float(np.clip(p_hi_j, clip_eps, 1.0 - clip_eps))
                p_lo_j = float(np.clip(p_lo_j, clip_eps, 1.0 - clip_eps))

                # Bernoulli sampling
                if rows_sig.any():
                    mask[rows_sig, j] = (torch.rand(rows_sig.sum(), device=device) < p_hi_j)
                rows_rest = ~rows_sig
                if rows_rest.any():
                    mask[rows_rest, j] = (torch.rand(rows_rest.sum(), device=device) < p_lo_j)

        else:  # mask_columns == "row"
            # row-level signal
            rows_sig = torch.zeros(n, dtype=torch.bool, device=device)
            for j in feats:
                xj = Xz[:, j]
                q_hi = torch.quantile(xj, top_quantile)
                q_lo = torch.quantile(xj, 1.0 - top_quantile)
                if which == "high":
                    rows_sig |= (xj >= q_hi)
                elif which == "low":
                    rows_sig |= (xj <= q_lo)
                else:
                    rows_sig |= ((xj >= q_hi) | (xj <= q_lo))

            s = rows_sig.float().mean().item()

            if s >= 1.0 - 1e-12:
                p_hi_row, p_lo_row = float(np.clip(p_miss, 0.0, 1.0)), 0.0
            elif s <= 1e-12:
                p_lo_row = float(np.clip(p_miss, 0.0, 1.0))
                p_hi_row = min(1.0, max(p_lo_row + 1e-3, p_hi))
            else:
                if calibration == "ratio":
                    r = np.inf if p_lo <= 0 else (p_hi / max(p_lo, 1e-12))
                    if not np.isfinite(r) or r <= 1.0:
                        r = 2.0
                    denom = (s * r + (1 - s))
                    p_lo_row = float(np.clip(p_miss / max(denom, 1e-12), 0.0, 1.0))
                    p_hi_row = float(np.clip(r * p_lo_row, 0.0, 1.0))
                    if p_hi_row <= p_lo_row:
                        p_hi_row = min(1.0, p_lo_row + 1e-3)
                else:
                    p_hi_row = float(np.clip(p_hi, 0.0, 1.0))
                    p_lo_row = float(np.clip((p_miss - s * p_hi_row) / max(1 - s, 1e-12), 0.0, 1.0))
                    if p_lo_row >= p_hi_row:
                        p_hi_row = min(1.0, max(p_lo_row + 1e-3, p_hi_row - 1e-3))

            p_hi_row = float(np.clip(p_hi_row, clip_eps, 1.0 - clip_eps))
            p_lo_row = float(np.clip(p_lo_row, clip_eps, 1.0 - clip_eps))

            if rows_sig.any():
                mask[rows_sig, :] = (torch.rand(rows_sig.sum(), d, device=device) < p_hi_row)
            rows_rest = ~rows_sig
            if rows_rest.any():
                mask[rows_rest, :] = (torch.rand(rows_rest.sum(), d, device=device) < p_lo_row)

        # ensure no column is fully missing
        for j in range(d):
            if bool(mask[:, j].all()):
                i = torch.randint(0, n, (1,), device=device)
                mask[i, j] = False

        X_incomp = X.clone()
        X_incomp[mask] = float("nan")

        miss_global = torch.isnan(X_incomp).float().mean().item()
        summary = {
            "target_p_miss": float(p_miss),
            "achieved_p_miss": float(miss_global),
            "features_used": feats,
            "which": which,
            "top_quantile": float(top_quantile),
            "mask_columns": mask_columns,
            "calibration": calibration,
        }

        masks[seed] = {
            "X_incomp": X_incomp,
            "mask": mask,
            "summary": summary,
            "train_mask": train_mask_nodes,
            "val_mask": val_mask_nodes,
            "test_mask": test_mask_nodes,
        }

    return masks

def cdmnar(
    data,
    seeds,
    p_miss,                         # target % NaN (0..1) for train+val
    r_ratio=2.0,                    # ratio = p_hi / p_lo (>=1)
    clip_eps=1e-6,
    # Decision Tree
    tree_max_depth=2,
    tree_min_samples_leaf=1,
    class_positive=1,
    # --- OOD controls ---
    ood=False,                      # if True: test set uses different missingness scheme
    ood_test_mode="none",           # {"mcar", "none"}
    p_test_mcar=0.5                 # prob MCAR for test cells if ood_test_mode="mcar"
):
    """
    Class-Dependent MNAR (cell-wise):

    - Train a Decision Tree to predict class == class_positive vs others.
    - For each sample reaching a positive leaf, mark features along the path as 'important' (p_hi).
    - Other cells -> p_lo.

    Calibration:
      p_miss = s_total*(r*p_lo) + (1 - s_total)*p_lo = p_lo*(s_total*r + 1 - s_total)
      => p_lo = p_miss / (s_total*r + 1 - s_total), p_hi = r*p_lo

    OOD mode:
      - ood=False: standard MNAR on all splits.
      - ood=True:
          * train+val: MNAR calibrated on train+val only
          * test: if ood_test_mode="mcar" apply MCAR with prob p_test_mcar;
                  if "none", keep test features complete.
    """
    assert 0 <= p_miss <= 1
    assert r_ratio >= 1.0
    assert ood_test_mode in ("mcar", "none")

    X = data.x.clone()
    n, d = X.shape
    device = X.device

    if not hasattr(data, "y"):
        raise ValueError("data.y is required")
    y = data.y.to(device)
    y_bin = (y == class_positive).long()

    # prepare data for Decision Tree (temporary mean imputation for NaNs)
    X_np = X.detach().cpu().numpy()
    col_means = np.nanmean(X_np, axis=0, keepdims=True)
    inds = np.where(np.isnan(X_np))
    X_np[inds] = np.take(col_means, inds[1], axis=1)
    y_np = y_bin.cpu().numpy()

    # train Decision Tree
    tree = DecisionTreeClassifier(
        max_depth=tree_max_depth,
        min_samples_leaf=tree_min_samples_leaf,
        random_state=0,
        criterion="gini"
    )
    tree.fit(X_np, y_np)

    # extract structure
    T = tree.tree_
    feat = T.feature
    thr = T.threshold
    cl_counts = T.value.squeeze(1)  # shape [n_nodes, 2]
    is_leaf = (feat == -2)

    # mark leaf as positive if p(class1) > 0.5
    leaf_is_pos = np.zeros(T.node_count, dtype=bool)
    for n_id in range(T.node_count):
        if is_leaf[n_id]:
            c0, c1 = cl_counts[n_id]
            leaf_is_pos[n_id] = (c1 > c0)

    left, right = T.children_left, T.children_right

    def important_cells_for_sample(x_row):
        """Return feature indices along the path if the leaf is positive."""
        node = 0
        used_feats = []
        while not is_leaf[node]:
            j = feat[node]
            t = thr[node]
            node = left[node] if x_row[j] <= t else right[node]
            used_feats.append(j)
        return used_feats if leaf_is_pos[node] else []

    masks = {}
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # stratified split
        idx = np.arange(n)
        y_full_np = data.y.cpu().numpy()
        idx_train_val, idx_test = train_test_split(
            idx, test_size=0.2, stratify=y_full_np, random_state=seed
        )
        y_tv = data.y[idx_train_val].cpu().numpy()
        idx_train, idx_val = train_test_split(
            idx_train_val, test_size=0.125, stratify=y_tv, random_state=seed
        )

        train_mask_nodes = torch.zeros(n, dtype=torch.bool, device=device); train_mask_nodes[idx_train] = True
        val_mask_nodes   = torch.zeros(n, dtype=torch.bool, device=device); val_mask_nodes[idx_val]   = True
        test_mask_nodes  = torch.zeros(n, dtype=torch.bool, device=device); test_mask_nodes[idx_test] = True
        tv_mask_nodes    = train_mask_nodes | val_mask_nodes

        # build signal mask: features on positive paths
        signal_mask = torch.zeros((n, d), dtype=torch.bool, device=device)
        for i in range(n):
            feats_i = important_cells_for_sample(X_np[i])
            if feats_i:
                signal_mask[i, feats_i] = True

        # calibration: compute fraction of "important" cells
        if ood:
            s_cal = signal_mask[tv_mask_nodes].float().mean().item() if tv_mask_nodes.any() else 0.0
        else:
            s_cal = signal_mask.float().mean().item()

        # calibrate p_lo / p_hi
        denom = (s_cal * r_ratio + (1.0 - s_cal))
        p_lo = float(np.clip(p_miss / max(denom, 1e-12), 0.0, 1.0))
        p_hi = float(np.clip(r_ratio * p_lo, 0.0, 1.0))
        p_lo = float(np.clip(p_lo, clip_eps, 1.0 - clip_eps))
        p_hi = float(np.clip(p_hi, clip_eps, 1.0 - clip_eps))

        # custom hacks
        if p_miss in [0.7, 0.8, 0.9]: p_lo += 0.05
        elif p_miss == 0.99: p_lo = 0.98
        elif p_miss == 0.5:  p_hi, p_lo = 0.9, 0.5

        # MNAR masking
        mask = torch.zeros((n, d), dtype=torch.bool, device=device)
        scope_mask = tv_mask_nodes if ood else torch.ones(n, dtype=torch.bool, device=device)

        sig_scope = signal_mask & scope_mask.unsqueeze(1)
        rest_scope = (~signal_mask) & scope_mask.unsqueeze(1)

        if sig_scope.any():
            mask[sig_scope] = (torch.rand(int(sig_scope.sum().item()), device=device) < p_hi)
        if rest_scope.any():
            mask[rest_scope] = (torch.rand(int(rest_scope.sum().item()), device=device) < p_lo)

        # OOD test handling
        if ood:
            if ood_test_mode == "none":
                mask[test_mask_nodes] = False
            elif ood_test_mode == "mcar":
                n_test = int(test_mask_nodes.sum().item())
                if n_test > 0:
                    test_cells = n_test * d
                    mask_test = (torch.rand(test_cells, device=device) < float(np.clip(p_test_mcar, 0.0, 1.0)))
                    mask_test = mask_test.view(n_test, d)
                    mask[test_mask_nodes] = mask_test

        # apply mask
        X_incomp = data.x.clone()
        X_incomp[mask] = float("nan")

        # compute stats
        miss_global = torch.isnan(X_incomp).float().mean().item()
        miss_train  = torch.isnan(X_incomp[train_mask_nodes]).float().mean().item() if train_mask_nodes.any() else 0.0
        miss_val    = torch.isnan(X_incomp[val_mask_nodes]).float().mean().item() if val_mask_nodes.any() else 0.0
        miss_test   = torch.isnan(X_incomp[test_mask_nodes]).float().mean().item() if test_mask_nodes.any() else 0.0

        summary = {
            "target_p_miss_train_val": float(p_miss),
            "achieved_p_miss_global": float(miss_global),
            "achieved_p_miss_train": float(miss_train),
            "achieved_p_miss_val": float(miss_val),
            "achieved_p_miss_test": float(miss_test),
            "ratio_r": float(r_ratio),
            "p_lo_used": p_lo,
            "p_hi_used": p_hi,
            "signal_fraction_train_val" if ood else "signal_fraction_all": float(s_cal),
            "tree_max_depth": tree_max_depth,
            "tree_min_samples_leaf": tree_min_samples_leaf,
            "class_positive": int(class_positive),
            "ood": bool(ood),
            "ood_test_mode": ood_test_mode if ood else "n/a",
            "p_test_mcar": float(p_test_mcar) if (ood and ood_test_mode == "mcar") else 0.0,
        }

        masks[seed] = {
            "X_incomp": X_incomp,
            "mask": mask,
            "summary": summary,
            "train_mask": train_mask_nodes,
            "val_mask": val_mask_nodes,
            "test_mask": test_mask_nodes,
        }

    return masks

def produce_NA_ood(data, p_miss_dict, mech, seeds=seeds):
    """
    Generate missing values for specifics missing-data mechanism and proportion of missing values. 
    
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p_miss : float
        Proportion of missing values to generate for variables which will have missing values.
    mech : dict, 
            Indicates the missing-data mechanism to be used. "MCAR" by default, "MAR", "MNAR" or "MNARsmask"
    opt: str, 
         For mecha = "MNAR", it indicates how the missing-data mechanism is generated: using a logistic regression ("logistic"), quantile censorship ("quantile") or logistic regression for generating a self-masked MNAR mechanism ("selfmasked").
    p_obs : float
            If mecha = "MAR", or mecha = "MNAR" with opt = "logistic" or "quanti", proportion of variables with *no* missing values that will be used for the logistic masking model.
    q : float
        If mecha = "MNAR" and opt = "quanti", quantile level at which the cuts should occur.
    
    Returns
    ----------
    A dictionnary containing:
    'X_init': the initial data matrix.
    'X_incomp': the data with the generated missing values.
    'mask': a matrix indexing the generated missing values.s
    """
    
    to_torch = torch.is_tensor(data.x)
    if not to_torch:
        data.x = data.x.astype(np.float32)
        data.x = torch.from_numpy(data.x)

    if mech["mecha"] == "UMCAR":
        data.masks = umcar(data, p_miss_dict, seeds)
    elif mech["mecha"] == "LDMCAR":
        data.masks = ldmcar(data, p_miss_dict, seeds)
    elif mech["mecha"] == 'SMCAR':
        data.masks = smcar(data, p_miss_dict, seeds)
    elif mech["mecha"] == "FDMNAR":
        data.masks = fdmnar(data, seeds, p_miss_dict['train'])
    elif mech["mecha"] == "CDMNAR":
        data.masks = cdmnar(data, seeds, p_miss_dict['train'])
    return data


