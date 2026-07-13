#!/usr/bin/env python3
"""
Benchmark inference speed for GNN methods.
Measures pairs/second for each GNN model type.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# DeepChem for featurization
import deepchem as dc

# DGL for graph neural networks
import dgl
from dgl.nn import GraphConv, GATConv, GINConv
from dgl.nn.pytorch import Set2Set

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET = "bigsol1.0"  # Use first dataset for benchmarking
ALL_DATASETS_DIR = "all_datasets"
OUTPUT_DIR = "gnn_benchmark_results"
SEED = 42
NUM_INFERENCE_SAMPLES = 1000  # Number of samples for inference benchmarking

# Model config
HIDDEN_DIM = 64
NUM_GNN_LAYERS = 3
MLP_HIDDEN_DIM = 128
BATCH_SIZE = 64

# Device - DGL does not support MPS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GNN_MODELS = ["GCN", "GAT", "GIN", "MPNN"]

# ============================================================================
# GNN LAYERS (copied from baselining_gnn_methods.py)
# ============================================================================

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_dim, hidden_dim, activation=nn.ReLU()))
        for _ in range(num_layers - 1):
            self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU()))
        self.pool = dgl.nn.AvgPooling()

    def forward(self, g, feats):
        h = feats
        for layer in self.layers:
            h = layer(g, h)
        return self.pool(g, h)


class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_dim, hidden_dim // num_heads, num_heads, activation=nn.ELU()))
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(hidden_dim, hidden_dim // num_heads, num_heads, activation=nn.ELU()))
        self.pool = dgl.nn.AvgPooling()

    def forward(self, g, feats):
        h = feats
        for layer in self.layers:
            h = layer(g, h).flatten(1)
        return self.pool(g, h)


class GINEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layers.append(GINConv(mlp, learn_eps=True))
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.layers.append(GINConv(mlp, learn_eps=True))
        self.pool = dgl.nn.SumPooling()

    def forward(self, g, feats):
        h = feats
        for layer in self.layers:
            h = layer(g, h)
        return self.pool(g, h)


class MPNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, edge_dim=None):
        super().__init__()
        self.node_embed = nn.Linear(in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim) if edge_dim else None

        self.message_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.message_layers.append(GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU()))

        self.set2set = Set2Set(hidden_dim, n_iters=6, n_layers=3)
        self.out_dim = hidden_dim * 2

    def forward(self, g, feats, edge_feats=None):
        h = self.node_embed(feats)
        for layer in self.message_layers:
            h = layer(g, h)
        return self.set2set(g, h)


class DualGNNModel(nn.Module):
    def __init__(self, gnn_type, node_in_dim, edge_in_dim=None,
                 hidden_dim=64, num_layers=3, mlp_hidden=128, num_global_feats=4):
        super().__init__()
        self.gnn_type = gnn_type

        if gnn_type == "GCN":
            self.solute_encoder = GCNEncoder(node_in_dim, hidden_dim, num_layers)
            self.solvent_encoder = GCNEncoder(node_in_dim, hidden_dim, num_layers)
            emb_dim = hidden_dim
        elif gnn_type == "GAT":
            self.solute_encoder = GATEncoder(node_in_dim, hidden_dim, num_layers)
            self.solvent_encoder = GATEncoder(node_in_dim, hidden_dim, num_layers)
            emb_dim = hidden_dim
        elif gnn_type == "GIN":
            self.solute_encoder = GINEncoder(node_in_dim, hidden_dim, num_layers)
            self.solvent_encoder = GINEncoder(node_in_dim, hidden_dim, num_layers)
            emb_dim = hidden_dim
        elif gnn_type == "MPNN":
            self.solute_encoder = MPNNEncoder(node_in_dim, hidden_dim, num_layers, edge_in_dim)
            self.solvent_encoder = MPNNEncoder(node_in_dim, hidden_dim, num_layers, edge_in_dim)
            emb_dim = hidden_dim * 2
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

        mlp_in_dim = emb_dim * 2 + num_global_feats
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden // 2, 1)
        )

    def forward(self, solute_g, solvent_g, global_feats):
        solute_feats = solute_g.ndata['feat']
        solvent_feats = solvent_g.ndata['feat']

        solute_edge = solute_g.edata.get('feat', None)
        solvent_edge = solvent_g.edata.get('feat', None)

        if self.gnn_type == "MPNN":
            solute_emb = self.solute_encoder(solute_g, solute_feats, solute_edge)
            solvent_emb = self.solvent_encoder(solvent_g, solvent_feats, solvent_edge)
        else:
            solute_emb = self.solute_encoder(solute_g, solute_feats)
            solvent_emb = self.solvent_encoder(solvent_g, solvent_feats)

        combined = torch.cat([solute_emb, solvent_emb, global_feats], dim=1)
        return self.mlp(combined)


# ============================================================================
# GRAPH FEATURIZATION
# ============================================================================

def smiles_to_dgl_graph(smiles, featurizer):
    """Convert SMILES to DGL graph."""
    try:
        mol_graph = featurizer.featurize([smiles])[0]
        if mol_graph is None:
            return None

        node_feats = torch.tensor(mol_graph.node_features, dtype=torch.float32)
        edge_feats = torch.tensor(mol_graph.edge_features, dtype=torch.float32) if mol_graph.edge_features is not None else None

        src = mol_graph.edge_index[0]
        dst = mol_graph.edge_index[1]
        g = dgl.graph((src, dst), num_nodes=mol_graph.num_nodes)
        g = dgl.add_self_loop(g)
        g.ndata['feat'] = node_feats

        if edge_feats is not None:
            num_self_loops = mol_graph.num_nodes
            self_loop_feats = torch.zeros(num_self_loops, edge_feats.shape[1], dtype=torch.float32)
            g.edata['feat'] = torch.cat([edge_feats, self_loop_feats], dim=0)

        return g
    except Exception as e:
        return None


def featurize_molecules(smiles_list, featurizer):
    """Featurize list of SMILES."""
    graph_dict = {}
    for smiles in smiles_list:
        if smiles not in graph_dict:
            g = smiles_to_dgl_graph(smiles, featurizer)
            if g is not None:
                graph_dict[smiles] = g
    return graph_dict


class SolubilityDataset(Dataset):
    def __init__(self, df, solute_graphs, solvent_graphs):
        self.df = df.reset_index(drop=True)
        self.solute_graphs = solute_graphs
        self.solvent_graphs = solvent_graphs

        valid_mask = []
        for i in range(len(self.df)):
            sol = self.df.loc[i, 'Solute']
            solv = self.df.loc[i, 'Solvent']
            valid_mask.append(sol in solute_graphs and solv in solvent_graphs)

        self.df = self.df[valid_mask].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        solute_g = self.solute_graphs[row['Solute']]
        solvent_g = self.solvent_graphs[row['Solvent']]

        T = row['Temperature']
        global_feats = torch.tensor([
            T / 300.0,
            1000.0 / T,
            (T / 300.0) ** 2,
            np.log(T / 300.0)
        ], dtype=torch.float32)

        target = torch.tensor([row['LogS']], dtype=torch.float32)
        return solute_g, solvent_g, global_feats, target


def collate_fn(batch):
    solute_gs, solvent_gs, global_feats, targets = zip(*batch)
    batched_solute = dgl.batch(solute_gs)
    batched_solvent = dgl.batch(solvent_gs)
    global_feats = torch.stack(global_feats)
    targets = torch.stack(targets)
    return batched_solute, batched_solvent, global_feats, targets


# ============================================================================
# BENCHMARKING
# ============================================================================

def benchmark_gnn_model(model, test_loader, device):
    """Benchmark GNN model inference speed."""
    model.eval()
    model = model.to(device)

    # Warmup
    with torch.no_grad():
        for solute_g, solvent_g, global_feats, _ in test_loader:
            solute_g = solute_g.to(device)
            solvent_g = solvent_g.to(device)
            global_feats = global_feats.to(device)
            _ = model(solute_g, solvent_g, global_feats)
            break  # Just one warmup batch

    # Benchmark
    total_samples = 0
    with torch.no_grad():
        start = time.time()
        for solute_g, solvent_g, global_feats, _ in test_loader:
            solute_g = solute_g.to(device)
            solvent_g = solvent_g.to(device)
            global_feats = global_feats.to(device)
            _ = model(solute_g, solvent_g, global_feats)
            total_samples += len(global_feats)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start

    pairs_per_second = total_samples / elapsed
    return pairs_per_second


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("INFERENCE BENCHMARK - GNN METHODS")
    print("="*80)
    print(f"Dataset: {DATASET}")
    print(f"Inference samples: {NUM_INFERENCE_SAMPLES}")
    print(f"Device: {DEVICE}")
    print(f"GNN Models: {GNN_MODELS}")
    print("="*80)

    # Load data
    print("\nLoading test data...")
    test_file = os.path.join(ALL_DATASETS_DIR, DATASET, "test.csv")
    df_test = pd.read_csv(test_file).head(NUM_INFERENCE_SAMPLES)
    print(f"Test samples: {len(df_test)}")

    # Featurize molecules
    print("\nFeaturizing molecules...")
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

    all_solutes = set(df_test['Solute'].unique())
    all_solvents = set(df_test['Solvent'].unique())

    print(f"Unique solutes: {len(all_solutes)}")
    print(f"Unique solvents: {len(all_solvents)}")

    solute_graphs = featurize_molecules(all_solutes, featurizer)
    solvent_graphs = featurize_molecules(all_solvents, featurizer)

    print(f"Featurized {len(solute_graphs)} solutes, {len(solvent_graphs)} solvents")

    # Get feature dimensions
    sample_graph = next(iter(solute_graphs.values()))
    node_dim = sample_graph.ndata['feat'].shape[1]
    edge_dim = sample_graph.edata['feat'].shape[1] if 'feat' in sample_graph.edata else None
    print(f"Node feature dim: {node_dim}, Edge feature dim: {edge_dim}")

    # Create dataset
    test_dataset = SolubilityDataset(df_test, solute_graphs, solvent_graphs)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_fn)

    print(f"Dataset size after filtering: {len(test_dataset)}")

    results = []

    # ========================================================================
    # BENCHMARK GNN MODELS
    # ========================================================================
    print("\n" + "="*80)
    print("BENCHMARKING GNN MODELS")
    print("="*80)

    for gnn_type in GNN_MODELS:
        print(f"\n{gnn_type}:")
        print("-"*80)

        model_path = os.path.join(OUTPUT_DIR, DATASET, "gnn_models", gnn_type, f"model_seed{SEED}.pt")

        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")

            # Create model
            model = DualGNNModel(
                gnn_type=gnn_type,
                node_in_dim=node_dim,
                edge_in_dim=edge_dim,
                hidden_dim=HIDDEN_DIM,
                num_layers=NUM_GNN_LAYERS,
                mlp_hidden=MLP_HIDDEN_DIM,
                num_global_feats=4
            )

            # Load weights
            model.load_state_dict(torch.load(model_path, map_location='cpu'))

            # Benchmark
            pairs_per_sec = benchmark_gnn_model(model, test_loader, DEVICE)
            device_type = "GPU (CUDA)" if DEVICE.type == 'cuda' else "CPU"

            print(f"Inference speed: {pairs_per_sec:.2f} pairs/second")
            print(f"Device: {device_type}")

            results.append({
                'Model': gnn_type,
                'Pairs_per_Second': pairs_per_sec,
                'Device': device_type,
                'Device_Detail': f"DGL + PyTorch on {DEVICE}"
            })
        else:
            print(f"Model not found at: {model_path}")
            print("Please train models first using baselining_gnn_methods.py")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    if results:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Pairs_per_Second', ascending=False)

        print("\n" + results_df.to_string(index=False))

        # Save to CSV
        output_path = os.path.join(OUTPUT_DIR, "inference_benchmark_gnn.csv")
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

        # Print device summary
        print("\n" + "="*80)
        print("DEVICE USAGE SUMMARY")
        print("="*80)

        for _, row in results_df.iterrows():
            print(f"{row['Model']:<10} - {row['Device']:<15} - {row['Pairs_per_Second']:.2f} pairs/sec")

        print("\nNote: All GNN models use DGL + PyTorch")
        print("DGL does not support MPS (Apple Silicon GPU), only CUDA or CPU")
        if DEVICE.type == 'cuda':
            print("Currently running on CUDA GPU")
        else:
            print("Currently running on CPU (no CUDA available)")


if __name__ == "__main__":
    main()
