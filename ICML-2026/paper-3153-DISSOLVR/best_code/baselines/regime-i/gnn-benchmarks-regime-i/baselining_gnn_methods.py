#!/usr/bin/env python3
"""
GNN Baselining for Aqueous Solubility Prediction (Regime-I)
============================================================
Implements MPNN, GAT, GCN, and GIN models for single-molecule solubility prediction.
Uses DeepChem's MolGraphConvFeaturizer and DGL for graph neural networks.

Usage:
    conda activate dgl_env1
    python baselining_gnn_methods.py [--quick-test]
"""

import os
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, r2_score
from tqdm.auto import tqdm
import joblib
import time

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

DATASETS = ["aqsoldb", "esol", "sc2"]
ALL_DATASETS_DIR = "all_datasets"
OUTPUT_DIR = "gnn_benchmark_results"
SEEDS = [42, 101, 123, 456, 789]

# Model types to benchmark
GNN_MODELS = ["GCN", "GAT", "GIN", "MPNN"]

# Training hyperparameters
HIDDEN_DIM = 64
NUM_GNN_LAYERS = 3
MLP_HIDDEN_DIM = 128
BATCH_SIZE = 64
EPOCHS = 100
LR = 0.001
PATIENCE = 15  # Early stopping patience

# Device - Note: DGL does not support MPS, so we use CUDA or CPU only
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# GRAPH FEATURIZATION
# ============================================================================

def smiles_to_dgl_graph(smiles, featurizer):
    """Convert SMILES to DGL graph using DeepChem's MolGraphConvFeaturizer."""
    try:
        mol_graph = featurizer.featurize([smiles])[0]
        if mol_graph is None:
            return None
        
        # Extract node and edge features from GraphData
        node_feats = torch.tensor(mol_graph.node_features, dtype=torch.float32)
        edge_feats = torch.tensor(mol_graph.edge_features, dtype=torch.float32) if mol_graph.edge_features is not None else None
        
        # Create DGL graph from edge index
        src = mol_graph.edge_index[0]
        dst = mol_graph.edge_index[1]
        g = dgl.graph((src, dst), num_nodes=mol_graph.num_nodes)
        
        # Add self-loops
        g = dgl.add_self_loop(g)
        
        # Pad node features to match self-loop edges
        g.ndata['feat'] = node_feats
        
        if edge_feats is not None:
            # Pad edge features for self-loops
            num_self_loops = mol_graph.num_nodes
            self_loop_feats = torch.zeros(num_self_loops, edge_feats.shape[1], dtype=torch.float32)
            g.edata['feat'] = torch.cat([edge_feats, self_loop_feats], dim=0)
        
        return g
    except Exception as e:
        print(f"Error featurizing {smiles}: {e}")
        return None


def featurize_molecules(smiles_list, featurizer):
    """Featurize a list of SMILES strings, returning a dict mapping SMILES to DGL graphs."""
    graph_dict = {}
    for smiles in tqdm(smiles_list, desc="Featurizing molecules"):
        if smiles not in graph_dict:
            g = smiles_to_dgl_graph(smiles, featurizer)
            if g is not None:
                graph_dict[smiles] = g
    return graph_dict


# ============================================================================
# GNN LAYERS
# ============================================================================

class GCNEncoder(nn.Module):
    """Graph Convolutional Network encoder."""
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
    """Graph Attention Network encoder."""
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
            h = layer(g, h).flatten(1)  # Flatten multi-head output
        return self.pool(g, h)


class GINEncoder(nn.Module):
    """Graph Isomorphism Network encoder."""
    def __init__(self, in_dim, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        # First layer 
        mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layers.append(GINConv(mlp, learn_eps=True))
        # Hidden layers
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.layers.append(GINConv(mlp, learn_eps=True))
        self.pool = dgl.nn.SumPooling()  # Sum pooling is standard for GIN
    
    def forward(self, g, feats):
        h = feats
        for layer in self.layers:
            h = layer(g, h)
        return self.pool(g, h)


class MPNNEncoder(nn.Module):
    """Message Passing Neural Network encoder with Set2Set readout."""
    def __init__(self, in_dim, hidden_dim, num_layers, edge_dim=None):
        super().__init__()
        self.node_embed = nn.Linear(in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim) if edge_dim else None
        
        self.message_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.message_layers.append(GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU()))
        
        self.set2set = Set2Set(hidden_dim, n_iters=6, n_layers=3)
        self.out_dim = hidden_dim * 2  # Set2Set doubles the dimension
    
    def forward(self, g, feats, edge_feats=None):
        h = self.node_embed(feats)
        for layer in self.message_layers:
            h = layer(g, h)
        return self.set2set(g, h)


# ============================================================================
# SINGLE MOLECULE GNN MODEL
# ============================================================================

class SingleGNNModel(nn.Module):
    """
    Single-molecule GNN model for aqueous solubility prediction.
    GNN encoder for molecule, then feed embedding to MLP predictor.
    """
    def __init__(self, gnn_type, node_in_dim, edge_in_dim=None, 
                 hidden_dim=64, num_layers=3, mlp_hidden=128):
        super().__init__()
        self.gnn_type = gnn_type
        
        # Create GNN encoder
        if gnn_type == "GCN":
            self.encoder = GCNEncoder(node_in_dim, hidden_dim, num_layers)
            emb_dim = hidden_dim
        elif gnn_type == "GAT":
            self.encoder = GATEncoder(node_in_dim, hidden_dim, num_layers)
            emb_dim = hidden_dim
        elif gnn_type == "GIN":
            self.encoder = GINEncoder(node_in_dim, hidden_dim, num_layers)
            emb_dim = hidden_dim
        elif gnn_type == "MPNN":
            self.encoder = MPNNEncoder(node_in_dim, hidden_dim, num_layers, edge_in_dim)
            emb_dim = hidden_dim * 2  # Set2Set output
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # MLP predictor
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden // 2, 1)
        )
    
    def forward(self, g):
        """
        Args:
            g: DGL graph for molecule
        """
        # Get node features
        feats = g.ndata['feat']
        
        # Get edge features if available (for MPNN)
        edge_feats = g.edata.get('feat', None)
        
        # Encode molecule
        if self.gnn_type == "MPNN":
            emb = self.encoder(g, feats, edge_feats)
        else:
            emb = self.encoder(g, feats)
        
        # Predict
        return self.mlp(emb)


# ============================================================================
# DATASET
# ============================================================================

class AqueousSolubilityDataset(Dataset):
    """Dataset for aqueous solubility prediction."""
    def __init__(self, df, molecule_graphs):
        self.df = df.reset_index(drop=True)
        self.molecule_graphs = molecule_graphs
        
        # Filter out rows with missing graphs
        valid_mask = []
        for i in range(len(self.df)):
            smiles = self.df.loc[i, 'SMILES']
            valid_mask.append(smiles in molecule_graphs)
        
        self.df = self.df[valid_mask].reset_index(drop=True)
        print(f"  Dataset size after filtering: {len(self.df)}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        g = self.molecule_graphs[row['SMILES']]
        target = torch.tensor([row['LogS']], dtype=torch.float32)
        
        return g, target


def collate_fn(batch):
    """Collate function for batching DGL graphs."""
    graphs, targets = zip(*batch)
    
    batched_graphs = dgl.batch(graphs)
    targets = torch.stack(targets)
    
    return batched_graphs, targets


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_loader, val_loader, epochs, lr, patience, device):
    """Train model with early stopping."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for graphs, targets in train_loader:
            graphs = graphs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            preds = model(graphs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for graphs, targets in val_loader:
                graphs = graphs.to(device)
                targets = targets.to(device)
                
                preds = model(graphs)
                val_loss += criterion(preds, targets).item()
        
        val_loss /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for graphs, targets in test_loader:
            graphs = graphs.to(device)
            
            preds = model(graphs)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())
    
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    
    return rmse, r2


# ============================================================================
# MAIN
# ============================================================================

def main(quick_test=False):
    print("="*80)
    print("GNN BASELINING FOR AQUEOUS SOLUBILITY PREDICTION (REGIME-I)")
    print("="*80)
    print(f"Models: {GNN_MODELS}")
    print(f"Datasets: {DATASETS}")
    print(f"Seeds: {SEEDS}")
    print(f"Device: {DEVICE}")
    print("="*80)
    
    if quick_test:
        print("\n*** QUICK TEST MODE - Using subset of data ***\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize featurizer
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    
    all_results = {}
    
    for dataset_name in DATASETS:
        print(f"\n{'#'*80}")
        print(f"# DATASET: {dataset_name}")
        print(f"{'#'*80}")
        
        # Load data
        train_file = os.path.join(ALL_DATASETS_DIR, dataset_name, "train.csv")
        test_file = os.path.join(ALL_DATASETS_DIR, dataset_name, "test.csv")
        
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
        
        if quick_test:
            df_train = df_train.head(200)
            df_test = df_test.head(50)
        
        print(f"  Train samples: {len(df_train)}")
        print(f"  Test samples: {len(df_test)}")
        
        # Collect all unique molecules
        all_molecules = set(df_train['SMILES'].unique()) | set(df_test['SMILES'].unique())
        
        print(f"  Unique molecules: {len(all_molecules)}")
        
        # Featurize molecules
        print("\n  Featurizing molecules...")
        molecule_graphs = featurize_molecules(all_molecules, featurizer)
        
        print(f"  Featurized {len(molecule_graphs)} molecules")
        
        # Get feature dimensions from first graph
        sample_graph = next(iter(molecule_graphs.values()))
        node_dim = sample_graph.ndata['feat'].shape[1]
        edge_dim = sample_graph.edata['feat'].shape[1] if 'feat' in sample_graph.edata else None
        print(f"  Node feature dim: {node_dim}, Edge feature dim: {edge_dim}")
        
        # Create datasets
        train_dataset = AqueousSolubilityDataset(df_train, molecule_graphs)
        test_dataset = AqueousSolubilityDataset(df_test, molecule_graphs)
        
        # Create validation split from training data
        val_size = int(0.1 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        
        dataset_results = {}
        
        for gnn_type in GNN_MODELS:
            print(f"\n  --- GNN Type: {gnn_type} ---")
            
            results = {'rmse': [], 'r2': [], 'time': []}
            
            for seed in SEEDS:
                print(f"    Seed {seed}...", end=' ')
                
                # Set seeds
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                # Split train/val
                train_subset, val_subset = torch.utils.data.random_split(
                    train_dataset, [train_size, val_size],
                    generator=torch.Generator().manual_seed(seed)
                )
                
                train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, 
                                         shuffle=True, collate_fn=collate_fn)
                val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, 
                                       shuffle=False, collate_fn=collate_fn)
                test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                                        shuffle=False, collate_fn=collate_fn)
                
                # Create model
                model = SingleGNNModel(
                    gnn_type=gnn_type,
                    node_in_dim=node_dim,
                    edge_in_dim=edge_dim,
                    hidden_dim=HIDDEN_DIM,
                    num_layers=NUM_GNN_LAYERS,
                    mlp_hidden=MLP_HIDDEN_DIM
                )
                
                # Train
                epochs = 10 if quick_test else EPOCHS
                start_time = time.time()
                model = train_model(model, train_loader, val_loader, 
                                   epochs, LR, PATIENCE, DEVICE)
                training_time = time.time() - start_time
                results['time'].append(training_time)
                
                # Evaluate
                rmse, r2 = evaluate_model(model, test_loader, DEVICE)
                results['rmse'].append(rmse)
                results['r2'].append(r2)
                
                print(f"RMSE={rmse:.4f}, R²={r2:.4f}")
                
                # Save model
                model_dir = os.path.join(OUTPUT_DIR, dataset_name, "gnn_models", gnn_type)
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f"model_seed{seed}.pt")
                torch.save(model.state_dict(), model_path)
            
            dataset_results[gnn_type] = results
        
        all_results[dataset_name] = dataset_results
    
    # Save summary
    save_results(all_results)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR}/")


def save_results(all_results):
    """Save aggregated results to JSON and CSV."""
    summary_data = []
    
    for dataset_name, model_results in all_results.items():
        for model_name, metrics in model_results.items():
            rmse_mean = np.mean(metrics['rmse'])
            rmse_std = np.std(metrics['rmse'])
            r2_mean = np.mean(metrics['r2'])
            r2_std = np.std(metrics['r2'])
            time_mean = np.mean(metrics['time'])
            time_std = np.std(metrics['time'])
            
            summary_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'RMSE_mean': rmse_mean,
                'RMSE_std': rmse_std,
                'R2_mean': r2_mean,
                'R2_std': r2_std,
                'Time_mean': time_mean,
                'Time_std': time_std,
                'Seeds': len(metrics['rmse'])
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, "gnn_benchmark_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved summary to: {csv_path}")
    
    # Save to JSON
    json_path = os.path.join(OUTPUT_DIR, "gnn_benchmark_results.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved full results to: {json_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("GNN BENCHMARK SUMMARY (Mean ± Std)")
    print("="*80)
    
    for dataset_name in DATASETS:
        print(f"\n{dataset_name}:")
        print("-"*80)
        dataset_df = summary_df[summary_df['Dataset'] == dataset_name].sort_values('RMSE_mean')
        for _, row in dataset_df.iterrows():
            print(f"  {row['Model']:<10} RMSE: {row['RMSE_mean']:.4f} ± {row['RMSE_std']:.4f}  "
                  f"R²: {row['R2_mean']:.4f} ± {row['R2_std']:.4f}  "
                  f"Time: {row['Time_mean']:.2f}s ± {row['Time_std']:.2f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GNN Baselining for Aqueous Solubility Prediction (Regime-I)")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test with subset of data")
    args = parser.parse_args()
    
    main(quick_test=args.quick_test)
