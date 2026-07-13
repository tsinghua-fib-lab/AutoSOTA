#!/usr/bin/env python3
"""
GNN Baselining for Solubility Prediction
=========================================
Implements MPNN, GAT, GCN, and GIN models with dual solute/solvent architecture.
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

DATASETS = ["bigsol1.0", "bigsol2.0", "leeds"]
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
# DUAL GNN MODEL
# ============================================================================

class DualGNNModel(nn.Module):
    """
    Dual GNN model for solute-solvent pair prediction.
    Separate GNN encoders for solute and solvent, concatenate embeddings
    with global features (temperature), then feed to MLP predictor.
    """
    def __init__(self, gnn_type, node_in_dim, edge_in_dim=None, 
                 hidden_dim=64, num_layers=3, mlp_hidden=128, num_global_feats=4):
        super().__init__()
        self.gnn_type = gnn_type
        
        # Create GNN encoders for solute and solvent
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
            emb_dim = hidden_dim * 2  # Set2Set output
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # MLP predictor
        # Input: solute_emb + solvent_emb + global_feats
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
        """
        Args:
            solute_g: DGL graph for solute
            solvent_g: DGL graph for solvent
            global_feats: Tensor of global features [batch_size, num_global_feats]
                          Contains: Temperature, T_inv, T_red (if available), etc.
        """
        # Get node features
        solute_feats = solute_g.ndata['feat']
        solvent_feats = solvent_g.ndata['feat']
        
        # Get edge features if available (for MPNN)
        solute_edge = solute_g.edata.get('feat', None)
        solvent_edge = solvent_g.edata.get('feat', None)
        
        # Encode molecules
        if self.gnn_type == "MPNN":
            solute_emb = self.solute_encoder(solute_g, solute_feats, solute_edge)
            solvent_emb = self.solvent_encoder(solvent_g, solvent_feats, solvent_edge)
        else:
            solute_emb = self.solute_encoder(solute_g, solute_feats)
            solvent_emb = self.solvent_encoder(solvent_g, solvent_feats)
        
        # Concatenate embeddings with global features
        combined = torch.cat([solute_emb, solvent_emb, global_feats], dim=1)
        
        # Predict
        return self.mlp(combined)


# ============================================================================
# DATASET
# ============================================================================

class SolubilityDataset(Dataset):
    """Dataset for solute-solvent solubility prediction."""
    def __init__(self, df, solute_graphs, solvent_graphs):
        self.df = df.reset_index(drop=True)
        self.solute_graphs = solute_graphs
        self.solvent_graphs = solvent_graphs
        
        # Filter out rows with missing graphs
        valid_mask = []
        for i in range(len(self.df)):
            sol = self.df.loc[i, 'Solute']
            solv = self.df.loc[i, 'Solvent']
            valid_mask.append(sol in solute_graphs and solv in solvent_graphs)
        
        self.df = self.df[valid_mask].reset_index(drop=True)
        print(f"  Dataset size after filtering: {len(self.df)}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        solute_g = self.solute_graphs[row['Solute']]
        solvent_g = self.solvent_graphs[row['Solvent']]
        
        # Global features: Temperature, 1/T, T^2
        T = row['Temperature']
        global_feats = torch.tensor([
            T / 300.0,           # Normalized temperature
            1000.0 / T,          # Inverse temperature
            (T / 300.0) ** 2,    # Temperature squared
            np.log(T / 300.0)    # Log temperature
        ], dtype=torch.float32)
        
        target = torch.tensor([row['LogS']], dtype=torch.float32)
        
        return solute_g, solvent_g, global_feats, target


def collate_fn(batch):
    """Collate function for batching DGL graphs."""
    solute_gs, solvent_gs, global_feats, targets = zip(*batch)
    
    batched_solute = dgl.batch(solute_gs)
    batched_solvent = dgl.batch(solvent_gs)
    global_feats = torch.stack(global_feats)
    targets = torch.stack(targets)
    
    return batched_solute, batched_solvent, global_feats, targets


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
        for solute_g, solvent_g, global_feats, targets in train_loader:
            solute_g = solute_g.to(device)
            solvent_g = solvent_g.to(device)
            global_feats = global_feats.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            preds = model(solute_g, solvent_g, global_feats)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for solute_g, solvent_g, global_feats, targets in val_loader:
                solute_g = solute_g.to(device)
                solvent_g = solvent_g.to(device)
                global_feats = global_feats.to(device)
                targets = targets.to(device)
                
                preds = model(solute_g, solvent_g, global_feats)
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
        for solute_g, solvent_g, global_feats, targets in test_loader:
            solute_g = solute_g.to(device)
            solvent_g = solvent_g.to(device)
            global_feats = global_feats.to(device)
            
            preds = model(solute_g, solvent_g, global_feats)
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
    print("GNN BASELINING FOR SOLUBILITY PREDICTION")
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
        all_solutes = set(df_train['Solute'].unique()) | set(df_test['Solute'].unique())
        all_solvents = set(df_train['Solvent'].unique()) | set(df_test['Solvent'].unique())
        
        print(f"  Unique solutes: {len(all_solutes)}")
        print(f"  Unique solvents: {len(all_solvents)}")
        
        # Featurize molecules
        print("\n  Featurizing molecules...")
        solute_graphs = featurize_molecules(all_solutes, featurizer)
        solvent_graphs = featurize_molecules(all_solvents, featurizer)
        
        print(f"  Featurized {len(solute_graphs)} solutes, {len(solvent_graphs)} solvents")
        
        # Get feature dimensions from first graph
        sample_graph = next(iter(solute_graphs.values()))
        node_dim = sample_graph.ndata['feat'].shape[1]
        edge_dim = sample_graph.edata['feat'].shape[1] if 'feat' in sample_graph.edata else None
        print(f"  Node feature dim: {node_dim}, Edge feature dim: {edge_dim}")
        
        # Create datasets
        train_dataset = SolubilityDataset(df_train, solute_graphs, solvent_graphs)
        test_dataset = SolubilityDataset(df_test, solute_graphs, solvent_graphs)
        
        # Create validation split from training data
        val_size = int(0.1 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        
        dataset_results = {}
        
        for gnn_type in GNN_MODELS:
            print(f"\n  --- GNN Type: {gnn_type} ---")
            
            results = {'rmse': [], 'r2': []}
            
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
                model = DualGNNModel(
                    gnn_type=gnn_type,
                    node_in_dim=node_dim,
                    edge_in_dim=edge_dim,
                    hidden_dim=HIDDEN_DIM,
                    num_layers=NUM_GNN_LAYERS,
                    mlp_hidden=MLP_HIDDEN_DIM,
                    num_global_feats=4
                )
                
                # Train
                epochs = 10 if quick_test else EPOCHS
                model = train_model(model, train_loader, val_loader, 
                                   epochs, LR, PATIENCE, DEVICE)
                
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
            
            summary_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'RMSE_mean': rmse_mean,
                'RMSE_std': rmse_std,
                'R2_mean': r2_mean,
                'R2_std': r2_std,
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
                  f"R²: {row['R2_mean']:.4f} ± {row['R2_std']:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GNN Baselining for Solubility Prediction")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test with subset of data")
    args = parser.parse_args()
    
    main(quick_test=args.quick_test)
