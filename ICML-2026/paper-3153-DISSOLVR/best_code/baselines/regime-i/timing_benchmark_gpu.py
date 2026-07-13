#!/usr/bin/env python3
"""
GPU Timing Benchmark for Regime-I Methods
Run on Lightning.ai with T4/A10 GPU

Methods timed (EXACT original implementations):
- SolubNet (Original mtMolDes GCNNet, 500 epochs)
- Ulrich (DeepChem GraphConvModel, 130 epochs, graph_conv_layers=[64,128])
- GIN, GCN, GAT, MPNN (DGL, 50 epochs)
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import random
import torch
from tqdm import tqdm

# Silence warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_WARNINGS"] = "0"
warnings.filterwarnings("ignore")

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# ============================================================================
# CONFIG
# ============================================================================

SEED = 123
DATASET_NAME = "aqsoldb"
TRAIN_PATH = f"all_datasets/{DATASET_NAME}/train.csv"
TEST_PATH = f"all_datasets/{DATASET_NAME}/test.csv"
SMILES_COL = "SMILES"
TARGET_COL = "LogS"

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS = []

def log_result(method_name, train_time, inference_time=None, device_used=None, notes=""):
    """Log timing result."""
    dev = device_used if device_used else DEVICE
    n_test = pd.read_csv(TEST_PATH).shape[0]
    RESULTS.append({
        "Method": method_name,
        "Train Time (s)": train_time,
        "Inference Time (s)": inference_time,
        "ms/sample": (inference_time * 1000 / n_test) if inference_time else None,
        "Device": dev.upper(),
        "Notes": notes
    })
    inf_str = f", Inference: {inference_time:.3f}s" if inference_time else ""
    print(f"  ✓ {method_name} [{dev.upper()}]: Train={train_time:.2f}s{inf_str}")


# ============================================================================
# 1. Ulrich (DeepChem GraphConvModel) - EXACT original params
# ============================================================================

def time_ulrich():
    """Train Ulrich GCN via DeepChem - using MyGraphConvModel."""
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        import deepchem as dc
        from deepchem.feat.mol_graphs import ConvMol
        from deepchem.models import KerasModel
        from mygraphconvmodel import MyGraphConvModel
    except Exception as e:
        print(f"  ✗ Ulrich: import failed - {type(e).__name__}: {e}")
        return
    
    # Original Ulrich hyperparameters
    BATCH_SIZE = 50
    NUM_EPOCHS = 130
    NEURONS_LAYER1 = 64
    NEURONS_LAYER2 = 128
    DROPOUT = 0.1
    LEARNING_RATE = 3e-4
    
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    # Pre-filter invalid molecules (deepchem version issue)
    from rdkit import Chem
    print("  Pre-filtering valid molecules...")
    train_df = train_df[train_df[SMILES_COL].apply(lambda s: Chem.MolFromSmiles(s) is not None)]
    test_df = test_df[test_df[SMILES_COL].apply(lambda s: Chem.MolFromSmiles(s) is not None)]
    print(f"  After filter - Train: {len(train_df)}, Test: {len(test_df)}")
    
    print("  Featurizing molecules with CSVLoader...")
    featurizer = dc.feat.ConvMolFeaturizer(use_chirality=True)
    
    # Use CSVLoader
    train_df.to_csv("_tmp_train.csv", index=False)
    test_df.to_csv("_tmp_test.csv", index=False)
    
    loader = dc.data.CSVLoader(tasks=[TARGET_COL], feature_field=SMILES_COL, featurizer=featurizer)
    train_dataset = loader.featurize("_tmp_train.csv")
    test_dataset = loader.featurize("_tmp_test.csv")
    
    os.remove("_tmp_train.csv")
    os.remove("_tmp_test.csv")
    
    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Data generator for training
    def data_generator(dataset, batch_size, epochs=1):
        for _ in range(epochs):
            for X_b, y_b, w_b, ids_b in dataset.iterbatches(
                batch_size=batch_size, deterministic=True, pad_batches=True
            ):
                multiConvMol = ConvMol.agglomerate_mols(X_b)
                inputs = [
                    multiConvMol.get_atom_features(),
                    multiConvMol.deg_slice,
                    np.array(multiConvMol.membership)
                ]
                for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
                    inputs.append(multiConvMol.get_deg_adjacency_lists()[i])
                yield (inputs, [y_b], [w_b])
    
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    # Create model with original architecture
    model = KerasModel(
        MyGraphConvModel(
            batch_size=BATCH_SIZE,
            neuronslayer1=NEURONS_LAYER1,
            neuronslayer2=NEURONS_LAYER2,
            dropout=DROPOUT
        ),
        loss=dc.models.losses.L1Loss(),
        learning_rate=LEARNING_RATE,
    )
    
    start = time.time()
    
    print(f"  Training for {NUM_EPOCHS} epochs...")
    for epoch in tqdm(range(NUM_EPOCHS), desc="  Ulrich"):
        model.fit_generator(data_generator(train_dataset, BATCH_SIZE, epochs=1))
    
    train_time = time.time() - start
    
    # Inference timing
    start = time.time()
    pred = model.predict_on_generator(data_generator(test_dataset, BATCH_SIZE))
    inf_time = time.time() - start
    
    # DeepChem uses GPU via TensorFlow if available
    device_used = "gpu" if tf.config.list_physical_devices('GPU') else "cpu"
    log_result("Ulrich (GCN)", train_time, inf_time, device_used, f"MyGraphConvModel, {NUM_EPOCHS} epochs")


# ============================================================================
# 2. SolubNet (Original mtMolDes GCNNet)
# ============================================================================

def time_solubnet():
    """Train SolubNet using ORIGINAL mtMolDes.model.GCNNet architecture."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SolubNetD'))
    
    try:
        from mtMolDes import model, Utility
        import torch as th
        from torch.optim.lr_scheduler import ReduceLROnPlateau
    except ImportError:
        print("  ✗ SolubNet: mtMolDes not found in SolubNetD folder")
        return
    
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    device = th.device(DEVICE)
    
    # EXACT original hyperparameters from SolubNet paper
    NUM_FEATURES = 4
    NUM_LABELS = 1
    FEATURE_STR = 'h'
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    MAX_EPOCHS = 500
    
    print("  Converting molecules to graphs (original SolubNet featurization)...")
    train_data = []
    for idx, row in train_df.iterrows():
        try:
            graph = Utility.ParseSMILES(row['SMILES'], NUM_FEATURES, FEATURE_STR, device)
            train_data.append([row['SMILES'], graph, float(row['LogS'])])
        except:
            continue
    
    test_data = []
    for idx, row in test_df.iterrows():
        try:
            graph = Utility.ParseSMILES(row['SMILES'], NUM_FEATURES, FEATURE_STR, device)
            test_data.append(graph)
        except:
            continue
    
    print(f"  Train: {len(train_data)}/{len(train_df)}, Test: {len(test_data)}/{len(test_df)}")
    
    # Initialize model - EXACT original architecture
    solubnet = model.GCNNet(NUM_FEATURES, NUM_LABELS, FEATURE_STR)
    solubnet.to(device)
    
    th.manual_seed(SEED)
    np.random.seed(SEED)
    
    optimizer = th.optim.Adam(solubnet.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=3, min_lr=1e-6)
    criterion = th.nn.MSELoss()
    
    # EXACT original get_predictions function
    def get_predictions(graphs, labels, net):
        predictions = th.zeros(len(graphs), device=device)
        for i in range(len(graphs)):
            predictions[i] = th.sum(net(graphs[i]), dim=0)
        return predictions, labels
    
    num_samples = len(train_data)
    batch_idx = [[i * BATCH_SIZE, min((i + 1) * BATCH_SIZE, num_samples)] 
                 for i in range((num_samples + BATCH_SIZE - 1) // BATCH_SIZE)]
    
    start = time.time()
    
    for epoch in tqdm(range(MAX_EPOCHS), desc="  SolubNet"):
        solubnet.train()
        epoch_loss = 0
        
        for idx0, idx1 in batch_idx:
            batch_graphs = [train_data[i][1] for i in range(idx0, idx1)]
            batch_labels = th.tensor([train_data[i][2] for i in range(idx0, idx1)], dtype=th.float32, device=device)
            
            y_pred, y_true = get_predictions(batch_graphs, batch_labels, solubnet)
            
            # EXACT original loss: RMSE - 0.1 * R²
            rmse = th.sqrt(criterion(y_pred, y_true))
            target_mean = th.mean(y_true)
            ss_tot = th.sum((y_true - target_mean) ** 2)
            ss_res = th.sum((y_true - y_pred) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            loss = rmse - 0.1 * r2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step(epoch_loss / len(batch_idx))
        
        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1}/{MAX_EPOCHS}, Loss: {epoch_loss/len(batch_idx):.4f}")
    
    train_time = time.time() - start
    
    # Inference timing
    solubnet.eval()
    start = time.time()
    with th.no_grad():
        for graph in test_data:
            _ = th.sum(solubnet(graph), dim=0)
    inf_time = time.time() - start
    
    log_result("SolubNet", train_time, inf_time, DEVICE, f"Original GCNNet, {MAX_EPOCHS} epochs")


# ============================================================================
# 3. GNN Models (GIN, GCN, GAT, MPNN) using DGL
# ============================================================================

def time_gnn_models():
    """Train GNN models: GIN, GCN, GAT, MPNN."""
    try:
        import dgl
        from dgl.nn import GraphConv, GATConv, GINConv
    except ImportError:
        print("  ✗ GNN models: dgl not installed")
        return
    
    from rdkit import Chem
    
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    # Standard molecular graph featurization
    def smiles_to_dgl(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return None
        
        # 30-dim node features (one-hot atomic number + properties)
        node_feats = torch.zeros(num_atoms, 30)
        for i, atom in enumerate(mol.GetAtoms()):
            node_feats[i, atom.GetAtomicNum() % 30] = 1
            node_feats[i, 20] = atom.GetDegree()
            node_feats[i, 21] = atom.GetIsAromatic()
        
        src, dst = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src.extend([i, j])
            dst.extend([j, i])
        
        if len(src) == 0:
            src, dst = [0], [0]
        
        g = dgl.graph((src, dst), num_nodes=num_atoms)
        g = dgl.add_self_loop(g)
        g.ndata['feat'] = node_feats
        return g
    
    print("  Featurizing train molecules...")
    graphs, labels = [], []
    for i, row in train_df.iterrows():
        g = smiles_to_dgl(row[SMILES_COL])
        if g is not None:
            graphs.append(g)
            labels.append(row[TARGET_COL])
    
    print("  Featurizing test molecules...")
    test_graphs = []
    for i, row in test_df.iterrows():
        g = smiles_to_dgl(row[SMILES_COL])
        if g is not None:
            test_graphs.append(g)
    
    labels = torch.tensor(labels, dtype=torch.float32)
    device = torch.device(DEVICE)
    
    print(f"  Train: {len(graphs)}, Test: {len(test_graphs)}")
    
    # GCN Model (2-layer)
    class GCNModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GraphConv(30, 64, activation=torch.relu)
            self.conv2 = GraphConv(64, 64, activation=torch.relu)
            self.fc = torch.nn.Linear(64, 1)
        
        def forward(self, g, h):
            h = self.conv1(g, h)
            h = self.conv2(g, h)
            g.ndata['h'] = h
            return self.fc(dgl.mean_nodes(g, 'h'))
    
    # GIN Model (2-layer with MLPs)
    class GINModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            mlp1 = torch.nn.Sequential(torch.nn.Linear(30, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64))
            mlp2 = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64))
            self.conv1 = GINConv(mlp1, learn_eps=True)
            self.conv2 = GINConv(mlp2, learn_eps=True)
            self.fc = torch.nn.Linear(64, 1)
        
        def forward(self, g, h):
            h = self.conv1(g, h)
            h = self.conv2(g, h)
            g.ndata['h'] = h
            return self.fc(dgl.sum_nodes(g, 'h'))
    
    # GAT Model (2-layer with 4 attention heads)
    class GATModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GATConv(30, 16, num_heads=4)
            self.conv2 = GATConv(64, 16, num_heads=4)
            self.fc = torch.nn.Linear(64, 1)
        
        def forward(self, g, h):
            h = torch.relu(self.conv1(g, h).flatten(1))
            h = torch.relu(self.conv2(g, h).flatten(1))
            g.ndata['h'] = h
            return self.fc(dgl.mean_nodes(g, 'h'))
    
    # MPNN (3-layer GCN with message passing)
    class MPNNModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GraphConv(30, 64, activation=torch.relu)
            self.conv2 = GraphConv(64, 64, activation=torch.relu)
            self.conv3 = GraphConv(64, 64, activation=torch.relu)
            self.fc = torch.nn.Linear(64, 1)
        
        def forward(self, g, h):
            h = self.conv1(g, h)
            h = self.conv2(g, h)
            h = self.conv3(g, h)
            g.ndata['h'] = h
            return self.fc(dgl.mean_nodes(g, 'h'))
    
    models = {
        "GCN": GCNModel,
        "GIN": GINModel,
        "GAT": GATModel,
        "MPNN": MPNNModel,
    }
    
    NUM_EPOCHS = 50
    
    for name, ModelClass in models.items():
        print(f"  Training {name}...")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        
        model = ModelClass().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        start = time.time()
        for epoch in range(NUM_EPOCHS):
            model.train()
            for i, g in enumerate(graphs):
                g = g.to(device)
                h = g.ndata['feat'].to(device)
                y = labels[i:i+1].to(device)
                
                pred = model(g, h)
                loss = criterion(pred, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        train_time = time.time() - start
        
        # Inference timing
        model.eval()
        start = time.time()
        with torch.no_grad():
            for g in test_graphs:
                g = g.to(device)
                h = g.ndata['feat'].to(device)
                _ = model(g, h)
        inf_time = time.time() - start
        
        log_result(name, train_time, inf_time, DEVICE, f"DGL, {NUM_EPOCHS} epochs")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("GPU TIMING BENCHMARK (Regime-I)")
    print("=" * 70)
    print(f"Dataset: {DATASET_NAME}")
    print(f"PyTorch Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)
    
    print("\n[1/3] Ulrich (DeepChem GCN)...")
    time_ulrich()
    
    print("\n[2/3] SolubNet...")
    time_solubnet()
    
    print("\n[3/3] GNN Models (GIN, GCN, GAT, MPNN)...")
    time_gnn_models()
    
    # Print results
    print("\n" + "=" * 100)
    print("TIMING RESULTS (GPU)")
    print("=" * 100)
    
    if not RESULTS:
        print("No results collected! Make sure dependencies are installed:")
        print("  pip install deepchem dgl torch")
        print("  # SolubNetD folder must contain mtMolDes package")
        return
    
    print(f"{'Method':20} | {'Device':6} | {'Train (s)':10} | {'Inference (s)':12} | {'ms/sample':10} | Notes")
    print("-" * 100)
    
    results_df = pd.DataFrame(RESULTS)
    results_df = results_df.sort_values("Train Time (s)")
    
    for _, row in results_df.iterrows():
        inf_s = f"{row['Inference Time (s)']:.4f}" if row['Inference Time (s)'] else "N/A"
        ms = f"{row['ms/sample']:.4f}" if row['ms/sample'] else "N/A"
        notes = row['Notes'] if row['Notes'] else ""
        print(f"{row['Method']:20} | {row['Device']:6} | {row['Train Time (s)']:10.2f} | {inf_s:>12} | {ms:>10} | {notes}")
    
    results_df.to_csv("timing_results_gpu.csv", index=False)
    print(f"\nSaved to timing_results_gpu.csv")
    print("=" * 100)


if __name__ == "__main__":
    main()
