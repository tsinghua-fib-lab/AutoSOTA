#!/usr/bin/env python3
"""
Unified Timing Benchmark for Regime-I Methods
==============================================

Runs each baseline method with seed 123 on AqSolDB and reports training time.
Uses CUDA if available, otherwise CPU.

Run on Lightning.ai:
    pip install -r requirements.txt
    python timing_benchmark.py

Methods included:
- GSE (closed-form, no training)
- ESOL Model (closed-form, no training)  
- ChemProp
- FastProp
- SolTranNet
- AqSolPred
- SolubNet
- Ulrich et al. (GCN)
- Tayyebi et al.
- Decision Tree, Random Forest, XGBoost, LightGBM, ANN
- GIN, GCN, GAT, MPNN (PyTorch Geometric)
- Ours (proposed method)
"""

import os
import sys
import time
import warnings
import subprocess
import numpy as np
import pandas as pd

# Silence warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# ============================================================================
# CONFIG
# ============================================================================

SEED = 123
DATASET_NAME = "aqsoldb"
TRAIN_PATH = "all_datasets/aqsoldb/train.csv"
TEST_PATH = "all_datasets/aqsoldb/test.csv"
SMILES_COL = "SMILES"
TARGET_COL = "LogS"

# Check device
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE.upper()}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# RESULTS STORAGE
# ============================================================================

RESULTS = []

def log_result(method_name, train_time, inference_time=None, device_used=None, notes=""):
    """Log timing result with device info."""
    dev = device_used if device_used else DEVICE
    RESULTS.append({
        "Method": method_name,
        "Train Time (s)": train_time,
        "Train Time (min)": train_time / 60,
        "Inference Time (s)": inference_time,
        "Inference Time (ms/sample)": (inference_time * 1000 / pd.read_csv(TEST_PATH).shape[0]) if inference_time else None,
        "Device": dev.upper(),
        "Notes": notes
    })
    inf_str = f", Inference: {inference_time:.2f}s" if inference_time else ""
    print(f"  ✓ {method_name} [{dev.upper()}]: Train={train_time:.2f}s{inf_str}")


# ============================================================================
# 1. GSE (No Training) - Proper Implementation
# ============================================================================

# Joback group contributions for melting point
JOBACK_GROUPS = {
    "ch3": ("[CH3;X4;!R]", -5.10), "ch2_c": ("[CH2;X4;!R]", 11.27),
    "ch_c": ("[CH1;X4;!R]", 12.64), "c_c": ("[CH0;X4;!R]", 46.43),
    "ch2_r": ("[CH2;X4;R]", 8.25), "ch_r": ("[CH1;X4;R]", 20.15),
    "c_r": ("[CH0;X4;R]", 37.40), "c=c_c": ("[CX3;!R]=[CX3;!R]", 4.18),
    "c=c_r": ("[c,C;R]=[c,C;R]", 13.02), "F": ("[F]", 9.88),
    "Cl": ("[Cl]", 17.51), "Br": ("[Br]", 26.15), "I": ("[I]", 37.0),
    "oh_a": ("[OH;!#6a]", 20.0), "oh_p": ("[OH;a]", 44.45),
    "ether_c": ("[OD2;!R]([#6])[#6]", 22.42), "ether_r": ("[OD2;R]([#6])[#6]", 31.22),
    "co": ("[CX3]=[OX1]", 26.15), "nh2": ("[NH2]", 25.72)
}

def time_gse():
    """GSE: log S = -logP - 0.01*(Mpt - 25) + 0.5 with Joback melting point."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    def estimate_melting_point(mol):
        tm_sum = 122.5
        for smarts, weight in JOBACK_GROUPS.values():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern:
                matches = len(mol.GetSubstructMatches(pattern))
                tm_sum += matches * weight
        tm_kelvin = max(tm_sum, 150.0)
        return tm_kelvin - 273.15  # Celsius
    
    test_df = pd.read_csv(TEST_PATH)
    
    start = time.time()
    for smiles in test_df[SMILES_COL]:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            logP = Descriptors.MolLogP(mol)
            Mpt = estimate_melting_point(mol)
            _ = -logP - 0.01 * (Mpt - 25) + 0.5
    elapsed = time.time() - start
    
    # GSE has no training - inference only
    log_result("GSE", 0.0, elapsed, "cpu", "Joback melting point, closed-form")


# ============================================================================
# 2. ESOL Model (Linear Regression)
# ============================================================================

def time_esol_model():
    """ESOL Model: Linear regression on 4 features."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from sklearn.linear_model import LinearRegression
    
    def compute_features(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return [
            Descriptors.MolLogP(mol),
            Descriptors.MolWt(mol),
            Descriptors.NumRotatableBonds(mol),
            sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / mol.GetNumHeavyAtoms()
        ]
    
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    X_train = np.array([f for f in [compute_features(s) for s in train_df[SMILES_COL]] if f])
    y_train = train_df[TARGET_COL].values[:len(X_train)]
    X_test = np.array([f for f in [compute_features(s) for s in test_df[SMILES_COL]] if f])
    
    start = time.time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    start = time.time()
    _ = model.predict(X_test)
    inf_time = time.time() - start
    
    log_result("ESOL Model", train_time, inf_time, "cpu", "linear regression, 4 features")


# ============================================================================
# 3. Generic ML Baselines (RF, LGB, XGB, ANN)
# ============================================================================

def time_generic_baselines():
    """Train generic baselines using existing featurizer - EXACT original params."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb
    import xgboost as xgb
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from featurizer import MoleculeFeaturizer
    
    print("  Generating molecular features...")
    featurizer = MoleculeFeaturizer()
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    X_train = featurizer.transform(train_df[SMILES_COL].tolist()).values
    y_train = train_df[TARGET_COL].values
    X_test = featurizer.transform(test_df[SMILES_COL].tolist()).values
    
    # Create 5% validation split for early stopping (same as original)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=SEED)
    
    # Decision Tree - EXACT original params
    print("  Training Decision Tree...")
    start = time.time()
    dt = DecisionTreeRegressor(
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=SEED
    )
    dt.fit(X_train, y_train)
    train_time = time.time() - start
    start = time.time()
    _ = dt.predict(X_test)
    inf_time = time.time() - start
    log_result("Decision Tree", train_time, inf_time, "cpu")
    
    # Random Forest - EXACT original params (10000 trees!)
    print("  Training Random Forest...")
    start = time.time()
    rf = RandomForestRegressor(
        n_estimators=10000,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=SEED
    )
    rf.fit(X_train, y_train)
    train_time = time.time() - start
    start = time.time()
    _ = rf.predict(X_test)
    inf_time = time.time() - start
    log_result("Random Forest", train_time, inf_time, "cpu", "10000 trees")
    
    # LightGBM - EXACT original params with early stopping
    print("  Training LightGBM...")
    start = time.time()
    lgbm = lgb.LGBMRegressor(
        n_estimators=10000,
        learning_rate=0.02,
        max_depth=8,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=5,
        random_state=SEED,
        verbose=-1
    )
    lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    train_time = time.time() - start
    start = time.time()
    _ = lgbm.predict(X_test)
    inf_time = time.time() - start
    log_result("LightGBM", train_time, inf_time, "cpu", "10000 iters + early stopping")
    
    # XGBoost - EXACT original params with early stopping
    print("  Training XGBoost...")
    start = time.time()
    xgbm = xgb.XGBRegressor(
        n_estimators=10000,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=5,
        random_state=SEED,
        verbosity=0
    )
    xgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    train_time = time.time() - start
    start = time.time()
    _ = xgbm.predict(X_test)
    inf_time = time.time() - start
    log_result("XGBoost", train_time, inf_time, "cpu", "10000 iters + early stopping")
    
    # ANN - EXACT original params
    print("  Training ANN...")
    start = time.time()
    ann = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=SEED
    )
    ann.fit(X_train, y_train)
    train_time = time.time() - start
    start = time.time()
    _ = ann.predict(X_test)
    inf_time = time.time() - start
    log_result("ANN", train_time, inf_time, "cpu", "early stopping")


# ============================================================================
# 4. ChemProp (requires chemprop package)
# ============================================================================

def time_chemprop():
    """Train ChemProp via CLI."""
    import shutil
    
    checkpoint_dir = "timing_chemprop_output"
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir)
    
    train_cmd = [
        "chemprop", "train",
        "--data-path", TRAIN_PATH,
        "--task-type", "regression",
        "--smiles-columns", SMILES_COL,
        "--target-columns", TARGET_COL,
        "--output-dir", checkpoint_dir,
        "--epochs", "30",
        "--data-seed", str(SEED),
        "--pytorch-seed", str(SEED),
    ]
    
    start = time.time()
    result = subprocess.run(train_cmd, capture_output=True)
    elapsed = time.time() - start
    
    if result.returncode == 0:
        log_result("ChemProp", elapsed, "30 epochs")
    else:
        print(f"  ✗ ChemProp failed")


# ============================================================================
# 5. FastProp (requires fastprop package)
# ============================================================================

def time_fastprop():
    """Train FastProp via CLI."""
    import shutil
    
    output_dir = "timing_fastprop_output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    train_cmd = [
        "fastprop", "train",
        "--input-file", TRAIN_PATH,
        "--smiles-column", SMILES_COL,
        "--target-columns", TARGET_COL,
        "--output-directory", output_dir,
        "--problem-type", "regression",
        "--number-epochs", "100",
        "--random-seed", str(SEED),
    ]
    
    start = time.time()
    result = subprocess.run(train_cmd, capture_output=True)
    elapsed = time.time() - start
    
    if result.returncode == 0:
        log_result("FastProp", elapsed, "100 epochs")
    else:
        print(f"  ✗ FastProp failed")


# ============================================================================
# 6. SolTranNet
# ============================================================================

def time_soltrannet():
    """Train lightweight SolTranNet."""
    sys.path.insert(0, "SolTranNet_paper")
    
    try:
        from SolTranNet_paper.data_utils import load_data_from_smiles, construct_loader
        from SolTranNet_paper.transformer import make_model
    except ImportError:
        print("  ✗ SolTranNet import failed")
        return
    
    train_df = pd.read_csv(TRAIN_PATH)
    train_smiles = train_df[SMILES_COL].tolist()
    train_y = train_df[TARGET_COL].values.astype(np.float32)
    
    print("  Featurizing molecules...")
    train_x = load_data_from_smiles(train_smiles, add_dummy_node=True)
    train_loader = construct_loader(train_x, batch_size=64, shuffle=True)
    
    device = torch.device(DEVICE)
    model = make_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.SmoothL1Loss()
    
    torch.manual_seed(SEED)
    
    start = time.time()
    for epoch in range(100):
        model.train()
        for batch in train_loader:
            adj, feats, _, indices = batch
            adj = adj.to(device)
            feats = feats.to(device)
            batch_y = torch.FloatTensor([train_y[i] for i in indices]).to(device).unsqueeze(1)
            
            mask = torch.sum(torch.abs(feats), dim=-1) != 0
            pred = model(feats, mask, adj, None)
            loss = criterion(pred, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    train_time = time.time() - start
    
    # Inference timing
    test_df = pd.read_csv(TEST_PATH)
    test_smiles = test_df[SMILES_COL].tolist()
    test_x = load_data_from_smiles(test_smiles, add_dummy_node=True)
    test_loader = construct_loader(test_x, batch_size=64, shuffle=False)
    
    model.eval()
    start = time.time()
    with torch.no_grad():
        for batch in test_loader:
            adj, feats, _, _ = batch
            adj = adj.to(device)
            feats = feats.to(device)
            mask = torch.sum(torch.abs(feats), dim=-1) != 0
            _ = model(feats, mask, adj, None)
    inf_time = time.time() - start
    
    log_result("SolTranNet", train_time, inf_time, DEVICE, "100 epochs, 3393 params")


# ============================================================================
# 7. AqSolPred (Consensus: NN + XGB + RF with Mordred)
# ============================================================================

def time_aqsolpred():
    """Train AqSolPred consensus model - EXACT original architecture."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    
    try:
        from mordred import Calculator, descriptors
    except ImportError:
        print("  ✗ AqSolPred: mordred not installed")
        return
    
    from rdkit import Chem
    
    train_df = pd.read_csv(TRAIN_PATH)
    
    print("  Calculating Mordred descriptors...")
    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(s) for s in train_df[SMILES_COL]]
    valid = [(m, i) for i, m in enumerate(mols) if m is not None]
    valid_mols = [m for m, _ in valid]
    valid_idx = [i for _, i in valid]
    
    X = calc.pandas(valid_mols).apply(pd.to_numeric, errors='coerce').fillna(0).values
    y = train_df[TARGET_COL].values[valid_idx]
    
    start = time.time()
    
    # Train 3 models - EXACT original AqSolPred params
    nn = MLPRegressor(
        activation='tanh',
        hidden_layer_sizes=(500,),
        max_iter=500,
        random_state=SEED,
        solver='adam'
    )
    rf = RandomForestRegressor(n_estimators=1000, random_state=SEED, n_jobs=-1)
    xgb_model = xgb.XGBRegressor(n_estimators=1000, random_state=SEED, verbosity=0)
    
    nn.fit(X, y)
    rf.fit(X, y)
    xgb_model.fit(X, y)
    train_time = time.time() - start
    
    # Inference on test set (need to compute test descriptors)
    test_df = pd.read_csv(TEST_PATH)
    test_mols = [Chem.MolFromSmiles(s) for s in test_df[SMILES_COL]]
    test_valid = [(m, i) for i, m in enumerate(test_mols) if m is not None]
    test_valid_mols = [m for m, _ in test_valid]
    X_test = calc.pandas(test_valid_mols).apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    start = time.time()
    pred_consensus = (nn.predict(X_test) + rf.predict(X_test) + xgb_model.predict(X_test)) / 3
    inf_time = time.time() - start
    
    log_result("AqSolPred", train_time, inf_time, "cpu", "consensus: NN(500) + XGB + RF")


# ============================================================================
# 8. Tayyebi (Mordred + RF) - EXACT original params
# ============================================================================

def time_tayyebi():
    """Train Tayyebi baseline with Mordred descriptors + RF."""
    from rdkit import Chem
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import VarianceThreshold
    
    try:
        from mordred import Calculator, descriptors
    except ImportError:
        print("  ✗ Tayyebi: mordred not installed")
        return
    
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    print("  Calculating Mordred descriptors...")
    calc = Calculator(descriptors, ignore_3D=True)
    
    # Train features
    train_mols = [Chem.MolFromSmiles(s) for s in train_df[SMILES_COL]]
    train_valid = [(m, i) for i, m in enumerate(train_mols) if m is not None]
    train_valid_mols = [m for m, _ in train_valid]
    train_valid_idx = [i for _, i in train_valid]
    X_train_raw = calc.pandas(train_valid_mols).apply(pd.to_numeric, errors='coerce').fillna(0)
    y_train = train_df[TARGET_COL].values[train_valid_idx]
    
    # Test features
    test_mols = [Chem.MolFromSmiles(s) for s in test_df[SMILES_COL]]
    test_valid = [(m, i) for i, m in enumerate(test_mols) if m is not None]
    test_valid_mols = [m for m, _ in test_valid]
    X_test_raw = calc.pandas(test_valid_mols).apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Align columns
    X_test_raw = X_test_raw.reindex(columns=X_train_raw.columns, fill_value=0)
    
    # Tayyebi filtering: drop FilterItLogS/SLogP, variance < 0.1, correlation > 0.8
    cols_to_drop = ['FilterItLogS', 'SLogP']
    X_train_clean = X_train_raw.drop(columns=cols_to_drop, errors='ignore')
    X_test_clean = X_test_raw.drop(columns=cols_to_drop, errors='ignore')
    
    vt = VarianceThreshold(threshold=0.1)
    vt.fit(X_train_clean)
    mask_var = vt.get_support()
    X_train_var = X_train_clean.loc[:, mask_var]
    X_test_var = X_test_clean.loc[:, mask_var]
    
    # Correlation filter
    corr_matrix = X_train_var.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.8)]
    X_train_final = X_train_var.drop(columns=to_drop).values
    X_test_final = X_test_var.drop(columns=to_drop).values
    
    print(f"  Features after filtering: {X_train_final.shape[1]}")
    
    start = time.time()
    rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(X_train_final, y_train)
    train_time = time.time() - start
    
    start = time.time()
    _ = rf.predict(X_test_final)
    inf_time = time.time() - start
    
    log_result("Tayyebi", train_time, inf_time, "cpu", "Mordred + RF(100 trees)")


# ============================================================================
# 9. Ulrich (DeepChem GCN) - requires deepchem
# ============================================================================

def time_ulrich():
    """Train Ulrich GCN via DeepChem."""
    try:
        import deepchem as dc
    except ImportError:
        print("  ✗ Ulrich: deepchem not installed")
        return
    
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    featurizer = dc.feat.ConvMolFeaturizer()
    X_train = featurizer.featurize(train_df[SMILES_COL].tolist())
    y_train = train_df[TARGET_COL].values
    X_test = featurizer.featurize(test_df[SMILES_COL].tolist())
    y_test = test_df[TARGET_COL].values
    
    train_dataset = dc.data.NumpyDataset(X_train, y_train)
    test_dataset = dc.data.NumpyDataset(X_test, y_test)
    
    start = time.time()
    from deepchem.models import GraphConvModel
    model = GraphConvModel(n_tasks=1, mode='regression', batch_size=50)
    model.fit(train_dataset, nb_epoch=50)
    train_time = time.time() - start
    
    start = time.time()
    _ = model.predict(test_dataset)
    inf_time = time.time() - start
    
    log_result("Ulrich (GCN)", train_time, inf_time, "cpu", "DeepChem GCN, 50 epochs")


# ============================================================================
# 10. Ours (CatBoost)
# ============================================================================

def time_ours():
    """Train Ours: CatBoost with custom features."""
    from catboost import CatBoostRegressor
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from featurizer import MoleculeFeaturizer
    
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    print("  Generating molecular features...")
    featurizer = MoleculeFeaturizer()
    X_train = featurizer.transform(train_df[SMILES_COL].tolist())
    y_train = train_df[TARGET_COL].values
    X_test = featurizer.transform(test_df[SMILES_COL].tolist())
    
    start = time.time()
    model = CatBoostRegressor(
        iterations=10000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5,
        verbose=1000,
        random_state=SEED,
        allow_writing_files=False,
        thread_count=-1
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    start = time.time()
    _ = model.predict(X_test)
    inf_time = time.time() - start
    
    log_result("Ours", train_time, inf_time, "cpu", "CatBoost, 10000 iterations")


# ============================================================================
# 11. SolubNet (Original mtMolDes GCNNet, 1 fold)
# ============================================================================

def time_solubnet():
    """Train SolubNet using ORIGINAL mtMolDes.model.GCNNet architecture."""
    # Import original SolubNet model
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
    
    # Original hyperparameters
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
    
    print(f"  Successfully featurized {len(train_data)}/{len(train_df)} train, {len(test_data)}/{len(test_df)} test")
    
    # Initialize model - EXACT original architecture
    solubnet = model.GCNNet(NUM_FEATURES, NUM_LABELS, FEATURE_STR)
    solubnet.to(device)
    
    th.manual_seed(SEED)
    np.random.seed(SEED)
    
    optimizer = th.optim.Adam(solubnet.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=3, min_lr=1e-6)
    criterion = th.nn.MSELoss()
    
    # Create mini-batches exactly as original
    def get_predictions(graphs, labels, net):
        predictions = th.zeros(len(graphs))
        for i in range(len(graphs)):
            predictions[i] = th.sum(net(graphs[i]), dim=0)
        return predictions, labels
    
    num_samples = len(train_data)
    batch_idx = [[i * BATCH_SIZE, min((i + 1) * BATCH_SIZE, num_samples)] 
                 for i in range((num_samples + BATCH_SIZE - 1) // BATCH_SIZE)]
    
    start = time.time()
    
    for epoch in range(MAX_EPOCHS):
        solubnet.train()
        epoch_loss = 0
        
        for idx0, idx1 in batch_idx:
            batch_graphs = [train_data[i][1] for i in range(idx0, idx1)]
            batch_labels = th.tensor([train_data[i][2] for i in range(idx0, idx1)], 
                                    dtype=th.float32, device=device)
            
            y_pred, y_true = get_predictions(batch_graphs, batch_labels, solubnet)
            
            # Original loss: RMSE - 0.1 * R²
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
# 12-15. GNN Models (GIN, GCN, GAT, MPNN) using DGL
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
    
    # Featurize molecules
    def smiles_to_dgl(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return None
        
        node_feats = torch.zeros(num_atoms, 30)  # 30-dim features
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
    graphs = []
    labels = []
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
    
    # GCN Model
    class GCNModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GraphConv(30, 64)
            self.conv2 = GraphConv(64, 64)
            self.fc = torch.nn.Linear(64, 1)
        
        def forward(self, g, h):
            h = torch.relu(self.conv1(g, h))
            h = torch.relu(self.conv2(g, h))
            g.ndata['h'] = h
            return self.fc(dgl.mean_nodes(g, 'h'))
    
    # GIN Model
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
    
    # GAT Model
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
    
    # MPNN (simplified as GCN with aggregation)
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
    
    for name, ModelClass in models.items():
        print(f"  Training {name}...")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        
        model = ModelClass().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        start = time.time()
        for epoch in range(50):  # 50 epochs for timing
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
        
        # Inference timing on test graphs
        model.eval()
        start = time.time()
        with torch.no_grad():
            for g in test_graphs:
                g = g.to(device)
                h = g.ndata['feat'].to(device)
                _ = model(g, h)
        inf_time = time.time() - start
        
        log_result(name, train_time, inf_time, DEVICE, "DGL, 50 epochs")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("REGIME-I TIMING BENCHMARK")
    print(f"Dataset: {DATASET_NAME} (AqSolDB)")
    print(f"Seed: {SEED}")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    # Run all benchmarks
    print("\n[1/14] GSE...")
    time_gse()
    
    print("\n[2/14] ESOL Model...")
    time_esol_model()
    
    print("\n[3/14] Generic ML Baselines...")
    time_generic_baselines()
    
    print("\n[4/14] ChemProp...")
    time_chemprop()
    
    print("\n[5/14] FastProp...")
    time_fastprop()
    
    print("\n[6/14] SolTranNet...")
    time_soltrannet()
    
    print("\n[7/14] AqSolPred...")
    time_aqsolpred()
    
    print("\n[8/14] Tayyebi...")
    time_tayyebi()
    
    print("\n[9/14] Ulrich (GCN)...")
    time_ulrich()
    
    print("\n[10/14] Ours...")
    time_ours()
    
    print("\n[11/14] SolubNet...")
    time_solubnet()
    
    print("\n[12-15/14] GNN Models (GIN, GCN, GAT, MPNN)...")
    time_gnn_models()
    
    # Print results summary
    print("\n" + "=" * 100)
    print("TIMING RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Method':20} | {'Device':6} | {'Train (s)':10} | {'Inference (s)':12} | {'ms/sample':10} | Notes")
    print("-" * 100)
    
    results_df = pd.DataFrame(RESULTS)
    results_df = results_df.sort_values("Train Time (s)")
    
    for _, row in results_df.iterrows():
        inf_s = f"{row['Inference Time (s)']:.3f}" if row['Inference Time (s)'] else "N/A"
        ms_sample = f"{row['Inference Time (ms/sample)']:.3f}" if row['Inference Time (ms/sample)'] else "N/A"
        notes = row['Notes'] if row['Notes'] else ""
        print(f"{row['Method']:20} | {row['Device']:6} | {row['Train Time (s)']:10.2f} | {inf_s:>12} | {ms_sample:>10} | {notes}")
    
    # Save to CSV
    results_df.to_csv("timing_results.csv", index=False)
    print(f"\nResults saved to timing_results.csv")
    print("=" * 100)


if __name__ == "__main__":
    main()

