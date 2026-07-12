#!/usr/bin/env python3
"""Evaluation script for membership inference attack using pre-computed attack scores.
Adapted from attack_model_training.py to work with Combined_Attack_Scores data.

Reproduces: TPR@0.1%FPR, TPR@1%FPR, AUC, Balanced Accuracy
Settings: BCE-CrossAtt, CREMAD, late fusion, 6-client (approximated from pre-computed 5-client data)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score

# Add repo to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attack_models import CrossAttnGapMIA, GapGatedMIA, AffineGapMIA, SimpleMIA

# ==========================================
# Configuration
# ==========================================
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "random_cremad_5client_mix",
                         "Combined_Attack_Scores_Epoch_50.xlsx")

# Feature columns matching the attack_model_training.py setup
FEATURE_COLUMNS = [
    'LOSS_audio_Score',
    'LOSS_visual_Score',
    'LOSS_BASED_audio_Score',
    'LOSS_BASED_visual_Score',
]

LABEL_COLUMN = 'Sample_Type'
LABEL_MAPPING = {'Member': 1, 'Non_Member': 0}
NEW_FEATURE_NAME = 'Diff_Audio_Visual_Score'

BATCH_SIZE = 2056
LEARNING_RATE = 1e-5
EPOCHS = 100
SEED = 422134
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_and_prepare_data():
    """Load combined attack scores and prepare features/labels."""
    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)
    print(f"  Total samples: {len(df)}")
    print(f"  Member: {(df[LABEL_COLUMN] == 'Member').sum()}")
    print(f"  Non-Member: {(df[LABEL_COLUMN] == 'Non_Member').sum()}")

    # Compute Diff feature
    if 'LOSS_BASED_audio_Score' in df.columns and 'LOSS_BASED_visual_Score' in df.columns:
        df[NEW_FEATURE_NAME] = df['LOSS_BASED_audio_Score'] - df['LOSS_BASED_visual_Score']

    FINAL_FEATURE_LIST = FEATURE_COLUMNS + [NEW_FEATURE_NAME]

    # Process labels
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip().map(LABEL_MAPPING)
    df = df.dropna(subset=[LABEL_COLUMN])

    # Shuffle and split (80% train, 20% test)
    df = shuffle(df, random_state=SEED).reset_index(drop=True)
    split_idx = int(0.8 * len(df))
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    X_train = df_train[FINAL_FEATURE_LIST].values.astype(np.float32)
    y_train = df_train[LABEL_COLUMN].values.astype(np.float32)
    X_test = df_test[FINAL_FEATURE_LIST].values.astype(np.float32)
    y_test = df_test[LABEL_COLUMN].values.astype(np.float32)

    print(f"  Train: {len(X_train)} ({np.sum(y_train):.0f} members)")
    print(f"  Test:  {len(X_test)} ({np.sum(y_test):.0f} members)")
    print(f"  Features: {FINAL_FEATURE_LIST}")

    return X_train, y_train, X_test, y_test, FINAL_FEATURE_LIST


def get_tpr_at_fpr(target_fpr, fpr_arr, tpr_arr):
    """Get TPR at a specific FPR threshold."""
    idx = np.where(fpr_arr <= target_fpr)[0]
    return tpr_arr[idx[-1]] if len(idx) > 0 else 0.0


def evaluate_model(model, X_test_tensor, y_test_tensor, X_train_tensor, y_train_tensor):
    """Evaluate the attack model and compute all metrics."""
    model.eval()
    with torch.no_grad():
        y_prob = torch.sigmoid(model(X_test_tensor)).cpu().numpy().flatten()
        y_true = y_test_tensor.cpu().numpy().flatten()
        y_pred = (y_prob >= 0.5).astype(int)

        # Training predictions for balanced accuracy
        train_prob = torch.sigmoid(model(X_train_tensor)).cpu().numpy().flatten()
        train_pred = (train_prob >= 0.5).astype(int)
        y_train_np = y_train_tensor.cpu().numpy().flatten()

    # ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # TPR at specific FPR
    tpr_01_pct = get_tpr_at_fpr(0.001, fpr, tpr)
    tpr_1_pct = get_tpr_at_fpr(0.01, fpr, tpr)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Balanced accuracy
    tpr_all = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (tpr_all + tnr) / 2.0

    # Training balanced accuracy
    train_tn, train_fp, train_fn, train_tp = confusion_matrix(y_train_np, train_pred).ravel()
    train_acc_mem = train_tp / (train_tp + train_fn) if (train_tp + train_fn) > 0 else 0.0
    train_acc_non = train_tn / (train_tn + train_fp) if (train_tn + train_fp) > 0 else 0.0
    train_balanced_acc = (train_acc_mem + train_acc_non) / 2.0

    return {
        'TPR_0.1pct_FPR': tpr_01_pct * 100,
        'TPR_1pct_FPR': tpr_1_pct * 100,
        'AUC': roc_auc,
        'Balanced_Accuracy': balanced_acc * 100,
        'Accuracy': acc * 100,
        'Train_Balanced_Accuracy': train_balanced_acc * 100,
    }


def train_model(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor):
    """Train the attack model."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_test_loss = float('inf')
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor).item()

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}: train_loss={train_loss/len(train_loader):.4f}, test_loss={test_loss:.4f}")

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    print("=" * 60)
    print("MMIA Attack Model Evaluation")
    print("=" * 60)

    # 1. Load data
    X_train, y_train, X_test, y_test, feature_names = load_and_prepare_data()

    # 2. Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train).to(DEVICE)
    y_train_tensor = torch.tensor(y_train).unsqueeze(1).to(DEVICE)
    X_test_tensor = torch.tensor(X_test).to(DEVICE)
    y_test_tensor = torch.tensor(y_test).unsqueeze(1).to(DEVICE)

    input_dim = X_train.shape[1]

    # 3. Evaluate multiple attack model variants
    results = {}

    # CrossAttnGapMIA with BCE (paper's primary attack)
    print("\n--- CrossAttnGapMIA (BCE loss) ---")
    model_crossattn = CrossAttnGapMIA(input_dim, temperature=0.5).to(DEVICE)
    model_crossattn = train_model(model_crossattn, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
    results['BCE-CrossAtt'] = evaluate_model(model_crossattn, X_test_tensor, y_test_tensor, X_train_tensor, y_train_tensor)

    # GapGatedMIA with BCE
    print("\n--- GapGatedMIA (BCE loss) ---")
    model_gapgated = GapGatedMIA(input_dim).to(DEVICE)
    model_gapgated = train_model(model_gapgated, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
    results['BCE-GapGated'] = evaluate_model(model_gapgated, X_test_tensor, y_test_tensor, X_train_tensor, y_train_tensor)

    # AffineGapMIA with BCE
    print("\n--- AffineGapMIA (BCE loss) ---")
    model_affine = AffineGapMIA(input_dim, temperature=0.5).to(DEVICE)
    model_affine = train_model(model_affine, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
    results['BCE-Affine'] = evaluate_model(model_affine, X_test_tensor, y_test_tensor, X_train_tensor, y_train_tensor)

    # 4. Print results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'TPR@0.1%':>10} {'TPR@1%':>10} {'AUC':>10} {'BalAcc':>10}")
    print("-" * 60)

    for name, metrics in results.items():
        print(f"{name:<20} {metrics['TPR_0.1pct_FPR']:>9.2f}% {metrics['TPR_1pct_FPR']:>9.2f}% "
              f"{metrics['AUC']:>10.4f} {metrics['Balanced_Accuracy']:>9.2f}%")

    print("-" * 60)
    print("\nPaper reported (Table 3, BCE-CrossAtt on CREMAD-6):")
    print("  TPR @ 0.1% FPR: 7.08")
    print("  TPR @ 1% FPR:   16.67")
    print("  AUC:            0.7788")
    print("  Balanced Acc:   70.30")

    # 5. Save results
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results.json")
    import json
    serializable_results = {}
    for name, metrics in results.items():
        serializable_results[name] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                       for k, v in metrics.items()}
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    main()
