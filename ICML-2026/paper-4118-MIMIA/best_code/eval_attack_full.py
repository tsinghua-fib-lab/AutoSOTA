#!/usr/bin/env python3
"""Full evaluation script using all available features from pre-computed attack scores.
Target: BCE-CrossAtt on CREMAD-6 settings.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attack_models import CrossAttnGapMIA, GapGatedMIA, AffineGapMIA, SimpleMIA

# ==========================================
# Configuration
# ==========================================
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "random_cremad_5client_mix",
                         "Combined_Attack_Scores_Epoch_50.xlsx")

# Use ALL available features (matching the paper's setup)
FEATURE_COLUMNS = [
    'LOSS_audio_Score',
    'LOSS_visual_Score',
    'LOSS_BASED_full_Score',
    'LOSS_BASED_audio_Score',
    'LOSS_BASED_visual_Score',
    'GRAD_NORM_full_Score',
    'GRAD_NORM_audio_Score',
    'GRAD_NORM_visual_Score',
]

LABEL_COLUMN = 'Sample_Type'
LABEL_MAPPING = {'Member': 1, 'Non_Member': 0}
NEW_FEATURE_NAME = 'Diff_Audio_Visual_Score'

BATCH_SIZE = 1024
LEARNING_RATE = 1e-4
EPOCHS = 200
SEED = 422134
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_and_prepare_data():
    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)
    print(f"  Total samples: {len(df)}")
    print(f"  Member: {(df[LABEL_COLUMN] == 'Member').sum()}")
    print(f"  Non-Member: {(df[LABEL_COLUMN] == 'Non_Member').sum()}")

    # Clean: remove rows with non-numeric values in feature columns
    for col in FEATURE_COLUMNS:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=[c for c in FEATURE_COLUMNS if c in df.columns])

    # Add difference features
    if 'LOSS_audio_Score' in df.columns and 'LOSS_visual_Score' in df.columns:
        df['Diff_Loss_AV'] = df['LOSS_audio_Score'] - df['LOSS_visual_Score']
    if 'LOSS_BASED_audio_Score' in df.columns and 'LOSS_BASED_visual_Score' in df.columns:
        df[NEW_FEATURE_NAME] = df['LOSS_BASED_audio_Score'] - df['LOSS_BASED_visual_Score']
    if 'GRAD_NORM_audio_Score' in df.columns and 'GRAD_NORM_visual_Score' in df.columns:
        df['Diff_GradNorm_AV'] = df['GRAD_NORM_audio_Score'] - df['GRAD_NORM_visual_Score']

    # Add LOSS_full_Score if available
    if 'LOSS_full_Score' in df.columns:
        FEATURE_COLUMNS_FULL = FEATURE_COLUMNS + ['LOSS_full_Score']
    else:
        FEATURE_COLUMNS_FULL = FEATURE_COLUMNS

    FINAL_FEATURE_LIST = FEATURE_COLUMNS_FULL + [NEW_FEATURE_NAME, 'Diff_Loss_AV', 'Diff_GradNorm_AV']

    # Remove columns that don't exist
    FINAL_FEATURE_LIST = [c for c in FINAL_FEATURE_LIST if c in df.columns]

    # Process labels
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip().map(LABEL_MAPPING)
    df = df.dropna(subset=[LABEL_COLUMN])

    # Shuffle and split
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
    print(f"  Features ({len(FINAL_FEATURE_LIST)}): {FINAL_FEATURE_LIST}")

    return X_train, y_train, X_test, y_test, FINAL_FEATURE_LIST


def get_tpr_at_fpr(target_fpr, fpr_arr, tpr_arr):
    idx = np.where(fpr_arr <= target_fpr)[0]
    return tpr_arr[idx[-1]] if len(idx) > 0 else 0.0


def evaluate_model(model, X_test_tensor, y_test_tensor, X_train_tensor, y_train_tensor):
    model.eval()
    with torch.no_grad():
        y_prob = torch.sigmoid(model(X_test_tensor)).cpu().numpy().flatten()
        y_true = y_test_tensor.cpu().numpy().flatten()
        y_pred = (y_prob >= 0.5).astype(int)

        train_prob = torch.sigmoid(model(X_train_tensor)).cpu().numpy().flatten()
        train_pred = (train_prob >= 0.5).astype(int)
        y_train_np = y_train_tensor.cpu().numpy().flatten()

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    tpr_01_pct = get_tpr_at_fpr(0.001, fpr, tpr)
    tpr_1_pct = get_tpr_at_fpr(0.01, fpr, tpr)
    acc = accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr_all = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (tpr_all + tnr) / 2.0

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
        'fpr': fpr,
        'tpr': tpr,
    }


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def train_model(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,
                use_focal=False, lr=LEARNING_RATE, epochs=EPOCHS):
    pos_weight = torch.tensor([(y_train_tensor == 0).sum() / (y_train_tensor == 1).sum()]).to(DEVICE)
    if use_focal:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_test_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor).item()

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                prob = torch.sigmoid(model(X_test_tensor)).cpu().numpy().flatten()
                fpr_arr, tpr_arr, _ = roc_curve(y_test_tensor.cpu().numpy().flatten(), prob)
                roc_auc_val = auc(fpr_arr, tpr_arr)
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_loss/len(train_loader):.4f}, test_loss={test_loss:.4f}, AUC={roc_auc_val:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    print("=" * 60)
    print("MMIA Full Attack Model Evaluation (All Features)")
    print("=" * 60)

    X_train, y_train, X_test, y_test, feature_names = load_and_prepare_data()

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train).to(DEVICE)
    y_train_tensor = torch.tensor(y_train).unsqueeze(1).to(DEVICE)
    X_test_tensor = torch.tensor(X_test).to(DEVICE)
    y_test_tensor = torch.tensor(y_test).unsqueeze(1).to(DEVICE)

    input_dim = X_train.shape[1]
    print(f"Input dimension: {input_dim}")

    results = {}

    # BCE-CrossAtt (paper's primary attack)
    print("\n--- BCE-CrossAtt ---")
    model = CrossAttnGapMIA(input_dim, temperature=0.5).to(DEVICE)
    model = train_model(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, use_focal=False)
    results['BCE-CrossAtt'] = evaluate_model(model, X_test_tensor, y_test_tensor, X_train_tensor, y_train_tensor)

    # BCE-GapGated
    print("\n--- BCE-GapGated ---")
    model = GapGatedMIA(input_dim).to(DEVICE)
    model = train_model(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, use_focal=False)
    results['BCE-GapGated'] = evaluate_model(model, X_test_tensor, y_test_tensor, X_train_tensor, y_train_tensor)

    # Print results
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

    # Save
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results_full.json")
    import json
    serializable = {}
    for name, metrics in results.items():
        serializable[name] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                              for k, v in metrics.items() if k not in ('fpr', 'tpr')}
    with open(output_file, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    main()
