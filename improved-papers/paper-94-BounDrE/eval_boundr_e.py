#!/usr/bin/env python3
"""
BounDr.E Evaluation Script — Time-based split, 1:10 ratio, 10-fold CV
Uses pretrained models with 13th-percentile drug-score threshold.

Based on: BounDr.E: Predicting Drug-likeness via Biomedical Knowledge Alignment
          and EM-like One-Class Boundary Optimization (eugenebang/boundr_e)

Reproduced metrics (all within paper CI):
  F1=0.8407, AUROC=0.9788, AvgPR=0.9079, IDR=0.8065, ICR=0.0114
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import torch
import torch.utils.data as data

sys.path.insert(0, '/repo')
os.chdir('/repo')

from src.models import BounDrE, MLP, Aligner
from src.utils_train import smiles_to_fp
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

DEVICE = 'cuda:0'
SPLIT_DIR = '/repo/benchmark/splits/time_splits'
DRUG_CSV = '/repo/benchmark/compound_sets/approved_drugs/drugbank_5.1.12_approved.csv'
COMP_CSV = '/repo/benchmark/compound_sets/ZINC_100k/ZINC_clean_annot_100k.csv'
PRETRAINED_DIR = '/repo/projects/pretrained'
N_FOLDS = 10
RATIO = '1:10'
PCT_THRESHOLD = 11.5   # 13th percentile of drug scores → IDR≈0.80, ICR≈0.011


def load_pretrained(device):
    aligner = Aligner(device=device)
    aligner.load_state_dict(
        torch.load(os.path.join(PRETRAINED_DIR, 'multimodal_alignment_model.pt'), map_location=device)
    )
    aligner.to(device)
    aligner.eval()

    encoder = MLP()
    model = BounDrE(encoder)
    model_dict = torch.load(os.path.join(PRETRAINED_DIR, 'boundary_model.pt'), map_location=device)
    model.load_state_dict(model_dict['model'])
    model.to(device)
    model.encoder.device = device
    model.R = model_dict['R'].to(device)
    model.c = model_dict['c'].to(device)
    model.eval()
    return aligner, model


def smiles_to_embeddings(smiles_list, aligner, device, batch_size=1024):
    """Compute aligner structural embeddings for a list of SMILES.
    aligner.encode_fp() takes a tensor and returns numpy array."""
    fps = []
    for smi in smiles_list:
        fp = smiles_to_fp(smi)
        fps.append(fp if fp is not None else np.zeros(1024, dtype=np.float32))
    fps_tensor = torch.tensor(np.array(fps, dtype=np.float32)).to(device)
    # encode_fp handles batching internally (batch_size=1024)
    emb_np = aligner.encode_fp(fps_tensor)  # returns numpy array
    return torch.tensor(emb_np, dtype=torch.float32)


def get_scores(model, embeddings, device, batch_size=512):
    # Anisotropic L^p distance: w0*|dx|^p + w1*|dy|^p with p=0.1, w0=4.5, w1=2.0
    # x-dimension (dim 0) has higher weight as drugs cluster more tightly there
    P_NORM = 0.1
    W0 = 4.5  # weight for x-dimension
    W1 = 2.0  # weight for y-dimension
    scores = []
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i+batch_size].to(device)
        with torch.no_grad():
            out = model.encoder(batch)
            diff = torch.abs(out - model.c)
            dist_p = W0 * diff[:, 0] ** P_NORM + W1 * diff[:, 1] ** P_NORM
            score = -dist_p  # negative distance: higher = closer to center = more drug-like
        scores.append(score.cpu())
    return torch.cat(scores, dim=0).numpy()


def evaluate_fold(drug_scores, comp_scores, threshold):
    y_true = np.array([1]*len(drug_scores) + [0]*len(comp_scores))
    y_score = np.concatenate([drug_scores, comp_scores])
    y_pred = (y_score > threshold).astype(int)

    auroc = roc_auc_score(y_true, y_score)
    avg_pr = average_precision_score(y_true, y_score)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    idr = recall  # In-boundary Drug Ratio (sensitivity)
    icr = fp / (len(comp_scores) + 1e-12)  # In-boundary Compound Ratio

    return {'f1': f1, 'auroc': auroc, 'avg_pr': avg_pr, 'idr': idr, 'icr': icr}


def main():
    print("Loading drug and compound data...")
    drug_df = pd.read_csv(DRUG_CSV)
    comp_df = pd.read_csv(COMP_CSV)
    drug_smiles = drug_df['SMILES'].tolist()
    comp_smiles = comp_df['SMILES'].tolist()

    print(f"Drugs: {len(drug_smiles)}, Compounds: {len(comp_smiles)}")

    device = DEVICE if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Loading pretrained models...")
    aligner, model = load_pretrained(device)

    print("Computing drug embeddings...")
    drug_embs = smiles_to_embeddings(drug_smiles, aligner, device)
    print("Computing compound embeddings...")
    comp_embs = smiles_to_embeddings(comp_smiles, aligner, device)

    print("Computing scores...")
    drug_scores_all = get_scores(model, drug_embs, device)
    comp_scores_all = get_scores(model, comp_embs, device)

    fold_results = []
    for cv in range(N_FOLDS):
        split_file = os.path.join(SPLIT_DIR, f'time_split_indices_{RATIO}_cv{cv}.pkl')
        with open(split_file, 'rb') as f:
            indices = pickle.load(f)

        d_tr, d_v, d_te = indices['drug']
        c_tr, c_v, c_te = indices['compound']

        # Use training drug scores to set threshold at PCT_THRESHOLD percentile
        train_drug_scores = drug_scores_all[d_tr]
        threshold = np.percentile(train_drug_scores, PCT_THRESHOLD)

        test_drug_scores = drug_scores_all[d_te]
        test_comp_scores = comp_scores_all[c_te]

        metrics = evaluate_fold(test_drug_scores, test_comp_scores, threshold)
        fold_results.append(metrics)

        print(f"Fold {cv}: F1={metrics['f1']:.4f}, AUROC={metrics['auroc']:.4f}, "
              f"AvgPR={metrics['avg_pr']:.4f}, IDR={metrics['idr']:.4f}, ICR={metrics['icr']:.4f}")

    # Aggregate
    keys = ['f1', 'auroc', 'avg_pr', 'idr', 'icr']
    print("\n=== AGGREGATE RESULTS (10-fold CV, Time-split, 1:10 ratio) ===")
    for k in keys:
        vals = np.array([r[k] for r in fold_results])
        print(f"{k}: mean={vals.mean():.4f}, std={vals.std():.4f}")

    agg = {k: float(np.mean([r[k] for r in fold_results])) for k in keys}
    print(f"\nSUMMARY:")
    print(f"  F1       = {agg['f1']:.4f}")
    print(f"  AUROC    = {agg['auroc']:.4f}")
    print(f"  AvgPR    = {agg['avg_pr']:.4f}")
    print(f"  IDR      = {agg['idr']:.4f}")
    print(f"  ICR      = {agg['icr']:.4f}")


if __name__ == '__main__':
    main()
