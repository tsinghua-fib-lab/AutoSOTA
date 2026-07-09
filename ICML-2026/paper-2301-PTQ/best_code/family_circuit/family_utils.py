import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_data(family_id, embeddings, sequences, all_ids):
    """
    Extracts balanced positive/negative dataset for a family.
    """
    pos_mask = (all_ids == family_id)
    n_pos = pos_mask.sum()
    if n_pos < 6: return None, None, None, 0

    rng = np.random.RandomState(42)
    neg_pool = np.where(all_ids != family_id)[0]
    n_neg = min(len(neg_pool), n_pos * 4) 
    neg_indices = rng.choice(neg_pool, size=n_neg, replace=False)
    X = np.concatenate([embeddings[pos_mask], embeddings[neg_indices]])
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    seqs = np.concatenate([sequences[pos_mask], sequences[neg_indices]])
    perm = rng.permutation(len(y))
    return X[perm], y[perm], seqs[perm], n_pos

def train_probe(X_train, y_train, device):
    """
    Trains a Class-Balanced Logistic Regression and converts it to a Torch Linear layer.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(X_train_scaled, y_train)
    
    probe = nn.Linear(X_train.shape[1], 1).to(device)
    with torch.no_grad():
        s = torch.tensor(scaler.scale_, device=device, dtype=torch.float)
        m = torch.tensor(scaler.mean_, device=device, dtype=torch.float)
        c = torch.tensor(clf.coef_, device=device, dtype=torch.float)
        b = torch.tensor(clf.intercept_, device=device, dtype=torch.float)
        probe.weight.data = c / s
        probe.bias.data = b - (c / s @ m)
        
    return probe

def evaluate_circuit(discoverer, probe, seqs, y, circuit_nodes, batch_size, mean_pooled=True, gt_embeddings=None, sequential=False, freeze_attention=True, source="mlp_output"):
    """
    Unified evaluation function.
    - Computes F1 if 'probe' is provided.
    - Computes NMSE if 'gt_embeddings' are provided.
    - Uses 'mean_pooled' to toggle between pooled (B, H) and sequence (B, L, H) embeddings.
    - Uses 'sequential' to toggle between Direct (False) and Sequential (True) reconstruction.
    """
    preds_list = []
    all_recon_embs = []
    all_gt_embs = []
    
    if hasattr(discoverer, 'clt'):
        discoverer.clt.eval()
    else:
        discoverer.plt.eval()
    if probe is not None:
        probe.eval()
    
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i : i + batch_size]
            
            # 1. Reconstruct 
            recon_emb = discoverer.reconstruct_layer_embeddings(
                batch_seqs, 
                active_nodes=circuit_nodes, 
                mean_pool=mean_pooled,
                sequential=sequential,
                freeze_attention=freeze_attention,
                source=source
            )
            
            # 2. Probe Evaluation (F1)
            if probe is not None:
                logits = probe(recon_emb)
                if logits.dim() > 1 and logits.shape[-1] == 1:
                    logits = logits.squeeze(-1)                
                if logits.dim() > 1:
                    logits = logits.mean(dim=1)
                batch_preds = (torch.sigmoid(logits).cpu().numpy() > 0.5).astype(int)
                preds_list.extend(batch_preds)

            # 3. NMSE Evaluation 
            if gt_embeddings is not None:
                batch_gt = torch.as_tensor(gt_embeddings[i : i + batch_size], device=recon_emb.device)
                all_recon_embs.append(recon_emb)
                all_gt_embs.append(batch_gt)
    results = {}
    if probe is not None:
        results['f1'] = f1_score(y, preds_list)
    else:
        results['f1'] = 0.0
    if gt_embeddings is not None:
        full_recon = torch.cat(all_recon_embs, dim=0)
        full_gt = torch.cat(all_gt_embs, dim=0)
        mse = F.mse_loss(full_recon, full_gt) 
        target_var = torch.var(full_gt) + 1e-8
        results['nmse'] = (mse / target_var).item() #
    else:
        results['nmse'] = float('nan')
        
    return results

def split_data(X, y, seqs, test_size, val_limit):
    """
    Splits data into train/val/test sets.
    """
    n_total = len(y)
    holdout_size = max(int(n_total * test_size), 2)
    X_train, X_hold, y_train, y_hold, seq_train, seq_hold = train_test_split(
        X, y, seqs, test_size=holdout_size, stratify=y, random_state=42
    )
    n_hold_pos = y_hold.sum()
    if len(y_hold) > (val_limit + 1) and n_hold_pos > 1:
        val_count = min(len(y_hold), val_limit)
        X_val, X_test, y_val, y_test, seq_val, seq_test = train_test_split(
            X_hold, y_hold, seq_hold, train_size=val_count, stratify=y_hold, random_state=42
        )
    else:
        X_val, y_val, seq_val = X_hold, y_hold, seq_hold
        X_test, y_test, seq_test = X_hold, y_hold, seq_hold

    return X_train, X_val, X_test, y_train, y_val, y_test, seq_train, seq_val, seq_test