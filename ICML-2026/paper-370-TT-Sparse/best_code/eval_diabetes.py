"""Reproduction eval: TT-Sparse on diabetes with 5 seeds.

Paper: TT-Sparse (ICML 2026), Table 1, diabetes row
Protocol: 80-20 test split, grid-selected config, 5 training seeds
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tt_sparse import TabularEncoder, TTSparseModel, train, prune, explain

warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TEST_SEED = 42
TRAIN_SEEDS = [0, 1, 2, 3, 4]
# Best config from grid search over n_bits={4,5,6}, nodes={20,30,40}, tau={0.001,0.01,0.05}
N_BITS, NUM_NODES, TAU = 4, 30, 0.05
EPOCHS, BATCH_SIZE, LR, PATIENCE, VAL_SPLIT = 200, 1024, 0.005, 25, 0.2
NUM_BITS_ENC, MAX_DROP, FT_EPOCHS, MAX_ITER, MAX_FANIN = 9, 2.0, 30, 80, 16

def main():
    t0 = time.time()
    ds = fetch_openml(data_id=37, as_frame=True, parser='auto')
    df = ds.frame.copy()
    df = df.rename(columns={'class': 'target'})
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=TEST_SEED)
    print(f'Data: train={len(df_train)} test={len(df_test)}  Device={DEVICE}')

    results = []
    models_and_encoders = []
    for seed in TRAIN_SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        encoder = TabularEncoder(target='target', task_type='binary', num_bits=NUM_BITS_ENC)
        td = encoder.fit_transform(df_train)
        model = TTSparseModel(
            input_size=encoder.n_ltt_features, num_nodes=NUM_NODES,
            num_classes=1, n_bits=N_BITS, tau=TAU, task_type='binary',
            skip_size=encoder.n_skip_features)
        train(model, td, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
            val_split=VAL_SPLIT, patience=PATIENCE, device=DEVICE, seed=seed, verbose=False)
        prune(model, td, max_drop_pct=MAX_DROP, finetune_epochs=FT_EPOCHS,
            max_iterations=MAX_ITER, max_fanin=MAX_FANIN, device=DEVICE, seed=seed, verbose=False)
        rules = explain(model, encoder)
        td_test = encoder.transform(df_test)
        Xt = torch.tensor(td_test['X_ltt'], dtype=torch.float32, device=DEVICE)
        Xst = torch.tensor(td_test['X_skip'], dtype=torch.float32, device=DEVICE)
        yt = td_test['y']
        model.eval()
        with torch.no_grad():
            preds = model(Xt, Xst).cpu().numpy().ravel()
        auc = float(roc_auc_score(yt, preds))
        comp = int(rules.complexity)
        results.append({'seed': seed, 'auc': auc, 'complexity': comp})
        models_and_encoders.append((model, encoder, rules))
        print(f'  seed={seed}: AUC={auc:.4f} complexity={comp}')


    # Ensemble: average probabilities from all 5 models
    yt_ens = None
    all_preds = []
    for _model, _encoder, _rules in models_and_encoders:
        td_i = _encoder.transform(df_test)
        if yt_ens is None:
            yt_ens = td_i["y"]
        Xt_i = torch.tensor(td_i["X_ltt"], dtype=torch.float32, device=DEVICE)
        Xst_i = torch.tensor(td_i["X_skip"], dtype=torch.float32, device=DEVICE)
        _model.eval()
        with torch.no_grad():
            preds_i = torch.sigmoid(_model(Xt_i, Xst_i)).cpu().numpy().ravel()
        all_preds.append(preds_i)

    ensemble_preds = np.mean(all_preds, axis=0)
    ensemble_auc = float(roc_auc_score(yt_ens, ensemble_preds))
    max_complexity = max(r["complexity"] for r in results)

    aucs = [r['auc'] for r in results]
    comps = [r['complexity'] for r in results]
    m = {
        'AUC': round(float(np.mean(aucs)), 4),
        'AUC_std': round(float(np.std(aucs, ddof=1)), 4),
        'AUC_per_seed': [round(a, 4) for a in aucs],
        'Ensemble_AUC': round(ensemble_auc, 4),
        'Ensemble_Complexity': max_complexity,
        'Complexity': round(float(np.mean(comps)), 1),
        'Complexity_std': round(float(np.std(comps, ddof=1)), 1),
        'Complexity_per_seed': comps,
        'primary_metric': 'AUC',
        'metric_direction': 'higher',
        'elapsed_seconds': round(time.time() - t0, 1),
    }
    out = Path('/repo/outputs/metrics.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=2))
    print(f'\nAUC={m["AUC"]}+/-{m["AUC_std"]}  Complexity={m["Complexity"]}+/-{m["Complexity_std"]}')
    print(f"Ensemble: AUC={ensemble_auc:.4f}  Complexity={max_complexity}")
    print(json.dumps(m, indent=2))

if __name__ == '__main__':
    main()
