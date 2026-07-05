"""Reproduction: TT-Sparse diabetes benchmark with grid search and 5 seeds."""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tt_sparse import TabularEncoder, TTSparseModel, train, prune, explain

warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TEST_SEED = 42
N_SEEDS = 5
TRAIN_SEEDS = list(range(N_SEEDS))
GRID = {'n_bits': [4, 5, 6], 'num_nodes': [20, 30, 40], 'tau': [0.001, 0.01, 0.05]}
EPOCHS, BATCH_SIZE, LR, PATIENCE, VAL_SPLIT = 200, 1024, 0.005, 25, 0.2
NUM_BITS_ENC = 9
MAX_DROP, FT_EPOCHS, MAX_ITER, MAX_FANIN = 2.0, 30, 80, 16

def load_diabetes():
    ds = fetch_openml(data_id=37, as_frame=True, parser='auto')
    df = ds.frame.copy()
    df = df.rename(columns={'class': 'target'})
    return train_test_split(df, test_size=0.2, random_state=TEST_SEED)

def run_one(df_train, df_test, n_bits, num_nodes, tau, seed):
    encoder = TabularEncoder(target='target', task_type='binary', num_bits=NUM_BITS_ENC)
    td = encoder.fit_transform(df_train)
    model = TTSparseModel(
        input_size=encoder.n_ltt_features, num_nodes=num_nodes,
        num_classes=1, n_bits=n_bits, tau=tau, task_type='binary',
        skip_size=encoder.n_skip_features)
    train(model, td, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
        val_split=VAL_SPLIT, patience=PATIENCE,
        device=DEVICE, seed=seed, verbose=False)
    prune(model, td, max_drop_pct=MAX_DROP, finetune_epochs=FT_EPOCHS,
        max_iterations=MAX_ITER, max_fanin=MAX_FANIN,
        device=DEVICE, seed=seed, verbose=False)
    rules = explain(model, encoder)
    test_data = encoder.transform(df_test)
    Xt = torch.tensor(test_data['X_ltt'], dtype=torch.float32, device=DEVICE)
    Xst = torch.tensor(test_data['X_skip'], dtype=torch.float32, device=DEVICE)
    yt = test_data['y']
    model.eval()
    with torch.no_grad():
        preds = model(Xt, Xst).cpu().numpy().ravel()
    auc = roc_auc_score(yt, preds)
    return {
        'seed': seed, 'n_bits': n_bits, 'num_nodes': num_nodes, 'tau': tau,
        'auc': float(auc), 'complexity': int(rules.complexity),
        'num_rules': len(rules.rules)}

print(f'Device: {DEVICE}, GPUs: {torch.cuda.device_count() if DEVICE=="cuda" else 0}')
df_train, df_test = load_diabetes()
print(f'Train: {len(df_train)}, Test: {len(df_test)}')

configs = [(nb, nn, t) for nb in GRID['n_bits'] for nn in GRID['num_nodes'] for t in GRID['tau']]
total = len(configs) * N_SEEDS
print(f'Configs: {len(configs)}, Total runs: {total}')

all_r = []
cfg_groups = defaultdict(list)
idx = 0
t_start = time.time()

for nb, nn, t in configs:
    for s in TRAIN_SEEDS:
        idx += 1
        print(f'[{idx}/{total}] n_bits={nb} nodes={nn} tau={t} seed={s}', flush=True)
        try:
            r = run_one(df_train, df_test, nb, nn, t, s)
            all_r.append(r)
            cfg_groups[(nb, nn, t)].append(r)
            print(f'  AUC={r["auc"]:.4f} complexity={r["complexity"]} rules={r["num_rules"]}', flush=True)
        except Exception as e:
            print(f'  FAILED: {e}', flush=True)
            all_r.append({'seed': s, 'n_bits': nb, 'num_nodes': nn, 'tau': t,
                         'auc': None, 'complexity': None, 'error': str(e)})

elapsed = time.time() - t_start
print()
print('=' * 80)
print(f'SUMMARY (elapsed: {elapsed:.0f}s)')
print('=' * 80)

for key in sorted(cfg_groups.keys()):
    results = cfg_groups[key]
    aucs = [r['auc'] for r in results if r['auc'] is not None]
    comps = [r['complexity'] for r in results if r['complexity'] is not None]
    if aucs:
        print(f'n_bits={key[0]} nodes={key[1]} tau={key[2]}: '
              f'AUC={np.mean(aucs):.4f}+/-{np.std(aucs,ddof=1):.4f} '
              f'Complexity={np.mean(comps):.1f}+/-{np.std(comps,ddof=1):.1f} '
              f'(n={len(aucs)})')

# Best by mean AUC
best_auc = -1
best_key = None
for key, results in cfg_groups.items():
    aucs = [r['auc'] for r in results if r['auc'] is not None]
    if aucs:
        m = np.mean(aucs)
        if m > best_auc:
            best_auc = m
            best_key = key

if best_key and best_key in cfg_groups:
    results = cfg_groups[best_key]
    aucs = [r['auc'] for r in results]
    comps = [r['complexity'] for r in results]
    print(f'\nBEST CONFIG: n_bits={best_key[0]} nodes={best_key[1]} tau={best_key[2]}')
    print(f'  AUC: {np.mean(aucs):.4f} +/- {np.std(aucs,ddof=1):.4f}')
    print(f'  Complexity: {np.mean(comps):.1f} +/- {np.std(comps,ddof=1):.1f}')
    for r in results:
        print(f'  seed={r["seed"]}: AUC={r["auc"]:.4f} complexity={r["complexity"]}')

# Also compute Pareto-like best (best AUC among configs with complexity < 50)
print(f'\nPARETO CANDIDATES (complexity <= 50):')
pareto = []
for key, results in cfg_groups.items():
    for r in results:
        if r['auc'] is not None and r['complexity'] is not None and r['complexity'] <= 50:
            pareto.append(r)
pareto.sort(key=lambda x: x['auc'], reverse=True)
for r in pareto[:10]:
    print(f'  AUC={r["auc"]:.4f} complexity={r["complexity"]} '
          f'n_bits={r["n_bits"]} nodes={r["num_nodes"]} tau={r["tau"]} seed={r["seed"]}')

Path('/repo/outputs').mkdir(parents=True, exist_ok=True)
Path('/repo/outputs/diabetes_results.json').write_text(json.dumps({
    'all_results': all_r, 'elapsed_s': round(elapsed, 1),
}, indent=2, default=str))
print('\nDone. Results saved to /repo/outputs/diabetes_results.json')
