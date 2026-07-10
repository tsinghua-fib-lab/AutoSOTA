"""
Simple sequential reproduction script for paper 3108.
Computes: R2 Score, MSE (importance), ROC AUC for Friedman 1 benchmark.
Uses: MLP (64-32-8), Bagging (n=10), LOCO importance, SNR=1, n=512.
Ground truth: asymptotic at n=5000.
"""
import sys, time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.datasets import make_friedman1
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
from sklearn.base import clone
from copy import copy, deepcopy
from hidimstat import LOCO

RESULTS = Path('/repo/results')
RESULTS.mkdir(exist_ok=True)
SEEDS = [1, 2, 3]  # Run 3 seeds for validation
N_SAMPLES = 512
N_FEATURES = 20
SNR = 1.0
N_ENSEMBLE = 10
N_SPLITS = 5
HIDDEN = (64, 32, 8)
MAX_ITER = 500
ASYM_N = 5000  # Asymptotic sample size

def make_model(seed):
    return BaggingRegressor(
        estimator=MLPRegressor(hidden_layer_sizes=HIDDEN, max_iter=MAX_ITER,
                               early_stopping=True, random_state=seed),
        n_estimators=N_ENSEMBLE, random_state=seed, n_jobs=1)

def compute_loco_fold(model, X_train, y_train, X_test, y_test, fold_id):
    """Compute LOCO importances for ensemble and sub-models (single fold)."""
    results = []
    n_features = X_train.shape[1]

    # Ensemble LOCO
    t0 = time.time()
    print(f'  Fold {fold_id}: ensemble LOCO...')
    loco = LOCO(model, loss=mean_squared_error)
    loco.fit(X_train, y_train)
    imp_ens = loco.importance(X_test, y_test)
    results.append(pd.DataFrame({
        'feature': np.arange(n_features), 'importance': imp_ens,
        'fold': fold_id, 'model': 'ensemble'
    }))
    print(f'    ensemble LOCO: {time.time()-t0:.1f}s')

    # Sub-model LOCO
    sub_models = model.estimators_
    for i, sm in enumerate(sub_models):
        t0 = time.time()
        sm_c = copy(sm)
        loco_s = LOCO(sm_c, loss=mean_squared_error)
        loco_s.fit(X_train, y_train)
        imp_sub = loco_s.importance(X_test, y_test)
        results.append(pd.DataFrame({
            'feature': np.arange(n_features), 'importance': imp_sub,
            'fold': fold_id, 'model': f'sub_model_{i}'
        }))
        if i % 3 == 0:
            print(f'    sub-model {i}/9: {time.time()-t0:.1f}s')

    return results

def run_simulation(seed):
    """Run full simulation for one seed."""
    print(f'\n{"="*60}')
    print(f'SEED {seed}: Simulation (n={N_SAMPLES}, SNR={SNR})')
    print(f'{"="*60}')

    # Generate data
    X, y = make_friedman1(n_samples=N_SAMPLES, n_features=N_FEATURES,
                          noise=2.0/SNR, random_state=seed)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
    support = np.argsort(
        __import__('sklearn.feature_selection', fromlist=['mutual_info_regression'])
        .mutual_info_regression(X, y, random_state=seed)
    )[-5:]

    out_dir = RESULTS / f'friedman1_mlp_n{N_SAMPLES}_p{N_FEATURES}_bagging{N_ENSEMBLE}_snr{int(SNR)}'
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / f'support_friedman1_{seed}.npy', support)
    np.save(out_dir / f'support_bis_friedman1_{seed}.npy', support)

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    # Fit models and compute scores
    models = []
    scores = []
    print('Fitting models...')
    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X)):
        t0 = time.time()
        m = make_model(seed)
        m.fit(X[train_idx], y[train_idx])
        models.append(m)
        print(f'  Fold {fold_id}: fit in {time.time()-t0:.1f}s')

        yp_ens = m.predict(X[test_idx])
        scores.append({'score': r2_score(y[test_idx], yp_ens), 'model': 'ensemble',
                       'metric': 'r2', 'fold': fold_id, 'seed': seed})
        scores.append({'score': mean_squared_error(y[test_idx], yp_ens), 'model': 'ensemble',
                       'metric': 'mse', 'fold': fold_id, 'seed': seed})

        for sm in m.estimators_:
            yp_sub = sm.predict(X[test_idx])
            scores.append({'score': r2_score(y[test_idx], yp_sub), 'model': 'sub_models',
                           'metric': 'r2', 'fold': fold_id, 'seed': seed})
            scores.append({'score': mean_squared_error(y[test_idx], yp_sub), 'model': 'sub_models',
                           'metric': 'mse', 'fold': fold_id, 'seed': seed})

    pd.DataFrame(scores).to_csv(out_dir / f'scores_friedman1_{seed}.csv', index=False)
    print(f'R2 ensemble: {np.mean([s["score"] for s in scores if s["model"]=="ensemble" and s["metric"]=="r2"]):.4f}')

    # Compute LOCO
    print('Computing LOCO...')
    loco_path = out_dir / f'loco_friedman1_{seed}.csv'
    if not loco_path.exists():
        t0 = time.time()
        all_loco = []
        for fold_id, (train_idx, test_idx) in enumerate(cv.split(X)):
            all_loco.extend(compute_loco_fold(models[fold_id], X[train_idx], y[train_idx],
                                              X[test_idx], y[test_idx], fold_id))
        pd.concat(all_loco).to_csv(loco_path, index=False)
        print(f'LOCO total: {time.time()-t0:.1f}s')
    else:
        print('LOCO already exists')

    return out_dir

def run_asymptotic(seed):
    """Compute asymptotic ground-truth LOCO."""
    print(f'\n{"="*60}')
    print(f'ASYMPTOTIC (n={ASYM_N}, SNR={SNR})')
    print(f'{"="*60}')

    X, y = make_friedman1(n_samples=ASYM_N, n_features=N_FEATURES,
                          noise=2.0/SNR, random_state=seed)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()

    out_dir = RESULTS / f'asympt_n{ASYM_N}_friedman1_mlp_p{N_FEATURES}_bagging{N_ENSEMBLE}'
    out_dir.mkdir(exist_ok=True)
    models_dir = out_dir / 'models'
    models_dir.mkdir(exist_ok=True)

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    # Fit models
    models = []
    print('Fitting asymptotic models...')
    for fold_id, (train_idx, _) in enumerate(cv.split(X)):
        t0 = time.time()
        m = make_model(seed)
        m.fit(X[train_idx], y[train_idx])
        models.append(m)
        print(f'  Fold {fold_id}: fit in {time.time()-t0:.1f}s')

    # LOCO
    loco_path = out_dir / f'asympt_n{int(X.shape[0])}_loco_friedman1_{seed}.csv'
    if not loco_path.exists():
        print('Computing asymptotic LOCO...')
        t0 = time.time()
        all_loco = []
        for fold_id, (train_idx, test_idx) in enumerate(cv.split(X)):
            all_loco.extend(compute_loco_fold(models[fold_id], X[train_idx], y[train_idx],
                                              X[test_idx], y[test_idx], fold_id))
        pd.concat(all_loco).to_csv(loco_path, index=False)
        print(f'Asymptotic LOCO total: {time.time()-t0:.1f}s')
    else:
        print('Asymptotic LOCO already exists')

    return out_dir

def compute_metrics(sim_dir, asym_dir, seeds):
    """Compute R2, MSE importance, and ROC AUC."""
    import polars as pl

    print(f'\n{"="*60}')
    print(f'METRICS')
    print(f'{"="*60}')

    # R2 from scores
    all_scores = []
    for seed in seeds:
        sf = sim_dir / f'scores_friedman1_{seed}.csv'
        if sf.exists():
            df = pl.read_csv(sf).with_columns(pl.lit(seed).alias('seed'))
            all_scores.append(df)

    if all_scores:
        df_s = pl.concat(all_scores)
        r2_e = df_s.filter((pl.col('metric')=='r2') & (pl.col('model')=='ensemble'))
        r2_s = df_s.filter((pl.col('metric')=='r2') & (pl.col('model')=='sub_models'))
        print(f'\nR2 Score:')
        print(f'  Ensemble: median={r2_e["score"].median():.4f}, mean={r2_e["score"].mean():.4f}')
        print(f'  Sub-models: median={r2_s["score"].median():.4f}, mean={r2_s["score"].mean():.4f}')

    # MSE importance and ROC AUC
    loco_files = list(sim_dir.glob('loco_*.csv'))
    asym_files = list(asym_dir.glob('asympt_n*_loco_*.csv'))

    if loco_files and asym_files:
        all_loco = []
        for lf in loco_files:
            seed = int(lf.stem.split('_')[-1])
            df = pl.read_csv(lf).with_columns(pl.lit(seed).alias('seed'))
            all_loco.append(df)
        df_l = pl.concat(all_loco)

        asym_df = pl.read_csv(asym_files[0])
        asym_avg = (asym_df
            .with_columns(strategy=pl.when(pl.col('model')=='ensemble')
                          .then(pl.lit('ensemble')).otherwise(pl.lit('sub-models')))
            .drop('model')
            .group_by(['feature','fold','strategy'])
            .agg(pl.col('importance').mean().alias('asymptotic_importance')))

        support = np.load(sim_dir / 'support_bis_friedman1_1.npy')

        merged = (df_l
            .join(asym_avg, on=['feature','fold','strategy'], how='left')
            .with_columns(support=pl.col('feature').is_in(support.tolist()),
                          se=(pl.col('importance')-pl.col('asymptotic_importance'))**2))

        # MSE per seed per strategy
        mse_ps = merged.group_by(['seed','strategy']).agg(pl.col('se').mean().alias('mse'))
        print(f'\nMSE (importance):')
        for strat in ['ensemble', 'sub-models']:
            vals = mse_ps.filter(pl.col('strategy')==strat)['mse']
            if len(vals) > 0:
                print(f'  {strat}: median={vals.median():.6f}, mean={vals.mean():.6f}')

        # ROC AUC
        merged2 = merged.with_columns(
            y=(pl.col('asymptotic_importance')>1e-3).cast(pl.Int8),
            y_score=pl.col('importance'))
        auc_ps = (merged2.group_by(['seed','strategy']).agg([pl.col('y'),pl.col('y_score')])
            .with_columns(roc_auc=pl.struct(['y','y_score']).map_elements(
                lambda r: roc_auc_score(r['y'],r['y_score']) if len(set(r['y']))>1 else float('nan'),
            return_dtype=pl.Float64))
            .drop(['y','y_score']))

        print(f'\nROC AUC:')
        for strat in ['ensemble', 'sub-models']:
            vals = auc_ps.filter(pl.col('strategy')==strat)['roc_auc']
            if len(vals) > 0:
                print(f'  {strat}: median={vals.median():.4f}, mean={vals.mean():.4f}')

    print('\nDone!')

if __name__ == '__main__':
    t_total = time.time()

    # Run simulation for each seed
    sim_dir = None
    for seed in SEEDS:
        sim_dir = run_simulation(seed)

    # Run asymptotic
    asym_dir = run_asymptotic(1)

    # Compute metrics
    if sim_dir:
        compute_metrics(sim_dir, asym_dir, SEEDS)

    print(f'\nTotal time: {(time.time()-t_total)/60:.1f} minutes')
