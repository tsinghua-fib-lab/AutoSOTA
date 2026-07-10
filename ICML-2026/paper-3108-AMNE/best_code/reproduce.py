"""
Reproduction script for paper 3108: "Aggregate Models, Not Explanations"
Friedman 1 benchmark: MLP (64-32-8), Bagging (n=10), LOCO, SNR=1, n=512.
Computes: R2 Score, MSE (importance), ROC AUC.

Usage:
    python reproduce.py --n_seeds 10 --asym_n 20000 --n_jobs 5
"""
import argparse
import sys, time, numpy as np, pandas as pd
from pathlib import Path
from copy import copy
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.base import clone
from hidimstat import LOCO
from joblib import Parallel, delayed

from ensemble_vim.data import get_dataset
from ensemble_vim.simulation import get_model, get_sub_models


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n_seeds', type=int, default=10)
    p.add_argument('--asym_n', type=int, default=20000)
    p.add_argument('--n_jobs', type=int, default=5)
    p.add_argument('--results_dir', type=str, default='/repo/results')
    p.add_argument('--dataset', type=str, default='friedman1')
    p.add_argument('--model', type=str, default='mlp')
    p.add_argument('--ensemble', type=str, default='bagging')
    p.add_argument('--n_features', type=int, default=20)
    p.add_argument('--n_ensemble', type=int, default=10)
    p.add_argument('--n_samples', type=int, default=512)
    p.add_argument('--snr', type=float, default=1.0)
    p.add_argument('--n_splits', type=int, default=5)
    return p.parse_args()


def loco_one_fast(model, X_train, y_train, X_test, y_test, fold_id):
    """Compute LOCO for ensemble and sub-models (single fold)."""
    results = []
    nf = X_train.shape[1]

    # Ensemble LOCO
    loco = LOCO(model, loss=mean_squared_error)
    loco.fit(X_train, y_train)
    imp = loco.importance(X_test, y_test)
    results.append(pd.DataFrame({
        'feature': np.arange(nf), 'importance': imp,
        'fold': fold_id, 'model': 'ensemble'
    }))

    # Sub-model LOCO
    for i, sm in enumerate(get_sub_models(model)):
        sm_c = copy(sm)
        loco_s = LOCO(sm_c, loss=mean_squared_error)
        loco_s.fit(X_train, y_train)
        imp_s = loco_s.importance(X_test, y_test)
        results.append(pd.DataFrame({
            'feature': np.arange(nf), 'importance': imp_s,
            'fold': fold_id, 'model': f'sub_model_{i}'
        }))
    return results


def fit_one(X, y, train_idx, model, fold_id, seed):
    """Fit one model for a fold."""
    mc = clone(model)
    mc.fit(X[train_idx], y[train_idx])
    return mc


def run_seed(seed, args):
    """Run one seed of the simulation."""
    out_dir = Path(args.results_dir)
    ds = args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    X, y, support, support_bis = get_dataset(
        dataset_name=ds, n_samples=args.n_samples, n_features=args.n_features,
        random_state=seed, snr=args.snr)

    cv = KFold(n_splits=args.n_splits, shuffle=True, random_state=seed)

    # Build model
    model = get_model(args.model, args.n_ensemble, args.ensemble, seed)

    # Fit models in parallel across folds
    fitted = Parallel(n_jobs=args.n_jobs)(
        delayed(fit_one)(X, y, train_idx, model, fold_id, seed)
        for fold_id, (train_idx, _) in enumerate(cv.split(X, y)))

    # Compute scores
    scores = []
    for fold_id, (_, test_idx) in enumerate(cv.split(X, y)):
        yp_ens = fitted[fold_id].predict(X[test_idx])
        scores.append({'score': r2_score(y[test_idx], yp_ens),
                       'model': 'ensemble', 'metric': 'r2',
                       'fold': fold_id, 'seed': seed})
        scores.append({'score': mean_squared_error(y[test_idx], yp_ens),
                       'model': 'ensemble', 'metric': 'mse',
                       'fold': fold_id, 'seed': seed})
        for sm in get_sub_models(fitted[fold_id]):
            yp_sub = sm.predict(X[test_idx])
            scores.append({'score': r2_score(y[test_idx], yp_sub),
                           'model': 'sub_models', 'metric': 'r2',
                           'fold': fold_id, 'seed': seed})
            scores.append({'score': mean_squared_error(y[test_idx], yp_sub),
                           'model': 'sub_models', 'metric': 'mse',
                           'fold': fold_id, 'seed': seed})

    pd.DataFrame(scores).to_csv(
        out_dir / f'scores_{ds}_{seed}.csv', index=False)

    # Compute LOCO in parallel across folds
    print(f'  Seed {seed}: computing LOCO...', flush=True)
    t0 = time.time()
    loco_out = Parallel(n_jobs=args.n_jobs)(
        delayed(loco_one_fast)(
            fitted[fold_id], X[train_idx], y[train_idx],
            X[test_idx], y[test_idx], fold_id)
        for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y)))

    loco_df = pd.concat([item for sublist in loco_out for item in sublist])
    loco_df.to_csv(out_dir / f'loco_{ds}_{seed}.csv', index=False)
    print(f'  Seed {seed}: LOCO done in {time.time()-t0:.1f}s', flush=True)

    # Save support
    np.save(out_dir / f'support_{ds}_{seed}.npy', support)
    np.save(out_dir / f'support_bis_{ds}_{seed}.npy', support_bis)

    r2_e = np.mean([s['score'] for s in scores if s['model']=='ensemble' and s['metric']=='r2'])
    r2_s = np.mean([s['score'] for s in scores if s['model']=='sub_models' and s['metric']=='r2'])
    print(f'  Seed {seed}: R2 ensemble={r2_e:.4f}, sub-models={r2_s:.4f}', flush=True)
    return out_dir


def run_asymptotic(args):
    """Compute asymptotic ground-truth LOCO."""
    seed = 1
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = args.dataset

    print(f'Asymptotic: n={args.asym_n}, seed={seed}')
    X, y, support, support_bis = get_dataset(
        dataset_name=ds, n_samples=args.asym_n, n_features=args.n_features,
        random_state=seed, snr=args.snr)

    cv = KFold(n_splits=args.n_splits, shuffle=True, random_state=seed)
    model = get_model(args.model, args.n_ensemble, args.ensemble, seed)

    print('  Fitting asymptotic models...')
    t0 = time.time()
    fitted = Parallel(n_jobs=args.n_jobs)(
        delayed(fit_one)(X, y, train_idx, model, fold_id, seed)
        for fold_id, (train_idx, _) in enumerate(cv.split(X, y)))
    print(f'  Models fit in {time.time()-t0:.1f}s')

    # Scores
    scores = []
    for fold_id, (_, test_idx) in enumerate(cv.split(X, y)):
        yp = fitted[fold_id].predict(X[test_idx])
        scores.append({'score': r2_score(y[test_idx], yp), 'model': 'ensemble',
                       'metric': 'r2', 'fold': fold_id, 'seed': seed})
        for sm in get_sub_models(fitted[fold_id]):
            yp_s = sm.predict(X[test_idx])
            scores.append({'score': r2_score(y[test_idx], yp_s), 'model': 'sub_models',
                           'metric': 'r2', 'fold': fold_id, 'seed': seed})
    pd.DataFrame(scores).to_csv(
        out_dir / f'asympt_n{args.asym_n}_scores_{ds}_{seed}.csv', index=False)

    print(f'  Computing asymptotic LOCO...')
    t0 = time.time()
    loco_out = Parallel(n_jobs=args.n_jobs)(
        delayed(loco_one_fast)(
            fitted[fold_id], X[train_idx], y[train_idx],
            X[test_idx], y[test_idx], fold_id)
        for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y)))

    loco_df = pd.concat([item for sublist in loco_out for item in sublist])
    loco_df.to_csv(
        out_dir / f'asympt_n{args.asym_n}_loco_{ds}_{seed}.csv', index=False)
    print(f'  Asymptotic LOCO done in {time.time()-t0:.1f}s')

    # Save support
    np.save(out_dir / f'asympt_n{args.asym_n}_support_{ds}_{seed}.npy', support)
    np.save(out_dir / f'asympt_n{args.asym_n}_support_bis_{ds}_{seed}.npy', support_bis)


def compute_metrics(args):
    """Compute R2, MSE (importance), and ROC AUC across all seeds."""
    import polars as pl
    out_dir = Path(args.results_dir)
    ds = args.dataset

    score_files = sorted(out_dir.glob(f'scores_{ds}_*.csv'))
    if not score_files:
        print('No score files found!')
        return

    df_s = pl.concat([pl.read_csv(f) for f in score_files])

    print('\n' + '='*60)
    print('REPRODUCTION METRICS')
    print('='*60)

    for model_type in ['ensemble', 'sub_models']:
        r2_vals = df_s.filter((pl.col('metric')=='r2') & (pl.col('model')==model_type))['score']
        if len(r2_vals) > 0:
            print(f'\nR2 Score ({model_type}):')
            print(f'  Mean: {r2_vals.mean():.4f}')
            print(f'  Median: {r2_vals.median():.4f}')
            r2_std = r2_vals.std()
            print(f'  Std: {r2_std:.4f}' if r2_std is not None else '  Std: 0.0000')

    loco_files = sorted(out_dir.glob(f'loco_{ds}_*.csv'))
    asym_files = sorted(out_dir.glob(f'asympt_n{args.asym_n}_loco_{ds}_*.csv'))

    if not loco_files:
        print('No LOCO files found!')
        return

    df_l = pl.concat([pl.read_csv(f).with_columns(
        pl.lit(int(f.stem.split('_')[-1])).alias('seed')) for f in loco_files])

    df_l = df_l.with_columns(
        strategy=pl.when(pl.col('model')=='ensemble')
        .then(pl.lit('ensemble')).otherwise(pl.lit('sub-models')))

    if asym_files:
        asym_df = pl.read_csv(asym_files[0])
        asym_avg = (asym_df
            .with_columns(strategy=pl.when(pl.col('model')=='ensemble')
                          .then(pl.lit('ensemble')).otherwise(pl.lit('sub-models')))
            .drop('model')
            .group_by(['feature','fold','strategy'])
            .agg(pl.col('importance').mean().alias('asymptotic_importance')))

        merged = df_l.join(asym_avg, on=['feature','fold','strategy'], how='left')
        merged = merged.with_columns(
            se=(pl.col('importance')-pl.col('asymptotic_importance'))**2)

        for strat in ['ensemble', 'sub_models']:
            mse_vals = (merged.filter(pl.col('strategy')==strat)
                .group_by(['seed']).agg(pl.col('se').mean().alias('mse')))
            if len(mse_vals) > 0:
                mse_col = mse_vals['mse']
                mse_std_val = mse_col.std() if mse_col.std() is not None else 0.0
                print(f'\nMSE importance ({strat}):')
                print(f'  Mean: {mse_col.mean():.6f}')
                print(f'  Median: {mse_col.median():.6f}')
                print(f'  Std: {mse_std_val:.6f}')

        # ROC AUC
        merged2 = merged.with_columns(
            y=(pl.col('asymptotic_importance')>1e-3).cast(pl.Int8),
            y_score=pl.col('importance'))
        auc_ps = (merged2.group_by(['seed','strategy'])
            .agg([pl.col('y'),pl.col('y_score')])
            .with_columns(roc_auc=pl.struct(['y','y_score']).map_elements(
                lambda r: roc_auc_score(r['y'],r['y_score'])
                if len(set(r['y']))>1 else float('nan'),
                return_dtype=pl.Float64))
            .drop(['y','y_score']))

        for strat in ['ensemble', 'sub_models']:
            vals = auc_ps.filter(pl.col('strategy')==strat)['roc_auc']
            if len(vals) > 0:
                sval = vals.std() if vals.std() is not None else 0.0
                print(f'\nROC AUC ({strat}):')
                print(f'  Mean: {vals.mean():.4f}')
                print(f'  Median: {vals.median():.4f}')
                print(f'  Std: {sval:.4f}')
    else:
        print('No asymptotic files found - skipping importance metrics.')

    print('\nDone!')


if __name__ == '__main__':
    args = parse_args()
    t_total = time.time()

    # Step 1: Run simulation for each seed
    print(f'Running {args.n_seeds} seeds...')
    for seed in range(1, args.n_seeds + 1):
        t0 = time.time()
        run_seed(seed, args)
        print(f'Seed {seed} total: {time.time()-t0:.1f}s')

    # Step 2: Run asymptotic
    run_asymptotic(args)

    # Step 3: Compute metrics
    compute_metrics(args)

    print(f'\nTotal time: {(time.time()-t_total)/60:.1f} minutes')
