"""HAR experiment: ALGO-04 progressive q-annealing.
Start from high q (strong regularization, better global structure),
anneal to low q (weak regularization, better per-task fit).
"""
from __future__ import annotations
import time, sys, math
import numpy as np
import pandas as pd
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from scipy.special import expit
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler

import preprocessing
from ARMUL import ARMUL, Baselines
from path_setup import find_code_dir, find_har_dataset

from real_data_har import (
    _apply_local_preprocessing_patch, nll_and_grad_theta, reg_sqrt_value_and_grad
)
_apply_local_preprocessing_patch()

# --- Config ---
Q_GRID = [0.001]
OURS_Q = 0.001
Q_ANNEAL = [0.10, 0.05, 0.01, 0.001]  # ALGO-04: progressive annealing
N_SPLITS = 30
N_FOLD = 5
ETA = 0.1
MAXITER_GD = 1000
MAXITER_BFGS = 200  # Per annealing stage
MAXITER_CV = 50
POSITIVE_LABELS = [5]


def load_raw_minmax(path=None, label_list=None):
    if label_list is None:
        label_list = POSITIVE_LABELS
    code_dir = find_code_dir(mount_drive=False) if path is None else Path(path)
    base_path = find_har_dataset(code_dir)
    X_train = np.loadtxt(base_path / "train" / "X_train.txt")
    y_train = np.loadtxt(base_path / "train" / "y_train.txt")
    subject_train = np.loadtxt(base_path / "train" / "subject_train.txt")
    X_test = np.loadtxt(base_path / "test" / "X_test.txt")
    y_test = np.loadtxt(base_path / "test" / "y_test.txt")
    subject_test = np.loadtxt(base_path / "test" / "subject_test.txt")
    X_all = np.concatenate((X_train, X_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)
    subject_all = np.concatenate((subject_train, subject_test), axis=0).astype(int)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_all = scaler.fit_transform(X_all)
    X_raw, y_raw = [], []
    for task_id in range(1, subject_all.max() + 1):
        indices = np.where(subject_all == task_id)[0]
        X_raw.append(X_all[indices])
        labels = y_all[indices].astype(int) - 1
        y_raw.append(np.array([1 if (label + 1) in label_list else 0 for label in labels], dtype=int))
    return [X_raw, y_raw]


def fit_ours_bfgs_stage(data, q, x0, maxiter):
    X_list, y_list = data
    m = len(X_list)
    d = X_list[0].shape[1]
    n_list = [len(y) for y in y_list]
    sigmas = [(X.T @ X) / n for X, n in zip(X_list, n_list)]
    lam_list = [q * math.sqrt(d) * math.sqrt(n) for n in n_list]

    def objective(params):
        theta = params[:m * d].reshape(m, d)
        bar = params[m * d:]
        total_loss = 0.0
        theta_grad = np.zeros_like(theta)
        bar_grad = np.zeros_like(bar)
        for j in range(m):
            nll, grad = nll_and_grad_theta(X_list[j], y_list[j].ravel(), theta[j])
            total_loss += n_list[j] * nll
            theta_grad[j] += n_list[j] * grad
            reg_value, reg_grad = reg_sqrt_value_and_grad(theta[j] - bar, sigmas[j], lam_list[j])
            total_loss += reg_value
            theta_grad[j] += reg_grad
            bar_grad -= reg_grad
        gradient = np.concatenate([theta_grad.ravel(), bar_grad.ravel()])
        return total_loss, gradient

    result = minimize(objective, x0, method="L-BFGS-B", jac=True,
                      options={"maxiter": maxiter, "maxfun": maxiter * 20})
    theta = result.x[:m * d].reshape(m, d)
    return [theta[j] for j in range(m)], result.x


def fit_ours_annealed(data, q_sequence, maxiter_per_stage=200):
    """Progressive q-annealing: start from high q, anneal to low q."""
    m = len(data[0])
    d = data[0][0].shape[1]
    x0 = np.zeros(m * d + d)
    thetas = None
    for q in q_sequence:
        thetas, x0 = fit_ours_bfgs_stage(data, q, x0, maxiter_per_stage)
    return thetas


def evaluate_ours(data_test, thetas):
    X_list, y_list = data_test
    errors = []
    for X, y, theta in zip(X_list, y_list, thetas):
        logits = X @ theta
        y_hat = (expit(logits) >= 0.5).astype(int)
        errors.append(np.mean(y_hat != y.ravel()))
    return float(np.mean(errors))


def main():
    data = load_raw_minmax()
    m = len(data[0])
    d = data[0][0].shape[1]
    print(f"Loaded HAR data with m={m} tasks and d={d} raw features.", flush=True)
    print(f"ALGO-04 annealing: {Q_ANNEAL}, GD={MAXITER_GD}, BFGS/stage={MAXITER_BFGS}", flush=True)

    results = []
    total_start = time.time()

    for split_idx in range(N_SPLITS):
        split_start = time.time()
        seed = split_idx * 100
        prop = 0.2 * np.ones(m)
        data_train, data_test = preprocessing.split(data, prop=prop, seed=seed)

        t0 = time.time()
        baseline = Baselines(link="logistic", n_class=2)
        baseline.DP_train(data_train, eta=ETA, T=MAXITER_GD, standardization=False, intercept=True)
        dp_err = baseline.evaluate(data_test, model="DP")["average error"]

        baseline.STL_train(data_train, eta=ETA, T=MAXITER_GD, standardization=False, intercept=True)
        itl_err = baseline.evaluate(data_test, model="STL")["average error"]

        n_list = np.array([len(y) for y in data_train[1]])
        armul_splits = preprocessing.split_cv(n_list, N_FOLD, seed=np.random.randint(10000))
        best_q_armul, best_err_armul = None, float("inf")
        for q in Q_GRID:
            fold_errors = []
            for fold in range(N_FOLD):
                X_tr_fold, y_tr_fold, X_val_fold, y_val_fold = [], [], [], []
                for j in range(m):
                    idx_val = armul_splits[j][fold]
                    idx_tr = np.delete(np.arange(n_list[j]), idx_val)
                    X_tr_fold.append(data_train[0][j][idx_tr])
                    y_tr_fold.append(data_train[1][j][idx_tr])
                    X_val_fold.append(data_train[0][j][idx_val])
                    y_val_fold.append(data_train[1][j][idx_val])
                model = ARMUL(link="logistic", n_class=2, penalty="new")
                lbd_vec = q * np.sqrt((d + 1) / np.array([len(y) for y in y_tr_fold]))
                model.vanilla([X_tr_fold, y_tr_fold], lbd=lbd_vec, eta_global=ETA, eta_local=ETA,
                              T_global=MAXITER_CV, standardization=False, intercept=True)
                fold_errors.append(model.evaluate([X_val_fold, y_val_fold], model="vanilla")["average error"])
            avg_err = float(np.mean(fold_errors))
            if avg_err < best_err_armul:
                best_err_armul = avg_err
                best_q_armul = q
        armul_model = ARMUL(link="logistic", n_class=2, penalty="new")
        lbd_vec = best_q_armul * np.sqrt((d + 1) / n_list)
        armul_model.vanilla(data_train, lbd=lbd_vec, eta_global=ETA, eta_local=ETA,
                          T_global=MAXITER_GD, standardization=False, intercept=True)
        armul_err = armul_model.evaluate(data_test, model="vanilla")["average error"]

        t0_ours = time.time()
        X_tr = [np.hstack([np.ones((x.shape[0], 1)), x]) for x in data_train[0]]
        X_te = [np.hstack([np.ones((x.shape[0], 1)), x]) for x in data_test[0]]
        ours_thetas = fit_ours_annealed([X_tr, data_train[1]], Q_ANNEAL, MAXITER_BFGS)
        ours_err = evaluate_ours([X_te, data_test[1]], ours_thetas)
        ot = time.time()-t0_ours

        results.append({
            "split": split_idx, "DP": dp_err, "ITL": itl_err,
            "ARMUL": armul_err, "OURS": ours_err,
        })

        elapsed = time.time() - split_start
        total_elapsed = time.time() - total_start
        eta_rem = (total_elapsed / (split_idx + 1)) * (N_SPLITS - split_idx - 1) / 60
        print(f"  [{split_idx}] OURS={ours_err:.4f}({ot:.1f}s) total={elapsed:.1f}s ETA={eta_rem:.1f}min", flush=True)

    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print(f"ALGO-04 annealing: {Q_ANNEAL}, GD={MAXITER_GD}, BFGS/stage={MAXITER_BFGS}")
    print("=" * 60)
    for method in ["DP", "ITL", "ARMUL", "OURS"]:
        mean_v = df[method].mean()
        std_v = df[method].std()
        print(f"  {method:<8} mean={mean_v:.4f}  std={std_v:.4f}")
    print(f"Total: {(time.time()-total_start)/60:.1f} min")
    df.to_csv("/repo/har_results.csv", index=False)
    return df

if __name__ == "__main__":
    main()
