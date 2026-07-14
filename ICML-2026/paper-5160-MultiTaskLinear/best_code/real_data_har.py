"""Run the HAR real-data experiment used in the paper.

This script mirrors the protocol in `MTLR_Real-data.ipynb`. It uses raw
561-dimensional HAR features, global Min-Max scaling, 30 random train/test
splits, and 5-fold CV to tune the regularization multiplier for ARMUL and
the matrix-weighted estimator.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import MinMaxScaler

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import MTL
import preprocessing
from ARMUL import ARMUL, Baselines
try:
    from path_setup import find_code_dir, find_har_dataset
except ImportError:
    from MTLR_Codes.path_setup import find_code_dir, find_har_dataset


Q_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
N_SPLITS = 30
N_FOLD = 5
ETA = 0.1
MAXITER = 200
POSITIVE_LABELS = [5]  # HAR label 5 corresponds to "standing".


def safe_mtl_preprocessing(data, link="linear", intercept=True, n_class=1, standardization=True):
    """Local patch matching the notebook behavior when standardization is disabled."""
    if standardization:
        return _ORIGINAL_MTL_PREPROCESSING(data, link, intercept, n_class, standardization)

    m = len(data[0])
    d = data[0][0].shape[1]
    n_list = np.zeros(m).astype(int)
    X_means = np.zeros((d, 1))
    X_stds = np.ones((d, 1))

    if intercept:
        X_means = np.vstack((np.zeros((1, 1)), X_means))
        X_stds = np.vstack((np.ones((1, 1)), X_stds))

    y_mean, y_std = 0, 1
    X, Y = [], []
    for j in range(m):
        tmp = data[0][j]
        n_list[j] = tmp.shape[0]
        if intercept:
            tmp = np.hstack((np.ones((n_list[j], 1)), tmp))
        X.append(tmp)

    if link == "logistic" and n_class > 2:
        d_out = n_class
        for y_dat in data[1]:
            rows = np.arange(y_dat.shape[0])
            encoded = np.zeros((y_dat.shape[0], n_class))
            encoded[rows, y_dat.reshape(-1)] = 1
            Y.append(encoded)
    else:
        d_out = 1
        for y_dat in data[1]:
            Y.append(y_dat.reshape(-1, 1))

    return [X, Y, X_means, X_stds, y_mean, y_std, n_list, d_out]


_ORIGINAL_MTL_PREPROCESSING = preprocessing.MTL_preprocessing


def _apply_local_preprocessing_patch():
    preprocessing.MTL_preprocessing = safe_mtl_preprocessing
    MTL.MTL_preprocessing = safe_mtl_preprocessing


_apply_local_preprocessing_patch()


def load_raw_minmax(path: Path | None = None, label_list: list[int] | None = None):
    """Load HAR features, apply global Min-Max scaling, and split by subject."""
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
        y_raw.append(
            np.array([1 if (label + 1) in label_list else 0 for label in labels], dtype=int)
        )

    return [X_raw, y_raw]


def nll_and_grad_theta(X: np.ndarray, y: np.ndarray, theta: np.ndarray):
    """Return the logistic negative log-likelihood and its gradient."""
    z = X @ theta
    p = expit(z)
    nll = np.mean(np.logaddexp(0.0, z) - y * z)
    grad = (X.T @ (p - y)) / X.shape[0]
    return nll, grad


def reg_sqrt_value_and_grad(diff: np.ndarray, sigma: np.ndarray, lam: float):
    """Return lambda * ||diff||_sigma and its gradient with a small floor."""
    sigma_diff = sigma @ diff
    quad_form = float(diff @ sigma_diff)
    norm_val = math.sqrt(max(quad_form, 1e-12))
    value = lam * norm_val
    gradient = (lam / norm_val) * sigma_diff if norm_val > 1e-12 else np.zeros_like(diff)
    return value, gradient


def fit_ours_bfgs(data, q: float = 1.0, maxiter: int = MAXITER):
    """Fit the matrix-weighted logistic estimator with L-BFGS-B."""
    X_list, y_list = data
    m = len(X_list)
    d = X_list[0].shape[1]
    n_list = [len(y) for y in y_list]
    sigmas = [(X.T @ X) / n for X, n in zip(X_list, n_list)]
    lam_list = [q * math.sqrt(d) * math.sqrt(n) for n in n_list]

    def objective(params: np.ndarray):
        theta = params[: m * d].reshape(m, d)
        bar = params[m * d :]

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

    result = minimize(
        objective,
        np.zeros(m * d + d),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter},
    )
    theta = result.x[: m * d].reshape(m, d)
    return [theta[j] for j in range(m)]


def evaluate_ours(data_test, thetas):
    """Evaluate the matrix-weighted estimator on held-out tasks."""
    X_list, y_list = data_test
    errors = []
    for X, y, theta in zip(X_list, y_list, thetas):
        logits = X @ theta
        y_hat = (expit(logits) >= 0.5).astype(int)
        errors.append(np.mean(y_hat != y.ravel()))
    return float(np.mean(errors))


def run_cv_selection(
    data_train,
    model_type: str,
    q_grid: list[float],
    d_dim: int,
    n_fold: int = N_FOLD,
    eta: float = ETA,
    maxiter: int = MAXITER,
):
    """Select q by CV and refit either ARMUL or OURS on the full training set."""
    m = len(data_train[0])
    n_list = np.array([len(y) for y in data_train[1]])
    splits = preprocessing.split_cv(n_list, n_fold, seed=np.random.randint(10000))

    best_q = None
    best_error = float("inf")

    for q in q_grid:
        fold_errors = []
        for fold in range(n_fold):
            X_tr, y_tr, X_val, y_val = [], [], [], []
            for j in range(m):
                idx_val = splits[j][fold]
                idx_tr = np.delete(np.arange(n_list[j]), idx_val)
                X_tr.append(data_train[0][j][idx_tr])
                y_tr.append(data_train[1][j][idx_tr])
                X_val.append(data_train[0][j][idx_val])
                y_val.append(data_train[1][j][idx_val])

            if model_type == "ARMUL":
                model = ARMUL(link="logistic", n_class=2, penalty="new")
                lbd_vec = q * np.sqrt((d_dim + 1) / np.array([len(y) for y in y_tr]))
                model.vanilla(
                    [X_tr, y_tr],
                    lbd=lbd_vec,
                    eta_global=eta,
                    eta_local=eta,
                    T_global=maxiter,
                    standardization=False,
                    intercept=True,
                )
                fold_errors.append(model.evaluate([X_val, y_val], model="vanilla")["average error"])
            elif model_type == "OURS":
                X_tr_int = [np.hstack([np.ones((x.shape[0], 1)), x]) for x in X_tr]
                X_val_int = [np.hstack([np.ones((x.shape[0], 1)), x]) for x in X_val]
                thetas = fit_ours_bfgs([X_tr_int, y_tr], q=q, maxiter=maxiter)
                fold_errors.append(evaluate_ours([X_val_int, y_val], thetas))
            else:
                raise ValueError("model_type must be 'ARMUL' or 'OURS'.")

        avg_error = float(np.mean(fold_errors))
        if avg_error < best_error:
            best_error = avg_error
            best_q = q

    if best_q is None:
        raise RuntimeError("Cross-validation did not select a valid q.")

    if model_type == "ARMUL":
        model = ARMUL(link="logistic", n_class=2, penalty="new")
        lbd_vec = best_q * np.sqrt((d_dim + 1) / n_list)
        model.vanilla(
            data_train,
            lbd=lbd_vec,
            eta_global=eta,
            eta_local=eta,
            T_global=maxiter,
            standardization=False,
            intercept=True,
        )
        return model, best_q

    X_tr_int = [np.hstack([np.ones((x.shape[0], 1)), x]) for x in data_train[0]]
    thetas = fit_ours_bfgs([X_tr_int, data_train[1]], q=best_q, maxiter=maxiter)
    return thetas, best_q


def run_experiment(path: Path | None = None):
    """Run the full HAR experiment and print aggregate held-out errors."""
    if path is None:
        path = find_code_dir(mount_drive=False)

    data = load_raw_minmax(path=path, label_list=POSITIVE_LABELS)
    m = len(data[0])
    d = data[0][0].shape[1]
    print(f"Loaded HAR data with m={m} tasks and d={d} raw features.")

    results = []
    print(f"Starting experiment: {N_SPLITS} random splits with {N_FOLD}-fold CV.")

    for split_idx in range(N_SPLITS):
        if split_idx % 5 == 0:
            print(f"--- Split {split_idx + 1}/{N_SPLITS} ---")

        seed = split_idx * 100
        prop = 0.2 * np.ones(m)
        data_train, data_test = preprocessing.split(data, prop=prop, seed=seed)

        baseline = Baselines(link="logistic", n_class=2)
        baseline.DP_train(data_train, eta=ETA, T=MAXITER, standardization=False, intercept=True)
        dp_res = baseline.evaluate(data_test, model="DP")

        baseline.STL_train(data_train, eta=ETA, T=MAXITER, standardization=False, intercept=True)
        itl_res = baseline.evaluate(data_test, model="STL")

        armul_model, best_q_armul = run_cv_selection(
            data_train,
            "ARMUL",
            Q_GRID,
            d,
            n_fold=N_FOLD,
            eta=ETA,
            maxiter=MAXITER,
        )
        armul_res = armul_model.evaluate(data_test, model="vanilla")

        ours_thetas, best_q_ours = run_cv_selection(
            data_train,
            "OURS",
            Q_GRID,
            d,
            n_fold=N_FOLD,
            eta=ETA,
            maxiter=MAXITER,
        )
        X_test_int = [np.hstack([np.ones((x.shape[0], 1)), x]) for x in data_test[0]]
        ours_err = evaluate_ours([X_test_int, data_test[1]], ours_thetas)

        results.append(
            {
                "split": split_idx,
                "DP": dp_res["average error"],
                "ITL": itl_res["average error"],
                "ARMUL": armul_res["average error"],
                "q_ARMUL": best_q_armul,
                "OURS": ours_err,
                "q_OURS": best_q_ours,
            }
        )
        print(
            f"   [Split {split_idx}] ARMUL: {armul_res['average error']:.4f} | "
            f"OURS: {ours_err:.4f}"
        )

    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("FINAL RESULTS (Held-out Test Error)")
    print("Protocol: Global Min-Max [0,1] -> 5-fold CV -> Refit -> Test")
    print("=" * 60)
    for method in ["DP", "ITL", "ARMUL", "OURS"]:
        print(f"{method:<10} | mean={df[method].mean():.4f} | std={df[method].std():.4f}")
    print("=" * 60)
    return df


if __name__ == "__main__":
    run_experiment()
