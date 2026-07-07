import os
import csv
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc

from preprocessing import (
    gauss_noise_padding,
    softrank_preprocessing_correct,
    whiten_blocks,
)


# ============================================================
# Internal MI estimation (for training-time evaluation)
# ============================================================

def estimate_mi_xy(X, Y, model, max_dim, softrank_reg: float = 1e-3, gauss_copula=True):
    device = next(model.parameters()).device

    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have the same shape, got X: {X.shape}, Y: {Y.shape}")
    if len(X.shape) != 3:
        raise ValueError(f"X and Y must have shape [B, N, d], got {X.shape}")
    B, N, d = X.shape
    if d > max_dim:
        raise ValueError(f"Input dimension d={d} cannot be greater than max_dim={max_dim}")

    X = X.float()
    Y = Y.float()

    X_padded = gauss_noise_padding(X, aim_dim=max_dim, perm=False)
    Y_padded = gauss_noise_padding(Y, aim_dim=max_dim, perm=False)

    sample_xy = torch.cat([X_padded, Y_padded], dim=-1)
    sample_xy = softrank_preprocessing_correct(
        sample_xy, regularization_strength=softrank_reg, gauss_copula=gauss_copula
    ).to(device)

    # Match the model's training-time preprocessing: if it was trained with per-side
    # ZCA whitening, whiten the eval data too (stamped as model._whiten by
    # LightningWrapper.__init__ / infer.load_ckpt). Otherwise in-training BMI/sync
    # numbers under-estimate badly due to a train/eval preprocessing mismatch.
    _whiten = str(getattr(model, "_whiten", "none"))
    if _whiten == "eig":
        _eps = float(getattr(model, "_whiten_eps", 1e-3))
        sample_xy = whiten_blocks(sample_xy, max_dim=max_dim, eps_floor=_eps)

    with torch.no_grad():
        mi_est = model(sample_xy)
        if mi_est.shape[0] == B:
            return mi_est.cpu()
        else:
            return mi_est.cpu().item()


def compute_ksmi_mean(X, Y, projection_dim, model, proj_num: int, batchsize: int,
                      max_dim: int = 5, softrank_reg: float = 1e-3,
                      normalize_input: bool = True, gauss_copula: bool = True):
    model.eval()
    device = next(model.parameters()).device
    seq_len, dx = X.shape
    _, dy = Y.shape

    if seq_len != Y.shape[0]:
        raise ValueError(f"X and Y must have the same number of samples, got X: {X.shape}, Y: {Y.shape}")

    if normalize_input:
        X = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-6)
        Y = (Y - Y.mean(dim=0, keepdim=True)) / (Y.std(dim=0, keepdim=True) + 1e-6)

    results = []

    for i in range(0, proj_num, batchsize):
        current_batch_size = min(batchsize, proj_num - i)

        X_proj_batch = []
        Y_proj_batch = []
        for _ in range(current_batch_size):
            proj_matrix_x = torch.randn(dx, projection_dim, device="cpu")
            proj_matrix_y = torch.randn(dy, projection_dim, device="cpu")

            proj_matrix_x = proj_matrix_x / (torch.norm(proj_matrix_x, dim=0, keepdim=True) + 1e-12)
            proj_matrix_y = proj_matrix_y / (torch.norm(proj_matrix_y, dim=0, keepdim=True) + 1e-12)

            X_proj = X @ proj_matrix_x
            Y_proj = Y @ proj_matrix_y
            X_proj_batch.append(X_proj.unsqueeze(0))
            Y_proj_batch.append(Y_proj.unsqueeze(0))

        X_proj_batch = torch.cat(X_proj_batch, dim=0)
        Y_proj_batch = torch.cat(Y_proj_batch, dim=0)

        mi_est = estimate_mi_xy(X_proj_batch, Y_proj_batch, model, max_dim,
                                softrank_reg, gauss_copula=gauss_copula)

        if torch.is_tensor(mi_est) and mi_est.shape[0] == current_batch_size:
            results.append(mi_est)
        else:
            results.append(torch.tensor([mi_est], device=device))

    results = torch.cat(results) if len(results) > 1 else results[0]
    return torch.mean(results).cpu()


# ============================================================
# BMI evaluation (training-time)
# ============================================================

def evaluate_bmi(
    module,
    max_dim,
    number_test,
    training_step,
    softrank_reg,
    log_dir,
    *,
    sample_sizes=500,
    n_samples_to_use=5,
    data_root="bmi_task_data_full",
    seeds=range(42, 47),
    gauss_copula=True,
):
    """
    BMI evaluation using pre-saved data from data_root/<task_name>/X_seed_*.npz.

    CSV output: log_dir/bmi_results_full.csv
    """
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "bmi_results_full.csv")

    chosen_task_list = [
        '1v1-normal-0.75', 'normal_cdf-1v1-normal-0.75', '1v1-additive-0.1', '1v1-additive-0.75',
        'wiggly-1v1-normal-0.75', 'student-identity-1-1-1', 'asinh-student-identity-1-1-1',
        'multinormal-dense-2-2-0.5', 'multinormal-dense-3-3-0.5', 'multinormal-dense-5-5-0.5',
        'multinormal-sparse-2-2-2-2.0', 'multinormal-sparse-3-3-2-2.0', 'multinormal-sparse-5-5-2-2.0',
        'student-identity-2-2-1', 'student-identity-2-2-2', 'student-identity-3-3-2',
        'student-identity-3-3-3', 'student-identity-5-5-2', 'student-identity-5-5-3',
        'normal_cdf-multinormal-sparse-3-3-2-2.0', 'normal_cdf-multinormal-sparse-5-5-2-2.0',
        'half_cube-multinormal-sparse-3-3-2-2.0', 'half_cube-multinormal-sparse-5-5-2-2.0',
        'spiral-multinormal-sparse-3-3-2-2.0', 'spiral-multinormal-sparse-5-5-2-2.0',
        'spiral-normal_cdf-multinormal-sparse-3-3-2-2.0', 'spiral-normal_cdf-multinormal-sparse-5-5-2-2.0',
        'asinh-student-identity-2-2-1', 'asinh-student-identity-3-3-2', 'asinh-student-identity-5-5-2'
    ]

    # Keep only tasks the model can actually ingest: native input dim <= max_dim.
    # For max_dim >= 5 every task passes (byte-identical to before); for the 3d
    # model this drops the dim-4/5 tasks so estimate_mi_xy never raises mid-eval
    # (which would otherwise crash training at the first validation).
    def _native_dim(task_name):
        xp = os.path.join(data_root, task_name, "X_seed_42.npz")
        return int(np.load(xp)["X"].shape[1])

    _full = list(chosen_task_list)
    chosen_task_list = [t for t in chosen_task_list if _native_dim(t) <= max_dim]
    if len(chosen_task_list) < len(_full):
        print(f"[BMI eval] max_dim={max_dim}: using {len(chosen_task_list)}/{len(_full)} "
              f"tasks (native dim <= {max_dim}); dropped {len(_full) - len(chosen_task_list)}")

    if isinstance(sample_sizes, int):
        sample_sizes = [sample_sizes]
    sample_sizes = [int(x) for x in sample_sizes]

    seeds = list(seeds)
    if n_samples_to_use > len(seeds):
        raise ValueError(f"n_samples_to_use={n_samples_to_use} > available seeds={len(seeds)}")

    rng = np.random.default_rng(int(training_step))

    header = ["training_step", "sample_size"] + chosen_task_list + ["mean_bias"]

    gt_list = []
    for task_name in chosen_task_list:
        task_dir = os.path.join(data_root, task_name)
        gt_data = np.load(os.path.join(task_dir, "ground_truth.npz"))
        gt_mi = float(gt_data["mi"].item())
        gt_list.append(gt_mi)

    gt_row = ["gt", "gt"] + [f"{v:.10f}" for v in gt_list] + [""]

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerow(gt_row)

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if len(rows) == 0 or rows[0] != header:
        raise ValueError(
            f"Existing bmi_results.csv header mismatch.\n"
            f"Expected: {header}\nFound: {rows[0] if rows else 'EMPTY'}"
        )
    if len(rows) < 2 or rows[1][:2] != ["gt", "gt"]:
        rows = [header, gt_row]

    last_mean_bias = float("nan")

    for sample_size in sample_sizes:
        if sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got {sample_size}")

        task_estimates = []
        all_biases = []

        for task_idx, task_name in enumerate(chosen_task_list):
            task_dir = os.path.join(data_root, task_name)
            gt_mi = gt_list[task_idx]

            picked_seeds = rng.choice(seeds, size=n_samples_to_use, replace=False)

            mi_estimates = []
            for seed in picked_seeds:
                x_path = os.path.join(task_dir, f"X_seed_{seed}.npz")
                y_path = os.path.join(task_dir, f"Y_seed_{seed}.npz")

                X = np.load(x_path)["X"]
                Y = np.load(y_path)["Y"]

                if sample_size > X.shape[0] or sample_size > Y.shape[0]:
                    raise ValueError(
                        f"sample_size={sample_size} is larger than saved samples ({X.shape[0]}).\n"
                        f"Task={task_name}, seed={seed}"
                    )

                X_sub = X[:sample_size]
                Y_sub = Y[:sample_size]

                mi_est = estimate_mi_xy(
                    torch.from_numpy(X_sub).unsqueeze(0).cpu(),
                    torch.from_numpy(Y_sub).unsqueeze(0).cpu(),
                    module, max_dim,
                    softrank_reg=softrank_reg,
                    gauss_copula=gauss_copula,
                )
                mi_val = float(mi_est.item()) if hasattr(mi_est, "item") else float(mi_est)
                mi_estimates.append(mi_val)
                all_biases.append(abs(gt_mi - mi_val))

            mean_est = float(np.mean(mi_estimates)) if mi_estimates else float("nan")
            task_estimates.append(mean_est)

            print(f"[BMI] step={training_step} | N={sample_size} | {task_name} | GT={gt_mi:.6f} | Est={mean_est:.6f}")

        mean_bias = float(np.mean(all_biases)) if all_biases else float("nan")
        last_mean_bias = mean_bias
        print(f"[BMI] Finish step={training_step}, sample_size={sample_size}, mean_bias={mean_bias:.6f}")

        step_row = [str(int(training_step)), str(int(sample_size))] + \
                   [f"{v:.10f}" for v in task_estimates] + [f"{mean_bias:.10f}"]

        step_str = str(int(training_step))
        size_str = str(int(sample_size))
        replaced = False
        for i in range(2, len(rows)):
            if len(rows[i]) >= 2 and rows[i][0] == step_str and rows[i][1] == size_str:
                rows[i] = step_row
                replaced = True
                break
        if not replaced:
            rows.append(step_row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)

    return last_mean_bias


# ============================================================
# Independence testing data generators
# ============================================================

def gen_independent(d, seq_len):
    mean = np.zeros(d)
    sigma = np.eye(d)
    X = np.random.multivariate_normal(mean, sigma, seq_len)
    Z = np.random.multivariate_normal(mean, sigma, seq_len)
    return X, Z


def gen_test1(d, seq_len):
    mean = np.zeros(d)
    sigma = np.eye(d)
    X = np.random.multivariate_normal(mean, sigma, seq_len)
    Z = np.random.multivariate_normal(mean, sigma, seq_len)
    tmp = np.dot(np.ones(d), X.T)
    Y = (np.outer(tmp, np.ones(d)) / np.sqrt(d) + Z) / np.sqrt(2)
    return X, Y


def gen_test2(d, seq_len):
    mean = np.zeros(d)
    sigma = np.eye(d)
    X = np.random.multivariate_normal(mean, sigma, seq_len)
    Z = np.random.multivariate_normal(mean, sigma, seq_len)
    tmp1 = np.zeros(d // 2)
    tmp2 = np.ones(d // 2)
    Y = np.zeros((seq_len, d))
    for i in range(d):
        if i <= d // 2:
            Y[:, i] = np.dot(np.concatenate((tmp2, tmp1)).T, X.T) + Z[:, i]
        else:
            Y[:, i] = np.dot(np.concatenate((tmp2, tmp1)).T, X.T) + Z[:, i]
    Y = Y / (d * np.sqrt(2))
    return X, Y


def gen_test3(d, seq_len):
    mean = np.zeros(d)
    sigma = np.eye(d)
    X = np.random.multivariate_normal(mean, sigma, seq_len)
    Z = np.random.multivariate_normal(mean, sigma, seq_len)
    Y = (X + Z) / np.sqrt(2)
    return X, Y


# ============================================================
# Independence testing AUC evaluation
# ============================================================

def evaluate_inde_auc(
    module,
    seq_len: int,
    method: str,
    d: int,
    *,
    max_dim: int,
    softrank_reg: float,
    proj_num: int = 256,
    num_test: int = 20,
    batchsize: int = 64,
    gauss_copula: bool = True,
    normalize_input: bool = True,
    rng: np.random.Generator | None = None,
):
    assert num_test % 2 == 0, "num_test must be even (half dependent, half independent)."
    if rng is None:
        rng = np.random.default_rng()

    smi_means = []
    y_labels = np.concatenate((np.ones(num_test // 2), np.zeros(num_test // 2)))

    for j in range(num_test // 2):
        if method == "test1":
            X, Y = gen_test1(d, seq_len)
        elif method == "test2":
            X, Y = gen_test2(d, seq_len)
        elif method == "test3":
            X, Y = gen_test3(d, seq_len)
        else:
            raise ValueError(f"Unknown method: {method}")

        smi_mean = compute_ksmi_mean(
            torch.from_numpy(X).float(),
            torch.from_numpy(Y).float(),
            projection_dim=max_dim,
            model=module,
            proj_num=proj_num,
            batchsize=batchsize,
            max_dim=max_dim,
            softrank_reg=softrank_reg,
            normalize_input=normalize_input,
            gauss_copula=gauss_copula,
        )
        smi_means.append(float(smi_mean))

    for j in range(num_test // 2):
        X, Y = gen_independent(d, seq_len)
        smi_mean = compute_ksmi_mean(
            torch.from_numpy(X).float(),
            torch.from_numpy(Y).float(),
            projection_dim=max_dim,
            model=module,
            proj_num=proj_num,
            batchsize=batchsize,
            max_dim=max_dim,
            softrank_reg=softrank_reg,
            normalize_input=normalize_input,
            gauss_copula=gauss_copula,
        )
        smi_means.append(float(smi_mean))

    fpr, tpr, _ = roc_curve(y_labels, np.array(smi_means))
    return float(auc(fpr, tpr))


# ============================================================
# Independence evaluation grid + CSV logging
# ============================================================

def run_inde_evaluation_and_log(
    module,
    *,
    log_dir: str,
    training_step: int,
    max_dim: int,
    softrank_reg: float,
    proj_num: int = 128,
    num_test: int = 20,
    d_list=(16, 128),
    seq_lens=[50, 100, 150, 250, 400],
    methods=("test1", "test2", "test3"),
    batchsize: int = 128,
    normalize_input: bool = True,
    gauss_copula: bool = True,
):
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "inde_results.csv")

    col_names = []
    for m in methods:
        for d in d_list:
            for s in seq_lens:
                col_names.append(f"{m}_d-{d}_seq-{s}")
    header = ["training_step"] + col_names + ["mean_auc"]

    rng = np.random.default_rng(int(training_step))

    results = {}
    for m in methods:
        for d in d_list:
            for s in seq_lens:
                key = f"{m}_d-{d}_seq-{s}"
                print(f"[INDE] step={training_step} | {key} | proj_num={proj_num} | num_test={num_test}")
                auc_val = evaluate_inde_auc(
                    module=module, seq_len=s, method=m, d=d,
                    max_dim=max_dim, softrank_reg=softrank_reg,
                    proj_num=proj_num, num_test=num_test,
                    batchsize=batchsize, normalize_input=normalize_input,
                    rng=rng, gauss_copula=gauss_copula,
                )
                results[key] = auc_val

    mean_auc = float(np.mean(list(results.values()))) if results else float("nan")

    row = [str(int(training_step))] + [f"{results[c]:.10f}" for c in col_names] + [f"{mean_auc:.10f}"]

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerow(row)
        return results, mean_auc

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if len(rows) == 0:
        rows = [header]
    elif rows[0] != header:
        raise ValueError(
            f"Existing inde_results.csv header mismatch.\n"
            f"Expected: {header}\nFound: {rows[0]}"
        )

    step_str = str(int(training_step))
    replaced = False
    for i in range(1, len(rows)):
        if len(rows[i]) > 0 and rows[i][0] == step_str:
            rows[i] = row
            replaced = True
            break
    if not replaced:
        rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)

    return results, mean_auc
