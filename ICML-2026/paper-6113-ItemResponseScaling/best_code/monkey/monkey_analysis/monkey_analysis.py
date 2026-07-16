import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from tueplots import bundles
bundles.icml2024()
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
import warnings
from huggingface_hub import snapshot_download
warnings.filterwarnings("ignore")
np.random.seed(0)
torch.manual_seed(42)

def estimate_success_rate_at_k_per_problem(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def compute_pass_datk_gts(data2d: np.ndarray, idxs: list, max_k: int) -> np.ndarray:
    k_range = np.arange(1, max_k + 1)
    pass_matrix = []
    for i in idxs:
        arr = data2d[i]
        valid = ~np.isnan(arr)
        n = valid.sum()
        c = np.nansum(arr)
        pass_matrix.append([
            estimate_success_rate_at_k_per_problem(n, int(c), k)
            for k in k_range
        ])
    return np.nanmean(np.vstack(pass_matrix), axis=0)

def cal_passdatk(pass_i1s: np.ndarray, k: int) -> float:
    return float(np.nanmean(1 - (1 - pass_i1s) ** k))

if __name__ == "__main__":
    output_dir = "../../result/monkey_analysis"
    os.makedirs(output_dir, exist_ok=True)

    input_dir = snapshot_download(
        repo_id="stair-lab/monkey_3d_data", 
        repo_type="dataset",
    )
    for fname in sorted(os.listdir(input_dir)[4:]):
        if not fname.endswith("_tensor.pth"):
            continue
        path = os.path.join(input_dir, fname)
        data = torch.load(path, map_location="cpu")
        dataset_name = fname.split("_tensor.pth")[0]
        tensor3d = data["data_tensor"]  # (n_takers, n_items, n_samples)
        print(tensor3d.shape)
        helm_zs_all = data["z"]              # (n_items,)
        models = data["models"]
        
        for model_idx in tqdm(range(tensor3d.shape[0])):
            data2d = tensor3d[model_idx].cpu().numpy()
            model_name = models[model_idx]
            
            item_mask = ~np.all(np.isnan(data2d), axis=1)
            data2d = data2d[item_mask]
            helm_zs = helm_zs_all[item_mask]
            sample_mask = ~np.all(np.isnan(data2d), axis=0)
            data2d = data2d[:, sample_mask]

            n_items, max_k = data2d.shape
            print(data2d.shape)
            if n_items <= 5 or max_k==5:
                continue
            pass_i1 = np.nanmean(data2d, axis=1)
            k_range = np.arange(1, max_k + 1)

            # --- split items for train/test by Helm z ---
            temperature = 0.1
            split_idx = n_items // 2
            probs = (helm_zs - helm_zs.min() + 1e-6).pow(1.0 / temperature)
            probs /= probs.sum()
            train_idxs = torch.multinomial(probs, split_idx, replacement=False).tolist()
            test_idxs = [i for i in range(n_items) if i not in train_idxs]
            train_zs = helm_zs[train_idxs].numpy()
            test_zs  = helm_zs[test_idxs].numpy()

            # --- GT ---
            train_pass_datk_gts = compute_pass_datk_gts(data2d, train_idxs, max_k)
            test_pass_datk_gts  = compute_pass_datk_gts(data2d, test_idxs,  max_k)

            # --- distributional estimator ---
            train_pass_dist = np.array([cal_passdatk(pass_i1[train_idxs], k) for k in k_range])

            # --- logistic regression estimator ---
            X_train = np.repeat(train_zs, max_k).reshape(-1, 1)
            y_train = data2d[train_idxs].reshape(-1)
            mask = ~np.isnan(y_train)
            X_train, y_train = X_train[mask], y_train[mask].astype(int)
            lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000)
            lr.fit(X_train, y_train)
            X_test = np.repeat(test_zs, max_k).reshape(-1, 1)
            y_test = data2d[test_idxs].reshape(-1)
            mask_test = ~np.isnan(y_test)
            X_test, y_test = X_test[mask_test], y_test[mask_test].astype(int)
            train_probs = lr.predict_proba(train_zs.reshape(-1, 1))[:, 1]
            test_probs  = lr.predict_proba(test_zs.reshape(-1, 1))[:, 1]
            train_pass_lr = np.array([cal_passdatk(train_probs, k) for k in k_range])
            test_pass_lr  = np.array([cal_passdatk(test_probs,  k) for k in k_range])

            # --- Plot pass@1 vs HELM z ---
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                plt.figure(figsize=(6,6))
                plt.scatter(train_zs, pass_i1[train_idxs], label="Train", alpha=0.7)
                plt.scatter(test_zs,  pass_i1[test_idxs],  label="Test",  alpha=0.7)
                z_range = np.linspace(helm_zs.min(), helm_zs.max(), 200).reshape(-1,1)
                lr_curve = lr.predict_proba(z_range)[:,1]
                plt.plot(z_range, lr_curve, linestyle='-', linewidth=2, label="LR fit")
                plt.xlabel("HELM $z$", fontsize=16)
                plt.ylabel("Pass@1", fontsize=16)
                plt.legend(fontsize=14)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/passat1_vs_helmz_{model_name}_{dataset_name}.png", dpi=300)

            # --- Plot pass@k vs k with GT, distributional & LR ---
            mse_train_dist = mean_squared_error(train_pass_datk_gts, train_pass_dist)
            mse_test_dist  = mean_squared_error(test_pass_datk_gts,  train_pass_dist)
            mse_train_lr   = mean_squared_error(train_pass_datk_gts, train_pass_lr)
            mse_test_lr    = mean_squared_error(test_pass_datk_gts,  test_pass_lr)
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                fig, ax = plt.subplots(figsize=(8,6))
                ax.semilogx(k_range, train_pass_datk_gts, linestyle='-',  color='blue', linewidth=2, label=f'Train GT', alpha=0.5)
                ax.semilogx(k_range, test_pass_datk_gts,  linestyle='-',  color='red',  linewidth=2, label=f'Test GT',  alpha=0.5)
                ax.semilogx(k_range, train_pass_dist,    linestyle='--', color='blue', label=f'Dist (Train MSE={mse_train_dist:.2e}, Test MSE={mse_test_dist:.2e})', alpha=0.5)
                ax.semilogx(k_range, train_pass_lr,      linestyle=':',  color='blue', label=f'Train LR (MSE={mse_train_lr:.2e})', alpha=0.5)
                ax.semilogx(k_range, test_pass_lr,       linestyle=':',  color='red',  label=f'Test LR (MSE={mse_test_lr:.2e})', alpha=0.5)
                ax.set_xlabel('$k$', fontsize=20)
                ax.set_ylabel('pass@k', fontsize=20)
                ax.tick_params(axis='both', labelsize=14)
                ax.legend(fontsize=12, frameon=False)
                fig.tight_layout()
                fig.savefig(f"{output_dir}/passatk_vs_k_{model_name}_{dataset_name}.png", dpi=300, bbox_inches="tight")

            # --- save results summary ---
            results = {
                'train_pass_datk_gts': train_pass_datk_gts,
                'test_pass_datk_gts':  test_pass_datk_gts,
                'train_pass_dist':     train_pass_dist,
                'train_pass_lr':       train_pass_lr,
                'test_pass_lr':        test_pass_lr,
            }
            with open(f"{output_dir}/result_{model_name}_{dataset_name}.pkl", "wb") as f:
                pickle.dump(results, f)
