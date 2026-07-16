import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tueplots import bundles
bundles.icml2024()
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
import warnings
from huggingface_hub import snapshot_download
from tqdm import tqdm
from torch.optim import LBFGS
warnings.filterwarnings("ignore")
np.random.seed(0)
torch.manual_seed(42)

def estimate_success_rate_at_k_per_problem(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def compute_pass_datk_gts(scores_2d: np.ndarray, idxs: list, max_k: int) -> np.ndarray:
    k_range = np.arange(1, max_k + 1)
    pass_matrix = []
    for i in idxs:
        arr = scores_2d[i]
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

def trainer(params, optim, closure, n_iter=500, verbose=True):
    pbar = tqdm(range(n_iter)) if verbose else range(n_iter)
    for it in pbar:
        if it > 0:
            prev = [p.clone() for p in params]
            prev_loss = loss.clone()
        loss = optim.step(closure)
        if it > 0:
            dloss = (prev_loss - loss).item()
            dparam = sum(torch.norm(a-b).item() for a,b in zip(prev, params))
            gradn = sum(torch.norm(p.grad).item() for p in params if p.grad is not None)
            if verbose:
                pbar.set_postfix({"grad_norm": gradn, "d_param": dparam, "d_loss": dloss})
            if dloss < 1e-5 and dparam < 1e-5 and gradn < 1e-5:
                break
    return params

def fit_mse_rasch(resp_matrix, device, train_percentage=0.8):
    n_takers, n_items = resp_matrix.shape
    data_withnan = torch.tensor(resp_matrix, dtype=torch.float, device=device)
    data_withneg1 = data_withnan.nan_to_num(nan=-1.0)
    data_idtor = (data_withneg1 != -1).to(torch.float)
    data = data_withneg1 * data_idtor

    valid = False
    while not valid:
        train_idtor = torch.bernoulli(data_idtor * train_percentage).int()
        test_idtor = (data_idtor - train_idtor).int()
        valid = (train_idtor.sum(dim=1) != 0).all() and (train_idtor.sum(dim=0) != 0).all()

    thetas_nuisance = torch.randn(150, n_takers, device=device)
    z = torch.randn(n_items, requires_grad=True, device=device)
    optim_z = LBFGS([z], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    def closure_z():
        optim_z.zero_grad()
        preds = torch.sigmoid(thetas_nuisance[:, :, None] + z[None, None, :])
        loss = ((preds - data)**2 * train_idtor.float()).sum() / train_idtor.sum()
        loss.backward()
        return loss
    zs = trainer([z], optim_z, closure_z)[0].detach()

    thetas = torch.randn(n_takers, requires_grad=True, device=device)
    oopt_th = LBFGS([thetas], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    def closure_th():
        oopt_th.zero_grad()
        preds = torch.sigmoid(thetas[:, None] + zs[None, :])
        loss = ((preds - data)**2 * train_idtor.float()).sum() / train_idtor.sum()
        loss.backward()
        return loss
    thetas = trainer([thetas], oopt_th, closure_th)[0].detach()

    return thetas.cpu(), zs.cpu(), train_idtor.bool().cpu().numpy(), test_idtor.bool().cpu().numpy()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "../../result/monkey_analysis_denoise_data"
    os.makedirs(output_dir, exist_ok=True)

    cache_dir = snapshot_download(
        repo_id="stair-lab/denoise_eval_3d_data",
        repo_type="dataset"
    )
    input_path = f"{cache_dir}/resmat.pt"
    data = torch.load(input_path, map_location="cpu")
    scores, models, datasets = data["data_tensor"], data["models"], data["datasets"]
    
    exclude_models = ["gemma-3-12b-it", "Llama-3.3-70B-Instruct", "Mistral-Small-3.2-24B-Instruct-2506", "Phi-4-mini-reasoning"]
    mask_models = [name not in exclude_models for name in models]
    models = [m for m, keep in zip(models, mask_models) if keep]
    scores = scores[mask_models, :, :]
    print(f"Shape: {scores.shape}")
    print(f"NaN count: {torch.isnan(scores).sum()}, NaN percentage: {torch.isnan(scores).sum() / scores.numel():.2f}%")
    set_models = set(models)
    set_datasets = set(datasets)
    
    _, zs, _, _ = fit_mse_rasch(torch.nanmean(scores, axis=2), device, train_percentage=1)
    
    for model in tqdm(set_models):
        for dataset in set_datasets:
            print(model, dataset)
            model_idx = models.index(model)
            dataset_idxs = [i for i, d in enumerate(datasets) if d == dataset]
            scores_2d = scores[model_idx][dataset_idxs, :].numpy()
            sub_zs = zs[dataset_idxs]
            n_items, max_k = scores_2d.shape
            pass_i1 = np.nanmean(scores_2d, axis=1)
            k_range = np.arange(1, max_k + 1)

            # --- split items for train/test by z ---
            temperature = 0.1
            split_idx = n_items // 2
            probs = (sub_zs - sub_zs.min() + 1e-6).pow(1.0 / temperature)
            probs /= probs.sum()
            train_idxs = torch.multinomial(probs, split_idx, replacement=False).tolist()
            test_idxs = [i for i in range(n_items) if i not in train_idxs]
            train_zs = sub_zs[train_idxs].numpy()
            test_zs  = sub_zs[test_idxs].numpy()

            # --- GT ---
            train_pass_datk_gts = compute_pass_datk_gts(scores_2d, train_idxs, max_k)
            test_pass_datk_gts  = compute_pass_datk_gts(scores_2d, test_idxs,  max_k)

            # --- distributional estimator ---
            # train_pass_dist = np.array([cal_passdatk(pass_i1[train_idxs], k) for k in k_range])

            # --- logistic regression estimator ---
            X_train = np.repeat(train_zs, max_k).reshape(-1, 1)
            y_train = scores_2d[train_idxs].reshape(-1)
            mask = ~np.isnan(y_train)
            X_train, y_train = X_train[mask], y_train[mask].astype(int)
            lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000)
            lr.fit(X_train, y_train)
            X_test = np.repeat(test_zs, max_k).reshape(-1, 1)
            y_test = scores_2d[test_idxs].reshape(-1)
            mask_test = ~np.isnan(y_test)
            X_test, y_test = X_test[mask_test], y_test[mask_test].astype(int)
            train_probs = lr.predict_proba(train_zs.reshape(-1, 1))[:, 1]
            test_probs  = lr.predict_proba(test_zs.reshape(-1, 1))[:, 1]
            train_pass_lr = np.array([cal_passdatk(train_probs, k) for k in k_range])
            test_pass_lr  = np.array([cal_passdatk(test_probs,  k) for k in k_range])

            # --- Plot pass@1 vs z ---
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                plt.figure(figsize=(6,6))
                plt.scatter(train_zs, pass_i1[train_idxs], label="Train GT", alpha=0.7)
                plt.scatter(test_zs,  pass_i1[test_idxs],  label="Test GT",  alpha=0.7)
                z_range = np.linspace(sub_zs.min(), sub_zs.max(), 200).reshape(-1,1)
                lr_curve = lr.predict_proba(z_range)[:,1]
                plt.plot(z_range, lr_curve, linestyle='-', linewidth=2, label="IRT fit")
                plt.xlabel("$z$", fontsize=16)
                plt.ylabel("pass@1", fontsize=16)
                plt.tick_params(axis='both', labelsize=14)
                plt.legend(fontsize=14)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/passat1_vs_z_{model}_{dataset}.png", dpi=300)

            # --- Plot pass@k vs k with GT, distributional & LR ---
            # mse_train_dist = mean_squared_error(train_pass_datk_gts, train_pass_dist)
            # mse_test_dist  = mean_squared_error(test_pass_datk_gts,  train_pass_dist)
            mse_train_lr   = mean_squared_error(train_pass_datk_gts, train_pass_lr)
            mse_test_lr    = mean_squared_error(test_pass_datk_gts,  test_pass_lr)
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                fig, ax = plt.subplots(figsize=(8,6))
                ax.semilogx(k_range, train_pass_datk_gts, linestyle='-',  color='blue', linewidth=2, label=f'Train GT', alpha=0.5)
                ax.semilogx(k_range, test_pass_datk_gts,  linestyle='-',  color='red',  linewidth=2, label=f'Test GT',  alpha=0.5)
                # ax.semilogx(k_range, train_pass_dist,    linestyle='--', color='blue', label=f'Dist (Train MSE={mse_train_dist:.2e}, Test MSE={mse_test_dist:.2e})', alpha=0.5)
                ax.semilogx(k_range, train_pass_lr,      linestyle=':',  color='blue', label=f'Train IRT (MSE={mse_train_lr:.2e})', alpha=0.5)
                ax.semilogx(k_range, test_pass_lr,       linestyle=':',  color='red',  label=f'Test IRT (MSE={mse_test_lr:.2e})', alpha=0.5)
                ax.set_xlabel('$k$', fontsize=20)
                ax.set_ylabel('pass@k', fontsize=20)
                ax.tick_params(axis='both', labelsize=14)
                ax.legend(fontsize=12, frameon=False)
                fig.tight_layout()
                fig.savefig(f"{output_dir}/passatk_vs_k_{model}_{dataset}.png", dpi=300, bbox_inches="tight")

            # --- save results summary ---
            results = {
                'train_pass_datk_gts': train_pass_datk_gts,
                'test_pass_datk_gts':  test_pass_datk_gts,
                # 'train_pass_dist':     train_pass_dist,
                'train_pass_lr':       train_pass_lr,
                'test_pass_lr':        test_pass_lr,
            }
            with open(f"{output_dir}/result_{model}_{dataset}.pkl", "wb") as f:
                pickle.dump(results, f)
