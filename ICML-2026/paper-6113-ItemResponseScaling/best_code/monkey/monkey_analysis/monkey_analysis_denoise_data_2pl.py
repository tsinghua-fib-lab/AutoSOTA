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
    """Fit 1PL (Rasch) model: p = sigmoid(θ - z)"""
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

def fit_mse_2pl(resp_matrix, device, train_percentage=0.8):
    """Fit 2PL model: p = sigmoid(a*θ - z)"""
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

    # Initialize parameters
    thetas_nuisance = torch.randn(150, n_takers, device=device)
    z = torch.randn(n_items, requires_grad=True, device=device)
    a = torch.ones(n_items, requires_grad=True, device=device)  # discrimination parameters

    # Optimize z and a together
    optim_za = LBFGS([z, a], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    def closure_za():
        optim_za.zero_grad()
        # 2PL: sigmoid(a * theta - z) = sigmoid(a * (theta - z/a))
        preds = torch.sigmoid(a[None, None, :] * thetas_nuisance[:, :, None] + z[None, None, :])
        loss = ((preds - data)**2 * train_idtor.float()).sum() / train_idtor.sum()
        loss.backward()
        return loss
    z, a = trainer([z, a], optim_za, closure_za, verbose=True)
    zs = z.detach()
    alphas = a.detach()

    # Optimize theta with fixed z and a
    thetas = torch.randn(n_takers, requires_grad=True, device=device)
    oopt_th = LBFGS([thetas], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    def closure_th():
        oopt_th.zero_grad()
        preds = torch.sigmoid(alphas[None, :] * thetas[:, None] + zs[None, :])
        loss = ((preds - data)**2 * train_idtor.float()).sum() / train_idtor.sum()
        loss.backward()
        return loss
    thetas = trainer([thetas], oopt_th, closure_th)[0].detach()

    return thetas.cpu(), zs.cpu(), alphas.cpu(), train_idtor.bool().cpu().numpy(), test_idtor.bool().cpu().numpy()

# TwoPL_LogisticRegression class removed - was incorrectly fitting a single global
# discrimination parameter instead of using per-question parameters from fit_mse_2pl

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "/home/v-tatruong/generalized-scaling-laws/result/monkey_analysis_denoise_data_2pl"
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

    # Fit both 1PL and 2PL models for comparison
    print("\n=== Fitting 1PL (Rasch) model ===")
    thetas_1pl, zs_1pl, _, _ = fit_mse_rasch(torch.nanmean(scores, axis=2), device, train_percentage=1)

    print("\n=== Fitting 2PL model ===")
    thetas_2pl, zs_2pl, alphas_2pl, _, _ = fit_mse_2pl(torch.nanmean(scores, axis=2), device, train_percentage=1)

    print(f"\n2PL discrimination parameters (a):")
    print(f"  Mean: {alphas_2pl.mean():.3f}, Std: {alphas_2pl.std():.3f}")
    print(f"  Min: {alphas_2pl.min():.3f}, Max: {alphas_2pl.max():.3f}")

    for model in tqdm(set_models):
        for dataset in set_datasets:
            print(f"\n{model}, {dataset}")
            model_idx = models.index(model)
            dataset_idxs = [i for i, d in enumerate(datasets) if d == dataset]
            scores_2d = scores[model_idx][dataset_idxs, :].numpy()
            sub_zs_1pl = zs_1pl[dataset_idxs]
            sub_zs_2pl = zs_2pl[dataset_idxs]
            sub_alphas_2pl = alphas_2pl[dataset_idxs]
            n_items, max_k = scores_2d.shape
            pass_i1 = np.nanmean(scores_2d, axis=1)
            k_range = np.arange(1, max_k + 1)

            # Split items for train/test by z
            temperature = 0.1
            split_idx = n_items // 2
            probs = (sub_zs_1pl - sub_zs_1pl.min() + 1e-6).pow(1.0 / temperature)
            probs /= probs.sum()
            train_idxs = torch.multinomial(probs, split_idx, replacement=False).tolist()
            test_idxs = [i for i in range(n_items) if i not in train_idxs]

            train_zs_1pl = sub_zs_1pl[train_idxs].numpy()
            test_zs_1pl  = sub_zs_1pl[test_idxs].numpy()
            train_zs_2pl = sub_zs_2pl[train_idxs].numpy()
            test_zs_2pl  = sub_zs_2pl[test_idxs].numpy()
            train_alphas_2pl = sub_alphas_2pl[train_idxs].numpy()
            test_alphas_2pl  = sub_alphas_2pl[test_idxs].numpy()

            # Ground truth
            train_pass_datk_gts = compute_pass_datk_gts(scores_2d, train_idxs, max_k)
            test_pass_datk_gts  = compute_pass_datk_gts(scores_2d, test_idxs,  max_k)

            # --- Get model-specific ability parameters ---
            # Use the globally-fitted theta values (one per model)
            theta_1pl = thetas_1pl[model_idx].item()
            theta_2pl = thetas_2pl[model_idx].item()

            # --- 1PL (Rasch) model: p = sigmoid(θ + z) ---
            # Use per-question difficulty z and model ability θ
            train_probs_1pl = 1 / (1 + np.exp(-(theta_1pl + train_zs_1pl)))
            test_probs_1pl = 1 / (1 + np.exp(-(theta_1pl + test_zs_1pl)))
            train_pass_1pl = np.array([cal_passdatk(train_probs_1pl, k) for k in k_range])
            test_pass_1pl = np.array([cal_passdatk(test_probs_1pl, k) for k in k_range])

            # --- 2PL model: p = sigmoid(α*θ + z) ---
            # Use per-question discrimination α, difficulty z, and model ability θ
            train_probs_2pl = 1 / (1 + np.exp(-(train_alphas_2pl * theta_2pl + train_zs_2pl)))
            test_probs_2pl = 1 / (1 + np.exp(-(test_alphas_2pl * theta_2pl + test_zs_2pl)))
            train_pass_2pl = np.array([cal_passdatk(train_probs_2pl, k) for k in k_range])
            test_pass_2pl = np.array([cal_passdatk(test_probs_2pl, k) for k in k_range])

            # Compute MSEs
            mse_train_1pl = mean_squared_error(train_pass_datk_gts, train_pass_1pl)
            mse_test_1pl  = mean_squared_error(test_pass_datk_gts,  test_pass_1pl)
            mse_train_2pl = mean_squared_error(train_pass_datk_gts, train_pass_2pl)
            mse_test_2pl  = mean_squared_error(test_pass_datk_gts,  test_pass_2pl)

            print(f"  1PL: Train MSE={mse_train_1pl:.4f}, Test MSE={mse_test_1pl:.4f}")
            print(f"  2PL: Train MSE={mse_train_2pl:.4f}, Test MSE={mse_test_2pl:.4f}")
            print(f"  Improvement: {(mse_test_1pl - mse_test_2pl)/mse_test_1pl*100:.1f}%")

            # --- Plot pass@1 vs z (comparing 1PL and 2PL) ---
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                plt.figure(figsize=(8,6))
                plt.scatter(train_zs_1pl, pass_i1[train_idxs], label="Train GT", alpha=0.7, s=50)
                plt.scatter(test_zs_1pl,  pass_i1[test_idxs],  label="Test GT",  alpha=0.7, s=50)

                # Generate curves using the IRT model directly
                z_range = np.linspace(sub_zs_1pl.numpy().min(), sub_zs_1pl.numpy().max(), 200)
                # For 2PL curve, use mean discrimination parameter for visualization
                mean_alpha = sub_alphas_2pl.numpy().mean()
                curve_1pl = 1 / (1 + np.exp(-(theta_1pl + z_range)))
                curve_2pl = 1 / (1 + np.exp(-(mean_alpha * theta_2pl + z_range)))

                plt.plot(z_range, curve_1pl, linestyle='-', linewidth=2, label="1PL (Rasch)", color='blue')
                plt.plot(z_range, curve_2pl, linestyle='--', linewidth=2, label="2PL", color='red')

                plt.xlabel("$z$ (difficulty)", fontsize=16)
                plt.ylabel("pass@1", fontsize=16)
                plt.title(f"{model} on {dataset}", fontsize=14)
                plt.tick_params(axis='both', labelsize=14)
                plt.legend(fontsize=12)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/passat1_vs_z_{model}_{dataset}.png", dpi=300)
                plt.close()

            # --- Plot pass@k vs k (comparing 1PL and 2PL) ---
            with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
                fig, ax = plt.subplots(figsize=(10,6))
                ax.semilogx(k_range, train_pass_datk_gts, linestyle='-',  color='blue', linewidth=2, label='Train GT', alpha=0.5)
                ax.semilogx(k_range, test_pass_datk_gts,  linestyle='-',  color='red',  linewidth=2, label='Test GT',  alpha=0.5)
                ax.semilogx(k_range, train_pass_1pl,      linestyle=':',  color='blue', linewidth=2, label=f'1PL Train (MSE={mse_train_1pl:.4f})')
                ax.semilogx(k_range, test_pass_1pl,       linestyle=':',  color='red',  linewidth=2, label=f'1PL Test (MSE={mse_test_1pl:.4f})')
                ax.semilogx(k_range, train_pass_2pl,      linestyle='--', color='cyan', linewidth=2, label=f'2PL Train (MSE={mse_train_2pl:.4f})')
                ax.semilogx(k_range, test_pass_2pl,       linestyle='--', color='orange', linewidth=2, label=f'2PL Test (MSE={mse_test_2pl:.4f})')
                ax.set_xlabel('$k$ (samples)', fontsize=20)
                ax.set_ylabel('pass@k', fontsize=20)
                ax.set_title(f"{model} on {dataset}", fontsize=14)
                ax.tick_params(axis='both', labelsize=14)
                ax.legend(fontsize=10, frameon=False, loc='lower right')
                fig.tight_layout()
                fig.savefig(f"{output_dir}/passatk_vs_k_{model}_{dataset}.png", dpi=300, bbox_inches="tight")
                plt.close()

            # Save results
            results = {
                'train_pass_datk_gts': train_pass_datk_gts,
                'test_pass_datk_gts':  test_pass_datk_gts,
                'train_pass_1pl':      train_pass_1pl,
                'test_pass_1pl':       test_pass_1pl,
                'train_pass_2pl':      train_pass_2pl,
                'test_pass_2pl':       test_pass_2pl,
                'mse_train_1pl':       mse_train_1pl,
                'mse_test_1pl':        mse_test_1pl,
                'mse_train_2pl':       mse_train_2pl,
                'mse_test_2pl':        mse_test_2pl,
                'train_alphas_2pl':    train_alphas_2pl,
                'test_alphas_2pl':     test_alphas_2pl,
                'theta_1pl':           theta_1pl,
                'theta_2pl':           theta_2pl,
            }
            with open(f"{output_dir}/result_{model}_{dataset}.pkl", "wb") as f:
                pickle.dump(results, f)

    print("\n=== Analysis complete! ===")
    print(f"Results saved to: {output_dir}/")
