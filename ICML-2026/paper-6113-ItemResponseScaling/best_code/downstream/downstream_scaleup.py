import pickle
import torch
torch.manual_seed(0)
from torch.distributions import Bernoulli
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from scipy.optimize import curve_fit
from sklearn.kernel_ridge import KernelRidge
from huggingface_hub import snapshot_download

def irt_formula(theta, z, guess):
    return guess + (1-guess)*torch.sigmoid(theta[:, None] + z[None, :])

def estimate_theta_all(asked_ys, asked_zs, device, guess):
    def closure():
        optim.zero_grad()
        mask = ~torch.isnan(asked_ys)
        probs = irt_formula(theta, asked_zs, guess)
        loss = -Bernoulli(probs=probs[mask]).log_prob(asked_ys[mask]).mean()
        loss.backward()
        return loss

    theta = torch.zeros((asked_ys.shape[0],), requires_grad=True, device=device)
    optim = torch.optim.LBFGS([theta], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    
    for iteration in range(100):
        if iteration > 0:
            previous_theta = theta.clone()
            previous_loss = loss.clone()
        
        loss = optim.step(closure)
        
        if iteration > 0:
            d_loss = previous_loss - loss
            d_theta = torch.norm(previous_theta - theta, p=2)
            grad_norm = torch.norm(optim.param_groups[0]["params"][0].grad, p=2)
            if d_loss < 1e-5 and d_theta < 1e-5 and grad_norm < 1e-5:
                break
    
    return theta.detach()

def linear_func(flop, w, b):
    return w * flop + b

def moving_average(x, window=5):
    kernel = np.ones(window) / window
    # 'same' keeps the output the same length as x, with edge effects averaged
    return np.convolve(x, kernel, mode='same')    
    
if __name__ == "__main__":
    device = "cuda:1"
    
    # split_method = "random_55split"
    split_method = "hardeasy_split"
    # split_method = "small_subset_split"
    # split_method = "random_28split"
    
    repo_ids = [
        "EleutherAI/pythia-12b",
        # "EleutherAI/pythia-6.9b",
        # "EleutherAI/pythia-2.8b",
        # "EleutherAI/pythia-1.4b",
        # "EleutherAI/pythia-1b",
        # "EleutherAI/pythia-410m",
        # "EleutherAI/pythia-160m",
        # "EleutherAI/pythia-70m",
        # "EleutherAI/pythia-14m",
        # "LLM360/Amber",
        # "HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints",
        # "HuggingFaceTB/SmolLM2-360M-intermediate-checkpoints",
        # "HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints",
    ]
    model_names = [repo_id.split("/")[1] for repo_id in repo_ids]
    
    scenarios = ['wikifact']
    # scenarios = ['babi_qa', 'civil_comments', 'commonsense',
    #    'dyck_language_np=3', 'entity_data_imputation', 'entity_matching',
    #    'gsm', 'legal_support', 'legalbench', 'mmlu', 'raft',
    #    'synthetic_reasoning', 'wikifact'] # 'med_qa', 'boolq', 'imdb'
    scenario2guess = {
        'babi_qa': 0,
        'civil_comments': 0.5,
        'commonsense': 0.25,
        'dyck_language_np=3': 0,
        'entity_data_imputation': 0,
        'entity_matching': 0.5,
        'gsm': 0,
        'legal_support': 0.5,
        'legalbench': 0.2,
        'mmlu': 0.25,
        'raft': 0.5,
        'synthetic_reasoning': 0,
        'wikifact': 0,
    }

    results_dict = defaultdict(lambda: defaultdict(dict))
    for scenario in tqdm(scenarios):
        guess = scenario2guess[scenario]
        for model_name in model_names:
            cache_dir = snapshot_download(repo_id="stair-lab/irsl_downstream_resmat1_binary", repo_type="dataset")
            with open(f"{cache_dir}/results_{model_name}.pkl", "rb") as f:
                results = pickle.load(f)
            keep_cols = ~results.columns.get_level_values("z").isna()
            results = results.loc[:, keep_cols]
            results = results.loc[:, results.columns.get_level_values("scenario") == scenario]
            results = results[~results.isna().all(axis=1)]
            
            # discard first 10% ckpts
            n_discard = int(results.shape[0] * 0.1)
            results = results.iloc[n_discard:, :]

            ys = torch.tensor(results.values, dtype=torch.float, device=device)
            n_test_takers, n_items = ys.shape
            zs = results.columns.get_level_values("z").astype(float).to_numpy()
            zs = torch.tensor(zs, dtype=torch.float, device=device)
            time_steps = np.array([float(name.split("-")[-1]) for name in results.index])
            log_time_steps = np.log(time_steps+1e-6)

            if split_method == "hardeasy_split":
                # sorted_indices = torch.argsort(zs, descending=True)
                item_means = torch.nanmean(ys, dim=0)
                sorted_indices = torch.argsort(item_means)
                split_idx = n_items // 2
                # train_indices = sorted_indices[:split_idx]   # Smaller zs → train
                # test_indices = sorted_indices[split_idx:]    # Larger zs → test
                train_indices = sorted_indices[split_idx:]   # Smaller zs → train
                test_indices = sorted_indices[:split_idx]    # Larger zs → test

            elif split_method == "random_55split":
                perm = torch.randperm(n_items)
                split_idx = n_items // 2
                train_indices = perm[:split_idx]
                test_indices = perm[split_idx:]
            
            elif split_method == "small_subset_split":
                subset_size = 50
                if n_items < subset_size*2:
                    raise ValueError("No enough items")
                perm = torch.randperm(n_items)
                train_indices = perm[:subset_size]       # first 50 random items
                test_indices  = perm[subset_size:subset_size*2]    # next 50 random items
                
            elif split_method == "random_28split":
                perm = torch.randperm(n_items)
                split_idx = int(n_items * 0.2)           # 20% train
                train_indices = perm[:split_idx]
                test_indices = perm[split_idx:]          # 80% test
                    
            ys_train = ys[:, train_indices]
            ys_test = ys[:, test_indices]
            zs_train = zs[train_indices]
            zs_test = zs[test_indices]

            # gt
            gt_ctt_train = torch.nanmean(ys_train, dim=1).cpu().numpy()
            gt_ctt_test = torch.nanmean(ys_test, dim=1).cpu().numpy()
            # smooth gt
            # gt_ctt_train = moving_average(gt_ctt_train, window=5)
            # gt_ctt_test  = moving_average(gt_ctt_test,  window=5)
            
            # # classic linear
            # popt, _ = curve_fit(linear_func, log_time_steps, gt_ctt_train)
            # w, b = popt
            # train_linears = linear_func(log_time_steps, w, b)
            
            # classic kernel ridge
            X = log_time_steps.reshape(-1, 1)
            kr = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)
            kr.fit(X, gt_ctt_train)
            train_linears = kr.predict(X)

            # IRT
            train_theta_gt = estimate_theta_all(ys_train, zs_train, device, guess)
            # popt_irt, _ = curve_fit(linear_func, log_time_steps, train_theta_gt.cpu().numpy())
            # w_irt, b_irt = popt_irt
            # train_theta = linear_func(log_time_steps, w_irt, b_irt)
            kr_irt = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)
            kr_irt.fit(X, train_theta_gt.cpu().numpy())
            train_theta = kr_irt.predict(X)

            train_theta = torch.tensor(train_theta, dtype=torch.float, device=device)
            train_probs = irt_formula(train_theta, zs_train, guess)
            # train_irts = torch.bernoulli(train_probs).mean(dim=1).cpu().numpy()
            train_irts = train_probs.mean(dim=1).cpu().numpy()
            test_probs = irt_formula(train_theta, zs_test, guess)
            # test_irts = torch.bernoulli(test_probs).mean(dim=1).cpu().numpy()
            test_irts = test_probs.mean(dim=1).cpu().numpy()
            
            results_dict[scenario][model_name] = {
                "time_steps": time_steps,
                "gt_ctt_train": gt_ctt_train,
                "gt_ctt_test": gt_ctt_test,
                "train_linears": train_linears,
                "train_irts": train_irts,
                "test_irts": test_irts,
                "zs_train": zs_train,
                "zs_test": zs_test,
            }
            
    final_results_dict = {k: dict(v) for k, v in results_dict.items()}
    with open(f"downstream_binary_generalize_{split_method}.pkl", "wb") as f:
        pickle.dump(final_results_dict, f)