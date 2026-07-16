from tqdm import tqdm
from torchmetrics import AUROC
auroc = AUROC(task="binary")
import torch
from torch.distributions import Bernoulli
import pickle
from torch.optim import LBFGS
import pandas as pd
import json

def trainer(parameters, optim, closure, n_iter=100, verbose=True):
    pbar = tqdm(range(n_iter)) if verbose else range(n_iter)
    for iteration in pbar:
        if iteration > 0:
            # Clone each tensor individually for previous state
            previous_parameters = [p.clone() for p in parameters]
            previous_loss = loss.clone()
        
        loss = optim.step(closure)
        
        if iteration > 0:
            d_loss = (previous_loss - loss).item()
            d_parameters = sum(
                torch.norm(prev - curr, p=2).item()
                for prev, curr in zip(previous_parameters, parameters)
            )
            grad_norm = sum(torch.norm(p.grad, p=2).item() for p in parameters if p.grad is not None)
            if verbose:
                pbar.set_postfix({"grad_norm": grad_norm, "d_parameter": d_parameters, "d_loss": d_loss})
            
            if d_loss < 1e-5 and d_parameters < 1e-5 and grad_norm < 1e-5:
                break
            
    return parameters

if __name__ == "__main__":
    torch.manual_seed(0)
    B = 50000
    n_thetas_nuisance = 150
    device = "cuda:6"
    with open("../data/gather_helm_data/results_mmlu_subject.pkl", "rb") as f:
        results = pickle.load(f)
    data = torch.tensor(results.values, dtype=torch.float, device=device)
    n_test_takers, n_items = data.shape
    
    optimized_z = []
    thetas_nuisance = torch.randn(n_thetas_nuisance, n_test_takers, device=device)
    for i in tqdm(range(0, n_items, B)):
        data_batch = data[:, i:i+B]
        current_B = data_batch.shape[1]
        z = torch.randn(current_B, requires_grad=True, device=device)
        optim_z = LBFGS([z], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
        def closure_z():
            optim_z.zero_grad()
            mask = ~torch.isnan(data_batch).expand(n_thetas_nuisance, -1, -1)
            
            probs = torch.sigmoid(thetas_nuisance[:, :, None] + z[None, None, :])
            loss = -(Bernoulli(probs=probs[mask]).log_prob(
                data_batch.expand(n_thetas_nuisance, -1, -1)[mask]
            )).mean()
            loss.backward()
            return loss
        z_optimized = trainer([z], optim_z, closure_z)[0].detach()
        optimized_z.append(z_optimized)
    zs = torch.cat(optimized_z)

    # add a new column key to matrix: z
    new_columns = []
    for col, z_val in zip(results.columns, zs.cpu().numpy()):
        new_columns.append(col + (z_val,))  # Append z value to tuple
    results.columns = pd.MultiIndex.from_tuples(new_columns, names=["input.text", "references", "scenario", "benchmark", "subject", "z"])
    
    with open("../data/gather_helm_data/results_with_z_mmlu_subject.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # texts = results.columns.get_level_values("input.text")
    # zs = results.columns.get_level_values("z").astype(float)
    # z_dict = dict(zip(texts, zs))
    # with open("../data/gather_helm_data/input_to_z.json", "w") as fp:
    #     json.dump(z_dict, fp, ensure_ascii=False, indent=2)
    
    # thetas = torch.randn(n_test_takers, requires_grad=True, device=device)
    # optim_theta = LBFGS([thetas], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    # def closure_theta():
    #     optim_theta.zero_grad()
    #     mask = ~torch.isnan(data)
    #     probs = torch.sigmoid(thetas[:, None] + zs[None, :])
    #     loss = -(Bernoulli(probs=probs[mask]).log_prob(data[mask])).mean()
    #     loss.backward()
    #     return loss
    # thetas = trainer([thetas], optim_theta, closure_theta)[0].detach()