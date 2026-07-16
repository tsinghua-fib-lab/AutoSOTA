import pickle
import torch
torch.manual_seed(0)
torch.set_num_threads(1)
from joblib import Parallel, delayed
from collections import defaultdict
from tqdm import tqdm
import sys
sys.path.append("..")
from utils import cat_beta_2pl, cat_binary_2pl

if __name__ == "__main__":
    device = "cpu"
    with open(f"preprocessed_resmat2.pkl", "rb") as f:
        results_dict = pickle.load(f)
    results_dict = defaultdict(lambda: defaultdict(dict), results_dict)
        
    for dataset, fam_dict in results_dict.items():
        for model_family, value_dict in fam_dict.items():
            zs = torch.tensor(value_dict["zs"], dtype=torch.float, device=device)
            discris = torch.tensor(value_dict["discris"], dtype=torch.float, device=device)
            ys_pvocab = torch.tensor(value_dict["resmat_prob_vocab_correct"].values, dtype=torch.float, device=device)
            ys_pchoices = torch.tensor(value_dict["resmat_prob_choices_correct"].values, dtype=torch.float, device=device)
            ys_acc = torch.tensor(value_dict["resmat_acc"].values, dtype=torch.float, device=device)
            
            def _run_one_pvocab(i):
                return cat_beta_2pl(ys_pvocab[i], discris, zs, device)
            thetass_pvocab = Parallel(n_jobs=-1)(delayed(_run_one_pvocab)(i) for i in tqdm(range(ys_pvocab.shape[0])))
            thetass_pvocab = torch.tensor(thetass_pvocab, dtype=torch.float) # (n_models, budget)
            results_dict[dataset][model_family]["thetass_pvocab"] = thetass_pvocab.numpy()
            
            def _run_one_pchoices(i):
                return cat_beta_2pl(ys_pchoices[i], discris, zs, device)
            thetass_pchoices = Parallel(n_jobs=-1)(delayed(_run_one_pchoices)(i) for i in tqdm(range(ys_pchoices.shape[0])))
            thetass_pchoices = torch.tensor(thetass_pchoices, dtype=torch.float)
            results_dict[dataset][model_family]["thetass_pchoices"] = thetass_pchoices.numpy()
            
            def _run_one_acc(i):
                return cat_binary_2pl(ys_acc[i], discris, zs, device)
            thetass_acc = Parallel(n_jobs=-1)(delayed(_run_one_acc)(i) for i in tqdm(range(ys_acc.shape[0])))
            thetass_acc = torch.tensor(thetass_acc, dtype=torch.float)
            results_dict[dataset][model_family]["thetass_acc"] = thetass_acc.numpy()
        
    final_results_dict = {k: dict(v) for k, v in results_dict.items()}
    with open(f"withtheta_resmat2.pkl", "wb") as f:
        pickle.dump(final_results_dict, f)