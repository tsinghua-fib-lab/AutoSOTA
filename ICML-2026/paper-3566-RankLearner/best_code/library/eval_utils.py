import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from metrics import *
from models import *

####################################################################################

class EvalDataset(Dataset):
    def __init__(self, df, confounders):
        self.df = df
        self.confounders = confounders

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(row[self.confounders].astype(np.float32).to_numpy(), dtype=torch.float32) 
        cate = torch.tensor(float(row['cate']), dtype=torch.float32)
        M0 = torch.tensor(float(row['M0']), dtype=torch.float32)
        M1 = torch.tensor(float(row['M1']), dtype=torch.float32)
        
        return x, cate, M0, M1


####################################################################################


def get_estimates_all(rank_learner, pi_model, dr_model, m0_model, m1_model, test_loader, device):
    # put models to eval
    for model in [rank_learner, pi_model, dr_model, m0_model, m1_model]:
        model.eval()
    
    # collectors
    rank_learner_hat, pi_hat, DR_hat, T_hat, cate_true, M0_true, M1_true = [], [], [], [], [], [], []
    
    # loop over test loader
    with torch.inference_mode():
        for x, cate, M0, M1 in test_loader:
            x = x.to(device)
            preds = dr_model(x).squeeze(-1)

            pred_0 = m0_model(x)
            pred_1 = m1_model(x)
            t_preds = pred_1 - pred_0

            rl_preds = rank_learner(x).squeeze(-1)
            pi_preds = pi_model(x).squeeze(-1)
            
            # collect
            rank_learner_hat.append(rl_preds)
            pi_hat.append(pi_preds)
            DR_hat.append(preds)
            T_hat.append(t_preds)
            cate_true.append(cate.squeeze(-1))
            M0_true.append(M0.squeeze(-1))
            M1_true.append(M1.squeeze(-1))
    
    # store
    to_np = lambda parts: torch.cat(parts, dim=0).detach().cpu().numpy()
    df_eval = pd.DataFrame({
        "DR_hat": to_np(DR_hat),     
        "T_hat": to_np(T_hat),    
        "rank_learner_score": to_np(rank_learner_hat),     
        "plugin_score": to_np(pi_hat),    
        "cate":    to_np(cate_true),    
        "M0":      to_np(M0_true),
        "M1":      to_np(M1_true)})

    return df_eval


####################################################################################


def compute_metrics_all(eval_df):
    # check eval dataframe
    required = {
        "DR_hat",
        "T_hat",
        "rank_learner_score",
        "plugin_score",
        "cate"}

    missing = required - set(eval_df.columns)
    if missing:
        raise ValueError(f"eval_df is missing required columns: {sorted(missing)}")

    specs = [
        ("oracle", "cate"),
        ("DR", "DR_hat"),
        ("T", "T_hat"),
        ("rank_learner", "rank_learner_score"),
        ("plug_in", "plugin_score")]

    # evaluation
    rows = []
    for name, score_col in specs:
        ranked = eval_df.sort_values(score_col, ascending=False).copy()

        rows.append({
            "model": name,
            "autoc": autoc(ranked),
            "policy_value": policy_value(ranked),
        })

    return pd.DataFrame(rows)
