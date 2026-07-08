

import numpy as np
import torch
import time
import sys
import os
import argparse

from models import *
from data import *
import pandas as pd


from torch.utils.tensorboard import SummaryWriter


       


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--ydim', type=int, default=900, help='dimension of y')
    
    parser.add_argument('--alpha', type=float, default=100, help='alpha')
    parser.add_argument('--dual_cutoff', type=float, default=1e-3, help='dual cutoff')
    parser.add_argument('--slack_tol', type=float, default=1e-8, help='slack tolerance')

    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--learn_constraint', type=int, default=0, help='whether to learn constraint')
    parser.add_argument('--suffix', type=str, default="", help='suffix to the result directory')
    
    
    
    
    
    args = parser.parse_args()

    method = args.method
    seed = args.seed
    num_epochs = args.epochs
    learning_rate = args.lr
    alpha = args.alpha
    dual_cutoff = args.dual_cutoff
    slack_tol = args.slack_tol

    # Set random seed for reproducibility
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    

    input_dim   = 640
    ydim  = args.ydim
    n = ydim
    num_samples = 2000 #2048
    batch_size = args.batch_size

    train_loader, test_loader = genData(device, input_dim, ydim, num_samples, batch_size)

    METHODS = [
        "cvxpylayer",
        "qpth",
        "lpgd",
        "ffoqp_eq_schur",
        "ffocp_eq"]
    
    SEEDS=[3,4]
    SAVE_PATH = f"../synthetic_results_epoch_zero"
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    summary_df = pd.DataFrame(columns=["method", "seed", "train_df_loss"])
    
    for seed in SEEDS:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        for method in METHODS:
            print(f"######## Running method: {method}, seed: {seed}")
            model = OptModel(input_dim, ydim, layer_type=method, constraint_learnable=(args.learn_constraint==1), batch_size=batch_size, device=device, alpha=alpha, dual_cutoff=dual_cutoff, slack_tol=slack_tol).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
            loss_fn = torch.nn.MSELoss()
            


            model.train()
            train_df_loss_list = []
            
            for i, (x, y) in enumerate(train_loader):
                z, y_pred = model(x) # (opt solution, predicted q)
                ts_loss = loss_fn(y_pred, y)
                df_loss = torch.mean(y * z)
                loss = df_loss
                train_df_loss_list.append(loss.item())
            train_df_loss = np.mean(train_df_loss_list)
            print(f"seed={seed},method={method},train_df_loss={train_df_loss}")
            
            summary_df = pd.concat([
                summary_df, 
                pd.DataFrame([{
                    "method": method,
                    "seed": seed,
                    "avg_train_df_loss": train_df_loss
                }])
            ], ignore_index=True)
            
            print(summary_df)
    
    summary_df.to_csv(os.path.join(SAVE_PATH, "epoch_0.csv"), index=False)
    
    