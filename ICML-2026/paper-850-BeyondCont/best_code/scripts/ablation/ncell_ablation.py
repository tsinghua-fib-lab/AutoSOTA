from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from utils import compute_uot_plans, USB, SDE, wasserstein, wasserstein_with_weights, sample_from_ot_plan, ma, plot_k_values, plot_comparisions
import time as TIME


################################################## 
# Parameter setting 
################################################## 
batch_size = 512
nu = 0.001
steps = 3000     
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
delta = 15

################################################ 
# Data loading
################################################ 
print('Begin data loading...') 
data_name = 'mouse' 

data_raw = pd.read_csv('data/Weinreb_alltime.csv') 

output = 'ncell_ablation/'+data_name+'/'+data_name+'.txt' 

for ncell in [10000,15000,20000,25000,30000,35000,40000,45000,49302]: 
    for run in range(5):
        data = data_raw.sample(n=ncell, random_state=42)
        
        # Xs[k] are data points from time point k
        Xs = [] 
        mass_ratio = [] 
        max_sample_time = int(np.max(data['samples']))
        for k in range(max_sample_time + 1): 
            Xs.append(np.array(data[data['samples'] == k])[:, 1:]) 
            mass_ratio.append(Xs[k].shape[0]/Xs[0].shape[0]) 
        
        dim = Xs[0].shape[1] 
        t_train = np.arange(len(Xs)).tolist() 
        t_vals_tensor = torch.tensor(t_train, dtype=torch.float32, device=device)
        samples_per_interval = np.array(data).shape[0]
    
    
    
        print(f'=== ncell = {ncell} ===') 
        
        start_time = TIME.perf_counter()
        # 1. Compute UOT Plans (Running once per delta)
        print('Computing UOT plans...') 
        uot_plans, gamma0_plans, gamma1_plans, true_action = compute_uot_plans(
            Xs, t_train, delta=delta, cuda=True, use_mini_batch_uot = True, chunk_size = 2000,
        ) 
    
        # 2. Flow Matching Training
        for seed in [113]: 
            torch.manual_seed(seed) 
            print(f'Seed = {seed}')
            
            model = USB([dim + 1, 256, 256, 256, 256, 256], nu=nu).to(device) 
            
            # -------------------------------------------------------
            # -------------------------------------------------------
            print('Pre-sampling training data on CPU -> GPU...')
            
            all_x0, all_x1, all_m0, all_m1, all_t_start, all_dt = [], [], [], [], [], []
            
            for k in range(len(t_train)-1):
                gamma0_plan = gamma0_plans[k] 
                gamma1_plan = gamma1_plans[k] 
    
                x0_np, x1_np, idx_0, idx_1 = sample_from_ot_plan(gamma0_plan, Xs[k], Xs[k+1], samples_per_interval)
                ratio = (gamma1_plan[idx_0, idx_1] / gamma0_plan[idx_0, idx_1]).reshape(-1, 1)
                m1_np = np.log(1e-8 + ratio)
                m0_np = np.zeros_like(m1_np)
    
                t_s = t_train[k]
                d_t = t_train[k+1] - t_train[k]
                
    
                all_x0.append(torch.tensor(x0_np, dtype=torch.float32))
                all_x1.append(torch.tensor(x1_np, dtype=torch.float32))
                all_m0.append(torch.tensor(m0_np, dtype=torch.float32))
                all_m1.append(torch.tensor(m1_np, dtype=torch.float32))
                
    
                all_t_start.append(torch.full((len(x0_np), 1), t_s, dtype=torch.float32))
                all_dt.append(torch.full((len(x0_np), 1), d_t, dtype=torch.float32))
    
    
            train_x0 = torch.cat(all_x0).to(device)
            train_x1 = torch.cat(all_x1).to(device)
            train_m0 = torch.cat(all_m0).to(device)
            train_m1 = torch.cat(all_m1).to(device)
            train_t_start = torch.cat(all_t_start).to(device)
            train_dt = torch.cat(all_dt).to(device)
    
            train_dataset = TensorDataset(train_x0, train_x1, train_m0, train_m1, train_t_start, train_dt)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            
            print(f'Data prepared. Total samples: {len(train_dataset)}. Batches per epoch: {len(train_loader)}')
    
            # --------------------------------------------------------------------
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)
            
            vlosses, slosses, klosses, losses = [], [], [], []
            
            model.train()
            global_step = 0
            keep_training = True
            
            print('Begin Flow Matching (Vectorized)...') 
            
            while keep_training:
                for batch_idx, (b_x0, b_x1, b_m0, b_m1, b_ts, b_dt) in enumerate(train_loader):
                    optimizer.zero_grad()
                    
                    # Sample relative t in [0, 1] on GPU
                    relative_t = torch.rand((b_x0.shape[0], 1), device=device)
                    
                    # Compute Real t
                    real_t = b_dt * relative_t + b_ts
                    
                    # Model Sampling (Vectorized for all time intervals at once)
                    xts_samp, vts_samp, kts_samp, eps_samp, mts_samp = model.sample_comditional_flow(
                        b_x0, b_x1, b_m0, b_m1, relative_t
                    )
                    
                    # Time rescaling
                    kts_samp = kts_samp / b_dt
                    vts_samp = vts_samp / b_dt
                    
                    # Model Forward
                    v, s, k_out = model.forward(xts_samp, real_t)
                    
                    # Loss Computation
                    weights = torch.exp(mts_samp)
                    v_loss = torch.mean(torch.pow(v - vts_samp, 2) * weights)
                    
                    # s_loss scaling term
                    bridge_term = 2 * torch.sqrt(relative_t * (1 - relative_t)) / (nu + 1e-8)
                    s_loss = torch.mean(torch.pow(bridge_term * s + eps_samp, 2) * weights)
                    
                    k_loss = torch.mean(torch.pow(k_out - kts_samp, 2) * weights)
                    
                    loss = v_loss + s_loss + k_loss
                    
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                    global_step += 1
                    
                    # Logging
                    if global_step % 200 == 0:
                        print(f"Stef {global_step}/{steps} | Loss: {loss.item():.4f} (v: {v_loss.item():.4f}, s: {s_loss.item():.4f}, k: {k_loss.item():.4f})")
                        vlosses.append(v_loss.item())
                        slosses.append(s_loss.item())
                        klosses.append(k_loss.item())
                        losses.append(loss.item())
                    
                    if global_step >= steps:
                        keep_training = False
                        break
            
            end_time = TIME.perf_counter()
            
            with open(output, 'a') as f:
                print('###################################################', file=f) 
                print(f'delta = {delta}', file=f) 
                print(f'Time: {end_time - start_time:.4f} s', file=f)
        