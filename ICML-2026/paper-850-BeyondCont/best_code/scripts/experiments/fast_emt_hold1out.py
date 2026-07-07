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
from utils import compute_uot_plans, USB, SDE, wasserstein, wasserstein_with_weights, sample_from_ot_plan

################################################## 
# Parameter setting 
################################################## 
batch_size = 256     
nu = 0.001 
steps = 3000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

################################################ 
# Data loading
################################################ 
data_name = 'emt' 
print('Begin data loading...') 

data = pd.read_csv('data/emt.csv') 

output = 'stochastic/'+data_name+'/'+data_name+'_hold1out.txt' 

# Xs[k] are data points from time point k
Xs_whole = [] 
mass_ratio_whole = [] 
max_sample_time = int(np.max(data['samples']))

for k in range(max_sample_time + 1): 
    Xs_whole.append(np.array(data[data['samples'] == k])[:, 1:]) 
    mass_ratio_whole.append(Xs_whole[k].shape[0]/Xs_whole[0].shape[0]) 

dim = Xs_whole[0].shape[1] 
t_train_whole = np.arange(len(Xs_whole)).tolist() 

samples_per_interval = np.array(data).shape[0]

# -----------------------------------------------------------
# Loop over deltas
# -----------------------------------------------------------
# for delta in [2,5,7,10,15,20,25,30,35,40,45,50,55,60,80,100]: 
for delta in [1.4]*5+[1.5]*5+[1.6]*5: 
    print(f'#################### Delta = {delta} ####################') 
    
    wa_unnormalized_list = [] 
    wa_normalized_list = [] 
    RME_list = [] 
    total_mass_list = [] 
    action_list = [] 
    
    # -------------------------------------------------------
    # Loop over Hold-Out timepoints
    # -------------------------------------------------------
    for hold_out in range(1, len(Xs_whole)-1): 
        print(f'=== Hold Out Timepoint: {hold_out} ===')
        
        # 1. Reconstruct Dataset (Removing the held-out timepoint)
        Xs = [] 
        t_train = [] 
        mass_ratio = [] 
        for tmp in range(len(Xs_whole)): 
            if tmp != hold_out: 
                Xs.append(Xs_whole[tmp]) 
                t_train.append(t_train_whole[tmp]) 
                mass_ratio.append(mass_ratio_whole[tmp]) 
        
        # 2. Compute UOT Plans
        print('Computing UOT plans...') 
        
        uot_plans, gamma0_plans, gamma1_plans, true_action = compute_uot_plans(
            Xs, t_train, delta=delta, use_mini_batch_uot=True, chunk_size=10000, cuda=True
        ) 

        # 3. Initialize Model and Optimizer
        
        torch.manual_seed(113) 
        model = USB([dim + 1, 256, 256, 256, 256, 256], nu=nu).to(device) 
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)
        
        # -------------------------------------------------------
        
        # -------------------------------------------------------
        print('Pre-sampling training data...')
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

        
        train_dataset = TensorDataset(
            torch.cat(all_x0).to(device),
            torch.cat(all_x1).to(device),
            torch.cat(all_m0).to(device),
            torch.cat(all_m1).to(device),
            torch.cat(all_t_start).to(device),
            torch.cat(all_dt).to(device)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # -------------------------------------------------------
        # Vectorized Training Loop
        # -------------------------------------------------------
        print('Begin Flow Matching Training...')
        model.train()
        global_step = 0
        keep_training = True
        
        while keep_training:
            for batch_data in train_loader:
                b_x0, b_x1, b_m0, b_m1, b_ts, b_dt = batch_data
                
                optimizer.zero_grad()
                
                # Sample relative t
                relative_t = torch.rand((b_x0.shape[0], 1), device=device)
                real_t = b_dt * relative_t + b_ts
                
                # Model Sampling
                xts_samp, vts_samp, kts_samp, eps_samp, mts_samp = model.sample_comditional_flow(
                    b_x0, b_x1, b_m0, b_m1, relative_t
                )
                
                # Rescaling
                kts_samp = kts_samp / b_dt
                vts_samp = vts_samp / b_dt
                
                # Loss
                v, s, k_out = model.forward(xts_samp, real_t)
                weights = torch.exp(mts_samp)
                
                v_loss = torch.mean(torch.pow(v - vts_samp, 2) * weights)
                bridge_term = 2 * torch.sqrt(relative_t * (1 - relative_t)) / (nu + 1e-8)
                s_loss = torch.mean(torch.pow(bridge_term * s + eps_samp, 2) * weights)
                k_loss = torch.mean(torch.pow(k_out - kts_samp, 2) * weights)
                
                loss = v_loss + s_loss + k_loss
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                global_step += 1
                if global_step >= steps:
                    keep_training = False
                    break
        
        print(f'Training finished for Hold-Out {hold_out}. Final Loss: {loss.item():.4f}')

        # -------------------------------------------------------
        # Inference & Evaluation
        # -------------------------------------------------------
        print('Begin Inference...')
        model.eval() 
        model.to('cpu') 
        simulator = SDE(model, nu, mode = 1, positive=False) 
        
        x_source = torch.tensor(Xs_whole[0], dtype=torch.float32, device = 'cpu') 
        
        # Generate trajectory for the FULL time range
       
        xs, ms, action = simulator.trajectory(
            x = x_source, 
            m = torch.zeros([x_source.size(0),1], device = 'cpu'), 
            delta = delta, 
            T = int(np.max(data['samples'])), 
            N = int(np.max(data['samples']))*100
        )
        
        # Evaluation at the held-out time point
        idx_eval = hold_out * 100
        if idx_eval >= xs.shape[0]: idx_eval = xs.shape[0] - 1
        
        x1s = xs[idx_eval, :] 
        m1s = np.exp(ms[idx_eval, :]) 
        
        # Ground Truth is Xs_whole[hold_out]
        wa_unnormalized = wasserstein(
            torch.Tensor(np.array(x1s)), 
            torch.tensor(Xs_whole[hold_out], dtype=torch.float32), 
            power=1
        ) 
        
        # Wasserstein distance with mass
        wa_normalized = wasserstein_with_weights(
            torch.Tensor(np.array(x1s)), 
            np.array(m1s), 
            torch.Tensor(Xs_whole[hold_out]), 
            np.ones(Xs_whole[hold_out].shape[0]), 
            power=1
        ) 
        
        # Relative mass error
        m_ratio = Xs_whole[hold_out].shape[0] / Xs_whole[0].shape[0] 
        RME = np.abs((np.mean(m1s) - mass_ratio_whole[hold_out]) / mass_ratio_whole[hold_out]) 
    
        wa_unnormalized_list.append(wa_unnormalized) 
        wa_normalized_list.append(wa_normalized) 
        RME_list.append(RME) 
        total_mass_list.append(np.mean(m1s)) 
        action_list.append(action)
        
        # Clean up Memory
        del train_dataset, train_loader, uot_plans, gamma0_plans, gamma1_plans
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    # -------------------------------------------------------
    # Logging per Delta
    # -------------------------------------------------------
    with open(output, 'a') as file:
        print('###################################################', file=file) 
        print(f'delta = {delta}', file=file) 
        print(f'Wasserstein distance without mass: {wa_unnormalized_list}', file=file) 
        print(f'Wasserstein distance with mass: {wa_normalized_list}', file=file) 
        print(f'RME = {RME_list}', file=file) 
    
    print(f'Delta {delta} Done.')