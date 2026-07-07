import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd
import numpy as np
import math
import time
from copy import deepcopy
import os


class NNOptimizer(nn.Module):
    
    @staticmethod 
    def divide_train_val(x, y, ratio=0.80):
        n = len(x)
        n_train = int(ratio*n)
        x_train, y_train = x[0:n_train], y[0:n_train]
        x_val, y_val = x[n_train:n], y[n_train:n]
        return  x_train, y_train, x_val, y_val
    
    @staticmethod 
    def learn(net, x, y, shuffle=False):    
        # shuffle data
        if shuffle:
            idx = torch.randperm(len(x))
            x, y = x[idx].clone().detach(), y[idx].clone().detach()

        # hyperparams 
        T = 2000 if not hasattr(net, 'max_iteration') else net.max_iteration 
        bs = 200 if not hasattr(net, 'bs') else net.bs
        lr = 5e-4 if not hasattr(net, 'lr') else net.lr
        wd = 0e-5 if not hasattr(net, 'wd') else net.wd
        PRINTING = True if not hasattr(net, 'trace_learning') else net.trace_learning
        T_NO_IMPROVE_THRESHOLD = 200 if not hasattr(net, 't_patience') else net.t_patience

        # divide train & val
        n = len(x)
        x_train, y_train, x_val, y_val = NNOptimizer.divide_train_val(x, y)
        net.device = x.device

        # adapt batch size for small samples
        if len(x_train) < 1000:
            bs = max(len(x_train) // 2, 1)

        # learn in loops
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=wd)
        n_batch, n_val_batch = int(len(x_train)/bs), int(len(x_val)/1000) if len(x_val) > 1000 else 1
        best_val_loss, best_model_state_dict, best_t, no_improvement = math.inf, None, 0, 0
                
        for t in range(T):
            # shuffle the batch
            idx = torch.randperm(len(x_train)) 
            x_train, y_train = x_train[idx], y_train[idx]
            x_chunks, y_chunks = torch.chunk(x_train, n_batch), torch.chunk(y_train, n_batch)
            x_v_chunks, y_v_chunks = torch.chunk(x_val, n_val_batch), torch.chunk(y_val, n_val_batch)

            # gradient descend
            net.train()
            for i in range(len(x_chunks)):
                optimizer.zero_grad()
                loss = -net.objective_func(x_chunks[i], y_chunks[i])
                if t>0:
                    loss.backward()
                    optimizer.step()
              
            # early stopping if val loss does not improve after some epochs
            net.eval()
            loss_val = torch.zeros(1, device=x.device)
            with torch.no_grad():
                for j in range(len(x_v_chunks)):
                    loss_val += -net.objective_func(x_v_chunks[j], y_v_chunks[j])/len(x_v_chunks)
            if loss_val.item() < best_val_loss:
                no_improvement = 0 
                best_val_loss = loss_val.item() 
                best_model_state_dict = deepcopy(net.state_dict())
                best_t = t
            else:
                no_improvement += 1
                best_val_loss = best_val_loss
                best_model_state_dict = best_model_state_dict
            if no_improvement >= T_NO_IMPROVE_THRESHOLD: break
            # report
            if PRINTING and t%(T//20+1) == 0: 
               print('finished: t=', t, 'loss=', loss.item(), 'loss val=', loss_val.item(), 'best val loss=', best_val_loss, 'best t=', best_t)
        print('\n')
                
        # return the best snapshot in the history
        net.load_state_dict(best_model_state_dict)
        return best_val_loss


class NNOptimizer_save(nn.Module):
    
    @staticmethod 
    def divide_train_val(x, y, ratio=0.80):
        n = len(x)
        n_train = int(ratio*n)
        x_train, y_train = x[0:n_train], y[0:n_train]
        x_val, y_val = x[n_train:n], y[n_train:n]
        return x_train, y_train, x_val, y_val
    
    @staticmethod 
    def learn(net, x, y, shuffle=False, save_mi_history=False, output_dir="results/clip/mi_history", mi_interval=1):    
        # shuffle data
        if shuffle:
            idx = torch.randperm(len(x))
            x, y = x[idx].clone().detach(), y[idx].clone().detach()

        # hyperparams 
        T = 500 
        bs = 200 if not hasattr(net, 'bs') else net.bs 
        lr = 5e-4 if not hasattr(net, 'lr') else net.lr
        wd = 0e-5 if not hasattr(net, 'wd') else net.wd
        PRINTING = True if not hasattr(net, 'trace_learning') else net.trace_learning  
        T_NO_IMPROVE_THRESHOLD = 400
        
        # divide train & val
        n = len(x)
        x_train, y_train, x_val, y_val = NNOptimizer_save.divide_train_val(x, y)
        net.device = x.device

        # adapt batch size for small samples
        if len(x_train) < 1000:
            bs = max(len(x_train) // 2, 1)

        # initialize MI history on net object
        net.mi_history = []

        # learn in loops
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=wd)
        n_batch, n_val_batch = int(len(x_train)/bs), int(len(x_val)/1000) if len(x_val) > 1000 else 1
        best_val_loss, best_model_state_dict, best_t, no_improvement = math.inf, None, 0, 0
                
        for t in range(T):
            # shuffle the batch
            idx = torch.randperm(len(x_train)) 
            x_train, y_train = x_train[idx], y_train[idx]
            x_chunks, y_chunks = torch.chunk(x_train, n_batch), torch.chunk(y_train, n_batch)
            x_v_chunks, y_v_chunks = torch.chunk(x_val, n_val_batch), torch.chunk(y_val, n_val_batch)

            # gradient descent
            net.train()
            for i in range(len(x_chunks)):
                optimizer.zero_grad()
                loss = -net.objective_func(x_chunks[i], y_chunks[i])
                if t > 0:
                    loss.backward()
                    optimizer.step()
            
            
            # early stopping if val loss does not improve after some epochs
            net.eval()
            loss_val = torch.zeros(1, device=x.device)
            with torch.no_grad():
                for j in range(len(x_v_chunks)):
                    loss_val += -net.objective_func(x_v_chunks[j], y_v_chunks[j])/len(x_v_chunks)
            
            # record MI using net.MI(X, Y)
            if t % mi_interval == 0 or t == T - 1:
                net.eval()
                with torch.no_grad():
                    mi = net.MI(x, y)
                    # Handle MINDE's tuple output (mi, mi_sigma)
                    mi_value = mi[0] if isinstance(mi, tuple) else mi
                    mi_value = mi_value.item() if torch.is_tensor(mi_value) else mi_value
                    net.mi_history.append(mi_value)
                    
                    # print MI at specific iteration (e.g., 500)
                    if t == 500 - 1:  # 0-based indexing
                        print(f"Iteration {t + 1}: MI = {mi_value:.4f}")
            
            
            if loss_val.item() < best_val_loss:
                no_improvement = 0 
                best_val_loss = loss_val.item() 
                best_model_state_dict = deepcopy(net.state_dict())
                best_t = t
            else:
                no_improvement += 1
            if no_improvement >= T_NO_IMPROVE_THRESHOLD:
                break
            
            # report
            if PRINTING and t % (T // 20 + 1) == 0: 
                print(f'finished: t={t}, loss={loss.item():.4f}, best t={best_t}, MI={mi_value:.4f}')
        
        # save MI history
        if save_mi_history:
            os.makedirs(output_dir, exist_ok=True)
            np.savez(
                os.path.join(output_dir, f"mi_history_{net.__class__.__name__}.npz"),
                iterations=np.arange(0, T, mi_interval)[:len(net.mi_history)],
                mi=np.array(net.mi_history)
            )
            print(f"MI history saved to {os.path.join(output_dir, f'mi_history_{net.__class__.__name__}.npz')}")
        
        # return the best snapshot in the history
        net.load_state_dict(best_model_state_dict)
        return best_val_loss