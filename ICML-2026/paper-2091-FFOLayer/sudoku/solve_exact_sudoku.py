from models_sudoku import get_default_sudoku_params
from utils_sudoku import setup_cvx_qp_problem
import cvxpy as cp
import os


import numpy as np
import torch
import time
import os
import argparse
from torch.utils.data import DataLoader, TensorDataset, Subset

from utils_sudoku import decode_onehot, computeErr

n_threads = os.cpu_count()

n = 2
param_vals = get_default_sudoku_params(n=n, Qpenalty=0.1, get_equality=True)
A = param_vals["A"]
G = param_vals["G"]
h = param_vals["h"]
Q = param_vals["Q"]**0.5
b = param_vals["b"]

y_dim = (n**2)**3
num_ineq = param_vals["G"].shape[0]
num_eq = param_vals["A"].shape[0]

problem, objective, ineq_functions, eq_functions, params, variables = setup_cvx_qp_problem(opt_var_dim=y_dim, num_ineq=num_ineq, num_eq=num_eq)


 
batch_size = 1
device = torch.device('cpu') #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data_dir_path = f"sudoku/data/{n}"
features = torch.load(os.path.join(train_data_dir_path, "features.pt"))
labels = torch.load(os.path.join(train_data_dir_path, "labels.pt"))
features = torch.tensor(features, dtype=torch.float32).to(device)[:50000]
labels   = torch.tensor(labels, dtype=torch.float32).to(device)[:50000]
print(features.shape)
print(labels.shape)

dataset = TensorDataset(features, labels)   

num_samples = len(dataset)
train_split = 0.9
train_size = int(num_samples * train_split)
test_size = num_samples - train_size
train_indices = list(range(0, train_size))
test_indices = list(range(train_size, num_samples))
train_dataset = Subset(dataset, train_indices)
test_dataset  = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

###############################################
for i, (x, y) in enumerate(test_loader):
    x = x.to(device)
    y = y.to(device)
    print(x.shape)
    print(i)
    # params = [Q_cp, p_cp, G_cp, h_cp, A_cp, b_cp]
    
    
    params[0].value = Q.numpy()
    params[1].value = -(x.numpy().reshape(-1)).squeeze()
    params[2].value = G.numpy()
    params[3].value = h.numpy()
    params[4].value = A
    params[5].value = b.numpy()
    
    problem.solve(solver=cp.GUROBI, **{"Threads": n_threads, "OutputFlag": 0})
    
    puzzle_shape = x.shape
    pred = torch.tensor(variables[0].value.reshape(*puzzle_shape))
    print(pred.shape)
    print(f"pred logit: {pred}")
    A_torch = torch.tensor(A)
    b_torch = torch.tensor(b)
    assert(torch.matmul(A_torch, pred.reshape(-1)).allclose(b_torch))

    print(f"pred error: {computeErr(pred)}")
    print(f"GT error: {computeErr(y)}")
    print(f"num active: {torch.sum(pred<=1e-9)}")
    print(f"test example: {i}")
    print(f"input puzzle: \n{decode_onehot(x[0])}")
    print(f"gt sol:\n{decode_onehot(y[0])}")
    print(f"pred sol: \n{decode_onehot(pred[0])}")
    
    
    ### try active constraints
    print("######### Try active constraints ########")
    problem, objective, ineq_functions, eq_functions, params, variables = setup_cvx_qp_problem(opt_var_dim=y_dim, num_ineq=num_ineq, num_eq=104)

    active_mask = (pred.reshape(-1)<=1e-9) #(64,)
    active_constraints = torch.zeros((active_mask.shape[0], active_mask.shape[-1])) #(64, 64)
    print(f"active constraints shape: {active_constraints.shape}")
    active_constraints[active_mask, active_mask] = 1.0
    
    
    new_A = torch.cat([A_torch, active_constraints], dim=0)
    new_rank = np.linalg.matrix_rank(new_A.numpy(), tol=1e-12)
    print(f"new rank: {new_rank}, old rank: {np.linalg.matrix_rank(A, tol=1e-12)}")
    new_b = torch.cat([b_torch, torch.zeros(active_constraints.shape[0])], dim=0)
    
    params[0].value = Q.numpy()
    params[1].value = -(x.numpy().reshape(-1)).squeeze()
    params[2].value = G.numpy()
    params[3].value = h.numpy()
    params[4].value = new_A.numpy()
    params[5].value = new_b.numpy()
    
    problem.solve(solver=cp.GUROBI, **{"Threads": n_threads, "OutputFlag": 0})
    
    puzzle_shape = x.shape
    pred = torch.tensor(variables[0].value.reshape(*puzzle_shape))
    print(pred.shape)
    print(f"pred logit: {pred}")
    A_torch = torch.tensor(A)
    b_torch = torch.tensor(b)
    assert(torch.matmul(A_torch, pred.reshape(-1)).allclose(b_torch))

    print(f"pred error: {computeErr(pred)}")
    print(f"GT error: {computeErr(y)}")
    print(f"num active: {torch.sum(pred<=1e-9)}")
    print(f"test example: {i}")
    print(f"input puzzle: \n{decode_onehot(x[0])}")
    print(f"gt sol:\n{decode_onehot(y[0])}")
    print(f"pred sol: \n{decode_onehot(pred[0])}")
    

    input("Press Enter to continue...")

