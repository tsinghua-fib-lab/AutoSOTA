import torch
import numpy as np

from cvxpylayers.torch import CvxpyLayer
from utils_sudoku import setup_cvx_qp_problem, get_sudoku_matrix
from models_sudoku import get_default_sudoku_params
import cvxpy as cp

def debug_infeasible_solver():
    nBatch = 1
    
    A = torch.load("sudoku/bad_A.pt")
    z0_a = torch.load("sudoku/bad_z0_a.pt")
    b = torch.load("sudoku/bad_b.pt")
    p = torch.load("sudoku/bad_p.pt")
    print("A shape: ", A.shape)
    print("z0_a shape: ", z0_a.shape)
    print("b shape: ", b.shape)
    print("p shape: ", p.shape)
    
    rec_b = torch.matmul(A,z0_a)
    diff = torch.norm(b-rec_b).item()
    print("diff b: ", diff)
    assert(diff==0)
    
    n = 2
    y_dim = (n**2)**3
    num_ineq = y_dim

    
    param_vals = get_default_sudoku_params(n, Qpenalty=0.1, get_equality=True)
    num_ineq = y_dim
    num_eq = A.shape[0]
    Q = param_vals["Q"]**0.5
    G = param_vals["G"]
    h = param_vals["h"]
    
    
    problem, objective, ineq_functions, eq_functions, params, variables = setup_cvx_qp_problem(opt_var_dim=y_dim, num_ineq=num_ineq, num_eq=num_eq)
    
    Q_cp, p_cp, G_cp, h_cp, A_cp, b_cp = params
    y_cp = variables[0]
    
    Q_cp.value = Q.double().cpu().numpy()
    p_cp.value = p.reshape(-1).double().cpu().numpy()
    G_cp.value = G.double().cpu().numpy()
    h_cp.value = h.double().cpu().numpy()
    A_cp.value = A.double().cpu().numpy()
    b_cp.value = b.double().cpu().numpy()
    
    y_cp = problem.solve(solver=cp.GUROBI)
    print("status: ", problem.status)
    print("optimal value: ", problem.value)
    print("optimal var: ", y_cp.value)
    print("optimal var shape: ", y_cp.value.shape)
    print("check eq constraints (should be close to 0): ", np.linalg.norm(A_cp.value @ y_cp.value - b_cp.value))
    print("check ineq constraints (should be <=0): ", G_cp.value @ y_cp.value - h_cp.value)
    print("p: ", p_cp.value)


    # optlayer = CvxpyLayer(problem, parameters=params, variables=variables)
    # Q_batched = Q.unsqueeze(0).expand(nBatch, -1, -1)   # (batch, y_dim, y_dim)
    # G_batched = G.unsqueeze(0).expand(nBatch, -1, -1)   # (batch, num_ineq, y_dim)
    # h_batched = h.unsqueeze(0).expand(nBatch, -1)       # (batch, num_ineq)
    # A_batched = A.unsqueeze(0).expand(nBatch, -1, -1)   # (batch, num_eq, y_dim)
    # b_batched = b.unsqueeze(0).expand(nBatch, -1)       # (batch, num_eq)
    # params_batched = [Q_batched, p, G_batched, h_batched, A_batched, b_batched]
    # sol, = optlayer(*params_batched)
    
    
    
    
    
    
    # rec_z0_a = torch.matmul(torch.linalg.inv(A), b)
    # diff = torch.norm(z0_a-rec_z0_a).item()
    # assert(diff==0)
    
if __name__=="__main__":
    debug_infeasible_solver()