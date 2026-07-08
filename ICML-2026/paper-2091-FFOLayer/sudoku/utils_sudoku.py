import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import time
import os
import argparse
import logging

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset, Subset

import cvxpy as cp


def setup_cvx_qp_problem(opt_var_dim, num_ineq, num_eq, ignore_ineq=False):
    '''
    set up quadratic program functions for cvxpy, Q_cp is the square root of Q in the QP
    '''
    Q_cp = cp.Parameter((opt_var_dim, opt_var_dim), PSD=True)
    p_cp = cp.Parameter(opt_var_dim)
    G_cp = cp.Parameter((num_ineq, opt_var_dim))
    h_cp = cp.Parameter(num_ineq)
    A_cp = cp.Parameter((num_eq, opt_var_dim))
    b_cp = cp.Parameter(num_eq)
    
    y_cp = cp.Variable(opt_var_dim)
    
    ### Q_cp is the square root of the actual Q in the QP, assuming Q_cp is symmetric
    objective_fn = 0.5 * cp.sum_squares(Q_cp @ y_cp) + p_cp.T @ y_cp
    
    # objective_fn = 0.5 * cp.quad_form(y_cp, Q_cp) + p_cp.T @ y_cp
    
    ### Construct xx^T and form quadratic expression
    # X = cp.reshape(y_cp, (opt_var_dim,1)) @ cp.reshape(y_cp, (1,opt_var_dim))
    # quad_form = cp.trace(Q_cp @ X)
    # objective_fn = 0.5 * quad_form + p_cp.T @ y_cp
    
    
    ineq_functions = [G_cp @ y_cp - h_cp]
    eq_functions = [A_cp @ y_cp -b_cp]
    ineq_constraints = [f<=0 for f in ineq_functions]
    if ignore_ineq:
        ineq_constraints=[]
    eq_constraints = [f==0 for f in eq_functions]
    
    problem = cp.Problem(cp.Minimize(objective_fn), eq_constraints+ineq_constraints)
    assert problem.is_dpp()
    
    params = [Q_cp, p_cp, G_cp, h_cp, A_cp, b_cp]
    if ignore_ineq:
        params = [Q_cp, p_cp, A_cp, b_cp]
    variables = [y_cp]
    
    return problem, objective_fn, ineq_functions, eq_functions, params, variables




def decode_onehot(encoded_board):
    """
    Take the unique argmax of the one-hot encoded board.
    
    encoded_board: (n**2, n**2, n**2), one-hot
        - dimensions correspond to (row, column, digit)
    """
    v,I = torch.max(encoded_board, -1) #(row, column)
    return ((v>0).long()*(I+1)).squeeze()



def get_sudoku_matrix(n):
    '''
    sudoku board size is n**2 x n**2
    
    return the equality constraint matrix A for the LP of sudoku, the shape is (K, n**6) for some K
    '''
    X = np.array([[cp.Variable(n**2) for i in range(n**2)] for j in range(n**2)]) #(row, col, digit), row and column are square number
    cons = ([x >= 0 for row in X for x in row] + # one-hot (#row * #col * #digits constraints)
            [cp.sum(x) == 1 for row in X for x in row] + # each cell contains exactly one digit (#row * #col), [1, 1, 1, 1, 0, 0, 0, 0...]
            [sum(row) == np.ones(n**2) for row in X] + # each row contains n**2 unique digits (#row * #digit) [1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0]
            [sum([row[i] for row in X]) == np.ones(n**2) for i in range(n**2)] + # each column contains n**2 unique digit (#col * #digit) [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.. 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1]
            [sum([sum(row[i:i+n]) for row in X[j:j+n]]) == np.ones(n**2) for i in range(0,n**2,n) for j in range(0, n**2, n)]) # n**2 unique digits in n x n blocks (n**2 * n**2 * n**2)
    f = sum([cp.sum(x) for row in X for x in row])
    prob = cp.Problem(cp.Minimize(f), cons)

    A = np.asarray(prob.get_problem_data(cp.GUROBI)[0]["A"].todense())
    # print(f"A shape: {A.shape}")
    A0 = [A[0]]
    rank = 1
    for i in range(1,A.shape[0]):
        if np.linalg.matrix_rank(A0+[A[i]], tol=1e-12) > rank:
            A0.append(A[i])
            rank += 1
        # else:
        #     print(i, A[i])


    return np.array(A0)



def computeErr(pred):
    '''
    return how many predicted boards are incorrect
    '''
    # pred has size (B, n^2, n^2, n^2)
    batchSz = pred.size(0)
    nsq = int(pred.size(1))
    n = int(np.sqrt(nsq))
    s = (nsq-1)*nsq//2 # 0 + 1 + ... + n^2-1
    I = torch.max(pred, 3)[1].squeeze().view(batchSz, nsq, nsq) # decode last axis, shape (B, n^2, n^2)

    def invalidGroups(x):
        '''
        x is a (batchSz, nsq) slice (row, col, or block).

        a valid group must:
        1. contain all digits from 0 to nsq-1
        2. each digit appears exactly once
        
        return true if group is not valid, boolean tensor of shape (batchSz,)
        '''
        # valid = (x.min(1)[0] == 0)
        # valid *= (x.max(1)[0] == nsq-1)
        # valid *= (x.sum(1) == s)
        counts = torch.stack([torch.bincount(row, minlength=nsq) for row in x])
        valid = (counts == 1).all(dim=1)
        return ~valid

    boardCorrect = torch.ones(batchSz).type_as(pred)
    for j in range(nsq):
        # Check the jth row and column.
        boardCorrect[invalidGroups(I[:,j,:])] = 0 
        boardCorrect[invalidGroups(I[:,:,j])] = 0 

        # Check the jth block.
        row, col = n*(j // n), n*(j % n)
        M = invalidGroups(I[:,row:row+n,col:col+n].contiguous().view(batchSz,-1))
        boardCorrect[M] = 0

        if boardCorrect.sum() == 0:
            return batchSz

    return batchSz-boardCorrect.sum().item()


def create_logger(logging_root, log_name):
    logger = logging.getLogger('my_logger')
    # Set the default logging level (this can be adjusted as needed)
    logger.setLevel(logging.DEBUG)
    # Create two handlers for logging to two different files
    file_handler1 = logging.FileHandler(os.path.join(logging_root, log_name))
    # Set the log level for each handler (optional)
    file_handler1.setLevel(logging.DEBUG)   # For detailed logging
    # Create a formatter to define the log message format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Assign the formatter to both handlers
    file_handler1.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(file_handler1)

    # Example logging
    # self.logger.debug("This is a debug message")
    # self.logger.info("This is an info message")
    # self.logger.warning("This is a warning message")
    # self.logger.error("This is an error message")
    # self.logger.critical("This is a critical message")
    logger.info("\n")
    logger.info(">>>>>>> START LOGGING <<<<<<<<")
    return logger

if __name__=="__main__":
    ######### board encoding #######
    # encoded_board = torch.zeros((9,4,4))
    # encoded_board[-1,:,:]=1
    # encoded_board[-1,1,1]=0
    # print(decode_onehot(encoded_board))
    
    ######### sudoku matrix #########
    A_new = get_sudoku_matrix(3)
    print(A_new.shape)