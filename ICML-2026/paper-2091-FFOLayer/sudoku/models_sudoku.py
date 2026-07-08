import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np

from src.ffolayer.ffocp_eq import FFOLayer
from src.ffolayer.ffoqp_eq import FFOQPLayer

from dqp import dQP
from baselines.BPQP import BPQPLayer
from baselines.AltDiff import AltDiffLayer
from baselines.qpthlocal.qp import QPFunction
from baselines.cvxpylayers_local.cvxpylayer import CvxpyLayer

from utils_sudoku import setup_cvx_qp_problem, get_sudoku_matrix
from constants import FFOCP_EQ, QPTH, LPGD, CVXPY_LAYER, FFOQP_EQ, LPGD_QP, BPQP, DQP, FFOQP_EQ_SCHUR, ALTDIFF


def get_default_sudoku_params(n, Qpenalty=0.1, get_equality=True):
    '''
    return the correct value for the Sudoku LP's params: Q, G, h. Optionally return A, b.
    
    Args:
        n: n**2 is the Sudoku board length
        QPenalty: small coefficient for the quadratic term for QP
        get_equality: whether get A, b
    '''
    y_dim = (n**2)**3
    num_ineq = y_dim
    
    Q = Qpenalty * torch.eye(y_dim, dtype=torch.double)
    G = -torch.eye(num_ineq, dtype=torch.double)
    h = torch.zeros(num_ineq, dtype=torch.double)
    
    if get_equality:
        A = get_sudoku_matrix(n)
        num_eq = A.shape[0]
        b = torch.ones(num_eq, dtype=torch.double)
        
        return {"Q":Q, "G":G, "h":h, "A":A, "b":b}
        
    return {"Q":Q, "G":G, "h":h, "b":b}

def get_feasible_b(A, z0):
    '''
    get a vector b such that the equality constraint Ay=b can be satisfied
    
    i.e. return b = A(z0)
    
    Args:
        - A: (num_eq, y_dim)
        - z0: (y_dim,)
    '''
    return torch.matmul(A, z0)
    
def get_feasible_h(G, z0, s0):
    '''
    get a vector h such that the inequality constraint Gy<=h can be satisfied
    
    i.e. return h = G(z0) + s0 where s0 are all positive
    
    Args:
        - G: (num_ineq, y_dim)
        - s0: (num_ineq, )
        - z0: (y_dim,)
    '''
    assert(not torch.any(s0<0))
    return torch.matmul(G, z0) + s0

def get_Q_from_L(self, L, eps):
    '''
    get the Q matrix (Hessian) from lower triangular square matrix L
    
    Args:
        - L: lower triangular matrix of shape (output_dim, output_dim)
        - eps: a small number keeping Q positive definite
    '''
    L_mask =  torch.tril(torch.ones(L.shape[0], L.shape[0]))
    assert(torch.sum(L[(L_mask==0)])==0) # check upper triangular part of L is zero
    I = torch.eye(L.shape[0], device=L.device, dtype=L.dtype)
    Q = L.mm(L.t()) + eps*I
    return Q

class SingleOptLayerSudoku(nn.Module):
    def __init__(self, n, learnable_parts, layer_type, Qpenalty=0.1, alpha=100, init_learnable_vals=None, dual_cutoff=1e-3, slack_tol=1e-6, batch_size=32, warm_start=True):
        '''
        The architecture is {parameter - optLayer}.
        
        solve sudoku , represented as the linear program (It is QP here because we add a small quadratic term):
        min_y eps/2*y^Ty-p^Ty
            s.t. Ay=b
                Gy<=h
            
        Args:
            - delta = 1/alpha, which is the perturbation constant for finite difference
            - learnable_parts: a list of strings chosen from ["ineq", "eq"]
            - QPenalty: eps in the LP
            - init_learnable_vals: a dictionary containing initial values for learnable parts, keys can be "A", "G", "z0_a", "z0_g", "s0"
        '''
        super().__init__()
        self.layer_type = layer_type
        assert(layer_type in [QPTH, FFOCP_EQ, FFOQP_EQ_SCHUR, LPGD, CVXPY_LAYER, FFOQP_EQ, LPGD_QP, BPQP, DQP, ALTDIFF])
       
        param_vals = get_default_sudoku_params(n, Qpenalty=Qpenalty, get_equality=True)
        self.warm_start = warm_start

        self.y_dim = (n**2)**3
        self.num_ineq = param_vals["G"].shape[0]
        self.num_eq = param_vals["A"].shape[0]

        ## whether objective is learnable, or inequalit or equality is learnable
        expected_parts = ["ineq", "eq"]
        self.ineq_learnable = "ineq" in learnable_parts
        self.eq_learnable = "eq" in learnable_parts
        for part in learnable_parts:
                if part not in expected_parts:
                    raise Exception(f"unexpected input: {part}")
                
        assert(len(learnable_parts)!=0)
        assert(len(learnable_parts)==1)
        
        if self.layer_type in [QPTH, FFOQP_EQ, FFOQP_EQ_SCHUR, LPGD_QP, BPQP, DQP, ALTDIFF]:
            self.register_buffer("Q", param_vals["Q"])
        else:
            self.register_buffer("Q", param_vals["Q"]**0.5) ## due to the setup cvxpy problem method
        
        ######## set learnable parameters and constant parameters
        if self.eq_learnable:
            if init_learnable_vals is not None:
                self.A = Parameter(init_learnable_vals["A"])
                self.z0_a = Parameter(init_learnable_vals["z0_a"])
            else:
                self.A = Parameter(torch.rand((self.num_eq, self.y_dim)).double())

                self.z0_a = Parameter(torch.zeros((self.y_dim,)).double())
        else:
            raise Exception("must be equality learnable")
            _A = torch.zeros(self.num_eq, self.y_dim)
            _idx = torch.arange(self.num_eq)
            _A[_idx, _idx] = 1.0
            self.A = Parameter(_A.double())
            self.register_buffer("b", param_vals["b"])

            
        if self.ineq_learnable:
            if init_learnable_vals is not None:
                self.G = Parameter(init_learnable_vals["G"])
                self.z0_g = Parameter(init_learnable_vals["z0_g"])
                self.log_s0 = Parameter((init_learnable_vals["log_s0"]))
            else:
                self.G = Parameter(torch.rand((self.num_ineq, self.y_dim)).double())
                self.z0_g = Parameter(torch.zeros((self.y_dim,)).double())
                self.log_s0 = Parameter(torch.rand((self.num_ineq,)).double())
        else:
            for name in ["G", "h"]:
                self.register_buffer(name, param_vals[name])
                

        if self.layer_type not in [QPTH, LPGD_QP, BPQP, DQP, ALTDIFF]:
            problem, objective, ineq_functions, eq_functions, params, variables = setup_cvx_qp_problem(opt_var_dim=self.y_dim, num_ineq=self.num_ineq, num_eq=self.num_eq)
            
            if layer_type==FFOCP_EQ:
                self.optlayer = FFOLayer(problem, parameters=params, variables=variables, alpha=alpha, dual_cutoff=dual_cutoff, slack_tol=slack_tol, eps=1e-6, backward_eps=1e-5)

            elif layer_type==FFOQP_EQ or layer_type==FFOQP_EQ_SCHUR:
                self.optlayer = FFOQPLayer(alpha=alpha, chunk_size=1, solver='qpsolvers')
            elif layer_type==CVXPY_LAYER:
                self.optlayer = CvxpyLayer(problem, parameters=params, variables=variables)
            elif layer_type==LPGD:
                # problem, objective, ineq_functions, eq_functions, params, variables = setup_cvx_qp_problem(opt_var_dim=self.y_dim, num_ineq=self.num_ineq, num_eq=self.num_eq, ignore_ineq=True)
                self.optlayer = CvxpyLayer(problem, parameters=params, variables=variables, lpgd=True)
        else:
            if self.layer_type==QPTH:
                self.optlayer = QPFunction(verbose=-1)
            elif self.layer_type==ALTDIFF:
                self.optlayer = AltDiffLayer()
            elif self.layer_type==BPQP:
                self.optlayer = BPQPLayer()
            elif self.layer_type==DQP:
                dQP_settings = dQP.build_settings(
                    solve_type="dense",
                    qp_solver="gurobi",
                    # lin_solver="scipy LU",
                )
                self.optlayer = dQP.dQP_layer(settings=dQP_settings)
        
    def forward(self, x):
        puzzle_shape = x.shape
        nBatch = x.size(0)
        x = x.view(nBatch, -1) #(B, y_dim)
        p = -x.double() #(batch, y_dim)
        
        if self.ineq_learnable:
            h = get_feasible_h(self.G, self.z0_g, torch.exp(self.log_s0)) #torch.matmul(self.G,self.z0_g) + torch.exp(self.log_s0)
        else:
            h = self.h
        
        if self.eq_learnable:
            # b = get_feasible_b(self.A, torch.exp(torch.clamp(self.z0_a, -1, 1))) #torch.matmul(self.A, self.z0_a)
            # b = get_feasible_b(self.A, torch.clamp(torch.exp(self.z0_a), 0, 1)) #torch.matmul(self.A, self.z0_a)
            b = get_feasible_b(self.A, self.z0_a.exp())
            # b = get_feasible_b(self.A, torch.clamp(self.z0_a, 0, 1)) #torch.matmul(self.A, self.z0_a)
        else:
            b = self.b
        
        # torch.save(self.A.data.clone().detach().cpu(), "sudoku/bad_A.pt")
        # torch.save(self.z0_a.data.clone().detach().cpu(), "sudoku/bad_z0_a.pt")
        # torch.save(b.clone().detach().cpu(), "sudoku/bad_b.pt")
        # torch.save(p.clone().detach().cpu(), "sudoku/bad_p.pt")
        # print(f"rank A: {np.linalg.matrix_rank(self.A.cpu().detach().numpy())}")

        if self.layer_type in [QPTH, FFOQP_EQ, FFOQP_EQ_SCHUR, LPGD_QP]:
            sol = self.optlayer(
                self.Q, p, self.G, h, self.A, b
            )
        elif self.layer_type==BPQP:
            Q_batched = self.Q.unsqueeze(0).expand(nBatch, -1, -1)   # (batch, y_dim, y_dim)
            G_batched = self.G.unsqueeze(0).expand(nBatch, -1, -1)   # (batch, num_ineq, y_dim)
            h_batched = h.unsqueeze(0).expand(nBatch, -1)       # (batch, num_ineq)
            A_batched = self.A.unsqueeze(0).expand(nBatch, -1, -1)   # (batch, num_eq, y_dim)
            b_batched = b.unsqueeze(0).expand(nBatch, -1)       # (batch, num_eq)
            sol = self.optlayer(
                Q_batched, p, G_batched, h_batched, A_batched, b_batched
            )
        else:
            # Expand constant params along batch dimension
            Q_batched = self.Q.unsqueeze(0).expand(nBatch, -1, -1)   # (batch, y_dim, y_dim)
            G_batched = self.G.unsqueeze(0).expand(nBatch, -1, -1)   # (batch, num_ineq, y_dim)
            h_batched = h.unsqueeze(0).expand(nBatch, -1)       # (batch, num_ineq)
            A_batched = self.A.unsqueeze(0).expand(nBatch, -1, -1)   # (batch, num_eq, y_dim)
            b_batched = b.unsqueeze(0).expand(nBatch, -1)       # (batch, num_eq)
            
            params_batched = [Q_batched, p, G_batched, h_batched, A_batched, b_batched]
            
            if self.layer_type==LPGD:
                sol, = self.optlayer(*params_batched, solver_args={"eps": 1e-6}) #, solver_args={"eps": 1e-8, "max_iters": 10000, "acceleration_lookback": 0}
                # sol, = self.optlayer(*params_batched)
            elif self.layer_type==ALTDIFF:
                sol = self.optlayer(*params_batched)
            elif self.layer_type==DQP:
                if Q_batched.device.type == 'cuda':
                    params_batched = [Q_batched.cpu(), p.cpu(), G_batched.cpu(), h_batched.cpu(), A_batched.cpu(), b_batched.cpu()]
                sol, _, _, _, _ = self.optlayer(*params_batched)
                if Q_batched.device.type == 'cuda':
                    sol = sol.to(Q_batched.device)
            elif self.layer_type==CVXPY_LAYER:
                solver_args={"mode": "lsqr", "max_iters": 100, "eps": 1e-6,}
                sol, = self.optlayer(*params_batched, solver_args=solver_args)
                # sol, = self.optlayer(*params_batched)
            elif self.layer_type==FFOCP_EQ:
                ws_keys = None
                if self.warm_start:
                    with torch.no_grad():
                        p_np = p.detach().cpu().numpy()
                        ws_keys = [hash(p_np[i].tobytes()) for i in range(nBatch)]
                sol, = self.optlayer(*params_batched, solver_args={"warm_start": self.warm_start, "ws_keys": ws_keys})
            else:
                sol, = self.optlayer(*params_batched)
            

            #print(f"sol: {sol}")

        return sol.to(x.dtype).reshape(*puzzle_shape)


    
class OptNetSudokuLearnA(nn.Module):
    def __init__(self, n, Qpenalty=0.1, init_A=None):
        ## Qpenalty: ridge term's coefficient
        super().__init__()
        assert(n==2)
        y_dim = (n ** 2) ** 3 ## feature's dimension for a board of n**2 x n**2

        # Fixed tensors → use register_buffer
        self.register_buffer("Q", Qpenalty * torch.eye(y_dim, dtype=torch.double))
        self.register_buffer("G", -torch.eye(y_dim, dtype=torch.double))
        self.register_buffer("h", torch.zeros(y_dim, dtype=torch.double))
        self.register_buffer("b", torch.ones(40, dtype=torch.double))

        # Trainable parameter
        A_shape = (40, y_dim)  # from true solution

        if init_A is None:
            self.A = Parameter(torch.rand(A_shape, dtype=torch.double))
        else:
            self.A = Parameter(init_A)

    def forward(self, puzzles):
        nBatch = puzzles.size(0)
        p = -puzzles.view(nBatch, -1).to(torch.double) #(batch, nx)
        
        print(f"rank A: {np.linalg.matrix_rank(self.A.cpu().detach().numpy(), tol=1e-10)}")

        sol = QPFunction(verbose=-1)(
            self.Q, p, self.G, self.h, self.A, self.b
        )
        return sol.to(puzzles.dtype).view_as(puzzles)
    
class CvxpyLayerSudokuLearnA(nn.Module):
    def __init__(self, n, Qpenalty=0.1, init_A=None):
        super().__init__()
       
        param_vals = get_default_sudoku_params(n, Qpenalty=Qpenalty, get_equality=True)
        
        self.y_dim = (n**2)**3
        self.num_ineq = param_vals["G"].shape[0]
        self.num_eq = param_vals["A"].shape[0]
        
       
        self.register_buffer("Q", param_vals["Q"]**0.5) ## due to the setup cvxpy problem method
        if init_A is not None:
            self.A = Parameter(init_A)
        else:
            self.A = Parameter(torch.rand((self.num_eq, self.y_dim)).double())
            
        for name in ["b"]:
            self.register_buffer(name, param_vals[name])
            
        for name in ["G", "h"]:
            self.register_buffer(name, param_vals[name])
                
        ######## set up optimization layer
       
        problem, objective, ineq_functions, eq_functions, params, variables = setup_cvx_qp_problem(opt_var_dim=self.y_dim, num_ineq=self.num_ineq, num_eq=self.num_eq)
        self.optlayer = CvxpyLayer(problem, parameters=params, variables=variables)
            
        
    def forward(self, x):
        puzzle_shape = x.shape
        nBatch = x.size(0)
        x = x.view(nBatch, -1) #(B, y_dim)
        p = -x.double() #(batch, y_dim)
        
        h = self.h
        b = self.b

        # Expand constant params along batch dimension
        Q_batched = self.Q.unsqueeze(0).expand(nBatch, -1, -1)   # (batch, y_dim, y_dim)
        G_batched = self.G.unsqueeze(0).expand(nBatch, -1, -1)   # (batch, num_ineq, y_dim)
        h_batched = h.unsqueeze(0).expand(nBatch, -1)       # (batch, num_ineq)
        A_batched = self.A.unsqueeze(0).expand(nBatch, -1, -1)   # (batch, num_eq, y_dim)
        b_batched = b.unsqueeze(0).expand(nBatch, -1)       # (batch, num_eq)
        
        params_batched = [Q_batched, p, G_batched, h_batched, A_batched, b_batched]
        
        sol, = self.optlayer(*params_batched)
        
        return sol.to(x.dtype).reshape(*puzzle_shape)
    
    
class BLOSudokuLearnA(nn.Module):
    def __init__(self, n, Qpenalty=0.1, alpha=100):
        '''
        delta = 1/alpha, which is the perturbation constant for finite difference
        '''
        super().__init__()
        assert(n==2)
        
        y_dim = (n**2)**3
        num_eq = 40
        num_ineq = y_dim
        
        self.alpha = alpha
        
        problem, objective, ineq_functions, eq_functions, params, variables = setup_cvx_qp_problem(opt_var_dim=y_dim, num_ineq=num_ineq, num_eq=num_eq)
        self.blolayer1 = FFOLayer(problem, parameters=params, variables=variables, alpha=self.alpha)
        
        self.register_buffer("Q", Qpenalty * torch.eye(y_dim, dtype=torch.double))
        self.register_buffer("G", -torch.eye(num_ineq, dtype=torch.double))
        self.register_buffer("h", torch.zeros(num_ineq, dtype=torch.double))
        self.register_buffer("b", torch.ones(num_eq, dtype=torch.double))
        
        A = torch.rand((num_eq, y_dim), dtype=torch.double)
        #####TODO: problem infeasible or unbounded if I use random A, if I use zero A, all good. Maybe because of Gurobi solver.
        # for i in range(len(A)):
        #         A[i,i] = 1
        self.A = Parameter(A)
        
    def forward(self, puzzles):
        nBatch = puzzles.size(0)
        p = -puzzles.view(nBatch, -1).to(torch.double) #(batch, y_dim)
        
        # Expand constant params along batch dimension
        Q_batched = self.Q.unsqueeze(0).expand(nBatch, -1, -1)   # (batch, y_dim, y_dim)
        G_batched = self.G.unsqueeze(0).expand(nBatch, -1, -1)   # (batch, num_ineq, y_dim)
        h_batched = self.h.unsqueeze(0).expand(nBatch, -1)       # (batch, num_ineq)
        A_batched = self.A.unsqueeze(0).expand(nBatch, -1, -1)   # (batch, num_eq, y_dim)
        b_batched = self.b.unsqueeze(0).expand(nBatch, -1)       # (batch, num_eq)
        
        print("####### rank: ", np.linalg.matrix_rank(self.A.cpu().detach().numpy()))

        params_batched = [Q_batched, p, G_batched, h_batched, A_batched, b_batched]
        
        blo_solution, = self.blolayer1(*params_batched)
        # print('blo_layer output', blo_solution)
        
        return blo_solution.to(puzzles.dtype).view_as(puzzles)

class MLP(nn.Module):
    '''
    2 layers of {FC-ReLU-BN} and a final layer of FC
    '''
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.activation(self.batch_norm1(self.fc1(x)))
        x = self.activation(self.batch_norm2(self.fc2(x)))
        x = self.fc3(x)
        
        return x
    
# class OptLayerSudoku(nn.Module):
#     def __init__(self, y_dim, num_ineq, num_eq, 
#                  init_params, learnable_parts, use_MLP, input_dim, eps=1e-4, layer_type="qpth"):
#         '''
#         The architecture is 1){MLP - optLayer} or 2){parameter - optLayer}.
        
#         Args:
#             - y_dim: dim of optimization variable of opt layer
#             - init_QP_params: initialize constant params of QP
#             - learnable_parts: a list that can contain ["obj", "ineq", "eq"], specifying which QP parameters are learnable
            
#             - use_MLP: use architecture 1) or 2)
#             - input_dim: input dim of MLP
#             - output_dim: the output dim of MLP
#             - hidden_dim: hidden_dim of the MLP
            
#             - eps: when we use MLP, eps keeps the Q positive definite
#             ...
#         '''
#         super().__init__()
        
#         self.num_ineq = num_ineq
#         self.num_eq = num_eq
#         self.input_dim = input_dim
        
#         self.y_dim = y_dim
#         self.use_MLP = use_MLP
        
#         self.Q_shape = (y_dim, y_dim)
#         self.p_shape = (y_dim,)
#         self.G_shape = (num_ineq, y_dim)
#         self.h_shape = (num_ineq,)
#         self.A_shape = (num_eq, y_dim)
#         self.b_shape = (num_eq,)
#         self.opt_var_shape = (y_dim,)
        
#         learnable_params = []
#         constant_params = []
#         expected_parts = ["obj", "ineq", "eq"]
        
#         self.obj_learnable = "obj" in learnable_parts
#         self.ineq_learnable = "ineq" in learnable_parts
#         self.eq_learnable = "eq" in learnable_parts
        
#         for part in learnable_parts:
#                 if part not in expected_parts:
#                     raise Exception(f"unexpected input: {part}")
                
#         assert(len(learnable_parts)!=0)
        
#         self.MLP_output_dim = 0
        
#         if not self.use_MLP:
#             #### all learnable parameters are not dependent on input
#             if self.obj_learnable: # Q learnable
#                 self.L = Parameter(torch.tril(torch.rand(y_dim, y_dim)).double())
#             else:
#                 self.register_buffer("Q", constant_params["Q"].double())
#                 assert(self.Q.shape==self.Q_shape)
                
#             if self.ineq_learnable: # G and h=G(z0)+s0 learnable
#                 self.G = Parameter(torch.rand(self.G_shape).double())
#                 self.z0 = Parameter(torch.rand(self.y_dim).double())
#                 self.log_s0 = Parameter(torch.rand(self.h_shape).double())
#             else:
#                 self.register_buffer("G", constant_params["G"].double())
#                 self.register_buffer("h", constant_params["h"].double())
#                 assert(self.G.shape==self.G_shape)
#                 assert(self.h.shape==self.h_shape)
            
#             if self.eq_learnable:  # A and b=A(z0) learnable
#                 self.A = Parameter(torch.rand(self.A_shape).double())
#                 self.z0 = Parameter(torch.rand(self.y_dim).double())
#             else:
#                 self.register_buffer("A", constant_params["A"].double())
#                 self.register_buffer("b", constant_params["b"].double())
#                 assert(self.A.shape==self.A_shape)
#                 assert(self.b.shape==self.b_shape)
                
#         else:
#             #### learnable parameters can depend on input
#             if not self.obj_learnable:
#                 self.register_buffer("Q", constant_params["Q"].double())
#             else:
#                 self.MLP_output_dim += math.prod(self.Q_shape)
#             if not self.ineq_learnable:
#                 self.register_buffer("G", constant_params["G"].double())
#                 self.register_buffer("h", constant_params["h"].double())
#                 assert(self.G.shape==self.G_shape)
#                 assert(self.h.shape==self.h_shape)
#             else:
#                 self.MLP_output_dim += (math.prod(self.G_shape) + math.prod(self.opt_var_shape) + math.prod(self.h_shape))
#             if not self.eq_learnable:
#                 self.register_buffer("A", constant_params["A"].double())
#                 self.register_buffer("b", constant_params["b"].double())
#                 assert(self.A.shape==self.A_shape)
#                 assert(self.b.shape==self.b_shape)
#             else:
#                 self.MLP_output_dim += (math.prod(self.A_shape) + math.prod(self.opt_var_shape))

#         if use_MLP:
#             self.mlp = MLP(input_dim, output_dim=self.MLP_output_dim, hidden_dim=128)
#         else:
#             assert(init_learnable_QP_params is not None)

        
#         nCls = output_dim

#         self.register_buffer("L_mask", torch.tril(torch.ones(self.y_dim, self.y_dim))) # have ones in lower triangular part, others are zeros
        
#         self.L = Parameter(torch.tril(torch.rand(nCls, nCls).cuda()))
#         self.G = Parameter(torch.Tensor(nineq,nCls).uniform_(-1,1).cuda())
#         self.z0 = Parameter(torch.zeros(nCls).cuda())
#         self.s0 = Parameter(torch.ones(nineq).cuda())

                
#     def forward(self, x):
#         nBatch = x.size(0)
#         x = x.view(nBatch, -1)
        
#         if self.use_MLP:
#             x = self.mlp(x) #(B, output_dim)
#             L = x[:math.prod(self.Q_shape)]
#             p = x[math.prod(self.Q_shape):math.prod(self.Q_shape)+self.y_dim]

#         L = self.M*self.L
#         Q = L.mm(L.t()) + self.eps*Variable(torch.eye(self.nCls)).cuda()
#         h = self.G.mv(self.z0)+self.s0
#         e = Variable(torch.Tensor())
#         x = QPFunction(verbose=False)(Q, x, G, h, e, e)

#         return F.log_softmax(x)