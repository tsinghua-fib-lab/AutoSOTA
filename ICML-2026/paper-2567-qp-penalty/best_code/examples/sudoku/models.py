import numpy as np

import scipy.sparse as spa
import scipy.linalg as sla

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import cvxpy as cp
from qpth.qp import QPFunction

import os, sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from src.dXPP import dXPPLayer


# modified to be on CPU
class OptNetEq(nn.Module):
    def __init__(self, n, Qpenalty, qp_solver, trueInit=False):
        super().__init__()

        self.qp_solver = qp_solver

        nx = (n**2)**3
        self.Q = Variable(Qpenalty*torch.eye(nx).double())
        # self.Q = Variable(Qpenalty*torch.eye(nx).double().cuda())
        self.Q_idx = spa.csc_matrix(self.Q.detach().cpu().numpy()).nonzero()

        # self.G = Variable(-torch.eye(nx).double().cuda())
        # self.h = Variable(torch.zeros(nx).double().cuda())
        self.G = Variable(-torch.eye(nx).double())
        self.h = Variable(torch.zeros(nx).double())
        t = get_sudoku_matrix(n)

        if trueInit:
            # self.A = Parameter(torch.DoubleTensor(t).cuda())
            self.A = Parameter(torch.DoubleTensor(t))
        else:
            # self.A = Parameter(torch.rand(t.shape).double().cuda())
            self.A = Parameter(torch.rand(t.shape).double())

        # self.log_z0 = Parameter(torch.zeros(nx).double().cuda())
        self.log_z0 = Parameter(torch.zeros(nx).double())
        # self.b = Variable(torch.ones(self.A.size(0)).double().cuda())

        if self.qp_solver == 'osqpth':
            t = torch.cat((self.A, self.G), dim=0)
            self.AG_idx = spa.csc_matrix(t.detach().cpu().numpy()).nonzero()

    # @profile
    def forward(self, puzzles):
        nBatch = puzzles.size(0)

        p = -puzzles.view(nBatch, -1)
        b = self.A.mv(self.log_z0.exp())

        if self.qp_solver == 'qpth_1':
            y = QPFunction(verbose=-1,eps=1e-6)(
                self.Q, p.double(), self.G, self.h, self.A, b
            ).float().view_as(puzzles)
        # elif self.qp_solver == 'osqpth':
        #     _l = torch.cat(
        #         (b, torch.full(self.h.shape, float('-inf'),
        #                     device=self.h.device, dtype=self.h.dtype)),
        #         dim=0)
        #     _u = torch.cat((b, self.h), dim=0)
        #     Q_data = self.Q[self.Q_idx[0], self.Q_idx[1]]
        #
        #     AG = torch.cat((self.A, self.G), dim=0)
        #     AG_data = AG[self.AG_idx[0], self.AG_idx[1]]
        #     y = OSQP(self.Q_idx, self.Q.shape, self.AG_idx, AG.shape,
        #              diff_mode=DiffModes.FULL)(
        #         Q_data, p.double(), AG_data, _l, _u).float().view_as(puzzles)
        # else:
        #     assert False

        return y




















class dXPPEq(nn.Module):
    def __init__(self, n, Qpenalty, trueInit=False):

        super().__init__()

        nx = (n**2)**3
        self.Q = Variable(Qpenalty*torch.eye(nx).double())
        self.G = Variable(-torch.eye(nx).double())
        self.h = Variable(torch.zeros(nx).double())
        t = get_sudoku_matrix(n)

        if trueInit:
            self.A = Parameter(torch.DoubleTensor(t))
        else:
            self.A = Parameter(torch.rand(t.shape).double())

        self.log_z0 = Parameter(torch.zeros(nx).double())

        self.solve = dXPPLayer(
            beta=1e-6, penalty_coeff=10.0,
            eps_abs=1e-6, eps_rel=0.0,
            solve_type="dense", qp_solver="gurobi",
        )

    def forward(self, puzzles):
        nBatch = puzzles.size(0)

        p = -puzzles.view(nBatch, -1)
        b = self.A.mv(self.log_z0.exp())

        y, _, _ = self.solve(Q=self.Q, q=torch.squeeze(p.double()), G=self.G, h=self.h, A=self.A, b=b)
        y = torch.squeeze(y).float().view_as(puzzles)

        return y


class FC(nn.Module):
    def __init__(self, nFeatures, nHidden, bn=False):
        super().__init__()
        self.bn = bn

        fcs = []
        prevSz = nFeatures
        for sz in nHidden:
            fc = nn.Linear(prevSz, sz)
            prevSz = sz
            fcs.append(fc)
        for sz in list(reversed(nHidden))+[nFeatures]:
            fc = nn.Linear(prevSz, sz)
            prevSz = sz
            fcs.append(fc)
        self.fcs = nn.ModuleList(fcs)

    def __call__(self, x):
        nBatch = x.size(0)
        Nsq = x.size(1)
        in_x = x
        x = x.view(nBatch, -1)

        for fc in self.fcs:
            x = F.relu(fc(x))

        x = x.view_as(in_x)
        ex = x.exp()
        exs = ex.sum(3).expand(nBatch, Nsq, Nsq, Nsq)
        x = ex/exs

        return x

class Conv(nn.Module):
    def __init__(self, boardSz):
        super().__init__()

        self.boardSz = boardSz

        convs = []
        Nsq = boardSz**2
        prevSz = Nsq
        szs = [512]*10 + [Nsq]
        for sz in szs:
            conv = nn.Conv2d(prevSz, sz, kernel_size=3, padding=1)
            convs.append(conv)
            prevSz = sz

        self.convs = nn.ModuleList(convs)

    def __call__(self, x):
        nBatch = x.size(0)
        Nsq = x.size(1)

        for i in range(len(self.convs)-1):
            x = F.relu(self.convs[i](x))
        x = self.convs[-1](x)

        ex = x.exp()
        exs = ex.sum(3).expand(nBatch, Nsq, Nsq, Nsq)
        x = ex/exs

        return x

def get_sudoku_matrix(n):
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"sudoku_matrix_n{n}.npy")
    if os.path.isfile(cache_path):
        return np.load(cache_path)

    X = np.array([[cp.Variable(n**2) for i in range(n**2)] for j in range(n**2)])
    cons = ([x >= 0 for row in X for x in row] +
            [cp.sum(x) == 1 for row in X for x in row] +
            [sum(row) == np.ones(n**2) for row in X] +
            [sum([row[i] for row in X]) == np.ones(n**2) for i in range(n**2)] +
            [sum([sum(row[i:i+n]) for row in X[j:j+n]]) == np.ones(n**2) for i in range(0,n**2,n) for j in range(0, n**2, n)])
    f = sum([cp.sum(x) for row in X for x in row])
    prob = cp.Problem(cp.Minimize(f), cons)

    A = np.asarray(prob.get_problem_data(cp.ECOS)[0]["A"].todense())
    # Select a maximal set of linearly independent rows in one shot (faster than
    # repeated matrix_rank calls).
    _, R, piv = sla.qr(A.T, pivoting=True, mode="economic")
    rank = int(np.sum(np.abs(np.diag(R)) > 1e-12))
    rows = piv[:rank]
    A0 = A[rows]
    np.save(cache_path, A0)
    return A0