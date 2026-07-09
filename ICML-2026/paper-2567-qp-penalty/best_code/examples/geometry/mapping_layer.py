import numpy as np

import os, sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.dXPP import dXPPLayer
from src.sparse_helper import csc_scipy_to_torch
import torch_geometric

import torch
from torch import nn

################# Tolerances #########################
tolerance = 1e-5
eps_active = 1e-4
import time
#############################################


class QPLaplacianLayer(nn.Module):
    ''' a differentiable QP layer for harmonic mapping with Laplacian deformation parameter
    '''

    def __init__(self,L_init,edges,nv,nc,nb,A,b,G_precompute):
        super().__init__()

        self.edges = torch.tensor(edges.T,dtype=torch.int64)
        self.symedges = torch.cat((self.edges,torch.flip(self.edges,dims=(0,))),1)
        self.nv = nv
        self.nc = nc
        self.nb = nb
        self.dim = 2

        Linds = np.hstack((np.expand_dims(L_init.row,-1),np.expand_dims(L_init.col,-1)))
        idx = np.where((Linds == edges[:, None]).all(-1))[1]
        Lvals = L_init.data[idx]
        self.Lvals = nn.Parameter(torch.tensor(Lvals,dtype=torch.float64)) # non-zero values of Laplacian are parameters of layer

        self.A = csc_scipy_to_torch(A).requires_grad_(False)
        self.b = torch.tensor(b,requires_grad=False,dtype=torch.float64)
        self.G_precompute = torch.tensor(G_precompute,requires_grad=False,dtype=torch.float64).to_sparse_coo().coalesce() # note .to_sparse_coo() is new ; doesn't need grad but pretend to have one for spspmm
        self.q = torch.zeros(self.dim * self.nv,1,requires_grad=False,dtype=torch.float64)
        self.h = torch.zeros(self.dim * self.nc,1,requires_grad=False,dtype=torch.float64)

        # ===== dXPP layer (main solver — participates in gradient graph) =====
        self.dXPP_layer = dXPPLayer(
            beta=1e-4, penalty_coeff=10.0,
            eps_abs=tolerance, eps_rel=0.0,
            solve_type="sparse", qp_solver="piqp", lin_solver="scipy SPLU"
        )

        self.ReLU = torch.nn.ReLU()

    def forward(self):
        Lvals = -self.ReLU(-self.Lvals) - 1e-2 # thresohld to be negative
        Lvals = torch.cat((Lvals,Lvals)) # duplicate for symmetry
        Linds, Lvals = torch_geometric.utils.get_laplacian(self.symedges, Lvals)
        Lvals = -Lvals # PD convention
        Lvals[Linds[0, :] == Linds[1, :]] += 1e-4 # perturb by eps*I to make PD for QP solver
        Linds = torch.cat((Linds,Linds+self.nv),1)
        Lvals = torch.cat((Lvals,Lvals)) # duplicate for block diagonal
        L = torch.sparse_coo_tensor(Linds, Lvals.double(), (self.nv*self.dim,self.nv*self.dim))
        G = torch.matmul(self.G_precompute,L)

        # ==================== dXPP forward ====================
        t = time.time()
        x_star, mu_star, nu_star = self.dXPP_layer(
            Q=L.to_sparse_csc(), q=self.q.squeeze(),
            G=G.to_sparse_csc(), h=self.h.squeeze(),
            A=self.A, b=self.b.squeeze()
        )
        dXPP_forward_time = time.time() - t
        print("dXPP forward: " + str(dXPP_forward_time))

        return (L.to_sparse_csc(), x_star, nu_star,
                dXPP_forward_time)
