import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import math
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import sys
sys.path.append("..")  
from utils import *
from sklearn.cluster import KMeans
try:
    from grakel import GraphKernel, Graph
except ImportError:
    pass



def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)

class GINConv(torch.nn.Module):
	def __init__(self, input_dim, output_dim):
		super().__init__()
		self.linear = torch.nn.Linear(input_dim, output_dim)

	def forward(self, A, X):
		X = self.linear(X + A @ X)
		X = torch.nn.functional.relu(X)
		return X


class GIN(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, n_layers,
				 use_input_augmentation):
		super().__init__()
		self.in_proj = torch.nn.Linear(input_dim, hidden_dim)
		self.convs = torch.nn.ModuleList()
		self.use_input_agumentation = use_input_augmentation
		if(use_input_augmentation):
			self.hidden_input_dim = input_dim+hidden_dim
		else:
			self.hidden_input_dim = hidden_dim
		for _ in range(n_layers):
			self.convs.append(GINConv(self.hidden_input_dim, hidden_dim))
		self.out_proj = torch.nn.Linear(hidden_dim * (1 + n_layers), output_dim)

	def forward(self, A, X):
		initial_X = torch.empty_like(X).copy_(X)
		X = self.in_proj(X)
		hidden_states = [X]
		for layer in self.convs:
			if(self.use_input_agumentation):
				X = layer(A, torch.cat([initial_X,X],dim=1))
			else:
				X = layer(A, X)
			hidden_states.append(X)
		X = torch.cat(hidden_states, dim=1)
		X = self.out_proj(X)
		return X




class GConv_tide(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, self_loop, tide=False, k=300, alpha=1.0, beta=1.0):
        super(GConv_tide, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        self.tide = tide
        self.self_loop = self_loop
        self.proj = nn.Linear(input_dim, hidden_dim)
        if self.tide:
            self.t = nn.Parameter(torch.Tensor(1))
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
            self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
        
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim, add_self_loops=self.self_loop))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=self.self_loop))
        

    def diffusion(self, x, phi, lamb):        
        clamped_t = torch.clamp(self.t, min=0, max=20)  # prevent overflow
        t = torch.nn.functional.softplus(clamped_t)    

        phi_t_x = torch.matmul(phi.T, x)  # [k, d]

        # element-wise exp(-t_i * lambda_i)
        heat_coeff = torch.exp(-t * lamb)  # [k]

        modifier = heat_coeff.unsqueeze(1) - self.alpha  # [k, 1]
        spectral_term = modifier * phi_t_x  # [k, d]

        out = torch.matmul(phi, spectral_term) + self.beta * x
    
        return out 
    
              
    def forward(self, x, edge_index, edge_index2, phi, lamb, edge_weight=None, edge_weight2=None):
        z = x
        x_res = self.proj(x)
        for conv in self.layers:
            z = conv(z, edge_index, edge_index2, edge_weight, edge_weight2)
            if self.tide:
                z = self.diffusion(z, phi, lamb)
            z = z + x_res
            z = self.activation(z)
        return z



class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, self_loop):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        self.self_loop = self_loop
        self.proj = nn.Linear(input_dim, hidden_dim)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv2(input_dim, hidden_dim, add_self_loops=self.self_loop))
            else:
                self.layers.append(GCNConv2(hidden_dim, hidden_dim, add_self_loops=self.self_loop))
        

                
    def forward(self, x, edge_index, edge_index2, edge_weight=None, edge_weight2=None):
        z = x
        x_res = self.proj(x)
        for conv in self.layers:
            z = conv(z, edge_index, edge_index2, edge_weight, edge_weight2)
            z = z + x_res
            z = self.activation(z)
        return z

   
  
  
 

class Encoder_GAE_FM(nn.Module):
    def __init__(self, encoder1, encoder2, hidden_dim, k):
        super(Encoder_GAE_FM, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.C12 = nn.Linear(k, k)
        self.C21 = nn.Linear(k, k)       
        
    @staticmethod
    def corruption(x, edge_index, edge_weight):
        return x[torch.randperm(x.size(0))], edge_index, edge_weight

    def forward(self, x1, edge_index1, x2, edge_index2, edge_weight1=None, edge_weight2=None):
        Z1_h, Z1_l = self.encoder1(x1, edge_index1)
        Z2_h, Z2_l = self.encoder2(x2, edge_index2)
                   
        return Z1_h, Z1_l, Z2_h, Z2_l

    
    def compute_FM(self, z1, z2, phi1, phi2, lam1, lam2, C, alpha = 1e-3, beta=1e-2):
        Lambda_1 = torch.diag(lam1)
        Lambda_2 = torch.diag(lam2)
        
        AA = torch.mm(phi1.T, z1)
        BB = torch.mm(phi2.T, z2)
               
        W = C.weight  # assuming C is a linear layer
        A1_mapped = torch.mm(W,AA)
        map_loss = torch.norm(A1_mapped - BB, p='fro')**2
        
        # Orthogonality constraint: C^T C =~ I
        orth_loss =0
        orth_loss = torch.norm(torch.mm(W.T, W) - torch.eye(W.shape[1], device=W.device), p='fro')**2
        
        ## ----Laplacian commutativity regularizer----
        reg_loss = torch.norm(torch.mm(Lambda_2, W) - torch.mm(W, Lambda_1), p='fro')**2
      
        loss_fm = alpha * map_loss  + beta * orth_loss +  reg_loss 
        
        return  loss_fm
    
       
        
    def DF_map(self, z1, z2, phi1, phi2, lam1, lam2, args):

        phi1 = orthonormalize_basis(phi1)
        phi2 = orthonormalize_basis(phi2)
        lam1 = lam1 / (lam1.max() + 1e-8)  
        lam2 = lam2 / (lam2.max() + 1e-8)
        
        loss_fm_1 = self.compute_FM(z1, z2, phi1, phi2, lam1, lam2, self.C12, args.alpha, args.beta)
        
        loss_fm_2 = self.compute_FM(z2, z1, phi2, phi1, lam2, lam1, self.C21, args.alpha, args.beta)
            
        loss_fm = loss_fm_1+ loss_fm_2

        W12 = self.C12.weight
        W21 = self.C21.weight

        I = torch.eye(W12.shape[0], device=W12.device)

        # - Cycle consistency
        bij_1 = torch.norm(torch.mm(W12, W21) - I, p='fro')**2
        bij_2 = torch.norm(torch.mm(W21, W12) - I, p='fro')**2

        # - Orthonormality
        ortho_1 = torch.norm(torch.mm(W12.T, W12) - I, p='fro')**2
        ortho_2 = torch.norm(torch.mm(W21.T, W21) - I, p='fro')**2
               
        loss_map = args.l_bij*(bij_1 + bij_2) + args.l_ort*(ortho_1 + ortho_2) 
   
        return loss_fm, loss_map


    
    
    
class GConv_GADL(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, self_loop, residual=False):
        super(GConv_GADL, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        self.self_loop = self_loop
        self.residual = residual
        self.proj = nn.Linear(input_dim, hidden_dim)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim, normalize=False, add_self_loops=False))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim, normalize=False, add_self_loops=False))
        

                
    def forward(self, x, edge_index, edge_weight=None):
        z = x
        x_res = self.proj(x)
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = z + x_res if self.residual else z
        return z
    

class GAE_GADL(nn.Module):
    def __init__(self, num_hidden_layers, input_dim, hidden_dim, output_dim, residual=False):
        super(GAE_GADL, self).__init__()
        self.input_dim = input_dim
        self.base_gcn_l = GConv_GADL(input_dim, hidden_dim, num_hidden_layers, self_loop=True, residual=residual)
        self.base_gcn_h = GConv_GADL(input_dim, hidden_dim, num_hidden_layers, self_loop=True, residual=residual)
        self.mlp = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        
    def _scipy_to_torch_sparse(self, scipy_sparse):
        """Convert scipy sparse matrix to torch sparse tensor"""
        scipy_coo = scipy_sparse.tocoo()
        indices = torch.from_numpy(np.vstack((scipy_coo.row, scipy_coo.col))).long()
        values = torch.from_numpy(scipy_coo.data).float()
        shape = scipy_coo.shape
        return torch.sparse_coo_tensor(indices, values, torch.Size(shape))     
    
    def _sparse_to_edge_index(self, sparse_tensor):
        """Convert sparse tensor to edge_index format for GCNConv"""
        sparse_tensor = sparse_tensor.coalesce()
        edge_index = sparse_tensor.indices()
        edge_weight = sparse_tensor.values()
        return edge_index, edge_weight

    def forward(self, initial_X, adj):
        if not isinstance(adj, torch.Tensor):
            adj = self._scipy_to_torch_sparse(adj).to(initial_X.device)
       
        num_nodes = adj.shape[0]
        identity = torch.sparse_coo_tensor(indices=torch.arange(num_nodes, device=adj.device).repeat(2, 1), values=torch.ones(num_nodes, device=adj.device), size=(num_nodes, num_nodes))

        A_tilde = adj + identity
        # A_tilde = adj
        
        deg_values = torch.sparse.sum(A_tilde, dim=1).to_dense()
        D_tilde = torch.sparse_coo_tensor(indices=torch.arange(num_nodes, device=adj.device).repeat(2, 1), values=deg_values, size=(num_nodes, num_nodes))

        adj_h = (D_tilde - A_tilde) / 2
        adj_l = (D_tilde + A_tilde) / 2


        deg_inv_sqrt = torch.pow(deg_values, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.sparse_coo_tensor(indices=torch.arange(num_nodes, device=adj.device).repeat(2, 1),
            values=deg_inv_sqrt,size=(num_nodes, num_nodes))
        
        # Apply symmetric normalization: D^{-1/2} A_* D^{-1/2}
        adj_h = torch.sparse.mm(torch.sparse.mm(D_inv_sqrt, adj_h), D_inv_sqrt)
        adj_l = torch.sparse.mm(torch.sparse.mm(D_inv_sqrt, adj_l), D_inv_sqrt)
        adj = torch.sparse.mm(torch.sparse.mm(D_inv_sqrt, A_tilde), D_inv_sqrt)
                                            
        Z_h = self.base_gcn_h(initial_X, adj_h)
        Z_l = self.base_gcn_l(initial_X, adj_l)
                
        return Z_h, Z_l
    
    
    
 
class GConv_GADL_vl(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super(GConv_GADL_vl, self).__init__()
        self.num_layers = num_layers
        self.initial_proj = nn.Linear(input_dim, hidden_dim)

        self.scale_convs = nn.ModuleList([GCNConv4(hidden_dim, hidden_dim)for _ in range(num_layers)])
        self.activation = nn.PReLU(hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        x_in = x
        x = self.initial_proj(x)  # [N, hidden_dim]

        for conv in self.scale_convs:
            x = conv(x, edge_index, edge_weight)
            x = self.activation(x)
            
        return x



class GAE_GConv_vl(nn.Module):
    def __init__(self, num_hidden_layers, input_dim, hidden_dim, output_dim):
        super(GAE_GConv_vl, self).__init__()
        self.base_gcn_l = GConv_GADL_vl(input_dim, hidden_dim, num_hidden_layers)
        self.base_gcn_h = GConv_GADL_vl(input_dim, hidden_dim, num_hidden_layers)

        self.mlp = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        
    
    def forward(self, initial_X, adj):
        # convert to torch
        if not isinstance(adj, torch.Tensor):
            adj = self._scipy_to_torch_sparse(adj).to(initial_X.device)
        
        num_nodes = adj.shape[0]
        
        identity = torch.sparse_coo_tensor(indices=torch.arange(num_nodes, device=adj.device).repeat(2, 1), values=torch.ones(num_nodes, device=adj.device), size=(num_nodes, num_nodes))
        A_tilde = adj + identity
        # A_tilde = adj.clone()
        
        deg_values = torch.sparse.sum(A_tilde, dim=1).to_dense()
        D_tilde = torch.sparse_coo_tensor(indices=torch.arange(num_nodes, device=adj.device).repeat(2, 1), values=deg_values, size=(num_nodes, num_nodes))

        adj_h = (D_tilde - A_tilde) / 2
        adj_l = (D_tilde + A_tilde) / 2
   
                                    
        adj_h_norm = self.normalize_adj_I(adj_h, deg_values)
        adj_l_norm = self.normalize_adj_I(adj_l, deg_values)
        adj_norm = self.normalize_adj_I(adj, deg_values)
        
        Z_h = self.base_gcn_h(initial_X, adj_h_norm)
        Z_l = self.base_gcn_l(initial_X, adj_l_norm)
        
        return Z_h, Z_l    
    
    def _scipy_to_torch_sparse(self, scipy_sparse):
        """Convert scipy sparse matrix to torch sparse tensor"""
        scipy_coo = scipy_sparse.tocoo()
        indices = torch.from_numpy(np.vstack((scipy_coo.row, scipy_coo.col))).long()
        values = torch.from_numpy(scipy_coo.data).float()
        shape = scipy_coo.shape
        return torch.sparse_coo_tensor(indices, values, torch.Size(shape))       
    
       
    def normalize_adj_I(self, adj, deg):
        device = adj.device
        adj = adj.coalesce()

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.

        # D^{-1/2} * A * D^{-1/2}
        row, col = adj.indices()
        norm_values = adj.values() * deg_inv_sqrt[row] * deg_inv_sqrt[col]

        adj_norm = torch.sparse_coo_tensor(adj.indices(), norm_values, adj.size(), device=device).coalesce().transpose(0, 1)
        
        return adj_norm
