## Copyright (c) 2017 Robert Bosch GmbH
## All rights reserved.
##
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from torch.nn.utils import weight_norm
import pickle
import sys
from termcolor import colored

from modules.hierarchical_embedding import HierarchicalEmbedding
from modules.embeddings import LearnableEmbedding, SineEmbedding


def sqdist(A,B):
    return (A**2).sum(dim=2)[:,:,None] + (B**2).sum(dim=2)[:,None,:] - 2*torch.bmm(A,B.transpose(1,2))


class ResidualBlock(nn.Module):
    def __init__(self, d_in, d_out, groups=1, dropout=0.0):
        super().__init__()
        assert d_in % groups == 0, "Input dimension must be a multiple of groups"
        assert d_out % groups == 0, "Output dimension must be a multiple of groups"
        self.d_in = d_in
        self.d_out = d_out
        self.proj = nn.Sequential(nn.Conv1d(d_in, d_out, kernel_size=1, groups=groups),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(dropout),
                                  nn.Conv1d(d_out, d_out, kernel_size=1, groups=groups),
                                  nn.Dropout(dropout))
        if d_in != d_out:
            self.downsample = nn.Conv1d(d_in, d_out, kernel_size=1, groups=groups)
    
    def forward(self, x):
        assert x.size(1) == self.d_in, "x dimension does not agree with d_in"
        return x + self.proj(x) if self.d_in == self.d_out else self.downsample(x) + self.proj(x)

    
class GraphLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_head, dropout=0.0, attn_dropout=0.0, wnorm=False, use_quad=False, lev=0):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_head = d_head
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.lev = lev
        self.use_quad = use_quad
        
        # To produce the query-key-value for the self-attention computation
        self.qkv_net = nn.Linear(d_model, 3*d_model)
        self.o_net = nn.Linear(n_head*d_head, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.proj1 = nn.Linear(d_model, d_inner)
        self.proj2 = nn.Linear(d_inner, d_model)
        self.gamma = nn.Parameter(torch.ones(4,4))   # For different sub-matrices of D
        self.sqrtd = np.sqrt(d_head)
        
        if wnorm:
            self.wnorm()
    
    def wnorm(self):
        self.qkv_net = weight_norm(self.qkv_net, name="weight")
        self.o_net = weight_norm(self.o_net, name="weight")
        self.proj1 = weight_norm(self.proj1, name="weight")
        self.proj2 = weight_norm(self.proj2, name="weight")
        
    def forward(self, Z, D, new_mask, mask, RA, RB, RT, RQ, store=False):
        # RA = slice(0,N), RB = slice(N,N+M), RT = slice(N+M, N+M+P)
        bsz, n_elem, nhid = Z.size()
        n_head, d_head, d_model = self.n_head, self.d_head, self.d_model
        assert nhid == d_model, "Hidden dimension of Z does not agree with d_model"
        
        # Create gamma mask
        gamma_mask = torch.ones_like(D)
        all_slices = [RA, RB, RT, RQ] if self.use_quad else [RA, RB, RT]
        for i, slice_i in enumerate(all_slices):
            for j, slice_j in enumerate(all_slices):
                gamma_mask[:, slice_i, slice_j] = self.gamma[i, j]
    
        # Self-attention
        inp = Z
        Z = self.norm1(Z)
        Z2, Z3, Z4 = self.qkv_net(Z).view(bsz, n_elem, n_head, 3*d_head).chunk(3, dim=3)     # "V, Q, K"
        W = -(gamma_mask*D)[:,None] + torch.einsum('bnij, bmij->binm', Z3, Z4).type(D.dtype) / self.sqrtd + new_mask[:,None]
        W = self.attn_dropout(F.softmax(W, dim=3).type(mask.dtype) * mask[:,None])           # softmax(-gamma*D + Q^TK)
        if store:
            pickle.dump(W.cpu().detach().numpy(), open(f'analysis/layer_{self.lev}_W.pkl', 'wb'))
        attn_out = torch.einsum('binm,bmij->bnij', W, Z2.type(W.dtype)).contiguous().view(bsz, n_elem, d_model)
        attn_out = self.dropout(self.o_net(F.leaky_relu(attn_out)))
        Z = attn_out + inp
        
        # Position-wise feed-forward
        inp = Z
        Z = self.norm2(Z)
        return self.proj2(self.dropout(F.relu(self.proj1(Z)))) + inp

    
class GraphTransformer(nn.Module):
    def __init__(self, dim, n_layers, final_dim, d_inner, 
                 fdim=30,
                 dropout=0.0, 
                 dropatt=0.0,
                 final_dropout=0.0, 
                 n_head=10,
                 num_atom_types=[5,13,27], 
                 num_bond_types=[28,53,69],
                 num_triplet_types=[29,118],
                 num_quad_types=[62],
                 min_bond_dist=0.9586,
                 max_bond_dist=3.9244,
                 dist_embedding="sine",
                 atom_angle_embedding="learnable",
                 trip_angle_embedding="learnable",
                 quad_angle_embedding="learnable",
                 wnorm=False,
                 use_quad=False
                 ):
        super().__init__()
        self.fdim = fdim
        num_atom_types = np.array(num_atom_types)
        num_bond_types = np.array(num_bond_types)
        num_triplet_types = np.array(num_triplet_types)
        num_quad_types = np.array(num_quad_types)
        self.atom_embedding = LearnableEmbedding(len(num_atom_types), num_atom_types+1, 
                                                 d_embeds=dim-self.fdim, d_feature=self.fdim, n_feature=2) \
            if atom_angle_embedding == "learnable" else SineEmbedding(len(num_atom_types), num_atom_types+1, dim, n_feature=2)
        self.bond_embedding = LearnableEmbedding(len(num_bond_types), num_bond_types+1, 
                                                 d_embeds=dim-self.fdim, d_feature=self.fdim, n_feature=1) \
            if dist_embedding == "learnable" else SineEmbedding(len(num_bond_types), num_bond_types+1, dim, n_feature=1)
        self.triplet_embedding = LearnableEmbedding(len(num_triplet_types), num_triplet_types+1, 
                                                 d_embeds=dim-self.fdim, d_feature=self.fdim, n_feature=1) \
            if trip_angle_embedding == "learnable" else SineEmbedding(len(num_triplet_types), num_triplet_types+1, dim)
        
        if use_quad:
            self.quad_embedding = LearnableEmbedding(len(num_quad_types), num_quad_types+1, 
                                                     d_embeds=dim-self.fdim, d_feature=self.fdim, n_feature=1) \
                if quad_angle_embedding == "learnable" else SineEmbedding(len(num_quad_types), num_quad_types+1, dim)

        self.dim = dim
        self.min_bond_dist = min_bond_dist
        self.max_bond_dist = max_bond_dist
        self.wnorm = wnorm
        self.use_quad = use_quad
        print(f"{'' if use_quad else colored('Not ', 'cyan')}Using Quadruplet Features")

        self.n_head = n_head
        assert dim % n_head == 0, "dim must be a multiple of n_head"
        self.layers = nn.ModuleList([GraphLayer(d_model=dim, d_inner=d_inner, n_head=n_head, d_head=dim//n_head, dropout=dropout,
                                                attn_dropout=dropatt, wnorm=wnorm, use_quad=use_quad, lev=i+1) for i in range(n_layers)])

        self.final_norm = nn.LayerNorm(dim)
        
        # TODO: Warning: we are predicting with the second-hierarchy bond (sub)types!!!!!
        self.final_dropout = final_dropout
        self.final_dim = num_bond_types[1]*final_dim
        self.final_lin1 = nn.Conv1d(dim, self.final_dim, kernel_size=1)
        self.final_res = nn.Sequential(
                             # ResidualBlock(self.final_dim, self.final_dim, groups=int(num_bond_types[1]), dropout=final_dropout),
                             ResidualBlock(self.final_dim, self.final_dim, groups=int(num_bond_types[1]), dropout=final_dropout),
                             nn.Conv1d(self.final_dim, num_bond_types[1], kernel_size=1, groups=int(num_bond_types[1]))
                         )
        self.apply(self.weights_init)
        
    def forward(self,x_atom,x_atom_pos, x_bond, x_bond_dist, x_triplet, x_triplet_angle, x_quad, x_quad_angle):
        # PART I: Form the embeddings and the distance matrix
        bsz = x_atom.shape[0]
        N = x_atom.shape[1]
        M = x_bond.shape[1]
        P = x_triplet.shape[1]
        Q = x_quad.shape[1] if self.use_quad else 0

        D = torch.zeros(x_atom.shape[0], N+M+P+Q, N+M+P+Q, device=x_atom.device)
        RA = slice(0,N)
        RB = slice(N,N+M)
        RT = slice(N+M, N+M+P)
        RQ = slice(N+M+P, N+M+P+Q)

        D[:,RA,RA] = sqdist(x_atom_pos[:,:,:3], x_atom_pos[:,:,:3])    # Only the x,y,z information, not charge/angle

        for i in range(D.shape[0]):
            # bonds
            a1,a2 = x_bond[i,:,3], x_bond[i,:,4]
            D[i, RA, RB] = torch.min(D[i, RA, a1], D[i, RA, a2])
            D[i, RB, RA] = D[i, RA, RB].transpose(0,1)
            D[i, RB, RB] = (D[i,a1,RB] + D[i,a2,RB])/2
            D[i, RB ,RB] = (D[i,RB,RB] + D[i,RB,RB].transpose(0,1))/2

            # triplets
            a1,a2,a3 = x_triplet[i,:,1], x_triplet[i,:,2], x_triplet[i,:,3]
            b1,b2 = x_triplet[i,:,4], x_triplet[i,:,5]
            D[i,RA,RT] = torch.min(torch.min(D[i,RA,a1], D[i,RA,a2]), D[i,RA, a3]) + D[i,RA,a1]
            D[i,RT,RA] = D[i,RA,RT].transpose(0,1)
            D[i,RB,RT] = torch.min(D[i,RB,b1], D[i,RB,b2])
            D[i,RT,RB] = D[i,RB,RT].transpose(0,1)
            D[i,RT,RT] = (D[i,b1,RT] + D[i,b2,RT]) / 2
            D[i,RT,RT] = (D[i,RT,RT] + D[i,RT,RT].transpose(0,1)) / 2
            
            if self.use_quad:
                # quad
                a1,a2,a3,a4 = x_quad[i,:,1], x_quad[i,:,2], x_quad[i,:,3], x_quad[i,:,4]
                b1,b2,b3 = x_quad[i,:,5], x_quad[i,:,6], x_quad[i,:,7]
                t1,t2 = x_quad[i,:,8], x_quad[i,:,9]
                D[i,RA,RQ] = torch.min(torch.min(torch.min(D[i,RA,a1], D[i,RA,a2]), D[i,RA, a3]), D[i,RA,a4]) + \
                             torch.min(D[i,RA,a1], D[i,RA,a2])
                D[i,RQ,RA] = D[i,RA,RQ].transpose(0,1)
                D[i,RB,RQ] = torch.min(torch.min(D[i,RB,b1], D[i,RB,b2]), D[i,RB, b3]) + D[i,RB,b1]
                D[i,RQ,RB] = D[i,RB,RQ].transpose(0,1)
                D[i,RT,RQ] = torch.min(D[i,RT,t1], D[i,RT,t2])
                D[i,RQ,RT] = D[i,RT,RQ].transpose(0,1)
                D[i,RQ,RQ] = (D[i,t1,RQ] + D[i,t2,RQ]) / 2
                D[i,RQ,RQ] = (D[i,RQ,RQ] + D[i,RQ,RQ].transpose(0,1))/2
        
        # No interaction (as in attention = 0) if query or key is the zero padding...
        if self.use_quad:
            mask = torch.cat([x_atom[:,:,0] > 0, x_bond[:,:,0] > 0, x_triplet[:,:,0] > 0, x_quad[:,:,0] > 0], dim=1).type(x_atom_pos.dtype)
        else:
            mask = torch.cat([x_atom[:,:,0] > 0, x_bond[:,:,0] > 0, x_triplet[:,:,0] > 0], dim=1).type(x_atom_pos.dtype)
        mask = torch.einsum('bi, bj->bij', mask, mask)
        new_mask = -1e20 * torch.ones_like(mask).to(mask.device)
        new_mask[mask > 0] = 0
        if self.use_quad:
            Z = torch.cat([
                self.atom_embedding(x_atom[:,:,:3], x_atom_pos[:,:,3:]),    
                self.bond_embedding(x_bond[:,:,:3], x_bond_dist),           
                self.triplet_embedding(x_triplet[:,:,:2], x_triplet_angle),  
                self.quad_embedding(x_quad[:,:,:1], x_quad_angle),           
                ], dim=1)
        else:
            Z = torch.cat([
                self.atom_embedding(x_atom[:,:,:3], x_atom_pos[:,:,3:]),    
                self.bond_embedding(x_bond[:,:,:3], x_bond_dist),           
                self.triplet_embedding(x_triplet[:,:,:2], x_triplet_angle),          
                ], dim=1)
        
        # PART II: Pass through a bunch of self-attention and position-wise feed-forward blocks
        seed = np.random.uniform(0,1)
        for i in range(len(self.layers)):
            Z = self.layers[i](Z, D, new_mask, mask, RA, RB, RT, RQ, store=False)
            
        # PART III: Coupling type based (grouped) transformations
        Z = self.final_norm(Z)
        Z_group = self.final_lin1(Z.transpose(1,2)[:,:,RB])
        return self.final_res(Z_group), Z

    @staticmethod
    def init_weight(weight):
        nn.init.uniform_(weight, -0.1, 0.1)

    @staticmethod
    def init_bias(bias):
        nn.init.constant_(bias, 0.0)

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1 or classname.find('Conv1d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                GraphTransformer.init_weight(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                GraphTransformer.init_bias(m.bias)
   
