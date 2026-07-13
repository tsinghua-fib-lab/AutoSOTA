
# Import all the packages
# Import all the packages
import torch
import torch.nn as nn
from spectral_clustering import Spectral_clustering_init
from sklearn import metrics
import numpy as np

import math




class LSM(nn.Module,Spectral_clustering_init):
    def __init__(self,sparse_i,sparse_j, input_size,latent_dim,graph_type,non_sparse_i=None,non_sparse_j=None,sparse_i_rem=None,sparse_j_rem=None,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), neg_ratio=5.0):
        super(LSM, self).__init__()
        # initialization
        # initialization
        Spectral_clustering_init.__init__(self,num_of_eig=latent_dim,method='Normalized')
        self.input_size=input_size
        self.neg_ratio = neg_ratio
     
       
        self.bias=nn.Parameter(torch.randn(1,device=device))
        self.scaling_factor=nn.Parameter(torch.randn(1,device=device))
        self.latent_dim=latent_dim
        #degs=torch.cat((sparse_i,sparse_j)).unique(return_counts=True)[1]
        #self.gamma=nn.Parameter(degs/degs.max())

        self.gamma=nn.Parameter(torch.randn(self.input_size,device=device))

        #self.alpha=nn.Parameter(torch.randn(self.input_size,device=device))
        self.scaling=1
        #create indices to index properly the receiver and senders variable
        self.sparse_i_idx=sparse_i
        self.sparse_j_idx=sparse_j
        self.pdist = nn.PairwiseDistance(p=2,eps=0)
        self.softplus=nn.Softplus()
        self.sampling_weights=torch.ones(self.input_size,device=device)
        self.device=device

       
        self.non_sparse_i_idx_removed=non_sparse_i
     
        self.non_sparse_j_idx_removed=non_sparse_j
           
        self.sparse_i_idx_removed=sparse_i_rem
        self.sparse_j_idx_removed=sparse_j_rem
        if self.non_sparse_i_idx_removed!=None:
            # total sample of missing dyads with i<j
            self.removed_i=torch.cat((self.non_sparse_i_idx_removed,self.sparse_i_idx_removed))
            self.removed_j=torch.cat((self.non_sparse_j_idx_removed,self.sparse_j_idx_removed))

        self.Softmax=nn.Softmax(1)
        

        
        H = torch.zeros((latent_dim, latent_dim - 1),  dtype=torch.float32)
        for j in range(latent_dim - 1):
            denom = torch.sqrt(torch.tensor((j + 1) * (j + 2),
                                             dtype=torch.float32))
            H[: j + 1, j] = 1.0 / denom
            H[j + 1, j] = -(j + 1) / denom
        
        #self.V=H
        
        # Unconstrained parameter matrix
        self.W = nn.Parameter(torch.randn(latent_dim, latent_dim-1,device=device))

    

       
        norm=True
        if norm:

            self.spectral_data=self.spectral_clustering()

            spectral_centroids_to_z=self.spectral_data
            if self.spectral_data.shape[1]>latent_dim:
                self.latent_z1=nn.Parameter(spectral_centroids_to_z[:,0:latent_dim])
            elif self.spectral_data.shape[1]==latent_dim:
                self.latent_z1=nn.Parameter(spectral_centroids_to_z)
            else:
                self.latent_z1=nn.Parameter(torch.zeros(self.input_size,latent_dim,device=device))
                self.latent_z1.data[:,0:self.spectral_data.shape[1]]=spectral_centroids_to_z
        else:
            self.latent_z1=nn.Parameter(torch.rand(self.input_size,latent_dim,device=device))

            
    def forward_V(self):
        # 1. Zero-sum each column
        W = self.W - self.W.mean(dim=0, keepdim=True)
    
        # 2. Orthonormalize via QR
        Q, R = torch.linalg.qr(W, mode="reduced")
        d = torch.sign(torch.diag(R))
        d[d == 0] = 1
        Q = Q * d.unsqueeze(0)
        return Q[:, :self.latent_dim-1]

                

    
    
    
    def sample_uniform_pairs(self, N, M_neg, device, return_q=False, symmetric=True):
        """
        Uniform negative sampling over ordered pairs (i != j).
        If symmetric=True, also returns the mirrored pairs (j,i) so that
        negatives match the undirected edge convention (stored both ways).
        """
        # sample M_neg ordered pairs with i != j
        i = torch.randint(0, N, (M_neg,), device=device)
        j = torch.randint(0, N - 1, (M_neg,), device=device)
        j = j + (j >= i)  # skip self-pairs by shifting >= i
    
        if symmetric:
            i_all = torch.cat([i, j], dim=0)
            j_all = torch.cat([j, i], dim=0)
        else:
            i_all, j_all = i, j
    
        if return_q:
            # uniform over all ordered pairs i != j
            q = torch.full((i_all.numel(),), 1.0 / (N * (N - 1)), device=device)
            return i_all, j_all, q
        return i_all, j_all
        
    
    def helmert_basis(self,K) -> torch.Tensor:
        """
        Construct the (K x (K-1)) Helmert submatrix for ILR basis.
        """
        H = torch.zeros((K, K - 1),  dtype=torch.float32)
        for j in range(K - 1):
            denom = torch.sqrt(torch.tensor((j + 1) * (j + 2),
                                             dtype=torch.float32))
            H[: j + 1, j] = 1.0 / denom
            H[j + 1, j] = -(j + 1) / denom
        return H
    
    def ilr_transform(self, x, V, eps=1e-12):
        x = x.clamp_min(eps)
        return x.log() @ V


        
  

     #introduce the likelihood function 
    def LSM_likelihood_bias(self,epoch,euclidean):
        '''
        Bernoulli log-likelihood ignoring the log(k!) constant
        
        '''

        M_neg=int(self.neg_ratio * self.sparse_i_idx.shape[0])
        
        
        
        neg_i,neg_j=self.sample_uniform_pairs(self.input_size,M_neg, device=self.device)
        scale=(self.input_size*(self.input_size-1))/(neg_i.shape[0])

        self.euclidean=euclidean
        if euclidean:
            self.latent_z=self.Softmax(self.latent_z1)

        else:
            self.latent_z_=self.Softmax(self.latent_z1)
            
            #self.V=self.helmert_basis(self.latent_dim)
            self.V=self.forward_V()

            self.latent_z = self.ilr_transform(self.latent_z_, self.V)
        
        if self.scaling:
    
              
                mat=self.softplus(self.gamma[neg_i]+self.gamma[neg_j])                
                z_pdist1=scale*mat.sum()#(mat-torch.diag(torch.diagonal(mat))).sum()
    
                
                z_pdist2=((self.gamma[self.sparse_i_idx]+self.gamma[self.sparse_j_idx])).sum()
    
                
    
               
                log_likelihood_sparse=z_pdist2-z_pdist1
            
           
        else:
            mat_0= -(((self.latent_z[neg_i]-self.latent_z[neg_j]+1e-06)**2).sum(-1)**0.5) +(self.gamma[neg_i]+self.gamma[neg_j])
            
            
            
            

                            
            mat_1=-(((self.latent_z[self.sparse_i_idx]-self.latent_z[self.sparse_j_idx]+1e-06)**2).sum(-1)**0.5) +(self.gamma[self.sparse_i_idx]+self.gamma[self.sparse_j_idx])

                    
            
            mat=self.softplus(mat_0)
            z_pdist2=mat_1.sum()
            z_pdist1=scale*mat.sum()

           
        


            log_likelihood_sparse=z_pdist2-z_pdist1
        
        
        return log_likelihood_sparse
    
    
    
    
   
    
    def link_prediction(self):
        if self.euclidean:
            self.latent_z=self.Softmax(self.latent_z1)

        else:
            self.latent_z_=self.Softmax(self.latent_z1)
            
            #self.V=self.helmert_basis(self.latent_dim)
            self.V=self.forward_V()
            self.latent_z = self.ilr_transform(self.latent_z_, self.V)
            #self.gamma=torch.ones(self.input_size)*self.bias


        with torch.no_grad():
            
            z_pdist_miss=(((self.latent_z[self.removed_i]-self.latent_z[self.removed_j])**2).sum(-1))**0.5
            if self.scaling:
                logit_u_miss=self.gamma[self.removed_i]+self.gamma[self.removed_j]

            else:
                logit_u_miss=-z_pdist_miss+self.gamma[self.removed_i]+self.gamma[self.removed_j]
            rates=logit_u_miss
            self.rates=rates

            target=torch.cat((torch.zeros(self.non_sparse_i_idx_removed.shape[0]),torch.ones(self.sparse_i_idx_removed.shape[0])))
            #fpr, tpr, thresholds = metrics.roc_curve(target.cpu().data.numpy(), rates.cpu().data.numpy())
            precision, tpr, thresholds = metrics.precision_recall_curve(target.cpu().data.numpy(), rates.cpu().data.numpy())

           
        return metrics.roc_auc_score(target.cpu().data.numpy(),rates.cpu().data.numpy()),metrics.auc(tpr,precision)
    

   






    
