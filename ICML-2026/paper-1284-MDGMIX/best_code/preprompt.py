import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL, Lp, GcnLayers
from layers import GCN, AvgReadout 
import tqdm
import numpy as np
#import dgl
from sklearn.decomposition import PCA
from layers import Attentivemod
from tools import *
from models.gcn import GCNPYG
import copy
from utils import process

from downprompt import composedtoken
from torch_geometric.nn.conv.gcn_conv import gcn_norm
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class MDGMIX(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_domains, domain_center, device):
        super(MDGMIX, self).__init__()
        self.gnn =  GCNPYG(num_features=input_dim, hidden=hidden_dim,num_conv_layers=2, residual=True, dropout=0.1, gfn=False).to(device)

        self.domain_tokens = torch.tensor(domain_center, dtype=torch.float32)

        self.num_domains = num_domains
        self.pre_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)) # binary classification: same domain or not
   
        self.criterion = nn.CrossEntropyLoss()

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.grl = GradientReversalLayer()

        self.composition_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_domains),
            nn.Softmax(dim=-1)  
        )

        self.loss = CompositionLoss()



    def forward(self, merged_graphs, labels, domain_lables):
        """Pre-training forward pass"""

        graph_embeddings = []

        l_y = []
        r_y = []
        lam = []
        mix_loss = 0
        for graph in merged_graphs:
     
            source_g = copy.deepcopy(graph)
          
            h = self.gnn(source_g)
    
            graph_emb = torch.mean(h, dim=0).reshape(1,-1)
            
            graph_embeddings.append(graph_emb)
            
        graph_embeddings = torch.cat(graph_embeddings, dim=0)
        
        predictions = self.pre_proj(graph_embeddings)


        domain_loss = self.criterion(predictions, domain_lables.squeeze(1))
        pred_mask = torch.argmax(predictions,dim=1)
        

        inv_graph_embeddings = graph_embeddings
        inv_graph_embeddings = self.grl.apply(inv_graph_embeddings, 1.0)#梯度反转的位置还得具体分析

    
        composition_probs = self.composition_predictor(inv_graph_embeddings)

        loss = self.loss(composition_probs[pred_mask], labels[pred_mask])
    
        return loss + domain_loss 


    def embed(self, target_data):
        graph_emb, h = self.gnn(target_data)
        return h

    
def pca_compression(seq,k):
    pca = PCA(n_components=k)
    seq = pca.fit_transform(seq)
    
    print(pca.explained_variance_ratio_.sum())
    return seq

def svd_compression(seq, k):
    res = np.zeros_like(seq)
    U, Sigma, VT = np.linalg.svd(seq)
    print(U[:,:k].shape)
    print(VT[:k,:].shape)
    res = U[:,:k].dot(np.diag(Sigma[:k]))
 
    return res



class CompositionLoss(nn.Module):
    def __init__(self):
        super(CompositionLoss, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, pred_probs, target_probs):
     
        pred_log_probs = torch.log(pred_probs + 1e-8)
        loss = self.kl_loss(pred_log_probs, target_probs)
        return loss



def mixup_criterion(logits, l_y, r_y, lam):
    loss = lam * F.cross_entropy(logits, l_y) + (1 - lam) * F.cross_entropy(logits, r_y)
    return loss.mean()