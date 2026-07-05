import numpy as np
import torch
import ot
def compute_true_Wasserstein(X,Y,p=2):
    M = ot.dist(X.cpu().detach().numpy(), Y.cpu().detach().numpy())
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]
    return ot.emd2(a, b, M)
def compute_Wasserstein(M,device='cpu',e=0):
    if(e==0):
        pi = ot.emd([],[],M.cpu().detach().numpy()).astype('float32')
    else:
        pi = ot.sinkhorn([], [], M.cpu().detach().numpy(),reg=e).astype('float32')
    pi = torch.from_numpy(pi).to(device)
    return torch.sum(pi*M)

