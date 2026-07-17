import torch
import numpy as np 
from scipy.stats import norm
import sys
import os
sys.path.append(os.path.abspath('../..'))
from dataloader import load_data
from scipy.spatial.distance import pdist, squareform

def compute_bandwidths(name, N1, rs, number_bandwidths):
    X, Y = load_data(name, N1, rs + 283, 1)
    Z = np.concatenate((X, Y))
    pairwise_distances = squareform(pdist(Z))
    distances = pairwise_distances[np.triu_indices(pairwise_distances.shape[0])]
    median = np.median(distances)
    
    if number_bandwidths == 1:
        return median
    else:
        distances = distances + (distances == 0) * median
        dd = np.sort(distances)
        lambda_min = dd[(np.floor(len(dd) * 0.05).astype(int))] / 2
        lambda_max = dd[(np.floor(len(dd) * 0.95).astype(int))] * 2
        bandwidths = np.linspace(lambda_min, lambda_max, number_bandwidths)
        return bandwidths

def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x

def get_item(x, device):
    """get the numpy value from a torch tensor."""
    if device == torch.device("cpu"):
        x = x.detach().numpy()
    else:
        x = x.cpu().detach().numpy()
    return x

def Pdist2(x, y):
    """compute the paired distance between x and y."""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist<0]=0
    return Pdist

def h1_mean_var_gram(Kx, Ky, Kxy):
    """compute value of MMD and std of MMD using kernel matrix."""
    Kxxy = torch.cat((Kx,Kxy),1)
    Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
    Kxyxy = torch.cat((Kxxy,Kyxy),0)
    nx = Kx.shape[0]
    ny = Ky.shape[0]
    
    xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
    yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
    xy = torch.div((torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (ny - 1)))
    mmd2 = xx - 2 * xy + yy
    return mmd2, Kxyxy, 4 - xx - yy

def MMDu(Fea, len_s, sigma0):
    """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
    X = Fea[0:len_s, :] # fetch the sample 1 (features of deep networks)
    Y = Fea[len_s:, :] # fetch the sample 2 (features of deep networks)
    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)

    Kx = torch.exp(-Dxx / sigma0**2)
    Ky = torch.exp(-Dyy / sigma0**2)
    Kxy = torch.exp(-Dxy / sigma0**2)
    return h1_mean_var_gram(Kx, Ky, Kxy)

def TV_cal(ind1, ind2, N):
    Count1 = torch.zeros(N)
    Count2 = torch.zeros(N)
    Count11 = torch.zeros(N)
    Count22 = torch.zeros(N)
    
    for i in range(len(ind1)//2):
        Count1[ind1[i]] += 1
        Count2[ind2[i]] += 1
    for i in range(len(ind1)//2, len(ind1)):
        Count11[ind1[i]] += 1
        Count22[ind2[i]] += 1
    
    fi = torch.zeros(N)
    Zi = torch.zeros(N)
    stat = torch.tensor(0.0)
    for i in range(N):
        fi[i] = max(Count11[i] + Count22[i], 1)
        Zi[i] = (Count1[i] - Count2[i])**2 - Count1[i] - Count2[i]

        stat += Zi[i]/fi[i]
    
    return stat.item()
    
def TV_testing(Z, P1, P2, N, rs, sigma0, n_test, n_per, alpha, device, dtype):
    np.random.seed(seed=rs)
    torch.manual_seed(rs)
    torch.cuda.manual_seed(rs)
    
    TV_values = torch.zeros(n_per)
    P_uniform = (torch.ones(len(Z))/len(Z)).to(device, dtype)
    for k in range(n_per):
        ind_uni = torch.multinomial(P_uniform, N, replacement=True)
        ind1 = torch.multinomial(P1, N, replacement=True)
        TV_values[k] = TV_cal(ind_uni, ind1, len(Z))
    
    TV_values = torch.sort(TV_values)[0]
    
    pos = int(n_per * (1-alpha))
    TV_thres = TV_values[pos]

    H_TV = np.zeros(n_test)
    for k in range(n_test):
        ind_uni = torch.multinomial(P_uniform, N, replacement=True)
        ind2 = torch.multinomial(P2, N, replacement=True)
        TV_value = TV_cal(ind_uni, ind2, len(Z))
        H_TV[k] = int(TV_value>=TV_thres)

    return H_TV