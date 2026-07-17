import torch
import numpy as np 
import sys
import os
sys.path.append(os.path.abspath('../..'))
from scipy.stats import norm
from kernel_selection import set_random_seed


def NAMMD_discrete_P(Z, P1, P2, N, sigma0, M_matrix, K = 1):
    def h1_mean_var_gram(Kzz, P11, P12, P22, K = 1):
        """compute value of VMD and std of VMD using kernel matrix."""
        xx = torch.sum(Kzz * P11)
        yy = torch.sum(Kzz * P22)
        xy = torch.sum(Kzz * P12)

        MMD = xx - 2 * xy + yy
        Reg = 4 * K - xx - yy
        
        return MMD, Reg
    
    def Pdist2(x, y, M_matrix):
        """compute the paired distance between x and y."""

        n = x.shape[0]
        d = x.shape[1]
        assert x.shape[0]==y.shape[0]
        
        x_Mat = torch.matmul(x, M_matrix)
        y_Mat = torch.matmul(y, M_matrix)    
        
        x_Mat_x = torch.sum(torch.mul(x_Mat, x), 1).view(-1, 1)
        y_Mat_y = torch.sum(torch.mul(y_Mat, y), 1).view(1, -1)
        x_Mat_y = torch.sum(torch.mul(x_Mat.unsqueeze(1).expand(n,n,d), y), 2)
        y_Mat_x = torch.sum(torch.mul(y_Mat.unsqueeze(1).expand(n,n,d), x), 2).t()
        
        Pdist = x_Mat_x + y_Mat_y - x_Mat_y - y_Mat_x
        Pdist[Pdist<0]=0

        return Pdist
    
    def Prob(P1, P2, N):
        """compute the paired distance between x and y."""
        P1 = P1.reshape(N,1)
        P2 = P2.reshape(N,1)
        
        return P1 @ P2.t()

    Dzz = Pdist2(Z, Z, M_matrix)
    Kzz = torch.exp(-Dzz / sigma0**2)
    
    P11 = Prob(P1,P1,N)
    P12 = Prob(P1,P2,N)
    P22 = Prob(P2,P2,N)

    return h1_mean_var_gram(Kzz, P11, P12, P22, K)

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

def Pdist(x, y):
    """compute the paired distance between x and y."""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist<0]=0
    return Pdist

def Pdist2(x, y, M_matrix):
    """compute the paired distance between x and y."""
    n = x.shape[0]
    d = x.shape[1]
    assert x.shape[0]==y.shape[0]
    
    x_Mat = torch.matmul(x, M_matrix)
    y_Mat = torch.matmul(y, M_matrix)    
    
    x_Mat_x = torch.sum(torch.mul(x_Mat, x), 1).view(-1, 1)
    y_Mat_y = torch.sum(torch.mul(y_Mat, y), 1).view(1, -1)
    x_Mat_y = torch.sum(torch.mul(x_Mat.unsqueeze(1).expand(n,n,d), y), 2)
    y_Mat_x = torch.sum(torch.mul(y_Mat.unsqueeze(1).expand(n,n,d), x), 2).t()
    
    Pdist = x_Mat_x + y_Mat_y - x_Mat_y - y_Mat_x
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

def MMDu(Fea, len_s, sigma0, M_matrix):
    """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
    X = Fea[0:len_s, :] # fetch the sample 1 (features of deep networks)
    Y = Fea[len_s:, :] # fetch the sample 2 (features of deep networks)
    Dxx = Pdist2(X, X, M_matrix)
    Dyy = Pdist2(Y, Y, M_matrix)
    Dxy = Pdist2(X, Y, M_matrix)

    Kx = torch.exp(-Dxx / sigma0**2)
    Ky = torch.exp(-Dyy / sigma0**2)
    Kxy = torch.exp(-Dxy / sigma0**2)
    return h1_mean_var_gram(Kx, Ky, Kxy)

def training(Z, P1, P2, N1, rs, ne_MMD, bs_MMD, lr_MMD, ne_NAMMD, bs_NAMMD, lr_NAMMD, b_NAMMD, device, dtype):
    set_random_seed(rs)
    P_uniform = (torch.ones(len(Z))/len(Z)).to(device, dtype)
    ind1 = torch.multinomial(P_uniform, N1, replacement=True)
    Z1 = Z[ind1]
    ind2 = torch.multinomial(P2, N1, replacement=True)
    Z2 = Z[ind2]
    S_train = torch.cat((Z1, Z2))
    sigma0, M_matrix = MMD_fit(S_train, N1, lr_MMD, ne_MMD, bs_MMD, device, dtype, seed=rs)
    sigma0, M_matrix = NAMMD_fit(S_train, N1, sigma0, M_matrix, lr_NAMMD, ne_NAMMD, bs_NAMMD, b_NAMMD, device, seed=rs)
    return sigma0.detach(), M_matrix.detach()
    
def testing(Z, P1, P2, N1, rs, sigma0, M_matrix, n_test, n_per, alpha, device, dtype):
    H_MMD = np.zeros(n_test)
    H_NAMMD = np.zeros(n_test)
    set_random_seed(rs)

    H_NAMMD = np.zeros(n_test)
    H_MMD = np.zeros(n_test)
    P_uniform = (torch.ones(len(Z))/len(Z)).to(device, dtype)

    NAMMD_values = torch.zeros(n_per)
    MMD_values = torch.zeros(n_per)
    for k in range(n_per):
        ind_uni = torch.multinomial(P_uniform, N1, replacement=True)
        Z_uni = Z[ind_uni]
        ind1 = torch.multinomial(P1, N1, replacement=True)
        Z1 = Z[ind1]
        Fea = torch.cat((Z_uni, Z1))
        TEMP = MMDu(Fea, N1, sigma0, M_matrix)
        NAMMD_values[k] = TEMP[0]/TEMP[2]
        MMD_values[k] = TEMP[0]
    NAMMD_values = torch.sort(NAMMD_values)[0]
    MMD_values = torch.sort(MMD_values)[0]
    pos = int(n_per * (1-alpha))
    NAMMD_thres = NAMMD_values[pos]
    MMD_thres = MMD_values[pos]
    for k in range(n_test):
        ind_uni = torch.multinomial(P_uniform, N1, replacement=True)
        Z_uni = Z[ind_uni]
        ind2 = torch.multinomial(P2, N1, replacement=True)
        Z2 = Z[ind2]
        Fea = torch.cat((Z_uni, Z2))
        TEMP = MMDu(Fea, N1, sigma0, M_matrix)
        NAMMD_value = TEMP[0]/TEMP[2]
        MMD_value = TEMP[0]
        H_NAMMD[k] = int(NAMMD_value>=NAMMD_thres)
        H_MMD[k] = int(MMD_value>=MMD_thres)

    return H_MMD, H_NAMMD

def MMD_fit(S, N1, learning_rate, N_epoch, batch_size, device, dtype, seed=None):
    seed = int(torch.initial_seed() % (2**32 - 1) if seed is None else seed)
    set_random_seed(seed)

    Dxy = Pdist(S[:N1, :], S[N1:, :])
    sigma0 = torch.sqrt(Dxy.median().clamp_min(1e-12))
    M_matrix = np.identity(S.shape[1])
    # M_matrix = np.tile(M_matrix, (1, 1)).reshape((-1, S.shape[1], S.shape[1]))
    M_matrix = MatConvert(M_matrix, device, dtype)
    
    sigma0.requires_grad = True
    M_matrix.requires_grad = True
    optimizer = torch.optim.Adam([sigma0]+[M_matrix], lr=learning_rate)

    for t in range(N_epoch):
        S1 = S[:N1,:]
        S2 = S[N1:,:]
        epochs = max(min(int(len(S1) / batch_size) * 2, int(len(S2) / batch_size) * 2), 1)
        for i in range(epochs):
            if int( len(S1) / batch_size ) * 2 <= 1 or int( len(S2) / batch_size) * 2 <= 1:
                ind1 = np.random.choice(np.arange(len(S1)), min(len(S1),len(S2)), replace=False)
                ind2 = np.random.choice(np.arange(len(S2)), min(len(S1),len(S2)), replace=False)
            else:
                ind1 = np.random.choice(np.arange(len(S1)), int(batch_size/2), replace=False)
                ind2 = np.random.choice(np.arange(len(S2)), int(batch_size/2), replace=False)

            S_batch = torch.cat([S1[ind1], S2[ind2]], 0)
            if device == torch.device("cpu"):
                S1 = torch.index_select(S1, 0, torch.tensor(np.delete(np.arange(len(S1)),ind1,0), dtype=torch.long))
                S2 = torch.index_select(S2, 0, torch.tensor(np.delete(np.arange(len(S2)), ind2, 0), dtype=torch.long))
            else:
                S1 = torch.index_select(S1,0, torch.tensor(np.delete(np.arange(len(S1)),ind1,0), dtype=torch.long).cuda())
                S2 = torch.index_select(S2, 0, torch.tensor(np.delete(np.arange(len(S2)), ind2, 0), dtype=torch.long).cuda())
            TEMPa = MMDu(S_batch, int(len(S_batch)/2), sigma0, M_matrix)
            mmd_value_tempa = -1 * TEMPa[0]
            optimizer.zero_grad()
            mmd_value_tempa.backward(retain_graph=True)
            optimizer.step()
        if (t + 1) % 100 == 0:
            with torch.no_grad():
                eigvalues, eigvectors = torch.linalg.eig(M_matrix)
                eigvalues = torch.max(eigvalues.real, torch.tensor(1e-5).to(device, dtype))
                eigvectors = eigvectors.real
                eigvectors = eigvectors.t().reshape(eigvectors.shape[0], -1, eigvectors.shape[1])
                M_matrix = eigvalues[0] * eigvectors[0].t() * eigvectors[0]
                for i in range(1, len(eigvalues)):
                    M_matrix += eigvalues[i] * eigvectors[i] * eigvectors[i].t()
        if (t+1) % 500 == 0 or t == 0:
            print("MMD: ", -1 * mmd_value_tempa.item())
    return sigma0, M_matrix

def NAMMD_fit(S, N1, sigma0, M_matrix, learning_rate, N_epoch, batch_size, b, device, seed=None):
    seed = int(torch.initial_seed() % (2**32 - 1) if seed is None else seed)
    set_random_seed(seed)

    sigma0.requires_grad = True
    M_matrix.requires_grad = True
    optimizer = torch.optim.Adam([sigma0]+[M_matrix], lr=learning_rate)

    for t in range(N_epoch):
        S1 = S[:N1,:]
        S2 = S[N1:,:]
        epochs = max(min(int(len(S1) / batch_size) * 2, int(len(S2) / batch_size) * 2), 1)
        for i in range(epochs):
            if int( len(S1) / batch_size ) * 2 <= 1 or int( len(S2) / batch_size) * 2 <= 1:
                ind1 = np.random.choice(np.arange(len(S1)), min(len(S1),len(S2)), replace=False)
                ind2 = np.random.choice(np.arange(len(S2)), min(len(S1),len(S2)), replace=False)
            else:
                ind1 = np.random.choice(np.arange(len(S1)), int(batch_size/2), replace=False)
                ind2 = np.random.choice(np.arange(len(S2)), int(batch_size/2), replace=False)

            S_batch = torch.cat([S1[ind1], S2[ind2]], 0)
            if device == torch.device("cpu"):
                S1 = torch.index_select(S1, 0, torch.tensor(np.delete(np.arange(len(S1)),ind1,0), dtype=torch.long))
                S2 = torch.index_select(S2, 0, torch.tensor(np.delete(np.arange(len(S2)), ind2, 0), dtype=torch.long))
            else:
                S1 = torch.index_select(S1,0, torch.tensor(np.delete(np.arange(len(S1)),ind1,0), dtype=torch.long).cuda())
                S2 = torch.index_select(S2, 0, torch.tensor(np.delete(np.arange(len(S2)), ind2, 0), dtype=torch.long).cuda())
            
            TEMP = MMDu(S_batch, len(S_batch)//2, sigma0, M_matrix)
            NAMMD_value_temp = -1 * TEMP[0] + b * TEMP[2]

            # Initialize optimizer and Compute gradient
            optimizer.zero_grad()
            NAMMD_value_temp.backward(retain_graph=True)
            # Update weights using gradient descent
            optimizer.step()
            # Print NAMMD, std of NAMMD and J
        if (t+1) % 500 == 0 or t == 0:
            print("NAMMD_value: ", -1 * NAMMD_value_temp.item())
    return sigma0, M_matrix
