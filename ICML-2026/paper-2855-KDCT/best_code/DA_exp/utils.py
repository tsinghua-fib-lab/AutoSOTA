import torch
import numpy as np 
import sys
import os
sys.path.append(os.path.abspath('../..'))
from scipy.stats import norm
from kernel_selection import (
    select_gaussian_bandwidths_reference,
    set_random_seed,
    split_selected_bandwidths,
)
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

###### MMD
def MMD_TEST(Fea, N_per, N1, sigma0, alpha, device):
    mmd_vector = np.zeros(N_per)
    TEMP = MMDu(Fea, N1, sigma0)
    mmd_value = get_item(TEMP[0], device)
    Kxyxy = TEMP[1]
    count = 0
    nxy = Fea.shape[0]
    nx = N1

    for r in range(N_per):
        # print r
        ind = np.random.choice(nxy, nxy, replace=False)
        # divide into new X, Y
        indx = ind[:nx]
        indy = ind[nx:]
        Kx = Kxyxy[np.ix_(indx, indx)]
        Ky = Kxyxy[np.ix_(indy, indy)]
        Kxy = Kxyxy[np.ix_(indx, indy)]

        TEMP = h1_mean_var_gram(Kx, Ky, Kxy)
        mmd_vector[r] = TEMP[0]
        if mmd_vector[r] > mmd_value:
            count = count + 1
    if count > np.ceil(N_per * alpha):
        h = 0
        threshold = "NaN"
    else:
        h = 1
        S_mmd_vector = np.sort(mmd_vector)
        threshold = S_mmd_vector[np.int(np.ceil(N_per * (1 - alpha)))]

    return h, count/N_per

def MMD_fit(S, N1, learning_rate, N_epoch, batch_size, device, seed=None):
    seed = int(torch.initial_seed() % (2**32 - 1) if seed is None else seed)
    sigma_mmd, _, _ = select_gaussian_bandwidths_reference(
        S,
        N1,
        seed=seed,
        max_reference_samples=min(N1, max(int(batch_size), 500)),
        num_bandwidths=25,
        verbose=False,
    )
    return sigma_mmd

##### NAMMD
def NAMMD_TEST(Fea, N_per, N1, sigma0, alpha, device):
    NAMMD_vector = np.zeros(N_per)
    TEMP = MMDu(Fea, N1, sigma0)
    MMD = get_item(TEMP[0], device)
    Reg = get_item(TEMP[2], device)
    NAMMD_value = MMD/Reg
    Kxyxy = TEMP[1]
    count = 0
    nxy = Fea.shape[0]
    nx = N1

    for r in range(N_per):
        # print r
        ind = np.random.choice(nxy, nxy, replace=False)
        # divide into new X, Y
        indx = ind[:nx]
        indy = ind[nx:]
        Kx = Kxyxy[np.ix_(indx, indx)]
        Ky = Kxyxy[np.ix_(indy, indy)]
        Kxy = Kxyxy[np.ix_(indx, indy)]

        TEMP = h1_mean_var_gram(Kx, Ky, Kxy)
        MMD = get_item(TEMP[0], device)
        Reg = get_item(TEMP[2], device)
        NAMMD_vector[r] = MMD/Reg
        if NAMMD_vector[r] > NAMMD_value:
            count = count + 1
    if count > np.ceil(N_per * alpha):
        h = 0
        threshold = "NaN"
    else:
        h = 1
        S_mmd_vector = np.sort(NAMMD_vector)
        threshold = S_mmd_vector[np.int(np.ceil(N_per * (1 - alpha)))]

    return h, count/N_per

def NAMMD_fit(S, N1, sigma0, learning_rate, N_epoch, batch_size, b, device, seed=None):
    seed = int(torch.initial_seed() % (2**32 - 1) if seed is None else seed)
    _, sigma_nammd, _ = select_gaussian_bandwidths_reference(
        S,
        N1,
        seed=seed,
        max_reference_samples=min(N1, max(int(batch_size), 500)),
        num_bandwidths=25,
        verbose=False,
    )
    return sigma_nammd

def _asymptotic_values(Fea, N1, sigma0, device):
    TEMP = MMDu(Fea, N1, sigma0)
    NAMMD_value = TEMP[0]/TEMP[2]
    MMD_value = TEMP[0]
    Kxyxy = TEMP[1]
    ind = np.arange(2 * N1)
    indx = ind[:N1]
    indy = ind[N1:]
    Kx = Kxyxy[np.ix_(indx, indx)]
    Ky = Kxyxy[np.ix_(indy, indy)]
    Kxy = Kxyxy[np.ix_(indx, indy)]

    EE = torch.ones(N1).to(device)
    Kx_ = Kx * (1-torch.eye(N1)).to(device)
    Ky_ = Ky * (1-torch.eye(N1)).to(device)

    Xxi1 = (N1*(N1-1)*(N1-2))**(-1)*(torch.norm(Kx_@EE,p=2)**2-torch.norm(Kx_,p='fro')**2) - \
    (N1*(N1-1)*(N1-2)*(N1-3))**(-1)*((EE@Kx_@EE)**2-4*(torch.norm(Kx_@EE,p=2)**2+2*torch.norm(Kx_,p='fro')**2))
    Yxi1 = (N1*(N1-1)*(N1-2))**(-1)*(torch.norm(Ky_@EE,p=2)**2-torch.norm(Ky_,p='fro')**2) - \
    (N1*(N1-1)*(N1-2)*(N1-3))**(-1)*((EE@Ky_@EE)**2-4*(torch.norm(Ky_@EE,p=2)**2+2*torch.norm(Ky_,p='fro')**2))

    varxi1=Xxi1 + Yxi1 + (N1**2*(N1-1))**(-1)*(torch.norm(Kxy@EE,p=2)**2-torch.norm(Kxy,p='fro')**2) - \
        2*(N1**2*(N1-1)**2)**(-1)*((EE@Kxy@EE)**2-torch.norm(Kxy.T@EE,p=2)**2-torch.norm(Kxy@EE,p=2)**2+torch.norm(Kxy,p='fro')**2) + \
        (N1**2*(N1-1))**(-1)*(torch.norm(Kxy.T@EE,p=2)**2-torch.norm(Kxy,p='fro')**2) - \
        2*(N1**2*(N1-1))**(-1)*EE@Kx_@Kxy@EE + 2*(N1*N1*(N1-1)*(N1-2))**(-1)*(EE@Kx_@EE*EE@Kxy@EE-2*EE@Kx_@Kxy@EE) - \
        2*(N1**2*(N1-1))**(-1)*EE@Ky_@Kxy.T@EE + 2*(N1*N1*(N1-1)*(N1-2))**(-1)*(EE@Ky_@EE*EE@Kxy.T@EE-2*EE@Ky_@Kxy.T@EE)
    
    varxi2 = Xxi1 + Yxi1 +2* N1**(-2)*torch.norm(Kxy,p='fro')**2- 2*(N1**2*(N1-1)**2)**(-1)*((EE@Kxy@EE)**2-torch.norm(Kxy.T@EE,p=2)**2-torch.norm(Kxy@EE,p=2)**2+torch.norm(Kxy,p='fro')**2) - \
    4*(N1**2*(N1-1))**(-1)*EE@Kx_@Kxy@EE+4*(N1*N1*(N1-1)*(N1-2))**(-1)*(EE@Kx_@EE*EE@Kxy@EE-2*EE@Kx_@Kxy@EE) -\
    4*(N1**2*(N1-1))**(-1)*EE@Ky_@Kxy.T@EE + 4*(N1*N1*(N1-1)*(N1-2))**(-1)*(EE@Ky_@EE*EE@Kxy.T@EE-2*EE@Ky_@Kxy.T@EE)

    varEst = (4*(N1-2)/(N1*(N1-1)) * varxi1 + 2/(N1*(N1-1)) * varxi2)
    Var_all = varEst/TEMP[2]**2
    return MMD_value, NAMMD_value, varEst, Var_all

def testing(X, Y, MMD, NAMMD, N1, rs, sigma0, n_test, alpha, device):
    H_MMD = np.zeros(n_test)
    H_NAMMD = np.zeros(n_test)
    P_MMD = np.zeros(n_test)
    P_NAMMD = np.zeros(n_test)
    set_random_seed(rs)
    sigma_mmd, sigma_nammd = split_selected_bandwidths(sigma0)

    threshold = norm.ppf(1-alpha)
    for k in range(n_test):
        indices_X = torch.randint(0, len(X), (N1,))
        X_test = X[indices_X]
        indices_Y = torch.randint(0, len(Y), (N1,))
        Y_test = Y[indices_Y]

        Fea = torch.cat((X_test, Y_test))
        MMD_value, _, varEst, _ = _asymptotic_values(Fea, N1, sigma_mmd, device)
        _, NAMMD_value, _, Var_all = _asymptotic_values(Fea, N1, sigma_nammd, device)
        NAMMD_test = (NAMMD_value.item()-NAMMD) / torch.sqrt(Var_all).item()
        MMD_Test = (MMD_value.item()-MMD) / torch.sqrt(varEst).item()
        H_NAMMD[k] = int(NAMMD_test>threshold)
        H_MMD[k] = int(MMD_Test>threshold)
        P_MMD[k] = norm.sf(MMD_Test.item())
        P_NAMMD[k] = norm.sf(NAMMD_test.item())
    return H_MMD, H_NAMMD, P_MMD, P_NAMMD

def testing_per(X, Y, N_per, N1, rs, sigma0, n_test, alpha, device):
    H_MMD = np.zeros(n_test)
    H_NAMMD = np.zeros(n_test)
    P_MMD = np.zeros(n_test)
    P_NAMMD = np.zeros(n_test)
    set_random_seed(rs)
    sigma_mmd, sigma_nammd = split_selected_bandwidths(sigma0)

    for k in range(n_test):
        indices_X = torch.randint(0, len(X), (N1,))
        X_test = X[indices_X]
        indices_Y = torch.randint(0, len(Y), (N1,))
        Y_test = Y[indices_Y]

        Fea = torch.cat((X_test, Y_test))

        H_NAMMD[k], P_NAMMD[k] = NAMMD_TEST(Fea, N_per, N1, sigma_nammd, alpha, device)
        H_MMD[k], P_MMD[k] = MMD_TEST(Fea, N_per, N1, sigma_mmd, alpha, device)
    return H_MMD, H_NAMMD, P_MMD, P_NAMMD
