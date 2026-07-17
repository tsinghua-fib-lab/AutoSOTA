import torch
import numpy as np 
import sys
import os
sys.path.append(os.path.abspath('../..'))
from dataloader import load_data
from scipy.stats import norm
from kernel_selection import (
    select_gaussian_bandwidths_reference,
    set_random_seed,
    split_selected_bandwidths,
)

def NAMMD_discrete(X, Y, N1, sigma0, K = 1):
    def h1_mean_var_gram(Kx, Ky, Kxy, K = 1):
        """compute value of VMD and std of VMD using kernel matrix."""
        nx = Kx.shape[0]
        ny = Ky.shape[0]
        
        xx = torch.div(torch.sum(Kx), (nx * nx))
        yy = torch.div(torch.sum(Ky), (ny * ny ))
        xy = torch.div(torch.sum(Kxy), (nx * ny))

        MMD = xx - 2 * xy + yy
        Reg = 4 * K - xx - yy
        
        return MMD, Reg
    
    def Pdist2(x, y):
        # """compute the paired distance between x and y."""
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)

        Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
        Pdist[Pdist<0]=0
        return Pdist

    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)
    Kx = torch.exp(-Dxx / sigma0**2)
    Ky = torch.exp(-Dyy / sigma0**2)
    Kxy = torch.exp(-Dxy / sigma0**2)

    return h1_mean_var_gram(Kx, Ky, Kxy, K)

def construct_distributions(name, N, rs, eps, eps_gap, learning_rate, sigma0, K, device, dtype, scale=True):
    sigma_mmd, sigma_nammd = split_selected_bandwidths(sigma0)
    if eps == 0:
        X, _ = load_data(name, N, rs, 0, scale)
        X = MatConvert(X, device, dtype)
        Y = X
        MMD_mmd, _ = NAMMD_discrete(X, Y, N, sigma_mmd, K)
        MMD_nammd, Reg = NAMMD_discrete(X, Y, N, sigma_nammd, K)
        NAMMD = MMD_nammd / Reg
        return X, Y, MMD_mmd.item(), NAMMD.item(), X, Y, MMD_mmd.item(), NAMMD.item()
    else:
        ts = 100
        while True:
            X, Y = load_data(name, N, rs+ts, 1, scale)

            set_random_seed(rs + ts)

            X = MatConvert(X, device, dtype)
            Y = MatConvert(Y, device, dtype)
            
            X.requires_grad = True
            Y.requires_grad = True
            optimizer = torch.optim.Adam([X,Y], lr=learning_rate)
            MMD, Reg = NAMMD_discrete(X, Y, N, sigma_nammd, K)

            t = 0
            while abs((MMD/Reg).item() - eps) >= 10**(-7):
                    STAT_u = (MMD/Reg - eps)**2
                    # Initialize optimizer and Compute gradient
                    optimizer.zero_grad()
                    STAT_u.backward(retain_graph=True)
                    # Update weights using gradient descent
                    optimizer.step()

                    if t % 100 == 0:
                        print("MMD_value: ", MMD.item(), "Reg_value: ", Reg.item())
                    MMD, Reg = NAMMD_discrete(X, Y, N, sigma_nammd, K)
                    t += 1
            X1 = X.detach()
            Y1 = Y.detach()
            MMD1_mmd = NAMMD_discrete(X1, Y1, N, sigma_mmd, K)[0].item()
            MMD1_nammd, Reg1_nammd = NAMMD_discrete(X1, Y1, N, sigma_nammd, K)
            NAMMD1 = (MMD1_nammd / Reg1_nammd).item()

            t = 0
            while abs((MMD/Reg).item() - eps - eps_gap) >= 10**(-7):
                    STAT_u = (MMD/Reg - eps - eps_gap)**2
                    # Initialize optimizer and Compute gradient
                    optimizer.zero_grad()
                    STAT_u.backward(retain_graph=True)
                    # Update weights using gradient descent
                    optimizer.step()

                    if t % 100 == 0:
                        print("MMD_value: ", MMD.item(), "Reg_value: ", Reg.item())
                    MMD, Reg = NAMMD_discrete(X, Y, N, sigma_nammd, K)
                    t += 1
            X2 = X.detach()
            Y2 = Y.detach()
            MMD2_mmd = NAMMD_discrete(X2, Y2, N, sigma_mmd, K)[0].item()
            MMD2_nammd, Reg2_nammd = NAMMD_discrete(X2, Y2, N, sigma_nammd, K)
            NAMMD2 = (MMD2_nammd / Reg2_nammd).item()
            if Reg1_nammd.item() > Reg2_nammd.item() or eps_gap==0:
                break
            ts +=1
    return X1, Y1, MMD1_mmd, NAMMD1, X2, Y2, MMD2_mmd, NAMMD2

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
    return mmd2, Kxyxy, 4- xx - yy

def MMDu(Fea, len_s, sigma0):
    """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
    X = Fea[0:len_s, :] # fetch the sample 1 (features of deep networks)
    Y = Fea[len_s:, :] # fetch the sample 2 (features of deep networks)
    Dxx = Pdist(X, X)
    Dyy = Pdist(Y, Y)
    Dxy = Pdist(X, Y)
    Kx = torch.exp(-Dxx / sigma0**2)
    Ky = torch.exp(-Dyy / sigma0**2)
    Kxy = torch.exp(-Dxy / sigma0**2)
    return h1_mean_var_gram(Kx, Ky, Kxy)

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

def training(name, N1, rs, check, ne_MMD, bs_MMD, lr_MMD, ne_NAMMD, bs_NAMMD, lr_NAMMD, b_NAMMD, device, dtype):
    set_random_seed(rs)
    X_train, Y_train = load_data(name, N1, rs, check)
    S_train = np.concatenate((X_train, Y_train), axis=0)
    S_train = MatConvert(S_train, device, dtype)
    sigma_mmd = MMD_fit(S_train, N1, lr_MMD, ne_MMD, bs_MMD, device, seed=rs)
    sigma_nammd = NAMMD_fit(S_train, N1, sigma_mmd, lr_NAMMD, ne_NAMMD, bs_NAMMD, b_NAMMD, device, seed=rs)
    print("Selected bandwidths: MMD = {:.6g}, NAMMD = {:.6g}".format(sigma_mmd.item(), sigma_nammd.item()))
    return sigma_mmd.detach(), sigma_nammd.detach()
    

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


def testing(X, Y, MMD, NAMMD, N1, rs, sigma0, n_test, n_per, alpha, device, dtype):
    H_MMD = np.zeros(n_test)
    H_NAMMD = np.zeros(n_test)
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
        NAMMD_test = (NAMMD_value-NAMMD) / torch.sqrt(Var_all)
        MMD_Test = (MMD_value-MMD) / torch.sqrt(varEst)
        H_NAMMD[k] = int(NAMMD_test>threshold)
        H_MMD[k] = int(MMD_Test>threshold)

    return H_MMD, H_NAMMD
