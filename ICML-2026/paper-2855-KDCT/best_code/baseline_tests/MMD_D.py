import torch
import numpy as np 
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath('..'))
import time
from dataloader import load_data
class ModelLatentF(torch.nn.Module):
    """define deep networks."""
    def __init__(self, x_in, H, x_out):
        """Init latent features."""
        super(ModelLatentF, self).__init__()
        self.restored = False

        self.latent = torch.nn.Sequential(
            torch.nn.Linear(x_in, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, x_out, bias=True),
        )
    def forward(self, input):
        """Forward the LeNet."""
        fealant = self.latent(input)
        return fealant

class Featurizer(nn.Module):
    def __init__(self,channels,img_size,n):
        super(Featurizer, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2), nn.Dropout2d(0)] #0.25
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, n))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        feature = self.adv_layer(out)

        return feature

def MatConvert(S, device, dtype):
    """convert the numpy to a torch tensor."""
    S = torch.from_numpy(S).to(device, dtype)
    return S

def get_item(x, device):
    """get the numpy value from a torch tensor."""
    if device == torch.device("cpu"):
        x = x.detach().numpy()
    else:
        x = x.cpu().detach().numpy()
    return x

def Pdist2(X, Y):
    """compute the paired distance between x and y."""
    X_norm = (X ** 2).sum(1).view(-1, 1)
    if Y is not None:
        Y_norm = (Y ** 2).sum(1).view(1, -1)
    else:
        Y = X
        Y_norm = X_norm.view(1, -1)
    Pdist = X_norm + Y_norm - 2.0 * torch.mm(X, torch.transpose(Y, 0, 1))
    Pdist[Pdist<0]=0
    return Pdist

def h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U=True):
    """compute value of MMD and std of MMD using kernel matrix."""
    Kxxy = torch.cat((Kx,Kxy),1)
    Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
    Kxyxy = torch.cat((Kxxy,Kyxy),0)
    nx = Kx.shape[0]
    ny = Ky.shape[0]
    is_unbiased = True
    if is_unbiased:
        xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
        yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (ny - 1)))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    else:
        xx = torch.div((torch.sum(Kx)), (nx * nx))
        yy = torch.div((torch.sum(Ky)), (ny * ny))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy)), (nx * ny))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    if not is_var_computed:
        return mmd2, None, Kxyxy
    hh = Kx+Ky-Kxy-Kxy.transpose(0,1)
    V1 = torch.dot(hh.sum(1)/ny,hh.sum(1)/ny) / ny
    V2 = (hh).sum() / (nx) / nx
    varEst = 4*(V1 - V2**2)
    if  varEst == 0.0:
        print('error_var!!'+str(V1))
    return mmd2, varEst, Kxyxy

def MMDu(Fea, len_s, Fea_org, sigma, sigma0=0.1, epsilon=10**(-10), is_smooth=True, is_var_computed=True, use_1sample_U=True):
    """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
    X = Fea[0:len_s, :] # fetch the sample 1 (features of deep networks)
    Y = Fea[len_s:, :] # fetch the sample 2 (features of deep networks)
    X_org = Fea_org[0:len_s, :] # fetch the original sample 1
    Y_org = Fea_org[len_s:, :] # fetch the original sample 2
    L = 1 # generalized Gaussian (if L>1)
    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)
    Dxx_org = Pdist2(X_org, X_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)
    if is_smooth:
        Kx = (1-epsilon) * torch.exp(-(Dxx / sigma0)**L -Dxx_org / sigma) + epsilon * torch.exp(-Dxx_org / sigma)
        Ky = (1-epsilon) * torch.exp(-(Dyy / sigma0)**L -Dyy_org / sigma) + epsilon * torch.exp(-Dyy_org / sigma)
        Kxy = (1-epsilon) * torch.exp(-(Dxy / sigma0)**L -Dxy_org / sigma) + epsilon * torch.exp(-Dxy_org / sigma)
    else:
        Kx = torch.exp(-Dxx / sigma0)
        Ky = torch.exp(-Dyy / sigma0)
        Kxy = torch.exp(-Dxy / sigma0)
    return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)

def mmd2_permutations(K, n_X, permutations):
    """
        Fast implementation of permutations using kernel matrix.
    """
    K = torch.as_tensor(K)
    n = K.shape[0]
    assert K.shape[0] == K.shape[1]
    n_Y = n_X
    assert n == n_X + n_Y
    w_X = 1
    w_Y = -1
    ws = torch.full((permutations + 1, n), w_Y, dtype=K.dtype, device=K.device)
    ws[-1, :n_X] = w_X
    for i in range(permutations):
        ws[i, torch.randperm(n)[:n_X].numpy()] = w_X
    biased_ests = torch.einsum("pi,ij,pj->p", ws, K, ws)
    if True:  # u-stat estimator
        # need to subtract \sum_i k(X_i, X_i) + k(Y_i, Y_i) + 2 k(X_i, Y_i)
        # first two are just trace, but last is harder:
        is_X = ws > 0
        X_inds = is_X.nonzero()[:, 1].view(permutations + 1, n_X)
        Y_inds = (~is_X).nonzero()[:, 1].view(permutations + 1, n_Y)
        del is_X, ws
        cross_terms = K.take(Y_inds * n + X_inds).sum(1)
        del X_inds, Y_inds
        ests = (biased_ests - K.trace() + 2 * cross_terms) / (n_X * (n_X - 1))
    est = ests[-1]
    rest = ests[:-1]
    p_val = (rest > est).float().mean()
    return est.item(), p_val.item(), rest

def MMD_D_TEST(Fea, N_per, N1, Fea_org, sigma, sigma0, ep, alpha, device, is_smooth=True):
    """run two-sample test (TST) using deep kernel kernel."""
    mmd_vector = np.zeros(N_per)
    TEMP = MMDu(Fea, N1, Fea_org, sigma, sigma0, ep, is_smooth)
    mmd_value = get_item(TEMP[0], device)
    Kxyxy = TEMP[2]
    count = 0
    nxy = Fea.shape[0]
    nx = N1

    for r in range(N_per):
        # print r
        ind = np.random.choice(nxy, nxy, replace=False)
        # divide into new X, Y
        indx = ind[:nx]
        # print(indx)
        indy = ind[nx:]
        Kx = Kxyxy[np.ix_(indx, indx)]
        # print(Kx)
        Ky = Kxyxy[np.ix_(indy, indy)]
        Kxy = Kxyxy[np.ix_(indx, indy)]

        TEMP = h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed=False)
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

    return h, threshold, mmd_value.item()

def MMD_D_fit(name, S, N1, learning_rate, x_in, H, x_out, N_epoch, batch_size, device, dtype):
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)

    epsilonOPT = MatConvert(np.random.rand(1) * (10 ** (-10)), device, dtype)
    epsilonOPT.requires_grad = True
    if name in ['HDGM', 'Poker', 'Dna', 'agnos', 'har','krkp','bank']:
        sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * x_in), device, dtype)
        sigmaOPT.requires_grad = True
        sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.1), device, dtype)
        sigma0OPT.requires_grad = True
        if device == torch.device("cpu"):
            model_u = ModelLatentF(x_in, H, x_out)
        else:
            model_u = ModelLatentF(x_in, H, x_out).cuda()
    elif name == "MNIST":
        sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * 32 * 32), device, dtype)
        sigmaOPT.requires_grad = True
        sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.005), device, dtype)
        sigma0OPT.requires_grad = True
        if device == torch.device("cpu"):
            model_u = Featurizer(1, 32, 100)
        else:
            model_u = Featurizer(1, 32, 100).cuda()
        S = S.resize(len(S), 1, 32, 32)
    elif name == "CIFAR10":
        sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * 32 * 32), device, dtype)
        sigmaOPT.requires_grad = True
        sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.005), device, dtype)
        sigma0OPT.requires_grad = True
        if device == torch.device("cpu"):
            model_u = Featurizer(3, 32, 300)
        else:
            model_u = Featurizer(3, 32, 300).cuda()
        S = S.resize(len(S), 3, 32, 32)
    else:
        sigmaOPT = MatConvert(np.sqrt(np.random.rand(1) * 0.3), device, dtype)
        sigmaOPT.requires_grad = True
        sigma0OPT = MatConvert(np.sqrt(np.random.rand(1) * 0.002), device, dtype)
        sigma0OPT.requires_grad = True
        if device == torch.device("cpu"):
            model_u = ModelLatentF(x_in, H, x_out)
        else:
            model_u = ModelLatentF(x_in, H, x_out).cuda()
    # Setup optimizer for training deep kernel
    optimizer_mmd = torch.optim.Adam(list(model_u.parameters())+[epsilonOPT]+[sigmaOPT]+[sigma0OPT], lr=learning_rate) #    
    torch.autograd.set_detect_anomaly(True)
    # optimize for MMD
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
        
            ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
            sigma = sigmaOPT ** 2
            sigma0 = sigma0OPT ** 2
            
            modelu_output1 = model_u(S_batch)

            TEMP = MMDu(modelu_output1, int(S_batch.shape[0]/2), S_batch.view(S_batch.shape[0],-1), sigma, sigma0, ep)
            mmd_value_temp = -1 * TEMP[0]
            optimizer_mmd.zero_grad()
            mmd_value_temp.backward(retain_graph=True)
            optimizer_mmd.step()
        if t % 500 == 0:
            print("MMD: ", -1 * mmd_value_temp.item())
    
    return sigma, sigma0, model_u, ep

def TST_MMD_D(name, N1, rs, check, n_test, n_per, alpha, device, dtype, x_in, H, x_out, N_epoch, batch_size, learning_rate):
    np.random.seed(rs)
    X_train, Y_train = load_data(name, N1, rs, check)

    S_train = np.concatenate((X_train, Y_train), axis=0)
    S_train = MatConvert(S_train, device, dtype)

    start_time = time.time()
    sigma, sigma0_u, model_u, ep = MMD_D_fit(name, S_train, N1, learning_rate, x_in, H, x_out, N_epoch, batch_size, device, dtype)
    train_time = time.time() - start_time
    
    H_MMD_D= np.zeros(n_test)
    N_test_all = 10 * N1
    X_test_all, Y_test_all = load_data(name, N_test_all, rs + 283, check)
    test_time = 0
    for k in range(n_test):
        ind_test = np.random.choice(N_test_all, N1, replace=False)
        X_test = X_test_all[ind_test]
        Y_test = Y_test_all[ind_test]

        S_test = np.concatenate((X_test, Y_test), axis=0)
        S_test = MatConvert(S_test, device, dtype)
        
        if name == "MNIST":
            S_test = S_test.resize(len(S_test), 1, 32, 32)
        elif name == "CIFAR10":
            S_test = S_test.resize(len(S_test), 3, 32, 32)
        
        start_time = time.time()
        h_mmd_d, _, _ = MMD_D_TEST(model_u(S_test), n_per, N1, S_test.view(S_test.shape[0],-1), sigma, sigma0_u, ep, alpha, device)
        test_time += time.time() - start_time
        H_MMD_D[k] = h_mmd_d

    return H_MMD_D, train_time, test_time
