import torch
import torch.nn as nn
import numpy as np 
import sys
import os
sys.path.append(os.path.abspath('../..'))
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

def MMDu(Fea, Fea_org, len_s, sigma0, sigma, epsilon):
    """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
    X = Fea[:len_s, :] # fetch the sample 1 (features of deep networks)
    Y = Fea[len_s:, :] # fetch the sample 2 (features of deep networks)
    X_org = Fea_org[:len_s, :] # fetch the original sample 1
    Y_org = Fea_org[len_s:, :] # fetch the original sample 2

    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)
    Dxx_org = Pdist2(X_org, X_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)

    ep = torch.exp(epsilon)/(1+torch.exp(epsilon))
    
    Kx = ((1-ep) * torch.exp(-(Dxx / sigma0**2)) + ep) * torch.exp(-Dxx_org / sigma**2)
    Ky = ((1-ep) * torch.exp(-(Dyy / sigma0**2)) + ep) * torch.exp(-Dyy_org / sigma**2)
    Kxy = ((1-ep) * torch.exp(-(Dxy / sigma0**2)) + ep) * torch.exp(-Dxy_org / sigma**2)
    
    return h1_mean_var_gram(Kx, Ky, Kxy)

def training(name, N1, rs, check, sigma, sigma0, epsilon, H, x_out, ne_MMD, bs_MMD, lr_MMD, ne_NAMMD, bs_NAMMD, lr_NAMMD, b_NAMMD, device, dtype):
    np.random.seed(rs)
    X_train, Y_train = load_data(name, N1, rs, check)
    S_train = np.concatenate((X_train, Y_train), axis=0)
    S_train = MatConvert(S_train, device, dtype)
    sigma = MatConvert(sigma, device, dtype)
    sigma0 = MatConvert(sigma0, device, dtype)
    epsilon = MatConvert(epsilon, device, dtype)
    sigma, sigma0, epsilon, model = MMD_fit(name, S_train, N1, sigma, sigma0, epsilon, H, x_out, lr_MMD, ne_MMD, bs_MMD, device)
    sigma, sigma0, epsilon, model = NAMMD_fit(name, S_train, N1, sigma, sigma0, epsilon, model, lr_NAMMD, ne_NAMMD, bs_NAMMD, b_NAMMD, device)
    return sigma, sigma0, epsilon, model
    
def testing(name, N1, rs, check, sigma, sigma0, epsilon, model, n_test, n_per, alpha, device, dtype):
    H_MMD = np.zeros(n_test)
    H_NAMMD = np.zeros(n_test)
    
    N_test_all = 10 * N1
    X_test_all, Y_test_all = load_data(name, N_test_all, rs + 283, check)
    
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
        
        modelu_output = model(S_test)
        
        h_MMD, _, _ = MMD_TEST(modelu_output, S_test.view(S_test.shape[0],-1), n_per, N1, sigma, sigma0, epsilon, alpha, device)
        h_NAMMD, _, _ = NAMMD_TEST(modelu_output, S_test.view(S_test.shape[0],-1), n_per, N1, sigma, sigma0, epsilon, alpha, device)

        H_MMD[k] = h_MMD
        H_NAMMD[k] = h_NAMMD

    return H_MMD, H_NAMMD


###### MMD
def MMD_TEST(Fea, Fea_org, N_per, N1, sigma, sigma0, epsilon, alpha, device):
    mmd_vector = np.zeros(N_per)
    TEMP = MMDu(Fea, Fea_org, N1, sigma0, sigma, epsilon)
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

    return h, threshold, mmd_value.item()

def MMD_fit(name, S, N1, sigma, sigma0, epsilon, H, x_out, learning_rate, N_epoch, batch_size, device):
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    
    epsilon.requires_grad = True
    sigma.requires_grad = True
    sigma0.requires_grad = True
    
    if name == "MNIST":
        if device == torch.device("cpu"):
            model = Featurizer(1, 32, 100)
        else:
            model = Featurizer(1, 32, 100).cuda()
        S = S.resize(len(S), 1, 32, 32)
    elif name == "CIFAR10":
        if device == torch.device("cpu"):
            model = Featurizer(3, 32, 300)
        else:
            model = Featurizer(3, 32, 300).cuda()
        S = S.resize(len(S), 3, 32, 32)
    else:
        if device == torch.device("cpu"):
            model = ModelLatentF(S.shape[1], H, x_out)
        else:
            model = ModelLatentF(S.shape[1], H, x_out).cuda()
    # Setup optimizer for training deep kernel
    optimizer = torch.optim.Adam(list(model.parameters())+[epsilon]+[sigma]+[sigma0], lr=learning_rate)

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
        
            modelu_output1 = model(S_batch)
            TEMP =  MMDu(modelu_output1, S_batch.view(S_batch.shape[0],-1), int(S_batch.shape[0]/2), sigma0, sigma, epsilon)
            mmd_value_temp = -1 * TEMP[0]
            optimizer.zero_grad()
            mmd_value_temp.backward(retain_graph=True)
            optimizer.step()
            
        if (t+1) % 500 == 0 or t == 0:
            print("MMD: ", -1 * mmd_value_temp.item())
    
    return sigma, sigma0, epsilon, model

##### NAMMD
def NAMMD_TEST(Fea, Fea_org, N_per, N1, sigma, sigma0, epsilon, alpha, device):
    NAMMD_vector = np.zeros(N_per)
    TEMP = MMDu(Fea, Fea_org, N1, sigma0, sigma, epsilon)
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

    return h, threshold, NAMMD_value.item()

def NAMMD_fit(name, S, N1, sigma, sigma0, epsilon, model, learning_rate, N_epoch, batch_size, b, device):
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    
    if name == "MNIST":
        S = S.resize(len(S), 1, 32, 32)
    elif name == "CIFAR10":
        S = S.resize(len(S), 3, 32, 32)
    
    epsilon.requires_grad = True
    sigma.requires_grad = True
    sigma0.requires_grad = True
    
    # Setup optimizer for training deep kernel
    optimizer = torch.optim.Adam(list(model.parameters())+[epsilon]+[sigma]+[sigma0], lr=learning_rate)

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
            
            modelu_output1 = model(S_batch)
            TEMP =  MMDu(modelu_output1, S_batch.view(S_batch.shape[0],-1), int(S_batch.shape[0]/2), sigma0, sigma, epsilon)
            
            NAMMD_value_temp = -1 * TEMP[0] + b * TEMP[2]

            # Initialize optimizer and Compute gradient
            optimizer.zero_grad()
            NAMMD_value_temp.backward(retain_graph=True)
            # Update weights using gradient descent
            optimizer.step()
            # Print NAMMD, std of NAMMD and J
        if (t+1) % 500 == 0 or t == 0:
            print("NAMMD_value: ", -1 * NAMMD_value_temp.item())
    
    return sigma, sigma0, epsilon, model