import numpy as np
import torch
from sklearn.utils import check_random_state
from sklearn.preprocessing import MinMaxScaler
import pickle
import torchvision.transforms as transforms
from torchvision import datasets
import sys
import os
def MMscaler(X, Y):
    scaler = MinMaxScaler()
    scaler.fit(np.concatenate((X, Y), axis=0))
    return scaler.transform(X), scaler.transform(Y)

def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x

def sample_BLOB(N, rs, check, scale):
    """#Feat. 2  # Inst. inf"""
    rs = check_random_state(rs)
    rows = 3
    cols = 3
    if check == 0:
        """Generate Blob-S for testing type-I error"""
        sep = 1
        correlation = 0
        # generate within-blob variation
        mu = np.zeros(2)
        sigma = np.eye(2)
        X = rs.multivariate_normal(mu, sigma, size=N)
        corr_sigma = np.array([[1, correlation], [correlation, 1]])
        Y = rs.multivariate_normal(mu, corr_sigma, size=N)
        # assign to blobs
        X[:, 0] += rs.randint(rows, size=N) * sep
        X[:, 1] += rs.randint(cols, size=N) * sep
        Y[:, 0] += rs.randint(rows, size=N) * sep
        Y[:, 1] += rs.randint(cols, size=N) * sep
    else:
        """Generate Blob-D for testing type-II error (or test power)"""
        sigma_mx_2_standard = np.array([[0.03, 0], [0, 0.03]])
        sigma_mx_2 = np.zeros([9, 2, 2])
        for i in range(9):
            sigma_mx_2[i] = sigma_mx_2_standard
            if i < 4:
                sigma_mx_2[i][0, 1] = -0.02 - 0.002 * i
                sigma_mx_2[i][1, 0] = -0.02 - 0.002 * i
            if i == 4:
                sigma_mx_2[i][0, 1] = 0.00
                sigma_mx_2[i][1, 0] = 0.00
            if i > 4:
                sigma_mx_2[i][1, 0] = 0.02 + 0.002 * (i - 5)
                sigma_mx_2[i][0, 1] = 0.02 + 0.002 * (i - 5)

        mu = np.zeros(2)
        sigma = np.eye(2) * 0.03
        X = rs.multivariate_normal(mu, sigma, size=N)
        Y = rs.multivariate_normal(mu, np.eye(2), size=N)
        # assign to blobs
        X[:, 0] += rs.randint(rows, size=N)
        X[:, 1] += rs.randint(cols, size=N)
        Y_row = rs.randint(rows, size=N)
        Y_col = rs.randint(cols, size=N)
        locs = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
        for i in range(9):
            corr_sigma = sigma_mx_2[i]
            L = np.linalg.cholesky(corr_sigma)
            ind = np.expand_dims((Y_row == locs[i][0]) & (Y_col == locs[i][1]), 1)
            ind2 = np.concatenate((ind, ind), 1)
            Y = np.where(ind2, np.matmul(Y, L) + locs[i], Y)
    if scale:
        X, Y = MMscaler(X, Y)
    return X, Y

def sample_HDGM(N, rs, check, scale):
    """#Feat. 10  # Inst. inf"""
    d = 10 # data dim
    Num_clusters = 2  # number of modes
    n = int(N / Num_clusters)
    mu_mx = np.zeros([Num_clusters, d])
    mu_mx[1] = mu_mx[1] + 0.5
    sigma_mx_1 = np.identity(d)
    X = np.zeros([n * Num_clusters, d])
    Y = np.zeros([n * Num_clusters, d])
    # Generate HDGM-D
    for i in range(Num_clusters):
        np.random.seed(seed=rs + i + 283)
        X[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
    for i in range(Num_clusters):
        np.random.seed(seed=rs + i)
        if check == 0:
            Y[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
        else:
            sigma_mx_2 = [np.identity(d), np.identity(d)]
            sigma_mx_2[0][0, 1] = 0.5
            sigma_mx_2[0][1, 0] = 0.5
            sigma_mx_2[1][0, 1] = -0.5
            sigma_mx_2[1][1, 0] = -0.5
            Y[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_2[i], n)
    if scale:
        X, Y = MMscaler(X, Y)
    return X, Y

def sample_HIGGS(N, rs, check, scale):
    """#Feat. 4  #Class 2  #Inst. [5170877,5829123]"""
    np.random.seed(seed=rs)
    try:
        data = pickle.load(open('../../data/HIGGS_TST.pckl', 'rb'))
    except Exception as e:
        data = pickle.load(open('../data/HIGGS_TST.pckl', 'rb'))
    if check == 0:
        dataX = data[0]
        dataY = data[0]
    else:
        dataX = data[0]
        dataY = data[1]
    del data

    N1_T = dataX.shape[0]
    N2_T = dataY.shape[0]
    ind1 = np.random.choice(N1_T, N, replace=False)
    ind2 = np.random.choice(N2_T, N, replace=False)
    X = dataX[ind1, :4]
    Y = dataY[ind2, :4]
    if scale:
        X, Y = MMscaler(X, Y)
    return X, Y

def sample_MNIST(N, rs, check, scale):
    """#Feat. 28*28  #Class 2  #Inst. [10000,10000]"""
    np.random.seed(seed=rs)

    # True_MNIST
    img_size = 32
    try:
        dataloader_FULL_te = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=False,
            download=False,
            transform=transforms.Compose(
                [transforms.Resize(img_size),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5], [0.5]),
                 ]
            ),
        ),
        batch_size=10000,
        shuffle=True,
    )
    except Exception as e:
        dataloader_FULL_te = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data/mnist",
            train=False,
            download=False,
            transform=transforms.Compose(
                [transforms.Resize(img_size),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5], [0.5]),
                 ]
            ),
        ),
        batch_size=10000,
        shuffle=True,
    )

    for i, (imgs, Labels) in enumerate(dataloader_FULL_te):
        dataX = np.array(imgs.view(len(imgs), -1))

    # Fake_MNIST
    try:
        Fake_MNIST = pickle.load(open('../../data/Fake_MNIST_data_EP100_N10000.pckl', 'rb'))
    except Exception as e:
        Fake_MNIST = pickle.load(open('../data/Fake_MNIST_data_EP100_N10000.pckl', 'rb'))
    dataY = torch.from_numpy(Fake_MNIST[0][:])
    dataY = np.array(dataY.view(len(dataY), -1))
    if check == 0:
        N1_T = dataX.shape[0]
        ind1 = np.random.choice(N1_T, N, replace=False)
        ind2 = np.random.choice(N1_T, N, replace=False)
        X = dataX[ind1, :]
        Y = dataX[ind2, :]
    else:
        N1_T = dataX.shape[0]
        N2_T = dataY.shape[0]
        ind1 = np.random.choice(N1_T, N, replace=False)
        ind2 = np.random.choice(N2_T, N, replace=False)
        X = dataX[ind1, :]
        Y = dataY[ind2, :]

    if scale:
        X, Y = MMscaler(X, Y)
        
    return X, Y

def sample_CIFAR10(N, rs, check, scale):
    """#Feat. 64*64  #Class 2  #Inst. [10000,2021]"""
    np.random.seed(seed=rs)

    img_size = 32
    try:
        dataset_test = datasets.CIFAR10(root='../../data/cifar10', download=False, train=False,
                                        transform=transforms.Compose([
                                            transforms.Resize(img_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            # transforms.Grayscale(),
                                        ]))
    except Exception as e:
        dataset_test = datasets.CIFAR10(root='../data/cifar10', download=False, train=False,
                                        transform=transforms.Compose([
                                            transforms.Resize(img_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            # transforms.Grayscale(),
                                        ]))

    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=10000,
                                                  shuffle=True)
    # Obtain CIFAR10 images
    for i, (imgs, Labels) in enumerate(dataloader_test):
        data_all = np.array(imgs.view(len(imgs), -1))

    try:
        data_new = np.load('../../data/cifar10_X_adversarial.npy')
    except Exception as e:
        data_new = np.load('../data/cifar10_X_adversarial.npy')
    data_T = data_new.reshape((-1, 3, img_size, img_size))
    ind_M = np.random.choice(len(data_T), len(data_T), replace=False)
    data_T = data_T[ind_M]
    TT = transforms.Compose([transforms.Resize(img_size),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             transforms.Grayscale(),
                             ])
    trans = transforms.ToPILImage()
    data_trans = torch.zeros([len(data_T), 3, img_size, img_size])
    data_T_tensor = torch.from_numpy(data_T)
    for i in range(len(data_T)):
        d0 = trans(data_T_tensor[i])
        data_trans[i] = TT(d0)
    data_trans = np.array(data_trans.view(len(data_trans), -1))

    if check == 0:
        Ind = np.random.choice(len(data_all), N, replace=False)
        X = data_all[Ind]
        Ind_v4 = np.random.choice(len(data_all), N, replace=False)
        Y = data_all[Ind_v4]
    else: 
        Ind = np.random.choice(len(data_all), N, replace=False)
        X = data_all[Ind]
        Ind_v4 = np.random.choice(len(data_trans), N, replace=False)
        Y = data_trans[Ind_v4]

    if scale:
        X, Y = MMscaler(X, Y)

    #"""transform to tensor"""
    # X = torch.from_numpy(X)
    # X = X.resize(len(X),3,img_size,img_size)

    return X, Y

def load_data(name, N, rs, check=1, scale=True):
    if name == 'BLOB':
        X, Y = sample_BLOB(N, rs, check, scale)
    elif name == 'HDGM':
        X, Y = sample_HDGM(N, rs, check, scale)
    elif name == 'HIGGS':
        X, Y = sample_HIGGS(N, rs, check, scale)
    elif name == 'MNIST':
        X, Y = sample_MNIST(N, rs, check, scale)   
    elif name == 'CIFAR10':
        X, Y = sample_CIFAR10(N, rs, check, scale)
    else:
        print('No Dataset: ', name)

    return X, Y

def NAMMD_discrete(X, Y, N1, sigma0, K = 1):
    def h1_mean_var_gram(Kx, Ky, Kxy, K = 1):
        """compute value of VMD and std of VMD using kernel matrix."""
        nx = Kx.shape[0]
        ny = Ky.shape[0]
        
        xx = torch.div(torch.sum(Kx), (nx * nx))
        yy = torch.div(torch.sum(Ky), (ny * ny))
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

def construct_distributions(name, N, rs, eps, learning_rate, sigma0, K, device, dtype, scale=True):
    if eps == 0:
        X, _ = load_data(name, N, rs, 0, scale)
        X = MatConvert(X, device, dtype)
        Y = X
        MMD, Reg = NAMMD_discrete(X, Y, N, sigma0, K)
    else:
        X, Y = load_data(name, N, rs, 1, scale)

        np.random.seed(seed=1102)
        torch.manual_seed(1102)
        torch.cuda.manual_seed(1102)

        X = MatConvert(X, device, dtype)
        Y = MatConvert(Y, device, dtype)
        
        X.requires_grad = True
        Y.requires_grad = True
        optimizer = torch.optim.Adam([X,Y], lr=learning_rate)

        MMD, Reg = NAMMD_discrete(X, Y, N, sigma0, K)
        t = 0

        while abs((MMD/Reg).item() - eps) >= 10**(-7):
                MMD, Reg = NAMMD_discrete(X, Y, N, sigma0, K)
                STAT_u = (MMD/Reg - eps)**2
                # Initialize optimizer and Compute gradient
                optimizer.zero_grad()
                STAT_u.backward(retain_graph=True)
                # Update weights using gradient descent
                optimizer.step()

                if t % 100 == 0:
                    print("MMD_value: ", MMD.item(), "Reg_value: ", Reg.item())
                t += 1
    return X.detach(), Y.detach(), MMD.item(), Reg.item()

def construct_distributions_norm(name, N, rs, eps, learning_rate, sigma0, K, device, dtype, scale=True):
    def Pdist2(x, y):
        """compute the paired distance between x and y."""
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)

        Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
        Pdist[Pdist<0]=0
        return Pdist

    X, _ = load_data(name, N, rs, 0, scale)
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    X = MatConvert(X, device, dtype)

    X.requires_grad = True
    optimizer = torch.optim.Adam([X], lr=learning_rate)

    Norm = None
    t = 0
    while Norm == None or abs(Norm - eps) >= 10**(-5):
        Dxx = Pdist2(X, X)
        Kx = torch.exp(-Dxx / sigma0**2)
        Norm = torch.div(torch.sum(Kx), (N * N))
        STAT = (Norm - eps)**2
        optimizer.zero_grad()
        STAT.backward(retain_graph=True)
        # Update weights using gradient descent
        optimizer.step()

        if t % 100 == 0:
            print("Norm: ", Norm.item())
        t += 1
    return X.detach(), Norm.detach()

def NAMMD_discrete_P(Z, P1, P2, N, sigma0, K = 1):
    def h1_mean_var_gram(Kzz, P11, P12, P22, K = 1):
        """compute value of VMD and std of VMD using kernel matrix."""
        xx = torch.sum(Kzz * P11)
        yy = torch.sum(Kzz * P22)
        xy = torch.sum(Kzz * P12)

        MMD = xx - 2 * xy + yy
        Reg = 4 * K - xx - yy
        
        return MMD, Reg
    
    def Pdist2(x, y):
        """compute the paired distance between x and y."""
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)

        Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
        Pdist[Pdist<0]=0
        return Pdist
    
    def Prob(P1, P2, N):
        """compute the paired distance between x and y."""
        P1 = P1.reshape(N,1)
        P2 = P2.reshape(N,1)
        
        return P1 @ P2.t()

    Dzz = Pdist2(Z, Z)
    Kzz = torch.exp(-Dzz / sigma0**2)
    
    P11 = Prob(P1,P1,N)
    P12 = Prob(P1,P2,N)
    P22 = Prob(P2,P2,N)

    return h1_mean_var_gram(Kzz, P11, P12, P22, K)

def construct_distributions_tv(name, N, rs, delt1, delt2, sigma0, K, device, dtype, way="uni", num = 2, scale=True):
    np.random.seed(seed=rs)
    torch.manual_seed(rs)
    torch.cuda.manual_seed(rs)
    
    Z = None
    P1 = None
    P2 = None
    MMD1 = None
    Reg1 = None
    MMD2 = None
    Reg2 = None
    if way == "uni":
        tt = 0
        while True:
            Z, _ = load_data(name, N, rs + tt, 0, scale)
            P_uniform = np.ones(N) / N
            P1 = np.ones(N) / N
            P2 = np.ones(N) / N
            ind = np.random.choice(N, N, replace=False)
            # divide into new X, Y
            indx = ind[:N//2]
            indy = ind[N//2:]
            
            P1[indx] += delt1/N
            P1[indy] -= delt1/N
            
            P2[indx] += delt2/N
            P2[indy] -= delt2/N
            
            P_uniform = MatConvert(P_uniform, device, dtype)
            P1 = MatConvert(P1, device, dtype)
            P2 = MatConvert(P2, device, dtype)
            Z = MatConvert(Z, device, dtype)
            
            if delt1 == delt2:
                break
            
            MMD1, Reg1 = NAMMD_discrete_P(Z, P_uniform, P1, N, sigma0, K)
            MMD2, Reg2 = NAMMD_discrete_P(Z, P_uniform, P2, N, sigma0, K)

            if MMD1/Reg1 < MMD2/Reg2:
                break
            tt += 1
    else:
        tt = 0
        while True:
            Z, _ = load_data(name, N, rs+tt, 0, scale)
            P_uniform = np.ones(N) / N
            P1 = np.ones(N) / N
            P2 = np.ones(N) / N

            distances = np.sum(abs(Z-Z[0])**2,axis=1)
            sorted_indices = np.argsort(distances)
            
            delt = delt1
            idx = N//2
            while delt > 0:
                idxx = sorted_indices[idx]
                change = min(P1[idxx], delt/2)
                P1[idxx] = P1[idxx] - change
                delt -= change * 2
                if P1[idxx] == 0:
                    if idx >= N//2:
                        idx = N//2 - (idx-N//2) - 1
                    else:
                        idx = N//2 + (N//2-idx)
                for i in range(num):
                    P1[sorted_indices[i]] += change/(2 * num)
                    P1[sorted_indices[-i-1]] += change/(2 * num)
                    
            delt = delt2
            idx = N//2
            while delt > 0:
                idxx = sorted_indices[idx]
                change = min(P2[idxx], delt/2)
                P2[idxx] = P2[idxx] - change
                delt -= change * 2
                if P2[idxx] == 0:
                    if idx >= N//2:
                        idx = N//2 - (idx-N//2) - 1
                    else:
                        idx = N//2 + (N//2-idx)
                for i in range(num):
                    P2[sorted_indices[i]] += change/(2 * num)
                    P2[sorted_indices[-i-1]] += change/(2 * num)
                    
            P_uniform = MatConvert(P_uniform, device, dtype)
            P1 = MatConvert(P1, device, dtype)
            P2 = MatConvert(P2, device, dtype)
            Z = MatConvert(Z, device, dtype)
            
            MMD1, Reg1 = NAMMD_discrete_P(Z, P_uniform, P1, N, sigma0, K)
            MMD2, Reg2 = NAMMD_discrete_P(Z, P_uniform, P2, N, sigma0, K)

            if delt1 == delt2:
                break
            if MMD1 < MMD2 and MMD1/Reg1 < MMD2/Reg2:
                break
            tt += 1
    try:
        return Z.detach(), P1, P2, MMD1.item(), Reg1.item(), MMD2.item(), Reg2.item()
    except Exception as e:
        return  Z.detach(), P1, P2,None, None,None, None