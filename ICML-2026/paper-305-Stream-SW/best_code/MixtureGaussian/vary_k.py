import numpy as np
import torch
from libs import SW, StreamSW
from mg import sample_gmm
import tqdm
np.random.seed(1)
torch.manual_seed(1)

n=10000
d = 2
ks = [2,5,10, 20, 50, 100, 200, 500, 1000]
times=10
L=1000
device='cuda'
for k in ks:
    print('K {}'.format(k))
    a = torch.ones(n,device=device) / n
    b = torch.ones(n,device=device) / n
    MAEs=[]
    num_floats=[]
    trues=[]
    SWks = []
    for _ in range(times):
        means1 = torch.tensor([[0.0, 0.0], [3.0, 3.0], [-3.0, 3.0]]).to(device)
        covs1 = torch.tensor([[[0.5, 0.2], [0.2, 0.5]],
                              [[0.8, -0.3], [-0.3, 0.5]],
                              [[0.6, 0.1], [0.1, 0.6]]]).to(device)
        weights1 = torch.tensor([0.3, 0.4, 0.3]).to(device)

        # Define the second Gaussian mixture (3 components in 2D)                                       mn m m          ?mIOE0929611443
        means2 = torch.tensor([[-3.0, -3.0], [3.0, -3.0], [0.0, 3.0]]).to(device)
        covs2 = torch.tensor([[[0.7, 0.1], [0.1, 0.7]],
                              [[0.5, -0.2], [-0.2, 0.5]],
                              [[0.6, 0.2], [0.2, 0.6]]]).to(device)
        weights2 = torch.tensor([0.4, 0.3, 0.3]).to(device)
        X = sample_gmm(means1, covs1, weights1, num_samples=n)
        Y = sample_gmm(means2, covs2, weights2, num_samples=n)
        thetas = torch.randn(L,d,device=device)
        thetas = thetas.data/torch.sqrt(torch.sum(thetas.data**2,dim=1,keepdim=True))
        true_SW = SW(X,Y,a,b,L=L,thetas=thetas)
        c = torch.ones(int(3 * k + 2 * np.log2(n / (2 / 3 * k))) + 1, device=device) / (
                    int(3 * k + 2 * np.log2(n / (2 / 3 * k))) + 1)
        SWk = SW(X[np.random.choice(n, int(3*k+2*np.log2(n/(2/3*k)))+1)],
                 Y[np.random.choice(n, int(3*k+2*np.log2(n/(2/3*k)))+1)], c, c, L=L, thetas=thetas)
        SWks.append(SWk.detach().cpu().numpy())
        stream_sw = StreamSW(L=L,d=d,k=k,p=2, thetas=thetas.detach().cpu().numpy())
        for i in tqdm.tqdm(range(n)):
            stream_sw.update_mu(X[i].detach().cpu().numpy().reshape(1,-1))
            stream_sw.update_nu(Y[i].detach().cpu().numpy().reshape(1,-1))
        num_mu,num_nu = stream_sw.sizes()
        value= stream_sw.compute_distance()
        MAEs.append(torch.abs(value-true_SW.cpu()).detach().cpu().numpy())
        trues.append(true_SW.cpu().detach().cpu().numpy())
        num_floats.append(num_mu/L)
    np.savetxt('result_varying_ks/MAEs_k{}.csv'.format(k), np.array(MAEs).reshape(-1,), delimiter=',')
    np.savetxt('result_varying_ks/num_points_k{}.csv'.format(k), np.array(num_floats).reshape(-1, ), delimiter=',')
    np.savetxt('result_varying_ks/trueSW_k{}.csv'.format(k), np.array(trues).reshape(-1, ), delimiter=',')
    np.savetxt('result_varying_ks/SW_k{}.csv'.format(k), np.array(SWks).reshape(-1, ), delimiter=',')




