import numpy as np
import torch
from libs import SW, StreamSW
import tqdm
np.random.seed(1)
torch.manual_seed(1)

n=10000
d = 2
ns = [500,2000,5000,10000,20000,50000]
times=10
L=1000
k=100
device='cuda'
for n in ns:
    print('N {}'.format(n))
    a = torch.ones(n,device=device) / n
    b = torch.ones(n,device=device) / n
    MAEs=[]
    num_floats=[]
    trues=[]
    SWks=[]
    for _ in range(times):
        X = torch.randn(n,d,device=device)-1
        Y = torch.randn(n,d,device=device)+2
        thetas = torch.randn(L,d,device=device)
        thetas = thetas.data/torch.sqrt(torch.sum(thetas.data**2,dim=1,keepdim=True))
        true_SW = SW(X,Y,a,b,L=L,thetas=thetas)
        c = torch.ones(int(3 * k + 2 * np.log2(n / (2 / 3 * k))) + 1, device=device) / (
                    int(3 * k + 2 * np.log2(n / (2 / 3 * k))) + 1)
        if(int(3 * k + np.log2(n)) + 1<n):
            SWk = SW(X[np.random.choice(n, int(3*k+2*np.log2(n/(2/3*k)))+1)],
                     Y[np.random.choice(n, int(3*k+2*np.log2(n/(2/3*k)))+1)], c, c, L=L, thetas=thetas)
        else:
            SWk=true_SW
        stream_sw = StreamSW(L=L,d=d,k=k,p=2, thetas=thetas.detach().cpu().numpy())
        for i in tqdm.tqdm(range(n)):
            stream_sw.update_mu(X[i].detach().cpu().numpy().reshape(1,-1))
            stream_sw.update_nu(Y[i].detach().cpu().numpy().reshape(1,-1))
        num_mu,num_nu = stream_sw.sizes()
        value= stream_sw.compute_distance()
        MAEs.append(torch.abs(value-true_SW.cpu()).detach().cpu().numpy())
        trues.append(true_SW.cpu().detach().cpu().numpy())
        SWks.append(SWk.detach().cpu().numpy())
        num_floats.append(num_mu/L)
    np.savetxt('result_varying_ns/MAEs_n{}.csv'.format(n), np.array(MAEs).reshape(-1,), delimiter=',')
    np.savetxt('result_varying_ns/num_points_n{}.csv'.format(n), np.array(num_floats).reshape(-1, ), delimiter=',')
    np.savetxt('result_varying_ns/trueSW_n{}.csv'.format(n), np.array(trues).reshape(-1, ), delimiter=',')
    np.savetxt('result_varying_ns/SW_k_n{}.csv'.format(n), np.array(SWks).reshape(-1, ), delimiter=',')



