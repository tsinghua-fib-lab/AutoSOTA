import random
import time
from utils import *
import numpy as np
import torch
from libs import OnesidedStreamSW,SW
for _ in range(1000):
    torch.randn(100)
    np.random.rand(100)
A = np.load("reconstruct_random_50_shapenetcore55.npy")
ind1=30
ind2=31
target=A[ind2]
source=A[ind1]
device='cuda'
learning_rate = 0.001
N_step=5000

print_steps = [0]+list(np.arange(100,N_step+1,100)-1)
Y = torch.from_numpy(target).to(device)
N=target.shape[0]


ks=[100,200]
seeds=[1,2,3]
Ls=[100,1000]

for L in Ls:
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        Z = (1 - (2 * torch.arange(1, L + 1) - 1) / L).view(-1, 1)
        theta1 = torch.arccos(Z)
        theta2 = torch.remainder(1.8 * np.sqrt(L) * theta1, 2 * np.pi)
        theta = torch.cat(
            [torch.sin(theta1) * torch.cos(theta2), torch.sin(theta1) * torch.sin(theta2), torch.cos(theta1)],
            dim=1)
        theta = theta.to(device)

        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        X=torch.tensor(source, requires_grad=True,device=device)
        optimizer = torch.optim.SGD([X], lr=learning_rate)
        points=[]
        caltimes=[]
        distances=[]
        start = time.time()
        for i in range(N_step):
            if (i in print_steps):
                distance,cal_time=compute_true_Wasserstein(X, Y), time.time() - start
                print("Full SW {}:{} ({}s)".format(i + 1, distance,np.round(cal_time,2)))
                points.append(X.cpu().detach().clone().data.numpy())
                caltimes.append(cal_time)
                distances.append(distance)

            optimizer.zero_grad()

            sw= SW(X,Y,torch.ones(X.shape[0],device=device)/X.shape[0],torch.ones(Y.shape[0],device=device)/Y.shape[0],L=L,thetas=theta)
            loss= N*sw
            loss.backward()
            optimizer.step()
        points.append(Y.cpu().detach().clone().data.numpy())
        np.save("saved/SW_full_L{}_{}_{}_points_seed{}.npy".format(L,ind1,ind2,seed),np.stack(points))
        np.savetxt("saved/SW_full_L{}_{}_{}_distances_seed{}.txt".format(L,ind1,ind2,seed), np.array(distances), delimiter=",")
        np.savetxt("saved/SW_full_L{}_{}_{}_times_seed{}.txt".format(L,ind1,ind2,seed), np.array(caltimes), delimiter=",")

        for k in ks:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            inds= np.random.choice(N, int(3*k+2*np.log2(N/(2/3*k)))+1)
            Y_subset = Y[inds]
            X=torch.tensor(source, requires_grad=True,device=device)
            optimizer = torch.optim.SGD([X], lr=learning_rate)
            points=[]
            caltimes=[]
            distances=[]
            start = time.time()
            for i in range(N_step):
                if (i in print_steps):
                    distance,cal_time=compute_true_Wasserstein(X, Y), time.time() - start
                    print("SW (Random Sampling) {}:{} ({}s)".format(i + 1, distance,np.round(cal_time,2)))
                    points.append(X.cpu().detach().clone().data.numpy())
                    caltimes.append(cal_time)
                    distances.append(distance)

                optimizer.zero_grad()

                sw= SW(X,Y_subset,torch.ones(X.shape[0],device=device)/X.shape[0],torch.ones(Y_subset.shape[0],device=device)/Y_subset.shape[0],L=L,thetas=theta)
                loss= N*sw
                loss.backward()
                optimizer.step()
            points.append(Y.cpu().detach().clone().data.numpy())
            np.save("saved/SW_L{}_k{}_{}_{}_points_seed{}.npy".format(L,k,ind1,ind2,seed),np.stack(points))
            np.savetxt("saved/SW_L{}_k{}_{}_{}_distances_seed{}.txt".format(L,k,ind1,ind2,seed), np.array(distances), delimiter=",")
            np.savetxt("saved/SW_L{}_k{}_{}_{}_times_seed{}.txt".format(L,k,ind1,ind2,seed), np.array(caltimes), delimiter=",")



            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            streamsw= OnesidedStreamSW(L=L,d=3,k=k,p=2, thetas=theta.detach().cpu().numpy())
            streamsw.update(Y.cpu().detach().numpy())
            X=torch.tensor(source, requires_grad=True,device=device)
            optimizer = torch.optim.SGD([X], lr=learning_rate)
            points=[]
            caltimes=[]
            distances=[]
            start = time.time()
            for i in range(N_step):
                if (i in print_steps):
                    distance,cal_time=compute_true_Wasserstein(X, Y), time.time() - start
                    print("Stream SW {}:{} ({}s)".format(i + 1, distance,np.round(cal_time,2)))
                    points.append(X.cpu().detach().clone().data.numpy())
                    caltimes.append(cal_time)
                    distances.append(distance)

                optimizer.zero_grad()
                a=np.ones(X.shape[0]) / X.shape[0]
                sw= streamsw.compute_distance_torch(X,a)
                loss= N*sw
                loss.backward()
                optimizer.step()
            points.append(Y.cpu().detach().clone().data.numpy())
            np.save("saved/StreamSW_L{}_k{}_{}_{}_points_seed{}.npy".format(L,k,ind1,ind2,seed),np.stack(points))
            np.savetxt("saved/StreamSW_L{}_k{}_{}_{}_distances_seed{}.txt".format(L,k,ind1,ind2,seed), np.array(distances), delimiter=",")
            np.savetxt("saved/StreamSW_L{}_k{}_{}_{}_times_seed{}.txt".format(L,k,ind1,ind2,seed), np.array(caltimes), delimiter=",")