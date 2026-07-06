import numpy as np
import matplotlib.pyplot as plt
import torch
from libs import SW, StreamSW
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
print('---------------------------')
input_method = input("Enter Testing Method: ")
input_user = input("User (1-4): ")
dtype = torch.float
device = torch.device("cpu")
data_bending = np.load('data_bending_{}.npy'.format(input_user))
data_throwing = np.load('data_throwing_{}.npy'.format(input_user))


N_throwing, D = np.shape(data_throwing)
data_total = np.concatenate((data_bending, data_throwing), axis=0)
W = 100

X_Tr = data_total[0:W,:]
Y_Tr = data_total[W:2*W,:]
alpha = 0.05


eta_reg = 100


num_trial = 1000
Stat_hist = np.zeros(num_trial)
L=1000
theta = np.random.randn(L,D)
theta = theta/np.sqrt(np.sum(theta**2,axis=1,keepdims=True))
if input_method == "SW":
    a = np.float64(np.ones(W)/W)
    b = np.float64(np.ones(W)/W)
    for trial in range(num_trial):
        np.random.seed(trial*44 + 5)
        idx_XY = np.sort(np.random.choice(300, 2*W, replace=False))
        X_Te = data_total[idx_XY[:W], :]
        Y_Te = data_total[idx_XY[W:], :]
        stat = SW(torch.from_numpy(X_Te).float(),torch.from_numpy(Y_Te).float(),torch.ones(X_Te.shape[0])/X_Te.shape[0],torch.ones(Y_Te.shape[0])/Y_Te.shape[0],thetas=torch.from_numpy(theta).float(),L=1000).item()
        print('Trial: ',trial, '\t Stat: ',stat)
        Stat_hist[trial] = stat
    np.save('SW_Guest_stat_hist',Stat_hist)
    Threshold = np.quantile(Stat_hist, 0.95)
    print(Threshold)

elif input_method=='StreamSW':

    a = np.float64(np.ones(W) / W)
    b = np.float64(np.ones(W) / W)
    for trial in range(num_trial):
        np.random.seed(trial * 44 + 5)
        idx_XY = np.random.choice(300, 2 * W, replace=False)
        X_Te = data_total[idx_XY[:W], :]
        Y_Te = data_total[idx_XY[W:], :]
        stat = SW(torch.from_numpy(X_Te).float(),torch.from_numpy(Y_Te).float(),torch.ones(X_Te.shape[0])/X_Te.shape[0],torch.ones(Y_Te.shape[0])/Y_Te.shape[0],thetas=torch.from_numpy(theta).float(),L=1000).item()
        print('Trial: ', trial, '\t Stat: ', stat)
        Stat_hist[trial] = stat
    np.save('StreamSW_Guest_stat_hist', Stat_hist)
    Threshold = np.quantile(Stat_hist, 0.95)  # Threshold = np.max(Stat_hist)

Time_length =100 + N_throwing
SW_Test_Stat_hist = np.zeros(Time_length)

StreamSW_Test_Stat_hist = np.zeros(Time_length)

np.random.seed(100)
idx_X = np.sort(np.random.choice(3*W, W, replace=False))
X_Te = data_total[idx_X, :]


if input_method == 'StreamSW':
    streamsw = StreamSW(L=L, d=D, k=30, p=2, c=2.0 / 3.0, thetas=theta, lazy=True, alternate=True)
    streamsw.update_mu(X_Te)
    streamsw.update_nu(X_Te)
flag=True
for t in range(Time_length):
    Time_idx = 400+t
    Y_Te = data_total[Time_idx-W:Time_idx,:]
    if input_method == "SW":
        stat = SW(torch.from_numpy(X_Te).float(),torch.from_numpy(Y_Te).float(),torch.ones(X_Te.shape[0])/X_Te.shape[0],torch.ones(Y_Te.shape[0])/Y_Te.shape[0],thetas=torch.from_numpy(theta).float(),L=1000).item()
        if stat > Threshold:
            decision = 1
            print(flag)
            if(flag):
                detect_time = Time_idx
                print(detect_time)
                flag=False
        else:
            decision = 0
        print('Time Index: ',Time_idx,'\t Decision: ',decision, '\t Stat: ',stat)
        SW_Test_Stat_hist[t] = stat
    elif input_method == 'StreamSW':
        streamsw.update_nu(data_total[Time_idx,:].reshape(1,-1))
        stat = streamsw.compute_distance()
        if stat > Threshold:
            decision = 1
            if (flag):
                detect_time = Time_idx
                flag = False
        else:
            decision = 0
        print('Time Index: ',Time_idx,'\t Decision: ',decision, '\t Stat: ',stat)
        StreamSW_Test_Stat_hist[t] = stat


if input_method == "SW":
    np.save('SW_Guest_Test_Stat_hist',SW_Test_Stat_hist)

    fig = plt.figure(figsize= (6,6))
    plt.plot(np.arange(len(SW_Test_Stat_hist))+400,SW_Test_Stat_hist,'.-')
    plt.title('SW (Sliding Window)',fontsize=20)
    plt.ylabel('Test Statistic',fontsize=20)
    plt.xlabel('Sample Index', fontsize=20)
    plt.axhline(y=Threshold, color='black', linestyle=':', label='Threshold', linewidth=2)
    plt.axvline(x=detect_time, color='tab:red', linestyle='-.', label='Detected Index', linewidth=2)
    print(abs(detect_time-500))
    plt.axvline(x=500, color='tab:green', linestyle='--', label='True Index ', linewidth=2)
    plt.legend(fontsize=18)
    plt.grid()
    plt.tight_layout()
    plt.show()
elif input_method == "StreamSW":
    np.save('StreamSW_Guest_Test_Stat_hist',StreamSW_Test_Stat_hist)

    fig = plt.figure(figsize= (6,6))
    plt.plot(np.arange(len(StreamSW_Test_Stat_hist))+400,StreamSW_Test_Stat_hist,'.-')
    plt.title('Stream-SW',fontsize=20)
    plt.xlabel('Test Statistic',fontsize=20)
    plt.ylabel('Sample Index', fontsize=20)
    plt.axhline(y=Threshold, color='black', linestyle=':', label='Threshold', linewidth=2)
    plt.axvline(x=detect_time, color='tab:red', linestyle='-.', label='Detected Index', linewidth=2)
    print(abs(detect_time - 500))
    plt.axvline(x=500, color='tab:green', linestyle='--', label='True Index ', linewidth=2)
    plt.legend(fontsize=18)
    plt.grid()
    plt.tight_layout()
    plt.show()