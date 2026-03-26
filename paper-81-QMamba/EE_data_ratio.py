from q_mamba import Q_Mamba
from dataset import My_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import pickle
from tensorboardX import SummaryWriter
from env.optimizer_mamba import Optimizer
import os
import numpy as np
import time, warnings

def mk_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def rollout(q_mamba_good, test_envs, epo, repeat=19):
    results = 0
    # pbar = tqdm(total=8*19, desc=f'Epoch {epo} testing')
    q_mamba_good.eval()
    for k in range(8):
        env = test_envs[k]
        reward = 0
        for i in range(repeat):
            env.seed(i+1)
            # print('reward = {}'.format(i,q_mamba_good.rollout_trajectory(env,500)))
            reward += q_mamba_good.rollout_trajectory(env,500)
            # pbar.update()
        results += reward / repeat
    # pbar.close()
    results /= 8
    return results
'''
training
'''

# Change the rate of good data here
rate = 0.75


warnings.filterwarnings("ignore")
torch.set_num_threads(1)
os.environ['KMP_DUPLICATE_LIB_OK']='True'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.multiprocessing.set_sharing_strategy('file_system')
time_stamp = time.strftime("%Y%m%dT%H%M%S")
log_dir = 'log/train/' + time_stamp
model_save_dir = 'model/' + time_stamp
device = 'cuda:0'
q_mamba = Q_Mamba(device=device)


# exit()
mk_dir(log_dir)
mk_dir(model_save_dir)
logger = SummaryWriter(log_dir)

with open(f'trajectory_files/trajectory_set_0_CfgX.pkl', 'rb') as f:
    trajectories_cfgx = pickle.load(f)
with open(f'trajectory_files/trajectory_set_0_Rand.pkl', 'rb') as f:
    trajectories_rand = pickle.load(f)
trajectories_unit = []

for i in range(16):
    trajectories_unit += trajectories_cfgx[i * 700:int(i* 700 + 700*rate)]
    trajectories_unit += trajectories_rand[i * 700:int(i* 700 + 700*(1 - rate))]
traj_dataset = My_Dataset('./trajectory_files/trajectory_set_0_Unit.pkl', device=device)
traj_dataset.data = trajectories_unit

batch_size = 64
with open('./task_set_for_mamba_test.pkl', 'rb') as f:
    test_envs = pickle.load(f)
q_mamba.train()
num_epoch = 300
best_loss = 1e9
best_reward = rollout(q_mamba, test_envs, 0, 3)
logger.add_scalar('testing',best_reward,0)
best_epoch = 0
dataloader = DataLoader(traj_dataset, batch_size=batch_size, shuffle=False) #todo, close shuffle 
q_mamba.save_model(model_save_dir + '/model_0.pth')
pbar = tqdm(total=int(num_epoch*np.ceil(len(traj_dataset)/batch_size)), desc=time_stamp)
for epoch in range(num_epoch):
    q_mamba.train()
    loss = 0
    loss1 = 0
    loss2 = 0
    for bid, (states, actions, rewards) in enumerate(dataloader):
        batch_traj = {'state': states, 'action': actions, 'reward': rewards}
        loss_, loss_q, loss_cql = q_mamba.learn_from_trajectory(batch_traj)
        loss += loss_
        loss1 += loss_q
        loss2 += loss_cql
        pbar.update()
    if loss < best_loss:
        best_loss = loss
    q_mamba.save_model(model_save_dir + '/model_{}.pth'.format(epoch+1))
    if q_mamba.lr_decay:
        q_mamba.lr_adapter.step()
    if (epoch + 1) % 5 == 0:
        rol = rollout(q_mamba, test_envs, epoch+1, 3)
        logger.add_scalar('testing',rol,epoch+1)
        if rol > best_reward:
            best_reward = rol
            best_epoch = epoch + 1
    logger.add_scalar('loss',loss,epoch+1)
    logger.add_scalar('q_loss',loss1,epoch+1)
    logger.add_scalar('CQL_loss',loss2,epoch+1)

    pbar.set_postfix(total_loss = loss, best_reward=best_reward, best_epoch=best_epoch)
    
print(best_epoch, best_reward)
