from q_mamba import Q_Mamba

from dataset import My_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import pickle
from tensorboardX import SummaryWriter
from env.optimizer_mamba import Optimizer
from options import get_options
import os
import json
import numpy as np
import random
import time

def mk_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def save_args(args, log_path):
    argsDict = args.__dict__
    txt_path = log_path + '/setting.txt'
    print("txt_path:", txt_path)
    with open(txt_path, 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    json_path = os.path.join(log_path, 'setting.json')
    print("json_path:", json_path)
    with open(json_path, 'w') as f:
        json.dump(argsDict, f, indent=4)
        
def append_args(args_2, log_path):
    argsDict = args_2.__dict__
    txt_path = log_path + '/setting.txt'
    print("txt_path:", txt_path)
    with open(txt_path, 'a') as f:
        f.writelines('\n' + '------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    json_path = os.path.join(log_path, 'setting.json')
    print("json_path:", json_path)
    with open(json_path, 'a') as f:
        json.dump(argsDict, f, indent=4)

'''training'''

# def test_for_one_epoch(q_model, envs, algorithm_num, eval_num=1):
    
#     q_model.eval()
#     question_rewards = {}
#     # pbar = tqdm(total=8*19)
#     j = algorithm_num
#     for k in range(8):
#         env = envs[j * 8 + k]
#         rewards = []
#         costs = []
#         start_time = time.time()
#         for i in range(eval_num):
#             env.seed(i + 1)
#             reward, cost = q_model.rollout_trajectory(env, 500, need_return_cost=True)
#             rewards.append(reward)
#             costs.append(cost)
#             # pbar.update()
#         mean_reward = np.mean(rewards)
#         std_reward = np.std(rewards)
#         # var_reward = np.var(rewards)
#         mean_cost = np.mean(costs)
#         std_cost = np.std(costs)
        
#         # dict_reward = {'mean': mean_reward, 'std': std_reward, 'var': var_reward}
#         mean_cost_time = (time.time() - start_time) / eval_num
#         start_time = time.time()
        
#         dict_reward_cost = {'rewards': rewards, 'costs': costs, 'cost_time': mean_cost_time}
#         question_rewards["quesition_{}".format(k)] = dict_reward_cost
#     return question_rewards
        
        # print(f"algorithm:{j}, question: {k}, mean reward: {mean_reward}, std reward: {std_reward}, mean cost: {mean_cost}, std cost: {std_cost}, cost time: {mean_cost_time}")
   
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(cfg):

    file_name = os.path.basename(cfg.trajectory_file_path)
    file_name_without_ext = os.path.splitext(file_name)[0]
    cfg.task_name = file_name_without_ext
    cfg.task_id = int(file_name_without_ext.split('_')[-2])
    
    print(f"------- this algorithm is start training: {cfg.task_id} -------")

    cfg.log_path = cfg.log_path + "/"+ cfg.task_name + "/" +  cfg.log_name + "/"
    cfg.model_dir = cfg.model_dir + "/"+ cfg.task_name + "/" + cfg.log_name + "/"
    trajectory_file_path = cfg.trajectory_file_path
    mk_dir(cfg.log_path)
    mk_dir(cfg.model_dir)
    if os.path.exists(trajectory_file_path):
        assert os.path.exists(trajectory_file_path), "trajectory_file_path does not exist"
    save_args(cfg, cfg.log_path)
    
    set_seed(cfg.seed) 
    
    
    logger = SummaryWriter(cfg.log_path)

    q_model = Q_Mamba(state_dim=cfg.state_dim,
                actions_dim=cfg.actions_dim,
                action_bins=cfg.action_bins,
                d_state=cfg.d_state,
                d_conv=cfg.d_conv,
                expand=cfg.expand,
                num_hidden_mlp=cfg.num_hidden_mlp,
                device=cfg.device,
                mamba_num = cfg.mamba_num,
                gamma = cfg.gamma,)

    print(f"----------model: {cfg.model} is initialized----------")
    q_model.train()
    
    # traj_dataset = My_Dataset('./trajectory_files/trajectory_set_0_CfgX.pkl')
    traj_dataset = My_Dataset(trajectory_file_path)
    dataloader = DataLoader(traj_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
    
    print(f"-----dataset is loaded, batch_size: {cfg.batch_size}, shuffle: {cfg.shuffle}, dataset length: {len(traj_dataset)}-----")
    
    with open('./task_set_for_mamba.pkl', 'rb') as f:
        test_envs = pickle.load(f)
    
    q_model.save_model(cfg.model_dir + '/epoch_start.pth')
    num_epoch = cfg.num_epoch
    has_conservative_reg_loss = cfg.has_conservative_reg_loss
    pbar = tqdm(total=num_epoch)
    epoch_info = {}
    best_loss = 1e10
    for epoch in range(num_epoch):
        loss , td_loss, cql_loss = 0, 0, 0
        for states, actions, rewards in dataloader:
            batch_traj = {'state': states, 'action': actions, 'reward': rewards}
            if has_conservative_reg_loss:
                total_loss, td_loss_, cql_loss_ = q_model.learn_from_trajectory(batch_traj, has_conservative_reg_loss = has_conservative_reg_loss,beta = cfg.beta,alpha = getattr(cfg, "lambda", 0.1))
                loss += total_loss
                td_loss += td_loss_
                cql_loss += cql_loss_
            else:
                loss += q_model.learn_from_trajectory(batch_traj, has_conservative_reg_loss = has_conservative_reg_loss,beta = cfg.beta,alpha = getattr(cfg, "lambda", 0.1))
        if has_conservative_reg_loss:
            logger.add_scalar('td_loss',td_loss,epoch)
            logger.add_scalar('conservative_reg_loss',cql_loss,epoch)
            pbar.set_postfix(total_loss=loss, td_loss = td_loss, conservative_reg_loss = cql_loss)
        else:
            pbar.set_postfix(loss = loss)
        
        logger.add_scalar('total_loss',loss,epoch)
        
        # pbar.update()
        # epoch_test_info = test_for_one_epoch(q_model, test_envs, cfg.task_id, eval_num=1)
        # epoch_info[epoch] = epoch_test_info
        # mean_reward = np.mean([np.mean(v['rewards']) for v in epoch_test_info.values()]).item()
        # mean_cost = np.mean([np.mean(v['costs']) for v in epoch_test_info.values()]).item()
        # # print(f"epoch: {epoch}, mean_reward: {mean_reward}, mean_cost: {mean_cost}")
        # logger.add_scalar('test_mean_reward', mean_reward, epoch)
        # logger.add_scalar('test_mean_cost', mean_cost, epoch)
        
        # test_key = epoch_test_info.keys()
        # test_rewards = [np.mean(v['rewards']) for v in epoch_test_info.values()]
        # test_costs = [np.mean(v['costs']) for v in epoch_test_info.values()]
        # logger.add_scalars('test_rewards', dict(zip(test_key,test_rewards)), epoch)
        # logger.add_scalars('test_costs', dict(zip(test_key,test_costs)), epoch)
        
        
        
        q_model.save_model(cfg.model_dir + '/epoch_{}.pth'.format(epoch))
        
    q_model.save_model(cfg.model_dir + '/epoch_{}.pth'.format(num_epoch))
    with open(cfg.log_dir + '/epoch_info.pkl', 'wb') as f:
        pickle.dump(epoch_info, f)

def get_args_from_json(json_path):
    with open(json_path, 'r') as f:
        args = json.load(f)
    return args

def find_last_epoch(model_dir):
    epochs = [f.split('.')[0].split('_')[-1] for f in os.listdir(model_dir) if f.endswith('.pth')]
    epochs = [int(e) for e in epochs if e != 'best' and e != 'copy']
    return max(epochs)

def get_best_loss(model,dataloader,model_dir,has_conservative_reg_loss):
    if model == 'q_mamba':
        q_model = Q_Mamba(from_pretrain=model_dir + '/model_best.pth')
    
    q_model.eval()
    loss = 0
    for states, actions, rewards in dataloader:
        batch_traj = {'state': states, 'action': actions, 'reward': rewards}
        if has_conservative_reg_loss:
            total_loss, td_loss_, cql_loss_ = q_model.learn_from_trajectory(batch_traj, has_conservative_reg_loss = has_conservative_reg_loss)
            loss += total_loss
        else:
            loss += q_model.learn_from_trajectory(batch_traj, has_conservative_reg_loss = has_conservative_reg_loss)

    return loss

def train_resume(cfg):
    # read json
    # cfg.resume = r'/home/data3/ZhouJiang2/AAAAAAAAA/Gong/Mamba-DAC-v3/log/trajectory_set_3_Unit/20240912T014630/setting.json'
    args = get_args_from_json(cfg.resume)
    import pprint
    pprint.pprint(args)
    last_epoch=find_last_epoch(args['model_dir'])
    last_model_dir = args['model_dir'] + '/epoch_' + str(last_epoch) + '.pth'
    if args['model'] == 'q_mamba':
        q_model = Q_Mamba(from_pretrain=last_model_dir)
    q_model.train()
    num_epoch = args['num_epoch']
    logger = SummaryWriter(args['log_path'])
    traj_dataset = My_Dataset(args['trajectory_file_path'])
    dataloader = DataLoader(traj_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
    
    num_epoch = args['num_epoch']
    has_conservative_reg_loss = args['has_conservative_reg_loss']
    pbar = tqdm(total=num_epoch)
    pbar.update(last_epoch)
    best_loss = get_best_loss(args['model'],dataloader,args['model_dir'],has_conservative_reg_loss)
    print("best_loss:",best_loss)
    for epoch in range(last_epoch, num_epoch):
        loss , td_loss, cql_loss = 0, 0, 0
        for states, actions, rewards in dataloader:
            batch_traj = {'state': states, 'action': actions, 'reward': rewards}
            if has_conservative_reg_loss:
                total_loss, td_loss_, cql_loss_ = q_model.learn_from_trajectory(batch_traj, has_conservative_reg_loss = has_conservative_reg_loss)
                loss += total_loss
                td_loss += td_loss_
                cql_loss += cql_loss_
            else:
                loss += q_model.learn_from_trajectory(batch_traj, has_conservative_reg_loss = has_conservative_reg_loss)
        logger.add_scalar('total_loss',loss,epoch)
        if has_conservative_reg_loss:
            logger.add_scalar('td_loss',td_loss,epoch)
            logger.add_scalar('conservative_reg_loss',cql_loss,epoch)
            pbar.set_postfix(total_loss=loss, td_loss = td_loss, conservative_reg_loss = cql_loss)
        else:
            pbar.set_postfix(loss = loss)
        pbar.update()
        if epoch % 10 == 0:
            q_model.save_model(args['model_dir'] + '/epoch_{}.pth'.format(epoch))
        if loss < best_loss:
            best_loss = loss
            q_model.save_model(args['model_dir'] +  '/model_best' + '.pth')
    q_model.save_model(args['model_dir'] + '/epoch_{}.pth'.format(num_epoch))

def get_parameter_number(self): 
    total_num = sum(p.numel() for p in self.parameters())
    trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
    return {'Actor: Total': total_num, 'Trainable': trainable_num}

def train_online(cfg):
    file_name = os.path.basename(cfg.trajectory_file_path)
    file_name_without_ext = os.path.splitext(file_name)[0]
    cfg.algorithm_id =int(file_name_without_ext.split('_')[-2]) 
    cfg.task_name = file_name_without_ext
    

    cfg.log_path = cfg.log_path + "/"+"online_"+ cfg.task_name + "/" +  cfg.time_stamp + "/"
    cfg.model_dir = cfg.model_dir + "/"+"online_"+ cfg.task_name + "/" + cfg.time_stamp + "/"

    trajectory_file_path = cfg.trajectory_file_path
    mk_dir(cfg.log_path)
    mk_dir(cfg.model_dir)
    if os.path.exists(trajectory_file_path):
        assert os.path.exists(trajectory_file_path), "trajectory_file_path does not exist"
    save_args(cfg, cfg.log_path)
    
    logger = SummaryWriter(cfg.log_path)
    

    print("training q_mamba")
    q_model = Q_Mamba(state_dim=cfg.state_dim,
                actions_dim=cfg.actions_dim,
                action_bins=cfg.action_bins,
                d_state=cfg.d_state,
                d_conv=cfg.d_conv,
                expand=cfg.expand,
                num_hidden_mlp=cfg.num_hidden_mlp,
                device=cfg.device,
                mamba_num = cfg.mamba_num,
                gamma = cfg.gamma,)
   
    # print(get_parameter_number(q_model.dac_block))
    
    
    q_model.train()
    
    # traj_dataset = My_Dataset('./trajectory_files/trajectory_set_0_CfgX.pkl')
    # traj_dataset = My_Dataset(trajectory_file_path)
    # dataloader = DataLoader(traj_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
    
    q_model.save_model(cfg.model_dir + '/epoch_0.pth')
    num_epoch = cfg.num_epoch
    has_conservative_reg_loss = cfg.has_conservative_reg_loss
    
    
    
    best_loss = 1e10
    
    with open('./task_set_for_mamba.pkl', 'rb') as f:
        envs = pickle.load(f)
    print("len:",len(envs))
    
    pbar = tqdm(total=num_epoch)
    for epoch in range(num_epoch):
        loss , td_loss, cql_loss = 0, 0, 0
        for quesition_id in range(16): 
            # print("algorithm_id:",cfg.algorithm_id)
            # print("cfg.algorithm_id + 16 * quesition_id:",cfg.algorithm_id + 16 * quesition_id)
            env = envs[16 * cfg.algorithm_id +  quesition_id]
            if has_conservative_reg_loss:
                total_loss, td_loss_, cql_loss_ = q_model.learn_from_online_trajectory(env, 500, has_conservative_reg_loss = has_conservative_reg_loss)
                loss += total_loss
                td_loss += td_loss_
                cql_loss += cql_loss_
            else:
                loss += q_model.learn_from_online_trajectory(env, 500, has_conservative_reg_loss = has_conservative_reg_loss)
        logger.add_scalar('total_loss',loss,epoch)
        if has_conservative_reg_loss:
            logger.add_scalar('td_loss',td_loss,epoch)
            logger.add_scalar('conservative_reg_loss',cql_loss,epoch)
            pbar.set_postfix(total_loss=loss, td_loss = td_loss, conservative_reg_loss = cql_loss)
        else:
            pbar.set_postfix(loss = loss)
        pbar.update()
        if epoch % 100 == 0:
            q_model.save_model(cfg.model_dir + '/epoch_{}.pth'.format(epoch))
        if loss < best_loss:
            best_loss = loss
            q_model.save_model(cfg.model_dir +  '/model_best' + '.pth')
    q_model.save_model(cfg.model_dir + '/epoch_{}.pth'.format(num_epoch))

'''testing'''
def random_rollout_trajectory(env, maxGens):
        state = env.reset() # 1, 1, 9
        config_space = env.get_config_space()
        total_reward = 0
        for gen in range(maxGens):
            actions_for_this_gen = {}           
            for action_step, key in enumerate(config_space.keys()):
                rge = config_space[key]
                if isinstance(rge[0], float):
                    config_range = 16
                else:
                    config_range = len(rge)
                action_id = torch.randint(config_range,(1,))
                actions_for_this_gen[key] = action_id

                # give the selected action to env
            state, reward, _, _ = env.mamba_step(actions_for_this_gen)
            total_reward += reward
        return total_reward

def test(cfg):
    cfg.log_path = cfg.log_path + "/test/" +"/"+ cfg.model + "/"+ cfg.time_stamp + "/" 
    mk_dir(cfg.log_path)
    
    with open('./task_set_for_mamba.pkl', 'rb') as f:
        envs = pickle.load(f)

    q_mamba_for_test = Q_Mamba(from_pretrain=cfg.load_path)

    question_rewards = {}
    j = cfg.algorithm_id
    pbar = tqdm(total=8 * 19)
    for k in range(8):
        env = envs[j * 8 + k]
        rewards = []
        costs = []
        start_time = time.time()
        for i in range(19):
            env.seed(i + 1)
            reward, cost = q_mamba_for_test.rollout_trajectory(env,500,need_return_cost=True)
            # print(f"algorithm:{j}, question: {k}, reward: {reward}, cost: {cost}")
            rewards.append(reward)
            costs.append(cost)
            pbar.update()
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)

        mean_cost_time = (time.time() - start_time) / 19
        start_time = time.time()
        
        dict_reward_cost = {'rewards': rewards, 'costs': costs, 'cost_time': mean_cost_time}
        question_rewards["quesition_{}".format(k)] = dict_reward_cost
        
        print(f"algorithm:{j}, question: {k}, mean reward: {mean_reward}, std reward: {std_reward}, mean cost: {mean_cost}, std cost: {std_cost},mean cost time: {mean_cost_time}")

    with open(cfg.log_path + '/test_rewards.pkl', 'wb') as f:
        pickle.dump(question_rewards, f)    
        
if __name__ == '__main__':
    cfg = get_options()
    
    assert cfg.train or cfg.test or cfg.train_online, "Please specify at least one of the following options: --train, --test, --train_online"
    
    if cfg.train:
        train(cfg)
        
    if cfg.train_online:
        train_online(cfg)
        
    if cfg.test:
        test(cfg)








