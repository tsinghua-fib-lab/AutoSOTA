import numpy as np
import torch
import gymnasium as gym
import argparse
import os
import copy
import torch.nn as nn
import torch.nn.functional as F



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


size1 = 256
size2 = 256


class SpikingNeuron(nn.Module):
    def __init__(self, width, mode="ann"):
        super(SpikingNeuron, self).__init__()
        self.mode = mode
        self.rate = 0

        self.register_buffer("thre", torch.zeros(width))

        self.register_buffer("act_buffer", torch.zeros(1,width))

        self.T = 0
        self.mem = torch.zeros((1,width)).to(device)
        self.init = torch.zeros((1,width)).to(device)
        self.op = 0.
        self.fire = torch.zeros((1,width)).to(device)
        self.width = width
        self.count = torch.zeros(width).to(device)


    @torch.no_grad()
    def optimize(self, x):
        y = F.relu(x)
        self.count += y[0] > 0
        pos_act = y.detach()

        if pos_act.numel() > 0:
            self.act_buffer = torch.cat([self.act_buffer, pos_act])

        return y

    @torch.no_grad()
    def finalize(self):

        if self.act_buffer.numel() == 0 or self.act_buffer.shape[0] <= 1:
            print("[Warning] No activation collected. thre remains zero.")
            return

        # Strip initial zero row (CODE-01: avoids biasing percentile estimates)
        act = self.act_buffer[1:]

        if act.shape[0] == 0:
            print("[Warning] No activation after stripping zero row. thre remains zero.")
            return

        # ALGO-01 v2: Adaptive percentile threshold
        # Per-neuron percentile adjusted by activation frequency:
        #   - Frequently active neurons -> use percentile (handle outliers)
        #   - Rarely active neurons -> use max (conservative)
        # From the paper's commented-out code (convert.py original lines 57-65)
        percentile = 0.999
        n_steps = act.shape[0]  # number of calibration forward passes
        aaa = 1.0 - (1.0 - percentile) * (F.relu(self.count - 1)) / n_steps
        # aaa[j] = 1.0 when count[j]<=1 (use max), = percentile when count[j]>=n_steps

        try:
            # Per-column quantile with adaptive q values
            for j in range(self.width):
                q_j = max(min(aaa[j].item(), 1.0), 0.0)
                self.thre[j] = torch.quantile(act[:, j], q_j)
        except Exception:
            # Fallback to max if adaptive quantile fails
            self.thre = torch.max(act, dim=0).values

        # Ensure threshold is at least a small positive value
        self.thre = torch.clamp(self.thre, min=1e-6)


    def forward(self, x):
        if self.mode == "snn":
            if self.T == 0:
                self.init = self.init + 0.5 * self.thre
                self.mem = self.init
            self.mem = self.mem + x
            self.fire += (self.mem - self.thre) > 0
            x = ((self.mem - self.thre) > 0) * self.thre
            self.mem = self.mem - x
            self.T += 1
        else:
            x = self.optimize(x)
            # self.delta = torch.norm(y - x, p="fro").item()
        return x

    def reset(self):
        min = - self.thre * self.fire
        self.init = torch.clamp(self.mem - self.init,min,max=None) 
        max = 0.5 * self.thre
        self.init = torch.clamp(self.init* self.rate,-max,max) 
        self.mem = self.init
        self.T = 0
        self.op = 0.
        self.fire = torch.zeros((1,self.width)).to(device)
        return
    
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, ts):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, size1)
        self.l2 = nn.Linear(size1, size2)
        self.l3 = nn.Linear(size2, action_dim)

        self.max_action = max_action
        self.neuron1 = SpikingNeuron(size1)
        self.neuron2 = SpikingNeuron(size2)

        self.timestep = ts
        self.mode = 'ANN'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state):
        if self.mode == 'ANN':
            a = self.neuron1(self.l1(state))
            a = self.neuron2(self.l2(a))
            return self.max_action * torch.tanh(self.l3(a))
        else:
            reset_model(self)
            out=torch.zeros(1,size2).to(self.device)
            for t in range(self.timestep):
                ttmp = self.neuron1(self.l1(state))
                ttmp2 = self.neuron2(self.l2(ttmp))
                # print("t: ", t, " -> ", ttmp.mean())
                out+=ttmp2
            return self.max_action * torch.tanh(self.l3(out/self.timestep))

class SAC_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, ts):
        super(SAC_Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256,256)
        self.fc_mu = nn.Linear(256, action_dim)
        self.fc_std = nn.Linear(256, action_dim)

        self.max_action = max_action
        self.neuron1 = SpikingNeuron(256)
        self.neuron2 = SpikingNeuron(256)

        self.timestep = ts
        self.mode = 'ANN'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state):
        if self.mode == 'ANN':
            a = self.neuron1(self.fc1(state))
            a = self.neuron2(self.fc2(a))
            return self.max_action * torch.tanh(self.fc_mu(a))
        else:
            reset_model(self)
            out=torch.zeros(1,size2).to(self.device)
            for t in range(self.timestep):
                ttmp = self.neuron1(self.fc1(state))
                ttmp2 = self.neuron2(self.fc2(ttmp))
                # print("t: ", t, " -> ", ttmp.mean())
                out+=ttmp2
            return self.max_action * torch.tanh(self.fc_mu(out/self.timestep))
              
def reset_model(model):
    for module in model.modules():
        if isinstance(module, SpikingNeuron):
            module.reset()
    return

def select_action(actor, state):
    state = torch.FloatTensor(state.reshape(1, -1)).to(device)
    return actor(state).cpu().data.numpy().flatten()

def eval_policy(policy, env_name, eval_seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.reset(seed=eval_seed + 100)

    returns = np.zeros(eval_episodes)
    for i in range(eval_episodes):
        sum_reward = 0.
        reset_model(policy)
        state, done = eval_env.reset(), False
        state = state[0]
        while not done:
            action = select_action(policy,np.array(state))
            state, reward, done1, done2, _ = eval_env.step(action)
            done = done1 + done2
            returns[i] += reward

    return returns

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--TIME", default=10000, type=int)
    parser.add_argument("--SNN_ts", default=32, type=int)
    parser.add_argument("--policy_name", default="TD3")  # OpenAI gym environment name
    parser.add_argument("--env", default="Hopper-v4")  # OpenAI gym environment name
    parser.add_argument("--eval_seed", default=0, type=int) # Sets Gym, PyTorch and Numpy seeds

    args = parser.parse_known_args()[0]
    
    policy_name = args.policy_name

    if policy_name == "DDPG":
        size1 = 400
        size2 = 300
    
    eval_seed = args.eval_seed

    env_name = args.env
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    SNN = Actor(state_dim, action_dim, max_action, args.SNN_ts).to(device)
    if policy_name=="SAC":
        SNN = SAC_Actor(state_dim, action_dim, max_action, args.SNN_ts).to(device)
    SNN.mode = 'ANN'
    filename = "./models/" + policy_name + "_"+env_name+"_actor"
    # SNN.load_state_dict(torch.load(filename,map_location=device), strict=False)
    # try:
    SNN.load_state_dict(torch.load(filename,map_location=device),strict=False)
    env.reset(seed=eval_seed)
    env.action_space.seed(eval_seed)

    state, done = env.reset(), False
    state = state[0]

    print("convert")

    R=0
    for t in range(args.TIME):

        if done:
            print(R)
            R=0
            state, done = env.reset(), False
            state = state[0]

        action = select_action(SNN,np.array(state))
        action = action.clip(-max_action, max_action)

        # Perform action
        next_state, reward, done1, done2, _ = env.step(action)
        done = done1 + done2
        state = next_state
        R+=reward

    for module in SNN.modules():
        if isinstance(module, SpikingNeuron):
            module.mode = "snn"
            module.finalize()

    SNN.mode='SNN'
    print('eval')


    SNN.timestep=args.SNN_ts

    ans = np.zeros((11,10))
    for i in range(11):
        for module in SNN.modules():
            if isinstance(module, SpikingNeuron):
                module.rate = i / 10.0
        with torch.no_grad():
            ans[i] = eval_policy(SNN, env_name, eval_seed)

    if not os.path.exists(f"./IF-3M/{str(args.SNN_ts)}"):
        os.makedirs(f"./IF-3M/{str(args.SNN_ts)}")
    np.save(f"./IF-3M/{str(args.SNN_ts)}/{policy_name}_{args.env}_{eval_seed}", ans)

