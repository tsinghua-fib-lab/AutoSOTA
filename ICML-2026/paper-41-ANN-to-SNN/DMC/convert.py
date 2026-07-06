import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import drqv2
import dmc
import torch
import utils
import argparse
import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. IF neuron
# ============================================================

class SpikingNeuron(nn.Module):
    def __init__(self, v_th):
        super().__init__()
        self.register_buffer("thre", v_th)
        self.mem = None
        self.rate = 0.
        self.fire = None
        self.T=0
        self.init=None

    def reset(self):
        if self.mem is None:
            return
        if self.init is None:
            return
        if self.fire is None:
            return
        
        min = - self.thre * self.fire
        self.init = torch.clamp(self.mem - self.init,min,max=None) 
        max = 0.5 * self.thre
        self.init = torch.clamp(self.init* self.rate,-max,max) 
        self.mem = self.init
        # print(self.rate)
        self.T = 0
        self.op = 0.
        self.fire = torch.zeros_like(self.fire) 
        # print(self.init)
        return

    def forward(self, x):
        if self.mem is None:
            self.mem = torch.zeros_like(x) 
        if self.init is None:
            self.init = torch.zeros_like(x) 
        if self.fire is None:
            self.fire = torch.zeros_like(x) 
        if self.T == 0:
            self.init = self.init + 0.5 * self.thre
            self.mem = self.init

        self.mem = self.mem + x
        self.fire += (self.mem - self.thre) > 0
        x = ((self.mem - self.thre) > 0) * self.thre
        self.mem = self.mem - x
        self.T += 1
        
        return x


# ============================================================
# 2. SNN layers
# ============================================================

class ConvIF(nn.Module):
    def __init__(self, conv: nn.Conv2d, v_th):
        super().__init__()
        self.conv = conv
        self.ifn = SpikingNeuron(v_th.view(-1, 1, 1))

    def reset(self):
        self.ifn.reset()

    def forward(self, x):
        return self.ifn(self.conv(x))


class LinearIF(nn.Module):
    def __init__(self, linear: nn.Linear, v_th):
        super().__init__()
        self.linear = linear
        self.ifn = SpikingNeuron(v_th)

    def reset(self):
        self.ifn.reset()

    def forward(self, x):
        return self.ifn(self.linear(x))


# ============================================================
# 3. Activation recorder
# ============================================================

class ActivationRecorder:
    def __init__(self):
        self.storage = defaultdict(list)

    def hook(self, name):
        def fn(module, inp, out):
            self.storage[name].append(out.detach().cpu())
        return fn


# ============================================================
# 4. Threshold computation
# ============================================================

def conv_channel_threshold(act_list):
    # act_list: list of [B, C, H, W]
    C = act_list[0].shape[1]

    channel_values = [[] for _ in range(C)]

    for acts in act_list:
        # acts: [B, C, H, W]
        acts = acts.permute(1, 0, 2, 3).reshape(C, -1)
        for c in range(C):
            channel_values[c].append(acts[c].cpu())

    v_th = []
    for c in range(C):
        vals = torch.cat(channel_values[c], dim=0)
        v_th.append(torch.max(vals))

    return torch.stack(v_th)

def fc_neuron_threshold(act_list):
    D = act_list[0].shape[1]
    neuron_values = [[] for _ in range(D)]

    for acts in act_list:
        # acts: [B, D]
        for d in range(D):
            neuron_values[d].append(acts[:, d].cpu())

    v_th = []
    for d in range(D):
        vals = torch.cat(neuron_values[d], dim=0)
        v_th.append(torch.max(vals))

    return torch.stack(v_th)


# ============================================================
# 5. ANN activation collection
# ============================================================

def register_encoder_hooks(encoder, recorder):
    idx = 0
    for m in encoder.convnet:
        if isinstance(m, nn.ReLU):
            m.register_forward_hook(recorder.hook(f"encoder.relu{idx}"))
            idx += 1


def register_actor_hooks(actor, recorder):
    idx = 0
    for m in actor.trunk:
        if isinstance(m, nn.ReLU):
            m.register_forward_hook(recorder.hook(f"actor.trunk.relu{idx}"))
            idx += 1

    idx = 0
    for m in actor.policy:
        if isinstance(m, nn.ReLU):
            m.register_forward_hook(recorder.hook(f"actor.policy.relu{idx}"))
            idx += 1


@torch.no_grad()
def collect_activations(agent, env, num_iters):
    recorder = ActivationRecorder()
    register_encoder_hooks(agent.encoder, recorder)
    register_actor_hooks(agent.actor, recorder)



    time_step = env.reset()
    for _ in range(num_iters):
        if time_step.last():
            time_step = env.reset()

        with utils.eval_mode(agent):
            action = agent.act(time_step.observation,0,eval_mode=True)
        time_step = env.step(action)

    return recorder.storage


def compute_thresholds(storage):
    th = {}
    for k, v in storage.items():
        if v[0].dim() == 4:
            th[k] = conv_channel_threshold(v)
        else:
            th[k] = fc_neuron_threshold(v)
    return th


# ============================================================
# 6. SNN Encoder
# ============================================================

class SNNEncoder(nn.Module):
    def __init__(self, ann_encoder, th_dict):
        super().__init__()
        layers = []
        relu_idx = 0

        for m in ann_encoder.convnet:
            if isinstance(m, nn.Conv2d):
                layers.append(m)
            elif isinstance(m, nn.ReLU):
                conv = layers.pop()
                v_th = th_dict[f"encoder.relu{relu_idx}"]
                layers.append(ConvIF(conv, v_th))
                relu_idx += 1

        self.layers = nn.ModuleList(layers)

    def reset(self):
        for m in self.layers:
            if hasattr(m, "reset"):
                m.reset()

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x.view(x.size(0), -1)


# ============================================================
# 7. SNN Actor
# ============================================================

class SNNActor(nn.Module):
    def __init__(self, ann_actor, th_dict):
        super().__init__()

        self.trunk = nn.ModuleList()
        idx = 0
        prev = None
        for m in ann_actor.trunk:
            if isinstance(m, nn.Linear):
                prev = m
            elif isinstance(m, nn.ReLU):
                v_th = th_dict[f"actor.trunk.relu{idx}"]
                print(v_th.shape)
                self.trunk.append(LinearIF(prev, v_th))
                idx += 1

        self.policy = nn.ModuleList()
        idx = 0
        prev = None
        for m in ann_actor.policy:
            if isinstance(m, nn.Linear):
                prev = m
            elif isinstance(m, nn.ReLU):
                v_th = th_dict[f"actor.policy.relu{idx}"]
                print(v_th.shape)
                self.policy.append(LinearIF(prev, v_th))
                idx += 1

        self.out = ann_actor.policy[-1]

    def reset(self):
        for m in self.trunk:
            m.reset()
        for m in self.policy:
            m.reset()

    def forward(self, x):
        for m in self.trunk:
            x = m(x)
        for m in self.policy:
            x = m(x)
        return self.out(x)


# ============================================================
# 8. Time-stepped SNN inference
# ============================================================

class SNNAgent:
    def __init__(self, encoder, actor, T):
    
        self.encoder = encoder
        self.actor = actor
        self.T = T

    @torch.no_grad()
    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=device).unsqueeze(0)
        obs = obs / 255.0 - 0.5
        self.encoder.reset()
        self.actor.reset()
        out_sum = 0.0
        for _ in range(self.T):
            h = self.encoder(obs)
            # obs = self.encoder(obs.unsqueeze(0))
            a = self.actor(h)
            out_sum += a

        action = torch.tanh(out_sum / self.T)
        return action.cpu().numpy()[0]
    
    def set_reset_rate(self, rate):
        for m in self.encoder.layers:
            if hasattr(m, "reset"):
                m.ifn.rate=rate
                # print(m.rate)
        for m in self.actor.trunk:
            if hasattr(m, "reset"):
                m.ifn.rate=rate
                # print(m.rate)
        for m in self.actor.policy:
            if hasattr(m, "reset"):
                m.ifn.rate=rate
                # print(m.rate)
        
# ============================================================
# 9. Example main
# ============================================================

def eval_ann(env_name, agent, seed, eval_episodes = 10):
    env = dmc.make(env_name, 3, 2, seed)
    step, total_reward = 0, 0 
    for _ in range(eval_episodes):
        time_step = env.reset()
        while not time_step.last():
            with torch.no_grad(), utils.eval_mode(agent):
                action = agent.act(time_step.observation,0,eval_mode=True)
            time_step = env.step(action)
            total_reward += time_step.reward
            step += 1
    return total_reward/eval_episodes

def eval_snn(env_name, agent, seed, eval_episodes = 10):
    env = dmc.make(env_name, 3, 2, seed)
    step, total_reward = 0, 0 
    for _ in range(eval_episodes):
        time_step = env.reset()
        agent.encoder.reset()
        agent.actor.reset()
        while not time_step.last():
            with torch.no_grad():
                action = agent.act(time_step.observation,0,eval_mode=True)
            time_step = env.step(action)
            total_reward += time_step.reward
            step += 1
    return total_reward/eval_episodes


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--TIME", default=5000, type=int)
    parser.add_argument("--SNN_ts", default=32, type=int)
    parser.add_argument("--env_name", default="finger_spin")  
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_known_args()[0]

    snapshot = f"./exp_local/{args.env_name}/snapshot.pt"
    torch.serialization.add_safe_globals([drqv2.DrQV2Agent])
    payload = torch.load(snapshot, map_location=device, weights_only=False)
    agent = payload["agent"]

    env = dmc.make(args.env_name, 3, 2, args.seed)
    
    storage = collect_activations(agent, env, num_iters=args.TIME)
    Vth = compute_thresholds(storage)

    snn_encoder = SNNEncoder(agent.encoder, Vth).to(device)
    snn_actor = SNNActor(agent.actor, Vth).to(device)

    snn_agent = SNNAgent(snn_encoder, snn_actor, args.SNN_ts) 

    
    returns=[]
    for i in range(11):
        snn_agent.set_reset_rate(i/10.0)
        returns.append(eval_snn(args.env_name, snn_agent, args.seed))
    if not os.path.exists(f"./IF/{args.SNN_ts}"):
        os.makedirs(f"./IF/{args.SNN_ts}")
    np.save(f"./IF/{args.SNN_ts}/{args.env_name}_{args.seed}", returns)
    print(f"./IF/{args.SNN_ts}/{args.env_name}_{args.seed}", returns)