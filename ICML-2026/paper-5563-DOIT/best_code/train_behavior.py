import os
import d4rl
import gym
import scipy
import tqdm
import functools
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from diffusion_SDE.loss import loss_fn
from diffusion_SDE.schedule import marginal_prob_std
from diffusion_SDE.model import ScoreNet
from utils import get_args
from dataset.dataset import D4RL_dataset

from pipeline import Pipeline
from search.configs import Arguments
from search.utils import get_hyper_params

def train_behavior(args, score_model, data_loader, start_epoch=0, action_sampler=None, eval_policy=None, lr=1e-4):
    def datas_():
        while True:
            yield from data_loader
    datas = datas_()
    n_epochs = 20
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    save_interval = 1
    
    optimizer = torch.optim.Adam(score_model.parameters(), lr=lr)

    if eval_policy is not None:
        envs = eval_policy(score_model.select_actions)
        mean = np.mean([envs[i].buffer_return for i in range(args.seed_per_evaluation)])
        std = np.std([envs[i].buffer_return for i in range(args.seed_per_evaluation)])
        print("Evaluation over {} episodes: {:.3f} +- {:.3f}".format(args.seed_per_evaluation, mean, std))
    
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for _ in tqdm.tqdm(range(60)):
            data = next(datas)
            data = {k: d.to(args.device) for k, d in data.items()}

            s = data['s']
            a = data['a'] if action_sampler is None else torch.tensor(np.array(action_sampler(data['s'])), device=s.device, dtype=s.dtype)
            score_model.condition = s
            loss = loss_fn(score_model, a, args.marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            score_model.condition = None

            avg_loss += loss
            num_items += 1
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        
        envs = eval_policy(score_model.select_actions)
        mean = np.mean([envs[i].buffer_return for i in range(args.seed_per_evaluation)])
        std = np.std([envs[i].buffer_return for i in range(args.seed_per_evaluation)])
        print("Evaluation over {} episodes: {:.3f} +- {:.3f}".format(args.seed_per_evaluation, mean, std))
        args.writer.add_scalar("eval/mean", mean, global_step=epoch)
        args.writer.add_scalar("eval/std", std, global_step=epoch)

        if (epoch % save_interval == 0) or epoch == 599:
            torch.save(score_model.state_dict(), os.path.join("./models_rl", str(args.expid), "behavior_ckpt{}.pth".format(epoch+1)))
            
        args.writer.add_scalar("actor/loss", avg_loss / num_items, global_step=epoch)

def behavior(args):
    # The diffusion behavior training pipeline is copied directly from https://github.com/ChenDRAG/SfBC/blob/master/train_behavior.py
    for dir in ["./models_rl", "./logs"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    if not os.path.exists(os.path.join("./models_rl", str(args.expid))):
        os.makedirs(os.path.join("./models_rl", str(args.expid)))

    writer = SummaryWriter("./logs/" + str(args.expid))
    print("Env:", args.env, "Device", args.device, "Id:", args.expid)
    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    args.writer = writer
    
    marginal_prob_std_fn = functools.partial(marginal_prob_std, device=args.device)
    args.marginal_prob_std_fn = marginal_prob_std_fn
    score_model= ScoreNet(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
    
    dataset = D4RL_dataset(args)
    data_loader = DataLoader(dataset, batch_size=16384, shuffle=True)

    tfg_args = Arguments()
    tfg_args.dataset = args.env
    tfg_args.device = args.device
    tfg_args.seed = args.seed
    tfg_args.inference_steps = args.diffusion_steps
    tfg_args = get_hyper_params(tfg_args)[0] 
    pipeline = Pipeline(args=tfg_args)
    sampler = pipeline.sampler

    eval_policy = pipeline.env_args.eval_func

    score_model.load_state_dict(torch.load(pipeline.env_args.actor_load_path,map_location=args.device), strict=True)

    print("training behavior")
    train_behavior(args, score_model, data_loader, start_epoch=0, action_sampler=sampler, eval_policy=eval_policy)
    print("finished")

if __name__ == "__main__":
    args = get_args()
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args.expid = "ft-" + args.env + current_time
    # args.env = "hopper-medium-v2"
    behavior(args)