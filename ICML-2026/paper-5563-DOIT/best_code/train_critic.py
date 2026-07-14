import os
import gym
import d4rl
import scipy
import tqdm
import functools

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
from utils import get_args, pallaral_eval_policy
from dataset.dataset import D4RL_dataset
LOAD_FAKE=True

def train_critic(args, score_model, data_loader, start_epoch=0):
    def datas_():
        while True:
            yield from data_loader
    datas = datas_()
    n_epochs = 100
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    evaluation_inerval = 4
    save_interval = 20

    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for _ in range(10000):
            data = next(datas)
            data = {k: d.to(args.device) for k, d in data.items()}
            if (epoch <50):
                score_model.q[0].update_q0(data)
            loss2 = score_model.q[0].update_qt(data)
            avg_loss += loss2
            num_items += 1
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        if (epoch % evaluation_inerval == (evaluation_inerval -1)) or epoch==0:
            # for guidance_scale in args.s:
            #     score_model.q[0].guidance_scale = guidance_scale

            #     envs = args.eval_func(score_model.select_actions)
            #     mean = np.mean([envs[i].buffer_return for i in range(args.seed_per_evaluation)])
            #     std = np.std([envs[i].buffer_return for i in range(args.seed_per_evaluation)])

            #     args.writer.add_scalar("eval/rew{}".format(guidance_scale), mean, global_step=epoch)
            #     args.writer.add_scalar("eval/std{}".format(guidance_scale), std, global_step=epoch)
            score_model.q[0].guidance_scale = 1.0

        if args.save_model and ((epoch % save_interval == (save_interval - 1)) or epoch==0):
            torch.save(score_model.q[0].state_dict(), os.path.join("./models_rl", str(args.expid), "critic_ckpt{}.pth".format(epoch+1)))
        args.writer.add_scalar("critic/loss", avg_loss / num_items, global_step=epoch)
        data_p = [0, 10, 25, 50, 75, 90, 100]
        if args.debug:
            args.writer.add_scalars("target/mean", {str(p): d for p, d in zip(data_p, np.percentile(score_model.q[0].all_mean, data_p))}, global_step=epoch)
            args.writer.add_scalars("target/std", {str(p): d for p, d in zip(data_p, np.percentile(score_model.q[0].all_std, data_p))}, global_step=epoch)
            args.writer.add_scalars("target/debug", {str(p): d for p, d in zip(data_p, np.percentile(score_model.q[0].debug_used, data_p))}, global_step=epoch)

def critic(args):
    for dir in ["./models_rl", "./logs"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    if not os.path.exists(os.path.join("./models_rl", str(args.expid))):
        os.makedirs(os.path.join("./models_rl", str(args.expid)))
    writer = SummaryWriter("./logs/" + str(args.expid))
    
    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.eval_func = functools.partial(pallaral_eval_policy, env_name=args.env, seed=args.seed, eval_episodes=args.seed_per_evaluation, diffusion_steps=args.diffusion_steps)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    args.writer = writer
    
    marginal_prob_std_fn = functools.partial(marginal_prob_std, device=args.device)
    args.marginal_prob_std_fn = marginal_prob_std_fn
    score_model= ScoreNet(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
    score_model.q[0].to(args.device)

    print("loading actor...")
    ckpt = torch.load(args.actor_load_path, map_location=args.device)
    score_model.load_state_dict(ckpt)
    
    dataset = D4RL_dataset(args)
    data_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    score_model.q[0].guidance_scale = 0.0
    
    # generate support action set
    if os.path.exists(args.actor_load_path+ "actions{}_raw.npy".format(args.diffusion_steps)) and LOAD_FAKE:
        dataset.fake_actions = torch.Tensor(np.load(args.actor_load_path+ "actions{}_raw.npy".format(args.diffusion_steps)).astype(np.float32)).to(args.device)
    else:
        allstates = dataset.states[:].cpu().numpy()
        actions = []
        for states in tqdm.tqdm(np.array_split(allstates, allstates.shape[0] // 256 + 1)):
            actions.append(score_model.sample(states, sample_per_state=args.M, diffusion_steps=args.diffusion_steps))
        actions = np.concatenate(actions)
        dataset.fake_actions = torch.Tensor(actions.astype(np.float32)).to(args.device)
        if LOAD_FAKE:
            np.save(args.actor_load_path+ "actions{}_raw.npy".format(args.diffusion_steps), actions)

    # fake next action
    if os.path.exists(args.actor_load_path+ "next_actions{}_raw.npy".format(args.diffusion_steps)) and LOAD_FAKE:
        dataset.fake_next_actions = torch.Tensor(np.load(args.actor_load_path+ "next_actions{}_raw.npy".format(args.diffusion_steps)).astype(np.float32)).to(args.device)
    else:
        allstates = dataset.next_states[:].cpu().numpy()
        actions = []
        for states in tqdm.tqdm(np.array_split(allstates, allstates.shape[0] // 256 + 1)):
            actions.append(score_model.sample(states, sample_per_state=args.M, diffusion_steps=args.diffusion_steps))
        actions = np.concatenate(actions)
        dataset.fake_next_actions = torch.Tensor(actions.astype(np.float32)).to(args.device)
        if LOAD_FAKE:
            np.save(args.actor_load_path+ "next_actions{}_raw.npy".format(args.diffusion_steps), actions)

    print("training critic")
    train_critic(args, score_model, data_loader, start_epoch=0)
    print("finished")

if __name__ == "__main__":
    args = get_args()
    args.env = "walker2d-medium-replay-v2"
    args.actor_load_path = "models_rl/walker2d-medium-replay-v20large_actor/behavior_ckpt600.pth"
    if "antmaze" not in args.env:
        args.M = 16
    else:
        args.M = 32
    if args.s is None:
        args.s = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0] if "antmaze" in args.env else [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    critic(args)