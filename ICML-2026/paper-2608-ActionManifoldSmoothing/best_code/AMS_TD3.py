# -*- coding: utf-8 -*-
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
# 在文件开头添加
import shimmy  # ← 添加这一行！必须要import才能注册DMC环境
from gymnasium.wrappers import FlattenObservation, TransformObservation, TransformReward

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    K_neighbors: int = 10
    neighborhood_radius: float = 0.6
    use_neighborhood_td: bool = True
    refresh_every: int = 30000
    huber_beta: float = 0.3
    eval_frequency: int = 50000
    eval_episodes: int = 10


def make_env(env_id, seed, idx, capture_video, run_name, eval_mode=False):
    def thunk():
        if env_id.startswith("dm_control/"):
            env = gym.make(env_id)
            env = FlattenObservation(env)
            env = TransformObservation(env, lambda obs: obs.astype(np.float32), None)
            env = TransformReward(env, lambda reward: np.float32(reward))
        elif env_id.startswith("h1hand-") or env_id.startswith("h1-"):
            import humanoid_bench
            env = gym.make(env_id)
        else:
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id)

        if not eval_mode:
            env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk
# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, hidden_dim=256):
        super().__init__()
        obs_dim = np.array(env.single_observation_space.shape).prod()
        action_dim = np.prod(env.single_action_space.shape)

        # Layer 1
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        # Layer 2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Output Layer
        self.fc3 = nn.Linear(hidden_dim, 1)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x, a):
        if x.dtype != torch.float32:
            x = x.float()
        if a.dtype != torch.float32:
            a = a.float()

        xu = torch.cat([x, a], 1)

        x = self.fc1(xu)
        x = self.ln1(x)
        x = F.silu(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = F.silu(x)

        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


class OrthogonalSampler:
    def __init__(self, action_dim, K, device, refresh_every=1000):
        self.action_dim = action_dim
        self.K = min(K, action_dim)
        self.device = device
        self.refresh_every = refresh_every
        self.call_count = 0
        self._refresh_basis()

    def _refresh_basis(self):
        random_matrix = torch.randn(self.action_dim, self.action_dim, device=self.device)
        Q, _ = torch.linalg.qr(random_matrix)
        self.basis = Q[:, :self.K]  # [action_dim, K]

    def sample(self, batch_size):
        self.call_count += 1
        if self.call_count % self.refresh_every == 0:
            self._refresh_basis()

        signs = (torch.randint(0, 2, (batch_size, self.K), device=self.device) * 2 - 1).float()
        dirs = self.basis.T.unsqueeze(0) * signs.unsqueeze(-1)
        return dirs  # [batch_size, K, action_dim]

def evaluate(envs_eval, actor, device, eval_episodes=10):
    returns = []
    obs, _ = envs_eval.reset()
    obs = obs.astype(np.float32)
    ep_reward = 0.0

    while len(returns) < eval_episodes:
        with torch.no_grad():
            actions = actor(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy().clip(
                envs_eval.single_action_space.low,
                envs_eval.single_action_space.high)

        obs, rewards, terminations, truncations, infos = envs_eval.step(actions)
        obs = obs.astype(np.float32)
        ep_reward += rewards[0]

        if terminations[0] or truncations[0]:
            returns.append(ep_reward)
            ep_reward = 0.0
            obs, _ = envs_eval.reset()
            obs = obs.astype(np.float32)

    return np.mean(returns), np.std(returns)

if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )


    eval_envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + 1000, 0, False, run_name, eval_mode=True)]
    )

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    # EMA for evaluation stability
    ema_decay = 0.999
    actor_ema = Actor(envs).to(device)
    qf1_ema = QNetwork(envs).to(device)
    qf2_ema = QNetwork(envs).to(device)
    actor_ema.load_state_dict(actor.state_dict())
    qf1_ema.load_state_dict(qf1.state_dict())
    qf2_ema.load_state_dict(qf2.state_dict())
    for p in actor_ema.parameters(): p.requires_grad = False
    for p in qf1_ema.parameters(): p.requires_grad = False
    for p in qf2_ema.parameters(): p.requires_grad = False


    action_dim = envs.single_action_space.shape[0]
    orth_sampler = OrthogonalSampler(action_dim, K=args.K_neighbors, device=device, refresh_every=args.refresh_every)
    action_low = envs.single_action_space.low[0]
    action_high = envs.single_action_space.high[0]

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    obs = obs.astype(np.float32)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        next_obs = next_obs.astype(np.float32)
        rewards = rewards.astype(np.float32)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break
        elif "episode" in infos:
            ep_data = infos["episode"]
            ep_returns = ep_data["r"]
            ep_lengths = ep_data["l"]
            for i in range(len(ep_returns)):
                if ep_data["_r"][i]:
                    print(f"global_step={global_step}, episodic_return={ep_returns[i]}")
                    writer.add_scalar("charts/episodic_return", ep_returns[i], global_step)
                    writer.add_scalar("charts/episodic_length", ep_lengths[i], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx].astype(np.float32)
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            with torch.no_grad():

                mu_next = target_actor(data.next_observations)

                batch_size = data.next_observations.shape[0]
                K_eff = orth_sampler.K


                orthogonal_dirs = orth_sampler.sample(batch_size)


                epsilon = args.neighborhood_radius * target_actor.action_scale
                # Symmetric neighborhood: both + and - directions (matching AMS_SAC.py and paper eq.)
                a_plus = mu_next.unsqueeze(1) + epsilon * orthogonal_dirs
                a_minus = mu_next.unsqueeze(1) - epsilon * orthogonal_dirs
                a_plus = a_plus.clamp(action_low, action_high)
                a_minus = a_minus.clamp(action_low, action_high)

                obs_expanded = data.next_observations.unsqueeze(1).expand(-1, K_eff, -1)
                obs_flat = obs_expanded.reshape(batch_size * K_eff, -1)
                action_dim = mu_next.shape[-1]

                q_plus = torch.min(
                    qf1_target(obs_flat, a_plus.reshape(-1, action_dim)),
                    qf2_target(obs_flat, a_plus.reshape(-1, action_dim))
                ).view(batch_size, K_eff)

                q_minus = torch.min(
                    qf1_target(obs_flat, a_minus.reshape(-1, action_dim)),
                    qf2_target(obs_flat, a_minus.reshape(-1, action_dim))
                ).view(batch_size, K_eff)

                min_qf_next_target = 0.5 * (q_plus + q_minus).mean(dim=1)

                next_q_value = data.rewards.flatten() + (
                        1 - data.dones.flatten()
                ) * args.gamma * min_qf_next_target


            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            # qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            # qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf1_loss = F.smooth_l1_loss(qf1_a_values, next_q_value, beta=args.huber_beta)
            qf2_loss = F.smooth_l1_loss(qf2_a_values, next_q_value, beta=args.huber_beta)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update EMA parameters
                with torch.no_grad():
                    for ema_p, p in zip(actor_ema.parameters(), actor.parameters()):
                        ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
                    for ema_p, p in zip(qf1_ema.parameters(), qf1.parameters()):
                        ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
                    for ema_p, p in zip(qf2_ema.parameters(), qf2.parameters()):
                        ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)



            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )


            if global_step % args.eval_frequency == 0 and global_step > 0:
                # Use EMA parameters for evaluation
                actor_backup = {k: v.clone() for k, v in actor.state_dict().items()}
                actor.load_state_dict(actor_ema.state_dict())
                eval_mean, eval_std = evaluate(eval_envs, actor, device, args.eval_episodes)
                actor.load_state_dict(actor_backup)
                print(f"📊 Eval step={global_step}: mean={eval_mean:.1f}, std={eval_std:.1f}")
                writer.add_scalar("eval/mean_return", eval_mean, global_step)
                writer.add_scalar("eval/std_return", eval_std, global_step)




    envs.close()
    writer.close()
