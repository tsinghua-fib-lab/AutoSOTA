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
import shimmy
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
    wandb_project_name: str = "GCR"
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
    """ the discount factor gamma"""
    tau: float = 0.0025
    """target smoothing coefficient"""
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
    huber_beta: float = 0.3
    """huber_beta"""
    n_step: int = 3
    """N-step return"""
    save_every : int = 100000
    eval_frequency: int = 50000
    eval_episodes: int = 10

def make_env(env_id, seed, idx, capture_video, run_name, eval_mode=False):
    def thunk():
        if env_id.startswith("dm_control/"):
            env = gym.make(env_id)
            env = FlattenObservation(env)
            env = TransformObservation(env, lambda obs: obs.astype(np.float32))
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


class NStepReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space,
                 device, n_envs, n_step=3, gamma=0.99):

        self.rb = ReplayBuffer(
            buffer_size, observation_space, action_space,
            device, n_envs=1, handle_timeout_termination=False,
        )
        self.n_step = n_step
        self.gamma = gamma
        self.n_envs = n_envs
        self.temp_buffers = [[] for _ in range(n_envs)]

    def add(self, obs, next_obs, actions, rewards, dones, infos):
        for i in range(self.n_envs):
            self.temp_buffers[i].append((
                obs[i], next_obs[i], actions[i],
                rewards[i], dones[i]
            ))

            if dones[i]:
                self._flush_env(i)
            elif len(self.temp_buffers[i]) >= self.n_step:
                self._store_nstep(i)

    def _store_nstep(self, env_idx):
        buf = self.temp_buffers[env_idx]
        # n-step cumulative reward
        n_step_reward = 0.0
        for k in range(self.n_step):
            # Rt(n)=rt+γrt+1+⋯+γn−1rt+n−1
            n_step_reward += (self.gamma ** k) * buf[k][3]  # rewards


        obs_0 = buf[0][0]
        next_obs_n = buf[self.n_step - 1][1]

        action_0 = buf[0][2]
        done_n = buf[self.n_step - 1][4]

        self.rb.add(
            obs_0.reshape(1, -1),
            next_obs_n.reshape(1, -1),
            action_0.reshape(1, -1),
            np.array([n_step_reward], dtype=np.float32),
            np.array([done_n], dtype=np.float32),
            [{}],
        )
        buf.pop(0)

    def _flush_env(self, env_idx):
        buf = self.temp_buffers[env_idx]
        while len(buf) > 0:
            steps = len(buf)
            n_step_reward = 0.0
            for k in range(steps):
                n_step_reward += (self.gamma ** k) * buf[k][3]

            self.rb.add(
                buf[0][0].reshape(1, -1),
                buf[steps - 1][1].reshape(1, -1),
                buf[0][2].reshape(1, -1),
                np.array([n_step_reward], dtype=np.float32),
                np.array([buf[steps - 1][4]], dtype=np.float32),
                [{}],
            )
            buf.pop(0)
        self.temp_buffers[env_idx] = []

    def sample(self, batch_size):
        return self.rb.sample(batch_size)

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


    timestamp = int(time.time())
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{timestamp}"

    if args.track:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    if args.save_model:
        if args.track:
            ckpt_dir = f"checkpoints/{timestamp}"
        else:
            ckpt_dir = "checkpoints/local"
        os.makedirs(ckpt_dir, exist_ok=True)

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

    # EVAL
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

    envs.single_observation_space.dtype = np.float32
    rb = NStepReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        n_step=args.n_step,
        gamma=args.gamma,
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

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
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

                clipped_noise = (torch.randn_like(mu_next) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                ) * target_actor.action_scale

                next_state_actions = (mu_next + clipped_noise).clamp(
                    envs.single_action_space.low[0],
                    envs.single_action_space.high[0]
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target).squeeze(-1)


            # yt=Rt(n)+γnQ(st+n,at+n)
            next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
            ) * (args.gamma ** args.n_step) * min_qf_next_target

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

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)


            if global_step % args.save_every == 0 and global_step > 0 and args.save_model:
                env_short = args.env_id.replace("dm_control/", "").replace("-", "_")
                ckpt_path = os.path.join(
                    ckpt_dir,
                    f"{env_short}_step{global_step // 1000}k_seed{args.seed}.pt"
                )
                torch.save({
                    "global_step": global_step,
                    "actor": actor.state_dict(),
                    "qf1": qf1.state_dict(),
                    "qf2": qf2.state_dict(),
                    "target_actor": target_actor.state_dict(),
                    "qf1_target": qf1_target.state_dict(),
                    "qf2_target": qf2_target.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "q_optimizer": q_optimizer.state_dict(),
                    "args": vars(args),
                    "wandb_run_id": run.id if args.track else "local",
                }, ckpt_path)
                print(f"✅ Checkpoint saved: {ckpt_path}")

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
                eval_mean, eval_std = evaluate(eval_envs, actor, device, args.eval_episodes)
                print(f"📊 Eval step={global_step}: mean={eval_mean:.1f}, std={eval_std:.1f}")
                writer.add_scalar("eval/mean_return", eval_mean, global_step)
                writer.add_scalar("eval/std_return", eval_std, global_step)

    envs.close()
    writer.close()
