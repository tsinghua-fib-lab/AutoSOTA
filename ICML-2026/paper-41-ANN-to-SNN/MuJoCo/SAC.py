import numpy as np
import torch
import gymnasium as gym
import argparse
import os
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256,256)
        self.fc_mu = nn.Linear(256, action_dim)
        self.fc_std = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        action = action * self.max_action
        return action, log_prob.sum(1,keepdim=True)
    
    def deter(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        action = torch.tanh(mu) * self.max_action
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):

        sa1 = torch.cat([state, action], 1)
        q = F.relu(self.l1(sa1))
        q = F.relu(self.l2(q))
        q = self.l3(q)

        return q

    def Q1(self, state, action):
        sa1 = torch.cat([state, action], 1)
        q = F.relu(self.l1(sa1))
        q = F.relu(self.l2(q))
        q = self.l3(q)

class SAC(object):

    def __init__(self, state_dim , action_dim, max_action,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)  
        self.critic_1 = Critic(state_dim, action_dim).to(device)  
        self.critic_2 = Critic(state_dim, action_dim).to(device)   
        self.target_critic_1 = Critic(state_dim, action_dim).to(device) 
        self.target_critic_2 = Critic(state_dim, action_dim).to(device)  
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
       
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],lr=alpha_lr)
        self.target_entropy = target_entropy  
        self.update_interval=1
        self.gamma = gamma
        self.tau = tau

        self.total_it = 0


    def select_action(self, state, deter=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        if deter:
            action_ = self.actor.deter(state)
        else:
            action_ , _= self.actor(state)
        return action_.cpu().data.numpy().flatten()

    def train(self, replaybuffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_actions, log_prob = self.actor(next_state)
            entropy = -log_prob
            q1_value = self.target_critic_1(next_state, next_actions)
            q2_value = self.target_critic_2(next_state, next_actions)
            next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
            target_Q = reward + not_done * self.gamma * next_value

        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)
        critic_1_loss = F.mse_loss(current_Q1, target_Q)
        critic_2_loss = F.mse_loss(current_Q2, target_Q)
        critic_loss = critic_1_loss + critic_2_loss
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        new_actions, log_prob = self.actor(state)
        entropy = -log_prob
        q1_value = self.critic_1(state, new_actions)
        q2_value = self.critic_2(state, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        if self.total_it % self.update_interval == 0:
            # Update the frozen target models
            for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        return float(critic_loss), float(-actor_loss)

    def save(self, filename):
        torch.save(self.critic_1.state_dict(), filename + "_critic_1")
        torch.save(self.critic_1_optimizer.state_dict(), filename + "_critic_1_optimizer")

        torch.save(self.critic_2.state_dict(), filename + "_critic_2")
        torch.save(self.critic_2_optimizer.state_dict(), filename + "_critic_2_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


def eval_policy(policy, env_name, eval_seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.reset(seed=eval_seed + 100)

    avg_reward1 = 0.
    avg_reward2 = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        state = state[0]
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done1, done2, _ = eval_env.step(action)
            done = done1 + done2
            avg_reward1 += reward

    avg_reward1 /= eval_episodes

    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        state = state[0]
        while not done:
            action = policy.select_action(np.array(state),deter=True)
            state, reward, done1, done2, _ = eval_env.step(action)
            done = done1 + done2
            avg_reward2 += reward

    avg_reward2 /= eval_episodes

    #print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward1:.3f} , {avg_reward2:.3f}")
    #print("---------------------------------------")
    return avg_reward1,avg_reward2



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="SAC")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="Ant-v4")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=50e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=3e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_known_args()[0]


    file_name = f"{args.policy}_{args.env}_{args.seed}"
    file = open(file_name + '.csv', "w")
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "alpha_lr": 3e-4,
        "target_entropy": -env.action_space.shape[0],
        "tau": args.tau,
        "gamma": args.discount,
    }
    # Initialize policy

    policy = SAC(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]

    state, done = env.reset(), False
    state = state[0]
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    actor_update = 0
    critic_update = 0
    actor_loss = 0
    critic_loss = 0

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(state)).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done1, done2, _ = env.step(action)
        done = done1 + done2
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            [loss1, loss2] = policy.train(replay_buffer, args.batch_size)
            critic_loss += loss1
            critic_update += 1
            actor_loss += loss2
            actor_update += 1

        if done:
            if actor_update != 0:
                actor_loss /= actor_update
            if critic_update != 0:
                critic_loss /= critic_update

            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Q loss:{critic_loss:.5f}  actor Q:{actor_loss:.5f}")
            # Reset environment
            file.write(
                str(f"{t + 1}\t{episode_num + 1}\t{episode_timesteps}\t{episode_reward:.3f}\t{critic_loss:.5f}\t{actor_loss:.5f}\n"))
            state, done = env.reset(), False
            state = state[0]
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            actor_update = 0
            critic_update = 0
            actor_loss = 0
            critic_loss = 0

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/{file_name}", evaluations)
            policy.save(f"./models/{file_name}-{str((t + 1) / args.eval_freq)}")

            #if args.save_model: policy.save(f"./models/{file_name}")

    file.close()
    policy.save(f"./models/{file_name}")