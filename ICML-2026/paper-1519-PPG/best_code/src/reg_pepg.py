import copy

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from src.envs.gridworld import Gridworld
from src.policies.policies import *


class RegPePG:
    def __init__(
        self,
        env: Gridworld,
        max_iterations,
        lamda,
        reg,
        gradient,
        eta,
        sampling,
        n_sample,
        policy_gradient,
        nu,
        unregularized_obj,
        lagrangian,
        N,
        delta,
        B,
        feature_dim=32,
        nn_lr=0.001,
        warmup=1,
        num_trajectories=30,
        value_lr=0.1,
        entropy_reg_lambda=0.1,
        seed=1,
    ):
        np.random.seed(seed)

        self.env = env
        self.max_iterations = max_iterations
        self.lamda = lamda
        self.reg = reg
        self.gradient = gradient
        self.eta = eta
        self.policy_gradient = policy_gradient
        self.nu = nu

        self.feature_dim = feature_dim
        self.nn_lr = nn_lr
        self.value_lr = value_lr
        self.warmup = warmup
        self.num_trajectories = num_trajectories

        # Entropy regularization parameter (λ in the theorem)
        self.entropy_reg_lambda = entropy_reg_lambda

        if not wandb.run:
            wandb.init(
                project="regpepg",
                name=f"regpepg_eta{eta}_nu{nu}_warmup_{warmup}_entreg{entropy_reg_lambda}",
            )

        self.reset()

    def reset(self):
        self.env.reset()
        self.agents = self.env.agents
        self.d_diff = []
        self.sub_gap = []
        self.v_values = []
        self.iteration = 0

        self._init_policy_parameterization()
        self._init_value_function()
        self.warmup_data = []

    def _init_policy_parameterization(self):
        num_actions = len(self.env.agents[1].actions)

        # Policy parameters and features
        self.theta = np.random.normal(0, 0.1, self.feature_dim)
        self.phi = np.random.normal(
            0, 0.2, (self.env.dim, num_actions, self.feature_dim)
        )

        self.f_r = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.env.dim * num_actions),
        )

        self.f_p = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.env.dim * num_actions * self.env.dim),
        )

        self.f_r_optimizer = optim.Adam(self.f_r.parameters(), lr=self.nn_lr)
        self.f_p_optimizer = optim.Adam(self.f_p.parameters(), lr=self.nn_lr)

        self.pi_theta = self._compute_policy()

    def _init_value_function(self):
        """Initialize the learned value function baseline."""
        self.state_dim = self.env.dim

        self.value_network = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.value_optimizer = optim.Adam(
            self.value_network.parameters(), lr=self.value_lr
        )

    def _state_to_tensor(self, state):
        """Convert state to one-hot tensor representation."""
        state_tensor = torch.zeros(self.state_dim)
        state_tensor[state] = 1.0
        return state_tensor

    def _compute_value_estimates(self, trajectories):
        """Compute value function estimates for all states in trajectories."""
        value_estimates = {}

        self.value_network.eval()
        with torch.no_grad():
            for trajectory in trajectories:
                for state, _, _ in trajectory:
                    if state not in value_estimates:
                        state_tensor = self._state_to_tensor(state).unsqueeze(0)
                        value_estimates[state] = self.value_network(state_tensor).item()

        return value_estimates

    def _fit_value_function(self, trajectories):
        """Fit value function using collected trajectories."""
        if not trajectories:
            return

        states = []
        returns = []

        for trajectory in trajectories:
            if not trajectory:
                continue

            trajectory_returns = []
            G = 0
            for t in reversed(range(len(trajectory))):
                _, _, reward = trajectory[t]
                G = reward + self.env.gamma * G
                trajectory_returns.insert(0, G)

            for (state, _, _), G_t in zip(trajectory, trajectory_returns):
                states.append(self._state_to_tensor(state))
                returns.append(G_t)

        if not states:
            return

        states_tensor = torch.stack(states)
        returns_tensor = torch.FloatTensor(returns)

        self.value_network.train()
        total_loss = 0
        num_epochs = 10

        for epoch in range(num_epochs):
            self.value_optimizer.zero_grad()
            value_predictions = self.value_network(states_tensor).squeeze()
            loss = nn.MSELoss()(value_predictions, returns_tensor)
            loss.backward()
            self.value_optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / num_epochs

        wandb.log(
            {
                "value_function_loss": avg_loss,
                "value_training_samples": len(states),
                "iteration": self.iteration,
            }
        )

        if self.iteration % 10 == 0:
            print(
                f"  Value function training: loss = {avg_loss:.6f}, samples = {len(states)}"
            )

    def _compute_policy(self):
        """
        πθ(a|s) = softmax(θ^T φ(s,a))
        """
        num_actions = len(self.env.agents[1].actions)
        policy = np.zeros((self.env.dim, num_actions))

        for s in range(self.env.dim):
            logits = np.array(
                [np.dot(self.theta, self.phi[s, a]) for a in range(num_actions)]
            )
            logits = logits - np.max(logits)
            exp_logits = np.exp(logits)
            policy[s] = exp_logits / np.sum(exp_logits)

        return policy

    def _get_performative_rewards(self):
        """R^πθ(s,a) = f_r(θ)"""
        with torch.no_grad():
            theta_tensor = torch.FloatTensor(self.theta)
            rewards_flat = self.f_r(theta_tensor).numpy()

        num_actions = len(self.env.agents[1].actions)
        return rewards_flat.reshape(self.env.dim, num_actions)

    def _get_performative_transitions(self):
        """P^πθ(s'|s,a) = softmax(f_p(θ))"""
        with torch.no_grad():
            theta_tensor = torch.FloatTensor(self.theta)
            logits_flat = self.f_p(theta_tensor).numpy()

        num_actions = len(self.env.agents[1].actions)
        transitions = np.zeros((self.env.dim, num_actions, self.env.dim))

        idx = 0
        for s in range(self.env.dim):
            for a in range(num_actions):
                logits = logits_flat[idx : idx + self.env.dim]
                logits = logits - np.max(logits)
                exp_logits = np.exp(logits)
                transitions[s, a] = exp_logits / np.sum(exp_logits)
                idx += self.env.dim

        return transitions

    def _collect_trajectories(self):
        """Collect trajectories using current policy."""
        trajectories = []

        for _ in range(self.num_trajectories):
            trajectory = []
            state = np.random.choice(self.env.dim, p=self.env.rho)

            for t in range(200):
                if self.env.is_terminal(state):
                    break

                action = np.random.choice(
                    len(self.env.agents[1].actions), p=self.pi_theta[state]
                )

                if self.iteration >= self.warmup:
                    reward = self._get_performative_rewards()[state, action]
                    trans_probs = self._get_performative_transitions()[state, action]
                else:
                    reward = self.R[state, action]
                    trans_probs = self.T[state, action]

                try:
                    next_state = np.random.choice(self.env.dim, p=trans_probs)
                except ValueError:
                    trans_probs = trans_probs / np.sum(trans_probs)
                    next_state = np.random.choice(self.env.dim, p=trans_probs)

                trajectory.append((state, action, reward))
                state = next_state

            trajectories.append(trajectory)

        return trajectories

    def execute(self):
        self.R, self.T = self.env._get_RT()
        d_first = self.env._get_d(self.T, self.agents[1])
        self.d_last = d_first

        for _ in range(self.max_iterations):
            self.retrain1()
            self.retrain2()

            if self.iteration >= self.warmup:
                self.R = self._get_performative_rewards()
                self.T = self._get_performative_transitions()

    def retrain1(self):
        if not self.policy_gradient:
            return self._retrain_standard()

        # Collect trajectories
        trajectories = self._collect_trajectories()

        # Fit value function (baseline)
        self._fit_value_function(trajectories)

        if self.iteration < self.warmup:
            self.warmup_data.append(
                {
                    "theta": self.theta.copy(),
                    "trajectories": trajectories,
                    "rewards": self.R.copy(),
                }
            )
            gradient = self._compute_entropy_regularized_gradient(
                trajectories, performative=False
            )

        elif self.iteration == self.warmup:
            print(
                f"Initial training of performative models at iteration {self.iteration}..."
            )
            self._train_performative_models()
            gradient = self._compute_entropy_regularized_gradient(
                trajectories, performative=True
            )

        else:  # self.iteration > self.warmup
            self._update_performative_models_online(trajectories)
            gradient = self._compute_entropy_regularized_gradient(
                trajectories, performative=True
            )

        # Gradient ascent step
        self.theta += self.eta * gradient
        self.pi_theta = self._compute_policy()
        self.agents[1].policy = Tabular(self.agents[1].actions, self.pi_theta)

        self._update_metrics()

        if self.iteration % 10 == 0:
            grad_norm = np.linalg.norm(gradient)
            theta_norm = np.linalg.norm(self.theta)
            print(
                f"Iter {self.iteration}: d_diff = {self.d_diff[-1]:.6f}, grad_norm = {grad_norm:.6f}, theta_norm = {theta_norm:.6f}"
            )

        self.iteration += 1

    def _compute_entropy_regularized_advantage(
        self, state, action, return_val, value_estimates
    ):
        """
        Compute regularized advantage: Ã^πθ_πθ(s,a) = Q^πθ_πθ(s,a) - V^πθ_πθ(s) - λ log πθ(a|s)
        This corresponds to the tilde A term from equation (b) in the theorem.
        """
        baseline = value_estimates.get(state, 0.0)

        # Standard advantage
        advantage = return_val - baseline

        # Subtract entropy regularization term: -λ log πθ(a|s)
        log_pi = np.log(
            self.pi_theta[state, action] + 1e-8
        )  # Add small epsilon for numerical stability
        regularized_advantage = advantage - self.entropy_reg_lambda * log_pi

        return regularized_advantage

    def _compute_entropy_regularized_gradient(self, trajectories, performative=False):
        """
        Compute entropy-regularized performative gradient using equation (b) from the theorem:
        ∇θ Ṽ^πθ_πθ(τ) = E[∑_t γ^t (Ã^πθ_πθ(s_t,a_t) ∇θ log πθ(a_t|s_t) +
                                      Ã^πθ_πθ(s_t,a_t) ∇θ log P_πθ(s_{t+1}|s_t,a_t) +
                                      ∇θ r_πθ(s_t,a_t) - λ ∇θ log πθ(a_t|s_t))]
        """
        gradient = np.zeros(self.feature_dim)

        # Get value estimates for baseline
        value_estimates = self._compute_value_estimates(trajectories)

        trajectory_data = []
        all_returns = []
        all_reg_advantages = []

        # Compute returns and regularized advantages
        for trajectory in trajectories:
            if not trajectory:
                continue

            # Compute returns
            returns = []
            G = 0
            for t in reversed(range(len(trajectory))):
                _, _, reward = trajectory[t]
                G = reward + self.env.gamma * G
                returns.insert(0, G)

            # Compute regularized advantages
            reg_advantages = []
            for t, (state, action, _) in enumerate(trajectory):
                reg_advantage = self._compute_entropy_regularized_advantage(
                    state, action, returns[t], value_estimates
                )
                reg_advantages.append(reg_advantage)

            trajectory_data.append((trajectory, returns, reg_advantages))
            all_returns.extend(returns)
            all_reg_advantages.extend(reg_advantages)

        # Compute gradient components
        policy_grad_total = np.zeros(self.feature_dim)
        reward_grad_total = np.zeros(self.feature_dim)
        transition_grad_total = np.zeros(self.feature_dim)
        entropy_grad_total = np.zeros(self.feature_dim)

        for trajectory, returns, reg_advantages in trajectory_data:
            for t, ((state, action, reward), reg_advantage) in enumerate(
                zip(trajectory, reg_advantages)
            ):
                discount = self.env.gamma**t

                # Term 1: Ã^πθ_πθ(s_t,a_t) ∇θ log πθ(a_t|s_t)
                grad_log_pi = self.phi[state, action].copy()
                for a_prime in range(len(self.env.agents[1].actions)):
                    grad_log_pi -= (
                        self.pi_theta[state, a_prime] * self.phi[state, a_prime]
                    )

                policy_term = discount * reg_advantage * grad_log_pi
                policy_grad_total += policy_term
                gradient += policy_term

                # Term 4: -λ ∇θ log πθ(a_t|s_t) (entropy regularization term)
                entropy_term = -self.entropy_reg_lambda * discount * grad_log_pi
                entropy_grad_total += entropy_term
                gradient += entropy_term

                if performative:
                    # Term 3: ∇θ r_πθ(s_t,a_t) (performative reward gradient)
                    reward_grad = self._compute_reward_gradient(state, action)
                    reward_term = discount * reward_grad
                    reward_grad_total += reward_term
                    gradient += reward_term

                    # Term 2: Ã^πθ_πθ(s_t,a_t) ∇θ log P_πθ(s_{t+1}|s_t,a_t) (performative transition gradient)
                    if t < len(trajectory) - 1:
                        next_state = trajectory[t + 1][0]
                        trans_grad = self._compute_transition_gradient(
                            state, action, next_state
                        )
                        trans_term = reg_advantage * trans_grad
                        transition_grad_total += trans_term
                        gradient += trans_term

        # Normalize by number of trajectories
        gradient /= max(len(trajectory_data), 1)
        policy_grad_total /= max(len(trajectory_data), 1)
        reward_grad_total /= max(len(trajectory_data), 1)
        transition_grad_total /= max(len(trajectory_data), 1)
        entropy_grad_total /= max(len(trajectory_data), 1)

        # Log gradient components and advantage statistics
        if all_reg_advantages:
            wandb.log(
                {
                    "avg_regularized_advantage": np.mean(all_reg_advantages),
                    "regularized_advantage_std": np.std(all_reg_advantages),
                    "avg_return": np.mean(all_returns) if all_returns else 0,
                    "gradient_norm_policy": np.linalg.norm(policy_grad_total),
                    "gradient_norm_reward": np.linalg.norm(reward_grad_total),
                    "gradient_norm_transition": np.linalg.norm(transition_grad_total),
                    "gradient_norm_entropy": np.linalg.norm(entropy_grad_total),
                    "gradient_norm_total": np.linalg.norm(gradient),
                    "iteration": self.iteration,
                }
            )

        # Debug logging
        if self.iteration % 10 == 0 and all_returns:
            avg_return = np.mean(all_returns)
            avg_reg_advantage = np.mean(all_reg_advantages)
            reg_advantage_std = np.std(all_reg_advantages)
            print(
                f"  Entropy-regularized gradient: avg_return={avg_return:.4f}, avg_reg_advantage={avg_reg_advantage:.4f}, reg_advantage_std={reg_advantage_std:.4f}"
            )

        return gradient

    def _update_performative_models_online(self, trajectories):
        # Add current data point
        current_data = {
            "theta": self.theta.copy(),
            "trajectories": trajectories,
            "rewards": self.R.copy(),
        }
        self.warmup_data.append(current_data)

        # Keep only recent data
        max_history = 50
        if len(self.warmup_data) > max_history:
            self.warmup_data = self.warmup_data[-max_history:]

        X_theta = []
        y_rewards = []

        for data in self.warmup_data:
            X_theta.append(data["theta"])
            y_rewards.append(data["rewards"].flatten())

        X_theta = np.array(X_theta)
        y_rewards = np.array(y_rewards)

        self._train_network_online(self.f_r, self.f_r_optimizer, X_theta, y_rewards)

    def _train_network_online(self, network, optimizer, X, y):
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        for epoch in range(10):
            optimizer.zero_grad()
            pred = network(X_tensor)
            loss = nn.MSELoss()(pred, y_tensor)
            loss.backward()
            optimizer.step()

        wandb.log({"network_loss": loss.item(), "iteration": self.iteration})

        if self.iteration % 10 == 0:
            final_loss = loss.item()
            print(f"  Online training: loss = {final_loss:.6f}, data_points = {len(X)}")

    def _compute_reward_gradient(self, state, action):
        theta_tensor = torch.FloatTensor(self.theta)
        theta_tensor.requires_grad_(True)

        rewards = self.f_r(theta_tensor)
        num_actions = len(self.env.agents[1].actions)
        reward_sa = rewards[state * num_actions + action]

        reward_sa.backward()
        return theta_tensor.grad.detach().numpy()

    def _compute_transition_gradient(self, state, action, next_state):
        theta_tensor = torch.FloatTensor(self.theta)
        theta_tensor.requires_grad_(True)

        logits = self.f_p(theta_tensor)
        num_actions = len(self.env.agents[1].actions)

        start_idx = (state * num_actions + action) * self.env.dim
        state_logits = logits[start_idx : start_idx + self.env.dim]
        log_probs = torch.log_softmax(state_logits, dim=0)
        log_prob = log_probs[next_state]

        log_prob.backward()
        return theta_tensor.grad.detach().numpy()

    def _train_performative_models(self):
        if len(self.warmup_data) < 2:
            return

        X_theta = []
        y_rewards = []

        for data in self.warmup_data:
            X_theta.append(data["theta"])
            y_rewards.append(data["rewards"].flatten())

        X_theta = np.array(X_theta)
        y_rewards = np.array(y_rewards)

        self._train_network(self.f_r, self.f_r_optimizer, X_theta, y_rewards)

    def _train_network(self, network, optimizer, X, y):
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        for epoch in range(100):
            optimizer.zero_grad()
            pred = network(X_tensor)
            loss = nn.MSELoss()(pred, y_tensor)
            loss.backward()
            optimizer.step()

    def _update_metrics(self):
        d_new = self.env._get_d(self.T, self.agents[1])
        d_diff = np.linalg.norm(d_new - self.d_last) / (
            np.linalg.norm(self.d_last) + 1e-8
        )
        self.d_diff.append(d_diff)
        self.d_last = copy.deepcopy(d_new)

        # Compute current policy value (no solver needed for RegPePG itself)
        if self.iteration >= self.warmup:
            R_performative = self._get_performative_rewards()
            T_performative = self._get_performative_transitions()
            d_performative = self.env._get_d(T_performative, self.agents[1])

            # Pure gradient-based: just compute the current value
            V_pi = np.sum(d_performative * R_performative)

        else:
            V_pi = np.sum(d_new * self.R)

        self.v_values.append(V_pi)
        wandb.log({"v_pi": V_pi, "iteration": self.iteration})

        # # Optional: Compute V* using solver only for performance evaluation (not for the algorithm itself)
        # try:
        #     opt_d = cp.Variable((self.env.dim, len(self.agents[1].actions)), nonneg=True)

        #     # For comparison, we can still compute the optimal solution
        #     if self.reg == 'ER':  # Entropy regularization
        #         opt_obj = cp.Maximize(cp.sum(cp.multiply(opt_d, self.R)) +
        #                              self.entropy_reg_lambda * cp.sum(cp.entr(opt_d)))
        #     else:  # Default L2 regularization
        #         opt_obj = cp.Maximize(cp.sum(cp.multiply(opt_d, self.R)) -
        #                              self.lamda/2 * cp.power(cp.pnorm(opt_d, 2), 2))

        #     constraints = []
        #     for s in range(self.env.dim):
        #         if self.env.is_terminal(s):
        #             continue
        #         constraints.append(cp.sum(opt_d[s]) == self.env.rho[s] +
        #                          self.env.gamma * cp.sum(cp.multiply(opt_d, self.T[:,:,s])))

        #     problem = cp.Problem(opt_obj, constraints)
        #     problem.solve(solver=cp.SCS, eps=1e-5, verbose=False)

        #     if problem.value is not None and np.isfinite(problem.value):
        #         V_star = problem.value

        #         if abs(V_star) > 1e-6:
        #             sub_gap = max((V_star - V_pi) / abs(V_star), 0)
        #         else:
        #             sub_gap = 0.0

        #         self.sub_gap.append(sub_gap)
        #         wandb.log({
        #             "suboptimality_gap": sub_gap,
        #             "v_star": V_star,
        #             "iteration": self.iteration
        #         })
        #     else:
        #         self.sub_gap.append(self.sub_gap[-1] if self.sub_gap else 0.0)

        # except:
        #     self.sub_gap.append(self.sub_gap[-1] if self.sub_gap else 0.0)

        # if self.iteration % 10 == 0:
        print(f"Iter {self.iteration}: V^π={V_pi:.4f}")

    def _retrain_standard(self):
        """
        Standard occupancy measure optimization (only used when policy_gradient=False).
        For RegPePG, we typically use policy_gradient=True, so this is mainly for fallback.
        """
        d = cp.Variable((self.env.dim, len(self.agents[1].actions)), nonneg=True)

        if self.reg == "L2":
            objective = cp.Maximize(
                cp.sum(cp.multiply(d, self.R))
                - self.lamda / 2 * cp.power(cp.pnorm(d, 2), 2)
            )
        elif self.reg == "ER":
            objective = cp.Maximize(
                cp.sum(cp.multiply(d, self.R))
                + self.entropy_reg_lambda * cp.sum(cp.entr(d))
            )
        else:
            raise ValueError("Unknown regularizer.")

        constraints = []
        for s in range(self.env.dim):
            if self.env.is_terminal(s):
                continue
            constraints.append(
                cp.sum(d[s])
                == self.env.rho[s]
                + self.env.gamma * cp.sum(cp.multiply(d, self.T[:, :, s]))
            )

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, eps=1e-5, verbose=False)

        if d.value is not None:
            self.d_diff.append(
                np.linalg.norm(d.value - self.d_last)
                / (np.linalg.norm(self.d_last) + 1e-8)
            )
            self.d_last = d.value
            self.agents[1].policy = RandomizedD_Policy(self.agents[1].actions, d.value)

        self.iteration += 1

    def retrain2(self):
        self.agents[2].policy = self.env.response_model(self.agents)
