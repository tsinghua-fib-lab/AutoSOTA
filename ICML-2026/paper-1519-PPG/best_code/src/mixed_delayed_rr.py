import copy

import cvxpy as cp
import numpy as np

from src.envs.gridworld import Gridworld
from src.policies.policies import *


class Mixed_Delayed_RR:
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
        warmup=0,
        v=1.1,
        k=3,
        seed=1,
    ):
        np.random.seed(seed)

        self.env = env
        self.max_iterations = max_iterations
        self.lamda = lamda
        self.reg = reg
        self.gradient = gradient
        self.eta = eta
        self.sampling = sampling
        self.n_sample = n_sample
        self.policy_gradient = policy_gradient
        self.nu = nu
        self.unregularized_obj = unregularized_obj
        self.lagrangian = lagrangian
        # number of rounds for lagrangian method / FTRL steps
        self.N = N
        # parameter delta for lagrangian
        self.delta = delta
        # parameter B for lagrangian / coverage constraint
        self.B = B

        # warmup parameter for PePG
        self.warmup = warmup
        # v parameter for MDRR - controls sample weighting
        self.v = v
        # k parameter for MDRR - number of delayed rounds
        self.k = k

        self.sampling = True
        self.n_sample = 100
        self.N = 10
        self.B = 10
        self.lamda = 0.1

        self.solver_kwargs = {"solver": cp.SCS, "eps": 1e-5}

        print(f"MDRR Init: sampling={sampling}, n_sample={n_sample}, k={k}, v={v}")

        self.reset()

    def reset(self):
        env = self.env
        env.reset()

        self.agents = env.agents
        self.d_diff = []
        self.sub_gap = []
        self.iteration = 0
        self.v_values = []

        self._n_steps = 0  # Steps within current k-round cycle
        self.n_iterations = 0  # Number of retraining iterations
        self.empirical_occupancy_list = []  # Occupancy measures from k rounds
        self.trajectories_list = []  # Trajectories from k rounds
        self.num_samples_occ_list = []  # Sample counts from k rounds

        # Initial occupancy
        d_first = env._get_d(env._get_RT()[1], self.agents[1])
        self.d_last = d_first

        return

    def execute(self):
        print(" MDRR Is Running (k deployments per iteration)! ")
        env = self.env

        self.R, self.T = env._get_RT()

        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration}")
            self.iteration = iteration

            # Perform k deployments in this iteration
            self.k_deployment_cycle()

            self.retrain2()
            self.R, self.T = env._get_RT()

        return

    def k_deployment_cycle(self):
        env = self.env

        # Reset for k-cycle
        self.empirical_occupancy_list = []
        self.trajectories_list = []
        self.num_samples_occ_list = []

        # Collect data over k deployments
        for deployment in range(self.k):
            print(f"  Deployment {deployment + 1}/{self.k}")

            trajectories = []
            if self.sampling and self.n_sample > 0:
                trajectories = self._collect_trajectories()
                print(f"    Collected {len(trajectories)} trajectories")
            else:
                print(
                    f"    No sampling (sampling={self.sampling}, n_sample={self.n_sample})"
                )

            num_samples_total = sum(len(traj) for traj in trajectories)
            self.num_samples_occ_list.append(num_samples_total)

            emp_occ, reward_value = self.empirical_occupancy_n_reward(trajectories)

            self.empirical_occupancy_list.append(emp_occ)
            self.trajectories_list.append(trajectories)

            print(f"    {len(trajectories)} trajectories, {num_samples_total} samples")
            print(
                f"    Empirical occupancy norm: {np.linalg.norm(emp_occ):.4f}, reward: {reward_value:.4f}"
            )

            if deployment < self.k - 1:
                self.retrain2()
                self.R, self.T = env._get_RT()

        print(f"  Retraining after {self.k} deployments (iteration {self.iteration})")

        total_samples = sum(self.num_samples_occ_list)
        if total_samples == 0:
            print(
                "  No samples collected across all deployments, using fallback policy update"
            )
            self._fallback_update()
        else:
            sample_list, occupancy_list = self._apply_mdrr_weighting()
            new_occupancy = self._solve_ftrl_optimization(sample_list, occupancy_list)

            if new_occupancy is not None:
                self._update_policy_and_metrics(new_occupancy)
            else:
                print("  FTRL optimization failed, using fallback")
                self._fallback_update()

    def _update_policy_and_metrics(self, new_occupancy):
        env = self.env
        agent = self.agents[1]

        policy_probs = np.zeros_like(new_occupancy)
        for s in range(self.env.dim):
            policy_probs[s] = new_occupancy[s] / np.sum(new_occupancy[s])

        agent.policy = Tabular(agent.actions, policy_probs)

        # Compute occupancy from the updated policy
        actual_d = self.env._get_d(self.T, agent)
        d_diff_value = np.linalg.norm(actual_d - self.d_last) / (
            np.linalg.norm(self.d_last) + 1e-8
        )
        self.d_diff.append(d_diff_value)
        self.d_last = copy.deepcopy(actual_d)

        V_pi = (
            np.sum(actual_d * self.R) - self.lamda / 2 * np.linalg.norm(actual_d) ** 2
        )
        self.v_values.append(V_pi)

        print(f"MDRR Iteration {self.iteration}: V^π={V_pi:.4f}")
        print(f"  d_diff = {d_diff_value:.6f}")
        print(f"  Parameters: v={self.v}, k={self.k}, λ={self.lamda}, B={self.B}")

    def _fallback_update(self):
        env = self.env
        agent = self.agents[1]

        current_d = env._get_d(self.T, agent)

        d = cp.Variable((env.dim, len(agent.actions)), nonneg=True)

        if self.reg == "L2":
            objective = cp.Maximize(
                cp.sum(cp.multiply(d, self.R))
                - self.lamda / 2 * cp.power(cp.pnorm(d, 2), 2)
            )
        else:
            objective = cp.Maximize(cp.sum(cp.multiply(d, self.R)))

        constraints = []
        for s in env.state_ids:
            if env.is_terminal(s):
                continue
            constraints.append(
                cp.sum(d[s])
                == env.rho[s] + env.gamma * cp.sum(cp.multiply(d, self.T[:, :, s]))
            )

        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(**self.solver_kwargs)
            if d.value is not None:
                self._update_policy_and_metrics(d.value)
                print("  Fallback optimization successful")
                return
        except:
            pass

        print("  Using current occupancy as fallback")
        self._update_policy_and_metrics(current_d)

    def _collect_trajectories(self):
        trajectories = []

        agent = self.agents[1]

        for _ in range(self.n_sample):
            trajectory = []
            state = np.random.choice(self.env.dim, p=self.env.rho)

            for t in range(200):  # Max trajectory length
                if self.env.is_terminal(state):
                    break

                if hasattr(agent.policy, "action_prob"):
                    action_probs = np.array(
                        [
                            agent.policy.action_prob(state, a)
                            for a in range(len(agent.actions))
                        ]
                    )
                    action = np.random.choice(len(agent.actions), p=action_probs)
                else:
                    action_probs = agent.policy._get_probs(state)
                    action = np.random.choice(len(agent.actions), p=action_probs)

                reward = self.R[state, action]
                trans_probs = self.T[state, action]

                try:
                    next_state = np.random.choice(self.env.dim, p=trans_probs)
                except ValueError:
                    trans_probs = trans_probs / np.sum(trans_probs)
                    next_state = np.random.choice(self.env.dim, p=trans_probs)

                trajectory.append((state, action, next_state, reward))
                state = next_state

            if len(trajectory) > 0:
                trajectories.append(trajectory)

        return trajectories

    def meta_step(self):
        env = self.env

        self._n_steps += 1

        trajectories = []
        if self.sampling and self.n_sample > 0:
            trajectories = self._collect_trajectories()
            print(f"  Collected {len(trajectories)} trajectories using custom method")
        else:
            print(f"  No sampling (sampling={self.sampling}, n_sample={self.n_sample})")

        num_samples_total = sum(len(traj) for traj in trajectories)
        self.num_samples_occ_list.append(num_samples_total)

        emp_occ, reward_value = self.empirical_occupancy_n_reward(trajectories)

        self.empirical_occupancy_list.append(emp_occ)
        self.trajectories_list.append(trajectories)

        print(
            f"  Step {self._n_steps}/{self.k}: {len(trajectories)} trajectories, {num_samples_total} samples"
        )
        print(
            f"  Empirical occupancy norm: {np.linalg.norm(emp_occ):.4f}, reward: {reward_value:.4f}"
        )

        if self._n_steps == self.k:
            print(f"  Retraining after {self.k} rounds (iteration {self.n_iterations})")
            self._n_steps = 0

            total_samples = sum(self.num_samples_occ_list)
            if total_samples == 0:
                print(
                    "  No samples collected across all rounds, using fallback policy update"
                )
                self._fallback_update()
            else:
                sample_list, occupancy_list = self._apply_mdrr_weighting()
                new_occupancy = self._solve_ftrl_optimization(
                    sample_list, occupancy_list
                )

                if new_occupancy is not None:
                    self._update_policy_and_metrics(new_occupancy)
                else:
                    print("  FTRL optimization failed, using fallback")
                    self._fallback_update()

            self.num_samples_occ_list = []
            self.trajectories_list = []
            self.empirical_occupancy_list = []
            self.n_iterations += 1
        else:
            current_d = self.env._get_d(self.T, self.agents[1])
            d_diff_value = np.linalg.norm(current_d - self.d_last) / (
                np.linalg.norm(self.d_last) + 1e-8
            )
            self.d_diff.append(d_diff_value)

            V_pi = (
                np.sum(current_d * self.R)
                - self.lamda / 2 * np.linalg.norm(current_d) ** 2
            )
            self.v_values.append(V_pi)

    def empirical_occupancy_n_reward(self, trajectories):
        env = self.env
        agent = self.agents[1]

        occupancy = np.zeros((env.dim, len(agent.actions)))
        total_mass = 0.0
        total_reward = 0.0

        for trajectory in trajectories:
            for t, (state, action, next_state, reward) in enumerate(trajectory):
                gamma_factor = env.gamma**t
                occupancy[state, action] += gamma_factor
                total_mass += gamma_factor
                total_reward += gamma_factor * reward

        if total_mass > 0:
            occupancy /= total_mass * (1.0 - env.gamma)

        if len(trajectories) > 0:
            total_reward /= len(trajectories)

        return occupancy, total_reward

    def _apply_mdrr_weighting(self):
        sample_list = []
        occupancy_list = []
        num_samples_opt = np.inf
        num_samples_used = 0
        cur_total_weight = 0

        for t in range(self.k, 0, -1):
            trajectories_t = self.trajectories_list[t - 1]
            num_samples_t = self.num_samples_occ_list[t - 1]
            occupancy_t = self.empirical_occupancy_list[t - 1]

            occupancy_list.append(occupancy_t)

            # Calculate MDRR weight: w_t = (v-1)/v^k-1 * v^(t-1)
            if self.v == 1.0:
                weight_t = 1.0 / self.k
            else:
                weight_t = ((self.v - 1) / (self.v**self.k - 1)) * (self.v ** (t - 1))

            if num_samples_opt <= num_samples_used + num_samples_t:
                truncated_samples = self._take_n_samples(
                    trajectories_t, int(num_samples_opt - num_samples_used)
                )
                sample_list.append(truncated_samples)
                break

            sample_list.append(trajectories_t)
            num_samples_used += num_samples_t
            cur_total_weight += weight_t

            if num_samples_used - cur_total_weight * num_samples_opt < 0:
                num_samples_opt = num_samples_used / cur_total_weight

        sample_list.reverse()
        occupancy_list.reverse()

        return sample_list, occupancy_list

    def _take_n_samples(self, trajectories, num_samples):
        if num_samples <= 0:
            return []

        result = []
        result_num_samples = 0

        for trajectory in trajectories:
            cur_num_samples = len(trajectory)

            if num_samples >= result_num_samples + cur_num_samples:
                result.append(trajectory)
                result_num_samples += cur_num_samples
            else:
                num_samples_left = num_samples - result_num_samples
                if num_samples_left > 0:
                    result.append(trajectory[:num_samples_left])
                return result

            if num_samples == result_num_samples:
                return result

        return result

    def _solve_ftrl_optimization(self, sample_list, occupancy_list):
        env = self.env
        agent = self.agents[1]
        num_states = env.dim
        num_actions = len(agent.actions)

        initial_d = np.ones((num_states, num_actions)) / (
            (1 - env.gamma) * num_states * num_actions
        )

        # FTRL tracking
        h_vals = np.empty((self.N, num_states))
        d_vals = np.empty((self.N, num_states, num_actions))

        num_samples = 0
        for trajectories in sample_list:
            for trajectory in trajectories:
                num_samples += len(trajectory)

        if num_samples == 0:
            print("No samples available for optimization")
            return None

        print(f"  FTRL optimization with {num_samples} total samples")

        for step in range(self.N):
            h = cp.Variable(num_states)
            constraints = [
                cp.pnorm(h, 2) <= 3 * num_states / cp.power((1 - env.gamma), 2)
            ]

            h_linear_factors = np.zeros(num_states)

            if step == 0:
                d_val_list = [initial_d]
            else:
                d_val_list = d_vals[:step]

            for i_step, trajectories in enumerate(sample_list):
                for trajectory in trajectories:
                    for state, action, next_state, reward in trajectory:
                        for d_val in d_val_list:
                            d_bar = occupancy_list[i_step]

                            if d_bar[state, action] > 1e-10:
                                h_linear_factors[state] -= d_val[state, action] / (
                                    d_bar[state, action] * (1 - env.gamma)
                                )
                                h_linear_factors[next_state] += (
                                    env.gamma
                                    * d_val[state, action]
                                    / (d_bar[state, action] * (1 - env.gamma))
                                )

            h_linear_factors /= num_samples
            h_linear_factors += env.rho

            lagrangian_h = cp.scalar_product(h, h_linear_factors)
            lagrangian_h += self.delta / 2 * cp.power(cp.pnorm(h, 2), 2)

            h_problem = cp.Problem(cp.Minimize(lagrangian_h), constraints)

            try:
                h_problem.solve(**self.solver_kwargs)
                h_val = h.value
                if h_val is None:
                    raise ValueError("h optimization failed")
            except:
                print(f"h optimization failed at step {step}")
                return None

            d = cp.Variable((num_states, num_actions), nonneg=True)
            constraints = []

            for s in range(num_states):
                for a in range(num_actions):
                    constraints.append(d[s, a] >= 0)
                    for occupancy in occupancy_list:
                        if occupancy[s, a] > 1e-10:
                            constraints.append(d[s, a] <= self.B * occupancy[s, a])

            d_linear_factors = np.zeros((num_states, num_actions))

            for i_step, trajectories in enumerate(sample_list):
                for trajectory in trajectories:
                    for state, action, next_state, reward in trajectory:
                        d_bar = occupancy_list[i_step]

                        if d_bar[state, action] > 1e-10:
                            advantage = (
                                reward - h_val[state] + env.gamma * h_val[next_state]
                            ) / (d_bar[state, action] * (1 - env.gamma))
                            d_linear_factors[state, action] += advantage

            d_linear_factors /= num_samples

            lagrangian_d = cp.scalar_product(d, d_linear_factors)
            lagrangian_d -= self.lamda / 2 * cp.power(cp.norm(d, "fro"), 2)

            d_problem = cp.Problem(cp.Maximize(lagrangian_d), constraints)

            try:
                d_problem.solve(**self.solver_kwargs)
                d_val = d.value
                if d_val is None:
                    raise ValueError("d optimization failed")
            except:
                print(f"d optimization failed at step {step}")
                return None

            h_vals[step] = h_val
            d_vals[step] = d_val

        return np.mean(d_vals, axis=0)

    def retrain2(self):
        env = self.env
        agent = self.agents[2]
        agent.policy = env.response_model(self.agents)
        return

    def retrain1(self):
        pass

    def retrain1_mdrr(self):
        pass
