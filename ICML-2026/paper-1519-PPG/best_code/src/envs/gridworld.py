import copy
import itertools
import json

import numpy as np
from numpy.testing import assert_almost_equal

from src.agents.gw_agent import Agent
from src.policies.policies import *


class Gridworld:
    def __init__(
        self,
        beta,
        eps,
        gamma,
        num_followers,
        sampling=False,
        n_sample=500,
        seed=1,
        max_sample_steps=100,
    ):
        # grid
        h = -0.5
        f = -0.02
        self.grid = np.array(
            [
                [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01],
                [-0.01, -0.01, f, -0.01, h, -0.01, -0.01, -0.01],
                [-0.01, -0.01, -0.01, h, -0.01, -0.01, f, -0.01],
                [-0.01, f, -0.01, -0.01, -0.01, h, -0.01, f],
                [-0.01, -0.01, -0.01, h, -0.01, -0.01, f, -0.01],
                [-0.01, h, h, -0.01, f, -0.01, h, -0.01],
                [-0.01, h, -0.01, -0.01, h, -0.01, h, -0.01],
                [-0.01, -0.01, -0.01, h, -0.01, f, -0.01, +1],
            ]
        )

        self.dim = np.prod(self.grid.shape)
        self.state_ids = range(self.dim)
        self.terminal_state_id = self.dim - 1
        self.reward_space = [-0.01, f, h, 1]
        # initial states
        self.initial_states = [
            s
            for s in self.state_ids
            if (state := self._get_state(s)) and (state[0] == 0 or state[1] == 0)
        ]
        self.rho = self._get_initial_state_distribution()
        # move: left, right, up, down
        self.moves = [0, 1, 2, 3]
        self.move_mapping = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        self.move_meaning = {0: "left", 1: "right", 2: "up", 3: "down"}
        # intervene: left, right, up, down
        self.interventions = [0, 1, 2, 3, 4]
        self.interventions_mapping = {
            move: self.move_mapping[move] for move in self.moves
        }
        self.interventions_mapping[4] = False
        self.interventions_meaning = {
            move: self.move_meaning[move] for move in self.moves
        }
        self.interventions_meaning[4] = "nope"
        # cost of intervention
        self.C = -0.05
        # agents
        self.num_of_agents = 2
        self.agents = {}
        self.action_space = {}
        # agent 1 is the main agent (controls actor movement based on the correct gird)
        self.action_space[1] = self.moves
        self.agents[1] = Agent(1, self.action_space[1])
        # agent 2 is the secondary agent (makes interventions based on one or more pertrubed grids) and it is considered as part of the environment
        # followers are basically integrated in agent 2
        self.action_space[2] = self.interventions
        self.agents[2] = Agent(2, self.action_space[2])
        # pertrub grids for agent 2/followers -- the pertrubed grids stay the same for all experiments
        self.num_followers = num_followers
        self.eps = eps
        self.pertrubed_grids = []
        for pertrubation_seed in range(num_followers):
            pertrubation_rng = np.random.default_rng(pertrubation_seed)
            self.pertrubed_grids.append(self.pertrub_grid(pertrubation_rng))
        # pertrubed grid that computational functions solve for
        self.active_pertrubed_grid = self.pertrubed_grids[0]
        # parameters
        self.gamma = gamma
        self.beta = beta
        # sampling
        self.sampling = sampling
        self.n_sample = n_sample
        self.max_sample_steps = max_sample_steps
        # set random generator (for trajectory sampling)
        self.rng = np.random.default_rng(seed)

        self.reset()

    def reset(self):
        """ """
        self.initialize_policies()

        return

    def get_next_state(self, state_id, move):
        """ """
        assert move in self.moves, f"Illegal move {move} was given."

        state = self._get_state(state_id)

        if self.is_terminal(state_id):
            return state_id

        next_state = self.vector_add(state, self.move_mapping[move])

        if not self.is_valid(next_state):
            next_state = state

        next_state_id = self._get_state_id(next_state)

        return next_state_id

    def get_rewards(self, state_id, next_state_id):
        """ """
        rewards = {}

        if self.is_terminal(state_id):
            rewards[1] = 0
            rewards[2] = 0
            return rewards

        state = self._get_state(next_state_id)
        # agent 1
        rewards[1] = self.grid[state[0], state[1]]
        # agent 2
        rewards[2] = self.active_pertrubed_grid[state[0], state[1]]

        return rewards

    def _get_initial_state_distribution(self):
        """ """
        rho = np.zeros(self.dim, dtype="float64")
        initial_states = self.initial_states
        for s in initial_states:
            rho[s] = 1 / len(initial_states)

        return rho

    def value_iteration(self, agent, tol=1e-5):
        """ """
        gamma = self.gamma

        # initialize all state values to zero
        U = np.zeros(self.dim, dtype="float64")
        while True:
            U_old = copy.deepcopy(U)
            delta = 0
            for s in self.state_ids:
                if self.is_terminal(s):
                    U[s] = 0
                else:
                    bell = np.zeros_like(self.moves, dtype="float64")
                    for move in self.moves:
                        s_pr = self.get_next_state(s, move)
                        r = self.get_rewards(s, s_pr)
                        bell[move] = r[agent.id] + gamma * U_old[s_pr]
                    U[s] = np.max(bell)
                delta = max(delta, abs(U[s] - U_old[s]))

            if delta < tol:
                break

        return U

    def _get_Q(self, U, agent):
        """ """
        gamma = self.gamma

        # initialize all state-move values to zero
        Q = np.zeros(shape=(self.dim, len(self.moves)), dtype="float64")
        for s, move in itertools.product(self.state_ids, self.moves):
            if self.is_terminal(s):
                Q[s, move] = 0
            else:
                s_pr = self.get_next_state(s, move)
                r = self.get_rewards(s, s_pr)
                Q[s, move] = r[agent.id] + gamma * U[s_pr]

        return Q

    def get_mnext_state(self, state_id, actions):
        """
        actions: dict {agent.id: action}
        """
        for agent in self.agents.values():
            assert actions[agent.id] in agent.actions, (
                f"Illegal action {actions[agent.id]} was given."
            )

        if not self.intervention_happened(actions):
            next_state_id = self.get_next_state(state_id, actions[1])
        else:
            next_state_id = self.get_next_state(state_id, actions[2])

        return next_state_id

    def get_mrewards(self, state_id, actions, next_state_id):
        """ """
        rewards = self.get_rewards(state_id, next_state_id)
        if self.intervention_happened(actions):
            rewards[2] += self.C

        return rewards

    def _build_next_state_table(self):
        """Precompute next_state[s, a1, a2] for fast value iteration."""
        if hasattr(self, '_next_state_cache'):
            return self._next_state_cache
        n_states = self.dim
        n_a1 = len(self.moves)
        n_a2 = len(self.interventions)
        ns_table = np.zeros((n_states, n_a1, n_a2), dtype=np.int32)
        for s in self.state_ids:
            for a1_idx, a1 in enumerate(self.moves):
                for a2_idx, a2 in enumerate(self.interventions):
                    actions = {1: a1, 2: a2}
                    ns_table[s, a1_idx, a2_idx] = self.get_mnext_state(s, actions)
        self._next_state_cache = ns_table
        return ns_table

    def _build_reward_table(self, agent_id=2):
        """Precompute reward[s, a1, a2] for agent_id given current perturbed grid."""
        n_states = self.dim
        n_a1 = len(self.moves)
        n_a2 = len(self.interventions)
        r_table = np.zeros((n_states, n_a1, n_a2), dtype="float64")
        ns_table = self._build_next_state_table()
        for s in self.state_ids:
            for a1_idx in range(n_a1):
                for a2_idx in range(n_a2):
                    actions = {1: self.moves[a1_idx], 2: self.interventions[a2_idx]}
                    s_pr = ns_table[s, a1_idx, a2_idx]
                    r_dict = self.get_mrewards(s, actions, s_pr)
                    r_table[s, a1_idx, a2_idx] = r_dict[agent_id]
        return r_table

    def best_response_value_iteration(self, agent, fixed_agent, tol=1e-5):
        """Vectorized best-response value iteration."""
        gamma = self.gamma
        ns_table = self._build_next_state_table()  # [S, A1, A2]
        r_table = self._build_reward_table(agent.id)  # [S, A1, A2]
        is_term = np.array([self.is_terminal(s) for s in range(self.dim)])

        # Precompute agent1 probs for all states [S, A1]
        a1_probs = np.array([fixed_agent._get_probs(s) for s in range(self.dim)])

        U = np.zeros(self.dim, dtype="float64")
        while True:
            U_old = U.copy()
            # Vectorized Bellman for all non-terminal states
            # expected[s, a1, a2] = r[s,a1,a2] + gamma * U_old[next_state[s,a1,a2]]
            expected = r_table + gamma * U_old[ns_table]  # [S, A1, A2]
            # bell[s, a2] = sum_{a1} p[a1] * expected[s, a1, a2]
            bell = np.einsum('sa,sab->sb', a1_probs, expected)  # [S, A2]
            # Optimal value
            U_new = np.max(bell, axis=1)  # [S]
            U_new[is_term] = 0.0
            delta = np.max(np.abs(U_new - U_old))
            U = U_new
            if delta < tol:
                break
        return U

    def _get_mU(self, agent, fixed_agent, tol=1e-5):
        """Vectorized expected value function (uses agent's policy, not optimal)."""
        gamma = self.gamma
        ns_table = self._build_next_state_table()
        r_table = self._build_reward_table(agent.id)
        is_term = np.array([self.is_terminal(s) for s in range(self.dim)])

        a1_probs = np.array([fixed_agent._get_probs(s) for s in range(self.dim)])
        a2_probs = np.array([agent._get_probs(s) for s in range(self.dim)])

        U = np.zeros(self.dim, dtype="float64")
        while True:
            U_old = U.copy()
            expected = r_table + gamma * U_old[ns_table]  # [S, A1, A2]
            bell = np.einsum('sa,sab->sb', a1_probs, expected)  # [S, A2]
            U_new = np.sum(a2_probs * bell, axis=1)  # [S]
            U_new[is_term] = 0.0
            delta = np.max(np.abs(U_new - U_old))
            U = U_new
            if delta < tol:
                break
        return U

    def _get_mQ(self, U, agent, fixed_agent):
        """Vectorized Q-value computation."""
        gamma = self.gamma
        ns_table = self._build_next_state_table()
        r_table = self._build_reward_table(agent.id)

        a1_probs = np.array([fixed_agent._get_probs(s) for s in range(self.dim)])

        # expected[s, a1, a2] = r[s,a1,a2] + gamma * U[next_state]
        expected = r_table + gamma * U[ns_table]  # [S, A1, A2]
        Q = np.einsum('sa,sab->sb', a1_probs, expected)  # [S, A2]
        for s in self.state_ids:
            if self.is_terminal(s):
                Q[s, :] = 0
        return Q

    def _get_RT(self):
        """
        Get R and T functions for agent 1, given the fixed policy of agent 2.
        R(s,a) is the expected reward that agent 1 will receive after taking action a in state s
        T(s, a, s_pr) is the probability of the actor transitioning from state s to state s_pr after agent 1 takes action a
        """
        if not self.sampling:
            return self._get_exact_RT()
        else:
            return self._get_approximate_RT()

    def _get_exact_RT(self):
        """ """
        agent = self.agents[1]
        fixed_agent = self.agents[2]

        T = np.zeros(shape=(self.dim, len(agent.actions), self.dim), dtype="float64")
        R = np.zeros(shape=(self.dim, len(agent.actions)), dtype="float64")
        for s, a in itertools.product(self.state_ids, agent.actions):
            p = fixed_agent._get_probs(s)
            for fa in fixed_agent.actions:
                actions = {fixed_agent.id: fa, agent.id: a}
                s_pr = self.get_mnext_state(s, actions)
                T[s, a, s_pr] += p[fa]
                r = self.get_mrewards(s, actions, s_pr)
                R[s, a] += p[fa] * r[agent.id]

            assert_almost_equal(np.sum(T[s, a, :]), 1)

        return R, T

    def _get_approximate_RT(self):
        """ """
        agent = self.agents[1]
        n_sample = self.n_sample

        # total values
        T_tot = np.zeros(
            shape=(self.dim, len(agent.actions), self.dim), dtype="float64"
        )
        R_tot = np.zeros(shape=(self.dim, len(agent.actions)), dtype="float64")
        # visitattion count
        V = np.zeros(shape=(self.dim, len(agent.actions)), dtype="int")
        # compute empirical data
        for _ in range(n_sample):
            trajectory = self.sample_trajectory()
            for s, a, s_pr, r in trajectory:
                # update visitation count
                V[s, a] += 1
                # update total values
                T_tot[s, a, s_pr] += 1
                R_tot[s, a] += r

        # approximated values
        T_hat = np.zeros(
            shape=(self.dim, len(agent.actions), self.dim), dtype="float64"
        )
        R_hat = np.zeros(shape=(self.dim, len(agent.actions)), dtype="float64")
        # remove +1 from reward space
        reward_space = [r for r in self.reward_space if r != 1]
        for s in self.state_ids:
            r_states = self._get_reachable_states(s)
            for a in agent.actions:
                if self.is_terminal(s):
                    R_hat[s, a] = 0
                    T_hat[s, a, self.terminal_state_id] = 1
                elif V[s, a]:
                    R_hat[s, a] = R_tot[s, a] / V[s, a]
                    for s_pr in r_states:
                        T_hat[s, a, s_pr] = T_tot[s, a, s_pr] / V[s, a]
                else:
                    # if s, a not seen during sampling assign to R[s,a] the mean of all negative rewards
                    R_hat[s, a] = np.mean(reward_space)
                    # if if s, a not seen during sampling assign to T[s, a] the uniform distiribution over all reachable states
                    for s_pr in r_states:
                        T_hat[s, a, s_pr] = 1 / len(r_states)

                assert_almost_equal(np.sum(T_hat[s, a, :]), 1)

        return R_hat, T_hat

    def _get_d(self, T, agent, tol=1e-5):
        """
        Computes state-action occupancy measure relative to some agent and transitions probability T
        """
        gamma = self.gamma
        rho = self.rho

        d = np.zeros(shape=(self.dim, len(agent.actions)), dtype="float64")
        while True:
            d_old = copy.deepcopy(d)
            delta = 0
            for s, a in itertools.product(self.state_ids, agent.actions):
                if self.is_terminal(s):
                    d[s, a] = 0
                else:
                    p = agent._get_probs(s)
                    d[s, a] = p[a] * (rho[s] + gamma * np.sum(d_old * T[:, :, s]))
                delta = max(delta, abs(d[s, a] - d_old[s, a]))
            if delta < tol:
                break

        return d

    def is_valid(self, state):
        """ """
        grid_shape = self.grid.shape

        return bool(0 <= state[0] < grid_shape[0] and 0 <= state[1] < grid_shape[1])

    def is_terminal(self, state_id):
        """ """

        return bool(state_id == self.terminal_state_id)

    def intervention_happened(self, actions):
        """ """

        return bool(self.interventions_mapping[actions[2]])

    def _get_state(self, state_id):
        """ """
        grid_shape = self.grid.shape
        state = (state_id // grid_shape[0], state_id % grid_shape[1])

        return state

    def _get_state_id(self, state):
        """ """
        grid_shape = self.grid.shape
        state_id = state[0] * grid_shape[0] + state[1]

        return state_id

    def _get_reachable_states(self, state_id):
        """ """
        r_states = []

        for move in self.moves:
            next_state_id = self.get_next_state(state_id, move)
            if next_state_id not in r_states:
                r_states.append(next_state_id)

        return r_states

    def pertrub_grid(self, pertrubation_rng):
        """ """
        rng = pertrubation_rng
        grid_shape = self.grid.shape

        grid_values = np.unique(self.grid)
        grid_values = np.delete(grid_values, np.where(grid_values == 1))

        new_grid = np.zeros(shape=grid_shape)
        r = rng.random(grid_shape)
        for i, j in itertools.product(range(grid_shape[0]), range(grid_shape[1])):
            # initial states stay the same
            if self._get_state_id([i, j]) in self.initial_states or r[i, j] > self.eps:
                new_grid[i, j] = self.grid[i, j]
            else:
                new_grid[i, j] = rng.choice(grid_values)

        # goal value remains the same
        new_grid[-1, -1] = self.grid[-1, -1]

        return new_grid

    def sample_trajectory(self):
        """ """
        agent = self.agents[1]
        fixed_agent = self.agents[2]
        rho = self.rho
        rng = self.rng

        # trajectory quartets (state, action, next state, reward)
        trajectory = []
        n_steps = 0
        # initial state
        s = rng.choice(np.arange(self.dim), p=rho)
        while not self.is_terminal(s) and n_steps < self.max_sample_steps:
            # actions
            actions = {}
            actions[agent.id] = agent.take_action(s, rng)
            actions[fixed_agent.id] = fixed_agent.take_action(s, rng)
            a = actions[agent.id]
            # next state
            s_pr = self.get_mnext_state(s, actions)
            # rewards
            r = self.get_mrewards(s, actions, s_pr)
            # update trajectory
            trajectory.append((s, a, s_pr, r[agent.id]))
            # prepare for next time-step
            s = s_pr
            n_steps += 1

        return trajectory

    def _get_policy_array(self, agent):
        """
        Given the (initial) agent's policy return the probability array
        """
        pi = np.zeros(shape=(self.dim, len(agent.actions)), dtype="float64")

        for s in self.state_ids:
            p = agent._get_probs(s)
            for a in agent.actions:
                pi[s, a] = p[a]

        return pi

    def initialize_policies(self):
        """ """
        agents = self.agents

        # agent 1
        agent = agents[1]

        U_star = self.value_iteration(agent)
        Q_star = self._get_Q(U_star, agent)
        agent.policy = Eps_Greedy_Policy(agent.actions, Q_star)

        # agent 2
        agent = agents[2]

        agent.policy = self.response_model(agents)

        # store initial visualization of env (seed-specific to prevent conflicts)
        seed_str = str(getattr(self.rng, '_seed', '0'))
        config_name = "initial_" + f"beta={self.beta}_gamma={self.gamma}_seed={seed_str}"
        vis = self._get_env_vis()
        with open(f"limiting_envs/{config_name}.json", "w") as f:
            json.dump(vis, f, indent=4)

        return

    def response_model(self, agents):
        """
        Computes response policy of agent 2 (environment) w.r.t. the policy of agent 1
        """
        fixed_agent = agents[1]
        agent = agents[2]

        # compute best response Q values for all pertrubed grids
        lst_Q_br = []
        for pertrubed_grid in self.pertrubed_grids:
            self.active_pertrubed_grid = pertrubed_grid
            U_br = self.best_response_value_iteration(agent, fixed_agent)
            lst_Q_br.append(self._get_mQ(U_br, agent, fixed_agent))
        # compute the average over all Q_br
        avg_Q_br = np.average(lst_Q_br, axis=0)
        # apply softmax
        response_policy = SoftMaxQ_Policy(agent.actions, avg_Q_br, self.beta)

        return response_policy

    def _get_env_vis(self):
        """
        Returns a visualization of the environment/policy of agent 2
        """
        agent = self.agents[2]
        grid_shape = self.grid.shape

        vis = []
        for i in range(grid_shape[0]):
            vis_row = []
            for j in range(grid_shape[1]):
                s = self._get_state_id([i, j])
                p = agent._get_probs(s)
                a = np.argmax(p)
                # a should have probability higher than nope
                if p[a] == p[4]:
                    a = 4
                vis_row.append(self.interventions_meaning[a])
            vis.append(vis_row)

        return vis

    @staticmethod
    def vector_add(x, y):
        return (x[0] + y[0], x[1] + y[1])
