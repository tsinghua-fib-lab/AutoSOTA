import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm


class MCTS:
    def __init__(self, exploration_coeff, algorithm, tau, alpha, number_of_atoms, step_size, gamma, update_type,
                 varde_temp=None, variance_floor=0.0, use_dp_value=True):
        self._exploration_coeff = exploration_coeff  # This is 'C' for the optimistic bonus
        self._algorithm = algorithm
        self._tau = tau
        self._alpha = alpha  # This is 'p' for power mean in the paper
        self._number_of_atoms = number_of_atoms
        self._step_size = step_size
        self._gamma = gamma  # discount factor
        self._update_type = update_type
        self._varde_temp = self._tau if varde_temp is None else varde_temp
        self._variance_floor = variance_floor
        self._use_dp_value = use_dp_value
        if algorithm == 'alpha-divergence' and alpha == 1:
            self._algorithm = 'ments'
        if algorithm == 'alpha-divergence' and alpha == 2:
            self._algorithm = 'tents'

    def run(self, tree_env, n_simulations):
        v_hat = np.zeros(n_simulations)
        regret = np.zeros_like(v_hat)
        for i in range(n_simulations):
            tree_env.reset()
            v_hat[i], regret[i] = self._simulation(tree_env)

        return v_hat, regret.cumsum()

    @staticmethod
    def _compute_prob_max(mean_list, sigma_list):
        n_actions = len(mean_list)
        lower_limit = mean_list - 8 * sigma_list
        upper_limit = mean_list + 8 * sigma_list
        epsilon = 1e-5
        _epsilon = 1e-25
        n_trapz = 100
        x = np.zeros(shape=(n_trapz, n_actions))
        y = np.zeros(shape=(n_trapz, n_actions))
        integrals = np.zeros(n_actions)
        for j in range(n_actions):
            if sigma_list[j] < epsilon:
                p = 1
                for k in range(n_actions):
                    if k != j:
                        p *= norm.cdf(mean_list[j], loc=mean_list[k], scale=sigma_list[k] + _epsilon)
                integrals[j] = p
            else:
                x[:, j] = np.linspace(lower_limit[j], upper_limit[j], n_trapz)
                y[:, j] = norm.pdf(x[:, j], loc=mean_list[j], scale=sigma_list[j] + _epsilon)
                for k in range(n_actions):
                    if k != j:
                        y[:, j] *= norm.cdf(x[:, j], loc=mean_list[k], scale=sigma_list[k] + _epsilon)
                integrals[j] = (upper_limit[j] - lower_limit[j]) / (2 * (n_trapz - 1)) * (y[0, j] + y[-1, j] + 2 * np.sum(y[1:-1, j]))
        with np.errstate(divide='raise'):
            try:
                return integrals / (np.sum(integrals))
            except FloatingPointError:
                print(integrals)
                print(mean_list)
                print(sigma_list)
                input()

    def _simulation(self, tree_env):
        path = self._navigate(tree_env)
        leaf_node = tree_env.tree.nodes[path[-1][1]]
        reward = tree_env.rollout(path[-1][1])

        if self._algorithm == "dng":
            cumulative_reward = 0

        leaf_node['V'] = (leaf_node['V'] * leaf_node['N'] + reward) / (leaf_node['N'] + 1)
        leaf_node['N'] += 1

        if self._algorithm == "w-mcts":
            leaf_node['v_mean'] = (leaf_node['v_mean'] * leaf_node['N'] + reward) / (leaf_node['N'] + 1)
            if (leaf_node['N'] == 1):
                leaf_node['v_variance'] = 1.0
            else:
                leaf_node['v_variance'] = (leaf_node['v_variance'] * (leaf_node['N'] - 1) + (reward - leaf_node['v_mean'])**2) / leaf_node['N']
        elif self._algorithm == "varde":
            leaf_node['dp'] = leaf_node['V']

        for e in reversed(path):
            current_node = tree_env.tree.nodes[e[0]]
            action = tree_env.tree.nodes[e[1]]
            next_state = e[2]
            next_node = tree_env.tree.nodes[e[2]]
            N = tree_env.tree[e[0]][e[2]]['N']
            Q = tree_env.tree[e[0]][e[2]]['Q']

            # Update Q value (empirical mean)
            tree_env.tree[e[0]][e[2]]['Q'] = (Q * N + next_node['V']) / (N + 1)
            tree_env.tree[e[0]][e[2]]['N'] += 1

            # Update value for a specific state-action pair
            updated_state_action = []
            for each_action, each_state, each_value in tree_env.tree.nodes[e[0]]['p_sa']:
                if each_action == action and each_state == next_state:
                    updated_state_action.append((each_action, each_action, each_value + 1))
                else:
                    updated_state_action.append((each_action, each_state, each_value))

            # Replace the old state_action list with the updated one
            tree_env.tree.nodes[e[0]]['p_sa'] = updated_state_action

            if self._algorithm == 'catso':
                # CATSO: Categorical Thompson Sampling with Optimistic Bonus
                # Calculate intermediate reward (Q_bar in the paper)
                intermediate_reward = reward + self._gamma * next_node['V']

                support = tree_env.tree[e[0]][e[2]]['support']
                dir_alpha = tree_env.tree[e[0]][e[2]]['dir_alpha']

                # Check if we need to expand the support
                if intermediate_reward < support[0] or intermediate_reward > support[-1]:
                    # Expand support to include new value
                    new_min = min(support[0], intermediate_reward)
                    new_max = max(support[-1], intermediate_reward)
                    new_support = np.linspace(new_min, new_max, self._number_of_atoms)

                    # Initialize new dir_alpha with ones (prior)
                    new_dir_alpha = np.ones(self._number_of_atoms)

                    # Map old atoms to nearest new atoms
                    for i, old_value in enumerate(support):
                        nearest_idx = np.argmin(np.abs(new_support - old_value))
                        # Add the old counts minus 1 (since we initialized with 1s)
                        new_dir_alpha[nearest_idx] += dir_alpha[i] - 1

                    tree_env.tree[e[0]][e[2]]['support'] = new_support
                    tree_env.tree[e[0]][e[2]]['dir_alpha'] = new_dir_alpha.tolist()

                # Find nearest atom and update its count
                nearest_bin_index = np.argmin(np.abs(tree_env.tree[e[0]][e[2]]['support'] - intermediate_reward))
                tree_env.tree[e[0]][e[2]]['dir_alpha'][nearest_bin_index] += 1

            elif self._algorithm == 'patso':
                # PATSO: Particle Thompson Sampling with Optimistic Bonus
                # Calculate intermediate reward (Q_bar in the paper)
                intermediate_reward = reward + self._gamma * next_node['V']

                support = tree_env.tree[e[0]][e[2]]['support']
                dir_alpha = tree_env.tree[e[0]][e[2]]['dir_alpha']

                # Check if particle already exists
                if intermediate_reward in support:
                    idx = support.index(intermediate_reward)
                    tree_env.tree[e[0]][e[2]]['dir_alpha'][idx] += 1
                else:
                    # Add new particle
                    tree_env.tree[e[0]][e[2]]['support'].append(intermediate_reward)
                    tree_env.tree[e[0]][e[2]]['dir_alpha'].append(1)

            elif self._algorithm == 'w-mcts':
                q_mean = tree_env.tree[e[0]][e[2]]['q_mean']
                q_variance = tree_env.tree[e[0]][e[2]]['q_variance']
                v_mean = next_node['v_mean']
                v_variance = next_node['v_variance']

                t = tree_env.tree[e[0]][e[2]]['N']
                _stepsize = 1. / np.power(t, self._step_size)

                tree_env.tree[e[0]][e[2]]['q_mean'] = _stepsize * q_mean + \
                                                      (1 - _stepsize) * (reward + self._gamma * v_mean)
                tree_env.tree[e[0]][e[2]]['q_variance'] = _stepsize * q_variance + \
                                                          (1 - _stepsize) * (self._gamma * v_variance)

                out_edges = [e for e in tree_env.tree.edges(e[0])]

                mean_next_all = [tree_env.tree[e[0]][e[1]]['q_mean'] for e in out_edges]
                variance_next_all = [tree_env.tree[e[0]][e[1]]['q_variance'] for e in out_edges]

                mean_next_all = np.array(mean_next_all)
                variance_next_all = np.array(variance_next_all)

                if self._update_type == 'max':
                    best = np.random.choice(np.argwhere(mean_next_all == np.max(mean_next_all)).ravel())
                    current_node['v_mean'] = mean_next_all[best]
                    current_node['v_variance'] = variance_next_all[best]
                else:
                    prob = self._compute_prob_max(mean_next_all, variance_next_all)
                    current_node['v_mean'] = np.sum(mean_next_all * prob)
                    current_node['v_variance'] = np.sum(variance_next_all * prob)

            elif self._algorithm == "dng":
                cumulative_reward = reward + self._gamma * cumulative_reward
                current_node["alpha"] += .5
                current_node["beta"] += .5 * (current_node["lambda"] * (cumulative_reward - current_node["mu"])**2
                                              / (current_node["lambda"] + 1))
                current_node["mu"] = ((current_node["lambda"] * current_node["mu"] + cumulative_reward)
                                      / (current_node["lambda"] + 1))
                current_node["lambda"] += 1
            elif self._algorithm == "varde":
                edge_data = tree_env.tree[e[0]][e[2]]
                ret_mean = edge_data.get('ret_mean', 0.0)
                ret_m2 = edge_data.get('ret_m2', 0.0)
                ret_n = edge_data.get('ret_n', 0)
                delta = reward - ret_mean
                ret_mean += delta / (ret_n + 1)
                delta2 = reward - ret_mean
                ret_m2 += delta * delta2
                edge_data['ret_mean'] = ret_mean
                edge_data['ret_m2'] = ret_m2
                edge_data['ret_n'] = ret_n + 1
                edge_data['dp'] = next_node.get('dp', next_node['V'])

                current_node['V'] = (current_node['V'] * current_node['N'] + reward) / (current_node['N'] + 1)

                out_edges = [oe for oe in tree_env.tree.edges(e[0])]
                dp_candidates = []
                for oe in out_edges:
                    ed = tree_env.tree[oe[0]][oe[1]]
                    if ed['N'] > 0:
                        dp_candidates.append(ed.get('dp', ed['Q']))
                if dp_candidates:
                    current_node['dp'] = np.max(dp_candidates)
            elif self._algorithm in {'uct', 'fixed-depth-mcts'}:
                current_node['V'] = (current_node['V'] * current_node['N'] +
                                     tree_env.tree[e[0]][e[2]]['Q']) / (current_node['N'] + 1)

            elif self._algorithm in {'power-uct', 'catso', 'patso'}:
                # Power mean backup for V nodes
                out_edges = [e for e in tree_env.tree.edges(e[0])]
                counts = np.array([tree_env.tree[e[0]][e[1]]['N'] for e in out_edges])
                qvalues = np.array([tree_env.tree[e[0]][e[1]]['Q'] for e in out_edges])

                # Avoid division by zero
                if np.sum(counts) > 0:
                    weights = counts / np.sum(counts)
                    # Power mean: (sum(w_i * q_i^p))^(1/p)
                    power_sum = np.sum(weights * np.power(qvalues, self._alpha))
                    current_node['V'] = np.power(power_sum, 1.0 / self._alpha)
                else:
                    current_node['V'] = 0

            else:
                out_edges = [e for e in tree_env.tree.edges(e[0])]
                qs = np.array([tree_env.tree[e[0]][e[1]]['Q'] for e in out_edges])

                if self._algorithm in {'max-ments', 'dents', 'bts'}:
                    current_node['V'] = np.max(qs)
                elif self._algorithm == 'ments':
                    current_node['V'] = self._tau * logsumexp(qs / self._tau)
                elif self._algorithm == 'rents':
                    qs_tau = qs / self._tau
                    weighted_logsumexp_qs = qs_tau.max() + np.log(
                        np.sum(current_node['prior'] * np.exp(qs_tau - qs_tau.max()))
                    )
                    current_node['V'] = self._tau * weighted_logsumexp_qs
                elif self._algorithm == 'tents':
                    q_tau = qs / self._tau
                    temp_q_tau = q_tau.copy()
                    sorted_q = np.flip(np.sort(temp_q_tau))
                    kappa = list()
                    for i in range(1, len(sorted_q) + 1):
                        if 1 + i * sorted_q[i - 1] > sorted_q[:i].sum():
                            idx = np.argwhere(temp_q_tau == sorted_q[i - 1]).ravel()[0]
                            temp_q_tau[idx] = np.nan
                            kappa.append(idx)
                    kappa = np.array(kappa)
                    sparse_max = q_tau[kappa] ** 2 / 2 - (q_tau[kappa].sum() - 1) ** 2 / (2 * len(kappa) ** 2)
                    sparse_max = sparse_max.sum() + .5
                    current_node['V'] = self._tau * sparse_max
                elif self._algorithm == 'alpha-divergence':
                    q_tau = qs / self._tau
                    temp_q_tau = q_tau.copy()

                    sorted_q = np.flip(np.sort(temp_q_tau))
                    kappa = list()
                    for i in range(1, len(sorted_q) + 1):
                        if 1 + i * sorted_q[i - 1] > sorted_q[:i].sum() + i * (1 - (1 / (self._alpha - 1))):
                            idx = np.argwhere(temp_q_tau == sorted_q[i - 1]).ravel()[0]
                            temp_q_tau[idx] = np.nan
                            kappa.append(idx)
                    kappa = np.array(kappa)
                    c_s_tau = ((q_tau[kappa].sum() - 1) / len(kappa)) + (1 - (1 / (self._alpha - 1)))
                    max_omega = np.maximum(q_tau - c_s_tau, np.zeros(len(q_tau)))
                    max_omega = np.power(max_omega * (self._alpha - 1), 1 / (self._alpha - 1))
                    max_omega = max_omega / np.sum(max_omega)
                    sparse_max_tmp = max_omega * q_tau
                    sparse_max = sparse_max_tmp.sum()
                    current_node['V'] = self._tau * sparse_max
                else:
                    raise ValueError

            current_node['N'] += 1

            # Update reward for next iteration in the path (for algorithms that accumulate rewards)
            if self._algorithm in ['catso', 'patso']:
                reward = intermediate_reward

        v_hat = 0
        if self._algorithm == 'w-mcts':
            v_hat = tree_env.tree.nodes[0]['v_mean']
        elif self._algorithm == 'dng':
            v_hat = tree_env.tree.nodes[0]['mu']
        else:
            v_hat = tree_env.tree.nodes[0]['V']

        max_a = self._select(tree_env=tree_env, state=0)
        regret = tree_env.q_root.max() - tree_env.q_root[max_a]
        return v_hat, regret

    def _navigate(self, tree_env):
        state = tree_env.state
        action = self._select(tree_env, state)
        next_state = tree_env.step(action)
        if next_state not in tree_env.leaves:
            return [[state, action, next_state]] + self._navigate(tree_env)
        else:
            return [[state, action, next_state]]

    def _select(self, tree_env, state):
        out_edges = [e for e in tree_env.tree.edges(state)]
        n_state_action = np.array([tree_env.tree[e[0]][e[1]]['N'] for e in out_edges])
        qs = np.array([tree_env.tree[e[0]][e[1]]['Q'] for e in out_edges])

        if self._algorithm == 'w-mcts':
            qvalues = []
            for edge in out_edges:
                # Sample from normal gamma distribution
                mu = tree_env.tree[edge[0]][edge[1]]['q_mean']
                delta = tree_env.tree[edge[0]][edge[1]]['q_variance']
                x = np.random.normal(mu, delta)
                qvalues.append(x)
            qvalues = np.array(qvalues)
            chosen_action = np.random.choice(np.argwhere(qvalues == np.max(qvalues)).ravel())
            return chosen_action

        elif self._algorithm == "dng":
            q_values = []
            for edge in out_edges:
                # Sample from normal gamma distribution
                mu = tree_env.tree.nodes[edge[1]]["mu"]
                alpha = tree_env.tree.nodes[edge[1]]["alpha"]
                beta = tree_env.tree.nodes[edge[1]]["beta"]

                tau = np.random.gamma(alpha, beta)
                x = np.random.normal(mu, np.sqrt(1 / tau))

                q_values.append(x)
            qvalues = []
            action = 0
            for e in out_edges:
                state = e[0]
                next_state = e[1]
                p_sa = []
                for each_action, each_state, each_value in tree_env.tree.nodes[state]['p_sa']:
                    if each_action == action and each_state == next_state:
                        p_sa.append(each_value)
                if np.array(p_sa).sum() == 0:
                    qvalues.append(np.inf)
                    continue

                p_sa = np.array(p_sa)
                alphas = p_sa / p_sa.sum()
                # sample from a dirichlet of p_sa
                samples = np.random.dirichlet(alphas)
                R = np.dot(samples, q_values)
                qvalues.append(R)

                action += 1

            chosen_action = np.random.choice(np.argwhere(qvalues == np.max(qvalues)).ravel())
            return chosen_action

        elif self._algorithm in {'uct', 'power-uct'}:
            n_state = np.sum(n_state_action)
            if n_state > 0:
                ucb_values = qs + self._exploration_coeff * np.sqrt(
                    np.log(n_state) / (n_state_action + 1e-10)
                )
            else:
                ucb_values = np.ones(len(n_state_action)) * np.inf

            chosen_action = np.random.choice(np.argwhere(ucb_values == np.max(ucb_values)).ravel())
            probs = np.zeros_like(ucb_values)
            probs[chosen_action] += 1

            return chosen_action

        elif self._algorithm in {'fixed-depth-mcts', 'stochastic-power-uct'}:
            n_state = np.sum(n_state_action)
            if n_state > 0:
                ucb_values = qs + self._exploration_coeff * np.sqrt(
                    np.sqrt(n_state) / (n_state_action + 1e-10)
                )
            else:
                ucb_values = np.ones(len(n_state_action)) * np.inf

            chosen_action = np.random.choice(np.argwhere(ucb_values == np.max(ucb_values)).ravel())
            probs = np.zeros_like(ucb_values)
            probs[chosen_action] += 1

            return chosen_action

        elif self._algorithm == 'catso':
            # CATSO: Thompson Sampling with Optimistic Bonus
            phi_values = []
            n_state = np.sum(n_state_action)

            for i, edge in enumerate(out_edges):
                support = np.array(tree_env.tree[edge[0]][edge[1]]['support'])
                dir_alpha = np.array(tree_env.tree[edge[0]][edge[1]]['dir_alpha'])

                # Sample from Dirichlet distribution
                if len(dir_alpha) > 0:
                    L = np.random.dirichlet(dir_alpha)
                    # Compute phi = support^T * L (weighted sum)
                    phi = np.dot(support, L)
                else:
                    # If no samples yet, use a default value
                    phi = 0.0

                # Add optimistic bonus: B(n, s, a) = C * n_s^(1/4) / n_sa^(1/2)
                if n_state_action[i] > 0 and n_state > 0:
                    bonus = self._exploration_coeff * np.power(n_state, 0.25) / np.power(n_state_action[i], 0.5)
                else:
                    bonus = np.inf  # Ensure unvisited actions are tried

                phi_with_bonus = phi + bonus
                phi_values.append(phi_with_bonus)

            # Select action with maximum phi + bonus
            phi_values = np.array(phi_values)
            chosen_action = np.random.choice(np.argwhere(phi_values == np.max(phi_values)).ravel())
            return chosen_action

        elif self._algorithm == 'patso':
            # PATSO: Thompson Sampling with Optimistic Bonus
            phi_values = []
            n_state = np.sum(n_state_action)

            for i, edge in enumerate(out_edges):
                support = tree_env.tree[edge[0]][edge[1]]['support']
                dir_alpha = tree_env.tree[edge[0]][edge[1]]['dir_alpha']

                # Sample from Dirichlet distribution
                if len(dir_alpha) > 0 and len(support) > 0:
                    L = np.random.dirichlet(dir_alpha)
                    # Compute phi = support^T * L (weighted sum)
                    phi = np.dot(support, L)
                else:
                    # If no samples yet, use a default value
                    phi = 0.0

                # Add optimistic bonus: B(n, s, a) = C * n_s^(1/4) / n_sa^(1/2)
                if n_state_action[i] > 0 and n_state > 0:
                    bonus = self._exploration_coeff * np.power(n_state, 0.25) / np.power(n_state_action[i], 0.5)
                else:
                    bonus = np.inf  # Ensure unvisited actions are tried

                phi_with_bonus = phi + bonus
                phi_values.append(phi_with_bonus)

            # Select action with maximum phi + bonus
            phi_values = np.array(phi_values)
            chosen_action = np.random.choice(np.argwhere(phi_values == np.max(phi_values)).ravel())
            return chosen_action

        elif self._algorithm == 'varde':
            unvisited = []
            for i, edge in enumerate(out_edges):
                if tree_env.tree[edge[0]][edge[1]]['N'] == 0:
                    unvisited.append(i)
            if unvisited:
                return np.random.choice(unvisited)

            edge_data = [tree_env.tree[edge[0]][edge[1]] for edge in out_edges]
            if self._use_dp_value:
                q_est = np.array([ed.get('dp', ed['Q']) for ed in edge_data], dtype=float)
            else:
                q_est = np.array([ed.get('ret_mean', ed['Q']) for ed in edge_data], dtype=float)

            temp = max(self._varde_temp, 1e-8)
            scaled = q_est / temp
            max_scaled = np.max(scaled)
            weights = np.exp(scaled - max_scaled)
            denom = np.sum(weights)
            if denom <= 0 or not np.isfinite(denom):
                influence = np.ones_like(weights) / len(weights)
            else:
                influence = weights / denom

            variances = []
            for ed in edge_data:
                ret_n = ed.get('ret_n', 0)
                if ret_n < 2:
                    variances.append(np.inf)
                else:
                    ret_m2 = ed.get('ret_m2', 0.0)
                    variances.append(ret_m2 / (ret_n - 1))
            variances = np.array(variances, dtype=float)
            effective_variance = np.maximum(variances, self._variance_floor)

            n_visits = np.array([ed['N'] for ed in edge_data], dtype=float)
            exploration_bonus = 1.0 / (n_visits * (n_visits + 1.0))
            scores = influence * effective_variance * exploration_bonus

            if np.any(np.isinf(scores)):
                candidates = np.argwhere(np.isinf(scores)).ravel()
            else:
                max_score = np.max(scores)
                candidates = np.argwhere(scores == max_score).ravel()
            return np.random.choice(candidates)

        else:
            n_actions = len(out_edges)
            lambda_coeff = np.clip(self._exploration_coeff * n_actions / np.log(
                np.sum(n_state_action) + 1 + 1e-10), 0, 1)

            if self._algorithm in {'ments', 'dents', 'bts', 'max-ments'}:
                q_exp_tau = np.exp(qs / self._tau)
                probs = (1 - lambda_coeff) * q_exp_tau / q_exp_tau.sum() + lambda_coeff / n_actions
            elif self._algorithm == 'rents':
                qs_tau = qs / self._tau
                prior_q_exp_tau = tree_env.tree.nodes[state]['prior'] * np.exp(qs_tau - qs_tau.max())
                probs = (1 - lambda_coeff) * prior_q_exp_tau / (prior_q_exp_tau.sum()) + lambda_coeff / n_actions
            elif self._algorithm == 'tents':
                q_tau = qs / self._tau
                temp_q_tau = q_tau.copy()

                sorted_q = np.flip(np.sort(temp_q_tau))
                kappa = list()
                for i in range(1, len(sorted_q) + 1):
                    if 1 + i * sorted_q[i - 1] > sorted_q[:i].sum():
                        idx = np.argwhere(temp_q_tau == sorted_q[i - 1]).ravel()[0]
                        temp_q_tau[idx] = np.nan
                        kappa.append(idx)
                kappa = np.array(kappa)

                max_omega = np.maximum(q_tau - (q_tau[kappa].sum() - 1) / len(kappa),
                                       np.zeros(len(q_tau)))
                probs = (1 - lambda_coeff) * max_omega + lambda_coeff / n_actions
            elif self._algorithm == 'alpha-divergence':
                q_tau = qs / self._tau
                temp_q_tau = q_tau.copy()

                sorted_q = np.flip(np.sort(temp_q_tau))
                kappa = list()
                for i in range(1, len(sorted_q) + 1):
                    if 1 + i * sorted_q[i - 1] > sorted_q[:i].sum() + i * (1 - (1 / (self._alpha - 1))):
                        idx = np.argwhere(temp_q_tau == sorted_q[i - 1]).ravel()[0]
                        temp_q_tau[idx] = np.nan
                        kappa.append(idx)
                kappa = np.array(kappa)
                c_s_tau = ((q_tau[kappa].sum() - 1) / len(kappa)) + (1 - (1 / (self._alpha - 1)))

                max_omega = np.maximum(q_tau - c_s_tau, np.zeros(len(q_tau)))
                max_omega = np.power(max_omega * (self._alpha - 1), 1 / (self._alpha - 1))
                max_omega = max_omega / np.sum(max_omega)
                probs = (1 - lambda_coeff) * max_omega + lambda_coeff / n_actions
            else:
                raise ValueError

            return np.random.choice(np.arange(n_actions), p=probs)
