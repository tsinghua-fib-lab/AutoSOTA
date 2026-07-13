import networkx as nx
import numpy as np

import random

from scipy.special import logsumexp

def equally_random_number(k):
    return random.randint(0, k - 1)

def random_number(k):
    # Randomly choose between 0 and other numbers
    if random.random() < 0.5:
        return 0
    else:
        return random.randint(1, k - 1)

class SyntheticTree:
    def __init__(self, k, d, algorithm, tau, alpha, number_of_atom, gamma, step_size):
        self._k = k
        self._d = d
        self._algorithm = algorithm
        self._tau = tau
        self._alpha = alpha
        self._number_of_atom = number_of_atom
        self._gamma = gamma
        self._step_size = step_size

        if algorithm == 'alpha-divergence' and alpha == 1:
            self._algorithm = 'ments'

        if algorithm == 'alpha-divergence' and alpha == 2:
            self._algorithm = 'tents'

        if alpha == 0:
            self._algorithm = 'uct'
        elif alpha == -1:
            self._algorithm = 'dng'
        elif alpha == -2:
            self._algorithm = 'ments'

        self._tree = nx.balanced_tree(k, d, create_using=nx.DiGraph)
        random_weights = np.random.rand(len(self._tree.edges))
        for i, e in enumerate(self._tree.edges):
            self._tree[e[0]][e[1]]['weight'] = random_weights[i]
            self._tree[e[0]][e[1]]['N'] = 0.
            self._tree[e[0]][e[1]]['Q'] = 0.

            if algorithm == "w-mcts":
                self._tree[e[0]][e[1]]['q_mean'] = 0.
                self._tree[e[0]][e[1]]['q_variance'] = 0.
            elif algorithm == "catso":
                self._tree[e[0]][e[1]]['dir_alpha'] = [1] * self._number_of_atom
                self._tree[e[0]][e[1]]['min_value'] = 0
                self._tree[e[0]][e[1]]['max_value'] = 0.0001
                self._tree[e[0]][e[1]]['support'] = np.linspace(self._tree[e[0]][e[1]]['min_value'],
                                                                self._tree[e[0]][e[1]]['max_value'],
                                                                self._number_of_atom)
            elif algorithm == "patso":
                self._tree[e[0]][e[1]]['dir_alpha'] = []
                self._tree[e[0]][e[1]]['support'] = []
            elif algorithm == "varde":
                self._tree[e[0]][e[1]]['ret_mean'] = 0.
                self._tree[e[0]][e[1]]['ret_m2'] = 0.
                self._tree[e[0]][e[1]]['ret_n'] = 0
                self._tree[e[0]][e[1]]['dp'] = 0.

        for n in self._tree.nodes:
            self._tree.nodes[n]['N'] = 0.
            self._tree.nodes[n]['V'] = 0.

            edges = [e for e in self._tree.edges(n)]
            number_of_actions = len(edges)
            state_action = []
            for action in range(number_of_actions):
                state = edges[action][1]
                state_action.append((action, state, 0))

            self._tree.nodes[n]['p_sa'] = state_action

            if algorithm == "w-mcts":
                self._tree.nodes[n]['v_mean'] = 0.
                self._tree.nodes[n]['v_variance'] = 0.
            elif algorithm == "dng":
                self._tree.nodes[n]["mu"] = 0.
                self._tree.nodes[n]["lambda"] = 1e-2
                self._tree.nodes[n]["alpha"] = 1.
                self._tree.nodes[n]["beta"] = 100.
            elif algorithm == "varde":
                self._tree.nodes[n]['dp'] = 0.
        self.leaves = [x for x in self._tree.nodes() if
                       self._tree.out_degree(x) == 0 and self._tree.in_degree(x) == 1]

        self._compute_mean()
        means = np.array([self._tree.nodes[n]['mean'] for n in self.leaves])
        means = (means - means.min()) / (means.max() - means.min()) if len(means) > 1 else [0.]
        for i, n in enumerate(self.leaves):
            self._tree.nodes[n]['mean'] = means[i]

        self._assign_priors_maxs()
        self.max_mean = self._tree.nodes[0]['mean']
        self.optimal_v_root, self.q_root = self._solver()
        self.state = None
        self.reset()

    def reset(self, state=None):
        if state is not None:
            self.state = state
        else:
            self.state = 0

        return self.state

    def calculate_probabilities(self, K, idx):
        element = 0.5 / (K-1)  # Calculate the value for other elements

        # Create the list
        my_list = [element if i != idx else 0.5 for i in range(K)]

        return my_list

    def step(self, action):
        edges = [e for e in self._tree.edges(self.state)]
        number_of_actions = len(edges)
        prob = self.calculate_probabilities(number_of_actions, action)
        rand_action = np.random.choice(number_of_actions, p=prob)

        self.state = edges[rand_action][1]

        return self.state

    def rollout(self, state):
        return np.random.normal(self._tree.nodes[state]['mean'], scale=.5)

    @property
    def tree(self):
        return self._tree

    def _compute_mean(self, node=0, weight=0):
        if node not in self.leaves:
            for e in self._tree.edges(node):
                self._compute_mean(e[1],
                                   weight + self._tree[e[0]][e[1]]['weight'])
        else:
            self._tree.nodes[node]['mean'] = weight

    def _compute_v(self, means):
        mean_list = []
        num_action = len(means)
        for mean in means:
            temp_sum = sum(means) - mean
            mean_list.append(0.5*mean + (0.5 * temp_sum)/(num_action - 1))
        return max(mean_list)

    def _assign_priors_maxs(self, node=0):
        successors = [n for n in self._tree.successors(node)]
        if successors[0] not in self.leaves:
            means = np.array([self._assign_priors_maxs(s) for s in successors])

            max_mean  = self._compute_v(means)

            self._tree.nodes[node]['prior'] = means / means.sum()
            self._tree.nodes[node]['mean'] = max_mean

            return max_mean
        else:
            means = np.array([self._tree.nodes[s]['mean'] for s in successors])
            max_mean  = self._compute_v(means)

            self._tree.nodes[node]['prior'] = means / means.sum()
            self._tree.nodes[node]['mean'] = max_mean

            return max_mean

    def _solver(self, node=0):
        # For algorithms that use power mean backup
        if self._algorithm in {'catso', 'patso'}:
            successors = [n for n in self._tree.successors(node)]
            means = np.array([self._tree.nodes[s]['mean'] for s in successors])
            return self.max_mean, means

        elif self._algorithm == 'w-mcts':
            successors = [n for n in self._tree.successors(node)]
            means = np.array([self._tree.nodes[s]['mean'] for s in successors])
            return self.max_mean, means

        elif self._algorithm == 'dng':
            successors = [n for n in self._tree.successors(node)]
            means = np.array([self._tree.nodes[s]['mean'] for s in successors])
            return self.max_mean, means

        elif self._algorithm in {'uct', 'power-uct', 'fixed-depth-mcts', 'max-ments', 'dents', 'bts', 'varde'}:
            successors = [n for n in self._tree.successors(node)]
            means = np.array([self._tree.nodes[s]['mean'] for s in successors])
            return self.max_mean, means

        else:
            successors = [n for n in self._tree.successors(node)]
            if self._algorithm == 'ments':
                if successors[0] in self.leaves:
                    x = np.array([self._tree.nodes[n]['mean'] for n in self._tree.successors(node)])
                    return self._tau * logsumexp(x / self._tau), x
                else:
                    x = np.array([self._solver(n)[0] for n in self._tree.successors(node)])
                    return self._tau * logsumexp(x / self._tau), x
            elif self._algorithm == 'rents':
                if successors[0] in self.leaves:
                    x = np.array([self._tree.nodes[n]['mean'] for n in self._tree.successors(node)])
                    return self._tau * np.log(np.sum(self._tree.nodes[node]['prior'] * np.exp(x / self._tau))), x
                else:
                    x = np.array([self._solver(n)[0] for n in self._tree.successors(node)])
                    return self._tau * np.log(np.sum(self._tree.nodes[node]['prior'] * np.exp(x / self._tau))), x
            elif self._algorithm == 'alpha-divergence':
                def sparse_max_alpha_divergence(means_tau):
                    temp_means_tau = means_tau.copy()
                    sorted_means = np.flip(np.sort(temp_means_tau))
                    kappa = list()
                    for i in range(1, len(sorted_means) + 1):
                        if 1 + i * sorted_means[i-1] > sorted_means[:i].sum() + i * (1 - (1/(self._alpha-1))):
                            idx = np.argwhere(temp_means_tau == sorted_means[i-1]).ravel()[0]
                            temp_means_tau[idx] = np.nan
                            kappa.append(idx)
                    kappa = np.array(kappa)

                    c_s_tau = ((means_tau[kappa].sum() - 1) / len(kappa)) + (1 - (1/(self._alpha-1)))

                    max_omega_tmp = np.maximum(means_tau - c_s_tau, np.zeros(len(means_tau)))
                    max_omega = np.power(max_omega_tmp * (self._alpha - 1), 1/(self._alpha-1))
                    max_omega = max_omega/np.sum(max_omega)

                    sparse_max_tmp = max_omega * means_tau

                    sparse_max = sparse_max_tmp.sum()

                    return sparse_max

                if successors[0] in self.leaves:
                    x = np.array([self._tree.nodes[n]['mean'] for n in self._tree.successors(node)])

                    return self._tau * sparse_max_alpha_divergence(x / self._tau), x
                else:
                    x = np.array([self._solver(n)[0] for n in self._tree.successors(node)])

                    return self._tau * sparse_max_alpha_divergence(np.array(x / self._tau)), x
            elif self._algorithm == 'tents':
                def sparse_max(means_tau):
                    temp_means_tau = means_tau.copy()
                    sorted_means = np.flip(np.sort(temp_means_tau))
                    kappa = list()
                    for i in range(1, len(sorted_means) + 1):
                        if 1 + i * sorted_means[i-1] > sorted_means[:i].sum():
                            idx = np.argwhere(temp_means_tau == sorted_means[i-1]).ravel()[0]
                            temp_means_tau[idx] = np.nan
                            kappa.append(idx)
                    kappa = np.array(kappa)

                    sparse_max = means_tau[kappa] ** 2 / 2 - (
                            means_tau[kappa].sum() - 1) ** 2 / (2 * len(kappa) ** 2)
                    sparse_max = sparse_max.sum() + .5

                    return sparse_max

                if successors[0] in self.leaves:
                    x = np.array([self._tree.nodes[n]['mean'] for n in self._tree.successors(node)])
                    return self._tau * sparse_max(x / self._tau), x
                else:
                    x = np.array([self._solver(n)[0] for n in self._tree.successors(node)])
                    return self._tau * sparse_max(np.array(x / self._tau)), x
            else:
                raise ValueError
