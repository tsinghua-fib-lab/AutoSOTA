from re import L
from xml.dom import NotSupportedErr
from matplotlib.pyplot import vlines
import numpy as np
import copy
from pandas import isna
import torch
from torch.distributions import Normal, Categorical
from sklearn.cluster import KMeans
from scipy.stats import qmc
import torch
import scipy.stats as stats


def discreterize(value, bits=16):
    rge = np.arange(bits)/(bits-1)
    dv = np.argmin((value - rge) ** 2)
    return dv

class Module:
    def __init__(self) -> None:
        self.controllble = True
        self.type_match = None
        self.alg_match = None
        self.op_match = None
    
    def gen_id(self, id):
        pass


class Operator:
    allow_qbest=False
    allow_archive=False
    Nxn = 0
    control_param = []
    description = ""
    op_id = -1
    type_match = 'Operator'

    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        self.use_qbest = use_qbest and self.allow_qbest
        self.use_archive = use_archive and self.allow_archive
        self.united_qbest = united_qbest and self.use_qbest and self.use_archive
        self.archive_id = archive_id
        self.united_id = united_id
        self.feature = self.get_feature()

    def qbest(self, q, population, cost):
        NP, dim = population.shape
        order = np.argsort(cost)
        population = population[order]
        cost = cost[order]
        if isinstance(q, float):
            qN = min(population.shape[0], max(2, int(np.round(q * NP))))
        else:
            qN = np.minimum(population.shape[0], np.maximum(2, np.round(q * NP)))
        return qN, population, cost
    
    def with_archive(self, population, archive, cost=None, a_cost=None):
        if cost is not None and a_cost is not None:
            if len(archive) < 1:
                return population, cost
            return np.concatenate((population, archive), 0), np.concatenate((cost, a_cost), 0)
        if len(archive) < 1:
            return population
        return np.concatenate((population, archive), 0)

    def get_xn(self, env, num_xn, rng=None):
        if rng is None:
            rng = np.random
        group = env['group']
        NP, dim = group.shape
        if 'trail' in env.keys():
            group = env['trail']
        rn = np.ones((NP, num_xn)) * -1
        xn = np.zeros((NP, num_xn, dim))
        arange = np.arange(NP)
        if self.use_archive:
            archive = np.array(env['archive'])
            united_archive = self.with_archive(group, archive)
            NPa = archive.shape[0]
            NPu = united_archive.shape[0]

        for i in range(num_xn):
            if self.use_archive and ((len(self.archive_id) > 0 and i in self.archive_id and NPa > 0) or (len(self.united_id) > 0 and i in self.united_id)):
                continue
            r = rng.randint(NP, size=NP)
            duplicate = np.where((r == arange) + np.sum([rn[:,j] == r for j in range(i)], 0))[0]
            while duplicate.shape[0] > 0 and NP > num_xn:
                r[duplicate] = rng.randint(NP, size=duplicate.shape[0])
                duplicate = np.where((r == arange) + np.sum([rn[:,j] == r for j in range(i)], 0))[0]
            rn[:,i] = r
            xn[:, i, :] = group[r]

        if self.use_archive and len(self.archive_id) > 0 and NPa > 0:
            for ia in self.archive_id:
                if ia >= num_xn:
                    continue
                r = rng.randint(NPa, size=(NP))
                rn[:,ia] = r
                xn[:, ia, :] = archive[r]

        if self.use_archive and len(self.united_id) > 0:
            for iu in self.united_id:
                if iu >= num_xn:
                    continue
                r = rng.randint(NPu, size=(NP))
                duplicate = np.where((r == arange) + np.sum([rn[:,j] == r for j in range(num_xn)], 0))[0]
                while duplicate.shape[0] > 0:
                    r[duplicate] = rng.randint(NPu, size=duplicate.shape[0])
                    duplicate = np.where((r == arange) + np.sum([rn[:,j] == r for j in range(num_xn)], 0))[0]
                rn[:,iu] = r
                xn[:, iu, :] = united_archive[r]

        return xn

    def get_xqb(self, env, q, rng=None):
        if rng is None:
            rng = np.random
        group = env['group']
        NP, dim = group.shape
        if self.united_qbest:
            group, _ = self.with_archive(env['group'], env['archive'], env['cost'], env['a_cost'])
        qN, qpop, q_cost = self.qbest(q, group, env['cost'])
        xqb = qpop[rng.randint(qN, size=NP)]
        return xqb

    def selection(self, action, group, cost, rng=None):
        if rng is None:
            rng = np.random
        NP, dim = group.shape
        if action == 'rand':
            permu = rng.permutation(NP)
            return permu[:NP//2], permu[NP//2:]
        elif action == 'inter':
            arange = np.arange(NP//2)
            return arange * 2, arange * 2 + 1
        elif action == 'parti':
            return np.arange(NP//2), np.arange(NP//2) + NP//2
        elif action == 'rank_inter':
            order = np.argsort(cost)
            return order[:NP//2], order[NP//2:]
        else:
            order = np.argsort(cost)
            arange = np.arange(NP//2)
            return order[arange * 2], order[arange * 2 + 1]

    def action_interpret(self, logits, softmax=False, fixed_action=None):
        actions = {}
        action_value = []
        logp = []
        entropy = []
        for i, param in enumerate(self.control_param):
            mu, sigma = logits[i*2:i*2+2]
            policy = Normal(mu, sigma)
            if param['type'] == 'float':
                if fixed_action is None:
                    action = policy.sample()
                else:
                    action = fixed_action[i]
                logp.append(policy.log_prob(action))
                entropy.append(policy.entropy())
                action_value.append(discreterize(action))
                actions[param['name']] = torch.clip(action, 0, 1).item() * (param['range'][1] - param['range'][0]) + param['range'][0]
            else:
                rang = (torch.arange(len(param['range']))) / len(param['range'])
                probs = torch.exp(policy.log_prob(rang))
                if softmax:
                    cate = Categorical(torch.softmax(probs/torch.sum(probs), -1))
                else:
                    cate = Categorical(probs/torch.sum(probs))
                if fixed_action is None:
                    if torch.max(probs) < 1e-3:
                        action = torch.argmin((rang - mu) ** 2)
                    else:
                        action = cate.sample()
                else:
                    action = fixed_action[i]
                
                logp.append(cate.log_prob(action))
                action_value.append(action)
                entropy.append(policy.entropy())
                actions[param['name']] = param['range'][action]
        return actions, action_value, logp, entropy

    def discrete_action_interpret(self, logits, e_greedy=0, rng=None):
        if rng is None:
            rng = np.random
        action_value = []
        for i, param in enumerate(self.control_param):
            if e_greedy > 0 and torch.rand(1) < e_greedy:
                action_value.append(rng.choice(param['range']))
            else:
                action_value.append(np.argmax(logits[i][:len(param['range'])]))
        return action_value

    def __call__(self, logits, env, softmax=False, rng=None, had_action=None):
        raise NotImplementedError

    def update_description(self):
        if self.use_archive:
            self.description += " It uses the archive for the eliminated individuals, some random individuals are sampled from the archive or the union of the population and archive."
            self.op_id += 1
        if self.use_qbest:
            self.description += " It uses qbest mechanism which select the best individual from a set of top individuals."
            self.op_id += 2
            if self.united_qbest:
                self.description += " The top set is extracted from the union of population and archive."
                self.op_id += 4
            else:
                self.description += " The top set is extracted from the population."
            self.description += " It has a prameter q to determine the range of the top."
        self.feature = self.get_feature()
    
    def get_feature(self, maxLen=24):
        feature = torch.zeros(maxLen)
        j = 1
        for i in range(maxLen-1, -1, -1):
            if self.op_id & j:
                feature[i] = 1
            j *= 2
        return feature



class DE_op(Operator):
    alg_match = 'DE'
    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)

class GA_op(Operator):
    alg_match = 'GA'
    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)

class PSO_op(Operator):
    alg_match = 'PSO'
    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)


"""
Mutation
"""

# DE ======================================================================
class DE_Mutation(DE_op):
    op_match = 'DE_Mutation'
    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)

class best2(DE_Mutation):
    allow_qbest=True
    allow_archive=True
    Nxn = 4
    description = "This is a mutation operator DE/best/2, it uses the weighted sum of the global best individual and the differences of two pairs of random individuls to generate new individual. It has two weights F1 and F2 for the random individuals."
    op_id = 1 * 8

    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)
        self.control_param = [
            {'name': 'F1', 'type': 'float', 'range': [0, 1], 'default': 0.5},
            {'name': 'F2', 'type': 'float', 'range': [0, 1], 'default': 0.5},
        ]
        if self.use_qbest:
            self.control_param.append({'name': 'q', 'type': 'float', 'range': [0, 1], 'default': 0.1})
        self.update_description()

    def __call__(self, logits, env, softmax=False, fixed_action=None, rng=None, had_action=None, random=False):
        if rng is None:
            rng = np.random
        action_value = []
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = self.action_interpret(logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        elif random:  # both None -> default
            action = {}
            for param in self.control_param:
                # action[param['name']] = param['default']
                ac = rng.rand()
                action[param['name']] = ac * (param['range'][-1] - param['range'][0]) + param['range'][0]
                action_value.append(discreterize(ac))
        else:
            action = {}
            for param in self.control_param:
                action[param['name']] = param['default']
                action_value.append(param['default'])
        xn = self.get_xn(env, 4, rng)
        if self.use_qbest:
            xqb = self.get_xqb(env, action['q'], rng)
            trail = xqb + action['F1'] * (xn[:,0] - xn[:,1]) + action['F2'] * (xn[:,2] - xn[:,3])
        else:
            trail = env['gbest_solutions'] + action['F1'] * (xn[:,0] - xn[:,1]) + action['F2'] * (xn[:,2] - xn[:,3])

        return {'result': trail, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

class rand2(DE_Mutation):
    allow_qbest=False
    allow_archive=True
    Nxn = 5
    description = "This is a mutation operator DE/rand/2, it uses the weighted sum of a randomly selected individuals and the differences of two pairs of random individuls to generate new individual. It has two weights F1 and F2."
    op_id = 2 * 8

    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)
        self.control_param = [
            {'name': 'F1', 'type': 'float', 'range': [0, 1], 'default': 0.5},
            {'name': 'F2', 'type': 'float', 'range': [0, 1], 'default': 0.5},
        ]
        self.update_description()

    def __call__(self, logits, env, softmax=False, fixed_action=None, rng=None, had_action=None, random=False):
        if rng is None:
            rng = np.random
        action_value = []
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = self.action_interpret(logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        elif random:  # both None -> default
            action = {}
            for param in self.control_param:
                # action[param['name']] = param['default']
                ac = rng.rand()
                action[param['name']] = ac * (param['range'][-1] - param['range'][0]) + param['range'][0]
                action_value.append(discreterize(ac))
        else:
            action = {}
            for param in self.control_param:
                action[param['name']] = param['default']
                action_value.append(param['default'])
        xn = self.get_xn(env, 5, rng)

        NP, dim = env['group'].shape

        if isinstance(action['F1'], np.ndarray) and len(action['F1'].shape) < 2:
            action['F1'] = action['F1'][:,None].repeat(dim, -1)
        if isinstance(action['F2'], np.ndarray) and len(action['F2'].shape) < 2:
            action['F2'] = action['F2'][:,None].repeat(dim, -1)

        trail = xn[:,0] + action['F1'] * (xn[:,1] - xn[:,2]) + action['F2'] * (xn[:,3] - xn[:,4])

        return {'result': trail, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

class current2rand(DE_Mutation):
    allow_qbest=False
    allow_archive=True
    Nxn = 3   
    description = "This is a mutation operator DE/current-to-rand/1, it uses the weighted sum of the current individual, the difference of a randomly selected individual and current individual, and the differences of a pair of random individuls to generate new individual. It has two weights F1 and F2 for the random individuals."
    op_id = 3 * 8

    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)
        self.control_param = [
            {'name': 'F1', 'type': 'float', 'range': [0, 1], 'default': 0.5},
            {'name': 'F2', 'type': 'float', 'range': [0, 1], 'default': 0.5},
        ]
        self.update_description()

    def __call__(self, logits, env, softmax=False, fixed_action=None, rng=None, had_action=None, random=False):
        if rng is None:
            rng = np.random
        action_value = []
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = self.action_interpret(logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        elif random:  # both None -> default
            action = {}
            for param in self.control_param:
                # action[param['name']] = param['default']
                ac = rng.rand()
                action[param['name']] = ac * (param['range'][-1] - param['range'][0]) + param['range'][0]
                action_value.append(discreterize(ac))
        else:
            action = {}
            for param in self.control_param:
                action[param['name']] = param['default']
                action_value.append(param['default'])
        xn = self.get_xn(env, 3, rng)
        group = env['group']
        if 'trail' in env.keys():
            group = env['trail']
        NP, dim = group.shape

        if isinstance(action['F1'], np.ndarray) and len(action['F1'].shape) < 2:
            action['F1'] = action['F1'][:,None].repeat(dim, -1)
        if isinstance(action['F2'], np.ndarray) and len(action['F2'].shape) < 2:
            action['F2'] = action['F2'][:,None].repeat(dim, -1)

        trail = group + action['F1'] * (xn[:,0] - group) + action['F2'] * (xn[:,1] - xn[:,2])

        return {'result': trail, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

class current2best(DE_Mutation):
    allow_qbest=True
    allow_archive=True
    Nxn = 2
    description = "This is a mutation operator DE/current-to-best/1, it uses the weighted sum of the current individual, the difference of the global best individual and current individual, and the differences of a pair of random individuls to generate new individual. It has two weights F1 and F2 for the best and the random individuals."
    op_id = 4 * 8

    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)
        self.control_param = [
            {'name': 'F1', 'type': 'float', 'range': [0, 1], 'default': 0.5},
            {'name': 'F2', 'type': 'float', 'range': [0, 1], 'default': 0.5},
        ]
        if self.use_qbest:
            self.control_param.append({'name': 'q', 'type': 'float', 'range': [0, 1], 'default': 2/15})
        self.update_description()

    def __call__(self, logits, env, softmax=False, fixed_action=None, rng=None, had_action=None, random=False):
        if rng is None:
            rng = np.random
        action_value = []
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = self.action_interpret(logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        elif random:  # both None -> default
            action = {}
            for param in self.control_param:
                # action[param['name']] = param['default']
                ac = rng.rand()
                action[param['name']] = ac * (param['range'][-1] - param['range'][0]) + param['range'][0]
                action_value.append(discreterize(ac))
        else:
            action = {}
            for param in self.control_param:
                action[param['name']] = param['default']
                action_value.append(param['default'])
        xn = self.get_xn(env, 2, rng)

        group = env['group']
        if 'trail' in env.keys():
            group = env['trail']
        NP, dim = group.shape

        if isinstance(action['F1'], np.ndarray) and len(action['F1'].shape) < 2:
            action['F1'] = action['F1'][:,None].repeat(dim, -1)
        if isinstance(action['F2'], np.ndarray) and len(action['F2'].shape) < 2:
            action['F2'] = action['F2'][:,None].repeat(dim, -1)

        if self.use_qbest:
            xqb = self.get_xqb(env, action['q'])
            trail = group + action['F1'] * (xqb - group) + action['F2'] * (xn[:,0] - xn[:,1])
        else:
            trail = group + action['F1'] * (env['gbest_solutions'] - group) + action['F2'] * (xn[:,0] - xn[:,1])

        return {'result': trail, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

class rand2best(DE_Mutation):
    allow_qbest=True
    allow_archive=True
    Nxn = 3
    description = "This is a mutation operator DE/current-to-best/1, it uses the weighted sum of a random individual, the difference of the global best individual and current individual, and the differences of a pair of random individuls to generate new individual. It has two weights F1 and F2 for the best and the random individuals."
    op_id = 5 * 8

    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)
        self.control_param = [
            {'name': 'F0', 'type': 'float', 'range': [0, 1], 'default': 0.5},
            {'name': 'F1', 'type': 'float', 'range': [0, 1], 'default': 0.5},
            {'name': 'F2', 'type': 'float', 'range': [0, 1], 'default': 0.5},
        ]
        if self.use_qbest:
            self.control_param.append({'name': 'q', 'type': 'float', 'range': [0, 1], 'default': 2/15})
        self.update_description()

    def __call__(self, logits, env, softmax=False, fixed_action=None, rng=None, had_action=None, random=False):
        if rng is None:
            rng = np.random
        action_value = []
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = self.action_interpret(logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        elif random:  # both None -> default
            action = {}
            for param in self.control_param:
                # action[param['name']] = param['default']
                ac = rng.rand()
                action[param['name']] = ac * (param['range'][-1] - param['range'][0]) + param['range'][0]
                action_value.append(discreterize(ac))
        else:
            action = {}
            for param in self.control_param:
                action[param['name']] = param['default']
                action_value.append(param['default'])
        xn = self.get_xn(env, 3, rng)

        group = env['group']
        if 'trail' in env.keys():
            group = env['trail']
        NP, dim = group.shape

        if isinstance(action['F1'], np.ndarray) and len(action['F1'].shape) < 2:
            action['F1'] = action['F1'][:,None].repeat(dim, -1)
        if isinstance(action['F2'], np.ndarray) and len(action['F2'].shape) < 2:
            action['F2'] = action['F2'][:,None].repeat(dim, -1)

        if self.use_qbest:
            xqb = self.get_xqb(env, action['q'])
            trail = action['F0'] * (xn[:,0] + action['F1'] * (xqb - group) + action['F2'] * (xn[:,1] - xn[:,2]))
        else:
            trail = action['F0'] * (xn[:,0] + action['F1'] * (env['gbest_solutions'] - group) + action['F2'] * (xn[:,1] - xn[:,2]))

        return {'result': trail, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

# GA ======================================================================
class GA_Mutation(GA_op):
    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)

class gaussian(GA_Mutation):
    allow_qbest=False
    allow_archive=False
    Nxn = 0
    description = "This is a gaussian mutation operator, it applies gaussian noise with parameter sigma on current individual to generate a new one."
    op_id = 6 * 8

    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)
        self.control_param = [
            {'name': 'sigma', 'type': 'float', 'range': [0, 1], 'default': 2/15},
        ]
        self.update_description()

    def __call__(self, logits, env, softmax=False, fixed_action=None, rng=None, had_action=None, random=False):
        if rng is None:
            rng = np.random
        action_value = []
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = self.action_interpret(logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        elif random:  # both None -> default
            action = {}
            for param in self.control_param:
                # action[param['name']] = param['default']
                ac = rng.rand()
                action[param['name']] = ac * (param['range'][-1] - param['range'][0]) + param['range'][0]
                action_value.append(discreterize(ac))
        else:
            action = {}
            for param in self.control_param:
                action[param['name']] = param['default']
                action_value.append(param['default'])
        NP, dim = env['group'].shape

        group = env['group']
        if 'trail' in env.keys():
            group = env['trail']

        trail = group + rng.normal(0, action['sigma'], size=(NP, dim))
        return {'result': trail, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

class polynomial(GA_Mutation):
    allow_qbest=False
    allow_archive=False
    Nxn = 0
    description = "This is a polynomial mutation operator, it randomly shift current individual to upperbound or lowerbound in random step lengths to generate a new one. It has a discrete parameter on the step length"
    op_id = 7 * 8

    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)
        self.control_param = [
            {'name': 'n', 'type': 'discrete', 'range': [1, 2 ,3], 'default': 2},
        ]
        self.update_description()

    def __call__(self, logits, env, softmax=False, fixed_action=None, rng=None, had_action=None, random=False):
        if rng is None:
            rng = np.random
        action_value = []
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = self.action_interpret(logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        elif random:  # both None -> default
            action = {}
            for param in self.control_param:
                # action[param['name']] = param['default']
                ac = rng.randint(len(param['range']))
                action[param['name']] = param['range'][ac]
                action_value.append(ac)
        else:
            action = {}
            for param in self.control_param:
                action[param['name']] = param['default']
                action_value.append(1)
        NP, dim = env['group'].shape
        group = env['group']
        if 'trail' in env.keys():
            group = env['trail']

        rvs = rng.rand(NP, dim)
        trail = np.zeros((NP, dim))
        d1 = np.power(2 * rvs, 1/(action['n']+1)) - 1
        d2 = 1 - np.power(2 * (1 - rvs), 1/(action['n']+1))
        C1 = group + d1 * (group - env['bounds'][0])
        C2 = group + d2 * (env['bounds'][1] - group)

        trail[rvs <= 0.5] = C1[rvs <= 0.5]
        trail[rvs > 0.5] = C2[rvs > 0.5]
        return {'result': trail, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}


"""
Crossover
"""

# DE ======================================================================
class DE_Crossover(DE_op):
    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)

class binomial(DE_Crossover):
    allow_qbest=True
    allow_archive=True
    Nxn = 0
    description = "This is a binomial crossover operator, it randomly switch values between the father and child individuals. A randomly selected child value will not be switched make the child different from its father. It has a crossover rate parameter."
    op_id = 8 * 8

    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)
        self.use_archive = self.united_qbest
        self.control_param = [
            {'name': 'Cr', 'type': 'float', 'range': [0, 1], 'default': 0.9},
        ]
        if self.use_qbest:
            self.control_param.append({'name': 'q', 'type': 'float', 'range': [0, 1], 'default': 2/15})
        self.update_description()

    def __call__(self, logits, env, softmax=False, fixed_action=None, rng=None, had_action=None, random=False):
        if rng is None:
            rng = np.random
        action_value = []
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = self.action_interpret(logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        elif random:  # both None -> default
            action = {}
            for param in self.control_param:
                # action[param['name']] = param['default']
                ac = rng.rand()
                action[param['name']] = ac * (param['range'][-1] - param['range'][0]) + param['range'][0]
                action_value.append(discreterize(ac))
        else:
            action = {}
            for param in self.control_param:
                action[param['name']] = param['default']
                action_value.append(param['default'])
        NP, dim = env['group'].shape

        if isinstance(action['Cr'], np.ndarray) and len(action['Cr'].shape) < 2:
            action['Cr'] = action['Cr'][:,None].repeat(dim, -1)

        jrand = rng.randint(dim, size=NP)
        parent = env['group']
        if self.use_qbest:
            parent = self.get_xqb(env, action['q'])
        u = np.where(rng.rand(NP, dim) < action['Cr'], env['trail'], parent)
        u[np.arange(NP), jrand] = env['trail'][np.arange(NP), jrand]
        return {'result': u, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

class exponential(DE_Crossover):
    allow_qbest=True
    allow_archive=True
    Nxn = 0
    description = "This is a exponential crossover operator, it switch a random segment of the father and child individuals. It has a crossover rate parameter."
    op_id = 9 * 8

    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)
        self.use_archive = self.united_qbest
        self.control_param = [
            {'name': 'Cr', 'type': 'float', 'range': [0, 1], 'default': 0.9},
        ]
        if self.use_qbest:
            self.control_param.append({'name': 'q', 'type': 'float', 'range': [0, 1], 'default': 2/15})
        self.update_description()

    def __call__(self, logits, env, softmax=False, fixed_action=None, rng=None, had_action=None, random=False):
        if rng is None:
            rng = np.random
        action_value = []
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = self.action_interpret(logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        elif random:  # both None -> default
            action = {}
            for param in self.control_param:
                # action[param['name']] = param['default']
                ac = rng.rand()
                action[param['name']] = ac * (param['range'][-1] - param['range'][0]) + param['range'][0]
                action_value.append(discreterize(ac))
        else:
            action = {}
            for param in self.control_param:
                action[param['name']] = param['default']
                action_value.append(param['default'])
        NP, dim = env['group'].shape
        if isinstance(action['Cr'], np.ndarray) and len(action['Cr'].shape) < 2:
            action['Cr'] = action['Cr'][:,None].repeat(dim, -1)

        parent = env['group']
        if self.use_qbest:
            parent = self.get_xqb(env, action['q'])
        u = parent.copy()
        L = rng.randint(dim-1, size=(NP, 1))
        L = np.repeat(L, dim, -1)
        rvs = rng.rand(NP, dim)
        index = np.repeat([np.arange(dim)], NP, 0)
        rvs[index < L] = -1 # masked as no crossover (left side)
        R = np.argmax(rvs > action['Cr'], -1)
        R[np.max(rvs, -1) < action['Cr']] = dim
        R = np.repeat(R[:,None], dim, -1)
        rvs[index >= R] = -1 # masked as no crossover (right side)
        u[rvs > 0] = env['trail'][rvs > 0]
        return {'result': u, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

# PSO ======================================================================
class PSO_Operator(PSO_op):
    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)

class currentxbest3(PSO_Operator):
    allow_qbest=True
    allow_archive=True
    Nxn = 0
    description = "This is a PSO update operator, it uses the weighted sum of the current velocity, the difference of the global best individual and current individual, the difference of the local best individual and current individual, and the differences of the personal best individual and current individual to generate new velocity. Then it uses the velocity to update the individual. It has a weight on current velocity and three wieghts for the three differences."
    op_id = 11 * 8

    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)
        self.use_archive = self.united_qbest
        self.control_param = [
                {'name': 'w', 'type': 'float', 'range': [0, 1], 'default': 0.9},
                {'name': 'c1', 'type': 'float', 'range': [0, 2], 'default': 2.0},
                {'name': 'c2', 'type': 'float', 'range': [0, 2], 'default': 2.0},
                {'name': 'c3', 'type': 'float', 'range': [0, 2], 'default': 2.0},
            ]
        if self.use_qbest:
            self.control_param.append({'name': 'q', 'type': 'float', 'range': [0, 1], 'default': 2/15})
        self.update_description()

    def __call__(self, logits, env, softmax=False, fixed_action=None, rng=None, had_action=None, random=False):
        if rng is None:
            rng = np.random
        NP, dim = env['group'].shape
        action_value = []
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = self.action_interpret(logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        elif random:  # both None -> default
            action = {}
            for param in self.control_param:
                # action[param['name']] = param['default']
                ac = rng.rand()
                action[param['name']] = ac * (param['range'][-1] - param['range'][0]) + param['range'][0]
                action_value.append(discreterize(ac))
        else:
            action = {}
            for param in self.control_param:
                action[param['name']] = param['default']
                action_value.append(param['default'])
        group = env['group']
        if 'trail' in env.keys():
            group = env['trail']

        for k in action.keys():
            if isinstance(action[k], np.ndarray) and len(action[k].shape) < 2:
                action[k] = action[k][:,None].repeat(dim, -1)

        gbest_solutions = env['gbest_solutions']
        if self.use_qbest:
            gbest_solutions = self.get_xqb(env, action['q'])
        vel = action['w'] * env['velocity'] + \
              action['c1'] * np.repeat(rng.rand(NP, 1), dim, -1) * (gbest_solutions - group) + \
              action['c2'] * np.repeat(rng.rand(NP, 1), dim, -1) * (env['lbest_solutions'] - group) + \
              action['c3'] * np.repeat(rng.rand(NP, 1), dim, -1) * (env['pbest_solutions'] - group)
        vel = np.clip(vel, -env['Vmax'], env['Vmax'])
        new_x = group + vel
        return {'result': new_x, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

# GA ======================================================================
class GA_Crossover(GA_op):
    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)

class sbx(GA_Crossover):
    allow_qbest=False
    allow_archive=False
    Nxn = 0
    description = "This is a simulated binary crossover operator, it uses the weighted sum of the two selected parent to generate two children. The selection can be random selection, divide current population into two parts, selection at regular intervals, selection based on ranking or selection at regular intervals based on ranking. It also has a discrete parameter for the weights."
    op_id = 12 * 8

    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)
        self.control_param = [
            {'name': 'n', 'type': 'discrete', 'range': [1, 2 ,3], 'default': 2},
            {'name': 'select', 'type': 'discrete', 'range': ['rand', 'inter', 'parti', 'rank_inter', 'rank_parti'], 'default': 'rand'},
        ]
        self.update_description()

    def __call__(self, logits, env, softmax=False, fixed_action=None, rng=None, had_action=None, random=False):
        if rng is None:
            rng = np.random
        action_value = []
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = self.action_interpret(logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        elif random:  # both None -> default
            action = {}
            for param in self.control_param:
                # action[param['name']] = param['default']
                ac = rng.randint(len(param['range']))
                action[param['name']] = param['range'][ac]
                action_value.append(ac)
        else:
            action = {}
            action[self.control_param[0]['name']] = self.control_param[0]['default']
            action_value.append(1)
            action[self.control_param[1]['name']] = self.control_param[1]['default']
            action_value.append(0)

        NP, dim = env['group'].shape
        group = env['group']
        if 'trail' in env.keys():
            group = env['trail']
        
        index1, index2 = self.selection(action['select'], group, env['cost'])
        ext = None
        if len(index1) != len(index2):  # odd number population
            if len(index1) > len(index2):
                ext = index1[0]
                index1 = index1[1:]
            else:
                ext = index2[0]
                index2 = index2[1:]
        rvs = rng.rand(NP//2, dim)
        b = np.power(2 * rvs, 1 / (action['n'] + 1))
        b[rvs > 0.5] = np.power(2 * (1 - rvs[rvs > 0.5]), -1 / (action['n'] + 1))
        C1 = 0.5 * (group[index2] + group[index1]) - 0.5 * b * (group[index2] - group[index1])
        C2 = 0.5 * (group[index2] + group[index1]) + 0.5 * b * (group[index2] - group[index1])

        new_x = np.zeros((NP, dim))
        new_x[index1] = C1
        new_x[index2] = C2
        if ext is not None:
            new_x[ext] = env['group'][ext]
        return {'result': new_x, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}

class mpx(GA_Crossover):
    allow_qbest=False
    allow_archive=False
    Nxn = 0
    description = "This is a multiple point crossover operator, it randomly switch values of the two selected parent to generate two children. The selection can be random selection, divide current population into two parts, selection at regular intervals, selection based on ranking or selection at regular intervals based on ranking. It also has a crossover rate parameter."
    op_id = 13 * 8

    def __init__(self, use_qbest=False, united_qbest=False, use_archive=False, archive_id=[], united_id=[]) -> None:
        super().__init__(use_qbest, united_qbest, use_archive, archive_id, united_id)
        self.control_param = [
            {'name': 'Cr', 'type': 'float', 'range': [0, 1], 'default': 0.5},
            {'name': 'select', 'type': 'discrete', 'range': ['rand', 'inter', 'parti', 'rank_inter', 'rank_parti'], 'default': 'rand'},
        ]
        self.update_description()

    def __call__(self, logits, env, softmax=False, fixed_action=None, rng=None, had_action=None, random=False):
        if rng is None:
            rng = np.random
        action_value = []
        logp = 0
        entropy = [0]
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = self.action_interpret(logits, softmax, fixed_action)
        elif had_action is not None:
            action = had_action
        elif random:  # both None -> default
            action = {}
            for param in self.control_param:
                # action[param['name']] = param['default']
                if param['type'] == 'discrete':
                    ac = rng.randint(len(param['range']))
                    action[param['name']] = param['range'][ac]
                    action_value.append(ac)
                else:
                    ac = rng.rand()
                    action[param['name']] = ac * (param['range'][-1] - param['range'][0]) + param['range'][0]
                    action_value.append(discreterize(ac))
        else:
            action = {}
            action[self.control_param[0]['name']] = self.control_param[0]['default']
            action_value.append(self.control_param[0]['default'])
            action[self.control_param[1]['name']] = self.control_param[1]['default']
            action_value.append(0)
        NP, dim = env['group'].shape
        if 'trail' in env.keys():
            rvs = rng.rand(NP, dim)
            trail = copy.deepcopy(env['trail'])
            trail[rvs < action['Cr']] = env['group'][rvs < action['Cr']]
            return {'result': trail, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}
        else:
            index1, index2 = self.selection(action['select'], env['group'], env['cost'])
            ext = None
            if len(index1) != len(index2):  # odd number population
                if len(index1) > len(index2):
                    ext = index1[0]
                    index1 = index1[1:]
                else:
                    ext = index2[0]
                    index2 = index2[1:]
            rvs = rng.rand(NP//2, dim)
            C1 = env['group'][index1]
            C2 = env['group'][index2]
            C1[rvs < action['Cr']], C2[rvs < action['Cr']] = C2[rvs < action['Cr']], C1[rvs < action['Cr']]

            new_x = np.zeros((NP, dim))
            new_x[index1] = C1
            new_x[index2] = C2
            if ext is not None:
                new_x[ext] = env['group'][ext]
            return {'result': new_x, 'actions': action_value, 'had_action': action, 'logp': logp, 'entropy': entropy}


class Multi_op(Module):
    def __init__(self, op_list) -> None:
        self.ops = op_list
        self.nop = len(op_list)
        self.description = f"This is a multi-operator block which contains {self.nop} operators. "
        ordinals = ['first', 'second', 'third']
        self.op_id = 0
        self.default_op = None
        for i, op in enumerate(self.ops):
            self.description += f"The {ordinals[i]} one " + op.description[5:]
            self.op_id = 256 * self.op_id + op.op_id
        self.feature = self.get_feature()

    def get_op(self, logits, softmax=False, fixed_action=None):
        mu, sigma = logits
        policy = Normal(mu, sigma)
        rang = (torch.arange(self.nop)) / self.nop
        probs = torch.exp(policy.log_prob(rang))
        entropy = policy.entropy()
        if softmax:
            cate = Categorical(torch.softmax(probs/torch.sum(probs), -1))
        else:
            cate = Categorical(probs/torch.sum(probs))
        if fixed_action is None:
            if torch.max(probs) < 1e-4:
                action = torch.argmin((rang - mu) ** 2)
            else:
                action = cate.sample()
        else:
            action = fixed_action[0]
        logp = cate.log_prob(action)
        return action, logp, entropy
    
    def get_feature(self, maxLen = 24):
        
        feature = torch.zeros(maxLen)
        j = 1
        for i in range(maxLen-1, -1, -1):
            if self.op_id & j:
                feature[i] = 1
            j *= 2
        return feature

    def action_interpret(self, logits, softmax=False, fixed_action=None):
        logps = []
        entropys = []
        iop, logp1, e = self.get_op(logits[:2], softmax, fixed_action)
        logps.append(logp1)
        entropys.append(e)
        op = self.ops[iop]
        _, _, logp, entropy = op.action_interpret(logits[2:], softmax=softmax, fixed_action=fixed_action[1:])
        logps += logp
        entropys += entropy
        return 0, 0, logps, entropys

    def __call__(self, logits, env, softmax=False, fixed_action=None, rng=None, had_action=None, random=False):
        if rng is None:
            rng = np.random
        
        def get_subenv(iop, i, env):
            subenv = copy.deepcopy(env)
            subenv['group'] = env['group'][iop == i]
            subenv['cost'] = env['cost'][iop == i]
            if 'trail' in env.keys():
                subenv['trail'] = env['trail'][iop == i]
            subenv['pbest_solutions'] = env['pbest_solutions'][iop == i]
            subenv['pbest'] = env['pbest'][iop == i]
            subenv['velocity'] = env['velocity'][iop == i]
            return subenv
        logp = []
        actions = []
        entropy = []
        had_action_rec = {}

        if logits is not None and had_action is None:
            iop, logp1, e = self.get_op(logits[:2], softmax, fixed_action)
            actions.append(iop)
            had_action_rec['op_select'] = iop
            logp.append(logp1)
            entropy.append(e)
            op = self.ops[iop]
            res = op(logits[2:], env, softmax, fixed_action[1:] if fixed_action is not None else None, rng=rng)
            actions += res['actions']
            had_action_rec['op_list'] = [None for _ in range(self.nop)]
            had_action_rec['op_list'][iop] = res['had_action']
            logp += res['logp']
            entropy += res['entropy']
            trail = res['result']
        elif had_action is not None:
            iop = had_action['op_select']
            if isinstance(iop, np.ndarray):
                trail = np.zeros_like(env['group'])
                for i in range(self.nop):
                    if np.sum(iop == i) < 1:
                        continue
                    # subenv = get_subenv(iop, i, env)
                    res = self.ops[i](None, env, softmax, fixed_action=None, rng=rng, had_action=had_action['op_list'][i])
                    trail[iop == i] = res['result'][iop == i]
            else:
                # assert isinstance(iop, int)
                res = self.ops[iop](None, env, softmax, fixed_action=None, rng=rng, had_action=had_action['op_list'][iop])
                trail = res['result']
        else:
            iop = rng.randint(self.nop) if random else self.default_op
            actions.append(iop)
            # trail = np.zeros_like(env['group'])
            # for i in range(self.nop):
            #     if np.sum(iop == i) < 1:
            #         continue
            #     subenv = get_subenv(iop, i, env)

            #     res = self.ops[i](None, subenv, softmax, fixed_action=None, rng=rng, had_action=None)
            #     trail[iop == i] = res['result']
            res = self.ops[iop](None, env, softmax, fixed_action=None, rng=rng, had_action=None)
            trail = res['result']
            actions += res['actions']
        return {'result': trail, 'actions': actions, 'had_action': had_action_rec, 'logp': logp, 'entropy': entropy}


"""
Bound
"""
class Bounding(Module):
    description = """This is a bound control operator, it restricts the values of individuals into a specific range. It has six modes:
clip is to clip values at the bound;
rand is to assign a random value for those values that out of bound;
periodic is to map the individual back to the space using the modulo function;
reflect is to resemble the reflection characteristic of a mirror and reflect the individual back;
halving is to halve the distance between the original position and the crossed bound."""
    op_id = 14 * 8

    def __init__(self) -> None:
        self.control_param = [
            {'name': 'bound', 'type': 'discrete', 'range': ['clip', 'rand', 'periodic', 'reflect', 'halving', ], 'default': 'clip'},
        ]
        self.feature = self.get_feature()

    def get_feature(self, maxLen=24):
        
        feature = torch.zeros(maxLen)
        j = 1
        for i in range(maxLen-1, -1, -1):
            if self.op_id & j:
                feature[i] = 1
            j *= 2
        return feature
    
    def action_interpret(self, logits, softmax=False, fixed_action=None):
        actions = {}
        action_value = []
        logp = []
        entropy = []
        for i, param in enumerate(self.control_param):
            mu, sigma = logits[i*2:i*2+2]
            policy = Normal(mu, sigma)
            if param['type'] == 'float':
                if fixed_action is None:
                    action = policy.sample()
                else:
                    action = fixed_action[i]
                logp.append(policy.log_prob(action))
                entropy.append(policy.entropy())
                action_value.append(action)
                actions[param['name']] = torch.clip(action, 0, 1).item() * (param['range'][1] - param['range'][0]) + param['range'][0]
            else:
                rang = (torch.arange(len(param['range']))) / len(param['range'])
                probs = torch.exp(policy.log_prob(rang))
                if softmax:
                    cate = Categorical(torch.softmax(probs/torch.sum(probs), -1))
                else:
                    cate = Categorical(probs/torch.sum(probs))
                if fixed_action is None:
                    if torch.max(probs) < 1e-4:
                        action = torch.argmin((rang - mu) ** 2)
                    else:
                        action = cate.sample()
                else:
                    action = fixed_action[i]
                
                logp.append(cate.log_prob(action))
                action_value.append(action)
                entropy.append(policy.entropy())
                actions[param['name']] = param['range'][action]
        return actions, action_value, logp, entropy
                
    def __call__(self, logits, env, softmax=False, had_action=None, fixed_action=None, rng=None, default=False):
        if rng is None:
            rng = np.random
        # if had_action is None:
        #     action, action_value, logp, entropy = self.action_interpret(logits, softmax, fixed_action)
        #     type = self.control_param[0]['range'][action[0]]
        # else:
        #     type = had_action['bound']
        entropy = None
        logp = None
        if had_action is None and logits is not None:
            action, action_value, logp, entropy = self.action_interpret(logits, softmax, fixed_action)
        elif had_action is not None:
            action_value = action = had_action
        else:  # both None -> default
            action = {}
            if default:
               action['bound'] = self.control_param[0]['default']
               action_value = [0]
            else:
                action_value = rng.randint(len(self.control_param[0]['range']))
                action['bound'] = self.control_param[0]['range'][action_value]
                action_value = [action_value]
        type = action['bound']
        trail = copy.deepcopy(env['trail'])
        group = env['group']
        bounds = env['bounds']
        if type == 'clip':
            trail = np.clip(trail, bounds[0], bounds[1])
        elif type == 'rand':
            trail[(trail < bounds[0]) + (trail > bounds[1])] = rng.rand(np.sum((trail < bounds[0]) + (trail > bounds[1]))) * (bounds[1] - bounds[0]) + bounds[0]
        elif type == 'periodic':
            while np.max(trail) > bounds[1] or np.min(trail) < bounds[0]:
                trail[trail > bounds[1]] += - bounds[1] + bounds[0]
                trail[trail < bounds[0]] += - bounds[0] + bounds[1]
        elif type == 'reflect':
            while np.max(trail) > bounds[1] or np.min(trail) < bounds[0]:
                trail[trail > bounds[1]] = 2*bounds[1] - trail[trail > bounds[1]]
                trail[trail < bounds[0]] = 2*bounds[0] - trail[trail < bounds[0]]
        elif type == 'halving':
            trail[trail < bounds[0]] = (group[trail < bounds[0]] - bounds[0]) / 2 + bounds[0]
            trail[trail > bounds[1]] = (group[trail > bounds[1]] - bounds[1]) / 2 + bounds[1]
        else:
            raise NotSupportedErr
            # trail = trail
        # if had_action is None and logits is not None:
        return {'result': trail, 'actions': action_value, 'had_action': {'bound': type}, 'logp': logp, 'entropy': entropy}
        # else:
        #     return {'result': trail}
    

"""
Selection
"""    
class Selection(Module):
    description = """This is a selection operator, it selects individuals from parents and children to form the next population. It has seven modes:
direct is to directly compare the parent and corresponding child and select the better one;
crowding is to compare the child and its closest individual;
inherit is to keep the child and remove parent regardless of their performance;
rank is to rank the union of parent and children populations and select the top individuals;
tournament is to compare two randomly selected individual and append the better one into the next population;
roulette is to select individuals randomly with the performance based probabilities;
rand is to randomly select individuals to form the next population."""
    op_id = 15 * 8

    def __init__(self, strategy=None) -> None:
        self.control_param = [
            {'name': 'select', 'type': 'discrete', 'range': ['direct', 'crowding', 'inherit', 'rank', 'tournament', 'roulette', ], 'default': 'inherit'},
        ]
        self.strategy = strategy if strategy is not None else self.control_param[0]['default']

    def __call__(self, env, rng=None):
        if rng is None:
            rng = np.random
        # if had_action is None:
        #     action, action_value, logp, entropy = self.action_interpret(logits, softmax, fixed_action)
        #     type = self.control_param[0]['range'][action[0]]
        # else:
        #     type = had_action['select']

        type = self.strategy
        new_x = copy.deepcopy(env['group'])
        new_y = copy.deepcopy(env['cost'])
        trail = env['trail']
        trail_cost = env['trail_cost']
        NP, dim = new_x.shape
        if type == 'direct':  # directly replace parent in DE-like
            replace_id = np.where(trail_cost < env['cost'])[0]
            replaced_x = new_x[replace_id]
            replaced_y = new_y[replace_id]
            new_x[replace_id] = trail[replace_id]
            new_y[replace_id] = trail_cost[replace_id]
            
        elif type == 'crowding':
            replaced_x, replaced_y = [], []
            have_replaced = np.zeros(NP)
            for i in range(NP):
                dist = np.sum((trail[i][None, :] - new_x) ** 2, -1)
                replace_id = np.argmin(dist)
                if trail_cost[i] < new_y[replace_id]:
                    if not have_replaced[replace_id]:
                        replaced_x.append(new_x[replace_id])
                        replaced_y.append(new_y[replace_id])
                    have_replaced[replace_id] = True
                    new_x[replace_id] = trail[i]
                    new_y[replace_id] = trail_cost[i]
            replaced_x = np.array(replaced_x)
            replaced_y = np.array(replaced_y)

        elif type == 'inherit':  # PSO-like
            replaced_x = copy.deepcopy(env['group'])
            replaced_y = copy.deepcopy(env['cost'])
            new_x = copy.deepcopy(trail)
            new_y = copy.deepcopy(trail_cost)

        elif type == 'rank':
            total = np.concatenate((new_x, trail), 0)
            total_cost = np.concatenate((new_y, trail_cost), 0)
            order = np.argsort(total_cost)
            new_x = total[order[:NP]]
            new_y = total_cost[order[:NP]]
            replace_id = order[NP:]
            replaced_x = env['group'][replace_id[replace_id < NP]]
            replaced_y = env['cost'][replace_id[replace_id < NP]]
        
        elif type == 'tournament':
            total = np.concatenate((new_x, trail), 0)
            total_cost = np.concatenate((new_y, trail_cost), 0)
            new_x, new_y = [], []
            indices = np.arange(total.shape[0])
            for i in range(NP):
                ia, ib = rng.choice(indices.shape[0], size=2, replace=False)
                a, b = indices[ia], indices[ib]
                if total_cost[a] < total_cost[b]:
                    new_x.append(total[a])
                    new_y.append(total_cost[a])
                    np.delete(indices, ia)
                else:
                    new_x.append(total[b])
                    new_y.append(total_cost[b])
                    np.delete(indices, ib)
            new_x, new_y = np.array(new_x), np.array(new_y)
            replaced_x = env['group'][indices[indices < NP]]
            replaced_y = env['cost'][indices[indices < NP]]

        elif type == 'roulette':
            total = np.concatenate((new_x, trail), 0)
            total_cost = np.concatenate((new_y, trail_cost), 0)
            prob = (np.max(total_cost) - total_cost + 1e-8)/(np.sum(np.max(total_cost) - total_cost) + 1e-8)
            prob /= np.sum(prob)
            # prob = np.maximum(0, total_cost)/(np.sum(np.maximum(0, total_cost)))
            if np.sum(np.isnan(prob)) > 0:
                print(np.sum(np.isnan(prob)), np.sum(np.isnan(total_cost)), np.max(total_cost), np.min(total_cost), np.sum(np.max(total_cost) - total_cost))
                print(total_cost)
            # if np.min(prob) < 0:
            #     print(np.min(total_cost))
            #     print(total[np.argmin(total_cost)])
            #     print(env['problem'].eval(total[np.argmin(total_cost)]))
            new_id = rng.choice(total.shape[0], size=NP, replace=False, p=prob)
            new_x = total[new_id]
            new_y = total_cost[new_id]
            replace_id = np.delete(np.arange(NP), new_id[new_id < NP])
            replaced_x, replaced_y = env['group'][replace_id], env['cost'][replace_id]

        else:  # rand
            total = np.concatenate((new_x, trail), 0)
            total_cost = np.concatenate((new_y, trail_cost), 0)
            new_id = rng.choice(total.shape[0], size=NP, replace=False)
            new_x = total[new_id]
            new_y = total_cost[new_id]
            replace_id = np.delete(np.arange(NP), new_id[new_id < NP])
            replaced_x, replaced_y = env['group'][replace_id], env['cost'][replace_id]

        return new_x, new_y, replaced_x, replaced_y


"""
Communicate
"""
class Comm(Module):
    op_id = 16 * 8
    def __init__(self) -> None:
        # self.ops = ['ring', 'broadcast']
        # self.nop = len(self.ops)
        self.nop = 0
        self.default_action = -1
        self.description = """This is a communication operator, it exchanges individuals across sub-populations. It has two modes:
ring is to replace the worst individual of the latter sub-population with the best individual of the previous sub-population in a ring manner;
broadcast is to replace the worst individual of the sub-populations with the global best individuals."""
        self.feature = self.get_feature()

    def get_op(self, logits, softmax=False, fixed_action=None):
        mu, sigma = logits[:2]
        policy = Normal(mu, sigma)
        rang = (torch.arange(self.nop)) / self.nop
        probs = torch.exp(policy.log_prob(rang))
        if softmax:
            cate = Categorical(torch.softmax(probs/torch.sum(probs), -1))
        else:
            cate = Categorical(probs/torch.sum(probs))
        if fixed_action is None:
            if torch.max(probs) < 1e-4:
                action = torch.argmin((rang - mu) ** 2)
            else:
                action = cate.sample()
        else:
            action = fixed_action[0]
        logp = cate.log_prob(action)
        if torch.max(probs) < 1e-4:
            logp = 1.
        entropy = policy.entropy()
        return action, [logp], [entropy]
    
    def get_feature(self, maxLen=24):
        
        feature = torch.zeros(maxLen)
        j = 1
        for i in range(maxLen-1, -1, -1):
            if self.op_id & j:
                feature[i] = 1
            j *= 2
        return feature

    def __call__(self, logits, pid, population, softmax=False, fixed_action=None, rng=None, had_action=None, random=False):
        if rng is None:
            rng = np.random
        logp = []
        actions = []
        entropy = []
        self.nop = population.multiPop
        if logits is not None and had_action is None:
            iop, logp1, entropy = self.get_op(logits[:2], softmax, fixed_action)
            actions.append(iop)
            logp.append(logp1)
        elif had_action is not None:
            iop = had_action
        else:
            # iop = pid
            # if self.default_action >= 0:
            #     iop = self.default_action
            iop = rng.randint(self.nop) if random else pid
            actions.append(iop)
        # if topo == 'ring':
        #     for i in range(population.multiPop):
        #         next = (i + 1) % population.multiPop
        #         mask_cost = copy.deepcopy(population.cost)
        #         mask_cost[population.pop_id != next] = np.inf
        #         lw = np.argmax(mask_cost)
        #         population.cost[lw] = population.lbest[i]
        #         population.group[lw] = population.lbest_solution[i]
        # else: # broadcast
        #     for i in range(population.multiPop):
        #         if population.lbest[i] <= population.gbest:
        #             continue
        #         mask_cost = copy.deepcopy(population.cost)
        #         mask_cost[population.pop_id != i] = np.inf
        #         lw = np.argmax(mask_cost)
        #         population.cost[lw] = population.gbest
        #         population.group[lw] = population.gbest_solution
        if iop != pid:
            tar_x = population.lbest_solution[iop]
            tar_f = population.lbest[iop]
            tar_v = population.lbest_velocity[iop]
            cost = population.cost[pid].copy()
            worst = np.argmax(cost)
            population.group[pid][worst] = tar_x.copy()
            population.cost[pid][worst] = tar_f
            population.velocity[pid][worst] = tar_v.copy()
        
        return {'result': population, 'actions': actions, 'had_action': iop, 'logp': logp, 'entropy': entropy}


"""
Regrouping
"""
class Regroup(Module):
    op_id = 17 * 8
    def __init__(self) -> None:
        self.ops = ['rank', 'nearest', 'rand', 'none']
        self.nop = len(self.ops)
        self.default_action = 2
        self.description = """This is a regrouping operator, it reorganizes the population into multiple sub-populations. It has three modes:
rank is to divide population based on ranks;
rank_avg is to group at regular intervals based on ranking;
rand is to randomly group."""
        self.feature = self.get_feature()

    def get_op(self, logits, softmax=False, fixed_action=None):
        mu, sigma = logits[:2]
        policy = Normal(mu, sigma)
        rang = (torch.arange(self.nop)) / self.nop
        probs = torch.exp(policy.log_prob(rang))
        entropy = policy.entropy()
        if softmax:
            cate = Categorical(torch.softmax(probs/torch.sum(probs), -1))
        else:
            cate = Categorical(probs/torch.sum(probs))
        if fixed_action is None:
            if torch.max(probs) < 1e-4:
                action = torch.argmin((rang - mu) ** 2)
            else:
                action = cate.sample()
        else:
            action = fixed_action[0]
        logp = cate.log_prob(action)
        return action, logp, [entropy]
    
    def get_feature(self, maxLen=24):
        
        feature = torch.zeros(maxLen)
        j = 1
        for i in range(maxLen-1, -1, -1):
            if self.op_id & j:
                feature[i] = 1
            j *= 2
        return feature

    def __call__(self, logits, population, softmax=False, fixed_action=None, subsize=None, rng=None, had_action=None, random=False):
        if rng is None:
            rng = np.random
        logp = []
        actions = []
        entropy = []
        if logits is not None and had_action is None:
            iop, logp1, entropy = self.get_op(logits[:2], softmax, fixed_action)
            actions.append(iop)
            logp.append(logp1)
            grouping = self.ops[iop]
        elif had_action is not None:
            grouping = had_action
        else:
            iop = rng.randint(self.nop) if random else self.default_action
            actions.append(iop)
            # grouping = self.default_action
            grouping = self.ops[iop]
        if grouping != 'none':
            population.grouping(grouping, subsize=subsize, rng=rng)
        # if subsize is None:
        #     subsize = np.ones(population.multiPop) * (population.NP // population.multiPop)
        # if grouping == 'rank':
        #     rank = np.argsort(population.cost)
        #     pre = 0
        #     for i in range(population.multiPop):
        #         population.pop_id[(pre <= rank) * (rank < pre+subsize[i])] = i
        #         pre += subsize[i]
        # if grouping == 'rank_avg':
        #     rank = np.argsort(population.cost)
        #     population.pop_id = rank % population.multiPop
        # # elif grouping == 'Kmeans':
        # #     est = KMeans(n_clusters=population.multiPop, n_init='auto')
        # #     est.fit(population.group)
        # #     population.pop_id = est.labels_
        # else: # rand
        #     pre = 0
        #     for i in range(population.multiPop):
        #         population.pop_id[int(pre): int(pre+subsize[i])] = i
        #         pre += subsize[i]
        
        return {'result': population, 'actions': actions, 'had_action': grouping, 'logp': logp, 'entropy': entropy}


class Restart(Module):
    op_id = 18 * 8
    conditions = ['stagnation', 'conver_x', 'conver_f', 'conver_fx']
    def __init__(self, strategy, max_stag=50, Myeqs=0.25) -> None:
        self.eps_x = 0.005
        self.eps_f = 1e-8
        self.Myeqs = Myeqs
        self.stagnation_count = 0
        self.strategy = strategy
        self.max_stag = max_stag
        self.gbest = None

    def get_feature(self, maxLen=24):
        
        feature = torch.zeros(maxLen)
        j = 1
        for i in range(maxLen-1, -1, -1):
            if self.op_id & j:
                feature[i] = 1
            j *= 2
        return feature

    def get_op(self, logits, softmax=False, fixed_action=None):
        logp = []
        entropy = []

        mu1, sigma1 = logits[:2]
        policy1 = Normal(mu1, sigma1)
        rang = (torch.arange(self.nop)) / self.nop
        probs = torch.exp(policy1.log_prob(rang))
        if softmax:
            cate = Categorical(torch.softmax(probs/torch.sum(probs), -1))
        else:
            cate = Categorical(probs/torch.sum(probs))
        if fixed_action is None:
            if torch.max(probs) < 1e-4:
                sample = torch.argmin((rang - mu1) ** 2)
            else:
                sample = cate.sample()
        else:
            sample = fixed_action[0]
        logp.append(cate.log_prob(sample))
        entropy.append(policy1.entropy())

        mu2, sigma2 = logits[2:4]
        policy2 = Normal(mu2, sigma2)
        if fixed_action is None:
            ratio = policy2.sample()
        else:
            ratio = fixed_action[0]
        logp.append(policy2.log_prob(ratio))
        entropy.append(policy2.entropy())

        return sample, ratio, logp, entropy
    
    def __call__(self, env, rng=None):
        if rng is None:
            rng = np.random
        if self.gbest is None:
            self.gbest = env['lbest']
            return False
        init = False
        if self.strategy == 'stagnation':
            if np.abs(self.gbest - env['lbest']) < self.eps_f:
                self.stagnation_count += 1
            if self.stagnation_count >= self.max_stag:
                self.stagnation_count = 0
                self.gbest = env['lbest']
                init = True
        elif self.strategy == 'conver_x':
            dist = np.sqrt(np.sum((np.max(env['group'], 0) - np.min(env['group'], 0)) ** 2))
            if dist < self.eps_x * (env['bounds'][1] - env['bounds'][0]):
                init = True
        elif self.strategy == 'conver_f':
            scost = np.sort(env['cost'])[:int(self.Myeqs * env['NP'])]
            if np.max(scost) - np.min(scost) < self.eps_f:
                init = True
        elif self.strategy == 'conver_fx':
            dist = np.sqrt(np.sum((np.max(env['group'], 0) - np.min(env['group'], 0)) ** 2))
            if dist < self.eps_x * (env['bounds'][1] - env['bounds'][0]) and np.max(env['cost']) - np.min(env['cost']) < self.eps_f:
                init = True
        else:
            return False
        return init        

            
    
"""
Parameter Adaption
"""        
class Param_adaption:
    def __init__(self) -> None:
        self.history_value = None

    def reset(self):
        self.history_value = None

    def get(self,):
        pass

    def update(self, old_cost, new_cost, ratio=None):
        self.history_value = None


class SHA(Param_adaption):
    def __init__(self, size, sample='Cauchy', init_value=0.5, fail_value=-1, sigma=0.1, bound=[0, 1], ubc='clip', lbc='clip', p=2, m=1, self_param_ada=None) -> None:
        super().__init__()
        self.size = size
        self.sample = sample
        self.sigma = sigma
        self.bound = bound
        self.ubc = ubc
        self.lbc = lbc
        self.fail_value = fail_value
        self.init_p = self.p = p
        self.init_m = self.m = m
        self.self_param_ada = None if self_param_ada is None else eval(self_param_ada['class'])(*self_param_ada['args'])
        self.init_value = init_value
        self.memory = np.ones(size) * self.init_value
        self.update_index = 0
        self.history_value = None
        self.history_id = None

    def reset(self, size=None):
        if size is not None:
            self.size = size
        self.memory = np.ones(self.size) * self.init_value
        self.update_index = 0
        self.history_value = None
        self.history_id = None
        self.p = self.init_p
        self.m = self.init_m

    def update(self, old_cost, new_cost, ratio=None):
        old_cost = np.concatenate(old_cost)
        new_cost = np.concatenate(new_cost)
        if self.history_value is None:
            return
        updated_id = np.where((new_cost < old_cost) * (self.history_value > 1e-8))[0]
        succ_param = self.history_value[updated_id]
        d_fitness = (old_cost[updated_id] - new_cost[updated_id]) / (old_cost[updated_id] + 1e-9)
        if succ_param.shape[0] < 1 or np.max(succ_param) < 1e-8:
            self.memory[self.update_index] = self.fail_value if self.fail_value > 0 else self.memory[self.update_index]
        else:
            w = d_fitness / np.sum(d_fitness)
            if ratio is not None and self.self_param_ada is not None:
                self.p = self.self_param_ada.get(ratio)
            self.memory[self.update_index] = np.sum(w * (succ_param ** self.p)) / np.sum(w * (succ_param ** (self.p - self.m)))
        self.update_index = (self.update_index + 1) % self.size
        self.history_value = None
        self.history_id = None

    def get(self, size, ids=None, rng=None):
        if rng is None:
            rng = np.random

        if ids is None:
            ids = rng.randint(self.size, size=size)

        mrs = self.memory[ids]

        def cal_value(mr, sz):
            if self.sample == 'Cauchy':
                values = stats.cauchy.rvs(loc=mr, scale=self.sigma, size=sz, random_state=rng)
            else:
                values = rng.normal(loc=mr, scale=self.sigma, size=sz)
            return values
        
        values = cal_value(mrs, size)

        if self.ubc == 'regenerate' and self.lbc == 'regenerate':
            exceed = np.where((values < self.bound[0]) + (values > self.bound[1]))[0]
            while exceed.shape[0] > 0:
                values[exceed] = cal_value(mrs[exceed], exceed.shape[0])
                exceed = np.where((values < self.bound[0]) + (values > self.bound[1]))[0]
        elif self.ubc == 'regenerate':
            exceed = np.where(values > self.bound[1])[0]
            while exceed.shape[0] > 0:
                values[exceed] = cal_value(mrs[exceed], exceed.shape[0])
                exceed = np.where(values > self.bound[1])[0]
        elif self.lbc == 'regenerate':
            exceed = np.where(values < self.bound[0])[0]
            while exceed.shape[0] > 0:
                values[exceed] = cal_value(mrs[exceed], exceed.shape[0])
                exceed = np.where(values < self.bound[0])[0]

        values = np.clip(values, self.bound[0], self.bound[1])
        self.history_value = values.copy()
        self.history_id = ids
        return values
    
    def get_ids(self, size, rng=None):
        if rng is None:
            rng = np.random
        ids = rng.randint(self.size, size=size)
        return ids


class jDE(Param_adaption):
    def __init__(self, size, init_value=None, bound=[0.1, 0.9], tau=0.1) -> None:
        super().__init__()
        self.size = size
        self.bound = bound
        self.tau = tau
        self.init_value = init_value
        if init_value is None:
            self.init_value = (bound[0] + bound[1]) / 2
        self.memory = np.zeros(size) + self.init_value

    def reset(self):
        self.memory = np.zeros(self.size) + self.init_value

    def update(self, old_cost, new_cost, rng=None):
        if rng is None:
            rng = np.random
        updated_id = np.where(new_cost < old_cost)[0]
        rvs = rng.rand(len(updated_id))
        self.memory[updated_id[rvs < self.tau]] = rng.uniform(low=self.bound[0], high=self.bound[1], size=len(updated_id))

    def get(self, size=None):
        if size is None:
            size = self.size
        return self.memory[:size]
    
    def reduce(self, reduce_id):
        assert np.max(reduce_id) < self.size
        self.memory = np.delete(self.memory, reduce_id)
        self.size = self.memory.shape[0]

    def reorder(self, new_order):
        self.memory = self.memory[new_order]
        

class Linear(Param_adaption):
    def __init__(self, init_value, final_value) -> None:
        super().__init__()
        self.init_value = init_value
        self.final_value = final_value

    def get(self, ratio, size=1):
        value = self.init_value + ratio * (self.final_value - self.init_value)
        if size == 1:
            return value
        return np.ones(size) * value


class Bound_rand(Param_adaption):
    def __init__(self, low, high) -> None:
        super().__init__()
        self.low = low
        self.high = high

    def get(self, size=1, rng=None):
        if rng is None:
            rng = np.random
        return rng.uniform(low=self.low, high=self.high, size=size)


# for DMS-PSO
class DMS(Param_adaption):
    def __init__(self, init, final, switch_point=0.9) -> None:
        super().__init__()
        self.init = init
        self.final = final
        self.switch_point = switch_point

    def get(self, ratio, size=1):
        if ratio < self.switch_point:
            value = self.init
        else:
            value = self.final
        if size == 1:
            return value
        return np.ones(size) * value

"""
Operator Selection
"""
class Probability_rand():
    def __init__(self, opm, probs) -> None:
        self.opm = opm
        self.probs = probs
        assert len(probs) >= opm
        self.probs = self.probs[:opm]
        assert np.sum(probs) > 0
        self.probs /= np.sum(self.probs)
        self.init_probs = self.probs.copy()
        self.history_value = None
    
    def reset(self):
        self.probs = self.init_probs.copy()
        self.history_value = None

    def get(self, size=1, rng=None):
        if rng is None:
            rng = np.random
        value = rng.choice(self.opm, size=size, p=self.probs)
        self.history_value = value.copy()
        return value
    
    def update(self, old_cost, new_cost, ratio=None):
        # assert self.history_value is not None
        # d_fitness = np.zeros(self.opm)
        # for i in range(self.opm):
        #     id = np.where(self.history_value == i)[0]
        #     d_fitness[i] = np.mean(np.maximum(0, (old_cost[id] - new_cost[id]) / (old_cost[id] + 1e-9)))
        # self.probs = d_fitness / np.sum(d_fitness)
        # self.history_value = None
        pass


class Fitness_rand():
    def __init__(self, opm, prob_bound=[0.1, 0.9]) -> None:
        self.opm = opm
        self.probs = np.ones(opm) / opm
        self.prob_bound = prob_bound
        self.history_value = None

    def reset(self):
        self.probs = np.ones(self.opm) / self.opm
        self.history_value = None

    def get(self, size=1, rng=None):
        if rng is None:
            rng = np.random
        value = rng.choice(self.opm, size=size, p=self.probs)
        self.history_value = value.copy()
        return value
    
    def update(self, old_cost, new_cost, ratio=None):
        if self.history_value is None:
            return
        df = np.maximum(0, old_cost - new_cost)
        count_S = np.zeros(self.opm)
        for i in range(self.opm):
            count_S[i] = np.mean(df[self.history_value == i] / old_cost[self.history_value == i])
        if np.sum(count_S) > 0:
            self.probs = np.maximum(self.prob_bound[0], np.minimum(self.prob_bound[1], count_S / np.sum(count_S)))
            self.probs /= np.sum(self.probs)
        else:
            self.probs = np.ones(self.opm) / self.opm
        self.history_value = None


class Random_select():
    def __init__(self, opm) -> None:
        self.opm = opm

    def reset(self):
        pass

    def get(self, size, rng=None):
        if rng is None:
            rng = np.random
        value = rng.randint(self.opm) * np.ones(size)
        self.history_value = value.copy()
        return value
    
    def update(self, old_cost, new_cost, ratio):
        pass