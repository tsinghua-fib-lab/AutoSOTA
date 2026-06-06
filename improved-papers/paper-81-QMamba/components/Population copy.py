import scipy.stats as stats
import numpy as np
import copy
from sklearn.cluster import KMeans
from scipy.stats import qmc


class Population:
    def __init__(self, dim, NPmax=200, NPmin=20, NA=3, Xmax=100, Vmax=0.2, sample='uniform', multiPop=1, regroup=0, arch_replace='oldest', pop_reduce=None, rng=None, seed=None):
        if rng is None:
            rng = np.random
        self.NPmax = NPmax                         # the upperbound of population size
        self.NPmin = NPmin                          # the lowerbound of population size
        self.multiPop = max(1, multiPop)
        self.NPmax -= self.NPmax % self.multiPop
        self.NPmin -= self.NPmin % self.multiPop
        assert self.NPmin > 0 and self.NPmax >= self.NPmin
        self.NP = NPmax                            # the population size
        self.NA = int(NA * self.NP)                          # the size of archive(collection of replaced individuals)
        self.dim = dim                          # the dimension of individuals
        self.Vmax = Vmax
        self.cost = np.zeros(self.NP)           # the cost of individuals
        self.Xmin = -Xmax         # the upperbound of individual value
        self.Xmax = Xmax          # the lowerbound of individual value
        self.sample = sample
        self.regroup = regroup
        self.pop_reduce = pop_reduce
        self.regroup_type = None

        self.cbest = np.inf                       # the best cost in current population, initialize as 1e15
        self.cbest_solution = np.zeros(dim)                      # the index of individual with the best cost
        self.cbest_velocity = np.zeros(dim)
        self.gbest = self.init_max = self.pre_gb = np.inf                       # the global best cost
        self.gbest_solution = np.zeros(dim)     # the individual with global best cost
        self.gbest_velocity = np.zeros(dim)     # the individual with global best cost
        self.lbest = np.ones(self.multiPop) * np.inf                       # the global best cost
        self.lbest_solution = np.zeros((self.multiPop, dim))
        self.lbest_velocity = np.zeros((self.multiPop, dim))
        self.pbest = np.ones(self.NP) * np.inf                        # the global best cost
        self.pbest_solution = np.zeros((self.NP, dim))

        self.group = self.initialize_group(sample, rng=rng, seed=seed)    # the population numpop x np x dim
        self.velocity = (rng.rand(self.NP, self.dim) * 2 - 1) * self.Vmax
        self.archive = []             # the archive(collection of replaced individuals)
        self.archive_cost = []             # the archive(collection of replaced individuals)
        self.archive_index = -1
        self.arch_replace = arch_replace
        self.pop_id = np.zeros(self.NP)

        self.minimal_subsize = 5 

    # generate an initialized population with size(default self population size)
    def initialize_group(self, sample, size=None, rng=None, seed=None):
        if size is None:
            size = self.NP
        if rng is None:
            rng = np.random
        if sample == 'gaussian':
            return rng.normal(loc=(self.Xmax + self.Xmin) / 2, scale=(self.Xmax - self.Xmin) * 0.2, size=(size, self.dim))
        elif sample == 'sobol':
            sampler = qmc.Sobol(d=self.dim, seed=seed)
            return sampler.random(size) * (self.Xmax - self.Xmin) + self.Xmin
        elif sample == 'lhs':
            sampler = qmc.LatinHypercube(d=self.dim, seed=seed)
            return sampler.random(size) * (self.Xmax - self.Xmin) + self.Xmin
        elif sample == 'halton':
            sampler = qmc.Halton(d=self.dim, seed=seed)
            return sampler.random(size) * (self.Xmax - self.Xmin) + self.Xmin
        else: # uniform
            return rng.rand(size, self.dim) * (self.Xmax - self.Xmin) + self.Xmin

    # initialize cost
    def initialize_costs(self, problem):
        self.cost = problem.eval(self.group)
        self.gbest = self.cbest = self.pre_gb = np.min(self.cost)
        self.init_max = np.max(self.cost)
        self.cbest_solution = self.gbest_solution = self.group[np.argmin(self.cost)]
        self.gbest_velocity = self.cbest_velocity = self.velocity[np.argmin(self.cost)]
        self.pbest = copy.deepcopy(self.cost)
        self.pbest_solution = copy.deepcopy(self.group)
        self.lbest = np.ones(self.multiPop) * self.gbest
        self.lbest_solution = np.zeros((self.multiPop, self.dim)) + self.gbest_solution
        self.lbest_solution = np.zeros((self.multiPop, self.dim)) + self.gbest_velocity

    def grouping(self, grouping, subsize=None, rng=None):
        if rng is None:
            rng = np.random
        self.regroup_type = grouping
        if subsize is None:
            subsize = np.ones(self.multiPop, dtype=np.int32) * int(self.NP // self.multiPop)
        if grouping == 'rank':
            rank = np.argsort(self.cost)
            pre = 0
            for i in range(self.multiPop):
                self.pop_id[(pre <= rank) * (rank < pre+subsize[i])] = i
                pre += subsize[i]
        # if grouping == 'rank_avg':
        #     rank = np.argsort(self.cost)
        #     self.pop_id = rank % self.multiPop
        if grouping == 'nearest':
            ban = []
            for i in range(self.multiPop):
                for j in range(self.NP):
                    if j not in ban:
                        ban.append(j)
                        break
                dist = np.argsort(np.sum((self.group[j,None] - self.group) ** 2, -1))
                count = 1
                for k in range(self.NP):
                    if dist[k] not in ban:
                        self.pop_id[dist[k]] = i
                        ban.append(dist[k])
                        count += 1
                    if count >= subsize[i]:
                        break
        # elif grouping == 'Kmeans':
        #     est = KMeans(n_clusters=self.multiPop, n_init='auto')
        #     est.fit(self.group)
        #     self.pop_id = est.labels_
        else: # rand
            # pre = 0
            # for i in range(self.multiPop):
            #     self.pop_id[pre: ] = i
            #     pre += subsize[i]
            order = rng.permutation(len(self.cost))
            pre = 0
            for i in range(self.multiPop):
                self.pop_id[order[pre:pre+subsize[i]]] = i
                pre += subsize[i]

    def replace_individuals(self, new_group, rng=None):
        if rng is None:
            rng = np.random
        size = new_group.shape[0]
        replaced = np.argsort(self.cost)[-size:]
        if self.group is None:
            self.group = new_group
            self.velocity = (rng.rand(size, self.dim) * 2 - 1) * self.Vmax
        else:    
            self.group[replaced] = new_group
            self.velocity[replaced] = (rng.rand(size, self.dim) * 2 - 1) * self.Vmax

    # update archive, join new individual
    def update_archive(self, x, y, rng=None):
        if rng is None:
            rng = np.random
        if len(self.archive) < self.NA:
            self.archive.append(x)
            self.archive_cost.append(y)
        else:
            if self.arch_replace == 'worst':
                archive_index = np.argmax(self.archive_cost)
            elif self.arch_replace == 'oldest':
                archive_index = self.archive_index
                self.archive_index = (self.archive_index + 1) % self.NA
            else:
                archive_index = rng.randint(self.NA)
            self.archive[archive_index] = x
            self.archive_cost[archive_index] = y

    def update_lbest(self):
        for i in range(self.multiPop):
            mask_cost = copy.deepcopy(self.pbest)
            mask_cost[self.pop_id != i] = np.inf
            lb = np.argmin(mask_cost)
            self.lbest[i] = self.pbest[lb]
            self.lbest_solution[i] = self.pbest_solution[lb].copy()
            self.lbest_velocity[i] = self.velocity[lb].copy()

    def get_subpops(self, ipop=None):
        subpops = []
        for id in range(self.multiPop):
            if ipop is not None and id != ipop:
                continue
            ret = {}
            ret['group'] = self.group[self.pop_id == id]
            ret['cost'] = self.cost[self.pop_id == id]
            ret['velocity'] = self.velocity[self.pop_id == id]
            ret['NP'] = ret['group'].shape[0]
            ret['Vmax'] = self.Vmax
            ret['init_max'] = self.init_max

            ret['cbest_solutions'] = self.cbest_solution
            ret['cbest'] = self.cbest
            ret['cbest_velocity'] = self.cbest_velocity
            ret['gbest_solutions'] = self.gbest_solution
            ret['gbest'] = self.gbest
            ret['gbest_velocity'] = self.gbest_velocity
            ret['lbest_solutions'] = self.lbest_solution[id]
            ret['lbest'] = self.lbest[id]
            ret['lbest_velocity'] = self.lbest_velocity
            ret['pbest_solutions'] = self.pbest_solution[self.pop_id == id]
            ret['pbest'] = self.pbest[self.pop_id == id]

            ret['archive'] = self.archive
            ret['a_cost'] = self.archive_cost

            ret['bounds'] = [self.Xmin, self.Xmax]

            subpops.append(ret)
        
        return subpops if ipop is None else subpops[0]

    def update_subpop(self, new_xs, new_ys, ignore_vel=False):
        old_x = self.group.copy()
        old_y = self.cost.copy()

        for id in range(self.multiPop):
            new_x, new_y = new_xs[id], new_ys[id]
            self.group[self.pop_id == id] = new_x
            self.cost[self.pop_id == id] = new_y
            if self.lbest[id] > np.min(new_y):
                self.lbest_solution[id] = new_x[np.argmin(new_y)]
                self.lbest[id] = np.min(new_y)
        
        if not ignore_vel:
            self.velocity = self.group - old_x

        self.cbest = np.min(self.cost)
        self.cbest_solution = self.group[np.argmin(self.cost)].copy()
        
        self.pre_gb = self.gbest
        if self.gbest > self.cbest:
            self.gbest = self.cbest
            self.gbest_solution = self.cbest_solution.copy()

        replace_id = np.where(self.cost < self.pbest)[0]
        self.pbest[replace_id] = self.cost[replace_id]
        self.pbest_solution[replace_id] = self.group[replace_id]

    def reduction(self, step_ratio, reduce_num=None, specific_id=None, rng=None):
        for i in range(self.multiPop):
            if reduce_num is None or (isinstance(reduce_num, list) and reduce_num[i] is None):
                if self.pop_reduce[i] == 'linear':
                    num = max(0, self.subNP[i] - int((self.NPmax[i] - self.NPmin[i]) * (1 - step_ratio) + self.NPmin[i]))
                elif self.pop_reduce == 'nonlinear':
                    num = max(0, self.subNP[i] - int(self.NPmax[i] + (self.NPmin[i] - self.NPmax[i]) * np.power(step_ratio, 1-step_ratio)))
                else:
                    continue
            if num < 1:
                continue
            mask_cost = copy.deepcopy(self.cost)
            if specific_id is not None:
                mask_cost[self.pop_id != specific_id] = -np.inf
            order = np.argsort(mask_cost)
            removed_id = order[-num:]
            for id in removed_id:
                self.update_archive(self.group[id], self.cost[id])
            self.group = np.delete(self.group, removed_id, 0)
            self.cost = np.delete(self.cost, removed_id, 0)
            self.pbest = np.delete(self.pbest, removed_id, 0)
            self.pbest_solution = np.delete(self.pbest_solution, removed_id, 0)
            self.velocity = np.delete(self.velocity, removed_id, 0)
            self.pop_id = np.delete(self.pop_id, removed_id, 0)
            self.NP -= num
        # balance subpops
        # if self.multiPop > 1 and not self.check_balance():
            # print(self.NP, self.multiPop, reduce_num, step_ratio, self.NPmax, self.NPmin)
            # self.grouping(self.regroup_type, rng=rng)

    def check_balance(self):
        for id in range(self.multiPop):
            if np.sum(self.pop_id == id) < self.minimal_subsize:
                # print(np.sum(self.pop_id == id))
                return False
        return True
    

class Sequence:
    def __init__(self, op_list: list) -> None:
        self.op_list = op_list

    def __call__(self, population):
        for op in self.op_list:
            population = op(population)
        return population
    

class Iteration:
    def __init__(self, op_list: list, callback: callable) -> None:
        self.op_list = op_list
        self.callback = callback

    def __call__(self, population):
        while not self.callback(population):
            for op in self.op_list:
                population = op(population)
        return population
    

class Condition:
    def __init__(self, op_list: list, conditions: list[callable]) -> None:
        assert len(op_list) == len(conditions)
        self.op_list = op_list
        self.conditions = conditions

    def __call__(self, population):
        for ic, condition in enumerate(self.conditions):
            if condition(population):
                return self.op_list[ic](population)
        return population
    
# class Scatter:
#     def __init__(self, op_list: list, scatter_rule: list[callable]) -> None: