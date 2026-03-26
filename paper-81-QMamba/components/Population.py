import scipy.stats as stats
import numpy as np
import copy
from sklearn.cluster import KMeans
from scipy.stats import qmc


class Population:
    def __init__(self, dim, NPmax=[100], NPmin=[4], NA=3, Xmax=100, Vmax=0.2, sample='uniform', multiPop=1, regroup=0, arch_replace='oldest', pop_reduce=None, rng=None, seed=None):
        self.NPmax = NPmax                         # the upperbound of population size
        self.NPmin = NPmin                          # the lowerbound of population size
        self.multiPop = max(1, multiPop)
        # self.NPmax -= self.NPmax % self.multiPop
        # self.NPmin -= self.NPmin % self.multiPop
        # assert self.NPmin > 0 and self.NPmax >= self.NPmin
        self.NP = np.sum(NPmax)                            # the population size
        self.subNP = NPmax
        self.NA = int(NA * self.NP)                          # the size of archive(collection of replaced individuals)
        self.dim = dim                          # the dimension of individuals
        self.Vmax = Vmax
        self.cost = []           # the cost of individuals
        self.Xmin = -Xmax         # the upperbound of individual value
        self.Xmax = Xmax          # the lowerbound of individual value
        self.sample = sample
        self.regroup = regroup
        self.pop_reduce = pop_reduce
        self.regroup_type = None
        self.arch_replace = arch_replace

        if rng is None:
            rng = np.random
        self.cbest = np.inf                       # the best cost in current population, initialize as 1e15
        self.cbest_solution = np.zeros(self.dim)                      # the index of individual with the best cost
        self.cbest_velocity = np.zeros(self.dim)
        self.gbest = self.init_max = self.pre_gb = np.inf                       # the global best cost
        self.gbest_solution = np.zeros(self.dim)     # the individual with global best cost
        self.gbest_velocity = np.zeros(self.dim)     # the individual with global best cost
        self.lbest = np.ones(self.multiPop) * np.inf                       # the global best cost
        self.lbest_solution = np.zeros((self.multiPop, self.dim))
        self.lbest_velocity = np.zeros((self.multiPop, self.dim))
        self.pbest = [np.ones(self.subNP[i]) * np.inf for i in range(self.multiPop)]                        # the global best cost
        self.pbest_solution = [np.zeros((self.subNP[i], self.dim)) for i in range(self.multiPop)]

        self.group = [self.initialize_group(self.sample, size=self.subNP[i], rng=rng, seed=seed) for i in range(self.multiPop)]    # the population numpop x np x dim
        self.velocity = [(rng.rand(self.subNP[i], self.dim) * 2 - 1) * self.Vmax for i in range(self.multiPop)]
        self.archive = []             # the archive(collection of replaced individuals)
        self.archive_cost = []             # the archive(collection of replaced individuals)
        self.archive_index = -1
        # self.pop_id = np.zeros(self.NP)


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
        self.cost = []
        for i in range(self.multiPop):
            self.cost.append(problem.eval(self.group[i]))
        self.gbest = self.cbest = self.pre_gb = np.min(np.concatenate(self.cost))
        self.init_max = np.max(np.concatenate(self.cost))
        self.cbest_solution = self.gbest_solution = np.concatenate(self.group)[np.argmin(np.concatenate(self.cost))]
        self.gbest_velocity = self.cbest_velocity = np.concatenate(self.velocity)[np.argmin(np.concatenate(self.cost))]
        self.pbest = copy.deepcopy(self.cost)
        self.pbest_solution = copy.deepcopy(self.group)
        self.pbest_velocity = copy.deepcopy(self.velocity)
        self.update_lbest()
        # self.lbest = np.ones(self.multiPop) * self.gbest
        # self.lbest_solution = np.zeros((self.multiPop, self.dim)) + self.gbest_solution
        # self.lbest_velocity = np.zeros((self.multiPop, self.dim)) + self.gbest_velocity


    def grouping(self, grouping, subsize=None, rng=None):
        if rng is None:
            rng = np.random
        self.regroup_type = grouping
        if subsize is None:
            subsize = self.subNP
        pop_id = np.zeros(self.NP)
        group = np.concatenate(self.group)
        cost = np.concatenate(self.cost)

        if grouping == 'rank':
            rank = np.argsort(cost)
            pre = 0
            for i in range(self.multiPop):
                pop_id[(pre <= rank) * (rank < pre+subsize[i])] = i
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
                dist = np.argsort(np.sum((group[j,None] - group) ** 2, -1))
                pop_id[j] = i
                count = 1
                for k in range(self.NP):
                    if dist[k] not in ban:
                        pop_id[dist[k]] = i
                        ban.append(dist[k])
                        count += 1
                    if count >= subsize[i]:
                        break
        # elif grouping == 'Kmeans':
        #     est = KMeans(n_clusters=self.multiPop, n_init='auto')
        #     est.fit(self.group)
        #     self.pop_id = est.labels_
        else:  # rand
            order = rng.permutation(self.NP)
            pre = 0
            for i in range(self.multiPop):
                pop_id[order[pre:pre+subsize[i]]] = i
                pre += subsize[i]

        velocity = np.concatenate(self.velocity)
        pbest = np.concatenate(self.pbest)
        pbest_solution = np.concatenate(self.pbest_solution)
        pbest_velocity = np.concatenate(self.pbest_velocity)
        self.group = []
        self.cost = []
        self.velocity = []
        self.pbest = []
        self.pbest_solution = []
        self.pbest_velocity = []
        for i in range(self.multiPop):
            self.group.append(group[pop_id == i])
            self.cost.append(cost[pop_id == i])
            self.velocity.append(velocity[pop_id == i])
            self.pbest_solution.append(pbest_solution[pop_id == i])
            self.pbest.append(pbest[pop_id == i])
            self.pbest_velocity.append(pbest_velocity[pop_id == i])
        self.update_lbest()

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
            lb = np.argmin(self.cost[i])
            self.lbest[i] = self.cost[i][lb]
            self.lbest_solution[i] = self.group[i][lb].copy()
            self.lbest_velocity[i] = self.velocity[i][lb].copy()

    def get_subpops(self, ipop=None):
        subpops = []
        for id in range(self.multiPop):
            if ipop is not None and id != ipop:
                continue
            ret = {}
            ret['id'] = id
            ret['group'] = self.group[id]
            ret['cost'] = self.cost[id]
            ret['velocity'] = self.velocity[id]
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
            ret['pbest_solutions'] = self.pbest_solution[id]
            ret['pbest'] = self.pbest[id]

            ret['archive'] = self.archive
            ret['a_cost'] = self.archive_cost

            ret['bounds'] = [self.Xmin, self.Xmax]

            subpops.append(ret)
        
        return subpops if ipop is None else subpops[0]

    def update_subpop(self, new_xs, new_ys, ignore_vel=False):
        old_x = copy.deepcopy(self.group)
        old_y = copy.deepcopy(self.cost)
        
        for id in range(self.multiPop):
            new_x, new_y = new_xs[id], new_ys[id]
            self.group[id] = new_x
            self.cost[id] = new_y
            replace_id = np.where(self.cost[id] < self.pbest[id])[0]
            self.pbest[id][replace_id] = self.cost[id][replace_id]
            self.pbest_solution[id][replace_id] = self.group[id][replace_id]      

            if not ignore_vel:
                self.velocity[id] = np.clip(self.group[id] - old_x[id], -self.Vmax, self.Vmax)
        self.update_lbest()

        self.cbest = np.min(np.concatenate(self.cost))
        self.cbest_solution = np.concatenate(self.group)[np.argmin(np.concatenate(self.cost))]
        
        self.pre_gb = self.gbest
        if self.gbest > self.cbest:
            self.gbest = self.cbest
            self.gbest_solution = self.cbest_solution.copy()


    def reduction(self, step_ratio, reduce_num=None, specific_id=None, rng=None):
        for i in range(self.multiPop):
            if specific_id is not None and i != specific_id:
                continue
            if reduce_num is None or (isinstance(reduce_num, list) and reduce_num[i] is None):
                if self.pop_reduce[i] == 'linear':
                    num = max(0, self.subNP[i] - int(np.round((self.NPmax[i] - self.NPmin[i]) * (1 - step_ratio) + self.NPmin[i])))
                elif self.pop_reduce[i] == 'nonlinear':
                    num = max(0, self.subNP[i] - int(np.round(self.NPmax[i] + (self.NPmin[i] - self.NPmax[i]) * np.power(step_ratio, 1-step_ratio))))
                else:
                    continue
            if num < 1:
                continue
            order = np.argsort(self.cost[i])
            removed_id = order[-num:]
            for id in removed_id:
                self.update_archive(self.group[i][id], self.cost[i][id])
            self.group[i] = np.delete(self.group[i], removed_id, 0)
            self.cost[i] = np.delete(self.cost[i], removed_id, 0)
            self.pbest[i] = np.delete(self.pbest[i], removed_id, 0)
            self.pbest_solution[i] = np.delete(self.pbest_solution[i], removed_id, 0)
            self.pbest_velocity[i] = np.delete(self.pbest_velocity[i], removed_id, 0)
            self.velocity[i] = np.delete(self.velocity[i], removed_id, 0)
            self.NP -= num
            self.subNP[i] -= num
