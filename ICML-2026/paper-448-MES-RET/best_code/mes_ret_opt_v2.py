#!/usr/bin/env python3
"""Optimized MES-RET for reproduction - single run with controlled threading."""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import numpy as np
import cma
import time
import json
import sys

# ============================================================================
# Task Setup
# ============================================================================

def build_cec2017_tasks():
    """Build CEC 2017 many-task benchmark."""
    import opfunu.cec_based as cec
    func_classes = []
    func_indices = list(range(1, 31))
    func_indices.remove(2)  # F2 excluded in CEC 2017
    for idx in func_indices:
        cls_name = "F%d2017" % idx
        cls = getattr(cec, cls_name, None)
        if cls is not None:
            func_classes.append(cls)
    dims = [10, 30, 50]
    tasks = []
    for dim in dims:
        for func_cls in func_classes:
            func = func_cls(ndim=dim)
            tasks.append({
                'func': func, 'dim': dim,
                'lb': np.array(func.lb, dtype=float),
                'ub': np.array(func.ub, dtype=float),
            })
    return tasks


def evaluate(func_obj, x, lb, ub):
    """Evaluate solutions on a CEC 2017 task."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    xc = np.clip(x, lb, ub)
    return np.array([func_obj.evaluate(xi) for xi in xc])


# ============================================================================
# MES-RET Algorithm
# ============================================================================

class MESRET:
    def __init__(self, tasks, seed=42, sigma0=0.3, tau=1, popsize=100, max_fe=4350000):
        self.tasks = tasks
        self.K = len(tasks)
        self.seed = seed
        self.sigma0 = sigma0
        self.tau = tau
        self.popsize = popsize
        self.mu = popsize // 2
        self.max_fe = max_fe
        self.dims = [t['dim'] for t in tasks]
        self.max_dim = max(self.dims)
        self.fe = 0
        self.gen = 0
        self.best_objs = np.full(self.K, np.inf)
        self.init_objs = np.full(self.K, np.inf)
        self.stop_flag = np.zeros(self.K, dtype=bool)
        self.cma_instances = []
        self.prev_objs = np.full(self.K, np.inf)
        self.prev_sigmas = np.ones(self.K)
        self.prev_stds = [np.ones(self.dims[t]) for t in range(self.K)]
        self.agg_mdec = np.full(self.max_dim, np.nan)
        self.agg_std_ratio = np.full(self.max_dim, np.nan)
        self.agg_sigma_ratio = 1.0

    def run(self):
        np.random.seed(self.seed)
        # Initialize CMA-ES instances
        for t in range(self.K):
            task = self.tasks[t]
            dim = task['dim']
            x0 = task['lb'] + np.random.rand(dim) * (task['ub'] - task['lb'])
            sigma0_val = self.sigma0 * (task['ub'] - task['lb'])
            es = cma.CMAEvolutionStrategy(x0, sigma0_val[0], {
                'popsize': self.popsize, 'maxfevals': np.inf,
                'verbose': -9, 'seed': self.seed + t * 10000,
                'CMA_diagonal': False,
            })
            self.cma_instances.append(es)

        # Initial evaluation for all tasks
        for t in range(self.K):
            es = self.cma_instances[t]
            sols = es.ask()
            x = np.array(sols)
            objs = evaluate(self.tasks[t]['func'], x,
                          self.tasks[t]['lb'], self.tasks[t]['ub'])
            es.tell(sols, objs.tolist())
            self.fe += len(sols)
            self.best_objs[t] = np.min(objs)
            self.init_objs[t] = self.best_objs[t]
        self.prev_objs = self.best_objs.copy()

        # Main loop
        while self.fe < self.max_fe and not np.all(self.stop_flag):
            self.gen += 1

            # Phase 1: Self evolution
            for t in range(self.K):
                if self.stop_flag[t]:
                    continue
                es = self.cma_instances[t]
                self.prev_objs[t] = self.best_objs[t]
                self.prev_sigmas[t] = es.sigma
                self.prev_stds[t] = np.sqrt(np.diag(es.C))
                sols = es.ask()
                if self.tau > 0 and self.gen > 10:
                    self._transfer(t, sols)
                x = np.array(sols)
                objs = evaluate(self.tasks[t]['func'], x,
                              self.tasks[t]['lb'], self.tasks[t]['ub'])
                es.tell(sols, objs.tolist())
                self.fe += len(sols)
                bi = np.argmin(objs)
                if objs[bi] < self.best_objs[t]:
                    self.best_objs[t] = objs[bi]
                if es.sigma * max(np.max(np.abs(es.pc)),
                                  np.max(np.sqrt(np.diag(es.C)))) < 1e-12:
                    self.stop_flag[t] = True

            if np.all(self.stop_flag):
                self.stop_flag[:] = False

            # Phase 2: Reward calculation
            if np.random.random() < 1 - self.fe / self.max_fe:
                rewards = self._reward_fit()
            else:
                rewards = self._reward_div()

            # Phase 3: Knowledge aggregation
            self._aggregate(rewards)

            # Phase 4: Reward-weighted evaluation
            for _ in range(self.K):
                t = np.random.choice(self.K, p=rewards)
                if self.stop_flag[t]:
                    continue
                es = self.cma_instances[t]
                self.prev_objs[t] = self.best_objs[t]
                self.prev_sigmas[t] = es.sigma
                self.prev_stds[t] = np.sqrt(np.diag(es.C))
                sols = es.ask()
                if self.tau > 0 and self.gen > 10:
                    self._transfer(t, sols)
                x = np.array(sols)
                objs = evaluate(self.tasks[t]['func'], x,
                              self.tasks[t]['lb'], self.tasks[t]['ub'])
                es.tell(sols, objs.tolist())
                self.fe += len(sols)
                bi = np.argmin(objs)
                if objs[bi] < self.best_objs[t]:
                    self.best_objs[t] = objs[bi]

            if self.gen % 25 == 0:
                active = np.sum(~self.stop_flag)
                if active > 0:
                    avg_b = np.mean(self.best_objs[~self.stop_flag])
                else:
                    avg_b = np.inf
                print("  Gen %d: FE=%d/%d, active=%d, avg_best=%.2e" % (
                    self.gen, self.fe, self.max_fe, active, avg_b))

        return {
            'task_best': self.best_objs.tolist(),
            'total_fe': self.fe,
            'generations': self.gen,
        }

    def _reward_fit(self):
        imp = np.zeros(self.K)
        for t in range(self.K):
            if self.stop_flag[t]:
                continue
            imp[t] = max(0, self.prev_objs[t] - self.best_objs[t]) / (
                abs(self.init_objs[t] - self.prev_objs[t]) + 1e-12)
        active = ~self.stop_flag
        if not np.any(active):
            return np.ones(self.K) / self.K
        vals = imp[active]
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin < 1e-9:
            norm = np.zeros_like(vals)
        else:
            norm = (vals - vmin) / (vmax - vmin)
        exps = np.exp(norm)
        probs = exps / exps.sum()
        reward = np.zeros(self.K)
        reward[active] = probs
        if np.any(np.isnan(reward)):
            reward = np.ones(self.K) / self.K
        return reward

    def _reward_div(self):
        div = np.zeros(self.K)
        for t in range(self.K):
            if self.stop_flag[t]:
                continue
            div[t] = (
                self.prev_sigmas[t] * np.sum(self.prev_stds[t] ** 2)
                / self.dims[t]
            )
        active = ~self.stop_flag
        if not np.any(active):
            return np.ones(self.K) / self.K
        dmin, dmax = div[active].min(), div[active].max()
        if dmax - dmin < 1e-12:
            return np.ones(self.K) / self.K
        nd = (div[active] - dmin) / (dmax - dmin + 1e-12)
        reward = np.zeros(self.K)
        reward[active] = nd / (nd.sum() + 1e-12)
        if np.any(np.isnan(reward)):
            reward = np.ones(self.K) / self.K
        return reward

    def _transfer(self, t, sols):
        task = self.tasks[t]
        dim = task['dim']
        es = self.cma_instances[t]
        m = es.mean
        sigma = es.sigma
        B = es.B
        D = es.D
        tr_num = max(1, int(np.round(self.tau)))
        # Mean aggregation transfer
        am = self.agg_mdec[:dim].copy()
        if np.any(np.isnan(am)):
            am = m.copy()
        x_nat = np.array(sols)
        if len(x_nat) > 0:
            m_step = np.mean(np.sqrt(np.sum(
                (x_nat[:, :dim] - m[:dim]) ** 2, axis=1)))
        else:
            m_step = sigma * np.sqrt(dim)
        for i in range(min(tr_num, len(sols))):
            z = B @ (D * np.random.randn(dim))
            u = am - m[:dim] + sigma * z
            nu = np.linalg.norm(u)
            if nu > 0:
                sols[i] = m[:dim] + (u / nu) * m_step
            else:
                sols[i] = m[:dim]
        # Covariance aggregation transfer
        v = self.agg_sigma_ratio * self.agg_std_ratio[:dim]
        if np.any(np.isnan(v)):
            v = np.ones(dim)
        for i in range(tr_num, min(2 * tr_num, len(sols))):
            u = v * sigma * np.sqrt(np.diag(es.C)[:dim]) * np.random.randn(dim)
            sols[i] = m[:dim] + u

    def _aggregate(self, rewards):
        active = ~self.stop_flag
        mdecs = np.full((self.K, self.max_dim), np.nan)
        std_ratios = np.full((self.K, self.max_dim), np.nan)
        sigma_ratios = np.ones(self.K)
        for t in range(self.K):
            if self.stop_flag[t]:
                continue
            es = self.cma_instances[t]
            dim = self.dims[t]
            mdecs[t, :dim] = es.mean[:dim]
            cs = np.sqrt(np.diag(es.C)[:dim])
            ps = self.prev_stds[t][:dim]
            r = np.ones(dim)
            nz = ps > 1e-12
            r[nz] = cs[nz] / ps[nz]
            std_ratios[t, :dim] = r
            sigma_ratios[t] = (
                es.sigma / self.prev_sigmas[t]
                if self.prev_sigmas[t] > 1e-12 else 1.0
            )
        for j in range(self.max_dim):
            idx = active & ~np.isnan(mdecs[:, j])
            if np.any(idx):
                r = rewards[idx]
                rn = r / r.sum()
                self.agg_mdec[j] = np.sum(rn * mdecs[idx, j])
        rw = rewards[active]
        if rw.sum() > 0:
            rw = rw / rw.sum()
        else:
            rw = np.ones(np.sum(active)) / max(1, np.sum(active))
        self.agg_sigma_ratio = np.sum(rw * sigma_ratios[active])
        for j in range(self.max_dim):
            idx = active & ~np.isnan(std_ratios[:, j])
            if np.any(idx):
                r = rewards[idx]
                rn = r / r.sum()
                self.agg_std_ratio[j] = np.sum(rn * std_ratios[idx, j])


# ============================================================================
# CMA-ES Baseline
# ============================================================================

class CMAES:
    def __init__(self, tasks, seed=42, sigma0=0.3, popsize=100, max_fe=4350000):
        self.tasks = tasks
        self.K = len(tasks)
        self.seed = seed
        self.sigma0 = sigma0
        self.popsize = popsize
        self.max_fe = max_fe
        self.max_fe_per_task = max_fe // self.K

    def run(self):
        np.random.seed(self.seed)
        total_fe = 0
        bests = np.full(self.K, np.inf)
        for t in range(self.K):
            task = self.tasks[t]
            dim = task['dim']
            x0 = task['lb'] + np.random.rand(dim) * (task['ub'] - task['lb'])
            s0 = self.sigma0 * (task['ub'] - task['lb'])
            es = cma.CMAEvolutionStrategy(x0, s0[0], {
                'popsize': self.popsize,
                'maxfevals': self.max_fe_per_task,
                'verbose': -9,
                'seed': self.seed + t * 10000,
                'CMA_diagonal': False,
            })
            bo = np.inf
            while not es.stop():
                sols = es.ask()
                x = np.array(sols)
                x = np.clip(x, task['lb'], task['ub'])
                objs = np.array([task['func'].evaluate(xi) for xi in x])
                es.tell(sols, objs.tolist())
                total_fe += len(sols)
                bi = np.argmin(objs)
                if objs[bi] < bo:
                    bo = objs[bi]
            bests[t] = bo
            if (t + 1) % 28 == 0:
                print("  CMA-ES: %d/%d tasks done, FE=%d" % (
                    t + 1, self.K, total_fe))
        return {'task_best': bests.tolist(), 'total_fe': total_fe}


# ============================================================================
# Main Experiment
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("MES-RET Reproduction - Paper Budget")
    print("Paper: Breaking Multi-Task Curse (Li et al., ICML 2026)")
    print("=" * 60)

    t0_total = time.time()
    tasks = build_cec2017_tasks()
    K = len(tasks)
    n_funcs = K // 3
    max_fe = 3000 * 50 * n_funcs
    print("Tasks: %d (%d functions x 3 dims), maxFE=%d" % (K, n_funcs, max_fe))

    # MES-RET
    print("\n[1/2] Running MES-RET...")
    t0 = time.time()
    mes = MESRET(tasks, seed=42, sigma0=0.3, tau=1, popsize=100, max_fe=max_fe)
    mes_res = mes.run()
    t_mes = time.time() - t0
    print("MES-RET done: %.0fs, FE=%d" % (t_mes, mes_res['total_fe']))

    # CMA-ES
    print("\n[2/2] Running CMA-ES baseline...")
    t0 = time.time()
    cma_es = CMAES(tasks, seed=42, sigma0=0.3, popsize=100, max_fe=max_fe)
    cma_res = cma_es.run()
    t_cma = time.time() - t0
    print("CMA-ES done: %.0fs, FE=%d" % (t_cma, cma_res['total_fe']))

    # Metrics
    mes_best = np.array(mes_res['task_best'])
    cma_best = np.array(cma_res['task_best'])
    better = int(np.sum(mes_best < cma_best))
    worse = int(np.sum(mes_best > cma_best))
    tie = int(np.sum(np.abs(mes_best - cma_best) < 1e-12))

    sep = "=" * 60
    print("\n" + sep)
    print("RESULTS (pairwise MES-RET vs CMA-ES)")
    print(sep)
    print("  #Better (MES-RET > CMA-ES): %d/%d" % (better, K))
    print("  #Worse  (CMA-ES > MES-RET): %d/%d" % (worse, K))
    print("  #Tie:                         %d/%d" % (tie, K))
    print("  Total time: %.0fs" % (time.time() - t0_total))
    print("  MES-RET time: %.0fs" % t_mes)
    print("  CMA-ES time: %.0fs" % t_cma)
    print(sep)

    results = {
        'paper_id': 448,
        'n_runs': 1,
        'n_tasks': K,
        'max_fe': max_fe,
        'mes_ret_time_s': t_mes,
        'cma_es_time_s': t_cma,
        '#Better': better,
        '#Worse': worse,
        '#Tie': tie,
        'mes_ret_task_bests': mes_res['task_best'],
        'cma_task_bests': cma_res['task_best'],
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open('/repo/reproduction_full.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to /repo/reproduction_full.json")
