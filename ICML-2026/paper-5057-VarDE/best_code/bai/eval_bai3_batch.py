"""Batch eval: test 3 VarDE configs on BAI.3 in one pass (n=2000 sub-sample)."""
import os, time, random
import numpy as np
from env import Bandit
from baseline import CRA
from VarDE import VarDE_lse
from tqdm import tqdm

random.seed(42)

means = [0.4]
stds = [0.4]
for i in range(10, 23):
    means.append(0.4 - 0.9**i)
    stds.append(0.4 - 0.9**i)
random.shuffle(stds)
true_best = 0
T = 200
n = 2000

env = Bandit(distribution="gaussian", means=means, stds=stds)

configs = [
    ("VarDE(tau=0.10, ve=0.5) [current best]", {"tau": 0.10, "variance_exponent": 0.5}),
    ("VarDE(tau=0.07, ve=0.5)", {"tau": 0.07, "variance_exponent": 0.5}),
    ("VarDE(tau=0.05, ve=0.5)", {"tau": 0.05, "variance_exponent": 0.5}),
]

rec = {"CR-A": [], **{name: [] for name, _ in configs}}

print(f"BAI.3 Batch: K={env.K} arms, T={T} budget, n={n} runs, {len(configs)} VarDE configs")
t0 = time.time()

for seed in tqdm(range(n)):
    agent_cra = CRA(env, T=T)
    random.seed(seed)
    env.seed(seed)
    agent_cra.run()
    rec["CR-A"].append(agent_cra.rec_history)

    for name, kwargs in configs:
        agent = VarDE_lse(env, T=T, **kwargs)
        random.seed(seed)
        env.seed(seed)
        agent.run()
        rec[name].append(agent.rec_history)

elapsed = time.time() - t0
print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

names = ["CR-A"] + [name for name, _ in configs]
for name in names:
    runs = rec[name]
    min_len = min(len(r) for r in runs)
    final_errors = [1 if (list(r) if hasattr(r, "tolist") else r)[min_len-1] != true_best else 0 for r in runs]
    error_pct = np.mean(final_errors) * 100
    print(f"{name}: Error Probability = {error_pct:.4f}%")
