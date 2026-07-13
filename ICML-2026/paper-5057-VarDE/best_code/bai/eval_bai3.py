"""Reproduce BAI.3 metric from VarDE paper (ICML 2026).
Settings: 14 arms, Gaussian rewards, budget T=200, 20000 independent runs.
Target: Error Probability for VarDE_lse-0.1 and CR-A baseline.
Matches the original exp3.py structure exactly.
"""
import os, time, random
import numpy as np
from env import Bandit
from baseline import CRA
from VarDE import VarDE_lse
from tqdm import tqdm

random.seed(42)

# BAI.3: Geometric progression distribution (Appendix F.1)
means = [0.4]
stds = [0.4]
for i in range(10, 23):
    means.append(0.4 - 0.9**i)
    stds.append(0.4 - 0.9**i)
random.shuffle(stds)
true_best = 0
T = 200
n = 20000

env = Bandit(distribution="gaussian", means=means, stds=stds)

rec = {
    "CR-A": [],
    "VarDE_lse-0.1": [],
}

def run(name, agent, seed):
    random.seed(seed)
    env.seed(seed)
    agent.run()
    rec[name].append(agent.rec_history)

print(f"BAI.3: K={env.K} arms, T={T} budget, {n} runs")
t0 = time.time()

for seed in tqdm(range(n)):
    agent = CRA(env, T=T)
    run("CR-A", agent=agent, seed=seed)

    agent = VarDE_lse(env, T=T, tau=0.10, variance_exponent=0.70, ema_alpha=0.10)
    run("VarDE_lse-0.1", agent=agent, seed=seed)

elapsed = time.time() - t0
print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

for name in ["CR-A", "VarDE_lse-0.1"]:
    runs = rec[name]
    min_len = min(len(r) for r in runs)
    final_errors = []
    for arr in runs:
        final_errors.append(1 if (list(arr) if hasattr(arr, "tolist") else arr)[min_len-1] != true_best else 0)
    error_pct = np.mean(final_errors) * 100
    print(f"{name}: Error Probability = {error_pct:.4f}%")

# Output single-line summary for parsing
os.makedirs("results", exist_ok=True)
with open("results/eval_bai3_results.txt", "w") as f:
    f.write(f"BAI.3 Reproduction (n={n}, T={T}, K={env.K})\n")
    for name in ["CR-A", "VarDE_lse-0.1"]:
        runs = rec[name]
        min_len = min(len(r) for r in runs)
        final_errors = [1 if (list(r) if hasattr(r, "tolist") else r)[min_len-1] != true_best else 0 for r in runs]
        error_pct = np.mean(final_errors) * 100
        f.write(f"{name}: {error_pct:.4f}%\n")
    f.write("Paper reported: VarDE_lse-0.1 = 7.34%, CR-A = 9.83%\n")
print("Results saved.")
