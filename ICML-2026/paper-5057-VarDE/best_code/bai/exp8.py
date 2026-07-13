from env import Bandit
from baseline import *
from VarDE import *
from plotexp import plot_experiment_results
from tqdm import tqdm
import random
random.seed(42)

expnum = 8

# Experiment setup: Arithmetic progression
means = []
stds = []
for i in range(15):
	means.append(0.4 - i * 0.025)
	stds.append(0.4 - i * 0.025)
random.shuffle(stds)
true_best = 0
T = 2000
env = Bandit(distribution='gaussian', means=means, stds=stds)

# Number of runs per algorithm
n = 20000

# Record of recommendations
rec = {
	'VarDE_0.05_0.01': [],
	'VarDE_var': [],
	'VarDE_wei': [],
}

def run(name, agent, seed, env=env):
	random.seed(seed)
	env.seed(seed)
	agent.run()
	rec[name].append(agent.rec_history)

for seed in tqdm(range(n)):
	agent = VarDE_lse(env, T=T, tau=0.05, var_floor=0.01)
	run('VarDE_0.05_0.01', agent=agent, seed=seed)
	
	agent = VarDE_lse(env, T=T, tau=0.05, var_floor=0.01, use_influence_weights=False)   
	run('VarDE_var', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.05, var_floor=0.01, use_empirical_variance=False)
	run('VarDE_wei', agent=agent, seed=seed)

# Plotting results
plot_experiment_results(rec, true_best, expnum=expnum)