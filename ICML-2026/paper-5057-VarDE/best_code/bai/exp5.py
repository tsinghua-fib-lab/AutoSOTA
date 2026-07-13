from env import Bandit
from baseline import *
from VarDE import *
from plotexp import plot_experiment_results
from tqdm import tqdm
import random
random.seed(42)

expnum = 5

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
	'VarDE_lse': [],
	'VarDE_nesterov': [],
	'VarDE_entmax': [],
	'VarDE_softplus': [],
	'VarDE_powermean': [],
}

def run(name, agent, seed, env=env):
	random.seed(seed)
	env.seed(seed)
	agent.run()
	rec[name].append(agent.rec_history)

for seed in tqdm(range(n)):
	agent = VarDE_lse(env, T=T, tau=0.05, warm_start=5)
	run('VarDE_lse', agent=agent, seed=seed)

	agent = VarDE_nesterov(env, T=T, mu=0.5, warm_start=5)
	run('VarDE_nesterov', agent=agent, seed=seed)

	agent = VarDE_entmax(env, T=T, mu=0.1, alpha=2.0, warm_start=5)
	run('VarDE_entmax', agent=agent, seed=seed)

	agent = VarDE_pairwise_softplus(env, T=T, delta=0.1, warm_start=5)
	run('VarDE_softplus', agent=agent, seed=seed)

	agent = VarDE_power_mean(env, T=T, p=5.0, warm_start=5)
	run('VarDE_powermean', agent=agent, seed=seed)

# Plotting results
plot_experiment_results(rec, true_best, expnum=expnum)