from env import Bandit
from baseline import *
from VarDE import *
from plotexp import plot_experiment_results
from tqdm import tqdm
import random
random.seed(42)

expnum = 7

# Experiment setup: Arithmetic progression
means = []
stds = []
for i in range(15):
	means.append(0.4 - i * 0.025)
	stds.append(0.4 - i * 0.025)
random.shuffle(stds)
true_best = 0
T = 500
env = Bandit(distribution='gaussian', means=means, stds=stds)

# Number of runs per algorithm
n = 20000

# Record of recommendations
rec = {
	'VarDE_0.01_0.0001': [],
	'VarDE_0.01_0.001': [],
	'VarDE_0.01_0.01': [],
	'VarDE_0.01_0.1': [],
	'VarDE_0.01_0.2': [],
	'VarDE_0.03_0.0001': [],
	'VarDE_0.03_0.001': [],
	'VarDE_0.03_0.01': [],
	'VarDE_0.03_0.1': [],
	'VarDE_0.03_0.2': [],
	'VarDE_0.05_0.0001': [],
	'VarDE_0.05_0.001': [],
	'VarDE_0.05_0.01': [],
	'VarDE_0.05_0.1': [],
	'VarDE_0.05_0.2': [],
	'VarDE_0.1_0.0001': [],
	'VarDE_0.1_0.001': [],
	'VarDE_0.1_0.01': [],
	'VarDE_0.1_0.1': [],
	'VarDE_0.1_0.2': [],
	'VarDE_0.2_0.0001': [],
	'VarDE_0.2_0.001': [],
	'VarDE_0.2_0.01': [],
	'VarDE_0.2_0.1': [],
	'VarDE_0.2_0.2': [],
	'VarDE_0.5_0.0001': [],
	'VarDE_0.5_0.001': [],
	'VarDE_0.5_0.01': [],
	'VarDE_0.5_0.1': [],
	'VarDE_0.5_0.2': [],
	'VarDE_1.0_0.0001': [],
	'VarDE_1.0_0.001': [],
	'VarDE_1.0_0.01': [],
	'VarDE_1.0_0.1': [],
	'VarDE_1.0_0.2': [],
	'VarDE_2.0_0.0001': [],
	'VarDE_2.0_0.001': [],
	'VarDE_2.0_0.01': [],
	'VarDE_2.0_0.1': [],
	'VarDE_2.0_0.2': [],
}

def run(name, agent, seed, env=env):
	random.seed(seed)
	env.seed(seed)
	agent.run()
	rec[name].append(agent.rec_history)

for seed in tqdm(range(n)):
	agent = VarDE_lse(env, T=T, tau=0.01, var_floor=0.0001)
	run('VarDE_0.01_0.0001', agent=agent, seed=seed)
	
	agent = VarDE_lse(env, T=T, tau=0.01, var_floor=0.001)   
	run('VarDE_0.01_0.001', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.01, var_floor=0.01)
	run('VarDE_0.01_0.01', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.01, var_floor=0.1)
	run('VarDE_0.01_0.1', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.01, var_floor=0.2)
	run('VarDE_0.01_0.2', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.03, var_floor=0.0001)
	run('VarDE_0.03_0.0001', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.03, var_floor=0.001)
	run('VarDE_0.03_0.001', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.03, var_floor=0.01)
	run('VarDE_0.03_0.01', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.03, var_floor=0.1)
	run('VarDE_0.03_0.1', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.03, var_floor=0.2)
	run('VarDE_0.03_0.2', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.05, var_floor=0.0001)
	run('VarDE_0.05_0.0001', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.05, var_floor=0.001)
	run('VarDE_0.05_0.001', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.05, var_floor=0.01)
	run('VarDE_0.05_0.01', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.05, var_floor=0.1)
	run('VarDE_0.05_0.1', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.05, var_floor=0.2)
	run('VarDE_0.05_0.2', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.1, var_floor=0.0001)
	run('VarDE_0.1_0.0001', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.1, var_floor=0.001)
	run('VarDE_0.1_0.001', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.1, var_floor=0.01)
	run('VarDE_0.1_0.01', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.1, var_floor=0.1)
	run('VarDE_0.1_0.1', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.1, var_floor=0.2)
	run('VarDE_0.1_0.2', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.2, var_floor=0.0001)
	run('VarDE_0.2_0.0001', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.2, var_floor=0.001)
	run('VarDE_0.2_0.001', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.2, var_floor=0.01)
	run('VarDE_0.2_0.01', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.2, var_floor=0.1)
	run('VarDE_0.2_0.1', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.2, var_floor=0.2)
	run('VarDE_0.2_0.2', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.5, var_floor=0.0001)
	run('VarDE_0.5_0.0001', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.5, var_floor=0.001)
	run('VarDE_0.5_0.001', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.5, var_floor=0.01)
	run('VarDE_0.5_0.01', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.5, var_floor=0.1)
	run('VarDE_0.5_0.1', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.5, var_floor=0.2)
	run('VarDE_0.5_0.2', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=1.0, var_floor=0.0001)
	run('VarDE_1.0_0.0001', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=1.0, var_floor=0.001)
	run('VarDE_1.0_0.001', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=1.0, var_floor=0.01)
	run('VarDE_1.0_0.01', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=1.0, var_floor=0.1)
	run('VarDE_1.0_0.1', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=1.0, var_floor=0.2)
	run('VarDE_1.0_0.2', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=2.0, var_floor=0.0001)
	run('VarDE_2.0_0.0001', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=2.0, var_floor=0.001)
	run('VarDE_2.0_0.001', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=2.0, var_floor=0.01)
	run('VarDE_2.0_0.01', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=2.0, var_floor=0.1)
	run('VarDE_2.0_0.1', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=2.0, var_floor=0.2)
	run('VarDE_2.0_0.2', agent=agent, seed=seed)

# Plotting results
plot_experiment_results(rec, true_best, expnum=expnum)