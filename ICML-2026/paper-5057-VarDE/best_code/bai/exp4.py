from env import Bandit
from baseline import *
from VarDE import *
from plotexp import plot_experiment_results
from tqdm import tqdm
import random
random.seed(42)

expnum = 4

# Experiment setup: 10 arms divided in three groups
means = [0.3] + [0.22]*3 + [0.2]*3 + [0.15]*3
stds = [0.3] + [0.22]*3 + [0.2]*3 + [0.15]*3
# randomize stds
random.shuffle(stds)
true_best = 0
T = 150
env = Bandit(distribution='gaussian', means=means, stds=stds)

# Number of runs per algorithm
n = 20000

# Record of recommendations
rec = {
    'Uniform': [],
	'UCBE-2': [],
	'UCBE-4': [],
	'UCBE-8': [],
	'UGapE-2': [],
	'UGapE-4': [],
	'UGapE-8': [],
	'SH': [],
	'SR': [],
	'CR-A': [],
	'CR-C': [],
	'VarDE_lse-0.05': [],
	'VarDE_lse-0.1': [],
	'VarDE_lse-0.15': [],
}

def run(name, agent, seed, env=env):
	random.seed(seed)
	env.seed(seed)
	agent.run()
	rec[name].append(agent.rec_history)

for seed in tqdm(range(n)):
	agent = UniformSampler(env, T=T)
	run('Uniform', agent=agent, seed=seed)

	agent = UCBE(env, T=T, a=2.0)
	run('UCBE-2', agent=agent, seed=seed)

	agent = UCBE(env, T=T, a=4.0)
	run('UCBE-4', agent=agent, seed=seed)

	agent = UCBE(env, T=T, a=8.0)
	run('UCBE-8', agent=agent, seed=seed)

	agent = UGapE(env, T=T, a=2.0)
	run('UGapE-2', agent=agent, seed=seed)

	agent = UGapE(env, T=T, a=4.0)
	run('UGapE-4', agent=agent, seed=seed)

	agent = UGapE(env, T=T, a=8.0)
	run('UGapE-8', agent=agent, seed=seed)

	agent = SH(env, T=T)
	run('SH', agent=agent, seed=seed)

	agent = SuccessiveRejects(env, T=T)
	run('SR', agent=agent, seed=seed)

	agent = CRA(env, T=T)
	run('CR-A', agent=agent, seed=seed)

	agent = CRC(env, T=T)
	run('CR-C', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.05)
	run('VarDE_lse-0.05', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.1)
	run('VarDE_lse-0.1', agent=agent, seed=seed)

	agent = VarDE_lse(env, T=T, tau=0.15)
	run('VarDE_lse-0.15', agent=agent, seed=seed)

# Plotting results
plot_experiment_results(rec, true_best, expnum=expnum)