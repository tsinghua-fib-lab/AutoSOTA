from env import Bandit
import VarDE
import baseline

# Environment setup
means = [0.5, 0.42, 0.4, 0.4, 0.35, 0.35]
stds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.9]
true_best = 0
T = 20000

# Initialize bandit environment
env = Bandit(distribution='gaussian', means=means, stds=stds)

# Initialize diagnostic agent
diagnostic_agent = VarDE.VarDE_lse(env, T=T, warm_start=10, tau=0.1)
#diagnostic_agent = baseline.UCBE(env, T=T, warm_start=10, a=4.0)

# Run diagnostic agent
best_arm = diagnostic_agent.run()

# Print results
print(f"Identified best arm: {best_arm}, True best arm: {true_best}")
diagnostic_agent.plot_diagnostics()