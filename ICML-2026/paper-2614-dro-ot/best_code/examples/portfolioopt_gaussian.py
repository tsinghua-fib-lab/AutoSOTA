import numpy as np
import time
import os
import matplotlib.pyplot as plt
from trainable_ot_dro.utils.gelbrich_distance import gelbrich_gradient_L, gelbrich_distance_L
from trainable_ot_dro.reformulations import reform_po_dro_gaussian_ref
from trainable_ot_dro.bilevel_optimization import optimize_transportation_matrix
from trainable_ot_dro.utils.risk_measures import gaussian_portfolio_CVaR

SEED = 0

# True distribution (Gaussian)
mean_true = np.array([-0.5, -0.5, 0.0])
cov_true = np.array([[0.5, 0.1, 0.0],
                     [0.1, 0.9, 0.0],
                     [0.0, 0.0, 0.05]])

# Number of samples to use from true distribution
n_samples = 50
# How many bootstrap distributions to generate
n_bootstrap = 5

# Dimension of the randomness
dim_randomness = mean_true.shape[0]

# Initial transportation cost matrix
L_init = np.eye(dim_randomness)

# CVaR confidence level
beta_cvar = 0.05

# Results folder preparation
results_folder_name = "results/res_gauss_"
results_folder_name += str(n_samples) + "samp_"
results_folder_name += str(n_bootstrap) + "boots_"
# Create the results folder
os.makedirs(results_folder_name, exist_ok=True)

# Penalization parameters
penalization = {}
penalization["lambda"] = 10.0
penalization["eta"] = 100.0

# Tolerances for stopping criteria
stopping_tols = {}
stopping_tols["diff_decision"] = 0.0
stopping_tols["diff_objective"] = 0.0
stopping_tols["diff_objective_pen"] = 0.0
stopping_tols["diff_weightmat"] = 0.0
stopping_tols["weightmat_gradient"] = 0.0
stopping_tols["rel_impr_obj"] = 1e-6
stopping_tols["rel_diff_obj_pen"] = 0.0

# Optimization parameters
opt_params = {}
opt_params["n_iter_max"] = 1_000_000
opt_params["learning_rate"] = 1e-4
opt_params["store_every"] = 100
opt_params["stopping_tols"] = stopping_tols
opt_params["penalization"] = penalization

# Type of Optimal Transport distance
distribution_distance = {}
distribution_distance["gradient"] = gelbrich_gradient_L
distribution_distance["type"] = 2
assert distribution_distance["type"] == 2

# Parameters for the coverage constraint
constraint_params = {}
constraint_params["gamma"] = 0.1

# Random sampling from the true distribution
rng = np.random.default_rng(seed=SEED)
samples = rng.multivariate_normal(mean_true, cov_true, n_samples)

# Create the reference distribution from the samples
mean_samples = np.mean(samples, axis=0)
cov_samples = np.cov(samples.T) + np.eye(dim_randomness) * 1e-6
ref_dist = (mean_samples, cov_samples)

# Create the bootstrap distributions
bootstrap_dists = []
bootstrap_initial_distances = []
for i in range(n_bootstrap):
  # Generate bootstrap samples
  indices = rng.choice(n_samples, n_samples, replace=True)
  bootstrap_samples = samples[indices, :]
  # Compute mean and covariance of bootstrap samples
  mean_bootstrap = np.mean(bootstrap_samples, axis=0)
  cov_bootstrap = np.cov(bootstrap_samples.T) + np.eye(dim_randomness) * 1e-6
  bootstrap_dist = (mean_bootstrap, cov_bootstrap)
  #
  bootstrap_dists.append(bootstrap_dist)
  bootstrap_initial_distances.append(gelbrich_distance_L(ref_dist, bootstrap_dist, L_init))

# Bundle the distributions
distributions = {}
distributions["reference"] = ref_dist
distributions["bootstrap"] = bootstrap_dists

# choose as initial epsilon the (1-gamma)-quantile of the bootstrap distances
eps_0 = np.quantile(bootstrap_initial_distances, 1 - constraint_params["gamma"])
constraint_params["eps"] = eps_0

# Reformulate the portfolio optimization DRO problem with Gaussian reference distribution
conic_program = reform_po_dro_gaussian_ref(mean_samples, cov_samples,
                                           beta_cvar, eps_0, L_init)

# Run the optimization
start_opt = time.time()
result = optimize_transportation_matrix(L_init,
                                        conic_program,
                                        distributions,
                                        distribution_distance,
                                        constraint_params,
                                        opt_params)
end_opt = time.time()
print("Time elapsed for bilevel optimization: ", end_opt - start_opt, " seconds")

result["mean_true"] = mean_true
result["cov_true"] = cov_true
result["beta_cvar"] = beta_cvar
result["L_init"] = L_init
result["mean_samples"] = mean_samples
result["cov_samples"] = cov_samples
result["eps"] = eps_0

# create result filename with timestamp
timestamp = time.time()
datetime_stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(timestamp))
result_filename = "res_gauss_" + datetime_stamp + ".npy"
# save the result
np.save(os.path.join(os.getcwd(), results_folder_name, result_filename),
        result, allow_pickle=True)



# ================================ PLOT ===================================
opt_decisions = result["decisions"]
worst_case_objective_values = result["objective_values"]
true_objective_values = []
for k in range(len(opt_decisions)):
  w_opt = opt_decisions[k]
  true_obj = gaussian_portfolio_CVaR(mean_true, cov_true, w_opt, beta_cvar)
  true_objective_values.append(true_obj)

iters = range(1, len(worst_case_objective_values) + 1)
iters = [opt_params["store_every"] * i for i in iters]

fig, ax = plt.subplots(figsize=(7,4))
# Left y-axis: worst-case
l1, = ax.plot(iters, worst_case_objective_values, lw=2, label="Worst-case CVaR (upper bound)")
ax.set_xlabel("Iteration")
ax.set_ylabel("Worst-case CVaR")
ax.grid(True, alpha=0.3)

# Right y-axis: true (red)
ax2 = ax.twinx()
l2, = ax2.plot(iters, true_objective_values, lw=2, color="red", label="True CVaR")
ax2.set_ylabel("True CVaR", color="red")
ax2.tick_params(axis="y", colors="red")

# One legend for both
lines = [l1, l2]
ax.legend(lines, [ln.get_label() for ln in lines], loc="best")

ax.set_title("Portfolio CVaR Optimization")
fig.tight_layout()

# save figure using the filename of the results
fig_filename = result_filename.replace(".npy", ".png")
fig.savefig(os.path.join(os.getcwd(), results_folder_name, fig_filename))

plt.show()
