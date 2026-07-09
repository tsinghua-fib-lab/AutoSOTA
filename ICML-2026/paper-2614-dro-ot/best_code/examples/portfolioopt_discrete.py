import numpy as np
import time
import os
import matplotlib.pyplot as plt
from trainable_ot_dro.utils.wasserstein_distance import wasserstein_gradient_L, wasserstein_distance_L
from trainable_ot_dro.utils.sampling_from_distributions import discrete_sample
from trainable_ot_dro.bilevel_optimization import optimize_transportation_matrix
from trainable_ot_dro.reformulations import reform_po_dro

SEED = 0

# True distribution (discrete)
# support points are of shape n_supp x dim_randomness
# probabilities are of shape n_supp
supp_true = np.array([[0.0, 0.0, -0.1],
                      [2.0, 0.0, -0.1],
                      [0.5, -5.0, 0.0],
                      [0.0, 1.0, 0.1],
                      [5.0, 1.0, 0.1],
                      [0.0, 0.0, 0.5],
                      [-1.0, 0.0, 0.5],
                      [0.0, -1.0, 0.5],
                      [-1.0, -1.0, 0.5],
                      [1.0, 1.0, 0.5]])
prob_true = np.array([0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])
assert np.isclose(np.sum(prob_true), 1.0), "Probabilities must sum to 1."

# Number of samples to use from true distribution
n_samples = 50
# How many bootstrap distributions to generate
n_bootstrap = 10

# Dimension of the randomness
dim_randomness = supp_true.shape[1]

# Initial transportation cost matrix
L_init = np.eye(dim_randomness)

# Type of Wasserstein distance
wasserstein_type = 1

# Results folder preparation
results_folder_name = "results/res_disc_"
results_folder_name += str(n_samples) + "samp_"
results_folder_name += str(n_bootstrap) + "boots"
results_folder_name += str(wasserstein_type) + "wass"
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
distribution_distance["gradient"] = wasserstein_gradient_L
distribution_distance["type"] = wasserstein_type

# Parameters for the coverage constraint
constraint_params = {}
constraint_params["gamma"] = 0.1 # conf. level for true-prob-dist-constraint

# Random sampling from the true distribution
rng = np.random.default_rng(SEED)
samples = discrete_sample(supp_true, prob_true, n_samples, seed=SEED)

# Create the reference distribution from the samples
supp_ref = np.unique(samples, axis=0)
prob_ref = np.zeros(supp_ref.shape[0])
for i in range(supp_ref.shape[0]):
  prob_ref[i] = np.sum(np.all(samples == supp_ref[i], axis=1)) / n_samples
ref_dist = (supp_ref, prob_ref)

# Create bootstrap distributions
bootstrap_dists = []
bootstrap_initial_distances = []
for i in range(n_bootstrap):
  # Generate bootstrap samples
  indices = rng.choice(n_samples, n_samples, replace=True)
  bootstrap_samples = samples[indices, :]
  # Create the bootstrap distribution
  supp_boots = np.unique(bootstrap_samples, axis=0)
  prob_boots = np.zeros(supp_boots.shape[0])
  for j in range(supp_boots.shape[0]):
    prob_boots[j] = np.sum(np.all(bootstrap_samples == supp_boots[j], axis=1)) / n_samples
  bootstrap_dist = (supp_boots, prob_boots)
  bootstrap_dists.append(bootstrap_dist)
  # Fill initial distances for eps_0 computation
  boots_distance, _, _ = wasserstein_distance_L(ref_dist, bootstrap_dist, L_init, wasserstein_type=wasserstein_type)
  bootstrap_initial_distances.append(boots_distance)

# Bundle the distributions
distributions = {}
distributions["reference"] = ref_dist
distributions["bootstrap"] = bootstrap_dists

# choose as initial epsilon the (1-gamma)-quantile of the bootstrap distances
eps_0 = np.quantile(bootstrap_initial_distances, 1 - constraint_params["gamma"])
constraint_params["eps"] = eps_0

# Reformulate the DRO problem as a conic program
conic_program = reform_po_dro(samples, eps_0, L_init,
                              wasserstein_type=wasserstein_type)

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

result["supp_true"] = supp_true
result["prob_true"] = prob_true
result["wasserstein_type"] = wasserstein_type
result["L_init"] = L_init
result["samples"] = samples
result["supp_ref"] = supp_ref
result["prob_ref"] = prob_ref
result["eps"] = eps_0

# create the result filename with timestamp
timestamp = time.time()
datetime_stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(timestamp))
result_filename = "res_gauss_" + datetime_stamp + ".npy"
# save the result
np.save(os.path.join(os.getcwd(), results_folder_name, result_filename),
        result, allow_pickle=True)



# ============================ PLOT =======================================
opt_decisions = result["decisions"]
worst_case_objective_values = result["objective_values"]
overall_mean = np.zeros(dim_randomness)
# calculate mean of discrete model true dist. by multiplying support points with their probabilities
for i in range(supp_true.shape[0]):
  overall_mean += prob_true[i] * supp_true[i, :]
true_objective_values = []
for i in range(len(opt_decisions)):
  w_opt = opt_decisions[i]
  # negative because we minimize expected loss but w^T xi is the expected return
  true_obj = - np.dot(w_opt, overall_mean)
  true_objective_values.append(true_obj)

iters = range(1, len(worst_case_objective_values) + 1)
iters = [opt_params["store_every"] * i for i in iters]

fig, ax = plt.subplots(figsize=(7,4))
# Left y-axis: worst-case
l1, = ax.plot(iters, worst_case_objective_values, lw=2, label="Worst-case (upper bound)")
ax.set_xlabel("Iteration")
ax.set_ylabel("Worst-case")
ax.grid(True, alpha=0.3)

# Right y-axis: true (red)
ax2 = ax.twinx()
l2, = ax2.plot(iters, true_objective_values, lw=2, color="red", label="True")
ax2.set_ylabel("True", color="red")
ax2.tick_params(axis="y", colors="red")

# One legend for both
lines = [l1, l2]
ax.legend(lines, [ln.get_label() for ln in lines], loc="best")

ax.set_title("Portfolio Expected Loss Optimization")
fig.tight_layout()

# save figure using the filename of the results
fig_filename = result_filename.replace(".npy", ".png")
fig.savefig(os.path.join(os.getcwd(), results_folder_name, fig_filename))

plt.show()
