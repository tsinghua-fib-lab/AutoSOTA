import numpy as np
import time
import os
import matplotlib.pyplot as plt
from trainable_ot_dro.utils.wasserstein_distance import wasserstein_gradient_L, wasserstein_distance_L
from trainable_ot_dro.reformulations import reform_linreg_dro
from trainable_ot_dro.bilevel_optimization import optimize_transportation_matrix

SEED = 0

# True weight vector and noise variance for the linear function
w_true = np.array([2.0, -3.0])
x_min = -10.0
x_max = 10.0
var_eps = 500.0

# Dimension of the randomness (including the output variable)
dim_randomness = w_true.shape[0] + 1
# Number of features (not including the output variable)
k = dim_randomness - 1

# Number of samples to use from true distribution
n_samples = 10
# How many bootstrap distributions to generate
n_bootstrap = 5

# Initial transportation cost matrix
L_init = np.eye(dim_randomness)

# Type of Wasserstein distance
wasserstein_type = 1

# Results folder preparation
results_folder_name = "results/res_linreg_"
results_folder_name += str(n_samples) + "samp_"
results_folder_name += str(n_bootstrap) + "boots_"
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
constraint_params["gamma"] = 0.1

# Generate the random dataset
rng = np.random.default_rng(seed=SEED)
X = rng.uniform(x_min, x_max, (n_samples, k))
y = np.dot(X, w_true) + rng.normal(0, np.sqrt(var_eps), n_samples)
samples = np.hstack((X, y.reshape(-1, 1)))

# Reference distribution
supp_ref = np.unique(samples, axis=0)
prob_ref = np.zeros(supp_ref.shape[0])
for i in range(supp_ref.shape[0]):
  prob_ref[i] = np.sum(np.all(samples == supp_ref[i], axis=1)) / n_samples
ref_dist = (supp_ref, prob_ref)

# Bootstrap distributions
bootstrap_dists = []
bootstrap_initial_distances = []
for i in range(n_bootstrap):
  # Generate bootstrap samples
  indices = rng.choice(n_samples, n_samples, replace=True)
  bootstrap_samples = samples[indices, :]
  # Create bootstrap distribution
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
conic_program = reform_linreg_dro(X, y, eps_0, L_init,
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

result["w_true"] = w_true
result["var_eps"] = var_eps
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
worst_case_objective_values = np.array(result["objective_values"])
# due to reformulation, need to square obj. values if wasserstein_type == 2
if wasserstein_type == 2:
  worst_case_objective_values = worst_case_objective_values**2

oos_errors = []
n_samples_OOS = 1e6
x_samples = np.random.uniform(x_min, x_max, (int(n_samples_OOS), k))
e_samples = np.random.normal(0, np.sqrt(var_eps), int(n_samples_OOS))
for decision in opt_decisions:
  decision_diff = decision - w_true
  if wasserstein_type == 1:
    loss_samles = np.abs(np.dot(x_samples, decision_diff) + e_samples)
  else:
    loss_samles = (np.dot(x_samples, decision_diff) + e_samples)**2
  oos_error = np.mean(loss_samles)
  oos_errors.append(oos_error)

iters = range(1, len(worst_case_objective_values) + 1)
iters = [opt_params["store_every"] * i for i in iters]

fig, ax = plt.subplots(figsize=(7,4))
# Left y-axis: worst-case
l1, = ax.plot(iters, worst_case_objective_values, lw=2, label="Worst-case Error (upper bound)")
ax.set_xlabel("Iteration")
ax.set_ylabel("Worst-case Error")
ax.grid(True, alpha=0.3)

# Right y-axis: true (red)
ax2 = ax.twinx()
l2, = ax2.plot(iters, oos_errors, lw=2, color="red", label="True Error")
ax2.set_ylabel("True Error", color="red")
ax2.tick_params(axis="y", colors="red")

# One legend for both
lines = [l1, l2]
ax.legend(lines, [ln.get_label() for ln in lines], loc="best")

ax.set_title("Linear Regression DRO")
fig.tight_layout()

# save figure using the filename of the results
fig_filename = result_filename.replace(".npy", ".png")
fig.savefig(os.path.join(os.getcwd(), results_folder_name, fig_filename))

plt.show()
