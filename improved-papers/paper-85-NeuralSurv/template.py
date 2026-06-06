import jax
import jax.numpy as jnp
import jax.random as jr

from sksurv.nonparametric import kaplan_meier_estimator

from model.model import NMLP, MLP
from neuralsurv import NeuralSurv
from postprocessing.plot import (
    plot_posterior_phi,
    plot_hazard_function,
    plot_survival_function_vs_km_curve,
)


# Create key
key = jr.PRNGKey(12)

# Prior specifications
alpha_prior = 1.0
beta_prior = 1.0
rho = jnp.float32(1.0)

# Posterior sampling specifications
num_samples = 1000

# Algorithm specifications
max_iter_em = 20
max_iter_cavi = 20
num_points_integral_em = 1000
num_points_integral_cavi = 1000

# Training specifications
batch_size = 1000

# Model specifications
n_hidden = 2
n_layers = 2
activation = jax.nn.relu

# Directories
output_dir = "/Users/Monod/Downloads/temp"
plot_dir = "/Users/Monod/Downloads/temp"

# Get observed data from the train and test set (they must be jnp arrays)
n_train = 10
n_test = 5
p = 3
key, k_time_train, k_event_train, k_x_train, k_time_test, k_event_test, k_x_test = (
    jax.random.split(key, 7)
)
time_train = jax.random.normal(k_time_train, (n_train,), dtype=jnp.float32) * 100 + 150
event_train = jax.random.bernoulli(k_event_train, 0.5, (n_train,)).astype(jnp.int32)
x_train = jax.random.normal(k_x_train, (n_train, p), dtype=jnp.float32)

time_test = jax.random.normal(k_time_test, (n_test,), dtype=jnp.float32) * 100 + 150
event_test = jax.random.bernoulli(k_event_test, 0.5, (n_test,)).astype(jnp.int32)
x_test = jax.random.normal(k_x_test, (n_test, p), dtype=jnp.float32)

# Rescale time
min_time_train = time_train.min()
time_train = time_train - min_time_train
time_test = time_test - min_time_train

# Get model
model = NMLP(mlp_main=MLP(n_hidden=n_hidden, n_layers=n_layers, activation=activation))

# Instantiate parameters
key, step_rng = jr.split(key)
model_params_init = model.init(step_rng, jnp.array([0]), jnp.zeros(p))

# Instantiate neuralsurv
neuralsurv = NeuralSurv.load_or_create(
    model,
    model_params_init,
    alpha_prior,
    beta_prior,
    rho,
    num_points_integral_em,
    num_points_integral_cavi,
    batch_size,
    max_iter_em,
    max_iter_cavi,
    output_dir,
)

# Fit neuralsurv
neuralsurv.fit(
    time_train,
    event_train,
    x_train,
)

# Get prior and posterior samples
key, step_rng = jr.split(key)
neuralsurv.get_posterior_samples(step_rng, num_samples)

key, step_rng = jr.split(key)
neuralsurv.get_prior_samples(step_rng, num_samples)

# Plot phi posterior distribution
plot_posterior_phi(
    neuralsurv.posterior_params["alpha"],
    neuralsurv.posterior_params["beta"],
    plot_dir,
)

# Compute c-index, auc, brier score
neuralsurv.compute_evaluation_metrics(
    time_train, event_train, time_test, event_test, x_test, plot_dir
)

# New times
time_max = max(time_train.max(), time_test.max())
delta_time = time_max / 20
num = int(time_max // delta_time) + 1
times = jnp.linspace(1e-6, time_max, num=num)

# Plot hazard function
plot_hazard_function(
    neuralsurv.predict_hazard_function(times, x_train, aggregate=(0, 2)),
    plot_dir,
    "train",
)
plot_hazard_function(
    neuralsurv.predict_hazard_function(times, x_test, aggregate=(0, 2)),
    plot_dir,
    "test",
)

# Plot survival function
km_times_train, km_survival_train = kaplan_meier_estimator(
    event_train.astype(bool), time_train
)
km_times_test, km_survival_test = kaplan_meier_estimator(
    event_test.astype(bool), time_test
)
surv_train = neuralsurv.predict_survival_function(times, x_train, aggregate=(0, 2))
surv_test = neuralsurv.predict_survival_function(times, x_test, aggregate=(0, 2))
plot_survival_function_vs_km_curve(
    surv_train,
    km_times_train,
    km_survival_train,
    plot_dir,
    "train",
)
plot_survival_function_vs_km_curve(
    surv_test,
    km_times_test,
    km_survival_test,
    plot_dir,
    "test",
)
