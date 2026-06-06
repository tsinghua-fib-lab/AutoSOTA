# %%
"""Autotuning pipeline.

We recommend using this script as a Jupyter notebook.
To easily do so, run:
pip install jupytext
jupytext --set-formats ipynb,py:percent --sync src/hcnn/autotuning.py
"""

# %%
import os

import jax
import jax.numpy as jnp
from tqdm import tqdm

from benchmarks.QP.load_QP import SimpleQPDataset, create_dataloaders, dc3_dataloader
from pinet import (
    AffineInequalityConstraint,
    EqualityConstraint,
    EqualityConstraintsSpecification,
    Project,
    ProjectionInstance,
)

jax.config.update("jax_enable_x64", True)

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# NOTE: Change this to the absolute path of your datasets directory.
dataset_dir = "absolute/path/to/datasets"


def load_data(
    use_DC3_dataset,
    use_convex,
    problem_seed,
    problem_var,
    problem_nineq,
    problem_neq,
    problem_examples,
):
    """Load problem data."""
    if not use_DC3_dataset:
        # Choose problem parameters
        if use_convex:
            filename = (
                f"SimpleQP_seed{problem_seed}_var{problem_var}_ineq{problem_nineq}"
                f"_eq{problem_neq}_examples{problem_examples}.npz"
            )
        else:
            raise NotImplementedError()
        dataset_path = os.path.join(dataset_dir, filename)

        QPDataset = SimpleQPDataset(dataset_path)
        train_loader, valid_loader, test_loader = create_dataloaders(
            dataset_path, batch_size=2048, val_split=0.1, test_split=0.1
        )
        Q, p, A, G, h = QPDataset.const
        p = p[0, :, :]
        X = QPDataset.X
    else:
        # Choose the filename here
        if use_convex:
            filename = (
                f"dc3_random_simple_dataset_var{problem_var}_ineq{problem_nineq}"
                f"_eq{problem_neq}_ex{problem_examples}"
            )
        else:
            filename = (
                f"dc3_random_nonconvex_dataset_var{problem_var}_ineq{problem_nineq}"
                f"_eq{problem_neq}_ex{problem_examples}"
            )
        filename_train = filename + "train.npz"
        dataset_path_train = os.path.join(dataset_dir, filename_train)
        filename_valid = filename + "valid.npz"
        dataset_path_valid = os.path.join(dataset_dir, filename_valid)
        filename_test = filename + "test.npz"
        dataset_path_test = os.path.join(dataset_dir, filename_test)
        train_loader = dc3_dataloader(dataset_path_train, use_convex, batch_size=2048)
        valid_loader = dc3_dataloader(
            dataset_path_valid, use_convex, batch_size=1024, shuffle=False
        )
        test_loader = dc3_dataloader(
            dataset_path_test, use_convex, batch_size=1024, shuffle=False
        )
        Q, p, A, G, h = train_loader.dataset.const
        p = p[0, :, :]
        X = train_loader.dataset.X

    return (filename, Q, p, A, G, h, X, train_loader, valid_loader, test_loader)


# %%
# Load a batch of data for autotuning
filename, Q, p, A, G, h, X, train_loader, valid_loader, test_loader = load_data(
    use_DC3_dataset=True,
    use_convex=True,
    problem_seed=42,
    problem_var=1000,
    problem_nineq=500,
    problem_neq=500,
    problem_examples=10000,
)
n_samples = 150
X_batch, _ = next(iter(valid_loader))
X_batch = X_batch[:n_samples]


# %%


def build_evaluate_params(x, b, n_iter, project, compute_cv):
    """Return function that evaluates hyperparameters."""

    def evaluate_params(init, sigma):
        y, init = project(init=init, x=x, b=b, sigma=sigma, n_iter=n_iter)
        cvs = compute_cv(y)
        values = jnp.linalg.norm(y.x - x.x)
        return init, jnp.max(cvs), jnp.mean(values)

    return evaluate_params


# %%
# Setup the projection layer
eq_constraint = EqualityConstraint(A=A, b=X_batch, method=None, var_b=True)
ineq_constraint = AffineInequalityConstraint(C=G, ub=h, lb=-jnp.inf * jnp.ones_like(h))
projection_layer = Project(
    ineq_constraint=ineq_constraint,
    eq_constraint=eq_constraint,
    unroll=False,
    equilibrate={
        "max_iter": 25,
        "tol": 1.0e-3,
        "ord": 2.0,
        "col_scaling": False,
        "update_mode": "Gauss",
        "safeguard": False,
    },
)

# %%
omega = 1.7


def project(init, x, b, sigma, n_iter):
    """Projection layer wrapper."""
    yraw = ProjectionInstance(x=x, eq=EqualityConstraintsSpecification(b=b))
    return projection_layer.call(
        s0=init,
        yraw=yraw,
        sigma=sigma,
        omega=omega,
        n_iter=n_iter,
    )


def compute_cv(y):
    """Compute constraint violation."""
    return projection_layer.cv(y).reshape(
        -1,
    )


x = 1 * jax.random.normal(
    jax.random.PRNGKey(0), (X_batch.shape[0], projection_layer.dim)
)  # batch of random points to project

# %%
# Target values for sigma tuning
target_cv_sigma = 5e-2
target_rs_sigma = 1e-1
# Target values for n_iter tuning
target_cv_n_iter = 1e-3
target_rs_n_iter = 1e-2
# Fixed n_iter for the first stage
fixed_n_max_iter = 100
fixed_n_iter_step = 100
fixed_n_iter_candidates = fixed_n_max_iter // fixed_n_iter_step
# n_iter candidates for the second stage
n_max_iter = 400
n_iter_step = 50
n_iter_candidates = n_max_iter // n_iter_step

tie_breaker = "cv"
if tie_breaker == "cv":
    id_tie_breaker = 0
elif tie_breaker == "rs":
    id_tie_breaker = 1

sigma_candidates = jnp.logspace(-3, jnp.log10(5.05), num=100)

init_shape = (X_batch.shape[0], projection_layer.dim_lifted, 1)

# Evaluate the first stage
fixed_eval_fn = jax.jit(
    build_evaluate_params(x, X_batch, fixed_n_iter_step, project, compute_cv)
)
# Evaluate the second stage
eval_fn = jax.jit(build_evaluate_params(x, X_batch, n_iter_step, project, compute_cv))


# %%
def generate_results(sigma_candidates, n_iter_candidates, eval_fn):
    """Generate results for the given sigma candidates and n_iter candidates."""
    # Initialize results array
    results = jnp.inf * jnp.ones((len(sigma_candidates), n_iter_candidates, 2))

    def body_fun(i, r):
        sigma = sigma_candidates[i]

        def body_fun_i(j, state_i):
            ri, init = state_i
            init, cv, val = eval_fn(init, sigma)
            return ri.at[j, :].set(jnp.stack([cv, val])), init

        init = jnp.zeros(init_shape)
        _r, _ = jax.lax.fori_loop(0, n_iter_candidates, body_fun_i, (r[i, ...], init))
        return r.at[i, ...].set(_r)

    # Wrap the range with tqdm to display a progress bar
    for i in tqdm(range(len(sigma_candidates)), desc="Processing candidates"):
        results = body_fun(i, results)

    return results


# %%
def get_best(results, sigma_candidates, n_iter_step, target_cv, target_rs):
    """Returns the best combination of hyperparameters."""
    # Use the best result as proxy for the optimal value
    opt = jnp.min(results[:, :, 1])
    # Compute the relative suboptimality
    rs = (results[:, :, 1] - opt) / (opt + 1e-20)
    # Compute which entries satisfy both target conditions
    mask = (results[:, :, 0] < target_cv) * (rs < target_rs)
    mask_valid_sigma = jnp.any(mask, axis=1)
    if jnp.sum(mask) == 0:
        raise ValueError("No valid sigma found for the given target conditions.")
    # For each row, find the first column index where the condition is met
    first_valid_idx = jnp.argmax(mask, axis=1)
    # Find the minimum number of iterations across all rows
    min_iter_idx = jnp.min(first_valid_idx[mask_valid_sigma])
    # Find the best sigma values
    best_sigma_mask = (first_valid_idx == min_iter_idx) * mask_valid_sigma

    if jnp.sum(best_sigma_mask) > 1:
        # Tie breaking
        mask = mask * best_sigma_mask[:, None]
        min_val = jnp.min(results[mask, id_tie_breaker])
        best_sigma_mask = best_sigma_mask & jnp.any(
            results[..., id_tie_breaker] == min_val, axis=1
        )
        if jnp.sum(best_sigma_mask) > 1:
            # Other tie breaking
            mask = mask * best_sigma_mask[:, None]
            min_val = jnp.min(results[mask, 1 - id_tie_breaker])
            best_sigma_mask = best_sigma_mask & jnp.any(
                results[..., id_tie_breaker] == min_val, axis=1
            )

    # Find the (first) index non-zero in best_sigma_mask
    best_sigma_idx = jnp.argmax(best_sigma_mask)
    best_sigma = sigma_candidates[best_sigma_idx]
    best_n_iter = n_iter_step * (min_iter_idx + 1)

    return best_sigma, best_n_iter, results[best_sigma_idx, min_iter_idx, :]


# %%
results_sigma = generate_results(
    sigma_candidates, fixed_n_iter_candidates, fixed_eval_fn
)

# %%
print("=========== Results for fixed n_iter ===========")
best_sigma, best_n_iter, best_result = get_best(
    results_sigma, sigma_candidates, fixed_n_iter_step, target_cv_sigma, target_rs_sigma
)
print(f"Best sigma: {best_sigma}")
best_sigma = jnp.array([best_sigma])

# %%
results_n_iter = generate_results(best_sigma, n_iter_candidates, eval_fn)

# %%
print("=========== Results for n_iter tuning ===========")
best_sigma, best_n_iter, best_result = get_best(
    results_n_iter, best_sigma, n_iter_step, target_cv_n_iter, target_rs_n_iter
)
print(f"Best sigma: {best_sigma}")
print(f"Best n_iter: {best_n_iter}")
