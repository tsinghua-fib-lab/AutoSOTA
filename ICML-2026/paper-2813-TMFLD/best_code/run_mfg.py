"""
Mean-Field Game (first-order, deterministic) particle solver (no random features)

We solve (approximately) the MFG via a forward–backward characteristic iteration:

HJB:    -∂t φ + 1/2 ||∇φ||^2 = f(t,x),   f(t,x) = (K * ρ_t)(x)
FP:     ∂t ρ - ∇·(ρ ∇φ) = 0

Using characteristics / Pontryagin system:
    ẋ = p
    ṗ = -∇_x f(t, x)
Terminal condition: p(T,x) = ∇ψ(x), with ψ(x)=10||x||^2 => ∇ψ(x)=20x
Initial condition:  x(0) ~ ρ0 (mixture of 8 Gaussians on an octagon)

We represent ρ_t by particles, and iterate:
  1) given a trajectory X[t_n, i], compute ∇f(t_n, X[t_n, i]) from all particles at that time
  2) backward integrate p from T to 0 using ṗ = -∇f
  3) forward integrate x from 0 to T using ẋ = p
  4) relax trajectories to stabilize

This is a practical computational experiment setup; it is not a proof-level PDE solver.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils.kernel import gaussian_kernel
from utils.evaluate import eval_mfg
import os
import argparse
import time
import pickle
from tqdm import tqdm
from functools import partial
from goodpoints import compress
from goodpoints.jax.compress import kt_compresspp
from goodpoints.jax.sliceable_points import SliceablePoints

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import sys

def get_config():
    parser = argparse.ArgumentParser(description='thinned_mfld')

    # Args settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--kernel', type=str, default='sobolev')
    parser.add_argument('--relax', type=float, default=0.3)
    parser.add_argument('--g', type=int, default=0)
    parser.add_argument('--bandwidth', type=float, default=1.0)
    parser.add_argument('--step_num', type=int, default=100)
    parser.add_argument('--particle_num', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='./results/')
    parser.add_argument('--thinning', type=str, default='kt')
    parser.add_argument('--d', type=int, default=2)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--kt_function', type=str, default='compresspp_kt')
    parser.add_argument("--skip_swap", action="store_true")
    args = parser.parse_args()  
    return args

def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f"mean_field_games/thinning_{args.thinning}/"
    args.save_path += f"kernel_{args.kernel}__relax_{args.relax}__bandwidth_{args.bandwidth}__step_num_{args.step_num}"
    args.save_path += f"__g_{args.g}__particle_num_{args.particle_num}__d_{args.d}"
    args.save_path += f"__kt_function_{args.kt_function}__skip_swap_{args.skip_swap}"
    args.save_path += f"__seed_{args.seed}"
    os.makedirs(args.save_path, exist_ok=True)
    with open(f'{args.save_path}/configs', 'wb') as handle:
        pickle.dump(vars(args), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return args

# -------------------------
# Problem / experiment setup
# -------------------------
def sample_rho0(key, N, d=2, std=0.1):
    """
    Sample N particles in R^d from a mixture of 8 Gaussians centered on a unit octagon
    embedded in first two coordinates.
    """
    # Octagon vertices in R^2
    js = jnp.arange(1, 9)
    centers2 = jnp.stack([jnp.cos(2 * jnp.pi * js / 8.0), jnp.sin(2 * jnp.pi * js / 8.0)], axis=1)  # (8,2)

    # Mix components uniformly
    key_comp, key_noise = jax.random.split(key, 2)
    comp = jax.random.randint(key_comp, shape=(N,), minval=0, maxval=8)  # in {0..7}
    means2 = centers2[comp]  # (N,2)

    noise = std * jax.random.normal(key_noise, shape=(N, d))
    x0 = jnp.zeros((N, d))
    x0 = x0.at[:, :2].set(means2)
    x0 = x0 + noise
    return x0


@jax.jit
def terminal_grad_psi(x):
    # psi(x)=10||x||^2 => ∇psi = 20 x
    return 20.0 * x


# -------------------------
# Kernel and mean-field force
# -------------------------
# @partial(jit, static_argnums=(2,3))
def grad_f_from_particles(X_t, scale, kernel, thin_fn, rng_key):
    """
    Inputs:
      X_t: (N,d)
    Output:
      grad_f: (N,d)
    """
    N, d = X_t.shape
    grad_kernel = jax.grad(kernel, argnums=0)
    X_t_prime = thin_fn(X_t, rng_key)
    grad = vmap(lambda x: vmap(lambda y: grad_kernel(x, y))(X_t_prime))(X_t)
    grad = scale * grad.mean(1)
    return grad


# -------------------------
# Forward–backward iteration
# -------------------------
def initial_guess_trajectory(x0, M, T=1.0):
    """
    A simple initial guess: linear interpolation from x0 to 0
    """
    # ts = jnp.linspace(0.0, T, M + 1)
    # X = (1.0 - ts[:, None, None]) * x0[None, :, :]
    X = jnp.broadcast_to(x0, (M + 1, *x0.shape))
    return X


def one_fbs_iteration(X_old, x0, dt, kernel, thin_fn, rng_key):
    """
    One forward-backward sweep producing a new trajectory X_new.
    """
    Np1, N, d = X_old.shape
    M = Np1 - 1

    # Compute grad f at each time and particle location
    scale = 10.0
    grad_f = jnp.stack([grad_f_from_particles(X_old[n], scale, kernel, thin_fn, rng_key) for n in range(Np1)], axis=0)
    # grad_f shape: (M+1,N,d)

    # Backward integrate p: ṗ = -∇f, terminal p(T)=∇psi(X_T)
    P = jnp.zeros((Np1, N, d))
    P = P.at[M].set(terminal_grad_psi(X_old[M]))
    # backward Euler:
    # p_{n} = p_{n+1} + dt * ∇f_n   (since p_{n+1} = p_n - dt ∇f_n)
    for n in range(M - 1, -1, -1):
        P = P.at[n].set(P[n + 1] + dt * grad_f[n])

    # Forward integrate x: ẋ = p, x(0)=x0
    X_new = jnp.zeros_like(X_old)
    X_new = X_new.at[0].set(x0)
    for n in range(M):
        X_new = X_new.at[n + 1].set(X_new[n] - dt * P[n])

    return X_new, P


def main(args):
    rng_key = jax.random.PRNGKey(args.seed)
    rng_key, _ = jax.random.split(rng_key)
    x0 = sample_rho0(rng_key, N=args.particle_num, d=args.d, std=0.1)

    dt = args.dt
    T = 1.0
    M = int(T / dt)
    ts = jnp.linspace(0.0, T, M + 1)

    # Initialize trajectory
    X = initial_guess_trajectory(x0, M=M, T=T)
    kernel = jax.jit(gaussian_kernel(args.bandwidth))

    if args.thinning == 'kt':
        if args.kt_function == 'compresspp_kt':
            def thin_fn(X, rng_key):
                rng_key, _ = jax.random.split(rng_key)
                seed = jax.random.randint(rng_key, (), 0, 2**31 - 1).item()
                x_cpu = np.array(np.asarray(X))
                kernel_type = "gaussian"
                k_params = np.array([args.bandwidth])
                coresets = compress.compresspp_kt(x_cpu, kernel_type=kernel_type.encode("utf-8"), 
                                                k_params=k_params, seed=seed, g=args.g)
                return jax.device_put(x_cpu[coresets, :])
            
        elif args.kt_function == 'compress_kt':
            print(f"Using compress_kt with skip_swap={args.skip_swap}")

            def thin_fn(X, rng_key):
                rng_key, _ = jax.random.split(rng_key)
                seed = jax.random.randint(rng_key, (), 0, 2**31 - 1).item()
                x_cpu = np.array(np.asarray(X))
                kernel_type = "gaussian"
                k_params = np.array([args.bandwidth])
                coresets = compress.compress_kt(x_cpu, kernel_type=kernel_type.encode("utf-8"), 
                                                k_params=k_params, seed=seed, g=args.g, skip_swap=args.skip_swap)
                return jax.device_put(x_cpu[coresets, :])
        else:
            raise ValueError(f'Unknown kt_function: {args.kt_function}')
    elif args.thinning == 'false':
        def thin_fn(X, rng_key):
            return X
    elif args.thinning == 'random':
        def thin_fn(X, rng_key):
            N = X.shape[0]
            key_thin, _ = jax.random.split(rng_key)
            indices = jax.random.choice(key_thin, N, (int(jnp.sqrt(N)),), replace=False)
            return X[indices]
    else:
        raise ValueError(f'Unknown thinning method: {args.thinning}')
    # Keep some snapshots for animation (store on CPU as numpy)
    loading_freq = 2
    x_history = [jnp.array(X)]
    p_history = [jnp.array(jnp.zeros_like(X))]
    time_history = [0.0]
    for k in tqdm(range(args.step_num)):
        time_now = time.time()
        rng_key, _ = jax.random.split(rng_key)
        X_new, P_new = one_fbs_iteration(X, x0, dt, kernel, thin_fn, rng_key)
        X = (1.0 - args.relax) * X + args.relax * X_new
        if k % loading_freq == 0:
            x_history.append(np.array(X))
            p_history.append(np.array(P_new))
            time_history.append(time.time() - time_now)

    kernel = jit(gaussian_kernel(args.bandwidth).make_distance_matrix) # for evaluation only
    eval_mfg(args, ts, X, kernel, x_history, p_history, time_history)
    return 0


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)  # optional for stability
    args = get_config()
    args = create_dir(args)
    new_save_path = args.save_path + '__complete'
    if os.path.exists(new_save_path):
        print(f"Job already completed. Folder exists: {new_save_path}")
        sys.exit(0)
    
    print('Program started!')
    print(vars(args))
    main(args)
    print('Program finished!')
    new_save_path = args.save_path + '__complete'
    import shutil
    if os.path.exists(new_save_path):
        shutil.rmtree(new_save_path)  # Deletes existing folder
    os.rename(args.save_path, new_save_path)
    print(f'Job completed. Renamed folder to: {new_save_path}')

