"""
Flexible training script for CINOC FKPP 1D optimization.
Accepts CLI arguments for loss weights, LR schedule, noise, architecture, etc.
"""
import jax
import jax.numpy as jnp
import sys
import os
from pathlib import Path
import optax
import time
from functools import partial
import argparse
import flax.serialization

script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics
from models.policy import DecentralizedControlNet
from data_utils import generate_grf

def parse_args():
    p = argparse.ArgumentParser()
    # Loss weights
    p.add_argument("--lambda-track", type=float, default=5.0)
    p.add_argument("--lambda-effort", type=float, default=0.001)
    p.add_argument("--lambda-bound", type=float, default=100.0)
    p.add_argument("--lambda-coll", type=float, default=1.0)
    p.add_argument("--lambda-accel", type=float, default=0.1)
    # LR schedule
    p.add_argument("--lr-schedule", choices=["exp", "cosine"], default="exp")
    p.add_argument("--lr-peak", type=float, default=1e-3)
    p.add_argument("--lr-end", type=float, default=1e-6)
    p.add_argument("--warmup-epochs", type=int, default=0)
    # Noise
    p.add_argument("--noise-u", type=float, default=0.0)
    p.add_argument("--noise-z", type=float, default=0.0)
    p.add_argument("--noise-anneal", action="store_true", default=False)
    p.add_argument("--noise-u-init", type=float, default=0.1)
    p.add_argument("--noise-z-init", type=float, default=0.05)
    # Architecture
    p.add_argument("--branch-features", type=str, default="64,64")
    p.add_argument("--trunk-features", type=str, default="32,32")
    # Time decay
    p.add_argument("--time-decay-alpha", type=float, default=0.0)
    # Data
    p.add_argument("--n-samples", type=int, default=5000)
    p.add_argument("--init-scale-min", type=float, default=0.2)
    p.add_argument("--init-scale-max", type=float, default=0.2)
    p.add_argument("--target-scale-min", type=float, default=0.4)
    p.add_argument("--target-scale-max", type=float, default=0.4)
    # Training
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default="decentralized_params.msgpack")
    p.add_argument("--save-dir", type=str, default=".")
    return p.parse_args()

args = parse_args()
n_pde, n_agents = 100, 20
T_steps = 300

# Parse architecture
branch_feat = tuple(int(x) for x in args.branch_features.split(","))
trunk_feat = tuple(int(x) for x in args.trunk_features.split(","))

model = DecentralizedControlNet(features=branch_feat)
key = jax.random.PRNGKey(args.seed)

params = model.init(key, jnp.zeros((n_pde,)), jnp.zeros((n_pde,)), jnp.zeros((n_agents,)))

# LR schedule
steps_per_epoch = args.n_samples // args.batch_size
total_steps = args.epochs * steps_per_epoch
if args.lr_schedule == "cosine":
    warmup_steps = args.warmup_epochs * steps_per_epoch if args.warmup_epochs > 0 else 0
    decay_steps = total_steps - warmup_steps
    if warmup_steps > 0:
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=args.lr_peak,
            warmup_steps=warmup_steps, decay_steps=decay_steps,
            end_value=args.lr_end
        )
    else:
        lr_schedule = optax.cosine_decay_schedule(args.lr_peak, decay_steps, alpha=args.lr_end/args.lr_peak)
else:
    lr_schedule = optax.exponential_decay(args.lr_peak, 2000, 0.5)

optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))
opt_state = optimizer.init(params)

def loss_fn(params, z_init, xi_init, z_target, dynamics, noise_u_val, noise_z_val):
    z_traj, xi_traj, u_traj, v_traj = dynamics.unroll_controlled(
        z_init, xi_init, z_target, params, T_steps,
            noise_u=noise_u_val, noise_z=noise_z_val
    )
    l_track = jnp.mean((z_traj - z_target[None, :]) ** 2)
    l_effort = jnp.mean(u_traj ** 2) + 0.1 * jnp.mean(v_traj ** 2)
    margin = 0.02
    l_bound = jnp.mean(jnp.maximum(0, margin - xi_traj)**2 + jnp.maximum(0, xi_traj - (1.0 - margin))**2)
    dists = jnp.abs(xi_traj[:, :, None] - xi_traj[:, None, :])
    mask = jnp.eye(n_agents)[None, :, :]
    l_coll = jnp.mean(jnp.maximum(0, 0.05 - (dists + mask * 1.0)) ** 2)
    l_accel = jnp.mean(jnp.diff(v_traj, axis=0)**2)

    # Apply time decay to tracking loss if alpha > 0
    if args.time_decay_alpha > 0:
        time_weights = jnp.exp(-args.time_decay_alpha * (T_steps - 1 - jnp.arange(T_steps)))
        time_weights = time_weights / jnp.sum(time_weights)
        l_track = jnp.sum(time_weights[:, None] * (z_traj - z_target[None, :]) ** 2)

    total_loss = (args.lambda_track * l_track + args.lambda_effort * l_effort +
                  args.lambda_bound * l_bound + args.lambda_coll * l_coll +
                  args.lambda_accel * l_accel)
    return total_loss, (l_track, l_effort, l_coll, l_bound)

def noise_u_schedule(epoch):
    if args.noise_anneal:
        return args.noise_u_init * (1.0 - epoch / args.epochs)
    return args.noise_u

def noise_z_schedule(epoch):
    if args.noise_anneal:
        return args.noise_z_init * (1.0 - epoch / args.epochs)
    return args.noise_z

@partial(jax.jit, static_argnames="dynamics")
def train_step(params, opt_state, z_init_batch, xi_init_batch, z_target_batch, dynamics, noise_u_val, noise_z_val):
    batched_loss_fn = jax.vmap(lambda p, zi, xi, zt: loss_fn(p, zi, xi, zt, dynamics, noise_u_val, noise_z_val), in_axes=(None, 0, 0, 0))
    def mean_loss_fn(p):
        losses, auxs = batched_loss_fn(p, z_init_batch, xi_init_batch, z_target_batch)
        return jnp.mean(losses), jax.tree_util.tree_map(jnp.mean, auxs)
    (loss, aux), grads = jax.value_and_grad(mean_loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux

dynamics = PDEDynamics(policy_apply_fn=model.apply)

print(f"Generating {args.n_samples} samples...")
all_keys = jax.random.split(key, args.n_samples + 2)

# Generate with random length scales if range specified
if args.init_scale_min == args.init_scale_max:
    _, z_init_all = jax.vmap(partial(generate_grf, n_points=n_pde, length_scale=args.init_scale_min))(all_keys[:args.n_samples])
else:
    init_scales = args.init_scale_min + (args.init_scale_max - args.init_scale_min) * jax.random.uniform(all_keys[args.n_samples], (args.n_samples,))
    _, z_init_all = jax.vmap(lambda k, s: generate_grf(k, n_points=n_pde, length_scale=s))(all_keys[:args.n_samples], init_scales)

if args.target_scale_min == args.target_scale_max:
    _, z_target_all = jax.vmap(partial(generate_grf, n_points=n_pde, length_scale=args.target_scale_min))(all_keys[args.n_samples:2*args.n_samples])
else:
    target_scales = args.target_scale_min + (args.target_scale_max - args.target_scale_min) * jax.random.uniform(all_keys[2*args.n_samples], (args.n_samples,))
    _, z_target_all = jax.vmap(lambda k, s: generate_grf(k, n_points=n_pde, length_scale=s))(all_keys[args.n_samples:2*args.n_samples], target_scales)

xi_init_batch = jnp.tile(jnp.linspace(0.2, 0.8, n_agents), (args.batch_size, 1))

print(f"Training {args.epochs} epochs...")
start_time = time.time()
for epoch in range(args.epochs):
    key, subkey = jax.random.split(key)
    idx = jax.random.randint(subkey, (args.batch_size,), 0, args.n_samples)
    z_init_b, z_target_b = z_init_all[idx], z_target_all[idx]
    params, opt_state, loss, aux = train_step(params, opt_state, z_init_b, xi_init_batch, z_target_b, dynamics, noise_u_schedule(epoch), noise_z_schedule(epoch))
    if epoch % 50 == 49 or epoch == 0:
        print(f"Epoch {epoch+1:03d} | Loss: {loss:.6f} | Track: {aux[0]:.6f} | NoiseU: {noise_u_schedule(epoch):.4f}")

elapsed = time.time() - start_time
print(f"Training finished in {elapsed:.1f}s.")

# Save
output_path = os.path.join(args.save_dir, args.output)
with open(output_path, "wb") as f:
    f.write(flax.serialization.to_bytes(params))
print(f"Params saved to {output_path}")
