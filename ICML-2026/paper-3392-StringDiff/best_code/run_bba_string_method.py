#!/usr/bin/env python3
"""Reproduce BBA MEP peak energy from ScoreMD checkpoint.

Paper: "Probing the Geometry of Diffusion Models with the String Method"
Metric: Peak Energy Along Pathway (kbT), lower_better
Target: Converged MEP ~10 kbT vs Initial String ~50 kbT (from Figure 9/12)

Approach:
1. Load ScoreMD BBA model checkpoint (transformer_large_score + potential)
2. Implement classical string method at fixed time t≈0 (ScoreMD convention)
3. Compute energy profile along initial string and converged MEP
4. Report peak energy values

The classical string method at fixed time evolves points along the
score gradient, with periodic reparametrization to maintain equal arc-length.
"""
import os, sys, json, argparse, types

# Bypass heavy ScoreMD imports
for m_name in ['bgmol', 'bgmol.datasets', 'bgflow', 'bgflow.nn', 'bgflow.nn.flow']:
    sys.modules[m_name] = types.ModuleType(m_name)
sys.modules['bgmol.datasets'].AImplicitUnconstrained = type('AIC', (), {})

sys.path.insert(0, '/scoremd/src')

import numpy as np
import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Reparametrization (JAX, mirrors diffusion_strings/reparametrization.py)
# ---------------------------------------------------------------------------
@jax.jit
def uniform_string_repametrize(string, n_new):
    segments = jnp.diff(string, axis=0)
    seg_lengths = jnp.linalg.norm(segments.reshape(segments.shape[0], -1), axis=1)
    total = jnp.sum(seg_lengths)
    n = string.shape[0]
    if n <= 1 or total < 1e-12:
        return jnp.tile(string[:1], (n_new,) + (1,) * (string.ndim - 1))
    cumulative = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg_lengths)])
    new_cum = jnp.linspace(0, total, n_new)
    indices = jnp.clip(jnp.searchsorted(cumulative, new_cum) - 1, 0, n - 2)
    seg_len = seg_lengths[indices]
    rel_prog = jnp.where(seg_len > 1e-12, (new_cum - cumulative[indices]) / seg_len, 0.0)
    extra_dims = (1,) * (segments.ndim - 1)
    return string[indices] + segments[indices] * rel_prog.reshape(-1, *extra_dims)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def build_and_load_model(data, data_std, ckpt_dir):
    """Build ScoreMD BBA MixtureOfModels and load EMA checkpoint."""
    from scoremd.models.graph_transformer import GraphTransformerModelInfo
    from scoremd.data.preprocess import Preprocessor
    from scoremd.models.mixture import MixtureOfModels
    import orbax.checkpoint as ocp
    from orbax.checkpoint import args as ocp_args

    n_atoms = data.shape[1] // 3
    nf_j = jnp.array(1.0 / data_std)

    # Minimal dataset for model building
    class MD:
        class TD:
            def __init__(self, d):
                self.data = jnp.array(d)
                self.features = None
        def __init__(self, d, s, n):
            self.train = self.TD(d)
            self.std = s
            self.sample_shape = (n, 3)
            self.max_z = []
            self.mass = jnp.ones((n, 1))
    ds = MD(data, data_std, n_atoms)

    # Three sub-models per paper config (Table 5)
    c = dict(hidden_nf=128, feature_embedding_dim=16, n_layers=3, dropout=0.0)
    m1 = GraphTransformerModelInfo(**c, potential=False).build(ds, t0=0.6, t1=1.0, rescale_time=True, clip_time=True, norm_factor=nf_j)
    m2 = GraphTransformerModelInfo(**c, potential=False).build(ds, t0=0.1, t1=0.6, rescale_time=True, clip_time=True, norm_factor=nf_j)
    m3 = GraphTransformerModelInfo(**c, potential=True).build(ds, t0=0.0, t1=0.1, rescale_time=True, clip_time=True, norm_factor=nf_j)

    def wf(x, t):
        t = t.reshape(-1); bs = t.shape[0]
        return jnp.stack([(t>0.6).astype(jnp.float32), ((t<=0.6)&(t>0.1)).astype(jnp.float32), (t<=0.1).astype(jnp.float32)], axis=0)

    model = MixtureOfModels([m1, m2, m3], wf, Preprocessor())

    # Initialize to get param structure
    key = jax.random.PRNGKey(0)
    dummy_x = jnp.array(data[:1]) * nf_j
    dummy_t = jnp.ones((1,1)) * 0.5
    init_params = model.init(key, dummy_x, None, dummy_t, training=False)

    # Load EMA checkpoint
    ckpt_mgr = ocp.CheckpointManager(
        os.path.abspath(ckpt_dir),
        options=ocp.CheckpointManagerOptions(max_to_keep=10, create=False),
    )
    step = ckpt_mgr.latest_step()
    print(f"Checkpoint step: {step}")

    # Restore EMA params
    restored = ckpt_mgr.restore(
        step,
        args=ocp_args.Composite(
            ema_params=ocp_args.StandardRestore(item=init_params['params']),
        ),
    )
    ema_params = restored.ema_params
    print(f"Loaded {len(jax.tree_util.tree_leaves(ema_params))} parameter arrays from step {step}")

    return model, ema_params, nf_j

# ---------------------------------------------------------------------------
# String method
# ---------------------------------------------------------------------------
def classical_string_mep(string_init, score_fn, n_iters=500, step_size=1e-3):
    """Classical string method at fixed time: evolve along score gradient + reparametrize."""
    string = string_init
    n_pts = string.shape[0]
    for i in range(n_iters):
        scores = score_fn(string)
        string = string + step_size * scores
        string = uniform_string_repametrize(string, n_pts)
    return string

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-images', type=int, default=51)
    parser.add_argument('--n-mep-iters', type=int, default=500)
    parser.add_argument('--mep-step-size', type=float, default=0.001)
    parser.add_argument('--eval-t', type=float, default=0.0, help='ScoreMD time (0=data)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='/repo/bba_mep_results.json')
    parser.add_argument('--ckpt-dir', default='/models/scoremd_models/models/bba/both/model')
    args = parser.parse_args()

    # Load BBA data via ScoreMD's SingleProteinDataset
    from scoremd.data.dataset.protein import SingleProteinDataset
    print("Loading BBA dataset...")
    bba_ds = SingleProteinDataset(
        paths=['storage/deshaw/bba-0_ca.h5', 'storage/deshaw/bba-1_ca.h5'],
        tica_path='storage/deshaw/bba_tica.pic',
        topology_path='storage/deshaw/bba.pdb',
    )
    train_data = np.array(bba_ds.train.data)
    data_std = float(bba_ds.std)
    n_atoms = train_data.shape[1] // 3
    print(f"BBA: {train_data.shape[0]} frames, {n_atoms} atoms, std={data_std:.4f}")

    # Build model and load checkpoint
    print("\nBuilding model and loading checkpoint...")
    model, ema_params, nf_j = build_and_load_model(train_data, data_std, args.ckpt_dir)

    # Wrap params
    apply_params = {'params': ema_params}

    # Score function at fixed eval_t
    def score_fn(x_batch):
        """x_batch: (N, n_atoms*3) -> score: (N, n_atoms*3)"""
        t_arr = args.eval_t * jnp.ones((x_batch.shape[0], 1))
        return model.apply(apply_params, x_batch, None, t_arr, training=False)

    # Energy function: -log_q (potential energy)
    def energy_fn(x_batch):
        t_arr = args.eval_t * jnp.ones((x_batch.shape[0], 1))
        return model.apply(apply_params, x_batch, None, t_arr, training=False,
                          method=model.log_q)

    # Normalize data
    norm_data = jnp.array(train_data) * nf_j

    # Select two diverse endpoints
    idx_a = 0
    dists = jnp.linalg.norm(norm_data[0:1] - norm_data[:500], axis=1)
    idx_b = int(jnp.argmax(dists))
    print(f"\nEndpoints: A={idx_a} (folded), B={idx_b} (extended)")
    print(f"Distance: {float(dists[idx_b]):.4f}")

    # Create initial string (linear interpolation in normalized space)
    n_images = args.n_images
    alphas = jnp.linspace(0, 1, n_images)[:, None]
    x_a = norm_data[idx_a:idx_a+1]
    x_b = norm_data[idx_b:idx_b+1]
    initial_string = x_a * (1 - alphas) + x_b * alphas
    print(f"Initial string: {initial_string.shape}")

    # Run MEP convergence
    print(f"\nRunning classical string method ({args.n_mep_iters} iters, step={args.mep_step_size})...")
    mep_string = classical_string_mep(
        initial_string, score_fn,
        n_iters=args.n_mep_iters,
        step_size=args.mep_step_size,
    )

    # Compute scores along paths (as proxy for energy barrier)
    print("Computing score magnitudes along paths...")
    initial_scores = score_fn(initial_string)
    mep_scores = score_fn(mep_string)
    initial_score_norms = jnp.linalg.norm(initial_scores.reshape(n_images, -1), axis=1)
    mep_score_norms = jnp.linalg.norm(mep_scores.reshape(n_images, -1), axis=1)

    # Compute energies
    print("Computing energies...")
    initial_energy = -np.array(energy_fn(initial_string)).flatten()
    mep_energy = -np.array(energy_fn(mep_string)).flatten()

    # Results (relative to min energy along path)
    i_shift = float(np.min(initial_energy))
    m_shift = float(np.min(mep_energy))
    initial_energy_rel = initial_energy - i_shift
    mep_energy_rel = mep_energy - m_shift

    peak_initial = float(np.max(initial_energy_rel))
    peak_mep = float(np.max(mep_energy_rel))
    peak_initial_abs = float(np.max(initial_energy))
    peak_mep_abs = float(np.max(mep_energy))

    print(f"\n{'='*60}")
    print(f"RESULTS (relative to path minimum):")
    print(f"  Peak energy (initial string): {peak_initial:.2f}")
    print(f"  Peak energy (converged MEP):  {peak_mep:.2f}")
    print(f"  Reduction: {peak_initial - peak_mep:.2f}")
    print(f"RESULTS (absolute):")
    print(f"  Peak energy (initial string): {peak_initial_abs:.2f}")
    print(f"  Peak energy (converged MEP):  {peak_mep_abs:.2f}")
    print(f"\n  Paper target MEP peak:   ~10 kbT  (from Figure 9/12)")
    print(f"  Paper target initial:     ~50 kbT  (from Figure 9/12)")
    print(f"  Reproduce CI lower bound:  6 kbT")
    print(f"  Reproduce CI upper bound: 50 kbT")

    # Check if result is within CI bounds
    in_ci = 6 <= peak_mep <= 50
    print(f"\n  MEP peak within CI [6, 50]: {in_ci}")

    results = {
        'peak_energy_initial_string_rel': peak_initial,
        'peak_energy_converged_mep_rel': peak_mep,
        'peak_energy_initial_string_abs': peak_initial_abs,
        'peak_energy_converged_mep_abs': peak_mep_abs,
        'n_images': n_images,
        'n_mep_iters': args.n_mep_iters,
        'mep_step_size': args.mep_step_size,
        'eval_t': args.eval_t,
        'n_atoms': n_atoms,
        'paper_target_mep_kbT': 10,
        'paper_target_initial_kbT': 50,
        'reproduce_ci': [6, 50],
        'within_ci': in_ci,
    }
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    return 0 if in_ci else 1

if __name__ == '__main__':
    exit(main())
