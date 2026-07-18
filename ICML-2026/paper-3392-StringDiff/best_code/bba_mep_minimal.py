#!/usr/bin/env python3
"""Minimal BBA MEP reproduction from ScoreMD checkpoint.
Avoids heavy ScoreMD dependencies (deeptime, bgmol, etc.).
"""
import os, sys, json, argparse
import h5py
import numpy as np
import jax, jax.numpy as jnp
from jax import grad
import flax.linen as nn

# Add scoremd source for model architecture (bypass __init__.py)
sys.path.insert(0, '/scoremd/src')

# ---------------------------------------------------------------------------
# Data loading (manual, avoids SingleProteinDataset heaviness)
# ---------------------------------------------------------------------------
def load_bba_data(data_dir='/repo/storage/deshaw'):
    """Load BBA protein data from HDF5 files."""
    paths = [
        os.path.join(data_dir, 'bba-0_ca.h5'),
        os.path.join(data_dir, 'bba-1_ca.h5'),
    ]
    all_data = []
    all_features = []
    for p in paths:
        with h5py.File(p, 'r') as f:
            all_data.append(np.array(f['data']))
            if 'features' in f:
                all_features.append(np.array(f['features']))

    data = np.concatenate(all_data, axis=0)
    features = np.concatenate(all_features, axis=0) if all_features else None

    # BBA has 24 atoms (C-alpha only), so 24*3 = 72 coords
    n_atoms = data.shape[1] // 3
    print(f"BBA data: {data.shape}, {n_atoms} atoms")
    return data, features, n_atoms

def compute_norm(data):
    """Compute std for normalization (per-coordinate)."""
    std = float(np.std(data))
    print(f"Data std: {std:.4f}")
    return std

# ---------------------------------------------------------------------------
# Reparametrization (JAX)
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
# Model building (manual, avoids ScoreMD heavy config path)
# ---------------------------------------------------------------------------
class GraphTransformerLucid(nn.Module):
    """Simplified graph transformer from ScoreMD."""
    depth: int = 3
    hidden_nf: int = 128
    heads: int = 8
    dim_head: int = 64
    with_feedforwards: bool = True
    dropout: float = 0.0

    @nn.compact
    def __call__(self, nodes, edge_attr, training, mask=None):
        for _ in range(self.depth):
            # Self-attention
            h = nn.LayerNorm()(nodes)
            h_attn = nn.MultiHeadDotProductAttention(
                num_heads=self.heads,
                qkv_features=self.heads * self.dim_head,
                out_features=self.hidden_nf,
                dropout_rate=self.dropout,
            )(h, h, mask=mask)
            nodes = nodes + h_attn
            # Feedforward
            if self.with_feedforwards:
                h = nn.LayerNorm()(nodes)
                h_ff = nn.Dense(self.hidden_nf * 2)(h)
                h_ff = nn.gelu(h_ff)
                h_ff = nn.Dense(self.hidden_nf)(h_ff)
                h_ff = nn.Dropout(rate=self.dropout, deterministic=not training)(h_ff)
                nodes = nodes + h_ff
        return nodes, edge_attr


class BBAEnergyModel(nn.Module):
    """Minimal ScoreMD graph transformer for BBA."""
    hidden_nf: int = 128
    n_layers: int = 3
    potential: bool = True
    dropout: float = 0.0

    def setup(self):
        self.edge_embedding = nn.Dense(self.hidden_nf)
        self.node_embedding = nn.Dense(self.hidden_nf)
        self.transformer = GraphTransformerLucid(
            depth=self.n_layers,
            hidden_nf=self.hidden_nf,
            with_feedforwards=True,
            dropout=self.dropout,
        )
        if self.potential:
            self.node_decoder = nn.Dense(1)
        else:
            self.node_decoder = nn.Dense(3)  # per-atom output

    def _edge_attr(self, x):
        bs, n_nodes, _ = x.shape
        xa = jnp.expand_dims(x, axis=1)
        xb = jnp.expand_dims(x, axis=2)
        return xa - xb  # intrinsic coords

    def _time_features(self, t):
        """Sinusoidal time features."""
        t = t.reshape(-1, 1)
        features = [
            jnp.cos(2 * jnp.pi * t),
            jnp.sin(2 * jnp.pi * t),
            -jnp.cos(4 * jnp.pi * t),
        ]
        return jnp.concatenate(features, axis=-1)

    def __call__(self, x, t, training=False):
        """Forward pass: x (bs, n_atoms*3) -> output.
        For potential=True: returns score (negative gradient of energy).
        For potential=False: returns direct score prediction.
        """
        bs = x.shape[0]
        n_nodes = x.shape[1] // 3
        x = x.reshape(bs, n_nodes, 3)

        t_feat = self._time_features(t)
        t_feat = jnp.tile(t_feat[:, None, :], (1, n_nodes, 1))

        # Node features (one-hot identity)
        h = jnp.eye(n_nodes)
        h = jnp.tile(h[None], (bs, 1, 1))

        # Edge features from coordinates
        edge_attr = self._edge_attr(x)
        edge_attr = self.edge_embedding(edge_attr)

        # Node encoding
        nodes = jnp.concatenate([h, t_feat], axis=-1)
        nodes = self.node_embedding(nodes)

        # Transformer
        nodes, _ = self.transformer(nodes, edge_attr, training)

        if self.potential:
            # Output scalar energy per node, score = -grad(energy_sum)
            energy_per_node = self.node_decoder(nodes)  # (bs, n_nodes, 1)
            return -grad(lambda x_: self._energy_forward(x_, training).sum())(x)
        else:
            return self.node_decoder(nodes.reshape(bs, -1)).reshape(bs, n_nodes, 3).reshape(bs, -1)

    def _energy_forward(self, x, training=False):
        """Compute energy for grad."""
        bs = x.shape[0]
        n_nodes = x.shape[1] // 3
        x = x.reshape(bs, n_nodes, 3)
        t_zero = jnp.zeros((bs, 1))
        t_feat = self._time_features(t_zero)
        t_feat = jnp.tile(t_feat[:, None, :], (1, n_nodes, 1))
        h = jnp.eye(n_nodes)
        h = jnp.tile(h[None], (bs, 1, 1))
        edge_attr = self._edge_attr(x)
        edge_attr = self.edge_embedding(edge_attr)
        nodes = jnp.concatenate([h, t_feat], axis=-1)
        nodes = self.node_embedding(nodes)
        nodes, _ = self.transformer(nodes, edge_attr, training)
        return self.node_decoder(nodes)

    def energy(self, x, t, training=False):
        """Compute energy = -log_q."""
        return self._energy_forward(x, training).sum(axis=(1, 2))


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_bba_checkpoint(ckpt_dir, model, init_vars):
    """Load EMA params from ScoreMD checkpoint."""
    import orbax.checkpoint as ocp
    from orbax.checkpoint import args as ocp_args

    ckpt_path = os.path.join(ckpt_dir, 'model')
    options = ocp.CheckpointManagerOptions(max_to_keep=10, create=False)
    ckpt_mgr = ocp.CheckpointManager(os.path.abspath(ckpt_path), options=options)
    step = ckpt_mgr.latest_step()
    print(f"Checkpoint step: {step}")

    # Try direct restore first
    try:
        restored = ckpt_mgr.restore(step)
        if hasattr(restored, 'ema_params'):
            return restored.ema_params
        return restored['ema_params']
    except Exception as e:
        print(f"Direct restore failed: {e}")

    # Try with Composite args
    try:
        restored = ckpt_mgr.restore(
            step,
            args=ocp_args.Composite(
                ema_params=ocp_args.StandardRestore(item=init_vars['params']),
            ),
        )
        return restored.ema_params
    except Exception as e2:
        print(f"Composite restore failed: {e2}")

    # Last resort: try PyTreeCheckpointHandler directly
    print("Trying direct PyTree restore...")
    from orbax.checkpoint import PyTreeCheckpointHandler
    handler = PyTreeCheckpointHandler()
    ema_params = handler.restore(
        os.path.join(ckpt_path, str(step), 'ema_params'),
    )
    print(f"Restored {len(jax.tree_util.tree_leaves(ema_params))} parameter leaves")
    return ema_params


# ---------------------------------------------------------------------------
# String method
# ---------------------------------------------------------------------------
def classical_mep(string_init, score_fn, n_iters=500, step_size=1e-3, eval_t=1e-5):
    """Classical string method at fixed time t."""
    string = string_init.copy()
    n_pts = string.shape[0]
    for i in range(n_iters):
        score_vals = score_fn(string, eval_t)
        string = string + step_size * score_vals
        string = uniform_string_repametrize(string, n_pts)
        if i % 100 == 0:
            max_score = float(jnp.max(jnp.abs(score_vals)))
            print(f"  iter {i:4d}: max|score|={max_score:.6f}")
    return string


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-images', type=int, default=51)
    parser.add_argument('--n-mep-iters', type=int, default=500)
    parser.add_argument('--mep-step-size', type=float, default=0.001)
    parser.add_argument('--eval-t', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='/repo/bba_mep_results.json')
    args = parser.parse_args()

    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    key = jax.random.PRNGKey(args.seed)

    print("=" * 60)
    print("Loading BBA data...")
    data, features, n_atoms = load_bba_data()
    data_std = compute_norm(data)
    norm_factor = 1.0 / data_std
    print(f"Normalization factor: {norm_factor:.4f}")

    print("\nBuilding BBA energy model...")
    model = BBAEnergyModel(hidden_nf=128, n_layers=3, potential=True, dropout=0.0)

    # Initialize with dummy data
    dummy_x = jnp.array(data[:2]) / data_std
    dummy_t = jnp.array([[0.5], [0.5]])
    init_vars = model.init(key, dummy_x, dummy_t, training=False)
    print(f"Model params: {len(jax.tree_util.tree_leaves(init_vars['params']))} arrays")

    print("\nLoading checkpoint...")
    ckpt_dir = '/models/scoremd_models/models/bba/both'
    ema_params = load_bba_checkpoint(ckpt_dir, model, init_vars)

    # Wrap params
    params = {'params': ema_params}

    # Test forward pass
    test_x = jnp.array(data[:4]) / data_std
    test_t = args.eval_t
    test_out = model.apply(params, test_x, test_t, training=False)
    print(f"Test forward pass shape: {test_out.shape}")
    print(f"Test output range: [{test_out.min():.4f}, {test_out.max():.4f}]")

    # Score function
    def score_fn(x_batch, t_val):
        t_arr = t_val * jnp.ones((x_batch.shape[0], 1))
        return model.apply(params, x_batch, t_arr, training=False)

    # Get diverse endpoints from training data
    norm_data = jnp.array(data[:100]) / data_std
    # Find two most distant conformations
    n_check = min(100, len(data))
    idx_a = 0
    # Find farthest from idx_a
    dists = jnp.linalg.norm(norm_data[0:1] - norm_data[:n_check], axis=1)
    idx_b = int(jnp.argmax(dists))
    print(f"\nEndpoints: A={idx_a}, B={idx_b}")
    print(f"Distance: {dists[idx_b]:.4f}")

    x_a = norm_data[idx_a:idx_a+1]
    x_b = norm_data[idx_b:idx_b+1]

    # Linear interpolation as initial string
    n_images = args.n_images
    alphas = jnp.linspace(0, 1, n_images)[:, None]
    initial_string = x_a * (1 - alphas) + x_b * alphas
    print(f"Initial string shape: {initial_string.shape}")

    # Run MEP
    print(f"\nRunning classical string method...")
    mep_string = classical_mep(
        initial_string, score_fn,
        n_iters=args.n_mep_iters,
        step_size=args.mep_step_size,
        eval_t=args.eval_t,
    )

    # Compute energy profiles
    print("\nComputing energy profiles...")
    def compute_energy(string_pts, t_val):
        t_arr = t_val * jnp.ones((string_pts.shape[0], 1))
        return model.apply(params, string_pts, t_arr, training=False,
                          method=model.energy)

    initial_energy_raw = compute_energy(initial_string, args.eval_t)
    mep_energy_raw = compute_energy(mep_string, args.eval_t)

    # Scale back to physical units: energy is in units of kT
    # The model output needs to be unscaled
    initial_energy = np.array(initial_energy_raw)
    mep_energy = np.array(mep_energy_raw)

    peak_initial = float(np.max(initial_energy))
    peak_mep = float(np.max(mep_energy))
    mean_initial = float(np.mean(initial_energy))
    mean_mep = float(np.mean(mep_energy))

    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Peak energy (initial string): {peak_initial:.2f}")
    print(f"  Peak energy (converged MEP): {peak_mep:.2f}")
    print(f"  Mean energy (initial string): {mean_initial:.2f}")
    print(f"  Mean energy (converged MEP): {mean_mep:.2f}")
    print(f"  Energy reduction: {peak_initial - peak_mep:.2f} ({100*(1-peak_mep/peak_initial) if peak_initial > 0 else 0:.1f}%)")
    print(f"\n  Paper target MEP peak: ~10 kbT")
    print(f"  Paper target initial peak: ~50 kbT")

    results = {
        'peak_energy_initial_string': peak_initial,
        'peak_energy_converged_mep': peak_mep,
        'mean_energy_initial_string': mean_initial,
        'mean_energy_converged_mep': mean_mep,
        'paper_target_mep_kbT': 10,
        'paper_target_initial_kbT': 50,
        'n_images': n_images,
        'n_mep_iters': args.n_mep_iters,
        'mep_step_size': args.mep_step_size,
        'eval_t': args.eval_t,
        'n_atoms': n_atoms,
    }
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    return 0

if __name__ == '__main__':
    exit(main())
